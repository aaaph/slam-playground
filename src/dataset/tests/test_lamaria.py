from pathlib import Path

import cv2
import numpy as np

from dataset.euroc import EurocConfig, EurocDataset, decode_stereo_pair
from dataset.lamaria import LamariaDatasetAdapter, LamariaStreamLoader
from dataset.manifest import DatasetManifest
from dataset.registry import DatasetRegistry


def _write_asl(root: Path) -> DatasetManifest:
    for cam in ("cam0", "cam1"):
        (root / cam / "data").mkdir(parents=True)
        (root / cam / "data.csv").write_text("100,100.png\n200,200.png\n")
        image = np.zeros((4, 6), dtype=np.uint8)
        image[0, 0] = 255  # top-left marker
        for ts in (100, 200):
            cv2.imwrite(str(root / cam / "data" / f"{ts}.png"), image)
    (root / "imu0").mkdir()
    (root / "imu0" / "data.csv").write_text(
        "90,0.1,0.2,0.3,-9.8,0,0\n95,0.1,0.2,0.3,-9.8,0,0\n150,0.1,0.2,0.3,-9.8,0,0\n190,0.1,0.2,0.3,-9.8,0,0\n"
    )
    (root / "gt.txt").write_text("100 1 2 3 0 0 0 1\n200 4 5 6 0.5 0.5 0.5 0.5\n")
    return DatasetManifest(
        name="lamaria_test",
        type="lamaria",
        root=root,
        rig=Path("config/dataset_rig/lamaria_r01.yaml"),
        streams={
            "cam0": "cam0/data.csv",
            "cam1": "cam1/data.csv",
            "imu0": "imu0/data.csv",
            "ground_truth": "gt.txt",
        },
    )


class TestLamariaDataset:
    """Tests for the LaMAria stream loader and canonical adapter."""

    def test_materialize_groups_imu_and_keeps_gt_xyzw(self, tmp_path: Path) -> None:
        """Header-less ASL csv + TUM GT should become the canonical stereo/IMU/GT schema."""
        manifest = _write_asl(tmp_path)
        hf_dataset = LamariaDatasetAdapter().materialize(LamariaStreamLoader().load(manifest))
        dataset = EurocDataset(hf_dataset, config=None)  # ty: ignore

        frames = dataset.imu_and_stereo(decode_images=False)[:]
        assert frames["timestamp"].tolist() == [100, 200]
        assert [len(ts) for ts in frames["imu_ts"]] == [1, 2]

        # 90 deg clockwise: (H, W) -> (W, H), top-left goes to top-right.
        left, _right = decode_stereo_pair(frames["stereo"][0])
        assert left.shape == (6, 4)
        assert left[0, 3] == 255

        ground_truth = dataset.ground_truth()[:]
        np.testing.assert_allclose(np.stack(ground_truth["gt_position"]), [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_allclose(ground_truth["gt_orientation"][1], [0.5, 0.5, 0.5, 0.5])
        assert np.isnan(np.asarray(ground_truth["gt_velocity"], dtype=np.float32)).all()

    def test_rig_matches_rotated_frames(self) -> None:
        """Rig must describe upright frames: xyzw qvec (~75 deg between axes) and cam1 to the right of cam0."""
        rig = DatasetRegistry(repo_root=Path.cwd()).resolve("lamaria_r01").rig
        config = EurocConfig.from_rig_config(rig)
        t_body_cam0 = config.cam0.body_sensor_transform
        t_body_cam1 = config.cam1.body_sensor_transform
        cam1_in_cam0 = (np.linalg.inv(t_body_cam0) @ t_body_cam1)[:3, 3]

        # ~14 deg would mean qvec was read as wxyz.
        assert abs(np.rad2deg(np.arccos(t_body_cam0[:3, 2] @ t_body_cam1[:3, 2])) - 75.2) < 0.5
        # Horizontal baseline: the stereo matcher expects cam1 at +x and no vertical offset.
        assert cam1_in_cam0[0] > 0.1
        assert abs(cam1_in_cam0[1]) < 0.01
        assert config.cam1.resolution == (569, 757)
