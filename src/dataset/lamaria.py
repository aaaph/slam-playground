from pathlib import Path

import cv2
import pandas as pd

from dataset.builder import RawStream, RawStreamBundle, RawStreamLoader
from dataset.euroc import EurocDatasetBuilder
from dataset.manifest import DatasetManifest
from datasets import Dataset, Features, Image, Sequence, Value

# LaMAria ASL csv files have no header; ground truth is TUM-style "ts tx ty tz qx qy qz qw".
_STREAM_FORMATS: dict[str, tuple[str, list[str]]] = {
    "cam0": (",", ["timestamp", "filename"]),
    "cam1": (",", ["timestamp", "filename"]),
    "imu0": (",", ["timestamp", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"]),
    "ground_truth": (
        " ",
        ["timestamp", "gt_x", "gt_y", "gt_z", "gt_qx", "gt_qy", "gt_qz", "gt_qw"],
    ),
}

_VECTOR = Sequence(Value("float64"))
_FEATURES = Features(
    {
        "timestamp": Value("int64"),
        "stereo": Sequence(Value("string"), length=2),
        "gyro": _VECTOR,
        "acc": _VECTOR,
        "gt_position": _VECTOR,
        "gt_orientation": _VECTOR,
        "gt_velocity": _VECTOR,
        "gt_gyro_bias": _VECTOR,
        "gt_acc_bias": _VECTOR,
        "has_imu": Value("bool"),
        "has_ground_truth": Value("bool"),
    }
)


def _upright_png(path: str | None) -> dict[str, object] | None:
    """Store a frame rotated 90 deg clockwise, since Aria SLAM cameras are mounted sideways."""
    # The rig config (config/dataset_rig/lamaria_*.yaml) holds K, resolution and T_BS of the rotated frames.
    if path is None:
        return None
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if image is None:
        msg = f"Failed to read LaMAria frame {path}"
        raise ValueError(msg)
    upright = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return {"bytes": cv2.imencode(".png", upright)[1].tobytes(), "path": Path(path).name}


class LamariaStreamLoader(RawStreamLoader):
    """Load header-less LaMAria ASL streams with explicit column names."""

    def load(self, manifest: DatasetManifest) -> RawStreamBundle:
        """Load manifest streams into named raw dataframes."""
        root = self.resolve_path(manifest.root)
        cache = self.resolve_path(manifest.cache) if manifest.cache is not None else root / "cache"
        streams = {
            name: RawStream(path=path, frame=self._read(name, path))
            for name, path in self.resolve_stream_paths(manifest, root).items()
        }
        return RawStreamBundle(manifest=manifest, root=root, cache=cache, streams=streams)

    @staticmethod
    def _read(name: str, path: Path) -> pd.DataFrame:
        sep, columns = _STREAM_FORMATS[name]
        return pd.read_csv(path, sep=sep, header=None, names=columns)


class LamariaDatasetAdapter:
    """Map raw LaMAria streams into the canonical HuggingFace dataset schema."""

    def materialize(self, streams: RawStreamBundle) -> Dataset:
        """Join camera, IMU and ground truth streams into one timestamp-sorted table."""
        left = self._cam_frame(streams.require("cam0"), "left_image")
        right = self._cam_frame(streams.require("cam1"), "right_image")
        frame = (
            left.merge(right, on="timestamp", how="inner")
            .merge(streams.require("imu0").frame, on="timestamp", how="outer")
            .merge(streams.require("ground_truth").frame, on="timestamp", how="outer")
            .sort_values("timestamp", ignore_index=True)
        )
        # LaMAria pseudo-GT has no velocity or IMU biases: keep the canonical columns as nulls.
        unknown = [[None, None, None]] * len(frame)
        table = pd.DataFrame(
            {
                "timestamp": frame["timestamp"].astype("int64"),
                "stereo": self._rows(frame, ["left_image", "right_image"]),
                "gyro": self._rows(frame, ["gyro_x", "gyro_y", "gyro_z"]),
                "acc": self._rows(frame, ["acc_x", "acc_y", "acc_z"]),
                "gt_position": self._rows(frame, ["gt_x", "gt_y", "gt_z"]),
                "gt_orientation": self._rows(frame, ["gt_qx", "gt_qy", "gt_qz", "gt_qw"]),
                "gt_velocity": unknown,
                "gt_gyro_bias": unknown,
                "gt_acc_bias": unknown,
                "has_imu": frame["gyro_x"].notna(),
                "has_ground_truth": frame["gt_x"].notna(),
            }
        )
        dataset = Dataset.from_pandas(table, features=_FEATURES, preserve_index=False)
        features = dataset.features.copy()
        features["stereo"] = Sequence(Image(decode=False), length=2)
        return dataset.map(
            lambda batch: {"stereo": [[_upright_png(path) for path in pair] for pair in batch["stereo"]]},
            batched=True,
            features=features,
            desc="Rotate Aria frames",
        )

    @staticmethod
    def _cam_frame(stream: RawStream, image_column: str) -> pd.DataFrame:
        image_prefix = stream.path.parent / "data"
        return pd.DataFrame(
            {
                "timestamp": stream.frame["timestamp"],
                image_column: [f"{image_prefix}/{filename}" for filename in stream.frame["filename"]],
            }
        )

    @staticmethod
    def _rows(frame: pd.DataFrame, columns: list[str]) -> list[list[object]]:
        # Missing stream values become [None, ...], matching the EuRoC canonical rows.
        missing = [None] * len(columns)
        present = frame[columns[0]].notna().to_numpy()
        values = frame[columns].to_numpy().tolist()
        return [row if ok else missing for row, ok in zip(values, present, strict=True)]


def lamaria_dataset_builder(repo_root: Path | None = None) -> EurocDatasetBuilder:
    """Cache-first builder wired with the LaMAria stream loader and adapter."""
    return EurocDatasetBuilder(
        repo_root=repo_root,
        stream_loader=LamariaStreamLoader(repo_root=repo_root),
        adapter=LamariaDatasetAdapter(),
    )
