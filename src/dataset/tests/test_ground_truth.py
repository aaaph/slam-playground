from pathlib import Path

import numpy as np
import pytest

from dataset.ground_truth import GroundTruthIndex
from datasets import Dataset


def _minimal_euroc_dataset(timestamps: list[int] | None = None) -> Dataset:
    timestamps = timestamps or [1]
    return Dataset.from_dict(
        {
            "timestamp": timestamps,
            "stereo": [[None, None] for _ in timestamps],
            "gyro": [[0.0, 0.0, 0.0] for _ in timestamps],
            "acc": [[0.0, 0.0, 0.0] for _ in timestamps],
            "gt_position": [[float(t), float(t + 1), float(t + 2)] for t in timestamps],
            "gt_orientation": [[0.0, 0.0, 0.0, 1.0] for _ in timestamps],
            "gt_velocity": [[float(t), 0.0, 0.0] for t in timestamps],
            "gt_gyro_bias": [[0.0, 0.0, 0.0] for _ in timestamps],
            "gt_acc_bias": [[0.0, 0.0, 0.0] for _ in timestamps],
        }
    )


class TestGroundTruth:
    """Test GroundTruth class."""

    def test_from_dataset(self) -> None:
        """Test that the GroundTruth can be created from a dataset."""
        dataset = _minimal_euroc_dataset()
        ground_truth = GroundTruthIndex.from_dataset(dataset)
        np.testing.assert_array_equal(ground_truth.timestamps_ns, np.array([1], dtype=np.int64))
        np.testing.assert_array_equal(ground_truth.positions, np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
        np.testing.assert_array_equal(
            ground_truth.orientations, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        )
        np.testing.assert_array_equal(ground_truth.velocities, np.array([[1.0, 0.0, 0.0]], dtype=np.float32))

    def test_from_dataset_sorts_by_timestamp(self) -> None:
        """Test that the GroundTruth sorts source rows by timestamp."""
        dataset = _minimal_euroc_dataset([30, 10, 20])
        ground_truth = GroundTruthIndex.from_dataset(dataset)
        np.testing.assert_array_equal(ground_truth.timestamps_ns, np.array([10, 20, 30], dtype=np.int64))
        np.testing.assert_array_equal(
            ground_truth.positions,
            np.array([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]], dtype=np.float32),
        )

    @pytest.mark.parametrize(
        ("timestamp_ns", "expected_idx"),
        [
            (0, 0),
            (10, 0),
            (14, 0),
            (15, 0),
            (16, 1),
            (25, 1),
            (26, 2),
            (40, 2),
        ],
    )
    def test_nearest_index(self, timestamp_ns: int, expected_idx: int) -> None:
        """Test that nearest_index finds the closest timestamp."""
        ground_truth = GroundTruthIndex.from_dataset(_minimal_euroc_dataset([10, 20, 30]))
        assert ground_truth.nearest_index(timestamp_ns) == expected_idx

    def test_nearest_returns_sample(self) -> None:
        """Test that nearest returns the closest ground truth sample."""
        ground_truth = GroundTruthIndex.from_dataset(_minimal_euroc_dataset([10, 20, 30]))
        sample = ground_truth.nearest(24)
        assert sample.timestamp_ns == 20
        np.testing.assert_array_equal(sample.position, ground_truth.positions[1])
        np.testing.assert_array_equal(sample.orientation, ground_truth.orientations[1])
        np.testing.assert_array_equal(sample.velocity, ground_truth.velocities[1])

    def test_empty_index_raises_on_nearest_index(self) -> None:
        """Test that nearest lookup fails explicitly for an empty index."""
        ground_truth = GroundTruthIndex(
            timestamps_ns=np.array([], dtype=np.int64),
            positions=np.empty((0, 3), dtype=np.float32),
            orientations=np.empty((0, 4), dtype=np.float32),
            velocities=np.empty((0, 3), dtype=np.float32),
        )
        with pytest.raises(ValueError, match="Ground truth index is empty"):
            ground_truth.nearest_index(1)

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Test that a GroundTruthIndex can be cached and loaded."""
        cache_path = tmp_path / "ground_truth_index_v1.npz"
        ground_truth = GroundTruthIndex.from_dataset(_minimal_euroc_dataset([10, 20, 30]))
        ground_truth.save(cache_path)

        loaded = GroundTruthIndex.load(cache_path)

        np.testing.assert_array_equal(loaded.timestamps_ns, ground_truth.timestamps_ns)
        np.testing.assert_array_equal(loaded.positions, ground_truth.positions)
        np.testing.assert_array_equal(loaded.orientations, ground_truth.orientations)
        np.testing.assert_array_equal(loaded.velocities, ground_truth.velocities)

    def test_load_or_build_uses_cache_without_building_dataset(self, tmp_path: Path) -> None:
        """Test that cache hits avoid building the source dataset."""
        cache_path = tmp_path / "ground_truth_index_v1.npz"
        ground_truth = GroundTruthIndex.from_dataset(_minimal_euroc_dataset([10]))
        ground_truth.save(cache_path)

        def fail_if_called() -> Dataset:
            raise AssertionError("dataset builder should not be called on cache hit")

        loaded = GroundTruthIndex.load_or_build(cache_path, fail_if_called)

        np.testing.assert_array_equal(loaded.timestamps_ns, np.array([10], dtype=np.int64))

    def test_load_or_build_rebuilds_invalid_cache(self, tmp_path: Path) -> None:
        """Test that invalid cache files are replaced by a rebuilt index."""
        cache_path = tmp_path / "ground_truth_index_v1.npz"
        cache_path.write_text("not a numpy archive", encoding="utf-8")

        loaded = GroundTruthIndex.load_or_build(cache_path, lambda: _minimal_euroc_dataset([20]))

        np.testing.assert_array_equal(loaded.timestamps_ns, np.array([20], dtype=np.int64))
        reloaded = GroundTruthIndex.load(cache_path)
        np.testing.assert_array_equal(reloaded.timestamps_ns, np.array([20], dtype=np.int64))

    def test_se3(self) -> None:
        """Test that the SE3 transformation for the ground truth sample is correct."""
        ground_truth = GroundTruthIndex.from_dataset(_minimal_euroc_dataset([10, 20, 30]))
        sample = ground_truth.nearest(24)
        se3 = sample.se3()
        assert se3 is not None
        np.testing.assert_allclose(se3.rotation().as_quat(), np.array([0.0, 0.0, 0.0, 1.0]))
        np.testing.assert_allclose(se3.translation(), np.array([20.0, 21.0, 22.0]))
