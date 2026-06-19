from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from numpy.typing import NDArray

from core.transformations.special_euclidian_3_dim import SE3
from datasets import Dataset

GROUND_TRUTH_INDEX_SCHEMA_VERSION = 1
GROUND_TRUTH_INDEX_CACHE_FILENAME = "ground_truth_index_v1.npz"


@dataclass(frozen=True)
class GroundTruthSample:
    """Ground truth sample."""

    timestamp_ns: int
    position: NDArray[np.float32]
    orientation: NDArray[np.float32]
    velocity: NDArray[np.float32]

    def se3(self) -> SE3:
        """Get the SE3 transformation for the ground truth sample."""
        return SE3.from_quat_and_translation(self.orientation.astype(np.float64), self.position.astype(np.float64))


@dataclass(frozen=True)
class GroundTruthIndex:
    """Ground truth index."""

    timestamps_ns: NDArray[np.int64]
    positions: NDArray[np.float32]
    orientations: NDArray[np.float32]
    velocities: NDArray[np.float32]

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> Self:
        """Create a ground truth index from a dataset."""
        ds = dataset.sort("timestamp").with_format("numpy")
        return cls(
            timestamps_ns=np.array(ds["timestamp"], dtype=np.int64),
            positions=np.array(ds["gt_position"], dtype=np.float32),
            orientations=np.array(ds["gt_orientation"], dtype=np.float32),
            velocities=np.array(ds["gt_velocity"], dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> Self:
        """Load a ground truth index from a NumPy cache file."""
        with np.load(path, allow_pickle=False) as data:
            schema_version = int(data["schema_version"].item())
            if schema_version != GROUND_TRUTH_INDEX_SCHEMA_VERSION:
                msg = (
                    f"Unsupported ground truth index schema version {schema_version}; "
                    f"expected {GROUND_TRUTH_INDEX_SCHEMA_VERSION}"
                )
                raise ValueError(msg)
            return cls(
                timestamps_ns=np.array(data["timestamps_ns"], dtype=np.int64),
                positions=np.array(data["positions"], dtype=np.float32),
                orientations=np.array(data["orientations"], dtype=np.float32),
                velocities=np.array(data["velocities"], dtype=np.float32),
            )

    @classmethod
    def load_or_build(cls, path: Path, dataset_factory: Callable[[], Dataset]) -> Self:
        """Load a cached ground truth index, rebuilding it when the cache is missing or invalid."""
        if path.exists():
            try:
                return cls.load(path)
            except (OSError, ValueError, KeyError, TypeError):
                pass

        index = cls.from_dataset(dataset_factory())
        index.save(path)
        return index

    def save(self, path: Path) -> None:
        """Save a ground truth index to a NumPy cache file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
        np.savez(
            tmp_path,
            schema_version=np.array(GROUND_TRUTH_INDEX_SCHEMA_VERSION, dtype=np.int64),
            timestamps_ns=self.timestamps_ns,
            positions=self.positions,
            orientations=self.orientations,
            velocities=self.velocities,
        )
        tmp_path.replace(path)

    def nearest_index(self, timestamp_ns: int) -> int:
        """Find the index of the ground truth sample closest to the timestamp."""
        if len(self.timestamps_ns) == 0:
            msg = "Ground truth index is empty"
            raise ValueError(msg)

        insert_idx = int(np.searchsorted(self.timestamps_ns, timestamp_ns))
        if insert_idx == 0:
            return 0
        if insert_idx == len(self.timestamps_ns):
            return len(self.timestamps_ns) - 1

        before_idx = insert_idx - 1
        before_delta = abs(int(self.timestamps_ns[before_idx]) - timestamp_ns)
        after_delta = abs(int(self.timestamps_ns[insert_idx]) - timestamp_ns)
        return before_idx if before_delta <= after_delta else insert_idx

    def nearest(self, timestamp_ns: int) -> GroundTruthSample:
        """Find the ground truth sample closest to the timestamp."""
        idx = self.nearest_index(timestamp_ns)
        return GroundTruthSample(
            timestamp_ns=int(self.timestamps_ns[idx]),
            position=self.positions[idx],
            orientation=self.orientations[idx],
            velocity=self.velocities[idx],
        )
