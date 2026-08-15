from collections import OrderedDict
from enum import IntEnum
from typing import ClassVar, Self

import numpy as np
from numpy.typing import NDArray

from core.pose_tracker.hypothesis_estimator import HypothesisEstimator

FeatureId = int
Vector3d = NDArray[np.float64]
FRONTEND_OBSERVATION_NDIM = 2


class LocalMapPointStatus(IntEnum):
    """Lifecycle status of a local-map landmark."""

    FRONTEND_CANDIDATE = 0
    FRONTEND_MATURE = 1
    BACKEND_OPTIMIZED = 2


class LocalMapSchema:
    """Local map schema."""

    FEAT_ID = 0
    X = 1
    Y = 2
    Z = 3
    COV_XX = 4
    COV_XY = 5
    COV_XZ = 6
    COV_YX = 7
    COV_YY = 8
    COV_YZ = 9
    COV_ZX = 10
    COV_ZY = 11
    COV_ZZ = 12
    DEPTH_SIGMA = 13
    STATUS = 14
    HEALTH = 15
    OBS_COUNT = 16
    FIRST_SEEN_TS = 17
    LAST_OBSERVED_TS = 18
    LAST_UPDATED_TS = 19
    BACKEND_VERSION = 20

    XYZ = slice(X, Z + 1)
    COV = slice(COV_XX, COV_ZZ + 1)

    @classmethod
    def count(cls) -> int:
        """Return the number of columns in the local map schema."""
        return cls.BACKEND_VERSION + 1


class CandidateHistorySchema:
    """Candidate history schema."""

    FEAT_ID: ClassVar[int] = 0
    TIMESTAMP_NS: ClassVar[int] = 1
    X = 2
    Y = 3
    Z = 4

    COV_XX = 5
    COV_XY = 6
    COV_XZ = 7
    COV_YX = 8
    COV_YY = 9
    COV_YZ = 10
    COV_ZX = 11
    COV_ZY = 12
    COV_ZZ = 13
    DEPTH_SIGMA: ClassVar[int] = 14

    LEFT_U = 15
    LEFT_V = 16
    RIGHT_U = 17
    RIGHT_V = 18

    CAM_X = 19
    CAM_Y = 20
    CAM_Z = 21

    XYZ: ClassVar[slice] = slice(X, Z + 1)
    COV: ClassVar[slice] = slice(COV_XX, COV_ZZ + 1)
    CAM_XYZ: ClassVar[slice] = slice(CAM_X, CAM_Z + 1)
    LEFT_UV = slice(LEFT_U, LEFT_V + 1)
    RIGHT_UV = slice(RIGHT_U, RIGHT_V + 1)

    @classmethod
    def count(cls) -> int:
        """Return the number of columns in the candidate history schema."""
        return cls.CAM_Z + 1

    @classmethod
    def covariance_matrix(cls, row: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert row-major covariance columns to a symmetric 3x3 covariance matrix."""
        return row[cls.COV].reshape(3, 3)


class LocalMap:
    """Local map backed by a fixed-size table."""

    def __init__(self, capacity: int = 1000, history_size: int = 5) -> None:
        """Initialize the local map."""
        self.hypothesis_estimator = HypothesisEstimator(CandidateHistorySchema)
        self.capacity = capacity
        self.stable_health_threshold = -3
        # Public attribute is kept for compatibility with the current tests and LRU checks.
        self.landmarks: OrderedDict[FeatureId, int] = OrderedDict()
        self._data = np.full((capacity, LocalMapSchema.count()), np.nan, dtype=np.float64)

        self._history = np.full((capacity, history_size, CandidateHistorySchema.count()), np.nan, dtype=np.float64)
        self._history_versions: dict[FeatureId, int] = {}
        self._history_head = np.zeros(capacity, dtype=np.int16)
        self._history_count = np.zeros(capacity, dtype=np.int16)

        self._feat_id_to_idx: dict[FeatureId, int] = {}
        self._free_slots: list[int] = []
        self._next_slot = 0
        self.iteration = 0

    @classmethod
    def from_capacity(cls, capacity: int) -> Self:
        """Create a local map from a capacity."""
        return cls(capacity)

    def _get_free_slot(self) -> int:
        """Get a free slot or evict the least recently used point."""
        if self._free_slots:
            return self._free_slots.pop()
        if self._next_slot < self.capacity:
            slot = self._next_slot
            self._next_slot += 1
            return slot

        feat_id, idx = self.landmarks.popitem(last=False)
        self._feat_id_to_idx.pop(feat_id, None)
        self._data[idx].fill(np.nan)
        return idx

    def _get_existing_slot(self, feat_id: FeatureId) -> int:
        """Get an existing slot for the feature id."""
        idx = self._feat_id_to_idx.get(feat_id)
        if idx is None:
            msg = f"Feature with ID {feat_id} not found"
            raise ValueError(msg)
        return idx

    def _find_slots(self, feat_ids: NDArray[np.int32]) -> NDArray[np.int32]:
        """Find the slots of the features in the local map."""
        if len(feat_ids) != len(np.unique(feat_ids)):
            raise ValueError("Duplicate feature IDs")
        output = np.zeros_like(feat_ids, dtype=np.int32)
        for i, feat_id in enumerate(feat_ids):
            feat_id_val = feat_id.item()
            idx = self._feat_id_to_idx.get(feat_id_val, None)
            if idx is None:
                msg = f"Feature with ID {feat_id_val} not found"
                raise ValueError(msg)
            output[i] = idx
        return output

    def add_point(self, feat_id: FeatureId, point_3d: Vector3d) -> None:
        """Add or update a frontend candidate point in the local map."""
        row = np.full((1, CandidateHistorySchema.count()), np.nan, dtype=np.float64)
        row[0, CandidateHistorySchema.FEAT_ID] = feat_id
        row[0, CandidateHistorySchema.TIMESTAMP_NS] = 0.0
        row[0, CandidateHistorySchema.XYZ] = point_3d
        self.add_frontend_observations(row)

    def add_points(self, new_points: dict[FeatureId, Vector3d]) -> None:
        """Add frontend candidate points to the local map."""
        for feat_id, point_3d in new_points.items():
            self.add_point(feat_id, point_3d)

    def get_point(self, feat_id: FeatureId) -> Vector3d | None:
        """Get a point from the local map."""
        idx = self._feat_id_to_idx.get(feat_id)
        if idx is None:
            return None

        self.landmarks.move_to_end(feat_id)
        return self._data[idx, LocalMapSchema.X : LocalMapSchema.Z + 1].copy()

    def increase_health(self, feat_ids: NDArray[np.int32]) -> NDArray[np.int32]:
        """Increase the health score for an existing landmark."""
        if feat_ids.size == 0:
            return np.empty(0, dtype=np.int32)
        indexes = self._find_slots(feat_ids)
        self._data[indexes, LocalMapSchema.HEALTH] += 1.0
        for feat_id in feat_ids:
            self.landmarks.move_to_end(int(feat_id))
        return self._data[indexes, LocalMapSchema.HEALTH].astype(np.int32, copy=False)

    def decrease_health(self, feat_ids: NDArray[np.int32]) -> NDArray[np.int32]:
        """Decrease the health score for an existing landmark."""
        if feat_ids.size == 0:
            return np.empty(0, dtype=np.int32)
        indexes = self._find_slots(feat_ids)
        self._data[indexes, LocalMapSchema.HEALTH] -= 1.0
        for feat_id in feat_ids:
            self.landmarks.move_to_end(int(feat_id))
        return self._data[indexes, LocalMapSchema.HEALTH].astype(np.int32, copy=False)

    def get_batch(self, feat_ids: NDArray[np.int32]) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """Get a batch of rows aligned to input feat ids as [feat_id, x, y, z]."""
        mask = np.zeros(feat_ids.shape[0], dtype=bool)
        points = np.full((feat_ids.shape[0], LocalMapSchema.Z + 1), np.nan, dtype=np.float64)
        points[:, LocalMapSchema.FEAT_ID] = feat_ids.astype(np.float32, copy=False)

        for i, feat_id in enumerate(feat_ids):
            idx = self._feat_id_to_idx.get(int(feat_id))
            if idx is None:
                continue
            mask[i] = True
            points[i, LocalMapSchema.X : LocalMapSchema.Z + 1] = self._data[
                idx, LocalMapSchema.X : LocalMapSchema.Z + 1
            ]

        return mask, points

    def get_stable_batch(self, feat_ids: NDArray[np.int32]) -> tuple[NDArray[np.bool_], NDArray[np.float64]]:
        """Get stable rows aligned to input feat ids using the full local-map schema."""
        mask = np.zeros(feat_ids.shape[0], dtype=bool)
        points = np.full((feat_ids.shape[0], LocalMapSchema.count()), np.nan, dtype=np.float64)
        points[:, LocalMapSchema.FEAT_ID] = feat_ids.astype(np.float32, copy=False)

        for i, feat_id in enumerate(feat_ids):
            idx = self._feat_id_to_idx.get(int(feat_id))
            if idx is None:
                continue
            health = self._data[idx, LocalMapSchema.HEALTH]
            if health < self.stable_health_threshold:
                continue
            mask[i] = True
            points[i] = self._data[idx]

        return mask, points

    def get_points_with_covariance(self) -> NDArray[np.float64]:
        """Return active local-map rows that have finite XYZ and covariance."""
        if self.empty():
            return np.empty((0, LocalMapSchema.count()), dtype=np.float64)

        indexes = np.fromiter(self.landmarks.values(), dtype=np.int32)
        points = self._data[indexes].copy()
        valid_mask = np.isfinite(points[:, LocalMapSchema.FEAT_ID])
        valid_mask &= np.all(np.isfinite(points[:, LocalMapSchema.XYZ]), axis=1)
        valid_mask &= np.all(np.isfinite(points[:, LocalMapSchema.COV]), axis=1)
        return points[valid_mask]

    def exists(self, feat_id: FeatureId) -> bool:
        """Check if a point exists in the local map."""
        return feat_id in self._feat_id_to_idx

    def empty(self) -> bool:
        """Check if the local map is empty."""
        return len(self._feat_id_to_idx) == 0

    def clear(self) -> None:
        """Clear the local map."""
        self.landmarks.clear()
        self._feat_id_to_idx.clear()
        self._data.fill(np.nan)
        self._free_slots.clear()
        self._next_slot = 0
        self.iteration = 0

    def add_ndarray(self, ndarray: NDArray[np.float64]) -> None:
        """Add rows shaped as [feat_id, x, y, z, ...] as frontend candidate observations."""
        self.add_frontend_observations(ndarray)

    def add_frontend_observations(self, ndarray: NDArray[np.float64]) -> None:  # noqa: C901, PLR0915
        """Add frontend triangulated observations without overwriting backend-owned landmarks."""
        ndarray = self._normalize_frontend_observation_rows(ndarray)
        if ndarray.shape[0] == 0:
            return
        if ndarray.shape[0] > self.capacity:
            raise ValueError("Too many points to add")

        for row in ndarray:
            if not np.isfinite(row[CandidateHistorySchema.FEAT_ID]) or not np.all(
                np.isfinite(row[CandidateHistorySchema.XYZ])
            ):
                continue

            feat_id = int(row[CandidateHistorySchema.FEAT_ID])
            point_3d = row[CandidateHistorySchema.XYZ].astype(np.float64, copy=False)
            covariance = CandidateHistorySchema.covariance_matrix(row)
            depth_sigma = row[CandidateHistorySchema.DEPTH_SIGMA].astype(np.float64, copy=False)
            timestamp_ns = self._timestamp_or_default(row[CandidateHistorySchema.TIMESTAMP_NS])

            idx, is_new = self._ensure_slot(feat_id)
            if is_new:
                # new point in local map -> FRONTEND_CANDIDATE initialization
                self._data[idx, LocalMapSchema.FEAT_ID] = feat_id
                self._data[idx, LocalMapSchema.XYZ] = point_3d
                self._data[idx, LocalMapSchema.COV] = covariance.reshape(9).astype(np.float64, copy=False)
                self._data[idx, LocalMapSchema.DEPTH_SIGMA] = depth_sigma
                self._data[idx, LocalMapSchema.STATUS] = LocalMapPointStatus.FRONTEND_CANDIDATE.value
                self._data[idx, LocalMapSchema.HEALTH] = 0.0
                self._data[idx, LocalMapSchema.OBS_COUNT] = 1.0
                self._data[idx, LocalMapSchema.FIRST_SEEN_TS] = timestamp_ns
                self._data[idx, LocalMapSchema.LAST_OBSERVED_TS] = timestamp_ns
                self._data[idx, LocalMapSchema.LAST_UPDATED_TS] = timestamp_ns
                history_head = self._history_head[idx]
                self._history[idx, history_head] = row
                self._history_head[idx] = (history_head + 1) % self._history.shape[1]
                self._history_count[idx] = min(self._history_count[idx] + 1, self._history.shape[1])

                continue

            feat_status = LocalMapPointStatus(int(self._data[idx, LocalMapSchema.STATUS]))
            self._data[idx, LocalMapSchema.OBS_COUNT] += 1.0
            self._data[idx, LocalMapSchema.LAST_OBSERVED_TS] = timestamp_ns

            match feat_status:
                case LocalMapPointStatus.FRONTEND_CANDIDATE:
                    history_head = self._history_head[idx]
                    self._history[idx, history_head] = row
                    self._history_head[idx] = (history_head + 1) % self._history.shape[1]
                    self._history_count[idx] = min(self._history_count[idx] + 1, self._history.shape[1])
                    feat_history = self._history[idx, : self._history_count[idx]]
                    hypothesis = self.hypothesis_estimator.estimate(feat_history)
                    self._data[idx, LocalMapSchema.HEALTH] += hypothesis.health_delta

                    if hypothesis.pnp_eligible:
                        self._data[idx, LocalMapSchema.XYZ] = hypothesis.xyz
                        self._data[idx, LocalMapSchema.COV] = hypothesis.covariance_row_major
                        self._data[idx, LocalMapSchema.DEPTH_SIGMA] = hypothesis.depth_sigma
                        self._data[idx, LocalMapSchema.LAST_UPDATED_TS] = timestamp_ns

                    if hypothesis.promote_to_mature:
                        self._data[idx, LocalMapSchema.STATUS] = LocalMapPointStatus.FRONTEND_MATURE.value

                    if hypothesis.promote_to_mature or hypothesis.pnp_eligible:
                        self._data[idx, LocalMapSchema.LAST_UPDATED_TS] = timestamp_ns

                case LocalMapPointStatus.FRONTEND_MATURE:
                    continue
                case LocalMapPointStatus.BACKEND_OPTIMIZED:
                    continue

        self.iteration += 1

    @staticmethod
    def _normalize_frontend_observation_rows(ndarray: NDArray[np.float64]) -> NDArray[np.float64]:
        """Normalize legacy frontend rows to the candidate-history schema."""
        ndarray = np.asarray(ndarray, dtype=np.float64)
        if ndarray.ndim != FRONTEND_OBSERVATION_NDIM:
            raise ValueError("Frontend observations must be a 2D ndarray")
        if ndarray.shape[1] == CandidateHistorySchema.count():
            return ndarray
        if ndarray.shape[1] <= LocalMapSchema.Z:
            raise ValueError("Frontend observation rows must contain at least feat_id, x, y, z")

        rows = np.full((ndarray.shape[0], CandidateHistorySchema.count()), np.nan, dtype=np.float64)
        rows[:, CandidateHistorySchema.FEAT_ID] = ndarray[:, LocalMapSchema.FEAT_ID]
        rows[:, CandidateHistorySchema.TIMESTAMP_NS] = 0.0
        rows[:, CandidateHistorySchema.XYZ] = ndarray[:, LocalMapSchema.XYZ]

        if ndarray.shape[1] > LocalMapSchema.COV_ZZ:
            rows[:, CandidateHistorySchema.COV] = ndarray[:, LocalMapSchema.COV]
        if ndarray.shape[1] > LocalMapSchema.DEPTH_SIGMA:
            rows[:, CandidateHistorySchema.DEPTH_SIGMA] = ndarray[:, LocalMapSchema.DEPTH_SIGMA]
        return rows

    def apply_backend_landmarks(self, ndarray: NDArray[np.float64], timestamp_ns: float | None = None) -> None:
        """Apply backend optimized landmarks and make backend the owner of those landmarks."""
        if ndarray.shape[0] == 0:
            return
        if ndarray.shape[0] > self.capacity:
            raise ValueError("Too many points to add")
        if ndarray.shape[1] <= LocalMapSchema.Z:
            raise ValueError("Backend landmark rows must contain at least feat_id, x, y, z")

        for row in ndarray:
            if not np.isfinite(row[LocalMapSchema.FEAT_ID]) or not np.all(np.isfinite(row[LocalMapSchema.XYZ])):
                continue

            feat_id = int(row[LocalMapSchema.FEAT_ID])
            idx, is_new = self._ensure_slot(feat_id)
            if is_new:
                self._initialize_slot(idx, feat_id, LocalMapPointStatus.BACKEND_OPTIMIZED, timestamp_ns)

            self._data[idx, LocalMapSchema.XYZ] = row[LocalMapSchema.XYZ]
            covariance = self._read_covariance(row)
            if covariance is not None:
                self._data[idx, LocalMapSchema.COV] = covariance
            depth_sigma = self._read_optional_scalar(row, LocalMapSchema.DEPTH_SIGMA)
            if np.isfinite(depth_sigma):
                self._data[idx, LocalMapSchema.DEPTH_SIGMA] = depth_sigma

            self._data[idx, LocalMapSchema.STATUS] = LocalMapPointStatus.BACKEND_OPTIMIZED.value
            self._data[idx, LocalMapSchema.HEALTH] = max(
                self._finite_or_default(idx, LocalMapSchema.HEALTH, 1.0), 1.0
            )
            self._data[idx, LocalMapSchema.LAST_OBSERVED_TS] = self._timestamp_or_default(timestamp_ns)
            self._data[idx, LocalMapSchema.LAST_UPDATED_TS] = self._timestamp_or_default(timestamp_ns)
            self._data[idx, LocalMapSchema.BACKEND_VERSION] = (
                self._finite_or_default(idx, LocalMapSchema.BACKEND_VERSION, 0.0) + 1.0
            )
            self.landmarks.move_to_end(feat_id)
            self.iteration += 1

    @staticmethod
    def covariance_to_matrix(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert row-major covariance columns to a symmetric 3x3 covariance matrix."""
        covariance_matrix = np.asarray(covariance, dtype=np.float64).reshape(3, 3)
        return 0.5 * (covariance_matrix + covariance_matrix.T)

    @staticmethod
    def matrix_to_covariance(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert a 3x3 covariance matrix to row-major covariance columns."""
        covariance_matrix = 0.5 * (covariance + covariance.T)
        return covariance_matrix.reshape(9).astype(np.float64, copy=False)

    def _ensure_slot(self, feat_id: FeatureId) -> tuple[int, bool]:
        idx = self._feat_id_to_idx.get(feat_id)
        if idx is not None:
            self.landmarks.move_to_end(feat_id)
            return idx, False

        idx = self._get_free_slot()
        self._feat_id_to_idx[feat_id] = idx
        self.landmarks[feat_id] = idx
        return idx, True

    def _initialize_slot(
        self,
        idx: int,
        feat_id: FeatureId,
        status: LocalMapPointStatus,
        timestamp_ns: float | None,
    ) -> None:
        timestamp = self._timestamp_or_default(timestamp_ns)
        self._data[idx].fill(np.nan)
        self._data[idx, LocalMapSchema.FEAT_ID] = feat_id
        self._data[idx, LocalMapSchema.STATUS] = status.value
        self._data[idx, LocalMapSchema.HEALTH] = 1.0
        self._data[idx, LocalMapSchema.OBS_COUNT] = 0.0
        self._data[idx, LocalMapSchema.FIRST_SEEN_TS] = timestamp
        self._data[idx, LocalMapSchema.LAST_OBSERVED_TS] = timestamp
        self._data[idx, LocalMapSchema.LAST_UPDATED_TS] = timestamp
        self._data[idx, LocalMapSchema.BACKEND_VERSION] = 0.0

    def _update_frontend_candidate(
        self,
        idx: int,
        point_3d: Vector3d,
        covariance: NDArray[np.float64] | None,
        depth_sigma: float,
    ) -> bool:
        old_count = self._finite_or_default(idx, LocalMapSchema.OBS_COUNT, 0.0)
        old_covariance = self._read_covariance(self._data[idx])
        if old_count > 0 and covariance is not None and old_covariance is not None:
            merged_point, merged_covariance = self._merge_measurement_with_covariance_inflation(
                self._data[idx, LocalMapSchema.XYZ],
                old_covariance,
                point_3d,
                covariance,
                old_count,
            )
            self._data[idx, LocalMapSchema.XYZ] = merged_point
            self._data[idx, LocalMapSchema.COV] = merged_covariance
            depth_sigma = self._depth_sigma_from_covariance(merged_covariance)
        elif old_count > 0 and covariance is None:
            weight = 1.0 / (old_count + 1.0)
            self._data[idx, LocalMapSchema.XYZ] = (1.0 - weight) * self._data[
                idx, LocalMapSchema.XYZ
            ] + weight * point_3d
        else:
            self._data[idx, LocalMapSchema.XYZ] = point_3d
            if covariance is not None:
                self._data[idx, LocalMapSchema.COV] = covariance.reshape(9).astype(np.float64, copy=False)

        if np.isfinite(depth_sigma):
            self._data[idx, LocalMapSchema.DEPTH_SIGMA] = depth_sigma
        self._data[idx, LocalMapSchema.STATUS] = LocalMapPointStatus.FRONTEND_CANDIDATE.value
        return True

    def _merge_measurement_with_covariance_inflation(
        self,
        old_point: Vector3d,
        old_covariance: NDArray[np.float64],
        measured_point: Vector3d,
        measured_covariance: NDArray[np.float64],
        old_count: float,
    ) -> tuple[Vector3d, NDArray[np.float64]]:
        old_cov = self.covariance_to_matrix(old_covariance)
        measured_cov = self.covariance_to_matrix(measured_covariance)
        old_weight = max(old_count, 1.0)
        measured_weight = 1.0
        total_weight = old_weight + measured_weight
        old_ratio = old_weight / total_weight
        measured_ratio = measured_weight / total_weight

        merged_point = old_ratio * old_point + measured_ratio * measured_point
        old_residual = old_point - merged_point
        measured_residual = measured_point - merged_point
        merged_cov = old_ratio * (old_cov + np.outer(old_residual, old_residual)) + measured_ratio * (
            measured_cov + np.outer(measured_residual, measured_residual)
        )
        return merged_point, self.matrix_to_covariance(merged_cov)

    @staticmethod
    def _depth_sigma_from_covariance(covariance: NDArray[np.float64]) -> float:
        covariance_matrix = LocalMap.covariance_to_matrix(covariance)
        return float(np.sqrt(max(covariance_matrix[2, 2], 0.0)))

    def _status(self, idx: int) -> LocalMapPointStatus:
        value = self._data[idx, LocalMapSchema.STATUS]
        if not np.isfinite(value):
            return LocalMapPointStatus.FRONTEND_CANDIDATE
        return LocalMapPointStatus(int(value))

    def _mark_observed(
        self,
        idx: int,
        timestamp_ns: float | None,
        *,
        updated: bool,
        accepted: bool,
    ) -> None:
        timestamp = self._timestamp_or_default(timestamp_ns)
        self._data[idx, LocalMapSchema.LAST_OBSERVED_TS] = timestamp
        if accepted:
            self._data[idx, LocalMapSchema.OBS_COUNT] = (
                self._finite_or_default(idx, LocalMapSchema.OBS_COUNT, 0.0) + 1.0
            )
        if updated:
            self._data[idx, LocalMapSchema.LAST_UPDATED_TS] = timestamp

    @staticmethod
    def _timestamp_or_default(timestamp_ns: float | None) -> float:
        if timestamp_ns is None or not np.isfinite(timestamp_ns):
            return 0.0
        return float(timestamp_ns)

    def _finite_or_default(self, idx: int, column: int, default: float) -> float:
        value = self._data[idx, column]
        if not np.isfinite(value):
            return default
        return float(value)

    @staticmethod
    def _read_optional_scalar(row: NDArray[np.float64], column: int) -> float:
        if row.shape[0] <= column or not np.isfinite(row[column]):
            return np.nan
        return float(row[column])

    @staticmethod
    def _read_covariance(row: NDArray[np.float64]) -> NDArray[np.float64] | None:
        if row.shape[0] <= LocalMapSchema.COV_ZZ:
            return None
        covariance = row[LocalMapSchema.COV].astype(np.float64, copy=False)
        if not np.all(np.isfinite(covariance)):
            return None
        covariance_matrix = LocalMap.covariance_to_matrix(covariance)
        if np.any(np.diag(covariance_matrix) <= 0.0):
            return None
        return LocalMap.matrix_to_covariance(covariance_matrix)
