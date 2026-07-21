from collections import OrderedDict
from enum import IntEnum
from typing import Self

import numpy as np
from numpy.typing import NDArray

FeatureId = int
Vector3d = NDArray[np.float64]


class LocalMapPointSource(IntEnum):
    """Owner/source of a local-map landmark."""

    FRONTEND_CANDIDATE = 0
    BACKEND_OPTIMIZED = 1


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
    SOURCE = 14
    HEALTH = 15
    OBS_COUNT = 16
    FIRST_SEEN_TS = 17
    LAST_OBSERVED_TS = 18
    LAST_UPDATED_TS = 19
    BACKEND_VERSION = 20

    XYZ = slice(X, Z + 1)
    COV = slice(COV_XX, DEPTH_SIGMA)

    @classmethod
    def count(cls) -> int:
        """Return the number of columns in the local map schema."""
        return cls.BACKEND_VERSION + 1


class LocalMap:
    """Local map backed by a fixed-size table."""

    frontend_fusion_mahalanobis_threshold: float = 16.27

    def __init__(self, capacity: int = 1000) -> None:
        """Initialize the local map."""
        self.capacity = capacity
        self.stable_health_threshold = -3
        # Public attribute is kept for compatibility with the current tests and LRU checks.
        self.landmarks: OrderedDict[FeatureId, int] = OrderedDict()
        self._data = np.full((capacity, LocalMapSchema.count()), np.nan, dtype=np.float64)
        self._feat_id_to_idx: dict[FeatureId, int] = {}
        self._free_slots = list(range(capacity - 1, -1, -1))

    @classmethod
    def from_capacity(cls, capacity: int) -> Self:
        """Create a local map from a capacity."""
        return cls(capacity)

    def _get_free_slot(self) -> int:
        """Get a free slot or evict the least recently used point."""
        if self._free_slots:
            return self._free_slots.pop()

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
        row = np.full((1, LocalMapSchema.count()), np.nan, dtype=np.float64)
        row[0, LocalMapSchema.FEAT_ID] = feat_id
        row[0, LocalMapSchema.XYZ] = point_3d
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
        self._free_slots = list(range(self.capacity - 1, -1, -1))

    def add_ndarray(self, ndarray: NDArray[np.float64]) -> None:
        """Add rows shaped as [feat_id, x, y, z, ...] as frontend candidate observations."""
        self.add_frontend_observations(ndarray)

    def add_frontend_observations(self, ndarray: NDArray[np.float64], timestamp_ns: float | None = None) -> None:
        """Add frontend triangulated observations without overwriting backend-owned landmarks."""
        if ndarray.shape[0] == 0:
            return
        if ndarray.shape[0] > self.capacity:
            raise ValueError("Too many points to add")
        if ndarray.shape[1] <= LocalMapSchema.Z:
            raise ValueError("Local map rows must contain at least feat_id, x, y, z")

        for row in ndarray:
            if not np.isfinite(row[LocalMapSchema.FEAT_ID]) or not np.all(np.isfinite(row[LocalMapSchema.XYZ])):
                continue

            feat_id = int(row[LocalMapSchema.FEAT_ID])
            point_3d = row[LocalMapSchema.XYZ].astype(np.float64, copy=False)
            covariance = self._read_covariance(row)
            depth_sigma = self._read_optional_scalar(row, LocalMapSchema.DEPTH_SIGMA)

            idx, is_new = self._ensure_slot(feat_id)
            if is_new:
                self._initialize_slot(idx, feat_id, LocalMapPointSource.FRONTEND_CANDIDATE, timestamp_ns)

            source = self._source(idx)
            if source == LocalMapPointSource.BACKEND_OPTIMIZED:
                self._mark_observed(idx, timestamp_ns, updated=False, accepted=True)
                continue

            accepted = self._update_frontend_candidate(idx, point_3d, covariance, depth_sigma)
            self._mark_observed(idx, timestamp_ns, updated=accepted, accepted=accepted)

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
                self._initialize_slot(idx, feat_id, LocalMapPointSource.BACKEND_OPTIMIZED, timestamp_ns)

            self._data[idx, LocalMapSchema.XYZ] = row[LocalMapSchema.XYZ]
            covariance = self._read_covariance(row)
            if covariance is not None:
                self._data[idx, LocalMapSchema.COV] = covariance
            depth_sigma = self._read_optional_scalar(row, LocalMapSchema.DEPTH_SIGMA)
            if np.isfinite(depth_sigma):
                self._data[idx, LocalMapSchema.DEPTH_SIGMA] = depth_sigma

            self._data[idx, LocalMapSchema.SOURCE] = LocalMapPointSource.BACKEND_OPTIMIZED.value
            self._data[idx, LocalMapSchema.HEALTH] = max(
                self._finite_or_default(idx, LocalMapSchema.HEALTH, 1.0), 1.0
            )
            self._data[idx, LocalMapSchema.LAST_OBSERVED_TS] = self._timestamp_or_default(timestamp_ns)
            self._data[idx, LocalMapSchema.LAST_UPDATED_TS] = self._timestamp_or_default(timestamp_ns)
            self._data[idx, LocalMapSchema.BACKEND_VERSION] = (
                self._finite_or_default(idx, LocalMapSchema.BACKEND_VERSION, 0.0) + 1.0
            )
            self.landmarks.move_to_end(feat_id)

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
        source: LocalMapPointSource,
        timestamp_ns: float | None,
    ) -> None:
        timestamp = self._timestamp_or_default(timestamp_ns)
        self._data[idx].fill(np.nan)
        self._data[idx, LocalMapSchema.FEAT_ID] = feat_id
        self._data[idx, LocalMapSchema.SOURCE] = source.value
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
            fused = self._fuse_measurement(
                self._data[idx, LocalMapSchema.XYZ],
                old_covariance,
                point_3d,
                covariance,
            )
            if fused is None:
                self._data[idx, LocalMapSchema.HEALTH] -= 1.0
                return False
            fused_point, fused_covariance = fused
            self._data[idx, LocalMapSchema.XYZ] = fused_point
            self._data[idx, LocalMapSchema.COV] = fused_covariance
        elif old_count > 0 and covariance is None:
            weight = 1.0 / (old_count + 1.0)
            self._data[idx, LocalMapSchema.XYZ] = (1.0 - weight) * self._data[
                idx, LocalMapSchema.XYZ
            ] + weight * point_3d
        else:
            self._data[idx, LocalMapSchema.XYZ] = point_3d
            if covariance is not None:
                self._data[idx, LocalMapSchema.COV] = covariance

        if np.isfinite(depth_sigma):
            self._data[idx, LocalMapSchema.DEPTH_SIGMA] = depth_sigma
        self._data[idx, LocalMapSchema.SOURCE] = LocalMapPointSource.FRONTEND_CANDIDATE.value
        return True

    def _fuse_measurement(
        self,
        old_point: Vector3d,
        old_covariance: NDArray[np.float64],
        measured_point: Vector3d,
        measured_covariance: NDArray[np.float64],
    ) -> tuple[Vector3d, NDArray[np.float64]] | None:
        old_cov = self.covariance_to_matrix(old_covariance)
        measured_cov = self.covariance_to_matrix(measured_covariance)
        innovation = measured_point - old_point
        innovation_cov = old_cov + measured_cov
        # print(f"mahalanobis distance: {self._mahalanobis_distance(innovation, innovation_cov)}")
        if self._mahalanobis_distance(innovation, innovation_cov) > self.frontend_fusion_mahalanobis_threshold:
            return None

        old_info = self._safe_inverse(old_cov)
        measured_info = self._safe_inverse(measured_cov)
        fused_cov = self._safe_inverse(old_info + measured_info)
        fused_point = fused_cov @ (old_info @ old_point + measured_info @ measured_point)
        return fused_point, self.matrix_to_covariance(fused_cov)

    @staticmethod
    def _mahalanobis_distance(vector: Vector3d, covariance: NDArray[np.float64]) -> float:
        return float(vector.T @ LocalMap._safe_inverse(covariance) @ vector)

    @staticmethod
    def _safe_inverse(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        jitter = np.eye(matrix.shape[0], dtype=np.float64) * 1e-9
        return np.linalg.pinv(matrix + jitter)

    def _source(self, idx: int) -> LocalMapPointSource:
        value = self._data[idx, LocalMapSchema.SOURCE]
        if not np.isfinite(value):
            return LocalMapPointSource.FRONTEND_CANDIDATE
        return LocalMapPointSource(int(value))

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
