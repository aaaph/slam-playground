from typing import Self

import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext
from core.front_end.landmark_cache import LandmarkCache
from core.front_end.landmark_triangulation import (
    LandmarkTriangulator,
    LandmarkTriangulatorProtocol,
    TriangulationStatus,
)
from core.front_end.observation_store import (
    CompressPolicy,
    ObservationSchema,
    ObservationSlots,
    ObservationStore,
    SelectPolicy,
)
from logger import spawn_logger

type LandmarkObservations = NDArray[np.float64]


class InitializedLandmarkSchema:
    """Schema for initialized landmarks."""

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

    XYZ = slice(X, Z + 1)
    COV = slice(COV_XX, COV_ZZ + 1)

    @classmethod
    def count(cls) -> int:
        """Return the number of columns in the initialized landmark schema."""
        return cls.DEPTH_SIGMA + 1


class ObservationTrackStatus:
    """Status of the observation of landmarks."""

    UNINITIALIZED = 0
    INITIALIZED = 1


class LandmarkInitialization:
    """Component for collecting tracking information for landmark initialization."""

    def __init__(
        self,
        store: ObservationStore,
        cache: LandmarkCache,
        triangulator: LandmarkTriangulatorProtocol,
    ) -> None:
        """Initialize the landmark initialization class."""
        self._store = store
        self._cache = cache
        self.logger = spawn_logger(__name__)
        self._triangulator = triangulator

    @classmethod
    def default_factory(cls, stereo_ctx: StereoContext, capacity: int = 1000) -> Self:
        """Create a default landmark initialization class."""
        return cls(
            ObservationStore.default_factory(
                compress_policy=CompressPolicy.UNIFORM_RECENT,
                k_inv=np.linalg.inv(stereo_ctx.stereo_k),
                select_policy=SelectPolicy.P90_PARALLAX,
                capacity=capacity,
            ),
            LandmarkCache.default_factory(capacity=capacity),
            LandmarkTriangulator.default_factory(stereo_ctx),
        )

    def add_observation(self, observations: LandmarkObservations) -> ObservationSlots:
        """Add an observation to the landmark initialization class."""
        used_slots, _history_slots = self._store.add_observations(observations)
        self.logger.trace(f"Added {observations.shape[0]} observations to the landmark initialization")
        observing_feat_ids = observations[:, ObservationSchema.FEAT_ID].astype(np.int32, copy=False)
        self._cache.apply_observing(observing_feat_ids, used_slots)

        ready_slots, ready_history_versions, ready_feat_ids = self._store.ready_slots(used_slots)
        cached_ready = self._cache.apply_ready(ready_feat_ids, ready_slots, ready_history_versions)
        return cached_ready[:20]

    def remove_lost_features(self, lost_features: NDArray[np.int32]) -> None:
        """Remove lost features from the landmark initialization class."""
        removed_slots = self._store.remove_features(lost_features)
        self._cache.clear_slots(removed_slots)
        self.logger.trace(f"Removed {lost_features.shape[0]} lost features from the landmark initialization")

    def get_initialized_landmarks(self) -> NDArray[np.float64]:
        """Get initialized landmarks from the cache."""
        return self._cache.get_completed_landmarks()

    def triangulate_ready_observations(self, ready_slots: ObservationSlots) -> None:
        """Triangulate ready observations and write the results to the landmark cache."""
        ready_observations, history_mask = self._store.get_feature_slice_by_slots(ready_slots)
        ready_feat_ids = self._store.get_feat_ids_by_slots(ready_slots)
        ready_history_versions = self._store.get_history_versions_by_slots(ready_slots)
        initialized_count = 0

        for i in range(ready_slots.shape[0]):
            rows = ready_observations[i, history_mask[i], :]
            slot_slice = ready_slots[i : i + 1]
            feat_id_slice = ready_feat_ids[i : i + 1]
            history_version_slice = ready_history_versions[i : i + 1]
            left_uvs = rows[:, ObservationSchema.LEFT_UV]
            right_uvs = rows[:, ObservationSchema.RIGHT_UV]
            left_poses = rows[:, ObservationSchema.CAM0_MATRIX].reshape(-1, 4, 4)

            status, point_in_world = self._triangulator.triangulate_mixed(left_uvs, right_uvs, left_poses)
            if status != TriangulationStatus.SUCCESS:
                self._cache.apply_failed(slot_slice, history_version_slice)
                continue

            self._cache.apply_completed(
                feat_id_slice,
                slot_slice,
                point_in_world.reshape(1, 3),
                np.zeros((1, 3, 3), dtype=np.float64),
            )
            initialized_count += 1

        if initialized_count > 0:
            self.logger.trace(f"Initialized {initialized_count} landmarks")
