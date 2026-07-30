from typing import Self

import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext
from core.front_end.landmark_cache import LandmarkCache
from core.front_end.landmark_refiner import LandmarkRefiner, LandmarkRefineStatus, Refiner
from core.front_end.observation_store import (
    CompressPolicy,
    ObservationSchema,
    ObservationSlots,
    ObservationStore,
    SelectPolicy,
)
from core.front_end.ray_triangulation import LandmarkTriangulator, RayTriangulation, TriangulationStatus
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger

type TrackingInfo = NDArray[np.float64]


class InitializedLandmarkSchema:
    """Schema for initialized landmarks."""

    FEAT_ID = 0
    X = 1
    Y = 2
    Z = 3

    XYZ = slice(X, Z + 1)

    @classmethod
    def count(cls) -> int:
        """Return the number of columns in the initialized landmark schema."""
        return cls.Z + 1


class ObservationTrackStatus:
    """Status of the observation of landmarks."""

    UNINITIALIZED = 0
    INITIALIZED = 1


class LandmarkInitialization:
    """Component for collecting tracking information for landmark initialization."""

    def __init__(
        self,
        store: ObservationStore,
        triangulator: LandmarkTriangulator,
        refiner: Refiner,
        cache: LandmarkCache,
        stereo_ctx: StereoContext,
    ) -> None:
        """Initialize the landmark initialization class."""
        self._store = store
        self._k_inv = np.linalg.inv(stereo_ctx.stereo_k)
        self._triangulator = triangulator
        self._refiner = refiner
        self._cache = cache
        self.logger = spawn_logger(__name__)

        self._point_in_world_by_feat_id: dict[int, NDArray[np.float64]] = {}
        self.rect0_from_rect1 = np.eye(4, dtype=np.float64)
        self.rect0_from_rect1[0, 3] = stereo_ctx.baseline

    @classmethod
    def default_factory(cls, stereo_ctx: StereoContext, capacity: int = 1000) -> Self:
        """Create a default landmark initialization class."""
        return cls(
            ObservationStore.default_factory(
                compress_policy=CompressPolicy.TOP_DISPLACEMENT,
                k_inv=np.linalg.inv(stereo_ctx.stereo_k),
                select_policy=SelectPolicy.P90_PARALLAX,
                capacity=capacity,
            ),
            RayTriangulation.default_factory(stereo_ctx),
            LandmarkRefiner(stereo_ctx),
            LandmarkCache.default_factory(capacity=capacity),
            stereo_ctx,
        )

    def add_observation(self, tracking_info: TrackingInfo, pose_estimate: SE3) -> ObservationSlots:
        """Add an observation to the landmark initialization class."""
        pose_matrix = pose_estimate.as_matrix()

        observations = np.full((tracking_info.shape[0], ObservationSchema.size()), np.nan, dtype=np.float64)
        observations[:, ObservationSchema.FEAT_ID] = tracking_info[:, StereoTriangulationSchema.FEAT_ID]
        observations[:, ObservationSchema.CAM0_MATRIX] = pose_matrix.flatten()
        observations[:, ObservationSchema.LEFT_U] = tracking_info[:, StereoTriangulationSchema.LEFT_U]
        observations[:, ObservationSchema.LEFT_V] = tracking_info[:, StereoTriangulationSchema.LEFT_V]
        observations[:, ObservationSchema.RIGHT_U] = tracking_info[:, StereoTriangulationSchema.RIGHT_U]
        observations[:, ObservationSchema.RIGHT_V] = tracking_info[:, StereoTriangulationSchema.RIGHT_V]
        observations[:, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT] = 0

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

            left_uv = rows[:, ObservationSchema.LEFT_UV]
            # right_uv = rows[:, ObservationSchema.RIGHT_UV]
            world_from_cam0 = rows[:, ObservationSchema.CAM0_MATRIX].reshape(-1, 4, 4)
            world_from_anchor = world_from_cam0[0]
            anchor_from_world_rot = world_from_anchor[:3, :3].T
            anchor_from_world_t = -anchor_from_world_rot @ world_from_anchor[:3, 3]

            anchor_from_cam0 = np.tile(np.eye(4, dtype=np.float64), (world_from_cam0.shape[0], 1, 1))
            anchor_from_cam0[:, :3, :3] = np.einsum(
                "ij,njk->nik", anchor_from_world_rot, world_from_cam0[:, :3, :3]
            )
            anchor_from_cam0[:, :3, 3] = (
                np.einsum("ij,nj->ni", anchor_from_world_rot, world_from_cam0[:, :3, 3]) + anchor_from_world_t
            )

            # right_valid_mask = np.all(np.isfinite(right_uv), axis=1)
            right_num = 0  # right_uv[right_valid_mask, :].shape[0]

            uvs = np.full((left_uv.shape[0] + right_num, 2), np.nan, dtype=np.float64)
            uvs[: left_uv.shape[0], :] = left_uv
            # uvs[left_uv.shape[0] :, :] = right_uv[right_valid_mask, :]

            poses = np.full((left_uv.shape[0] + right_num, 4, 4), np.nan, dtype=np.float64)
            poses[: left_uv.shape[0], :, :] = anchor_from_cam0[: left_uv.shape[0], :, :]
            # poses[left_uv.shape[0] :, :, :] = anchor_from_cam0[right_valid_mask, :, :] @ self.rect0_from_rect1

            ray_cast_status, point_in_anchor = self._triangulator.triangulate_feature_observations(uvs, poses)
            if ray_cast_status != TriangulationStatus.SUCCESS:
                self._cache.apply_failed(slot_slice, history_version_slice)
                continue
            refine_status, point_in_anchor = self._refiner.refine_point_gn(point_in_anchor, uvs, poses)
            if refine_status != LandmarkRefineStatus.SUCCESS:
                self._cache.apply_failed(slot_slice, history_version_slice)
                continue

            point_in_world = world_from_anchor[:3, :3] @ point_in_anchor + world_from_anchor[:3, 3]
            self._cache.apply_completed(feat_id_slice, slot_slice, point_in_world.reshape(1, 3))
            initialized_count += 1

        if initialized_count > 0:
            self.logger.trace(f"Initialized {initialized_count} landmarks")
