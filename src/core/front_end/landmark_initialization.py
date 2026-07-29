from typing import Self

import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext
from core.front_end.landmark_refiner import LandmarkRefiner, LandmarkRefineStatus, Refiner
from core.front_end.observation_store import (
    CompressPolicy,
    ObservationSchema,
    ObservationSlots,
    ObservationStore,
    ReadyObservationCriteria,
)
from core.front_end.ray_triangulation import LandmarkTriangulator, RayTriangulation, TriangulationStatus
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger
from logger.decorators import timeit

type TrackingInfo = NDArray[np.float64]
type InitializedLandmarks = NDArray[np.float64]


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
        ready_criteria: ReadyObservationCriteria,
        refiner: Refiner,
        stereo_ctx: StereoContext,
    ) -> None:
        """Initialize the landmark initialization class."""
        self._store = store
        self._k_inv = np.linalg.inv(stereo_ctx.stereo_k)
        self._triangulator = triangulator
        self._refiner = refiner
        self.logger = spawn_logger(__name__)

        self._ready_criteria = ready_criteria
        self._point_in_world_by_feat_id: dict[int, NDArray[np.float64]] = {}
        self.rect0_from_rect1 = np.eye(4, dtype=np.float64)
        self.rect0_from_rect1[0, 3] = stereo_ctx.baseline

    @classmethod
    def default_factory(cls, stereo_ctx: StereoContext) -> Self:
        """Create a default landmark initialization class."""
        return cls(
            ObservationStore.default_factory(
                compress_policy=CompressPolicy.TOP_DISPLACEMENT,
                k_inv=np.linalg.inv(stereo_ctx.stereo_k),
            ),
            RayTriangulation.default_factory(stereo_ctx),
            ReadyObservationCriteria(min_history_size=5, min_parallax_rad=0.01),
            LandmarkRefiner(stereo_ctx),
            stereo_ctx,
        )

    @timeit
    def add_observation(self, tracking_info: TrackingInfo, pose_estimate: SE3) -> InitializedLandmarks:
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

        _ready_slots, _ready_history, _ready_feat_ids = self._ready_slots(used_slots)

        return np.empty((0, InitializedLandmarkSchema.count()), dtype=np.float64)

    def remove_lost_features(self, lost_features: NDArray[np.int32]) -> None:
        """Remove lost features from the landmark initialization class."""
        self._store.remove_features(lost_features)
        for feat_id in lost_features:
            self._point_in_world_by_feat_id.pop(int(feat_id), None)
        self.logger.trace(f"Removed {lost_features.shape[0]} lost features from the landmark initialization")

    def _ready_slots(
        self, candidate_slots: ObservationSlots
    ) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32]]:
        """Get ready store slots, history versions, and feature IDs."""
        return self._store.ready_slots(self._ready_criteria, candidate_slots)

    @timeit
    def _try_to_triangulate_observation(
        self,
        ready_feat_ids: NDArray[np.int32],
        history_mask: NDArray[np.bool_],
        ready_features: NDArray[np.float64],
        current_world_from_cam0: NDArray[np.float64],
    ) -> InitializedLandmarks:
        """Try to triangulate an observation."""
        feat_num = ready_features.shape[0]
        initialized_landmarks = np.full((feat_num, InitializedLandmarkSchema.count()), np.nan, dtype=np.float64)
        initialized_count = 0

        current_from_world_rot = current_world_from_cam0[:3, :3].T
        current_from_world_t = -current_from_world_rot @ current_world_from_cam0[:3, 3]

        for i in range(feat_num):
            feat_id = int(ready_feat_ids[i])
            feature_history_mask = history_mask[i]
            rows = ready_features[i, feature_history_mask, :]

            left_uv = rows[:, ObservationSchema.LEFT_UV]
            right_uv = rows[:, ObservationSchema.RIGHT_UV]
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

            right_valid_mask = np.all(np.isfinite(right_uv), axis=1)
            right_num = right_uv[right_valid_mask, :].shape[0]

            uvs = np.full((left_uv.shape[0] + right_num, 2), np.nan, dtype=np.float64)
            uvs[: left_uv.shape[0], :] = left_uv
            uvs[left_uv.shape[0] :, :] = right_uv[right_valid_mask, :]

            poses = np.full((left_uv.shape[0] + right_num, 4, 4), np.nan, dtype=np.float64)
            poses[: left_uv.shape[0], :, :] = anchor_from_cam0[: left_uv.shape[0], :, :]
            poses[left_uv.shape[0] :, :, :] = anchor_from_cam0[right_valid_mask, :, :] @ self.rect0_from_rect1

            ray_cast_status, point_in_anchor = self._triangulator.triangulate_feature_observations(uvs, poses)
            if ray_cast_status != TriangulationStatus.SUCCESS:
                continue
            refine_status, point_in_anchor = self._refiner.refine_point_gn(point_in_anchor, uvs, poses)
            if refine_status != LandmarkRefineStatus.SUCCESS:
                continue

            point_in_world = world_from_anchor[:3, :3] @ point_in_anchor + world_from_anchor[:3, 3]

            point_in_current = current_from_world_rot @ point_in_world + current_from_world_t
            initialized_landmarks[initialized_count, InitializedLandmarkSchema.FEAT_ID] = feat_id
            initialized_landmarks[initialized_count, InitializedLandmarkSchema.XYZ] = point_in_current
            initialized_count += 1

        if initialized_count > 0:
            self.logger.trace(f"Initialized {initialized_count} landmarks")

        return initialized_landmarks[:initialized_count]
