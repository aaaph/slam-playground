from typing import Protocol, Self

import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext
from core.front_end.observation_store import (
    CompressPolicy,
    ObservationHistories,
    ObservationHistoryMask,
    ObservationSchema,
    ObservationStore,
    ReadyObservationCriteria,
)
from core.front_end.ray_triangulation import RayTriangulation, TriangulationStatus
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger

type TrackingInfo = NDArray[np.float64]
type InitializedLandmarks = NDArray[np.float64]


class _LandmarkTriangulator(Protocol):
    """Internal triangulator contract for landmark initialization."""

    def triangulate_feature_observations(
        self,
        left_uv: NDArray[np.float64],
        right_uv: NDArray[np.float64],
        cam0_poses: NDArray[np.float64],
    ) -> tuple[TriangulationStatus, NDArray[np.float64]]:
        """Triangulate one feature observation history."""


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
        triangulator: _LandmarkTriangulator,
        ready_criteria: ReadyObservationCriteria,
    ) -> None:
        """Initialize the landmark initialization class."""
        self._store = store
        self._triangulator = triangulator
        self.logger = spawn_logger(__name__)

        self._ready_criteria = ready_criteria
        self._point_in_world_by_feat_id: dict[int, NDArray[np.float64]] = {}

    @classmethod
    def default_factory(cls, stereo_ctx: StereoContext) -> Self:
        """Create a default landmark initialization class."""
        return cls(
            ObservationStore.default_factory(compress_policy=CompressPolicy.TOP_DISPLACEMENT),
            RayTriangulation.default_factory(stereo_ctx),
            ReadyObservationCriteria(min_history_size=5, min_pixel_displacement=1.5),
        )

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

        self._store.add_observations(observations)
        self.logger.trace(f"Added {observations.shape[0]} observations to the landmark initialization")

        ready_feat_ids, ready_observations, ready_history_mask = self._ready_observations()
        return self._try_to_triangulate_observation(
            ready_feat_ids, ready_history_mask, ready_observations, pose_matrix
        )

    def remove_lost_features(self, lost_features: NDArray[np.int32]) -> None:
        """Remove lost features from the landmark initialization class."""
        self._store.remove_features(lost_features)
        for feat_id in lost_features:
            self._point_in_world_by_feat_id.pop(int(feat_id), None)
        self.logger.trace(f"Removed {lost_features.shape[0]} lost features from the landmark initialization")

    def _ready_observations(self) -> tuple[NDArray[np.int32], ObservationHistories, ObservationHistoryMask]:
        """Get ready observations from the landmark initialization class."""
        return self._store.get_ready_feature_slice(self._ready_criteria)

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
            point_in_world = self._point_in_world_by_feat_id.get(feat_id)

            if point_in_world is None:
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
                status, point_in_anchor = self._triangulator.triangulate_feature_observations(
                    left_uv, right_uv, anchor_from_cam0
                )
                if status != TriangulationStatus.SUCCESS:
                    continue
                point_in_world = world_from_anchor[:3, :3] @ point_in_anchor + world_from_anchor[:3, 3]
                self._point_in_world_by_feat_id[feat_id] = point_in_world

            point_in_current = current_from_world_rot @ point_in_world + current_from_world_t
            initialized_landmarks[initialized_count, InitializedLandmarkSchema.FEAT_ID] = feat_id
            initialized_landmarks[initialized_count, InitializedLandmarkSchema.XYZ] = point_in_current
            initialized_count += 1

        if initialized_count > 0:
            self.logger.trace(f"Initialized {initialized_count} landmarks")

        return initialized_landmarks[:initialized_count]
