from typing import Self

import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext
from core.front_end.landmark_cache import LandmarkCache, LandmarkCacheSchema, LandmarkCacheStatus
from core.front_end.landmark_triangulation import (
    LandmarkTriangulationFlags,
    LandmarkTriangulator,
    LandmarkTriangulatorProtocol,
    TriangulationStatus,
)
from core.front_end.observation_store import (
    CompressPolicy,
    ObservationSchema,
    ObservationStore,
    ReadyObservationCriteria,
    SelectPolicy,
)
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema, StereoTriangulationStatus
from logger import spawn_logger

type LandmarkObservations = NDArray[np.float64]
type LandmarkFeatureFrame = NDArray[np.float64]
type TriangulateReadyObservationsResult = tuple[NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]


class LandmarkInitializationFrameSchema:
    """Frame-aligned schema that extends stereo triangulation rows with landmark init results."""

    FEAT_ID = StereoTriangulationSchema.FEAT_ID
    TIMESTAMP = StereoTriangulationSchema.TIMESTAMP
    LEFT_U = StereoTriangulationSchema.LEFT_U
    LEFT_V = StereoTriangulationSchema.LEFT_V
    RIGHT_U = StereoTriangulationSchema.RIGHT_U
    RIGHT_V = StereoTriangulationSchema.RIGHT_V
    LIFECYCLE = StereoTriangulationSchema.LIFECYCLE
    AGE = StereoTriangulationSchema.AGE
    STEREO_SCORE = StereoTriangulationSchema.STEREO_SCORE
    FRAME_PIXEL_DISPLACEMENT = StereoTriangulationSchema.FRAME_PIXEL_DISPLACEMENT
    LEFT_BEARING_X = StereoTriangulationSchema.LEFT_BEARING_X
    LEFT_BEARING_Y = StereoTriangulationSchema.LEFT_BEARING_Y
    LEFT_BEARING_Z = StereoTriangulationSchema.LEFT_BEARING_Z
    LEFT_UV = StereoTriangulationSchema.LEFT_UV
    RIGHT_UV = StereoTriangulationSchema.RIGHT_UV
    LEFT_BEARING = StereoTriangulationSchema.LEFT_BEARING
    STEREO_X = StereoTriangulationSchema.STEREO_X
    STEREO_Y = StereoTriangulationSchema.STEREO_Y
    STEREO_Z = StereoTriangulationSchema.STEREO_Z
    STEREO_STATUS = StereoTriangulationSchema.STEREO_STATUS
    STEREO_XYZ = StereoTriangulationSchema.XYZ

    LANDMARK_STATUS = StereoTriangulationSchema.count()
    LANDMARK_X = LANDMARK_STATUS + 1
    LANDMARK_Y = LANDMARK_X + 1
    LANDMARK_Z = LANDMARK_Y + 1
    LANDMARK_SLOT = LANDMARK_Z + 1
    LANDMARK_HISTORY_VERSION = LANDMARK_SLOT + 1
    TRACKED = LANDMARK_HISTORY_VERSION + 1

    LANDMARK_XYZ = slice(LANDMARK_X, LANDMARK_Z + 1)
    TRACKER = StereoTriangulationSchema.TRACKER
    STEREO = slice(0, StereoTriangulationSchema.count())

    @classmethod
    def count(cls) -> int:
        """Return the number of columns in the landmark initialization frame schema."""
        return cls.TRACKED + 1


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
        self._max_tri_per_frame = 6

    @classmethod
    def default_factory(cls, stereo_ctx: StereoContext, capacity: int = 1000) -> Self:
        """Create a default landmark initialization class."""
        return cls(
            ObservationStore.default_factory(
                compress_policy=CompressPolicy.UNIFORM_RECENT,
                k_inv=np.linalg.inv(stereo_ctx.stereo_k),
                select_policy=SelectPolicy.COMPARE_ANCHOR_TO_LATEST,
                capacity=capacity,
                history_size=20,
                compressed_history_size=5,
                ready_criteria=ReadyObservationCriteria(min_history_size=3),
            ),
            LandmarkCache.default_factory(capacity=capacity),
            LandmarkTriangulator.default_factory(stereo_ctx, flags=LandmarkTriangulationFlags.DEFAULT),
        )

    def apply_observation_frame(
        self,
        cam0_in_world: NDArray[np.float64],
        tracking_mask: NDArray[np.bool_],
        stereo_frame: NDArray[np.float32] | NDArray[np.float64],
    ) -> tuple[NDArray[np.bool_], LandmarkFeatureFrame]:
        """Apply observation frame to the landmark initialization class."""
        frame_size = stereo_frame.shape[0]
        # (N, LandmarkInitializationFrameSchema.count())
        landmark_frame = np.full((frame_size, LandmarkInitializationFrameSchema.count()), np.nan, dtype=np.float64)
        landmark_frame[:, LandmarkInitializationFrameSchema.STEREO] = stereo_frame
        landmark_frame[:, LandmarkInitializationFrameSchema.TRACKED] = tracking_mask.astype(np.float64, copy=False)
        if frame_size == 0:
            return np.zeros((frame_size,), dtype=np.bool_), landmark_frame

        frame_feat_ids = landmark_frame[:, LandmarkInitializationFrameSchema.FEAT_ID].astype(np.int32, copy=False)
        slots = self._store._get_feature_slots(frame_feat_ids)  # noqa: SLF001
        landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_SLOT] = slots
        landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_HISTORY_VERSION] = (
            self._cache.get_history_version(slots)
        )

        lost_mask = np.logical_not(tracking_mask)
        self.remove_lost_features(landmark_frame, lost_mask)

        cache_lookup = self._cache.lookup(frame_feat_ids, slots)
        landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_XYZ] = cache_lookup[
            :, LandmarkCacheSchema.XYZ
        ]

        cache_status = cache_lookup[:, LandmarkCacheSchema.STATUS]
        cached_ready_mask = cache_status == LandmarkCacheStatus.READY.value
        landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_STATUS] = cache_status

        observation_mask = (
            (cache_status != LandmarkCacheStatus.COMPLETED.value)
            & (cache_status != LandmarkCacheStatus.FAILED_HARD.value)
            & tracking_mask
        )
        triangulation_candidate_mask = cached_ready_mask & observation_mask
        cache_commit_mask = np.zeros((frame_size,), dtype=np.bool_)
        if np.any(observation_mask):
            observed_rows = np.flatnonzero(observation_mask)
            observations = self._build_observations(cam0_in_world, frame_feat_ids, landmark_frame, observed_rows)
            index_slots = self._store.add_observations(observations)
            history_versions = self._store.get_history_versions_by_slots(index_slots)
            slots[observed_rows] = index_slots
            landmark_frame[observed_rows, LandmarkInitializationFrameSchema.LANDMARK_SLOT] = index_slots
            landmark_frame[observed_rows, LandmarkInitializationFrameSchema.LANDMARK_HISTORY_VERSION] = (
                history_versions
            )

            ready_after_append_mask = self._store.pull_ready_mask(index_slots)
            if np.any(ready_after_append_mask):
                ready_after_append_rows = observed_rows[ready_after_append_mask]
                retry_ready_mask = (
                    cache_status[ready_after_append_rows] != LandmarkCacheStatus.FAILED_SOFT.value
                ) | (
                    history_versions[ready_after_append_mask]
                    >= cache_lookup[ready_after_append_rows, LandmarkCacheSchema.RETRY_AFTER_VERSION]
                )
                triangulation_candidate_mask[ready_after_append_rows[retry_ready_mask]] = True

            accumulating_observation_mask = (
                observation_mask
                & ~triangulation_candidate_mask
                & (
                    (cache_status == LandmarkCacheStatus.EMPTY.value)
                    | (cache_status == LandmarkCacheStatus.OBSERVING.value)
                )
            )
            landmark_frame[accumulating_observation_mask, LandmarkInitializationFrameSchema.LANDMARK_STATUS] = (
                LandmarkCacheStatus.OBSERVING.value
            )
            cache_commit_mask |= accumulating_observation_mask

        if np.any(triangulation_candidate_mask):
            landmark_frame[triangulation_candidate_mask, LandmarkInitializationFrameSchema.LANDMARK_STATUS] = (
                LandmarkCacheStatus.READY.value
            )
            tri_failed_mask, tri_success_mask, tri_xyz = self.triangulate_ready_observations(
                landmark_frame, triangulation_candidate_mask
            )
            if np.any(tri_failed_mask):
                failed_slots = landmark_frame[
                    tri_failed_mask, LandmarkInitializationFrameSchema.LANDMARK_SLOT
                ].astype(np.int32, copy=False)
                failed_history_versions = landmark_frame[
                    tri_failed_mask, LandmarkInitializationFrameSchema.LANDMARK_HISTORY_VERSION
                ].astype(np.int32, copy=False)
                landmark_frame[tri_failed_mask, LandmarkInitializationFrameSchema.LANDMARK_STATUS] = (
                    self._cache.resolve_failed_attempt_statuses(failed_slots, failed_history_versions)
                )
            landmark_frame[tri_success_mask, LandmarkInitializationFrameSchema.LANDMARK_STATUS] = (
                LandmarkCacheStatus.COMPLETED.value
            )
            landmark_frame[tri_success_mask, LandmarkInitializationFrameSchema.LANDMARK_XYZ] = tri_xyz
            cache_commit_mask |= triangulation_candidate_mask

        self._cache.commit(
            landmark_frame[cache_commit_mask, LandmarkInitializationFrameSchema.FEAT_ID],
            landmark_frame[cache_commit_mask, LandmarkInitializationFrameSchema.LANDMARK_SLOT],
            landmark_frame[cache_commit_mask, LandmarkInitializationFrameSchema.LANDMARK_STATUS],
            landmark_frame[cache_commit_mask, LandmarkInitializationFrameSchema.LANDMARK_HISTORY_VERSION],
            landmark_frame[cache_commit_mask, LandmarkInitializationFrameSchema.LANDMARK_XYZ],
        )
        success_mask = (
            landmark_frame[:, LandmarkInitializationFrameSchema.LANDMARK_STATUS]
            == LandmarkCacheStatus.COMPLETED.value
        )

        return success_mask, landmark_frame

    def _build_observations(
        self,
        cam0_in_world: NDArray[np.float64],
        frame_feat_ids: NDArray[np.int32],
        landmark_frame: LandmarkFeatureFrame,
        observed_rows: NDArray[np.int64],
    ) -> LandmarkObservations:
        """Build observation rows from tracked stereo frame rows."""
        observations = np.full((observed_rows.shape[0], ObservationSchema.size()), np.nan, dtype=np.float64)
        observations[:, ObservationSchema.FEAT_ID] = frame_feat_ids[observed_rows]
        observations[:, ObservationSchema.FRAME_ID] = 0.0
        observations[:, ObservationSchema.CAM0_MATRIX] = cam0_in_world.reshape(-1)
        observations[:, ObservationSchema.LEFT_UV] = landmark_frame[
            observed_rows, LandmarkInitializationFrameSchema.LEFT_UV
        ]
        triangulated_stereo_mask = (
            landmark_frame[observed_rows, LandmarkInitializationFrameSchema.STEREO_STATUS]
            == StereoTriangulationStatus.TRIANGULATED.value
        )
        observations[triangulated_stereo_mask, ObservationSchema.RIGHT_UV] = landmark_frame[
            observed_rows[triangulated_stereo_mask], LandmarkInitializationFrameSchema.RIGHT_UV
        ]
        return observations

    def remove_lost_features(self, landmark_frame: LandmarkFeatureFrame, lost_mask: NDArray[np.bool_]) -> None:
        """Remove lost features from the landmark initialization class."""
        lost_feat_ids = landmark_frame[lost_mask, LandmarkInitializationFrameSchema.FEAT_ID].astype(
            np.int32, copy=False
        )
        removed_slots = self._store.remove_features(lost_feat_ids)
        self._cache.clear_slots(removed_slots)
        self.logger.trace(f"Removed {lost_feat_ids.shape[0]} lost features from the landmark initialization")

    def triangulate_ready_observations(
        self, landmark_frame: LandmarkFeatureFrame, ready_mask: NDArray[np.bool_]
    ) -> TriangulateReadyObservationsResult:
        """Triangulate ready landmarks and return frame-aligned result masks plus compact XYZ."""
        failed_mask = np.zeros_like(ready_mask)
        success_mask = np.zeros_like(ready_mask)
        ready_rows = np.flatnonzero(ready_mask)
        triangulation_rows = ready_rows[: self._max_tri_per_frame]
        if triangulation_rows.shape[0] == 0:
            return failed_mask, success_mask, np.empty((0, 3), dtype=np.float64)

        ready_slots = landmark_frame[triangulation_rows, LandmarkInitializationFrameSchema.LANDMARK_SLOT].astype(
            np.int32, copy=False
        )
        ready_observations, history_mask = self._store.get_feature_slice_by_slots(ready_slots)

        xyz = np.empty((ready_slots.shape[0], 3), dtype=np.float64)
        success_count = 0
        for i in range(ready_observations.shape[0]):
            rows = ready_observations[i, history_mask[i], :]
            left_uvs = rows[:, ObservationSchema.LEFT_UV]
            right_uvs = rows[:, ObservationSchema.RIGHT_UV]
            left_poses = rows[:, ObservationSchema.CAM0_MATRIX].reshape(-1, 4, 4)

            tri_status, point_in_world = self._triangulator.triangulate_mixed(left_uvs, right_uvs, left_poses)
            if tri_status == TriangulationStatus.SUCCESS:
                success_mask[triangulation_rows[i]] = True
                xyz[success_count] = point_in_world
                success_count += 1
            else:
                failed_mask[triangulation_rows[i]] = True

        return failed_mask, success_mask, xyz[:success_count]
