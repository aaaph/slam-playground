from typing import Self

import numpy as np
from numpy.typing import NDArray

from core.front_end.observation_store import CompressPolicy, ObservationSchema, ObservationStore
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger

type TrackingInfo = NDArray[np.float64]


class ObservationTrackStatus:
    """Status of the observation of landmarks."""

    UNINITIALIZED = 0
    INITIALIZED = 1


class LandmarkInitialization:
    """Component for collecting tracking information for landmark initialization."""

    def __init__(
        self, store: ObservationStore, history_threshold: int = 5, min_pixel_displacement: float = 1.5
    ) -> None:
        """Initialize the landmark initialization class."""
        self._store = store
        self.logger = spawn_logger(__name__)

        self._history_threshold = history_threshold
        self._min_pixel_displacement = min_pixel_displacement

    @classmethod
    def default_factory(cls) -> Self:
        """Create a default landmark initialization class."""
        return cls(ObservationStore.default_factory(compress_policy=CompressPolicy.TOP_DISPLACEMENT))

    def add_observation(self, tracking_info: TrackingInfo, pose_estimate: SE3) -> None:
        """Add an observation to the landmark initialization class."""
        pose_flat = pose_estimate.as_flat_ndarray()
        observations = np.full((tracking_info.shape[0], ObservationSchema.size()), np.nan, dtype=np.float32)
        observations[:, ObservationSchema.FEAT_ID] = tracking_info[:, StereoTriangulationSchema.FEAT_ID]
        observations[:, ObservationSchema.CAM0_FLAT_POSE] = pose_flat
        observations[:, ObservationSchema.LEFT_U] = tracking_info[:, StereoTriangulationSchema.LEFT_U]
        observations[:, ObservationSchema.LEFT_V] = tracking_info[:, StereoTriangulationSchema.LEFT_V]
        observations[:, ObservationSchema.RIGHT_U] = tracking_info[:, StereoTriangulationSchema.RIGHT_U]
        observations[:, ObservationSchema.RIGHT_V] = tracking_info[:, StereoTriangulationSchema.RIGHT_V]
        observations[:, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT] = 0

        self._store.add_observations(observations)
        self.logger.info(f"Added {observations.shape[0]} observations to the landmark initialization")

        self.logger.info("READY FEATURES:")
        ready_features = self._store.get_feat_by_criteria(self._history_threshold, self._min_pixel_displacement)
        self.logger.info(f"Ready features: {ready_features}")
        feat_22 = self._store.get_feat_history(22)
        self.logger.info(f"Feature 22: {feat_22[:, ObservationSchema.ANCHOR_PIXEL_DISPLACEMENT]}")

    def remove_lost_features(self, lost_features: NDArray[np.int32]) -> None:
        """Remove lost features from the landmark initialization class."""
        self._store.remove_features(lost_features)
        self.logger.info(f"Removed {lost_features.shape[0]} lost features from the landmark initialization")
