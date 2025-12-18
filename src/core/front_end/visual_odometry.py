import numpy as np

from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature_tracker import FeatureTracker
from core.pose_tracker.pose_tracker import PoseTracker
from core.transformations.special_euclidian_3_dim import SE3


class VisualOdometry:
    """Stereo visual odometry."""

    def __init__(
        self, feature_tracker: FeatureTracker, pose_tracker: PoseTracker, stereo_ctx: StereoContext
    ) -> None:
        """Initialize the stereo visual odometry."""
        self.feature_tracker = feature_tracker
        self.pose_tracker = pose_tracker
        self.stereo_ctx = stereo_ctx

    @classmethod
    def default_factory(cls, initial_pose: SE3, stereo_ctx: StereoContext) -> "VisualOdometry":
        """Create a default `VisualOdometry` with a new feature tracker and pose tracker."""
        feature_tracker = FeatureTracker.default_factory(stereo_ctx)
        pose_tracker = PoseTracker.default_factory(initial_pose, stereo_ctx)
        return cls(feature_tracker, pose_tracker, stereo_ctx)

    def feed(self, timestamp: float, stereo: tuple[np.ndarray, np.ndarray]) -> None:
        """Feed the stereo images to the visual odometry."""
        self.feature_tracker.feed(timestamp, stereo)
        self.pose_tracker.feed(timestamp, stereo)
