import numpy as np

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature_tracker import FeatureTracker
from core.front_end.front_end_result import FrontendResult
from core.front_end.increment_decorator import increment_counter
from core.front_end.keyframe import Keyframe
from core.front_end.keyframe_selector import KeyframeSelector
from core.pose_tracker.pose_tracker import PoseTracker
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger


class FrontEnd:
    """
    Front-End SLAM.

    The front end is responsible for selecting keyframes and visual odometry.
    1. Rectify the stereo images using the camera model.
    2. Track features in the stereo images using the feature tracker.
    3. Estimate the pose and landmarks using the pose tracker.
    4. Select keyframes using the keyframe selector.
    The keyframes are used to push them into the factor graph
    """

    def __init__(
        self,
        camera_model: StereoCameraModel,
        feature_tracker: FeatureTracker,
        pose_tracker: PoseTracker,
        keyframe_selector: KeyframeSelector,
    ) -> None:
        """Initialize the front end."""
        self.camera_model = camera_model
        self.feature_tracker = feature_tracker
        self.pose_tracker = pose_tracker
        self.keyframe_selector = keyframe_selector
        self.stereo_ctx = camera_model.as_stereo_ctx()
        self.iteration_id = 0
        self.logger = spawn_logger(app="slam_front_end")

    @classmethod
    def default_factory(cls, camera_model: StereoCameraModel, initial_pose: SE3) -> "FrontEnd":
        """Create a default `FrontEnd` with a new feature tracker, pose tracker and keyframe selector."""
        feature_tracker = FeatureTracker.default_factory(camera_model.as_stereo_ctx())
        pose_tracker = PoseTracker.default_factory(initial_pose, camera_model.as_stereo_ctx())
        keyframe_selector = KeyframeSelector.default_factory()
        return cls(camera_model, feature_tracker, pose_tracker, keyframe_selector)

    @increment_counter(prop_name="iteration_id")
    def feed(self, timestamp: float, stereo: tuple[np.ndarray, np.ndarray]) -> FrontendResult:
        """Feed the stereo images to the front end."""
        left, right = np.array(stereo[0]), np.array(stereo[1])
        left, right = self.camera_model.process_stereo(left, right)

        active_features = self.feature_tracker.feed(timestamp, (left, right))
        active_feature_ids = self.feature_tracker.active_features_ids()

        camera_in_world_se3, new_landmarks = self.pose_tracker.estimate(timestamp, list(active_features.values()))

        good_keyframe, select_reason = self.keyframe_selector.check(camera_in_world_se3, active_feature_ids)
        if good_keyframe:
            self.keyframe_selector.update(timestamp, camera_in_world_se3, active_feature_ids)
            self.logger.info(
                f"Selected keyframe at {timestamp:.0f}, id: {self.iteration_id}, reason: {select_reason}"
            )
        keyframe = Keyframe(select_reason, timestamp, camera_in_world_se3) if good_keyframe else None

        return FrontendResult(
            result_id=self.iteration_id,
            timestamp=timestamp,
            camera_in_world_se3=camera_in_world_se3,
            new_landmarks=new_landmarks,
            active_features=active_features,
            keyframe=keyframe,
        )
