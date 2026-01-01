import numpy as np

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature_tracker import FeatureTracker
from core.front_end.front_end_result import FrontendResult
from core.front_end.increment_decorator import increment_counter
from core.front_end.keyframe import Keyframe
from core.front_end.keyframe_selector import KeyframeSelector
from core.pose_tracker.feature_triangulation import FeatureTriangulation
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
        feature_triangulation: FeatureTriangulation,
    ) -> None:
        """Initialize the front end."""
        self.camera_model = camera_model
        self.feature_tracker = feature_tracker
        self.pose_tracker = pose_tracker
        self.keyframe_selector = keyframe_selector
        self.stereo_ctx = camera_model.as_stereo_ctx()
        self.iteration_id = 0
        self.logger = spawn_logger(app="slam_front_end")
        self.feature_triangulation = feature_triangulation

    @classmethod
    def default_factory(cls, camera_model: StereoCameraModel, initial_pose: SE3) -> "FrontEnd":
        """Create a default `FrontEnd` with a new feature tracker, pose tracker and keyframe selector."""
        feature_tracker = FeatureTracker.default_factory(camera_model.as_stereo_ctx())
        pose_tracker = PoseTracker.default_factory(initial_pose, camera_model.as_stereo_ctx())
        keyframe_selector = KeyframeSelector.default_factory()
        feature_triangulation = FeatureTriangulation.from_stereo_camera_ctx(camera_model.as_stereo_ctx())
        return cls(camera_model, feature_tracker, pose_tracker, keyframe_selector, feature_triangulation)

    @increment_counter(prop_name="iteration_id")
    def feed(self, timestamp: float, stereo: tuple[np.ndarray, np.ndarray]) -> FrontendResult:
        """Feed the stereo images to the front end."""
        left, right = np.array(stereo[0]), np.array(stereo[1])
        left, right = self.camera_model.process_stereo(left, right)

        _ = self.feature_tracker.feed(timestamp, (left, right))
        good_features = self.feature_tracker.active_features_dict(states=["stable", "tracked", "new"])
        good_feature_ids = set(good_features.keys())
        good_features_list = self.feature_tracker.active_features_list(
            states=["stable", "tracked", "new", "unstable"]
        )

        camera_in_world_se3, new_landmarks = self.pose_tracker.estimate(timestamp, good_features_list)

        good_keyframe, select_reason = self.keyframe_selector.check(camera_in_world_se3, good_feature_ids)
        if good_keyframe:
            self.keyframe_selector.update(timestamp, camera_in_world_se3, good_feature_ids)
            self.logger.info(
                f"Selected keyframe at {timestamp:.0f}, id: {self.iteration_id}, reason: {select_reason}"
            )

        keyframe = None
        if good_keyframe:
            active_landmarks = {}

            for feat_id in good_feature_ids:
                if feat_id in new_landmarks:
                    active_landmarks[feat_id] = new_landmarks[feat_id]
                else:
                    # triangulate landmark by current values
                    feat = good_features[feat_id]
                    if feat.get_active_measurement().is_left_only():
                        continue
                    good_feature, initial_guess = self.feature_triangulation.make_initial_guess_by_stereo_pair(
                        feat
                    )
                    if not good_feature:
                        continue
                    active_landmarks[feat_id] = camera_in_world_se3 @ initial_guess
            successed_triangulated_ids = set(active_landmarks.keys())
            active_features = {}
            for feat_id in successed_triangulated_ids:
                active_features[feat_id] = good_features[feat_id]
                # msg = f"re-triangulation for {feat_id}, 3D point: {active_landmarks[feat_id]}"
                # self.logger.debug(msg)

            keyframe = Keyframe(
                keyframe_id=self.iteration_id,
                select_reason=select_reason,
                timestamp=timestamp,
                pose=camera_in_world_se3,
                active_features=active_features,
                active_landmarks=active_landmarks,
            )

        return FrontendResult(
            result_id=self.iteration_id,
            timestamp=timestamp,
            camera_in_world_se3=camera_in_world_se3,
            new_landmarks=new_landmarks,
            active_features=self.feature_tracker.active_features_dict(),
            keyframe=keyframe,
            left=left,
            right=right,
        )
