from pathlib import Path

import cv2
import numpy as np
import pytest

from core.camera_model.stereo_camera_ctx import StereoContext
from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature_tracker import FeatureTracker
from core.pose_tracker.feature_triangulation import FeatureTriangulation

from .config_helper import CAMERA_CONFIG_0, CAMERA_CONFIG_1


class TestFeatureTrackerStaticFeed:
    """Test for feature tracker static feed."""

    @pytest.fixture
    def test_data_dir(self) -> Path:
        """Fixture for test data directory."""
        return Path(__file__).resolve().parent / "test_data"

    @pytest.fixture
    def camera_model(self) -> StereoCameraModel:
        """Fixture for camera model."""
        return StereoCameraModel.from_cameras_config(CAMERA_CONFIG_0, CAMERA_CONFIG_1)

    @pytest.fixture
    def stereo_ctx(self, camera_model: StereoCameraModel) -> StereoContext:
        """Fixture for stereo context."""
        return camera_model.as_stereo_ctx()

    @pytest.fixture
    def feature_tracker(self, stereo_ctx: StereoContext) -> FeatureTracker:
        """Fixture for feature tracker."""
        return FeatureTracker.default_factory(stereo_ctx, feat_amount_per_region=20, feat_retrack_threshold=1)

    @pytest.fixture
    def feature_triangulator(self, stereo_ctx: StereoContext) -> FeatureTriangulation:
        """Fixture for feature triangulator."""
        return FeatureTriangulation.from_stereo_camera_ctx(stereo_ctx)

    @pytest.fixture
    def stereo_frame(self, test_data_dir: Path, camera_model: StereoCameraModel) -> tuple[np.ndarray, np.ndarray]:
        """Fixture for stereo frame."""
        testing_image_left = cv2.imread(str(test_data_dir / "testing_image_left.png"))
        testing_image_left = cv2.cvtColor(testing_image_left, cv2.COLOR_BGR2GRAY)
        testing_image_right = cv2.imread(str(test_data_dir / "testing_image_right.png"))
        testing_image_right = cv2.cvtColor(testing_image_right, cv2.COLOR_BGR2GRAY)
        left, right = camera_model.process_stereo(testing_image_left, testing_image_right)
        return left, right

    def test_feature_tracker_static_frame_keeping_features(
        self, stereo_frame: tuple[np.ndarray, np.ndarray], feature_tracker: FeatureTracker
    ):
        """Test that the feature tracker output is successful for a static frame and keeps features."""
        left, right = stereo_frame

        features = feature_tracker.feed(1, (left, right))
        assert features is not None
        feature_size_first = feature_tracker.feat_count()
        feature_tracker.feed(2, (left, right))

        feature_size_second = feature_tracker.feat_count()
        assert feature_size_second == feature_size_first

        feature_tracker.feed(3, (left, right))

        feature_size_third = feature_tracker.feat_count()
        assert feature_size_third == feature_size_second

    def test_feature_tracker_features_keep_in_same_position_but_could_have_noise(
        self, stereo_frame: tuple[np.ndarray, np.ndarray], feature_tracker: FeatureTracker
    ):
        """Test that the features keep in same position with minimal noise."""
        left, right = stereo_frame

        n_frames = 20
        for ts in range(1, n_frames + 1):
            feature_tracker.feed(ts, (left, right))
        feature_id = 1
        feature = feature_tracker.get_feature_by_id(feature_id)
        assert feature is not None

        left_mask = feature.cam_id == 0
        u_values = feature.u[left_mask]
        v_values = feature.v[left_mask]
        u_sigma = np.std(u_values)
        v_sigma = np.std(v_values)
        noise_threshold = 0.1
        assert u_sigma < noise_threshold
        assert v_sigma < noise_threshold

        drift_threshold = 0.1
        drift_u = abs(u_values[0] - u_values[-1])
        assert drift_u < drift_threshold

    def test_feature_tracker_stereo_close_vertical_points(
        self, stereo_frame: tuple[np.ndarray, np.ndarray], feature_tracker: FeatureTracker
    ):
        """Test that the feature tracker do rectification correctly by validating v values of features."""
        left, right = stereo_frame

        feature_tracker.feed(1, (left, right))

        for feature in feature_tracker.iterate_through_features():
            _, left_uv, right_uv = feature.get_active_stereo_pair()
            assert right_uv is not None
            diff = abs(left_uv[1] - right_uv[1])
            assert diff < 1.0

    def test_feature_tracker_stereo_disparity_stability(
        self,
        stereo_frame: tuple[np.ndarray, np.ndarray],
        feature_tracker: FeatureTracker,
        feature_triangulator: FeatureTriangulation,
    ):
        """Test that the features have stable disparity."""
        left, right = stereo_frame

        feature_tracker.feed(1, (left, right))

        good_features = []
        for feature in feature_tracker.iterate_through_features():
            good, _ = feature_triangulator.make_initial_guess_by_stereo_pair(feature)
            if good:
                good_features.append(feature)

        disparity_stability_threshold = 0.9

        good_features_rate = len(good_features) / feature_tracker.feat_count()
        assert good_features_rate > disparity_stability_threshold
