import numpy as np
import pytest

from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature import FeatureStatus
from core.feature_tracker.feature_tracker import FeatureTracker
from core.transformations.special_euclidian_3_dim import SE3


class TestFeatureTracker:
    """Unit test for feature tracker."""

    @pytest.fixture
    def stereo_ctx(self) -> StereoContext:
        """Create a stereo camera DTO."""
        return StereoContext(
            resolution=(100, 100),
            stereo_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            cam0_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            cam1_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            baseline=1.0,
            cam0_in_body_se3=SE3.identity(),
            cam1_in_body_se3=SE3.identity(),
        )

    @pytest.fixture
    def feature_tracker(self, stereo_ctx: StereoContext) -> FeatureTracker:
        """Create a feature tracker."""
        return FeatureTracker.default_factory(stereo_ctx)

    def test_feature_tracker_get_active_features_ids(self, feature_tracker: FeatureTracker):
        """Test that the feature tracker has a get_active_features_ids method."""
        for i in range(10):
            feature_tracker.tensor.add(i, 1, (0, 0), (1, 1), FeatureStatus.NEW)
        active_features_ids = feature_tracker.active_frame().ids
        assert len(active_features_ids) == 10
        for i in range(10):
            assert i in active_features_ids
