import numpy as np

from core.feature_tracker.feature import Feature, FeatureStatus
from core.front_end.front_end_result import FrontendResult
from core.transformations.special_euclidian_3_dim import SE3


class TestFrontendResult:
    """Unit test for frontend result."""

    def test_lost_features_ndarrays(self):
        """Test that the lost features can be converted to a NDArray."""
        feature = Feature.spawn_from_left_and_right(1, 0, (0, 0), (1, 1))
        feature.apply_left_only(2, (2, 2))
        feature.state = FeatureStatus.LOST
        frontend_result = FrontendResult(
            result_id=1,
            timestamp=0.0,
            camera_in_world_se3=SE3.identity(),
            new_landmarks={},
            active_features={},
            lost_features={1: feature},
            left=np.array([[1.0, 2.0], [3.0, 4.0]]),
            right=np.array([[5.0, 6.0], [7.0, 8.0]]),
        )

        count, lost_features_ndarrays = frontend_result.lost_features_ndarrays()
        assert count == 1
        np.testing.assert_almost_equal(lost_features_ndarrays, np.array([[1, 2.0, 2.0, np.nan, np.nan, 2]]))
