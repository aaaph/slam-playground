import numpy as np
import pytest
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature_schema import FeatureLifecycle, FeatureSchema
from core.pose_tracker.local_map import LocalMap
from core.pose_tracker.pnp_pose_tracker import PnpPoseTracker
from core.transformations.special_euclidian_3_dim import SE3


class TestPnpPoseTracker:
    """Unit tests for PnpPoseTracker."""

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

    def test_find_pose_builds_visual_features_only_from_good_ids(self, stereo_ctx: StereoContext, mocker) -> None:
        """Collect visual features only for PnP inliers."""
        pose_tracker = PnpPoseTracker(stereo_ctx)
        local_map = LocalMap.from_capacity(10)
        local_map.add_points(
            {
                10: np.array([1.0, 2.0, 3.0], dtype=np.float64),
                20: np.array([4.0, 5.0, 6.0], dtype=np.float64),
                30: np.array([7.0, 8.0, 9.0], dtype=np.float64),
                40: np.array([10.0, 11.0, 12.0], dtype=np.float64),
            }
        )

        def feature_row(
            feat_id: float, left_u: float, left_v: float, right_u: float, right_v: float
        ) -> NDArray[np.float32]:
            row = np.full(FeatureSchema.count(), np.nan, dtype=np.float32)
            row[FeatureSchema.FEAT_ID] = feat_id
            row[FeatureSchema.LEFT_U] = left_u
            row[FeatureSchema.LEFT_V] = left_v
            row[FeatureSchema.RIGHT_U] = right_u
            row[FeatureSchema.RIGHT_V] = right_v
            row[FeatureSchema.LIFECYCLE] = float(FeatureLifecycle.ACTIVE.value)
            return row

        active_track = np.stack(
            [
                feature_row(10.0, 100.0, 200.0, 90.0, 200.5),
                feature_row(20.0, 110.0, 210.0, 100.0, 210.5),
                feature_row(99.0, 120.0, 220.0, 110.0, 220.5),
                feature_row(30.0, 130.0, 230.0, np.nan, np.nan),
                feature_row(40.0, 140.0, 240.0, 130.0, 240.5),
            ]
        )

        pnp_mock = mocker.patch.object(
            pose_tracker,
            "_resolve_pnp_pose",
            return_value=(SE3.identity(), np.array([10, 30], dtype=np.int32), np.array([20, 40], dtype=np.int32)),
        )
        ba_mock = mocker.patch.object(pose_tracker, "_resolve_ba_correction", return_value=SE3.identity())

        pose_tracker.find_pose(active_track, local_map)

        pnp_args, _ = pnp_mock.call_args
        _, _, valid_feat_ids = pnp_args
        np.testing.assert_array_equal(valid_feat_ids, np.array([10, 20, 30, 40], dtype=np.int32))

        ba_args, _ = ba_mock.call_args
        _, visual_features = ba_args
        expected_visual_features = np.array(
            [
                [10.0, 100.0, 200.0, 90.0, 200.5, 1.0, 2.0, 3.0],
                [30.0, 130.0, 230.0, np.nan, np.nan, 7.0, 8.0, 9.0],
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(visual_features, expected_visual_features, equal_nan=True)
