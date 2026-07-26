import numpy as np

from core.feature_tracker.feature_schema import FeatureLifecycle, FeatureSchema
from core.front_end.feature_manager import FeatureManager
from core.pose_tracker.feature_triangulation import (
    FeatureTriangulation,
    StereoTriangulationSchema,
    StereoTriangulationStatus,
)


class TestFeatureManager:
    """Unit tests for aligned feature manager outputs."""

    def test_should_triangulate_active_track_and_preserve_order(self, mocker) -> None:
        """The triangulated table should keep the same row order as the input frame."""
        triangulator = mocker.Mock(spec=FeatureTriangulation)
        feature_manager = FeatureManager(triangulator)

        active_track = np.full((4, FeatureSchema.count()), np.nan, dtype=np.float32)
        active_track[:, FeatureSchema.FEAT_ID] = [10, 20, 30, 40]
        active_track[:, FeatureSchema.TIMESTAMP] = 1.0
        active_track[:, FeatureSchema.LEFT_U] = [100.0, 200.0, 300.0, 400.0]
        active_track[:, FeatureSchema.LEFT_V] = [101.0, 201.0, 301.0, 401.0]
        active_track[:, FeatureSchema.RIGHT_U] = [90.0, 190.0, 280.0, 390.0]
        active_track[:, FeatureSchema.RIGHT_V] = [101.5, 201.5, 301.5, 401.5]
        active_track[:, FeatureSchema.LIFECYCLE] = [
            FeatureLifecycle.ACTIVE.value,
            FeatureLifecycle.LOST.value,
            FeatureLifecycle.ACTIVE.value,
            FeatureLifecycle.ACTIVE.value,
        ]
        active_track[:, FeatureSchema.AGE] = [2, 5, 8, 13]
        active_track[:, FeatureSchema.STEREO_SCORE] = 0.0
        tracking_mask = np.array([True, False, True, True], dtype=np.bool_)
        tracked_points = np.full((3, StereoTriangulationSchema.count()), np.nan, dtype=np.float32)
        tracked_points[:, StereoTriangulationSchema.FEAT_ID] = [10, 30, 40]
        tracked_points[:, StereoTriangulationSchema.XYZ] = (
            [1.0, 2.0, 3.0],
            [np.nan, np.nan, np.nan],
            [4.0, 5.0, 6.0],
        )
        tracked_points[:, StereoTriangulationSchema.STATUS] = [
            StereoTriangulationStatus.TRIANGULATED.value,
            StereoTriangulationStatus.BAD_STEREO.value,
            StereoTriangulationStatus.TRIANGULATED.value,
        ]
        tracked_points[:, StereoTriangulationSchema.LEFT_UV] = active_track[
            tracking_mask, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1
        ]
        tracked_points[:, StereoTriangulationSchema.RIGHT_UV] = active_track[
            tracking_mask, FeatureSchema.RIGHT_U : FeatureSchema.RIGHT_V + 1
        ]
        tracked_good_mask = np.array([True, False, True], dtype=np.bool_)
        triangulator.make_initial_guess_by_stereo_batch.return_value = tracked_good_mask, tracked_points

        result_good_mask, result_points = feature_manager.triangulate_active_track(active_track, tracking_mask)

        np.testing.assert_array_equal(result_good_mask, np.array([True, False, False, True]))
        assert result_points.shape == (active_track.shape[0], StereoTriangulationSchema.count())
        np.testing.assert_allclose(result_points[0], tracked_points[0], equal_nan=True)
        np.testing.assert_allclose(result_points[2], tracked_points[1], equal_nan=True)
        np.testing.assert_allclose(result_points[3], tracked_points[2], equal_nan=True)
        assert result_points[1, StereoTriangulationSchema.STATUS] == StereoTriangulationStatus.UNTRACKED.value
        assert np.all(np.isnan(result_points[1, : StereoTriangulationSchema.STATUS]))
        assert np.all(np.isnan(result_points[1, StereoTriangulationSchema.STATUS + 1 :]))
        np.testing.assert_allclose(
            result_points[2, StereoTriangulationSchema.LEFT_UV],
            active_track[2, FeatureSchema.LEFT_U : FeatureSchema.LEFT_V + 1],
        )
        np.testing.assert_allclose(
            result_points[2, StereoTriangulationSchema.RIGHT_UV],
            active_track[2, FeatureSchema.RIGHT_U : FeatureSchema.RIGHT_V + 1],
        )
        assert np.all(np.isnan(result_points[2, StereoTriangulationSchema.XYZ]))
        candidates = triangulator.make_initial_guess_by_stereo_batch.call_args.args[0]
        np.testing.assert_allclose(candidates[:, 0], active_track[tracking_mask, FeatureSchema.FEAT_ID])
        np.testing.assert_allclose(candidates[:, 1], active_track[tracking_mask, FeatureSchema.LEFT_U])
        np.testing.assert_allclose(candidates[:, 2], active_track[tracking_mask, FeatureSchema.LEFT_V])
        np.testing.assert_allclose(candidates[:, 3], active_track[tracking_mask, FeatureSchema.RIGHT_U])
        np.testing.assert_allclose(candidates[:, 4], active_track[tracking_mask, FeatureSchema.RIGHT_V])

    def test_should_return_frame_aligned_empty_result_when_no_features_are_tracked(self, mocker) -> None:
        """No tracked rows should return frame-aligned empty triangulation without calling triangulator."""
        triangulator = mocker.Mock(spec=FeatureTriangulation)
        feature_manager = FeatureManager(triangulator)

        active_track = np.full((2, FeatureSchema.count()), np.nan, dtype=np.float32)
        tracking_mask = np.zeros((2,), dtype=np.bool_)

        result_good_mask, result_points = feature_manager.triangulate_active_track(active_track, tracking_mask)

        np.testing.assert_array_equal(result_good_mask, tracking_mask)
        assert result_points.shape == (active_track.shape[0], StereoTriangulationSchema.count())
        np.testing.assert_array_equal(
            result_points[:, StereoTriangulationSchema.STATUS],
            np.full((active_track.shape[0],), StereoTriangulationStatus.UNTRACKED.value),
        )
        assert np.all(np.isnan(result_points[:, : StereoTriangulationSchema.STATUS]))
        assert np.all(np.isnan(result_points[:, StereoTriangulationSchema.STATUS + 1 :]))
        triangulator.make_initial_guess_by_stereo_batch.assert_not_called()
