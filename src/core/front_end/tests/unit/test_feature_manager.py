import numpy as np

from core.feature_tracker.feature_schema import FeatureSchema
from core.front_end.feature_manager import FeatureManager
from core.pose_tracker.feature_triangulation import FeatureTriangulation


class TestFeatureManager:
    """Unit tests for aligned feature manager outputs."""

    def test_should_triangulate_active_track_and_preserve_order(self, mocker) -> None:
        """The triangulated table should keep the same row order as the input track."""
        triangulator = mocker.Mock(spec=FeatureTriangulation)
        feature_manager = FeatureManager(triangulator)

        active_track = np.full((3, FeatureSchema.count()), np.nan, dtype=np.float32)
        active_track[:, FeatureSchema.FEAT_ID] = [10, 20, 30]
        active_track[:, FeatureSchema.TIMESTAMP] = 1.0
        active_track[:, FeatureSchema.LEFT_U] = [100.0, 200.0, 300.0]
        active_track[:, FeatureSchema.LEFT_V] = [101.0, 201.0, 301.0]
        active_track[:, FeatureSchema.RIGHT_U] = [90.0, np.nan, 280.0]
        active_track[:, FeatureSchema.RIGHT_V] = [101.5, np.nan, 301.5]
        active_track[:, FeatureSchema.LIFECYCLE] = [0, 1, 1]
        active_track[:, FeatureSchema.AGE] = [2, 5, 8]
        active_track[:, FeatureSchema.STEREO_SCORE] = [0.0, 0.0, 0.0]
        aligned_points = np.array(
            [
                [10, 1.0, 2.0, 3.0, 1],
                [20, np.nan, np.nan, np.nan, 0],
                [30, 4.0, 5.0, 6.0, 1],
            ],
            dtype=np.float32,
        )
        good_mask = np.array([True, False, True], dtype=np.bool_)
        triangulator.make_initial_guess_by_stereo_batch.return_value = good_mask, aligned_points

        result_good_mask, result_points = feature_manager.triangulate_active_track(active_track)

        np.testing.assert_array_equal(result_good_mask, good_mask)
        np.testing.assert_allclose(result_points, aligned_points, equal_nan=True)
        candidates = triangulator.make_initial_guess_by_stereo_batch.call_args.args[0]
        np.testing.assert_allclose(candidates[:, 0], active_track[:, 0])
        np.testing.assert_allclose(candidates[:, 1], active_track[:, 2])
        np.testing.assert_allclose(candidates[:, 2], active_track[:, 3])
        np.testing.assert_allclose(candidates[:, 3], active_track[:, 4], equal_nan=True)
        np.testing.assert_allclose(candidates[:, 4], active_track[:, 5], equal_nan=True)

    def test_should_merge_active_track_with_xyz(self, mocker) -> None:
        """Merged track should append XYZ columns while keeping the original track untouched."""
        triangulator = mocker.Mock(spec=FeatureTriangulation)
        feature_manager = FeatureManager(triangulator)

        active_track = np.full((2, FeatureSchema.count()), np.nan, dtype=np.float32)
        active_track[:, FeatureSchema.FEAT_ID] = [10, 20]
        active_track[:, FeatureSchema.TIMESTAMP] = 1.0
        active_track[:, FeatureSchema.LEFT_U] = [100.0, 200.0]
        active_track[:, FeatureSchema.LEFT_V] = [101.0, 201.0]
        active_track[:, FeatureSchema.RIGHT_U] = [90.0, np.nan]
        active_track[:, FeatureSchema.RIGHT_V] = [101.5, np.nan]
        active_track[:, FeatureSchema.LIFECYCLE] = [0, 1]
        active_track[:, FeatureSchema.AGE] = [2, 5]
        active_track[:, FeatureSchema.STEREO_SCORE] = [0.0, 0.0]
        aligned_points = np.array(
            [
                [10, 1.0, 2.0, 3.0, 1],
                [20, np.nan, np.nan, np.nan, 0],
            ],
            dtype=np.float32,
        )
        triangulator.make_initial_guess_by_stereo_batch.return_value = (
            np.array([True, False], dtype=np.bool_),
            aligned_points,
        )

        merged_track = feature_manager.merge_active_track_and_points(active_track)

        assert merged_track.shape == (2, FeatureSchema.count() + 3)
        np.testing.assert_allclose(merged_track[:, : FeatureSchema.count()], active_track, equal_nan=True)
        np.testing.assert_allclose(
            merged_track[:, FeatureSchema.count() : FeatureSchema.count() + 3],
            np.array([[1.0, 2.0, 3.0], [np.nan, np.nan, np.nan]], dtype=np.float32),
            equal_nan=True,
        )
