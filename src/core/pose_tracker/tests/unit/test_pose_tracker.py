from collections.abc import Iterator

import numpy as np
import pytest

from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature import Feature
from core.pose_tracker.feature_triangulation import FeatureTriangulation
from core.pose_tracker.local_map import LocalMap
from core.pose_tracker.pose_tracker import PoseTracker
from core.transformations.special_euclidian_3_dim import SE3


class TestPoseTracker:
    """Unit test for pose tracker."""

    @pytest.fixture
    def stereo_ctx(self) -> StereoContext:
        """Create a stereo camera DTO."""
        return StereoContext(
            stereo_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            cam0_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            cam1_k=np.array([[1000, 0, 0], [0, 1000, 0], [0, 0, 1]]),
            baseline=1.0,
            cam0_in_body=SE3.identity(),
            cam1_in_body=SE3.identity(),
        )

    @pytest.fixture
    def pose_tracker(self, mocker, stereo_ctx: StereoContext) -> PoseTracker:
        """Create a pose tracker."""
        local_map_mock = mocker.Mock(spec=LocalMap)
        feat_triangulation_mock = mocker.Mock(spec=FeatureTriangulation)
        return PoseTracker(
            initial_pose=SE3.identity(),
            stereo_ctx=stereo_ctx,
            local_map=local_map_mock,
            feat_triangulation=feat_triangulation_mock,
        )

    @pytest.fixture
    def feature_iterator(self) -> Iterator[Feature]:
        """Create a feature iterator."""
        features = [Feature.spawn_from_left_and_right(i, i, (i, i), (i, i)) for i in range(10)]
        # feat_5_is_mono
        features[5].apply_left_only(6, (6, 6))
        return iter(features)

    @pytest.fixture
    def features(self, feature_iterator: Iterator[Feature]) -> list[Feature]:
        """Create a feature list."""
        return list(feature_iterator)

    def test_pose_tracker_sanity(self, pose_tracker: PoseTracker):
        """Test that the pose tracker can be created."""
        assert pose_tracker is not None
        assert pose_tracker.active_pose is not None
        assert pose_tracker.local_map is not None
        assert pose_tracker.feat_triangulation is not None
        assert pose_tracker.body_in_cam0 is not None
        assert pose_tracker.body_in_cam0 == pose_tracker.cam0_in_body.inverse()

    def test_pose_tracker_bootstrapping(self, pose_tracker: PoseTracker, features: list[Feature], mocker):
        """Test that the pose tracker can be bootstrapped."""
        mocker.patch.object(pose_tracker.local_map, "empty", return_value=True)
        expected_new_landmarks = {1: np.array([1, 1, 1])}
        mocker.patch.object(pose_tracker, "_landmark_triangulation", return_value=expected_new_landmarks)
        pose, landmarks = pose_tracker.estimate(10.0, features)
        assert pose is not None
        assert pose == pose_tracker.active_pose
        assert landmarks is not None
        assert landmarks == expected_new_landmarks

    def test_pose_tracker_local_map_not_empty(self, pose_tracker: PoseTracker, features: list[Feature], mocker):
        """Test case when local is not empty."""
        mocker.patch.object(pose_tracker.local_map, "empty", return_value=False)
        mocker.patch.object(pose_tracker, "_pnp_pose_prediction", return_value=(SE3.identity(), np.array([1])))
        refined_pose = SE3.from_rpy_xyz(np.array([0, 0, 0]), np.array([1.0, 0, 0]))
        mocker.patch.object(pose_tracker, "_ba_pose_correction", return_value=refined_pose)
        mocker.patch.object(pose_tracker, "_landmark_triangulation", return_value={1: np.array([1, 1, 1])})
        pose, landmarks = pose_tracker.estimate(10.0, features)
        assert pose is not None
        assert pose == refined_pose
        assert pose == pose_tracker.active_pose
        assert landmarks is not None
        assert len(landmarks) == 1

    def test_pose_tracker_pnp_use_local_map(self, pose_tracker: PoseTracker, features: list[Feature], mocker):
        """Test case when pnp pose prediction uses points from local map."""
        expected_pose = SE3.from_rpy_xyz(np.array([0, 0, 0]), np.array([1.0, 0, 0]))
        mocker.patch.object(pose_tracker.local_map, "empty", return_value=False)
        mocker.patch.object(pose_tracker.local_map, "exists", side_effect=lambda x: x in [0, 2, 3])
        mocker.patch.object(pose_tracker.local_map, "get_point", side_effect=lambda x: np.array([x, x, x]))
        mocker.patch.object(pose_tracker, "_ba_pose_correction", return_value=expected_pose)
        mocker.patch.object(pose_tracker, "_landmark_triangulation", return_value={1: np.array([1, 1, 1])})

        expected_inliners = np.array([0])
        pnp_mock_resolve = mocker.patch.object(
            PoseTracker, "_resolve_pnp_pose", return_value=(expected_pose, expected_inliners)
        )

        pose_tracker.estimate(10.0, features)
        args, _ = pnp_mock_resolve.call_args
        obj_points, img_points, feat_ids, _ = args
        np.testing.assert_array_equal(feat_ids, np.array([0, 2, 3]))
        np.testing.assert_array_equal(obj_points, np.array([[0, 0, 0], [2, 2, 2], [3, 3, 3]]))
        np.testing.assert_array_equal(img_points, np.array([[0, 0], [2, 2], [3, 3]]))

    def test_pose_tracker_ba_data_formatting(self, pose_tracker: PoseTracker, features: list[Feature], mocker):
        """Test case when ba data formatting is correct."""
        inliners = np.array([0, 2, 3, 5])
        mocker.patch.object(pose_tracker.local_map, "empty", return_value=False)
        mocker.patch.object(pose_tracker.local_map, "exists", side_effect=lambda x: x in [0, 2, 3])
        mocker.patch.object(pose_tracker.local_map, "get_point", side_effect=lambda x: np.array([x, x, x]))
        mocker.patch.object(pose_tracker, "_landmark_triangulation", return_value={1: np.array([1, 1, 1])})
        mocker.patch.object(pose_tracker, "_pnp_pose_prediction", return_value=(SE3.identity(), inliners))
        ba_mock_resolve = mocker.patch.object(PoseTracker, "_resolve_ba_correction", return_value=SE3.identity())
        pose_tracker.estimate(10.0, features)

        args, _ = ba_mock_resolve.call_args
        pose_k_initial_guess, obj_points, img_points, _, _ = args
        np.testing.assert_array_equal(pose_k_initial_guess, SE3.identity())
        assert len(obj_points) == 4
        assert len(img_points) == 4
        mono_feat = img_points[3]
        assert np.isnan(mono_feat[2])

    def test_pose_tracker_triangulate_new_landmarks(
        self, pose_tracker: PoseTracker, features: list[Feature], mocker
    ):
        """Test case when triangulate new landmarks is correct."""
        mocker.patch.object(pose_tracker.local_map, "empty", return_value=False)
        mocker.patch.object(pose_tracker.local_map, "exists", side_effect=lambda x: x in [0, 2, 3])
        mocker.patch.object(pose_tracker.local_map, "get_point", side_effect=lambda x: np.array([x, x, x]))
        mocker.patch.object(
            pose_tracker, "_pnp_pose_prediction", return_value=(SE3.identity(), np.array([0, 2, 3, 5]))
        )
        mocker.patch.object(pose_tracker, "_ba_pose_correction", return_value=SE3.identity())

        def feat_eight_is_bad_stereo(feat: Feature) -> tuple[bool, np.ndarray]:
            if feat.feat_id == 8:
                return False, np.array([1, 1, 1])
            return True, np.array([1, 1, 1])

        def feat_eight_is_bad_bearing(feat: Feature, pose: SE3) -> tuple[bool, np.ndarray]:
            if feat.feat_id == 8:
                return False, np.array([1, 1, 1])
            return True, np.array([1, 1, 1])

        mocker.patch.object(
            pose_tracker.feat_triangulation,
            "compute_feature_linear_system_update",
            return_value=(np.eye(3), np.array([1, 1, 1])),
        )
        mocker.patch.object(
            pose_tracker.feat_triangulation,
            "make_initial_guess_by_stereo_pair",
            side_effect=feat_eight_is_bad_stereo,
        )
        mocker.patch.object(
            pose_tracker.feat_triangulation,
            "make_linear_triangulation_guess",
            side_effect=feat_eight_is_bad_bearing,
        )

        add_points_resolve = mocker.patch.object(pose_tracker.local_map, "add_points", return_value=None)

        pose_tracker.estimate(10.0, features)

        args, _ = add_points_resolve.call_args
        new_landmarks = args[0]
        assert new_landmarks is not None
        assert new_landmarks.get(8, None) is None
        assert new_landmarks.get(0, None) is None
        assert new_landmarks.get(2, None) is None
        assert new_landmarks.get(3, None) is None
