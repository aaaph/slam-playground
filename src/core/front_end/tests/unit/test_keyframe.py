import numpy as np

from core.feature_tracker.feature import Feature
from core.front_end.keyframe import Keyframe
from core.front_end.keyframe_selector import SelectReason
from core.transformations.special_euclidian_3_dim import SE3


class TestKeyframe:
    """Unit test for keyframe."""

    def test_as_soa(self):
        """Test that the keyframe can be converted to a SOA dictionary."""
        some_feat = Feature.spawn_from_left_and_right(5, 0, (0, 0), (1, 1))
        some_feat.apply_left_only(1, (2, 2))
        keyframe = Keyframe(
            keyframe_id=1,
            select_reason=SelectReason.LOW_CONNECTIVITY,
            timestamp=0.0,
            pose=SE3.identity(),
            active_features={
                1: Feature.spawn_from_left_and_right(1, 0, (0, 0), (1, 1)),
                2: Feature.spawn_from_left_and_right(2, 0, (0, 0), (1, 1)),
                5: some_feat,
            },
            active_landmarks={
                1: np.array([1.0, 2.0, 3.0]),
                2: np.array([4.0, 5.0, 6.0]),
                5: np.array([7.0, 8.0, 9.0]),
            },
        )
        soa = keyframe.as_soa()
        np.testing.assert_almost_equal(soa["keyframe_id"], np.array([1], dtype=np.int32))
        np.testing.assert_almost_equal(soa["select_reason"], np.array([4], dtype=np.int32))
        np.testing.assert_almost_equal(soa["timestamp"], np.array([0.0], dtype=np.float32))
        pose_matrix = SE3.identity().as_matrix()
        np.testing.assert_almost_equal(soa["pose"], pose_matrix)
        expected_landmarks = np.array([[1, 1.0, 2.0, 3.0], [2, 4.0, 5.0, 6.0], [5, 7.0, 8.0, 9.0]])
        np.testing.assert_almost_equal(soa["active_landmarks"], expected_landmarks)
        extected_active_features = np.array(
            [[1, 0.0, 0.0, 1.0, 1.0, 0], [2, 0.0, 0.0, 1.0, 1.0, 0], [5, 2.0, 2.0, np.nan, np.nan, 1]]
        )
        np.testing.assert_almost_equal(soa["active_features"], extected_active_features)
        np.testing.assert_almost_equal(soa["active_features_count"], np.array([3], dtype=np.int32))
        np.testing.assert_almost_equal(soa["active_landmarks_count"], np.array([3], dtype=np.int32))

    def test_from_soa(self):
        """Test that the keyframe can be created from a SOA dictionary."""
        keyframe = Keyframe(
            keyframe_id=1,
            select_reason=SelectReason.LOW_CONNECTIVITY,
            timestamp=0.0,
            pose=SE3.identity(),
            active_features={
                1: Feature.spawn_from_left_and_right(1, 0, (0, 0), (1, 1)),
                2: Feature.spawn_from_left_and_right(2, 0, (0, 0), (1, 1)),
            },
            active_landmarks={1: np.array([1.0, 2.0, 3.0]), 2: np.array([4.0, 5.0, 6.0])},
        )
        soa = keyframe.as_soa()

        new_keyframe = Keyframe.from_soa(soa)
        assert new_keyframe.keyframe_id == keyframe.keyframe_id
        assert new_keyframe.select_reason == keyframe.select_reason
        assert new_keyframe.timestamp == keyframe.timestamp
        assert new_keyframe.pose == keyframe.pose
        assert new_keyframe.active_landmarks.keys() == keyframe.active_landmarks.keys()
        for lm_id in keyframe.active_landmarks:
            np.testing.assert_allclose(new_keyframe.active_landmarks[lm_id], keyframe.active_landmarks[lm_id])

        assert new_keyframe.active_features.keys() == keyframe.active_features.keys()
        for feat_id in keyframe.active_features:
            _, new_left, new_right = new_keyframe.active_features[feat_id].get_active_stereo_pair()
            _, old_left, old_right = keyframe.active_features[feat_id].get_active_stereo_pair()
            np.testing.assert_allclose(new_left, old_left)
            np.testing.assert_allclose(new_right, old_right)
