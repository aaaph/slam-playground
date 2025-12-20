import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from core.front_end.keyframe_selector import KeyframeSelector, KeyFrameSelectThresholds
from core.transformations.special_euclidian_3_dim import SE3


class TestKeyframeSelector:
    """Unit test for keyframe selector."""

    @pytest.fixture
    def keyframe_selector(self) -> KeyframeSelector:
        """Create a keyframe selector."""
        return KeyframeSelector(
            thresholds=KeyFrameSelectThresholds(
                min_distance=0.1,
                min_angle=10.0,
                max_time_delta=0.2,
                min_connectivity_ratio=0.5,
            )
        )

    @pytest.fixture
    def feat_ids(self) -> set[int]:
        """Create a set of feature IDs."""
        return set(range(10))

    def test_keyframe_selector_big_distance(self, keyframe_selector: KeyframeSelector, feat_ids: set[int]):
        """Test that the keyframe selector returns True if the distance is too big."""
        prev_pose = SE3(t=np.array([0.0, 0.0, 0.0]))
        next_pose = SE3(t=np.array([0.101, 0.0, 0.0]))

        keyframe_selector.update(0.0, prev_pose, feat_ids)
        good_keyframe, creation_reason = keyframe_selector.check(next_pose, feat_ids)
        assert good_keyframe
        assert creation_reason == "big_distance"

    def test_keyframe_selector_big_angle(self, keyframe_selector: KeyframeSelector, feat_ids: set[int]):
        """Test that the keyframe selector returns True if the angle is too big."""
        prev_pose = SE3(t=np.array([0.0, 0.0, 0.0]))
        next_pose = SE3(t=np.array([0.0, 0.0, 0.0]), r=Rotation.from_euler("xyz", [0.0, 0.0, np.deg2rad(10.01)]))

        keyframe_selector.update(0.0, prev_pose, feat_ids)
        good_keyframe, creation_reason = keyframe_selector.check(next_pose, feat_ids)
        assert good_keyframe
        assert creation_reason == "big_angle"

    def test_keyframe_selector_low_connectivity(self, keyframe_selector: KeyframeSelector, feat_ids: set[int]):
        """Test that the keyframe selector returns True if the connectivity is too low."""
        prev_pose = SE3(t=np.array([0.0, 0.0, 0.0]))
        next_pose = SE3(t=np.array([0.0, 0.0, 0.0]))

        keyframe_selector.update(0.0, prev_pose, feat_ids)
        next_feat_ids = set(range(20, 30))
        good_keyframe, creation_reason = keyframe_selector.check(next_pose, next_feat_ids)
        assert good_keyframe
        assert creation_reason == "low_connectivity"
