import numpy as np
import pytest
from numpy.typing import NDArray

from core.front_end.keyframe_selector import KeyframeSelector, KeyFrameSelectThresholds, SelectReason
from core.transformations.special_euclidian_3_dim import SE3


class TestKeyframeSelector:
    """Unit test for keyframe selector."""

    @pytest.fixture
    def keyframe_selector(self) -> KeyframeSelector:
        """Create a keyframe selector."""
        return KeyframeSelector(
            thresholds=KeyFrameSelectThresholds(
                ignore_time_until_sec=0.2,
                max_time_delta_sec=2.0,
                min_connectivity_ratio=0.5,
                min_parallax_pts=15,
            )
        )

    @pytest.fixture
    def feat_ids(self) -> set[int]:
        """Create a set of feature IDs."""
        return set(range(10))

    def test_keyframe_selector_low_connectivity(
        self, keyframe_selector: KeyframeSelector, active_track: NDArray[np.float32]
    ):
        """Test that the keyframe selector returns True if the connectivity is too low."""
        prev_pose = SE3(t=np.array([0.0, 0.0, 0.0]))
        next_pose = SE3(t=np.array([0.0, 0.0, 0.0]))

        keyframe_selector.set_new_keyframe(0.0, prev_pose, active_track)
        next_active_track = active_track.copy()
        next_active_track[:, 0] += 100.0
        good_keyframe, creation_reason, _metrics = keyframe_selector.check(1.0 * 1e9, next_pose, next_active_track)
        assert good_keyframe
        assert SelectReason.LOW_CONNECTIVITY in creation_reason

    def test_keyframe_selector_parallax(
        self, keyframe_selector: KeyframeSelector, active_track: NDArray[np.float32]
    ):
        """Test that the keyframe selector returns True if the parallax is too high."""
        prev_pose = SE3(t=np.array([0.0, 0.0, 0.0]))
        next_pose = SE3(t=np.array([0.0, 0.0, 0.0]))

        keyframe_selector.set_new_keyframe(0.0, prev_pose, active_track)
        next_active_track = active_track.copy()
        next_active_track[:, 1:3] += 200
        good_keyframe, creation_reason, _metrics = keyframe_selector.check(1.0 * 1e9, next_pose, next_active_track)
        assert good_keyframe
        assert SelectReason.PARALLAX in creation_reason

    def test_keyframe_selector_time_elapsed(
        self, keyframe_selector: KeyframeSelector, active_track: NDArray[np.float32]
    ):
        """Test that the keyframe selector returns True if the time elapsed is too high."""
        prev_pose = SE3(t=np.array([0.0, 0.0, 0.0]))
        next_pose = SE3(t=np.array([0.0, 0.0, 0.0]))

        keyframe_selector.set_new_keyframe(0.0, prev_pose, active_track)
        good_keyframe, creation_reason, _metrics = keyframe_selector.check(10.0 * 1e9, next_pose, active_track)
        assert good_keyframe
        assert SelectReason.TIME_ELAPSED in creation_reason

    def test_keyframe_selector_time_ignored(
        self, keyframe_selector: KeyframeSelector, active_track: NDArray[np.float32]
    ):
        """Test that the keyframe selector returns False if the time is ignored."""
        prev_pose = SE3(t=np.array([0.0, 0.0, 0.0]))
        next_pose = SE3(t=np.array([0.0, 0.0, 0.0]))

        keyframe_selector.set_new_keyframe(0.0, prev_pose, active_track)
        good_keyframe, creation_reason, _metrics = keyframe_selector.check(0.001 * 1e9, next_pose, active_track)
        assert not good_keyframe
        assert SelectReason.TIME_IGNORED in creation_reason
