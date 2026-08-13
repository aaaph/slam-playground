import numpy as np
import pytest
from numpy.typing import NDArray

from core.front_end.keyframe_selector import KeyframeSelector, KeyFrameSelectThresholds, SelectReason
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema


def make_stereo_frame(active_track: NDArray[np.float32]) -> NDArray[np.float32]:
    """Build a stereo frame from the compact selector fixture."""
    stereo_frame = np.full((active_track.shape[0], StereoTriangulationSchema.count()), np.nan, dtype=np.float32)
    stereo_frame[:, StereoTriangulationSchema.FEAT_ID] = active_track[:, 0]
    stereo_frame[:, StereoTriangulationSchema.TIMESTAMP] = active_track[:, 1]
    stereo_frame[:, StereoTriangulationSchema.LEFT_UV] = active_track[:, 2:4]
    stereo_frame[:, StereoTriangulationSchema.RIGHT_UV] = active_track[:, 4:6]
    return stereo_frame


class TestKeyframeSelector:
    """Unit test for keyframe selector."""

    @pytest.fixture
    def keyframe_selector(self) -> KeyframeSelector:
        """Create a keyframe selector."""
        ks = KeyframeSelector(
            thresholds=KeyFrameSelectThresholds(
                ignore_time_until_sec=0.2,
                max_time_delta_sec=2.0,
                min_connectivity_ratio=0.5,
                min_parallax_pts=15,
            )
        )
        ks.initialize()
        return ks

    @pytest.fixture
    def feat_ids(self) -> set[int]:
        """Create a set of feature IDs."""
        return set(range(10))

    def test_keyframe_selector_low_connectivity(
        self, keyframe_selector: KeyframeSelector, active_track: NDArray[np.float32]
    ):
        """Test that the keyframe selector returns True if the connectivity is too low."""
        stereo_frame = make_stereo_frame(active_track)
        keyframe_selector.set_new_keyframe(0.0, stereo_frame)
        lost_count = stereo_frame.shape[0] // 2 + 1
        good_keyframe, creation_reason, _metrics = keyframe_selector.check(1.0 * 1e9, stereo_frame[lost_count:])
        assert good_keyframe
        assert SelectReason.LOW_CONNECTIVITY in creation_reason

    def test_keyframe_selector_parallax(
        self, keyframe_selector: KeyframeSelector, active_track: NDArray[np.float32]
    ):
        """Test that the keyframe selector returns True if the parallax is too high."""
        stereo_frame = make_stereo_frame(active_track)
        keyframe_selector.set_new_keyframe(0.0, stereo_frame)
        next_stereo_frame = stereo_frame.copy()
        next_stereo_frame[:, StereoTriangulationSchema.LEFT_UV] += 200
        good_keyframe, creation_reason, _metrics = keyframe_selector.check(1.0 * 1e9, next_stereo_frame)
        assert good_keyframe
        assert SelectReason.PARALLAX in creation_reason

    def test_keyframe_selector_time_elapsed(
        self, keyframe_selector: KeyframeSelector, active_track: NDArray[np.float32]
    ):
        """Test that the keyframe selector returns True if the time elapsed is too high."""
        stereo_frame = make_stereo_frame(active_track)
        keyframe_selector.set_new_keyframe(0.0, stereo_frame)
        good_keyframe, creation_reason, _metrics = keyframe_selector.check(10.0 * 1e9, stereo_frame)
        assert good_keyframe
        assert SelectReason.TIME_ELAPSED in creation_reason

    def test_keyframe_selector_time_ignored(
        self, keyframe_selector: KeyframeSelector, active_track: NDArray[np.float32]
    ):
        """Test that the keyframe selector returns False if the time is ignored."""
        stereo_frame = make_stereo_frame(active_track)
        keyframe_selector.set_new_keyframe(0.0, stereo_frame)
        good_keyframe, creation_reason, _metrics = keyframe_selector.check(0.001 * 1e9, stereo_frame)
        assert not good_keyframe
        assert SelectReason.TIME_IGNORED in creation_reason
