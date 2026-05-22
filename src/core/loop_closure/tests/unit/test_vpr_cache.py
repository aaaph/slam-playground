import numpy as np
import pytest

from core.loop_closure.vpr_cache import InMemoryFrameCache
from core.loop_closure.vpr_frame import VPRFrame, VPRGeometrySchema


def make_vpr_frame(frame_id: int) -> VPRFrame:
    """Make a VPR frame with deterministic payload."""
    return VPRFrame(
        frame_id=frame_id,
        kf_id=frame_id + 100,
        timestamp=float(frame_id),
        geometry=np.full((2, VPRGeometrySchema.count()), frame_id, dtype=np.float32),
        descriptors=np.full((2, 32), frame_id, dtype=np.uint8),
    )


class TestInMemoryFrameCache:
    """Unit tests for InMemoryFrameCache."""

    @pytest.fixture
    def cache(self) -> InMemoryFrameCache:
        """Create a cache."""
        return InMemoryFrameCache()

    def test_should_be_possible_to_add_and_get_frame(self, cache: InMemoryFrameCache):
        """Test that the InMemoryFrameCache can add and get a frame."""
        frame = make_vpr_frame(frame_id=0)
        cache.add(frame)
        assert cache.get(0) is frame

    def test_cache_index_should_match_frame_id(self, cache: InMemoryFrameCache):
        """Test that the returned cache index always matches frame.frame_id."""
        frames = [make_vpr_frame(frame_id=index) for index in range(3)]

        for frame in frames:
            cache_index = cache.add(frame)

            assert cache_index == frame.frame_id
            assert cache.get(cache_index) is frame

    def test_should_reject_frame_when_frame_id_does_not_match_cache_index(self, cache: InMemoryFrameCache):
        """Test that the cache rejects frames with non-sequential frame ids."""
        cache.add(make_vpr_frame(frame_id=0))
        frame = make_vpr_frame(frame_id=2)

        with pytest.raises(KeyError, match="Frame ID 2 does not match cache ID 1"):
            cache.add(frame)
