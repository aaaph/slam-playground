from typing import Protocol

from core.loop_closure.vpr_frame import VPRFrame


class VPRFrameCache(Protocol):
    """VPR frame cache."""

    def add(self, frame: VPRFrame) -> int: ...  # noqa: D102
    def get(self, frame_id: int) -> VPRFrame | None: ...  # noqa: D102
    def __len__(self) -> int: ...  # noqa: D105


class InMemoryFrameCache(VPRFrameCache):
    """In-memory cache."""

    def __init__(self) -> None:
        """Initialize the in-memory cache."""
        self.cache: list[VPRFrame] = []

    def add(self, frame: VPRFrame) -> int:
        """Add a frame to the cache."""
        frame_id = frame.frame_id
        if frame_id != len(self.cache):
            msg = f"Frame ID {frame_id} does not match cache ID {len(self.cache)}"
            raise KeyError(msg)
        self.cache.append(frame)
        return frame_id

    def get(self, frame_id: int) -> VPRFrame:
        """Get a frame from the cache."""
        if frame_id < 0 or frame_id >= len(self.cache):
            msg = f"Frame ID {frame_id} not found in cache"
            raise KeyError(msg)
        return self.cache[frame_id]

    def __len__(self) -> int:
        """Get the number of frames in the cache."""
        return len(self.cache)
