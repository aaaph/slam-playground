from dataclasses import dataclass
from typing import NamedTuple

from core.feature_tracker.feature import Feature


@dataclass
class KeyFrameSelectThresholds(NamedTuple):
    """Keyframe selection thresholds."""

    max_distance: float = 0.1
    max_angle: float = 10.0
    max_time_delta: float = 0.1
    max_feature_count: int = 100


class KeyframeSelector:
    """Keyframe selector."""

    def __init__(self, thresholds: KeyFrameSelectThresholds) -> None:
        """Initialize the keyframe selector."""
        self.thresholds = thresholds

    def select(self, _features: None | list[Feature] = None) -> list[Feature]:
        """Select keyframes."""
        return []
