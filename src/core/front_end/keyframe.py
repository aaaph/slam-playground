from dataclasses import dataclass

from core.front_end.keyframe_selector import SelectReason
from core.transformations.special_euclidian_3_dim import SE3


@dataclass
class Keyframe:
    """Front-End keyframe."""

    select_reason: SelectReason
    timestamp: float
    pose: SE3
