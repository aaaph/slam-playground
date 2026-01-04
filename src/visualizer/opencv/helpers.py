import cv2
import numpy as np
from cv2.typing import MatLike

from core.feature_tracker.feature import Feature


def draw_features_on_left(
    left_image: np.ndarray,
    active_features: dict[int, Feature],
    debug_features: list[int] | None = None,
    *,
    put_text: bool = True,
    draw_stereo_baseline: bool = True,
) -> MatLike:
    """Draw the features on the left image."""
    if debug_features is None:
        debug_features = []
    left_out = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    for feat in active_features.values():
        meas = feat.get_active_measurement()
        lx, ly = meas.left
        color = feat.feature_color() if feat.feat_id not in debug_features else feat.debug_color
        size = 2 if feat.feat_id not in debug_features else 12
        if draw_stereo_baseline and meas.right is not None:
            rx, ry = meas.right
            cv2.line(left_out, (int(lx), int(ly)), (int(rx), int(ry)), color, 1)
        cv2.circle(left_out, (int(lx), int(ly)), size, color, -1)
    if put_text:
        cv2.putText(
            left_out,
            f"feat count: {len(active_features)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
    return left_out
