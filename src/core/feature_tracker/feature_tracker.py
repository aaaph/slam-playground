from typing import Literal

import cv2
import jax
import jax.numpy as jnp
import numpy as np

from core.feature_tracker.feature import Feature
from core.feature_tracker.feature_tracker_region import FeatureTrackerRegion
from core.feature_tracker.helper import grid_factor


class FeatureTracker:
    """
    Feature tracker.

    1. image preprocessing - equalizeHist, remap, undistort
    2. create regions based on h and w of image
    3. FAST in each region, get MAX_PER_REGION features
    4. spawn Feature instances for each feature - make ORB descriptors
    -----
    5. next frame
    6. use KLT to track features
    7. If feature is lost, try to match via ORB descriptors
    8. if ORB no returns -> skip feature
    9. Try to RANSAC to find inliers
    """

    def __init__(
        self,
        shift_margin: int = 10,
        region_amount: int = 8,
        klt_win_size: tuple[int, int] = (8, 8),
        feat_amount_per_region: int = 100,
        image_shape: tuple[int, int] = (752, 480),
    ) -> None:
        """Initialize the feature tracker."""
        if region_amount % 2 != 0:
            raise ValueError("Region must be a multiple of two")
        self.features: dict[int, Feature] = {}
        self.SHIFT_MARGIN = shift_margin  # how many pixels are cut off from the all sides of the image
        self.REGION_AMOUNT = region_amount
        self.FEAT_PER_REGION = feat_amount_per_region
        self.klt_params = {
            "winSize": klt_win_size,
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01),
        }
        self.IMAGE_SHAPE: dict[Literal["w", "h"], int] = {"w": image_shape[0], "h": image_shape[1]}
        self.grid: list[FeatureTrackerRegion] = self._spawn_grid()

    def _spawn_grid(self) -> list[FeatureTrackerRegion]:
        """Spawn a grid of regions."""
        w, h = self.IMAGE_SHAPE["w"], self.IMAGE_SHAPE["h"]
        rows, cols = grid_factor(self.REGION_AMOUNT)
        shift_mask = (
            jnp.zeros((h, w), dtype=np.uint8)
            .at[self.SHIFT_MARGIN : h - self.SHIFT_MARGIN, self.SHIFT_MARGIN : w - self.SHIFT_MARGIN]
            .set(1)
        )

        # create regions
        rows_per_region = h // rows
        cols_per_region = w // cols
        region_masks: list[FeatureTrackerRegion] = []
        for row_index in range(rows):
            for col_index in range(cols):
                row_start = (row_index) * rows_per_region
                row_end = (row_index + 1) * rows_per_region
                col_start = (col_index) * cols_per_region
                col_end = (col_index + 1) * cols_per_region

                mask = (
                    jnp.zeros((h, w), dtype=np.uint8)
                    .at[row_start:row_end, col_start:col_end]
                    .set(1)
                    .at[shift_mask == 0]
                    .set(0)
                )
                region = FeatureTrackerRegion((row_index, col_index), mask)
                region_masks.append(region)
        # apply shift margin

        return region_masks

    def add_feature(self, feature: Feature) -> None:
        """Add a feature to the feature tracker."""
        if not feature.obs_count() > 0:
            raise ValueError("Feature has no observations, feat_id: ", feature.feat_id)
        self.features[feature.feat_id] = feature

    def feed_first(self, _timestamp: float, _stereo: tuple[jax.Array, jax.Array]) -> None:
        """Feed the first frame."""
        return

    def feed(self) -> None:
        """Feed the second frame."""
        return
