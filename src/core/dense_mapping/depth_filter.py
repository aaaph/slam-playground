from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from core.dense_mapping.stereo_depth_estimator import DepthEstimate


@dataclass(frozen=True, slots=True)
class DepthFilterConfig:
    """Depth filter configuration."""

    min_depth_m: float = 0.4
    max_depth_m: float = 6.0
    min_disparity_px: float = 2.0
    min_confidence: float = 0.0
    median_kernel_size: int = 3
    mask_open_kernel_size: int = 3


@dataclass(frozen=True, slots=True)
class FilteredDepth:
    """Filtered depth image data."""

    disparity: NDArray[np.float32]
    depth_m: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    left_image: NDArray[np.uint8]


class DepthFilter:
    """Depth filter."""

    def __init__(self, config: DepthFilterConfig, *, focal_baseline_m: float | None = None) -> None:
        """Initialize the depth filter."""
        self.config = config
        self.focal_baseline_m = focal_baseline_m

    def apply(self, estimate: DepthEstimate) -> FilteredDepth:
        """Apply mapping-specific depth filtering policy to a stereo depth estimate."""
        disparity = estimate.disparity.astype(np.float32, copy=True)
        if estimate.valid_mask.shape != disparity.shape:
            msg = f"valid_mask shape {estimate.valid_mask.shape} does not match disparity shape {disparity.shape}"
            raise ValueError(msg)

        valid_mask = estimate.valid_mask.copy()
        invalid_disparity = ~np.isfinite(disparity) | (disparity <= self.config.min_disparity_px)
        disparity[invalid_disparity] = 0.0

        if self.config.median_kernel_size > 1:
            disparity = cv2.medianBlur(disparity, self.config.median_kernel_size)
            disparity = disparity.astype(np.float32, copy=False)

        depth = self._depth_from_disparity(disparity, estimate.depth_m)
        valid_mask &= (
            np.isfinite(depth)
            & (disparity > self.config.min_disparity_px)
            & (depth >= self.config.min_depth_m)
            & (depth <= self.config.max_depth_m)
        )
        if estimate.confidence is not None:
            if estimate.confidence.shape != valid_mask.shape:
                msg = (
                    f"confidence shape {estimate.confidence.shape} "
                    f"does not match disparity shape {valid_mask.shape}"
                )
                raise ValueError(msg)
            valid_mask &= np.isfinite(estimate.confidence) & (estimate.confidence > self.config.min_confidence)

        if self.config.mask_open_kernel_size > 1:
            kernel_size = self.config.mask_open_kernel_size
            kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            valid_mask = cv2.morphologyEx(valid_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel).astype(bool)

        filtered_depth = np.zeros_like(depth, dtype=np.float32)
        filtered_depth[valid_mask] = depth[valid_mask]
        filtered_disparity = np.zeros_like(disparity, dtype=np.float32)
        filtered_disparity[valid_mask] = disparity[valid_mask]
        return FilteredDepth(
            disparity=filtered_disparity,
            depth_m=filtered_depth,
            valid_mask=valid_mask,
            left_image=estimate.left_image,
        )

    def _depth_from_disparity(
        self,
        disparity: NDArray[np.float32],
        original_depth: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Return depth consistent with the filtered disparity when calibration is available."""
        if self.focal_baseline_m is None:
            return original_depth.astype(np.float32, copy=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            return (self.focal_baseline_m / disparity).astype(np.float32, copy=False)
