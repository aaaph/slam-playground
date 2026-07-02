from dataclasses import dataclass, field
from enum import Flag, auto

import cv2
import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_model import StereoCameraModel


@dataclass(frozen=True, slots=True)
class DepthEstimate:
    """Depth estimate data class."""

    disparity: NDArray[np.float32]
    depth_m: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    left_image: NDArray[np.uint8]
    confidence: NDArray[np.float32] | None = None


class PreprocessingMode(Flag):
    """Preprocessing mode."""

    RAW = 0
    EQUALIZATION = auto()
    CLAHE = auto()
    BLUR = auto()


class PostprocessingMode(Flag):
    """Postprocessing mode."""

    NONE = 0
    WLS = auto()


@dataclass(frozen=True)
class StereoSGBMConfig:
    """Stereo SGBM configuration."""

    min_disparity: int = 16
    num_disparities: int = 96
    block_size: int = 5
    disp12_max_diff: int = 1
    uniqueness_ratio: int = 15
    speckle_window_size: int = 150
    speckle_range: int = 2
    pre_filter_cap: int = 63
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY


@dataclass(frozen=True, slots=True)
class StereoDepthEstimatorConfig:
    """Stereo depth estimator configuration."""

    preprocessing_mode: PreprocessingMode = PreprocessingMode.RAW
    postprocessing_mode: PostprocessingMode = PostprocessingMode.NONE
    sgbm: StereoSGBMConfig = field(default_factory=StereoSGBMConfig)


class StereoDepthEstimator:
    """Stereo depth estimator."""

    def __init__(self, camera_model: StereoCameraModel, config: StereoDepthEstimatorConfig) -> None:
        """Initialize the stereo depth estimator."""
        self.camera_model = camera_model
        self.config = config
        window_size = self.config.sgbm.block_size
        p1 = 8 * 1 * window_size * window_size
        p2 = 32 * 1 * window_size * window_size
        self.left_matcher = cv2.StereoSGBM.create(
            minDisparity=self.config.sgbm.min_disparity,
            numDisparities=self.config.sgbm.num_disparities,
            blockSize=self.config.sgbm.block_size,
            P1=p1,
            P2=p2,
            disp12MaxDiff=self.config.sgbm.disp12_max_diff,
            uniquenessRatio=self.config.sgbm.uniqueness_ratio,
            speckleWindowSize=self.config.sgbm.speckle_window_size,
            speckleRange=self.config.sgbm.speckle_range,
            preFilterCap=self.config.sgbm.pre_filter_cap,
            mode=self.config.sgbm.mode,
        )
        self.right_matcher = None
        self.wls_filter = None
        if self.config.postprocessing_mode & PostprocessingMode.WLS:
            if not self.supports_wls():
                msg = "WLS postprocessing requires cv2.ximgproc.createRightMatcher and createDisparityWLSFilter"
                raise RuntimeError(msg)
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(self.left_matcher)
            self.wls_filter.setLambda(8000)
            self.wls_filter.setSigmaColor(1.5)

    @staticmethod
    def supports_wls() -> bool:
        """Return whether the installed OpenCV build exposes the WLS disparity filter API."""
        ximgproc = getattr(cv2, "ximgproc", None)
        return (
            ximgproc is not None
            and hasattr(ximgproc, "createRightMatcher")
            and hasattr(ximgproc, "createDisparityWLSFilter")
        )

    @classmethod
    def default_factory(
        cls,
        camera_model: StereoCameraModel,
    ) -> "StereoDepthEstimator":
        """Create a default stereo depth estimator."""
        return cls(camera_model, StereoDepthEstimatorConfig())

    def _preprocess_stereo(
        self, left_image: NDArray[np.uint8], right_image: NDArray[np.uint8]
    ) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
        """Preprocess the stereo images."""
        left_image, right_image = self.camera_model.process_stereo_rectify_only(left_image, right_image)

        if self.config.preprocessing_mode & PreprocessingMode.EQUALIZATION:
            left_image = cv2.equalizeHist(left_image).astype(np.uint8, copy=False)
            right_image = cv2.equalizeHist(right_image).astype(np.uint8, copy=False)
        if self.config.preprocessing_mode & PreprocessingMode.CLAHE:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            left_image = clahe.apply(left_image).astype(np.uint8, copy=False)
            right_image = clahe.apply(right_image).astype(np.uint8, copy=False)
        if self.config.preprocessing_mode & PreprocessingMode.BLUR:
            left_image = cv2.GaussianBlur(left_image, (3, 3), 0).astype(np.uint8, copy=False)
            right_image = cv2.GaussianBlur(right_image, (3, 3), 0).astype(np.uint8, copy=False)
        return left_image, right_image

    def estimate_depth(self, left_image: NDArray[np.uint8], right_image: NDArray[np.uint8]) -> DepthEstimate:
        """Estimate the depth from a stereo image pair."""
        left_rect, right_rect = self._preprocess_stereo(left_image, right_image)

        if self.config.postprocessing_mode == PostprocessingMode.NONE:
            disp = self.left_matcher.compute(left_rect, right_rect)
            disp = disp.astype(np.float32) / 16.0
            confidence = None
        elif self.config.postprocessing_mode & PostprocessingMode.WLS:
            if self.right_matcher is None or self.wls_filter is None:
                msg = "WLS postprocessing requires initialized right matcher and WLS filter"
                raise RuntimeError(msg)
            left_disp = self.left_matcher.compute(left_rect, right_rect)
            right_disp = self.right_matcher.compute(right_rect, left_rect)
            disp = self.wls_filter.filter(left_disp, left_rect, None, right_disp)
            disp = disp.astype(np.float32) / 16.0
            confidence = self.wls_filter.getConfidenceMap().astype(np.float32)
        else:
            msg = f"Invalid postprocessing mode: {self.config.postprocessing_mode}"
            raise ValueError(msg)

        invalid_disparity = ~np.isfinite(disp) | (disp <= self.config.sgbm.min_disparity)
        disp[invalid_disparity] = 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            depth = self.camera_model.stereo_k[0, 0] * self.camera_model.baseline / disp

        valid_mask = np.isfinite(depth) & (disp > self.config.sgbm.min_disparity)
        if confidence is not None and confidence.shape != valid_mask.shape:
            msg = f"Confidence shape {confidence.shape} does not match depth shape {valid_mask.shape}"
            raise ValueError(msg)

        return DepthEstimate(
            disparity=disp,
            depth_m=depth,
            valid_mask=valid_mask,
            left_image=left_rect,
            confidence=confidence,
        )
