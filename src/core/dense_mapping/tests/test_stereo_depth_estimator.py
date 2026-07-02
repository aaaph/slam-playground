from typing import Any

import numpy as np
import pytest

from core.dense_mapping import stereo_depth_estimator as estimator_module
from core.dense_mapping.stereo_depth_estimator import (
    PostprocessingMode,
    PreprocessingMode,
    StereoDepthEstimator,
    StereoDepthEstimatorConfig,
    StereoSGBMConfig,
)


class FakeCameraModel:
    """Camera model stub with identity rectification."""

    stereo_k = np.array(
        [
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    baseline = 1.0

    def process_stereo_rectify_only(
        self, left_image: np.ndarray, right_image: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return already-rectified test images."""
        return left_image, right_image


class FakeMatcher:
    """Stereo matcher stub returning fixed-point OpenCV disparity."""

    def __init__(self, disparity_raw: np.ndarray) -> None:
        """Initialize the matcher stub."""
        self.disparity_raw = disparity_raw.astype(np.int16)
        self.compute_count = 0

    def compute(self, _left_image: np.ndarray, _right_image: np.ndarray) -> np.ndarray:
        """Return the configured fixed-point disparity."""
        self.compute_count += 1
        return self.disparity_raw.copy()


class FakeWLSFilter:
    """WLS filter stub returning a filtered disparity and confidence map."""

    def __init__(self, filtered_disparity_raw: np.ndarray, confidence: np.ndarray) -> None:
        """Initialize the WLS filter stub."""
        self.filtered_disparity_raw = filtered_disparity_raw.astype(np.int16)
        self.confidence = confidence.astype(np.float32)
        self.filter_count = 0
        self.confidence_count = 0

    def filter(
        self,
        _left_disparity: np.ndarray,
        _left_image: np.ndarray,
        _filtered_disparity: Any,
        _right_disparity: np.ndarray,
    ) -> np.ndarray:
        """Return the configured filtered fixed-point disparity."""
        self.filter_count += 1
        return self.filtered_disparity_raw.copy()

    def getConfidenceMap(self) -> np.ndarray:  # noqa: N802
        """Return the configured confidence map."""
        self.confidence_count += 1
        return self.confidence.copy()


def make_estimator_with_components(
    config: StereoDepthEstimatorConfig,
    left_disparity_raw: np.ndarray,
    right_disparity_raw: np.ndarray | None = None,
    wls_filter: FakeWLSFilter | None = None,
) -> tuple[StereoDepthEstimator, FakeMatcher, FakeMatcher, FakeWLSFilter | None]:
    """Create an estimator and return its fake OpenCV components."""
    estimator = StereoDepthEstimator.__new__(StereoDepthEstimator)
    estimator.camera_model = FakeCameraModel()
    estimator.config = config
    left_matcher = FakeMatcher(left_disparity_raw)
    right_raw = right_disparity_raw if right_disparity_raw is not None else left_disparity_raw
    right_matcher = FakeMatcher(right_raw)
    estimator.left_matcher = left_matcher
    estimator.right_matcher = right_matcher
    estimator.wls_filter = wls_filter
    return estimator, left_matcher, right_matcher, wls_filter


def make_estimator(
    config: StereoDepthEstimatorConfig,
    left_disparity_raw: np.ndarray,
    right_disparity_raw: np.ndarray | None = None,
    wls_filter: FakeWLSFilter | None = None,
) -> StereoDepthEstimator:
    """Create an estimator with fake OpenCV components."""
    estimator, _, _, _ = make_estimator_with_components(
        config,
        left_disparity_raw,
        right_disparity_raw,
        wls_filter,
    )
    return estimator


def make_preprocess_estimator(preprocessing_mode: PreprocessingMode) -> StereoDepthEstimator:
    """Create an estimator with only preprocessing dependencies configured."""
    estimator = StereoDepthEstimator.__new__(StereoDepthEstimator)
    estimator.camera_model = FakeCameraModel()
    estimator.config = StereoDepthEstimatorConfig(
        preprocessing_mode=preprocessing_mode,
        postprocessing_mode=PostprocessingMode.NONE,
        sgbm=StereoSGBMConfig(min_disparity=0),
    )
    estimator.left_matcher = FakeMatcher(np.full((2, 2), 32, dtype=np.int16))
    estimator.right_matcher = None
    estimator.wls_filter = None
    return estimator


class FakeCLAHE:
    """CLAHE stub that records apply calls."""

    def __init__(self, calls: list[str]) -> None:
        """Initialize the CLAHE spy."""
        self.calls = calls

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Record a CLAHE application."""
        self.calls.append("clahe")
        return image


def install_preprocessing_spies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Install cv2 preprocessing spies and return the ordered call log."""
    calls: list[str] = []

    def equalize_hist(image: np.ndarray) -> np.ndarray:
        calls.append("equalization")
        return image

    def create_clahe(**_kwargs: Any) -> FakeCLAHE:
        calls.append("create_clahe")
        return FakeCLAHE(calls)

    def gaussian_blur(image: np.ndarray, _kernel_size: tuple[int, int], _sigma: int) -> np.ndarray:
        calls.append("blur")
        return image

    monkeypatch.setattr(estimator_module.cv2, "equalizeHist", equalize_hist)
    monkeypatch.setattr(estimator_module.cv2, "createCLAHE", create_clahe)
    monkeypatch.setattr(estimator_module.cv2, "GaussianBlur", gaussian_blur)
    return calls


@pytest.mark.parametrize(
    ("preprocessing_mode", "expected_calls"),
    [
        pytest.param(
            PreprocessingMode.EQUALIZATION,
            ["equalization", "equalization"],
            id="equalization",
        ),
        pytest.param(
            PreprocessingMode.CLAHE,
            ["create_clahe", "clahe", "clahe"],
            id="clahe",
        ),
        pytest.param(
            PreprocessingMode.BLUR,
            ["blur", "blur"],
            id="blur",
        ),
    ],
)
def test_preprocessing_mode_calls_configured_strategy(
    monkeypatch: pytest.MonkeyPatch,
    preprocessing_mode: PreprocessingMode,
    expected_calls: list[str],
) -> None:
    """A single preprocessing mode should call only its configured strategy."""
    calls = install_preprocessing_spies(monkeypatch)
    estimator = make_preprocess_estimator(preprocessing_mode)

    estimator.estimate_depth(
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
    )

    assert calls == expected_calls


def test_preprocessing_mode_combination_calls_all_configured_strategies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Combined preprocessing flags should call every configured strategy."""
    calls = install_preprocessing_spies(monkeypatch)
    estimator = make_preprocess_estimator(
        PreprocessingMode.EQUALIZATION | PreprocessingMode.CLAHE | PreprocessingMode.BLUR
    )

    estimator.estimate_depth(
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
    )

    assert calls == [
        "equalization",
        "equalization",
        "create_clahe",
        "clahe",
        "clahe",
        "blur",
        "blur",
    ]


def test_none_postprocessing_valid_mask_keeps_positive_finite_disparity() -> None:
    """Regular SGBM mode should mark positive finite disparity as valid."""
    disparity_raw = np.array(
        [
            [32, 0],
            [16, 64],
        ],
        dtype=np.int16,
    )
    config = StereoDepthEstimatorConfig(
        postprocessing_mode=PostprocessingMode.NONE,
        sgbm=StereoSGBMConfig(min_disparity=0),
    )
    estimator = make_estimator(config, disparity_raw)

    estimate = estimator.estimate_depth(
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
    )

    assert estimate.confidence is None
    assert np.array_equal(
        estimate.valid_mask,
        np.array(
            [
                [True, False],
                [True, True],
            ]
        ),
    )


def test_none_postprocessing_calls_left_matcher_only() -> None:
    """NONE postprocessing should use the left matcher without WLS dependencies."""
    disparity_raw = np.full((2, 2), 32, dtype=np.int16)
    config = StereoDepthEstimatorConfig(
        postprocessing_mode=PostprocessingMode.NONE,
        sgbm=StereoSGBMConfig(min_disparity=0),
    )
    estimator, left_matcher, right_matcher, _ = make_estimator_with_components(config, disparity_raw)

    estimate = estimator.estimate_depth(
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
    )

    assert estimate.confidence is None
    assert left_matcher.compute_count == 1
    assert right_matcher.compute_count == 0


def test_wls_confidence_is_returned_without_filtering_estimator_valid_mask() -> None:
    """WLS confidence should be exposed while map policy remains outside the estimator."""
    raw_disparity = np.full((2, 3), 32, dtype=np.int16)
    confidence = np.array(
        [
            [0.0, 101.0, 100.0],
            [np.nan, 255.0, 99.0],
        ],
        dtype=np.float32,
    )
    config = StereoDepthEstimatorConfig(
        postprocessing_mode=PostprocessingMode.WLS,
        sgbm=StereoSGBMConfig(min_disparity=0),
    )
    estimator = make_estimator(
        config,
        left_disparity_raw=raw_disparity,
        right_disparity_raw=-raw_disparity,
        wls_filter=FakeWLSFilter(raw_disparity, confidence),
    )

    estimate = estimator.estimate_depth(
        np.zeros((2, 3), dtype=np.uint8),
        np.zeros((2, 3), dtype=np.uint8),
    )

    assert estimate.confidence is not None
    assert np.array_equal(estimate.confidence, confidence, equal_nan=True)
    assert np.array_equal(
        estimate.valid_mask,
        np.ones((2, 3), dtype=bool),
    )


def test_wls_postprocessing_calls_matchers_filter_and_confidence_strategy() -> None:
    """WLS postprocessing should use left/right matchers, WLS filter, and confidence map."""
    left_disparity_raw = np.full((2, 2), 32, dtype=np.int16)
    filtered_disparity_raw = np.full((2, 2), 64, dtype=np.int16)
    confidence = np.full((2, 2), 255.0, dtype=np.float32)
    wls_filter = FakeWLSFilter(filtered_disparity_raw, confidence)
    config = StereoDepthEstimatorConfig(
        postprocessing_mode=PostprocessingMode.WLS,
        sgbm=StereoSGBMConfig(min_disparity=0),
    )
    estimator, left_matcher, right_matcher, wls_filter = make_estimator_with_components(
        config,
        left_disparity_raw=left_disparity_raw,
        right_disparity_raw=-left_disparity_raw,
        wls_filter=wls_filter,
    )

    estimate = estimator.estimate_depth(
        np.zeros((2, 2), dtype=np.uint8),
        np.zeros((2, 2), dtype=np.uint8),
    )

    assert wls_filter is not None
    assert left_matcher.compute_count == 1
    assert right_matcher.compute_count == 1
    assert wls_filter.filter_count == 1
    assert wls_filter.confidence_count == 1
    assert np.all(estimate.disparity == 4.0)
    assert estimate.confidence is not None
    assert np.array_equal(estimate.confidence, confidence)
