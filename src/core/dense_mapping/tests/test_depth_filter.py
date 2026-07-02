import numpy as np

from core.dense_mapping.depth_filter import DepthFilter, DepthFilterConfig
from core.dense_mapping.stereo_depth_estimator import DepthEstimate


def make_estimate(
    disparity: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
) -> DepthEstimate:
    """Create a depth estimate using a test focal-baseline of 10."""
    disparity = disparity.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = 10.0 / disparity
    if valid_mask is None:
        valid_mask = np.isfinite(depth) & (disparity > 0.0)
    return DepthEstimate(
        disparity=disparity,
        depth_m=depth.astype(np.float32, copy=False),
        valid_mask=valid_mask,
        left_image=np.zeros(disparity.shape, dtype=np.uint8),
        confidence=confidence,
    )


def test_depth_filter_rejects_invalid_and_out_of_range_depth() -> None:
    """DepthFilter should apply mapping range and disparity policy."""
    depth_filter = DepthFilter(
        DepthFilterConfig(
            min_depth_m=0.25,
            max_depth_m=3.0,
            min_disparity_px=1.0,
            median_kernel_size=1,
            mask_open_kernel_size=1,
        ),
        focal_baseline_m=10.0,
    )
    estimate = make_estimate(
        np.array(
            [
                [0.0, 5.0, 50.0],
                [np.nan, 10.0, 2.0],
            ],
            dtype=np.float32,
        )
    )

    filtered = depth_filter.apply(estimate)

    expected_mask = np.array(
        [
            [False, True, False],
            [False, True, False],
        ]
    )
    assert np.array_equal(filtered.valid_mask, expected_mask)
    assert filtered.disparity[0, 1] == 5.0
    assert filtered.disparity[1, 1] == 10.0
    assert filtered.depth_m[0, 1] == 2.0
    assert filtered.depth_m[1, 1] == 1.0
    assert np.all(filtered.depth_m[~expected_mask] == 0.0)


def test_depth_filter_removes_isolated_valid_depth_with_mask_opening() -> None:
    """DepthFilter should apply morphological cleanup to the valid mask."""
    depth_filter = DepthFilter(
        DepthFilterConfig(
            min_depth_m=0.25,
            max_depth_m=3.0,
            min_disparity_px=1.0,
            median_kernel_size=1,
            mask_open_kernel_size=3,
        ),
        focal_baseline_m=10.0,
    )
    disparity = np.zeros((5, 5), dtype=np.float32)
    disparity[2, 2] = 10.0

    filtered = depth_filter.apply(make_estimate(disparity))

    assert not np.any(filtered.valid_mask)
    assert np.all(filtered.disparity == 0.0)
    assert np.all(filtered.depth_m == 0.0)


def test_depth_filter_applies_confidence_threshold_when_available() -> None:
    """DepthFilter should reject otherwise valid low-confidence pixels."""
    confidence = np.array(
        [
            [0.0, 101.0, 100.0],
            [np.nan, 255.0, 99.0],
        ],
        dtype=np.float32,
    )
    depth_filter = DepthFilter(
        DepthFilterConfig(
            min_depth_m=0.0,
            max_depth_m=100.0,
            min_disparity_px=1.0,
            min_confidence=100.0,
            median_kernel_size=1,
            mask_open_kernel_size=1,
        ),
        focal_baseline_m=10.0,
    )
    estimate = make_estimate(
        np.full((2, 3), 32.0, dtype=np.float32),
        confidence=confidence,
    )

    filtered = depth_filter.apply(estimate)

    assert np.array_equal(
        filtered.valid_mask,
        np.array(
            [
                [False, True, False],
                [False, True, False],
            ]
        ),
    )
