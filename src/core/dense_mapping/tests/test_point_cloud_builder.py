import numpy as np

from core.dense_mapping.depth_filter import FilteredDepth
from core.dense_mapping.point_cloud_builder import PointCloudBuilder
from core.transformations.special_euclidian_3_dim import SE3


def test_build_from_depth_keeps_only_stride_aligned_valid_pixels() -> None:
    """PointCloudBuilder should preserve valid pixels only on the sampling grid."""
    builder = PointCloudBuilder.default_factory(np.eye(3, dtype=np.float32), sample_stride=2)
    valid_mask = np.array(
        [
            [True, True, False],
            [True, True, True],
            [False, True, True],
        ],
        dtype=bool,
    )
    depth_estimate = FilteredDepth(
        disparity=np.ones((3, 3), dtype=np.float32),
        depth_m=np.ones((3, 3), dtype=np.float32),
        valid_mask=valid_mask,
        left_image=np.array(
            [
                [10, 11, 12],
                [20, 21, 22],
                [30, 31, 50],
            ],
            dtype=np.uint8,
        ),
    )

    point_cloud = builder.build_from_depth(SE3.identity(), depth_estimate)

    assert np.allclose(
        point_cloud,
        np.array(
            [
                [0.0, 0.0, 1.0, 10.0, 10.0, 10.0],
                [2.0, 2.0, 1.0, 50.0, 50.0, 50.0],
            ],
            dtype=np.float32,
        ),
    )


def test_build_from_depth_backprojects_sampled_pixels_and_promotes_grayscale_colors() -> None:
    """PointCloudBuilder should output odom-frame XYZ points with RGB colors."""
    k_matrix = np.array(
        [
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    builder = PointCloudBuilder.default_factory(k_matrix, sample_stride=2)
    depth = np.full((3, 3), 2.0, dtype=np.float32)
    valid_mask = np.ones((3, 3), dtype=bool)
    left_image = np.array(
        [
            [10, 11, 20],
            [30, 31, 32],
            [40, 41, 50],
        ],
        dtype=np.uint8,
    )
    depth_estimate = FilteredDepth(
        disparity=np.ones((3, 3), dtype=np.float32),
        depth_m=depth,
        valid_mask=valid_mask,
        left_image=left_image,
    )
    cam0_in_odom = SE3(t=np.array([1.0, 10.0, 100.0], dtype=np.float64))

    point_cloud = builder.build_from_depth(cam0_in_odom, depth_estimate)

    expected_points = np.array(
        [
            [0.0, 9.0, 102.0],
            [2.0, 9.0, 102.0],
            [0.0, 11.0, 102.0],
            [2.0, 11.0, 102.0],
        ],
        dtype=np.float32,
    )
    expected_colors = np.array(
        [
            [10.0, 10.0, 10.0],
            [20.0, 20.0, 20.0],
            [40.0, 40.0, 40.0],
            [50.0, 50.0, 50.0],
        ],
        dtype=np.float32,
    )
    assert point_cloud.shape == (4, 6)
    assert point_cloud.dtype == np.float32
    assert np.allclose(point_cloud[:, :3], expected_points)
    assert np.array_equal(point_cloud[:, 3:], expected_colors)


def test_build_from_depth_returns_empty_point_cloud_when_no_sampled_pixels_are_valid() -> None:
    """PointCloudBuilder should preserve the Nx6 point-cloud contract for empty output."""
    builder = PointCloudBuilder.default_factory(np.eye(3, dtype=np.float32), sample_stride=1)
    depth_estimate = FilteredDepth(
        disparity=np.zeros((2, 2), dtype=np.float32),
        depth_m=np.zeros((2, 2), dtype=np.float32),
        valid_mask=np.zeros((2, 2), dtype=bool),
        left_image=np.zeros((2, 2), dtype=np.uint8),
    )

    point_cloud = builder.build_from_depth(SE3.identity(), depth_estimate)

    assert point_cloud.shape == (0, 6)
    assert point_cloud.dtype == np.float32
