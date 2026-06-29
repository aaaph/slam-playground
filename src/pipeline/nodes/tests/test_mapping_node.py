from types import SimpleNamespace

import numpy as np

from core.transformations.special_euclidian_3_dim import SE3
from pipeline.context import PipelineContext
from pipeline.nodes.mapping import DepthFilterConfig, MappingNode, VoxelConfig


def make_mapping_node(config: DepthFilterConfig) -> MappingNode:
    node = MappingNode.__new__(MappingNode)
    node.stereo_ctx = SimpleNamespace(
        stereo_k=np.array(
            [
                [100.0, 0.0, 0.0],
                [0.0, 100.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
        baseline=0.1,
    )
    node.depth_filter = config
    node.voxel_config = VoxelConfig(
        voxel_size_m=1.0,
        depth_stride_px=1,
        min_confirmed_hits=3,
        min_confirmed_observations=2,
    )
    node.voxels = {}
    return node


def test_filter_disparity_rejects_invalid_and_out_of_range_depth() -> None:
    node = make_mapping_node(
        DepthFilterConfig(
            min_depth_m=0.25,
            max_depth_m=3.0,
            min_disparity_px=1.0,
            median_kernel_size=1,
            mask_open_kernel_size=1,
        )
    )
    disparity = np.array(
        [
            [0.0, 5.0, 50.0],
            [np.nan, 10.0, 2.0],
        ],
        dtype=np.float32,
    )

    filtered_disparity, depth, valid_mask = node.filter_disparity(disparity)

    expected_mask = np.array(
        [
            [False, True, False],
            [False, True, False],
        ]
    )
    assert np.array_equal(valid_mask, expected_mask)
    assert filtered_disparity[0, 1] == 5.0
    assert filtered_disparity[1, 1] == 10.0
    assert depth[0, 1] == 2.0
    assert depth[1, 1] == 1.0
    assert np.all(depth[~expected_mask] == 0.0)


def test_filter_disparity_removes_isolated_valid_depth_with_mask_opening() -> None:
    node = make_mapping_node(
        DepthFilterConfig(
            min_depth_m=0.25,
            max_depth_m=3.0,
            min_disparity_px=1.0,
            median_kernel_size=1,
            mask_open_kernel_size=3,
        )
    )
    disparity = np.zeros((5, 5), dtype=np.float32)
    disparity[2, 2] = 10.0

    filtered_disparity, depth, valid_mask = node.filter_disparity(disparity)

    assert not np.any(valid_mask)
    assert np.all(filtered_disparity == 0.0)
    assert np.all(depth == 0.0)


def test_depth_to_odom_points_back_projects_valid_pixels() -> None:
    node = make_mapping_node(
        DepthFilterConfig(
            min_depth_m=0.25,
            max_depth_m=3.0,
            min_disparity_px=1.0,
            median_kernel_size=1,
            mask_open_kernel_size=1,
        )
    )
    depth = np.array(
        [
            [1.0, 2.0],
            [0.0, 3.0],
        ],
        dtype=np.float32,
    )
    valid_mask = depth > 0.0

    points = node.depth_to_odom_points(depth, valid_mask, SE3.identity())

    assert np.allclose(
        points,
        np.array(
            [
                [0.0, 0.0, 1.0],
                [0.02, 0.0, 2.0],
                [0.03, 0.03, 3.0],
            ],
            dtype=np.float32,
        ),
    )


def test_integrate_points_aggregates_points_by_voxel_key() -> None:
    node = make_mapping_node(
        DepthFilterConfig(
            min_depth_m=0.25,
            max_depth_m=3.0,
            min_disparity_px=1.0,
            median_kernel_size=1,
            mask_open_kernel_size=1,
        )
    )
    points = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.9, 0.2, 0.3],
            [1.2, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    updated = node.integrate_points(points, timestamp_ns=123.0)

    assert updated == 2
    assert set(node.voxels) == {(0, 0, 0), (1, 0, 0)}
    assert node.voxels[(0, 0, 0)].hits == 2
    assert node.voxels[(0, 0, 0)].observations == 1
    assert np.allclose(node.voxels[(0, 0, 0)].centroid, np.array([0.5, 0.2, 0.3], dtype=np.float32))
    assert np.allclose(node.voxels[(0, 0, 0)].color_rgb, np.array([155.0, 155.0, 155.0], dtype=np.float32))
    assert np.allclose(node.voxel_center_from_key((0, 0, 0)), np.array([0.5, 0.5, 0.5], dtype=np.float32))
    assert np.allclose(node.voxel_center_from_key((-1, 0, 1)), np.array([-0.5, 0.5, 1.5], dtype=np.float32))
    assert node.voxels[(1, 0, 0)].hits == 1


def test_depth_to_odom_points_with_colors_promotes_grayscale_pixels_to_rgb() -> None:
    node = make_mapping_node(
        DepthFilterConfig(
            min_depth_m=0.25,
            max_depth_m=3.0,
            min_disparity_px=1.0,
            median_kernel_size=1,
            mask_open_kernel_size=1,
        )
    )
    depth = np.array(
        [
            [1.0, 0.0],
            [2.0, 3.0],
        ],
        dtype=np.float32,
    )
    image = np.array(
        [
            [10, 20],
            [30, 40],
        ],
        dtype=np.uint8,
    )
    valid_mask = depth > 0.0

    points, colors = node.depth_to_odom_points_with_colors(depth, valid_mask, image, SE3.identity())

    assert points.shape == (3, 3)
    assert np.array_equal(
        colors,
        np.array(
            [
                [10.0, 10.0, 10.0],
                [30.0, 30.0, 30.0],
                [40.0, 40.0, 40.0],
            ],
            dtype=np.float32,
        ),
    )


def test_confirmed_voxel_count_requires_hits_and_repeated_observations() -> None:
    node = make_mapping_node(
        DepthFilterConfig(
            min_depth_m=0.25,
            max_depth_m=3.0,
            min_disparity_px=1.0,
            median_kernel_size=1,
            mask_open_kernel_size=1,
        )
    )
    points = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.3, 0.2, 0.3],
        ],
        dtype=np.float32,
    )

    node.integrate_points(points, timestamp_ns=123.0)

    assert node.confirmed_voxel_count() == 0

    node.integrate_points(points, timestamp_ns=456.0)

    assert node.confirmed_voxel_count() == 1


def test_append_voxel_outputs_exports_raw_and_confirmed_pointclouds() -> None:
    node = make_mapping_node(
        DepthFilterConfig(
            min_depth_m=0.25,
            max_depth_m=3.0,
            min_disparity_px=1.0,
            median_kernel_size=1,
            mask_open_kernel_size=1,
        )
    )
    node.integrate_points(
        np.array(
            [
                [0.1, 0.2, 0.3],
                [0.2, 0.2, 0.3],
                [1.2, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        timestamp_ns=123.0,
        colors_rgb=np.array(
            [
                [10.0, 20.0, 30.0],
                [30.0, 40.0, 50.0],
                [100.0, 100.0, 100.0],
            ],
            dtype=np.float32,
        ),
    )
    node.integrate_points(
        np.array(
            [
                [0.1, 0.2, 0.3],
                [0.3, 0.2, 0.3],
            ],
            dtype=np.float32,
        ),
        timestamp_ns=456.0,
        colors_rgb=np.array(
            [
                [50.0, 50.0, 50.0],
                [70.0, 70.0, 70.0],
            ],
            dtype=np.float32,
        ),
    )

    ctx = node.append_voxel_outputs(PipelineContext.from_timestamp(456.0)).reassemble()

    assert ctx.get_scalar("mapping_voxels_size", int) == 2
    assert ctx.get_scalar("mapping_confirmed_voxels_size", int) == 1
    raw_voxels = ctx.get_ndarray("mapping_voxels", (2, 8))
    confirmed_voxels = ctx.get_ndarray("mapping_confirmed_voxels", (1, 8))
    assert np.allclose(raw_voxels[:, 0], np.array([0.0, 1.0], dtype=np.float32))
    assert np.allclose(confirmed_voxels[0, 1:4], np.array([0.5, 0.5, 0.5], dtype=np.float32))
    assert confirmed_voxels[0, 4] == 4.0
    assert np.allclose(confirmed_voxels[0, 5:8], np.array([40.0, 45.0, 50.0], dtype=np.float32))


def test_append_depth_output_exports_filtered_metric_depth_image() -> None:
    node = make_mapping_node(
        DepthFilterConfig(
            min_depth_m=0.25,
            max_depth_m=3.0,
            min_disparity_px=1.0,
            median_kernel_size=1,
            mask_open_kernel_size=1,
        )
    )
    depth = np.array(
        [
            [0.0, 1.5],
            [2.5, 0.0],
        ],
        dtype=np.float32,
    )

    ctx = node.append_depth_output(PipelineContext.from_timestamp(123.0), depth).reassemble()

    assert np.array_equal(ctx.get_ndarray("mapping_depth", (2, 2)), depth)
