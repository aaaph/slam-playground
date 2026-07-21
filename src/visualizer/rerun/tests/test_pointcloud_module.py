import numpy as np
import pytest
import rerun as rr

from pipeline.context import PipelineContext
from visualizer.rerun.modules.pointcloud_module import PointcloudModule


class TestPointcloudModule:
    """Unit tests for PointcloudModule."""

    def test_process_logs_configured_colors_labels_and_radii(self, mocker) -> None:
        """Pointcloud options should control voxel visualization styling."""
        log_mock = mocker.patch("visualizer.rerun.modules.pointcloud_module.rr.log")
        points3d_mock = mocker.patch(
            "visualizer.rerun.modules.pointcloud_module.rr.Points3D",
            side_effect=lambda **kwargs: ("points3d", kwargs),
        )
        module = PointcloudModule(
            "mapping_voxels",
            "/world/estimates/local_map/mapping/voxels/raw",
            {
                "points_size_prop_name": "mapping_voxels_size",
                "default_color": [80, 160, 255],
                "label_prefix": "voxel",
                "radii": 0.025,
            },
        )
        ctx = (
            PipelineContext.from_timestamp(1.0)
            .set_scalar("mapping_voxels_size", 2)
            .set_ndarray(
                "mapping_voxels",
                np.array(
                    [
                        [10.0, 1.0, 2.0, 3.0, 4.0],
                        [11.0, 4.0, 5.0, 6.0, 7.0],
                    ],
                    dtype=np.float32,
                ),
            )
            .reassemble()
        )

        module.process(ctx)

        assert log_mock.call_args.args[0] == "/world/estimates/local_map/mapping/voxels/raw"
        assert log_mock.call_args.args[1] == (
            "points3d",
            points3d_mock.call_args.kwargs,
        )
        kwargs = points3d_mock.call_args.kwargs
        assert np.allclose(kwargs["positions"], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
        assert np.array_equal(kwargs["colors"], np.array([[80, 160, 255], [80, 160, 255]], dtype=np.uint8))
        assert np.array_equal(kwargs["labels"], np.array(["voxel_10", "voxel_11"]))
        assert np.allclose(kwargs["radii"], np.array([0.025, 0.025], dtype=np.float32))

    def test_process_logs_xyz_pointcloud_without_labels(self, mocker) -> None:
        """Pointcloud module should support raw Nx3 position arrays."""
        log_mock = mocker.patch("visualizer.rerun.modules.pointcloud_module.rr.log")
        points3d_mock = mocker.patch(
            "visualizer.rerun.modules.pointcloud_module.rr.Points3D",
            side_effect=lambda **kwargs: ("points3d", kwargs),
        )
        module = PointcloudModule(
            "points_in_odom",
            "/world/estimates/local_map/mapping/depth_points",
            {
                "points_size_prop_name": "points_in_odom_size",
                "position_columns": [0, 1, 2],
                "id_column": None,
                "show_labels": False,
                "default_color": [255, 220, 40],
                "radii": 0.01,
            },
        )
        ctx = (
            PipelineContext.from_timestamp(1.0)
            .set_scalar("points_in_odom_size", 2)
            .set_ndarray(
                "points_in_odom",
                np.array(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                    ],
                    dtype=np.float32,
                ),
            )
            .reassemble()
        )

        module.process(ctx)

        assert log_mock.call_args.args[0] == "/world/estimates/local_map/mapping/depth_points"
        kwargs = points3d_mock.call_args.kwargs
        assert np.allclose(kwargs["positions"], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
        assert np.array_equal(kwargs["colors"], np.array([[255, 220, 40], [255, 220, 40]], dtype=np.uint8))
        assert "labels" not in kwargs
        assert np.allclose(kwargs["radii"], np.array([0.01, 0.01], dtype=np.float32))

    def test_process_logs_covariance_ellipsoids_for_strict_schema(self, mocker) -> None:
        """Pointcloud covariance visualization should use the strict row-major 3x3 schema."""
        log_mock = mocker.patch("visualizer.rerun.modules.pointcloud_module.rr.log")
        mocker.patch(
            "visualizer.rerun.modules.pointcloud_module.rr.Points3D",
            side_effect=lambda **kwargs: ("points3d", kwargs),
        )
        ellipsoids_mock = mocker.patch(
            "visualizer.rerun.modules.pointcloud_module.rr.Ellipsoids3D",
            side_effect=lambda **kwargs: ("ellipsoids3d", kwargs),
        )
        module = PointcloudModule(
            "local_map_points",
            "/world/local_map/points",
            {
                "points_size_prop_name": "local_map_points_size",
                "visualize_covariance": True,
                "covariance_color": [10, 20, 30],
                "show_labels": False,
            },
        )
        ctx = (
            PipelineContext.from_timestamp(1.0)
            .set_scalar("local_map_points_size", 2)
            .set_ndarray(
                "local_map_points",
                np.array(
                    [
                        [7.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0],
                        [8.0, 4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 9.0],
                    ],
                    dtype=np.float32,
                ),
            )
            .reassemble()
        )

        module.process(ctx)

        assert log_mock.call_args_list[1].args[0] == "/world/local_map/points/covariance"
        assert log_mock.call_args_list[1].args[1] == (
            "ellipsoids3d",
            ellipsoids_mock.call_args.kwargs,
        )
        kwargs = ellipsoids_mock.call_args.kwargs
        assert np.allclose(kwargs["centers"], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32))
        assert np.allclose(kwargs["half_sizes"], np.array([[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]], dtype=np.float32))
        assert np.allclose(kwargs["quaternions"], np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]))
        assert np.array_equal(kwargs["colors"], np.array([[10, 20, 30], [10, 20, 30]], dtype=np.uint8))
        assert kwargs["fill_mode"] == rr.components.FillMode.MajorWireframe
        assert kwargs["show_labels"] is False

    def test_process_rejects_covariance_visualization_without_strict_schema(self) -> None:
        """Covariance visualization should fail if the publisher did not prepare the strict schema."""
        module = PointcloudModule(
            "points",
            "/world/points",
            {
                "points_size_prop_name": "points_size",
                "visualize_covariance": True,
            },
        )
        ctx = (
            PipelineContext.from_timestamp(1.0)
            .set_scalar("points_size", 1)
            .set_ndarray("points", np.array([[1.0, 2.0, 3.0, 4.0, 1.0]], dtype=np.float32))
            .reassemble()
        )

        with pytest.raises(ValueError, match="strict schema"):
            module.process(ctx)
