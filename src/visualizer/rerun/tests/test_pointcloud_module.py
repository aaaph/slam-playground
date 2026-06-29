import numpy as np

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
