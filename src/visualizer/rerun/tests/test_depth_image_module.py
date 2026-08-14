from unittest.mock import call

import numpy as np

from pipeline.context import PipelineContext
from visualizer.rerun.modules.depth_image_module import DepthImageModule


class TestDepthImageModule:
    """Unit tests for DepthImageModule."""

    def test_process_logs_metric_depth_image(self, mocker) -> None:
        """Depth images should be logged through Rerun's DepthImage archetype."""
        log_mock = mocker.patch("visualizer.rerun.modules.depth_image_module.rr.log")
        set_time_mock = mocker.patch("visualizer.rerun.modules.depth_image_module.rr.set_time")
        depth_image_mock = mocker.patch("visualizer.rerun.modules.depth_image_module.rr.DepthImage")
        compressed_depth = depth_image_mock.return_value.compress.return_value
        module = DepthImageModule(
            "mapping_depth",
            "/mapping/depth",
            {
                "width_field": "width",
                "height_field": "height",
                "meter": 1.0,
                "compress_level": 1,
            },
        )
        depth = np.array(
            [
                [0.0, 1.5],
                [2.5, 0.0],
            ],
            dtype=np.float32,
        )
        ctx = (
            PipelineContext.from_timestamp(1_000_000_000.0)
            .set_scalar("width", 2)
            .set_scalar("height", 2)
            .set_ndarray("mapping_depth", depth)
            .reassemble()
        )

        module.process(ctx)

        assert set_time_mock.call_args_list == [
            call("sim_time", timestamp=1.0),
            call("frame_time", timestamp=1.0),
        ]
        assert log_mock.call_args.args[0] == "/mapping/depth"
        assert log_mock.call_args.args[1] is compressed_depth
        depth_mm = depth_image_mock.call_args_list[-1].args[0]
        np.testing.assert_array_equal(depth_mm, np.array([[0, 1500], [2500, 0]], dtype=np.uint16))
        assert depth_image_mock.call_args_list[-1].kwargs == {"meter": 1000.0}
        depth_image_mock.return_value.compress.assert_called_once_with(compress_level=1)
