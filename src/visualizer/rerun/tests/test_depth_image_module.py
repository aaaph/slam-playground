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
        depth_image_mock = mocker.patch(
            "visualizer.rerun.modules.depth_image_module.rr.DepthImage",
            side_effect=lambda image, **kwargs: ("depth_image", image.copy(), kwargs),
        )
        module = DepthImageModule(
            "mapping_depth",
            "/mapping/depth",
            {
                "width_field": "width",
                "height_field": "height",
                "meter": 1.0,
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
        logged_depth_image = log_mock.call_args.args[1]
        assert logged_depth_image[0] == "depth_image"
        assert np.array_equal(logged_depth_image[1], depth)
        assert logged_depth_image[2] == {"meter": 1.0}
        assert depth_image_mock.call_args.kwargs == {"meter": 1.0}
