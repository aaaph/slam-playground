from unittest.mock import call

import numpy as np

from pipeline.context import PipelineContext
from visualizer.rerun.factories.rerun_config_factory import RerunConfigFactory
from visualizer.rerun.modules.plot_3d_vector_module import Plot3DVectorModule
from visualizer.rerun.schemas import (
    ColorsSchema,
    EntitySchema,
    ModuleType,
    RerunConfigSchema,
    ViewSchema,
    ViewType,
)


class TestPlot3DVectorModule:
    """Unit tests for Plot3DVectorModule."""

    def test_setup_and_process_logs_three_scalar_series(self, mocker) -> None:
        """It should configure and log x/y/z values as separate time-series."""
        log_mock = mocker.patch("visualizer.rerun.modules.plot_3d_vector_module.rr.log")
        set_time_mock = mocker.patch("visualizer.rerun.modules.plot_3d_vector_module.rr.set_time")
        series_lines_mock = mocker.patch(
            "visualizer.rerun.modules.plot_3d_vector_module.rr.SeriesLines",
            side_effect=lambda **kwargs: ("series", kwargs),
        )
        mocker.patch(
            "visualizer.rerun.modules.plot_3d_vector_module.rr.Scalars",
            side_effect=lambda value: ("scalar", value.tolist()),
        )

        module = Plot3DVectorModule(
            "pim_velocity",
            "/world/odom/base_link/pim_velocity",
            {"axis_colors": [[255, 60, 60], [60, 255, 160], [30, 210, 255]]},
        )
        module.setup()

        assert log_mock.call_args_list == [
            call(
                "/world/odom/base_link/pim_velocity",
                (
                    "series",
                    {
                        "widths": [1.5, 1.5, 1.5],
                        "colors": [[255, 60, 60], [60, 255, 160], [30, 210, 255]],
                        "names": ["pim_velocity_x", "pim_velocity_y", "pim_velocity_z"],
                    },
                ),
                static=True,
            )
        ]
        assert series_lines_mock.call_args_list == [
            call(
                widths=[1.5, 1.5, 1.5],
                colors=[[255, 60, 60], [60, 255, 160], [30, 210, 255]],
                names=["pim_velocity_x", "pim_velocity_y", "pim_velocity_z"],
            )
        ]

        log_mock.reset_mock()

        ctx = (
            PipelineContext.from_timestamp(1_000_000_000.0)
            .set_ndarray("pim_velocity", np.array([1.0, 2.0, 3.0], dtype=np.float64))
            .reassemble()
        )
        module.process(ctx)

        assert set_time_mock.call_args_list == [
            call("sim_time", timestamp=1.0),
            call("frame_time", timestamp=1.0),
        ]
        assert log_mock.call_args_list == [
            call("/world/odom/base_link/pim_velocity", ("scalar", [1.0, 2.0, 3.0])),
        ]

    def test_factory_registers_plot_3d_vector_module(self) -> None:
        """The config factory should build plot_3d_vector modules from YAML-like config."""
        config = RerunConfigSchema(
            views=[
                ViewSchema(
                    name="PIM velocity",
                    type=ViewType.TIME_SERIES,
                    origin="/world/odom/base_link/pim_velocity",
                    streams=[
                        EntitySchema(
                            id="pim_velocity",
                            module=ModuleType.PLOT_3D_VECTOR,
                            entity="/world/odom/base_link/pim_velocity",
                        )
                    ],
                )
            ]
        )

        visualizer = RerunConfigFactory.from_config(config)

        assert len(visualizer.modules) == 1
        assert isinstance(visualizer.modules[0], Plot3DVectorModule)

    def test_factory_injects_axis_colors_from_root_config(self) -> None:
        """Root config colors should be passed into plot_3d_vector defaults."""
        config = RerunConfigSchema(
            colors=ColorsSchema(
                x_axis_default=[1, 2, 3],
                y_axis_default=[4, 5, 6],
                z_axis_default=[7, 8, 9],
            ),
            views=[
                ViewSchema(
                    name="PIM velocity",
                    type=ViewType.TIME_SERIES,
                    origin="/world/odom/base_link/pim_velocity",
                    streams=[
                        EntitySchema(
                            id="pim_velocity",
                            module=ModuleType.PLOT_3D_VECTOR,
                            entity="/world/odom/base_link/pim_velocity",
                        )
                    ],
                )
            ],
        )

        visualizer = RerunConfigFactory.from_config(config)
        module = visualizer.modules[0]

        assert isinstance(module, Plot3DVectorModule)
        assert module._series_colors == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]  # noqa: SLF001

    def test_factory_resolves_dot_entity_to_view_origin(self) -> None:
        """A dot entity path should reuse the current view origin."""
        config = RerunConfigSchema(
            views=[
                ViewSchema(
                    name="PIM velocity",
                    type=ViewType.TIME_SERIES,
                    origin="/estimates/velocity/pim",
                    streams=[
                        EntitySchema(
                            id="pim_velocity",
                            module=ModuleType.PLOT_3D_VECTOR,
                            entity=".",
                        )
                    ],
                )
            ]
        )

        visualizer = RerunConfigFactory.from_config(config)
        module = visualizer.modules[0]

        assert isinstance(module, Plot3DVectorModule)
        assert module.entity_path == "/estimates/velocity/pim"
