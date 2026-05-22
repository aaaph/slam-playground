import rerun as rr

from visualizer.rerun.factories.rerun_config_factory import RerunConfigFactory
from visualizer.rerun.schemas import RerunConfigSchema, ViewSchema, ViewType


class TestViewCoordinatesSetupLogs:
    """Unit tests for Spatial3D view coordinate setup logs."""

    def test_factory_adds_view_coordinates_setup_log_for_spatial_3d_options(self) -> None:
        """The config factory should create a static setup log for Spatial3D view coordinates."""
        config = RerunConfigSchema(
            views=[
                ViewSchema(
                    name="Camera frame",
                    type=ViewType.SPATIAL_3D,
                    origin="/world/camera",
                    options={"view_coordinates": "RDF"},
                )
            ]
        )

        visualizer = RerunConfigFactory.from_config(config)

        assert len(visualizer.modules) == 0
        assert len(visualizer.setup_logs) == 1
        assert visualizer.setup_logs[0].entity_path == "/world/camera"
        assert repr(visualizer.setup_logs[0].archetype) == repr(rr.ViewCoordinates.RDF)

    def test_factory_resolves_right_hand_view_coordinates_alias(self) -> None:
        """Short coordinate aliases should resolve to right-handed Rerun constants."""
        config = RerunConfigSchema(
            views=[
                ViewSchema(
                    name="X-down frame",
                    type=ViewType.SPATIAL_3D,
                    origin="/world/x_down",
                    options={"view_coordinates": "X_DOWN"},
                )
            ]
        )

        visualizer = RerunConfigFactory.from_config(config)

        assert repr(visualizer.setup_logs[0].archetype) == repr(rr.ViewCoordinates.RIGHT_HAND_X_DOWN)
