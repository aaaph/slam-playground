from visualizer.rerun.factories.rerun_config_factory import RerunConfigFactory
from visualizer.rerun.modules.plot_scalars_module import PlotScalarsModule
from visualizer.rerun.schemas import EntitySchema, LayoutType, ModuleType, RerunConfigSchema, ViewSchema, ViewType


class TestBranchRouting:
    """Unit tests for branch assignment in Rerun visualizer config."""

    def test_factory_uses_default_branch_for_unscoped_streams(self) -> None:
        """Streams without a branch should use the config default branch."""
        config = RerunConfigSchema(
            default_branch="sync_branch",
            views=[
                ViewSchema(
                    name="Metric",
                    type=ViewType.TIME_SERIES,
                    origin="/metric",
                    streams=[
                        EntitySchema(
                            id="value",
                            module=ModuleType.PLOT_SCALAR,
                            entity=".",
                            options={"label": "value"},
                        )
                    ],
                )
            ],
        )

        visualizer = RerunConfigFactory.from_config(config)

        assert set(visualizer.modules_by_branch) == {"sync_branch"}
        assert isinstance(visualizer.modules_by_branch["sync_branch"][0], PlotScalarsModule)

    def test_factory_inherits_view_branch_and_allows_stream_override(self) -> None:
        """Streams should inherit the nearest view branch unless they override it."""
        config = RerunConfigSchema(
            views=[
                ViewSchema(
                    name="Mixed Metrics",
                    type=ViewType.TIME_SERIES,
                    branch="frontend_frame",
                    origin="/metrics",
                    streams=[
                        EntitySchema(
                            id="frontend_value",
                            module=ModuleType.PLOT_SCALAR,
                            entity="/metrics/frontend",
                            options={"label": "frontend"},
                        ),
                        EntitySchema(
                            id="fixedlag_value",
                            module=ModuleType.PLOT_SCALAR,
                            entity="/metrics/fixedlag",
                            branch="fixedlag_frame",
                            options={"label": "fixedlag"},
                        ),
                    ],
                )
            ],
        )

        visualizer = RerunConfigFactory.from_config(config)

        assert set(visualizer.modules_by_branch) == {"frontend_frame", "fixedlag_frame"}
        assert len(visualizer.modules_by_branch["frontend_frame"]) == 1
        assert len(visualizer.modules_by_branch["fixedlag_frame"]) == 1

    def test_factory_inherits_container_branch_into_child_views(self) -> None:
        """A container branch should scope all descendant views by default."""
        config = RerunConfigSchema(
            views=[
                ViewSchema(
                    name="Frontend Metrics",
                    type=ViewType.CONTAINER,
                    branch="frontend_frame",
                    layout=LayoutType.VERTICAL,
                    views=[
                        ViewSchema(
                            name="Metric",
                            type=ViewType.TIME_SERIES,
                            origin="/metric",
                            streams=[
                                EntitySchema(
                                    id="value",
                                    module=ModuleType.PLOT_SCALAR,
                                    entity=".",
                                    options={"label": "value"},
                                )
                            ],
                        )
                    ],
                )
            ],
        )

        visualizer = RerunConfigFactory.from_config(config)

        assert set(visualizer.modules_by_branch) == {"frontend_frame"}
