from collections.abc import Callable
from dataclasses import dataclass
from uuid import uuid4

import numpy as np
import rerun.blueprint as rrb
from pydantic import BaseModel, Field

from visualizer.rerun.modules.abc_module import IVizModule
from visualizer.rerun.modules.dynamic_transform_module import DynamicTransformModule
from visualizer.rerun.modules.features_module import FeaturesModule
from visualizer.rerun.modules.image_module import ImageModule
from visualizer.rerun.modules.plot_column_module import PlotColumnModule
from visualizer.rerun.modules.plot_scalars_module import PlotScalarsModule
from visualizer.rerun.modules.pointcloud_module import PointcloudModule
from visualizer.rerun.modules.static_transform_module import StaticTransformModule
from visualizer.rerun.rerun_vizualizer import RerunVizualizer
from visualizer.rerun.schemas import LayoutType, ModuleType, RerunConfigSchema, ViewSchema, ViewType

type ModuleFactory = Callable[[str, str, dict], IVizModule]

VIEW_CLASS_MAP = {
    ViewType.SPATIAL_2D: rrb.Spatial2DView,
    ViewType.SPATIAL_3D: rrb.Spatial3DView,
    ViewType.TIME_SERIES: rrb.TimeSeriesView,
}

MODULE_CLASS_MAP: dict[ModuleType, ModuleFactory] = {
    ModuleType.IMAGE: ImageModule,
    ModuleType.FEATURES: FeaturesModule,
    ModuleType.POINTCLOUD: PointcloudModule,
    ModuleType.PLOT_COLUMN: PlotColumnModule,
    ModuleType.PLOT_SCALAR: PlotScalarsModule,
    ModuleType.DYNAMIC_TRANSFORM: DynamicTransformModule,
    ModuleType.STATIC_TRANSFORM: StaticTransformModule,
}

LAYOUT_CLASS_MAP = {
    LayoutType.VERTICAL: rrb.Vertical,
    LayoutType.HORIZONTAL: rrb.Horizontal,
}


class PlotLegendOptions(BaseModel):
    """Plot legend options."""

    visible: bool = True


class TimeSeriesViewOptions(BaseModel):
    """Time series view options."""

    plot_legend: PlotLegendOptions = Field(default_factory=PlotLegendOptions)


@dataclass(frozen=True)
class BuildResult:
    """Factory node build result."""

    blueprint: rrb.BlueprintPart
    modules: list[IVizModule]


class RerunConfigFactory:
    """Rerun config factory."""

    def __init__(self, config: RerunConfigSchema) -> None:
        """Initialize the rerun config factory."""
        self.config = config
        self.resolution = config.resolution
        self.app_name = config.app_name or f"rerun_{uuid4()}"

    @classmethod
    def from_config(cls, config: RerunConfigSchema) -> RerunVizualizer:
        """Create a rerun config from a visualizer config."""
        return cls(config).create()

    def create(self) -> RerunVizualizer:
        """Create a rerun config from a visualizer config."""
        rerun_vizualizer = RerunVizualizer(app_name=self.app_name)

        for view in self.config.views:
            res = self._build_node(view)
            rerun_vizualizer.add_bluepint_part(res.blueprint)
            for module in res.modules:
                rerun_vizualizer.add_module(module)
        return rerun_vizualizer

    def _build_node(self, view: ViewSchema) -> BuildResult:
        """Build a rerun blueprint part in a recursive manner."""
        if view.type == ViewType.CONTAINER:
            return self._build_container(view)
        return self._build_view(view)

    def _build_container(self, view: ViewSchema) -> BuildResult:
        """Build a container node."""
        container_cls: type[rrb.BlueprintPart] = LAYOUT_CLASS_MAP[view.layout]

        all_modules: list[IVizModule] = []
        all_blueprints: list[rrb.BlueprintPart] = []

        for child in view.views:
            res = self._build_node(child)
            all_modules.extend(res.modules)
            all_blueprints.append(res.blueprint)

        return BuildResult(
            blueprint=container_cls(contents=all_blueprints, name=view.name),  # ty: ignore
            modules=all_modules,
        )

    def _build_view(self, view: ViewSchema) -> BuildResult:
        """Build a view node."""
        view_kwargs = {"name": view.name, "origin": view.origin}
        modules: list[IVizModule] = []
        if view.type == ViewType.SPATIAL_3D:
            view_kwargs["background"] = rrb.Background(color=np.array([0, 0, 0]))
            view_kwargs["line_grid"] = rrb.LineGrid3D(visible=False)
        if view.type == ViewType.TIME_SERIES:
            options = TimeSeriesViewOptions(**view.options)
            view_kwargs["plot_legend"] = rrb.PlotLegend(visible=options.plot_legend.visible)

        for entity in view.streams:
            module_cls = MODULE_CLASS_MAP[entity.module]
            mod = module_cls(entity.id, entity.entity, entity.options)
            modules.append(mod)
            if entity.module == ModuleType.FEATURES and self.resolution:
                view_kwargs["visual_bounds"] = rrb.VisualBounds2D(
                    x_range=[0, self.resolution[0]], y_range=[0, self.resolution[1]]
                )
        view_class = VIEW_CLASS_MAP[view.type]
        return BuildResult(
            blueprint=view_class(**view_kwargs),  # ty: ignore
            modules=modules,
        )
