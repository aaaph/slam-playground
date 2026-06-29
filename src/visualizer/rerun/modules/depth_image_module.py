from typing import Any

import rerun as rr
from pydantic import BaseModel

from logger import spawn_logger
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class DepthImageModuleOptions(BaseModel):
    """Depth image module options."""

    throw_on_nothing: bool = False
    width_field: str = "width"
    height_field: str = "height"
    meter: float = 1.0


class DepthImageModule(IVizModule):
    """Depth image module."""

    def __init__(self, property_name: str, entity_path: str, raw_options: dict[str, Any]) -> None:
        """Initialize the depth image module."""
        self.options = DepthImageModuleOptions(**raw_options)
        self.property_name = property_name
        self.entity_path = entity_path
        self.throw_on_nothing = self.options.throw_on_nothing
        self.logger = spawn_logger(DepthImageModule.__name__)
        self.width_field = self.options.width_field
        self.height_field = self.options.height_field

    def setup(self) -> None:
        """Set up the depth image module."""

    def process(self, context: Ctx) -> None:
        """Process the metric depth image."""
        exists = context.exists(self.property_name)
        if not exists and self.throw_on_nothing:
            msg = f"Depth image data not found in context: {self.property_name}"
            self.logger.warning(msg)
            raise KeyError(msg)
        if not exists and not self.throw_on_nothing:
            return

        width = context.get_scalar(self.width_field, int)
        height = context.get_scalar(self.height_field, int)
        depth = context.get_ndarray(self.property_name, (height, width))
        rr.set_time("sim_time", timestamp=context.get_scalar("timestamp", float) / 1e9)
        rr.set_time("frame_time", timestamp=context.get_scalar("timestamp", float) / 1e9)
        rr.log(
            self.entity_path,
            rr.DepthImage(depth, meter=self.options.meter),
        )

    def __repr__(self) -> str:
        """Return the string representation of the depth image module."""
        return f"DepthImageModule(property_name={self.property_name}, entity_path={self.entity_path})"
