from typing import Any

import rerun as rr
from pydantic import BaseModel

from logger import spawn_logger
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class ImageModuleOptions(BaseModel):
    """Image module options."""

    throw_on_nothing: bool = False
    width_field: str = "width"
    height_field: str = "height"
    channels: int | None = None


class ImageModule(IVizModule):
    """Image module."""

    def __init__(self, property_name: str, entity_path: str, raw_options: dict[str, Any]) -> None:
        """Initialize the image module."""
        self.options = ImageModuleOptions(**raw_options)
        self.property_name = property_name
        self.entity_path = entity_path
        self.throw_on_nothing = self.options.throw_on_nothing
        self.logger = spawn_logger(ImageModule.__name__)
        self.width_field = self.options.width_field
        self.height_field = self.options.height_field
        self.channels = self.options.channels

    def setup(self) -> None:
        """Set up the image module."""

    def process(self, context: Ctx) -> None:
        """Process the image data."""
        exists = context.exists(self.property_name)
        if not exists and self.throw_on_nothing:
            msg = f"Image data not found in context: {self.property_name}"
            self.logger.warning(msg)
            raise KeyError(msg)
        if not exists and not self.throw_on_nothing:
            return
        width = context.get_scalar(self.width_field, int)
        heigth = context.get_scalar(self.height_field, int)
        image_shape = (heigth, width) if self.channels is None else (heigth, width, self.channels)
        image = context.get_image(self.property_name, image_shape)
        rr.set_time("sim_time", timestamp=context.get_scalar("timestamp", float) / 1e9)
        rr.set_time("frame_time", timestamp=context.get_scalar("timestamp", float) / 1e9)
        rr.log(
            self.entity_path,
            rr.Image(image),
        )

    def __repr__(self) -> str:
        """Return the string representation of the image module."""
        return f"ImageModule(property_name={self.property_name}, entity_path={self.entity_path})"
