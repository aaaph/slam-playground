import rerun as rr

from logger import spawn_logger
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class ImageModule(IVizModule):
    """Image module."""

    def __init__(self, property_name: str, entity_path: str, *, throw_on_nothing: bool = False) -> None:
        """Initialize the image module."""
        self.property_name = property_name
        self.entity_path = entity_path
        self.throw_on_nothing = throw_on_nothing
        self.logger = spawn_logger(ImageModule.__name__)

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
        width = context.get_scalar("width", int)
        heigth = context.get_scalar("height", int)
        image = context.get_image(self.property_name, (heigth, width))
        rr.set_time("sim_time", timestamp=context.get_scalar("timestamp", float) / 1e9)
        rr.set_time("frame_time", timestamp=context.get_scalar("timestamp", float) / 1e9)
        rr.log(
            self.entity_path,
            rr.Image(image),
        )

    def __repr__(self) -> str:
        """Return the string representation of the image module."""
        return f"ImageModule(property_name={self.property_name}, entity_path={self.entity_path})"
