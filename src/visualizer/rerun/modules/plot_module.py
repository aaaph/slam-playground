import rerun as rr

from logger import spawn_logger
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class PlotModule(IVizModule):
    """Plot module."""

    def __init__(self, property_name: str, entity_path: str, *, throw_on_nothing: bool = False) -> None:
        """Initialize the plot module."""
        self.property_name = property_name
        self.entity_path = entity_path
        self.throw_on_nothing = throw_on_nothing
        self.logger = spawn_logger(PlotModule.__name__)

    def setup(self) -> None:
        """Set up the image module."""
        rr.log(self.entity_path, rr.SeriesLines(widths=1.4), static=True)

    def process(self, context: Ctx) -> None:
        """Process the plot data."""
        exists = context.exists(self.property_name)
        if not exists and self.throw_on_nothing:
            msg = f"Data not found in context: {self.property_name}"
            self.logger.warning(msg)
            raise KeyError(msg)
        if not exists and not self.throw_on_nothing:
            return
        ts = context.get_scalar("timestamp", float) / 1e9
        value = context.get_scalar(self.property_name)
        rr.set_time("sim_time", timestamp=ts)
        rr.set_time("frame_time", timestamp=ts)
        rr.log(self.entity_path, rr.Scalars(value))

    def __repr__(self) -> str:
        """Return the string representation of the image module."""
        return f"Plot(property_name={self.property_name}, entity_path={self.entity_path})"
