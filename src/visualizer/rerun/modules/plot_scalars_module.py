from typing import Any, Literal

import rerun as rr
from pydantic import BaseModel, Field

from logger import spawn_logger
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class PlotScalarsModuleOptions(BaseModel):
    """Plot column module options."""

    arrow_type: Literal["RecordBatch"] = "RecordBatch"
    arrow_field: str
    width: float = 1.5
    label: str
    color: list[int] | None = Field(default=None)


class PlotScalarsModule(IVizModule):
    """Plot column module."""

    def __init__(self, property_name: str, entity_path: str, raw_options: dict[str, Any]) -> None:
        """Initialize the plot module."""
        self.options = PlotScalarsModuleOptions(**raw_options)
        self.property_name = property_name
        self.entity_path = entity_path
        self.logger = spawn_logger(PlotScalarsModule.__name__)

    def setup(self) -> None:
        """Set up the image module."""
        rr.log(self.entity_path, rr.SeriesLines(widths=self.options.width, colors=self.options.color), static=True)

    def process(self, context: Ctx) -> None:
        """Process the plot data."""
        timestamp = context.get_scalar("timestamp", float) / 1e9

        match self.options.arrow_type:
            case "RecordBatch":
                record_batch = context.get_record_batch(self.property_name)
                value = record_batch.column(self.options.arrow_field)[0]

        rr.set_time("sim_time", timestamp=timestamp)
        rr.set_time("frame_time", timestamp=timestamp)
        rr.log(self.entity_path, rr.Scalars(value))

    def __repr__(self) -> str:
        """Return the string representation of the plot column module."""
        return f"PlotScalarsModule(property_name={self.property_name}, entity_path={self.entity_path})"
