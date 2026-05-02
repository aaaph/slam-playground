from typing import Any

import numpy as np
import rerun as rr
from pydantic import BaseModel, Field

from logger import spawn_logger
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class Plot3DVectorModuleOptions(BaseModel):
    """Plot 3D vector module options."""

    width: float = 1.5
    label: str | None = None
    color: list[int] | None = Field(default=None)
    axis_colors: list[list[int]] | None = Field(default=None)
    throw_on_nothing: bool = False


class Plot3DVectorModule(IVizModule):
    """Plot a 3D vector as three scalar time-series."""

    def __init__(self, property_name: str, entity_path: str, raw_options: dict[str, Any]) -> None:
        """Initialize the plot module."""
        self.options = Plot3DVectorModuleOptions(**raw_options)
        self.property_name = property_name
        self.entity_path = entity_path
        self.logger = spawn_logger(Plot3DVectorModule.__name__)
        self.throw_on_nothing = self.options.throw_on_nothing
        label_root = self.options.label or self.property_name
        self._series_names = [f"{label_root}_{axis}" for axis in ("x", "y", "z")]
        if self.options.axis_colors is not None:
            self._series_colors = [list(color) for color in self.options.axis_colors]
        elif self.options.color is not None:
            self._series_colors = [self.options.color] * len(self._series_names)
        else:
            self._series_colors = None

    def setup(self) -> None:
        """Set up the vector plot module."""
        series_kwargs: dict[str, Any] = {
            "widths": [self.options.width] * len(self._series_names),
            "names": self._series_names,
        }
        if self._series_colors is not None:
            series_kwargs["colors"] = self._series_colors

        rr.log(self.entity_path, rr.SeriesLines(**series_kwargs), static=True)

    def process(self, context: Ctx) -> None:
        """Process the plot data."""
        timestamp = context.get_scalar("timestamp", float) / 1e9
        exists = context.exists(self.property_name)
        if not exists and self.throw_on_nothing:
            msg = f"Property {self.property_name} not found in context"
            self.logger.warning(msg)
            raise KeyError(msg)
        if not exists and not self.throw_on_nothing:
            return
        value = np.asarray(context.get_ndarray(self.property_name, (3,)), dtype=np.float64)

        rr.set_time("sim_time", timestamp=timestamp)
        rr.set_time("frame_time", timestamp=timestamp)
        rr.log(self.entity_path, rr.Scalars(value))

    def __repr__(self) -> str:
        """Return the string representation of the plot column module."""
        return f"Plot3DVectorModule(property_name={self.property_name}, entity_path={self.entity_path})"
