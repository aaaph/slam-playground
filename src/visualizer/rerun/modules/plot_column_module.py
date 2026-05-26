from typing import Any

import numpy as np
import rerun as rr
from pydantic import BaseModel, Field

from logger import spawn_logger
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class ColumnMapping(BaseModel):
    """Column mapping."""

    index: int
    label: str
    color: list[int] = Field(default_factory=lambda: [255, 0, 0])
    width: float = 1.5


class PlotColumnModuleOptions(BaseModel):
    """Plot column module options."""

    time_idx: str = "imu_ts"
    timeline: str = "sim_time"
    mapping: list[ColumnMapping] = Field(default_factory=list)


class PlotColumnModule(IVizModule):
    """Plot column module."""

    def __init__(self, property_name: str, entity_path: str, raw_options: dict[str, Any]) -> None:
        """Initialize the plot module."""
        self.options = PlotColumnModuleOptions(**raw_options)
        self.property_name = property_name
        self.entity_path = entity_path
        self.logger = spawn_logger(PlotColumnModule.__name__)
        self.time_idx = self.options.time_idx
        self.timeline = self.options.timeline
        self.mapping = self.options.mapping
        self._cache: dict[int, str] = {}

    def setup(self) -> None:
        """Set up the image module."""
        for mapping in self.mapping:
            path = f"{self.entity_path}/{mapping.label}"
            index = mapping.index
            width = mapping.width
            rr.log(path, rr.SeriesLines(colors=mapping.color, names=[mapping.label], widths=[width]), static=True)
            self._cache[index] = path

    def process(self, context: Ctx) -> None:
        """Process the plot data."""
        exists = context.exists(self.time_idx)
        if not exists:
            return
        ts_array = context.get_scalar(self.time_idx, np.ndarray)
        ts_array = np.asarray(ts_array, dtype=np.int64)
        ts_array = ts_array / 1e9
        ts_index = rr.TimeColumn(self.timeline, timestamp=ts_array)
        data_rows = len(ts_array)

        data_columns = len(self.mapping)
        data_scalars = context.get_ndarray(self.property_name, (data_rows, data_columns))

        for index, path in self._cache.items():
            rr.send_columns(
                path,
                indexes=[ts_index],
                columns=rr.Scalars.columns(scalars=data_scalars[:, index]),
            )

    def __repr__(self) -> str:
        """Return the string representation of the plot column module."""
        return f"PlotColumnModule(property_name={self.property_name}, entity_path={self.entity_path})"
