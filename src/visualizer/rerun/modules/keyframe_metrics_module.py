import numpy as np
import rerun as rr

from core.front_end.keyframe_selector import KeyFrameSelectThresholds, SelectMetrics
from logger import spawn_logger
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class KeyframeMetricsModule(IVizModule):
    """KeyframeMetricsModule module."""

    def __init__(
        self, property_name: str, entity_path: str, metrics: list[str], *, throw_on_nothing: bool = False
    ) -> None:
        """Initialize the plot module."""
        self.property_name = property_name
        self.entity_path = entity_path
        self.throw_on_nothing = throw_on_nothing
        self.metrics = metrics
        self.logger = spawn_logger(KeyframeMetricsModule.__name__)
        self.thresholds = KeyFrameSelectThresholds()

    def setup(self) -> None:
        """Set up the image module."""
        for metric in self.metrics:
            rr.log(f"{self.entity_path}/{metric}", rr.SeriesLines(widths=1.75), static=True)
        rr.log(
            f"{self.entity_path}/median_parallax/threshold",
            rr.SeriesLines(widths=2.5, colors=np.array([255, 0, 0])),
            static=True,
        )
        rr.log(
            f"{self.entity_path}/connectivity_ratio/threshold",
            rr.SeriesLines(widths=2.5, colors=np.array([255, 0, 0])),
            static=True,
        )
        rr.log(
            f"{self.entity_path}/time_diff/min_threshold",
            rr.SeriesLines(widths=1.0, colors=np.array([255, 0, 0])),
            static=True,
        )
        rr.log(
            f"{self.entity_path}/time_diff/max_threshold",
            rr.SeriesLines(widths=2.5, colors=np.array([255, 0, 0])),
            static=True,
        )

    def process(self, context: Ctx) -> None:
        """Process the keyframe metrics data."""
        exists = context.exists(self.property_name)
        if not exists and self.throw_on_nothing:
            msg = f"Keyframe metrics data not found in context: {self.property_name}"
            self.logger.warning(msg)
            raise KeyError(msg)
        if not exists and not self.throw_on_nothing:
            return
        ts = context.get_scalar("timestamp", float) / 1e9

        select_metrics = SelectMetrics.from_arrow(
            context.get_record_batch(self.property_name, SelectMetrics.schema())
        )
        rr.set_time("sim_time", timestamp=ts)
        rr.set_time("frame_time", timestamp=ts)
        for metric in self.metrics:
            rr.log(f"{self.entity_path}/{metric}", rr.Scalars(getattr(select_metrics, f"keyframe_{metric}")))
        rr.log(f"{self.entity_path}/median_parallax/threshold", rr.Scalars(self.thresholds.min_parallax_pts))
        rr.log(
            f"{self.entity_path}/connectivity_ratio/threshold", rr.Scalars(self.thresholds.min_connectivity_ratio)
        )
        rr.log(f"{self.entity_path}/time_diff/min_threshold", rr.Scalars(self.thresholds.ignore_time_until_sec))
        rr.log(f"{self.entity_path}/time_diff/max_threshold", rr.Scalars(self.thresholds.max_time_delta_sec))

    def __repr__(self) -> str:
        """Return the string representation of the image module."""
        return f"KeyframeMetrics(property_name={self.property_name}, entity_path={self.entity_path})"
