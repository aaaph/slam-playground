from dataclasses import dataclass, field

import numpy as np
import rerun as rr
from numpy.typing import NDArray

from core.feature_tracker.feature import FeatureLifecycle
from core.feature_tracker.feature_schema import FeatureSchema
from core.feature_tracker.feature_tensor import FeatureTensor
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule

_DEFAULT_COLOR_PALETTE = {
    "new": [0, 255, 0],
    "tracked": [255, 0, 0],
    "lost": [128, 128, 128],
}


def _default_color_palette() -> dict[str, list[int]]:
    """Return a fresh default color palette."""
    return _DEFAULT_COLOR_PALETTE.copy()


@dataclass(slots=True)
class FeaturesModuleOptions:
    """Features module configuration."""

    point_size: float = 1.5
    color_palette: dict[str, list[int]] = field(default_factory=_default_color_palette)
    left: bool = True
    show_stereo_baseline: bool = False
    show_feature_labels: bool = False


class FeaturesModule(IVizModule):
    """Features module."""

    def __init__(
        self,
        entity_path: str,
        property_name: str,
        options: FeaturesModuleOptions | None = None,
    ) -> None:
        """Initialize the features module."""
        options = options or FeaturesModuleOptions()
        self.entity_path = entity_path
        self.property_name = property_name
        self.point_size = options.point_size
        self.show_stereo_baseline = options.show_stereo_baseline
        self.show_feature_labels = options.show_feature_labels
        if options.left:
            self.points_index_start = 2
            self.points_index_end = 4
        else:
            self.points_index_start = 4
            self.points_index_end = 6
        self.color_palette = options.color_palette

    def setup(self) -> None:
        """Set up the features module."""

    def process(self, context: Ctx) -> None:
        """Process the features data."""
        exists = context.exists(self.property_name)
        if not exists:
            msg = f"Features data not found in context: {self.property_name}"
            raise KeyError(msg)
        tensor = FeatureTensor.from_arrow(context.get_record_batch(self.property_name, FeatureTensor.schema))

        active_data = tensor.active_frame.ndarray
        points = active_data[:, self.points_index_start : self.points_index_end]
        features_ids = active_data[:, 0].astype(np.int32)
        colors = self._default_color_strategy(active_data)
        labels = np.array([f"{feat_id}" for feat_id in features_ids])
        radii = np.full(len(features_ids), self.point_size)
        rr.set_time("sim_time", timestamp=context.get_scalar("timestamp", float) / 1e9)
        rr.set_time("frame_time", timestamp=context.get_scalar("timestamp", float) / 1e9)
        if self.show_stereo_baseline:
            stereo_mask = ~np.isnan(active_data[:, FeatureSchema.RIGHT_U]) & (
                active_data[:, FeatureSchema.LIFECYCLE] == FeatureLifecycle.ACTIVE.value
            )
            stereo_data = active_data[stereo_mask]
            strips = np.stack([stereo_data[:, 2:4], stereo_data[:, 4:6]], axis=1)
            stereo_colors = colors[stereo_mask]
            stereo_radii = np.full(len(stereo_mask), max(self.point_size - 1, 1.0))
            rr.log(
                f"{self.entity_path}/baseline",
                rr.LineStrips2D(
                    strips=strips,
                    colors=stereo_colors,
                    radii=stereo_radii,
                ),
            )

        rr.log(
            self.entity_path,
            rr.Points2D(points, radii=radii, colors=colors, labels=labels, show_labels=self.show_feature_labels),
        )
        rr.log(f"{self.entity_path}/count", rr.TextLog(f"{len(features_ids)}"))

    def __repr__(self) -> str:
        """Return the string representation of the features module."""
        return f"FeaturesModule(entity_path={self.entity_path}, property_name={self.property_name})"

    def _default_color_strategy(self, data: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Return the default color strategy for features."""
        n = data.shape[0]
        colors = np.full((n, 3), [255, 255, 255], dtype=np.uint8)

        lifecycle = data[:, FeatureSchema.LIFECYCLE].astype(np.int32)
        age = data[:, FeatureSchema.AGE].astype(np.int32)

        is_active = lifecycle == FeatureLifecycle.ACTIVE.value
        is_new = is_active & (age == 0)
        is_tracked = is_active & (age > 0)
        is_lost = lifecycle == FeatureLifecycle.LOST.value

        colors[is_new] = self.color_palette.get("new", [0, 255, 0])
        colors[is_tracked] = self.color_palette.get("tracked", [255, 0, 0])
        colors[is_lost] = self.color_palette.get("lost", [128, 128, 128])

        return colors
