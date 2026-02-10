import numpy as np
import rerun as rr

from core.feature_tracker.feature_tensor import FeatureTensor
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class FeaturesModule(IVizModule):
    """Features module."""

    def __init__(self, entity_path: str, property_name: str, point_size: float = 1.5) -> None:
        """Initialize the features module."""
        self.entity_path = entity_path
        self.property_name = property_name
        self.point_size = point_size

    def setup(self) -> None:
        """Set up the features module."""

    def process(self, context: Ctx) -> None:
        """Process the features data."""
        exists = context.exists(self.property_name)
        if not exists:
            msg = f"Features data not found in context: {self.property_name}"
            raise KeyError(msg)
        tensor = FeatureTensor.from_arrow(context.get_record_batch(self.property_name, FeatureTensor.schema))

        active_data = tensor.active_features()
        points = active_data[:, 2:4]
        features_ids = active_data[:, 0].astype(np.int32)
        colors = FeatureTensor.to_color_array(active_data)
        labels = np.array([f"feat_{feat_id}" for feat_id in features_ids])
        radii = np.full(len(features_ids), self.point_size)

        rr.log(
            self.entity_path,
            rr.Points2D(
                points,
                radii=radii,
                colors=colors,
                labels=labels,
            ),
        )

    def __repr__(self) -> str:
        """Return the string representation of the features module."""
        return f"FeaturesModule(entity_path={self.entity_path}, property_name={self.property_name})"
