import rerun as rr

from core.transformations.special_euclidian_3_dim import SE3
from visualizer.rerun.modules.abc_module import IVizModule
from visualizer.visualizer_context import VisualizerContext


class StaticTransformModule(IVizModule):
    """Static transform module."""

    def __init__(self, entity_path: str, se3: SE3) -> None:
        """Initialize the static transform module."""
        self.entity_path = entity_path
        self.se3 = se3

    def setup(self) -> None:
        """Set up the static transform module."""
        rr.log(
            self.entity_path,
            rr.Transform3D(
                translation=self.se3.translation(),
                quaternion=self.se3.rotation().as_quat(),
            ),
            static=True,
        )

    def process(self, context: VisualizerContext) -> None:
        """Process the static transform data."""
