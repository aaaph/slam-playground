import rerun as rr

from visualizer.rerun.modules.abc_module import IVizModule
from visualizer.visualizer_context import VisualizerContext


class DynamicTransformModule(IVizModule):
    """Dynamic transform module."""

    def __init__(
        self,
        entity_path: str,
        property_name: str,
        axes_length: float = 0.425,
        *,
        show_axes: bool = True,
    ) -> None:
        """Initialize the dynamic transform module."""
        self.entity_path = entity_path
        self.property_name = property_name
        self.show_axes = show_axes
        self.axes_length = axes_length

    def setup(self) -> None:
        """Set up the dynamic transform module."""

    def process(self, context: VisualizerContext) -> None:
        """Process the static transform data."""
        if getattr(context, self.property_name) is None:  # with dict we could check my dynamic field
            msg = f"Property {self.property_name} not found in data"
            raise ValueError(msg)
        pose_se3 = getattr(context, self.property_name)
        if pose_se3 is None:
            raise ValueError("Pose data not found")
        vec = pose_se3.translation()
        quat = pose_se3.rotation().as_quat()
        rr.log(
            self.entity_path,
            rr.Transform3D(
                translation=vec,
                quaternion=quat,
            ),
            rr.TransformAxes3D(self.axes_length),
        )
