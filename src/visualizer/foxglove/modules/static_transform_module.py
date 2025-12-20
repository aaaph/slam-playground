from foxglove.schemas import FrameTransform, Quaternion, Vector3

from core.transformations.special_euclidian_3_dim import SE3
from visualizer.foxglove.modules.abc_module import IVizModule
from visualizer.visualizer_context import VisualizerContext


class StaticTransformModule(IVizModule):
    """Static transform module."""

    def __init__(self, parent_frame_id: str, child_frame_id: str, se3: SE3) -> None:
        """Initialize the static transforms module."""
        self.parent_frame_id = parent_frame_id
        self.child_frame_id = child_frame_id
        self.se3 = se3

    def setup(self) -> None:
        """Set up the static transforms module."""

    def process(self, _ctx: VisualizerContext) -> list[FrameTransform]:
        """Process the static transforms data."""
        vec = self.se3.translation()
        quat = self.se3.rotation().as_quat()
        return [
            FrameTransform(
                parent_frame_id=self.parent_frame_id,
                child_frame_id=self.child_frame_id,
                translation=Vector3(x=vec[0], y=vec[1], z=vec[2]),
                rotation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
            )
        ]
