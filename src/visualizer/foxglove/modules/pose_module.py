from foxglove.channels import PoseChannel
from foxglove.schemas import FrameTransform, Pose, Quaternion, Vector3

from visualizer.foxglove.modules.abc_module import IVizModule
from visualizer.visualizer_context import VisualizerContext


class PoseModule(IVizModule):
    """Pose module."""

    def __init__(self, channel_name: str, property_name: str, parent_frame_id: str, child_frame_id: str) -> None:
        """Initialize the pose module."""
        self.channel_name = channel_name
        self.property_name = property_name
        self.parent_frame_id = parent_frame_id
        self.child_frame_id = child_frame_id

    def setup(self) -> None:
        """Set up the pose module visualization."""
        self.pose_channel = PoseChannel(self.channel_name)

    def process(self, context: VisualizerContext) -> list[FrameTransform]:
        """Process the pose data."""
        if getattr(context, self.property_name) is None:  # with dict we could check my dynamic field
            msg = f"Property {self.property_name} not found in data"
            raise ValueError(msg)
        pose_se3 = getattr(context, self.property_name)
        if pose_se3 is None:
            raise ValueError("Pose data not found")
        vec = pose_se3.translation()
        quat = pose_se3.rotation().as_quat()
        pose_message = Pose(
            position=Vector3(x=vec[0], y=vec[1], z=vec[2]),
            orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
        )
        self.pose_channel.log(pose_message)
        return [
            FrameTransform(
                parent_frame_id=self.parent_frame_id,
                child_frame_id=self.child_frame_id,
                translation=Vector3(x=vec[0], y=vec[1], z=vec[2]),
                rotation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
            )
        ]
