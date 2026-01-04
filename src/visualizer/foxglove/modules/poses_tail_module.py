from foxglove.channels import PosesInFrameChannel
from foxglove.schemas import FrameTransform, Pose, PosesInFrame, Quaternion, Vector3

from visualizer.foxglove.modules.abc_module import IVizModule
from visualizer.visualizer_context import VisualizerContext


class PosesTailModule(IVizModule):
    """Poses tail module."""

    def setup(self) -> None:
        """Set up the poses tail module."""
        self.poses_tail_channel = PosesInFrameChannel("/poses_tail")

    def process(self, context: VisualizerContext) -> list[FrameTransform]:
        """Process the poses tail data."""
        if context.pose_history is None:
            return []
        poses_in_frame = []
        for pose in context.pose_history:
            vec = pose.translation()
            quat = pose.rotation().as_quat()
            pose_message = Pose(
                position=Vector3(x=vec[0], y=vec[1], z=vec[2]),
                orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
            )
            poses_in_frame.append(pose_message)
        self.poses_tail_channel.log(PosesInFrame(frame_id="poses_tail", poses=poses_in_frame))

        return [
            FrameTransform(
                parent_frame_id="world",
                child_frame_id="poses_tail",
                translation=Vector3(x=0, y=0, z=0),
            )
        ]
