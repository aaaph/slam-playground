from foxglove.channels import PosesInFrameChannel
from foxglove.schemas import FrameTransform, Pose, PosesInFrame, Quaternion, Vector3

from visualizer.foxglove.modules.abc_module import IVizModule
from visualizer.visualizer_context import VisualizerContext


class SelectedKeyframesModule(IVizModule):
    """Selected keyframes module."""

    def setup(self) -> None:
        """Set up the selected keyframes module."""
        self.selected_keyframes_channel = PosesInFrameChannel("/selected_keyframes")

    def process(self, context: VisualizerContext) -> list[FrameTransform]:
        """Process the selected keyframes data."""
        if context.selected_keyframes is None:
            return []
        poses_in_frame = []
        for keyframe in context.selected_keyframes:
            vec = keyframe.pose.translation()
            quat = keyframe.pose.rotation().as_quat()
            pose_message = Pose(
                position=Vector3(x=vec[0], y=vec[1], z=vec[2]),
                orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
            )
            poses_in_frame.append(pose_message)
        self.selected_keyframes_channel.log(PosesInFrame(frame_id="selected_keyframes", poses=poses_in_frame))

        return [
            FrameTransform(
                parent_frame_id="world",
                child_frame_id="selected_keyframes",
                translation=Vector3(x=0, y=0, z=0),
            )
        ]
