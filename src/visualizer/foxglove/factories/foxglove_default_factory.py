from collections.abc import Generator

from core.camera_model.stereo_camera_ctx import StereoContext
from core.transformations.special_euclidian_3_dim import SE3
from visualizer.foxglove.foxglove_visualizer import FoxgloveVisualizer
from visualizer.foxglove.modules.image_module import ImageModule
from visualizer.foxglove.modules.point_cloud_module import PointCloudModule
from visualizer.foxglove.modules.pose_module import PoseModule
from visualizer.foxglove.modules.poses_tail_module import PosesTailModule
from visualizer.foxglove.modules.selected_keyframes_module import SelectedKeyframesModule
from visualizer.foxglove.modules.static_transform_module import StaticTransformModule
from visualizer.visualizer_context import VisualizerContext


class FoxgloveFactory:
    """Foxglove default factory."""

    def __init__(self) -> None:
        """Initialize the Foxglove default factory."""

    def create_default_viz(
        self, stereo_ctx: StereoContext, viz_type: str = "websocket", mcap_file_name: str = "default"
    ) -> Generator[None, VisualizerContext]:
        """
        Create a default Foxglove visualizer with the default modules.

        The default modules are:
        - PointCloudModule
        - PoseModule(Base link, Ground truth, Optimized pose)
        - StaticTransformModule
        - PosesTailModule
        - SelectedKeyframesModule
        - ImageModule(Only left image is published)

        Args:
            stereo_ctx: The stereo context.
            viz_type: The type of visualizer to create.
            mcap_file_name: The name of the mcap file to create.

        Returns:
            A generator that yields the visualizer context.

        """
        foxglove_visualizer = FoxgloveVisualizer()
        foxglove_visualizer.add_module(PointCloudModule())
        foxglove_visualizer.add_module(
            PoseModule(
                channel_name="/base_link",
                property_name="body_in_world_se3",
                parent_frame_id="odom",
                child_frame_id="base_link",
            )
        )
        foxglove_visualizer.add_module(
            StaticTransformModule(
                parent_frame_id="world",
                child_frame_id="odom",
                se3=SE3.identity(),
            )
        )
        foxglove_visualizer.add_module(
            StaticTransformModule(
                parent_frame_id="base_link",
                child_frame_id="cam0",
                se3=stereo_ctx.cam0_in_body_se3,
            )
        )
        foxglove_visualizer.add_module(
            StaticTransformModule(
                parent_frame_id="base_link",
                child_frame_id="cam1",
                se3=stereo_ctx.cam1_in_body_se3,
            )
        )
        foxglove_visualizer.add_module(
            PoseModule(
                channel_name="/ground_truth",
                property_name="ground_truth_se3",
                parent_frame_id="world",
                child_frame_id="ground_truth",
            )
        )
        foxglove_visualizer.add_module(
            PoseModule(
                channel_name="/optimized_pose",
                property_name="optimized_pose_se3",
                parent_frame_id="world",
                child_frame_id="optimized_pose",
            )
        )
        foxglove_visualizer.add_module(PosesTailModule())
        foxglove_visualizer.add_module(ImageModule())
        foxglove_visualizer.add_module(SelectedKeyframesModule())
        if viz_type == "websocket":
            viz = foxglove_visualizer.websocket_viz_gen()
        elif viz_type == "mcap":
            viz = foxglove_visualizer.mcap_gen(mcap_file_name)
        else:
            msg = f"Invalid viz type: {viz_type}"
            raise ValueError(msg)
        return viz
