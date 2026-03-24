import numpy as np
from dora import Node

from core.camera_model.stereo_camera_ctx import StereoContext
from core.camera_model.stereo_camera_model import StereoCameraModel
from core.front_end.front_end import FrontEnd
from core.transformations.special_euclidian_3_dim import SE3
from dataset.euroc import EurocDataset
from logger import node_logger
from pipeline.annotations import Ctx
from pipeline.decorators import on_input, reactive, to_output


@reactive
class StereoFronEndNode:
    """Stereo front end node."""

    def run(self) -> None: ...  # noqa: D102

    def __init__(self) -> None:
        """Initialize the stereo front end node."""
        self.node = Node()
        self.logger = node_logger(app="stereo_front_end_node")

        euroc = EurocDataset.mh_01_easy()
        self.camera_model = StereoCameraModel.from_cameras_config(euroc.config.cam0, euroc.config.cam1)
        self.stereo_ctx = self.camera_model.as_stereo_ctx()
        initial_pose = StereoFronEndNode.get_init_cam0_in_world_se3(euroc, self.stereo_ctx)
        self.front_end: FrontEnd = FrontEnd.default_factory(self.camera_model, initial_pose=initial_pose)

    @staticmethod
    def get_init_cam0_in_world_se3(euroc_dataset: EurocDataset, stereo_ctx: StereoContext) -> SE3:
        """Get the initial camera pose in the world frame."""
        first_stereo_data = next(iter(euroc_dataset.stereo().to_iterable_dataset()))
        initial_body_in_world = euroc_dataset.find_nearest_ground_truth_by_timestamp(
            float(first_stereo_data["timestamp"])
        )
        initial_body_in_world_quat = initial_body_in_world["gt_orientation"]
        initial_body_in_world_vec = initial_body_in_world["gt_position"]
        initial_body_in_world_se3 = SE3.from_quat_and_translation(
            np.array(initial_body_in_world_quat), np.array(initial_body_in_world_vec)
        )
        cam0_in_body_se3: SE3 = stereo_ctx.cam0_in_body_se3
        return initial_body_in_world_se3 * cam0_in_body_se3

    @on_input("ctx")
    @to_output("ctx")
    def handle_ctx(self, ctx: Ctx) -> Ctx:
        """Handle the ctx event."""
        timestamp = ctx.get_scalar("timestamp", float)
        width = ctx.get_scalar("width", int)
        height = ctx.get_scalar("height", int)
        left = ctx.get_image("left", (height, width))
        right = ctx.get_image("right", (height, width))
        result = self.front_end.feed(timestamp, (left, right))
        soa = result.as_soa()
        lost_features_count, lost_features = result.lost_features_ndarrays()
        if lost_features_count > 0:
            ctx.set_scalar("lost_features_count", lost_features_count)
            ctx.set_ndarray("lost_features", lost_features)
        if result.keyframe:
            keyframe = result.keyframe.as_soa()
            ctx.set_scalar("keyframe_active_landmarks_count", keyframe["active_landmarks_count"][0])
            ctx.set_scalar("keyframe_active_features_count", keyframe["active_features_count"][0])
            ctx.set_scalar("keyframe_id", keyframe["keyframe_id"][0])
            ctx.set_ndarray("keyframe_select_reason", keyframe["select_reason"][0])
            ctx.set_scalar("keyframe_timestamp", keyframe["timestamp"][0])
            ctx.set_ndarray("keyframe_pose", keyframe["pose"])
            ctx.set_ndarray("keyframe_active_landmarks", keyframe["active_landmarks"])
            ctx.set_ndarray("keyframe_active_features", keyframe["active_features"])

        return (
            ctx.set_ndarray("active_feat", soa["active_feat"])
            .set_scalar("active_feat_count", soa["active_feat_count"][0])
            .set_ndarray("new_landmarks", soa["new_landmarks"])
            .set_scalar("new_landmarks_count", soa["new_landmarks_count"][0])
            .set_ndarray("camera_in_world_se3", result.camera_in_world_se3.as_matrix())
            .set_scalar("result_id", result.result_id)
            .set_image("rect_left", result.left)
            .set_image("rect_right", result.right)
            .reassemble()
        )

    @on_input("helthcheck")
    def handle_helthcheck(self) -> None:
        """Handle the tick event."""
        self.logger.trace(f"Tick event received: {self.front_end.iteration_id}")


if __name__ == "__main__":
    StereoFronEndNode().run()
