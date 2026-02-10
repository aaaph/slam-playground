import numpy as np

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.front_end.keyframe import Keyframe
from core.graph_optimizer.fixed_lag_optimizer import FixedLagOptimizer
from dataset.euroc import EurocDataset
from logger import node_logger
from pipeline.annotations import Ctx
from pipeline.decorators import on_input, on_stop, reactive, to_output


@reactive
class SlidingWindowBackEndNode:
    """Sliding window back end node."""

    def run(self) -> None: ...  # noqa: D102

    def __init__(self) -> None:
        """Initialize the sliding window back end node."""
        self.logger = node_logger(app="sliding_window_back_end_node")
        euroc = EurocDataset.mh_01_easy()
        self.model = StereoCameraModel.from_cameras_config(euroc.config.cam0, euroc.config.cam1)
        self.stereo_ctx = self.model.as_stereo_ctx()
        self.fixed_lag_optimizer = FixedLagOptimizer.from_stereo_ctx(self.stereo_ctx)

    @on_input("ctx")
    @to_output("ctx")
    def handle_ctx(self, ctx: Ctx) -> Ctx:
        """Handle the ctx event."""
        # timestamp = ctx.get_scalar("timestamp", float)
        lost_features_exists = ctx.exists("lost_features")
        if lost_features_exists:
            lost_features_count = ctx.get_scalar("lost_features_count", int)
            lost_features = ctx.get_ndarray("lost_features", (lost_features_count, 6))
            lost_features_ids = lost_features[:, 0].astype(int)
            erased_any, erased_landmarks = self.fixed_lag_optimizer.update_by_lost_features_ids(lost_features_ids)
            if erased_any:
                ctx.set_ndarray("erased_landmarks", np.array(erased_landmarks, dtype=np.int32))
        keyframe_exists = ctx.exists("keyframe_id")
        if keyframe_exists:
            keyframe_id = ctx.get_scalar("keyframe_id", int)
            keyframe_select_reason = ctx.get_scalar("keyframe_select_reason", int)
            keyframe_timestamp = ctx.get_scalar("keyframe_timestamp", float)
            keyframe_pose = ctx.get_ndarray("keyframe_pose", (4, 4))
            keyframe_active_landmarks_count = ctx.get_scalar("keyframe_active_landmarks_count", int)
            keyframe_active_features_count = ctx.get_scalar("keyframe_active_features_count", int)
            keyframe_active_landmarks = ctx.get_ndarray(
                "keyframe_active_landmarks", (keyframe_active_landmarks_count, 4)
            )
            keyframe_active_features = ctx.get_ndarray(
                "keyframe_active_features", (keyframe_active_features_count, 6)
            )
            keyframe = Keyframe.from_soa(
                {
                    "keyframe_id": np.array([keyframe_id], dtype=np.int32),
                    "select_reason": np.array([keyframe_select_reason], dtype=np.int32),
                    "timestamp": np.array([keyframe_timestamp], dtype=np.float32),
                    "pose": keyframe_pose,
                    "active_landmarks": keyframe_active_landmarks,
                    "active_features": keyframe_active_features,
                    "active_features_count": keyframe_active_features_count,
                    "active_landmarks_count": keyframe_active_landmarks_count,
                }
            )
            corrected_pose = self.fixed_lag_optimizer.update_by_keyframe(keyframe)
            ctx.set_ndarray("corrected_pose", corrected_pose.as_matrix())
        corrected_landmarks = self.fixed_lag_optimizer.get_landmarks()
        corrected_landmarks_count = len(corrected_landmarks)
        corrected_landmarks_pack = np.zeros((corrected_landmarks_count, 4), dtype=np.float32)
        for i, (landmark_id, landmark) in enumerate(corrected_landmarks.items()):
            corrected_landmarks_pack[i, 0] = landmark_id
            corrected_landmarks_pack[i, 1:4] = landmark
        ctx.set_scalar("corrected_landmarks_count", corrected_landmarks_count)
        ctx.set_ndarray("corrected_landmarks", corrected_landmarks_pack)
        return ctx

    @on_input("helthcheck")
    def handle_helthcheck(self) -> None:
        """Handle the helthcheck event."""
        self.logger.trace("Still alive")

    @on_stop
    def handle_shutdown(self) -> None:
        """Handle the shutdown event."""
        self.logger.info("Sliding window back end node stopped")


if __name__ == "__main__":
    SlidingWindowBackEndNode().run()
