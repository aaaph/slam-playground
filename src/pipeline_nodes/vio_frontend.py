from collections import deque

import numpy as np

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature_tracker import FeatureTracker, FeatureTrackerMode
from core.front_end.feature_manager import FeatureManager
from core.front_end.keyframe_selector import KeyframeSelector
from core.pose_tracker.inertial_integration import InertialIntegration
from core.transformations.special_euclidian_3_dim import SE3
from dataset.euroc import EurocDataset
from logger import spawn_logger
from pipeline.annotations import Ctx
from pipeline.decorators import on_input, on_stop, reactive, to_output


@reactive
class VIOFrontend:
    """VIO frontend."""

    def run(self) -> None: ...  # noqa: D102

    def __init__(self) -> None:
        """Initialize the VIO frontend."""
        self.logger = spawn_logger(app="vio_frontend")
        euroc = EurocDataset.mh_01_easy()
        self.camera_model = StereoCameraModel.from_cameras_config(euroc.config.cam0, euroc.config.cam1)
        self.stereo_ctx = self.camera_model.as_stereo_ctx()
        self.ft = FeatureTracker.default_factory(
            self.stereo_ctx,
            feat_amount_per_region=6,
            feat_retrack_threshold=4,
            region_amount=12,
            mode=FeatureTrackerMode.STEREO,
        )
        self.imu_integration = InertialIntegration.from_ground_truth(
            gravity=np.array([0.0, 0.0, -9.81]), ground_truth=euroc.first_ground_truth()
        )
        self.feature_manager = FeatureManager.from_stereo_camera_ctx(self.stereo_ctx)
        self.disparity_window = deque(maxlen=5)
        self.keyframe_selector = KeyframeSelector.default_factory()

    @on_input("ctx")
    @to_output("ctx")
    def handle_ctx(self, ctx: Ctx) -> Ctx:
        """Handle the ctx event."""
        width = ctx.get_scalar("width")
        height = ctx.get_scalar("height")
        left = ctx.get_image("left", (height, width))
        right = ctx.get_image("right", (height, width))
        timestamp = ctx.get_scalar("timestamp")
        left, right = self.camera_model.process_stereo(left, right)
        active_features = self.ft.feed(timestamp, (left, right))
        self.disparity_window.append(self.ft.median_disparity)
        avg_disparity = sum(self.disparity_window) / len(self.disparity_window)

        imu_rows = ctx.get_scalar("imu_rows", int)
        gyro = ctx.get_ndarray("gyro", (imu_rows, 3))
        accel = ctx.get_ndarray("accel", (imu_rows, 3))
        imu_ts = ctx.get_ndarray("imu_ts", (imu_rows,))
        nav_state = self.imu_integration.integrate_and_predict(accel, gyro, imu_ts)

        points = self.feature_manager.add_active_features(active_features)
        """ points[:, 1:4] = (
            self.stereo_ctx.cam0_in_body_se3.rotation().as_matrix() @ points[:, 1:4].T
        ).T + self.stereo_ctx.cam0_in_body_se3.translation() """
        points_size = points.shape[0]
        predicted_pose = SE3.from_matrix(nav_state.pose().matrix())
        good_kf, _, select_metrics = self.keyframe_selector.check(
            timestamp, predicted_pose, active_features.good_features()
        )
        if good_kf:
            self.keyframe_selector.set_new_keyframe(timestamp, predicted_pose, active_features.good_features())

        return (
            ctx.set_image("left_rect", left)
            .set_record_batch("active_feat", self.ft.tensor.as_arrow())
            .set_ndarray("vio_pose", nav_state.pose().matrix())
            .set_ndarray("points", points)
            .set_scalar("points_size", points_size)
            .set_scalar("interframe_disparity", self.ft.median_disparity)
            .set_scalar("avg_disparity", avg_disparity)
            .set_scalar("keyframe_time_diff", select_metrics["keyframe_time_diff"])
            .set_scalar("keyframe_median_parallax", select_metrics["keyframe_median_parallax"])
            .set_scalar("keyframe_connectivity_ratio", select_metrics["keyframe_connectivity_ratio"])
            .set_scalar("keyframe_common_feat_count", select_metrics["keyframe_common_feat_count"])
            .set_scalar("keyframe_distance_diff", select_metrics["keyframe_distance_diff"])
            .set_scalar("keyframe_angle_diff", select_metrics["keyframe_angle_diff"])
            .reassemble()
        )

    @on_input("helthcheck")
    def handle_helthcheck(self) -> None:
        """Handle the helthcheck event."""
        self.logger.trace("Still alive")

    @on_stop
    def handle_shutdown(self) -> None:
        """Handle the shutdown event."""
        self.logger.info("VIO frontend stopping...")


if __name__ == "__main__":
    VIOFrontend().run()
