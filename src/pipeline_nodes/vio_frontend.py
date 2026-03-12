import numpy as np

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature_tracker import FeatureTracker, FeatureTrackerMode
from core.front_end.feature_manager import FeatureManager
from core.pose_tracker.inertial_integration import InertialIntegration
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

        return (
            ctx.set_image("left_rect", left)
            .set_record_batch("active_feat", self.ft.tensor.as_arrow())
            .set_ndarray("vio_pose", nav_state.pose().matrix())
            .set_ndarray("points", points)
            .set_scalar("points_size", points_size)
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
