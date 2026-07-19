from enum import IntEnum
from typing import TYPE_CHECKING

import gtsam
import numpy as np
from dora import Node
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.camera_model.vio_context import VioContext
from core.feature_tracker.feature_frame import FeatureFrame
from core.feature_tracker.feature_tracker import FeatureTracker, FeatureTrackerMode
from core.feature_tracker.zero_velocity_tracker import ZeroVelocityTrackerState
from core.front_end.feature_manager import FeatureManager
from core.front_end.front_end_bootstrap import FrontEndBootstrap, FrontEndBootstrapInput, FrontEndBootstrapResult
from core.front_end.keyframe import KF
from core.front_end.keyframe_selector import KeyframeSelector, KeyFrameSelectThresholds, SelectReason
from core.graph_optimizer.optimizer_types import PredictionMode
from core.pose_tracker.inertial_integration import ImuBuffer
from core.pose_tracker.local_map import LocalMap
from core.pose_tracker.pnp_pose_tracker import PnpPoseTracker
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger
from pipeline.annotations import Ctx, Metadata
from pipeline.decorators import handle, on_input, on_stop, reactive, send_pipeline_context_output
from pipeline.nodes.base import PipelineNode

if TYPE_CHECKING:
    from gtsam.gtsam import NavState


class FrontEndMode(IntEnum):
    """Front end mode."""

    SILENT_AWAIT = 0
    VIBRATION_AWAIT = 1
    ZERO_MOTION_INITIALIZATION = 2
    DYNAMIC_INITIALIZATION = 3
    NOMINAL = 4


@reactive
class VIOFrontend(PipelineNode):
    """VIO frontend."""

    def __init__(self, camera_model: StereoCameraModel, vio_ctx: VioContext) -> None:
        """Initialize the VIO frontend."""
        self.node = Node()
        self.mode = FrontEndMode.SILENT_AWAIT
        self.estimation_mode = PredictionMode.PNP
        self.logger = spawn_logger(app="vio_frontend")
        self.camera_model = camera_model
        self.vio_ctx = vio_ctx
        self.ft = FeatureTracker.default_factory(
            self.vio_ctx.stereo,
            feat_amount_per_region=12,
            feat_retrack_threshold=4,
            region_amount=12,
            mode=FeatureTrackerMode.STEREO,
        )
        self.feature_manager = FeatureManager.from_stereo_camera_ctx(self.vio_ctx.stereo)
        self.bootstrap = FrontEndBootstrap()
        self.kf_selector = KeyframeSelector.from_thresholds(
            KeyFrameSelectThresholds(min_parallax_pts=50, max_time_delta_sec=3.0)
        )
        self.quiet_imu_buffer = ImuBuffer(capacity=1000)
        self.quiet_statistics = {"mean": None, "var": None}
        self.six_dof_quiet_threshold = chi2.ppf(0.95, 6)
        self.quiet_ratio_threshold = 10.0

        self.imu_buffer = ImuBuffer(capacity=10000)
        self.state = np.zeros(16)  # quat(4) + t(3) + v(3) + ba(3) + bg(3) = 16
        self.state[:4] = Rotation.identity().as_quat()

        self.vo_state = np.zeros(11)  # quat(4) + t(3) + v(3) + ts(1) = 11
        self.vo_state[:4] = Rotation.identity().as_quat()
        self.pim = gtsam.PreintegratedImuMeasurements(
            self.vio_ctx.imu.pim_params(), gtsam.imuBias.ConstantBias(self.state[10:13], self.state[13:16])
        )
        self.local_map = LocalMap.from_capacity(capacity=100000)
        self.pnp_pose_tracker = PnpPoseTracker.default_factory(self.vio_ctx.stereo, motion_only_ba_enabled=False)

    @handle("sensor_frame", "frame")
    def handle_sensor_frame(self, ctx: Ctx, metadata: Metadata) -> Ctx:
        """Handle the sensor frame event."""
        frame_id = self.ft.iterator_count
        timestamp = ctx.get_scalar("timestamp")

        motion_in_static_detected = self.process_image(frame_id, ctx)
        vibration_in_static_detected = self.process_imu_data(ctx)
        current_frame = self.ft.active_frame()

        current_points = self.feature_manager.triangulate_frame(current_frame)
        active_track = np.column_stack((current_frame.good_features(), current_points[:, 1:4]))
        bootstrap_result = self.update_bootstrap(frame_id, timestamp, current_frame, current_points, ctx)

        if not self.local_map.empty():
            good_features = current_frame.good_features()
            self.estimate_pnp_pose(timestamp, good_features)
        keyframes: list[KF] = []

        if vibration_in_static_detected:
            # shoube be a first keyframe in factor graph
            self.mode = FrontEndMode.ZERO_MOTION_INITIALIZATION
            self.logger.info("[FE:MODE]: from VIBRATION_AWAIT to ZERO_MOTION_INITIALIZATION")
            kf = KF(
                keyframe_id=frame_id,
                timestamp=timestamp,
                select_reasons=[SelectReason.STATIC_INITIALIZATION],
                state=self.state.copy(),
                imu_batch=self.imu_buffer.buffer[: self.imu_buffer.size, :].copy(),
                active_track=active_track,
                non_zero_velocity_detected=False,
            )
            keyframes.append(kf)
            self.kf_selector.set_new_keyframe(timestamp, current_frame.good_features())
            self.kf_selector.initialize()

        if motion_in_static_detected:
            self.mode = FrontEndMode.DYNAMIC_INITIALIZATION
            self.logger.info("[FE:MODE]: from VIBRATION_AWAIT to DYNAMIC_INITIALIZATION")
            kf = KF(
                keyframe_id=frame_id,
                timestamp=timestamp,
                select_reasons=[SelectReason.MOTION_INITIALIZATION],
                state=self.state.copy(),
                imu_batch=self.imu_buffer.buffer[: self.imu_buffer.size, :].copy(),
                active_track=active_track,
                non_zero_velocity_detected=True,
            )

            keyframes.append(kf)
            self.kf_selector.set_new_keyframe(timestamp, current_frame.good_features())
            # self.ks.initialize()

        good_kf, select_reasons, select_metrics = self.kf_selector.check(timestamp, current_frame.good_features())

        if good_kf:
            kf_state = self.state.copy()
            kf_state[:10] = self.vo_state[:10]
            kf = KF(
                keyframe_id=frame_id,
                timestamp=timestamp,
                select_reasons=select_reasons,
                state=kf_state,
                imu_batch=self.imu_buffer.buffer[: self.imu_buffer.size, :].copy(),
                active_track=active_track,
                non_zero_velocity_detected=(self.mode != FrontEndMode.ZERO_MOTION_INITIALIZATION),
            )
            keyframes.append(kf)
            self.kf_selector.set_new_keyframe(timestamp, current_frame.good_features())
            if self.mode == FrontEndMode.DYNAMIC_INITIALIZATION:
                self.kf_selector.switch_thresholds(KeyFrameSelectThresholds())
                self.mode = FrontEndMode.NOMINAL
                self.logger.info("[FE:MODE]: from DYNAMIC_INITIALIZATION to NOMINAL")

        nav_state: NavState = self.pim.predict(self.nav_from_state(), self.bias_from_state())

        pose_estimate = (
            nav_state.pose().matrix()
            if self.estimation_mode == PredictionMode.PIM
            else SE3.from_flat_ndarray(self.vo_state[:7]).as_matrix()
        )

        (
            ctx.set_ndarray("points", current_points)
            .set_scalar("front_end_mode", self.mode.value)
            .set_scalar("frontend_bootstrap_decision", bootstrap_result.decision.value)
            .set_scalar("frontend_bootstrap_ready", float(bootstrap_result.ready))
            .set_scalar("frontend_bootstrap_confidence", bootstrap_result.confidence)
            .set_scalar("frontend_bootstrap_window_frames", bootstrap_result.window_metrics.frame_count)
            .set_scalar("frontend_bootstrap_window_duration_sec", bootstrap_result.window_metrics.duration_sec)
            .set_scalar(
                "frontend_bootstrap_median_parallax_px",
                bootstrap_result.window_metrics.median_temporal_parallax_px,
            )
            .set_scalar(
                "frontend_bootstrap_p90_parallax_px",
                bootstrap_result.window_metrics.p90_temporal_parallax_px,
            )
            .set_scalar(
                "frontend_bootstrap_median_good_features",
                bootstrap_result.window_metrics.median_good_feature_count,
            )
            .set_scalar(
                "frontend_bootstrap_median_triangulated_features",
                bootstrap_result.window_metrics.median_triangulated_feature_count,
            )
            .set_scalar(
                "frontend_bootstrap_median_stereo_ok_ratio",
                bootstrap_result.window_metrics.median_stereo_ok_ratio,
            )
            .set_scalar(
                "frontend_bootstrap_median_gyro_std_norm",
                bootstrap_result.window_metrics.median_gyro_std_norm,
            )
            .set_scalar(
                "frontend_bootstrap_median_accel_norm_std",
                bootstrap_result.window_metrics.median_accel_norm_std,
            )
            .set_scalar("points_size", current_points.shape[0])
            .set_record_batch("keyframe_metrics", select_metrics.as_arrow())
            .set_ndarray("cam0_in_body", self.vio_ctx.stereo.cam0_in_body_se3.as_matrix())
            .set_ndarray("pim_pose", nav_state.pose().matrix())
            .set_ndarray("pim_velocity", nav_state.velocity())
            .set_ndarray("pnp_pose", SE3.from_flat_ndarray(self.vo_state[:7]).as_matrix())
            .set_ndarray("pnp_velocity", self.vo_state[7:10])
            .set_ndarray("pose_estimate", pose_estimate)
        )

        if len(keyframes) > 0:
            # need to push array of keyframes to the ctx
            self.logger.info(f"[FE:KF_LIST]: {keyframes}")
            keyframe_ctx = ctx.reassemble().set_record_batch("keyframes", KF.to_record_batch(keyframes))
            self.reset_pim(timestamp, nav_state)
            send_pipeline_context_output(self.node, "keyframes", keyframe_ctx, metadata)

        return ctx

    def update_bootstrap(
        self,
        frame_id: int,
        timestamp: float,
        current_frame: FeatureFrame,
        current_points: NDArray[np.float32],
        ctx: Ctx,
    ) -> FrontEndBootstrapResult:
        """Update the frontend bootstrap observer and return its current result."""
        imu_rows = ctx.get_scalar("imu_rows", int)
        accel = ctx.get_ndarray("accel", (imu_rows, 3))
        gyro = ctx.get_ndarray("gyro", (imu_rows, 3))
        imu_ts = ctx.get_ndarray("imu_ts", (imu_rows,))
        bootstrap_input = FrontEndBootstrapInput.from_frame_and_imu(
            frame_id=frame_id,
            timestamp_ns=timestamp,
            frame=current_frame,
            temporal_parallax_px=self.ft.metrics.temporal_pixel_displacement,
            temporal_parallax_p90_px=self.ft.metrics.temporal_pixel_displacement_p90,
            accel=accel,
            gyro=gyro,
            imu_timestamps_ns=imu_ts,
            triangulated_points=current_points,
        )
        result = self.bootstrap.feed(bootstrap_input)
        self.logger.debug(
            "[FE:BOOTSTRAP]: "
            f"frame={frame_id}, decision={result.decision.name}, ready={result.ready}, "
            f"confidence={result.confidence:.3f}, reasons={result.reasons}, "
            f"window_frames={result.window_metrics.frame_count}, "
            f"median_parallax={result.window_metrics.median_temporal_parallax_px:.3f}, "
            f"p90_parallax={result.window_metrics.p90_temporal_parallax_px:.3f}, "
            f"median_good_features={result.window_metrics.median_good_feature_count:.1f}"
        )
        return result

    def estimate_pnp_pose(self, timestamp: float, good_features: NDArray[np.float32]) -> None:
        """Estimate the PnP pose."""
        last_vo_timestamp = self.vo_state[10].copy()
        last_vo_vector = self.vo_state[4:7].copy()
        successed, reason, pnp_pose = self.pnp_pose_tracker.find_pose(good_features, self.local_map)
        if not successed:
            self.logger.warning(f"[PNP]: PnP pose estimation failed: {reason}")
            return
        pnp_pose_array = pnp_pose.as_flat_ndarray()
        self.vo_state[:7] = pnp_pose_array[:7]
        self.vo_state[10] = timestamp
        dt_sec = (timestamp - last_vo_timestamp) / 1e9
        pnp_velocity = (pnp_pose_array[4:7] - last_vo_vector) / dt_sec
        self.vo_state[7:10] = pnp_velocity
        self.logger.trace(
            f"[PNP]: pnp pose set to pnp_pose: {pnp_pose}, current_vel: {pnp_velocity}, dt: {dt_sec}"
        )

    def process_image(self, frame_id: int, ctx: Ctx) -> bool:
        """Process the image data."""
        width = ctx.get_scalar("width")
        height = ctx.get_scalar("height")
        left = ctx.get_image("left", (height, width))
        right = ctx.get_image("right", (height, width))
        timestamp = ctx.get_scalar("timestamp")
        left, right = self.camera_model.process_stereo(left, right)

        self.ft.feed(timestamp, (left, right))
        zero_velocity_state = self.ft.metrics.zero_velocity_state
        its_time_to_dynamic_init = (
            self.mode == FrontEndMode.ZERO_MOTION_INITIALIZATION
            and zero_velocity_state == ZeroVelocityTrackerState.NON_ZERO_VELOCITY
        )

        if its_time_to_dynamic_init:
            self.mode = FrontEndMode.DYNAMIC_INITIALIZATION
            self.logger.info("[FE:MODE]: from ZERO_MOTION_INITIALIZATION to DYNAMIC_INITIALIZATION")

        (
            ctx.set_scalar("frame_id", frame_id)
            .set_image("left_rect", left)
            .set_image("right_rect", right)
            .set_record_batch("active_feat", self.ft.tensor.as_arrow())
            .set_scalar("features_count", self.ft.metrics.active_count)
            .set_scalar("all_features_count", self.ft.metrics.active_count)
            .set_scalar("good_features_count", self.ft.metrics.good_count)
            .set_scalar("lost_features_count", self.ft.metrics.lost_count)
            .set_scalar("stereo_ok_count", self.ft.metrics.stereo_ok_count)
            .set_scalar("stereo_ok_ratio", self.ft.metrics.stereo_ok_ratio)
            .set_scalar("inner_frame_median_disparity", self.ft.metrics.temporal_pixel_displacement)
            .set_scalar("inner_frame_p90_disparity", self.ft.metrics.temporal_pixel_displacement_p90)
            .set_scalar("zero_velocity_state", zero_velocity_state)
        )
        return its_time_to_dynamic_init

    def process_imu_data(self, sensor_ctx: Ctx) -> bool:
        """Process the IMU data and update the mode."""
        imu_rows = sensor_ctx.get_scalar("imu_rows", int)
        accel = sensor_ctx.get_ndarray("accel", (imu_rows, 3))
        gyro = sensor_ctx.get_ndarray("gyro", (imu_rows, 3))
        imu_ts = sensor_ctx.get_ndarray("imu_ts", (imu_rows,))
        self.batch_integrate(accel, gyro, imu_ts)
        match self.mode:
            case FrontEndMode.SILENT_AWAIT:
                self.quiet_imu_buffer.add_batch(accel, gyro, imu_ts)
                if not self.ft.iterator_count > 1:
                    return False
                self.mode = FrontEndMode.VIBRATION_AWAIT
                self.quiet_statistics["mean"] = np.mean(self.quiet_imu_buffer.get_6d(), axis=0)
                self.quiet_statistics["var"] = np.var(self.quiet_imu_buffer.get_6d(), axis=0)
                self.logger.info("[FE:MODE]: from SILENT_AWAIT to VIBRATION_AWAIT")

            case FrontEndMode.VIBRATION_AWAIT:
                current_batch = np.column_stack((accel, gyro))
                current_mean = np.mean(current_batch, axis=0)
                carent_var = np.var(current_batch, axis=0)
                diff = current_mean - self.quiet_statistics["mean"]
                d2 = np.sum((diff**2) / self.quiet_statistics["var"])
                vibration_ratio = np.max(carent_var / self.quiet_statistics["var"])
                vibration_detected = (
                    d2 > self.six_dof_quiet_threshold or vibration_ratio > self.quiet_ratio_threshold
                )
                if not vibration_detected:
                    self.quiet_imu_buffer.add_batch(accel, gyro, imu_ts)
                    self.quiet_statistics["mean"] = np.mean(self.quiet_imu_buffer.get_6d(), axis=0)
                    self.quiet_statistics["var"] = np.var(self.quiet_imu_buffer.get_6d(), axis=0)
                    return False

                # this moment should be the first keyframe
                initial_state = self.quiet_imu_buffer.create_initial_state()
                self.state[:4] = initial_state.rotation.as_quat()
                self.vo_state[:4] = self.state[:4]
                self.state[10:13] = initial_state.accel_bias
                self.state[13:16] = initial_state.gyro_bias
                return True
        return False

    def batch_integrate(self, accel_batch: np.ndarray, gyro_batch: np.ndarray, imu_ts_batch: np.ndarray) -> None:
        """Batch integrate the IMU data."""
        self.logger.trace("pim integrate batch")
        self.imu_buffer.add_batch(accel_batch, gyro_batch, imu_ts_batch)
        for accel, gyro, dt in self.imu_buffer.iterate_last_batch():
            self.pim.integrateMeasurement(accel, gyro, dt)

    def reset_pim(self, timestamp: float, nav_state: gtsam.NavState) -> None:
        """Reset the pim."""
        imu_buffer_info = self.imu_buffer.info()
        self.logger.info(
            f"[FE:PIM]: reset pim and imu buffer: accel_bias: {self.state[10:13]} gyro_bias: {self.state[13:16]} "
            f"first buffer ts: {imu_buffer_info.first_buffer_ts:.0f} "
            f"last buffer ts: {imu_buffer_info.last_buffer_ts:.0f} "
            f"buffer size: {imu_buffer_info.size}"
        )
        self.pim.resetIntegration()
        self.imu_buffer.reset(timestamp)
        new_state = self.nav_state_to_flat_ndarray(nav_state)
        self.state[:10] = new_state[:10]

    def apply_new_bias_and_reintegrate(self, bias: gtsam.imuBias.ConstantBias) -> None:
        """Apply the new bias and reintegrate the IMU data."""
        self.pim.resetIntegrationAndSetBias(bias)
        self.state[10:13] = bias.accelerometer()
        self.state[13:16] = bias.gyroscope()
        # need to again reintegrate the IMU buffer

        buffer_info = self.imu_buffer.info()
        if buffer_info.size == 0:
            self.logger.warning("[FE:PIM]: no IMU data to reintegrate")
        else:
            first_buffer_ts = buffer_info.first_buffer_ts
            last_buffer_ts = buffer_info.last_buffer_ts
            msg = f"[FE:PIM]: Reintegrating buffer from {first_buffer_ts:.0f} to {last_buffer_ts:.0f}"
            self.logger.info(msg)
        for accel, gyro, dt in self.imu_buffer.iterate_full_buffer():
            self.pim.integrateMeasurement(accel, gyro, dt)

    @on_input("feedback")
    def handle_backend_feedback(self, ctx: Ctx) -> None:
        """Handle the backend feedback event."""
        points_size = int(ctx.get_scalar("optimized_points_size"))
        points = ctx.get_ndarray("optimized_points", (points_size, 5))
        actual_bias = ctx.get_ndarray("actual_bias", (6,))
        pose_matrix = ctx.get_ndarray("pose_matrix", (4, 4))
        actual_velocity = ctx.get_ndarray("optimized_velocity", (3,))
        self.estimation_mode = PredictionMode(ctx.get_scalar("prediction_mode"))
        pose = SE3.from_matrix(pose_matrix)
        self.state[:4] = pose.rotation().as_quat()
        self.state[4:7] = pose.translation()
        self.state[7:10] = actual_velocity
        self.local_map.add_ndarray(points)
        self.logger.info(
            f"[FE:FEEDBACK_LOOP]: added {points_size} points to the local map"
            f"bias: {actual_bias}, pose: {pose} "
            f"mode: {self.estimation_mode}"
        )
        bias = gtsam.imuBias.ConstantBias(actual_bias[:3], actual_bias[3:])
        self.apply_new_bias_and_reintegrate(bias)

    @on_input("helthcheck")
    def handle_helthcheck(self) -> None:
        """Handle the helthcheck event."""
        self.logger.trace("Still alive")

    @on_stop
    def handle_shutdown(self) -> None:
        """Handle the shutdown event."""
        self.logger.info("VIO frontend stopping...")

    def nav_from_state(self) -> gtsam.NavState:
        """Get the navigation state from the state."""
        quat = self.state[:4]
        t = self.state[4:7]
        v = self.state[7:10]
        # print(f"[NAV]: quat: {quat}, t: {t}, v: {v}")
        rot = Rotation.from_quat(quat)
        pose = gtsam.Pose3(gtsam.Rot3(rot.as_matrix()), t)
        return gtsam.NavState(pose, v)

    def nav_state_to_flat_ndarray(self, nav_state: gtsam.NavState) -> np.ndarray:
        """Convert the navigation state to a flat numpy array."""
        quat = Rotation.from_matrix(nav_state.pose().rotation().matrix()).as_quat()
        vec = nav_state.pose().translation()
        vel = nav_state.velocity()
        return np.array([quat[0], quat[1], quat[2], quat[3], vec[0], vec[1], vec[2], vel[0], vel[1], vel[2]])

    def bias_from_state(self) -> gtsam.imuBias.ConstantBias:
        """Get the bias from the state."""
        return gtsam.imuBias.ConstantBias(self.state[10:13], self.state[13:16])


if __name__ == "__main__":
    VIOFrontend(camera_model=VIOFrontend.create_stereo_camera_model(), vio_ctx=VIOFrontend.create_vio_ctx()).run()
