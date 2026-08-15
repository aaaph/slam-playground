from enum import IntEnum

import gtsam
import numpy as np
from dora import Node
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.camera_model.vio_context import VioContext
from core.feature_tracker.feature_tracker import FeatureTracker, FeatureTrackerMode
from core.feature_tracker.zero_velocity_tracker import ZeroVelocityTrackerState
from core.front_end.feature_manager import FeatureManager
from core.front_end.front_end_bootstrap import FrontEndBootstrap, FrontEndBootstrapDecision
from core.front_end.front_end_estimates import FrontEndPoseEstimates, MotionEstimate
from core.front_end.keyframe import KF
from core.front_end.keyframe_selector import (
    KeyframeSelector,
    KeyFrameSelectThresholds,
    SelectMetrics,
    SelectReason,
)
from core.graph_optimizer.optimizer_types import PredictionMode
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema, StereoTriangulationStatus
from core.pose_tracker.frame_to_frame_pnp_estimator import FrameToFramePnPEstimator
from core.pose_tracker.frame_to_frame_pnp_store import PnPMapSchema
from core.pose_tracker.inertial_integration import ImuBatch, ImuBuffer
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger
from pipeline.annotations import Ctx, Metadata
from pipeline.decorators import handle, on_input, on_stop, reactive, send_pipeline_context_output
from pipeline.nodes.base import PipelineNode


class FrontEndMode(IntEnum):
    """Front end mode."""

    BOOTSTRAP = 0
    NOMINAL = 1


@reactive
class VIOFrontend(PipelineNode):
    """VIO frontend."""

    def __init__(self, camera_model: StereoCameraModel, vio_ctx: VioContext) -> None:
        """Initialize the VIO frontend."""
        self.node = Node()
        self.mode = FrontEndMode.BOOTSTRAP
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
        self.pnp_estimator = FrameToFramePnPEstimator.default_factory(self.vio_ctx.stereo)
        self.feature_manager = FeatureManager.from_stereo_camera_ctx(self.vio_ctx.stereo)
        self.bootstrap = FrontEndBootstrap()
        self.bootstrap_outcome: FrontEndBootstrapDecision | None = None
        self.kf_selector = KeyframeSelector.from_thresholds(
            KeyFrameSelectThresholds(min_parallax_pts=50, max_time_delta_sec=3.0)
        )
        self.imu_buffer = ImuBuffer(capacity=10000)
        self.state = np.zeros(16, dtype=np.float32)  # quat(4) + t(3) + v(3) + ba(3) + bg(3) = 16
        self.state[:4] = Rotation.identity().as_quat()

        self.vo_state = np.zeros(11, dtype=np.float64)  # quat(4) + t(3) + v(3) + ts(1) = 11
        self.vo_state[:4] = Rotation.identity().as_quat()
        self.pim = gtsam.PreintegratedImuMeasurements(
            self.vio_ctx.imu.pim_params(), gtsam.imuBias.ConstantBias(self.state[10:13], self.state[13:16])
        )

    @handle("sensor_frame", "frame")
    def handle_sensor_frame(self, ctx: Ctx, metadata: Metadata) -> Ctx:
        """Handle the sensor frame event."""
        frame_id = self.ft.iterator_count
        timestamp = ctx.get_scalar("timestamp")

        tracking_mask, tracked_frame = self.process_image(ctx)
        imu_batch = self.process_imu_data(ctx)

        stereo_mask, stereo_frame = self.feature_manager.triangulate_active_track(tracked_frame, tracking_mask)

        self.process_bootstrap(frame_id, timestamp, stereo_frame, imu_batch)

        self.vo_state[:] = self.estimate_pnp_pose(timestamp, stereo_frame, tracking_mask)
        poses_estimates = self.get_poses_estimates()
        tracked_stereo_frame = stereo_frame[tracking_mask]
        keyframe_state = self.state.copy()
        keyframe_state[:4] = poses_estimates.selected.pose.rotation().as_quat()
        keyframe_state[4:7] = poses_estimates.selected.pose.translation()
        keyframe_state[7:10] = poses_estimates.selected.velocity
        keyframes, kf_metrics = self.select_keyframes(
            frame_id,
            timestamp,
            tracked_stereo_frame,
            keyframe_state,
        )

        stereo_points = self.build_stereo_points_for_visualization(stereo_mask, stereo_frame)
        (
            ctx.set_ndarray("stereo_points", stereo_points)
            .set_scalar("stereo_points_size", stereo_points.shape[0])
            .set_scalar("front_end_mode", self.mode.value)
            .set_record_batch("keyframe_metrics", kf_metrics.as_arrow())
            .set_ndarray("cam0_in_body", self.vio_ctx.stereo.cam0_in_body_se3.as_matrix())
            .set_ndarray("pim_pose", poses_estimates.pim.pose_matrix())
            .set_ndarray("pim_velocity", poses_estimates.pim.velocity)
            .set_ndarray("pnp_pose", poses_estimates.pnp.pose_matrix())
            .set_ndarray("pnp_velocity", poses_estimates.pnp.velocity)
            .set_ndarray("pose_estimate", poses_estimates.selected.pose_matrix())
        )

        if len(keyframes) > 0:
            # need to push array of keyframes to the ctx
            self.logger.info(f"[FE:KF_LIST]: {keyframes}")
            keyframe_ctx = ctx.reassemble().set_record_batch("keyframes", KF.to_record_batch(keyframes))
            self.state[:4] = poses_estimates.selected.pose.rotation().as_quat()
            self.state[4:7] = poses_estimates.selected.pose.translation()
            self.state[7:10] = poses_estimates.selected.velocity
            self.reset_pim(timestamp)
            send_pipeline_context_output(self.node, "keyframes", keyframe_ctx, metadata)

        return ctx

    def select_keyframes(
        self,
        frame_id: int,
        timestamp: float,
        stereo_frame: NDArray[np.float32],
        keyframe_state: NDArray[np.float32],
    ) -> tuple[list[KF], SelectMetrics]:
        """Select at most one keyframe after bootstrap has committed initialization."""
        select_metrics = SelectMetrics.zero(self.kf_selector.thresholds)

        if self.mode != FrontEndMode.NOMINAL:
            return [], select_metrics

        if not self.kf_selector.initialized:
            if self.bootstrap_outcome is None:
                msg = "NOMINAL frontend has no committed bootstrap outcome"
                raise RuntimeError(msg)
            non_zero_velocity_detected = (
                self.ft.metrics.zero_velocity_state == ZeroVelocityTrackerState.NON_ZERO_VELOCITY
            )
            keyframe = KF(
                keyframe_id=frame_id,
                timestamp=timestamp,
                select_reasons=[SelectReason.INITIALIZED],
                state=keyframe_state.copy(),
                imu_batch=self.imu_buffer.buffer[: self.imu_buffer.size, :].copy(),
                stereo_frame=stereo_frame.copy(),
                non_zero_velocity_detected=non_zero_velocity_detected,
            )
            self.kf_selector.set_new_keyframe(timestamp, stereo_frame)
            self.kf_selector.initialize()
            return [keyframe], select_metrics

        good_kf, select_reasons, select_metrics = self.kf_selector.check(timestamp, stereo_frame)
        if not good_kf:
            return [], select_metrics
        non_zero_velocity_detected = (
            self.ft.metrics.zero_velocity_state == ZeroVelocityTrackerState.NON_ZERO_VELOCITY
        )
        keyframe = KF(
            keyframe_id=frame_id,
            timestamp=timestamp,
            select_reasons=select_reasons,
            state=keyframe_state.copy(),
            imu_batch=self.imu_buffer.buffer[: self.imu_buffer.size, :].copy(),
            stereo_frame=stereo_frame.copy(),
            non_zero_velocity_detected=non_zero_velocity_detected,
        )
        self.kf_selector.set_new_keyframe(timestamp, stereo_frame)
        return [keyframe], select_metrics

    def process_bootstrap(
        self,
        frame_id: int,
        timestamp: float,
        stereo_frame: NDArray[np.float32],
        imu_batch: ImuBatch,
    ) -> None:
        """Process bootstrap evidence and commit frontend initialization."""
        if self.mode != FrontEndMode.BOOTSTRAP:
            return

        self.bootstrap.feed(frame_id, timestamp, stereo_frame, self.ft.metrics, imu_batch)
        result = self.bootstrap.evaluate()
        if result.initial_rotation is not None:
            rotation = result.initial_rotation
            self.logger.info(f"[FE:BOOTSTRAP:INIT_ROTATION]: {rotation.as_quat()}")
            quat = rotation.as_quat()
            self.state[:4] = quat.copy()
            self.vo_state[:4] = quat.copy()

        if result.gyro_bias is not None:
            bias = gtsam.imuBias.ConstantBias(np.zeros(3), result.gyro_bias)
            self.logger.info(f"[FE:BOOTSTRAP:GYRO_BIAS]: {result.gyro_bias}")
            self.apply_new_bias_and_reintegrate(bias)

        if result.decision == FrontEndBootstrapDecision.UNKNOWN:
            return

        self.bootstrap_outcome = result.decision
        self.mode = FrontEndMode.NOMINAL
        self.logger.info(f"[FE:BOOTSTRAP:DECISION]: {result.decision.name}")
        self.bootstrap.commit(timestamp)

    def get_poses_estimates(self) -> FrontEndPoseEstimates:
        """Get the poses from the state."""
        nav_state = self.pim.predict(self.nav_from_state(), self.bias_from_state())

        pim_pose = SE3.from_matrix(nav_state.pose().matrix())
        pim_velocity = nav_state.velocity()
        pim_estimate = MotionEstimate(pim_pose, pim_velocity)

        pnp_pose = SE3.from_flat_ndarray(self.vo_state[:7])
        pnp_velocity = self.vo_state[7:10]
        pnp_estimate = MotionEstimate(pnp_pose, pnp_velocity)

        return FrontEndPoseEstimates(pim_estimate, pnp_estimate, self.estimation_mode)

    def estimate_pnp_pose(
        self, timestamp: float, active_points: NDArray[np.float32], tracking_mask: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        """Estimate the PnP pose."""
        next_state = np.zeros(11, dtype=np.float64)

        visual_points = active_points[tracking_mask]
        visual_features = np.full((visual_points.shape[0], PnPMapSchema.count()), np.nan, dtype=np.float32)
        visual_features[:, PnPMapSchema.FEAT_ID] = visual_points[:, StereoTriangulationSchema.FEAT_ID]
        visual_features[:, PnPMapSchema.XYZ] = visual_points[:, StereoTriangulationSchema.XYZ]
        visual_features[:, PnPMapSchema.LEFT_U] = visual_points[:, StereoTriangulationSchema.LEFT_U]
        visual_features[:, PnPMapSchema.LEFT_V] = visual_points[:, StereoTriangulationSchema.LEFT_V]
        visual_features[:, PnPMapSchema.RIGHT_U] = visual_points[:, StereoTriangulationSchema.RIGHT_U]
        visual_features[:, PnPMapSchema.RIGHT_V] = visual_points[:, StereoTriangulationSchema.RIGHT_V]

        bad_stereo_mask = (
            visual_points[:, StereoTriangulationSchema.STEREO_STATUS] == StereoTriangulationStatus.BAD_STEREO.value
        )
        visual_features[bad_stereo_mask, PnPMapSchema.RIGHT_UV] = np.nan

        prev_pose_se3 = SE3.from_flat_ndarray(self.vo_state[:7])
        prev_timestamp = float(self.vo_state[10].copy())
        next_pose_se3 = self.pnp_estimator.estimate_pose(prev_pose_se3, visual_features.astype(np.float64))

        next_state[:7] = next_pose_se3.as_flat_ndarray()
        dt_sec = (timestamp - prev_timestamp) / 1e9

        pnp_velocity = (next_pose_se3.translation() - prev_pose_se3.translation()) / dt_sec
        next_state[7:10] = pnp_velocity
        next_state[10] = timestamp
        return next_state.astype(np.float64)

    @staticmethod
    def build_stereo_points_for_visualization(
        stereo_mask: NDArray[np.bool_],
        stereo_frame: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Build cam0-frame one-shot stereo points for Rerun pointcloud visualization."""
        one_shot_stereo_points = stereo_frame[stereo_mask]
        stereo_points = np.full((one_shot_stereo_points.shape[0], 4), np.nan, dtype=np.float32)
        if one_shot_stereo_points.shape[0] == 0:
            return stereo_points

        stereo_points[:, 0] = one_shot_stereo_points[:, StereoTriangulationSchema.FEAT_ID]
        stereo_points[:, 1:4] = one_shot_stereo_points[:, StereoTriangulationSchema.XYZ]
        return stereo_points

    def process_image(self, ctx: Ctx) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
        """Process the image data."""
        frame_id = self.ft.iterator_count
        width = ctx.get_scalar("width")
        height = ctx.get_scalar("height")
        left = ctx.get_image("left", (height, width))
        right = ctx.get_image("right", (height, width))
        timestamp = ctx.get_scalar("timestamp")
        left, right = self.camera_model.process_stereo(left, right)

        tracking_mask, active_features = self.ft.feed(timestamp, (left, right))

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
            .set_scalar("zero_velocity_state", self.ft.metrics.zero_velocity_state)
        )
        return tracking_mask, active_features

    def process_imu_data(self, sensor_ctx: Ctx) -> ImuBatch:
        """Process the IMU data and update the mode."""
        imu_rows = sensor_ctx.get_scalar("imu_rows", int)
        accel_batch = sensor_ctx.get_ndarray("accel", (imu_rows, 3))
        gyro_batch = sensor_ctx.get_ndarray("gyro", (imu_rows, 3))
        imu_ts_batch = sensor_ctx.get_ndarray("imu_ts", (imu_rows,))
        self.imu_buffer.add_batch(accel_batch, gyro_batch, imu_ts_batch)
        for accel, gyro, dt in self.imu_buffer.iterate_last_batch():
            self.pim.integrateMeasurement(accel, gyro, dt)
        return self.imu_buffer.get_last_batch()

    def reset_pim(self, timestamp: float) -> None:
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
        actual_bias = ctx.get_ndarray("actual_bias", (6,))
        pose_matrix = ctx.get_ndarray("pose_matrix", (4, 4))
        vo_pose_correction = SE3.from_matrix(ctx.get_ndarray("vo_pose_correction", (4, 4)))
        actual_velocity = ctx.get_ndarray("optimized_velocity", (3,))
        estimation_mode = PredictionMode(ctx.get_scalar("prediction_mode"))
        self.estimation_mode = estimation_mode
        pose = SE3.from_matrix(pose_matrix)
        self.state[:4] = pose.rotation().as_quat()
        self.state[4:7] = pose.translation()
        self.state[7:10] = actual_velocity
        corrected_vo_pose = vo_pose_correction * SE3.from_flat_ndarray(self.vo_state[:7])
        self.vo_state[:7] = corrected_vo_pose.as_flat_ndarray()
        self.vo_state[7:10] = vo_pose_correction.rotation().apply(self.vo_state[7:10])
        self.logger.info(f"[FE:FEEDBACK_LOOP]: bias: {actual_bias}, pose: {pose} mode: {self.estimation_mode}")
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

    def bias_from_state(self) -> gtsam.imuBias.ConstantBias:
        """Get the bias from the state."""
        return gtsam.imuBias.ConstantBias(self.state[10:13], self.state[13:16])


if __name__ == "__main__":
    VIOFrontend(camera_model=VIOFrontend.create_stereo_camera_model(), vio_ctx=VIOFrontend.create_vio_ctx()).run()
