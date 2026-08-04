from enum import IntEnum

import gtsam
import numpy as np
from dora import Node
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from scipy.stats import chi2

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.camera_model.vio_context import VioContext
from core.feature_tracker.feature_tracker import FeatureTracker, FeatureTrackerMode
from core.front_end.feature_manager import FeatureManager
from core.front_end.front_end_bootstrap import FrontEndBootstrap
from core.front_end.front_end_estimates import FrontEndPoseEstimates, MotionEstimate
from core.front_end.keyframe import KF
from core.front_end.keyframe_selector import KeyframeSelector, KeyFrameSelectThresholds
from core.front_end.landmark_initialization import (
    LandmarkFeatureFrame,
    LandmarkInitialization,
    LandmarkInitializationFrameSchema,
)
from core.graph_optimizer.optimizer_types import PredictionMode
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema, StereoTriangulationStatus
from core.pose_tracker.frame_to_frame_pnp_estimator import FrameToFramePnPEstimator
from core.pose_tracker.frame_to_frame_pnp_store import PnPMapSchema
from core.pose_tracker.inertial_integration import ImuBuffer
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger
from pipeline.annotations import Ctx, Metadata
from pipeline.decorators import handle, on_input, on_stop, reactive, send_pipeline_context_output
from pipeline.nodes.base import PipelineNode


class FrontEndMode(IntEnum):
    """Front end mode."""

    # SILENT_AWAIT = 0
    # VIBRATION_AWAIT = 1
    # ZERO_MOTION_INITIALIZATION = 2
    # DYNAMIC_INITIALIZATION = 3
    NOMINAL = 4
    BOOTSTRAP = 5


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
        self.kf_selector = KeyframeSelector.from_thresholds(
            KeyFrameSelectThresholds(min_parallax_pts=50, max_time_delta_sec=3.0)
        )
        self.quiet_imu_buffer = ImuBuffer(capacity=1000)
        self.quiet_statistics = {"mean": None, "var": None}
        self.six_dof_quiet_threshold = chi2.ppf(0.95, 6)
        self.quiet_ratio_threshold = 10.0

        self.imu_buffer = ImuBuffer(capacity=10000)
        self.state = np.zeros(16, dtype=np.float32)  # quat(4) + t(3) + v(3) + ba(3) + bg(3) = 16
        self.state[:4] = Rotation.identity().as_quat()

        self.vo_state = np.zeros(11, dtype=np.float64)  # quat(4) + t(3) + v(3) + ts(1) = 11
        self.vo_state[:4] = Rotation.identity().as_quat()
        self.pim = gtsam.PreintegratedImuMeasurements(
            self.vio_ctx.imu.pim_params(), gtsam.imuBias.ConstantBias(self.state[10:13], self.state[13:16])
        )
        self.landmark_init = LandmarkInitialization.default_factory(self.vio_ctx.stereo)

    @handle("sensor_frame", "frame")
    def handle_sensor_frame(self, ctx: Ctx, metadata: Metadata) -> Ctx:
        """Handle the sensor frame event."""
        frame_id = self.ft.iterator_count
        timestamp = ctx.get_scalar("timestamp")

        tracking_mask, tracked_frame = self.process_image(frame_id, ctx)
        _vibration_in_static_detected = self.process_imu_data(ctx)

        stereo_mask, stereo_frame = self.feature_manager.triangulate_active_track(tracked_frame, tracking_mask)

        if self.mode == FrontEndMode.BOOTSTRAP:
            metrics = self.ft.metrics
            imu_batch = self.imu_buffer.get_last_batch()
            bootstrap_result = self.bootstrap.feed(frame_id, timestamp, metrics, imu_batch)

            if bootstrap_result.rotation_ready:
                self.state[:4] = bootstrap_result.rotation_quat.copy()
                self.vo_state[:4] = self.state[:4].copy()
                self.logger.info(f"[FE:BOOTSTRAP]: set rotation to {bootstrap_result.rotation_quat}")

        self.vo_state[:] = self.estimate_pnp_pose(timestamp, stereo_frame, tracking_mask)
        poses_estimates = self.get_poses_estimates()
        cam0_in_world = poses_estimates.selected.pose * self.vio_ctx.stereo.cam0_in_body_se3
        landmark_mask, landmark_frame = self.landmark_init.apply_observation_frame(
            cam0_in_world.as_matrix(),
            tracking_mask,
            stereo_frame,
        )
        keyframes: list[KF] = []
        """
        if vibration_in_static_detected:
            # shoube be a first keyframe in factor graph
            # self.mode = FrontEndMode.ZERO_MOTION_INITIALIZATION
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
            # self.mode = FrontEndMode.DYNAMIC_INITIALIZATION
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
        """
        _good_kf, _select_reasons, select_metrics = self.kf_selector.check(timestamp, stereo_frame[tracking_mask])

        """ if good_kf:
            kf_state = self.state.copy()
            kf_state[:10] = self.vo_state[:10]
            kf = KF(
                keyframe_id=frame_id,
                timestamp=timestamp,
                select_reasons=select_reasons,
                state=kf_state,
                imu_batch=self.imu_buffer.buffer[: self.imu_buffer.size, :].copy(),
                active_track=active_track,
                non_zero_velocity_detected=False,
            )
            keyframes.append(kf)
            self.kf_selector.set_new_keyframe(timestamp, current_frame.good_features())
            # if self.mode == FrontEndMode.DYNAMIC_INITIALIZATION:
            #    self.kf_selector.switch_thresholds(KeyFrameSelectThresholds())
            #    self.mode = FrontEndMode.NOMINAL
            #    self.logger.info("[FE:MODE]: from DYNAMIC_INITIALIZATION to NOMINAL") """

        stereo_points = self.build_stereo_points_for_visualization(stereo_mask, stereo_frame)
        landmarks = self.build_landmarks_for_visualization(landmark_mask, landmark_frame, cam0_in_world)
        (
            ctx.set_ndarray("stereo_points", stereo_points)
            .set_scalar("stereo_points_size", stereo_points.shape[0])
            .set_ndarray("initialized_landmarks", landmarks)
            .set_scalar("initialized_landmarks_size", landmarks.shape[0])
            .set_scalar("front_end_mode", self.mode.value)
            .set_record_batch("keyframe_metrics", select_metrics.as_arrow())
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
            self.commit_pim_estimate(poses_estimates.selected)
            self.reset_pim(timestamp)
            send_pipeline_context_output(self.node, "keyframes", keyframe_ctx, metadata)

        return ctx

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
    ) -> NDArray[np.float32]:
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
        next_pose_se3 = self.pnp_estimator.estimate_pose(prev_pose_se3, visual_features)

        next_state[:7] = next_pose_se3.as_flat_ndarray()
        dt_sec = (timestamp - prev_timestamp) / 1e9

        pnp_velocity = (next_pose_se3.translation() - prev_pose_se3.translation()) / dt_sec
        next_state[7:10] = pnp_velocity
        next_state[10] = timestamp
        return next_state

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

    @staticmethod
    def build_landmarks_for_visualization(
        success_mask: NDArray[np.bool_],
        landmark_frame: LandmarkFeatureFrame,
        cam0_in_world: SE3,
    ) -> NDArray[np.float32]:
        """Build cam0-frame cached landmarks selected by the frame-aligned success mask."""
        visualizable_landmarks = landmark_frame[success_mask]
        landmarks = np.full((visualizable_landmarks.shape[0], 4), np.nan, dtype=np.float32)
        if visualizable_landmarks.shape[0] == 0:
            return landmarks

        landmarks[:, 0] = visualizable_landmarks[:, LandmarkInitializationFrameSchema.FEAT_ID]
        world_xyz = visualizable_landmarks[:, LandmarkInitializationFrameSchema.LANDMARK_XYZ].astype(
            np.float64, copy=False
        )
        cam0_from_world = cam0_in_world.inverse()
        landmarks[:, 1:4] = (cam0_from_world.rotation().apply(world_xyz) + cam0_from_world.translation()).astype(
            np.float32, copy=False
        )
        return landmarks

    def process_image(self, frame_id: int, ctx: Ctx) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
        """Process the image data."""
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

    def process_imu_data(self, sensor_ctx: Ctx) -> bool:
        """Process the IMU data and update the mode."""
        imu_rows = sensor_ctx.get_scalar("imu_rows", int)
        accel = sensor_ctx.get_ndarray("accel", (imu_rows, 3))
        gyro = sensor_ctx.get_ndarray("gyro", (imu_rows, 3))
        imu_ts = sensor_ctx.get_ndarray("imu_ts", (imu_rows,))
        self.batch_integrate(accel, gyro, imu_ts)
        """ match self.mode:
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
                return True """
        return False

    def batch_integrate(self, accel_batch: np.ndarray, gyro_batch: np.ndarray, imu_ts_batch: np.ndarray) -> None:
        """Batch integrate the IMU data."""
        self.logger.trace("pim integrate batch")
        self.imu_buffer.add_batch(accel_batch, gyro_batch, imu_ts_batch)
        for accel, gyro, dt in self.imu_buffer.iterate_last_batch():
            self.pim.integrateMeasurement(accel, gyro, dt)

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

    def commit_pim_estimate(self, pim_estimate: MotionEstimate) -> None:
        """Commit the pim estimate."""
        self.state[:4] = pim_estimate.pose.rotation().as_quat()
        self.state[4:7] = pim_estimate.pose.translation()
        self.state[7:10] = pim_estimate.velocity

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
        _points = ctx.get_ndarray("optimized_points", (points_size, 5))
        actual_bias = ctx.get_ndarray("actual_bias", (6,))
        pose_matrix = ctx.get_ndarray("pose_matrix", (4, 4))
        actual_velocity = ctx.get_ndarray("optimized_velocity", (3,))
        self.estimation_mode = PredictionMode(ctx.get_scalar("prediction_mode"))
        pose = SE3.from_matrix(pose_matrix)
        self.state[:4] = pose.rotation().as_quat()
        self.state[4:7] = pose.translation()
        self.state[7:10] = actual_velocity
        # self.local_map.apply_backend_landmarks(points, timestamp_ns=ctx.get_scalar("timestamp", float))
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
