from collections import deque
from typing import cast

import gtsam
import numpy as np
from numpy.typing import NDArray

from core.camera_model.vio_context import VioContext
from core.feature_tracker.feature_schema import FeatureLifecycle
from core.front_end.keyframe import ImuBatchSchema
from core.graph_optimizer.optimizer_types import (
    PredictionMode,
    SmartStereoProjectionPoseFactor,
    StereoMeasurement,
    VioKeyframe,
)
from core.graph_optimizer.sub_graph_builder import GraphContext, SubGraph, SubGraphBuilder
from core.pose_tracker.feature_triangulation import StereoTriangulationSchema
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger
from logger.decorators import timeit

X = gtsam.symbol_shorthand.X
V = gtsam.symbol_shorthand.V
B = gtsam.symbol_shorthand.B


class SmartFactorVIOOptimizer:
    """Smart factor VIO optimizer."""

    def __init__(self, ctx: GraphContext, lag: float = 10.0 * 1e9) -> None:
        """Initialize the explicit VIO optimizer."""
        self.ctx = ctx
        self.lag = lag

        self.smoother = gtsam.IncrementalFixedLagSmoother(lag)

        self.logger = spawn_logger(app="smart_factor_vio_optimizer")
        self.result = gtsam.Values()

        self.post_fit_error = 0.0
        self.live_factor_count = 0
        self.smart_factor_error_ratio = 0.0
        self.smart_factor_whitened_rmse = 0.0
        self.smart_factor_max_whitened_rmse = 0.0
        self.last_keyframe_id = -1
        self.alpha = 0.0

        self.smart_factors: dict[int, int] = {}
        self.measurement_history: dict[int, deque[StereoMeasurement]] = {}
        self.sliding_window_poses: dict[int, float] = {}
        self.sliding_window_poses_dq = deque()

    @classmethod
    def from_vio_ctx(cls, vio_ctx: VioContext, lag: float = 10.0 * 1e9) -> "SmartFactorVIOOptimizer":
        """Initialize the smart factor VIO optimizer from a VIO context."""
        return cls(GraphContext(vio_ctx), lag)

    def optimize(self, subgraph: SubGraph) -> None:
        """Optimize the subgraph."""
        self.logger.info(f"[FG:BEFORE]: apply subgraph: {subgraph}")
        self.smoother.update(subgraph.factors, subgraph.values, subgraph.timestamp_map, subgraph.delete_slots)
        self.result = self.smoother.calculateEstimate()
        self._update_fit_metrics()
        self.logger.info(f"[FG:AFTER]: post fit error: {self.post_fit_error}")

    def _update_fit_metrics(self) -> None:
        """Update total and smart-factor normalized post-fit errors."""
        factors = self.smoother.getFactors()
        smart_slots = {slot: feat_id for feat_id, slot in self.smart_factors.items()}
        smart_error = 0.0
        smart_dof = 0
        worst_feat_id = -1
        self.post_fit_error = 0.0
        self.live_factor_count = 0
        self.smart_factor_max_whitened_rmse = 0.0

        for slot in range(factors.size()):
            factor = factors.at(slot)
            if factor is None:
                continue
            factor_error = factor.error(self.result)
            self.post_fit_error += factor_error
            self.live_factor_count += 1

            feat_id = smart_slots.get(slot)
            if feat_id is None:
                continue
            factor_dof = 3 * (len(factor.keys()) - 1)
            if factor_dof <= 0:
                continue
            smart_error += factor_error
            smart_dof += factor_dof
            factor_rmse = float(np.sqrt(2.0 * factor_error / factor_dof))
            if factor_rmse > self.smart_factor_max_whitened_rmse:
                self.smart_factor_max_whitened_rmse = factor_rmse
                worst_feat_id = feat_id

        self.smart_factor_whitened_rmse = float(np.sqrt(2.0 * smart_error / smart_dof)) if smart_dof else 0.0
        self.smart_factor_error_ratio = smart_error / self.post_fit_error if self.post_fit_error else 0.0
        self.logger.debug(
            f"[FG:SMART_ERROR]: rmse={self.smart_factor_whitened_rmse:.3f}, "
            f"ratio={self.smart_factor_error_ratio:.3f}, worst_feat_id={worst_feat_id}, "
            f"worst_rmse={self.smart_factor_max_whitened_rmse:.3f}"
        )

    def apply_subgraph(self, subgraph: SubGraph) -> None:
        """Apply a subgraph to the optimizer."""
        self.optimize(subgraph)
        self.last_keyframe_id = subgraph.keyframe_id

    def get_nav_state(self) -> gtsam.NavState:
        """Return the latest optimized navigation state."""
        pose = self.result.atPose3(X(self.last_keyframe_id))
        velocity = self.result.atVector(V(self.last_keyframe_id))
        return gtsam.NavState(pose, velocity)

    def get_actual_bias_ndarray(self) -> NDArray[np.float32]:
        """Return the latest optimized accelerometer and gyroscope bias."""
        bias = self.result.atConstantBias(B(self.last_keyframe_id))
        return np.concatenate((bias.accelerometer(), bias.gyroscope())).astype(np.float32)

    def get_accel_bias_sigma(self) -> NDArray[np.float64]:
        """Return the accelerometer bias marginal standard deviation."""
        covariance = self.smoother.marginalCovariance(B(self.last_keyframe_id))
        return np.sqrt(np.diag(covariance)[:3])

    def post_fit_avg_error(self) -> float:
        """Return the average post-fit factor error."""
        return self.post_fit_error / self.live_factor_count if self.live_factor_count else 0.0

    def _extend_sliding_window(self, kfid: int, timestamp: float) -> None:
        """Extend the sliding window."""
        pose_key = X(kfid)
        self.sliding_window_poses_dq.append(pose_key)
        self.sliding_window_poses[pose_key] = timestamp

    def _remove_from_sliding_window(self, marginalized_poses: list[int]) -> None:
        """Remove the marginilize candidates from the sliding window."""
        for pose_key in marginalized_poses:
            self.sliding_window_poses.pop(pose_key, None)
            if self.sliding_window_poses_dq and self.sliding_window_poses_dq[0] == pose_key:
                self.sliding_window_poses_dq.popleft()

    def _get_nullptr_slots(self) -> deque[int]:
        """Get the nullptr slots."""
        null_slots = deque()
        factors = self.smoother.getFactors()
        size = factors.size()
        for i in range(size):
            factor = factors.at(i)
            if factor is None:
                null_slots.append(i)
        return null_slots

    def _new_builder(self, timestamp: float, keyframe_id: int) -> SubGraphBuilder:
        """Create a new builder."""
        return self.ctx.new_builder(
            timestamp=timestamp,
            null_slots=self._get_nullptr_slots(),
            factors_graph_size=self.smoother.getFactors().size(),
            keyframe_id=keyframe_id,
        )

    def get_out_of_sliding_window_poses(self, timestamp: float) -> list[int]:
        """Get the marginilize pose ids based on the sliding window poses."""
        window_diff = timestamp - self.lag
        marginilize_candidates: list[int] = []
        for pose_key in self.sliding_window_poses_dq:
            if self.sliding_window_poses[pose_key] < window_diff:
                marginilize_candidates.append(pose_key)
            else:
                break
        return marginilize_candidates

    def _calc_pim_batch(
        self, imu_batch: NDArray[np.float64], bias: gtsam.imuBias.ConstantBias
    ) -> gtsam.PreintegratedImuMeasurements:
        """Calculate the PIM."""
        pim = gtsam.PreintegratedImuMeasurements(self.ctx.pim_params, bias)
        for imu_item in imu_batch:
            accel = np.array(
                [
                    imu_item[ImuBatchSchema.ACCEL_X],
                    imu_item[ImuBatchSchema.ACCEL_Y],
                    imu_item[ImuBatchSchema.ACCEL_Z],
                ]
            )
            gyro = np.array(
                [
                    imu_item[ImuBatchSchema.GYRO_X],
                    imu_item[ImuBatchSchema.GYRO_Y],
                    imu_item[ImuBatchSchema.GYRO_Z],
                ]
            )
            dt = imu_item[ImuBatchSchema.DT]
            pim.integrateMeasurement(accel, gyro, dt)
        return pim

    def keyframe_to_subgraph(self, keyframe: VioKeyframe) -> SubGraph:
        """Convert a keyframe to a subgraph."""
        next_keyframe_id = keyframe.keyframe_id
        prev_keyframe_id = self.last_keyframe_id

        out_of_sliding_window = self.get_out_of_sliding_window_poses(keyframe.timestamp)
        builder = self._new_builder(keyframe.timestamp, next_keyframe_id)

        if prev_keyframe_id == -1:
            # need add priors for the first kf
            initial_pose = keyframe.pose_guess
            initial_velocity = np.asarray(keyframe.velocity_guess)
            initial_bias_vector = np.asarray(keyframe.bias_guess)
            initial_accel_bias = initial_bias_vector[:3]
            initial_gyro_bias = initial_bias_vector[3:]
            initial_bias = gtsam.imuBias.ConstantBias(initial_accel_bias, initial_gyro_bias)
            (
                builder.with_pose(next_keyframe_id, initial_pose)
                .add_pose_prior(next_keyframe_id, initial_pose, self.ctx.pose_prior_noise)
                .with_velocity(next_keyframe_id, initial_velocity)
                .add_velocity_prior(next_keyframe_id, initial_velocity, self.ctx.vel_prior_noise)
                .with_bias(next_keyframe_id, initial_bias)
                .add_bias_prior(next_keyframe_id, initial_bias, self.ctx.bias_prior_noise)
            )
        else:
            prev_bias = self.result.atConstantBias(B(prev_keyframe_id))
            pim = self._calc_pim_batch(keyframe.imu_batch, prev_bias)
            next_bias = prev_bias

            if keyframe.prediction_mode == PredictionMode.PNP:
                next_pose = keyframe.pose_guess or SE3.identity()
                next_velocity = np.asarray(keyframe.velocity_guess)
                # anchor_pose = next_pose.copy()
                self.logger.info(f"[FG:PNP]: pose: {next_pose}, velocity: {next_velocity}")

            if keyframe.prediction_mode == PredictionMode.PIM:
                prev_pose = self.result.atPose3(X(prev_keyframe_id))
                prev_vel = self.result.atVector(V(prev_keyframe_id))
                prev_nav_state = gtsam.NavState(prev_pose, prev_vel)

                next_nav_state = pim.predict(prev_nav_state, prev_bias)
                next_pose = next_nav_state.pose()
                next_velocity = next_nav_state.velocity()
                # anchor_pose = SE3.from_gtsam_pose(next_pose)
                self.logger.info(f"[FG:PIM]: pose: {SE3.from_gtsam_pose(next_pose)}, velocity: {next_velocity}")

            (
                builder.with_pose(next_keyframe_id, next_pose)
                .with_velocity(next_keyframe_id, next_velocity)
                .with_bias(next_keyframe_id, next_bias)
                .add_imu_factor(prev_keyframe_id, next_keyframe_id, pim)
                .add_between_bias_factor(prev_keyframe_id, next_keyframe_id, pim.deltaTij())
            )
            if keyframe.zupt:
                self.logger.info("[ZUPT]: adding zero velocity prior")
                builder.add_velocity_prior(next_keyframe_id, np.zeros(3), self.ctx.vel_prior_noise)

        self._apply_visual_frame_to_subgraph(next_keyframe_id, keyframe, builder, set(out_of_sliding_window))

        self._extend_sliding_window(next_keyframe_id, keyframe.timestamp)
        self._remove_from_sliding_window(out_of_sliding_window)
        return builder.build_subgraph()

    def _apply_visual_frame_to_subgraph(
        self,
        next_keyframe_id: int,
        keyframe: VioKeyframe,
        builder: SubGraphBuilder,
        outgoing_pose_keys: set[int],
    ) -> None:
        """Apply the visual frame to the subgraph."""
        active_mask = (
            keyframe.stereo_frame[:, StereoTriangulationSchema.LIFECYCLE] == FeatureLifecycle.ACTIVE.value
        )
        active_feat_ids = set(
            keyframe.stereo_frame[active_mask, StereoTriangulationSchema.FEAT_ID].astype(np.int32)
        )
        for feat_id, history in list(self.measurement_history.items()):
            if len(history) == 1 and feat_id not in active_feat_ids:
                del self.measurement_history[feat_id]

        for feat_id, history in list(self.measurement_history.items()):
            if history and history[0].pose_key in outgoing_pose_keys:
                del self.measurement_history[feat_id]
                self.smart_factors.pop(feat_id, None)

        for stereo_feature in keyframe.stereo_frame:
            feat_id = int(stereo_feature[StereoTriangulationSchema.FEAT_ID])
            if not StereoTriangulationSchema.active(stereo_feature):
                continue
            if not (
                StereoTriangulationSchema.stereo(stereo_feature)
                and StereoTriangulationSchema.good_stereo(stereo_feature)
            ):
                continue

            measurement = StereoMeasurement(
                next_keyframe_id,
                stereo_feature[StereoTriangulationSchema.LEFT_U],
                stereo_feature[StereoTriangulationSchema.RIGHT_U],
                stereo_feature[StereoTriangulationSchema.LEFT_V],
            )

            history = self.measurement_history.get(feat_id)
            if history is None:
                self.measurement_history[feat_id] = deque([measurement])
                continue

            slot = self.smart_factors.get(feat_id)
            if slot is not None:
                builder.push_delete_slot(slot)
            history.append(measurement)
            builder.add_smart_factor(feat_id, history)
            self.smart_factors[feat_id] = builder.smart_factor_slot(feat_id)

    @timeit
    def get_landmarks_ndarray(self) -> NDArray[np.float32]:
        """Get cached smart-factor landmarks."""
        landmarks = np.full((len(self.smart_factors), 5), np.nan, dtype=np.float32)
        factors = self.smoother.getFactors()

        for i, (feat_id, slot) in enumerate(self.smart_factors.items()):
            factor = cast("SmartStereoProjectionPoseFactor", factors.at(slot))
            point_result = factor.point()

            landmarks[i, 0] = feat_id
            landmarks[i, 4] = point_result.status.value
            if point_result.valid():
                landmarks[i, 1:4] = point_result.get()

        return landmarks
