from typing import NamedTuple, Self
from warnings import deprecated

import gtsam
import numpy as np

from core.camera_model.stereo_camera_ctx import StereoContext
from core.camera_model.vio_context import ImuContext, VioContext
from core.front_end.keyframe import ActiveTrackSchema
from core.front_end.keyframe_selector import SelectReason
from core.graph_optimizer.optimizer_types import ActiveTrack
from core.graph_optimizer.sub_graph_builder import GraphContext
from logger import spawn_logger

X = gtsam.symbol_shorthand.X
L = gtsam.symbol_shorthand.L
V = gtsam.symbol_shorthand.V
B = gtsam.symbol_shorthand.B


class VioKeyframe(NamedTuple):
    """
    Optimized keyframe.

    gyroscope_noise_density: 1.6968e-04     # [ rad / s / sqrt(Hz) ]   ( gyro "white noise" )
    gyroscope_random_walk: 1.9393e-05       # [ rad / s^2 / sqrt(Hz) ] ( gyro bias diffusion )
    accelerometer_noise_density: 2.0000e-3  # [ m / s^2 / sqrt(Hz) ]   ( accel "white noise" )
    accelerometer_random_walk: 3.0000e-3    # [ m / s^3 / sqrt(Hz) ].  ( accel bias diffusion )
    """

    keyframe_id: int
    select_reason: list[SelectReason]
    timestamp: float
    # Nx12 (feat_id, timestamp, left_u, left_v, right_u, right_v, state, age, stereo_score, x, y, z)
    active_track: ActiveTrack
    pim: gtsam.PreintegratedImuMeasurements
    nav_state: gtsam.NavState
    bias: gtsam.imuBias.ConstantBias


@deprecated("Batch optimizer is deprecated, use ExplicitVIOOptimizer or SmartVIOOptimizer instead")
class BatchOptimizer:
    """Batch optimizer."""

    def __init__(self, ctx: GraphContext) -> None:
        """Initialize the batch optimizer."""
        self.logger = spawn_logger(app="batch_optimizer")
        self.ctx = ctx
        self.solver_params = gtsam.LevenbergMarquardtParams()
        self.solver_params.setlambdaInitial(0.0)
        self.solver_params.setlambdaLowerBound(0.0)
        self.solver_params.setlambdaUpperBound(0.0)
        self.values_graph = gtsam.Values()
        self.factors_graph = gtsam.NonlinearFactorGraph()
        self.last_keyframe_id = -1
        self.first_keyframe_id = -1

    @classmethod
    def from_stereo_ctx(cls, ctx: StereoContext) -> Self:
        """Create a batch optimizer from a stereo context."""
        graph_ctx = GraphContext(
            VioContext(
                stereo=ctx,
                imu=ImuContext.empty(),
            )
        )
        return cls(graph_ctx)

    @classmethod
    def from_vio_ctx(cls, vio_ctx: VioContext) -> Self:
        """Create a batch optimizer from a VIO context."""
        graph_ctx = GraphContext(vio_ctx)
        return cls(graph_ctx)

    def solve(self) -> None:
        """Solve the batch optimizer."""
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.factors_graph, self.values_graph, self.solver_params)
        # print("before")
        # print(self.values_graph)
        # print("--------------------------------")
        _ = optimizer.optimize()

        # print(f"solve result: {result}")

    def add_new_keyframe(self, keyframe: VioKeyframe) -> None:
        """Update the batch optimizer with a keyframe."""
        next_pose_key = X(keyframe.keyframe_id)
        next_vel_key = V(keyframe.keyframe_id)

        self.values_graph.insert(next_pose_key, keyframe.nav_state.pose())
        self.values_graph.insert(next_vel_key, keyframe.nav_state.velocity())

        if self.last_keyframe_id == -1:
            self.first_keyframe_id = keyframe.keyframe_id
            static_bias_key = B(self.first_keyframe_id)
            self.values_graph.insert(static_bias_key, keyframe.bias)
            bias_prior = gtsam.noiseModel.Diagonal.Sigmas(self.ctx.silent_var)
            constant_bias_prior = gtsam.PriorFactorConstantBias(static_bias_key, keyframe.bias, bias_prior)

            pose_prior = gtsam.PriorFactorPose3(
                next_pose_key, keyframe.nav_state.pose(), self.ctx.pose_prior_noise
            )
            velocity_prior = gtsam.PriorFactorVector(
                next_vel_key, keyframe.nav_state.velocity(), self.ctx.vel_prior_noise
            )
            self.factors_graph.add(pose_prior)
            self.factors_graph.add(velocity_prior)
            self.factors_graph.add(constant_bias_prior)
        else:
            self._add_imu_factors(keyframe)
            zero_velocity_model = gtsam.noiseModel.Isotropic.Sigma(3, 1e-4)
            zero_velocity_factor = gtsam.PriorFactorVector(
                next_vel_key, keyframe.nav_state.velocity(), zero_velocity_model
            )

            static_pose_model = gtsam.noiseModel.Isotropic.Sigma(6, 1e-6)
            static_pose_factor = gtsam.BetweenFactorPose3(
                X(self.last_keyframe_id), next_pose_key, gtsam.Pose3(), static_pose_model
            )
            self.factors_graph.add(zero_velocity_factor)
            self.factors_graph.add(static_pose_factor)

        self.last_keyframe_id = keyframe.keyframe_id

        for item in keyframe.active_track:
            feat_id = int(item[ActiveTrackSchema.FEAT_ID])
            landmark_key = L(feat_id)

            has_stereo = np.isfinite(item[ActiveTrackSchema.RIGHT_U])
            has_xyz = np.isfinite(item[ActiveTrackSchema.X])
            landmark_in_graph = self.values_graph.exists(landmark_key)

            if not landmark_in_graph:
                if not has_xyz:
                    continue
                x = item[ActiveTrackSchema.X]
                y = item[ActiveTrackSchema.Y]
                z = item[ActiveTrackSchema.Z]
                self.values_graph.insert(landmark_key, gtsam.Point3(x, y, z))

            if has_stereo:
                # add stereo factor
                ul = item[ActiveTrackSchema.LEFT_U]
                ur = item[ActiveTrackSchema.RIGHT_U]
                v = item[ActiveTrackSchema.LEFT_V]
                stereo_point = gtsam.StereoPoint2(ul, ur, v)
                factor = gtsam.GenericStereoFactor3D(
                    stereo_point, self.ctx.static_stereo_noise, next_pose_key, landmark_key, self.ctx.stereo_k
                )
                self.factors_graph.add(factor)

    def _add_imu_factors(self, keyframe: VioKeyframe) -> None:
        """Add IMU factors to the batch optimizer."""
        # need to add between keyframe prior
        next_pose_key = X(keyframe.keyframe_id)
        next_vel_key = V(keyframe.keyframe_id)
        static_bias_key = B(self.first_keyframe_id)

        prev_keyframe_id = self.last_keyframe_id
        prev_pose_key = X(prev_keyframe_id)
        prev_vel_key = V(prev_keyframe_id)
        # prev_bias_key = B(prev_keyframe_id)

        # sqrt_delta_t = max(np.sqrt(keyframe.pim.deltaTij()), 1e-6)
        # acc_sigma = self.ctx.accel_random_walk * sqrt_delta_t
        # gyro_sigma = self.ctx.gyro_random_walk * sqrt_delta_t
        # bias_walk_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([acc_sigma] * 3 + [gyro_sigma] * 3))

        imu_factor = gtsam.ImuFactor(
            prev_pose_key, prev_vel_key, next_pose_key, next_vel_key, static_bias_key, keyframe.pim
        )
        # between_bias_factor = gtsam.BetweenFactorConstantBias(
        #    prev_bias_key, next_bias_key, gtsam.imuBias.ConstantBias(), bias_walk_model
        # )

        # print(f"imu_factor: {imu_factor}")
        # print(f"between_factor: {between_bias_factor}")
        self.factors_graph.add(imu_factor)
        # self.factors_graph.add(between_bias_factor)

    def pim_integration_required(self) -> bool:
        """Check if the PIM integration is required."""
        return self.last_keyframe_id != -1
