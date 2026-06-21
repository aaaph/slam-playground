from collections import deque

import gtsam
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from core.camera_model.vio_context import VioContext
from core.front_end.keyframe import ActiveTrackSchema, ImuBatchSchema
from core.graph_optimizer.optimizer_types import PredictionMode, VioKeyframe
from core.graph_optimizer.sub_graph_builder import GraphContext, SubGraph, SubGraphBuilder
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger

X = gtsam.symbol_shorthand.X
V = gtsam.symbol_shorthand.V
B = gtsam.symbol_shorthand.B
L = gtsam.symbol_shorthand.L


class ExplicitVIOOptimizer:
    """Explicit VIO optimizer."""

    def __init__(self, ctx: GraphContext, lag: float = 10.0) -> None:
        """Initialize the explicit VIO optimizer."""
        self.ctx = ctx
        self.lag = lag
        smoother_params = gtsam.ISAM2Params()
        smoother_params.cacheLinearizedFactors = True
        smoother_params.findUnusedFactorSlots = True
        smoother_params.setFactorization("CHOLESKY")
        smoother_params.evaluateNonlinearError = False
        smoother_params.enableDetailedResults = False
        smoother_params.setRelinearizeThreshold(0.01)
        smoother_params.relinearizeSkip = 1
        self.smoother = gtsam.IncrementalFixedLagSmoother(self.lag, smoother_params)
        self.logger = spawn_logger(app="explicit_vio_optimizer")
        self.result = gtsam.Values()
        self.post_fit_error = 0.0
        self.last_keyframe_id = -1
        self.alpha = 0.0

    @classmethod
    def from_vio_ctx(cls, vio_ctx: VioContext, lag: float = 10.0) -> "ExplicitVIOOptimizer":
        """Create an ExplicitVIOOptimizer from a VioContext."""
        return cls(GraphContext(vio_ctx), lag=lag)

    def optimize(
        self,
        factors: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
        timestamp_map: gtsam.FixedLagSmootherKeyTimestampMap,
        ts: float,
    ) -> None:
        """Optimize the graph."""
        self.logger.info(
            f"[FG:START]: new factors:{factors.size()}, new values:{values.size()}, ",
            f"new ts_size: {len(timestamp_map)}",
            f"ts:{ts}, old graph size: {self.result.size()}",
        )
        self.smoother.update(factors, values, timestamp_map)
        self.result = self.smoother.calculateEstimate()
        self.logger.info(f"[FG:DONE]: new graph size: {self.result.size()}")

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

    def keyframe_to_subgraph(
        self,
        keyframe: VioKeyframe,
    ) -> SubGraph:
        """Convert a keyframe to a subgraph."""
        next_keyframe_id = keyframe.keyframe_id
        prev_keyframe_id = self.last_keyframe_id

        builder = self._new_builder(keyframe.timestamp, next_keyframe_id)

        if prev_keyframe_id == -1:
            # need add priors for the first kf
            initial_pose = keyframe.pose_guess or SE3.identity()
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
            anchor_pose = initial_pose.copy()
        else:
            prev_bias = self.result.atConstantBias(B(prev_keyframe_id))
            pim = self._calc_pim_batch(keyframe.imu_batch, prev_bias)
            next_bias = prev_bias

            if keyframe.prediction_mode == PredictionMode.PNP:
                next_pose = keyframe.pose_guess or SE3.identity()
                next_velocity = np.asarray(keyframe.velocity_guess)
                anchor_pose = next_pose.copy()
                self.logger.info(f"[FG:PNP]: pose: {next_pose}, velocity: {next_velocity}")

            if keyframe.prediction_mode == PredictionMode.PIM:
                prev_pose = self.result.atPose3(X(prev_keyframe_id))
                prev_vel = self.result.atVector(V(prev_keyframe_id))
                prev_nav_state = gtsam.NavState(prev_pose, prev_vel)

                next_nav_state = pim.predict(prev_nav_state, prev_bias)
                next_pose = next_nav_state.pose()
                next_velocity = next_nav_state.velocity()
                anchor_pose = SE3.from_gtsam_pose(next_pose)
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
            # check for zupt and add zero velocity prior

        self._build_active_track_subgraph(next_keyframe_id, anchor_pose, keyframe, builder)

        return builder.build_subgraph()

    def _build_active_track_subgraph(
        self, next_keyframe_id: int, anchor_pose: SE3, keyframe: VioKeyframe, builder: SubGraphBuilder
    ) -> None:
        """Build the active track subgraph."""
        cam0_in_world = anchor_pose * self.ctx.cam0_in_body
        keyframe.active_track[:, ActiveTrackSchema.X : ActiveTrackSchema.Z + 1] = (
            keyframe.active_track[:, ActiveTrackSchema.X : ActiveTrackSchema.Z + 1]
            @ cam0_in_world.rotation().as_matrix().T
            + cam0_in_world.translation()
        )
        for visual_feature in keyframe.active_track:
            feat_id = int(visual_feature[ActiveTrackSchema.FEAT_ID])
            landmark_key = L(feat_id)

            has_stereo = np.isfinite(visual_feature[ActiveTrackSchema.RIGHT_U])
            has_xyz = np.isfinite(visual_feature[ActiveTrackSchema.X])
            existing_landmark_in_graph = self.result.exists(landmark_key)
            landmark_in_graph = existing_landmark_in_graph
            not_new = visual_feature[ActiveTrackSchema.AGE] > 0
            good_stereo = visual_feature[ActiveTrackSchema.STEREO_SCORE] == visual_feature[ActiveTrackSchema.AGE]

            if not landmark_in_graph and has_xyz and has_stereo and good_stereo and not_new:
                x = visual_feature[ActiveTrackSchema.X]
                y = visual_feature[ActiveTrackSchema.Y]
                z = visual_feature[ActiveTrackSchema.Z]
                builder.with_landmark(feat_id, np.array([x, y, z]))
                landmark_in_graph = True

            if has_stereo and has_xyz and landmark_in_graph:
                ul = visual_feature[ActiveTrackSchema.LEFT_U]
                ur = visual_feature[ActiveTrackSchema.RIGHT_U]
                v = visual_feature[ActiveTrackSchema.LEFT_V]
                if existing_landmark_in_graph:
                    measurement = np.array([ul, ur, v], dtype=np.float64)
                    reprojection_error = self._stereo_reprojection_error_px(
                        cam0_in_world,
                        np.asarray(self.result.atPoint3(landmark_key), dtype=np.float64),
                        measurement,
                    )
                    if reprojection_error > self.ctx.stereo_reprojection_gate_px:
                        self.logger.debug(
                            "[FG:LANDMARK_GATE]: skip inconsistent stereo factor "
                            f"feat_id={feat_id}, kf={next_keyframe_id}, error={reprojection_error:.2f}px"
                        )
                        continue
                stereo_point = gtsam.StereoPoint2(ul, ur, v)
                builder.add_stereo_factor(next_keyframe_id, feat_id, stereo_point)

    def _project_world_landmark_to_stereo(
        self, cam0_in_world: SE3, landmark_world: NDArray[np.float64]
    ) -> NDArray[np.float64] | None:
        """Project a world landmark into the current rectified stereo frame."""
        landmark_cam0 = cam0_in_world.inverse().act_on_vector(landmark_world)
        if not np.all(np.isfinite(landmark_cam0)):
            return None

        x, y, z = landmark_cam0
        if z < self.ctx.landmark_depth_min_m or z > self.ctx.landmark_depth_max_m:
            return None

        k_matrix = self.ctx.stereo_k_matrix
        fx = k_matrix[0, 0]
        fy = k_matrix[1, 1]
        cx = k_matrix[0, 2]
        cy = k_matrix[1, 2]
        ul = fx * x / z + cx
        ur = ul - fx * self.ctx.stereo_baseline / z
        v = fy * y / z + cy
        projected = np.array([ul, ur, v], dtype=np.float64)
        if not np.all(np.isfinite(projected)):
            return None
        return projected

    def _stereo_reprojection_error_px(
        self, cam0_in_world: SE3, landmark_world: NDArray[np.float64], measurement: NDArray[np.float64]
    ) -> float:
        """Return stereo reprojection error in pixels, or infinity for invalid projections."""
        projected = self._project_world_landmark_to_stereo(cam0_in_world, landmark_world)
        if projected is None:
            return np.inf
        return float(np.linalg.norm(projected - measurement))

    def apply_subgraph(
        self,
        subgraph: SubGraph,
    ) -> None:
        """Resolve the subgraph."""
        self.optimize(subgraph.factors, subgraph.values, subgraph.timestamp_map, subgraph.timestamp)
        self.last_keyframe_id = subgraph.keyframe_id

    def post_fit_avg_error(self) -> float:
        """Get the post-fit average error."""
        total_error = self.smoother.getFactors().error(self.result)
        n_factors = self.smoother.getFactors().size()
        return total_error / n_factors if n_factors > 0 else 0.0

    def add_keyframe(self, keyframe: VioKeyframe) -> None:
        """Add a keyframe to the optimizer."""
        subgraph = self.keyframe_to_subgraph(keyframe)
        self.apply_subgraph(subgraph)

    def keyframes_to_subgraph(self, keyframes: list[VioKeyframe]) -> SubGraph:
        """Convert multiple keyframes into a single subgraph for one optimization step."""
        if len(keyframes) == 0:
            raise ValueError("At least one keyframe is required")

        if len(keyframes) == 1:
            return self.keyframe_to_subgraph(keyframes[0])

        msg = "More then one keyframe is not supported for now"
        raise ValueError(msg)

    def add_keyframes(self, keyframes: list[VioKeyframe]) -> None:
        """Add multiple keyframes and optimize them as one subgraph."""
        subgraph = self.keyframes_to_subgraph(keyframes)
        self.apply_subgraph(subgraph)

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

    def get_landmarks_ndarray(self) -> NDArray[np.float32]:
        """
        Get the landmarks as a numpy array.

        Return schema: [feat_id, x, y, z, 1]
        """
        landmarks = []
        l_char = ord("l")
        for key in self.result.keys():  # noqa: SIM118
            sym = gtsam.Symbol(key)
            if sym.chr() == l_char:
                point_value = self.result.atPoint3(key)
                payload = [sym.index(), point_value[0], point_value[1], point_value[2], 1]
                landmarks.append(payload)
        return np.array(landmarks)

    def get_nav_state(self) -> gtsam.NavState:
        """Get the nav state."""
        pose = self.result.atPose3(self.actual_pose_index())
        vel = self.result.atVector(self.actual_velocity_index())
        return gtsam.NavState(pose, vel)

    def get_actual_bias(self) -> gtsam.imuBias.ConstantBias:
        """Get the actual bias."""
        return self.result.atConstantBias(self.actual_bias_index())

    def get_actual_bias_ndarray(self) -> NDArray[np.float32]:
        """Get the actual bias as a numpy array."""
        bias = self.get_actual_bias()
        accel = bias.accelerometer()
        gyro = bias.gyroscope()
        return np.array([accel[0], accel[1], accel[2], gyro[0], gyro[1], gyro[2]], dtype=np.float32)

    def get_nav_state_ndarray(self) -> NDArray[np.float32]:
        """Get the nav state as a numpy array."""
        nav_state = self.get_nav_state()
        quat = Rotation.from_matrix(nav_state.pose().rotation().matrix()).as_quat()
        vec = nav_state.pose().translation()
        vel = nav_state.velocity()
        return np.array([quat[0], quat[1], quat[2], quat[3], vec[0], vec[1], vec[2], vel[0], vel[1], vel[2]])

    def get_accel_bias_sigma(self) -> NDArray[np.float32]:
        """Get the acceleration bias sigma."""
        bias_covariance = self.get_covariance_in(self.actual_bias_index())
        return np.sqrt(np.array([bias_covariance[0, 0], bias_covariance[1, 1], bias_covariance[2, 2]]))

    def get_covariance_in(self, index: int) -> NDArray[np.float64]:
        """Get the covariance in the index."""
        return self.smoother.marginalCovariance(index)

    def actual_bias_index(self) -> int:
        """Get the index of the actual bias."""
        return B(self.last_keyframe_id)

    def actual_velocity_index(self) -> int:
        """Get the index of the actual velocity."""
        return V(self.last_keyframe_id)

    def actual_pose_index(self) -> int:
        """Get the index of the actual pose."""
        return X(self.last_keyframe_id)
