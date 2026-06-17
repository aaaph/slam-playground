from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, SupportsInt

import gtsam
import gtsam_unstable
import numpy as np

from core.camera_model.vio_context import VioContext
from core.graph_optimizer.optimizer_types import FactorType, StereoMeasurement
from core.transformations.special_euclidian_3_dim import SE3

X = gtsam.symbol_shorthand.X
L = gtsam.symbol_shorthand.L
V = gtsam.symbol_shorthand.V
B = gtsam.symbol_shorthand.B

SmartStereoProjectionPoseFactor: Any = getattr(gtsam_unstable, "SmartStereoProjectionPoseFactor")  # noqa: B009


@dataclass
class SubGraph:
    """Sub graph."""

    timestamp: float
    keyframe_id: int
    factors: gtsam.NonlinearFactorGraph
    values: gtsam.Values
    timestamp_map: gtsam.FixedLagSmootherKeyTimestampMap
    delete_slots: Sequence[SupportsInt]

    def __repr__(self) -> str:
        """Return a string with sub graph information."""
        return (
            f"SubGraph(timestamp={self.timestamp:.0f}, keyframe_id={self.keyframe_id}, "
            f"factors={self.factors.size()}, values={self.values.size()}, "
            f"timestamp_map={len(self.timestamp_map)}, "
            f"delete_slots={len(self.delete_slots)})"
        )

    def merge(self, other: "SubGraph") -> "SubGraph":
        """Merge two subgraphs."""
        merged_factors = gtsam.NonlinearFactorGraph()
        merged_factors.push_back(self.factors)
        merged_factors.push_back(other.factors)

        merged_values = gtsam.Values()
        merged_values.insert(self.values)
        self_keys = set(self.values.keys())
        duplicate_keys = self_keys.intersection(other.values.keys())
        if duplicate_keys:
            duplicate_symbols = [str(gtsam.Symbol(key)) for key in sorted(duplicate_keys)]
            msg = f"Cannot merge subgraphs with duplicate value keys: {duplicate_symbols}"
            raise ValueError(msg)
        merged_values.insert(other.values)

        merged_timestamp_map = gtsam.FixedLagSmootherKeyTimestampMap()
        for key, timestamp in self.timestamp_map.items():
            merged_timestamp_map.insert((key, timestamp))
        for key, timestamp in other.timestamp_map.items():
            merged_timestamp_map.insert((key, timestamp))

        merged_delete_slots = list(dict.fromkeys([*self.delete_slots, *other.delete_slots]))

        return SubGraph(
            timestamp=other.timestamp,
            keyframe_id=other.keyframe_id,
            factors=merged_factors,
            values=merged_values,
            timestamp_map=merged_timestamp_map,
            delete_slots=merged_delete_slots,
        )


class GraphContext:
    """Graph context."""

    def __init__(self, vio_ctx: VioContext) -> None:
        """Initialize the graph context."""
        huber = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
        self.static_stereo_noise = gtsam.noiseModel.Robust.Create(huber, gtsam.noiseModel.Isotropic.Sigma(3, 2.0))
        self.static_mono_noise = gtsam.noiseModel.Robust.Create(huber, gtsam.noiseModel.Isotropic.Sigma(2, 2.0))
        self.static_smart_noise = gtsam.noiseModel.Isotropic.Sigma(3, 2.0)
        self.freeze_prior_noise = gtsam.noiseModel.Constrained.All(6)
        self.smart_factor_params = gtsam.SmartProjectionParams()
        self.smart_factor_params.setDegeneracyMode(gtsam.DegeneracyMode.ZERO_ON_DEGENERACY)
        self.smart_factor_params.setRankTolerance(1.0)
        self.smart_factor_params.setLinearizationMode(gtsam.LinearizationMode.HESSIAN)
        self.body_sensor_transform = vio_ctx.stereo.cam0_in_body_se3.as_gtsam_pose()
        self.stereo_k_matrix = vio_ctx.stereo.stereo_k
        self.stereo_baseline = vio_ctx.stereo.baseline
        self.stereo_reprojection_gate_px = 30.0
        self.landmark_depth_min_m = 0.15
        self.landmark_depth_max_m = 40.0

        initial_position_sigma = 1e-05
        initial_velocity_sigma = 0.001
        initial_pose_sigmas = np.array(
            [
                np.radians(10.0),  # roll
                np.radians(10.0),  # pitch
                np.radians(0.1),  # yaw
                initial_position_sigma,  # x
                initial_position_sigma,  # y
                initial_position_sigma,  # z
            ]
        )
        self.pose_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(initial_pose_sigmas)
        self.vel_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([initial_velocity_sigma] * 3))

        self.stereo_k = vio_ctx.stereo.stereo_k_gtsam
        self.mono_k = vio_ctx.stereo.cam0_k_gtsam

        self.accel_random_walk = vio_ctx.imu.accel_random_walk
        self.gyro_random_walk = vio_ctx.imu.gyro_random_walk

        self.sigma_ba_value = 0.1
        self.sigma_ba = np.array([self.sigma_ba_value] * 3)
        self.sigma_bg_value = 0.01
        self.sigma_bg = np.array([self.sigma_bg_value] * 3)
        self.bias_prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.concatenate([self.sigma_ba, self.sigma_bg]))

        self.initial_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.initial_gyro_bias = np.array([0.0, 0.0, 0.0])
        self.silent_var = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        self.pim_params = vio_ctx.imu.pim_params()
        self.cam0_in_body = vio_ctx.stereo.cam0_in_body_se3.copy()

    def new_builder(
        self,
        null_slots: deque[int] | None = None,
        factors_graph_size: int = 0,
        timestamp: float = 0.0,
        keyframe_id: int = -1,
    ) -> "SubGraphBuilder":
        """Create a new builder."""
        return SubGraphBuilder(
            self,
            timestamp=timestamp,
            uppergraph_size=factors_graph_size,
            null_slots=null_slots,
            keyframe_id=keyframe_id,
        )


class SubGraphBuilder:
    """GTSAM subgraph builder."""

    def __init__(
        self,
        ctx: GraphContext,
        timestamp: float = 0.0,
        uppergraph_size: int = 0,
        null_slots: deque[int] | None = None,
        keyframe_id: int = -1,
    ) -> None:
        """Initialize the sub graph builder."""
        self.ctx = ctx
        self._values = gtsam.Values()
        self._factors = gtsam.NonlinearFactorGraph()
        self._timestamp = timestamp
        self._timestamp_map = gtsam.FixedLagSmootherKeyTimestampMap()
        self._null_slots = null_slots or deque()

        self._factor_slots_counter = 0
        self._factor_slots_map: dict[int, int] = {}
        self._subgraph_factor_slots_counter = 0
        self._subgraph_factor_slots: dict[int, int] = {}
        self._subgraph_factor_types: dict[int, FactorType] = {}

        self._upper_graph_size = uppergraph_size
        self._upper_factor_slots: dict[int, int] = {}
        self._upper_factor_types: dict[int, FactorType] = {}

        self._factor_slots_types: dict[int, FactorType] = {}
        self._delete_slots: set[SupportsInt] = set()
        self._keyframe_id = keyframe_id

    def set_keyframe_id(self, keyframe_id: int) -> None:
        """Set the keyframe id."""
        self._keyframe_id = keyframe_id

    def _add_factor_with_slots(
        self,
        factor: gtsam.NonlinearFactor,
        factor_type: FactorType,
        tracking_id: int | None = None,
    ) -> int:
        """
        Add a factor and keep local/upper slot counters in sync.

        Args:
            factor: Factor to add to the local sub-graph.
            factor_type: Semantic type of the factor for slot bookkeeping.
            tracking_id: Optional logical id used to look the factor slot up later.

        Returns:
            Slot index in the upper graph.

        """
        self._factors.add(factor)

        subgraph_slot = self._subgraph_factor_slots_counter
        self._subgraph_factor_types[subgraph_slot] = factor_type
        self._subgraph_factor_slots_counter += 1
        if tracking_id is not None:
            self._subgraph_factor_slots[tracking_id] = subgraph_slot

        upper_slot = self._allocate_upper_slot()
        self._upper_factor_types[upper_slot] = factor_type
        if tracking_id is not None:
            self._upper_factor_slots[tracking_id] = upper_slot

        return upper_slot

    def with_pose(self, frame_id: int, pose: gtsam.Pose3 | SE3) -> "SubGraphBuilder":
        """Add a pose to the sub graph."""
        if isinstance(pose, SE3):
            pose = pose.as_gtsam_pose()
        self._values.insert(X(frame_id), pose)
        self._timestamp_map.insert((X(frame_id), self._timestamp))
        return self

    def with_velocity(self, frame_id: int, velocity: np.ndarray) -> "SubGraphBuilder":
        """Add a velocity to the sub graph."""
        self._values.insert(V(frame_id), velocity)
        self._timestamp_map.insert((V(frame_id), self._timestamp))
        return self

    def with_bias(self, frame_id: int, bias: gtsam.imuBias.ConstantBias) -> "SubGraphBuilder":
        """Add a bias to the sub graph."""
        self._values.insert(B(frame_id), bias)
        self._timestamp_map.insert((B(frame_id), self._timestamp))
        return self

    def with_landmark(
        self, landmark_id: int, landmark: np.ndarray, uncertainty: np.ndarray | None = None
    ) -> "SubGraphBuilder":
        """Add a landmark to the sub graph."""
        point = gtsam.Point3(*landmark)
        self._values.insert(L(landmark_id), point)
        self._timestamp_map.insert((L(landmark_id), self._timestamp))
        if uncertainty is not None:
            uncertainty_factor = gtsam.noiseModel.Diagonal.Sigmas(uncertainty)
            factor = gtsam.PriorFactorPoint3(L(landmark_id), point, uncertainty_factor)
            self._add_factor_with_slots(factor, FactorType.PRIOR_FACTOR)
        return self

    def add_stereo_factor(
        self, frame_id: int, feat_id: int, stereo_point: gtsam.StereoPoint2
    ) -> "SubGraphBuilder":
        """Add a reprojection factor to the sub graph."""
        lmk_key = L(feat_id)
        pose_key = X(frame_id)
        self._timestamp_map.insert((lmk_key, self._timestamp))
        stereo_factor = gtsam.GenericStereoFactor3D(
            measured=stereo_point,
            noiseModel=self.ctx.static_stereo_noise,
            poseKey=pose_key,
            landmarkKey=lmk_key,
            K=self.ctx.stereo_k,
            body_P_sensor=self.ctx.body_sensor_transform,
        )
        self._add_factor_with_slots(stereo_factor, FactorType.LANDMARK, feat_id)
        return self

    def add_freeze_prior(self, frame_id: int) -> "SubGraphBuilder":
        """Add a free prior to the sub graph."""
        key = X(frame_id)
        prior_noise = gtsam.noiseModel.Constrained.All(6)
        factor = gtsam.PriorFactorPose3(key, gtsam.Pose3(), prior_noise)
        self._add_factor_with_slots(factor, FactorType.PRIOR_FACTOR, key)
        return self

    def add_pose_prior(
        self, frame_id: int, pose: gtsam.Pose3 | SE3, noise_model: gtsam.noiseModel.Base
    ) -> "SubGraphBuilder":
        """Add a pose prior to the sub graph."""
        key = X(frame_id)
        if isinstance(pose, SE3):
            pose = pose.as_gtsam_pose()
        prior = gtsam.PriorFactorPose3(key, pose, noise_model)
        self._add_factor_with_slots(prior, FactorType.PRIOR_FACTOR, key)
        return self

    def add_velocity_prior(
        self, frame_id: int, velocity: np.ndarray, noise_model: gtsam.noiseModel.Base
    ) -> "SubGraphBuilder":
        """Add a velocity prior to the sub graph."""
        key = V(frame_id)
        prior = gtsam.PriorFactorVector(key, velocity, noise_model)
        self._add_factor_with_slots(prior, FactorType.PRIOR_FACTOR, key)
        return self

    def add_bias_prior(
        self, frame_id: int, bias: gtsam.imuBias.ConstantBias, noise_model: gtsam.noiseModel.Base
    ) -> "SubGraphBuilder":
        """Add a bias prior to the sub graph."""
        key = B(frame_id)
        prior = gtsam.PriorFactorConstantBias(key, bias, noise_model)
        self._add_factor_with_slots(prior, FactorType.PRIOR_FACTOR, key)
        return self

    def add_freeze_prior_in_all_poses(self) -> "SubGraphBuilder":
        """Add a free prior to all poses in the sub graph."""
        x_symbol_char_value = ord("x")
        for key in self._values.keys():  # noqa: SIM118
            sym = gtsam.Symbol(key)
            if sym.chr() == x_symbol_char_value:
                frame_id = sym.index()
                pose = self._values.atPose3(X(frame_id))
                factor = gtsam.PriorFactorPose3(X(frame_id), pose, self.ctx.freeze_prior_noise)
                self._add_factor_with_slots(factor, FactorType.PRIOR_FACTOR, frame_id)

        return self

    def add_between_keyframe_prior(
        self, frame_id: int, prev_frame_id: int, prev_pose: gtsam.Pose3
    ) -> "SubGraphBuilder":
        """Add a between keyframe prior to the sub graph."""
        if not self._values.exists(X(frame_id)):
            msg = f"Frame {frame_id} does not exist in the sub graph."
            raise ValueError(msg)

        current_pose = self._values.atPose3(X(frame_id))

        between = prev_pose.between(current_pose)
        odometry_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.2, 0.1, 0.1, 0.1]))
        between_factor = gtsam.BetweenFactorPose3(X(prev_frame_id), X(frame_id), between, odometry_noise)
        self._add_factor_with_slots(between_factor, FactorType.BETWEEN_FACTOR, frame_id)
        return self

    def add_smart_factor(self, feat_id: int, measurements: deque[StereoMeasurement]) -> "SubGraphBuilder":
        """Add a smart factor to the sub graph."""
        smart_factor = SmartStereoProjectionPoseFactor(
            sharedNoiseModel=self.ctx.static_smart_noise,
            params=self.ctx.smart_factor_params,
            body_P_sensor=self.ctx.body_sensor_transform,
        )
        for smart_measurement in measurements:
            stereo_point = gtsam.StereoPoint2(smart_measurement.ul, smart_measurement.ur, smart_measurement.v)
            smart_factor.add(stereo_point, smart_measurement.pose_key, self.ctx.stereo_k)
        self._add_factor_with_slots(smart_factor, FactorType.SMART_FACTOR, feat_id)
        return self

    def add_imu_factor(
        self, prev_frame_id: int, next_frame_id: int, pim: gtsam.PreintegratedImuMeasurements
    ) -> "SubGraphBuilder":
        """Add an IMU factor to the sub graph."""
        prev_pose_key = X(prev_frame_id)
        prev_vel_key = V(prev_frame_id)
        next_pose_key = X(next_frame_id)
        next_vel_key = V(next_frame_id)
        prev_bias_key = B(prev_frame_id)
        imu_factor = gtsam.ImuFactor(prev_pose_key, prev_vel_key, next_pose_key, next_vel_key, prev_bias_key, pim)
        self._add_factor_with_slots(imu_factor, FactorType.IMU_FACTOR)
        return self

    def add_between_bias_factor(self, prev_frame_id: int, next_frame_id: int, delta_t: float) -> "SubGraphBuilder":
        """Add a between bias factor to the sub graph."""
        prev_bias_key = B(prev_frame_id)
        next_bias_key = B(next_frame_id)
        zero_bias = gtsam.imuBias.ConstantBias()
        sqrt_delta_t = max(np.sqrt(delta_t), 1e-6)
        acc_sigma = self.ctx.accel_random_walk * sqrt_delta_t
        gyro_sigma = self.ctx.gyro_random_walk * sqrt_delta_t
        bias_walk_model = gtsam.noiseModel.Diagonal.Sigmas(np.array([acc_sigma] * 3 + [gyro_sigma] * 3))
        between_bias_factor = gtsam.BetweenFactorConstantBias(
            prev_bias_key, next_bias_key, zero_bias, bias_walk_model
        )
        self._add_factor_with_slots(between_bias_factor, FactorType.BETWEEN_FACTOR)
        return self

    def _allocate_upper_slot(self) -> int:
        """Allocate a slot in the upper graph."""
        if self._null_slots:
            return self._null_slots.popleft()
        upper_slot = self._upper_graph_size
        self._upper_graph_size += 1
        return upper_slot

    def build_subgraph(self) -> SubGraph:
        """Build the sub graph."""
        if self._keyframe_id == -1:
            raise ValueError("Keyframe id is not set.")
        return SubGraph(
            timestamp=self._timestamp,
            keyframe_id=self._keyframe_id,
            factors=self._factors,
            values=self._values,
            timestamp_map=self._timestamp_map,
            delete_slots=self.get_delete_slots(),
        )

    def build(
        self,
    ) -> tuple[
        gtsam.NonlinearFactorGraph, gtsam.Values, gtsam.FixedLagSmootherKeyTimestampMap, Sequence[SupportsInt]
    ]:
        """Build the sub graph."""
        return self._factors, self._values, self._timestamp_map, list(self._delete_slots)

    def get_timestamp_map(self) -> gtsam.FixedLagSmootherKeyTimestampMap:
        """Get the timestamp map."""
        return self._timestamp_map

    def get_delete_slots(self) -> Sequence[SupportsInt]:
        """Get the delete slots."""
        return list(self._delete_slots)

    def factor_slot(self, feat_id: int) -> int:
        """Get the slot of the factor for a feature."""
        if feat_id not in self._upper_factor_slots:
            msg = f"Feature {feat_id} not found in the factor slots map."
            raise KeyError(msg)
        return self._upper_factor_slots[feat_id]

    def push_delete_slot(self, slot: SupportsInt) -> None:
        """Push a delete slot to the sub graph."""
        self._delete_slots.add(slot)
