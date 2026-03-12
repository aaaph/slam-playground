from collections import deque
from collections.abc import Sequence
from typing import SupportsInt

import gtsam_unstable
import numpy as np

import gtsam
from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature import Measurement
from core.graph_optimizer.optimizer_types import FactorType, StereoMeasurement
from core.transformations.special_euclidian_3_dim import SE3

X = gtsam.symbol_shorthand.X
L = gtsam.symbol_shorthand.L


class GraphContext:
    """Graph context."""

    def __init__(self, stereo_ctx: StereoContext) -> None:
        """Initialize the graph context."""
        huber = gtsam.noiseModel.mEstimator.Huber.Create(1.345)
        self.static_stereo_noise = gtsam.noiseModel.Robust.Create(huber, gtsam.noiseModel.Isotropic.Sigma(3, 2.0))
        self.static_mono_noise = gtsam.noiseModel.Robust.Create(huber, gtsam.noiseModel.Isotropic.Sigma(2, 2.0))
        self.static_smart_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1.0)
        self.freeze_prior_noise = gtsam.noiseModel.Constrained.All(6)
        self.smart_factor_params = gtsam.SmartProjectionParams()
        self.smart_factor_params.setDegeneracyMode(gtsam.DegeneracyMode.ZERO_ON_DEGENERACY)
        self.smart_factor_params.setRankTolerance(1.0)
        self.smart_factor_params.setLinearizationMode(gtsam.LinearizationMode.HESSIAN)
        self.body_sensor_transform = stereo_ctx.cam0_in_body_se3.as_gtsam_pose()

        self.stereo_k = stereo_ctx.stereo_k_gtsam
        self.mono_k = stereo_ctx.cam0_k_gtsam

    def new_builder(
        self, null_slots: deque[int] | None = None, factors_graph_size: int = 0, timestamp: float = 0.0
    ) -> "SubGraphBuilder":
        """Create a new builder."""
        return SubGraphBuilder(
            self, timestamp=timestamp, uppergraph_size=factors_graph_size, null_slots=null_slots
        )


class SubGraphBuilder:
    """GTSAM subgraph builder."""

    def __init__(
        self,
        ctx: GraphContext,
        timestamp: float = 0.0,
        uppergraph_size: int = 0,
        null_slots: deque[int] | None = None,
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

    def with_pose(self, frame_id: int, pose: gtsam.Pose3 | SE3) -> "SubGraphBuilder":
        """Add a pose to the sub graph."""
        if isinstance(pose, SE3):
            pose = pose.as_gtsam_pose()
        self._values.insert(X(frame_id), pose)
        self._timestamp_map.insert((X(frame_id), self._timestamp))
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
            self._factors.add(gtsam.PriorFactorPoint3(L(landmark_id), point, uncertainty_factor))
        return self

    def add_meas(self, frame_id: int, feat_id: int, meas: Measurement) -> "SubGraphBuilder":
        """Add a measurement to the sub graph."""
        self._timestamp_map.insert((L(feat_id), self._timestamp))
        if meas.is_stereo():
            uv_left, uv_right = meas.pair()
            ur, _ = uv_right
            ul, v = uv_left
            stereo_point = gtsam.StereoPoint2(ul, ur, v)
            stereo_factor = gtsam.GenericStereoFactor3D(
                stereo_point, self.ctx.static_stereo_noise, X(frame_id), L(feat_id), self.ctx.stereo_k
            )
            self._factors.add(stereo_factor)
        else:
            mono_point = gtsam.Point2(meas.left[0], meas.left[1])
            mono_factor = gtsam.GenericProjectionFactorCal3_S2(
                mono_point,
                self.ctx.static_mono_noise,
                X(frame_id),
                L(feat_id),
                self.ctx.mono_k,
            )
            self._factors.add(mono_factor)
        self._subgraph_factor_slots[feat_id] = self._subgraph_factor_slots_counter
        self._subgraph_factor_types[self._subgraph_factor_slots_counter] = FactorType.LANDMARK
        self._subgraph_factor_slots_counter += 1
        upper_slot = self._allocate_upper_slot()
        self._upper_factor_slots[feat_id] = upper_slot
        self._upper_factor_types[upper_slot] = FactorType.LANDMARK
        return self

    def add_stereo_factor(
        self, pose_key: int, feat_id: int, stereo_point: gtsam.StereoPoint2
    ) -> "SubGraphBuilder":
        """Add a reprojection factor to the sub graph."""
        self._timestamp_map.insert((L(feat_id), self._timestamp))
        stereo_factor = gtsam.GenericStereoFactor3D(
            stereo_point, self.ctx.static_stereo_noise, pose_key, L(feat_id), self.ctx.stereo_k
        )
        self._factors.add(stereo_factor)
        self._subgraph_factor_slots[feat_id] = self._subgraph_factor_slots_counter
        self._subgraph_factor_types[self._subgraph_factor_slots_counter] = FactorType.LANDMARK
        self._subgraph_factor_slots_counter += 1
        upper_slot = self._allocate_upper_slot()
        self._upper_factor_slots[feat_id] = upper_slot
        self._upper_factor_types[upper_slot] = FactorType.LANDMARK
        return self

    def add_freeze_prior(self, frame_id: int) -> "SubGraphBuilder":
        """Add a free prior to the sub graph."""
        prior_noise = gtsam.noiseModel.Constrained.All(6)
        self._factors.add(gtsam.PriorFactorPose3(X(frame_id), gtsam.Pose3(), prior_noise))
        self._subgraph_factor_slots[frame_id] = self._subgraph_factor_slots_counter
        self._subgraph_factor_types[self._subgraph_factor_slots_counter] = FactorType.PRIOR_FACTOR
        self._subgraph_factor_slots_counter += 1
        upper_slot = self._allocate_upper_slot()
        self._upper_factor_slots[frame_id] = upper_slot
        self._upper_factor_types[upper_slot] = FactorType.PRIOR_FACTOR
        return self

    def add_freeze_prior_in_all_poses(self) -> "SubGraphBuilder":
        """Add a free prior to all poses in the sub graph."""
        x_symbol_char_value = ord("x")
        for key in self._values.keys():  # noqa: SIM118
            sym = gtsam.Symbol(key)
            if sym.chr() == x_symbol_char_value:
                frame_id = sym.index()
                pose = self._values.atPose3(X(frame_id))
                self._factors.add(gtsam.PriorFactorPose3(X(frame_id), pose, self.ctx.freeze_prior_noise))
                self._subgraph_factor_slots[frame_id] = self._subgraph_factor_slots_counter
                self._subgraph_factor_types[self._subgraph_factor_slots_counter] = FactorType.PRIOR_FACTOR
                self._subgraph_factor_slots_counter += 1
                upper_slot = self._allocate_upper_slot()
                self._upper_factor_slots[frame_id] = upper_slot
                self._upper_factor_types[upper_slot] = FactorType.PRIOR_FACTOR

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
        self._factors.add(between_factor)
        self._subgraph_factor_slots[frame_id] = self._subgraph_factor_slots_counter
        self._subgraph_factor_types[self._subgraph_factor_slots_counter] = FactorType.BETWEEN_FACTOR
        self._subgraph_factor_slots_counter += 1
        upper_slot = self._allocate_upper_slot()
        self._upper_factor_slots[frame_id] = upper_slot
        self._upper_factor_types[upper_slot] = FactorType.BETWEEN_FACTOR
        return self

    def add_smart_factor(self, feat_id: int, measurements: deque[StereoMeasurement]) -> "SubGraphBuilder":
        """Add a smart factor to the sub graph."""
        smart_factor = gtsam_unstable.SmartStereoProjectionPoseFactor(
            sharedNoiseModel=self.ctx.static_smart_noise,
            params=self.ctx.smart_factor_params,
            body_P_sensor=self.ctx.body_sensor_transform,
        )
        for smart_measurement in measurements:
            stereo_point = gtsam.StereoPoint2(smart_measurement.ul, smart_measurement.ur, smart_measurement.v)
            smart_factor.add(stereo_point, smart_measurement.pose_key, self.ctx.stereo_k)
        self._factors.add(smart_factor)
        self._subgraph_factor_slots[feat_id] = self._subgraph_factor_slots_counter
        self._subgraph_factor_types[self._subgraph_factor_slots_counter] = FactorType.SMART_FACTOR
        self._subgraph_factor_slots_counter += 1
        upper_slot = self._allocate_upper_slot()
        self._upper_factor_slots[feat_id] = upper_slot
        self._upper_factor_types[upper_slot] = FactorType.SMART_FACTOR
        return self

    def _allocate_upper_slot(self) -> int:
        """Allocate a slot in the upper graph."""
        if self._null_slots:
            return self._null_slots.popleft()
        upper_slot = self._upper_graph_size
        self._upper_graph_size += 1
        return upper_slot

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
