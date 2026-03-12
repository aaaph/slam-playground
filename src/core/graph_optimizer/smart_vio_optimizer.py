from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING, SupportsInt, cast

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

import gtsam
from core.camera_model.stereo_camera_ctx import StereoContext
from core.graph_optimizer.optimizer_types import (
    FeatureId,
    FeatureStatus,
    FeatureTrack,
    OptimizedPose,
    OptKeyframe,
    StereoMeasurement,
)
from core.graph_optimizer.sub_graph_builder import GraphContext, SubGraphBuilder
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger

if TYPE_CHECKING:
    from gtsam_unstable import SmartStereoProjectionPoseFactor
X = gtsam.symbol_shorthand.X
L = gtsam.symbol_shorthand.L


class SmartVIOOptimizer:
    """Smart VIO optimizer."""

    def __init__(
        self,
        ctx: GraphContext,
        stereo_k: gtsam.Cal3_S2Stereo,
        body_sensor_transform: gtsam.Pose3,
        lag: float = 10.0,
    ) -> None:
        """Initialize the smart VIO optimizer."""
        self.logger = spawn_logger(app="smart_vio_optimizer")
        self.ctx = ctx
        self.smart_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        self.smart_params = gtsam.SmartProjectionParams()
        self.stereo_k = stereo_k
        self.body_sensor_transform = body_sensor_transform
        self.smoother = gtsam.IncrementalFixedLagSmoother(lag)
        self.result = gtsam.Values()
        self.lag = lag

        self.last_keyframe_id = -1
        self.sliding_window_poses: dict[int, float] = {}
        self.sliding_window_poses_dq = deque()
        self.tracks: dict[FeatureId, FeatureTrack] = {}

    @classmethod
    def from_stereo_ctx(cls, stereo_ctx: StereoContext, lag: float = 10.0) -> "SmartVIOOptimizer":
        """Create a SmartVIOOptimizer from a stereo context."""
        graph_ctx = GraphContext(stereo_ctx)
        return cls(graph_ctx, stereo_ctx.stereo_k_gtsam, stereo_ctx.cam0_in_body_se3.as_gtsam_pose(), lag=lag)

    def optimize(
        self,
        factors: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
        timestamp_map: gtsam.FixedLagSmootherKeyTimestampMap,
        ts: float,
        delete_slots: Sequence[SupportsInt],
    ) -> None:
        """Optimize the VIO."""
        size_before = self.result.size()
        ts_size = len(timestamp_map)
        factor_size = factors.size()
        value_size = values.size()
        msg = f"f:{factor_size}, v:{value_size},ts_size: {ts_size} ts:{ts}, size before: {size_before}"
        self.logger.info(msg)
        _ = self.smoother.update(factors, values, timestamp_map, delete_slots)
        self.result = self.smoother.calculateEstimate()
        self.logger.debug(f"Optimized graph with {self.result.size()} values")

    def init_keyframe_builder(self, keyframe: OptKeyframe) -> SubGraphBuilder:
        """Initialize the keyframe builder."""
        sub_graph_builder = self.ctx.new_builder(
            timestamp=keyframe.timestamp,
            factors_graph_size=self.smoother.getFactors().size(),
            null_slots=self.get_nullptr_slots(),
        )
        sub_graph_builder.with_pose(keyframe.keyframe_id, keyframe.pose)
        if self.last_keyframe_id == -1:
            sub_graph_builder.add_freeze_prior_in_all_poses()
        else:
            sub_graph_builder.add_between_keyframe_prior(
                keyframe.keyframe_id, self.last_keyframe_id, self.result.atPose3(X(self.last_keyframe_id))
            )
        self.last_keyframe_id = keyframe.keyframe_id
        return sub_graph_builder

    def add_new_keyframe(self, keyframe: OptKeyframe) -> OptimizedPose:
        """Add a new keyframe to the optimizer."""
        self.clear_marginilized_feat_ids()
        pose_key: int = X(keyframe.keyframe_id)
        sub_graph_builder = self.init_keyframe_builder(keyframe)

        pose_to_mariginalize = self.get_marginilize_candidates(keyframe.timestamp)
        self._promote_smart_factors(pose_to_mariginalize)

        for fid, left_u, left_v, right_u, _ in keyframe.active_track:
            feat_id = int(fid)
            feat_track = self.tracks.get(feat_id)
            if not feat_track:
                feat_track = FeatureTrack(feat_id)
                self.tracks[feat_id] = feat_track
            new_measurement = StereoMeasurement(pose_key, left_u, right_u, left_v)
            match feat_track.status:
                case FeatureStatus.EMPTY | FeatureStatus.MARGNILIZED:
                    self.logger.trace(f"{feat_id}: from EMPTY to SMART_FACTOR")
                    feat_track.history.append(new_measurement)
                    sub_graph_builder.add_smart_factor(feat_id, feat_track.history)
                    feat_track.slot = sub_graph_builder.factor_slot(feat_id)
                    feat_track.status = FeatureStatus.SMART_FACTOR
                case FeatureStatus.SMART_FACTOR:
                    self.logger.trace(f"{feat_id}: from SMART_FACTOR to SMART_FACTOR")
                    sub_graph_builder.push_delete_slot(feat_track.slot)
                    feat_track.history.append(new_measurement)
                    sub_graph_builder.add_smart_factor(feat_id, feat_track.history)
                    feat_track.slot = sub_graph_builder.factor_slot(feat_id)
                case FeatureStatus.SMART_TO_MARGNILIZED:
                    self.logger.trace(f"{feat_id}: from SMART_TO_MARGNILIZED to SMART_FACTOR")
                    sub_graph_builder.push_delete_slot(feat_track.slot)
                    feat_track.history.append(new_measurement)
                    sub_graph_builder.add_smart_factor(feat_id, feat_track.history)
                    feat_track.slot = sub_graph_builder.factor_slot(feat_id)
                    feat_track.status = FeatureStatus.MARGNILIZED
                case FeatureStatus.SMART_TO_EXPLICIT:
                    self.logger.trace(f"{feat_id}: from SMART_TO_EXPLICIT to SMART_FACTOR")
                    feat_track.history.append(new_measurement)
                    while feat_track.history and feat_track.history[0].pose_key in pose_to_mariginalize:
                        feat_track.history.popleft()  # keep only actual measurements for sliding window
                    sub_graph_builder.with_landmark(feat_id, feat_track.cached_point)
                    for meas in feat_track.history:
                        stereo_point = gtsam.StereoPoint2(meas.ul, meas.ur, meas.v)
                        sub_graph_builder.add_stereo_factor(meas.pose_key, feat_id, stereo_point)
                    feat_track.status = FeatureStatus.EXPLICIT_LANDMARK
                    feat_track.slot = -1
                case FeatureStatus.EXPLICIT_LANDMARK:
                    self.logger.trace(f"{feat_id}: from EXPLICIT_LANDMARK to EXPLICIT_LANDMARK")
                    stereo_point = gtsam.StereoPoint2(left_u, right_u, left_v)
                    sub_graph_builder.add_stereo_factor(pose_key, feat_id, stereo_point)

        factors, values, timestamp_map, delete_slots = sub_graph_builder.build()

        self.optimize(factors, values, timestamp_map, keyframe.timestamp, delete_slots)

        self._extend_sliding_window(pose_key, keyframe.timestamp)
        self._remove_from_sliding_window(pose_to_mariginalize)
        return SE3.from_gtsam_pose(self.result.atPose3(X(keyframe.keyframe_id)))

    def _promote_smart_factors(self, pose_to_mariginalize: list[int]) -> None:
        """Promote the smart factors to explicit landmarks."""
        for feat_track in self.tracks.values():
            feat_id = feat_track.feat_id
            match feat_track.status:
                case FeatureStatus.SMART_FACTOR:
                    history = feat_track.history
                    if history and history[0].pose_key in pose_to_mariginalize:
                        slot = feat_track.slot
                        factor = self.smoother.getFactors().at(slot)
                        factor = cast("SmartStereoProjectionPoseFactor", factor)
                        if factor.isValid():
                            point_result = factor.point(self.result)  # ty:ignore
                            feat_track.status = FeatureStatus.SMART_TO_EXPLICIT
                            feat_track.cached_point = point_result.get()
                            self.logger.trace(f"{feat_id}: from SMART_FACTOR to SMART_TO_EXPLICIT")
                        else:
                            self.logger.trace(f"{feat_id}: from SMART_FACTOR to SMART_TO_MARGNILIZED")
                            feat_track.status = FeatureStatus.SMART_TO_MARGNILIZED

    def _extend_sliding_window(self, pose_key: int, timestamp: float) -> None:
        """Extend the sliding window."""
        self.sliding_window_poses_dq.append(pose_key)
        self.sliding_window_poses[pose_key] = timestamp

    def _remove_from_sliding_window(self, marginalized_poses: list[int]) -> None:
        """Remove the marginilize candidates from the sliding window."""
        for pose_key in marginalized_poses:
            self.sliding_window_poses.pop(pose_key, None)
            if self.sliding_window_poses_dq and self.sliding_window_poses_dq[0] == pose_key:
                self.sliding_window_poses_dq.popleft()

    def get_marginilize_candidates(self, timestamp: float) -> list[int]:
        """Get the marginilize pose ids based on the sliding window poses."""
        window_diff = timestamp - self.lag
        marginilize_candidates: list[int] = []
        for pose_key in self.sliding_window_poses_dq:
            if self.sliding_window_poses[pose_key] < window_diff:
                marginilize_candidates.append(pose_key)
            else:
                break

        return marginilize_candidates

    def clear_marginilized_feat_ids(self) -> None:
        """
        Clear the marginilized feat ids before adding a new keyframe.

        Need to remove items from the dict to avoid memory leak.
        """
        to_clear = [
            feat_track.feat_id
            for feat_track in self.tracks.values()
            if feat_track.status == FeatureStatus.MARGNILIZED
        ]
        for feat_id in to_clear:
            del self.tracks[feat_id]

    def get_points(self) -> dict[int, np.ndarray]:
        """Get the points from the optimizer."""
        points = {}
        for feat_track in self.tracks.values():
            feat_id = feat_track.feat_id
            match feat_track.status:
                case FeatureStatus.SMART_FACTOR:
                    slot = feat_track.slot
                    smart_factor = self.smoother.getFactors().at(slot)
                    smart_factor = cast("SmartStereoProjectionPoseFactor", smart_factor)
                    point_result = smart_factor.point(self.result)  # ty:ignore
                    point_status = point_result.status

                    if point_status == gtsam.TriangulationResult.Status.VALID:
                        points[feat_id] = np.array([*point_result.get(), point_status.value])
                    else:
                        points[feat_id] = np.array([np.nan, np.nan, np.nan, point_status.value])
                case FeatureStatus.MARGNILIZED:
                    degenerate_status = gtsam.TriangulationResult.Status.DEGENERATE.value
                    points[feat_id] = np.array([np.nan, np.nan, np.nan, degenerate_status])

        return points

    def get_points_ndarray(self) -> NDArray[np.float32]:  # feat_id, x, y, z, feature_status
        """Get the points from the optimizer."""
        feat_count = len(self.tracks)
        points_ndarray = np.zeros((feat_count, 5), dtype=np.float32)

        for idx, feat_track in enumerate(self.tracks.values()):
            match feat_track.status:
                case FeatureStatus.SMART_FACTOR:
                    slot = feat_track.slot
                    smart_factor = self.smoother.getFactors().at(slot)
                    smart_factor = cast("SmartStereoProjectionPoseFactor", smart_factor)
                    point_result = smart_factor.point(self.result)  # ty:ignore
                    point_status = point_result.status
                    if point_status == gtsam.TriangulationResult.Status.VALID:
                        point_value = point_result.get()
                        points_ndarray[idx, 0] = feat_track.feat_id
                        points_ndarray[idx, 1] = point_value[0]
                        points_ndarray[idx, 2] = point_value[1]
                        points_ndarray[idx, 3] = point_value[2]
                        points_ndarray[idx, 4] = feat_track.status.value

                    else:
                        points_ndarray[idx, 0] = feat_track.feat_id
                        points_ndarray[idx, 1:4] = np.nan
                        points_ndarray[idx, 4] = feat_track.status.value
                case FeatureStatus.MARGNILIZED:
                    points_ndarray[idx, 0] = feat_track.feat_id
                    points_ndarray[idx, 1:4] = np.nan
                    points_ndarray[idx, 4] = feat_track.status.value
                case FeatureStatus.EXPLICIT_LANDMARK:
                    landmark_key = L(feat_track.feat_id)
                    landmark = self.result.atPoint3(landmark_key)
                    points_ndarray[idx, 0] = feat_track.feat_id
                    points_ndarray[idx, 1] = landmark[0]
                    points_ndarray[idx, 2] = landmark[1]
                    points_ndarray[idx, 3] = landmark[2]
                    points_ndarray[idx, 4] = feat_track.status.value
        return points_ndarray

    def get_nullptr_slots(self) -> deque[int]:
        """Get the nullptr slots."""
        null_slots = deque()
        factors = self.smoother.getFactors()
        size = factors.size()
        for i in range(size):
            factor = factors.at(i)
            if factor is None:
                null_slots.append(i)
        return null_slots

    def get_graph_arrow(self) -> dict[str, pa.Table]:
        """Get the graph in pyarrow."""
        node_ids = []
        node_types = []
        node_labels = []
        x_symbol_char_value = ord("x")
        l_symbol_char_value = ord("l")

        edges = []

        for val in self.result.keys():  # noqa: SIM118
            node_ids.append(val)
            sym = gtsam.Symbol(val)
            sym_index = sym.index()
            if sym.chr() == x_symbol_char_value:
                node_labels.append(f"x{sym_index}")
                node_types.append("pose")
            elif sym.chr() == l_symbol_char_value:
                node_labels.append(f"l{sym_index}")
                node_types.append("landmark")
            else:
                node_types.append("undefined")
                node_labels.append(f"undefined_{sym_index}")

        factors = self.smoother.getFactors()
        factors_size = factors.size()
        for i in range(factors_size):
            factor = factors.at(i)
            if factor is None:
                continue
            factor = cast("gtsam.NonlinearFactor", factor)
            node_ids.append(i)
            node_types.append("factor")
            node_labels.append(f"f{i}")
            edges.extend((i, key) for key in factor.keys())  # noqa: SIM118

        return {
            "nodes": pa.Table.from_pydict(
                {
                    "ids": node_ids,
                    "types": node_types,
                    "labes": node_labels,
                }
            ),
            "edges": pa.Table.from_pydict(
                {
                    "tuples": edges,
                }
            ),
        }
