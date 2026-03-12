import numpy as np

import gtsam
from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature import Feature, FeatureStatus
from core.front_end.keyframe import Keyframe
from core.graph_optimizer.sub_graph_builder import GraphContext, SubGraphBuilder
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger

L = gtsam.symbol_shorthand.L
X = gtsam.symbol_shorthand.X


class FixedLagOptimizer:
    """Fixed lag optimizer."""

    def __init__(self, ctx: GraphContext, lag: float = 10.0, ignoring_list: list[int] | None = None) -> None:
        """
        Initialize the Fixed Lag Smoother optimizer.

        Args:
            ctx: Graph context.
            lag: Lag time in nanoseconds.
            ignoring_list: List of landmark ids to ignore.

        """
        self.ctx = ctx
        self.lag = lag
        self.smoother = gtsam.IncrementalFixedLagSmoother(self.lag)
        self.logger = spawn_logger(app="fixed_lag_optimizer")
        self.result = gtsam.Values()
        self.last_keyframe_id = -1
        self.ignoring_list = ignoring_list or []

        self.factors_per_landmark_size = {}

    @classmethod
    def from_stereo_ctx(
        cls, ctx: StereoContext, lag: float = 10.0, ignoring_list: list[int] | None = None
    ) -> "FixedLagOptimizer":
        """Create a fixed lag optimizer from a stereo context."""
        graph_context = GraphContext(ctx)
        return cls(graph_context, lag=lag, ignoring_list=ignoring_list)

    def get_landmarks(self) -> dict[int, np.ndarray]:
        """Get the landmarks from the graph."""
        landmarks = {}
        l_char = ord("l")
        for key in self.result.keys():  # noqa: SIM118
            sym = gtsam.Symbol(key)
            if sym.chr() == l_char:
                index = sym.index()
                point_value = self.result.atPoint3(key)
                landmarks[index] = point_value
        return landmarks

    def get_poses(self) -> dict[int, SE3]:
        """Get the SE3 list from the graph."""
        se3_list = {}
        x_char = ord("x")
        for key in self.result.keys():  # noqa: SIM118
            sym = gtsam.Symbol(key)
            if sym == x_char:
                index = sym.index()
                se3_value = SE3.from_gtsam_pose(self.result.atPose3(key))
                se3_list[index] = se3_value
        return se3_list

    def _landmark_exists(self, landmark_id: int) -> bool:
        """Check if a landmark exists in the graph."""
        return self.result.exists(L(landmark_id))

    def update_by_keyframe(self, keyframe: Keyframe) -> SE3:
        """Update the graph with a keyframe."""
        self.logger.debug(f"Updating graph with {keyframe}")
        ts = keyframe.timestamp
        builder = self.keyframe_builder(keyframe, ts)
        if self.last_keyframe_id == -1:
            # not initiated yet -> add prior factor for initial pose
            builder.add_freeze_prior_in_all_poses()
        else:
            # add between keyframe prior
            prev_pose = self.result.atPose3(X(self.last_keyframe_id))
            builder.add_between_keyframe_prior(keyframe.keyframe_id, self.last_keyframe_id, prev_pose)
        self.last_keyframe_id = keyframe.keyframe_id
        factors, values, timestamp_map, _ = builder.build()
        self.optimize(factors, values, timestamp_map, ts)
        opt_gtsam_pose = self.result.atPose3(X(keyframe.keyframe_id))
        return SE3.from_gtsam_pose(opt_gtsam_pose)

    def update_by_lost_features(self, lost_features: dict[int, Feature]) -> tuple[bool, list[int]]:
        """
        Update the graph with lost features.

        Method removes singelton landmarks from the graph.
        Singelton landmarks are landmarks that are exists in the graph but have only one factor connecting to it.

        Returns:
            True if any landmark was erased, False otherwise.
            List of erased landmark ids.

        """
        erased_landmarks = []
        for landmark_id in lost_features:
            if not self._landmark_exists(landmark_id):
                continue
            if self.factors_per_landmark_size.get(landmark_id, 0) > 1:
                continue
            landmark_key = L(landmark_id)
            self.result.erase(landmark_key)
            self.factors_per_landmark_size.pop(landmark_id, None)
            erased_landmarks.append(landmark_id)
            self.logger.debug(f"Singleton landmark erased: {landmark_id}")
        return (len(erased_landmarks) > 0, erased_landmarks)

    def update_by_lost_features_ids(self, lost_features_ids: list[int]) -> tuple[bool, list[int]]:
        """Update the graph with lost features ids."""
        erased_landmarks = []
        for landmark_id in lost_features_ids:
            if not self._landmark_exists(landmark_id):
                continue
            if self.factors_per_landmark_size.get(landmark_id, 0) > 1:
                continue
            landmark_key = L(landmark_id)
            self.result.erase(landmark_key)
            self.factors_per_landmark_size.pop(landmark_id, None)
            erased_landmarks.append(landmark_id)
            self.logger.debug(f"Singleton landmark erased: {landmark_id}")
        return (len(erased_landmarks) > 0, erased_landmarks)

    def optimize(
        self,
        factors: gtsam.NonlinearFactorGraph,
        values: gtsam.Values,
        timestamp_map: gtsam.FixedLagSmootherKeyTimestampMap,
        ts: float,
    ) -> None:
        """Optimize the graph."""
        size_before = self.result.size()
        ts_size = len(timestamp_map)
        factor_size = factors.size()
        value_size = values.size()
        msg = f"f:{factor_size}, v:{value_size},ts_size: {ts_size} ts:{ts}, size before: {size_before}"
        self.logger.info(msg)
        self.smoother.update(factors, values, timestamp_map)
        self.result = self.smoother.calculateEstimate()

        self.logger.debug(f"Optimized graph with {self.result.size()} values")

    def keyframe_builder(self, keyframe: Keyframe, ts: float) -> SubGraphBuilder:
        """Build a subgraph associated to a keyframe."""
        camera_in_world_se3 = keyframe.pose
        keyframe_id = keyframe.keyframe_id
        active_features = keyframe.active_features
        active_landmarks = keyframe.active_landmarks

        builder = self.ctx.new_builder(ts).with_pose(keyframe_id, camera_in_world_se3)
        for landmark_id, landmark_initial_guess in active_landmarks.items():
            if landmark_id in self.ignoring_list:
                continue
            feat = active_features[landmark_id]
            if feat.state != FeatureStatus.STABLE:
                continue
            if not self._landmark_exists(landmark_id):
                state = feat.state
                msg = (
                    f"Inserting landmark {landmark_id} with initial guess {landmark_initial_guess}(state: {state})"
                )
                self.logger.trace(msg)
                builder.with_landmark(landmark_id, landmark_initial_guess)

        for feature_id, feature in active_features.items():
            if feature_id in self.ignoring_list:
                continue
            if feature.state != FeatureStatus.STABLE:
                continue
            meas = feature.get_active_measurement()
            # meas_type = "stereo" if meas.is_stereo() else "mono"
            """ self.logger.trace(
                f"Adding {meas_type} measurement for feature {feature_id}: {meas}(state: {feature.state})"
            ) """
            builder.add_meas(keyframe_id, feature_id, meas)
            self.factors_per_landmark_size[feature_id] = self.factors_per_landmark_size.get(feature_id, 0) + 1

        return builder
