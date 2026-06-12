from typing import Any, Self
from warnings import deprecated

import gtsam

from core.camera_model.stereo_camera_ctx import StereoContext
from core.camera_model.vio_context import ImuContext, VioContext
from core.graph_optimizer.sub_graph_builder import GraphContext, SubGraphBuilder
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger

X = gtsam.symbol_shorthand.X
L = gtsam.symbol_shorthand.L

Keyframe = Any


@deprecated("ISam2 optimizer is deprecated, use ExplicitVIOOptimizer or SmartVIOOptimizer instead")
class ISam2Optimizer:
    """ISam2 optimizer."""

    def __init__(
        self,
        ctx: GraphContext,
        isam_params: gtsam.ISAM2Params | None = None,
    ) -> None:
        """Initialize the optimizer."""
        self.ctx = ctx
        self.last_keyframe_id = -1

        if isam_params is None:
            isam_params = gtsam.ISAM2Params()
            isam_params.setRelinearizeThreshold(0.01)
            isam_params.relinearizeSkip = 1
        self.isam_params = isam_params
        self.isam = gtsam.ISAM2(isam_params)
        self.result = gtsam.Values()

        self.logger = spawn_logger(app="isam2_optimizer")

    @classmethod
    def from_stereo_ctx(cls, ctx: StereoContext, isam_params: gtsam.ISAM2Params | None = None) -> Self:
        """Create an optimizer from a stereo context."""
        graph_context = GraphContext(
            VioContext(
                stereo=ctx,
                imu=ImuContext.empty(),
            )
        )
        return cls(graph_context, isam_params=isam_params)

    def _landmark_exists(self, landmark_id: int) -> bool:
        """Check if a landmark exists in the graph."""
        return self.result.exists(L(landmark_id))

    def update_by_keyframe(self, keyframe: Keyframe) -> SE3:
        """Update the graph with a keyframe."""
        builder = self.keyframe_builder(keyframe)
        if self.last_keyframe_id == -1:
            # not initiated yet -> add prior factor for initial pose
            builder.add_freeze_prior_in_all_poses()
        else:
            # add between keyframe prior
            prev_pose = self.result.atPose3(X(self.last_keyframe_id))
            builder.add_between_keyframe_prior(keyframe.keyframe_id, self.last_keyframe_id, prev_pose)
        self.last_keyframe_id = keyframe.keyframe_id

        factors, values, _, _ = builder.build()
        self.optimize(factors, values)
        opt_gtsam_pose = self.result.atPose3(X(keyframe.keyframe_id))
        return SE3.from_gtsam_pose(opt_gtsam_pose)

    def optimize(self, factors: gtsam.NonlinearFactorGraph, values: gtsam.Values) -> None:
        """Optimize the graph."""
        msg = f"Optimizing graph, f:{factors.size()}, v:{values.size()}, size before: {self.result.size()}"
        self.logger.debug(msg)
        self.isam.update(factors, values)
        self.result = self.isam.calculateEstimate()
        self.logger.debug(f"Optimized graph with {self.result.size()} values")

    def keyframe_builder(self, keyframe: Keyframe) -> SubGraphBuilder:
        """Build a subgraph associated to a keyframe."""
        camera_in_world_se3 = keyframe.pose
        keyframe_id = keyframe.keyframe_id
        active_features = keyframe.active_features
        active_landmarks = keyframe.active_landmarks

        builder = self.ctx.new_builder().with_pose(keyframe_id, camera_in_world_se3)
        ignoring_list = [
            9132,
            9639,
            9165,
            12605,
            12607,
            14803,
            14804,
            14805,
            14801,
            8806,
            10504,
            13886,
            14942,
            15025,
        ]
        for landmark_id, landmark_initial_guess in active_landmarks.items():
            if landmark_id in ignoring_list:
                continue
            if not self._landmark_exists(landmark_id):
                self.logger.trace(f"Inserting landmark {landmark_id} with initial guess {landmark_initial_guess}")
                builder.with_landmark(landmark_id, landmark_initial_guess)

        for feature_id, feature in active_features.items():
            if feature_id in ignoring_list:
                continue
            meas = feature.get_active_measurement()
            # meas_type = "stereo" if meas.is_stereo() else "mono"
            # self.logger.trace(f"Adding {meas_type} measurement for feature {feature_id}: {meas}")
            builder.add_stereo_factor(keyframe_id, feature_id, meas)

        return builder
