import numpy as np

import gtsam
from core.camera_model.stereo_camera_ctx import StereoContext
from core.feature_tracker.feature import Measurement
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

        self.freeze_prior_noise = gtsam.noiseModel.Constrained.All(6)

        self.stereo_k = stereo_ctx.stereo_k_gtsam
        self.mono_k = stereo_ctx.cam0_k_gtsam

    def new_builder(self, timestamp: float = 0.0) -> "SubGraphBuilder":
        """Create a new builder."""
        return SubGraphBuilder(self, timestamp)


class SubGraphBuilder:
    """GTSAM subgraph builder."""

    def __init__(self, ctx: GraphContext, timestamp: float = 0.0) -> None:
        """Initialize the sub graph builder."""
        self.ctx = ctx
        self._values = gtsam.Values()
        self._factors = gtsam.NonlinearFactorGraph()
        self._timestamp = timestamp
        self._timestamp_map = gtsam.FixedLagSmootherKeyTimestampMap()

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
        return self

    def add_freeze_prior(self, frame_id: int) -> "SubGraphBuilder":
        """Add a free prior to the sub graph."""
        prior_noise = gtsam.noiseModel.Constrained.All(6)
        self._factors.add(gtsam.PriorFactorPose3(X(frame_id), gtsam.Pose3(), prior_noise))
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
        return self

    def build(self) -> tuple[gtsam.NonlinearFactorGraph, gtsam.Values]:
        """Build the sub graph."""
        return self._factors, self._values

    def get_timestamp_map(self) -> gtsam.FixedLagSmootherKeyTimestampMap:
        """Get the timestamp map."""
        return self._timestamp_map
