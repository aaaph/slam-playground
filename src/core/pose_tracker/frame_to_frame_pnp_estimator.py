from typing import Self

import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext
from core.pose_tracker.frame_to_frame_pnp_store import FrameToFramePnpStore, PnPMapSchema
from core.pose_tracker.pnp_solver import PnpPoseSolver, PnpSolverConfig
from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger

type VisualFeatures = NDArray[np.float64]


class FrameToFramePnPEstimator:
    """Estimator for PnP problem between two frames."""

    def __init__(self, pnp_store: FrameToFramePnpStore, solver: PnpPoseSolver, stereo_ctx: StereoContext) -> None:
        """Initialize the FrameToFramePnPEstimator."""
        self.pnp_store = pnp_store
        self.solver = solver
        self.stereo_ctx = stereo_ctx
        self.iteration = 0
        self.logger = spawn_logger("frame_to_frame_pnp_estimator")

    @classmethod
    def default_factory(cls, stereo_ctx: StereoContext, capacity: int = 400) -> Self:
        """Create a default instance of FrameToFramePnPEstimator."""
        pnp_store = FrameToFramePnpStore.default_factory(capacity)
        pnp_solver = PnpPoseSolver.default_factory(stereo_ctx, config=PnpSolverConfig(motion_only_ba_enabled=True))
        return cls(pnp_store, pnp_solver, stereo_ctx)

    def estimate_pose(self, prev_pose: SE3, visual_features: VisualFeatures) -> SE3:
        """
        Estimate the pose of the body in the current frame using the previous pose and the visual features.

        Method utilizes the pnp store to get the previous 3d landmarks which used to prepare the pnp input data.
        Method utilizes the stereo context convert body frame to camera frame.
        Method utilizes the solver to estimate the pose of the body in the current frame.

        If code is failed to estimate related pose -> return previous pose

        Args:
            prev_pose: The previous pose of the body.
            visual_features: The visual features of the frame.

        Returns:
            The pose of the body in the current frame.

        """
        if self.iteration == 0:
            # first pnp estimation -> we have zero previous points -> just store visual features and return prev
            self.pnp_store.add_features(visual_features)
            self.pnp_store.finish_frame_and_advance()
            self.iteration += 1
            return prev_pose

        prev_cam0_in_world = prev_pose * self.stereo_ctx.cam0_in_body_se3

        current_feat_ids = visual_features[:, PnPMapSchema.FEAT_ID].astype(np.int32, copy=False)
        previous_mask, previous_xyz = self.pnp_store.get_previous_xyz(current_feat_ids)

        matched_visual_features = visual_features[previous_mask].copy()
        matched_visual_features[:, PnPMapSchema.XYZ] = previous_xyz[previous_mask]
        matched_feat_ids = matched_visual_features[:, PnPMapSchema.FEAT_ID].astype(np.int32, copy=False)

        pnp_result = self.solver.solve_visual_features(matched_visual_features)

        if not pnp_result.ok:
            self.logger.error(f"PnP estimation failed: {pnp_result.reason}")
            self.iteration += 1
            return prev_pose

        next_cam0_in_world = prev_cam0_in_world * pnp_result.cam0_in_reference
        next_body_in_world = next_cam0_in_world * self.stereo_ctx.cam0_in_body_se3.inverse()

        self.pnp_store.add_features(visual_features)
        self.pnp_store.update_outlier_streak(matched_feat_ids, pnp_result.inlier_mask)
        self.pnp_store.finish_frame_and_advance()
        self.iteration += 1

        return next_body_in_world
