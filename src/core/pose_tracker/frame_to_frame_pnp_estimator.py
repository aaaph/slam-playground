import numpy as np
from numpy.typing import NDArray

from core.camera_model.stereo_camera_ctx import StereoContext
from core.pose_tracker.frame_to_frame_pnp_store import FrameToFramePnpStore
from core.pose_tracker.pnp_solver import PnpPoseSolver
from core.transformations.special_euclidian_3_dim import SE3

type VisualFeatures = NDArray[np.float64]


class FrameToFramePnPEstimator:
    """Estimator for PnP problem between two frames."""

    def __init__(self, pnp_store: FrameToFramePnpStore, solver: PnpPoseSolver, stereo_ctx: StereoContext) -> None:
        """Initialize the FrameToFramePnPEstimator."""
        self.pnp_store = pnp_store
        self.solver = solver
        self.stereo_ctx = stereo_ctx

    def estimate_pose(self, _prev_pose: SE3, _visual_features: VisualFeatures) -> SE3:
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
        return SE3.identity()
