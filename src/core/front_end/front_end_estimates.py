from dataclasses import dataclass

import gtsam
import numpy as np
from gtsam.gtsam import NavState
from numpy.typing import NDArray

from core.graph_optimizer.optimizer_types import PredictionMode
from core.transformations.special_euclidian_3_dim import SE3


@dataclass(slots=True, frozen=True)
class MotionEstimate:
    """Structure of motion estimate -> pose + velocity."""

    pose: SE3
    velocity: NDArray[np.float64]

    def pose_matrix(self) -> NDArray[np.float64]:
        """Get the pose matrix."""
        return self.pose.as_matrix()

    def nav_state(self) -> NavState:
        """Get the nav state."""
        return gtsam.NavState(self.pose.as_gtsam_pose(), self.velocity)


@dataclass(slots=True, frozen=True)
class FrontEndPoseEstimates:
    """Pose estimates from the front end."""

    pim: MotionEstimate
    pnp: MotionEstimate
    selected_mode: PredictionMode

    @property
    def selected(self) -> MotionEstimate:
        """Get the selected motion estimate."""
        return self.pim if self.selected_mode == PredictionMode.PIM else self.pnp
