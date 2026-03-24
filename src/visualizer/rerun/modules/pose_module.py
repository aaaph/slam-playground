import numpy as np
import rerun as rr

from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class PoseModule(IVizModule):
    """Image module."""

    def __init__(self, property_name: str, entity_path: str, axes_length: float = 0.425) -> None:
        """Initialize the image module."""
        self.property_name = property_name
        self.entity_path = entity_path
        self.logger = spawn_logger(PoseModule.__name__)
        self.axes_length = axes_length
        self.cam0_in_body_se3 = SE3.from_matrix(
            np.array(
                [
                    [0.01486554, -0.99988093, 0.0041403, -0.02164015],
                    [0.99955725, 0.01496721, 0.02571553, -0.06467699],
                    [-0.02577444, 0.00375619, 0.99966073, 0.00981073],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        )

    def setup(self) -> None:
        """Set up the pose module."""

    def process(self, context: Ctx) -> None:
        """Process the pose data."""
        exists = context.exists(self.property_name)
        if not exists:
            msg = f"Pose data not found in context: {self.property_name}"
            self.logger.warning(msg)
            raise KeyError(msg)
        timestamp = context.get_scalar("timestamp", float)
        transform = context.get_ndarray(self.property_name, (4, 4))
        se3 = SE3.from_matrix(transform)
        vec = se3.translation()
        quat = se3.rotation().as_quat()
        rr.set_time("sim_time", timestamp=timestamp / 1e9)
        rr.log(
            self.entity_path,
            rr.Transform3D(translation=vec, quaternion=quat),
            rr.TransformAxes3D(self.axes_length),
        )
        rr.log(
            f"{self.entity_path}/cam0",
            rr.Transform3D(
                translation=self.cam0_in_body_se3.translation(),
                quaternion=self.cam0_in_body_se3.rotation().as_quat(),
            ),
            rr.TransformAxes3D(self.axes_length),
        )

    def __repr__(self) -> str:
        """Return the string representation of the pose module."""
        return f"Pose(property_name={self.property_name}, entity_path={self.entity_path})"
