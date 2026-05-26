from typing import Any

import rerun as rr
from pydantic import BaseModel

from core.transformations.special_euclidian_3_dim import SE3
from logger import spawn_logger
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class StaticTransformModuleOptions(BaseModel):
    """Static transform module options."""

    axes_length: float = 0.425
    show_axes: bool = True
    throw_on_nothing: bool = False


class StaticTransformModule(IVizModule):
    """Image module."""

    def __init__(self, property_name: str, entity_path: str, raw_options: dict[str, Any]) -> None:
        """Initialize the image module."""
        self.options = StaticTransformModuleOptions(**raw_options)
        self.property_name = property_name
        self.entity_path = entity_path
        self.logger = spawn_logger(StaticTransformModule.__name__)
        self.axes_length = self.options.axes_length
        self.show_axes = self.options.show_axes
        self.throw_on_nothing = self.options.throw_on_nothing

    def setup(self) -> None:
        """Set up the static transform module."""

    def process(self, context: Ctx) -> None:
        """Process the pose data."""
        exists = context.exists(self.property_name)
        if not exists and not self.throw_on_nothing:
            return
        if not exists and self.throw_on_nothing:
            msg = f"Pose data not found in context: {self.property_name}"
            self.logger.warning(msg)
            raise KeyError(msg)
        timestamp = context.get_scalar("timestamp", float)
        transform = context.get_ndarray(self.property_name, (4, 4))
        se3 = SE3.from_matrix(transform)
        vec = se3.translation()
        quat = se3.rotation().as_quat()
        rr.set_time("frame_time", timestamp=timestamp / 1e9)
        rr.log(
            self.entity_path,
            rr.Transform3D(translation=vec, quaternion=quat),
            *([rr.TransformAxes3D(self.axes_length)] if self.show_axes else []),
        )

    def __repr__(self) -> str:
        """Return the string representation of the static transform module."""
        return (
            "StaticTransform("
            f"property_name={self.property_name}, "
            f"entity_path={self.entity_path}, "
            f"show_axes={self.show_axes}, "
            f"axes_length={self.axes_length})"
        )
