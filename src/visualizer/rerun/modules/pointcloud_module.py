from typing import Any

import numpy as np
import rerun as rr
from pydantic import BaseModel

from logger import spawn_logger
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class PointcloudModuleOptions(BaseModel):
    """Pointcloud module options."""

    throw_on_nothing: bool = False
    points_size_prop_name: str
    default_color: tuple[int, int, int] = (155, 155, 155)
    label_prefix: str = "feat"
    radii: float | None = None
    position_columns: tuple[int, int, int] = (1, 2, 3)
    id_column: int | None = 0
    color_columns: tuple[int, int, int] | None = None
    show_labels: bool = True


class PointcloudModule(IVizModule):
    """Pointcloud module."""

    def __init__(self, property_name: str, entity_path: str, raw_options: dict[str, Any]) -> None:
        """Initialize the pointcloud module."""
        self.options = PointcloudModuleOptions(**raw_options)
        self.entity_path = entity_path
        self.property_name = property_name
        self.logger = spawn_logger(PointcloudModule.__name__)
        if not self.options.points_size_prop_name:
            raise ValueError("points_size_prop_name is required")
        self.points_size_prop_name = self.options.points_size_prop_name
        self.throw_on_nothing = self.options.throw_on_nothing

    def setup(self) -> None:
        """Set up the pointcloud module."""

    def process(self, context: Ctx) -> None:
        """Process the pointcloud data."""
        exists = context.exists(self.property_name)
        if not exists and self.throw_on_nothing:
            msg = f"Pointcloud data not found in context: {self.property_name}"
            self.logger.warning(msg)
            raise KeyError(msg)
        if not exists and not self.throw_on_nothing:
            self.logger.trace(f"Pointcloud data not found in context: {self.property_name}")
            return
        pointcloud_size = int(context.get_scalar(self.points_size_prop_name))
        if pointcloud_size == 0:
            return

        pointcloud = context.get_ndarray(self.property_name, (pointcloud_size, -1))
        self.validate_columns(pointcloud)

        positions = pointcloud[:, self.options.position_columns].astype(np.float32, copy=False)
        ids = self.resolve_ids(pointcloud)
        colors = self.resolve_colors(pointcloud)
        labels = None
        if self.options.show_labels:
            labels = np.array([f"{self.options.label_prefix}_{point_id}" for point_id in ids])

        points3d_kwargs: dict[str, Any] = {
            "positions": positions,
            "colors": colors,
        }
        if labels is not None:
            points3d_kwargs["labels"] = labels
        if self.options.radii is not None:
            points3d_kwargs["radii"] = np.full(pointcloud_size, self.options.radii, dtype=np.float32)

        rr.log(self.entity_path, rr.Points3D(**points3d_kwargs))

    def validate_columns(self, pointcloud: np.ndarray) -> None:
        """Validate configured pointcloud column indexes."""
        max_position_column = max(self.options.position_columns)
        max_color_column = max(self.options.color_columns) if self.options.color_columns is not None else -1
        max_id_column = self.options.id_column if self.options.id_column is not None else -1
        max_required_column = max(max_position_column, max_color_column, max_id_column)
        if pointcloud.shape[1] <= max_required_column:
            msg = f"Pointcloud has {pointcloud.shape[1]} columns, but column {max_required_column} is required"
            raise ValueError(msg)

    def resolve_ids(self, pointcloud: np.ndarray) -> np.ndarray:
        """Resolve point ids used for labels."""
        if self.options.id_column is None:
            return np.arange(pointcloud.shape[0], dtype=np.int32)
        return pointcloud[:, self.options.id_column].astype(np.int32, copy=False)

    def resolve_colors(self, pointcloud: np.ndarray) -> np.ndarray:
        """Resolve point colors from configured columns or default color."""
        if self.options.color_columns is None:
            return np.full((pointcloud.shape[0], 3), self.options.default_color, dtype=np.uint8)
        return np.clip(pointcloud[:, self.options.color_columns], 0.0, 255.0).astype(np.uint8)

    def __repr__(self) -> str:
        """Return the string representation of the pointcloud module."""
        return (
            f"PointcloudModule(entity_path={self.entity_path}, "
            f"property_name={self.property_name}, "
            f"points_size_prop_name={self.points_size_prop_name})"
        )
