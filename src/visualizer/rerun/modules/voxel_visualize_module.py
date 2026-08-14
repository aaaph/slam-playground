from typing import Any, Literal

import numpy as np
import rerun as rr
from pydantic import BaseModel

from core.dense_mapping.voxel_schema import VoxelSchema
from logger import spawn_logger
from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule

type DrawMode = Literal["boxes", "points"]
type FillModeName = Literal[
    "densewireframe",
    "majorwireframe",
    "solid",
    "transparentfillmajorwireframe",
    "DenseWireframe",
    "MajorWireframe",
    "Solid",
    "TransparentFillMajorWireframe",
]

VOXEL_MIN_COLUMNS = 5
VOXEL_COLOR_COLUMNS = 8
VOXEL_SCHEMA_COLUMNS = VoxelSchema.count()


class VoxelVisualizeModuleOptions(BaseModel):
    """Voxel visualization module options."""

    throw_on_nothing: bool = False
    points_size_prop_name: str
    voxel_size_m: float = 0.1
    draw_mode: DrawMode = "boxes"
    point_radius: float = 0.025
    box_edge_radius: float = 0.0025
    fill_mode: FillModeName = "solid"
    label_prefix: str = "voxel"
    show_labels: bool = False
    fallback_color: tuple[int, int, int] = (155, 155, 155)


class VoxelVisualizeModule(IVizModule):
    """Visualize voxel rows with per-voxel color."""

    def __init__(self, property_name: str, entity_path: str, raw_options: dict[str, Any]) -> None:
        """Initialize the voxel visualization module."""
        self.options = VoxelVisualizeModuleOptions(**raw_options)
        self.entity_path = entity_path
        self.property_name = property_name
        self.logger = spawn_logger(VoxelVisualizeModule.__name__)
        if not self.options.points_size_prop_name:
            raise ValueError("points_size_prop_name is required")
        self.points_size_prop_name = self.options.points_size_prop_name
        self.throw_on_nothing = self.options.throw_on_nothing

    def setup(self) -> None:
        """Set up the voxel visualization module."""

    def process(self, context: Ctx) -> None:
        """Process voxel rows shaped by VoxelSchema, with legacy row fallback."""
        exists = context.exists(self.property_name)
        if not exists and self.throw_on_nothing:
            msg = f"Voxel data not found in context: {self.property_name}"
            self.logger.warning(msg)
            raise KeyError(msg)
        if not exists and not self.throw_on_nothing:
            self.logger.trace(f"Voxel data not found in context: {self.property_name}")
            return

        voxel_count = int(context.get_scalar(self.points_size_prop_name))
        if voxel_count == 0:
            return
        voxels = context.get_ndarray(self.property_name, (voxel_count, -1))
        if voxels.shape[1] < VOXEL_MIN_COLUMNS:
            msg = f"Voxel rows must have at least 5 columns, got {voxels.shape[1]}"
            raise ValueError(msg)

        centers = self.resolve_centers(voxels)
        labels = self.resolve_labels(voxels) if self.options.show_labels else None
        colors = self.resolve_colors(voxels)

        if self.options.draw_mode == "points":
            radii = np.full(voxel_count, self.options.point_radius, dtype=np.float32)
            rr.log(
                self.entity_path,
                rr.Points3D(
                    positions=centers,
                    colors=colors,
                    labels=labels,
                    radii=radii,
                    show_labels=self.options.show_labels,
                ),
            )
            return

        sizes = np.full((voxel_count, 3), self.options.voxel_size_m, dtype=np.float32)
        rr.log(
            self.entity_path,
            rr.Boxes3D(
                centers=centers,
                sizes=sizes,
                colors=colors,
                labels=labels,
                radii=self.options.box_edge_radius,
                fill_mode=self.options.fill_mode,
                show_labels=self.options.show_labels,
            ),
        )

    def has_voxel_schema(self, voxels: np.ndarray) -> bool:
        """Check whether rows follow the dense mapping VoxelSchema layout."""
        return voxels.shape[1] >= VOXEL_SCHEMA_COLUMNS

    def resolve_centers(self, voxels: np.ndarray) -> np.ndarray:
        """Resolve voxel centers from schema rows or legacy [id, x, y, z, ...] rows."""
        if self.has_voxel_schema(voxels):
            return voxels[:, VoxelSchema.VOXEL_CENTER].astype(np.float32, copy=False)
        return voxels[:, 1:4].astype(np.float32, copy=False)

    def resolve_labels(self, voxels: np.ndarray) -> np.ndarray:
        """Resolve stable labels from voxel keys or legacy ids."""
        if self.has_voxel_schema(voxels):
            keys = voxels[:, VoxelSchema.VOXEL_KEY].astype(np.int32, copy=False)
            return np.array(
                [f"{self.options.label_prefix}_{key_x}_{key_y}_{key_z}" for key_x, key_y, key_z in keys]
            )
        ids = voxels[:, 0].astype(np.int32, copy=False)
        return np.array([f"{self.options.label_prefix}_{voxel_id}" for voxel_id in ids])

    def resolve_colors(self, voxels: np.ndarray) -> np.ndarray:
        """Resolve RGB colors from schema rows, falling back for legacy rows."""
        if self.has_voxel_schema(voxels):
            return np.clip(voxels[:, VoxelSchema.VOXEL_COLOR], 0.0, 255.0).astype(np.uint8)
        if voxels.shape[1] >= VOXEL_COLOR_COLUMNS:
            return np.clip(voxels[:, 5:8], 0.0, 255.0).astype(np.uint8)
        return np.full((voxels.shape[0], 3), self.options.fallback_color, dtype=np.uint8)

    def __repr__(self) -> str:
        """Return the string representation of the voxel visualization module."""
        return (
            f"VoxelVisualizeModule(entity_path={self.entity_path}, "
            f"property_name={self.property_name}, "
            f"points_size_prop_name={self.points_size_prop_name}, "
            f"draw_mode={self.options.draw_mode})"
        )
