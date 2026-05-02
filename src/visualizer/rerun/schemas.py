from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class LayoutType(StrEnum):
    """Layout type for the rerun config."""

    VERTICAL = "Vertical"
    HORIZONTAL = "Horizontal"


class ViewType(StrEnum):
    """View type for the rerun config."""

    SPATIAL_2D = "Spatial2D"
    SPATIAL_3D = "Spatial3D"
    TIME_SERIES = "TimeSeries"
    CONTAINER = "Container"


class ModuleType(StrEnum):
    """Module type for the rerun config."""

    IMAGE = "image"
    FEATURES = "features"
    POINTCLOUD = "pointcloud"
    DYNAMIC_TRANSFORM = "dynamic_transform"
    STATIC_TRANSFORM = "static_transform"
    PLOT_COLUMN = "plot_column"
    PLOT_SCALAR = "plot_scalar"
    PLOT_3D_VECTOR = "plot_3d_vector"


class EntitySchema(BaseModel):
    """Entity schema for the rerun config."""

    id: str
    module: ModuleType
    entity: str
    options: dict[str, Any] = Field(default_factory=dict)


class ColorsSchema(BaseModel):
    """Shared color palette for rerun config."""

    x_axis_default: list[int] = Field(default_factory=lambda: [255, 60, 60])
    y_axis_default: list[int] = Field(default_factory=lambda: [60, 255, 160])
    z_axis_default: list[int] = Field(default_factory=lambda: [30, 210, 255])


class ViewSchema(BaseModel):
    """View schema for the rerun config."""

    name: str
    type: ViewType
    origin: str | None = None
    layout: LayoutType | None = None
    streams: list[EntitySchema] = Field(default_factory=list)
    views: list["ViewSchema"] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class RerunConfigSchema(BaseModel):
    """Rerun config schema."""

    app_name: str | None = None
    resolution: tuple[int, int] | None = None
    colors: ColorsSchema = Field(default_factory=ColorsSchema)
    views: list[ViewSchema]


ViewSchema.model_rebuild()
