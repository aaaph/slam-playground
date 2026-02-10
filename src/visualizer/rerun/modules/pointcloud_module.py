import numpy as np
import rerun as rr

from visualizer.rerun.modules.abc_module import IVizModule
from visualizer.visualizer_context import VisualizerContext


class PointcloudModule(IVizModule):
    """Pointcloud module."""

    def __init__(self, entity_path: str, property_name: str) -> None:
        """Initialize the pointcloud module."""
        self.entity_path = entity_path
        self.property_name = property_name

    def setup(self) -> None:
        """Set up the pointcloud module."""

    def process(self, context: VisualizerContext) -> None:
        """Process the pointcloud data."""
        pointcloud = context.pointcloud
        if pointcloud is None:
            raise ValueError("Pointcloud data not found")

        active_feat_colors = context.active_feat_colors
        positions = np.array(list(pointcloud.values()))
        default_color_gray: tuple[int, int, int] = (int(155), int(155), int(155))  # noqa: RUF046, UP018

        colors_dict = dict.fromkeys(pointcloud, default_color_gray)
        for feat_id, color in active_feat_colors.items():
            colors_dict[feat_id] = color

        colors = np.array([colors_dict[feat_id] for feat_id in pointcloud])
        labels = np.array([f"feat_{feat_id}" for feat_id in pointcloud])
        rr.log(
            self.entity_path,
            rr.Points3D(
                positions=positions,
                colors=colors,
                labels=labels,
            ),
        )
