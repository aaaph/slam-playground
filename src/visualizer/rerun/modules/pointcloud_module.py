import numpy as np
import rerun as rr

from pipeline.annotations import Ctx
from visualizer.rerun.modules.abc_module import IVizModule


class PointcloudModule(IVizModule):
    """Pointcloud module."""

    def __init__(self, entity_path: str, property_name: str) -> None:
        """Initialize the pointcloud module."""
        self.entity_path = entity_path
        self.property_name = property_name

    def setup(self) -> None:
        """Set up the pointcloud module."""

    def process(self, context: Ctx) -> None:
        """Process the pointcloud data."""
        exists = context.exists(self.property_name)
        if not exists:
            msg = f"Pointcloud data not found in context: {self.property_name}"
            self.logger.warning(msg)
            raise KeyError(msg)
        pointcloud_size = int(context.get_scalar("points_size"))
        pointcloud = context.get_ndarray(self.property_name, (pointcloud_size, 5))
        feat_ids = pointcloud[:, 0].astype(np.int32)

        # active_feat_colors = context.active_feat_colors
        positions = np.full((pointcloud_size, 3), np.nan)
        positions[:, 0:3] = pointcloud[:, 1:4]
        default_color_gray: tuple[int, int, int] = (int(155), int(155), int(155))  # noqa: RUF046, UP018

        colors_dict = dict.fromkeys(feat_ids, default_color_gray)
        """ for feat_id, color in active_feat_colors.items():
            colors_dict[feat_id] = color """

        colors = np.array([colors_dict[feat_id] for feat_id in feat_ids])
        labels = np.array([f"feat_{feat_id}" for feat_id in feat_ids])
        rr.log(
            self.entity_path,
            rr.Points3D(
                positions=positions,
                colors=colors,
                labels=labels,
            ),
        )

    def __repr__(self) -> str:
        """Return the string representation of the pointcloud module."""
        return f"PointcloudModule(entity_path={self.entity_path}, property_name={self.property_name})"
