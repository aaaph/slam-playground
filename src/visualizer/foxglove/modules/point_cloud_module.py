import struct

import numpy as np
from foxglove.channels import PointCloudChannel
from foxglove.schemas import (
    FrameTransform,
    PackedElementField,
    PackedElementFieldNumericType,
    PointCloud,
    Pose,
    Quaternion,
    Vector3,
)

from visualizer.foxglove.modules.abc_module import IVizModule
from visualizer.visualizer_context import VisualizerContext

PointCloudDict = dict[int, np.ndarray]
ActiveFeaturesColors = dict[int, tuple[int, int, int]]


class PointCloudModule(IVizModule):
    """Point cloud module."""

    def setup(self) -> None:
        """Set up the point cloud module visualization."""
        self.point_cloud_channel = PointCloudChannel("/pointcloud")

    def process(self, context: VisualizerContext) -> list[FrameTransform]:
        """Process the point cloud data."""
        if context.pointcloud is None:
            raise ValueError("Point cloud data not found")
        debug_features = context.debug_features
        if debug_features is None:
            debug_features = []
        active_feat_colors = context.active_feat_colors
        points: PointCloudDict = context.pointcloud
        point_cloud_message = PointCloudModule.make_point_cloud_message(points, active_feat_colors, debug_features)
        self.point_cloud_channel.log(point_cloud_message)
        return [
            FrameTransform(
                parent_frame_id="world",
                child_frame_id="pointcloud",
                translation=Vector3(x=0, y=0, z=0),
                rotation=Quaternion(x=0, y=0, z=0, w=1),
            )
        ]

    @staticmethod
    def make_point_cloud_message(
        points: PointCloudDict, active_feat_colors: ActiveFeaturesColors, debug_features: list[int]
    ) -> PointCloud:
        """Make a point cloud message using a 3d points array."""
        point_struct = struct.Struct("<fffBBBB")
        f32 = PackedElementFieldNumericType.Float32
        u8 = PackedElementFieldNumericType.Uint8
        points_array = np.array(list(points.values()))
        buffer = bytearray(point_struct.size * len(points_array))
        for i, (feat_id, point) in enumerate(zip(points.keys(), points_array, strict=False)):
            x, y, z = point
            r, g, b, a = 155, 155, 155, 255
            if feat_id in active_feat_colors:
                r, g, b = active_feat_colors[feat_id]
            if feat_id in debug_features:
                r, g, b = (0, 0, 255)
            r, g, b = PointCloudModule.rgb_to_bgr((r, g, b))
            point_struct.pack_into(buffer, i * point_struct.size, x, y, z, r, g, b, a)

        return PointCloud(
            frame_id="pointcloud",
            pose=Pose(
                position=Vector3(x=0, y=0, z=0),
                orientation=Quaternion(x=0, y=0, z=0, w=1),
            ),
            point_stride=16,
            fields=[
                PackedElementField(name="x", offset=0, type=f32),
                PackedElementField(name="y", offset=4, type=f32),
                PackedElementField(name="z", offset=8, type=f32),
                PackedElementField(name="red", offset=12, type=u8),
                PackedElementField(name="green", offset=13, type=u8),
                PackedElementField(name="blue", offset=14, type=u8),
                PackedElementField(name="alpha", offset=15, type=u8),
            ],
            data=bytes(buffer),
        )

    @staticmethod
    def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
        """Convert RGB to BGR."""
        return (rgb[2], rgb[1], rgb[0])
