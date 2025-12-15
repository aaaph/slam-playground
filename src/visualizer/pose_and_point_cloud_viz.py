import struct
from collections.abc import Callable, Generator
from functools import wraps

import cv2
import foxglove
import numpy as np
from cv2.typing import MatLike
from foxglove.channels import (
    FrameTransformsChannel,
    PointCloudChannel,
    PoseChannel,
    RawImageChannel,
)
from foxglove.schemas import (
    FrameTransform,
    FrameTransforms,
    PackedElementField,
    PackedElementFieldNumericType,
    PointCloud,
    Pose,
    PosesInFrame,
    Quaternion,
    RawImage,
    Vector3,
)
from foxglove.websocket import Capability

from core.transformations.special_euclidian_3_dim import SE3
from dataset.euroc import GroundTruth
from logger import spawn_logger
from visualizer.foxglove_listener import FoxgloveServerListener

TravelingPoses = dict[float, SE3]
ImageFrame = MatLike
PointCloudDict = dict[int, np.ndarray]
ActiveFeaturesColors = dict[int, tuple[int, int, int]]

PoseAndPointCloudData = tuple[SE3, PointCloudDict, ImageFrame, ActiveFeaturesColors, GroundTruth]


def coroutine(
    func: Callable[..., Generator[None, PoseAndPointCloudData]],
) -> Callable[..., Generator[None, PoseAndPointCloudData]]:
    """Coroutine decorator."""

    @wraps(func)
    def wrapper(*args, **kwargs) -> Generator[None, PoseAndPointCloudData]:  # noqa: ANN002, ANN003
        """Wrap the coroutine."""
        gen = func(*args, **kwargs)
        next(gen)
        return gen

    return wrapper


class FoxgloveVisualizer:
    """Pose and Point cloud Foxglove visualizer."""

    def __init__(self) -> None:
        """Initialize the Pose and Point cloud Foxglove visualizer."""
        self.logger = spawn_logger(app="foxglove_visualizer")

    @staticmethod
    @coroutine
    def pose_and_point_cloud_viz() -> Generator[None, PoseAndPointCloudData]:
        """
        Visualize the pose and point cloud.

        Args:
            se3_transforms: The SE3 transforms to visualize.
            point_cloud: The point cloud to visualize.
            image: The image to visualize.

        Returns:
            A generator that yields the pose and point cloud.

        """
        logger = spawn_logger(app="foxglove_pose_and_point_cloud_viz")
        if hasattr(foxglove, "start_server"):
            server = foxglove.start_server(
                server_listener=FoxgloveServerListener(),
                capabilities=[Capability.ClientPublish],
                supported_encodings=["json"],
            )
        else:
            raise AttributeError("start_server is not available in the foxglove module")

        """ mcap_file = "pnp_motion_only.mcap"
        mcap_path = Path(__file__).parent.parent.parent / mcap_file
        writer = foxglove.open_mcap(mcap_path, allow_overwrite=True) """

        frame_transforms_channel = FrameTransformsChannel("/tf")
        point_cloud_channel = PointCloudChannel("/pointcloud")
        image_channel = RawImageChannel("/frame")
        localization_pose_channel = PoseChannel("/localization")
        ground_truth_pose_channel = PoseChannel("/ground_truth")

        logger.info("Foxglove visualizer started")
        try:
            while True:
                incoming_data = yield
                if incoming_data is None:
                    break
                se3, point_cloud, image, active_feat_colors, ground_truth = incoming_data

                transforms_message: list[FrameTransform] = []
                quat = se3.rotation().as_quat()
                vec = se3.translation()
                transforms_message.append(
                    FrameTransform(
                        parent_frame_id="world",
                        child_frame_id="body",
                        translation=Vector3(x=vec[0], y=vec[1], z=vec[2]),
                        rotation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                    )
                )
                transforms_message.append(
                    FrameTransform(
                        parent_frame_id="world",
                        child_frame_id="ground_truth",
                        translation=Vector3(
                            x=ground_truth["gt_position"][0],
                            y=ground_truth["gt_position"][1],
                            z=ground_truth["gt_position"][2],
                        ),
                        rotation=Quaternion(
                            x=ground_truth["gt_orientation"][0],
                            y=ground_truth["gt_orientation"][1],
                            z=ground_truth["gt_orientation"][2],
                            w=ground_truth["gt_orientation"][3],
                        ),
                    )
                )
                transforms_message.append(
                    FrameTransform(
                        parent_frame_id="world",
                        child_frame_id="pointcloud",
                        translation=Vector3(x=0, y=0, z=0),
                    )
                )
                transforms_message.append(
                    FrameTransform(
                        parent_frame_id="world",
                        child_frame_id="passed_path",
                        translation=Vector3(x=0, y=0, z=0),
                    )
                )
                frame_transforms_channel.log(FrameTransforms(transforms=transforms_message))
                point_cloud_message = FoxgloveVisualizer.make_point_cloud_message(point_cloud, active_feat_colors)
                point_cloud_channel.log(point_cloud_message)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = image.tobytes()
                raw_image_message = RawImage(
                    data=image,
                    step=752 * 3,
                    width=752,
                    height=480,
                    encoding="rgb8",
                )
                image_channel.log(raw_image_message)

                localization_pose_message = Pose(
                    position=Vector3(x=vec[0], y=vec[1], z=vec[2]),
                    orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                )
                localization_pose_channel.log(localization_pose_message)
                ground_truth_pose_message = Pose(
                    position=Vector3(
                        x=ground_truth["gt_position"][0],
                        y=ground_truth["gt_position"][1],
                        z=ground_truth["gt_position"][2],
                    ),
                    orientation=Quaternion(
                        x=ground_truth["gt_orientation"][0],
                        y=ground_truth["gt_orientation"][1],
                        z=ground_truth["gt_orientation"][2],
                        w=ground_truth["gt_orientation"][3],
                    ),
                )
                ground_truth_pose_channel.log(ground_truth_pose_message)
        finally:
            logger.info("Foxglove visualizer stopping")
            server.stop()
            # writer.close()

    @staticmethod
    def make_point_cloud_message(points: PointCloudDict, active_feat_colors: ActiveFeaturesColors) -> PointCloud:
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
    def make_passed_path(poses: TravelingPoses) -> PosesInFrame:
        """Make a passed path message using a dictionary of poses."""
        poses_in_frame = []
        for pose in poses.values():
            quat = pose.rotation().as_quat()
            vec = pose.translation()
            poses_in_frame.append(
                Pose(
                    orientation=Quaternion(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                    position=Vector3(x=vec[0], y=vec[1], z=vec[2]),
                )
            )

        # last_ts = Timestamp(int(list(poses.keys())[-1]))
        return PosesInFrame(frame_id="passed_path", poses=poses_in_frame)
