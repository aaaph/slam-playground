import cv2
import numpy as np

from core.camera_model.stereo_camera_ctx import StereoContext
from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature import FeatureStatus
from core.feature_tracker.feature_tracker import FeatureTracker
from core.front_end.front_end import FrontEnd
from core.front_end.keyframe import Keyframe
from core.transformations.special_euclidian_3_dim import SE3
from dataset.euroc import EurocDataset
from visualizer.foxglove.foxglove_visualizer import FoxgloveVisualizer
from visualizer.foxglove.modules.image_module import ImageModule
from visualizer.foxglove.modules.point_cloud_module import PointCloudModule
from visualizer.foxglove.modules.pose_module import PoseModule
from visualizer.foxglove.modules.selected_keyframes_module import SelectedKeyframesModule
from visualizer.foxglove.modules.static_transform_module import StaticTransformModule
from visualizer.visualizer_context import VisualizerContext


def draw_left_features(left_out: np.ndarray, ft: FeatureTracker) -> None:
    """Draw the features on the concatenated image."""
    for feat in ft.iterate_through_features([FeatureStatus.NEW, FeatureStatus.TRACKED, FeatureStatus.STABLE]):
        _, left_uv, _ = feat.get_active_stereo_pair()
        lx, ly = left_uv
        cv2.circle(left_out, (int(lx), int(ly)), 2, feat.feature_color(), -1)
        cv2.putText(
            left_out, f"feat count: {ft.feat_count()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )


def get_init_cam0_in_world_se3(euroc_dataset: EurocDataset, stereo_ctx: StereoContext) -> SE3:
    """Get the initial camera pose in the world frame."""
    first_stereo_data = next(iter(euroc_dataset.stereo().to_iterable_dataset()))
    initial_body_in_world = euroc_dataset.find_nearest_ground_truth_by_timestamp(
        float(first_stereo_data["timestamp"])
    )
    initial_body_in_world_quat = initial_body_in_world["gt_orientation"]
    initial_body_in_world_vec = initial_body_in_world["gt_position"]
    initial_body_in_world_se3 = SE3.from_quat_and_translation(
        initial_body_in_world_quat, initial_body_in_world_vec
    )
    cam0_in_body_se3: SE3 = stereo_ctx.cam0_in_body_se3
    return initial_body_in_world_se3 * cam0_in_body_se3


euroc_dataset = EurocDataset.mh_01_easy()
stereo_iterator = euroc_dataset.stereo().to_iterable_dataset()
camera_model = StereoCameraModel.from_cameras_config(euroc_dataset.config.cam0, euroc_dataset.config.cam1)
stereo_ctx = camera_model.as_stereo_ctx()
initial_cam0_in_world_se3 = get_init_cam0_in_world_se3(euroc_dataset, stereo_ctx)

slam_fe = FrontEnd.default_factory(camera_model, initial_cam0_in_world_se3)

foxglove_visualizer = FoxgloveVisualizer()
foxglove_visualizer.add_module(PointCloudModule())
foxglove_visualizer.add_module(
    PoseModule(
        channel_name="/base_link",
        property_name="body_in_world_se3",
        parent_frame_id="odom",
        child_frame_id="base_link",
    )
)
foxglove_visualizer.add_module(
    StaticTransformModule(
        parent_frame_id="world",
        child_frame_id="odom",
        se3=SE3.identity(),
    )
)
foxglove_visualizer.add_module(
    StaticTransformModule(
        parent_frame_id="base_link",
        child_frame_id="cam0",
        se3=stereo_ctx.cam0_in_body_se3,
    )
)
foxglove_visualizer.add_module(
    StaticTransformModule(
        parent_frame_id="base_link",
        child_frame_id="cam1",
        se3=stereo_ctx.cam1_in_body_se3,
    )
)
foxglove_visualizer.add_module(
    PoseModule(
        channel_name="/ground_truth",
        property_name="ground_truth_se3",
        parent_frame_id="world",
        child_frame_id="ground_truth",
    )
)
foxglove_visualizer.add_module(ImageModule())
foxglove_visualizer.add_module(SelectedKeyframesModule())
viz = foxglove_visualizer.websocket_viz_gen()

keyframe_history: list[Keyframe] = []
landmarks_db: dict[int, np.ndarray] = {}

for stereo_data in stereo_iterator:
    ts = float(stereo_data["timestamp"])
    left, right = np.array(stereo_data["stereo"][0]), np.array(stereo_data["stereo"][1])

    result = slam_fe.feed(ts, (left, right))
    if result.keyframe is not None:
        keyframe_history.append(result.keyframe)
    body_in_world_se3 = result.camera_in_world_se3 * stereo_ctx.body_in_cam0_se3
    landmarks_db.update(result.new_landmarks)
    ground_truth_se3 = euroc_dataset.find_nearest_ground_truth_by_timestamp_se3(ts)

    active_features = result.active_features
    active_features_colors = {feat_id: feat.feature_color() for feat_id, feat in active_features.items()}

    viz_message = VisualizerContext(
        pointcloud=landmarks_db.copy(),
        body_in_world_se3=body_in_world_se3,
        ground_truth_se3=ground_truth_se3,
        frame=left,
        active_feat_colors=active_features_colors,
        selected_keyframes=keyframe_history,
    )
    viz.send(viz_message)

viz.close()
