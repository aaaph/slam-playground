import cv2
import numpy as np

import gtsam
from core.camera_model.stereo_camera_ctx import StereoContext
from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature import Feature
from core.front_end.front_end import FrontEnd
from core.front_end.keyframe import Keyframe
from core.graph_optimizer.isam2_optimizer import ISam2Optimizer
from core.transformations.special_euclidian_3_dim import SE3
from dataset.euroc import EurocDataset
from visualizer.foxglove.foxglove_visualizer import FoxgloveVisualizer
from visualizer.foxglove.modules.image_module import ImageModule
from visualizer.foxglove.modules.point_cloud_module import PointCloudModule
from visualizer.foxglove.modules.pose_module import PoseModule
from visualizer.foxglove.modules.selected_keyframes_module import SelectedKeyframesModule
from visualizer.foxglove.modules.static_transform_module import StaticTransformModule
from visualizer.visualizer_context import VisualizerContext


def draw_left_features(
    left_out: np.ndarray, active_features: dict[int, Feature], debug_features: list[int]
) -> None:
    """Draw the features on the concatenated image."""
    for feat in active_features.values():
        meas = feat.get_active_measurement()
        lx, ly = meas.left
        color = feat.feature_color() if feat.feat_id not in debug_features else feat.debug_color
        size = 2 if feat.feat_id not in debug_features else 8
        """ tail = feat.get_tail(0)
        lx_buffer = lx
        ly_buffer = ly
        for u, v in tail:
            cv2.line(left_out, (int(lx_buffer), int(ly_buffer)), (int(u), int(v)), color, 1)
            lx_buffer = u
            ly_buffer = v """
        if meas.is_stereo():
            rx, ry = meas.right
            cv2.line(left_out, (int(lx), int(ly)), (int(rx), int(ry)), color, 1)
        cv2.circle(left_out, (int(lx), int(ly)), size, color, -1)
        """ cv2.putText(
            left_out, f"feat count: {len(active_features)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        ) """


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
slam_opt = ISam2Optimizer.from_stereo_ctx(stereo_ctx)

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
foxglove_visualizer.add_module(
    PoseModule(
        channel_name="/optimized_pose",
        property_name="optimized_pose_se3",
        parent_frame_id="world",
        child_frame_id="optimized_pose",
    )
)
foxglove_visualizer.add_module(ImageModule())
foxglove_visualizer.add_module(SelectedKeyframesModule())
viz = foxglove_visualizer.websocket_viz_gen()

keyframe_history: list[Keyframe] = []
landmarks_db: dict[int, np.ndarray] = {}
debug_features = [9132, 9639, 9165, 12605, 12607, 14803, 14804, 14805, 8806, 13886, 14942, 15025]  # 173, 9639
opt_pose_se3 = get_init_cam0_in_world_se3(euroc_dataset, stereo_ctx)

input("Press Enter to start...")
for stereo_data in stereo_iterator:
    ts = float(stereo_data["timestamp"])
    left, right = np.array(stereo_data["stereo"][0]), np.array(stereo_data["stereo"][1])

    result = slam_fe.feed(ts, (left, right))
    if result.keyframe is not None:
        keyframe_history.append(result.keyframe)
        opt_result = slam_opt.update_by_keyframe(result.keyframe)
        opt_pose_se3 = opt_result * stereo_ctx.body_in_cam0_se3
        l_char = ord("l")
        for key in slam_opt.result.keys():  # noqa: SIM118
            sym = gtsam.Symbol(key)
            if sym.chr() == l_char:
                landmark_id = sym.index()
                landmark_point = slam_opt.result.atPoint3(key)
                landmarks_db[landmark_id] = landmark_point

        most_tall_feat_id = max(landmarks_db.keys(), key=lambda x: landmarks_db[x][2])
        most_tall_feat_point = landmarks_db[most_tall_feat_id]
        # print(f"most tall feat id: {most_tall_feat_id}, point: {most_tall_feat_point}")
    ground_truth_se3 = euroc_dataset.find_nearest_ground_truth_by_timestamp_se3(ts)

    odom_pose_se3 = result.camera_in_world_se3 * stereo_ctx.body_in_cam0_se3
    active_features = result.active_features
    active_features_colors = {feat.feat_id: feat.feature_color() for feat in active_features.values()}

    draw_left_features(result.left, active_features, debug_features)
    viz_points = landmarks_db.copy()
    viz_points.update(result.new_landmarks)
    viz_message = VisualizerContext(
        pointcloud=viz_points,
        body_in_world_se3=odom_pose_se3,
        ground_truth_se3=ground_truth_se3,
        optimized_pose_se3=opt_pose_se3,
        frame=result.left,
        active_feat_colors=active_features_colors,
        selected_keyframes=keyframe_history,
        debug_features=debug_features,
    )
    viz.send(viz_message)
viz.close()
