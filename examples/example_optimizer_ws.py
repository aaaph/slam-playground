import numpy as np

from core.camera_model.stereo_camera_ctx import StereoContext
from core.camera_model.stereo_camera_model import StereoCameraModel
from core.front_end.front_end import FrontEnd
from core.front_end.keyframe import Keyframe
from core.graph_optimizer.fixed_lag_optimizer import FixedLagOptimizer
from core.transformations.helpers import calculate_ape, calculate_rpe
from core.transformations.special_euclidian_3_dim import SE3
from dataset.euroc import EurocDataset
from visualizer.foxglove.factories.foxglove_default_factory import FoxgloveFactory
from visualizer.opencv.helpers import draw_features_on_left
from visualizer.visualizer_context import VisualizerContext


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
slam_opt = FixedLagOptimizer.from_stereo_ctx(stereo_ctx, lag=3.0 * 1e9, ignoring_list=[649])
viz = FoxgloveFactory().create_default_viz(stereo_ctx, viz_type="websocket")

keyframe_history: list[Keyframe] = []
landmarks_db: dict[int, np.ndarray] = {}

opt_pose_se3 = get_init_cam0_in_world_se3(euroc_dataset, stereo_ctx)
pose_history: list[SE3] = []
last_estimated = None
last_ground_truth = None
input("Press Enter to start...")
for stereo_data in stereo_iterator:
    ts = float(stereo_data["timestamp"])
    left, right = np.array(stereo_data["stereo"][0]), np.array(stereo_data["stereo"][1])

    result = slam_fe.feed(ts, (left, right))
    erased_any, erased_landmarks = slam_opt.update_by_lost_features(result.lost_features)
    if erased_any:
        for landmark_id in erased_landmarks:
            landmarks_db.pop(landmark_id, None)

    if result.keyframe is not None:
        opt_result = slam_opt.update_by_keyframe(result.keyframe)
        opt_pose_se3 = opt_result * stereo_ctx.body_in_cam0_se3

        # landmarks_db.clear()
        slam_fe.pose_tracker.local_map.clear()
        optimized_landmarks = slam_opt.get_landmarks()

        landmarks_db.update(optimized_landmarks)
        slam_fe.pose_tracker.local_map.add_points(optimized_landmarks)

        optimized_poses = slam_opt.get_poses()
        pose_history.clear()
        pose_history.extend(optimized_poses.values())

        # print(f"most tall feat id: {most_tall_feat_id}, point: {most_tall_feat_point}")
    ground_truth_se3 = euroc_dataset.find_nearest_ground_truth_by_timestamp_se3(ts)

    odom_pose_se3 = result.camera_in_world_se3 * stereo_ctx.body_in_cam0_se3
    active_features = result.active_features
    active_features_colors = {feat.feat_id: feat.feature_color() for feat in active_features.values()}

    left_out = draw_features_on_left(result.left, active_features)
    if last_estimated is not None and last_ground_truth is not None:
        delta_estimated = opt_pose_se3 * last_estimated.inverse()
        delta_ground_truth = ground_truth_se3 * last_ground_truth.inverse()
        relative_errors = calculate_rpe(delta_estimated, delta_ground_truth)
    else:
        relative_errors = (0.0, 0.0)
    ape = calculate_ape(opt_pose_se3, ground_truth_se3)
    viz_message = VisualizerContext(
        pointcloud=landmarks_db,
        body_in_world_se3=odom_pose_se3,
        ground_truth_se3=ground_truth_se3,
        optimized_pose_se3=opt_pose_se3,
        frame=left_out,
        active_feat_colors=active_features_colors,
        selected_keyframes=slam_fe.keyframe_history,
        pose_history=pose_history,
        errors=np.array([ape[0], ape[1], relative_errors[0], relative_errors[1]]),
    )
    last_estimated = opt_pose_se3
    last_ground_truth = ground_truth_se3
    viz.send(viz_message)
viz.close()
