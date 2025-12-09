import cv2
import numpy as np
from example_slam_front_end_helper import draw_left_features, resolve_pnp_pose

import gtsam
from core.feature_tracker.feature_tracker import FeatureTracker
from core.feature_tracker.feature_triangulation import FeatureTriangulation
from core.transformations.special_euclidian_3_dim import SE3
from dataset.euroc import EurocDataset
from visualizer.pose_and_point_cloud_viz import FoxgloveVisualizer

euroc_dataset = EurocDataset.mh_01_easy()
stereo = euroc_dataset.stereo()
stereo_iterator = stereo.to_iterable_dataset()
first_stereo_data = next(iter(stereo_iterator))
initial_body_in_world = euroc_dataset.find_nearest_ground_truth_by_timestamp(float(first_stereo_data["timestamp"]))
initial_body_in_world_se3 = SE3.from_quat_and_translation(
    initial_body_in_world["gt_orientation"], initial_body_in_world["gt_position"]
)
cam0_in_body_se3 = euroc_dataset.config.transform_tree().nodes["cam0"].t_bs
initial_cam0_in_world_se3 = initial_body_in_world_se3 * cam0_in_body_se3


ft = FeatureTracker(euroc_dataset.config.stereo, feat_amount_per_region=30, feat_retrack_threshold=10)
triang = FeatureTriangulation.from_euroc_config(euroc_dataset.config)

points_3d: dict[int, np.ndarray] = {}  # feat_id -> (x, y, z)
poses = {}
poses[float(first_stereo_data["timestamp"])] = initial_cam0_in_world_se3
smoother = gtsam.IncrementalFixedLagSmoother(smootherLag=5.0)
viz = FoxgloveVisualizer.pose_and_point_cloud_viz()

for stereo_data in stereo_iterator:
    ts = float(stereo_data["timestamp"])
    left, right = np.array(stereo_data["stereo"][0]), np.array(stereo_data["stereo"][1])
    left, right = ft.feed(ts, (left, right))

    left_out = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
    right_out = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
    concatenated = np.concatenate([left_out, right_out], axis=1)

    if ts not in poses:
        # do pnp and save the pose
        object_points = []
        image_points = []
        for feature in ft.iterate_through_features():
            feat_id = feature.feat_id
            _, uv_left, _ = feature.get_active_stereo_pair()
            if feat_id in points_3d:
                feat_3d = points_3d[feat_id]
                object_points.append(feat_3d)
                image_points.append((uv_left[0], uv_left[1]))
        object_points = np.array(object_points)
        image_points = np.array(image_points)
        cam0_in_world_se3 = resolve_pnp_pose(object_points, image_points, euroc_dataset.config.stereo.k_rect_left)
        poses[ts] = cam0_in_world_se3

    for feature in ft.iterate_through_active_features():
        _, left_uv, right_uv = feature.get_active_stereo_pair()
        lx, ly = left_uv
        if feature.feat_id not in points_3d:
            good, feat_in_cam0_vec = triang.make_initial_guess_by_stereo_pair(feature)
            if good:
                cam0_in_world_se3 = poses[ts]
                feat_in_world_vec = (
                    cam0_in_world_se3.rotation().as_matrix() @ feat_in_cam0_vec + cam0_in_world_se3.translation()
                )
                points_3d[feature.feat_id] = feat_in_world_vec

    draw_left_features(left_out, ft)

    points_3d_array = np.array(list(points_3d.values()))
    viz.send((poses[ts], points_3d_array, left_out))

    input()

viz.close()
