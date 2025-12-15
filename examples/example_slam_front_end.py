import cv2
import numpy as np
from example_slam_front_end_helper import draw_left_features

from core.feature_tracker.feature_tracker import FeatureTracker
from core.pose_tracker.feature_triangulation import FeatureTriangulation
from core.pose_tracker.pose_tracker import PoseTracker
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
cam0_in_body_se3: SE3 = euroc_dataset.config.transform_tree().nodes["cam0"].t_bs
initial_cam0_in_world_se3 = initial_body_in_world_se3 * cam0_in_body_se3

ft = FeatureTracker(euroc_dataset.config.stereo, feat_amount_per_region=30, feat_retrack_threshold=10)
triang = FeatureTriangulation.from_euroc_config(euroc_dataset.config)
pt = PoseTracker.default_factory(initial_cam0_in_world_se3, euroc_dataset.config.as_stereo_camera_dto())


viz = FoxgloveVisualizer.pose_and_point_cloud_viz()

input()
for stereo_data in stereo_iterator:
    ts = float(stereo_data["timestamp"])
    left, right = np.array(stereo_data["stereo"][0]), np.array(stereo_data["stereo"][1])
    left, right = ft.feed(ts, (left, right))
    left = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)

    features = list(ft.iterate_through_features())

    cam0_in_world_se3, _ = pt.estimate(ts, features)

    body_in_world_se3 = cam0_in_world_se3 * cam0_in_body_se3.inverse()

    draw_left_features(left, ft)
    ground_truth_in_k = euroc_dataset.find_nearest_ground_truth_by_timestamp(ts)

    active_features_colors = ft.get_active_features_colors()
    viz.send((body_in_world_se3, pt.local_map.landmarks.copy(), left, active_features_colors, ground_truth_in_k))

viz.close()
