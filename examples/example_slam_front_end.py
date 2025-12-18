import cv2
import numpy as np

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature_tracker import FeatureTracker
from core.pose_tracker.pose_tracker import PoseTracker
from core.transformations.special_euclidian_3_dim import SE3
from dataset.euroc import EurocDataset
from visualizer.pose_and_point_cloud_viz import FoxgloveVisualizer


def draw_left_features(left_out: np.ndarray, ft: FeatureTracker) -> None:
    """Draw the features on the concatenated image."""
    for feat in ft.iterate_through_features(["new", "tracked", "stable"]):
        _, left_uv, _ = feat.get_active_stereo_pair()
        lx, ly = left_uv
        cv2.circle(left_out, (int(lx), int(ly)), 2, feat.feature_color(), -1)
        cv2.putText(
            left_out, f"feat count: {ft.feat_count()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )


euroc_dataset = EurocDataset.mh_01_easy()
camera_model = StereoCameraModel.from_cameras_config(euroc_dataset.config.cam0, euroc_dataset.config.cam1)
stereo_ctx = camera_model.as_stereo_ctx()
stereo = euroc_dataset.stereo()
stereo_iterator = stereo.to_iterable_dataset()
first_stereo_data = next(iter(stereo_iterator))
initial_body_in_world = euroc_dataset.find_nearest_ground_truth_by_timestamp(float(first_stereo_data["timestamp"]))
initial_body_in_world_se3 = SE3.from_quat_and_translation(
    initial_body_in_world["gt_orientation"], initial_body_in_world["gt_position"]
)
cam0_in_body_se3: SE3 = stereo_ctx.cam0_in_body
initial_cam0_in_world_se3 = initial_body_in_world_se3 * cam0_in_body_se3

ft = FeatureTracker.default_factory(stereo_ctx, feat_amount_per_region=30, feat_retrack_threshold=10)
pt = PoseTracker.default_factory(initial_cam0_in_world_se3, stereo_ctx)


viz = FoxgloveVisualizer.pose_and_point_cloud_viz()

input()
for stereo_data in stereo_iterator:
    ts = float(stereo_data["timestamp"])
    left, right = np.array(stereo_data["stereo"][0]), np.array(stereo_data["stereo"][1])
    left, right = camera_model.process_stereo(left, right)
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
