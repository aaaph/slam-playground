import cv2
import numpy as np

from core.camera_model.stereo_camera_ctx import StereoContext
from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature_tracker import FeatureTracker
from core.front_end.front_end import FrontEnd
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
    cam0_in_body_se3: SE3 = stereo_ctx.cam0_in_body
    return initial_body_in_world_se3 * cam0_in_body_se3


euroc_dataset = EurocDataset.mh_01_easy()
stereo_iterator = euroc_dataset.stereo().to_iterable_dataset()
camera_model = StereoCameraModel.from_cameras_config(euroc_dataset.config.cam0, euroc_dataset.config.cam1)
stereo_ctx = camera_model.as_stereo_ctx()
initial_cam0_in_world_se3 = get_init_cam0_in_world_se3(euroc_dataset, stereo_ctx)

slam_fe = FrontEnd.default_factory(camera_model, initial_cam0_in_world_se3)

viz = FoxgloveVisualizer.pose_and_point_cloud_viz()

for stereo_data in stereo_iterator:
    ts = float(stereo_data["timestamp"])
    left, right = np.array(stereo_data["stereo"][0]), np.array(stereo_data["stereo"][1])

    keyframe = slam_fe.feed(ts, (left, right))
    input()

viz.close()
