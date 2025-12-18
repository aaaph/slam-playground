import cv2
import numpy as np

from core.camera_model.stereo_camera_model import StereoCameraModel
from core.feature_tracker.feature_tracker import FeatureTracker
from dataset.euroc import EurocDataset


def _bilinear_at(map_img: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Bilinear sample of a single-channel map at floating coords."""
    h, w = map_img.shape[:2]
    u = xy[:, 0]
    v = xy[:, 1]

    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    u1 = u0 + 1
    v1 = v0 + 1

    # clamp to image bounds (чтобы не выйти за край)
    u0 = np.clip(u0, 0, w - 1)
    u1 = np.clip(u1, 0, w - 1)
    v0 = np.clip(v0, 0, h - 1)
    v1 = np.clip(v1, 0, h - 1)

    i_a = map_img[v0, u0]
    i_b = map_img[v0, u1]
    i_c = map_img[v1, u0]
    i_d = map_img[v1, u1]

    du = u - u0
    dv = v - v0

    # билинейная интерполяция
    top = i_a * (1 - du) + i_b * du
    bot = i_c * (1 - du) + i_d * du
    return top * (1 - dv) + bot * dv


def uv_rect_to_uv_raw_via_maps(uv_rect: np.ndarray, map1: np.ndarray, map2: np.ndarray) -> np.ndarray:
    """Move points from rectified frame to original frame using maps."""
    uv_rect = np.asarray(uv_rect, dtype=np.float32).reshape(-1, 2)
    u_src = _bilinear_at(map1, uv_rect).astype(np.float64)
    v_src = _bilinear_at(map2, uv_rect).astype(np.float64)
    return np.stack([u_src, v_src], axis=1)


def distort_points(uv: tuple[float, float], map1: np.ndarray, map2: np.ndarray) -> tuple[float, float]:
    """Distort points."""
    uv_rect = np.array([uv]).reshape(-1, 1, 2)

    return uv_rect_to_uv_raw_via_maps(uv_rect, map1, map2)


euroc_dataset = EurocDataset.mh_01_easy()
stereo = euroc_dataset.stereo().with_format("numpy")
stereo_iterator = stereo.to_iterable_dataset()

camera_model = StereoCameraModel.from_cameras_config(euroc_dataset.config.cam0, euroc_dataset.config.cam1)
ft = FeatureTracker.default_factory(camera_model.as_stereo_ctx())

first_item = next(iter(stereo_iterator))
left, right = np.array(first_item["stereo"][0]), np.array(first_item["stereo"][1])
left_rect, right_rect = camera_model.process_stereo(left, right)
left_old, right_old = ft.feed_first(float(first_item["timestamp"]), (left_rect, right_rect))

left_out = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
right_out = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
concatenated = np.concatenate([left_out, right_out], axis=1)
for feature in ft.iterate_through_features():
    _, left_uv, right_uv = feature.get_active_stereo_pair()
    lx, ly = left_uv
    rx, ry = right_uv
    dlx, dly = distort_points((lx, ly), camera_model.map1_x, camera_model.map1_y).ravel()
    drx, dry = distort_points((rx, ry), camera_model.map2_x, camera_model.map2_y).ravel()
    cv2.circle(concatenated, (int(dlx), int(dly)), 2, feature.feature_color(), -1)
    cv2.circle(concatenated, (int(drx) + left_out.shape[1], int(dry)), 2, feature.feature_color(), -1)

cv2.imshow("concatenated", concatenated)
cv2.waitKey(0)
