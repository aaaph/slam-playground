from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from core.filter.state import CameraClone
from core.transformations.helpers import skew
from dataset.euroc import EurocDataset

euroc_dataset = EurocDataset.mh_01_easy()
camera_resolution = euroc_dataset.config.stereo.cam0.resolution
left_k_matrix = euroc_dataset.config.stereo.k_rect_left
right_k_matrix = euroc_dataset.config.stereo.k_rect_right
t_cam0_imu = euroc_dataset.config.stereo.cam0.body_sensor_transform
t_cam1_imu = euroc_dataset.config.stereo.cam1.body_sensor_transform

fx = left_k_matrix[0, 0]
baseline = -euroc_dataset.config.stereo.p2[0, 3] / fx
cx = left_k_matrix[0, 2]
cy = left_k_matrix[1, 2]
fy = left_k_matrix[1, 1]


def measurement_jacobian(
    uv: tuple[float, float], cam_id: Literal[0, 1], p_f_in_w: np.ndarray, clone: CameraClone
) -> np.ndarray:
    """Measurement Jacobian for the Multi-State Constraint Kalman Filter."""
    p_i_in_w = clone.p
    q_w_to_i = clone.q
    t_cam_imu = t_cam0_imu if cam_id == 0 else t_cam1_imu

    t_w_to_i = np.eye(4)
    t_w_to_i[:3, :3] = Rotation.from_quat(q_w_to_i).as_matrix()
    t_w_to_i[:3, 3] = p_i_in_w
    t_w_to_cam = t_w_to_i @ t_cam_imu
    p_cam_in_w = t_w_to_cam[:3, 3]
    rot_w_to_cam = t_w_to_cam[:3, :3]
    rot_cam_to_w = rot_w_to_cam.T

    p_f_in_cam = rot_w_to_cam.T @ (p_f_in_w - p_cam_in_w)
    x, y, z = p_f_in_cam
    dzn_dp = np.eye(2, 3, dtype=np.float32)
    dzn_dp[0, 0] = 1 / z
    dzn_dp[1, 1] = 1 / z
    dzn_dp[0, 2] = -x / (z * z)
    dzn_dp[1, 2] = -y / (z * z)

    dz_dzn = np.eye(2, 2, dtype=np.float32)
    dz_dzn[0, 0] = fx
    dz_dzn[1, 1] = fy
    dz_dp = dz_dzn @ dzn_dp  # 2x3

    dp_f_in_c_dxc = np.eye(3, 6, dtype=np.float32)
    dp_f_in_c_dxc[:, :3] = skew(p_f_in_cam)
    dp_f_in_c_dxc[:, 3:] = -rot_cam_to_w

    h_f = dz_dp @ rot_cam_to_w  # (2,3)
    h_x = dz_dp @ dp_f_in_c_dxc  # (2,6)

    a = h_x.copy()  # 2x6
    u = np.zeros(6, dtype=np.float32)  # 6x1
    g = np.array([0, 0, -9.81], dtype=np.float32)
    u[:3] = Rotation.from_quat(q_w_to_i).as_matrix() @ g
    u[3:] = skew(p_f_in_w - p_i_in_w) @ g

    h_x = a - (a @ u)[:, None] * u / (u @ u)

    resudial = uv - np.array([fx * (X / Z) + cx, fy * (Y / Z) + cy])
    return resudial, h_x, h_f


def parse_array(s: str) -> np.ndarray:
    """Parse a string into a numpy array."""
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    return np.fromstring(s, sep=" ")


feat_df = pd.read_csv("corrected_feat_data.csv")
feat_df = feat_df.head(50)

feat_df["q_wi"] = feat_df["q_wi"].apply(parse_array)
feat_df["p_iw"] = feat_df["p_iw"].apply(parse_array)
feat_df["timestamp"] = feat_df["timestamp"].apply(lambda x: float(x))
entries = feat_df[["cam_id", "timestamp", "u", "v", "p_iw", "q_wi"]].to_numpy()

fig = plt.figure()
ax = fig.add_subplot(111)

A = np.zeros((3, 3))
b = np.zeros(3)
for cam_id, ts, u, v, p_iw, q_wi in entries:
    ground_truth = euroc_dataset.find_nearest_ground_truth_by_timestamp(ts)
    p_i_in_w = np.array(p_iw)  # np.array(ground_truth["gt_position"])
    q_w_to_i = np.array(q_wi)  # np.array(ground_truth["gt_orientation"])
    k_matrix = left_k_matrix if cam_id == 0 else right_k_matrix
    t_cam_imu = t_cam0_imu if cam_id == 0 else t_cam1_imu

    t_wi = np.eye(4)
    t_wi[:3, :3] = Rotation.from_quat(q_w_to_i).as_matrix()
    t_wi[:3, 3] = p_i_in_w
    t_sw = t_wi @ t_cam_imu
    p_sw = t_sw[:3, 3]
    rot_ws = t_sw[:3, :3]

    pixel_homog = np.c_[u, v, np.ones_like(u)]
    uv_norm = (np.linalg.inv(left_k_matrix) @ pixel_homog.T).T
    uv_norm = uv_norm[:, :2] / uv_norm[:, 2:3]
    uv_norm = uv_norm.ravel()
    b_i = np.array([uv_norm[0], uv_norm[1], 1])
    b_i = rot_ws @ b_i
    b_i = b_i / np.linalg.norm(b_i)
    b_perp = skew(b_i)
    a_i = b_perp.T @ b_perp
    A += a_i
    b += a_i @ p_sw
    end_point = p_sw[:2] + 3.0 * b_i[:2]
    ax.plot([p_sw[0], end_point[0]], [p_sw[1], end_point[1]], "r-", linewidth=1, alpha=0.7)
    ax.plot(p_sw[0], p_sw[1], "bo")
    ax.plot(p_i_in_w[0], p_i_in_w[1], "go")


p_f_in_w = np.linalg.solve(A, b)
ax.plot(p_f_in_w[0], p_f_in_w[1], "go")

fig2 = plt.figure()
ax2 = fig2.add_subplot()

jacobian_row_size = 2 * 50
r_j = np.zeros(jacobian_row_size, dtype=np.float32)
h_fj = np.zeros((jacobian_row_size, 3), dtype=np.float32)
h_xj = np.zeros((jacobian_row_size, 15 + (25) * 6), dtype=np.float32)


for idx, (cam_id, timestamp, u, v, p_iw, q_wi) in enumerate(entries):
    ground_truth = euroc_dataset.find_nearest_ground_truth_by_timestamp(timestamp)
    p_i_in_w = np.array(p_iw)  # np.array(ground_truth["gt_position"])
    q_w_to_i = np.array(q_wi)  # np.array(ground_truth["gt_orientation"])
    k_matrix = left_k_matrix if cam_id == 0 else right_k_matrix
    t_cam_imu = t_cam0_imu if cam_id == 0 else t_cam1_imu

    t_w_to_i = np.eye(4)
    t_w_to_i[:3, :3] = Rotation.from_quat(q_w_to_i).as_matrix()
    t_w_to_i[:3, 3] = p_i_in_w
    t_w_to_cam = t_w_to_i @ t_cam_imu
    p_cam_in_w = t_w_to_cam[:3, 3]
    rot_w_to_cam = t_w_to_cam[:3, :3]

    p_f_in_cam = rot_w_to_cam.T @ (p_f_in_w - p_cam_in_w)
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    X, Y, Z = p_f_in_cam
    if Z <= 0:
        raise ValueError("Z is negative")
    u_returned = fx * (X / Z) + cx
    v_returned = fy * (Y / Z) + cy
    ax2.plot(u, v, "bo", markersize=3)
    ax2.plot(u_returned, v_returned, "ro", markersize=3)
    # plot line between uv and returned uv
    ax2.plot([u, u_returned], [v, v_returned], "g-", linewidth=2, alpha=0.7)
    r_i, h_xi, h_fi = measurement_jacobian(
        (u, v), cam_id, p_f_in_w, CameraClone(0, 1, p_i_in_w, q_w_to_i, p_i_in_w, q_w_to_i)
    )
    # should extend r_j by r_i and idx
    # idx = 0, r_j =
    r_j[idx * 2 : (idx + 1) * 2] = r_i
    h_fj[idx * 2 : (idx + 1) * 2, :3] = h_fi
    col = idx // 2
    h_xj[idx * 2 : (idx + 1) * 2, 15 + 6 * col : 15 + 6 * (col + 1)] = h_xi

Q, R = np.linalg.qr(h_xj)
A = Q[:, 3:]
h_x0 = A.T @ h_xj
r0 = A.T @ r_j

plt.legend(["original", "returned"])
# 123 ax2.set_xlim(0, camera_resolution[0])
# 13ax2.set_ylim(0, camera_resolution[1])
ax2.invert_yaxis()
plt.show()
