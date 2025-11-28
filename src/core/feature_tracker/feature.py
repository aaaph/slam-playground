from collections.abc import Iterator
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from core.filter.state import CameraClone
from core.transformations.helpers import skew


class Feature:
    """Represents a tracked feature with associated points and linear system matrices."""

    def __init__(self, feat_id: int, capacity: int = 60, spawned_timestamp: float = -1.0) -> None:
        """Initialize a feature with the given ID."""
        self.feat_id = feat_id
        self.capacity = capacity
        self.size = 0
        self.head = 0
        self.iteration_life = 0

        self.A = np.zeros((3, 3), dtype=np.float32)
        self.b = np.zeros(3, dtype=np.float32)

        self.p_fw = None
        self.valid = False
        self.state: Literal["new", "tracked", "lost", "stable"] = "new"

        self.ts = np.full(self.capacity, 0, np.int64)
        self.cam_id = np.full(self.capacity, -1, np.int32)
        self.u = np.full(self.capacity, np.nan, np.float32)
        self.v = np.full(self.capacity, np.nan, np.float32)

        self.active_timestamp = -1.0
        self.left_pair_idx: None | int = None
        self.right_pair_idx: None | int = None

        self.spawned_timestamp = spawned_timestamp
        self.max_cond_a = 10000.0
        self.max_depth = 60.0
        self.min_depth = 0.15

    def _add(self, ts: float, cam_id: Literal[0, 1], uv: tuple[float, float]) -> int:
        """Add a new observation to the feature."""
        index = self.head
        u, v = uv
        self.ts[index] = ts
        self.cam_id[index] = cam_id
        self.u[index] = u
        self.v[index] = v
        self.head = (self.head + 1) % self.capacity
        self.size = np.minimum(self.size + 1, self.capacity)
        self.active_timestamp = max(self.active_timestamp, ts)
        if self.size > 2:  # noqa: PLR2004
            self.state = "tracked"

        return index

    def apply_stereo_pair(self, ts: float, left_uv: tuple[float, float], right_uv: tuple[float, float]) -> None:
        """Apply a stereo pair to the feature."""
        left_idx = self._add(ts, 0, left_uv)
        right_idx = self._add(ts, 1, right_uv)
        self.active_timestamp = max(self.active_timestamp, ts)
        self.left_pair_idx = left_idx
        self.right_pair_idx = right_idx
        self.iteration_life += 1

    def apply_left_only(self, ts: float, left_uv: tuple[float, float]) -> None:
        """Apply a left point to the feature."""
        left_idx = self._add(ts, 0, left_uv)
        self.active_timestamp = max(self.active_timestamp, ts)
        self.left_pair_idx = left_idx
        self.right_pair_idx = None
        self.iteration_life += 1

    def get_uv_by_timestamp(self, ts: float) -> list[tuple[Literal[0, 1], float, float]]:
        """Get the uv by timestamp. Method could return 1 point or 2 per 1 timestamp."""
        result: list[tuple[Literal[0, 1], float, float]] = []
        mask = self.ts == ts
        for cam_id, u, v in zip(self.cam_id[mask], self.u[mask], self.v[mask], strict=True):
            result.append((cam_id, u, v))
        return result

    def get_active_stereo_pair(self) -> tuple[float, tuple[float, float], tuple[float, float] | None]:
        """Get the active stereo pair of the feature."""
        return self.get_active_stereo_pair_idx()

    def get_active_stereo_pair_idx(self) -> tuple[float, tuple[float, float], tuple[float, float] | None]:
        """Get the active stereo pair indexed of the feature."""
        if self.size < 1:
            raise ValueError("Feature has no active stereo pair")
        if self.left_pair_idx is None and self.right_pair_idx is None:
            msg = f"Feature has no active stereo pair, feat_id: {self.feat_id}"
            raise ValueError(msg)
        if self.right_pair_idx is not None and self.left_pair_idx is None:
            msg = f"Feature has no active left point, feat_id: {self.feat_id}"
            raise ValueError(msg)
        if self.left_pair_idx is not None and self.right_pair_idx is not None:
            left_u = self.u[self.left_pair_idx]
            left_v = self.v[self.left_pair_idx]
            right_u = self.u[self.right_pair_idx]
            right_v = self.v[self.right_pair_idx]
            timestamp = self.ts[self.left_pair_idx]
            return timestamp, (left_u, left_v), (right_u, right_v)
        if self.left_pair_idx is not None and self.right_pair_idx is None:
            left_u = self.u[self.left_pair_idx]
            left_v = self.v[self.left_pair_idx]
            timestamp = self.ts[self.left_pair_idx]
            return timestamp, (left_u, left_v), None

        left_u = self.u[self.left_pair_idx]
        left_v = self.v[self.left_pair_idx]
        right_u = self.u[self.right_pair_idx]
        right_v = self.v[self.right_pair_idx]
        timestamp = self.ts[self.left_pair_idx]
        return timestamp, (left_u, left_v), (right_u, right_v)

    def __repr__(self) -> str:
        """Return the representation of the feature."""
        rows = []
        state = self.state
        for i in range(self.size):
            rows.append(f"ts: {self.ts[i]}, cam_id: {self.cam_id[i]}, u: {self.u[i]}, v: {self.v[i]}")  # noqa: PERF401
        return f"Feature(feat_id={self.feat_id}, state: {state}, log:\n{'\n'.join(rows)})"

    def obs_count(self) -> int:
        """Get the number of observations for the feature."""
        return self.size

    def feature_color(self) -> tuple[int, int, int]:
        """Get the color of the feature."""
        match self.state:
            case "new":
                return (0, 255, 0)
            case "tracked":
                return (0, 0, 255)
            case "lost":
                return (255, 0, 0)
            case "stable":
                return (0, 255, 255)
            case _:
                return (255, 0, 0)

    def get_last_left(self) -> tuple[int, float, float, float]:
        """Get the last left observation for the feature by timestamp."""
        if self.size == 0:
            raise ValueError("Feature has no observations")
        index = self.left_pair_idx
        if index is None:
            raise ValueError("Feature has no left pair index")
        left_u, left_v = self.u[index], self.v[index]
        timestamp = self.ts[index]
        feat_id = self.feat_id
        return feat_id, timestamp, left_u, left_v

    def get_tail(self, cam_id: Literal[0, 1]) -> list[tuple[float, float]]:
        """Get the tail of the feature."""
        if self.size < 1:
            raise ValueError("Feature has no observations")
        camera_mask = self.cam_id == cam_id
        timestamp_mask = self.ts != self.active_timestamp
        mask = camera_mask & timestamp_mask
        return list(zip(self.u[mask], self.v[mask], strict=False))

    def iterate(self) -> Iterator[tuple[float, Literal[0, 1], float, float]]:
        """Iterate over the feature."""
        index = 0
        while index < self.size:
            yield self.ts[index], self.cam_id[index], self.u[index], self.v[index]
            index += 1

    def update_linear_system(
        self,
        clone: CameraClone,
        k_matrices_inv: tuple[np.ndarray, np.ndarray],
        t_bs_matrices: tuple[np.ndarray, np.ndarray],
    ) -> list[tuple[float, np.ndarray, np.ndarray, np.ndarray, Literal[0, 1], float, float]]:
        """Update the linear system of the feature."""
        k_left_inv, k_right_inv = k_matrices_inv
        t_bs_left, t_bs_right = t_bs_matrices
        timestamp = clone.timestamp
        p_iw = clone.p
        rot_wi = Rotation.from_quat(clone.q).as_matrix()

        t_wi = np.eye(4)
        t_wi[:3, :3] = rot_wi
        t_wi[:3, 3] = p_iw

        uv_list = self.get_uv_by_timestamp(timestamp)
        world_vectors: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, Literal[0, 1], float, float]] = []
        for cam_id, u, v in uv_list:
            k_matrix_inv = k_left_inv if cam_id == 0 else k_right_inv
            t_is = t_bs_left if cam_id == 0 else t_bs_right
            t_ws = t_wi @ t_is
            rot_ws = t_ws[:3, :3]
            p_sw = t_ws[:3, 3]
            pixel_homog = np.array([u, v, 1])
            uv_norm = k_matrix_inv @ pixel_homog
            b_i = np.array([uv_norm[0], uv_norm[1], 1])
            b_i = rot_ws @ b_i
            b_i = b_i / np.linalg.norm(b_i)
            b_perp = skew(b_i)
            a_i = b_perp.T @ b_perp
            self.A += a_i
            self.b += a_i @ p_sw
            value = (clone.timestamp, clone.p, clone.q, b_i, cam_id, u, v)
            world_vectors.append(value)

            if self.iteration_life > 3:  # noqa: PLR2004
                p_fw = np.linalg.solve(self.A, self.b)
                p_fs = rot_ws.transpose() @ (p_fw + p_sw)
                depth = p_fs[2]
                if depth < self.min_depth or depth > self.max_depth:
                    self.state = "tracked"
                    continue
                if np.isnan(np.linalg.norm(p_fw)):
                    self.state = "tracked"
                    continue
                _u, s, _vh = np.linalg.svd(self.A)
                cond_a = s[0] / s[-1]
                if cond_a > self.max_cond_a:
                    self.state = "tracked"
                    continue
                self.state = "stable"
                self.p_fw = p_fw
        return world_vectors

    def make_initial_guess(
        self, k_matrix: NDArray[np.float64], baseline: float
    ) -> NDArray[np.float64]:  # shape: (3,)
        """
        Make an initial guess for the feature using disparity.

        Should return a 3D position of the feature in camera frame.
        """
        stereo_pair_size = 2
        if self.size != stereo_pair_size:
            raise ValueError("Feature has no active stereo pair")
        left_u, left_v = self.u[self.left_pair_idx], self.v[self.left_pair_idx]
        right_u = self.u[self.right_pair_idx]

        disp = left_u - right_u
        if disp <= 0:
            raise ValueError("Disparity is non-positive")
        fx = k_matrix[0, 0]
        fy = k_matrix[1, 1]
        cx = k_matrix[0, 2]
        cy = k_matrix[1, 2]
        z = fx * baseline / disp
        x = (left_u - cx) * z / fx
        y = (left_v - cy) * z / fy
        return np.array([x, y, z])

    @staticmethod
    def spawn_from_left_and_right(
        feat_id: int, ts: float, left_uv: tuple[float, float], right_uv: tuple[float, float]
    ) -> "Feature":
        """Spawn a feature from a left and right observation."""
        feature = Feature(feat_id, spawned_timestamp=ts)
        feature.apply_stereo_pair(ts, left_uv, right_uv)
        return feature
