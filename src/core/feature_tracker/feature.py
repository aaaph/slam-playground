from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np


class Feature:
    """Represents a tracked feature with associated points and linear system matrices."""

    def __init__(self, feat_id: int, capacity: int = 10) -> None:
        """Initialize a feature with the given ID."""
        self.feat_id = feat_id
        self.capacity = capacity
        self.size = 0
        self.head = 0

        self.A = jnp.zeros((3, 3))
        self.B = jnp.zeros(3)

        self.p_F_in_G = None
        self.valid = False
        self.state: Literal["new", "tracked", "lost"] = "new"

        self.ts = np.full(self.capacity, np.nan, np.float32)
        self.cam_id = np.full(self.capacity, -1, np.int32)
        self.u = np.full(self.capacity, np.nan, np.float32)
        self.v = np.full(self.capacity, np.nan, np.float32)

        self.active_timestamp = -1.0
        self.left_pair_idx: None | int = None
        self.right_pair_idx: None | int = None

    def _add(self, ts: float, cam_id: Literal[0, 1], uv: tuple[float, float]) -> int:
        """Add a new observation to the feature."""
        index = self.head
        u, v = uv
        self.ts[index] = ts
        self.cam_id[index] = cam_id
        self.u[index] = u
        self.v[index] = v
        self.head = (self.head + 1) % self.capacity
        self.size = jnp.minimum(self.size + 1, self.capacity)
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

    def apply_left_only(self, ts: float, left_uv: tuple[float, float]) -> None:
        """Apply a left point to the feature."""
        left_idx = self._add(ts, 0, left_uv)
        self.active_timestamp = max(self.active_timestamp, ts)
        self.left_pair_idx = left_idx
        self.right_pair_idx = None

    def select(self, ts: float, cam_id: Literal[0, 1]) -> jax.Array:
        """Select a feature by timestamp and camera id."""
        mask = (self.ts == ts) & (self.cam_id == cam_id)
        set_u, set_v = self.u[mask], self.v[mask]
        # we have [0,0] and [1,1] -> need return [0,1] and [0,1]
        result = []
        for u, v in zip(set_u, set_v, strict=True):
            result.append((u, v))
        return jnp.array(result)

    def get_active_stereo_pair(self) -> tuple[float, tuple[float, float], tuple[float, float] | None]:
        """Get the active stereo pair of the feature."""
        return self.get_active_stereo_pair_idx()
        if self.size < 1:
            raise ValueError("Feature has no active stereo pair")
        active_timestamp = self.active_timestamp
        mask = self.ts == active_timestamp
        left_mask = mask & (self.cam_id == 0)
        right_mask = mask & (self.cam_id == 1)
        left_u = self.u[left_mask]
        left_v = self.v[left_mask]
        right_u = self.u[right_mask]
        right_v = self.v[right_mask]
        if len(left_u) == 0:
            msg = f"Feature has no active left point, feat_id: {self.feat_id}"
            raise ValueError(msg)
        if len(right_u) == 0:
            return (left_u[0], left_v[0]), None
        return (left_u[0], left_v[0]), (right_u[0], right_v[0])

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
        for i in range(self.size):
            rows.append(f"ts: {self.ts[i]}, cam_id: {self.cam_id[i]}, u: {self.u[i]}, v: {self.v[i]}")  # noqa: PERF401
        return f"Feature(feat_id={self.feat_id}, log:\n{'\n'.join(rows)})"

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

    @staticmethod
    def spawn_from_left_and_right(
        feat_id: int, ts: float, left_uv: tuple[float, float], right_uv: tuple[float, float]
    ) -> "Feature":
        """Spawn a feature from a left and right observation."""
        feature = Feature(feat_id)
        feature.apply_stereo_pair(ts, left_uv, right_uv)
        return feature
