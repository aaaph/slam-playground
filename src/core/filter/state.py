from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class InertialState:
    """Inertial state of the Multi-State Constraint Kalman Filter."""

    p: jax.Array
    q: jax.Array
    v: jax.Array
    b_a: jax.Array
    b_g: jax.Array

    def __repr__(self) -> str:
        """Return the representation of the inertial state."""
        return f"InertialState(p={self.p}, q={self.q}, v={self.v}, b_a={self.b_a}, b_g={self.b_g})"

    def map(self, f: Callable[[Self], Self]) -> "InertialState":
        """Map the inertial state."""
        result = f(self)
        return InertialState(result.p, result.q, result.v, result.b_a, result.b_g)

    def map_position(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the position."""
        return self.map(lambda x: InertialState(f(x[0]), x[1], x[2], x[3], x[4]))

    def map_orientation(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the orientation."""
        return self.map(lambda x: InertialState(x[0], f(x[1]), x[2], x[3], x[4]))

    def map_velocity(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the velocity."""
        return self.map(lambda x: InertialState(x[0], x[1], f(x[2]), x[3], x[4]))

    def map_acc_bias(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the acc bias."""
        return self.map(lambda x: InertialState(x[0], x[1], x[2], f(x[3]), x[4]))

    def map_gyro_bias(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the gyro bias."""
        return self.map(lambda x: InertialState(x[0], x[1], x[2], x[3], f(x[4])))

    def __getitem__(self, index: int) -> jax.Array:
        """Get the item at the index."""
        match index:
            case 0:
                return self.p
            case 1:
                return self.q
            case 2:
                return self.v
            case 3:
                return self.b_a
            case 4:
                return self.b_g
            case _:
                raise ValueError("Valid indices are 0-4")

    def get_pose(self) -> jax.Array:
        """Get the pose of the inertial state."""
        return jnp.concatenate([self.p, self.q], axis=0)

    @staticmethod
    def empty() -> "InertialState":
        """Create empty inertial state instance."""
        return InertialState(
            p=jnp.array([0, 0, 0]),
            q=jnp.array([1, 0, 0, 0]),
            v=jnp.array([0, 0, 0]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        )


@dataclass(frozen=True)
class CameraClone:
    """Camera clone of the Multi-State Constraint Kalman Filter."""

    clone_id: int
    timestamp: float
    p: np.ndarray
    q: np.ndarray
    p_fej: np.ndarray
    q_fej: np.ndarray


class SlidingWindow:
    """Sliding window of the Multi-State Constraint Kalman Filter."""

    def __init__(self, window_size: int = 20) -> None:
        """Initialize the sliding window."""
        self.window: OrderedDict[int, CameraClone] = OrderedDict()
        self.max_size = window_size
        self.ts_to_id: dict[float, int] = {}
        self.next_id = 0

    def add(self, timestamp: float, pose: np.ndarray) -> CameraClone:
        """Add a camera clone to the sliding window."""
        camera_id = self.next_id
        self.next_id += 1
        self.window[camera_id] = CameraClone(
            clone_id=camera_id,
            timestamp=timestamp,
            p=pose[0:3],
            q=pose[3:7],
            p_fej=pose[0:3],
            q_fej=pose[3:7],
        )
        self.ts_to_id[timestamp] = camera_id
        return self.window[camera_id]

    def get_candidate_for_removal(self) -> CameraClone | None:
        """Get the candidate for removal."""
        if self.size() <= self.max_size:
            return None
        _, candidate = self.window.popitem(last=False)
        self.ts_to_id.pop(candidate.timestamp, None)
        return candidate

    def get_by_id(self, camera_id: int) -> CameraClone | None:
        """Get a camera clone by its camera id."""
        return self.window.get(camera_id, None)

    def get_by_timestamp(self, timestamp: float) -> CameraClone | None:
        """Get the oldest camera clone by its timestamp."""
        camera_id = self.ts_to_id.get(timestamp, None)
        if camera_id is None:
            return None
        return self.window.get(camera_id, None)

    def get_oldest(self) -> CameraClone | None:
        """Get the oldest camera clone."""
        return next(iter(self.window.values()), None)

    def size(self) -> int:
        """Get the size of the sliding window."""
        return len(self.window)

    def get_oldest_than(self, oldest_ts: float) -> list[CameraClone]:
        """Get the camera clones older than the given timestamp."""
        return [clone for clone in self.window.values() if clone.timestamp < oldest_ts]

    def iterate(self) -> Iterator[CameraClone]:
        """Iterate over the sliding window."""
        return iter(self.window.values())

    def apply_clone(self, clone: CameraClone) -> Self:
        """Apply the clone to the sliding window. Method is used to update the sliding window."""
        self.window[clone.clone_id] = clone
        return self


@dataclass(frozen=True)
class Covariance:
    """Covariance of the Multi-State Constraint Kalman Filter."""

    sigma: np.ndarray

    def __init__(self, sigma: np.ndarray | None = None) -> None:
        """Initialize the covariance."""
        if sigma is not None:
            object.__setattr__(self, "sigma", sigma)
        else:
            sigma = np.eye(15)
            sigma[0:3, 0:3] = jnp.eye(3) * 1e-4
            sigma[3:6, 3:6] = jnp.deg2rad(0.01) ** 2 * jnp.eye(3)
            sigma[6:9, 6:9] = 1e-2 * jnp.eye(3)
            sigma[9:12, 9:12] = 1e-12 * jnp.eye(3)
            sigma[12:15, 12:15] = 1e-12 * jnp.eye(3)
            object.__setattr__(self, "sigma", sigma)

    def __repr__(self) -> str:
        """Return the representation of the covariance."""
        return f"Covariance(NxN={self.sigma.shape[0]}x{self.sigma.shape[1]})"

    def map(self, f: Callable[[np.ndarray], np.ndarray]) -> "Covariance":
        """Map the covariance."""
        return Covariance(f(self.sigma))

    def get_tuple_of_covariances(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get the tuple of covariances."""
        return (
            np.array(self.sigma[0:3, 0:3]),
            np.array(self.sigma[3:6, 3:6]),
            np.array(self.sigma[6:9, 6:9]),
            np.array(self.sigma[9:12, 9:12]),
            np.array(self.sigma[12:15, 12:15]),
        )


class State:
    """State of the Multi-State Constraint Kalman Filter."""

    def __init__(self) -> None:
        """Initialize the state."""
        self.ts = -1
        self.counter = 0
        self.inertial_state = InertialState.empty()
        self.covariance = Covariance()
        self.sliding_window: SlidingWindow = SlidingWindow()

    def initialize_inertial_state(
        self, p: jax.Array, q: jax.Array, v: jax.Array, b_a: jax.Array, b_g: jax.Array
    ) -> Self:
        """Initialize the inertial state."""
        self.inertial_state = InertialState(p, q, v, b_a, b_g)
        return self

    def map_inertial_state(self, f: Callable[[InertialState], InertialState]) -> Self:
        """Map the inertial state."""
        self.inertial_state = f(self.inertial_state)
        return self

    def initialize_covariance(self) -> Self:
        """Initialize the covariance."""
        self.covariance = Covariance()
        return self

    def apply_covariance(self, sigma: jax.Array) -> Self:
        """Apply the covariance."""
        self.covariance = self.covariance.map(lambda _x: sigma)
        return self

    def apply_timestamp(self, ts: float) -> Self:
        """Apply the timestamp."""
        self.ts = ts
        return self

    def map_poses_in_sliding_window(self, f: Callable[[CameraClone], tuple[np.ndarray, np.ndarray]]) -> Self:
        """Map the poses in the sliding window."""
        # Create a list of clones to avoid mutating during iteration
        clones = list(self.sliding_window.iterate())
        for clone in clones:
            new_p, new_q = f(clone)
            clone_id = clone.clone_id
            new_clone = CameraClone(
                clone_id=clone.clone_id,
                timestamp=clone.timestamp,
                p=new_p,
                q=new_q,
                p_fej=clone.p_fej,
                q_fej=clone.q_fej,
            )
            self.sliding_window.window.pop(clone_id, None)
            self.sliding_window.window[clone_id] = new_clone
        return self
