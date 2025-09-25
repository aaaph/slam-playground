from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp

InertialStateVector = tuple[float, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


@dataclass(frozen=True)
class InertialState:
    """Inertial state of the Multi-State Constraint Kalman Filter."""

    ts: float
    p: jax.Array
    q: jax.Array
    v: jax.Array
    b_a: jax.Array
    b_g: jax.Array

    def __init__(
        self,
        payload: InertialStateVector,
    ) -> None:
        """Initialize the inertial state."""
        object.__setattr__(self, "ts", payload[0])
        object.__setattr__(self, "p", jnp.array(payload[1]))
        object.__setattr__(self, "q", jnp.array(payload[2]))
        object.__setattr__(self, "v", jnp.array(payload[3]))
        object.__setattr__(self, "b_a", jnp.array(payload[4]))
        object.__setattr__(self, "b_g", jnp.array(payload[5]))

    def __repr__(self) -> str:
        """Return the representation of the inertial state."""
        return f"InertialState(ts={self.ts}, p={self.p}, q={self.q}, v={self.v}, b_a={self.b_a}, b_g={self.b_g})"

    def map(self, f: Callable[[InertialStateVector], InertialStateVector]) -> "InertialState":
        """Map the inertial state."""
        return InertialState(f((self.ts, self.p, self.q, self.v, self.b_a, self.b_g)))

    def map_position(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the position."""
        return self.map(lambda x: (x[0], f(x[1]), x[2], x[3], x[4], x[5]))

    def map_orientation(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the orientation."""
        return self.map(lambda x: (x[0], x[1], f(x[2]), x[3], x[4], x[5]))

    def map_velocity(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the velocity."""
        return self.map(lambda x: (x[0], x[1], x[2], f(x[3]), x[4], x[5]))

    def map_acc_bias(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the acc bias."""
        return self.map(lambda x: (x[0], x[1], x[2], x[3], f(x[4]), x[5]))

    def map_gyro_bias(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the gyro bias."""
        return self.map(lambda x: (x[0], x[1], x[2], x[3], x[4], f(x[5])))

    def apply_timestamp(self, ts: float) -> "InertialState":
        """Apply the timestamp."""
        return self.map(lambda x: (ts, x[1], x[2], x[3], x[4], x[5]))


@dataclass(frozen=True)
class Covariance:
    """Covariance of the Multi-State Constraint Kalman Filter."""

    cov: jax.Array

    def __init__(self) -> None:
        """Initialize the covariance."""
        object.__setattr__(self, "cov", jnp.eye(18))

    def __repr__(self) -> str:
        """Return the representation of the covariance."""
        return f"Covariance(NxN={self.cov.shape[0]}x{self.cov.shape[1]})"


class State:
    """State of the Multi-State Constraint Kalman Filter."""

    def __init__(self) -> None:
        """Initialize the state."""
        self.ts = -1
        self.counter = 0
        self.inertial_state = None
        self.covariance = None

    def initialize_inertial_state(self, payload: InertialStateVector) -> Self:
        """Initialize the inertial state."""
        self.inertial_state = InertialState(payload)
        self.ts = payload[0]
        return self

    def map_inertial_state(self, f: Callable[[InertialState], InertialState]) -> Self:
        """Map the inertial state."""
        self.inertial_state = f(self.inertial_state)
        return self

    def initialize_covariance(self) -> Self:
        """Initialize the covariance."""
        self.covariance = Covariance()
        return self
