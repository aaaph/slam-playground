from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp

InertialStateVector = tuple[float, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


@dataclass(frozen=True)
class InertialState:
    """Inertial state of the Multi-State Constraint Kalman Filter."""

    ts: float
    p: jax.Array
    q: jax.Array
    v: jax.Array
    b_a: jax.Array
    b_g: jax.Array
    g: jax.Array

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
        object.__setattr__(self, "g", jnp.array(payload[6]))

    def __repr__(self) -> str:
        """Return the representation of the inertial state."""
        return f"InertialState(ts={self.ts}, p={self.p}, q={self.q}, v={self.v}, b_a={self.b_a}, b_g={self.b_g})"

    def map(self, f: Callable[[InertialStateVector], InertialStateVector]) -> "InertialState":
        """Map the inertial state."""
        return InertialState(f((self.ts, self.p, self.q, self.v, self.b_a, self.b_g, self.g)))

    def map_position(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the position."""
        return self.map(lambda x: (x[0], f(x[1]), x[2], x[3], x[4], x[5], x[6]))

    def map_orientation(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the orientation."""
        return self.map(lambda x: (x[0], x[1], f(x[2]), x[3], x[4], x[5], x[6]))

    def map_velocity(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the velocity."""
        return self.map(lambda x: (x[0], x[1], x[2], f(x[3]), x[4], x[5], x[6]))

    def map_acc_bias(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the acc bias."""
        return self.map(lambda x: (x[0], x[1], x[2], x[3], f(x[4]), x[5], x[6]))

    def map_gyro_bias(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the gyro bias."""
        return self.map(lambda x: (x[0], x[1], x[2], x[3], x[4], f(x[5]), x[6]))

    def map_gravity(self, f: Callable[[jax.Array], jax.Array]) -> "InertialState":
        """Map the gravity."""
        return self.map(lambda x: (x[0], x[1], x[2], x[3], x[4], x[5], f(x[6])))

    def apply_timestamp(self, ts: float) -> "InertialState":
        """Apply the timestamp."""
        return self.map(lambda x: (ts, x[1], x[2], x[3], x[4], x[5], x[6]))

    @staticmethod
    def empty() -> "InertialState":
        """Create empty inertial state instance."""
        return InertialState(
            (
                0.0,
                jnp.array([0, 0, 0]),
                jnp.array([1, 0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, -9.81]),
            )
        )


@dataclass(frozen=True)
class Covariance:
    """Covariance of the Multi-State Constraint Kalman Filter."""

    sigma: jax.Array

    def __init__(self, sigma: jax.Array | None = None) -> None:
        """Initialize the covariance."""
        if sigma is not None:
            object.__setattr__(self, "sigma", sigma)
        else:
            sigma = jnp.eye(18)
            sigma = sigma.at[0:3, 0:3].set(1e-4 * jnp.eye(3))
            sigma = sigma.at[3:6, 3:6].set(jnp.deg2rad(0.01) ** 2 * jnp.eye(3))
            sigma = sigma.at[6:9, 6:9].set(1e-4 * jnp.eye(3))
            sigma = sigma.at[9:12, 9:12].set(1e-12 * jnp.eye(3))
            sigma = sigma.at[12:15, 12:15].set(1e-12 * jnp.eye(3))
            sigma = sigma.at[15:18, 15:18].set(1e-12 * jnp.eye(3))
            object.__setattr__(self, "sigma", sigma)

    def __repr__(self) -> str:
        """Return the representation of the covariance."""
        return f"Covariance(NxN={self.sigma.shape[0]}x{self.sigma.shape[1]})"

    def map(self, f: Callable[[jax.Array], jax.Array]) -> "Covariance":
        """Map the covariance."""
        return Covariance(f(self.sigma))


class State:
    """State of the Multi-State Constraint Kalman Filter."""

    def __init__(self) -> None:
        """Initialize the state."""
        self.ts = -1
        self.counter = 0
        self.inertial_state = InertialState.empty()
        self.covariance = Covariance()

    def initialize_inertial_state(self, payload: InertialStateVector) -> Self:
        """Initialize the inertial state."""
        self.inertial_state = InertialState(payload)
        self.ts = payload[0]
        return self

    def map_inertial_state(self, f: Callable[[InertialState], InertialState]) -> Self:
        """Map the inertial state."""
        self.inertial_state = f(self.inertial_state)
        self.ts = self.inertial_state.ts
        return self

    def initialize_covariance(self) -> Self:
        """Initialize the covariance."""
        self.covariance = Covariance()
        return self

    def apply_covariance(self, sigma: jax.Array) -> Self:
        """Apply the covariance."""
        self.covariance = self.covariance.map(lambda _x: sigma)
        return self
