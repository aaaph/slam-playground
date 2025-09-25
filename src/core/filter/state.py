from typing import Self

import jax
import jax.numpy as jnp


class InertialState:
    """Inertial state of the Multi-State Constraint Kalman Filter."""

    def __init__(self, p: jax.Array, q: jax.Array, v: jax.Array, b_a: jax.Array, b_g: jax.Array) -> None:
        """Initialize the inertial state."""
        self.p = p
        self.q = q
        self.v = v
        self.b_a = b_a
        self.b_g = b_g

    def __repr__(self) -> str:
        """Return the representation of the inertial state."""
        return f"InertialState(p={self.p}, q={self.q}, v={self.v}, b_a={self.b_a}, b_g={self.b_g})"


class Covariance:
    """Covariance of the Multi-State Constraint Kalman Filter."""

    def __init__(self) -> None:
        """Initialize the covariance."""
        self.cov = jnp.eye(18)

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

    def initialize_inertial_state(
        self, p: jax.Array, q: jax.Array, v: jax.Array, b_a: jax.Array, b_g: jax.Array
    ) -> Self:
        """Initialize the inertial state."""
        self.inertial_state = InertialState(p, q, v, b_a, b_g)
        return self

    def initialize_ts(self, ts: float) -> Self:
        """Initialize the timestamp."""
        self.ts = ts
        return self

    def initialize_covariance(self) -> Self:
        """Initialize the covariance."""
        self.covariance = Covariance()
        return self
