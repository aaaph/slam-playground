import time

import jax
import jax.numpy as jnp

from core.filter.state import State


class Initializer:
    """Initializer of the Multi-State Constraint Kalman Filter."""

    def __init__(self) -> None:
        """Initialize the initializer."""

    def initialize(self, state: State) -> State:
        """Initialize the state."""
        return state

    def zero_initialize(self, state: State) -> State:
        """Initialize the state to zero."""
        return state.initialize_inertial_state(
            payload=(
                time.time(),
                jnp.array([0, 0, 0]),
                jnp.array([1, 0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, -9.81]),
            )
        ).initialize_covariance()

    def initialize_from_row(
        self,
        state: State,
        row: tuple[float, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> State:
        """Initialize the state from a row."""
        timestamp = row[0]
        position = row[1]
        orientation = row[2]
        velocity = row[3]
        gyro_bias = row[4]
        acc_bias = row[5]
        return (
            state.initialize_inertial_state(
                payload=(timestamp, position, orientation, velocity, gyro_bias, acc_bias, jnp.array([0, 0, -9.81]))
            )
        ).initialize_covariance()
