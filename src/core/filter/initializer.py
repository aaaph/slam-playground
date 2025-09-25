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
        return (
            state.initialize_ts(ts=time.time())
            .initialize_inertial_state(
                p=jnp.array([0, 0, 0]),
                q=jnp.array([0, 0, 0, 1]),
                v=jnp.array([0, 0, 0]),
                b_a=jnp.array([0, 0, 0]),
                b_g=jnp.array([0, 0, 0]),
            )
            .initialize_covariance()
        )

    def initialize_from_row(
        self,
        state: State,
        timestamp: float,
        array: jax.Array,
    ) -> State:
        """Initialize the state from a row."""
        position = array[:3]
        orientation = array[3:7]
        velocity = array[7:10]
        gyro_bias = array[10:13]
        acc_bias = array[13:16]
        return (
            (
                state.initialize_inertial_state(
                    p=position,
                    q=orientation,
                    v=velocity,
                    b_a=acc_bias,
                    b_g=gyro_bias,
                )
            )
            .initialize_covariance()
            .initialize_ts(ts=timestamp)
        )
