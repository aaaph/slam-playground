import time
from typing import Literal

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
            state.apply_timestamp(time.time())
            .initialize_inertial_state(
                p=jnp.array([0, 0, 0]),
                q=jnp.array([0, 0, 0, 1]),
                v=jnp.array([0, 0, 0]),
                b_a=jnp.array([0, 0, 0]),
                b_g=jnp.array([0, 0, 0]),
            )
            .initialize_covariance()
        )

    def initialize_from_dict(
        self,
        state: State,
        timestamp: float,
        dictionary: dict[Literal["position", "orientation", "velocity", "acc_bias", "gyro_bias"], jax.Array],
    ) -> State:
        """Initialize the state from a row."""
        position = jnp.array(dictionary["position"])
        orientation = jnp.array(dictionary["orientation"])
        velocity = jnp.array(dictionary["velocity"])
        acc_bias = jnp.array(dictionary["acc_bias"])
        gyro_bias = jnp.array(dictionary["gyro_bias"])
        return (
            state.apply_timestamp(timestamp)
            .initialize_inertial_state(
                p=position,
                q=orientation,
                v=velocity,
                b_a=acc_bias,
                b_g=gyro_bias,
            )
            .initialize_covariance()
        )
