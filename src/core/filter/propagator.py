import jax

from core.filter.filter_interfaces import PredictNoise
from core.filter.state import State


class Propagator:
    """Propagator of the Multi-State Constraint Kalman Filter."""

    def __init__(self, noises: PredictNoise) -> None:
        """Initialize the propagator."""
        self.noises = noises

    def propagate(self, state: State, _imu_data: jax.Array) -> tuple[bool, State]:
        """
        Propagate the state.

        Args:
            state: State to propagate
            imu_data: IMU data[7x1]

        Returns:
            State: Propagated state

        """
        return (True, state)
