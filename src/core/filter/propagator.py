import jax

from core.filter.filter_interfaces import PredictNoise
from core.filter.state import State


class Propagator:
    """Propagator of the Multi-State Constraint Kalman Filter."""

    def __init__(self, ng: float, na: float, nba: float, nbg: float) -> None:
        """Initialize the propagator."""
        self.noises = PredictNoise(ng=ng, na=na, nba=nba, nbg=nbg)

    def propagate(self, state: State, _imu_data: tuple[float, jax.Array, jax.Array]) -> tuple[bool, State]:
        """
        Propagate the state.

        Args:
            state: State to propagate
            imu_data: IMU data[timestamp, gyro, acc]

        Returns:
            State: Propagated state

        """
        return (True, state)
