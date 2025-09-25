import jax

from core.filter.filter_interfaces import PredictNoise
from core.filter.state import State
from logger import spawn_logger


class Propagator:
    """Propagator of the Multi-State Constraint Kalman Filter."""

    logger = spawn_logger(app="filter_propagator")

    def __init__(self, ng: float, na: float, nba: float, nbg: float) -> None:
        """Initialize the propagator."""
        self.noises = PredictNoise(ng=ng, na=na, nba=nba, nbg=nbg)

    def state_propagation(self, state: State, imu_data: tuple[float, jax.Array, jax.Array]) -> tuple[bool, State]:
        """
        Propagate the state.

        Args:
            state: State to propagate
            imu_data: IMU data[timestamp, gyro, acc]

        Returns:
            State: Propagated state

        """
        timestamp_ns = imu_data[0]

        dt = timestamp_ns - state.ts
        dt = dt / 1e9

        return (True, state)

    def _nominal_state_propagation(
        self, _state: State, _dt: float, _acc_data: jax.Array, _gyro_data: jax.Array
    ) -> State:
        """Propagate the nominal state."""
        # 123

        return State()
