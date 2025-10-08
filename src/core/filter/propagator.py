import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from core.filter.filter_interfaces import PredictNoise
from core.filter.state import State
from core.transformations.helpers import omega, skew
from logger import spawn_logger


class Propagator:
    """Propagator of the Multi-State Constraint Kalman Filter."""

    logger = spawn_logger(app="filter_propagator")

    def __init__(self, ng: float, na: float, nba: float, nbg: float) -> None:
        """Initialize the propagator."""
        self.noises = PredictNoise(ng=ng, na=na, nba=nba, nbg=nbg)
        self.gyro_threshold = 0.00005

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
        if dt == 0:
            self.logger.warning("Timestamp is the same as the previous timestamp")
            return (
                False,
                state,
            )
        if dt < 0:
            self.logger.error("Timestamp is in the past")
            return (
                False,
                state,
            )
        gyro_data = jnp.array(imu_data[1])
        acc_data = jnp.array(imu_data[2])
        self.logger.warning(f"timestamp_ns: {timestamp_ns:.0f}, dt: {dt}, acc: {acc_data}, gyro: {gyro_data}")

        sigma_next = self._error_covariance_propagation(state, dt, acc_data, gyro_data)
        p_next, v_next, q_next = self._nominal_state_propagation(state, dt, acc_data, gyro_data)
        self.logger.debug(f"p_next: {p_next}, v_next: {v_next}, q_next: {q_next}")
        return (
            True,
            state.apply_timestamp(timestamp_ns)
            .map_inertial_state(
                lambda x: x.map_position(lambda _: p_next)
                .map_velocity(lambda _: v_next)
                .map_orientation(lambda _: q_next)
            )
            .apply_covariance(sigma_next),
        )

    def _nominal_state_propagation(
        self, state: State, dt: float, acc_data: jax.Array, gyro_data: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Propagate the nominal state."""
        p, q, v, accel_bias, gyro_bias = (
            state.inertial_state.p,
            state.inertial_state.q,
            state.inertial_state.v,
            state.inertial_state.b_a,
            state.inertial_state.b_g,
        )
        g = jnp.array([0, 0, -9.81])

        gyro_norm = jax.numpy.linalg.norm(gyro_data - gyro_bias)
        omega_matrix = omega(gyro_data - gyro_bias)
        theta = gyro_norm * dt

        if gyro_norm > self.gyro_threshold:
            dq_dt = (
                jnp.cos(theta * 0.5) * jnp.eye(4) + ((1 / gyro_norm) * jnp.sin(theta * 0.5)) * omega_matrix
            ) @ q
            dq_dt2 = (
                jnp.cos(theta * 0.25) * jnp.eye(4) + ((1 / gyro_norm) * jnp.sin(theta * 0.25)) * omega_matrix
            ) @ q
        else:
            dq_dt = (jnp.eye(4) + 0.5 * dt * omega_matrix) * jnp.cos(theta * 0.5) @ q
            dq_dt2 = (jnp.eye(4) + 0.25 * dt * omega_matrix) * jnp.cos(theta * 0.25) @ q

        q_next = dq_dt
        rotation_next = Rotation.from_quat(q_next).as_matrix()
        rotation_half_next = Rotation.from_quat(dq_dt2).as_matrix()
        rotation = Rotation.from_quat(q).as_matrix()

        # integrate velocity
        v_k1 = rotation @ (acc_data - accel_bias) + g
        v_k2 = rotation_half_next @ (acc_data - accel_bias) + g
        v_k3 = v_k2
        v_k4 = rotation_next @ (acc_data - accel_bias) + g
        v_next = v + dt * (v_k1 + 2 * v_k2 + 2 * v_k3 + v_k4) / 6

        # integrate position
        p_k1 = v
        # k2 = v_next_half = v + 0.5 * dt * v' = v + 0.5 * dt * v_k1(evaluate at t0)
        p_k2 = v + 0.5 * dt * v_k1
        p_k3 = v + 0.5 * dt * v_k2  # v_k2 is evaluated at t0 + 0.5*delta
        p_k4 = v + dt * v_k3
        p_next = p + dt * (p_k1 + 2 * p_k2 + 2 * p_k3 + p_k4) / 6
        return p_next, v_next, q_next / jnp.linalg.norm(q_next)

    def _error_covariance_propagation(
        self, state: State, dt: float, acc_data: jax.Array, gyro_data: jax.Array
    ) -> jax.Array:
        accel_bias, gyro_bias, q = (state.inertial_state.b_a, state.inertial_state.b_g, state.inertial_state.q)
        sigma = state.covariance.sigma.copy()
        rotation = Rotation.from_quat(q).as_matrix()

        f = jnp.zeros((15, 15))
        f = f.at[0:3, 6:9].set(jnp.eye(3))  # dp/dv
        f = f.at[3:6, 3:6].set(-skew(gyro_data - gyro_bias))  # δΘ/δΘ
        f = f.at[3:6, 12:15].set(-jnp.eye(3))  # δΘ/δb_w
        f = f.at[6:9, 3:6].set(-rotation @ skew(acc_data - accel_bias))  # dv/dθ
        f = f.at[6:9, 9:12].set(-rotation)  # dv/db_a
        # 123f = f.at[6:9, 15:18].set(jnp.eye(3))  # dv/dg

        # G is the projection of noise into state space
        g = jnp.zeros((15, 12))
        g = g.at[3:6, 0:3].set(jnp.eye(3))  # δΘ/gyro_noise
        g = g.at[6:9, 3:6].set(jnp.eye(3))  # dv/accel_noise
        g = g.at[9:12, 6:9].set(jnp.eye(3))  # δba/accel_bias_random_walk
        g = g.at[12:15, 9:12].set(jnp.eye(3))  # δbg/gyro_bias_random_walk

        q = jnp.zeros((12, 12))
        q = q.at[0:3, 0:3].set(jnp.eye(3) * self.noises.ng**2 * dt)
        q = q.at[3:6, 3:6].set(jnp.eye(3) * self.noises.na**2 * dt)
        q = q.at[6:9, 6:9].set(jnp.eye(3) * self.noises.nba**2 * dt)
        q = q.at[9:12, 9:12].set(jnp.eye(3) * self.noises.nbg**2 * dt)

        f_dt = f * dt
        f_dt_2 = f_dt @ f_dt
        f_dt_3 = f_dt_2 @ f_dt
        phi = jnp.eye(15) + f_dt + f_dt_2 / 2 + f_dt_3 / 6  # Taylor expansion of the state transition matrix

        q_k = phi @ g @ q @ g.T @ phi.T

        sigma_next = phi @ sigma @ phi.T + q_k
        sigma_next = (sigma_next + sigma_next.T) / 2

        return sigma_next  # noqa: RET504
