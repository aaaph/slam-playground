import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from core.filter.state import InertialState, State
from core.transformations.helpers import skew
from logger import spawn_logger


class Updater:
    """Updater of the Multi-State Constraint Kalman Filter."""

    logger = spawn_logger(app="filter_updater")

    def __init__(self) -> None:
        """Initialize the updater."""

    def state_update(self, state: State, measurements: tuple[jax.Array, jax.Array]) -> State:
        """Update the state of the MSCKF."""
        z_position = jnp.array(measurements[0])
        z_quaternion: Rotation = Rotation.from_quat(jnp.array(measurements[1]))
        self.logger.debug(f"z_position: {z_position}. z_quaternion: {z_quaternion}.")

        u = state.covariance.sigma.copy()
        h = jnp.zeros((6, 15))
        h = h.at[0:3, 0:3].set(jnp.eye(3))
        h = h.at[3:6, 3:6].set(jnp.eye(3))
        v = jnp.diag(
            jnp.array([1e-4, 1e-4, 1e-4, jnp.deg2rad(0.01) ** 2, jnp.deg2rad(0.01) ** 2, jnp.deg2rad(0.01) ** 2])
        )
        s = h @ u @ h.T + v
        k = u @ h.T @ jnp.linalg.inv(s)
        i_kh = jnp.eye(15) - k @ h

        u_new = i_kh @ u @ i_kh.T + k @ v @ k.T
        u_new = (u_new + u_new.T) / 2
        state.apply_covariance(u_new)

        r = jnp.zeros((6, 1))
        r = r.at[0:3, 0].set(z_position - state.inertial_state.p)
        q_pred = Rotation.from_quat(state.inertial_state.q)
        dq = z_quaternion * q_pred.inv()
        r = r.at[3:6, 0].set(dq.as_rotvec())

        errors = k @ r

        def apply_new_rotation(rotation_error: jax.Array, rotation_pred: Rotation) -> jax.Array:
            delta_rotation = Rotation.from_rotvec(rotation_error)
            new_q = (delta_rotation * rotation_pred).as_quat()
            return new_q / jnp.linalg.norm(new_q)

        def apply_inertial_state(inertial_state: InertialState, errors: jax.Array) -> InertialState:
            return (
                inertial_state.map_position(lambda x: jnp.add(x, jnp.array(errors[0:3, 0])))
                .map_velocity(lambda x: jnp.add(x, jnp.array(errors[6:9, 0])))
                .map_acc_bias(lambda x: jnp.add(x, jnp.array(errors[9:12, 0])))
                .map_gyro_bias(lambda x: jnp.add(x, jnp.array(errors[12:15, 0])))
                .map_orientation(lambda _: apply_new_rotation(jnp.array(errors[3:6, 0]), q_pred))
            )

        state.map_inertial_state(lambda inertial_state: apply_inertial_state(inertial_state, errors))

        self.logger.debug(f"state.inertial_state: {state.inertial_state}")

        g = jnp.eye(15)
        g = g.at[3:6, 3:6].set(jnp.eye(3) - skew(0.5 * jnp.array(errors[3:6, 0])))

        u_new = g @ state.covariance.sigma @ g.T
        u_new = (u_new + u_new.T) / 2
        state.apply_covariance(u_new)

        return state
