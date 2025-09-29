import jax
import jax.numpy as jnp


def omega(w: jax.Array) -> jax.Array:
    """
    Create a 4x4 skew-symmetric matrix from a 3D vector w.

    # [ -|w x| w ]
    # [  -w    0 ]
    """
    omega = jnp.zeros((4, 4))
    omega = omega.at[:3, :3].set(-skew(w))
    omega = omega.at[3, :3].set(-w)
    return omega.at[:3, 3].set(w)


def skew(vector: jax.Array) -> jax.Array:
    """
    Create a 3x3 skew-symmetric matrix from a 3D vector.

    # [ 0 -z y ]
    # [ z  0 -x ]
    # [ -y x  0 ]
    """
    x, y, z = vector
    return jnp.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
