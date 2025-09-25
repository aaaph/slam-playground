import jax.numpy as jnp

state = jnp.array(
    [
        1.0,
        [0, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
)

state = state.at[1].add(jnp.array([2.0, 0, 0]))
