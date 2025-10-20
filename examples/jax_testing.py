import jax.numpy as jnp

u = jnp.full(10, jnp.nan, jnp.float32)

u.at[0].set(1.0)
