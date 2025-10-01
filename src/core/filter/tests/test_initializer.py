import jax.numpy as jnp

from core.filter.initializer import Initializer
from core.filter.state import State


class TestUnitInitializer:
    """Unit test for initializer."""

    def test_should_be_possible_to_create(self):
        """Test that the initializer can be created."""
        initializer = Initializer()
        assert initializer is not None

    def test_should_have_zero_initialize_method(self):
        """Test that the initializer has a zero initialize method."""
        initializer = Initializer()
        assert hasattr(initializer, "zero_initialize")

    def test_zero_initialize_should_initialize_state_to_zero(self):
        """Test that the zero initialize method initializes the state to zero."""
        initializer = Initializer()
        state = initializer.zero_initialize(State())
        assert state is not None
        assert state.inertial_state is not None
        assert jnp.allclose(state.inertial_state.p, jnp.array([0, 0, 0]))
        assert jnp.allclose(state.inertial_state.q, jnp.array([0, 0, 0, 1]))
        assert jnp.allclose(state.inertial_state.v, jnp.array([0, 0, 0]))
        assert jnp.allclose(state.inertial_state.b_a, jnp.array([0, 0, 0]))
        assert jnp.allclose(state.inertial_state.b_g, jnp.array([0, 0, 0]))
        assert state.covariance is not None
        assert state.covariance.sigma.shape == (15, 15)
        assert state.ts is not None
        assert state.ts > 0

    def test_initializer_should_have_initialize_from_dict(self):
        """Test that the initializer has a initialize from row method."""
        initializer = Initializer()
        assert hasattr(initializer, "initialize_from_dict")

    def test_initialize_from_row_should_initialize_state_from_row(self):
        """Test that the initialize from row method initializes the state from a row."""
        initializer = Initializer()
        position = jnp.array([0, 0, 0])
        orientation = jnp.array([0, 0, 0, 1])
        velocity = jnp.array([0, 0, 0])
        acc_bias = jnp.array([0, 0, 0])
        gyro_bias = jnp.array([0, 0, 0])
        state = initializer.initialize_from_dict(
            State(),
            timestamp=1.0,
            dictionary={
                "position": position,
                "orientation": orientation,
                "velocity": velocity,
                "acc_bias": acc_bias,
                "gyro_bias": gyro_bias,
            },
        )
        assert state is not None
        assert state.inertial_state is not None
        assert jnp.allclose(state.inertial_state.p, jnp.array([0, 0, 0]))
        assert jnp.allclose(state.inertial_state.q, jnp.array([0, 0, 0, 1]))
        assert jnp.allclose(state.inertial_state.v, jnp.array([0, 0, 0]))
        assert jnp.allclose(state.inertial_state.b_a, jnp.array([0, 0, 0]))
        assert jnp.allclose(state.inertial_state.b_g, jnp.array([0, 0, 0]))
