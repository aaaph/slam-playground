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
        assert jnp.allclose(state.inertial_state.q, jnp.array([1, 0, 0, 0]))
        assert jnp.allclose(state.inertial_state.v, jnp.array([0, 0, 0]))
        assert jnp.allclose(state.inertial_state.b_a, jnp.array([0, 0, 0]))
        assert jnp.allclose(state.inertial_state.b_g, jnp.array([0, 0, 0]))
        assert state.covariance is not None
        assert jnp.allclose(state.covariance.cov, jnp.eye(18))
        assert state.ts is not None
        assert state.ts > 0

    def test_initializer_should_have_initialize_from_row(self):
        """Test that the initializer has a initialize from row method."""
        initializer = Initializer()
        assert hasattr(initializer, "initialize_from_row")

    def test_initialize_from_row_should_initialize_state_from_row(self):
        """Test that the initialize from row method initializes the state from a row."""
        initializer = Initializer()
        array = jnp.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        state = initializer.initialize_from_row(
            State(), (1.0, array[0:3], array[3:7], array[7:10], array[10:13], array[13:16])
        )
        assert state is not None
        assert state.inertial_state is not None
        assert jnp.allclose(state.inertial_state.p, jnp.array([0, 0, 0]))
        assert jnp.allclose(state.inertial_state.q, jnp.array([0, 0, 0, 1]))
        assert jnp.allclose(state.inertial_state.v, jnp.array([0, 0, 0]))
        assert jnp.allclose(state.inertial_state.b_a, jnp.array([0, 0, 0]))
        assert jnp.allclose(state.inertial_state.b_g, jnp.array([0, 0, 0]))
