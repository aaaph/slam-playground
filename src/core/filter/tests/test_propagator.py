import jax.numpy as jnp

from core.filter.propagator import Propagator
from core.filter.state import State


class TestUnitPropagator:
    """Unit test for propagator."""

    def test_should_be_possible_to_create(self):
        """Test that the propagator can be created."""
        propagator = Propagator(0.0, 0.0, 0.0, 0.0)
        assert propagator is not None

    def test_should_have_propagate_method(self):
        """Test that the propagator has a state_propagation method."""
        propagator = Propagator(0.0, 0.0, 0.0, 0.0)
        assert hasattr(propagator, "state_propagation")

    def test_propagate_should_return_tuple_with_boolean_and_state(self):
        """Test that the propagate method returns a tuple with a boolean and a state."""
        propagator = Propagator(0.0, 0.0, 0.0, 0.0)
        state = State()
        state.initialize_inertial_state(
            (
                1.0,
                jnp.array([0, 0, 0]),
                jnp.array([1, 0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
            )
        )
        boolean, state = propagator.state_propagation(state, (1.0, jnp.array([0, 0, 0]), jnp.array([0, 0, 0])))
        assert isinstance(boolean, bool)
        assert isinstance(state, State)
