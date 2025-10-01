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
            p=jnp.array([0, 0, 0]),
            q=jnp.array([1, 0, 0, 0]),
            v=jnp.array([0, 0, 0]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        ).apply_timestamp(1.0)
        boolean, state = propagator.state_propagation(state, (2.0, jnp.array([0, 0, 0]), jnp.array([0, 0, 0])))
        assert state.ts == 2.0
        assert isinstance(boolean, bool)
        assert isinstance(state, State)

    def test_propagate_should_return_false_and_state_if_timestamp_is_same_as_previous(
        self,
    ):
        """Test that propagate returns false and state if timestamp is same as previous."""
        propagator = Propagator(0.0, 0.0, 0.0, 0.0)
        state = State()
        state.initialize_inertial_state(
            p=jnp.array([0, 0, 0]),
            q=jnp.array([1, 0, 0, 0]),
            v=jnp.array([0, 0, 0]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        ).apply_timestamp(1.0)
        boolean, state = propagator.state_propagation(state, (0.5, jnp.array([0, 0, 0]), jnp.array([0, 0, 0])))
        assert not boolean
        assert state.ts == 1.0
        assert isinstance(state, State)
        assert jnp.allclose(state.inertial_state.p, jnp.array([0, 0, 0]))

    def test_propagate_should_return_not_same_state(self):
        """Test that propagate returns not same state."""
        propagator = Propagator(0.0, 0.0, 0.0, 0.0)
        state = State()
        state.initialize_inertial_state(
            p=jnp.array([0.5, 0.5, 0.5]),
            q=jnp.array([0, 0, 0, 1]),
            v=jnp.array([0.1, 0.1, 0.1]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        ).apply_timestamp(1.0)
        boolean, state = propagator.state_propagation(
            state, (100000.0, jnp.array([3.81, 4.81, 5.81]), jnp.array([100 * 6.81, 7.81, 8.81]))
        )
        assert boolean
        assert isinstance(state, State)
        assert not jnp.allclose(state.inertial_state.p, jnp.array([0.5, 0.5, 0.5]))
