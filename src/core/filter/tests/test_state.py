import jax.numpy as jnp

from core.filter.state import InertialState, State


class TestUnitInertialState:
    """Unit test for inertial state."""

    def test_should_be_possible_to_create(self):
        """Test that the inertial state can be created."""
        inertial_state = InertialState(
            p=jnp.array([0, 0, 0]),
            q=jnp.array([0, 0, 0, 0]),
            v=jnp.array([0, 0, 0]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        )
        assert inertial_state is not None


class TestUnitState:
    """Unit test for state."""

    def test_should_be_possible_to_create(self):
        """Test that the state can be created."""
        state = State()
        assert state is not None

    def test_should_have_inertial_state_field(self):
        """Test that the state has an inertial state field."""
        state = State()
        assert hasattr(state, "inertial_state")

    def test_should_have_initialize_inertial_state_method(self):
        """Test that the state has an initialize inertial state method."""
        state = State()
        assert hasattr(state, "initialize_inertial_state")

    def test_should_have_initialize_ts_method(self):
        """Test that the state has an initialize timestamp method."""
        state = State()
        assert hasattr(state, "initialize_ts")

    def test_should_initialize_inertial_state(self):
        """Test that the state can initialize the inertial state."""
        state = State()
        state.initialize_ts(140).initialize_inertial_state(
            p=jnp.array([0, 0, 0]),
            q=jnp.array([0, 0, 0, 0]),
            v=jnp.array([0, 0, 0]),
            b_a=jnp.array([0, 0, 0]),
            b_g=jnp.array([0, 0, 0]),
        )
        assert state.inertial_state is not None
        assert state.ts == 140

    def test_should_have_covariance_field(self):
        """Test that the state has a covariance field."""
        state = State()
        assert hasattr(state, "covariance")

    def test_should_have_initialize_covariance_method(self):
        """Test that the state has an initialize covariance method."""
        state = State()
        assert hasattr(state, "initialize_covariance")
        state = state.initialize_covariance()
        assert state.covariance is not None
