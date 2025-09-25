import jax.numpy as jnp

from core.filter.state import InertialState, State


class TestUnitInertialState:
    """Unit test for inertial state."""

    def test_should_be_possible_to_create(self):
        """Test that the inertial state can be created."""
        position = jnp.array([0, 0, 0])
        orientation = jnp.array([1, 0, 0, 0])
        velocity = jnp.array([0, 0, 0])
        acc_bias = jnp.array([0, 0, 0])
        gyro_bias = jnp.array([0, 0, 0])
        gravity = jnp.array([0, 0, 0])
        payload = (1.0, position, orientation, velocity, acc_bias, gyro_bias, gravity)
        inertial_state = InertialState(
            payload=payload,
        )
        assert inertial_state is not None

    def should_have_map_method(self):
        """Test that the inertial state has a map method."""
        inertial_state = InertialState(
            payload=(
                1.0,
                jnp.array([0, 0, 0]),
                jnp.array([1, 0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
            )
        )
        assert hasattr(inertial_state, "map")
        inertial_state = inertial_state.map(
            lambda x: (x[0] + 1.0, jnp.array([x[1][0] + 2.0, 0, 0]), x[2], x[3], x[4], x[5], x[6])
        )
        assert inertial_state.ts == 2.0
        assert jnp.allclose(inertial_state.p, jnp.array([2.0, 0, 0]))
        assert jnp.allclose(inertial_state.q, jnp.array([1, 0, 0, 0]))
        assert jnp.allclose(inertial_state.v, jnp.array([0, 0, 0]))
        assert jnp.allclose(inertial_state.b_a, jnp.array([0, 0, 0]))
        assert jnp.allclose(inertial_state.b_g, jnp.array([0, 0, 0]))

    def test_should_have_map_position_method(self):
        """Test that the inertial state has a map position method."""
        inertial_state = InertialState(
            payload=(
                1.0,
                jnp.array([15.0, 10.0, 0.5]),
                jnp.array([1, 0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
            )
        )
        assert hasattr(inertial_state, "map_position")
        inertial_state = inertial_state.map_position(lambda x: x + jnp.array([2.0, 1, 0]))
        assert jnp.allclose(inertial_state.p, jnp.array([17.0, 11, 0.5]))

    def test_should_have_apply_timestamp_method(self):
        """Test that the inertial state has a apply timestamp method."""
        inertial_state = InertialState(
            payload=(
                1.0,
                jnp.array([15.0, 10.0, 0.5]),
                jnp.array([1, 0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
            )
        )
        assert hasattr(inertial_state, "apply_timestamp")
        inertial_state = inertial_state.apply_timestamp(10.0).map_position(lambda x: x + jnp.array([2.0, 1, 0]))
        assert jnp.allclose(inertial_state.ts, 10.0)


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

    def test_should_initialize_inertial_state(self):
        """Test that the state can initialize the inertial state."""
        state = State()
        state.initialize_inertial_state(
            payload=(
                140.0,
                jnp.array([0, 0, 0]),
                jnp.array([1, 0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
            )
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

    def test_should_have_map_inertial_state_method(self):
        """Test that the state has a map inertial state method."""
        state = State()
        state.initialize_inertial_state(
            payload=(
                1.0,
                jnp.array([15.0, 10.0, 0.5]),
                jnp.array([1, 0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
                jnp.array([0, 0, 0]),
            )
        )
        assert hasattr(state, "map_inertial_state")
        state = state.map_inertial_state(
            lambda x: x.map_position(lambda x: x + jnp.array([1.0, 1, 0]))
            .map_position(lambda x: x + jnp.array([1.0, 0, 0]))
            .map_position(lambda x: x + jnp.array([0, 0, 0]))
        )
        assert jnp.allclose(state.inertial_state.p, jnp.array([17.0, 11, 0.5]))
