import jax.numpy as jnp

from core.filter.state import State
from core.filter.updater import Updater


class TestUnitUpdater:
    """Unit test for updater."""

    def test_should_be_possible_to_create(self):
        """Test that the updater can be created."""
        updater = Updater()
        assert updater is not None

    def test_should_have_update_method(self):
        """Test that the updater has an update method."""
        updater = Updater()
        assert hasattr(updater, "state_update")

    def test_update_should_return_state(self):
        """Test that the update method returns a state."""
        updater = Updater()
        state = updater.state_update(State(), (jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0, 1.0])))
        assert state is not None
