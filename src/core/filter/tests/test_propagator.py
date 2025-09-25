from core.filter.filter_interfaces import PredictNoise
from core.filter.propagator import Propagator


class TestUnitPropagator:
    """Unit test for propagator."""

    def test_should_be_possible_to_create(self):
        """Test that the propagator can be created."""
        propagator = Propagator(noises=PredictNoise(ng=0.0, na=0.0, nba=0.0, nbg=0.0))
        assert propagator is not None

    def test_should_have_propagate_method(self):
        """Test that the propagator has a propagate method."""
        propagator = Propagator(noises=PredictNoise(ng=0.0, na=0.0, nba=0.0, nbg=0.0))
        assert hasattr(propagator, "propagate")
