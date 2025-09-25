from core.filter.msckf import MSCKF


class TestUnitMSCKF:
    """Unit test for MSCKF."""

    def test_should_be_possible_to_create(self):
        """Test that the MSCKF can be created."""
        msckf = MSCKF()
        assert msckf is not None

    def test_should_have_predict_method(self):
        """Test that the MSCKF has a predict method."""
        msckf = MSCKF()
        assert hasattr(msckf, "predict")

    def test_should_have_update_method(self):
        """Test that the MSCKF has an update method."""
        msckf = MSCKF()
        assert hasattr(msckf, "update")
