from dataset.dataset_config import SensorConfig


class TestSensorConfig:
    """Test dataset configuration."""

    def test_should_have_file_path_as_constructor_argument(self):
        """Test that the dataset config has a file path as a constructor argument."""
        config = SensorConfig(file_path="test")
        assert config.file_path == "test"

    def test_should_have_dict_of_config_as_property(self):
        """Test that the dataset config has a dict of config as a property."""
        config = SensorConfig(file_path="test")
        assert config.payload is not None
