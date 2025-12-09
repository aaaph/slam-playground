from pathlib import Path

import pytest

from dataset.dataset_config import CameraConfig, SensorConfig


class TestUnitSensorConfig:
    """Unit test for dataset configuration."""

    def test_should_be_possible_to_create(self):
        """Test that the dataset config has a dict of config as a property."""
        config = SensorConfig({})
        assert config is not None

    def test_should_have_dict_of_config_as_property(self):
        """Test that the dataset config has a dict of config as a property."""
        config = SensorConfig({})
        assert config.payload is not None

    def test_should_check_file_exists(self, mocker):
        """Test that the dataset config checks if the file exists using pytest-mock."""
        mock_exists = mocker.patch("pathlib.Path.exists", return_value=True)

        # Method 1: Mock Path.open() with a proper file-like object
        mock_file = mocker.mock_open(read_data='{"test": "data"}')
        mock_open = mocker.patch("pathlib.Path.open", mock_file)

        config = SensorConfig.from_yaml("test_path")

        mock_exists.assert_called_once()
        mock_open.assert_called_once_with("test_path", "r")
        assert config.payload == {"test": "data"}

    def test_should_check_file_if_not_exist_throw_exception(self, mocker):
        """Test that the dataset config checks if the file exists using pytest-mock."""
        # Method 1: Mocking pathlib.Path.exists() method
        _mock_exists = mocker.patch("pathlib.Path.exists", return_value=False)

        # Assuming your SensorConfig will call Path(file_path).exists() in __init__
        with pytest.raises(FileNotFoundError, match="Config file does not exist"):
            _config = SensorConfig.from_yaml("test_path")

    def test_should_be_callable_via_get_item(self):
        """Test that the dataset config is callable via get item."""
        config = SensorConfig({"test": "data"})
        assert config["test"] == "data"

    def test_should_implement_contains(self):
        """Test that the dataset config implements contains."""
        config = SensorConfig({"test": "data"})
        assert "test" in config

    def test_should_work_without_model(self, mocker):
        """Test that the dataset config works without a model (uses raw dict)."""
        _mock_exists = mocker.patch("pathlib.Path.exists", return_value=True)
        mock_file = mocker.mock_open(read_data='{"hello": "data", "world": 1}')
        _mock_open = mocker.patch("pathlib.Path.open", mock_file)

        config = SensorConfig.from_yaml("test_path")
        assert config["hello"] == "data"
        assert config["world"] == 1
        assert isinstance(config.payload, dict)

    def test_should_work_with_model(self, mocker):
        """Test that the dataset config works with a model class."""
        _mock_exists = mocker.patch("pathlib.Path.exists", return_value=True)
        mock_file = mocker.mock_open(read_data='{"hello": "data", "world": 1}')
        _mock_open = mocker.patch("pathlib.Path.open", mock_file)

        config = SensorConfig.from_yaml("test_path")
        assert config["hello"] == "data"
        assert config["world"] == 1
        assert config.payload["hello"] == "data"
        assert config.payload["world"] == 1
        assert isinstance(config.payload, dict)


class TestIntegrationSensorConfig:
    """Integration test for dataset configuration."""

    def test_should_be_possible_to_create(self):
        """Test that the dataset config has a dict of config as a property."""
        current_dir = Path(__file__).parent
        config = SensorConfig.from_yaml(f"{current_dir}/testing_sensor_config.yaml")
        assert config is not None

    def test_should_be_possible_to_get_item(self):
        """Test that the dataset config has a dict of config as a property."""
        current_dir = Path(__file__).parent
        file_path = f"{current_dir}/testing_sensor_config.yaml"
        config = SensorConfig.from_yaml(file_path)
        assert config["sensor_type"] == "camera"
        assert config["comment"] == "VI-Sensor cam0 (MT9M034)"
        assert config["T_BS"] is not None
        assert config["rate_hz"] == 20
        assert config["resolution"] == [752, 480]
        assert config["camera_model"] == "pinhole"
        assert config["intrinsics"] is not None
        assert config["distortion_model"] == "radial-tangential"
        assert config["distortion_coefficients"] is not None

    def test_should_be_possible_to_get_camera_config(self):
        """Test that the dataset config has a dict of config as a property."""
        current_dir = Path(__file__).parent
        file_path = f"{current_dir}/testing_sensor_config.yaml"
        config = CameraConfig.from_yaml(file_path)
        assert config.resolution == [752, 480]
        assert config.body_sensor_transform is not None

    def test_camera_config_methods(self):
        """Test that the dataset config methods work."""
        current_dir = Path(__file__).parent
        file_path = f"{current_dir}/testing_sensor_config.yaml"
        camera_config = CameraConfig.from_yaml(file_path)

        assert camera_config.k is not None
        assert camera_config.k_matrix_in_gtsam() is not None
