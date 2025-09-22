class SensorConfig:
    """Sensor configuration."""

    def __init__(self, file_path: str) -> None:
        """Initialize the sensor configuration."""
        self.file_path = file_path
        self.payload = {}
