from dataclasses import dataclass


@dataclass(frozen=True)
class FeaturesStreamConfig:
    """Features stream configuration."""

    path: str
    show_stereo_baseline: bool = False


@dataclass(frozen=True)
class VisualizerConfig:
    """Visualizer configuration."""

    app_name: str
    image_streams: dict[str, str]
    features_streams: dict[str, str | FeaturesStreamConfig]
    image_resolution: tuple[int, int]
    imu_path: str
    imu_streams: list[str]
    pose_streams: dict[str, str]

    @property
    def image_stream_enabled(self) -> bool:
        """Check if the image stream is enabled."""
        return len(self.image_streams.keys()) > 0

    @property
    def features_stream_enabled(self) -> bool:
        """Check if the features stream is enabled."""
        return len(self.features_streams.keys()) > 0

    @property
    def imu_stream_enabled(self) -> bool:
        """Check if the imu stream is enabled."""
        return len(self.imu_streams) > 0

    @property
    def pose_stream_enabled(self) -> bool:
        """Check if the pose stream is enabled."""
        return len(self.pose_streams) > 0

    def feature_stream(self, name: str) -> FeaturesStreamConfig:
        """Get the features stream configuration."""
        raw_value = self.features_streams[name]
        if isinstance(raw_value, FeaturesStreamConfig):
            return raw_value
        if isinstance(raw_value, str):
            return FeaturesStreamConfig(path=raw_value)
        if isinstance(raw_value, dict):
            return FeaturesStreamConfig(**raw_value)  # ty: ignore

        msg = f"Unknown format for feature stream config: {type(raw_value)}"
        raise ValueError(msg)
