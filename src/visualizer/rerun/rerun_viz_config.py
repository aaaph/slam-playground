from dataclasses import dataclass


@dataclass(frozen=True)
class VisualizerConfig:
    """Visualizer configuration."""

    app_name: str
    image_streams: dict[str, str]
    features_streams: dict[str, str]

    @property
    def image_stream_enabled(self) -> bool:
        """Check if the image stream is enabled."""
        return len(self.image_streams.keys()) > 0

    @property
    def features_stream_enabled(self) -> bool:
        """Check if the features stream is enabled."""
        return len(self.features_streams.keys()) > 0
