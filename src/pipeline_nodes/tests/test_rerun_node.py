from pipeline_nodes.rerun_node import RerunNodeConfigProvider


class TestRerunNodeConfig:
    """Test rerun node configuration."""

    def test_image_streams(self, monkeypatch) -> None:
        """Test that the image streams are returned."""
        monkeypatch.setenv(
            "VISUALIZE_IMAGE_STREAMS",
            '{"left": "/world/odom/base_link/cam0/frame", "left_rect": "/world/odom/base_link/cam0/rect"}',
        )
        config = RerunNodeConfigProvider()
        assert config.image_stream_names == ["left", "left_rect"]

    def test_image_streams_empty(self, monkeypatch) -> None:
        """Test that the image streams are empty."""
        monkeypatch.setenv("VISUALIZE_IMAGE_STREAMS", "{}")
        config = RerunNodeConfigProvider()
        assert config.image_stream_names == []
