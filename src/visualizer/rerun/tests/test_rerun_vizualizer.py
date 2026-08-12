from pathlib import Path

from visualizer.rerun.rerun_vizualizer import RerunVizualizer


class TestRerunVizualizer:
    """Unit tests for rerun output sink behavior."""

    def test_pipeline_generator_saves_recording_to_file(self, mocker, tmp_path: Path) -> None:
        """File-backed sinks should attach rr.save before any frame is logged."""
        init = mocker.patch("visualizer.rerun.rerun_vizualizer.rr.init")
        save = mocker.patch("visualizer.rerun.rerun_vizualizer.rr.save")
        disconnect = mocker.patch("visualizer.rerun.rerun_vizualizer.rr.disconnect")
        save_path = tmp_path / "data.rrd"
        visualizer = RerunVizualizer("test_app", spawn=False, save_path=save_path)

        generator = visualizer.pipeline_generator()
        generator.close()

        init.assert_called_once()
        save.assert_called_once_with(save_path)
        disconnect.assert_called_once()

    def test_pipeline_generator_streams_to_viewer_and_file(self, mocker, tmp_path: Path) -> None:
        """Both sink should keep viewer and file outputs attached."""
        mocker.patch("visualizer.rerun.rerun_vizualizer.rr.init")
        save = mocker.patch("visualizer.rerun.rerun_vizualizer.rr.save")
        set_sinks = mocker.patch("visualizer.rerun.rerun_vizualizer.rr.set_sinks")
        grpc_sink = mocker.patch("visualizer.rerun.rerun_vizualizer.rr.GrpcSink", return_value="grpc")
        file_sink = mocker.patch("visualizer.rerun.rerun_vizualizer.rr.FileSink", return_value="file")
        save_path = tmp_path / "data.rrd"
        visualizer = RerunVizualizer("test_app", spawn=True, save_path=save_path)

        generator = visualizer.pipeline_generator()
        generator.close()

        grpc_sink.assert_called_once_with()
        file_sink.assert_called_once_with(save_path)
        assert set_sinks.call_args.args == ("grpc", "file")
        assert set_sinks.call_args.kwargs["default_blueprint"] is not None
        save.assert_not_called()

    def test_pipeline_generator_disabled_sink_does_not_connect_to_rerun(self, mocker) -> None:
        """Disabled sinks should drain events without initializing rerun."""
        init = mocker.patch("visualizer.rerun.rerun_vizualizer.rr.init")
        save = mocker.patch("visualizer.rerun.rerun_vizualizer.rr.save")
        disconnect = mocker.patch("visualizer.rerun.rerun_vizualizer.rr.disconnect")
        visualizer = RerunVizualizer("test_app", enabled=False)

        generator = visualizer.pipeline_generator()
        generator.close()

        init.assert_not_called()
        save.assert_not_called()
        disconnect.assert_not_called()
