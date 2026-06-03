from pathlib import Path

import pytest

from pipeline.profiles import PipelineProfileResolver, ProfileOverrides, RunMode, VisualizationSink


class TestPipelineProfileResolver:
    """Tests for pipeline profile resolution."""

    def test_resolve_quick_vio_euroc_profile(self) -> None:
        """Resolve a complete composite profile."""
        resolved = PipelineProfileResolver(repo_root=Path.cwd()).resolve(profile="quick_vio_euroc")

        assert resolved.profile == "quick_vio_euroc"
        assert resolved.dataset.name == "euroc_mh_01"
        assert resolved.dataset.type == "euroc"
        assert resolved.dataset.rig == Path("config/dataset_rig/euroc.yaml")
        assert resolved.dataset.streams.cam0 == Path("cam0/data.csv")
        assert resolved.rig.name == "euroc"
        assert resolved.rig.cam0.resolution == (752, 480)
        assert resolved.rig.cam0.body_sensor_transform.rows == 4
        assert resolved.dataflow.name == "vio-dataflow.yml"
        assert resolved.dataflow.template == Path("pipeline/vio-dataflow.yml")
        assert resolved.visualization.sink == VisualizationSink.FILE
        assert resolved.run.mode == RunMode.BATCH_FRACTION
        assert resolved.run.fraction == 0.05
        assert resolved.run.autostart_after_ready is True
        assert resolved.run.stop_after_dataset_done is True

    def test_load_profile_keeps_profile_name(self) -> None:
        """Composite profiles carry their own explicit name."""
        profile = PipelineProfileResolver(repo_root=Path.cwd()).load_profile("quick_vio_euroc")

        assert profile.name == "quick_vio_euroc"

    def test_cli_overrides_replace_only_explicit_fields(self) -> None:
        """CLI overrides should not erase unrelated profile fields."""
        resolved = PipelineProfileResolver(repo_root=Path.cwd()).resolve(
            profile="my_slam_euroc",
            overrides=ProfileOverrides(
                dataflow="vio-dataflow.yml",
                visualization_sink=VisualizationSink.BOTH,
            ),
        )

        assert resolved.dataset.name == "euroc_mh_01"
        assert resolved.dataflow.name == "vio-dataflow.yml"
        assert resolved.visualization.sink == VisualizationSink.BOTH
        assert resolved.run.mode == RunMode.MANUAL

    def test_fraction_override_implies_batch_fraction_mode(self) -> None:
        """A fraction override should switch a manual profile to batch-fraction mode."""
        resolved = PipelineProfileResolver(repo_root=Path.cwd()).resolve(
            profile="my_slam_euroc",
            overrides=ProfileOverrides(fraction=0.05),
        )

        assert resolved.run.mode == RunMode.BATCH_FRACTION
        assert resolved.run.fraction == 0.05

    def test_missing_required_selector_raises(self) -> None:
        """Without profile or explicit selectors, resolution cannot proceed."""
        with pytest.raises(ValueError, match="dataset must be provided"):
            PipelineProfileResolver(repo_root=Path.cwd()).resolve()
