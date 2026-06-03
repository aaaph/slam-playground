from typer.testing import CliRunner
from yaml import safe_load

from pipeline.cli import app


class TestPipelineCli:
    """Pipeline CLI tests."""

    def test_profile_resolve_outputs_resolved_profile(self) -> None:
        """Resolve a profile through the profile CLI namespace."""
        result = CliRunner().invoke(app, ["profile", "resolve", "--profile", "quick_vio_euroc"])

        assert result.exit_code == 0
        output = safe_load(result.output)
        assert output["profile"] == "quick_vio_euroc"
        assert output["dataset"]["name"] == "euroc_mh_01"
        assert output["dataflow"]["name"] == "vio-dataflow.yml"

    def test_pipeline_run_still_outputs_resolved_profile(self) -> None:
        """Resolve a pipeline run through the pipeline CLI namespace."""
        result = CliRunner().invoke(app, ["pipeline", "run", "--profile", "quick_vio_euroc"])

        assert result.exit_code == 0
        output = safe_load(result.output)
        assert output["profile"] == "quick_vio_euroc"
        assert output["run"]["mode"] == "batch_fraction"
