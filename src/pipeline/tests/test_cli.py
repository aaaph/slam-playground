import os
from pathlib import Path

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
        assert output["repo_root"] == str(Path.cwd().resolve())
        assert output["profile"] == "quick_vio_euroc"
        assert output["dataset"]["name"] == "euroc_mh_01"
        assert output["dataflow"]["name"] == "vio-dataflow.yml"
        assert output["dataflow"]["build"] is False

    def test_pipeline_run_uses_resolved_dataflow_without_build_by_default(self, monkeypatch) -> None:
        """Run a pipeline without building when profile dataflow.build is false."""
        calls = {"build": [], "run": []}
        monkeypatch.setattr("pipeline.cli.dora_build", lambda **kwargs: calls["build"].append(kwargs))
        monkeypatch.setattr("pipeline.cli.dora_run", lambda **kwargs: calls["run"].append(kwargs))

        result = CliRunner().invoke(app, ["pipeline", "run", "--profile", "quick_vio_euroc"])

        assert result.exit_code == 0
        assert os.environ["REPO_ROOT"] == str(Path.cwd().resolve())
        assert os.environ["PIPELINE_PROFILE"] == "quick_vio_euroc"
        assert os.environ["DATASET_NAME"] == "euroc_mh_01"
        assert calls["build"] == []
        assert calls["run"] == [{"dataflow_path": "pipeline/vio-dataflow.yml", "uv": True}]

    def test_pipeline_run_builds_when_profile_requests_build(self, monkeypatch, tmp_path) -> None:
        """Run a pipeline with build when profile dataflow.build is true."""
        profile_dir = tmp_path / "profile"
        profile_dir.mkdir()
        (profile_dir / "build_vio.yaml").write_text(
            """
name: build_vio
dataset: euroc_mh_01
dataflow:
  template: vio-dataflow.yml
  build: true
""".lstrip(),
            encoding="utf-8",
        )
        calls = {"build": [], "run": []}
        monkeypatch.setattr("pipeline.cli.dora_build", lambda **kwargs: calls["build"].append(kwargs))
        monkeypatch.setattr("pipeline.cli.dora_run", lambda **kwargs: calls["run"].append(kwargs))
        monkeypatch.setattr(
            "pipeline.cli.PipelineProfileResolver",
            lambda: __import__("pipeline.profiles").profiles.PipelineProfileResolver(profile_dir=profile_dir),
        )

        result = CliRunner().invoke(app, ["pipeline", "run", "--profile", "build_vio"])

        assert result.exit_code == 0
        assert os.environ["REPO_ROOT"] == str(Path.cwd().resolve())
        assert os.environ["PIPELINE_PROFILE"] == "build_vio"
        assert calls["build"] == [{"dataflow_path": "pipeline/vio-dataflow.yml", "uv": True}]
        assert calls["run"] == [{"dataflow_path": "pipeline/vio-dataflow.yml", "uv": True}]
