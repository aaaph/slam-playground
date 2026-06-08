import json
import os
from pathlib import Path
from typing import cast

from typer.testing import CliRunner
from yaml import safe_load

from pipeline.cli import app
from pipeline.runtime_config import DORA_NODE_ID_ENV, PIPELINE_NODE_CONFIG_ENV


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
        assert "parsed_dataflow" not in output
        assert "node_configs" not in output
        assert "runtime_dataflow" not in output
        runtime_nodes = cast("list[dict[str, object]]", output["dataflow"]["runtime_dataflow"]["nodes"])
        runtime_env_by_id = {str(node["id"]): cast("dict[str, object]", node["env"]) for node in runtime_nodes}
        assert PIPELINE_NODE_CONFIG_ENV in runtime_env_by_id["control"]
        assert PIPELINE_NODE_CONFIG_ENV in runtime_env_by_id["dataset"]
        control_config = json.loads(str(runtime_env_by_id["control"][PIPELINE_NODE_CONFIG_ENV]))
        dataset_config = json.loads(str(runtime_env_by_id["dataset"][PIPELINE_NODE_CONFIG_ENV]))
        assert control_config["node_id"] == "control"
        assert control_config["emit_ready_status"] is False
        assert control_config["run_mode"] == "batch_fraction"
        assert dataset_config["node_id"] == "dataset"
        assert dataset_config["dataset_rig_path"] == "config/dataset_rig/euroc.yaml"

    def test_pipeline_run_uses_resolved_dataflow_without_build_by_default(self, monkeypatch) -> None:
        """Run a pipeline without building when profile dataflow.build is false."""
        build_calls: list[dict[str, object]] = []
        run_calls: list[dict[str, object]] = []
        runtime_dataflow: dict[str, object] = {}
        monkeypatch.setattr("pipeline.cli.dora_build", lambda **kwargs: build_calls.append(kwargs))

        def fake_run(**kwargs) -> None:
            run_calls.append(kwargs)
            runtime_dataflow.update(
                cast(
                    "dict[str, object]",
                    safe_load(Path(str(kwargs["dataflow_path"])).read_text(encoding="utf-8")),
                )
            )

        monkeypatch.setattr("pipeline.cli.dora_run", fake_run)

        result = CliRunner().invoke(app, ["pipeline", "run", "--profile", "quick_vio_euroc"])

        assert result.exit_code == 0
        assert os.environ["REPO_ROOT"] == str(Path.cwd().resolve())
        assert os.environ["PIPELINE_PROFILE"] == "quick_vio_euroc"
        assert os.environ["DATASET_NAME"] == "euroc_mh_01"
        assert json.loads(os.environ["PIPELINE_READY_NODES"]) == [
            "dataset",
            "frontend",
            "fixed_lag_smoother",
            "rerun",
        ]
        assert build_calls == []
        assert len(run_calls) == 1
        runtime_dataflow_path = Path(str(run_calls[0]["dataflow_path"]))
        assert runtime_dataflow_path.name.endswith(".runtime.yml")
        assert runtime_dataflow_path.parent == Path.cwd() / "pipeline"
        assert not runtime_dataflow_path.exists()
        assert run_calls[0]["uv"] is True
        runtime_nodes = cast("list[dict[str, object]]", runtime_dataflow["nodes"])
        runtime_env_by_id = {str(node["id"]): cast("dict[str, object]", node["env"]) for node in runtime_nodes}
        assert {env[DORA_NODE_ID_ENV] for env in runtime_env_by_id.values()} == {
            "control",
            "dataset",
            "frontend",
            "fixed_lag_smoother",
            "rerun",
        }
        control_config = json.loads(str(runtime_env_by_id["control"][PIPELINE_NODE_CONFIG_ENV]))
        dataset_config = json.loads(str(runtime_env_by_id["dataset"][PIPELINE_NODE_CONFIG_ENV]))
        assert control_config["node_id"] == "control"
        assert control_config["emit_ready_status"] is False
        assert control_config["expected_ready_nodes"] == [
            "dataset",
            "frontend",
            "fixed_lag_smoother",
            "rerun",
        ]
        assert control_config["ready_inputs"] == {
            "dataset_status": "dataset",
            "fixed_lag_smoother_status": "fixed_lag_smoother",
            "frontend_status": "frontend",
            "rerun_status": "rerun",
        }
        assert control_config["run_mode"] == "batch_fraction"
        assert control_config["fraction"] == 0.05
        assert dataset_config["node_id"] == "dataset"
        assert dataset_config["emit_ready_status"] is True
        assert dataset_config["dataset_rig_path"] == "config/dataset_rig/euroc.yaml"

    def test_pipeline_run_embeds_control_config_for_ready_nodes(self, monkeypatch) -> None:
        """Runtime dataflow embeds control config for dynamic ready-node tracking."""
        run_calls: list[dict[str, object]] = []
        runtime_dataflow: dict[str, object] = {}
        monkeypatch.setattr("pipeline.cli.dora_build", lambda **_kwargs: None)

        def fake_run(**kwargs) -> None:
            run_calls.append(kwargs)
            runtime_dataflow.update(
                cast(
                    "dict[str, object]",
                    safe_load(Path(str(kwargs["dataflow_path"])).read_text(encoding="utf-8")),
                )
            )

        monkeypatch.setattr("pipeline.cli.dora_run", fake_run)

        result = CliRunner().invoke(app, ["pipeline", "run", "--profile", "dataset_viz"])

        assert result.exit_code == 0
        assert len(run_calls) == 1
        runtime_nodes = cast("list[dict[str, object]]", runtime_dataflow["nodes"])
        runtime_env_by_id = {str(node["id"]): cast("dict[str, object]", node["env"]) for node in runtime_nodes}
        control_config = json.loads(str(runtime_env_by_id["control"][PIPELINE_NODE_CONFIG_ENV]))
        dataset_config = json.loads(str(runtime_env_by_id["dataset"][PIPELINE_NODE_CONFIG_ENV]))
        rerun_config = json.loads(str(runtime_env_by_id["rerun"][PIPELINE_NODE_CONFIG_ENV]))

        assert control_config["node_id"] == "control"
        assert control_config["emit_ready_status"] is False
        assert control_config["expected_ready_nodes"] == ["dataset", "rerun"]
        assert control_config["ready_inputs"] == {
            "dataset_status": "dataset",
            "rerun_status": "rerun",
        }
        assert dataset_config["emit_ready_status"] is True
        assert rerun_config["emit_ready_status"] is True

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
        assert len(calls["build"]) == 1
        assert len(calls["run"]) == 1
        assert calls["build"][0]["dataflow_path"] == calls["run"][0]["dataflow_path"]
        assert Path(calls["run"][0]["dataflow_path"]).name.endswith(".runtime.yml")
        assert calls["build"][0]["uv"] is True
        assert calls["run"][0]["uv"] is True
