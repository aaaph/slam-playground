import json
import os
import re
import sys
from pathlib import Path
from typing import cast

from typer.testing import CliRunner
from yaml import safe_load

from pipeline import cli as pipeline_cli
from pipeline.cli import app
from pipeline.profiles import PipelineProfileResolver
from pipeline.runtime_config import DORA_NODE_ID_ENV, PIPELINE_NODE_CONFIG_ENV

ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class TestPipelineCli:
    """Pipeline CLI tests."""

    def test_pipeline_run_accepts_short_help_flag(self) -> None:
        """Pipeline run help should be available through just-friendly -h."""
        result = CliRunner().invoke(app, ["pipeline", "run", "-h"], color=True)
        output = ANSI_ESCAPE_RE.sub("", result.output)

        assert result.exit_code == 0
        assert "Resolve and launch the requested pipeline run." in output
        assert "--profile" in output
        assert "--dataset" in output
        assert "--viz" in output

    def test_pipeline_run_accepts_viz_alias(self, monkeypatch, euroc_mh_01_dataset_dir: Path) -> None:
        """Pipeline run should accept --viz as visualization sink override."""
        run_calls: list[dict[str, object]] = []
        runtime_dataflow: dict[str, object] = {}
        _patch_profile_resolver(monkeypatch, dataset_dir=euroc_mh_01_dataset_dir)
        monkeypatch.setattr("pipeline.cli.dora_build", lambda **_kwargs: None)
        monkeypatch.setattr("pipeline.cli._dataset_pre_cache", lambda _resolved_profile: None)

        def fake_run(dataflow_path: Path, **kwargs) -> None:
            kwargs["dataflow_path"] = dataflow_path
            run_calls.append(kwargs)
            runtime_dataflow.update(
                cast(
                    "dict[str, object]",
                    safe_load(Path(str(kwargs["dataflow_path"])).read_text(encoding="utf-8")),
                )
            )

        monkeypatch.setattr("pipeline.cli._run_dora_dataflow", fake_run)

        result = CliRunner().invoke(app, ["pipeline", "run", "--profile", "dataset_viz", "--viz", "off"])

        assert result.exit_code == 0
        assert len(run_calls) == 1
        runtime_nodes = cast("list[dict[str, object]]", runtime_dataflow["nodes"])
        runtime_env_by_id = {str(node["id"]): cast("dict[str, object]", node["env"]) for node in runtime_nodes}
        rerun_config = json.loads(str(runtime_env_by_id["rerun"][PIPELINE_NODE_CONFIG_ENV]))
        assert rerun_config["sink"] == "off"

    def test_profile_resolve_outputs_resolved_profile(
        self,
        monkeypatch,
        euroc_mh_01_dataset_dir: Path,
    ) -> None:
        """Resolve a profile through the profile CLI namespace."""
        _patch_profile_resolver(monkeypatch, dataset_dir=euroc_mh_01_dataset_dir)

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
        assert PIPELINE_NODE_CONFIG_ENV in runtime_env_by_id["rerun"]
        control_config = json.loads(str(runtime_env_by_id["control"][PIPELINE_NODE_CONFIG_ENV]))
        dataset_config = json.loads(str(runtime_env_by_id["dataset"][PIPELINE_NODE_CONFIG_ENV]))
        rerun_config = json.loads(str(runtime_env_by_id["rerun"][PIPELINE_NODE_CONFIG_ENV]))
        assert control_config["node_id"] == "control"
        assert control_config["emit_ready_status"] is False
        assert control_config["run_mode"] == "batch_fraction"
        assert control_config["transport"] == "none"
        assert dataset_config["node_id"] == "dataset"
        assert dataset_config["dataset_rig_path"] == "config/dataset_rig/euroc.yaml"
        assert rerun_config["node_id"] == "rerun"
        assert rerun_config["sink"] == "file"

    def test_pipeline_run_uses_resolved_dataflow_without_build_by_default(
        self,
        monkeypatch,
        euroc_mh_01_dataset_dir: Path,
    ) -> None:
        """Run a pipeline without building when profile dataflow.build is false."""
        build_calls: list[dict[str, object]] = []
        run_calls: list[dict[str, object]] = []
        runtime_dataflow: dict[str, object] = {}
        _patch_profile_resolver(monkeypatch, dataset_dir=euroc_mh_01_dataset_dir)
        monkeypatch.setattr("pipeline.cli.dora_build", lambda **kwargs: build_calls.append(kwargs))
        monkeypatch.setattr("pipeline.cli._dataset_pre_cache", lambda _resolved_profile: None)

        def fake_run(dataflow_path: Path, **kwargs) -> None:
            kwargs["dataflow_path"] = dataflow_path
            run_calls.append(kwargs)
            runtime_dataflow.update(
                cast(
                    "dict[str, object]",
                    safe_load(Path(str(kwargs["dataflow_path"])).read_text(encoding="utf-8")),
                )
            )

        monkeypatch.setattr("pipeline.cli._run_dora_dataflow", fake_run)

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
        assert run_calls[0]["repo_root"] == Path.cwd().resolve()
        assert run_calls[0]["stop_on_completed"] is True
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
        rerun_config = json.loads(str(runtime_env_by_id["rerun"][PIPELINE_NODE_CONFIG_ENV]))
        assert control_config["node_id"] == "control"
        assert control_config["emit_ready_status"] is False
        assert control_config["transport"] == "none"
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
        assert rerun_config["node_id"] == "rerun"
        assert rerun_config["sink"] == "file"

    def test_pipeline_run_ensures_dataset_cache_before_launch(
        self,
        monkeypatch,
        euroc_mh_01_dataset_dir: Path,
    ) -> None:
        """Pipeline run should prepare the resolved dataset before launching Dora."""
        seen_datasets: list[str] = []
        _patch_profile_resolver(monkeypatch, dataset_dir=euroc_mh_01_dataset_dir)
        monkeypatch.setattr("pipeline.cli.dora_build", lambda **_kwargs: None)
        monkeypatch.setattr(
            "pipeline.cli._dataset_pre_cache",
            lambda resolved_profile: seen_datasets.append(resolved_profile.dataset.name),
        )
        monkeypatch.setattr("pipeline.cli._run_dora_dataflow", lambda _dataflow_path, **_kwargs: None)

        result = CliRunner().invoke(app, ["pipeline", "run", "--profile", "quick_vio_euroc"])

        assert result.exit_code == 0
        assert seen_datasets == ["euroc_mh_01"]

    def test_pipeline_run_embeds_control_config_for_ready_nodes(
        self,
        monkeypatch,
        euroc_mh_01_dataset_dir: Path,
    ) -> None:
        """Runtime dataflow embeds control config for dynamic ready-node tracking."""
        run_calls: list[dict[str, object]] = []
        runtime_dataflow: dict[str, object] = {}
        _patch_profile_resolver(monkeypatch, dataset_dir=euroc_mh_01_dataset_dir)
        monkeypatch.setattr("pipeline.cli.dora_build", lambda **_kwargs: None)
        monkeypatch.setattr("pipeline.cli._dataset_pre_cache", lambda _resolved_profile: None)

        def fake_run(dataflow_path: Path, **kwargs) -> None:
            kwargs["dataflow_path"] = dataflow_path
            run_calls.append(kwargs)
            runtime_dataflow.update(
                cast(
                    "dict[str, object]",
                    safe_load(Path(str(kwargs["dataflow_path"])).read_text(encoding="utf-8")),
                )
            )

        monkeypatch.setattr("pipeline.cli._run_dora_dataflow", fake_run)

        result = CliRunner().invoke(app, ["pipeline", "run", "--profile", "dataset_viz"])

        assert result.exit_code == 0
        assert len(run_calls) == 1
        assert run_calls[0]["stop_on_completed"] is False
        runtime_nodes = cast("list[dict[str, object]]", runtime_dataflow["nodes"])
        runtime_env_by_id = {str(node["id"]): cast("dict[str, object]", node["env"]) for node in runtime_nodes}
        control_config = json.loads(str(runtime_env_by_id["control"][PIPELINE_NODE_CONFIG_ENV]))
        dataset_config = json.loads(str(runtime_env_by_id["dataset"][PIPELINE_NODE_CONFIG_ENV]))
        rerun_config = json.loads(str(runtime_env_by_id["rerun"][PIPELINE_NODE_CONFIG_ENV]))

        assert control_config["node_id"] == "control"
        assert control_config["emit_ready_status"] is False
        assert control_config["transport"] == "http"
        assert control_config["expected_ready_nodes"] == ["dataset", "rerun"]
        assert control_config["ready_inputs"] == {
            "dataset_status": "dataset",
            "rerun_status": "rerun",
        }
        assert dataset_config["emit_ready_status"] is True
        assert rerun_config["emit_ready_status"] is True
        assert rerun_config["sink"] == "app"

    def test_pipeline_run_builds_when_profile_requests_build(
        self,
        monkeypatch,
        tmp_path: Path,
        euroc_mh_01_dataset_dir: Path,
    ) -> None:
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
        monkeypatch.setattr("pipeline.cli._dataset_pre_cache", lambda _resolved_profile: None)
        monkeypatch.setattr(
            "pipeline.cli._run_dora_dataflow",
            lambda dataflow_path, **kwargs: calls["run"].append({"dataflow_path": dataflow_path, **kwargs}),
        )
        _patch_profile_resolver(monkeypatch, dataset_dir=euroc_mh_01_dataset_dir, profile_dir=profile_dir)

        result = CliRunner().invoke(app, ["pipeline", "run", "--profile", "build_vio"])

        assert result.exit_code == 0
        assert os.environ["REPO_ROOT"] == str(Path.cwd().resolve())
        assert os.environ["PIPELINE_PROFILE"] == "build_vio"
        assert len(calls["build"]) == 1
        assert len(calls["run"]) == 1
        assert calls["build"][0]["dataflow_path"] == str(calls["run"][0]["dataflow_path"])
        assert Path(str(calls["build"][0]["dataflow_path"])).name.endswith(".runtime.yml")
        assert calls["build"][0]["uv"] is True
        assert calls["run"][0]["uv"] is True
        assert calls["run"][0]["stop_on_completed"] is False

    def test_run_dora_dataflow_until_completed_stops_process_from_manifest(
        self,
        monkeypatch,
        tmp_path: Path,
    ) -> None:
        """Managed dora run should stop the process when the run manifest is completed."""
        state_file = tmp_path / "pipeline" / "out" / "current-run.json"
        state_file.parent.mkdir(parents=True)
        state_file.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
        dataflow_path = tmp_path / "dataflow.yml"
        dataflow_path.write_text("nodes: []", encoding="utf-8")

        class FakeProcess:
            pid = 1234

            def __init__(self) -> None:
                self.wait_calls = 0

            def poll(self):
                return None

            def wait(self, timeout=None):
                self.wait_calls += 1
                return 0

        process = FakeProcess()
        popen_calls = []
        killpg_calls = []
        monkeypatch.setattr(
            "pipeline.cli.subprocess.Popen",
            lambda command, **kwargs: popen_calls.append((command, kwargs)) or process,
        )
        monkeypatch.setattr("pipeline.cli.os.killpg", lambda pid, sig: killpg_calls.append((pid, sig)))

        def complete_after_first_poll(_seconds: float) -> None:
            state_file.write_text(json.dumps({"status": "completed"}), encoding="utf-8")

        monkeypatch.setattr("pipeline.cli.time.sleep", complete_after_first_poll)

        pipeline_cli._run_dora_dataflow_until_completed(  # noqa: SLF001 - unit-test CLI runner helper.
            dataflow_path,
            uv=True,
            state_file=state_file,
            poll_interval_seconds=0.0,
        )

        assert popen_calls == [
            (
                [
                    sys.executable,
                    "-c",
                    "import sys\nfrom dora import run\nrun(sys.argv[1], uv=sys.argv[2] == '1')",
                    str(dataflow_path),
                    "1",
                ],
                {"start_new_session": True},
            )
        ]
        assert killpg_calls == [(1234, pipeline_cli.signal.SIGINT)]
        assert process.wait_calls == 1


def _patch_profile_resolver(monkeypatch, *, dataset_dir: Path, profile_dir: Path | None = None) -> None:
    def factory() -> PipelineProfileResolver:
        return PipelineProfileResolver(dataset_dir=dataset_dir, profile_dir=profile_dir)

    monkeypatch.setattr("pipeline.cli.PipelineProfileResolver", factory)
