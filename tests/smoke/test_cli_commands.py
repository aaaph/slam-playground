"""Command-level smoke tests for user-facing CLI recipes."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from yaml import safe_load

COMMAND_TIMEOUT_SECONDS = 30


def run_just_command(*args: str) -> subprocess.CompletedProcess[str]:
    """Run a repository `just` command and return the completed process."""
    just = shutil.which("just")
    if just is None:
        pytest.fail("`just` executable is required for CLI smoke tests")

    return subprocess.run(  # noqa: S603 - smoke tests intentionally exercise the real CLI command.
        [just, *args],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )


def assert_command_succeeded(result: subprocess.CompletedProcess[str]) -> None:
    """Assert that a command succeeded with useful diagnostics on failure."""
    assert result.returncode == 0, (
        f"Command failed with exit code {result.returncode}: {' '.join(result.args)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.smoke
def test_dataset_list_reports_euroc_smoke_as_verified() -> None:
    """The dataset list command should see the committed smoke fixture as usable."""
    result = run_just_command("dataset", "list", "--format", "yaml")

    assert_command_succeeded(result)
    datasets = safe_load(result.stdout)
    datasets_by_name = {str(item["name"]): item for item in datasets}
    smoke = datasets_by_name["euroc_smoke"]

    assert smoke["type"] == "euroc"
    assert smoke["root"] == "datasets/euroc_smoke"
    assert smoke["local"]["exists"] is True
    assert smoke["local"]["verified"] is True
    assert smoke["local"]["issues"] == []


@pytest.mark.smoke
def test_profile_resolve_accepts_euroc_smoke_dataset_override() -> None:
    """The profile CLI should resolve the smoke dataset into a runnable pipeline config."""
    result = run_just_command(
        "profile",
        "resolve",
        "--profile",
        "slam_agent_profile",
        "--dataset",
        "euroc_smoke",
        "--fraction",
        "1",
    )

    assert_command_succeeded(result)
    resolved = safe_load(result.stdout)
    runtime_nodes = resolved["dataflow"]["runtime_dataflow"]["nodes"]
    node_ids = {str(node["id"]) for node in runtime_nodes}
    control_node = next(node for node in runtime_nodes if node["id"] == "control")
    control_config = json.loads(control_node["env"]["PIPELINE_NODE_CONFIG"])

    assert resolved["dataset"]["name"] == "euroc_smoke"
    assert resolved["dataset"]["root"] == "datasets/euroc_smoke"
    assert resolved["run"]["mode"] == "batch_fraction"
    assert resolved["run"]["fraction"] == 1.0
    assert {"control", "dataset", "frontend", "rerun"}.issubset(node_ids)
    assert control_config["dataset_name"] == "euroc_smoke"
    assert control_config["fraction"] == 1.0
