"""End-to-end SLAM regression test on EuRoC V1_01_easy."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from dataset.registry import DatasetRegistry
from evaluation.ape import evaluate_ape, format_ape_summary
from pipeline.result import PipelineResult

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "euroc_v101"
PROFILE_NAME = "slam_agent_profile"
DATASET_FRACTION = "0.05"
PIPELINE_TIMEOUT_SECONDS = 900

MIN_APE_SAMPLES = 20
MAX_RMSE_TRANSLATION_M = 0.30
MAX_MEAN_TRANSLATION_M = 0.20
MAX_TRANSLATION_M = 1.00


@pytest.mark.regression
def test_euroc_v101_pipeline_ape_regression() -> None:
    """Run the SLAM pipeline on EuRoC V1_01_easy and gate offline APE metrics."""
    _skip_if_dataset_is_unavailable(DATASET_NAME)

    pipeline_run = _run_pipeline(DATASET_NAME)
    assert pipeline_run.returncode == 0, _format_completed_process(pipeline_run)

    result = PipelineResult.current(repo_root=REPO_ROOT)
    assert result.require_dataset_name() == DATASET_NAME
    result.require_evo_rerun_inputs()

    artifacts = evaluate_ape(result, save_result=False)
    summary = format_ape_summary(artifacts)

    assert artifacts.samples_count >= MIN_APE_SAMPLES, summary
    assert artifacts.stats["rmse"] < MAX_RMSE_TRANSLATION_M, summary
    assert artifacts.stats["mean"] < MAX_MEAN_TRANSLATION_M, summary
    assert artifacts.stats["max"] < MAX_TRANSLATION_M, summary


def _skip_if_dataset_is_unavailable(dataset_name: str) -> None:
    registry = DatasetRegistry(repo_root=REPO_ROOT)
    manifest = registry.find(dataset_name)
    status = registry.local_status(manifest)
    if status.verified:
        return

    missing = ", ".join(str(path) for path in status.issues[:5])
    if len(status.issues) > 5:
        missing = f"{missing}, ..."
    pytest.skip(f"Dataset '{dataset_name}' is not locally available: {missing}")


def _run_pipeline(dataset_name: str) -> subprocess.CompletedProcess[str]:
    just = shutil.which("just")
    if just is None:
        pytest.fail("`just` executable is required for regression tests")

    return subprocess.run(  # noqa: S603 - regression test intentionally exercises the real pipeline CLI.
        [
            just,
            "pipeline",
            "run",
            "--profile",
            PROFILE_NAME,
            "--dataset",
            dataset_name,
            "--fraction",
            DATASET_FRACTION,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=PIPELINE_TIMEOUT_SECONDS,
        check=False,
    )


def _format_completed_process(result: subprocess.CompletedProcess[str]) -> str:
    return (
        f"Command failed with exit code {result.returncode}: {' '.join(result.args)}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
