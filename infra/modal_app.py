# ruff: noqa: INP001
"""
Modal entrypoints for running the pipeline without keeping datasets locally.

Anyone who clones this repo can run these against their own Modal workspace:

    pip install modal && modal setup
    modal run infra/modal_app.py::fetch_dataset --dataset euroc_v101
    modal run infra/modal_app.py::fetch_artifact --artifact orb_vocabulary
    modal run infra/modal_app.py::run --profile full_batch_profile --dataset euroc_v101
    modal volume get slam-playground-output latest

Volumes are created on first use in the caller's own workspace, so there is no
shared state and no credentials beyond `modal setup`.
"""

import subprocess
import sys
from pathlib import Path

import modal

root = Path(__file__).resolve().parents[1]

# Payloads live in volumes so they survive between runs and never touch the
# caller's disk. The manifests that describe them stay baked into the image, so
# they always match the checked-out revision instead of drifting in a volume.
MANIFESTS = "/repo/manifests"

datasets = modal.Volume.from_name("slam-playground-datasets", create_if_missing=True)
artifacts = modal.Volume.from_name("slam-playground-artifacts", create_if_missing=True)
output = modal.Volume.from_name("slam-playground-output", create_if_missing=True)

volumes = {
    "/repo/datasets": datasets,
    "/repo/artifacts": artifacts,
    "/repo/pipeline/out": output,
}

ignore = [
    "**",
    "!README.md",
    "!pyproject.toml",
    "!uv.lock",
    "!config/**",
    "!pipeline/**",
    "!src/**",
    "**/.DS_Store",
    "**/.uv_cache/**",
    "**/__pycache__/**",
    "pipeline/.*.runtime.yml",
    "pipeline/out/**",
]

app = modal.App("slam-playground")

image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync(uv_project_dir=str(root), extra_options="--no-dev")
    .env(
        {
            "PYTHONPATH": "/repo/src",
            "UV_NO_SYNC": "1",
            "UV_PROJECT_ENVIRONMENT": "/.uv/.venv",
        }
    )
    .workdir("/repo")
    .add_local_dir(root, "/repo", copy=True, ignore=ignore)
    .add_local_dir(root / "datasets", f"{MANIFESTS}/datasets", copy=True, ignore=["**", "!*.yaml"])
    .add_local_dir(root / "artifacts", f"{MANIFESTS}/artifacts", copy=True, ignore=["**", "!*.yaml"])
)


def _cli(module: str, *args: str) -> None:
    subprocess.run([sys.executable, "-m", module, *args], cwd="/repo", check=True)  # noqa: S603 - repo CLI module.


@app.function(image=image, cpu=4, memory=8192, timeout=7200, volumes=volumes)
def fetch_dataset(dataset: str, force: bool = False) -> str:  # noqa: FBT001, FBT002
    """Download one dataset into the datasets volume. Idempotent."""
    _cli(
        "dataset.cli",
        "fetch",
        dataset,
        "--dataset-dir",
        f"{MANIFESTS}/datasets",
        "--yes",
        *(["--force"] if force else []),
    )
    datasets.commit()
    return dataset


@app.function(image=image, cpu=2, memory=4096, timeout=1800, volumes=volumes)
def fetch_artifact(artifact: str, force: bool = False) -> str:  # noqa: FBT001, FBT002
    """Download one artifact into the artifacts volume. Idempotent."""
    _cli(
        "artifacts.cli",
        "fetch",
        artifact,
        "--artifact-dir",
        f"{MANIFESTS}/artifacts",
        "--yes",
        *(["--force"] if force else []),
    )
    artifacts.commit()
    return artifact


@app.function(image=image, cpu=4, memory=8192, timeout=7200, volumes=volumes)
def run(
    profile: str = "slam_agent_profile",
    dataset: str = "euroc_smoke",
    dataflow: str = "vio-dataflow.yml",
    fraction: str = "1",
) -> str:
    """Run one pipeline profile against one dataset and keep the result."""
    _cli(
        "pipeline.cli",
        "pipeline",
        "run",
        "--profile",
        profile,
        "--dataset",
        dataset,
        "--dataflow",
        dataflow,
        "--visualization-sink",
        "file",
        "--fraction",
        fraction,
    )
    output.commit()
    return Path("/repo/pipeline/out/current-run.json").read_text()
