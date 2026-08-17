import subprocess  # noqa: INP001
import sys
from pathlib import Path

import modal

root = Path(__file__).resolve().parents[1]
ignore = [
    "**",
    "!README.md",
    "!pyproject.toml",
    "!uv.lock",
    "!config/**",
    "!datasets/euroc_smoke/**",
    "!datasets/euroc_smoke.yaml",
    "!pipeline/**",
    "!src/**",
    "**/.DS_Store",
    "**/.uv_cache/**",
    "**/__pycache__/**",
    "pipeline/.*.runtime.yml",
    "pipeline/out/**",
]

app = modal.App("slam-playground")
output = modal.Volume.from_name("slam-playground-output", create_if_missing=True)

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
)


@app.function(
    image=image,
    cpu=4,
    memory=8192,
    timeout=3600,
    volumes={"/repo/pipeline/out": output},
)
def smoke() -> str:
    """Run the smoke VIO pipeline in Modal."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeline.cli",
            "pipeline",
            "run",
            "--profile",
            "slam_agent_profile",
            "--dataset",
            "euroc_smoke",
            "--dataflow",
            "vio-dataflow.yml",
            "--visualization-sink",
            "file",
            "--fraction",
            "1",
        ],
        cwd="/repo",
        check=True,
    )
    output.commit()
    return Path("/repo/pipeline/out/current-run.json").read_text()
