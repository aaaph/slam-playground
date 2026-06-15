from pathlib import Path

import pytest
from yaml import safe_dump


@pytest.fixture
def euroc_mh_01_dataset_dir(tmp_path: Path) -> Path:
    """Create a minimal test registry entry for profiles that select euroc_mh_01."""
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    manifest = {
        "name": "euroc_mh_01",
        "type": "euroc",
        "root": "datasets/euroc_mh_01",
        "rig": "config/dataset_rig/euroc.yaml",
        "streams": {
            "cam0": "cam0/data.csv",
            "cam1": "cam1/data.csv",
            "imu0": "imu0/data.csv",
            "ground_truth": "state_groundtruth_estimate0/data.csv",
        },
    }
    (dataset_dir / "euroc_mh_01.yaml").write_text(safe_dump(manifest), encoding="utf-8")
    return dataset_dir
