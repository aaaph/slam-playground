from pathlib import Path

import pytest
from yaml import safe_dump


@pytest.fixture
def euroc_mh_01_dataset_dir(tmp_path: Path) -> Path:
    """Create minimal test registry entries for EuRoC profiles."""
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir()
    for dataset_name in ("euroc_mh_01", "euroc_v101"):
        manifest = {
            "name": dataset_name,
            "type": "euroc",
            "root": f"datasets/{dataset_name}",
            "rig": "config/dataset_rig/euroc.yaml",
            "streams": {
                "cam0": "cam0/data.csv",
                "cam1": "cam1/data.csv",
                "imu0": "imu0/data.csv",
                "ground_truth": "state_groundtruth_estimate0/data.csv",
            },
        }
        (dataset_dir / f"{dataset_name}.yaml").write_text(safe_dump(manifest), encoding="utf-8")
    return dataset_dir
