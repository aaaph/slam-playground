from pathlib import Path

from dataset.factory import DatasetFactory
from dataset.registry import DatasetRegistry
from logger import spawn_logger

logger = spawn_logger(app="example_load_dataset")

registry = DatasetRegistry(repo_root=Path.cwd())

manifest = registry.find("euroc_v101")

factory = DatasetFactory(repo_root=Path.cwd())

ds = factory.load_dataset("euroc_v101").ground_truth()

logger.info(f"Dataset features: {ds.features}")
logger.info(f"Dataset length: {len(ds)}")
logger.info(f"Dataset first item: {ds[0]}")
