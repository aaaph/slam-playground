from collections.abc import Iterator

from datasets import Dataset, Image, IterableDataset

row_example: dict[str, list[str | None] | list[dict[str, list[float]]]] = {
    "image": ["./datasets/euroc_v_01_easy/cam0/data/1403715273262142976.png", None],
    "imu": [{"acc": [123], "gyro": [123]}, {"acc": [123], "gyro": [123]}],
}


def _gen() -> Iterator[dict]:
    yield row_example


ds: Dataset = Dataset.from_dict(row_example).cast_column("image", Image())

iterable: IterableDataset = ds.to_iterable_dataset()
