import os
import time
from enum import Enum, auto
from pathlib import Path
from typing import Literal, cast

import numpy as np
from dora import Node

from dataset.euroc import EurocDataset, decode_stereo_pair
from dataset.registry import DatasetRegistry
from datasets import Dataset
from logger import bind_trace_id, spawn_logger
from pipeline.annotations import (
    SYNC_EXECUTION_START_TIME_NS_METADATA_FIELD,
    Event,
    Metadata,
)
from pipeline.context import PipelineContext
from pipeline.decorators import on_input, reactive, to_output

type StepValue = int


class StepStrategy(Enum):
    """Step strategy."""

    INCREMENT = auto()
    SECONDS = auto()
    PERCENTAGE = auto()
    NOT_DEFINED = auto()


@reactive
class DatasetNode:
    """Dataset node."""

    def run(self) -> None: ...  # noqa: D102

    def __init__(self, ds: Dataset, node: Node | None = None) -> None:
        """Initialize the dataset node."""
        self.node: Node = node or Node()
        self.logger = spawn_logger(app="dataset_node")
        self.state: Literal["PAUSED", "PLAYING", "DONE", "IDLE", "STEPPING"] = cast(
            "Literal['PAUSED']", os.getenv("INITIAL_STATE", "PAUSED")
        )
        self.remaining_steps: int = 0
        self.ds = ds
        self.ds_iter = iter(ds.to_iterable_dataset())
        self.logger.info(f"Initial state: {self.state}")
        self.frame_id = 0

    def _bind_next_trace_id(self, metadata: Metadata) -> None:
        """Bind and advance the dataset frame trace id."""
        bind_trace_id(metadata, self.frame_id)
        self.frame_id += 1

    def _bind_dataflow_id(self, metadata: Metadata) -> None:
        """Bind dora dataflow id."""
        metadata["dataflow_id"] = self.node.dataflow_id()

    def _create_next_item(self, step: int | None = None) -> PipelineContext | None:
        """Send the next item from the dataset."""
        try:
            data = next(self.ds_iter)
        except StopIteration:
            self.logger.info(f"Dataset done.... {self.state} -> DONE")
            self.state = "DONE"
            return None

        timestamp = float(data["timestamp"])
        gyro_data = np.asarray(data["gyro_data"], dtype=np.float32)
        acc_data = np.asarray(data["acc_data"], dtype=np.float32)
        imu_ts = np.asarray(data["imu_ts"], dtype=np.int64)
        imu_rows = len(imu_ts)

        msg = (
            f"[{step + 1}] Sending next item: {timestamp:.0f}"
            if step is not None
            else f"Sending next item: {timestamp:.0f}"
        )

        self.logger.debug(msg)
        pipeline = PipelineContext.from_timestamp(timestamp)
        pipeline = pipeline.set_ndarray("gyro", gyro_data)
        pipeline = pipeline.set_ndarray("accel", acc_data)
        pipeline = pipeline.set_ndarray("imu_ts", imu_ts)
        pipeline = pipeline.set_scalar("imu_rows", imu_rows)
        pipeline = pipeline.set_ndarray("column_ts", np.array([timestamp]))

        left, right = decode_stereo_pair(data["stereo"])
        width = left.shape[1]
        height = left.shape[0]

        pipeline.set_image("left", left)
        pipeline.set_image("right", right)
        pipeline.set_scalar("width", width)
        pipeline.set_scalar("height", height)

        return pipeline.reassemble()

    @on_input("control")
    def handle_control_start(self, event: Event) -> None:
        """Handle the control start event."""
        prev_state = self.state
        arrow = event.get("value")
        command = arrow[0].as_py() if arrow is not None else None
        value = arrow[1].as_py() if arrow is not None and len(arrow) > 1 else None
        self.logger.trace(f"Control event: {command} {value}")
        if command == "start":
            next_state = "PLAYING"
        if command == "pause":
            next_state = "PAUSED"
        if command == "step":
            next_state = "STEPPING"
            if value is not None:
                steps, strategy = self.parse_step_value(value)
            else:
                steps, strategy = 1, StepStrategy.INCREMENT

            if strategy == StepStrategy.INCREMENT:
                self.logger.trace(f"Setting remaining steps to {steps}")
                self.remaining_steps = steps
            else:
                self.logger.warning(f"Strategy {strategy} not implemented")

        self.state = next_state
        self.logger.trace(f"Dataset state changed: {prev_state} -> {next_state}")

    @on_input("tick")
    @to_output("sensor_frame")
    def handle_tick(self, metadata: Metadata) -> PipelineContext | None:
        """Handle the tick event."""
        if self.state == "PLAYING":
            metadata[SYNC_EXECUTION_START_TIME_NS_METADATA_FIELD] = time.perf_counter_ns()
            self._bind_next_trace_id(metadata)
            self._bind_dataflow_id(metadata)
            next_item = self._create_next_item()
            if next_item is not None:
                metadata["timestamp_ns"] = next_item.get_scalar("timestamp")
                return next_item
        if self.state == "STEPPING":
            if self.remaining_steps <= 0:
                self.state = "PAUSED"
                self.logger.trace("Stepping done.... STEPPING -> PAUSED")
                return None
            self.remaining_steps -= 1
            metadata[SYNC_EXECUTION_START_TIME_NS_METADATA_FIELD] = time.perf_counter_ns()
            self._bind_next_trace_id(metadata)
            self._bind_dataflow_id(metadata)
            next_item = self._create_next_item(self.remaining_steps)
            if next_item is not None:
                metadata["timestamp_ns"] = next_item.get_scalar("timestamp")
                return next_item
        return None

    def parse_step_value(self, value: str) -> tuple[StepValue, StepStrategy]:
        """Parse the step value."""
        if value.isdigit():
            return int(value), StepStrategy.INCREMENT
        if value.endswith("s"):
            return int(value[:-1]), StepStrategy.SECONDS
        if value.endswith("%"):
            return int(value[:-1]), StepStrategy.PERCENTAGE
        return 0, StepStrategy.NOT_DEFINED


if __name__ == "__main__":
    dataset_name = os.getenv("DATASET_NAME")
    repo_root = os.getenv("REPO_ROOT")
    if dataset_name is None or repo_root is None:
        raise ValueError("DATASET_NAME or REPO_ROOT is not set")
    dataset_registry = DatasetRegistry(repo_root=Path(repo_root))
    manifest = dataset_registry.resolve(dataset_name)

    DatasetNode(EurocDataset.mh_01_easy().imu_and_stereo(decode_images=False)).run()
