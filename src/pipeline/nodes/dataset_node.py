import json
import os
import time
from enum import Enum, auto
from typing import Literal, cast

import numpy as np
import pyarrow as pa
from dora import Node

from dataset.euroc import decode_stereo_pair
from dataset.factory import DatasetFactory
from datasets import Dataset
from logger import bind_trace_id, spawn_logger
from pipeline.annotations import (
    SYNC_EXECUTION_START_TIME_NS_METADATA_FIELD,
    Event,
    Metadata,
)
from pipeline.context import PipelineContext
from pipeline.decorators import on_input, reactive, to_output
from pipeline.nodes.base import PipelineNode
from pipeline.runtime_config import DatasetNodeConfig

type StepValue = int


class StepStrategy(Enum):
    """Step strategy."""

    INCREMENT = auto()
    SECONDS = auto()
    PERCENTAGE = auto()
    NOT_DEFINED = auto()


type DatasetState = Literal["PAUSED", "PLAYING", "DONE", "IDLE", "STEPPING"]


@reactive
class DatasetNode(PipelineNode):
    """Dataset node."""

    def __init__(self, ds: Dataset, node: Node | None = None) -> None:
        """Initialize the dataset node."""
        self.node: Node = node or Node()
        self.node_name = self.runtime_config().node_id
        self.logger = spawn_logger(app="dataset_node")
        self.state: DatasetState = cast("Literal['PAUSED']", os.getenv("INITIAL_STATE", "PAUSED"))
        self.remaining_steps: int = 0
        self.ds = ds
        self.total_items = ds.num_rows
        self.ds_iter = iter(ds.to_iterable_dataset())
        self.logger.info(f"Initial state: {self.state}")
        self.frame_id = 0
        self.dataset_done_sent = False

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
        (
            pipeline.set_ndarray("gyro", gyro_data)
            .set_ndarray("accel", acc_data)
            .set_ndarray("imu_ts", imu_ts)
            .set_scalar("imu_rows", imu_rows)
            .set_ndarray("column_ts", np.array([timestamp]))
        )

        left, right = decode_stereo_pair(data["stereo"])
        width = left.shape[1]
        height = left.shape[0]

        (
            pipeline.set_image("left", left)
            .set_image("right", right)
            .set_scalar("width", width)
            .set_scalar("height", height)
        )

        return pipeline.reassemble()

    @on_input("control")
    def handle_control_start(self, event: Event) -> None:
        """Handle the control start event."""
        prev_state = self.state
        arrow = event.get("value")
        command = arrow[0].as_py() if arrow is not None else None
        value = arrow[1].as_py() if arrow is not None and len(arrow) > 1 else None
        self.logger.trace(f"Control event: {command} {value}")
        next_state = self.state
        if command == "start":
            next_state = "PLAYING"
            self.dataset_done_sent = False
        if command == "pause":
            next_state = "PAUSED"
        if command == "step":
            next_state = "STEPPING"
            self.dataset_done_sent = False
            if value is not None:
                steps, strategy = self.parse_step_value(value)
            else:
                steps, strategy = 1, StepStrategy.INCREMENT

            if strategy == StepStrategy.INCREMENT:
                self.logger.trace(f"Setting remaining steps to {steps}")
                self.remaining_steps = steps
            elif strategy == StepStrategy.PERCENTAGE:
                self.remaining_steps = int(self.total_items * steps / 100)
                self.logger.info(f"Setting remaining steps to {self.remaining_steps} (percentage: {steps}%)")
            else:
                self.logger.warning(f"Strategy {strategy} not implemented")

        self.set_status(next_state)
        self.logger.info(f"Dataset state changed: {prev_state} -> {next_state}")

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
            if self.state == "DONE":
                self.send_dataset_done(reason="dataset_exhausted")
        if self.state == "STEPPING":
            if self.remaining_steps <= 0:
                self.set_status("PAUSED")
                self.send_dataset_done(reason="steps_done")
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
            if self.state == "DONE":
                self.send_dataset_done(reason="dataset_exhausted")
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

    def set_status(self, state: DatasetState) -> None:
        """Set the status."""
        self.state = state
        self.send_status(self.state)

    def send_dataset_done(self, *, reason: str) -> None:
        """Send a one-shot dataset completion status."""
        if self.dataset_done_sent:
            return
        self.dataset_done_sent = True
        self.send_status("done", reason=reason)

    def send_status(self, state: str, *, reason: str | None = None) -> None:
        """Send a dataset status event."""
        payload = {
            "node": self.node_name,
            "state": state,
        }
        if reason is not None:
            payload["reason"] = reason
        self.node.send_output(
            "status",
            pa.array([json.dumps(payload, sort_keys=True)]),
            {},
        )


if __name__ == "__main__":
    runtime_config = DatasetNode.runtime_config_as(DatasetNodeConfig)
    DatasetNode(
        DatasetFactory(repo_root=runtime_config.repo_root)
        .load_vio_dataset(runtime_config.dataset_name)
        .imu_and_stereo(decode_images=False)
    ).run()
