import os
from typing import Literal

import numpy as np

from dataset.euroc import EurocDataset
from logger import node_logger
from pipeline.context import PipelineContext
from pipeline.decorators import on_input, reactive, to_output


@reactive
class DatasetNode:
    """Dataset node."""

    def run(self) -> None: ...  # noqa: D102

    def __init__(self) -> None:
        """Initialize the dataset node."""
        self.logger = node_logger(app="dataset_node")
        self.state: Literal["PAUSED", "PLAYING", "DONE", "IDLE"] = os.getenv("INITIAL_STATE", "PAUSED")

        self.euroc_dataset = EurocDataset.mh_01_easy()
        self.stereo = self.euroc_dataset.stereo()
        self.stereo_iterator = iter(self.stereo.to_iterable_dataset())
        self.logger.info(f"Initial state: {self.state}")

    def _send_next(self) -> PipelineContext | None:
        """Send the next item from the dataset."""
        try:
            stereo_data = next(self.stereo_iterator)
            timestamp = float(stereo_data["timestamp"])
            left = np.asarray(stereo_data["stereo"][0], dtype=np.uint8)
            right = np.asarray(stereo_data["stereo"][1], dtype=np.uint8)
            width = left.shape[1]
            height = left.shape[0]
            self.logger.trace(f"Sending next item: {timestamp:.0f}")
            return (
                PipelineContext.from_timestamp(timestamp)
                .set_image("left", left)
                .set_image("right", right)
                .set_scalar("width", width)
                .set_scalar("height", height)
                .reassemble()
            )
        except StopIteration:
            self.logger.info(f"Dataset done.... {self.state} -> DONE")
            self.state = "DONE"
            return None

    @on_input("control_start")
    def handle_control_start(self) -> None:
        """Handle the control start event."""
        self.logger.info(f"Control start event received: {self.state} -> PLAYING")
        self.state = "PLAYING"

    @on_input("control_pause")
    def handle_control_pause(self) -> None:
        """Handle the control pause event."""
        self.logger.info(f"Control pause event received: {self.state} -> PAUSED")
        self.state = "PAUSED"

    @on_input("control_step")
    @to_output("ctx")
    def handle_step(self) -> None:
        """Handle the step event."""
        if self.state not in ["PAUSED", "IDLE"]:
            return None
        return self._send_next()

    @on_input("tick")
    @to_output("ctx")
    def handle_tick(self) -> PipelineContext | None:
        """Handle the tick event."""
        if self.state != "PLAYING":
            return None
        return self._send_next()


if __name__ == "__main__":
    DatasetNode().run()
