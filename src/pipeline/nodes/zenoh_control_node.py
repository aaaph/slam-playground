import json
import os
from enum import Enum
from queue import Empty as QueueEmptyException
from queue import Queue
from typing import Literal, cast

import pyarrow as pa
from dora import Node
from zenoh import Config, Sample, Session
from zenoh import open as zenoh_open

from logger import spawn_logger
from pipeline.annotations import Event as DoraEvent
from pipeline.decorators import on_input, on_stop, reactive
from pipeline.nodes.base import PipelineNode
from pipeline.runtime_config import ControlNodeRuntimeConfig
from pipeline.utils import BackgroundPipelineRunState, PipelineRunState

type CommndValue = str | None
type Command = str | None


class CommandTarget(Enum):
    """Command target."""

    DATASET = "ds"
    UNKNOWN = "unknown"


@reactive
class ZenohControlNode(PipelineNode):
    """Zenoh control node."""

    def __init__(
        self,
        node: Node | None = None,
        session: Session | None = None,
        run_state: BackgroundPipelineRunState | PipelineRunState | None = None,
        nodes_to_watch: set[str] | None = None,
        config: ControlNodeRuntimeConfig | None = None,
    ) -> None:
        """Initialize the zenoh control node."""
        self.node: Node = node or Node()
        self.config = config or self.runtime_config_as(ControlNodeRuntimeConfig)

        self.session: Session = session or zenoh_open(Config())
        self.signal_queue: Queue[
            dict[Literal["target", "command", "value"], CommandTarget | Command | CommndValue]
        ] = Queue()
        self.logger = spawn_logger(app="zenoh_control_node")
        self.run_state = run_state or BackgroundPipelineRunState()
        self.nodes_to_watch = (
            nodes_to_watch if nodes_to_watch is not None else set(self.config.expected_ready_nodes)
        )
        self.ready_nodes = set()
        self.all_nodes_ready = False
        self.autostart_sent = False
        self.run_completed = False

        def callback(data: Sample) -> None:
            line = data.payload.to_bytes().decode("utf-8").strip().lower()
            if line:
                self.logger.trace(f"Received command: {line}")
                target, command, value = self.parse_command(line)
                self.signal_queue.put({"target": target, "command": command, "value": value})

        self.sub = self.session.declare_subscriber("pipeline/control", callback)
        self.logger.info(f"Zenoh control node initialized: zid: {self.session.zid()}")
        self._write_run_state(status="running")

    @on_input("transport_tick")
    def handle_transport_tick(self) -> None:
        """Pooling of queue for commands from zenoh, if there are commands in the queue, send them to dataflow."""
        try:
            while not self.signal_queue.empty():
                obj = self.signal_queue.get_nowait()
                target, command, value = obj["target"], obj["command"], obj["value"]
                array = pa.array([command, value]) if value is not None else pa.array([command])
                target = cast("CommandTarget", target)
                self.node.send_output(target.value, array)
                self.logger.info(f"Target: {target}, Command: {command}, Value: {value}")
        except QueueEmptyException:
            pass

    @on_input("startup_tick")
    def handle_startup_tick(self) -> None:
        """Handle startup tick."""
        if self.autostart_sent:
            return

        self.all_nodes_ready = self.nodes_to_watch.issubset(self.ready_nodes)
        if not self.all_nodes_ready:
            return

        autostart_configured = self.config.autostart_after_ready is not None and self.config.fraction is not None
        if not autostart_configured:
            return

        self.autostart_sent = True
        fraction = self.config.fraction
        target = CommandTarget.DATASET
        command = "step"
        value = f"{int((fraction) * 100)}%"
        array = pa.array([command, value])
        target = cast("CommandTarget", target)
        self.node.send_output(target.value, array)
        self.logger.info(f"Autostart sent: {command} {value}")

    @on_input("*_status")
    def handle_status(self, event: DoraEvent) -> None:
        """Handle status updates."""
        arrow = event["value"]
        raw = arrow[0].as_py()

        payload = json.loads(raw)
        event_id = event["id"]
        node = self.config.ready_inputs.get(event_id, payload["node"])
        status = str(payload["state"]).lower()
        if status == "ready":
            self.ready_nodes.add(node)

        should_stop = node == "dataset" and self.config.stop_after_dataset_done and status == "done"
        if should_stop:
            self._complete_pipeline()

    def _complete_pipeline(self) -> None:
        if self.run_completed:
            return
        self.run_completed = True
        self.logger.info("Completing pipeline")
        self._write_run_state(status="completed")

    @on_stop
    def graceful_shutdown(self) -> None:
        """Graceful shutdown."""
        self._write_run_state(status="completed" if self.run_completed else "stopped")
        self._close_runtime_resources()
        self.logger.info("Zenoh control node stopped")

    def _close_runtime_resources(self) -> None:
        self._close_run_state()
        self.sub.undeclare()
        self.session.close()

    def _write_run_state(self, *, status: str) -> None:
        try:
            self.run_state.write(status=status, node=self.node)
        except OSError as exc:
            self.logger.warning(f"Could not write pipeline run state: {exc}")

    def _close_run_state(self) -> None:
        close = getattr(self.run_state, "close", None)
        if callable(close):
            close(timeout=0.25)

    def parse_command(self, line: str) -> tuple[CommandTarget, Command, CommndValue]:
        """Parse the command."""
        try:
            target_raw, command, *values = line.split(":")
        except ValueError:
            return CommandTarget.UNKNOWN, None, None
        try:
            target = CommandTarget(target_raw)
        except ValueError:
            target = CommandTarget.UNKNOWN
        if not values or values == [""]:
            values = None
        return target, command, ":".join(values) if values is not None else None


if __name__ == "__main__":
    import os

    control_config = ZenohControlNode.runtime_config_as(ControlNodeRuntimeConfig)
    raw_nodes_to_watch = os.getenv("PIPELINE_READY_NODES") or os.getenv("CONTROL_NODE_EXTECTING_NODES")
    nodes_to_watch = set(json.loads(raw_nodes_to_watch)) if raw_nodes_to_watch is not None else None
    ZenohControlNode(config=control_config, nodes_to_watch=nodes_to_watch).run()
