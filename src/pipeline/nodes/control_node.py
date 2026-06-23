from __future__ import annotations

import json
from enum import Enum
from queue import Empty as QueueEmptyException
from queue import Queue
from typing import Literal, cast

import pyarrow as pa
from dora import Node

from logger import spawn_logger
from pipeline.annotations import (
    Event as DoraEvent,  # noqa: TC001 - reactive get_type_hints resolves this at runtime.
)
from pipeline.decorators import on_input, on_stop, reactive
from pipeline.nodes.base import PipelineNode
from pipeline.runtime_config import ControlNodeRuntimeConfig
from pipeline.transport import ControlTransport, ControlTransportFactory, ControlTransportSettings
from pipeline.utils import BackgroundPipelineRunState, PipelineRunState

type CommandValue = str | None
type Command = str | None


class CommandTarget(Enum):
    """Command target."""

    DATASET = "ds"
    UNKNOWN = "unknown"


type CommandMessage = dict[Literal["target", "command", "value"], CommandTarget | Command | CommandValue]


@reactive
class ControlNode(PipelineNode):
    """Control node with pluggable external command ingress."""

    def __init__(
        self,
        node: Node | None = None,
        transport: ControlTransport | None = None,
        run_state: BackgroundPipelineRunState | PipelineRunState | None = None,
        config: ControlNodeRuntimeConfig | None = None,
    ) -> None:
        """Initialize the control node."""
        self.node: Node = node or Node()
        self.config = config or self.runtime_config_as(ControlNodeRuntimeConfig)

        self.signal_queue: Queue[CommandMessage] = Queue()
        self.logger = spawn_logger(app="control_node")
        self.run_state = run_state or BackgroundPipelineRunState()
        self.nodes_to_watch = set(self.config.expected_ready_nodes)
        self.ready_nodes = set()
        self.all_nodes_ready = False
        self.all_nodes_ready_announced = False
        self.autostart_sent = False
        self.run_completed = False

        self.command_transport = transport or ControlTransportFactory.create(
            ControlTransportSettings(
                transport=self.config.transport,
                http_host=self.config.http_host,
                http_port=self.config.http_port,
            ),
        )

        self.command_transport.on(self.enqueue_command_line)
        self.command_transport.start()
        self._write_run_state(status="running")

    @on_input("transport_tick")
    def handle_transport_tick(self) -> None:
        """Poll queued external commands and forward them into the dataflow."""
        try:
            while not self.signal_queue.empty():
                obj = self.signal_queue.get_nowait()
                target, command, value = obj["target"], obj["command"], obj["value"]
                target = cast("CommandTarget", target)
                if target == CommandTarget.UNKNOWN or command is None:
                    self.logger.warning(f"Ignoring invalid control command: {obj}")
                    continue
                array = pa.array([command, value]) if value is not None else pa.array([command])
                self.node.send_output(target.value, array)
                self.logger.info(f"Target: {target}, Command: {command}, Value: {value}")
        except QueueEmptyException:
            pass

    @on_input("startup_tick")
    def handle_startup_tick(self) -> None:
        """Handle startup tick."""
        self.all_nodes_ready = self.nodes_to_watch.issubset(self.ready_nodes)
        if not self.all_nodes_ready:
            return

        self._announce_all_nodes_ready()

        if self.autostart_sent:
            return

        autostart_configured = self.config.autostart_after_ready and self.config.fraction is not None
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

    def _announce_all_nodes_ready(self) -> None:
        if self.all_nodes_ready_announced:
            return
        self.all_nodes_ready_announced = True
        self.logger.info(f"Nodes are ready, transport type: {self.config.transport}")

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
        self.logger.info("Control node stopped")

    def _close_runtime_resources(self) -> None:
        self._close_run_state()
        self.command_transport.close()

    def _write_run_state(self, *, status: str) -> None:
        try:
            self.run_state.write(status=status, node=self.node)
        except OSError as exc:
            self.logger.warning(f"Could not write pipeline run state: {exc}")

    def _close_run_state(self) -> None:
        close = getattr(self.run_state, "close", None)
        if callable(close):
            close(timeout=0.25)

    def enqueue_command_line(self, line: str) -> dict[str, object]:
        """Parse and enqueue a command line from an external transport."""
        normalized_line = line.strip().lower()
        if not normalized_line:
            msg = "empty control command"
            raise ValueError(msg)

        target, command, value = self.parse_command(normalized_line)
        if target == CommandTarget.UNKNOWN or command is None:
            msg = f"invalid control command: {line}"
            raise ValueError(msg)
        self.signal_queue.put({"target": target, "command": command, "value": value})
        return {
            "target": target.value,
            "command": command,
            "value": value,
        }

    def parse_command(self, line: str) -> tuple[CommandTarget, Command, CommandValue]:
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
    ControlNode(config=ControlNode.runtime_config_as(ControlNodeRuntimeConfig)).run()
