from enum import Enum
from queue import Empty as QueueEmptyException
from queue import Queue
from typing import Literal, cast

import pyarrow as pa
from dora import Node
from zenoh import Config, Sample, Session
from zenoh import open as zenoh_open

from logger import spawn_logger
from pipeline.decorators import on_input, on_stop, reactive

type CommndValue = str | None
type Command = str | None


class CommandTarget(Enum):
    """Command target."""

    DATASET = "ds"
    UNKNOWN = "unknown"


@reactive
class ZenohControlNode:
    """Zenoh control node."""

    def __init__(self, node: Node | None = None, session: Session | None = None) -> None:
        """Initialize the zenoh control node."""
        self.node: Node = node or Node()
        self.session: Session = session or zenoh_open(Config())
        self.signal_queue: Queue[
            dict[Literal["target", "command", "value"], CommandTarget | Command | CommndValue]
        ] = Queue()
        self.logger = spawn_logger(app="zenoh_control_node")

        def callback(data: Sample) -> None:
            line = data.payload.to_bytes().decode("utf-8").strip().lower()
            if line:
                target, command, value = self.parse_command(line)
                self.signal_queue.put({"target": target, "command": command, "value": value})

        self.sub = self.session.declare_subscriber("pipeline/control", callback)

    def run(self) -> None: ...  # noqa: D102

    @on_input("tick")
    def handle_timer(self) -> None:
        """Pooling of queue for commands from zenoh, if there are commands in the queue, send them to dataflow."""
        try:
            while not self.signal_queue.empty():
                obj = self.signal_queue.get_nowait()
                target, command, value = obj["target"], obj["command"], obj["value"]
                array = pa.array([command, value]) if value is not None else pa.array([command])
                target = cast("CommandTarget", target)
                self.node.send_output(target.value, array)
                self.logger.debug(f"Target: {target}, Command: {command}, Value: {value}")
        except QueueEmptyException:
            pass

    @on_stop
    def graceful_shutdown(self) -> None:
        """Graceful shutdown."""
        self.sub.undeclare()
        self.session.close()
        self.logger.info("Zenoh control node stopped")

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
    ZenohControlNode().run()
