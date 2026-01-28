from queue import Empty as QueueEmptyException
from queue import Queue

import pyarrow as pa
from dora import Node
from zenoh import Config, Sample, Session
from zenoh import open as zenoh_open

from logger import node_logger
from pipeline.decorators import on_input, on_stop, reactive


@reactive
class ZenohControlNode:
    """Zenoh control node."""

    def __init__(self, node: Node | None = None, session: Session | None = None) -> None:
        """Initialize the zenoh control node."""
        self.node: Node = node or Node()
        self.session: Session = session or zenoh_open(Config())
        self.signal_queue: Queue[str] = Queue()
        self.logger = node_logger(app="zenoh_control_node")

        def callback(data: Sample) -> None:
            cmd = data.payload.to_bytes().decode("utf-8").strip().lower()
            if cmd:
                self.signal_queue.put(cmd)
            self.logger.debug(f"Received command: {cmd}")

        self.sub = self.session.declare_subscriber("pipeline/control", callback)

    def run(self) -> None: ...  # noqa: D102

    @on_input("tick")
    def handle_timer(self) -> None:
        """Pooling of queue for commands from zenoh, if there are commands in the queue, send them to dataflow."""
        try:
            while not self.signal_queue.empty():
                command = self.signal_queue.get_nowait()
                self.node.send_output(command, pa.array([True]))
                self.logger.debug(f"Sent command: {command}")
        except QueueEmptyException:
            pass

    @on_stop
    def graceful_shutdown(self) -> None:
        """Graceful shutdown."""
        self.sub.undeclare()
        self.session.close()
        self.logger.info("Zenoh control node stopped")


if __name__ == "__main__":
    ZenohControlNode().run()
