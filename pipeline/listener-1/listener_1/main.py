"""Module to process node input events and print received messages."""

from dora import Node
from logger import spawn_logger


def main():
    """Listen for input events and print received messages."""
    logger = spawn_logger(app="listener_1")
    node = Node()
    for event in node:
        if event["type"] == "INPUT":
            message = event["value"][0].as_py()
            logger.info(f"""I heard {message} from {event["id"]}""")


if __name__ == "__main__":
    main()
