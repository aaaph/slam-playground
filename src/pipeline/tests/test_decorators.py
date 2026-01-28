from collections.abc import Callable
from unittest.mock import patch

from logger import spawn_logger
from pipeline.decorators import on_input, on_stop, reactive


class TestDecorators:
    """Test decorators."""

    def test_on_input(self) -> None:
        """Test on_input decorator."""

        @on_input("test")
        def test_function() -> None:
            """Test function."""

        assert test_function.dora_input_id == "test"

    def test_on_stop(self) -> None:
        """Test on_stop decorator."""

        @on_stop
        def test_function() -> None:
            """Test function."""

        assert isinstance(test_function.dora_stop_id, str)

    def test_reactive(self) -> None:
        """Test reactive decorator."""

        @reactive
        class TestNode:
            """Test node."""

            def run(self) -> None: ...
            def __init__(self) -> None:
                """Initialize the test node."""
                self.logger = spawn_logger(app="test_node")
                self.flag = False
                self.array = []
                self.stopped = False

            @on_input("test")
            def handle_test(self, data) -> None:
                self.flag = True
                self.array.append(data)

            @on_stop
            def graceful_shutdown(self) -> None:
                self.stopped = True

        assert isinstance(TestNode.__init__, Callable)

        with patch("pipeline.decorators.Node") as mocked_node:
            mocked_node.return_value = [{"type": "INPUT", "id": "test", "value": 1}]
            node = TestNode()
            node.run()
            assert node.flag
            assert node.array == [1]
            assert node.stopped is False

        with patch("pipeline.decorators.Node") as mocked_node:
            mocked_node.return_value = [{"type": "INPUT", "id": "test", "value": 1}, {"type": "STOP"}]
            node = TestNode()
            node.run()
            assert node.flag
            assert node.array == [1]
            assert node.stopped is True
