from typing import cast

import pyarrow as pa
import pytest
from dora import Node
from zenoh import Session

from pipeline_nodes.zenoh_control_node import ZenohControlNode


class TestZenohControlNode:
    """Unit tests for Zenoh control node."""

    @pytest.fixture
    def mock_deps(self, mocker) -> tuple[Node, Session]:
        """Mock dependencies."""
        mock_node = mocker.MagicMock(spec=Node)
        mock_session = mocker.MagicMock(spec=Session)
        mock_sub = mocker.MagicMock()
        mock_session.declare_subscriber.return_value = mock_sub

        return cast("Node", mock_node), cast("Session", mock_session)

    def test_command_propagation(self, mock_deps: tuple[Node, Session], mocker):
        """Test command propagation."""
        mock_node, mock_session = mock_deps

        cast("mocker.MagicMock", mock_node).__iter__.return_value = iter(
            [{"type": "INPUT", "id": "tick", "value": pa.array([0])}, {"type": "STOP"}]
        )
        node = ZenohControlNode(node=mock_node, session=mock_session)
        node.signal_queue.put("start_dataset")
        node.run()
        cast("mocker.MagicMock", mock_node).send_output.assert_any_call("start_dataset", pa.array([True]))
        assert node.signal_queue.empty()

    def test_graceful_shutdown(self, mock_deps: tuple[Node, Session], mocker):
        """Test graceful shutdown."""
        mock_node, mock_session = mock_deps
        node = ZenohControlNode(node=mock_node, session=mock_session)
        node.graceful_shutdown()
        cast("mocker.MagicMock", mock_session).close.assert_called_once()
        cast("mocker.MagicMock", node.sub).undeclare.assert_called_once()
