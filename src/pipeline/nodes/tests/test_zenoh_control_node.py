import json
from pathlib import Path
from typing import cast

import pyarrow as pa
import pytest
from dora import Node
from zenoh import Session

from pipeline.nodes.zenoh_control_node import (
    BackgroundPipelineRunState,
    CommandTarget,
    PipelineRunState,
    ZenohControlNode,
)


class DummyNode:
    """Small node double for run state serialization tests."""

    def dataflow_id(self) -> str:
        """Return a stable test dataflow id."""
        return "test-dataflow"

    def node_config(self) -> dict[str, str]:
        """Return a stable test node config."""
        return {"id": "control"}


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
        run_state = mocker.MagicMock(spec=PipelineRunState)
        node = ZenohControlNode(node=mock_node, session=mock_session, run_state=run_state)
        node.signal_queue.put({"target": CommandTarget.DATASET, "command": "start", "value": None})
        node.run()
        cast("mocker.MagicMock", mock_node).send_output.assert_any_call("ds", pa.array(["start"]))
        assert node.signal_queue.empty()
        run_state.write.assert_any_call(status="running", node=mock_node)
        run_state.write.assert_any_call(status="stopped", node=mock_node)

    def test_graceful_shutdown(self, mock_deps: tuple[Node, Session], mocker):
        """Test graceful shutdown."""
        mock_node, mock_session = mock_deps
        run_state = mocker.MagicMock(spec=PipelineRunState)
        node = ZenohControlNode(node=mock_node, session=mock_session, run_state=run_state)
        node.graceful_shutdown()
        cast("mocker.MagicMock", mock_session).close.assert_called_once()
        cast("mocker.MagicMock", node.sub).undeclare.assert_called_once()
        run_state.write.assert_any_call(status="running", node=mock_node)
        run_state.write.assert_any_call(status="stopped", node=mock_node)

    def test_parse_command_method(self, mock_deps: tuple[Node, Session], mocker):
        """Test command parsing."""
        mock_node, mock_session = mock_deps
        run_state = mocker.MagicMock(spec=PipelineRunState)
        node = ZenohControlNode(node=mock_node, session=mock_session, run_state=run_state)

        line = "ds:start_dataset"
        target, command, value = node.parse_command(line)
        assert target == CommandTarget.DATASET
        assert command == "start_dataset"
        assert value is None

        line = "12321312"
        target, command, value = node.parse_command(line)
        assert target == CommandTarget.UNKNOWN
        assert command is None
        assert value is None

        line = "ds:step:1"
        target, command, value = node.parse_command(line)
        assert target == CommandTarget.DATASET
        assert command == "step"
        assert value == "1"

        line = "ds:step"
        target, command, value = node.parse_command(line)
        assert target == CommandTarget.DATASET
        assert command == "step"
        assert value is None

        line = "ds:step:"
        target, command, value = node.parse_command(line)
        assert target == CommandTarget.DATASET
        assert command == "step"
        assert value is None

        line = "try:to:parse:this:command"
        target, command, value = node.parse_command(line)
        assert target == CommandTarget.UNKNOWN
        assert command == "to"
        assert value == "parse:this:command"


class TestPipelineRunState:
    """Unit tests for pipeline run state persistence."""

    def test_write_current_run_manifest_and_latest_link(self, tmp_path: Path) -> None:
        """Test that current-run state points to the newest log directory."""
        out_dir = tmp_path / "pipeline" / "out"
        stale_run_dir = out_dir / "019e6e5c-d033-746a-9bfd-358774c72bee"
        current_run_dir = out_dir / "019e6e61-d4ab-766e-a886-d44ab4e041cb"
        stale_run_dir.mkdir(parents=True)
        current_run_dir.mkdir()
        (current_run_dir / "log_control.txt").write_text("control log", encoding="utf-8")
        (current_run_dir / "log_frontend.txt").write_text("frontend log", encoding="utf-8")

        state = PipelineRunState(out_dir=out_dir, repo_root=tmp_path)
        state.write(status="running", node=cast("Node", DummyNode()))

        manifest = json.loads((out_dir / "current-run.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "running"
        assert manifest["dataflow_id"] == "test-dataflow"
        assert manifest["node_config"] == {"id": "control"}
        assert manifest["log_dir"] == "pipeline/out/019e6e61-d4ab-766e-a886-d44ab4e041cb"
        assert manifest["logs"] == {
            "control": "pipeline/out/019e6e61-d4ab-766e-a886-d44ab4e041cb/log_control.txt",
            "frontend": "pipeline/out/019e6e61-d4ab-766e-a886-d44ab4e041cb/log_frontend.txt",
        }
        assert (out_dir / "latest").is_symlink()
        assert (out_dir / "latest").readlink() == Path("019e6e61-d4ab-766e-a886-d44ab4e041cb")


class TestBackgroundPipelineRunState:
    """Unit tests for background pipeline run state persistence."""

    def test_write_runs_on_background_worker(self, tmp_path: Path) -> None:
        """Test that the background writer persists the manifest asynchronously."""
        out_dir = tmp_path / "pipeline" / "out"
        current_run_dir = out_dir / "019e6e61-d4ab-766e-a886-d44ab4e041cb"
        current_run_dir.mkdir(parents=True)
        (current_run_dir / "log_control.txt").write_text("control log", encoding="utf-8")
        state = PipelineRunState(out_dir=out_dir, repo_root=tmp_path)
        background_state = BackgroundPipelineRunState(state=state)

        done = background_state.write(status="running", node=cast("Node", DummyNode()))

        assert done.wait(timeout=1.0)
        manifest = json.loads((out_dir / "current-run.json").read_text(encoding="utf-8"))
        assert manifest["status"] == "running"
        assert manifest["dataflow_id"] == "test-dataflow"
        assert manifest["logs"] == {"control": "pipeline/out/019e6e61-d4ab-766e-a886-d44ab4e041cb/log_control.txt"}
        background_state.close(timeout=1.0)
