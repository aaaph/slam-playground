import json
from pathlib import Path
from typing import cast

import pyarrow as pa
import pytest
from dora import Node
from zenoh import Session

from pipeline.nodes.zenoh_control_node import (
    CommandTarget,
    ZenohControlNode,
)
from pipeline.runtime_config import ControlNodeRuntimeConfig
from pipeline.utils import BackgroundPipelineRunState, PipelineRunState


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
            [{"type": "INPUT", "id": "transport_tick", "value": pa.array([0])}, {"type": "STOP"}]
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

    def test_nodes_to_watch_from_control_config(self, mock_deps: tuple[Node, Session], mocker) -> None:
        """Control node should use expected ready nodes from runtime config."""
        mock_node, mock_session = mock_deps
        run_state = mocker.MagicMock(spec=PipelineRunState)
        config = ControlNodeRuntimeConfig(
            node_id="control",
            expected_ready_nodes=["dataset", "rerun"],
        )

        node = ZenohControlNode(node=mock_node, session=mock_session, run_state=run_state, config=config)

        assert node.nodes_to_watch == {"dataset", "rerun"}

    def test_handle_status_uses_ready_input_mapping(self, mock_deps: tuple[Node, Session], mocker) -> None:
        """Control node should map generated status input ids back to dataflow node ids."""
        mock_node, mock_session = mock_deps
        run_state = mocker.MagicMock(spec=PipelineRunState)
        config = ControlNodeRuntimeConfig(
            node_id="control",
            expected_ready_nodes=["frontend"],
            ready_inputs={"frontend_status": "frontend"},
        )
        node = ZenohControlNode(node=mock_node, session=mock_session, run_state=run_state, config=config)
        event = {
            "type": "INPUT",
            "id": "frontend_status",
            "value": pa.array([json.dumps({"node": "FrontendNode", "state": "ready"})]),
        }

        node.handle_status(event)

        assert node.ready_nodes == {"frontend"}

    def test_dataset_done_status_marks_pipeline_completed(
        self,
        mock_deps: tuple[Node, Session],
        mocker,
    ) -> None:
        """Control node should mark completion and leave stopping to the CLI runner."""
        mock_node, mock_session = mock_deps
        cast("mocker.MagicMock", mock_node).dataflow_id.return_value = "df-123"
        run_state = mocker.MagicMock(spec=PipelineRunState)
        config = ControlNodeRuntimeConfig(
            node_id="control",
            stop_after_dataset_done=True,
            ready_inputs={"dataset_status": "dataset"},
        )
        node = ZenohControlNode(node=mock_node, session=mock_session, run_state=run_state, config=config)
        event = {
            "type": "INPUT",
            "id": "dataset_status",
            "value": pa.array([json.dumps({"node": "dataset", "state": "done", "reason": "steps_done"})]),
        }

        node.handle_status(event)
        node.handle_status(event)

        assert node.run_completed is True
        run_state.write.assert_any_call(status="completed", node=mock_node)
        assert len([call for call in run_state.write.call_args_list if call.kwargs["status"] == "completed"]) == 1
        cast("mocker.MagicMock", node.sub).undeclare.assert_not_called()
        cast("mocker.MagicMock", mock_session).close.assert_not_called()

    def test_graceful_shutdown_preserves_completed_status(
        self,
        mock_deps: tuple[Node, Session],
        mocker,
    ) -> None:
        """Control shutdown should not downgrade completed runs to stopped."""
        mock_node, mock_session = mock_deps
        run_state = mocker.MagicMock(spec=PipelineRunState)
        node = ZenohControlNode(node=mock_node, session=mock_session, run_state=run_state)
        node.run_completed = True

        node.graceful_shutdown()

        run_state.write.assert_any_call(status="completed", node=mock_node)

    def test_startup_tick_syncs_all_nodes_ready(self, mock_deps: tuple[Node, Session], mocker) -> None:
        """Startup tick should materialize whether all expected nodes are ready."""
        mock_node, mock_session = mock_deps
        run_state = mocker.MagicMock(spec=PipelineRunState)
        config = ControlNodeRuntimeConfig(
            node_id="control",
            expected_ready_nodes=["dataset", "rerun"],
        )
        node = ZenohControlNode(node=mock_node, session=mock_session, run_state=run_state, config=config)

        node.ready_nodes.add("dataset")
        node.handle_startup_tick()
        assert node.all_nodes_ready is False

        node.ready_nodes.add("rerun")
        node.handle_startup_tick()
        assert node.all_nodes_ready is True

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
