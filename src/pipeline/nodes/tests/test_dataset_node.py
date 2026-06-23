import json

import pyarrow as pa
import pytest

from pipeline.annotations import SYNC_EXECUTION_START_TIME_NS_METADATA_FIELD
from pipeline.context import PipelineContext
from pipeline.nodes.dataset_node import DatasetNode, StepStrategy


class TestDatasetNode:
    """Test dataset node."""

    @pytest.fixture
    def dataset_node(self, mocker) -> DatasetNode:
        """Create a minimal initialized DatasetNode without dataset iteration."""
        node = DatasetNode.__new__(DatasetNode)
        node.node = mocker.Mock()
        node.node.dataflow_id.return_value = "test-dataflow"
        node.node_name = "dataset"
        node.logger = mocker.Mock()
        node.frame_id = 0
        node.dataset_done_sent = False
        return node

    def test_parse_step_value(self, dataset_node: DatasetNode) -> None:
        """Test that the step value is parsed correctly."""
        increment, strategy = dataset_node.parse_step_value("1")
        assert increment == 1
        assert strategy == StepStrategy.INCREMENT

        increment, strategy = dataset_node.parse_step_value("2s")
        assert increment == 2
        assert strategy == StepStrategy.SECONDS

        increment, strategy = dataset_node.parse_step_value("25%")
        assert increment == 25
        assert strategy == StepStrategy.PERCENTAGE

        increment, strategy = dataset_node.parse_step_value("dfdfsdfsfsd")
        assert increment == 0
        assert strategy == StepStrategy.NOT_DEFINED

    def test_handle_tick_records_sync_execution_start_time(
        self,
        dataset_node: DatasetNode,
        mocker,
    ) -> None:
        """Test that output ticks record synchronous execution start time in metadata."""
        dataset_node.state = "PLAYING"
        dataset_node._create_next_item = mocker.Mock(return_value=PipelineContext.from_timestamp(1.0))  # noqa: SLF001
        mocker.patch("pipeline.nodes.dataset_node.time.perf_counter_ns", return_value=123)

        metadata = {}
        result = dataset_node.handle_tick(metadata)

        assert result is not None
        assert metadata[SYNC_EXECUTION_START_TIME_NS_METADATA_FIELD] == 123
        assert metadata["trace_id"] == "0"
        assert metadata["dataflow_id"] == "test-dataflow"
        assert dataset_node.frame_id == 1

    def test_stepping_done_sends_one_dataset_done_status(
        self,
        dataset_node: DatasetNode,
    ) -> None:
        """Dataset node should emit one done status after a bounded step run."""
        dataset_node.state = "STEPPING"
        dataset_node.remaining_steps = 0

        result = dataset_node.handle_tick({})
        dataset_node.handle_tick({})

        assert result is None
        assert dataset_node.state == "PAUSED"
        status_outputs = [
            call.args for call in dataset_node.node.send_output.call_args_list if call.args[0] == "status"
        ]
        assert len(status_outputs) == 2
        paused_payload = json.loads(status_outputs[0][1][0].as_py())
        done_payload = json.loads(status_outputs[1][1][0].as_py())
        assert paused_payload == {"node": "dataset", "state": "PAUSED"}
        assert done_payload == {"node": "dataset", "reason": "steps_done", "state": "done"}

    def test_handle_control_percentage_sets_remaining_steps(
        self,
        dataset_node: DatasetNode,
    ) -> None:
        """Percentage steps should materialize against total dataset rows."""
        dataset_node.state = "IDLE"
        dataset_node.total_items = 200
        event = {"value": pa.array(["step", "5%"])}

        dataset_node.handle_control_start(event)

        assert dataset_node.state == "STEPPING"
        assert dataset_node.remaining_steps == 10

    def test_handle_control_stop_pauses_dataset(
        self,
        dataset_node: DatasetNode,
    ) -> None:
        """Stop control command should pause dataset playback."""
        dataset_node.state = "PLAYING"
        event = {"value": pa.array(["stop"])}

        dataset_node.handle_control_start(event)

        assert dataset_node.state == "PAUSED"
        status_payload = json.loads(dataset_node.node.send_output.call_args.args[1][0].as_py())
        assert status_payload == {"node": "dataset", "state": "PAUSED"}
