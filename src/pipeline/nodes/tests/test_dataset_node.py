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
        node.frame_id = 0
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
