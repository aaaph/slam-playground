from unittest.mock import Mock

import pyarrow as pa

from pipeline.annotations import EXECUTION_TIME_MS_METADATA_FIELD
from pipeline.context import PipelineContext
from pipeline.nodes.rerun_node import RerunNode


class TestRerunNode:
    """Test Rerun node."""

    def test_visualize_branch_materializes_execution_time_metadata(self) -> None:
        """RerunNode should store execution metadata in the context before visualization."""
        node = RerunNode.__new__(RerunNode)
        node.logger = Mock()
        node.vizualize = Mock()

        record_batch = pa.RecordBatch.from_pydict({"DatasetNode": [1.25]})
        node.visualize_branch("dataset_frame", PipelineContext.from_timestamp(1.0), record_batch)

        branch, sent_ctx = node.vizualize.send.call_args.args[0]
        assert branch == "dataset_frame"
        assert sent_ctx.get_record_batch(EXECUTION_TIME_MS_METADATA_FIELD).schema.names == ["DatasetNode"]

    def test_visualize_branch_accepts_empty_execution_time_metadata(self) -> None:
        """Empty execution metadata should still be a valid context field."""
        node = RerunNode.__new__(RerunNode)
        node.logger = Mock()
        node.vizualize = Mock()

        record_batch = pa.RecordBatch.from_arrays([], schema=pa.schema([]))
        node.visualize_branch("dataset_frame", PipelineContext.from_timestamp(1.0), record_batch)

        branch, sent_ctx = node.vizualize.send.call_args.args[0]
        assert branch == "dataset_frame"
        assert sent_ctx.get_record_batch(EXECUTION_TIME_MS_METADATA_FIELD).schema.names == []
