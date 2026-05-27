import json
from collections.abc import Callable
from unittest.mock import patch

from logger import current_trace_id, spawn_logger
from pipeline.annotations import (
    SYNC_EXECUTION_START_TIME_NS_METADATA_FIELD,
    ExecutionTimeMetadata,
    Metadata,
)
from pipeline.context import PipelineContext
from pipeline.decorators import handle, on_input, on_stop, reactive, to_output

_DORA_INPUT_ID_ATTR = "dora_input_id"
_DORA_STOP_ID_ATTR = "dora_stop_id"
_DORA_OUTPUT_ID_ATTR = "dora_output_id"


class TestDecorators:
    """Test decorators."""

    def test_on_input(self) -> None:
        """Test on_input decorator."""

        @on_input("test")
        def test_function() -> None:
            """Test function."""

        assert getattr(test_function, _DORA_INPUT_ID_ATTR) == "test"

    def test_on_stop(self) -> None:
        """Test on_stop decorator."""

        @on_stop
        def test_function() -> None:
            """Test function."""

        assert isinstance(getattr(test_function, _DORA_STOP_ID_ATTR), str)

    def test_handle(self) -> None:
        """Test combined input/output handler decorator."""

        @handle("input", "output")
        def test_function() -> None:
            """Test function."""

        assert getattr(test_function, _DORA_INPUT_ID_ATTR) == "input"
        assert getattr(test_function, _DORA_OUTPUT_ID_ATTR) == "output"

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

    def test_reactive_adds_to_output_duration_metadata(self) -> None:
        """Test that reactive handlers add output duration to dora metadata output."""

        class FakeNode:
            def __init__(self) -> None:
                self.outputs = []
                self.events = [
                    {
                        "type": "INPUT",
                        "id": "tick",
                        "value": PipelineContext.from_timestamp(1.0).get_struct(),
                        "metadata": {
                            "execution_time_ms": '{"UpstreamNode": 1.0}',
                            "timestamp": 42,
                        },
                    },
                    {"type": "STOP"},
                ]

            def __iter__(self):
                return iter(self.events)

            def send_output(self, output_id, value, metadata=None) -> None:
                self.outputs.append((output_id, value, metadata))

        seen_metadata = []

        @reactive
        class TestOutputNode:
            """Test output node."""

            def run(self) -> None: ...
            @on_input("tick")
            @to_output("ctx")
            def handle_tick(self, metadata: Metadata) -> PipelineContext:
                seen_metadata.append(metadata)
                return PipelineContext.from_timestamp(1.0)

        fake_node = FakeNode()
        with patch("pipeline.decorators.Node", return_value=fake_node):
            node = TestOutputNode()
            node.run()

        assert len(fake_node.outputs) == 1
        output_id, value, metadata = fake_node.outputs[0]
        assert output_id == "ctx"
        assert "timestamp" not in metadata
        assert seen_metadata[0]["execution_time_ms"]["UpstreamNode"] == 1.0
        execution_time_ms = json.loads(metadata["execution_time_ms"])
        assert execution_time_ms["UpstreamNode"] == 1.0
        assert execution_time_ms["TestOutputNode"] >= 0.0
        assert "TestOutputNode" not in metadata

        ctx = PipelineContext(value)
        assert not ctx.exists("TestOutputNode")

    def test_reactive_binds_trace_id_from_metadata(self) -> None:
        """Test that reactive handlers bind trace id from dora metadata."""

        class FakeNode:
            def __init__(self) -> None:
                self.events = [
                    {
                        "type": "INPUT",
                        "id": "tick",
                        "value": PipelineContext.from_timestamp(1.0).get_struct(),
                        "metadata": {"trace_id": "frame-42"},
                    },
                    {"type": "STOP"},
                ]

            def __iter__(self):
                return iter(self.events)

        seen_trace_ids = []

        @reactive
        class TestTraceNode:
            """Test trace node."""

            def run(self) -> None: ...
            @on_input("tick")
            def handle_tick(self) -> None:
                seen_trace_ids.append(current_trace_id())

        with patch("pipeline.decorators.Node", return_value=FakeNode()):
            node = TestTraceNode()
            node.run()

        assert seen_trace_ids == ["frame-42"]
        assert current_trace_id() is None

    def test_reactive_leaves_trace_id_unset_without_metadata(self) -> None:
        """Test that reactive handlers do not create trace ids for untraced events."""

        class FakeNode:
            def __init__(self) -> None:
                self.events = [
                    {
                        "type": "INPUT",
                        "id": "tick",
                        "value": PipelineContext.from_timestamp(1.0).get_struct(),
                        "metadata": {},
                    },
                    {"type": "STOP"},
                ]

            def __iter__(self):
                return iter(self.events)

        seen_trace_ids = []

        @reactive
        class TestTraceNode:
            """Test trace node."""

            def run(self) -> None: ...
            @on_input("tick")
            def handle_tick(self) -> None:
                seen_trace_ids.append(current_trace_id())

        with patch("pipeline.decorators.Node", return_value=FakeNode()):
            node = TestTraceNode()
            node.run()

        assert seen_trace_ids == [None]
        assert current_trace_id() is None

    def test_reactive_extracts_execution_time_metadata_annotations(self) -> None:
        """Test that reactive handlers can receive decoded execution time metadata."""

        class FakeNode:
            def __init__(self) -> None:
                self.events = [
                    {
                        "type": "INPUT",
                        "id": "tick",
                        "value": PipelineContext.from_timestamp(1.0).get_struct(),
                        "metadata": {
                            "execution_time_ms": '{"DatasetNode": 1.0, "VIOFrontend": 2.5}',
                            SYNC_EXECUTION_START_TIME_NS_METADATA_FIELD: 123,
                        },
                    },
                    {"type": "STOP"},
                ]

            def __iter__(self):
                return iter(self.events)

        seen_execution_time_metadata = []

        @reactive
        class TestMetadataNode:
            """Test metadata node."""

            def run(self) -> None: ...
            @on_input("tick")
            def handle_tick(
                self,
                execution_time_metadata: ExecutionTimeMetadata,
            ) -> None:
                seen_execution_time_metadata.append(execution_time_metadata)

        with patch("pipeline.decorators.Node", return_value=FakeNode()):
            node = TestMetadataNode()
            node.run()

        record_batch = seen_execution_time_metadata[0]
        assert record_batch.schema.names == ["DatasetNode", "VIOFrontend"]
        assert record_batch.column("DatasetNode")[0].as_py() == 1.0
        assert record_batch.column("VIOFrontend")[0].as_py() == 2.5

    def test_reactive_extracts_empty_execution_time_metadata(self) -> None:
        """ExecutionTimeMetadata should be an empty RecordBatch when metadata has no timings."""

        class FakeNode:
            def __init__(self) -> None:
                self.events = [
                    {
                        "type": "INPUT",
                        "id": "tick",
                        "value": PipelineContext.from_timestamp(1.0).get_struct(),
                        "metadata": {},
                    },
                    {"type": "STOP"},
                ]

            def __iter__(self):
                return iter(self.events)

        seen_execution_time_metadata = []

        @reactive
        class TestMetadataNode:
            """Test metadata node."""

            def run(self) -> None: ...
            @on_input("tick")
            def handle_tick(self, execution_time_metadata: ExecutionTimeMetadata) -> None:
                seen_execution_time_metadata.append(execution_time_metadata)

        with patch("pipeline.decorators.Node", return_value=FakeNode()):
            node = TestMetadataNode()
            node.run()

        record_batch = seen_execution_time_metadata[0]
        assert record_batch.schema.names == []
