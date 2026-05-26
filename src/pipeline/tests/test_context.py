import numpy as np
import pyarrow as pa
import pytest
from pyarrow import StructArray

from pipeline.context import PipelineContext


class TestPipelineContext:
    """Test PipelineContext class."""

    @pytest.fixture
    def empty_pipeline_context(self) -> PipelineContext:
        """Create a pipeline context."""
        data = StructArray.from_arrays([pa.array([0.0], type=pa.float64())], names=["empty"])
        return PipelineContext(data)

    @pytest.fixture
    def ndarray_pipeline_context(self) -> PipelineContext:
        """Create a pipeline context with an ndarray."""
        array = np.zeros((480, 752, 3), dtype=np.uint8)
        data = StructArray.from_arrays([pa.array([array.ravel()], type=pa.list_(pa.uint8()))], names=["array"])
        return PipelineContext(data)

    def test_get_scalar(self, empty_pipeline_context: PipelineContext) -> None:
        """Test that the PipelineContext can get a scalar value."""
        value = empty_pipeline_context.get_scalar("empty", float)
        assert value == 0.0

    def test_get_scalar_no_values(self, empty_pipeline_context: PipelineContext) -> None:
        """Test that the PipelineContext raises an error if the scalar value is not found."""
        with pytest.raises(KeyError, match="not_found"):
            empty_pipeline_context.get_scalar("not_found")

    def test_get_image(self, ndarray_pipeline_context: PipelineContext) -> None:
        """Test that the PipelineContext can get an image."""
        result = ndarray_pipeline_context.get_image("array", (480, 752, 3))
        assert result.shape == (480, 752, 3)
        assert result.dtype == np.uint8

        fields = ndarray_pipeline_context.data.type.names
        assert "array" in fields
        assert fields == ["array"]

    def test_get_image_no_values(self, ndarray_pipeline_context: PipelineContext) -> None:
        """Test that the PipelineContext raises an error if the image is not found."""
        with pytest.raises(KeyError, match="not_found"):
            ndarray_pipeline_context.get_image("not_found", (480, 752, 3))

    def test_set_scalar(self, empty_pipeline_context: PipelineContext) -> None:
        """Test that the PipelineContext can set a scalar value."""
        new_ctx = (
            empty_pipeline_context.set_scalar("two", 1.0, dtype=pa.int32())
            .set_scalar("three", 2.0, dtype=pa.float64())
            .reassemble()
        )
        assert new_ctx.get_scalar("two", int) == 1
        assert new_ctx.get_scalar("three", float) == 2.0

    def test_set_image(self, empty_pipeline_context: PipelineContext) -> None:
        """Test that the PipelineContext can set an image."""
        image = np.zeros((480, 752, 3), dtype=np.uint8)
        new_ctx = empty_pipeline_context.set_image("image", image).reassemble()

        assert new_ctx.get_image("image", (480, 752, 3)).shape == (480, 752, 3)
        assert new_ctx.get_image("image", (480, 752, 3)).dtype == np.uint8

    def test_from_timestamp(self) -> None:
        """Test that the PipelineContext can be created from a timestamp."""
        ctx = PipelineContext.from_timestamp(1.0)
        assert ctx.get_scalar("timestamp", float) == 1.0
        assert ctx.get_scalar("context_birth_time_ns", int) > 0

    def test_set_ndarray_empty(self, empty_pipeline_context: PipelineContext) -> None:
        """Test that the PipelineContext can set an empty ndarray."""
        new_ctx = empty_pipeline_context.set_ndarray("array", np.array([], dtype=np.uint8)).reassemble()
        assert new_ctx.get_ndarray("array").size == 0

    def test_exists(self, empty_pipeline_context: PipelineContext) -> None:
        """Test that the PipelineContext can check if a key exists."""
        assert empty_pipeline_context.exists("empty")
        assert not empty_pipeline_context.exists("not_found")

    def test_set_record_batch(self) -> None:
        """Test that the PipelineContext can set a record batch."""
        ctx = PipelineContext.from_timestamp(1.0)

        schema = pa.schema(
            [
                pa.field("id", pa.int32()),
                pa.field("name", pa.string()),
            ]
        )
        record_batch = pa.RecordBatch.from_arrays(
            [pa.array([1, 2, 3], type=pa.int32()), pa.array(["a", "b", "c"], type=pa.string())],
            schema=schema,
        )
        new_ctx = ctx.set_record_batch("record_batch", record_batch).reassemble()
        assert new_ctx.exists("record_batch")

        unpacked = new_ctx.get_record_batch("record_batch", schema)
        assert unpacked.column("id").equals(record_batch.column("id"))

    def test_set_empty_record_batch(self) -> None:
        """Test that the PipelineContext can set an empty record batch."""
        ctx = PipelineContext.from_timestamp(1.0)
        record_batch = pa.RecordBatch.from_arrays([], schema=pa.schema([]))

        new_ctx = ctx.set_record_batch("record_batch", record_batch).reassemble()

        unpacked = new_ctx.get_record_batch("record_batch")
        assert unpacked.schema.names == []
