import time
from typing import Any, TypeVar, cast

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

T = TypeVar("T")
CONTEXT_BIRTH_TIME_NS_FIELD = "context_birth_time_ns"


class PipelineContext:
    """Pipeline context."""

    def __init__(self, data: pa.StructArray) -> None:
        """Initialize the pipeline context."""
        self.data = data
        self._updates: dict[str, pa.Array] = {}

    def exists(self, key: str) -> bool:
        """Check if the given key exists in the pipeline context."""
        try:
            return self.data.field(key) is not None
        except (KeyError, pa.ArrowKeyError):
            return False
        except Exception as e:
            msg = f"Error checking if key {key} exists: {e}"
            raise ValueError(msg) from e

    def set_scalar(
        self,
        key: str,
        value: Any,  # noqa: ANN401
        dtype: pa.DataType | None = None,
    ) -> "PipelineContext":
        """Set the scalar value of the given key in the struct array."""
        if dtype is None:
            dtype = pa.float64()
        self._updates[key] = pa.array([value], type=dtype)
        return self

    def set_ndarray(self, key: str, value: NDArray[Any]) -> "PipelineContext":
        """Set the ndarray value of the given key in the struct array."""
        flat = value.ravel()
        self._updates[key] = pa.array([flat])
        return self

    def set_image(self, key: str, value: NDArray[np.uint8]) -> "PipelineContext":
        """Set the image value of the given key in the struct array."""
        return self.set_ndarray(key, value)

    def set_record_batch(self, key: str, value: pa.RecordBatch) -> "PipelineContext":
        """Set the record batch value of the given key in the struct array."""
        if value.num_columns == 0:
            self._updates[key] = pa.array([{}], type=pa.struct([]))
            return self

        arrays: list[pa.Array] = []
        fields: list[pa.Field] = []
        for f, col in zip(value.schema, value.columns, strict=True):
            offsets = pa.array([0, len(col)], type=pa.int32())
            list_col = pa.ListArray.from_arrays(offsets, col)
            arrays.append(list_col)
            fields.append(pa.field(f.name, pa.list_(f.type), nullable=True))
        self._updates[key] = pa.StructArray.from_arrays(arrays, fields=fields)
        return self

    def get_struct(self) -> pa.StructArray:
        """Get the struct array from the pipeline context."""
        return self.data

    def get_scalar(self, key: str, return_type: type[T] = Any) -> T:  # noqa: ARG002  # ty:ignore[invalid-parameter-default]
        """Get the scalar value of the given key from the struct array."""
        field = self.data.field(key)
        if len(field) < 1:
            msg = f"Field {key} has no values"
            raise ValueError(msg)
        value = field[0].as_py()
        return cast("T", value)

    def get_ndarray(self, key: str, shape: tuple[int, ...] | None = None) -> NDArray[Any]:
        """Get the ndarray value of the given key from the struct array."""
        field = self.data.field(key)
        if len(field) < 1:
            msg = f"Field {key} has no values"
            raise ValueError(msg)
        try:
            flat_array = field[0].values.to_numpy(zero_copy_only=True)
        except (AttributeError, ValueError):
            flat_array = field[0].values.to_numpy(zero_copy_only=False)
        if shape is not None:
            shape = tuple(map(int, shape))
        return flat_array.reshape(shape)

    def get_image(self, key: str, shape: tuple[int, ...]) -> NDArray[np.uint8]:
        """Get the image value of the given key from the struct array."""
        return cast("NDArray[np.uint8]", self.get_ndarray(key, shape))

    def get_record_batch(self, key: str, schema: pa.Schema | None = None) -> pa.RecordBatch:
        """Get the record batch value of the given key from the struct array."""
        field_array = self.data.field(key)
        if len(field_array) < 1:
            msg = f"Field {key} has no values"
            raise ValueError(msg)
        row0 = field_array[0]
        if schema is None:
            names = row0.type.names
            cols = []
            fields = []
            for name in names:
                list_scalar = row0[name]
                values_arr = list_scalar.values
                cols.append(values_arr)
                fields.append(pa.field(name, values_arr.type))
            return pa.RecordBatch.from_arrays(cols, schema=pa.schema(fields))
        cols: list[pa.Array] = []
        for f in schema:
            try:
                list_scalar = row0[f.name]
            except (KeyError, pa.ArrowKeyError) as e:
                msg = f"Field {f.name} not found in record batch"
                raise KeyError(msg) from e

            values_arr = list_scalar.values
            if values_arr.type != f.type:
                try:
                    values_arr = values_arr.cast(f.type)
                except Exception as e:
                    msg = f"Error casting field {f.name} to type {f.type}: {e}"
                    raise TypeError(msg) from e
            cols.append(values_arr)

        return pa.RecordBatch.from_arrays(cols, schema=schema)

    def reassemble(self) -> "PipelineContext":
        """Reassemble the pipeline context. Adds new updates to the existing data."""
        all_names = list(self.data.type.names)

        for key in self._updates:
            if key not in all_names:
                all_names.append(key)

        final_arrays = []
        for name in all_names:
            if name in self._updates:
                final_arrays.append(self._updates[name])
            else:
                final_arrays.append(self.data.field(name))
        return PipelineContext(pa.StructArray.from_arrays(arrays=final_arrays, names=all_names))

    @classmethod
    def from_timestamp(cls, timestamp: float) -> "PipelineContext":
        """Create a pipeline context from a scalar value."""
        arrays = [
            pa.array([timestamp], type=pa.float64()),
            pa.array([time.perf_counter_ns()], type=pa.int64()),
        ]
        return cls(pa.StructArray.from_arrays(arrays, names=["timestamp", CONTEXT_BIRTH_TIME_NS_FIELD]))

    def __repr__(self) -> str:
        """Return a string representation of the pipeline context."""
        packed_column_names = list(self.data.type.names)
        pending_column_names = list(self._updates.keys())
        return f"PipelineContext(packed={packed_column_names}, pending={pending_column_names})"
