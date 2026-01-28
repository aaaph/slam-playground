from typing import Any, TypeVar, cast

import numpy as np
import pyarrow as pa
from numpy.typing import NDArray

T = TypeVar("T")


class PipelineContext:
    """Pipeline context."""

    def __init__(self, data: pa.StructArray) -> None:
        """Initialize the pipeline context."""
        self.data = data
        self._updates: dict[str, pa.Array] = {}

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
        storage = pa.array(flat)
        self._updates[key] = pa.FixedSizeListArray.from_arrays(storage, value.size)
        return self

    def set_image(self, key: str, value: NDArray[np.uint8]) -> "PipelineContext":
        """Set the image value of the given key in the struct array."""
        return self.set_ndarray(key, value)

    def get_struct(self) -> pa.StructArray:
        """Get the struct array from the pipeline context."""
        return self.data

    def get_scalar(self, key: str, return_type: type[T] = Any) -> T:  # noqa: ARG002
        """Get the scalar value of the given key from the struct array."""
        field = self.data.field(key)
        if len(field) < 1:
            msg = f"Field {key} has no values"
            raise ValueError(msg)
        value = field[0].as_py()
        return cast("T", value)

    def get_ndarray(self, key: str, shape: tuple[int, ...]) -> NDArray[Any]:
        """Get the ndarray value of the given key from the struct array."""
        field = self.data.field(key)
        if len(field) < 1:
            msg = f"Field {key} has no values"
            raise ValueError(msg)
        try:
            flat_array = field[0].values.to_numpy(zero_copy_only=True)
        except (AttributeError, ValueError):
            flat_array = field[0].values.to_numpy(zero_copy_only=False)
        return flat_array.reshape(shape)

    def get_image(self, key: str, shape: tuple[int, ...]) -> NDArray[np.uint8]:
        """Get the image value of the given key from the struct array."""
        return cast("NDArray[np.uint8]", self.get_ndarray(key, shape))

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

        return PipelineContext(pa.StructArray.from_arrays(final_arrays, names=all_names))

    @classmethod
    def from_timestamp(cls, timestamp: float) -> "PipelineContext":
        """Create a pipeline context from a scalar value."""
        array = pa.array([timestamp], type=pa.float64())
        return cls(pa.StructArray.from_arrays([array], names=["timestamp"]))

    def __repr__(self) -> str:
        """Return a string representation of the pipeline context."""
        column_names = list(self.data.type.names)
        column_size = len(column_names)
        return f"PipelineContext(columns={column_names}, size={column_size})"
