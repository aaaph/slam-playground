import inspect
import json
import time
from collections.abc import Callable
from typing import Any, Protocol, cast, get_type_hints

import pyarrow as pa
import reactivex as rx
import reactivex.operators as ops
from dora import Node

from logger import TRACE_ID_METADATA_FIELD, reset_current_trace_id, set_current_trace_id, spawn_logger
from pipeline.annotations import (
    EXECUTION_TIME_MS_METADATA_FIELD,
    InCtx,
    InEvent,
    InExecutionTimeMetadata,
    InMetadata,
)
from pipeline.context import PipelineContext

F = Callable[..., Any]
_DoraEvent = dict[str, Any]
_ArgExtractor = Callable[[_DoraEvent, dict[str, Any]], Any]
_DORA_INPUT_ID_ATTR = "dora_input_id"
_DORA_STOP_ID_ATTR = "dora_stop_id"
_DORA_OUTPUT_ID_ATTR = "dora_output_id"
_INIT_ATTR = "__init__"
_SETUP_SUBSCRIPTIONS_ATTR = "_setup_subscriptions"
_SUBSCRIBE_INPUT_ATTR = "_subscribe_input"
_SUBSCRIBE_STOP_ATTR = "_subscribe_stop"
_CREATE_HANDLER_ATTR = "_create_handler"
_RUN_ATTR = "run"


class _ReactiveNode(Protocol):
    node: Node
    logger: object
    _subject: rx.Subject
    _event_stream: rx.Observable

    def _setup_subscriptions(self) -> None: ...
    def _subscribe_input(self, method: F) -> None: ...
    def _subscribe_stop(self, method: F) -> None: ...
    def _create_handler(self, method: F) -> Callable[[_DoraEvent], None]: ...


class _PipelineOutputNode(Protocol):
    def send_output(
        self,
        output_id: str,
        value: pa.Array,
        metadata: dict[Any, Any],
        /,
    ) -> None: ...


def _make_reactive_init[T: object](cls: type[T]) -> Callable[..., None]:
    original_init = getattr(cls, _INIT_ATTR)

    def new_init(self: _ReactiveNode, *args: object, **kwargs: object) -> None:
        """Initialize the node."""
        self.logger = spawn_logger(app=cls.__name__)
        original_init(self, *args, **kwargs)
        if not hasattr(self, "node"):
            self.node = Node()
        self._subject = rx.Subject()
        self._event_stream = self._subject.pipe(ops.share())
        self._setup_subscriptions()

    return new_init


def _reactive_setup_subscriptions(self: _ReactiveNode) -> None:
    """Iterate over the methods of the node and setup subscriptions to the event stream."""
    blacklist = {
        "run",
        "node",
        "logger",
        "_subject",
        "_event_stream",
        "_setup_subscriptions",
        "_subscribe_input",
        "_subscribe_stop",
        "_create_handler",
    }
    cls_type = type(self)
    for name in dir(cls_type):
        if name in blacklist or name.startswith("__"):
            continue

        method = getattr(self, name)

        if hasattr(method, _DORA_INPUT_ID_ATTR):
            self._subscribe_input(method)
        elif hasattr(method, _DORA_STOP_ID_ATTR):
            self._subscribe_stop(method)


def _reactive_subscribe_input(self: _ReactiveNode, method: F) -> None:
    input_id = getattr(method, _DORA_INPUT_ID_ATTR)
    handler = self._create_handler(method)

    self._event_stream.pipe(ops.filter(lambda e: e["type"] == "INPUT" and e["id"] == input_id)).subscribe(
        on_next=handler
    )


def _reactive_subscribe_stop(self: _ReactiveNode, method: F) -> None:
    stream = self._event_stream.pipe(ops.filter(lambda e: e["type"] == "STOP")).pipe(ops.take(1))
    handler = self._create_handler(method)
    stream.subscribe(on_next=handler)


def _decode_execution_time_ms(value: object) -> dict[str, float]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    if not isinstance(value, dict):
        return {}
    return {str(key): float(duration) for key, duration in value.items() if isinstance(duration, int | float)}


def _decode_metadata(metadata: object) -> dict[str, Any]:
    decoded: dict[str, Any] = cast("dict[str, Any]", metadata.copy()) if isinstance(metadata, dict) else {}
    if EXECUTION_TIME_MS_METADATA_FIELD in decoded:
        decoded[EXECUTION_TIME_MS_METADATA_FIELD] = _decode_execution_time_ms(
            decoded[EXECUTION_TIME_MS_METADATA_FIELD]
        )
    return decoded


def _execution_time_ms_record_batch(value: object) -> pa.RecordBatch:
    execution_time_ms = _decode_execution_time_ms(value)
    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []
    for node_name, duration_ms in execution_time_ms.items():
        arrays.append(pa.array([duration_ms], type=pa.float64()))
        fields.append(pa.field(node_name, pa.float64()))

    return pa.RecordBatch.from_arrays(arrays, schema=pa.schema(fields))


def _int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _encode_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    encoded = metadata.copy()
    encoded.pop("timestamp", None)
    execution_time_ms = encoded.get(EXECUTION_TIME_MS_METADATA_FIELD)
    if isinstance(execution_time_ms, dict):
        encoded[EXECUTION_TIME_MS_METADATA_FIELD] = json.dumps(execution_time_ms, sort_keys=True)
    return encoded


def send_pipeline_context_output(
    node: _PipelineOutputNode,
    output_id: str,
    ctx: PipelineContext,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Send a PipelineContext output with metadata encoded for dora."""
    encoded_metadata = _encode_metadata(metadata) if metadata is not None else {}
    node.send_output(output_id, ctx.reassemble().get_struct(), encoded_metadata)


def _add_output_duration_metadata(
    node: _ReactiveNode,
    metadata: dict[str, Any],
    duration_ms: float,
) -> None:
    execution_time_ms = _decode_execution_time_ms(metadata.get(EXECUTION_TIME_MS_METADATA_FIELD))
    execution_time_ms[node.__class__.__name__] = duration_ms
    metadata[EXECUTION_TIME_MS_METADATA_FIELD] = execution_time_ms


def _extractor_for_marker(marker: object) -> _ArgExtractor:
    marker_extractors: dict[object, _ArgExtractor] = {
        InEvent: lambda e, _metadata: e,
        InCtx: lambda e, _metadata: PipelineContext(e["value"]),
        InMetadata: lambda _event, metadata: metadata,
        InExecutionTimeMetadata: lambda _event, metadata: _execution_time_ms_record_batch(
            metadata.get(EXECUTION_TIME_MS_METADATA_FIELD)
        ),
    }
    return marker_extractors.get(marker, lambda e, _metadata: e.get("value"))


def _reactive_create_handler(self: _ReactiveNode, method: F) -> Callable[[_DoraEvent], None]:
    sig = inspect.signature(method)
    type_hints = get_type_hints(method, include_extras=True)
    params = list(sig.parameters.values())
    extractors = []
    for param in params:
        if param.name == "self":
            continue
        hint = type_hints.get(param.name, Any)
        metadata_list = getattr(hint, "__metadata__", [None])
        marker = metadata_list[0]
        extractors.append(_extractor_for_marker(marker))

    """ if not extractors:
        return lambda _: method() """

    """   if len(extractors) == 1:
        extractor = extractors[0]
        return lambda event: method(extractor(event))
    """
    output_id = getattr(method, _DORA_OUTPUT_ID_ATTR, None)

    def handler(event: _DoraEvent) -> None:
        metadata = _decode_metadata(event.get("metadata", {}))
        trace_id = metadata.get(TRACE_ID_METADATA_FIELD)
        token = set_current_trace_id(trace_id)
        try:
            args = [ext(event, metadata) for ext in extractors]
            start_time = time.perf_counter()
            result = method(*args)
            duration_ms = (time.perf_counter() - start_time) * 1000
            if result is not None and output_id is not None:
                metadata.pop("timestamp", None)
                _add_output_duration_metadata(self, metadata, duration_ms)
                metadata = _encode_metadata(metadata)
                if isinstance(result, PipelineContext):
                    final_result = result.reassemble()
                    self.node.send_output(output_id, final_result.get_struct(), metadata)
                elif isinstance(result, pa.Array):
                    self.node.send_output(output_id, result, metadata)
                else:
                    self.node.send_output(output_id, pa.array([result]), metadata)
        finally:
            reset_current_trace_id(token)

    return handler


def _reactive_run(self: _ReactiveNode) -> None:
    """Run the node."""
    getattr(self.logger, "info")(f"Running node {self.__class__.__name__}")  # noqa: B009
    try:
        for event in self.node:
            self._subject.on_next(event)
            if event["type"] == "STOP":
                break
        self._subject.on_completed()
    except Exception as e:
        self._subject.on_error(e)
        raise


def on_input(input_id: str) -> Callable[[F], F]:
    """Pipeline input decorator. Should handle dataflow input events."""

    def decorator(func: F) -> F:
        """Inner decorator."""
        setattr(func, _DORA_INPUT_ID_ATTR, input_id)
        return func

    return decorator


def on_stop(func: F) -> F:
    """Pipeline stop decorator. Should handle dataflow stop events."""
    setattr(func, _DORA_STOP_ID_ATTR, "__dora_reactive_stop_handler__")
    return func


def to_output(output_id: str) -> Callable[[F], F]:
    """Pipeline output decorator. Should handle dataflow output events."""

    def decorator(func: F) -> F:
        """Inner decorator."""
        setattr(func, _DORA_OUTPUT_ID_ATTR, output_id)
        return func

    return decorator


def handle(input_id: str, output_id: str) -> Callable[[F], F]:
    """Pipeline handler decorator that subscribes to input and sends returned values to output."""

    def decorator(func: F) -> F:
        """Inner decorator."""
        return on_input(input_id)(to_output(output_id)(func))

    return decorator


def reactive[T: object](cls: type[T]) -> type[T]:
    """Reactive node decorator."""
    setattr(cls, _INIT_ATTR, _make_reactive_init(cls))
    setattr(cls, _SETUP_SUBSCRIPTIONS_ATTR, _reactive_setup_subscriptions)
    setattr(cls, _SUBSCRIBE_INPUT_ATTR, _reactive_subscribe_input)
    setattr(cls, _CREATE_HANDLER_ATTR, _reactive_create_handler)
    setattr(cls, _SUBSCRIBE_STOP_ATTR, _reactive_subscribe_stop)
    setattr(cls, _RUN_ATTR, _reactive_run)
    return cls
