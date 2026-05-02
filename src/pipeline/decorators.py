import inspect
from collections.abc import Callable
from typing import Any, Protocol, get_type_hints

import pyarrow as pa
import reactivex as rx
import reactivex.operators as ops
from dora import Node

from logger import spawn_logger
from pipeline.annotations import InCtx, InEvent, InMetadata
from pipeline.context import PipelineContext

F = Callable[..., Any]


class _ReactiveNode(Protocol):
    node: Node
    logger: object
    _subject: rx.Subject
    _event_stream: rx.Observable

    def _setup_subscriptions(self) -> None: ...
    def _subscribe_input(self, method: F) -> None: ...
    def _subscribe_stop(self, method: F) -> None: ...
    def _create_handler(self, method: F) -> Callable[[dict[str, object]], object]: ...


def _make_reactive_init[T: object](cls: type[T]) -> Callable[..., None]:
    original_init = cls.__init__

    def new_init(self: _ReactiveNode, *args: tuple[object, ...], **kwargs: dict[str, object]) -> None:
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

        if hasattr(method, "dora_input_id"):
            self._subscribe_input(method)
        elif hasattr(method, "dora_stop_id"):
            self._subscribe_stop(method)


def _reactive_subscribe_input(self: _ReactiveNode, method: F) -> None:
    input_id = method.dora_input_id
    handler = self._create_handler(method)

    self._event_stream.pipe(ops.filter(lambda e: e["type"] == "INPUT" and e["id"] == input_id)).subscribe(
        on_next=handler
    )


def _reactive_subscribe_stop(self: _ReactiveNode, method: F) -> None:
    stream = self._event_stream.pipe(ops.filter(lambda e: e["type"] == "STOP")).pipe(ops.take(1))
    handler = self._create_handler(method)
    stream.subscribe(on_next=handler)


def _reactive_create_handler(self: _ReactiveNode, method: F) -> Callable[[dict[str, object]], None]:
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
        if marker is InEvent:
            extractors.append(lambda e: e)
        elif marker is InCtx:
            extractors.append(lambda e: PipelineContext(e["value"]))
        elif marker is InMetadata:
            extractors.append(lambda e: e.get("metadata", {}))
        else:
            extractors.append(lambda e: e.get("value"))

    """ if not extractors:
        return lambda _: method() """

    """   if len(extractors) == 1:
        extractor = extractors[0]
        return lambda event: method(extractor(event))
    """
    output_id = getattr(method, "dora_output_id", None)

    def handler(event: dict[str, Any]) -> None:
        args = [ext(event) for ext in extractors]
        result = method(*args)
        if result is not None and output_id is not None:
            metadata = event.get("metadata", {}).copy()
            metadata.pop("timestamp", None)
            if isinstance(result, PipelineContext):
                final_result = result.reassemble()
                self.node.send_output(output_id, final_result.get_struct(), metadata)
            elif isinstance(result, pa.Array):
                self.node.send_output(output_id, result, metadata)
            else:
                self.node.send_output(output_id, pa.array([result]), metadata)

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
        func.dora_input_id = input_id
        return func

    return decorator


def on_stop(func: F) -> F:
    """Pipeline stop decorator. Should handle dataflow stop events."""
    func.dora_stop_id = "__dora_reactive_stop_handler__"
    return func


def to_output(output_id: str) -> Callable[[F], F]:
    """Pipeline output decorator. Should handle dataflow output events."""

    def decorator(func: F) -> F:
        """Inner decorator."""
        func.dora_output_id = output_id
        return func

    return decorator


def reactive[T: object](cls: type[T]) -> type[T]:
    """Reactive node decorator."""
    cls.__init__ = _make_reactive_init(cls)
    cls._setup_subscriptions = _reactive_setup_subscriptions
    cls._subscribe_input = _reactive_subscribe_input
    cls._create_handler = _reactive_create_handler
    cls._subscribe_stop = _reactive_subscribe_stop
    cls.run = _reactive_run
    return cls
