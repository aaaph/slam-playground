from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlparse

from zenoh import Config
from zenoh import open as zenoh_open

if TYPE_CHECKING:
    from zenoh import Sample, Session


class ControlNodeTransport(StrEnum):
    """External command ingress for the control node."""

    ZENOH = "zenoh"
    HTTP = "http"
    NONE = "none"


type ControlCommandResult = dict[str, object]


class ControlCommandHandler(Protocol):
    """Handle a raw external control command line."""

    def __call__(self, line: str) -> ControlCommandResult:
        """Handle the raw command line and return a response payload."""
        ...


class ControlTransport(Protocol):
    """Lifecycle for a control command transport."""

    def on(self, command_handler: ControlCommandHandler) -> None:
        """Register a raw command handler."""
        ...

    def start(self) -> None:
        """Start the transport."""
        ...

    def close(self) -> None:
        """Close the transport."""
        ...


@dataclass(frozen=True)
class ControlTransportSettings:
    """Runtime settings for a control command transport."""

    transport: ControlNodeTransport
    http_host: str
    http_port: int


class _TransportSubscription(Protocol):
    def undeclare(self) -> None: ...


class _CommandEmitter:
    def __init__(self) -> None:
        self._command_handler: ControlCommandHandler | None = None

    def on(self, command_handler: ControlCommandHandler) -> None:
        """Register a raw command handler."""
        self._command_handler = command_handler

    def _handle_command(self, line: str) -> ControlCommandResult:
        if self._command_handler is None:
            msg = "control command handler is not registered"
            raise RuntimeError(msg)
        return self._command_handler(line)


class NoopControlTransport(_CommandEmitter):
    """Disabled external control command transport."""

    def __init__(self) -> None:
        """Initialize the disabled transport."""
        super().__init__()

    def start(self) -> None:
        """Start the disabled transport."""

    def close(self) -> None:
        """Close the disabled transport."""


class ZenohControlTransport(_CommandEmitter):
    """Zenoh subscriber transport for control commands."""

    def __init__(
        self,
        *,
        session: Session | None = None,
        key: str = "pipeline/control",
    ) -> None:
        """Initialize the zenoh control transport."""
        super().__init__()
        self.session = session
        self.key = key
        self._sub: _TransportSubscription | None = None

    def start(self) -> None:
        """Start the zenoh subscriber."""
        if self.session is None:
            config = Config()
            config.insert_json5("transport/link/tx/lease", "30000")
            config.insert_json5("transport/link/tx/keep_alive", "4")
            self.session = zenoh_open(config)

        def callback(data: Sample, /) -> None:
            line = data.payload.to_bytes().decode("utf-8").strip()
            if not line:
                return
            try:
                self._handle_command(line)
            except (RuntimeError, ValueError):
                return

        self._sub = self.session.declare_subscriber(self.key, callback)

    def close(self) -> None:
        """Close the zenoh subscriber and session."""
        if self._sub is not None:
            self._sub.undeclare()
            self._sub = None
        if self.session is not None:
            self.session.close()
            self.session = None


class HttpControlTransport(_CommandEmitter):
    """HTTP server transport for control commands."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
    ) -> None:
        """Initialize the HTTP control transport."""
        super().__init__()
        self.host = host
        self.port = port
        self._server: _ControlHttpServer | None = None
        self._thread: Thread | None = None

    @property
    def bound_address(self) -> tuple[str, int]:
        """Return the actual host/port the server is bound to."""
        if self._server is None:
            return self.host, self.port
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def start(self) -> None:
        """Start the HTTP control server."""
        self._server = _ControlHttpServer(
            (self.host, self.port),
            _ControlHttpHandler,
            self._handle_command,
        )
        self._thread = Thread(
            target=self._server.serve_forever,
            name="control-http-server",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stop the HTTP control server."""
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
        if self._thread is not None:
            self._thread.join(timeout=0.25)
            self._thread = None


class ControlTransportFactory:
    """Factory for control command transports."""

    @staticmethod
    def create(
        settings: ControlTransportSettings,
        *,
        zenoh_session: Session | None = None,
    ) -> ControlTransport:
        """Create a control transport from runtime configuration."""
        if zenoh_session is not None:
            return ZenohControlTransport(session=zenoh_session)
        if settings.transport == ControlNodeTransport.ZENOH:
            return ZenohControlTransport()
        if settings.transport == ControlNodeTransport.HTTP:
            return HttpControlTransport(
                host=settings.http_host,
                port=settings.http_port,
            )
        return NoopControlTransport()


class _ControlHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler: type[BaseHTTPRequestHandler],
        command_handler: ControlCommandHandler,
    ) -> None:
        super().__init__(server_address, request_handler)
        self.command_handler = command_handler


class _ControlHttpHandler(BaseHTTPRequestHandler):
    server: _ControlHttpServer

    def do_GET(self) -> None:
        """Serve a lightweight readiness endpoint."""
        if urlparse(self.path).path != "/health":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self._write_json(HTTPStatus.OK, {"status": "ok"})

    def do_POST(self) -> None:
        """Accept a control command and enqueue it for the dora tick path."""
        if urlparse(self.path).path != "/control":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        try:
            line = self._read_command_line()
            result = self.server.command_handler(line)
        except (RuntimeError, TypeError, ValueError) as exc:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return

        self._write_json(HTTPStatus.ACCEPTED, {"accepted": True, **result})

    def _read_command_line(self) -> str:
        content_length = self.headers.get("Content-Length")
        if content_length is None:
            msg = "missing Content-Length"
            raise ValueError(msg)
        try:
            length = int(content_length)
        except ValueError as exc:
            msg = "invalid Content-Length"
            raise ValueError(msg) from exc
        if length <= 0:
            msg = "empty command body"
            raise ValueError(msg)

        raw_body = self.rfile.read(length).decode("utf-8").strip()
        if self.headers.get_content_type() == "application/json":
            return self._command_line_from_json(raw_body)
        return raw_body

    @staticmethod
    def _command_line_from_json(raw_body: str) -> str:
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as exc:
            msg = "invalid JSON command body"
            raise ValueError(msg) from exc

        if isinstance(payload, str):
            return payload
        if not isinstance(payload, dict):
            msg = "JSON command body must be a string or object"
            raise TypeError(msg)

        raw_line = payload.get("line") or payload.get("command_line")
        if isinstance(raw_line, str):
            return raw_line

        command = payload.get("command")
        if not isinstance(command, str) or not command:
            msg = "JSON command body must include command"
            raise ValueError(msg)

        target = payload.get("target", "ds")
        if not isinstance(target, str) or not target:
            msg = "JSON command target must be a string"
            raise ValueError(msg)

        value = payload.get("value")
        if value is None:
            return f"{target}:{command}"
        return f"{target}:{command}:{value}"

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
