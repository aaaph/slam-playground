import json
import urllib.error
import urllib.request
from typing import cast

import pytest
from zenoh import Session

from pipeline.transport import (
    ControlNodeTransport,
    ControlTransportFactory,
    ControlTransportSettings,
    HttpControlTransport,
    NoopControlTransport,
    ZenohControlTransport,
)


def _settings(transport: ControlNodeTransport, *, port: int = 0) -> ControlTransportSettings:
    return ControlTransportSettings(
        transport=transport,
        http_host="127.0.0.1",
        http_port=port,
    )


def _post_control(
    transport: HttpControlTransport,
    *,
    body: bytes,
    content_type: str,
) -> tuple[int, dict[str, object]]:
    host, port = transport.bound_address
    request = urllib.request.Request(
        f"http://{host}:{port}/control",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2) as response:  # noqa: S310 - local test server
        return response.status, json.loads(response.read().decode("utf-8"))


class TestControlTransportFactory:
    """Unit tests for control transport creation."""

    def test_create_http_transport(self) -> None:
        """Factory should create the configured HTTP/Zenoh/None transport."""
        assert isinstance(
            ControlTransportFactory.create(_settings(ControlNodeTransport.HTTP)), HttpControlTransport
        )
        assert isinstance(
            ControlTransportFactory.create(_settings(ControlNodeTransport.ZENOH)), ZenohControlTransport
        )
        assert isinstance(
            ControlTransportFactory.create(_settings(ControlNodeTransport.NONE)), NoopControlTransport
        )


class TestHttpControlTransport:
    """Unit tests for HTTP control transport."""

    def test_post_text_command_dispatches_registered_handler(self) -> None:
        """Text command bodies should be passed to the registered handler."""
        transport = HttpControlTransport(host="127.0.0.1", port=0)
        received: list[str] = []
        transport.on(lambda line: received.append(line) or {"line": line})

        try:
            transport.start()
            status, payload = _post_control(
                transport,
                body=b"ds:step:2",
                content_type="text/plain",
            )
        finally:
            transport.close()

        assert status == 202
        assert payload == {"accepted": True, "line": "ds:step:2"}
        assert received == ["ds:step:2"]

    def test_post_json_command_dispatches_registered_handler(self) -> None:
        """JSON command objects should be normalized to the line protocol."""
        transport = HttpControlTransport(host="127.0.0.1", port=0)
        received: list[str] = []
        transport.on(lambda line: received.append(line) or {"line": line})

        try:
            transport.start()
            status, payload = _post_control(
                transport,
                body=json.dumps({"command": "step", "value": "10%"}).encode("utf-8"),
                content_type="application/json",
            )
        finally:
            transport.close()

        assert status == 202
        assert payload == {"accepted": True, "line": "ds:step:10%"}
        assert received == ["ds:step:10%"]

    def test_post_without_handler_returns_bad_request(self) -> None:
        """HTTP transport should reject commands before a handler is registered."""
        transport = HttpControlTransport(host="127.0.0.1", port=0)

        try:
            transport.start()
            with pytest.raises(urllib.error.HTTPError) as exc_info:
                _post_control(
                    transport,
                    body=b"ds:start",
                    content_type="text/plain",
                )
            assert exc_info.value.code == 400
            payload = json.loads(exc_info.value.read().decode("utf-8"))
        finally:
            transport.close()

        assert payload == {"error": "control command handler is not registered"}


class TestZenohControlTransport:
    """Unit tests for zenoh control transport."""

    def test_subscriber_dispatches_registered_handler(self, mocker) -> None:
        """Zenoh subscriber callback should pass decoded command lines to the handler."""
        session = mocker.MagicMock(spec=Session)
        transport = ZenohControlTransport(session=cast("Session", session))
        received: list[str] = []
        transport.on(lambda line: received.append(line) or {"line": line})

        transport.start()
        callback = session.declare_subscriber.call_args.args[1]
        sample = _sample(mocker, b"ds:pause\n")
        callback(sample)
        transport.close()

        session.declare_subscriber.assert_called_once()
        session.declare_subscriber.return_value.undeclare.assert_called_once()
        session.close.assert_called_once()
        assert received == ["ds:pause"]

    def test_subscriber_ignores_invalid_commands(self, mocker) -> None:
        """Zenoh transport should swallow handler validation errors."""
        session = mocker.MagicMock(spec=Session)
        transport = ZenohControlTransport(session=cast("Session", session))

        def invalid_handler(line: str) -> dict[str, object]:
            msg = f"invalid command: {line}"
            raise ValueError(msg)

        transport.on(invalid_handler)

        transport.start()
        callback = session.declare_subscriber.call_args.args[1]
        callback(_sample(mocker, b"bad-command"))
        transport.close()

        session.close.assert_called_once()


def _sample(mocker, payload: bytes):
    sample = mocker.MagicMock()
    sample.payload.to_bytes.return_value = payload
    return sample
