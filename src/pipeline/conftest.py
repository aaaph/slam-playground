from collections.abc import Iterator

import pytest

from logger import reset_current_trace_id, set_current_trace_id


@pytest.fixture(autouse=True)
def clean_trace_context() -> Iterator[None]:
    """Run pipeline tests with an isolated logging trace context."""
    token = set_current_trace_id(None)
    try:
        yield
    finally:
        reset_current_trace_id(token)
