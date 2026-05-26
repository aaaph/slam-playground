from typing import Annotated

import pyarrow as pa

from pipeline.context import PipelineContext

EXECUTION_TIME_MS_METADATA_FIELD = "execution_time_ms"
CONTEXT_BIRTH_TIME_NS_FIELD = "context_birth_time_ns"
SYNC_EXECUTION_START_TIME_NS_METADATA_FIELD = "sync_execution_start_time_ns"


class InEvent:
    """Event annotation. Used to mark a parameter as an event."""


class InCtx:
    """Context annotation. Used to mark a parameter as a context."""


class InMetadata:
    """Metadata annotation. Used to mark a parameter as a metadata."""


class InExecutionTimeMetadata:
    """Execution time metadata annotation."""


Event = Annotated[dict, InEvent]
Ctx = Annotated[PipelineContext, InCtx]
Metadata = Annotated[dict, InMetadata]
ExecutionTimeMetadata = Annotated[pa.RecordBatch, InExecutionTimeMetadata]
