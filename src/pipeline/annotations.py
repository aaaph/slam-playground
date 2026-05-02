from typing import Annotated

from pipeline.context import PipelineContext


class InEvent:
    """Event annotation. Used to mark a parameter as an event."""


class InCtx:
    """Context annotation. Used to mark a parameter as a context."""


class InMetadata:
    """Metadata annotation. Used to mark a parameter as a metadata."""


Event = Annotated[dict, InEvent]
Ctx = Annotated[PipelineContext, InCtx]
Metadata = Annotated[dict, InMetadata]
