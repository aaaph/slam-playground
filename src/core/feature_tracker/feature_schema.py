from enum import IntEnum

import pyarrow as pa


class FeatureSchema(IntEnum):
    """Feature schema."""

    FEAT_ID = 0
    TIMESTAMP = 1
    LEFT_U = 2
    LEFT_V = 3
    RIGHT_U = 4
    RIGHT_V = 5
    LIFECYCLE = 6
    AGE = 7

    @classmethod
    def count(cls) -> int:
        """Get the count of the feature schema."""
        return len(cls)


point2_schema = pa.struct(
    [
        pa.field("u", pa.float32(), nullable=True),
        pa.field("v", pa.float32(), nullable=True),
    ]
)

active_feat_arrow_schema = pa.schema(
    [
        pa.field("feat_id", pa.int32()),
        pa.field("timestamp", pa.float32()),
        pa.field(
            "stereo",
            pa.struct(
                [
                    pa.field("left", point2_schema),
                    pa.field("right", point2_schema),
                ]
            ),
        ),
        pa.field("state", pa.int32()),
        pa.field("age", pa.int32()),
    ]
)
