from unittest.mock import call

import pyarrow as pa
import pytest

from pipeline.context import PipelineContext
from visualizer.rerun.modules.plot_scalars_module import PlotScalarsModule


class TestPlotScalarsModule:
    """Unit tests for PlotScalarsModule."""

    def test_process_record_batch_scalar_field(self, mocker) -> None:
        """It should log a scalar from a selected RecordBatch column."""
        log_mock = mocker.patch("visualizer.rerun.modules.plot_scalars_module.rr.log")
        set_time_mock = mocker.patch("visualizer.rerun.modules.plot_scalars_module.rr.set_time")
        mocker.patch(
            "visualizer.rerun.modules.plot_scalars_module.rr.Scalars",
            side_effect=lambda value: ("scalar", value),
        )

        module = PlotScalarsModule(
            "execution_time_ms",
            "/metrics/reactive/execution_time_ms/dataset",
            {
                "arrow_type": "RecordBatch",
                "arrow_field": "DatasetNode",
                "label": "DatasetNode",
            },
        )
        record_batch = pa.RecordBatch.from_pydict({"DatasetNode": [1.25]})
        ctx = (
            PipelineContext.from_timestamp(1_000_000_000.0)
            .set_record_batch("execution_time_ms", record_batch)
            .reassemble()
        )

        module.process(ctx)

        assert set_time_mock.call_args_list == [
            call("sim_time", timestamp=1.0),
            call("frame_time", timestamp=1.0),
        ]
        assert log_mock.call_args_list == [
            call("/metrics/reactive/execution_time_ms/dataset", ("scalar", 1.25)),
        ]

    def test_record_batch_requires_arrow_field(self) -> None:
        """It should fail clearly when a RecordBatch scalar does not specify a column."""
        module = PlotScalarsModule(
            "execution_time_ms",
            "/metrics/reactive/execution_time_ms/dataset",
            {"arrow_type": "RecordBatch", "label": "DatasetNode"},
        )
        record_batch = pa.RecordBatch.from_pydict({"DatasetNode": [1.25]})
        ctx = (
            PipelineContext.from_timestamp(1_000_000_000.0)
            .set_record_batch("execution_time_ms", record_batch)
            .reassemble()
        )

        with pytest.raises(ValueError, match="arrow_field is required"):
            module.process(ctx)

    def test_process_record_batch_skips_missing_field_by_default(self, mocker) -> None:
        """Missing RecordBatch fields should be ignored unless strict mode is enabled."""
        log_mock = mocker.patch("visualizer.rerun.modules.plot_scalars_module.rr.log")

        module = PlotScalarsModule(
            "execution_time_ms",
            "/metrics/reactive/execution_time_ms/frontend",
            {
                "arrow_type": "RecordBatch",
                "arrow_field": "VIOFrontend",
                "label": "VIOFrontend",
            },
        )
        record_batch = pa.RecordBatch.from_pydict({"DatasetNode": [1.25]})
        ctx = (
            PipelineContext.from_timestamp(1_000_000_000.0)
            .set_record_batch("execution_time_ms", record_batch)
            .reassemble()
        )

        module.process(ctx)

        log_mock.assert_not_called()
