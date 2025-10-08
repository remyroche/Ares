import io
import json
from contextlib import contextmanager

import pytest

from src.training.steps.pre_training.logging_utils import (
    PreTrainingEventLogger,
    StepLogContext,
    configure_pre_training_logging,
)


@pytest.fixture
def structured_logger():
    return configure_pre_training_logging()


@contextmanager
def _capture_stream(handler):
    original_stream = handler.stream
    buffer = io.StringIO()
    handler.stream = buffer
    try:
        yield buffer
    finally:
        handler.stream = original_stream


def test_step_begin_logs_structured_payload(structured_logger):
    event_logger = PreTrainingEventLogger(structured_logger)
    context = StepLogContext(
        run_id="run-123",
        step="final_feature_selection",
        symbol="BTCUSDT",
        timeframe="1h",
        rows_in=100,
        rows_out=80,
    )

    handler = structured_logger.handlers[0]
    with _capture_stream(handler) as stream:
        event_logger.step_begin(context)
        handler.flush()
        output = stream.getvalue().strip().splitlines()[-1]

    payload = json.loads(output)
    assert payload["event"] == "step_begin"
    assert payload["phase"] == "begin"
    assert payload["run_id"] == "run-123"
    assert payload["step"] == "final_feature_selection"
    assert payload["symbol"] == "BTCUSDT"
    assert payload["timeframe"] == "1h"
    assert payload["rows_in"] == 100
    assert payload["rows_out"] == 80
    assert payload["duration_ms"] is None


def test_pipeline_end_includes_duration(structured_logger):
    event_logger = PreTrainingEventLogger(structured_logger)

    handler = structured_logger.handlers[0]
    with _capture_stream(handler) as stream:
        event_logger.pipeline_end(
            run_id="run-456",
            symbol="ETHUSDT",
            timeframe="15m",
            mode="full",
            success=True,
            duration_ms=1234.5,
            completed_steps=4,
            total_steps=4,
            metadata={"example": True},
        )
        handler.flush()
        output = stream.getvalue().strip().splitlines()[-1]

    payload = json.loads(output)
    assert payload["event"] == "pipeline_end"
    assert payload["run_id"] == "run-456"
    assert payload["symbol"] == "ETHUSDT"
    assert payload["timeframe"] == "15m"
    assert payload["duration_ms"] == pytest.approx(1234.5)
    assert payload["completed_steps"] == 4
    assert payload["total_steps"] == 4
    assert payload["success"] is True
