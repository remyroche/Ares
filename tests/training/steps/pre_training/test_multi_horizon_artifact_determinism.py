import asyncio
from pathlib import Path

import pandas as pd
import pytest

from src.training.steps.pre_training.components.base_component import ComponentConfig
from src.training.steps.pre_training.multi_horizon_profit_labeler import (
    MultiHorizonProfitLabelerComponent,
)


@pytest.fixture
def stub_labeler(monkeypatch):
    class _StubLabeler:
        def __init__(self, config):
            self.config = config
            self.quality_thresholds = {}

        async def execute_labeling(
            self,
            symbol: str,
            exchange: str,
            timeframe: str,
            data_dir: str = "historical_data",
            regime_data=None,
            quality_thresholds=None,
        ):
            df = pd.DataFrame({"timestamp": [1, 2, 3], "label": [1, 0, 1]}).set_index("timestamp")
            return {
                "multi_horizon_labeling_result": {
                    "labeled_data": df,
                    "method": "stubbed",
                },
                "labeling_report": {"status": "ok"},
            }

    monkeypatch.setattr(
        "src.training.steps.pre_training.multi_horizon_profit_labeler.MultiHorizonProfitLabeler",
        _StubLabeler,
    )
    return _StubLabeler


def test_multi_horizon_outcome_is_deterministic(tmp_path, monkeypatch, stub_labeler):
    monkeypatch.chdir(tmp_path)
    component = MultiHorizonProfitLabelerComponent(ComponentConfig())

    pipeline_state = {
        "symbol": "ETHUSDT",
        "exchange": "binance",
        "timeframe": "1h",
        "data_dir": "historical_data",
        "random_seed": 123,
    }

    first_result = asyncio.run(component.execute(None, pipeline_state))
    second_result = asyncio.run(component.execute(None, pipeline_state))

    assert first_result.success is True
    assert second_result.success is True

    assert first_result.metadata["artifact_digest"] == second_result.metadata["artifact_digest"]
    assert first_result.metadata["artifact_path"] == second_result.metadata["artifact_path"]

    outcome_path = Path(first_result.metadata["artifact_path"]).resolve()
    assert outcome_path.exists()
    with outcome_path.open() as handle:
        first_content = handle.read()
    with outcome_path.open() as handle:
        second_content = handle.read()
    assert first_content == second_content

    saved_files = sorted(Path("outcomes").glob("market_analysis_multi_horizon_profit_labeler_outcome_*.json"))
    assert len(saved_files) == 1
