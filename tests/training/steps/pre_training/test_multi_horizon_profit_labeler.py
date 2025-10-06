"""Tests for the multi-horizon profit labeler data loading and execution pipeline."""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.training.steps.pre_training.multi_horizon_profit_labeler import (
    MultiHorizonConfig,
    MultiHorizonProfitLabeler,
)


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def stub_volatility_labeler(monkeypatch):
    class _StubVolatilityLabeler:
        def __init__(self, config):
            self.config = config

        def generate_labels(self, market_data):
            raise NotImplementedError("stub - replace in tests")

    monkeypatch.setattr(
        "src.training.steps.pre_training.multi_horizon_profit_labeler.VolatilityAwareMultiHorizonLabeler",
        _StubVolatilityLabeler,
    )
    monkeypatch.setattr(
        MultiHorizonProfitLabeler,
        "_create_volatility_config",
        lambda self: None,
    )
    yield


def _create_sample_ohlcv_dataframe(rows: int = 240) -> pd.DataFrame:
    """Create a realistic OHLCV dataframe with a timestamp column."""
    start = datetime.utcnow() - timedelta(minutes=rows * 15)
    timestamps = pd.date_range(start=start, periods=rows, freq="15min")

    base_price = 100 + np.cumsum(np.random.randn(rows)) * 0.5
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": base_price + np.random.randn(rows) * 0.1,
            "high": base_price + np.abs(np.random.randn(rows) * 0.2),
            "low": base_price - np.abs(np.random.randn(rows) * 0.2),
            "close": base_price + np.random.randn(rows) * 0.1,
            "volume": np.random.rand(rows) * 1_000 + 100,
        }
    )

    return df


class _StubKlinesManager:
    """Minimal klines manager that returns a copy of prepared OHLCV data."""

    def __init__(self, dataframe: pd.DataFrame):
        self._dataframe = dataframe
        self.calls: list[tuple[str, str, str]] = []

    def read_data(
        self,
        symbol: str,
        interval: str,
        start_date=None,
        end_date=None,
        data_type: str = "raw",
        columns=None,
    ) -> pd.DataFrame:
        self.calls.append((symbol, interval, data_type))
        return self._dataframe.copy()


@pytest.mark.anyio("asyncio")
async def test_load_market_data_normalizes_dataframe(monkeypatch):
    """The loader should read OHLCV data and normalize the index/columns."""

    df = _create_sample_ohlcv_dataframe()
    stub_manager = _StubKlinesManager(df)

    monkeypatch.setattr(
        "src.training.steps.pre_training.multi_horizon_profit_labeler.get_klines_manager",
        lambda data_dir: stub_manager,
    )

    labeler = MultiHorizonProfitLabeler(MultiHorizonConfig(min_data_points=100))
    loaded = await labeler._load_market_data("ETHUSDT", "binance", "15m", "unused")

    assert not loaded.empty
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.is_monotonic_increasing
    for column in ("open", "high", "low", "close", "volume"):
        assert column in loaded.columns
    assert stub_manager.calls, "klines manager should be queried"


class _DummyLabelingResult:
    """Minimal labeling result used to validate execute_labeling wiring."""

    def __init__(self, index: pd.DatetimeIndex):
        labels = pd.DataFrame(
            {
                "small_target": np.random.choice([-1, 0, 1], len(index)),
                "medium_target": np.random.choice([-1, 0, 1], len(index)),
                "large_target": np.random.choice([-1, 0, 1], len(index)),
            },
            index=index,
        )

        confidence_scores = pd.DataFrame(
            {
                "small_target_conf": np.random.rand(len(index)),
                "medium_target_conf": np.random.rand(len(index)),
                "large_target_conf": np.random.rand(len(index)),
            },
            index=index,
        )

        eligibility_masks = pd.DataFrame(
            {
                "small_target_mask": np.ones(len(index), dtype=bool),
                "medium_target_mask": np.ones(len(index), dtype=bool),
                "large_target_mask": np.ones(len(index), dtype=bool),
            },
            index=index,
        )

        # Rename to patterns expected by the mapping logic
        self.labels = labels.rename(
            columns={
                "small_target": "small_h0.50_a1.00",
                "medium_target": "medium_h1.00_a2.00",
                "large_target": "high_h1.50_a3.00",
            }
        )
        self.confidence_scores = confidence_scores
        self.eligibility_masks = eligibility_masks
        self.quality_scores = {
            "small_h0.50_a1.00": {"overall_quality": 0.7},
            "medium_h1.00_a2.00": {"overall_quality": 0.65},
            "large_h1.50_a3.00": {"overall_quality": 0.6},
        }
        self.processing_time = 0.5
        self.n_samples = len(index)
        self.n_targets = self.labels.shape[1]
        self.n_horizons = 3


@pytest.mark.anyio("asyncio")
async def test_execute_labeling_produces_feature_lookback_artifacts(monkeypatch):
    """execute_labeling should run the volatility-aware path and expose mapped labels."""

    df = _create_sample_ohlcv_dataframe()
    stub_manager = _StubKlinesManager(df)

    monkeypatch.setattr(
        "src.training.steps.pre_training.multi_horizon_profit_labeler.get_klines_manager",
        lambda data_dir: stub_manager,
    )

    config = MultiHorizonConfig(min_data_points=100)
    labeler = MultiHorizonProfitLabeler(config)

    async def fake_report(*args, **kwargs):
        return {"status": "ok"}

    monkeypatch.setattr(
        MultiHorizonProfitLabeler,
        "_generate_comprehensive_report",
        fake_report,
        raising=False,
    )

    captured = {}

    def fake_generate_labels(market_data: pd.DataFrame):
        captured["market_data"] = market_data
        return _DummyLabelingResult(market_data.index)

    monkeypatch.setattr(labeler.volatility_labeler, "generate_labels", fake_generate_labels)

    artifacts = await labeler.execute_labeling("ETHUSDT", "binance", "15m", "unused")

    assert "market_data" in captured  # ensure volatility-aware pipeline executed

    mh_result = artifacts["multi_horizon_labeling_result"]
    assert "labeled_data" in mh_result and not mh_result["labeled_data"].empty
    assert "labels" in mh_result and mh_result["labels"].equals(mh_result["labeled_data"])
    expected_targets = {
        "immediate_opportunity",
        "short_term_opportunity",
        "leverage_adjusted_score",
    }
    assert expected_targets.issubset(set(mh_result["labeled_data"].columns))
    assert artifacts["labeling_report"]["status"] == "ok"
    assert stub_manager.calls, "klines manager should be used during execute_labeling"
