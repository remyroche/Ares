"""Tests for the multi-horizon profit labeler data loading and execution pipeline."""
from datetime import datetime, timedelta
import builtins

import numpy as np
import pandas as pd
import pytest
from typing import Optional

from src.training.steps.pre_training.multi_horizon_profit_labeler import (
    MultiHorizonConfig,
    MultiHorizonProfitLabeler,
    MultiHorizonProfitLabelerComponent,
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
        self.window_calls: list[tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = []

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
        if start_date or end_date:
            start_ts = pd.to_datetime(start_date) if start_date else None
            end_ts = pd.to_datetime(end_date) if end_date else None
            self.window_calls.append((start_ts, end_ts))
            df = self._dataframe.copy()
            if start_ts is not None:
                df = df[df['timestamp'] >= start_ts]
            if end_ts is not None:
                df = df[df['timestamp'] <= end_ts]
            return df
        return self._dataframe.copy()

    def get_data_info(self, symbol: str, interval: str, data_type: str = "raw") -> dict:
        start = self._dataframe['timestamp'].min()
        end = self._dataframe['timestamp'].max()
        return {
            'available': True,
            'files_count': 1,
            'total_records': int(len(self._dataframe)),
            'file_size_mb': 1.0,
            'date_range': {
                'start': start.isoformat() if hasattr(start, 'isoformat') else start,
                'end': end.isoformat() if hasattr(end, 'isoformat') else end,
            },
        }


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
    batches = []
    async for batch in labeler._load_market_data("ETHUSDT", "binance", "15m", "unused"):
        batches.append(batch)

    assert batches, "Expected at least one batch"
    loaded = pd.concat(batches)

    assert not loaded.empty
    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.is_monotonic_increasing
    for column in ("open", "high", "low", "close", "volume"):
        assert column in loaded.columns
    assert stub_manager.calls, "klines manager should be queried"


@pytest.mark.anyio("asyncio")
async def test_load_market_data_respects_batch_size_and_window(monkeypatch):
    df = _create_sample_ohlcv_dataframe(120)
    stub_manager = _StubKlinesManager(df)

    monkeypatch.setattr(
        "src.training.steps.pre_training.multi_horizon_profit_labeler.get_klines_manager",
        lambda data_dir: stub_manager,
    )

    labeler = MultiHorizonProfitLabeler(
        MultiHorizonConfig(
            min_data_points=50,
            market_data_batch_size=30,
            market_data_window_days=1,
        )
    )

    batches = []
    async for batch in labeler._load_market_data(
        "ETHUSDT",
        "binance",
        "15m",
        "unused",
        batch_size=30,
        window_days=1,
    ):
        batches.append(batch)

    assert stub_manager.window_calls, "Expected windowed reads when window_days is set"
    assert len(batches) >= 2, "Windowing should produce multiple batches"
    expected = labeler._prepare_market_data_frame(df)
    combined = pd.concat(batches).sort_index()
    combined = combined[~combined.index.duplicated(keep='first')]
    pd.testing.assert_frame_equal(combined, expected)


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
        self.normalization_factors = {}
        self.sigma_payoffs = pd.DataFrame(index=index)
        self.processing_time = 0.5
        self.n_samples = len(index)
        self.n_targets = self.labels.shape[1]
        self.n_horizons = 3
        self.execution_timing = {
            "signal_to_execution_delay_bars": 1,
            "entry_price_source": "next_open",
        }


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
    assert "execution_timing" in mh_result
    assert mh_result["execution_timing"]["signal_to_execution_delay_bars"] == 1
    assert mh_result["metadata"]["execution_timing"]["signal_to_execution_delay_bars"] == 1
    std_metadata = artifacts["standardized_output"]["metadata"]
    assert std_metadata["execution_timing"]["signal_to_execution_delay_bars"] == 1
    expected_targets = {
        "immediate_opportunity",
        "short_term_opportunity",
        "leverage_adjusted_score",
    }
    assert expected_targets.issubset(set(mh_result["labeled_data"].columns))
    assert artifacts["labeling_report"]["status"] == "ok"
    assert stub_manager.calls, "klines manager should be used during execute_labeling"


@pytest.mark.anyio("asyncio")
async def test_execute_labeling_chunked_matches_full(monkeypatch):
    df = _create_sample_ohlcv_dataframe(180)

    async def fake_report(*args, **kwargs):
        return {"status": "ok"}

    def setup_labeler(batch_size: Optional[int]) -> MultiHorizonProfitLabeler:
        stub_manager = _StubKlinesManager(df)
        monkeypatch.setattr(
            "src.training.steps.pre_training.multi_horizon_profit_labeler.get_klines_manager",
            lambda data_dir: stub_manager,
        )

        config = MultiHorizonConfig(min_data_points=50)
        config.market_data_batch_size = batch_size
        labeler = MultiHorizonProfitLabeler(config)

        monkeypatch.setattr(
            MultiHorizonProfitLabeler,
            "_generate_comprehensive_report",
            fake_report,
            raising=False,
        )

        def fake_generate_labels(market_data: pd.DataFrame) -> _DummyLabelingResult:
            return _DummyLabelingResult(market_data.index)

        monkeypatch.setattr(labeler.volatility_labeler, "generate_labels", fake_generate_labels)
        return labeler

    np.random.seed(7)
    baseline_labeler = setup_labeler(batch_size=None)
    baseline_artifacts = await baseline_labeler.execute_labeling("ETHUSDT", "binance", "15m", "unused")

    np.random.seed(7)
    chunked_labeler = setup_labeler(batch_size=45)
    chunked_artifacts = await chunked_labeler.execute_labeling("ETHUSDT", "binance", "15m", "unused")

    base_labels = baseline_artifacts["multi_horizon_labeling_result"]["labeled_data"]
    chunk_labels = chunked_artifacts["multi_horizon_labeling_result"]["labeled_data"]
    pd.testing.assert_frame_equal(chunk_labels, base_labels)

    chunk_batches = chunked_artifacts["multi_horizon_labeling_result"].get("market_data_batches")
    assert chunk_batches is not None and len(chunk_batches) >= 2


@pytest.mark.anyio("asyncio")
async def test_component_metadata_records_outcome_save_failure(monkeypatch):
    """Component metadata should flag when the persistent outcome cannot be saved."""

    component = MultiHorizonProfitLabelerComponent()

    async def fake_execute_labeling(*args, **kwargs):
        return {
            "multi_horizon_labeling_result": {"labels": []},
            "labeling_report": {"status": "ok"},
        }

    monkeypatch.setattr(component.labeler, "execute_labeling", fake_execute_labeling)

    real_open = builtins.open

    def failing_open(file, mode="r", *args, **kwargs):
        if "market_analysis_multi_horizon_profit_labeler_outcome" in str(file) and "w" in mode:
            raise OSError("disk full")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", failing_open)

    result = await component.execute(
        None,
        {"symbol": "ETHUSDT", "exchange": "binance", "timeframe": "1h"},
    )

    assert result.success is True
    assert result.metadata.get("artifacts_saved") is False
    assert result.metadata.get("artifact_save_error") == "disk full"


def test_calculate_target_distribution_includes_numeric_columns():
    """Numeric label columns should populate distribution statistics."""

    labeler = MultiHorizonProfitLabeler(MultiHorizonConfig(min_data_points=10))
    labels_df = pd.DataFrame(
        {
            "numeric_float": pd.Series([0.1, 0.2, np.nan, -0.1, 0.0], dtype=float),
            "numeric_int": pd.Series([1, 2, 3, 4, np.nan]),
            "non_numeric": pd.Series(["a", "b", "c", None, "d"]),
        }
    )

    distribution = labeler._calculate_target_distribution(labels_df)

    assert "numeric_float" in distribution
    assert "numeric_int" in distribution
    assert "non_numeric" not in distribution
    for metrics in (distribution["numeric_float"], distribution["numeric_int"]):
        assert set(metrics) >= {"mean", "std", "min", "max", "non_null_count", "class_balance"}
