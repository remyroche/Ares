import sys
import types

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

if "cvxpy" not in sys.modules:
    dummy_cvxpy = types.ModuleType("cvxpy")

    class _DummyProblem:
        def __init__(self):
            self.status = "optimal"

        def solve(self, *args, **kwargs):
            return None

    dummy_cvxpy.Variable = lambda *args, **kwargs: None
    dummy_cvxpy.Constraint = object
    dummy_cvxpy.Maximize = lambda expr: None
    dummy_cvxpy.Problem = lambda objective, constraints: _DummyProblem()
    dummy_cvxpy.Parameter = lambda *args, **kwargs: None
    dummy_cvxpy.sum = lambda *args, **kwargs: 0
    dummy_cvxpy.multiply = lambda *args, **kwargs: 0
    dummy_cvxpy.diag = lambda *args, **kwargs: 0
    dummy_cvxpy.CBC = "CBC"
    dummy_cvxpy.OPTIMAL = "optimal"

    sys.modules["cvxpy"] = dummy_cvxpy

if "pymc" not in sys.modules:
    pymc_stub = types.ModuleType("pymc")

    class _DummyModel:
        pass

    pymc_stub.Model = _DummyModel
    pymc_stub.sample = lambda *args, **kwargs: None

    sys.modules["pymc"] = pymc_stub

if "aesara" not in sys.modules:
    aesara_stub = types.ModuleType("aesara")
    tensor_stub = types.ModuleType("aesara.tensor")

    def _dummy_attr(*args, **kwargs):
        return None

    tensor_stub.__getattr__ = lambda name: _dummy_attr

    sys.modules["aesara"] = aesara_stub
    sys.modules["aesara.tensor"] = tensor_stub

if "arviz" not in sys.modules:
    sys.modules["arviz"] = types.ModuleType("arviz")

from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.pipeline import (
    PipelineConfig,
    CrossTimeframePipeline,
)
from src.training.steps.pre_training.interaction_feature_generator.cross_timeframe_generation.config import (
    SessionConfig,
)


def _build_trading_sessions(start_dates):
    tz = ZoneInfo("America/New_York")
    indices = []
    for day in start_dates:
        day = pd.Timestamp(day).tz_localize(tz)
        session_start = day.replace(hour=9, minute=30)
        session_end = day.replace(hour=16, minute=0)
        session_index = pd.date_range(
            start=session_start,
            end=session_end,
            freq="5min",
            tz=tz,
        )
        indices.append(session_index)
    return indices


def _build_sample_data():
    session_days = ["2023-03-10", "2023-03-13"]
    indices = _build_trading_sessions(session_days)
    full_index = indices[0].append(indices[1])

    ohlcv = pd.DataFrame(
        {
            "open": np.arange(len(full_index)),
            "high": np.arange(len(full_index)) + 0.5,
            "low": np.arange(len(full_index)) - 0.5,
            "close": np.arange(len(full_index)) + 1.0,
            "volume": np.random.randint(100, 200, size=len(full_index)),
        },
        index=full_index,
    )

    optional_index = full_index[::2]
    optional = pd.DataFrame(
        {
            "depth": np.linspace(0.0, 1.0, num=len(optional_index)),
        },
        index=optional_index,
    )

    return ohlcv, optional


def test_dst_sessions_and_alignment_contiguous():
    ohlcv, optional = _build_sample_data()
    config = PipelineConfig(
        session=SessionConfig(
            base_timeframe_minutes=5,
            session_start_hour=9,
            session_end_hour=16,
            dst_handling=True,
        )
    )
    pipeline = CrossTimeframePipeline(config)

    sessionized = pipeline._sessionize_and_align(ohlcv, {"depth": optional})

    sessions = sessionized["sessions"]
    assert len(sessions) == 2
    offsets = [session["open_dt"].utcoffset() for session in sessions]
    assert offsets[0] != offsets[1]
    assert offsets[0] == pd.Timedelta(hours=-5)
    assert offsets[1] == pd.Timedelta(hours=-4)

    aligned = sessionized["aligned_data"]
    assert "session_id" in aligned.columns

    freq_delta = pd.Timedelta(minutes=config.base_timeframe_minutes)
    for _, group in aligned.groupby("session_id"):
        diffs = group.index.to_series().diff().dropna().unique()
        assert len(diffs) == 1
        assert diffs[0] == freq_delta

    depth_columns = [col for col in aligned.columns if col.startswith("depth")]
    assert depth_columns
    depth_col = depth_columns[0]

    for _, group in aligned.groupby("session_id"):
        series = group[depth_col]
        if series.notna().any():
            first_valid = series[series.notna()].index[0]
            assert series.loc[first_valid:].isna().sum() == 0
