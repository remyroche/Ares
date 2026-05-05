import numpy as np
import pandas as pd
from numpy.typing import NDArray

from extreme_price_movements.periods_symbols_management import (
    EventSchema,
    PlannerState,
    SamplingPolicy,
    SlicePlannerConfig,
    SymbolPolicy,
    _resolve_outer_partition,
    apply_sampling_policy,
    validate_events,
)


def test_validate_events_deduplicates_same_base_quote_variants():
    events = pd.DataFrame(
        {
            "event_id": np.arange(6, dtype=np.int64),
            "symbol": [
                "ETH/USDT",
                "ETH/USDT",
                "ETH/USDC",
                "BTC/USDT",
                "BTC/USDC",
                "SOL/EUR",
            ],
            "t0": pd.date_range("2026-01-01", periods=6, freq="H", tz="UTC"),
            "t1": pd.date_range("2026-01-01", periods=6, freq="H", tz="UTC"),
        }
    )
    cfg = SlicePlannerConfig.fast_defaults(schema=EventSchema())
    out = validate_events(events, cfg.schema, cfg)

    assert "ETH/USDT" in set(out["symbol"].astype(str))
    assert "ETH/USDC" not in set(out["symbol"].astype(str))
    assert "BTC/USDT" in set(out["symbol"].astype(str))
    assert "BTC/USDC" not in set(out["symbol"].astype(str))
    assert "SOL/EUR" in set(out["symbol"].astype(str))


def test_subset_symbol_policy_rotates_symbols_across_outer_folds():
    symbol_codes = np.array([0, 1, 2], dtype=np.int32)
    state = PlannerState(
        events=pd.DataFrame(),
        schema=EventSchema(),
        t0_ns=np.array([0], dtype=np.int64),
        t1_ns=np.array([0], dtype=np.int64),
        symbol_codes=symbol_codes,
        symbol_labels=np.array(["A", "B", "C"], dtype=object),
        group_codes={},
        group_labels={},
        overlap=None,  # type: ignore[arg-type]
    )

    policy = SymbolPolicy(
        mode="subset_symbols",
        subset_fraction=0.66,
        min_symbols_per_split=2,
        random_state=42,
    )

    partitions = [
        _resolve_outer_partition(
            state,
            policy,
            fold_id,
            np.array([0, 1, 2]),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )
        for fold_id in range(3)
    ]
    subsets = [tuple(part.train_symbol_codes.tolist()) for part in partitions]

    assert len({tuple(p) for p in subsets}) == 3
    # Each fold should get exactly two symbols with fast-like subset size.
    assert all(len(p) == 2 for p in subsets)
    # Across three folds with three symbols and size-2 subsets, every symbol appears.
    all_symbols = {sym for subset in subsets for sym in subset}
    assert all_symbols == {0, 1, 2}


def _make_state_for_sampling(symbol_codes: NDArray[np.integer]) -> PlannerState:
    n = int(symbol_codes.size)
    return PlannerState(
        events=pd.DataFrame(),
        schema=EventSchema(),
        t0_ns=np.zeros(n, dtype=np.int64),
        t1_ns=np.zeros(n, dtype=np.int64),
        symbol_codes=np.asarray(symbol_codes, dtype=np.int32),
        symbol_labels=np.array([], dtype=object),
        group_codes={},
        group_labels={},
        overlap=None,  # type: ignore[arg-type]
    )


def test_symbol_sampling_rotates_for_train_and_keeps_min_symbols():
    state = _make_state_for_sampling(np.array([0, 0, 1, 1, 2, 2], dtype=np.int32))
    sampling = SamplingPolicy(mode="subsample", symbol_fraction=0.5, random_state=0)
    policy = SymbolPolicy(
        mode="subset_symbols", min_symbols_per_split=2, random_state=0
    )
    fold0, _ = apply_sampling_policy(
        state, np.arange(6), sampling, policy, {"split": "train", "outer_fold": 0}
    )
    fold1, _ = apply_sampling_policy(
        state, np.arange(6), sampling, policy, {"split": "train", "outer_fold": 1}
    )
    fold2, _ = apply_sampling_policy(
        state, np.arange(6), sampling, policy, {"split": "train", "outer_fold": 2}
    )

    syms0 = set(np.unique(state.symbol_codes[fold0]).tolist())
    syms1 = set(np.unique(state.symbol_codes[fold1]).tolist())
    syms2 = set(np.unique(state.symbol_codes[fold2]).tolist())

    assert len(syms0) == 2
    assert len(syms1) == 2
    assert len(syms2) == 2
    assert syms0 != syms1 != syms2
