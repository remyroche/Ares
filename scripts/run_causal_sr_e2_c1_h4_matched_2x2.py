#!/usr/bin/env python3
"""Matched E2/C1 population × continuation-contract H4 falsification.

This is an offline, research-only 2×2 designed to answer one narrow question:
did the earlier H4 result depend primarily on its E2-selected population, or
on the archived one-interval L1 continuation contract rather than the repaired
latched L2 contract?

Every arm uses the same frozen rich exact-one-minute policy, decision +5 minute
entry, one 100-bps policy cost, and normal chronological portfolio auction.
Only these two dimensions change:

* population: original target-free E2 q50 selection vs target-free C1 dual-40;
* controller: archived L1 one-interval target vs repaired L2 latched target.

No target/path field participates in population construction.  Minute paths are
joined only after both target-free populations have been persisted.  This is a
falsification report, not a live or canonical release.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_h4_overlay_research import (  # noqa: E402
    replay_exact_1m_h4_giveback20,
)
from scripts import run_strict_r3_p8u_15m_continuation_feature_contract_ablation as legacy_h4  # noqa: E402
from scripts.run_causal_sr_c1_exact1m_parent_portfolio_replay import (  # noqa: E402
    DEFAULT_MINUTE_ROOT,
    DEFAULT_POLICY,
    _outcomes,
    _portfolio,
)
from scripts.run_causal_sr_c1_h4_exact1m_transfer_replay import (  # noqa: E402
    _fields as legacy_fields,
    _state_features,
)
from scripts.run_strict_r3_p8u_15m_continuation_postfs_hpo import (  # noqa: E402
    SPECS as LEGACY_SPECS,
)
from scripts.run_strict_r3_p8u_exact_1m_simple_policy_optimiser import (  # noqa: E402
    ExactPaths,
    _materialize_exact_paths,
    _load_policy,
)


E2_SELECTION = ROOT / (
    "data_perp/artifacts/strict_r3_p8u_15m_entry_e2_demotion_residual_"
    "20260830_v1/E2_q50_agreement_selection_target_free.parquet"
)
C1_ROUTE = ROOT / (
    "data_perp/artifacts/causal_sr_c1_expanded_h4_route_t40_20260831_v1/"
    "target_free_c1_dual_route.parquet"
)
C1_LATCH_ROOT = ROOT / "data_perp/artifacts/causal_sr_c1_h4_expanded_support_20260831_v1"
LEGACY_FEATURE_CONTRACT = ROOT / (
    "data_perp/artifacts/strict_r3_p8u_15m_continuation_feature_contract_"
    "20260830_v2/stable_selected_features.parquet"
)
DEFAULT_OUT = ROOT / "data_perp/artifacts/causal_sr_e2_c1_h4_matched_2x2_20260831_v1"

EVAL_START = pd.Timestamp("2026-06-01T00:00:00Z")
EVAL_END = pd.Timestamp("2026-09-01T00:00:00Z")
SEED = 1729
H12 = pd.Timedelta(hours=12)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_frozen_parquet(path: Path) -> pd.DataFrame:
    """Read a single immutable Parquet receipt without dataset discovery.

    Some macOS/PyArrow dataset-wrapper reads intermittently misinterpret this
    small standalone receipt as a buffer after large exact-path allocations.
    ``ParquetFile`` reads the same frozen bytes directly and preserves its
    schema/order without any dataset inference.
    """
    return pq.ParquetFile(str(path)).read().to_pandas()


def _utc(values: object) -> pd.Series | pd.Timestamp:
    return pd.to_datetime(values, utc=True, errors="raise")


def _valid_window(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    result = frame.loc[
        frame[column].ge(EVAL_START) & frame[column].lt(EVAL_END)
    ].copy()
    if result.empty:
        raise RuntimeError("target-free route has no June--August 2026 rows")
    return result


def _load_e2_population(path: Path) -> pd.DataFrame:
    """Read the archived target-free E2 selection before touching outcomes."""
    required = {
        "candidate_id", "__decision_ts__", "__symbol__", "bcf_final_score",
        "bcf_mc1_expected_bps", "current_mc1_expected_bps",
    }
    frame = pd.read_parquet(path).copy()
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AssertionError(f"E2 selection is missing target-free fields: {missing}")
    forbidden = {
        "net_bps", "gross_bps", "outcome", "exit", "path", "label", "target",
        "portfolio_accepted", "realized",
    }
    observed = {column.lower() for column in frame.columns}
    if observed.intersection(forbidden):
        raise AssertionError("E2 target-free selection unexpectedly contains outcome fields")
    result = pd.DataFrame({
        "candidate_id": frame["candidate_id"].astype(str),
        "timestamp": _utc(frame["__decision_ts__"]),
        "symbol": frame["__symbol__"].astype(str),
        "side_name": "long",
        "bcf_final_score": pd.to_numeric(frame["bcf_final_score"], errors="raise"),
        "bcf_mc1_expected_bps": pd.to_numeric(frame["bcf_mc1_expected_bps"], errors="raise"),
        "current_mc1_expected_bps": pd.to_numeric(frame["current_mc1_expected_bps"], errors="raise"),
    })
    result["entry_ts"] = result["timestamp"] + pd.Timedelta(minutes=5)
    result["auction_priority_bps"] = result["bcf_mc1_expected_bps"]
    result["population"] = "E2_original_selection"
    result = _valid_window(result, "timestamp")
    if result["candidate_id"].duplicated().any():
        raise AssertionError("archived E2 selection duplicates candidate identity")
    return result.loc[:, [
        "population", "candidate_id", "timestamp", "entry_ts", "symbol", "side_name",
        "bcf_final_score", "bcf_mc1_expected_bps", "current_mc1_expected_bps",
        "auction_priority_bps",
    ]].sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True)


def _load_c1_population(path: Path) -> pd.DataFrame:
    """Read the repaired target-free C1 dual-40 population before paths."""
    required = {
        "candidate_id", "timestamp", "entry_ts", "symbol", "side_name",
        "bcf_final_score", "bcf_mc1_expected_bps", "current_mc1_expected_bps",
        "auction_priority_bps", "admission_threshold_bps",
    }
    frame = pd.read_parquet(path).copy()
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise AssertionError(f"C1 route is missing target-free fields: {missing}")
    result = frame.loc[:, [
        "candidate_id", "timestamp", "entry_ts", "symbol", "side_name", "bcf_final_score",
        "bcf_mc1_expected_bps", "current_mc1_expected_bps", "auction_priority_bps",
        "admission_threshold_bps",
    ]].copy()
    result["candidate_id"] = result["candidate_id"].astype(str)
    result["timestamp"] = _utc(result["timestamp"])
    result["entry_ts"] = _utc(result["entry_ts"])
    result["symbol"] = result["symbol"].astype(str)
    result["side_name"] = result["side_name"].astype(str).str.lower().str.strip()
    for column in ("bcf_final_score", "bcf_mc1_expected_bps", "current_mc1_expected_bps", "auction_priority_bps"):
        result[column] = pd.to_numeric(result[column], errors="raise")
    if not result["admission_threshold_bps"].eq(40.0).all():
        raise AssertionError("C1 source is not the declared dual-40 route")
    result = result.loc[result["side_name"].eq("long")].copy()
    result = _valid_window(result, "timestamp")
    result["population"] = "C1_dual40"
    if result["candidate_id"].duplicated().any():
        raise AssertionError("C1 target-free route duplicates candidate identity")
    return result.loc[:, [
        "population", "candidate_id", "timestamp", "entry_ts", "symbol", "side_name",
        "bcf_final_score", "bcf_mc1_expected_bps", "current_mc1_expected_bps",
        "auction_priority_bps",
    ]].sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True)


def _union_path_population(populations: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one path request per identity while preserving population routes."""
    frame = pd.concat(populations.values(), ignore_index=True)
    identity = ["candidate_id", "timestamp", "entry_ts", "symbol", "side_name"]
    for candidate, group in frame.groupby("candidate_id", sort=False):
        if len(group) > 1 and group.loc[:, identity[1:]].drop_duplicates().shape[0] != 1:
            raise AssertionError(f"cross-population candidate identity mismatch: {candidate}")
    return frame.loc[:, identity].drop_duplicates("candidate_id", keep="first").sort_values(
        ["timestamp", "candidate_id"], kind="stable"
    ).reset_index(drop=True)


def _windowed_exact_paths(
    population: pd.DataFrame,
    *,
    minute_root: Path,
    workers: int,
) -> tuple[ExactPaths, pd.DataFrame, pd.DataFrame]:
    """Materialise only the exact ATR-plus-H12 intervals candidates require.

    The generic exact-path helper is intentionally conservative and reads the
    entire min-to-max timestamp span for each symbol.  That is appropriate for
    dense policy HPO, but an E2/C1 falsification route has sparse entries over
    three months.  Here, candidate windows are coalesced only when their
    14-hour ATR warm-up and H12 paths overlap.  Each coalesced sub-population is
    still passed to the canonical exact materialiser, so price handling, ATR,
    coverage, and source-error semantics remain identical.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lookback = pd.Timedelta(hours=14, minutes=1)
    horizon = pd.Timedelta(minutes=719)
    groups = [(str(symbol), group.reset_index(drop=True)) for symbol, group in population.groupby("symbol", sort=True)]

    def windows(group: pd.DataFrame) -> list[pd.DataFrame]:
        records = group.copy()
        records["__window_start__"] = _utc(records["timestamp"]) - lookback
        records["__window_end__"] = _utc(records["entry_ts"]) + horizon
        records = records.sort_values(["__window_start__", "__window_end__", "candidate_id"], kind="stable")
        parts: list[list[int]] = []
        current: list[int] = []
        current_end: pd.Timestamp | None = None
        for index, row in records.iterrows():
            start = pd.Timestamp(row["__window_start__"])
            end = pd.Timestamp(row["__window_end__"])
            if current_end is None or start <= current_end:
                current.append(int(index))
                current_end = end if current_end is None else max(current_end, end)
            else:
                parts.append(current)
                current = [int(index)]
                current_end = end
        if current:
            parts.append(current)
        return [records.loc[indexes].drop(columns=["__window_start__", "__window_end__"]).reset_index(drop=True) for indexes in parts]

    def one(symbol: str, group: pd.DataFrame):
        valid_parts: list[ExactPaths] = []
        coverage_parts: list[pd.DataFrame] = []
        invalid_parts: list[pd.DataFrame] = []
        for ordinal, window in enumerate(windows(group), start=1):
            try:
                paths, coverage, invalid = _materialize_exact_paths(window, minute_root=minute_root)
                coverage = coverage.copy()
                coverage["coalesced_window_ordinal"] = ordinal
                coverage["coalesced_window_start"] = pd.to_datetime(window["timestamp"], utc=True).min() - lookback
                coverage["coalesced_window_end"] = pd.to_datetime(window["entry_ts"], utc=True).max() + horizon
                valid_parts.append(paths)
                coverage_parts.append(coverage)
                if not invalid.empty:
                    invalid_parts.append(invalid)
            except Exception as exc:
                bad = window.copy()
                bad["outcome_invalid_reason"] = f"exact_1m_source_error:{type(exc).__name__}"
                invalid_parts.append(bad)
                coverage_parts.append(pd.DataFrame([{
                    "symbol": symbol, "candidate_rows": int(len(window)), "valid_rows": 0,
                    "reason": f"exact_1m_source_error:{type(exc).__name__}",
                    "source_error": str(exc), "coalesced_window_ordinal": ordinal,
                    "coalesced_window_start": pd.to_datetime(window["timestamp"], utc=True).min() - lookback,
                    "coalesced_window_end": pd.to_datetime(window["entry_ts"], utc=True).max() + horizon,
                }]))
        if valid_parts:
            rows = pd.concat([part.rows for part in valid_parts], ignore_index=True)
            paths = ExactPaths(
                rows=rows,
                entry=np.concatenate([part.entry for part in valid_parts]),
                atr=np.concatenate([part.atr for part in valid_parts]),
                high=np.concatenate([part.high for part in valid_parts]),
                low=np.concatenate([part.low for part in valid_parts]),
                close=np.concatenate([part.close for part in valid_parts]),
            )
        else:
            paths = None
        coverage = pd.concat(coverage_parts, ignore_index=True)
        invalid = pd.concat(invalid_parts, ignore_index=True) if invalid_parts else pd.DataFrame(
            columns=[*group.columns, "outcome_invalid_reason"]
        )
        return symbol, paths, coverage, invalid

    completed: dict[str, tuple[ExactPaths | None, pd.DataFrame, pd.DataFrame]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="exact1m-windowed") as executor:
        futures = {executor.submit(one, symbol, group): symbol for symbol, group in groups}
        for ordinal, future in enumerate(as_completed(futures), start=1):
            symbol, paths, coverage, invalid = future.result()
            completed[symbol] = (paths, coverage, invalid)
            if ordinal == 1 or ordinal % 10 == 0 or ordinal == len(futures):
                print(f"windowed exact-1m source {ordinal}/{len(futures)} symbols completed", flush=True)
    valid_parts: list[ExactPaths] = []
    coverage_parts = []
    invalid_parts = []
    for symbol, _ in groups:
        paths, coverage, invalid = completed[symbol]
        if paths is not None:
            valid_parts.append(paths)
        coverage_parts.append(coverage)
        if not invalid.empty:
            invalid_parts.append(invalid)
    if not valid_parts:
        raise RuntimeError("no exact paths after target-free routing")
    rows = pd.concat([part.rows for part in valid_parts], ignore_index=True)
    paths = ExactPaths(
        rows=rows,
        entry=np.concatenate([part.entry for part in valid_parts]),
        atr=np.concatenate([part.atr for part in valid_parts]),
        high=np.concatenate([part.high for part in valid_parts]),
        low=np.concatenate([part.low for part in valid_parts]),
        close=np.concatenate([part.close for part in valid_parts]),
    )
    coverage = pd.concat(coverage_parts, ignore_index=True)
    invalid = pd.concat(invalid_parts, ignore_index=True) if invalid_parts else pd.DataFrame(
        columns=[*population.columns, "outcome_invalid_reason"]
    )
    valid_ids = set(paths.rows["candidate_id"].astype(str))
    invalid_ids = set(invalid["candidate_id"].astype(str))
    requested_ids = set(population["candidate_id"].astype(str))
    if paths.rows["candidate_id"].duplicated().any() or valid_ids.intersection(invalid_ids):
        raise AssertionError("windowed exact materialisation duplicated or conflicted identities")
    if valid_ids.union(invalid_ids) != requested_ids:
        raise AssertionError("windowed exact materialisation lost target-free identities")
    return paths, coverage, invalid


def _reuse_exact_paths(
    root: Path,
    population: pd.DataFrame,
) -> tuple[ExactPaths, pd.DataFrame, pd.DataFrame]:
    """Load a previously completed exact-path stage after exact identity checks.

    This is an execution-time reuse mechanism only.  It accepts a prior
    interrupted run *only* when its target-free union, valid path identities,
    and invalid identities exactly partition the current requested union.  The
    continuation models are still refit and replayed in the new immutable run.
    """
    required = (
        "target_free_union_path_population.parquet",
        "valid_exact_paths_rows.parquet",
        "invalid_outcomes_after_target_free_route.parquet",
        "exact_1m_source_coverage.parquet",
        "exact_paths.npz",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"reuse root lacks required exact-path artifacts: {missing}")
    identity = ["candidate_id", "timestamp", "entry_ts", "symbol", "side_name"]
    requested = population.loc[:, identity].copy()
    prior_requested = pd.read_parquet(root / "target_free_union_path_population.parquet")
    missing_identity = sorted(set(identity).difference(prior_requested.columns))
    if missing_identity:
        raise AssertionError(f"reused target-free union lacks {missing_identity}")
    prior_requested = prior_requested.loc[:, identity].copy()
    for frame in (requested, prior_requested):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["timestamp"] = _utc(frame["timestamp"])
        frame["entry_ts"] = _utc(frame["entry_ts"])
        frame["symbol"] = frame["symbol"].astype(str)
        frame["side_name"] = frame["side_name"].astype(str)
    requested = requested.sort_values(["candidate_id"], kind="stable").reset_index(drop=True)
    prior_requested = prior_requested.sort_values(["candidate_id"], kind="stable").reset_index(drop=True)
    if not requested.equals(prior_requested):
        raise AssertionError("reused exact-path target-free union does not match current request")
    rows = pd.read_parquet(root / "valid_exact_paths_rows.parquet").copy()
    invalid = pd.read_parquet(root / "invalid_outcomes_after_target_free_route.parquet").copy()
    rows["candidate_id"] = rows["candidate_id"].astype(str)
    invalid["candidate_id"] = invalid["candidate_id"].astype(str)
    valid_ids = set(rows["candidate_id"])
    invalid_ids = set(invalid["candidate_id"])
    requested_ids = set(requested["candidate_id"])
    if rows["candidate_id"].duplicated().any() or valid_ids.intersection(invalid_ids):
        raise AssertionError("reused exact-path stage has duplicate or conflicting identities")
    if valid_ids.union(invalid_ids) != requested_ids:
        raise AssertionError("reused exact-path stage does not partition the target-free request")
    payload = np.load(root / "exact_paths.npz", allow_pickle=True)
    payload_ids = payload["candidate_id"].astype(str)
    if not np.array_equal(payload_ids, rows["candidate_id"].to_numpy(str)):
        raise AssertionError("reused exact-path arrays do not align with persisted valid identities")
    entry, atr, high, low, close = (payload[field] for field in ("entry", "atr", "high", "low", "close"))
    n = len(rows)
    if not (len(entry) == len(atr) == len(high) == len(low) == len(close) == n):
        raise AssertionError("reused exact-path arrays have inconsistent cardinality")
    if high.shape != (n, 720) or low.shape != (n, 720) or close.shape != (n, 720):
        raise AssertionError("reused exact-path arrays do not satisfy the frozen 720-minute horizon")
    paths = ExactPaths(
        rows=rows,
        entry=np.asarray(entry, dtype=float),
        atr=np.asarray(atr, dtype=float),
        high=np.asarray(high, dtype=np.float32),
        low=np.asarray(low, dtype=np.float32),
        close=np.asarray(close, dtype=np.float32),
    )
    return paths, pd.read_parquet(root / "exact_1m_source_coverage.parquet"), invalid


def _raw_h4_states(paths, *, params: object, median_atr_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay parent states once, and prove parity with parent outcomes later."""
    state_parts: list[pd.DataFrame] = []
    parent_records: list[dict[str, object]] = []
    for position, row in paths.rows.iterrows():
        trace = replay_exact_1m_h4_giveback20(
            entry_price=float(paths.entry[position]), signal_atr=float(paths.atr[position]),
            entry_ts=row["entry_ts"], highs=paths.high[position], lows=paths.low[position],
            closes=paths.close[position], params=params, median_atr_fraction=float(median_atr_fraction),
            mc1_expected_bps=0.0, state_decider=None, giveback_tighten=0.20, emit_states=True,
        )
        parent_records.append({
            "candidate_id": str(row["candidate_id"]),
            "parent_h4_engine_net_bps": float(trace["net_bps"]),
            "parent_h4_engine_exit_ts": pd.Timestamp(trace["exit_timestamp"]),
            "parent_h4_engine_exit_reason": str(trace["exit_reason"]),
        })
        if trace["states"]:
            part = pd.DataFrame(trace["states"])
            part["candidate_id"] = str(row["candidate_id"])
            part["__symbol__"] = str(row["symbol"])
            part["entry_decision_ts"] = pd.Timestamp(row["timestamp"])
            part["entry_price"] = float(paths.entry[position])
            part["signal_atr"] = float(paths.atr[position])
            part["state_bar_15m"] = np.arange(len(part), dtype=np.int16)
            part["current_PnL"] = (
                part["current_pnl_atr"].to_numpy(float) * float(paths.atr[position])
                / float(paths.entry[position]) * 10_000.0 - 100.0
            )
            state_parts.append(part)
    if not state_parts:
        raise RuntimeError("exact paths produced no parent-observable H4 states")
    states = pd.concat(state_parts, ignore_index=True)
    keys = ["candidate_id", "state_decision_ts", "state_bar_15m"]
    if states.duplicated(keys).any():
        raise AssertionError("exact parent states duplicate identity")
    return states, pd.DataFrame(parent_records)


def _legacy_actions(states: pd.DataFrame, *, contract: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the original H4 L1 one-interval contract to one population."""
    old = legacy_h4._load_panel(legacy_h4.TARGET_PANEL, legacy_h4.VWAP_PANEL)
    selected_contract = contract.copy()
    required_contract = {"arm", "held_month", "position", "feature"}
    missing_contract = sorted(required_contract.difference(selected_contract.columns))
    if missing_contract:
        raise AssertionError(f"legacy feature contract missing {missing_contract}")
    required = {"entry_decision_ts", "policy_label_available_ts", "activation50_advantage_bps", "MC1_expected_bps"}
    missing = sorted(required.difference(old.columns))
    if missing:
        raise AssertionError(f"legacy H4 label panel missing {missing}")
    panel = states.copy()
    panel["entry_decision_ts"] = _utc(panel["entry_decision_ts"])
    action_parts: list[pd.DataFrame] = []
    support: list[dict[str, object]] = []
    for held_period in sorted(panel["entry_decision_ts"].dt.to_period("M").unique()):
        held = pd.Timestamp(held_period.start_time, tz="UTC")
        end = held + pd.offsets.MonthBegin(1)
        fields = tuple(
            selected_contract.loc[
                selected_contract["arm"].eq("C4_normalized_vwap_fs")
                & selected_contract["held_month"].eq(held.strftime("%Y-%m"))
            ].sort_values("position", kind="stable")["feature"].astype(str)
        )
        if len(fields) != 45 or len(set(fields)) != len(fields):
            raise AssertionError(f"{held:%Y-%m}: missing frozen 45-field C4 contract")
        train = old.loc[
            old["entry_decision_ts"].ge(held - pd.DateOffset(months=4))
            & old["entry_decision_ts"].lt(held)
            & old["policy_label_available_ts"].lt(held)
            & pd.to_numeric(old["MC1_expected_bps"], errors="coerce").ge(30.0)
        ].copy()
        test = panel.loc[
            panel["entry_decision_ts"].ge(held) & panel["entry_decision_ts"].lt(end)
        ].copy()
        missing = set(fields).difference(train.columns) | set(fields).difference(test.columns)
        if missing:
            raise AssertionError(f"{held:%Y-%m}: legacy fields unavailable: {sorted(missing)}")
        if train["candidate_id"].nunique() < 100:
            raise RuntimeError(f"{held:%Y-%m}: insufficient legacy H4 strict-prior support")
        spec = LEGACY_SPECS["H4_l1_d4_l15_leaf5_reg20"]
        child = max(8, int(np.ceil(len(train) * float(spec["min_child_fraction"]))))
        model = lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=int(spec["n_estimators"]),
            learning_rate=float(spec["learning_rate"]), max_depth=int(spec["max_depth"]),
            num_leaves=int(spec["num_leaves"]), min_child_samples=child,
            subsample=.80, colsample_bytree=.80, reg_lambda=float(spec["reg_lambda"]),
            random_state=SEED, n_jobs=2, verbosity=-1,
        )
        weight = 1.0 / train.groupby("candidate_id")["candidate_id"].transform("size").to_numpy(float)
        model.fit(train.loc[:, fields], train["activation50_advantage_bps"].to_numpy(float), sample_weight=weight)
        action = test.loc[:, ["candidate_id", "state_decision_ts"]].copy()
        action["h4_predicted_advantage_bps"] = model.predict(test.loc[:, fields])
        action["h4_enable"] = action["h4_predicted_advantage_bps"].ge(0.0)
        action["held_month"] = held.strftime("%Y-%m")
        action_parts.append(action)
        support.append({
            "controller": "legacy_l1_one_interval", "held_month": held.strftime("%Y-%m"),
            "train_rows": int(len(train)), "train_candidates": int(train["candidate_id"].nunique()),
            "test_states": int(len(test)), "enabled_states": int(action["h4_enable"].sum()),
        })
    result = pd.concat(action_parts, ignore_index=True)
    if result.duplicated(["candidate_id", "state_decision_ts"]).any():
        raise AssertionError("legacy H4 schedule duplicates state identity")
    return result, pd.DataFrame(support)


def _latched_fields(contracts: pd.DataFrame, held: pd.Timestamp) -> tuple[str, ...]:
    rows = contracts.loc[
        contracts["arm"].eq("F45_d4")
        & contracts["held_month"].eq(held.strftime("%Y-%m"))
    ].sort_values("position", kind="stable")
    fields = tuple(rows["feature"].astype(str))
    if len(fields) != 45 or len(set(fields)) != len(fields):
        raise AssertionError(f"{held:%Y-%m}: repaired L2 F45 contract is incomplete")
    return fields


def _fit_latched_l2(train: pd.DataFrame, fields: tuple[str, ...]) -> lgb.LGBMRegressor:
    child = max(8, int(np.ceil(len(train) * .05)))
    model = lgb.LGBMRegressor(
        objective="regression_l2", n_estimators=420, learning_rate=.025, max_depth=4,
        num_leaves=15, min_child_samples=child, subsample=.80, colsample_bytree=.80,
        reg_lambda=20.0, random_state=SEED, n_jobs=2, verbosity=-1,
    )
    weight = 1.0 / train.groupby("candidate_id")["candidate_id"].transform("size").to_numpy(float)
    model.fit(train.loc[:, fields], train["activation50_advantage_bps"].to_numpy(float), sample_weight=weight)
    return model


def _latched_actions(states: pd.DataFrame, *, label_root: Path, mfe_ready_atr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the repaired strictly-prequential latched L2 contract."""
    reference = pd.read_parquet(label_root / "all_h4_context_states_target_free.parquet")
    labels = pd.read_parquet(label_root / "latched_activation50_giveback20_labels.parquet")
    contracts = pd.read_parquet(label_root / "prequential_feature_contracts.parquet")
    keys = ["candidate_id", "state_decision_ts"]
    reference["candidate_id"] = reference["candidate_id"].astype(str)
    labels["candidate_id"] = labels["candidate_id"].astype(str)
    states = states.copy()
    states["candidate_id"] = states["candidate_id"].astype(str)
    for frame in (reference, labels, states):
        frame["state_decision_ts"] = _utc(frame["state_decision_ts"])
    reference["entry_decision_ts"] = _utc(reference["entry_decision_ts"])
    labels["policy_label_available_ts"] = _utc(labels["policy_label_available_ts"])
    train_panel = reference.merge(
        labels.loc[:, [*keys, "activation50_advantage_bps", "policy_label_available_ts"]],
        on=keys, how="inner", validate="one_to_one",
    )
    train_panel["h4_actionable_state"] = pd.to_numeric(
        train_panel["current_MFE_ATR"], errors="coerce"
    ).ge(float(mfe_ready_atr))
    states["entry_decision_ts"] = _utc(states["entry_decision_ts"])
    states["h4_actionable_state"] = pd.to_numeric(
        states["current_MFE_ATR"], errors="coerce"
    ).ge(float(mfe_ready_atr))
    action_parts: list[pd.DataFrame] = []
    support: list[dict[str, object]] = []
    for held_period in sorted(states["entry_decision_ts"].dt.to_period("M").unique()):
        held = pd.Timestamp(held_period.start_time, tz="UTC")
        end = held + pd.offsets.MonthBegin(1)
        fields = _latched_fields(contracts, held)
        train = train_panel.loc[
            train_panel["entry_decision_ts"].lt(held)
            & train_panel["policy_label_available_ts"].lt(held)
            & train_panel["h4_actionable_state"]
        ].copy()
        test = states.loc[
            states["entry_decision_ts"].ge(held) & states["entry_decision_ts"].lt(end)
        ].copy()
        missing = set(fields).difference(train.columns) | set(fields).difference(test.columns)
        if missing:
            raise AssertionError(f"{held:%Y-%m}: repaired L2 fields unavailable: {sorted(missing)}")
        actionable = test.loc[test["h4_actionable_state"]].copy()
        action = test.loc[:, ["candidate_id", "state_decision_ts"]].copy()
        action["h4_predicted_advantage_bps"] = float("-inf")
        if not actionable.empty:
            if train["candidate_id"].nunique() < 100:
                raise RuntimeError(f"{held:%Y-%m}: insufficient repaired L2 strict-prior support")
            model = _fit_latched_l2(train, fields)
            action.loc[actionable.index, "h4_predicted_advantage_bps"] = model.predict(actionable.loc[:, fields])
        action["h4_enable"] = action["h4_predicted_advantage_bps"].gt(0.0)
        action["held_month"] = held.strftime("%Y-%m")
        action_parts.append(action)
        support.append({
            "controller": "repaired_l2_latched", "held_month": held.strftime("%Y-%m"),
            "train_rows": int(len(train)), "train_candidates": int(train["candidate_id"].nunique()),
            "test_states": int(len(test)), "actionable_test_states": int(len(actionable)),
            "enabled_states": int(action["h4_enable"].sum()),
        })
    result = pd.concat(action_parts, ignore_index=True)
    if result.duplicated(["candidate_id", "state_decision_ts"]).any():
        raise AssertionError("repaired L2 H4 schedule duplicates state identity")
    return result, pd.DataFrame(support)


def _h4_outcomes(
    *,
    route: pd.DataFrame,
    paths,
    actions: pd.DataFrame,
    params: object,
    median_atr_fraction: float,
    latched: bool,
) -> pd.DataFrame:
    """Apply an H4 schedule to exact paths without changing entry identities."""
    position = pd.Series(np.arange(len(paths.rows), dtype=np.int64), index=paths.rows["candidate_id"].astype(str))
    enabled = {
        str(candidate): set(pd.to_datetime(group["state_decision_ts"], utc=True).view("int64").tolist())
        for candidate, group in actions.loc[actions["h4_enable"]].groupby("candidate_id", sort=False)
    }
    rows: list[dict[str, object]] = []
    for _, candidate in route.sort_values(["timestamp", "candidate_id"], kind="stable").iterrows():
        candidate_id = str(candidate["candidate_id"])
        index = int(position[candidate_id])
        selected = enabled.get(candidate_id, set())
        action_latched = False

        def decide(state: dict[str, float], selected: set[int] = selected) -> bool:
            nonlocal action_latched
            now = int(pd.Timestamp(state["state_decision_ts"]).value) in selected
            if latched:
                action_latched = action_latched or now
                return action_latched
            return now

        trace = replay_exact_1m_h4_giveback20(
            entry_price=float(paths.entry[index]), signal_atr=float(paths.atr[index]),
            entry_ts=paths.rows.iloc[index]["entry_ts"], highs=paths.high[index],
            lows=paths.low[index], closes=paths.close[index], params=params,
            median_atr_fraction=float(median_atr_fraction),
            mc1_expected_bps=float(candidate["bcf_mc1_expected_bps"]),
            state_decider=decide, giveback_tighten=.20, emit_states=False,
        )
        rows.append({
            "candidate_id": candidate_id, "timestamp": candidate["timestamp"],
            "entry_ts": paths.rows.iloc[index]["entry_ts"], "symbol": candidate["symbol"],
            "bcf_mc1_expected_bps": float(candidate["bcf_mc1_expected_bps"]),
            "auction_priority_bps": float(candidate["auction_priority_bps"]),
            "exact_entry_price": float(paths.entry[index]), "exact_net_bps": float(trace["net_bps"]),
            "exact_gross_bps": float(trace["gross_bps"]), "exact_exit_price": float(trace["exit_price"]),
            "exact_exit_ts": pd.Timestamp(trace["exit_timestamp"]),
            "exact_exit_minute": int(trace["exit_minute"]), "exact_exit_reason": str(trace["exit_reason"]),
        })
    result = pd.DataFrame(rows)
    if result["candidate_id"].duplicated().any() or len(result) != len(route):
        raise AssertionError("H4 replay lost candidate identities")
    return result


def _monthly(accepted: pd.DataFrame) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame(columns=["month", "trades", "net_bps_per_trade", "total_net_bps"])
    result = accepted.copy()
    result["month"] = _utc(result["decision_timestamp"]).dt.strftime("%Y-%m")
    return result.groupby("month", as_index=False, sort=True).agg(
        trades=("candidate_id", "size"), net_bps_per_trade=("net_bps", "mean"),
        total_net_bps=("net_bps", "sum"),
    )


def _parent_for_route(route: pd.DataFrame, parent_outcomes: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "candidate_id", "timestamp", "entry_ts", "symbol", "bcf_mc1_expected_bps",
        "auction_priority_bps",
    ]
    parent = route.loc[:, fields].merge(
        parent_outcomes.loc[:, [
            "candidate_id", "exact_entry_price", "exact_net_bps", "exact_gross_bps",
            "exact_exit_price", "exact_exit_ts", "exact_exit_minute", "exact_exit_reason",
        ]],
        on="candidate_id", how="inner", validate="one_to_one",
    )
    if len(parent) != len(route):
        raise AssertionError("exact parent outcomes do not cover a source-valid population")
    return parent


def _write_portfolio(
    out: Path,
    *,
    arm: str,
    outcome: pd.DataFrame,
    result_rows: list[dict[str, object]],
) -> pd.DataFrame:
    route = outcome.loc[:, [
        "candidate_id", "timestamp", "entry_ts", "symbol", "auction_priority_bps",
    ]].copy()
    realised = outcome.loc[:, [
        "candidate_id", "timestamp", "entry_ts", "symbol",
        "exact_entry_price", "exact_net_bps", "exact_gross_bps",
        "exact_exit_price", "exact_exit_ts", "exact_exit_minute", "exact_exit_reason",
    ]].copy()
    candidates, decisions, accepted, equity, metrics = _portfolio(route, realised, arm)
    candidates.to_parquet(out / f"{arm}_portfolio_candidates.parquet", index=False, compression="zstd")
    decisions.to_parquet(out / f"{arm}_portfolio_decisions.parquet", index=False, compression="zstd")
    accepted.to_parquet(out / f"{arm}_portfolio_accepted.parquet", index=False, compression="zstd")
    equity.to_parquet(out / f"{arm}_portfolio_equity.parquet", index=False, compression="zstd")
    _monthly(accepted).to_parquet(out / f"{arm}_monthly_portfolio_metrics.parquet", index=False, compression="zstd")
    result_rows.append({"arm": arm, **metrics})
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2-selection", type=Path, default=E2_SELECTION)
    parser.add_argument("--c1-route", type=Path, default=C1_ROUTE)
    parser.add_argument("--c1-latch-root", type=Path, default=C1_LATCH_ROOT)
    parser.add_argument("--legacy-feature-contract", type=Path, default=LEGACY_FEATURE_CONTRACT)
    parser.add_argument("--minute-root", type=Path, default=DEFAULT_MINUTE_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--source-workers", type=int, default=12)
    parser.add_argument(
        "--reuse-exact-root", type=Path, default=None,
        help=(
            "Completed interrupted-run exact-path checkpoint to reuse only after "
            "exact target-free identity and array-alignment validation."
        ),
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    if int(args.source_workers) < 1:
        raise ValueError("source workers must be positive")
    # Read this small immutable receipt before the exact-path arrays are
    # allocated.  The field values and order remain frozen; this only avoids a
    # process-local Arrow buffer failure observed after multi-gigabyte state
    # materialisation on macOS.
    legacy_contract = _read_frozen_parquet(args.legacy_feature_contract.resolve())

    # The two routes are written before any exact path, parent replay, state,
    # label, or portfolio dataset is read.
    populations = {
        "E2_original_selection": _load_e2_population(args.e2_selection.resolve()),
        "C1_dual40": _load_c1_population(args.c1_route.resolve()),
    }
    output_route = pd.concat(populations.values(), ignore_index=True)
    overlap = set(populations["E2_original_selection"]["candidate_id"]).intersection(
        set(populations["C1_dual40"]["candidate_id"])
    )
    path_population = _union_path_population(populations)
    if args.reuse_exact_root is None:
        paths, source_coverage, source_invalid = _windowed_exact_paths(
            path_population, minute_root=args.minute_root.resolve(), workers=int(args.source_workers),
        )
        reused_exact = None
    else:
        reused_exact = args.reuse_exact_root.resolve()
        paths, source_coverage, source_invalid = _reuse_exact_paths(reused_exact, path_population)
    params, median_atr_fraction, policy_receipt = _load_policy(args.policy.resolve())
    parent_outcomes = _outcomes(paths, args.policy.resolve())
    raw_states, h4_parent = _raw_h4_states(
        paths, params=params, median_atr_fraction=float(median_atr_fraction),
    )
    parent_check = parent_outcomes.merge(h4_parent, on="candidate_id", how="inner", validate="one_to_one")
    if not np.allclose(parent_check["exact_net_bps"], parent_check["parent_h4_engine_net_bps"], rtol=0.0, atol=1e-8):
        raise AssertionError("H4 state engine does not reproduce the exact parent net outcome")
    if not (pd.to_datetime(parent_check["exact_exit_ts"], utc=True) == pd.to_datetime(parent_check["parent_h4_engine_exit_ts"], utc=True)).all():
        raise AssertionError("H4 state engine does not reproduce exact parent exit timestamps")
    context_states, state_coverage = _state_features(raw_states)
    context_states["candidate_id"] = context_states["candidate_id"].astype(str)

    valid_ids = set(parent_outcomes["candidate_id"].astype(str))
    valid_populations = {
        name: route.loc[route["candidate_id"].isin(valid_ids)].copy().reset_index(drop=True)
        for name, route in populations.items()
    }
    if any(route.empty for route in valid_populations.values()):
        raise RuntimeError("a matched population has no valid exact paths")
    out.mkdir(parents=True, exist_ok=False)
    output_route.to_parquet(out / "target_free_population_routes.parquet", index=False, compression="zstd")
    path_population.to_parquet(out / "target_free_union_path_population.parquet", index=False, compression="zstd")
    source_coverage.to_parquet(out / "exact_1m_source_coverage.parquet", index=False, compression="zstd")
    source_invalid.to_parquet(out / "invalid_outcomes_after_target_free_route.parquet", index=False, compression="zstd")
    paths.rows.to_parquet(out / "valid_exact_paths_rows.parquet", index=False, compression="zstd")
    np.savez_compressed(
        out / "exact_paths.npz", candidate_id=paths.rows["candidate_id"].astype(str).to_numpy(),
        entry=paths.entry, atr=paths.atr, high=paths.high, low=paths.low, close=paths.close,
    )
    parent_outcomes.to_parquet(out / "exact_parent_outcomes.parquet", index=False, compression="zstd")
    raw_states.to_parquet(out / "all_parent_observable_h4_states_target_free.parquet", index=False, compression="zstd")
    context_states.to_parquet(out / "all_h4_context_states_target_free.parquet", index=False, compression="zstd")
    state_coverage.to_parquet(out / "h4_state_feature_coverage.parquet", index=False, compression="zstd")

    result_rows: list[dict[str, object]] = []
    support_parts: list[pd.DataFrame] = []
    action_parts: list[pd.DataFrame] = []
    changed_rows: list[dict[str, object]] = []
    for population_name, route in valid_populations.items():
        population_states = context_states.merge(
            route.loc[:, ["candidate_id", "bcf_mc1_expected_bps"]], on="candidate_id",
            how="inner", validate="many_to_one",
        ).copy()
        population_states["MC1_expected_bps"] = pd.to_numeric(
            population_states.pop("bcf_mc1_expected_bps"), errors="raise"
        )
        parent = _parent_for_route(route, parent_outcomes)
        parent_arm = f"{population_name}__parent_exact1m"
        parent_accepted = _write_portfolio(out, arm=parent_arm, outcome=parent, result_rows=result_rows)
        parent_exit = parent.set_index("candidate_id")["exact_exit_ts"]

        legacy_action, legacy_support = _legacy_actions(
            population_states, contract=legacy_contract,
        )
        legacy_action.insert(0, "population", population_name)
        legacy_action.to_parquet(out / f"{population_name}__legacy_l1_actions.parquet", index=False, compression="zstd")
        legacy_support.insert(0, "population", population_name)
        support_parts.append(legacy_support)
        action_parts.append(legacy_action.assign(controller="legacy_l1_one_interval"))
        legacy_outcome = _h4_outcomes(
            route=route, paths=paths, actions=legacy_action, params=params,
            median_atr_fraction=float(median_atr_fraction), latched=False,
        )
        legacy_arm = f"{population_name}__legacy_l1_one_interval"
        legacy_outcome.to_parquet(out / f"{legacy_arm}_exact1m_outcomes.parquet", index=False, compression="zstd")
        legacy_accepted = _write_portfolio(out, arm=legacy_arm, outcome=legacy_outcome, result_rows=result_rows)

        latched_action, latched_support = _latched_actions(
            population_states, label_root=args.c1_latch_root.resolve(),
            mfe_ready_atr=.5 * float(params.protection_activation_atr),
        )
        latched_action.insert(0, "population", population_name)
        latched_action.to_parquet(out / f"{population_name}__repaired_l2_actions.parquet", index=False, compression="zstd")
        latched_support.insert(0, "population", population_name)
        support_parts.append(latched_support)
        action_parts.append(latched_action.assign(controller="repaired_l2_latched"))
        latched_outcome = _h4_outcomes(
            route=route, paths=paths, actions=latched_action, params=params,
            median_atr_fraction=float(median_atr_fraction), latched=True,
        )
        latched_arm = f"{population_name}__repaired_l2_latched"
        latched_outcome.to_parquet(out / f"{latched_arm}_exact1m_outcomes.parquet", index=False, compression="zstd")
        latched_accepted = _write_portfolio(out, arm=latched_arm, outcome=latched_outcome, result_rows=result_rows)

        for controller, candidate_outcome, accepted in (
            ("legacy_l1_one_interval", legacy_outcome, legacy_accepted),
            ("repaired_l2_latched", latched_outcome, latched_accepted),
        ):
            changed = candidate_outcome.loc[
                pd.to_datetime(candidate_outcome["exact_exit_ts"], utc=True).ne(
                    candidate_outcome["candidate_id"].map(parent_exit)
                )
            ]
            changed_rows.append({
                "population": population_name, "controller": controller,
                "route_valid_candidates": int(len(route)), "candidate_exit_changes": int(len(changed)),
                "accepted_parent": int(len(parent_accepted)), "accepted_controller": int(len(accepted)),
            })

    summary = pd.DataFrame(result_rows)
    summary["population"] = summary["arm"].str.split("__").str[0]
    summary["controller"] = summary["arm"].str.split("__").str[1]
    for population_name in valid_populations:
        ref = summary.loc[summary["arm"].eq(f"{population_name}__parent_exact1m")].iloc[0]
        selector = summary["population"].eq(population_name)
        for field in (
            "portfolio_accepted", "net_bps_per_trade", "total_net_bps", "max_drawdown",
            "sortino", "worst_week", "compounded_return",
        ):
            summary.loc[selector, f"delta_vs_parent_{field}"] = summary.loc[selector, field] - ref[field]
    summary.to_parquet(out / "portfolio_summary.parquet", index=False)
    support = pd.concat(support_parts, ignore_index=True)
    support.to_parquet(out / "strict_prequential_training_support.parquet", index=False, compression="zstd")
    pd.concat(action_parts, ignore_index=True).to_parquet(out / "all_action_schedules.parquet", index=False, compression="zstd")
    pd.DataFrame(changed_rows).to_parquet(out / "exit_change_attribution.parquet", index=False, compression="zstd")

    audit = {
        "schema": "causal-sr-e2-c1-h4-matched-2x2-v1",
        "scope": "offline research-only population × continuation-contract falsification; no exchange or live mutation",
        "evaluation": [str(EVAL_START), str(EVAL_END)],
        "axes": {
            "population": ["E2 original target-free q50 selection", "C1 target-free dual MC1 >=40 bps"],
            "controller": [
                "archived L1 one-interval activation50-advantage contract",
                "repaired L2 latched activation50+giveback20 contract",
            ],
        },
        "fixed_parent": "frozen rich exact-1m policy; decision +5m entry; 100-bps cost exactly once",
        "fixed_portfolio": "normal global chronological auction; same constrained parameters and BCF priority per population",
        "target_free_sources": {
            str(args.e2_selection.resolve()): _sha256(args.e2_selection.resolve()),
            str(args.c1_route.resolve()): _sha256(args.c1_route.resolve()),
        },
        "label_sources": {
            "legacy": str(legacy_h4.TARGET_PANEL),
            "repaired": str(args.c1_latch_root.resolve() / "latched_activation50_giveback20_labels.parquet"),
        },
        "policy": str(args.policy.resolve()),
        "policy_sha256": _sha256(args.policy.resolve()),
        "routes_before_outcomes": {name: int(len(route)) for name, route in populations.items()},
        "valid_exact_paths_after_target_free_route": {name: int(len(route)) for name, route in valid_populations.items()},
        "cross_population_candidate_overlap": int(len(overlap)),
        "union_target_free_path_requests": int(len(path_population)),
        "union_exact_valid_paths": int(len(paths.rows)),
        "union_exact_invalid_after_route": int(len(source_invalid)),
        "reused_exact_path_checkpoint": (
            None if reused_exact is None else {
                "path": str(reused_exact),
                "target_free_union_sha256": _sha256(reused_exact / "target_free_union_path_population.parquet"),
                "valid_paths_sha256": _sha256(reused_exact / "valid_exact_paths_rows.parquet"),
                "exact_arrays_sha256": _sha256(reused_exact / "exact_paths.npz"),
                "source_coverage_sha256": _sha256(reused_exact / "exact_1m_source_coverage.parquet"),
            }
        ),
        "parent_engine_parity": "exact net and exit timestamp matched on every valid path",
        "causality": (
            "routes are loaded and persisted before exact paths; original L1 training uses prior-resolved historical labels; "
            "repaired L2 training uses prior-resolved C1 labels; action begins after completed state only"
        ),
        "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    print(out)


if __name__ == "__main__":
    main()
