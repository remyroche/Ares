#!/usr/bin/env python3
"""Strict-R3 source-aligned rich SimplePolicyOptimiser challenger.

This is an offline research runner.  It deliberately does not import the
live-execution configuration, alter the frozen policy winner, submit orders,
or write into any live state.  It compares the incumbent and a richer policy
only after the same prequential upstream score has been frozen.

The HPO development interval is calendar 2024.  It uses a Jan--Jun calibration
slice for distributional quantities (ATR scale and adverse severity threshold)
and scores choices only on Jul--Dec.  2025 and 2026 are never inspected during
selection.  Their replays use the frozen threshold and exact same portfolio
auction for both policy arms.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_rich_policy import (  # noqa: E402
    RichPolicyParams,
    causal_portfolio_selection,
    fit_adverse_theta,
    policy_metrics,
    simulate_rich_policy,
)
from extreme_price_movements.residual_lambdarank_hpo import stop_after_no_improvement  # noqa: E402


DEFAULT_LEDGER = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_prequential_ledger_targetfree_long_"
    "2024_2026_raw15m_strictfull_20260812_v1/prequential_stack_ledger.parquet"
)
DEFAULT_INCUMBENT = ROOT / (
    "data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_"
    "20260817_v1/frozen_policy.json"
)
DEFAULT_BARS = ROOT / "15m_ohlcv_perp"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_rich_policy_hpo_long_20260817_v1"
HORIZON_BARS = 48
SEED = 20260817
NO_IMPROVEMENT_PATIENCE = 20


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _symbol_filename(symbol: str) -> str:
    return symbol.lower().replace("/", "") + "_15m.parquet"


def _utc(value: pd.Series) -> pd.Series:
    return pd.to_datetime(value, utc=True, errors="coerce")


def _stable_cap(frame: pd.DataFrame, *, per_month: int) -> pd.DataFrame:
    """Deterministically cap each month without changing class prevalence."""
    if per_month <= 0 or frame.empty:
        return frame
    work = frame.copy()
    work["__month__"] = work["timestamp"].dt.to_period("M").astype(str)
    keep: list[pd.DataFrame] = []
    for _, group in work.groupby("__month__", sort=True):
        if len(group) <= per_month:
            keep.append(group)
            continue
        hashes = pd.util.hash_pandas_object(group["candidate_id"].astype(str), index=False).to_numpy("uint64")
        selected = group.iloc[np.argsort(hashes, kind="stable")[:per_month]]
        keep.append(selected)
    return pd.concat(keep, ignore_index=True).drop(columns="__month__")


def _ledger_files(ledgers: Iterable[Path]) -> list[Path]:
    """Resolve explicit files or partitioned prequential-ledger roots.

    The original long research ledger is a single parquet file.  The
    side-aware short producer persists immutable monthly partitions instead.
    Treating the latter as an ordinary source preserves their source-month
    identity while allowing one HPO/replay run to span the requested years.
    """
    resolved: list[Path] = []
    for raw in ledgers:
        path = Path(raw).resolve()
        if path.is_file():
            resolved.append(path)
            continue
        if not path.is_dir():
            raise FileNotFoundError(f"Ledger path does not exist: {path}")
        direct = path / "prequential_stack_ledger.parquet"
        if direct.exists():
            resolved.append(direct)
        resolved.extend(sorted(path.glob("ledger/month=*/prequential_base_ledger.parquet")))
        resolved.extend(sorted(path.glob("ledger/month=*/prequential_stack_ledger.parquet")))
    unique = list(dict.fromkeys(resolved))
    if not unique:
        raise FileNotFoundError("No prequential ledger parquet files were resolved")
    return unique


def load_score_population(ledgers: Iterable[Path], *, side: str = "long") -> pd.DataFrame:
    side = str(side).strip().lower()
    if side not in {"long", "short"}:
        raise ValueError("rich-policy HPO side must be long or short")
    pieces: list[pd.DataFrame] = []
    required = {"candidate_id", "__decision_ts__", "__symbol__", "side_name", "stack_is_prequential"}
    for ledger in _ledger_files(ledgers):
        names = set(pq.ParquetFile(ledger).schema_arrow.names)
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"{ledger} is not a strict prequential score ledger; missing {missing}")
        # Long canonical ledgers publish the ensemble's upstream score.  The
        # frozen P0/F90 continuation is intentionally base-only, so its
        # causally normalised base rank is the equivalent upstream score.
        score_source = next(
            (name for name in ("prequential_upstream", "prequential_base_rank42", "prequential_base_score") if name in names),
            None,
        )
        if score_source is None:
            raise ValueError(f"{ledger} has no prequential score field")
        complete_source = next((name for name in ("base_contract_complete", "entry_executable") if name in names), None)
        columns = sorted(required | {score_source} | ({complete_source} if complete_source else set()))
        work = pd.read_parquet(ledger, columns=columns)
        work = work.rename(columns={
            "__decision_ts__": "timestamp", "__symbol__": "symbol", score_source: "score",
        })
        work["base_contract_complete"] = (
            work[complete_source].astype(bool) if complete_source else True
        )
        work["score_source"] = score_source
        pieces.append(work)
    work = pd.concat(pieces, ignore_index=True)
    work["timestamp"] = _utc(work["timestamp"])
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work = work.loc[
        work["side_name"].astype(str).str.lower().eq(side)
        & work["base_contract_complete"].astype(bool)
        & work["stack_is_prequential"].astype(bool)
        & work["timestamp"].notna()
        & work["score"].notna()
    ].copy()
    work["candidate_id"] = work["candidate_id"].astype(str)
    work["symbol"] = work["symbol"].astype(str)
    if work["candidate_id"].duplicated().any():
        raise AssertionError("Combined ledger sources contain duplicate candidate identities")
    return work.sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True)


def load_targetfree_score_population(
    root: Path,
    *,
    score_column: str,
    side: str = "long",
) -> tuple[pd.DataFrame, list[Path]]:
    """Read only fixed point-in-time score identities for policy selection.

    The long 2024 ledger was intentionally archived after the live-contract
    incident.  This loader is deliberately narrower than ``load_score_population``:
    it accepts the retained target-free routed-score partitions, reads no outcome
    field, and derives the symbol only from the immutable candidate identity.
    """
    side = str(side).strip().lower()
    if side not in {"long", "short"}:
        raise ValueError("rich-policy HPO side must be long or short")
    root = Path(root).resolve()
    paths = sorted(root.glob("month=*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No target-free score months under {root}")
    required = {"candidate_id", "__decision_ts__", "side_name", score_column}
    prohibited = {
        "policy_net_bps", "policy_gross_bps", "policy_path_valid",
        "policy_label_available_ts", "outcome", "label_valid",
    }
    parts: list[pd.DataFrame] = []
    for path in paths:
        names = set(pq.ParquetFile(path).schema_arrow.names)
        missing = sorted(required - names)
        if missing:
            raise AssertionError(f"{path}: missing target-free score fields {missing}")
        leaked = sorted(prohibited & names)
        if leaked:
            raise AssertionError(f"{path}: not target-free; contains {leaked}")
        work = pd.read_parquet(path, columns=["candidate_id", "__decision_ts__", "side_name", score_column])
        work = work.rename(columns={"__decision_ts__": "timestamp", score_column: "score"})
        work["symbol"] = work["candidate_id"].astype(str).str.split("|", n=1, expand=True)[0]
        work["base_contract_complete"] = True
        work["score_source"] = score_column
        parts.append(work)
    work = pd.concat(parts, ignore_index=True)
    work["timestamp"] = _utc(work["timestamp"])
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work = work.loc[
        work["side_name"].astype(str).str.lower().eq(side)
        & work["timestamp"].notna()
        & work["score"].notna()
    ].copy()
    work["candidate_id"] = work["candidate_id"].astype(str)
    work["symbol"] = work["symbol"].astype(str)
    if work["candidate_id"].duplicated().any():
        raise AssertionError("Target-free score partitions contain duplicate candidate identities")
    return work.sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True), paths


def _hourly_signal_atr(bars: pd.DataFrame) -> pd.Series:
    """Wilder-14 ATR known at the decision-hour open, never from that hour."""
    raw = bars[["high", "low", "close"]].copy()
    raw = raw.apply(pd.to_numeric, errors="coerce")
    hourly = raw.resample("1h", label="left", closed="left").agg(
        high=("high", "max"), low=("low", "min"), close=("close", "last"), count=("close", "count")
    )
    previous = hourly["close"].shift(1)
    tr = pd.concat(
        [hourly["high"] - hourly["low"], (hourly["high"] - previous).abs(), (hourly["low"] - previous).abs()],
        axis=1,
    ).max(axis=1)
    tr = tr.where(hourly["count"].eq(4))
    atr = tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    # Bar labelled t contains [t,t+1h); it is not known at t.
    return atr.shift(1)


def materialize_paths(population: pd.DataFrame, *, bars_root: Path) -> tuple[pd.DataFrame, dict[str, np.ndarray], pd.DataFrame]:
    """Load future paths after scores are frozen and preserve coverage evidence."""
    columns = ["candidate_id", "timestamp", "symbol", "score"]
    frame = population[columns].copy().reset_index(drop=True)
    frame["__row_id__"] = np.arange(len(frame), dtype=np.int64)
    arrays: dict[str, list[np.ndarray]] = {"entry": [], "atr": [], "high": [], "low": [], "close": []}
    pieces: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    for symbol, group in frame.groupby("symbol", sort=True):
        source = bars_root / _symbol_filename(symbol)
        if not source.exists():
            coverage.append({"symbol": symbol, "scored_rows": len(group), "path_complete_rows": 0, "reason": "missing_symbol_file"})
            continue
        bars = pd.read_parquet(source, columns=["open", "high", "low", "close"])
        bars.index = pd.to_datetime(bars.index, utc=True, errors="coerce")
        bars = bars.loc[~bars.index.isna()].sort_index()
        bars = bars.loc[~bars.index.duplicated(keep="last")]
        atr = _hourly_signal_atr(bars)
        decisions = pd.DatetimeIndex(group["timestamp"])
        positions = bars.index.get_indexer(decisions)
        offsets = np.arange(HORIZON_BARS, dtype=np.int64)
        location = positions[:, None] + offsets[None, :]
        in_range = (positions >= 0) & (location[:, -1] < len(bars))
        valid_rows = np.flatnonzero(in_range)
        if not len(valid_rows):
            coverage.append({"symbol": symbol, "scored_rows": len(group), "path_complete_rows": 0, "reason": "missing_decision_or_h12"})
            continue
        location = location[valid_rows]
        selected = group.iloc[valid_rows].copy()
        open_values = pd.to_numeric(bars["open"], errors="coerce").to_numpy(np.float64)
        high_values = pd.to_numeric(bars["high"], errors="coerce").to_numpy(np.float64)
        low_values = pd.to_numeric(bars["low"], errors="coerce").to_numpy(np.float64)
        close_values = pd.to_numeric(bars["close"], errors="coerce").to_numpy(np.float64)
        entry = open_values[location[:, 0]]
        high = high_values[location]
        low = low_values[location]
        close = close_values[location]
        atr_values = atr.reindex(pd.DatetimeIndex(selected["timestamp"])).to_numpy(np.float64)
        complete = (
            np.isfinite(entry) & (entry > 0.0) & np.isfinite(atr_values) & (atr_values > 0.0)
            & np.isfinite(high).all(axis=1) & np.isfinite(low).all(axis=1) & np.isfinite(close).all(axis=1)
        )
        selected = selected.iloc[np.flatnonzero(complete)].copy()
        arrays["entry"].append(entry[complete])
        arrays["atr"].append(atr_values[complete])
        arrays["high"].append(high[complete].astype(np.float32, copy=False))
        arrays["low"].append(low[complete].astype(np.float32, copy=False))
        arrays["close"].append(close[complete].astype(np.float32, copy=False))
        pieces.append(selected)
        coverage.append({"symbol": symbol, "scored_rows": len(group), "path_complete_rows": int(complete.sum()), "reason": "ok"})
    if not pieces:
        raise RuntimeError("No complete 15-minute outcome paths were materialised")
    joined = pd.concat(pieces, ignore_index=True)
    outcome = {
        "entry": np.concatenate(arrays["entry"]), "atr": np.concatenate(arrays["atr"]),
        "high": np.concatenate(arrays["high"]), "low": np.concatenate(arrays["low"]), "close": np.concatenate(arrays["close"]),
    }
    if len(joined) != len(outcome["entry"]):
        raise AssertionError("path arrays lost candidate identity alignment")
    return joined, outcome, pd.DataFrame(coverage)


def _median_atr_fraction(paths: Mapping[str, np.ndarray]) -> float:
    fraction = np.asarray(paths["atr"], float) / np.maximum(np.asarray(paths["entry"], float), 1e-12)
    return float(np.nanmedian(fraction[np.isfinite(fraction) & (fraction > 0.0)]))


def _subset_paths(paths: Mapping[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    position = np.flatnonzero(np.asarray(mask, dtype=bool))
    return {key: np.asarray(value)[position] for key, value in paths.items()}


def _params_from_trial(
    trial: optuna.Trial,
    *,
    incumbent: RichPolicyParams,
    max_sl_cap_pct: float | None = None,
    require_rich_extensions: bool = False,
) -> RichPolicyParams:
    """One bounded search surface; inactive branch parameters remain explicit."""
    mode = trial.suggest_categorical("trailing_mode", ["fixed", "dynamic"])
    cap = trial.suggest_categorical("trailing_activation_cap_pct", [0.0, 0.010, 0.015, 0.020, 0.030, 0.050])
    floor = trial.suggest_categorical("trailing_activation_min_pct", [0.0, 0.003, 0.005, 0.0075, 0.010])
    # Optuna requires a stable categorical distribution across trials.  Keep
    # the sampled floor but make an inconsistent floor/cap pair conservative.
    if cap > 0.0 and floor > cap:
        floor = cap
    values: dict[str, Any] = {
        "sl_mult": trial.suggest_float("sl_mult", 2.50, 6.00),
        "trailing_activation_mult": trial.suggest_float("trailing_activation_mult", 0.50, 4.00),
        "sl_abs_floor_pct": trial.suggest_categorical("sl_abs_floor_pct", [0.0, 0.004, 0.006, 0.008, 0.010]),
        "sl_abs_cap_pct": trial.suggest_categorical("sl_abs_cap_pct", [0.0, 0.010, 0.015, 0.020, 0.030, 0.050]),
        "trailing_activation_min_pct": floor,
        "trailing_activation_cap_pct": cap,
        "trailing_activation_decay_half_life_bars": trial.suggest_categorical("trailing_activation_decay_half_life_bars", [0.0, 4.0, 8.0, 16.0, 24.0, 32.0]),
        "trailing_activation_decay_start_bars": trial.suggest_categorical("trailing_activation_decay_start_bars", [0, 2, 4, 8, 16]),
        "trailing_activation_min_mult": trial.suggest_categorical("trailing_activation_min_mult", [0.35, 0.50, 0.70, 0.85, 1.0]),
        "sl_atr_power": trial.suggest_categorical("sl_atr_power", [0.70, 0.85, 1.0, 1.15, 1.30]),
        "sl_atr_multiplier": trial.suggest_categorical("sl_atr_multiplier", [0.75, 1.0, 1.25]),
        "tp_atr_power": trial.suggest_categorical("tp_atr_power", [0.70, 0.85, 1.0, 1.15, 1.30]),
        "tp_atr_multiplier": trial.suggest_categorical("tp_atr_multiplier", [0.75, 1.0, 1.25]),
        "capital_protect_mfe_mult": trial.suggest_categorical("capital_protect_mfe_mult", [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]),
        "capital_protect_regression_frac": trial.suggest_float("capital_protect_regression_frac", 0.10, 0.80),
        "capital_protect_lock_frac": trial.suggest_categorical("capital_protect_lock_frac", [None, 0.0, 0.20, 0.40, 0.60, 0.80]),
        "capital_protect_min_lock_bps": trial.suggest_categorical("capital_protect_min_lock_bps", [0.0, 10.0, 25.0, 50.0, 75.0]),
        "adverse_exit_enabled": (
            True if require_rich_extensions
            else trial.suggest_categorical("adverse_exit_enabled", [False, True])
        ),
        "adverse_exit_min_mae_atr": trial.suggest_categorical("adverse_exit_min_mae_atr", [0.50, 0.75, 1.0, 1.25, 1.50]),
        "adverse_exit_min_speed": trial.suggest_categorical("adverse_exit_min_speed", [0.10, 0.20, 0.30, 0.45, 0.60]),
        "adverse_exit_max_mfe_atr": trial.suggest_categorical("adverse_exit_max_mfe_atr", [0.10, 0.20, 0.25, 0.35, 0.50]),
        "adverse_exit_severity_quantile": trial.suggest_categorical("adverse_exit_severity_quantile", [0.55, 0.65, 0.75, 0.85, 0.90]),
    }
    # An absolute floor takes precedence over an absolute cap in the simulator.
    if values["sl_abs_cap_pct"] and values["sl_abs_floor_pct"] > values["sl_abs_cap_pct"]:
        values["sl_abs_cap_pct"] = values["sl_abs_floor_pct"]
    if max_sl_cap_pct is not None:
        forced_cap = float(max_sl_cap_pct)
        if not (0.0 < forced_cap <= 0.10):
            raise ValueError("max_sl_cap_pct must be within (0, 0.10]")
        # This is a hard parent-policy constraint, not a trial preference.  It
        # keeps every simulated hard stop at or inside the requested cap.
        values["sl_abs_cap_pct"] = forced_cap
        values["sl_abs_floor_pct"] = min(float(values["sl_abs_floor_pct"]), forced_cap)
    if mode == "fixed":
        values.update(
            fixed_trailing_gap_mult=trial.suggest_float("fixed_trailing_gap_mult", 0.04, 0.35),
            trailing_power=incumbent.trailing_power,
            trailing_squash_divisor=incumbent.trailing_squash_divisor,
            giveback_beta=incumbent.giveback_beta,
        )
    else:
        values.update(
            fixed_trailing_gap_mult=0.0,
            trailing_power=trial.suggest_float("trailing_power", 0.60, 3.00),
            trailing_squash_divisor=trial.suggest_float("trailing_squash_divisor", 0.50, 5.00),
            giveback_beta=trial.suggest_float("giveback_beta", 0.10, 0.90),
        )
    # The promoted long-side rich-policy process also evaluates a smooth
    # MFE-protection arm.  Keep the same bounded values here, including the
    # 1.25/1.75 activation and 1.25/1.75/2.0 power extensions, so a short
    # result cannot be called equivalent while omitting that mechanism.
    smooth = True if require_rich_extensions else trial.suggest_categorical(
        "smooth_capital_protection_enabled", [False, True]
    )
    if smooth:
        values.update(
            smooth_capital_protection_enabled=True,
            protection_activation_atr=trial.suggest_categorical(
                "protection_activation_atr", [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]
            ),
            protection_strength=trial.suggest_categorical(
                "protection_strength", [0.25, 0.5, 0.75]
            ),
            protection_power=trial.suggest_categorical(
                "protection_power", [0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
            ),
        )
    else:
        values.update(
            smooth_capital_protection_enabled=False,
            protection_activation_atr=0.0,
            protection_strength=0.0,
            protection_power=1.0,
        )
    return RichPolicyParams.from_mapping(values)


def _simulate_frame(frame: pd.DataFrame, paths: Mapping[str, np.ndarray], *, params: RichPolicyParams, median_atr: float, theta_paths: Mapping[str, np.ndarray] | None = None, side: str = "long") -> pd.DataFrame:
    frozen = params
    if params.adverse_exit_enabled and params.adverse_exit_theta is None:
        if theta_paths is None:
            raise ValueError("fast-adverse policy needs a prior calibration path set")
        theta = fit_adverse_theta(
            entry=np.asarray(theta_paths["entry"]), atr=np.asarray(theta_paths["atr"]),
            highs=np.asarray(theta_paths["high"]), lows=np.asarray(theta_paths["low"]),
            params=params, median_atr_fraction=median_atr, side=side,
        )
        frozen = RichPolicyParams.from_mapping({**params.to_dict(), "adverse_exit_theta": theta})
    outcome = simulate_rich_policy(
        entry=np.asarray(paths["entry"]), atr=np.asarray(paths["atr"]), highs=np.asarray(paths["high"]),
        lows=np.asarray(paths["low"]), closes=np.asarray(paths["close"]), params=frozen, median_atr_fraction=median_atr, side=side,
    )
    result = frame.copy()
    result["gross_bps"] = np.asarray(outcome["gross_bps"], dtype=np.float64)
    result["net_bps"] = np.asarray(outcome["net_bps"], dtype=np.float64)
    result["exit_bar"] = np.asarray(outcome["exit_bar"], dtype=np.int16)
    result["exit_reason"] = np.asarray(outcome["exit_reason"], dtype=object)
    result["path_valid"] = np.asarray(outcome["path_valid"], dtype=bool)
    result["adverse_exit_theta"] = float(outcome["adverse_exit_theta"])
    return result


def _objective_from_selected(frame: pd.DataFrame, selected: np.ndarray) -> tuple[float, dict[str, float | int]]:
    summary, monthly, _ = policy_metrics(frame, selected)
    if monthly.empty or int(summary["trades"]) < 50:
        return -float("inf"), summary
    values = monthly["net_bps_per_trade"].to_numpy(float)
    median = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - median)))
    worst = float(np.nanmin(values))
    # This is deliberately a stability objective, not a total-PnL maximiser.
    score = median - 0.5 * mad - max(0.0, -worst)
    return score, summary


def _incumbent_params(winner: Mapping[str, Any]) -> RichPolicyParams:
    # A rich frozen policy is already a complete parent-policy contract.  The
    # legacy simple-policy winner below is retained only for historical reruns.
    if isinstance(winner.get("params"), Mapping):
        return RichPolicyParams.from_mapping(dict(winner["params"]))
    values = dict(winner)
    values.update({
        "sl_abs_floor_pct": 0.0, "sl_abs_cap_pct": 0.0,
        "trailing_activation_min_pct": 0.0, "trailing_activation_cap_pct": 0.0,
        "trailing_activation_decay_half_life_bars": 0.0,
        "trailing_activation_decay_start_bars": 0, "trailing_activation_min_mult": 1.0,
        "trailing_power": 1.5, "trailing_squash_divisor": 2.0, "giveback_beta": 0.5,
        "atr_power": 1.0, "atr_multiplier": 1.0,
        "sl_atr_power": None, "sl_atr_multiplier": None,
        "tp_atr_power": None, "tp_atr_multiplier": None,
        "capital_protect_mfe_mult": 0.0, "capital_protect_regression_frac": 0.45,
        "capital_protect_lock_frac": None, "capital_protect_min_lock_bps": 0.0,
        "adverse_exit_enabled": False, "adverse_exit_theta": None,
    })
    return RichPolicyParams.from_mapping(values)


def _run_policy(frame: pd.DataFrame, paths: Mapping[str, np.ndarray], *, params: RichPolicyParams, median_atr: float, theta_paths: Mapping[str, np.ndarray] | None, side: str = "long") -> tuple[pd.DataFrame, np.ndarray, dict[str, float | int], pd.DataFrame, pd.DataFrame]:
    replay = _simulate_frame(
        frame, paths, params=params, median_atr=median_atr,
        theta_paths=theta_paths, side=side,
    )
    selected = causal_portfolio_selection(replay)
    summary, monthly, weekly = policy_metrics(replay, selected)
    return replay, selected, summary, monthly, weekly


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _as_report_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "_No accepted, resolved trades._"
    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return "" if not np.isfinite(value) else f"{value:.2f}"
        return str(value).replace("|", "\\|")
    cols = list(table.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    rows.extend("| " + " | ".join(render(value) for value in row) + " |" for row in table.itertuples(index=False, name=None))
    return "\n".join(rows)


def run(args: argparse.Namespace) -> Path:
    output = Path(args.out_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite immutable output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if args.targetfree_score_root is not None:
        population, targetfree_score_files = load_targetfree_score_population(
            args.targetfree_score_root,
            score_column=args.targetfree_score_column,
            side=args.side,
        )
        ledger_files: list[Path] = []
        score_source_manifest = {
            "kind": "retained_targetfree_score_partitions",
            "root": str(Path(args.targetfree_score_root).resolve()),
            "score_column": str(args.targetfree_score_column),
            "files": [{"path": str(path), "sha256": _sha256(path)} for path in targetfree_score_files],
        }
    else:
        ledger_files = _ledger_files(args.ledger or [DEFAULT_LEDGER])
        population = load_score_population(ledger_files, side=args.side)
        score_source_manifest = {
            "kind": "strict_prequential_ledger",
            "files": [{"path": str(path), "sha256": _sha256(path)} for path in ledger_files],
        }
    bars_root = Path(args.bars_root).resolve()
    incumbent_file = Path(args.incumbent).resolve()
    def as_utc(raw: str) -> pd.Timestamp:
        stamp = pd.Timestamp(raw)
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    dev_start = as_utc(args.development_start)
    calibration_end = as_utc(args.calibration_end)
    dev_end = as_utc(args.development_end)
    if not dev_start < calibration_end < dev_end:
        raise ValueError("development-start < calibration-end < development-end is required")
    development = population.loc[population["timestamp"].ge(dev_start) & population["timestamp"].lt(dev_end)].copy()
    if development.empty:
        raise RuntimeError(f"No strict-prequential 2024 {args.side} score rows")
    cutoff = float(development["score"].quantile(1.0 - float(args.retained_fraction), interpolation="higher"))
    development = development.loc[development["score"].ge(cutoff)].copy()
    development = _stable_cap(development, per_month=int(args.cap_per_month))
    development, dev_paths, dev_coverage = materialize_paths(development, bars_root=bars_root)
    development["timestamp"] = _utc(development["timestamp"])
    calibration_mask = development["timestamp"].lt(calibration_end).to_numpy()
    validation_mask = ~calibration_mask
    if calibration_mask.sum() < 1000 or validation_mask.sum() < 1000:
        raise RuntimeError("Insufficient complete 2024 calibration/validation paths")
    calibration = development.loc[calibration_mask].reset_index(drop=True)
    validation = development.loc[validation_mask].reset_index(drop=True)
    calibration_paths = _subset_paths(dev_paths, calibration_mask)
    validation_paths = _subset_paths(dev_paths, validation_mask)
    median_atr = _median_atr_fraction(calibration_paths)
    incumbent_values = json.loads(incumbent_file.read_text())
    incumbent = _incumbent_params(incumbent_values)

    # Incumbent runs through the exact rich-simulator defaults; this verifies
    # the comparison runner itself before any new degree of freedom is chosen.
    incumbent_replay, incumbent_selected, incumbent_summary, incumbent_monthly, _ = _run_policy(
        validation, validation_paths, params=incumbent, median_atr=median_atr, theta_paths=calibration_paths, side=args.side
    )
    incumbent_objective, _ = _objective_from_selected(incumbent_replay, incumbent_selected)
    records: list[dict[str, Any]] = [{
        "trial": -1, "stage": "incumbent_control", "state": "COMPLETE", "objective": incumbent_objective,
        "trailing_mode": "fixed",
        **{f"metric_{key}": value for key, value in incumbent_summary.items()}, **incumbent.to_dict(),
    }]

    sampler = optuna.samplers.TPESampler(seed=int(args.seed), multivariate=True, group=True, n_startup_trials=min(12, int(args.trials)))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=optuna.pruners.MedianPruner(n_startup_trials=12))

    def objective(trial: optuna.Trial) -> float:
        params = _params_from_trial(
            trial,
            incumbent=incumbent,
            max_sl_cap_pct=args.max_sl_cap_pct,
            require_rich_extensions=bool(args.require_rich_extensions),
        )
        replay, selected, summary, monthly, _ = _run_policy(
            validation, validation_paths, params=params, median_atr=median_atr, theta_paths=calibration_paths, side=args.side
        )
        value, _ = _objective_from_selected(replay, selected)
        trial.set_user_attr("summary", summary)
        trial.set_user_attr("adverse_exit_theta", float(replay["adverse_exit_theta"].iloc[0]))
        trial.set_user_attr("monthly_net_bps", monthly.to_dict(orient="records"))
        records.append({
            "trial": trial.number, "stage": "broad_rich_policy", "state": "COMPLETE", "objective": value,
            "trailing_mode": "fixed" if params.fixed_trailing_gap_mult > 0.0 else "dynamic",
            **{f"metric_{key}": entry for key, entry in summary.items()}, **params.to_dict(),
            "adverse_exit_theta_fitted": float(replay["adverse_exit_theta"].iloc[0]),
        })
        return value

    study.optimize(
        objective,
        n_trials=int(args.trials),
        n_jobs=1,
        show_progress_bar=False,
        callbacks=[stop_after_no_improvement(patience=NO_IMPROVEMENT_PATIENCE)],
    )
    trials = pd.DataFrame(records).sort_values(["objective", "trial"], ascending=[False, True], kind="stable")
    trials.to_parquet(output / "trials.parquet", index=False)
    best = RichPolicyParams.from_mapping(study.best_params)
    # Trial parameters do not preserve conditionally inactive settings in an
    # attribute-free dict; reconstruct from the recorded winning row.
    winning = trials.loc[trials["trial"].eq(study.best_trial.number)].iloc[0].dropna().to_dict()
    best = RichPolicyParams.from_mapping(winning)
    best_theta = fit_adverse_theta(
        entry=np.asarray(dev_paths["entry"]), atr=np.asarray(dev_paths["atr"]),
        highs=np.asarray(dev_paths["high"]), lows=np.asarray(dev_paths["low"]),
        params=best, median_atr_fraction=_median_atr_fraction(dev_paths), side=args.side,
    ) if best.adverse_exit_enabled else float("nan")
    best = RichPolicyParams.from_mapping({**best.to_dict(), "adverse_exit_theta": best_theta})
    full_dev_median_atr = _median_atr_fraction(dev_paths)
    _write_json(output / "frozen_challenger.json", {
        "schema": "strict_r3_rich_simple_policy_challenger_v1",
        "side": args.side,
        "selection_window": f"{calibration_end.isoformat()}..{dev_end.isoformat()} validation; calibration {dev_start.isoformat()}..{calibration_end.isoformat()}",
        "score_cutoff_development_only": cutoff,
        "median_atr_fraction_fitted_on_complete_2024_development": full_dev_median_atr,
        "adverse_exit_theta_fitted_on_complete_2024_development": best_theta,
        "params": best.to_dict(),
        "objective": "median monthly net bps/trade - 0.5*monthly MAD - negative worst-month penalty",
        "cost_bps": 100.0,
    })
    dev_coverage.to_parquet(output / "development_path_coverage.parquet", index=False)

    all_comparisons: list[dict[str, Any]] = []
    all_months: list[pd.DataFrame] = []
    all_weeks: list[pd.DataFrame] = []
    all_exit_causes: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    correctness_rows: list[dict[str, Any]] = []
    for year in args.evaluation_year:
        start, end = pd.Timestamp(f"{year}-01-01", tz="UTC"), pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
        scored = population.loc[
            population["timestamp"].ge(start) & population["timestamp"].lt(end) & population["score"].ge(cutoff)
        ].copy()
        resolved, paths, coverage = materialize_paths(scored, bars_root=bars_root)
        coverage["year"] = year
        coverage_rows.append(coverage)
        if resolved.empty:
            continue
        year_median_atr = full_dev_median_atr
        for arm, params in (("incumbent", incumbent), ("rich_challenger", best)):
            replay, selected, summary, monthly, weekly = _run_policy(
                resolved, paths, params=params, median_atr=year_median_atr,
                theta_paths=dev_paths, side=args.side,
            )
            accepted = replay.loc[selected].copy()
            accepted["accepted"] = True
            accepted["arm"] = arm
            accepted["year"] = year
            cost_gap = pd.to_numeric(accepted["gross_bps"], errors="coerce") - pd.to_numeric(accepted["net_bps"], errors="coerce")
            exact_cost = bool(np.isclose(cost_gap.to_numpy(float), 100.0, atol=1e-8, rtol=0.0).all())
            if not exact_cost:
                raise AssertionError(f"{year} {arm}: policy cost is not applied exactly once")
            correctness_rows.append({
                "year": year, "arm": arm, "scored_before_outcome_join": int(len(scored)),
                "complete_outcome_rows": int(len(resolved)), "accepted_rows": int(len(accepted)),
                "candidate_ids_unique": bool(resolved["candidate_id"].is_unique),
                "cost_exactly_once": exact_cost,
                "all_entries_at_decision_open": True,
                "h12_path_bars": HORIZON_BARS,
            })
            accepted.to_parquet(output / f"accepted_{arm}_{year}.parquet", index=False)
            exits = accepted.groupby("exit_reason", dropna=False).agg(
                trades=("candidate_id", "size"),
                net_bps_per_trade=("net_bps", "mean"),
                total_net_bps=("net_bps", "sum"),
            ).reset_index()
            exits["year"] = year
            exits["arm"] = arm
            all_exit_causes.append(exits)
            all_comparisons.append({"year": year, "arm": arm, **summary})
            monthly["year"] = year; monthly["arm"] = arm
            weekly["year"] = year; weekly["arm"] = arm
            all_months.append(monthly); all_weeks.append(weekly)
    summary_frame = pd.DataFrame(all_comparisons)
    baseline = summary_frame.loc[summary_frame["arm"].eq("incumbent")].set_index("year")
    for metric in ("trades", "net_bps_per_trade", "gross_bps_per_trade", "total_net_bps", "max_drawdown_bps", "worst_month_net_bps_per_trade", "worst_week_net_bps_per_trade"):
        summary_frame[f"delta_vs_incumbent_{metric}"] = [
            value - float(baseline.loc[row.year, metric]) if row.arm == "rich_challenger" and row.year in baseline.index else 0.0
            for row, value in zip(summary_frame.itertuples(), summary_frame[metric])
        ]
    summary_frame.to_parquet(output / "comparison_yearly.parquet", index=False)
    monthly_frame = pd.concat(all_months, ignore_index=True) if all_months else pd.DataFrame()
    weekly_frame = pd.concat(all_weeks, ignore_index=True) if all_weeks else pd.DataFrame()
    monthly_frame.to_parquet(output / "comparison_monthly.parquet", index=False)
    weekly_frame.to_parquet(output / "comparison_weekly.parquet", index=False)
    (pd.concat(coverage_rows, ignore_index=True) if coverage_rows else pd.DataFrame()).to_parquet(
        output / "evaluation_path_coverage.parquet", index=False
    )
    exit_frame = pd.concat(all_exit_causes, ignore_index=True) if all_exit_causes else pd.DataFrame()
    exit_frame.to_parquet(output / "comparison_exit_causes.parquet", index=False)
    correctness = {
        "schema": "strict_r3_rich_simple_policy_correctness_v1",
        "status": "passed",
        "policy_fit": f"development only: calibration {dev_start.isoformat()}..{calibration_end.isoformat()}; selection {calibration_end.isoformat()}..{dev_end.isoformat()}. Evaluation years are never read by HPO.",
        "score_route": "strict-prequential upstream score only; no outcome validity field is used before path materialisation.",
        "entry": "decision timestamp open of the first 15-minute path bar",
        "horizon": "48 future 15-minute bars / H12",
        "cost": "gross_bps - net_bps equals exactly 100.0 bps for every accepted valid row",
        "portfolio": "identical score-only two-concurrent/two-per-timestamp/one-per-asset auction in both arms",
        "arm_checks": correctness_rows,
    }
    _write_json(output / "correctness_report.json", correctness)

    report = [
        "# Strict-R3 Rich SimplePolicyOptimiser — Offline Challenger",
        "",
        f"This is a source-aligned, side-local ({args.side}) policy-geometry comparison. It does not modify the live contract, the live exit monitor, or any open position.",
        "",
        "## Freeze and validation",
        "",
        f"- Development score cutoff (pre-2025 only): `{cutoff:.8f}`",
        (
            f"- HPO trials: `{len(study.trials)}` executed / `{args.trials}` ceiling; "
            f"no-improvement patience `{NO_IMPROVEMENT_PATIENCE}`; seed `{args.seed}`"
        ),
        f"- Complete-path 2024 development rows: `{len(development)}`",
        f"- Calibration: `{dev_start.date()}`..`{calibration_end.date()}`. Selection: `{calibration_end.date()}`..`{dev_end.date()}`. Evaluation: {', '.join(map(str, args.evaluation_year))} held score rows.",
        "- Entry is the decision-time next-bar 15-minute open; 48 future bars form the H12 outcome. Cost is 100 bps once.",
        "- Portfolio auction is fixed for both arms: two concurrent positions, two new entries per timestamp, one per asset; priority is the frozen prequential upstream score.",
        "",
        "## Frozen winning challenger parameters",
        "",
        "```json",
        json.dumps(best.to_dict(), indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## Matched yearly portfolio replay",
        "",
        _as_report_table(summary_frame),
        "",
        "## Month-level replay",
        "",
        _as_report_table(monthly_frame),
        "",
        "## Exit-cause reconciliation",
        "",
        _as_report_table(exit_frame),
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")
    _write_json(output / "run_manifest.json", {
        "schema": "strict_r3_rich_simple_policy_hpo_v1",
        "run_type": "offline_research_only",
        "side": args.side,
        "side_local": True,
        "score_source": score_source_manifest,
        "incumbent": str(incumbent_file), "incumbent_sha256": _sha256(incumbent_file),
        "bars_root": str(bars_root), "horizon_bars": HORIZON_BARS, "cost_bps_once": 100.0,
        "selection_window": {"start": dev_start.isoformat(), "calibration_end": calibration_end.isoformat(), "end": dev_end.isoformat()},
        "evaluation_years": list(map(int, args.evaluation_year)),
        "retained_fraction": float(args.retained_fraction), "score_cutoff": cutoff,
        "cap_per_month": int(args.cap_per_month),
        "requested_trials": int(args.trials),
        "executed_trials": int(len(study.trials)),
        "no_improvement_patience": NO_IMPROVEMENT_PATIENCE,
        "stop_reason": study.user_attrs.get("stop_reason", "trial_budget"),
        "trials_since_improvement": study.user_attrs.get("trials_since_improvement", 0),
        "seed": int(args.seed),
        "hard_max_sl_cap_pct": float(args.max_sl_cap_pct) if args.max_sl_cap_pct is not None else None,
        "require_rich_extensions": bool(args.require_rich_extensions),
        "portfolio": {"max_concurrent": 2, "max_new_per_timestamp": 2, "max_per_asset": 1},
        "hpo_objective": "median monthly net bps/trade - 0.5*monthly MAD - max(0,-worst_month)",
        "prohibitions": ["no_live_state", "no_exchange_io", "no_2025_2026_outcomes_in_hpo", "no_policy_path_valid_routing"],
    })
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger", type=Path, action="append", default=None,
        help="Strict-prequential parquet file or producer root; repeat for non-overlapping ledger roots.",
    )
    parser.add_argument(
        "--targetfree-score-root", type=Path,
        help="Alternative retained target-free score root with month=YYYY-MM parquet partitions.",
    )
    parser.add_argument(
        "--targetfree-score-column", default="router_primary_rank",
        help="Point-in-time score column used only for pre-outcome development routing.",
    )
    parser.add_argument("--bars-root", type=Path, default=DEFAULT_BARS)
    parser.add_argument("--incumbent", type=Path, default=DEFAULT_INCUMBENT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--trials", type=int, default=48)
    parser.add_argument("--retained-fraction", type=float, default=0.05)
    parser.add_argument("--cap-per-month", type=int, default=3500)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--max-sl-cap-pct", type=float,
        help="Hard absolute maximum loss distance for every HPO trial, e.g. 0.05 for 5%%.",
    )
    parser.add_argument(
        "--require-rich-extensions", action="store_true",
        help="Require smooth capital protection and fast-adverse exit in every trial.",
    )
    parser.add_argument("--development-start", default="2024-01-01")
    parser.add_argument("--calibration-end", default="2024-07-01")
    parser.add_argument("--development-end", default="2025-01-01")
    parser.add_argument(
        "--evaluation-year", type=int, action="append", default=[],
        help="Optional held calendar year for post-selection diagnostics; repeatable.",
    )
    parser.add_argument("--side", choices=("long", "short"), default="long")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.targetfree_score_root is not None and args.ledger:
        parser.error("--targetfree-score-root and --ledger are mutually exclusive")
    if not args.evaluation_year:
        args.evaluation_year = [2025, 2026]
    return args


if __name__ == "__main__":
    destination = run(parse_args())
    print(destination)
