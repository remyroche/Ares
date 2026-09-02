#!/usr/bin/env python3
"""Backtest opportunity-pressure exit tightening on accepted policy trades.

The experiment keeps the entry policy fixed, then re-simulates accepted trades
with a lightweight 1-minute OHLC path engine. The goal is to test whether
positions should be exited/tightened when the portfolio is crowded and better
same-timestamp candidates are waiting.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:  # pragma: no cover - exercised by runtime jobs.
    import optuna
except Exception:  # pragma: no cover
    optuna = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # pragma: no cover - exercised by runtime smoke in this repo.
    from numba import njit
except Exception:  # pragma: no cover
    njit = None

from extreme_price_movements.portfolio_policy_replay import (  # noqa: E402
    fit_hierarchical_ev_curves,
    normalise_candidate_table,
    portfolio_policy_params_from_live_config,
    replay_candidates,
)


DEFAULT_POLICY_CANDIDATES = Path(
    "data_perp/artifacts/20260629_050000_lgbm_mda"
    "/simple_policy_optimiser/simple_policy_candidates_deployable.parquet"
)
DEFAULT_EDGE_CANDIDATES = Path(
    "data_perp/artifacts/finalfit_candidate_mask_native_candidates_20260627_6mo"
    "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_POLICY = Path(
    "data_perp/artifacts/20260629_050000_lgbm_mda"
    "/portfolio_policy_replay/optimized_portfolio_policy_config_perps.json"
)
DEFAULT_ARTIFACT_DECISIONS = Path(
    "data_perp/artifacts/20260629_050000_lgbm_mda"
    "/portfolio_policy_replay/per_candidate_replay_decisions.parquet"
)
DEFAULT_OHLCV_ROOT = Path("data_perp/exchanges/krakenfutures/execution_1m/ohlcv")
DEFAULT_OUT = Path("data_perp/reports/exit_tightening_redeploy_mar_apr_jun_20260630")
PERIOD_WINDOWS = {
    "mar": ("2026-03-01T00:00:00+00:00", "2026-04-01T00:00:00+00:00"),
    "apr": ("2026-04-01T00:00:00+00:00", "2026-05-01T00:00:00+00:00"),
    "may": ("2026-05-01T00:00:00+00:00", "2026-06-01T00:00:00+00:00"),
    "jun": ("2026-06-01T00:00:00+00:00", "2026-06-27T00:00:00+00:00"),
}
HEAD_ALIASES = {"short_bollinger": "short_boll"}
HEAD_PREFIXES = ("long_bars", "long_dist", "short_asset", "short_boll")
MAX_PATH_MINUTES = 64
EPS = 1e-9


@dataclass(frozen=True)
class ExitTighteningConfig:
    config_id: str
    candidate_edge_quantile: float
    pressure_mode: str
    pressure_mid: float
    pressure_power: float
    churn_penalty_bps: float
    exit_hysteresis_bps: float
    base_stop_loss_bps: float
    min_stop_loss_bps: float
    base_trailing_gap_bps: float
    min_trailing_gap_bps: float
    base_tp_remaining_bps: float
    min_tp_remaining_bps: float
    pressure_use_mode: str


@dataclass(frozen=True)
class IndependentExitTighteningConfig:
    config_id: str
    sl_candidate_edge_quantile: float
    trail_candidate_edge_quantile: float
    sl_pressure_mode: str
    trail_pressure_mode: str
    sl_pressure_mid: float
    trail_pressure_mid: float
    sl_pressure_power: float
    trail_pressure_power: float
    sl_churn_penalty_bps: float
    trail_churn_penalty_bps: float
    sl_hysteresis_bps: float
    trail_hysteresis_bps: float
    sl_ev_weight: float
    trail_ev_weight: float
    trail_profit_weight: float
    base_stop_loss_bps: float
    min_stop_loss_bps: float
    base_trailing_gap_bps: float
    min_trailing_gap_bps: float
    base_tp_remaining_bps: float
    min_tp_remaining_bps: float
    sl_formula: str
    trail_formula: str
    force_exit_mode: str
    force_hysteresis_mult: float


def _utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _strategy_head(strategy_id: Any) -> str:
    raw = str(strategy_id)
    for alias, head in HEAD_ALIASES.items():
        if raw == alias or raw.startswith(f"{alias}_"):
            return head
    for prefix in HEAD_PREFIXES:
        if raw == prefix or raw.startswith(f"{prefix}_"):
            return prefix
    return raw.split("_", 1)[0] if raw else "unknown"


def _side_code(value: Any, strategy_id: Any = "") -> int:
    text = str(value).strip().lower()
    if text in {"1", "1.0", "long", "l"}:
        return 1
    if text in {"-1", "-1.0", "short", "s"}:
        return -1
    sid = str(strategy_id).strip().lower()
    return -1 if sid.startswith("short") or sid.startswith("s") else 1


def _symbol_dir_name(symbol: Any) -> str:
    return str(symbol).replace("/", "_")


def _load_policy(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    return portfolio_policy_params_from_live_config(payload), payload


def _candidate_columns(path: Path) -> list[str]:
    cols = pq.ParquetFile(path).schema.names
    wanted = [
        "timestamp",
        "symbol",
        "strategy_id",
        "side",
        "normalized_rank_score",
        "strategy_rank_pct",
        "rank_pct",
        "policy_rank_pct",
        "base_strategy_threshold",
        "calibrated_score",
        "entry_price",
        "policy_executable_entry_price",
        "delayed_entry_ts",
        "delayed_entry_effective_ts",
        "entry_delay_actual_minutes",
        "exit_timestamp",
        "exit_price",
        "net_return",
        "gross_return",
        "holding_bars",
        "barrier_pct",
        "policy_effective_barrier_pct",
        "expected_friction_bps",
        "expected_spread_bps",
        "expected_half_spread_bps",
        "spread_cost_bps",
        "exit_quote_half_spread_bps",
        "exit_spread_cost_bps",
        "fees_bps",
        "slippage_bps",
        "entry_slippage_proxy_bps",
        "price_gap_bps",
        "simple_policy_exit_reason",
        "liquidity_capacity_weight",
        "portfolio_fixed_position_size",
    ]
    return [c for c in wanted if c in cols]


def _load_candidates(path: Path, periods: list[str], *, global_floor: float) -> pd.DataFrame:
    frame = pq.read_table(path, columns=_candidate_columns(path)).to_pandas()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["exit_timestamp"] = pd.to_datetime(frame["exit_timestamp"], utc=True, errors="coerce")
    mask = np.zeros(len(frame), dtype=bool)
    for period in periods:
        start, end = PERIOD_WINDOWS[str(period)]
        mask |= frame["timestamp"].ge(_utc(start)) & frame["timestamp"].lt(_utc(end))
    out = frame.loc[mask].copy()
    out["symbol"] = out["symbol"].astype(str)
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["side"] = [
        "long" if _side_code(s, sid) > 0 else "short"
        for s, sid in zip(out.get("side", ""), out["strategy_id"])
    ]
    out["head"] = out["strategy_id"].map(_strategy_head)
    if "calibrated_score" not in out.columns:
        out["calibrated_score"] = pd.to_numeric(out["normalized_rank_score"], errors="coerce")
    if "base_strategy_threshold" not in out.columns:
        out["base_strategy_threshold"] = float(global_floor)
    out["base_strategy_threshold"] = (
        pd.to_numeric(out["base_strategy_threshold"], errors="coerce")
        .fillna(float(global_floor))
        .clip(0.0, 0.999)
    )
    out["normalized_rank_score"] = (
        pd.to_numeric(out["normalized_rank_score"], errors="coerce")
        .fillna(pd.to_numeric(out.get("rank_pct"), errors="coerce"))
    )
    if "liquidity_capacity_weight" not in out.columns:
        out["liquidity_capacity_weight"] = 1.0
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(
        subset=[
            "timestamp",
            "exit_timestamp",
            "symbol",
            "strategy_id",
            "normalized_rank_score",
            "base_strategy_threshold",
            "entry_price",
            "net_return",
            "gross_return",
        ]
    )
    out = out.sort_values(["timestamp", "strategy_id", "symbol"]).reset_index(drop=True)
    out["candidate_row_id"] = np.arange(len(out), dtype=np.int64)
    return out


def _period_mask(frame: pd.DataFrame, periods: list[str]) -> pd.Series:
    timestamps = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    mask = pd.Series(False, index=frame.index)
    for period in periods:
        if period not in PERIOD_WINDOWS:
            raise ValueError(f"Unknown period {period!r}; known periods={sorted(PERIOD_WINDOWS)}")
        start, end = PERIOD_WINDOWS[str(period)]
        mask |= timestamps.ge(_utc(start)) & timestamps.lt(_utc(end))
    return mask


def _entry_time_series(frame: pd.DataFrame) -> pd.Series:
    base = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for col in ("delayed_entry_effective_ts", "delayed_entry_ts"):
        if col in frame.columns:
            candidate = pd.to_datetime(frame[col], utc=True, errors="coerce")
            base = candidate.where(candidate.notna(), base)
            break
    return base


def _load_ev_candidates(path: Path, *, cutoff: pd.Timestamp, global_floor: float) -> pd.DataFrame:
    frame = pq.read_table(path, columns=_candidate_columns(path)).to_pandas()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["exit_timestamp"] = pd.to_datetime(frame["exit_timestamp"], utc=True, errors="coerce")
    frame = frame.loc[frame["timestamp"].lt(cutoff)].copy()
    if frame.empty:
        return frame
    frame["symbol"] = frame["symbol"].astype(str)
    frame["strategy_id"] = frame["strategy_id"].astype(str)
    frame["side"] = [
        "long" if _side_code(s, sid) > 0 else "short"
        for s, sid in zip(frame.get("side", ""), frame["strategy_id"])
    ]
    if "calibrated_score" not in frame.columns:
        frame["calibrated_score"] = pd.to_numeric(frame["normalized_rank_score"], errors="coerce")
    if "base_strategy_threshold" not in frame.columns:
        frame["base_strategy_threshold"] = float(global_floor)
    frame["base_strategy_threshold"] = (
        pd.to_numeric(frame["base_strategy_threshold"], errors="coerce")
        .fillna(float(global_floor))
        .clip(0.0, 0.999)
    )
    if "liquidity_capacity_weight" not in frame.columns:
        frame["liquidity_capacity_weight"] = 1.0
    return frame.dropna(
        subset=["timestamp", "exit_timestamp", "normalized_rank_score", "base_strategy_threshold", "net_return", "gross_return"]
    ).reset_index(drop=True)


def _edge_and_cost_columns(
    frame: pd.DataFrame,
    *,
    spread_floor_bps: float,
    execution_gap_bps: float,
) -> pd.DataFrame:
    out = frame.copy()
    rank = pd.to_numeric(out.get("normalized_rank_score"), errors="coerce").fillna(0.5)
    barrier = (
        pd.to_numeric(out.get("policy_effective_barrier_pct"), errors="coerce")
        .fillna(pd.to_numeric(out.get("barrier_pct"), errors="coerce"))
        .fillna(0.005)
        .clip(lower=1e-4)
    )
    fees = pd.to_numeric(out.get("fees_bps"), errors="coerce").fillna(20.0).clip(lower=0.0)
    spread = pd.to_numeric(out.get("expected_spread_bps"), errors="coerce").fillna(0.0)
    spread = np.maximum(spread.to_numpy(dtype=np.float64), float(spread_floor_bps))
    friction = pd.to_numeric(out.get("expected_friction_bps"), errors="coerce").fillna(0.0)
    friction = np.maximum(
        friction.to_numpy(dtype=np.float64),
        fees.to_numpy(dtype=np.float64) + spread + float(execution_gap_bps),
    )
    out["_edge_gross_bps"] = ((rank.to_numpy(dtype=np.float64) - 0.5) * 2.0 * barrier.to_numpy(dtype=np.float64) * 10_000.0).astype(np.float32)
    out["_friction_bps"] = np.maximum(friction, 0.0).astype(np.float32)
    out["_edge_net_bps"] = (out["_edge_gross_bps"].to_numpy(dtype=np.float32) - out["_friction_bps"].to_numpy(dtype=np.float32)).astype(np.float32)
    return out


def _timestamp_edge_table(frame: pd.DataFrame, *, global_floor: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eligible = frame.loc[
        pd.to_numeric(frame["normalized_rank_score"], errors="coerce").ge(float(global_floor))
    ]
    for ts, group in eligible.groupby("timestamp", sort=True):
        gross = pd.to_numeric(group["_edge_gross_bps"], errors="coerce").to_numpy(dtype=np.float64)
        friction = pd.to_numeric(group["_friction_bps"], errors="coerce").to_numpy(dtype=np.float64)
        gross = gross[np.isfinite(gross)]
        friction = friction[np.isfinite(friction)]
        rec = {"timestamp": pd.Timestamp(ts)}
        for q in (0.65, 0.75, 0.85):
            rec[f"candidate_edge_p{int(q * 100)}_bps"] = float(np.quantile(gross, q)) if gross.size else 0.0
        rec["candidate_fees_and_slippage_p75_bps"] = float(np.quantile(friction, 0.75)) if friction.size else 0.0
        rec["candidate_rows_above_floor"] = int(len(group))
        rows.append(rec)
    return pd.DataFrame(rows)


def _accepted_trades(candidates: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"]].copy()
    if accepted.empty:
        return candidates.iloc[0:0].copy()
    idx = pd.to_numeric(accepted["candidate_index"], errors="coerce").astype("int64").to_numpy()
    rows = candidates.iloc[idx].copy().reset_index(drop=True)
    decision_cols = [
        "candidate_index",
        "position_size",
        "open_positions_before",
        "open_positions_after",
        "wallet_before",
        "wallet_after",
        "open_notional_before",
        "open_notional_after",
        "dynamic_threshold",
        "portfolio_priority",
        "position_exit_timestamp",
        "position_net_return",
        "position_gross_return",
        "position_exit_reason",
        "position_exit_price",
    ]
    for col in decision_cols:
        if col in accepted.columns:
            rows[col] = accepted[col].to_numpy()
    if "position_net_return" in rows.columns:
        adjusted = pd.to_numeric(rows["position_net_return"], errors="coerce")
        rows["net_return"] = adjusted.where(adjusted.notna(), rows["net_return"])
    if "position_gross_return" in rows.columns:
        adjusted = pd.to_numeric(rows["position_gross_return"], errors="coerce")
        rows["gross_return"] = adjusted.where(adjusted.notna(), rows["gross_return"])
    if "position_exit_timestamp" in rows.columns:
        adjusted_ts = pd.to_datetime(rows["position_exit_timestamp"], utc=True, errors="coerce")
        rows["exit_timestamp"] = adjusted_ts.where(adjusted_ts.notna(), rows["exit_timestamp"])
    if "position_exit_price" in rows.columns:
        adjusted = pd.to_numeric(rows["position_exit_price"], errors="coerce")
        rows["exit_price"] = adjusted.where(adjusted.notna(), rows["exit_price"])
    if "position_exit_reason" in rows.columns:
        adjusted = rows["position_exit_reason"].astype(str)
        rows["simple_policy_exit_reason"] = adjusted.where(
            adjusted.str.len() > 0,
            rows.get("simple_policy_exit_reason", ""),
        )
    rows["baseline_net_pnl"] = (
        pd.to_numeric(rows["position_size"], errors="coerce").fillna(0.0)
        * pd.to_numeric(rows["net_return"], errors="coerce").fillna(0.0)
    )
    rows["baseline_gross_pnl"] = (
        pd.to_numeric(rows["position_size"], errors="coerce").fillna(0.0)
        * pd.to_numeric(rows["gross_return"], errors="coerce").fillna(0.0)
    )
    return rows


def _decision_keyed(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out["symbol"] = out["symbol"].astype(str)
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["side"] = [_side_code(s, sid) for s, sid in zip(out.get("side", ""), out["strategy_id"])]
    return out


def _max_abs_delta(merged: pd.DataFrame, column: str) -> float | None:
    left = f"{column}_local"
    right = f"{column}_artifact"
    if left not in merged.columns or right not in merged.columns:
        return None
    lhs = pd.to_numeric(merged[left], errors="coerce")
    rhs = pd.to_numeric(merged[right], errors="coerce")
    delta = (lhs - rhs).abs().replace([np.inf, -np.inf], np.nan).dropna()
    return float(delta.max()) if len(delta) else None


def _replay_parity_audit(
    *,
    local_decisions: pd.DataFrame,
    artifact_decisions_path: Path | None,
    periods: list[str],
    out_dir: Path,
) -> dict[str, Any]:
    if artifact_decisions_path is None or not Path(artifact_decisions_path).exists():
        return {
            "artifact_decisions_path": str(artifact_decisions_path) if artifact_decisions_path else "",
            "artifact_decisions_available": False,
        }
    artifact = pd.read_parquet(artifact_decisions_path)
    artifact = artifact.loc[_period_mask(artifact, periods)].copy()
    local = local_decisions.copy()
    keys = ["timestamp", "symbol", "side", "strategy_id"]
    local = _decision_keyed(local)
    artifact = _decision_keyed(artifact)
    local_dup = int(local.duplicated(keys).sum())
    artifact_dup = int(artifact.duplicated(keys).sum())
    merged = local.merge(
        artifact,
        on=keys,
        how="outer",
        suffixes=("_local", "_artifact"),
        indicator=True,
    )
    matched = merged.loc[merged["_merge"].eq("both")].copy()
    mismatch_cols = [
        "timestamp",
        "symbol",
        "side",
        "strategy_id",
        "_merge",
        "accepted_local",
        "accepted_artifact",
        "rejection_reason_local",
        "rejection_reason_artifact",
        "normalized_rank_score_local",
        "normalized_rank_score_artifact",
        "dynamic_threshold_local",
        "dynamic_threshold_artifact",
        "portfolio_priority_local",
        "portfolio_priority_artifact",
        "position_size_local",
        "position_size_artifact",
        "open_positions_before_local",
        "open_positions_before_artifact",
        "open_positions_after_local",
        "open_positions_after_artifact",
        "wallet_before_local",
        "wallet_before_artifact",
        "open_notional_before_local",
        "open_notional_before_artifact",
    ]
    mismatch_mask = ~merged["_merge"].eq("both")
    if len(matched):
        mismatch_mask |= merged.get("accepted_local").fillna(False).astype(bool) != merged.get(
            "accepted_artifact"
        ).fillna(False).astype(bool)
        for col in [
            "normalized_rank_score",
            "effective_rank_score",
            "base_threshold",
            "dynamic_threshold",
            "portfolio_priority",
            "position_size",
            "open_positions_before",
            "open_positions_after",
            "wallet_before",
            "wallet_after",
            "open_notional_before",
            "open_notional_after",
        ]:
            left = f"{col}_local"
            right = f"{col}_artifact"
            if left in merged.columns and right in merged.columns:
                lhs = pd.to_numeric(merged[left], errors="coerce")
                rhs = pd.to_numeric(merged[right], errors="coerce")
                mismatch_mask |= (lhs - rhs).abs().gt(1e-8).fillna(False)
    mismatch_cols = [c for c in mismatch_cols if c in merged.columns]
    mismatches = merged.loc[mismatch_mask, mismatch_cols].copy()
    if len(mismatches) > 50_000:
        mismatches = mismatches.head(50_000)
    mismatches.to_csv(out_dir / "replay_parity_mismatches.csv", index=False)

    accepted_match_rate = None
    rejection_match_rate = None
    if len(matched):
        local_acc = matched["accepted_local"].fillna(False).astype(bool)
        art_acc = matched["accepted_artifact"].fillna(False).astype(bool)
        accepted_match_rate = float((local_acc == art_acc).mean())
        rejected = ~(local_acc | art_acc)
        if rejected.any():
            rejection_match_rate = float(
                (
                    matched.loc[rejected, "rejection_reason_local"].astype(str)
                    == matched.loc[rejected, "rejection_reason_artifact"].astype(str)
                ).mean()
            )

    audit: dict[str, Any] = {
        "artifact_decisions_path": str(artifact_decisions_path),
        "artifact_decisions_available": True,
        "periods": periods,
        "local_rows": int(len(local)),
        "artifact_rows": int(len(artifact)),
        "matched_rows": int(len(matched)),
        "local_only_rows": int(merged["_merge"].eq("left_only").sum()),
        "artifact_only_rows": int(merged["_merge"].eq("right_only").sum()),
        "local_duplicate_keys": local_dup,
        "artifact_duplicate_keys": artifact_dup,
        "local_accepted": int(local.get("accepted", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "artifact_accepted": int(artifact.get("accepted", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "accepted_match_rate": accepted_match_rate,
        "rejection_reason_match_rate": rejection_match_rate,
        "mismatch_rows_written": int(len(mismatches)),
    }
    for col in [
        "normalized_rank_score",
        "effective_rank_score",
        "base_threshold",
        "dynamic_threshold",
        "portfolio_priority",
        "position_size",
        "wallet_before",
        "wallet_after",
        "open_notional_before",
        "open_notional_after",
    ]:
        audit[f"max_abs_delta_{col}"] = _max_abs_delta(matched, col)
    (out_dir / "replay_parity_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return audit


def _attach_artifact_position_sizes(
    candidates: pd.DataFrame,
    artifact_decisions_path: Path | None,
    periods: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if artifact_decisions_path is None or not Path(artifact_decisions_path).exists():
        return candidates, {
            "artifact_position_size_path": str(artifact_decisions_path) if artifact_decisions_path else "",
            "artifact_position_size_available": False,
        }
    decisions = pd.read_parquet(artifact_decisions_path)
    decisions = decisions.loc[_period_mask(decisions, periods)].copy()
    keys = ["timestamp", "symbol", "side", "strategy_id"]
    if decisions.empty:
        return candidates, {
            "artifact_position_size_path": str(artifact_decisions_path),
            "artifact_position_size_available": True,
            "artifact_position_size_decision_rows": 0,
            "artifact_position_size_accepted_rows": 0,
            "artifact_position_size_candidate_rows": int(len(candidates)),
            "artifact_position_size_matched_rows": 0,
            "artifact_position_size_matched_pct": 0.0,
        }
    size_frame = _decision_keyed(decisions)
    accepted_mask = size_frame.get("accepted", pd.Series(False, index=size_frame.index))
    size_frame = size_frame.loc[
        accepted_mask.fillna(False).astype(bool),
        keys + ["position_size"],
    ].copy()
    size_frame = size_frame.rename(columns={"position_size": "portfolio_fixed_position_size"})
    out = _decision_keyed(candidates)
    if "portfolio_fixed_position_size" in out.columns:
        out = out.drop(columns=["portfolio_fixed_position_size"])
    out = out.merge(size_frame, on=keys, how="left")
    out["side"] = ["long" if int(value) > 0 else "short" for value in out["side"]]
    fixed = pd.to_numeric(out["portfolio_fixed_position_size"], errors="coerce")
    matched = int(fixed.notna().sum())
    return out, {
        "artifact_position_size_path": str(artifact_decisions_path),
        "artifact_position_size_available": True,
        "artifact_position_size_decision_rows": int(len(decisions)),
        "artifact_position_size_accepted_rows": int(len(size_frame)),
        "artifact_position_size_candidate_rows": int(len(candidates)),
        "artifact_position_size_matched_rows": matched,
        "artifact_position_size_matched_pct": float(matched / max(len(candidates), 1) * 100.0),
    }


def _read_symbol_1m(root: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    sym_dir = root / f"symbol={_symbol_dir_name(symbol)}"
    if not sym_dir.exists():
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
    years = sorted({start.year, end.year})
    parts: list[pd.DataFrame] = []
    for year in years:
        year_dir = sym_dir / f"year={year}"
        if not year_dir.exists():
            continue
        files = sorted(year_dir.glob("*.parquet"))
        if not files:
            continue
        table = pq.read_table(
            files,
            columns=["ts", "open", "high", "low", "close"],
        )
        part = table.to_pandas()
        part["ts"] = pd.to_datetime(part["ts"], utc=True, errors="coerce")
        part = part.loc[part["ts"].ge(start) & part["ts"].le(end)].copy()
        if not part.empty:
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close"])
    out = pd.concat(parts, ignore_index=True)
    return out.dropna(subset=["ts"]).sort_values("ts").drop_duplicates("ts", keep="last")


def _build_paths(
    trades: pd.DataFrame,
    *,
    ohlcv_root: Path,
    max_path_minutes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(trades)
    shape = (n, int(max_path_minutes) + 1)
    opens = np.full(shape, np.nan, dtype=np.float32)
    highs = np.full(shape, np.nan, dtype=np.float32)
    lows = np.full(shape, np.nan, dtype=np.float32)
    closes = np.full(shape, np.nan, dtype=np.float32)
    coverage = np.zeros(n, dtype=np.int16)
    if trades.empty:
        return opens, highs, lows, closes, coverage
    entry_times = _entry_time_series(trades)
    start = _utc(entry_times.min()) - pd.Timedelta(minutes=2)
    end = _utc(entry_times.max()) + pd.Timedelta(minutes=int(max_path_minutes) + 2)
    for symbol, idx in trades.groupby("symbol", sort=False).groups.items():
        px = _read_symbol_1m(ohlcv_root, str(symbol), start, end)
        if px.empty:
            continue
        ts_ns = px["ts"].astype("int64").to_numpy()
        arr_o = pd.to_numeric(px["open"], errors="coerce").to_numpy(dtype=np.float32)
        arr_h = pd.to_numeric(px["high"], errors="coerce").to_numpy(dtype=np.float32)
        arr_l = pd.to_numeric(px["low"], errors="coerce").to_numpy(dtype=np.float32)
        arr_c = pd.to_numeric(px["close"], errors="coerce").to_numpy(dtype=np.float32)
        row_pos = np.asarray(list(idx), dtype=np.int64)
        entry_ns = entry_times.loc[row_pos].astype("int64").to_numpy()
        for out_i, start_ns in zip(row_pos, entry_ns):
            pos = int(np.searchsorted(ts_ns, int(start_ns), side="left"))
            if pos >= len(ts_ns):
                continue
            take = min(shape[1], len(ts_ns) - pos)
            # Require the first bar to be no more than two minutes after executable entry time.
            if int(ts_ns[pos]) - int(start_ns) > int(pd.Timedelta(minutes=2).value):
                continue
            opens[out_i, :take] = arr_o[pos : pos + take]
            highs[out_i, :take] = arr_h[pos : pos + take]
            lows[out_i, :take] = arr_l[pos : pos + take]
            closes[out_i, :take] = arr_c[pos : pos + take]
            coverage[out_i] = int(take)
    return opens, highs, lows, closes, coverage


if njit is not None:

    @njit(cache=True)
    def _simulate_exit_grid_numba(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        side: np.ndarray,
        entry_price: np.ndarray,
        hold_minutes: np.ndarray,
        cost_bps: np.ndarray,
        edge_net_bps: np.ndarray,
        candidate_edge_bps: np.ndarray,
        candidate_friction_bps: np.ndarray,
        count_ratio: np.ndarray,
        notional_ratio: np.ndarray,
        params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = entry_price.shape[0]
        k = params.shape[0]
        net = np.empty((k, n), dtype=np.float32)
        gross = np.empty((k, n), dtype=np.float32)
        exit_minute = np.empty((k, n), dtype=np.int16)
        reason = np.empty((k, n), dtype=np.int8)
        for c in range(k):
            edge_q = params[c, 0]
            pressure_mode = int(params[c, 1])
            pressure_mid = params[c, 2]
            pressure_power = params[c, 3]
            churn = params[c, 4]
            hysteresis = max(params[c, 5], 1e-6)
            base_sl = params[c, 6]
            min_sl = params[c, 7]
            base_trail = params[c, 8]
            min_trail = params[c, 9]
            base_tp = params[c, 10]
            min_tp = params[c, 11]
            use_mode = int(params[c, 12])
            for i in range(n):
                ent = entry_price[i]
                s = side[i]
                if not math.isfinite(ent) or ent <= 0.0 or s == 0:
                    net[c, i] = 0.0
                    gross[c, i] = 0.0
                    exit_minute[c, i] = 0
                    reason[c, i] = 6
                    continue
                hm = hold_minutes[i]
                if hm < 1:
                    hm = 1
                max_j = min(hm, opens.shape[1] - 1)
                if max_j < 1:
                    max_j = min(opens.shape[1] - 1, 1)
                pressure_ratio = count_ratio[i] if pressure_mode == 0 else notional_ratio[i]
                capital_pressure = (pressure_ratio - pressure_mid) / max(1.0 - pressure_mid, 1e-6)
                if capital_pressure < 0.0:
                    capital_pressure = 0.0
                if capital_pressure > 1.0:
                    capital_pressure = 1.0
                capital_pressure = capital_pressure ** pressure_power
                redeploy_edge = candidate_edge_bps[i] - candidate_friction_bps[i]
                if redeploy_edge < 0.0:
                    redeploy_edge = 0.0
                max_fav_bps = 0.0
                prev_close = closes[i, 0]
                if not math.isfinite(prev_close) or prev_close <= 0.0:
                    prev_close = ent
                out_gross = 0.0
                out_min = max_j
                out_reason = 3
                exited = False
                for j in range(1, max_j + 1):
                    o = opens[i, j]
                    h = highs[i, j]
                    l = lows[i, j]
                    cl = closes[i, j]
                    if not math.isfinite(o) or not math.isfinite(h) or not math.isfinite(l) or not math.isfinite(cl):
                        continue
                    prev_unrl = s * (prev_close / ent - 1.0) * 10000.0
                    remaining_frac = max(0.0, 1.0 - (j - 1) / max(float(hm), 1.0))
                    current_ev_remaining = edge_net_bps[i] * remaining_frac - max(prev_unrl, 0.0)
                    if current_ev_remaining < 0.0:
                        current_ev_remaining = 0.0
                    exit_adv = capital_pressure * redeploy_edge - current_ev_remaining - churn
                    pressure = exit_adv / hysteresis
                    if pressure < 0.0:
                        pressure = 0.0
                    elif pressure > 1.0:
                        pressure = 1.0
                    if use_mode == 2:
                        p_stop = pressure * pressure
                        p_trail = math.sqrt(pressure)
                        p_tp = pressure
                    elif use_mode == 1:
                        p_trail = pressure
                        p_stop = 0.0 if pressure < 0.5 else (pressure - 0.5) * 2.0
                        p_tp = 0.0 if pressure < 0.75 else (pressure - 0.75) * 4.0
                    else:
                        p_stop = pressure
                        p_trail = pressure
                        p_tp = pressure
                    sl_bps = base_sl - p_stop * (base_sl - min_sl)
                    trail_bps = base_trail - p_trail * (base_trail - min_trail)
                    tp_rem_bps = base_tp - p_tp * (base_tp - min_tp)
                    if exit_adv > hysteresis:
                        fill = o
                        out_gross = s * (fill / ent - 1.0)
                        out_min = j
                        out_reason = 4
                        exited = True
                        break
                    fav_high = (h / ent - 1.0) * 10000.0 if s > 0 else (ent / l - 1.0) * 10000.0
                    if fav_high > max_fav_bps:
                        max_fav_bps = fav_high
                    # Overlay semantics: do not replace the original policy exit
                    # unless a tightened rule is active.
                    if p_stop > 0.0 and s > 0:
                        stop_px = ent * (1.0 - sl_bps / 10000.0)
                        if l <= stop_px:
                            fill = o if o <= stop_px else stop_px
                            out_gross = fill / ent - 1.0
                            out_min = j
                            out_reason = 1
                            exited = True
                            break
                    elif p_stop > 0.0:
                        stop_px = ent * (1.0 + sl_bps / 10000.0)
                        if h >= stop_px:
                            fill = o if o >= stop_px else stop_px
                            out_gross = ent / fill - 1.0
                            out_min = j
                            out_reason = 1
                            exited = True
                            break
                    if p_tp > 0.0:
                        target_fav = max(prev_unrl, 0.0) + tp_rem_bps
                        if target_fav < min_tp:
                            target_fav = min_tp
                        if fav_high >= target_fav:
                            if s > 0:
                                fill = ent * (1.0 + target_fav / 10000.0)
                                out_gross = fill / ent - 1.0
                            else:
                                fill = ent / (1.0 + target_fav / 10000.0)
                                out_gross = ent / fill - 1.0
                            out_min = j
                            out_reason = 2
                            exited = True
                            break
                    if p_trail > 0.0 and max_fav_bps > trail_bps:
                        trail_fav = max_fav_bps - trail_bps
                        if s > 0:
                            trail_px = ent * (1.0 + trail_fav / 10000.0)
                            if l <= trail_px:
                                fill = o if o <= trail_px else trail_px
                                out_gross = fill / ent - 1.0
                                out_min = j
                                out_reason = 5
                                exited = True
                                break
                        else:
                            trail_px = ent / (1.0 + trail_fav / 10000.0)
                            if h >= trail_px:
                                fill = o if o >= trail_px else trail_px
                                out_gross = ent / fill - 1.0
                                out_min = j
                                out_reason = 5
                                exited = True
                                break
                    prev_close = cl
                if not exited:
                    cl = closes[i, out_min]
                    if not math.isfinite(cl) or cl <= 0.0:
                        cl = prev_close
                    out_gross = s * (cl / ent - 1.0)
                    out_reason = 3
                gross[c, i] = out_gross
                net[c, i] = out_gross - cost_bps[i] / 10000.0
                exit_minute[c, i] = out_min
                reason[c, i] = out_reason
        return net, gross, exit_minute, reason

    @njit(cache=True)
    def _simulate_exit_one_numba(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        side: int,
        entry_price: float,
        hold_minutes: int,
        cost_bps: float,
        edge_net_bps: float,
        candidate_edge_bps: float,
        candidate_friction_bps: float,
        count_ratio: float,
        notional_ratio: float,
        params: np.ndarray,
    ) -> tuple[float, float, int, int]:
        ent = entry_price
        s = side
        if not math.isfinite(ent) or ent <= 0.0 or s == 0:
            return 0.0, 0.0, 0, 6
        hm = hold_minutes
        if hm < 1:
            hm = 1
        max_j = min(hm, opens.shape[0] - 1)
        if max_j < 1:
            max_j = min(opens.shape[0] - 1, 1)
        pressure_mode = int(params[1])
        pressure_mid = params[2]
        pressure_power = params[3]
        churn = params[4]
        hysteresis = max(params[5], 1e-6)
        base_sl = params[6]
        min_sl = params[7]
        base_trail = params[8]
        min_trail = params[9]
        base_tp = params[10]
        min_tp = params[11]
        use_mode = int(params[12])
        pressure_ratio = count_ratio if pressure_mode == 0 else notional_ratio
        capital_pressure = (pressure_ratio - pressure_mid) / max(1.0 - pressure_mid, 1e-6)
        if capital_pressure < 0.0:
            capital_pressure = 0.0
        elif capital_pressure > 1.0:
            capital_pressure = 1.0
        capital_pressure = capital_pressure ** pressure_power
        redeploy_edge = candidate_edge_bps - candidate_friction_bps
        if redeploy_edge < 0.0:
            redeploy_edge = 0.0
        max_fav_bps = 0.0
        prev_close = closes[0]
        if not math.isfinite(prev_close) or prev_close <= 0.0:
            prev_close = ent
        out_gross = 0.0
        out_min = max_j
        out_reason = 3
        exited = False
        for j in range(1, max_j + 1):
            o = opens[j]
            h = highs[j]
            l = lows[j]
            cl = closes[j]
            if not math.isfinite(o) or not math.isfinite(h) or not math.isfinite(l) or not math.isfinite(cl):
                continue
            prev_unrl = s * (prev_close / ent - 1.0) * 10000.0
            remaining_frac = max(0.0, 1.0 - (j - 1) / max(float(hm), 1.0))
            current_ev_remaining = edge_net_bps * remaining_frac - max(prev_unrl, 0.0)
            if current_ev_remaining < 0.0:
                current_ev_remaining = 0.0
            exit_adv = capital_pressure * redeploy_edge - current_ev_remaining - churn
            pressure = exit_adv / hysteresis
            if pressure < 0.0:
                pressure = 0.0
            elif pressure > 1.0:
                pressure = 1.0
            if use_mode == 2:
                p_stop = pressure * pressure
                p_trail = math.sqrt(pressure)
                p_tp = pressure
            elif use_mode == 1:
                p_trail = pressure
                p_stop = 0.0 if pressure < 0.5 else (pressure - 0.5) * 2.0
                p_tp = 0.0 if pressure < 0.75 else (pressure - 0.75) * 4.0
            else:
                p_stop = pressure
                p_trail = pressure
                p_tp = pressure
            sl_bps = base_sl - p_stop * (base_sl - min_sl)
            trail_bps = base_trail - p_trail * (base_trail - min_trail)
            tp_rem_bps = base_tp - p_tp * (base_tp - min_tp)
            if exit_adv > hysteresis:
                fill = o
                out_gross = s * (fill / ent - 1.0)
                out_min = j
                out_reason = 4
                exited = True
                break
            fav_high = (h / ent - 1.0) * 10000.0 if s > 0 else (ent / l - 1.0) * 10000.0
            if fav_high > max_fav_bps:
                max_fav_bps = fav_high
            if p_stop > 0.0 and s > 0:
                stop_px = ent * (1.0 - sl_bps / 10000.0)
                if l <= stop_px:
                    fill = o if o <= stop_px else stop_px
                    out_gross = fill / ent - 1.0
                    out_min = j
                    out_reason = 1
                    exited = True
                    break
            elif p_stop > 0.0:
                stop_px = ent * (1.0 + sl_bps / 10000.0)
                if h >= stop_px:
                    fill = o if o >= stop_px else stop_px
                    out_gross = ent / fill - 1.0
                    out_min = j
                    out_reason = 1
                    exited = True
                    break
            if p_tp > 0.0:
                target_fav = max(prev_unrl, 0.0) + tp_rem_bps
                if target_fav < min_tp:
                    target_fav = min_tp
                if fav_high >= target_fav:
                    if s > 0:
                        fill = ent * (1.0 + target_fav / 10000.0)
                        out_gross = fill / ent - 1.0
                    else:
                        fill = ent / (1.0 + target_fav / 10000.0)
                        out_gross = ent / fill - 1.0
                    out_min = j
                    out_reason = 2
                    exited = True
                    break
            if p_trail > 0.0 and max_fav_bps > trail_bps:
                trail_fav = max_fav_bps - trail_bps
                if s > 0:
                    trail_px = ent * (1.0 + trail_fav / 10000.0)
                    if l <= trail_px:
                        fill = o if o <= trail_px else trail_px
                        out_gross = fill / ent - 1.0
                        out_min = j
                        out_reason = 5
                        exited = True
                        break
                else:
                    trail_px = ent / (1.0 + trail_fav / 10000.0)
                    if h >= trail_px:
                        fill = o if o >= trail_px else trail_px
                        out_gross = ent / fill - 1.0
                        out_min = j
                        out_reason = 5
                        exited = True
                        break
            prev_close = cl
        if not exited:
            cl = closes[out_min]
            if not math.isfinite(cl) or cl <= 0.0:
                cl = prev_close
            out_gross = s * (cl / ent - 1.0)
            out_reason = 3
        return out_gross - cost_bps / 10000.0, out_gross, out_min, out_reason

    @njit(cache=True)
    def _pressure_transform(value: float, formula: int) -> float:
        if value < 0.0:
            value = 0.0
        elif value > 1.0:
            value = 1.0
        if formula == 1:
            return value * value
        if formula == 2:
            return math.sqrt(value)
        if formula == 3:
            if value < 0.5:
                return 0.0
            return (value - 0.5) * 2.0
        return value

    @njit(cache=True)
    def _simulate_exit_one_independent_numba(
        opens: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        side: int,
        entry_price: float,
        hold_minutes: int,
        cost_bps: float,
        edge_net_bps: float,
        sl_candidate_edge_bps: float,
        trail_candidate_edge_bps: float,
        candidate_friction_bps: float,
        count_ratio: float,
        notional_ratio: float,
        params: np.ndarray,
    ) -> tuple[float, float, int, int]:
        ent = entry_price
        s = side
        if not math.isfinite(ent) or ent <= 0.0 or s == 0:
            return 0.0, 0.0, 0, 6
        hm = hold_minutes
        if hm < 1:
            hm = 1
        max_j = min(hm, opens.shape[0] - 1)
        if max_j < 1:
            max_j = min(opens.shape[0] - 1, 1)
        sl_pressure_mode = int(params[0])
        trail_pressure_mode = int(params[1])
        sl_mid = params[2]
        trail_mid = params[3]
        sl_power = params[4]
        trail_power = params[5]
        sl_churn = params[6]
        trail_churn = params[7]
        sl_hyst = max(params[8], 1e-6)
        trail_hyst = max(params[9], 1e-6)
        sl_ev_weight = params[10]
        trail_ev_weight = params[11]
        trail_profit_weight = params[12]
        base_sl = params[13]
        min_sl = params[14]
        base_trail = params[15]
        min_trail = params[16]
        base_tp = params[17]
        min_tp = params[18]
        sl_formula = int(params[19])
        trail_formula = int(params[20])
        force_mode = int(params[21])
        force_mult = max(params[22], 0.0)

        sl_ratio = count_ratio if sl_pressure_mode == 0 else notional_ratio
        trail_ratio = count_ratio if trail_pressure_mode == 0 else notional_ratio
        sl_cap_pressure = (sl_ratio - sl_mid) / max(1.0 - sl_mid, 1e-6)
        trail_cap_pressure = (trail_ratio - trail_mid) / max(1.0 - trail_mid, 1e-6)
        if sl_cap_pressure < 0.0:
            sl_cap_pressure = 0.0
        elif sl_cap_pressure > 1.0:
            sl_cap_pressure = 1.0
        if trail_cap_pressure < 0.0:
            trail_cap_pressure = 0.0
        elif trail_cap_pressure > 1.0:
            trail_cap_pressure = 1.0
        sl_cap_pressure = sl_cap_pressure ** sl_power
        trail_cap_pressure = trail_cap_pressure ** trail_power
        sl_redeploy = sl_candidate_edge_bps - candidate_friction_bps
        trail_redeploy = trail_candidate_edge_bps - candidate_friction_bps
        if sl_redeploy < 0.0:
            sl_redeploy = 0.0
        if trail_redeploy < 0.0:
            trail_redeploy = 0.0

        max_fav_bps = 0.0
        prev_close = closes[0]
        if not math.isfinite(prev_close) or prev_close <= 0.0:
            prev_close = ent
        out_gross = 0.0
        out_min = max_j
        out_reason = 3
        exited = False
        for j in range(1, max_j + 1):
            o = opens[j]
            h = highs[j]
            l = lows[j]
            cl = closes[j]
            if not math.isfinite(o) or not math.isfinite(h) or not math.isfinite(l) or not math.isfinite(cl):
                continue
            prev_unrl = s * (prev_close / ent - 1.0) * 10000.0
            remaining_frac = max(0.0, 1.0 - (j - 1) / max(float(hm), 1.0))
            current_ev_remaining = edge_net_bps * remaining_frac - max(prev_unrl, 0.0)
            if current_ev_remaining < 0.0:
                current_ev_remaining = 0.0
            positive_profit_bps = max(prev_unrl, 0.0)
            sl_adv = sl_cap_pressure * sl_redeploy - sl_ev_weight * current_ev_remaining - sl_churn
            trail_adv = (
                trail_cap_pressure * trail_redeploy
                + trail_profit_weight * positive_profit_bps
                - trail_ev_weight * current_ev_remaining
                - trail_churn
            )
            sl_pressure = _pressure_transform(sl_adv / sl_hyst, sl_formula)
            trail_pressure = _pressure_transform(trail_adv / trail_hyst, trail_formula)
            sl_bps = base_sl - sl_pressure * (base_sl - min_sl)
            trail_bps = base_trail - trail_pressure * (base_trail - min_trail)
            tp_rem_bps = base_tp
            if trail_pressure > 0.0:
                tp_rem_bps = base_tp - trail_pressure * (base_tp - min_tp)

            force_threshold = force_mult * max(sl_hyst, trail_hyst)
            if force_mode == 1 and sl_adv > force_threshold:
                fill = o
                out_gross = s * (fill / ent - 1.0)
                out_min = j
                out_reason = 4
                exited = True
                break
            if force_mode == 2 and max(sl_adv, trail_adv) > force_threshold:
                fill = o
                out_gross = s * (fill / ent - 1.0)
                out_min = j
                out_reason = 4
                exited = True
                break

            fav_high = (h / ent - 1.0) * 10000.0 if s > 0 else (ent / l - 1.0) * 10000.0
            if fav_high > max_fav_bps:
                max_fav_bps = fav_high
            if sl_pressure > 0.0 and s > 0:
                stop_px = ent * (1.0 - sl_bps / 10000.0)
                if l <= stop_px:
                    fill = o if o <= stop_px else stop_px
                    out_gross = fill / ent - 1.0
                    out_min = j
                    out_reason = 1
                    exited = True
                    break
            elif sl_pressure > 0.0:
                stop_px = ent * (1.0 + sl_bps / 10000.0)
                if h >= stop_px:
                    fill = o if o >= stop_px else stop_px
                    out_gross = ent / fill - 1.0
                    out_min = j
                    out_reason = 1
                    exited = True
                    break
            if trail_pressure > 0.0:
                target_fav = max(prev_unrl, 0.0) + tp_rem_bps
                if target_fav < min_tp:
                    target_fav = min_tp
                if fav_high >= target_fav:
                    if s > 0:
                        fill = ent * (1.0 + target_fav / 10000.0)
                        out_gross = fill / ent - 1.0
                    else:
                        fill = ent / (1.0 + target_fav / 10000.0)
                        out_gross = ent / fill - 1.0
                    out_min = j
                    out_reason = 2
                    exited = True
                    break
            if trail_pressure > 0.0 and max_fav_bps > trail_bps:
                trail_fav = max_fav_bps - trail_bps
                if s > 0:
                    trail_px = ent * (1.0 + trail_fav / 10000.0)
                    if l <= trail_px:
                        fill = o if o <= trail_px else trail_px
                        out_gross = fill / ent - 1.0
                        out_min = j
                        out_reason = 5
                        exited = True
                        break
                else:
                    trail_px = ent / (1.0 + trail_fav / 10000.0)
                    if h >= trail_px:
                        fill = o if o >= trail_px else trail_px
                        out_gross = ent / fill - 1.0
                        out_min = j
                        out_reason = 5
                        exited = True
                        break
            prev_close = cl
        if not exited:
            cl = closes[out_min]
            if not math.isfinite(cl) or cl <= 0.0:
                cl = prev_close
            out_gross = s * (cl / ent - 1.0)
            out_reason = 3
        return out_gross - cost_bps / 10000.0, out_gross, out_min, out_reason

else:
    _simulate_exit_grid_numba = None
    _simulate_exit_one_numba = None
    _simulate_exit_one_independent_numba = None


def _config_grid(max_configs: int | None = None) -> list[ExitTighteningConfig]:
    rows: list[ExitTighteningConfig] = []
    for q in (0.65, 0.75, 0.85):
        for pressure_mode in ("count", "notional"):
            for pressure_mid in (0.45, 0.50, 0.55):
                for pressure_power in (1.0, 1.5):
                    for churn in (3.0, 5.0, 8.0, 12.0):
                        for hysteresis in (15.0, 25.0, 40.0):
                            for min_sl in (15.0, 25.0, 35.0):
                                for min_trail in (10.0, 20.0, 35.0):
                                    for min_tp in (20.0, 35.0, 50.0):
                                        for use_mode in ("linear", "hierarchical", "convex"):
                                            rows.append(
                                                ExitTighteningConfig(
                                                    config_id=f"cfg_{len(rows):05d}",
                                                    candidate_edge_quantile=q,
                                                    pressure_mode=pressure_mode,
                                                    pressure_mid=pressure_mid,
                                                    pressure_power=pressure_power,
                                                    churn_penalty_bps=churn,
                                                    exit_hysteresis_bps=hysteresis,
                                                    base_stop_loss_bps=80.0,
                                                    min_stop_loss_bps=min_sl,
                                                    base_trailing_gap_bps=70.0,
                                                    min_trailing_gap_bps=min_trail,
                                                    base_tp_remaining_bps=120.0,
                                                    min_tp_remaining_bps=min_tp,
                                                    pressure_use_mode=use_mode,
                                                )
                                            )
                                            if max_configs is not None and len(rows) >= max_configs:
                                                return rows
    return rows


def _param_matrix(configs: list[ExitTighteningConfig]) -> np.ndarray:
    pressure_mode = {"count": 0, "notional": 1}
    use_mode = {"linear": 0, "hierarchical": 1, "convex": 2}
    out = np.zeros((len(configs), 13), dtype=np.float32)
    for i, cfg in enumerate(configs):
        out[i] = np.asarray(
            [
                cfg.candidate_edge_quantile,
                pressure_mode[cfg.pressure_mode],
                cfg.pressure_mid,
                cfg.pressure_power,
                cfg.churn_penalty_bps,
                cfg.exit_hysteresis_bps,
                cfg.base_stop_loss_bps,
                cfg.min_stop_loss_bps,
                cfg.base_trailing_gap_bps,
                cfg.min_trailing_gap_bps,
                cfg.base_tp_remaining_bps,
                cfg.min_tp_remaining_bps,
                use_mode[cfg.pressure_use_mode],
            ],
            dtype=np.float32,
        )
    return out


def _independent_param_row(cfg: IndependentExitTighteningConfig) -> np.ndarray:
    pressure_mode = {"count": 0, "notional": 1}
    formula = {"linear": 0, "convex": 1, "concave": 2, "delayed": 3}
    force_mode = {"none": 0, "sl": 1, "max": 2}
    return np.asarray(
        [
            pressure_mode[cfg.sl_pressure_mode],
            pressure_mode[cfg.trail_pressure_mode],
            cfg.sl_pressure_mid,
            cfg.trail_pressure_mid,
            cfg.sl_pressure_power,
            cfg.trail_pressure_power,
            cfg.sl_churn_penalty_bps,
            cfg.trail_churn_penalty_bps,
            cfg.sl_hysteresis_bps,
            cfg.trail_hysteresis_bps,
            cfg.sl_ev_weight,
            cfg.trail_ev_weight,
            cfg.trail_profit_weight,
            cfg.base_stop_loss_bps,
            cfg.min_stop_loss_bps,
            cfg.base_trailing_gap_bps,
            cfg.min_trailing_gap_bps,
            cfg.base_tp_remaining_bps,
            cfg.min_tp_remaining_bps,
            formula[cfg.sl_formula],
            formula[cfg.trail_formula],
            force_mode[cfg.force_exit_mode],
            cfg.force_hysteresis_mult,
        ],
        dtype=np.float32,
    )


def _optuna_independent_config(trial: Any) -> IndependentExitTighteningConfig:
    return IndependentExitTighteningConfig(
        config_id=f"independent_trial_{trial.number:04d}",
        sl_candidate_edge_quantile=float(trial.suggest_categorical("sl_candidate_edge_quantile", [0.65, 0.75, 0.85])),
        trail_candidate_edge_quantile=float(trial.suggest_categorical("trail_candidate_edge_quantile", [0.65, 0.75, 0.85])),
        sl_pressure_mode=str(trial.suggest_categorical("sl_pressure_mode", ["count", "notional"])),
        trail_pressure_mode=str(trial.suggest_categorical("trail_pressure_mode", ["count", "notional"])),
        sl_pressure_mid=float(trial.suggest_float("sl_pressure_mid", 0.35, 0.75, step=0.05)),
        trail_pressure_mid=float(trial.suggest_float("trail_pressure_mid", 0.35, 0.75, step=0.05)),
        sl_pressure_power=float(trial.suggest_categorical("sl_pressure_power", [0.75, 1.0, 1.5, 2.0])),
        trail_pressure_power=float(trial.suggest_categorical("trail_pressure_power", [0.5, 0.75, 1.0, 1.5, 2.0])),
        sl_churn_penalty_bps=float(trial.suggest_float("sl_churn_penalty_bps", 0.0, 15.0, step=1.0)),
        trail_churn_penalty_bps=float(trial.suggest_float("trail_churn_penalty_bps", 0.0, 15.0, step=1.0)),
        sl_hysteresis_bps=float(trial.suggest_float("sl_hysteresis_bps", 8.0, 60.0, step=2.0)),
        trail_hysteresis_bps=float(trial.suggest_float("trail_hysteresis_bps", 8.0, 60.0, step=2.0)),
        sl_ev_weight=float(trial.suggest_float("sl_ev_weight", 0.0, 2.0, step=0.25)),
        trail_ev_weight=float(trial.suggest_float("trail_ev_weight", 0.0, 2.0, step=0.25)),
        trail_profit_weight=float(trial.suggest_float("trail_profit_weight", 0.0, 1.5, step=0.25)),
        base_stop_loss_bps=80.0,
        min_stop_loss_bps=float(trial.suggest_float("min_stop_loss_bps", 10.0, 55.0, step=5.0)),
        base_trailing_gap_bps=70.0,
        min_trailing_gap_bps=float(trial.suggest_float("min_trailing_gap_bps", 5.0, 50.0, step=5.0)),
        base_tp_remaining_bps=120.0,
        min_tp_remaining_bps=float(trial.suggest_float("min_tp_remaining_bps", 20.0, 80.0, step=10.0)),
        sl_formula=str(trial.suggest_categorical("sl_formula", ["linear", "convex", "delayed"])),
        trail_formula=str(trial.suggest_categorical("trail_formula", ["linear", "convex", "concave", "delayed"])),
        force_exit_mode=str(trial.suggest_categorical("force_exit_mode", ["none", "sl", "max"])),
        force_hysteresis_mult=float(trial.suggest_float("force_hysteresis_mult", 1.0, 4.0, step=0.5)),
    )


def _prepare_sim_arrays(
    accepted: pd.DataFrame,
    edge_table: pd.DataFrame,
    *,
    params: Any,
    spread_floor_bps: float,
    execution_gap_bps: float,
) -> dict[str, np.ndarray]:
    work = accepted.merge(edge_table, on="timestamp", how="left")
    side = np.asarray([_side_code(s, sid) for s, sid in zip(work["side"], work["strategy_id"])], dtype=np.int8)
    ts = _entry_time_series(work)
    exit_ts = pd.to_datetime(work["exit_timestamp"], utc=True, errors="coerce")
    hold_minutes = np.ceil((exit_ts - ts).dt.total_seconds().fillna(3600.0).to_numpy(dtype=np.float64) / 60.0).astype(np.int16)
    hold_minutes = np.clip(hold_minutes, 1, MAX_PATH_MINUTES).astype(np.int16)
    entry = pd.to_numeric(work["entry_price"], errors="coerce").to_numpy(dtype=np.float32)
    position_size = pd.to_numeric(work["position_size"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    fees = pd.to_numeric(work.get("fees_bps"), errors="coerce").fillna(20.0).clip(lower=0.0)
    spread = pd.to_numeric(work.get("expected_spread_bps"), errors="coerce").fillna(0.0)
    roundtrip_cost = np.maximum(
        pd.to_numeric(work.get("expected_friction_bps"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
        fees.to_numpy(dtype=np.float64)
        + np.maximum(spread.to_numpy(dtype=np.float64), float(spread_floor_bps))
        + float(execution_gap_bps),
    )
    count_ratio = (
        pd.to_numeric(work.get("open_positions_after"), errors="coerce")
        .fillna(pd.to_numeric(work.get("open_positions_before"), errors="coerce").fillna(0.0) + 1.0)
        .to_numpy(dtype=np.float32)
        / max(float(params.max_concurrent_positions), 1.0)
    )
    wallet = pd.to_numeric(work.get("wallet_before"), errors="coerce").replace(0.0, np.nan).fillna(10_000.0)
    open_notional = pd.to_numeric(work.get("open_notional_after"), errors="coerce").fillna(0.0)
    cap = max(float(params.max_total_wallet_allocation_pct), 1e-6)
    notional_ratio = (open_notional.to_numpy(dtype=np.float64) / np.maximum(wallet.to_numpy(dtype=np.float64) * cap, 1e-9)).astype(np.float32)
    cand_edges = []
    for q in (65, 75, 85):
        cand_edges.append(
            pd.to_numeric(work.get(f"candidate_edge_p{q}_bps"), errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
    return {
        "side": side,
        "entry_timestamp_ns": ts.astype("int64").to_numpy(dtype=np.int64),
        "entry_price": entry,
        "hold_minutes": hold_minutes,
        "cost_bps": np.maximum(roundtrip_cost, 0.0).astype(np.float32),
        "edge_net_bps": pd.to_numeric(work["_edge_net_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        "candidate_edge_p65_bps": cand_edges[0],
        "candidate_edge_p75_bps": cand_edges[1],
        "candidate_edge_p85_bps": cand_edges[2],
        "candidate_friction_bps": pd.to_numeric(work.get("candidate_fees_and_slippage_p75_bps"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        "count_ratio": np.clip(count_ratio, 0.0, 5.0).astype(np.float32),
        "notional_ratio": np.clip(notional_ratio, 0.0, 5.0).astype(np.float32),
        "position_size": position_size,
    }


def _prepare_candidate_exit_arrays(
    candidates: pd.DataFrame,
    edge_table: pd.DataFrame,
    *,
    params: Any,
    spread_floor_bps: float,
    execution_gap_bps: float,
) -> dict[str, np.ndarray]:
    work = candidates.merge(edge_table, on="timestamp", how="left")
    side = np.asarray([_side_code(s, sid) for s, sid in zip(work["side"], work["strategy_id"])], dtype=np.int8)
    ts = _entry_time_series(work)
    exit_ts = pd.to_datetime(work["exit_timestamp"], utc=True, errors="coerce")
    hold_minutes = np.ceil((exit_ts - ts).dt.total_seconds().fillna(3600.0).to_numpy(dtype=np.float64) / 60.0).astype(np.int16)
    hold_minutes = np.clip(hold_minutes, 1, MAX_PATH_MINUTES).astype(np.int16)
    fees = pd.to_numeric(work.get("fees_bps"), errors="coerce").fillna(20.0).clip(lower=0.0)
    spread = pd.to_numeric(work.get("expected_spread_bps"), errors="coerce").fillna(0.0)
    roundtrip_cost = np.maximum(
        pd.to_numeric(work.get("expected_friction_bps"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64),
        fees.to_numpy(dtype=np.float64)
        + np.maximum(spread.to_numpy(dtype=np.float64), float(spread_floor_bps))
        + float(execution_gap_bps),
    )
    cand_edges = []
    for q in (65, 75, 85):
        cand_edges.append(
            pd.to_numeric(work.get(f"candidate_edge_p{q}_bps"), errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
    return {
        "side": side,
        "entry_timestamp_ns": ts.astype("int64").to_numpy(dtype=np.int64),
        "entry_price": pd.to_numeric(work["entry_price"], errors="coerce").to_numpy(dtype=np.float32),
        "hold_minutes": hold_minutes,
        "cost_bps": np.maximum(roundtrip_cost, 0.0).astype(np.float32),
        "edge_net_bps": pd.to_numeric(work["_edge_net_bps"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        "candidate_edge_p65_bps": cand_edges[0],
        "candidate_edge_p75_bps": cand_edges[1],
        "candidate_edge_p85_bps": cand_edges[2],
        "candidate_friction_bps": pd.to_numeric(work.get("candidate_fees_and_slippage_p75_bps"), errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
    }


def _exit_reason_text(code: int) -> str:
    return {
        1: "tightened_stop_loss",
        2: "tightened_take_profit",
        3: "timeout",
        4: "force_exit",
        5: "tightened_trailing_stop",
        6: "invalid_path",
    }.get(int(code), "unknown")


def _exit_reason_code(text: Any) -> int:
    value = str(text).strip().lower()
    if value in {"tightened_stop_loss", "stop_loss", "sl", "loss"}:
        return 1
    if value in {"tightened_take_profit", "take_profit", "tp"}:
        return 2
    if value in {"force_exit", "forced_exit"}:
        return 4
    if value in {"tightened_trailing_stop", "trailing_stop", "trailing"}:
        return 5
    if value in {"invalid_path", "invalid"}:
        return 6
    return 3


def _make_exit_adjust_callback(
    cfg: ExitTighteningConfig,
    arrays: dict[str, np.ndarray],
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    params: Any,
):
    if _simulate_exit_one_numba is None:
        raise RuntimeError("Numba is required for full portfolio replay exit adjustment.")
    cfg_row = _param_matrix([cfg])[0]
    edge_key = f"candidate_edge_p{int(cfg.candidate_edge_quantile * 100)}_bps"

    def _callback(
        idx: int,
        timestamp: pd.Timestamp,
        state: Any,
        _cache: Any,
        position_size: float,
        _capital_limit: float,
        _remaining_capital: float,
        _group_idx: np.ndarray,
    ) -> dict[str, Any] | None:
        if idx < 0 or idx >= len(arrays["entry_price"]):
            return None
        count_ratio = (float(len(state.open_positions)) + 1.0) / max(float(params.max_concurrent_positions), 1.0)
        cap = max(float(params.max_total_wallet_allocation_pct), EPS)
        next_notional = float(state.open_notional) + max(float(position_size), 0.0)
        notional_ratio = next_notional / max(float(state.wallet) * cap, EPS)
        net, gross, minute, reason = _simulate_exit_one_numba(
            paths[0][idx],
            paths[1][idx],
            paths[2][idx],
            paths[3][idx],
            int(arrays["side"][idx]),
            float(arrays["entry_price"][idx]),
            int(arrays["hold_minutes"][idx]),
            float(arrays["cost_bps"][idx]),
            float(arrays["edge_net_bps"][idx]),
            float(arrays[edge_key][idx]),
            float(arrays["candidate_friction_bps"][idx]),
            float(np.clip(count_ratio, 0.0, 5.0)),
            float(np.clip(notional_ratio, 0.0, 5.0)),
            cfg_row,
        )
        if int(reason) in {3, 6}:
            return None
        if int(minute) >= int(arrays["hold_minutes"][idx]):
            return None
        minute = max(1, int(minute))
        entry_ts = pd.Timestamp(int(arrays["entry_timestamp_ns"][idx]), tz="UTC")
        entry = float(arrays["entry_price"][idx])
        side = int(arrays["side"][idx])
        if side > 0:
            exit_price = entry * (1.0 + float(gross))
        else:
            exit_price = entry / max(1.0 + float(gross), EPS)
        return {
            "exit_timestamp": entry_ts + pd.Timedelta(minutes=minute),
            "net_return": float(net),
            "gross_return": float(gross),
            "exit_reason": _exit_reason_text(int(reason)),
            "exit_price": float(exit_price),
        }

    return _callback


def _make_independent_exit_adjust_callback(
    cfg: IndependentExitTighteningConfig,
    arrays: dict[str, np.ndarray],
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    params: Any,
):
    if _simulate_exit_one_independent_numba is None:
        raise RuntimeError("Numba is required for independent exit adjustment.")
    cfg_row = _independent_param_row(cfg)
    sl_edge_key = f"candidate_edge_p{int(cfg.sl_candidate_edge_quantile * 100)}_bps"
    trail_edge_key = f"candidate_edge_p{int(cfg.trail_candidate_edge_quantile * 100)}_bps"

    def _callback(
        idx: int,
        timestamp: pd.Timestamp,
        state: Any,
        _cache: Any,
        position_size: float,
        _capital_limit: float,
        _remaining_capital: float,
        _group_idx: np.ndarray,
    ) -> dict[str, Any] | None:
        if idx < 0 or idx >= len(arrays["entry_price"]):
            return None
        count_ratio = (float(len(state.open_positions)) + 1.0) / max(float(params.max_concurrent_positions), 1.0)
        cap = max(float(params.max_total_wallet_allocation_pct), EPS)
        next_notional = float(state.open_notional) + max(float(position_size), 0.0)
        notional_ratio = next_notional / max(float(state.wallet) * cap, EPS)
        net, gross, minute, reason = _simulate_exit_one_independent_numba(
            paths[0][idx],
            paths[1][idx],
            paths[2][idx],
            paths[3][idx],
            int(arrays["side"][idx]),
            float(arrays["entry_price"][idx]),
            int(arrays["hold_minutes"][idx]),
            float(arrays["cost_bps"][idx]),
            float(arrays["edge_net_bps"][idx]),
            float(arrays[sl_edge_key][idx]),
            float(arrays[trail_edge_key][idx]),
            float(arrays["candidate_friction_bps"][idx]),
            float(np.clip(count_ratio, 0.0, 5.0)),
            float(np.clip(notional_ratio, 0.0, 5.0)),
            cfg_row,
        )
        if int(reason) in {3, 6}:
            return None
        if int(minute) >= int(arrays["hold_minutes"][idx]):
            return None
        minute = max(1, int(minute))
        entry_ts = pd.Timestamp(int(arrays["entry_timestamp_ns"][idx]), tz="UTC")
        entry = float(arrays["entry_price"][idx])
        side = int(arrays["side"][idx])
        if side > 0:
            exit_price = entry * (1.0 + float(gross))
        else:
            exit_price = entry / max(1.0 + float(gross), EPS)
        return {
            "exit_timestamp": entry_ts + pd.Timedelta(minutes=minute),
            "net_return": float(net),
            "gross_return": float(gross),
            "exit_reason": _exit_reason_text(int(reason)),
            "exit_price": float(exit_price),
        }

    return _callback


def _reason_codes_from_accepted(accepted: pd.DataFrame) -> np.ndarray:
    if "position_exit_reason" in accepted.columns:
        source = accepted["position_exit_reason"]
    else:
        source = accepted.get("simple_policy_exit_reason", pd.Series(["timeout"] * len(accepted)))
    return np.asarray([_exit_reason_code(v) for v in source], dtype=np.int8)


def _simulate_configs(
    configs: list[ExitTighteningConfig],
    arrays: dict[str, np.ndarray],
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if _simulate_exit_grid_numba is None:
        raise RuntimeError("Numba is required for this ablation script.")
    edge_by_config = []
    for cfg in configs:
        if cfg.candidate_edge_quantile == 0.65:
            edge_by_config.append(arrays["candidate_edge_p65_bps"])
        elif cfg.candidate_edge_quantile == 0.85:
            edge_by_config.append(arrays["candidate_edge_p85_bps"])
        else:
            edge_by_config.append(arrays["candidate_edge_p75_bps"])
    # The Numba kernel expects one candidate-edge vector per config. Run in
    # quantile batches to avoid a large repeated matrix.
    net_all: list[np.ndarray] = []
    gross_all: list[np.ndarray] = []
    minute_all: list[np.ndarray] = []
    reason_all: list[np.ndarray] = []
    for q in (0.65, 0.75, 0.85):
        idx = [i for i, cfg in enumerate(configs) if cfg.candidate_edge_quantile == q]
        if not idx:
            continue
        sub_cfg = [configs[i] for i in idx]
        edge_key = f"candidate_edge_p{int(q * 100)}_bps"
        net, gross, minute, reason = _simulate_exit_grid_numba(
            paths[0],
            paths[1],
            paths[2],
            paths[3],
            arrays["side"],
            arrays["entry_price"],
            arrays["hold_minutes"],
            arrays["cost_bps"],
            arrays["edge_net_bps"],
            arrays[edge_key],
            arrays["candidate_friction_bps"],
            arrays["count_ratio"],
            arrays["notional_ratio"],
            _param_matrix(sub_cfg),
        )
        net_all.append((np.asarray(idx, dtype=np.int32), net))
        gross_all.append((np.asarray(idx, dtype=np.int32), gross))
        minute_all.append((np.asarray(idx, dtype=np.int32), minute))
        reason_all.append((np.asarray(idx, dtype=np.int32), reason))
    n_cfg = len(configs)
    n = len(arrays["entry_price"])
    net_out = np.empty((n_cfg, n), dtype=np.float32)
    gross_out = np.empty((n_cfg, n), dtype=np.float32)
    minute_out = np.empty((n_cfg, n), dtype=np.int16)
    reason_out = np.empty((n_cfg, n), dtype=np.int8)
    for idx, val in net_all:
        net_out[idx] = val
    for idx, val in gross_all:
        gross_out[idx] = val
    for idx, val in minute_all:
        minute_out[idx] = val
    for idx, val in reason_all:
        reason_out[idx] = val
    return net_out, gross_out, minute_out, reason_out


def _week_start(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True, errors="coerce").dt.to_period("W-SUN").dt.start_time.dt.tz_localize("UTC")


def _metrics_for_returns(frame: pd.DataFrame, net_return: np.ndarray, gross_return: np.ndarray, prefix: str = "") -> dict[str, Any]:
    size = pd.to_numeric(frame["position_size"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    net_pnl = size * np.asarray(net_return, dtype=np.float64)
    gross_pnl = size * np.asarray(gross_return, dtype=np.float64)
    weeks = _week_start(frame["timestamp"])
    weekly = pd.DataFrame({"week_start": weeks, "net_pnl": net_pnl})
    weekly_sum = weekly.groupby("week_start", observed=True)["net_pnl"].sum()
    q15 = float(weekly_sum.quantile(0.15)) if len(weekly_sum) else 0.0
    avg = float(weekly_sum.mean()) if len(weekly_sum) else 0.0
    return {
        f"{prefix}trade_count": int(len(frame)),
        f"{prefix}net_pnl": float(np.sum(net_pnl)),
        f"{prefix}gross_pnl": float(np.sum(gross_pnl)),
        f"{prefix}cost_pnl": float(np.sum(gross_pnl - net_pnl)),
        f"{prefix}hit_rate": float(np.mean(net_pnl > 0.0)) if len(net_pnl) else 0.0,
        f"{prefix}avg_week_pnl": avg,
        f"{prefix}q15_week_pnl": q15,
        f"{prefix}objective": float(avg + 0.5 * q15),
        f"{prefix}worst_week_pnl": float(weekly_sum.min()) if len(weekly_sum) else 0.0,
        f"{prefix}positive_week_share": float((weekly_sum > 0.0).mean()) if len(weekly_sum) else 0.0,
    }


def _summary_tables(
    accepted: pd.DataFrame,
    net_return: np.ndarray,
    gross_return: np.ndarray,
    reason: np.ndarray,
    *,
    config_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = accepted[["timestamp", "strategy_id", "head", "position_size"]].copy()
    size = pd.to_numeric(work["position_size"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    work["config_id"] = str(config_id)
    work["week_start"] = _week_start(work["timestamp"])
    work["month"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce").dt.to_period("M").astype(str)
    work["net_return"] = np.asarray(net_return, dtype=np.float64)
    work["gross_return"] = np.asarray(gross_return, dtype=np.float64)
    work["net_pnl"] = size * work["net_return"].to_numpy(dtype=np.float64)
    work["gross_pnl"] = size * work["gross_return"].to_numpy(dtype=np.float64)
    work["cost_pnl"] = work["gross_pnl"] - work["net_pnl"]
    work["win"] = work["net_pnl"] > 0.0
    work["exit_code"] = np.asarray(reason, dtype=np.int16)

    def agg(keys: list[str]) -> pd.DataFrame:
        rows = []
        for vals, g in work.groupby(keys, observed=True, dropna=False):
            if not isinstance(vals, tuple):
                vals = (vals,)
            rec = dict(zip(keys, vals))
            rec.update(
                {
                    "trade_count": int(len(g)),
                    "net_pnl": float(g["net_pnl"].sum()),
                    "gross_pnl": float(g["gross_pnl"].sum()),
                    "cost_pnl": float(g["cost_pnl"].sum()),
                    "hit_rate_pct": float(g["win"].mean() * 100.0) if len(g) else np.nan,
                    "mean_net_return_pct": float(g["net_return"].mean() * 100.0) if len(g) else np.nan,
                    "q05_net_return_pct": float(g["net_return"].quantile(0.05) * 100.0) if len(g) else np.nan,
                    "force_exit_rate_pct": float((g["exit_code"] == 4).mean() * 100.0) if len(g) else 0.0,
                    "tight_sl_rate_pct": float((g["exit_code"] == 1).mean() * 100.0) if len(g) else 0.0,
                    "trailing_rate_pct": float((g["exit_code"] == 5).mean() * 100.0) if len(g) else 0.0,
                    "timeout_rate_pct": float((g["exit_code"] == 3).mean() * 100.0) if len(g) else 0.0,
                }
            )
            rows.append(rec)
        return pd.DataFrame(rows)

    return agg(["config_id", "week_start"]), agg(["config_id", "head"]), agg(["config_id", "week_start", "head"])


def _config_lookup(configs: list[ExitTighteningConfig]) -> dict[str, ExitTighteningConfig]:
    return {cfg.config_id: cfg for cfg in configs}


def _shortlist_full_replay_configs(
    summary: pd.DataFrame,
    configs: list[ExitTighteningConfig],
    *,
    limit: int,
) -> list[ExitTighteningConfig]:
    lookup = _config_lookup(configs)
    ids: list[str] = []

    def add(values: pd.Series) -> None:
        for value in values.astype(str).tolist():
            if value in lookup and value not in ids:
                ids.append(value)

    ranked = summary.sort_values("objective", ascending=False)
    add(ranked["config_id"].head(max(1, limit // 2)))
    if "delta_vs_sim_base_net_pnl" in summary.columns:
        add(summary.sort_values("delta_vs_sim_base_net_pnl", ascending=False)["config_id"].head(max(1, limit // 4)))
    group_cols = [
        col
        for col in ["candidate_edge_quantile", "pressure_mode", "pressure_use_mode"]
        if col in summary.columns
    ]
    if group_cols:
        grouped = (
            summary.sort_values("objective", ascending=False)
            .groupby(group_cols, observed=True, dropna=False)
            .head(1)
        )
        add(grouped["config_id"])
    return [lookup[value] for value in ids[: max(1, int(limit))]]


def _returns_from_accepted(accepted: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    net = pd.to_numeric(accepted["net_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    gross = pd.to_numeric(accepted["gross_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    reason = _reason_codes_from_accepted(accepted)
    return net, gross, reason


def _run_full_portfolio_replays(
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    configs: list[ExitTighteningConfig],
    arrays: dict[str, np.ndarray],
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    market_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    weekly_parts: list[pd.DataFrame] = []
    head_parts: list[pd.DataFrame] = []
    weekly_head_parts: list[pd.DataFrame] = []
    accepted_by_config: dict[str, pd.DataFrame] = {}
    for pos, cfg in enumerate(configs, start=1):
        callback = _make_exit_adjust_callback(cfg, arrays, paths, params=params)
        decisions, equity, replay_metrics = replay_candidates(
            candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            accepted_position_callback=callback,
            market_mode=market_mode,
        )
        accepted = _accepted_trades(candidates, decisions)
        net, gross, reason = _returns_from_accepted(accepted)
        metric = _metrics_for_returns(accepted, net, gross)
        row = asdict(cfg)
        row.update(metric)
        row["replay_objective"] = float(replay_metrics.get("objective", np.nan))
        row["replay_final_wallet"] = float(replay_metrics.get("final_wallet", np.nan))
        row["replay_max_drawdown"] = float(replay_metrics.get("max_drawdown", np.nan))
        row["full_replay_rank"] = int(pos)
        rows.append(row)
        weekly, by_head, weekly_by_head = _summary_tables(
            accepted,
            net,
            gross,
            reason,
            config_id=cfg.config_id,
        )
        weekly_parts.append(weekly)
        head_parts.append(by_head)
        weekly_head_parts.append(weekly_by_head)
        accepted_by_config[cfg.config_id] = accepted
        print(
            "full replay "
            f"{pos}/{len(configs)} {cfg.config_id} "
            f"objective={metric.get('objective', np.nan):.6f} "
            f"trades={len(accepted)}",
            flush=True,
        )
    summary = pd.DataFrame(rows).sort_values("objective", ascending=False).reset_index(drop=True)
    weekly_all = pd.concat(weekly_parts, ignore_index=True) if weekly_parts else pd.DataFrame()
    head_all = pd.concat(head_parts, ignore_index=True) if head_parts else pd.DataFrame()
    weekly_head_all = pd.concat(weekly_head_parts, ignore_index=True) if weekly_head_parts else pd.DataFrame()
    return summary, weekly_all, head_all, weekly_head_all, accepted_by_config


def _run_independent_optuna_study(
    candidates: pd.DataFrame,
    params: Any,
    ev_curve: dict[str, Any],
    arrays: dict[str, np.ndarray],
    paths: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    market_mode: str,
    n_trials: int,
    seed: int,
    timeout_seconds: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if optuna is None:
        raise RuntimeError("Optuna is not installed; cannot run independent exit study.")
    trial_rows: list[dict[str, Any]] = []
    best: dict[str, Any] = {
        "objective": -np.inf,
        "config": None,
        "accepted": pd.DataFrame(),
        "net": np.asarray([], dtype=np.float32),
        "gross": np.asarray([], dtype=np.float32),
        "reason": np.asarray([], dtype=np.int8),
    }
    sampler = optuna.samplers.TPESampler(seed=int(seed), multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(5, min(10, int(n_trials) // 4)))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def objective(trial: Any) -> float:
        cfg = _optuna_independent_config(trial)
        callback = _make_independent_exit_adjust_callback(cfg, arrays, paths, params=params)
        decisions, _equity, replay_metrics = replay_candidates(
            candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            accepted_position_callback=callback,
            market_mode=market_mode,
        )
        accepted = _accepted_trades(candidates, decisions)
        net, gross, reason = _returns_from_accepted(accepted)
        metric = _metrics_for_returns(accepted, net, gross)
        value = float(metric.get("objective", -np.inf))
        row = asdict(cfg)
        row.update(metric)
        row["trial_number"] = int(trial.number)
        row["replay_objective"] = float(replay_metrics.get("objective", np.nan))
        row["replay_final_wallet"] = float(replay_metrics.get("final_wallet", np.nan))
        row["replay_max_drawdown"] = float(replay_metrics.get("max_drawdown", np.nan))
        row["force_exit_rate_pct"] = float(np.mean(reason == 4) * 100.0) if len(reason) else 0.0
        row["tight_sl_rate_pct"] = float(np.mean(reason == 1) * 100.0) if len(reason) else 0.0
        row["trailing_rate_pct"] = float(np.mean(reason == 5) * 100.0) if len(reason) else 0.0
        trial_rows.append(row)
        if value > float(best["objective"]):
            best.update(
                {
                    "objective": value,
                    "config": cfg,
                    "accepted": accepted.copy(),
                    "net": net.copy(),
                    "gross": gross.copy(),
                    "reason": reason.copy(),
                }
            )
        print(
            "independent optuna "
            f"trial={trial.number + 1}/{n_trials} "
            f"objective={value:.6f} trades={len(accepted)}",
            flush=True,
        )
        return value

    study.optimize(
        objective,
        n_trials=int(n_trials),
        timeout=int(timeout_seconds) if timeout_seconds else None,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    summary = pd.DataFrame(trial_rows).sort_values("objective", ascending=False).reset_index(drop=True)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    if best["config"] is None:
        return summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), trials
    weekly, by_head, weekly_by_head = _summary_tables(
        best["accepted"],
        best["net"],
        best["gross"],
        best["reason"],
        config_id=str(summary.iloc[0]["config_id"]),
    )
    return summary, weekly, by_head, weekly_by_head, trials


def _markdown_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_POLICY_CANDIDATES)
    parser.add_argument("--edge-candidates", type=Path, default=DEFAULT_EDGE_CANDIDATES)
    parser.add_argument("--policy-config", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--artifact-decisions", type=Path, default=DEFAULT_ARTIFACT_DECISIONS)
    parser.add_argument("--ohlcv-root", type=Path, default=DEFAULT_OHLCV_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--periods", default="mar,apr,jun")
    parser.add_argument("--ev-fit-end", default="2026-03-01T00:00:00+00:00")
    parser.add_argument("--spread-floor-bps", type=float, default=35.0)
    parser.add_argument("--execution-gap-bps", type=float, default=15.0)
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--config-chunk-size", type=int, default=512)
    parser.add_argument("--full-portfolio-replay", action="store_true")
    parser.add_argument("--full-replay-top-configs", type=int, default=48)
    parser.add_argument("--full-replay-config-summary", type=Path, default=None)
    parser.add_argument("--independent-optuna-trials", type=int, default=0)
    parser.add_argument("--independent-optuna-seed", type=int, default=42)
    parser.add_argument("--independent-optuna-timeout-seconds", type=int, default=0)
    parser.add_argument("--calibration-only", action="store_true")
    parser.add_argument("--no-artifact-position-sizes", action="store_true")
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    if not args.calibration_only and _simulate_exit_grid_numba is None:
        raise RuntimeError("Numba is not available; this script intentionally requires it.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    params, policy_payload = _load_policy(args.policy_config)
    periods = [p.strip().lower() for p in str(args.periods).split(",") if p.strip()]
    global_floor = float(params.global_threshold_floor)
    candidates = _edge_and_cost_columns(
        _load_candidates(args.candidates, periods, global_floor=global_floor),
        spread_floor_bps=float(args.spread_floor_bps),
        execution_gap_bps=float(args.execution_gap_bps),
    )
    candidates = normalise_candidate_table(candidates).reset_index(drop=True)
    candidates["head"] = candidates["strategy_id"].map(_strategy_head)
    candidates["candidate_row_id"] = np.arange(len(candidates), dtype=np.int64)
    artifact_size_audit: dict[str, Any] = {"artifact_position_size_enabled": False}
    if not args.no_artifact_position_sizes:
        candidates, artifact_size_audit = _attach_artifact_position_sizes(
            candidates,
            args.artifact_decisions,
            periods,
        )
        candidates = normalise_candidate_table(candidates).reset_index(drop=True)
        candidates["head"] = candidates["strategy_id"].map(_strategy_head)
        candidates["candidate_row_id"] = np.arange(len(candidates), dtype=np.int64)
        artifact_size_audit["artifact_position_size_enabled"] = True
    edge_candidates = _edge_and_cost_columns(
        _load_candidates(args.edge_candidates, periods, global_floor=global_floor),
        spread_floor_bps=float(args.spread_floor_bps),
        execution_gap_bps=float(args.execution_gap_bps),
    )
    edge_candidates = normalise_candidate_table(edge_candidates).reset_index(drop=True)
    edge_candidates["head"] = edge_candidates["strategy_id"].map(_strategy_head)

    ev_candidates = _edge_and_cost_columns(
        _load_ev_candidates(args.candidates, cutoff=_utc(args.ev_fit_end), global_floor=global_floor),
        spread_floor_bps=float(args.spread_floor_bps),
        execution_gap_bps=float(args.execution_gap_bps),
    )
    ev_curve = fit_hierarchical_ev_curves(ev_candidates if not ev_candidates.empty else candidates)
    decisions, equity, replay_metrics = replay_candidates(
        candidates,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=str(args.market_mode),
    )
    accepted = _accepted_trades(candidates, decisions)
    baseline_accepted_full = accepted.copy()
    if accepted.empty:
        raise RuntimeError("No accepted trades under the fixed portfolio policy.")
    parity_audit = _replay_parity_audit(
        local_decisions=decisions,
        artifact_decisions_path=args.artifact_decisions,
        periods=periods,
        out_dir=args.out_dir,
    )
    if args.calibration_only:
        accepted_net, accepted_gross, accepted_reason = _returns_from_accepted(accepted)
        weekly, by_head, weekly_by_head = _summary_tables(
            accepted,
            accepted_net,
            accepted_gross,
            accepted_reason,
            config_id="local_policy_replay",
        )
        weekly.to_csv(args.out_dir / "calibration_weekly_metrics.csv", index=False)
        by_head.to_csv(args.out_dir / "calibration_per_strategy_metrics.csv", index=False)
        weekly_by_head.to_csv(args.out_dir / "calibration_weekly_per_strategy_metrics.csv", index=False)
        accepted.to_parquet(args.out_dir / "calibration_accepted_trades.parquet", index=False)
        calibration_manifest = {
            "generated_by": "backtest_exit_tightening_redeploy",
            "mode": "calibration_only",
            "candidate_path": str(args.candidates),
            "edge_candidate_path": str(args.edge_candidates),
            "policy_config": str(args.policy_config),
            "artifact_decisions": str(args.artifact_decisions),
            "periods": periods,
            "candidate_rows": int(len(candidates)),
            "edge_candidate_rows": int(len(edge_candidates)),
            "accepted_rows": int(len(accepted)),
            "portfolio_replay_metrics": replay_metrics,
            "parity_audit": parity_audit,
            "artifact_position_size_audit": artifact_size_audit,
        }
        (args.out_dir / "manifest.json").write_text(
            json.dumps(calibration_manifest, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        lines = [
            "# Exit Tightening Replay Calibration",
            "",
            f"- Policy candidates: `{args.candidates}`",
            f"- Edge candidates: `{args.edge_candidates}`",
            f"- Candidate rows: {len(candidates):,}",
            f"- Edge candidate rows: {len(edge_candidates):,}",
            f"- Accepted rows: {len(accepted):,}",
            "",
            "## Parity Audit",
            "",
            _markdown_table(pd.DataFrame([parity_audit]).T.reset_index().rename(columns={"index": "metric", 0: "value"}), max_rows=80),
            "",
            "## Per-Strategy Metrics",
            "",
            _markdown_table(by_head.sort_values("net_pnl", ascending=False), max_rows=80),
        ]
        (args.out_dir / "exit_tightening_replay_calibration.md").write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
        print(
            "Calibration complete "
            f"candidates={len(candidates)} accepted={len(accepted)} "
            f"matched={parity_audit.get('matched_rows')} "
            f"accepted_match_rate={parity_audit.get('accepted_match_rate')}",
            flush=True,
        )
        return

    edge_table = _timestamp_edge_table(edge_candidates, global_floor=global_floor)
    paths = _build_paths(accepted, ohlcv_root=args.ohlcv_root, max_path_minutes=MAX_PATH_MINUTES)
    coverage = paths[4]
    path_ok = coverage >= np.clip(
        np.ceil(
            (
                pd.to_datetime(accepted["exit_timestamp"], utc=True, errors="coerce")
                - _entry_time_series(accepted)
            ).dt.total_seconds().fillna(3600.0)
            / 60.0
        ).to_numpy(dtype=np.float64),
        1,
        MAX_PATH_MINUTES,
    )
    accepted = accepted.loc[path_ok].reset_index(drop=True)
    paths = tuple(p[path_ok] for p in paths[:4])
    arrays = _prepare_sim_arrays(
        accepted,
        edge_table,
        params=params,
        spread_floor_bps=float(args.spread_floor_bps),
        execution_gap_bps=float(args.execution_gap_bps),
    )
    max_configs = int(args.max_configs or 0)
    configs = _config_grid(max_configs=max_configs if max_configs > 0 else None)
    print(
        f"Loaded candidates={len(candidates)} accepted={len(accepted)} "
        f"path_coverage={float(path_ok.mean() * 100.0):.2f}% configs={len(configs)}",
        flush=True,
    )
    position_size = arrays["position_size"].astype(np.float64)
    summary_rows: list[dict[str, Any]] = []
    baseline_metrics = _metrics_for_returns(
        accepted,
        pd.to_numeric(accepted["net_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        pd.to_numeric(accepted["gross_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        prefix="artifact_",
    )
    no_tight_cfg = ExitTighteningConfig(
        config_id="sim_no_tightening",
        candidate_edge_quantile=0.75,
        pressure_mode="count",
        pressure_mid=2.0,
        pressure_power=1.0,
        churn_penalty_bps=1_000_000.0,
        exit_hysteresis_bps=25.0,
        base_stop_loss_bps=80.0,
        min_stop_loss_bps=80.0,
        base_trailing_gap_bps=70.0,
        min_trailing_gap_bps=70.0,
        base_tp_remaining_bps=120.0,
        min_tp_remaining_bps=120.0,
        pressure_use_mode="linear",
    )
    base_net, base_gross, base_minute, base_reason = _simulate_configs(
        [no_tight_cfg],
        arrays,
        paths,
    )
    sim_baseline = _metrics_for_returns(accepted, base_net[0], base_gross[0], prefix="sim_base_")
    best_objective = -np.inf
    best_cfg: ExitTighteningConfig | None = None
    best_net: np.ndarray | None = None
    best_gross: np.ndarray | None = None
    best_reason: np.ndarray | None = None
    chunk_size = max(1, int(args.config_chunk_size))
    if args.full_replay_config_summary is not None and Path(args.full_replay_config_summary).exists():
        summary = pd.read_csv(args.full_replay_config_summary)
        print(f"Loaded row-level config summary from {args.full_replay_config_summary}", flush=True)
    else:
        for start_i in range(0, len(configs), chunk_size):
            chunk = configs[start_i : start_i + chunk_size]
            net, gross, exit_minute, reason = _simulate_configs(chunk, arrays, paths)
            for i, cfg in enumerate(chunk):
                m = _metrics_for_returns(accepted, net[i], gross[i])
                row = asdict(cfg)
                row.update(m)
                row["delta_vs_sim_base_net_pnl"] = float(
                    np.sum(position_size * net[i].astype(np.float64))
                    - np.sum(position_size * base_net[0].astype(np.float64))
                )
                row["delta_vs_artifact_net_pnl"] = float(
                    np.sum(position_size * net[i].astype(np.float64))
                    - float(accepted["baseline_net_pnl"].sum())
                )
                row["mean_exit_minute"] = float(np.mean(exit_minute[i])) if exit_minute.shape[1] else 0.0
                row["force_exit_rate_pct"] = float(np.mean(reason[i] == 4) * 100.0) if reason.shape[1] else 0.0
                row["tight_sl_rate_pct"] = float(np.mean(reason[i] == 1) * 100.0) if reason.shape[1] else 0.0
                row["trailing_rate_pct"] = float(np.mean(reason[i] == 5) * 100.0) if reason.shape[1] else 0.0
                summary_rows.append(row)
                objective = float(m.get("objective", -np.inf))
                if objective > best_objective:
                    best_objective = objective
                    best_cfg = cfg
                    best_net = net[i].copy()
                    best_gross = gross[i].copy()
                    best_reason = reason[i].copy()
            if start_i == 0 or (start_i + len(chunk)) % (chunk_size * 4) == 0 or start_i + len(chunk) >= len(configs):
                print(
                    "config progress "
                    f"{start_i + len(chunk)}/{len(configs)} "
                    f"best_objective={best_objective:.6f}",
                    flush=True,
                )
        summary = pd.DataFrame(summary_rows).sort_values("objective", ascending=False).reset_index(drop=True)
    best_id = str(summary.iloc[0]["config_id"])
    if best_cfg is None or best_net is None or best_gross is None or best_reason is None:
        best_cfg = _config_lookup(configs).get(best_id)
        if best_cfg is None:
            raise RuntimeError(f"Best config {best_id!r} is not present in this config grid.")
        best_net, best_gross, _best_minute, best_reason = _simulate_configs([best_cfg], arrays, paths)
        best_net = best_net[0].copy()
        best_gross = best_gross[0].copy()
        best_reason = best_reason[0].copy()
    if best_cfg.config_id != best_id:
        # Prefer the summary table's sort result if floating-point tie ordering
        # differed from the streaming tracker.
        best_cfg = next(cfg for cfg in configs if cfg.config_id == best_id)
        best_net, best_gross, _best_minute, best_reason = _simulate_configs([best_cfg], arrays, paths)
        best_net = best_net[0].copy()
        best_gross = best_gross[0].copy()
        best_reason = best_reason[0].copy()
    weekly_best, by_head_best, weekly_by_head_best = _summary_tables(
        accepted,
        best_net,
        best_gross,
        best_reason,
        config_id=best_id,
    )
    weekly_artifact, by_head_artifact, weekly_by_head_artifact = _summary_tables(
        accepted,
        pd.to_numeric(accepted["net_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        pd.to_numeric(accepted["gross_return"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        np.full(len(accepted), 0, dtype=np.int8),
        config_id="artifact_policy",
    )
    weekly_sim_base, by_head_sim_base, weekly_by_head_sim_base = _summary_tables(
        accepted,
        base_net[0],
        base_gross[0],
        base_reason[0],
        config_id="sim_no_tightening",
    )
    weekly = pd.concat([weekly_artifact, weekly_sim_base, weekly_best], ignore_index=True)
    by_head = pd.concat([by_head_artifact, by_head_sim_base, by_head_best], ignore_index=True)
    weekly_by_head = pd.concat([weekly_by_head_artifact, weekly_by_head_sim_base, weekly_by_head_best], ignore_index=True)

    full_manifest: dict[str, Any] = {}
    full_summary = pd.DataFrame()
    full_weekly = pd.DataFrame()
    full_by_head = pd.DataFrame()
    full_weekly_by_head = pd.DataFrame()
    independent_manifest: dict[str, Any] = {}
    independent_summary = pd.DataFrame()
    independent_weekly = pd.DataFrame()
    independent_by_head = pd.DataFrame()
    independent_weekly_by_head = pd.DataFrame()
    independent_trials = pd.DataFrame()
    candidate_paths = None
    candidate_coverage = np.asarray([], dtype=np.int16)
    full_arrays: dict[str, np.ndarray] | None = None
    if args.full_portfolio_replay or int(args.independent_optuna_trials or 0) > 0:
        candidate_paths = _build_paths(candidates, ohlcv_root=args.ohlcv_root, max_path_minutes=MAX_PATH_MINUTES)
        candidate_coverage = candidate_paths[4]
        full_arrays = _prepare_candidate_exit_arrays(
            candidates,
            edge_table,
            params=params,
            spread_floor_bps=float(args.spread_floor_bps),
            execution_gap_bps=float(args.execution_gap_bps),
        )
    if args.full_portfolio_replay:
        assert candidate_paths is not None and full_arrays is not None
        full_configs = _shortlist_full_replay_configs(
            summary,
            configs,
            limit=max(1, int(args.full_replay_top_configs)),
        )
        print(
            "Full portfolio replay shortlist "
            f"{len(full_configs)} configs; candidate path rows with coverage>1="
            f"{int((candidate_coverage > 1).sum())}/{len(candidate_coverage)}",
            flush=True,
        )
        artifact_full_net, artifact_full_gross, artifact_full_reason = _returns_from_accepted(baseline_accepted_full)
        artifact_full_metrics = _metrics_for_returns(
            baseline_accepted_full,
            artifact_full_net,
            artifact_full_gross,
            prefix="full_artifact_",
        )
        artifact_full_weekly, artifact_full_by_head, artifact_full_weekly_by_head = _summary_tables(
            baseline_accepted_full,
            artifact_full_net,
            artifact_full_gross,
            artifact_full_reason,
            config_id="full_artifact_policy",
        )
        no_tight_callback = _make_exit_adjust_callback(no_tight_cfg, full_arrays, tuple(candidate_paths[:4]), params=params)
        base_full_decisions, _base_full_equity, _base_full_replay_metrics = replay_candidates(
            candidates,
            params,
            mode="global_auction",
            ev_curve=ev_curve,
            accepted_position_callback=no_tight_callback,
            market_mode=str(args.market_mode),
        )
        base_full_accepted = _accepted_trades(candidates, base_full_decisions)
        base_full_net, base_full_gross, base_full_reason = _returns_from_accepted(base_full_accepted)
        base_full_metrics = _metrics_for_returns(
            base_full_accepted,
            base_full_net,
            base_full_gross,
            prefix="full_sim_base_",
        )
        base_full_weekly, base_full_by_head, base_full_weekly_by_head = _summary_tables(
            base_full_accepted,
            base_full_net,
            base_full_gross,
            base_full_reason,
            config_id="full_sim_no_tightening",
        )
        (
            full_summary,
            full_weekly_configs,
            full_by_head_configs,
            full_weekly_by_head_configs,
            full_accepted_by_config,
        ) = _run_full_portfolio_replays(
            candidates,
            params,
            ev_curve,
            full_configs,
            full_arrays,
            tuple(candidate_paths[:4]),
            market_mode=str(args.market_mode),
        )
        baseline_net_pnl = float(base_full_metrics.get("full_sim_base_net_pnl", 0.0))
        baseline_objective = float(base_full_metrics.get("full_sim_base_objective", 0.0))
        if not full_summary.empty:
            full_summary["delta_vs_full_baseline_net_pnl"] = (
                pd.to_numeric(full_summary["net_pnl"], errors="coerce") - baseline_net_pnl
            )
            full_summary["delta_vs_full_baseline_objective"] = (
                pd.to_numeric(full_summary["objective"], errors="coerce") - baseline_objective
            )
        full_weekly = pd.concat([artifact_full_weekly, base_full_weekly, full_weekly_configs], ignore_index=True)
        full_by_head = pd.concat([artifact_full_by_head, base_full_by_head, full_by_head_configs], ignore_index=True)
        full_weekly_by_head = pd.concat([artifact_full_weekly_by_head, base_full_weekly_by_head, full_weekly_by_head_configs], ignore_index=True)
        full_summary.to_csv(args.out_dir / "full_replay_config_summary.csv", index=False)
        full_weekly.to_csv(args.out_dir / "full_replay_weekly_metrics.csv", index=False)
        full_by_head.to_csv(args.out_dir / "full_replay_per_strategy_metrics.csv", index=False)
        full_weekly_by_head.to_csv(args.out_dir / "full_replay_weekly_per_strategy_metrics.csv", index=False)
        if not full_summary.empty:
            best_full_id = str(full_summary.iloc[0]["config_id"])
            best_full_accepted = full_accepted_by_config.get(best_full_id)
            if best_full_accepted is not None:
                best_full_accepted.to_parquet(args.out_dir / "full_replay_best_accepted.parquet", index=False)
        full_manifest = {
            "full_portfolio_replay": True,
            "full_replay_top_configs": int(args.full_replay_top_configs),
            "full_replay_config_ids": [cfg.config_id for cfg in full_configs],
            "candidate_path_coverage_gt1_pct": float((candidate_coverage > 1).mean() * 100.0) if len(candidate_coverage) else 0.0,
            "full_artifact_metrics": artifact_full_metrics,
            "full_sim_no_tightening_metrics": base_full_metrics,
            "full_sim_no_tightening_replay_metrics": _base_full_replay_metrics,
            "full_winning_config": full_summary.iloc[0].to_dict() if not full_summary.empty else {},
        }
    if int(args.independent_optuna_trials or 0) > 0:
        assert candidate_paths is not None and full_arrays is not None
        (
            independent_summary,
            independent_weekly,
            independent_by_head,
            independent_weekly_by_head,
            independent_trials,
        ) = _run_independent_optuna_study(
            candidates,
            params,
            ev_curve,
            full_arrays,
            tuple(candidate_paths[:4]),
            market_mode=str(args.market_mode),
            n_trials=int(args.independent_optuna_trials),
            seed=int(args.independent_optuna_seed),
            timeout_seconds=int(args.independent_optuna_timeout_seconds or 0) or None,
        )
        if not independent_summary.empty:
            if "full_sim_base_net_pnl" in full_manifest.get("full_sim_no_tightening_metrics", {}):
                baseline_net_pnl = float(full_manifest["full_sim_no_tightening_metrics"]["full_sim_base_net_pnl"])
                baseline_objective = float(full_manifest["full_sim_no_tightening_metrics"]["full_sim_base_objective"])
            else:
                no_tight_callback = _make_exit_adjust_callback(no_tight_cfg, full_arrays, tuple(candidate_paths[:4]), params=params)
                base_full_decisions, _base_full_equity, _base_full_replay_metrics = replay_candidates(
                    candidates,
                    params,
                    mode="global_auction",
                    ev_curve=ev_curve,
                    accepted_position_callback=no_tight_callback,
                    market_mode=str(args.market_mode),
                )
                base_full_accepted = _accepted_trades(candidates, base_full_decisions)
                base_full_net, base_full_gross, _base_full_reason = _returns_from_accepted(base_full_accepted)
                base_full_metrics = _metrics_for_returns(
                    base_full_accepted,
                    base_full_net,
                    base_full_gross,
                    prefix="full_sim_base_",
                )
                baseline_net_pnl = float(base_full_metrics.get("full_sim_base_net_pnl", 0.0))
                baseline_objective = float(base_full_metrics.get("full_sim_base_objective", 0.0))
                full_manifest.setdefault("full_sim_no_tightening_metrics", base_full_metrics)
                full_manifest.setdefault("full_sim_no_tightening_replay_metrics", _base_full_replay_metrics)
            independent_summary["delta_vs_full_baseline_net_pnl"] = (
                pd.to_numeric(independent_summary["net_pnl"], errors="coerce") - baseline_net_pnl
            )
            independent_summary["delta_vs_full_baseline_objective"] = (
                pd.to_numeric(independent_summary["objective"], errors="coerce") - baseline_objective
            )
        independent_summary.to_csv(args.out_dir / "independent_optuna_summary.csv", index=False)
        independent_trials.to_csv(args.out_dir / "independent_optuna_trials.csv", index=False)
        independent_weekly.to_csv(args.out_dir / "independent_optuna_best_weekly_metrics.csv", index=False)
        independent_by_head.to_csv(args.out_dir / "independent_optuna_best_per_strategy_metrics.csv", index=False)
        independent_weekly_by_head.to_csv(args.out_dir / "independent_optuna_best_weekly_per_strategy_metrics.csv", index=False)
        independent_manifest = {
            "independent_optuna_trials": int(args.independent_optuna_trials),
            "independent_optuna_seed": int(args.independent_optuna_seed),
            "independent_optuna_timeout_seconds": int(args.independent_optuna_timeout_seconds or 0),
            "independent_winning_config": independent_summary.iloc[0].to_dict() if not independent_summary.empty else {},
        }

    summary.to_csv(args.out_dir / "exit_tightening_config_summary.csv", index=False)
    weekly.to_csv(args.out_dir / "weekly_metrics.csv", index=False)
    by_head.to_csv(args.out_dir / "per_strategy_metrics.csv", index=False)
    weekly_by_head.to_csv(args.out_dir / "weekly_per_strategy_metrics.csv", index=False)
    accepted.to_parquet(args.out_dir / "accepted_trade_universe.parquet", index=False)
    pd.DataFrame([asdict(cfg) for cfg in configs]).to_csv(args.out_dir / "config_grid.csv", index=False)
    manifest = {
        "generated_by": "backtest_exit_tightening_redeploy",
        "candidate_path": str(args.candidates),
        "edge_candidate_path": str(args.edge_candidates),
        "policy_config": str(args.policy_config),
        "artifact_decisions": str(args.artifact_decisions),
        "ohlcv_root": str(args.ohlcv_root),
        "periods": periods,
        "global_floor": global_floor,
        "spread_floor_bps": float(args.spread_floor_bps),
        "execution_gap_bps": float(args.execution_gap_bps),
        "candidate_rows": int(len(candidates)),
        "edge_candidate_rows": int(len(edge_candidates)),
        "accepted_rows_before_path_filter": int(len(path_ok)),
        "accepted_rows": int(len(accepted)),
        "path_coverage_pct": float(path_ok.mean() * 100.0),
        "config_count": int(len(configs)),
        "winning_config": summary.iloc[0].to_dict(),
        "artifact_baseline_metrics": baseline_metrics,
        "sim_no_tightening_metrics": sim_baseline,
        "portfolio_replay_metrics": replay_metrics,
        "parity_audit": parity_audit,
        "artifact_position_size_audit": artifact_size_audit,
        "policy_selection": policy_payload.get("selection", {}),
        "assumption_note": (
            "The legacy row-level section directly simulates synthetic exits and "
            "is not live-equivalent. Use full_sim_no_tightening and independent "
            "Optuna metrics for calibrated overlay decisions. Those replay through "
            "chronological portfolio admission/capacity; adjusted exits can release "
            "open notional before later decision timestamps, while timeout/no-op "
            "paths preserve the frozen policy outcome."
        ),
    }
    manifest.update(full_manifest)
    manifest.update(independent_manifest)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    lines = [
        "# Exit Tightening Redeploy Backtest",
        "",
        "## Assumptions",
        "",
        f"- Candidate universe: `{args.candidates}`",
        f"- Periods: {', '.join(periods)}",
        f"- Accepted trades with 1m path coverage: {len(accepted)} / {len(path_ok)} ({path_ok.mean() * 100.0:.2f}%)",
        f"- Conservative spread floor: {float(args.spread_floor_bps):.1f} bps",
        f"- Extra execution gap: {float(args.execution_gap_bps):.1f} bps",
        "- Legacy tables keep entry selection fixed by the baseline portfolio replay.",
        "- Full-replay tables, when present, re-run chronological portfolio admission/capacity after adjusted exits.",
        "",
        "## Winning Config",
        "",
        _markdown_table(summary.head(10)),
        "",
        "## Weekly Metrics",
        "",
        _markdown_table(weekly.sort_values(["config_id", "week_start"])),
        "",
        "## Per-Strategy Metrics",
        "",
        _markdown_table(by_head.sort_values(["config_id", "net_pnl"], ascending=[True, False])),
        "",
        "## Weekly Per-Strategy Metrics",
        "",
        _markdown_table(weekly_by_head.sort_values(["config_id", "week_start", "head"]), max_rows=120),
        "",
    ]
    if args.full_portfolio_replay:
        lines.extend(
            [
                "## Full Portfolio Replay Winning Configs",
                "",
                _markdown_table(full_summary.head(20)),
                "",
                "## Full Portfolio Replay Weekly Metrics",
                "",
                _markdown_table(full_weekly.sort_values(["config_id", "week_start"]), max_rows=160),
                "",
                "## Full Portfolio Replay Per-Strategy Metrics",
                "",
                _markdown_table(full_by_head.sort_values(["config_id", "net_pnl"], ascending=[True, False]), max_rows=160),
                "",
            ]
        )
    if int(args.independent_optuna_trials or 0) > 0:
        lines.extend(
            [
                "## Independent SL/Trailing Optuna Study",
                "",
                _markdown_table(independent_summary.head(20)),
                "",
                "## Independent Optuna Best Weekly Metrics",
                "",
                _markdown_table(independent_weekly.sort_values(["config_id", "week_start"]), max_rows=160),
                "",
                "## Independent Optuna Best Per-Strategy Metrics",
                "",
                _markdown_table(independent_by_head.sort_values(["config_id", "net_pnl"], ascending=[True, False]), max_rows=80),
                "",
            ]
        )
    (args.out_dir / "exit_tightening_redeploy_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True, default=str))
    print("\nTop configs")
    print(summary.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
