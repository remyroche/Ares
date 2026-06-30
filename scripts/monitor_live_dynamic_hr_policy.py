#!/usr/bin/env python3
"""Monitor live dynamic HR-surprise policy parity and execution health."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from extreme_price_movements.inference.execution_reconciliation import run_reconciliation
from extreme_price_movements.simple_policy_optimiser import (
    DEFAULT_BAR_MINUTES,
    DEFAULT_FORWARD_BARS,
    DEFAULT_POLICY_PER_SIDE_COST_PCT,
    _apply_delayed_entry_execution_model,
    _build_simple_policy_candidate_rows,
    _fetch_policy_paths,
    _make_policy_replay_store,
    _policy_path_finite_mask,
)


DEFAULT_MODEL_RUN_ID = "20260617_090000_no_mkt4_labelhpo_final_fit"
DEFAULT_POLICY_RUN_ID = "20260620_185313_no_mkt4_evband002_policy_uncertainty_ev"


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_json_safe(dict(payload)), indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_json_safe(dict(payload)), sort_keys=True) + "\n")


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.DataFrame()


def _resolve_policy_deployment_payload(
    *,
    data_root: Path,
    policy_run_id: str,
    market_mode: str,
    explicit_path: str | Path | None,
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    run_root = data_root / "artifacts" / str(policy_run_id)
    candidates = [
        run_root / "policy_params" / f"best_policy_params_{market_mode}.json",
        run_root / "policy_params" / "best_policy_params.json",
        run_root / "simple_policy_optimiser" / "deployment" / "best_policy_params.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[1]


def _resolve_policy_reference_candidates(
    *,
    data_root: Path,
    policy_run_id: str,
    explicit_path: str | Path | None,
) -> Path:
    if explicit_path:
        return Path(explicit_path)
    return (
        data_root
        / "artifacts"
        / str(policy_run_id)
        / "simple_policy_optimiser"
        / "simple_policy_candidates.parquet"
    )


def _load_policy_strategy_params(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = _read_json(path)
    out: dict[str, dict[str, Any]] = {}
    for strategy in payload.get("strategies") or []:
        if not isinstance(strategy, Mapping):
            continue
        strategy_id = str(strategy.get("strategy_id") or "").strip()
        if not strategy_id or not bool(strategy.get("selected", True)):
            continue
        params = dict(strategy)
        try:
            size_power = float(strategy.get("best_size_power", 1.0) or 1.0)
        except (TypeError, ValueError):
            size_power = 1.0
        try:
            threshold = float(strategy.get("deployment_rank_threshold", 0.0) or 0.0)
        except (TypeError, ValueError):
            threshold = 0.0
        out[strategy_id] = {
            "params": params,
            "best_size_power": size_power,
            "deployment_rank_threshold": threshold,
        }
    diagnostics = {
        "path": str(path),
        "exists": path.exists(),
        "selected_strategy_count": int(len(out)),
        "strategy_ids": sorted(out),
        "schema_version": payload.get("schema_version"),
        "generated_by": payload.get("generated_by"),
    }
    return out, diagnostics


def _load_policy_barriers(path: Path) -> tuple[dict[str, float], dict[str, Any]]:
    diagnostics = {
        "path": str(path),
        "exists": path.exists(),
        "loaded": False,
        "strategy_count": 0,
        "fallback_barrier_pct": 0.02,
    }
    if not path.exists():
        return {}, diagnostics
    try:
        frame = pd.read_parquet(path, columns=["strategy_id", "barrier_pct"])
    except Exception as exc:
        diagnostics["error"] = str(exc)
        return {}, diagnostics
    if frame.empty or "strategy_id" not in frame.columns or "barrier_pct" not in frame.columns:
        return {}, diagnostics
    frame["strategy_id"] = frame["strategy_id"].astype(str)
    barriers = (
        pd.to_numeric(frame["barrier_pct"], errors="coerce")
        .groupby(frame["strategy_id"])
        .median()
        .dropna()
        .astype(float)
        .to_dict()
    )
    diagnostics.update(
        {
            "loaded": True,
            "strategy_count": int(len(barriers)),
            "barriers": barriers,
        }
    )
    return barriers, diagnostics


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce")


def _bool_series(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    values = frame[col]
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    return values.astype(str).str.lower().isin({"1", "true", "yes", "y", "on"})


def _summary(values: pd.Series) -> dict[str, Any]:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return {"n": 0}
    return {
        "n": int(len(vals)),
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "p05": float(vals.quantile(0.05)),
        "p95": float(vals.quantile(0.95)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def _side_to_numeric(values: pd.Series, strategy_id: pd.Series) -> pd.Series:
    side_text = values.astype(str).str.lower() if values is not None else pd.Series("", index=strategy_id.index)
    strategy_text = strategy_id.astype(str).str.lower()
    short_mask = side_text.str.startswith("short") | strategy_text.str.startswith("short")
    long_mask = side_text.str.startswith("long") | strategy_text.str.startswith("long")
    numeric = pd.to_numeric(values, errors="coerce") if values is not None else pd.Series(np.nan, index=strategy_id.index)
    numeric = numeric.where(numeric.notna(), np.where(short_mask, -1.0, np.where(long_mask, 1.0, 1.0)))
    return pd.Series(numeric, index=strategy_id.index, dtype="float64")


def _policy_replay_scope_mask(rows: pd.DataFrame, scope: str) -> pd.Series:
    if rows.empty:
        return pd.Series(False, index=rows.index, dtype=bool)
    decision = rows.get("portfolio_decision", pd.Series("", index=rows.index)).astype(str)
    reason = rows.get("portfolio_reject_reason", pd.Series("", index=rows.index)).astype(str)
    rejected = decision.ne("") & ~decision.isin({"accepted", "traded", "would_trade", "shadow_traded"})
    rank_rejected = decision.eq("rank_rejected") | reason.str.contains("rank_below", case=False, na=False)
    capacity_like = reason.str.contains(
        "capacity|concurrent|max_new_entries|entry_cap|cap_reached",
        case=False,
        na=False,
    )
    if scope == "disabled":
        return pd.Series(False, index=rows.index, dtype=bool)
    if scope == "all_rejects":
        return rejected
    if scope == "non_rank_rejects":
        return rejected & ~rank_rejected
    return rejected & ~rank_rejected & capacity_like


def _prepare_policy_replay_rows(
    rows: pd.DataFrame,
    *,
    barrier_by_strategy: Mapping[str, float],
) -> pd.DataFrame:
    out = rows.copy().reset_index(drop=False).rename(columns={"index": "source_row_index"})
    ts_source = "signal_bar_ts" if "signal_bar_ts" in out.columns else "timestamp"
    out["timestamp"] = pd.to_datetime(out[ts_source], utc=True, errors="coerce")
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["symbol"] = out["symbol"].astype(str)
    out["side"] = _side_to_numeric(
        out.get("side", pd.Series(np.nan, index=out.index)),
        out["strategy_id"],
    )
    if "calibrated_score" not in out.columns:
        for candidate in ("reliability_blend_score", "estimated_hit_rate", "raw_prediction_score"):
            if candidate in out.columns:
                out["calibrated_score"] = pd.to_numeric(out[candidate], errors="coerce")
                break
    rank_source = None
    for candidate in ("threshold_rank_score", "policy_rank_pct", "normalized_rank_score", "final_gate_rank_score"):
        if candidate in out.columns:
            rank = pd.to_numeric(out[candidate], errors="coerce")
            if rank.notna().any():
                out["rank_pct"] = rank
                rank_source = candidate
                break
    if "rank_pct" not in out.columns:
        out["rank_pct"] = np.nan
    if "barrier_pct" not in out.columns:
        out["barrier_pct"] = out["strategy_id"].map(dict(barrier_by_strategy)).astype("float64")
    out["barrier_pct"] = pd.to_numeric(out["barrier_pct"], errors="coerce").fillna(0.02)
    out["policy_replay_rank_source"] = rank_source or "missing"
    return out.dropna(subset=["timestamp", "symbol", "strategy_id", "calibrated_score", "rank_pct"]).reset_index(drop=True)


def _policy_path_maturity_mask(
    rows: pd.DataFrame,
    ds: Any,
    *,
    path_len: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    n = len(rows)
    mature = np.zeros(n, dtype=bool)
    available_bars = np.zeros(n, dtype=np.int32)
    details: list[dict[str, Any]] = []
    if n == 0:
        return mature, available_bars, pd.DataFrame()
    step = pd.Timedelta(minutes=int(DEFAULT_BAR_MINUTES))
    lookahead = step * max(int(path_len), 1)
    row_pos = pd.Series(np.arange(n, dtype=np.int64), index=rows.index)
    timestamps = pd.to_datetime(rows["timestamp"], utc=True, errors="coerce")
    for symbol, group in rows.groupby("symbol", sort=False):
        group_ts = timestamps.loc[group.index]
        valid_ts = group_ts.dropna()
        if valid_ts.empty:
            details.append({"symbol": str(symbol), "rows": int(len(group)), "mature_rows": 0, "reason": "missing_timestamp"})
            continue
        start_ts = valid_ts.min()
        end_ts = valid_ts.max() + lookahead
        try:
            klines = ds.load(str(symbol), start_ts=start_ts, end_ts=end_ts)
        except Exception as exc:
            details.append(
                {
                    "symbol": str(symbol),
                    "rows": int(len(group)),
                    "mature_rows": 0,
                    "reason": "load_failed",
                    "error": str(exc),
                }
            )
            continue
        if klines is None or len(klines) == 0:
            details.append({"symbol": str(symbol), "rows": int(len(group)), "mature_rows": 0, "reason": "missing_klines"})
            continue
        work = klines.reset_index()
        if "ts" not in work.columns and "index" in work.columns:
            work = work.rename(columns={"index": "ts"})
        if "ts" not in work.columns:
            details.append({"symbol": str(symbol), "rows": int(len(group)), "mature_rows": 0, "reason": "missing_ts_column"})
            continue
        k_ts = pd.to_datetime(work["ts"], utc=True, errors="coerce")
        valid = ~pd.isna(k_ts)
        if not bool(valid.any()):
            details.append({"symbol": str(symbol), "rows": int(len(group)), "mature_rows": 0, "reason": "empty_valid_ts"})
            continue
        k_ms = (k_ts.loc[valid].astype("int64").to_numpy() // 10**6).astype(np.int64)
        event_ms = (group_ts.astype("int64").to_numpy() // 10**6).astype(np.int64)
        start_pos = np.searchsorted(k_ms, event_ms)
        group_available = np.maximum(0, len(k_ms) - start_pos).astype(np.int32)
        group_mature = (start_pos >= 0) & ((start_pos + int(path_len) - 1) < len(k_ms))
        rel_idx = row_pos.loc[group.index].to_numpy(dtype=np.int64)
        mature[rel_idx] = group_mature
        available_bars[rel_idx] = np.minimum(group_available, int(path_len))
        latest_kline = pd.to_datetime(k_ts.loc[valid].max(), utc=True, errors="coerce")
        required_latest = pd.to_datetime(group_ts.max(), utc=True, errors="coerce") + step * max(int(path_len) - 1, 0)
        details.append(
            {
                "symbol": str(symbol),
                "rows": int(len(group)),
                "mature_rows": int(group_mature.sum()),
                "min_available_bars": int(np.nanmin(group_available)) if len(group_available) else 0,
                "max_available_bars": int(np.nanmax(group_available)) if len(group_available) else 0,
                "latest_kline_ts": latest_kline.isoformat() if not pd.isna(latest_kline) else None,
                "required_latest_ts": required_latest.isoformat() if not pd.isna(required_latest) else None,
                "reason": "ok" if bool(group_mature.all()) else "pending_future_bars",
            }
        )
    return mature, available_bars, pd.DataFrame(details)


def _policy_replay_summary(table: pd.DataFrame) -> dict[str, Any]:
    if table.empty:
        return {"rows": 0}
    status = table.get("policy_replay_status", pd.Series("", index=table.index)).astype(str)
    resolved = table.loc[status.eq("resolved")].copy()
    net = pd.to_numeric(resolved.get("net_return"), errors="coerce") if not resolved.empty else pd.Series(dtype="float64")
    by_head: dict[str, Any] = {}
    if "dynamic_hr_surprise_head" in table.columns:
        for head, group in table.groupby("dynamic_hr_surprise_head", dropna=False):
            g_status = group.get("policy_replay_status", pd.Series("", index=group.index)).astype(str)
            g_resolved = group.loc[g_status.eq("resolved")]
            g_net = pd.to_numeric(g_resolved.get("net_return"), errors="coerce") if not g_resolved.empty else pd.Series(dtype="float64")
            by_head[str(head)] = {
                "rows": int(len(group)),
                "resolved_rows": int(len(g_resolved)),
                "pending_rows": int(g_status.eq("pending_future_bars").sum()),
                "mean_net_return": float(g_net.mean()) if len(g_net) else None,
                "sum_net_return": float(g_net.sum()) if len(g_net) else None,
                "hit_rate": float((g_net > 0.0).mean()) if len(g_net) else None,
            }
    return {
        "rows": int(len(table)),
        "status_counts": status.value_counts(dropna=False).to_dict(),
        "resolved_rows": int(len(resolved)),
        "pending_rows": int(status.eq("pending_future_bars").sum()),
        "mean_net_return": float(net.mean()) if len(net) else None,
        "median_net_return": float(net.median()) if len(net) else None,
        "sum_net_return": float(net.sum()) if len(net) else None,
        "hit_rate": float((net > 0.0).mean()) if len(net) else None,
        "net_return": _summary(net),
        "by_head": by_head,
        "return_basis": "simple_policy_candidate_net_return_with_policy_fees_and_spread_adjustments",
        "mtm_proxy_used": False,
    }


def _counterfactual_policy_replay(
    *,
    latest_rows: pd.DataFrame,
    candle_dir: Path,
    data_root: Path,
    policy_run_id: str,
    market_mode: str,
    exchange: str,
    scope: str,
    deployment_payload_path: Path,
    reference_candidates_path: Path,
    path_len: int,
    max_rows: int,
    download_missing_15m: bool = False,
) -> dict[str, Any]:
    candle_dir.mkdir(parents=True, exist_ok=True)
    output_path = candle_dir / "counterfactual_policy_replay.parquet"
    detail_path = candle_dir / "counterfactual_policy_replay_path_maturity.parquet"
    summary_path = candle_dir / "counterfactual_policy_replay_summary.json"
    if latest_rows.empty:
        summary = {"status": "skipped_empty_latest_rows", "rows": 0}
        _write_json(summary_path, summary)
        return summary
    if scope == "disabled":
        summary = {"status": "disabled", "rows": 0}
        _write_json(summary_path, summary)
        return summary
    selected_mask = _policy_replay_scope_mask(latest_rows, scope)
    selected = latest_rows.loc[selected_mask].copy()
    rank_rejects = (
        latest_rows.get("portfolio_decision", pd.Series("", index=latest_rows.index)).astype(str).eq("rank_rejected")
        | latest_rows.get("portfolio_reject_reason", pd.Series("", index=latest_rows.index)).astype(str).str.contains("rank_below", case=False, na=False)
    )
    if int(max_rows) > 0 and len(selected) > int(max_rows):
        selected = selected.head(int(max_rows)).copy()
    if selected.empty:
        summary = {
            "status": "skipped_no_policy_eligible_rejects",
            "scope": scope,
            "latest_rows": int(len(latest_rows)),
            "rank_rejected_rows": int(rank_rejects.sum()),
            "selected_rows": 0,
            "output_path": str(output_path),
            "mtm_proxy_used": False,
        }
        pd.DataFrame().to_parquet(output_path, index=False)
        _write_json(summary_path, summary)
        return summary

    params_by_strategy, params_diag = _load_policy_strategy_params(deployment_payload_path)
    barrier_by_strategy, barrier_diag = _load_policy_barriers(reference_candidates_path)
    prepared = _prepare_policy_replay_rows(selected, barrier_by_strategy=barrier_by_strategy)
    if prepared.empty:
        summary = {
            "status": "skipped_no_replayable_rows_after_schema_normalisation",
            "scope": scope,
            "selected_rows": int(len(selected)),
            "policy_params": params_diag,
            "barriers": barrier_diag,
            "output_path": str(output_path),
            "mtm_proxy_used": False,
        }
        pd.DataFrame().to_parquet(output_path, index=False)
        _write_json(summary_path, summary)
        return summary

    os.environ["EPM_EXCHANGE"] = str(exchange)
    os.environ["EXCHANGE_NAME"] = str(exchange)
    os.environ["EPM_SIMPLE_POLICY_15M_DOWNLOAD"] = "1" if download_missing_15m else "0"
    ds = _make_policy_replay_store(data_root, market_mode)
    full_maturity_mask, available_bars, maturity_detail = _policy_path_maturity_mask(
        prepared,
        ds,
        path_len=int(path_len),
    )
    if not maturity_detail.empty:
        maturity_detail.to_parquet(detail_path, index=False)

    replay_rows = prepared.copy()
    simulation_mask = available_bars > 1
    replay_rows["policy_replay_status"] = np.where(
        simulation_mask,
        "pending_simulation",
        "pending_future_bars",
    )
    replay_rows["policy_replay_full_path_mature"] = full_maturity_mask
    replay_rows["policy_replay_available_forward_bars"] = available_bars
    replay_rows["policy_replay_scope"] = scope
    replay_rows["policy_replay_path_len"] = int(path_len)
    replay_rows["policy_replay_bar_minutes"] = int(DEFAULT_BAR_MINUTES)
    replay_rows["policy_replay_mtm_proxy_used"] = False

    resolved_frames: list[pd.DataFrame] = []
    simulated_rows = replay_rows.loc[simulation_mask].copy().reset_index(drop=True)
    for strategy_id, group in simulated_rows.groupby("strategy_id", sort=False):
        strategy_policy = params_by_strategy.get(str(strategy_id))
        if not strategy_policy:
            replay_rows.loc[replay_rows["strategy_id"].eq(strategy_id) & replay_rows["policy_replay_status"].eq("pending_simulation"), "policy_replay_status"] = "missing_strategy_policy_params"
            continue
        group = group.reset_index(drop=True)
        paths = _fetch_policy_paths(group, ds, path_len=int(path_len))
        finite_before = _policy_path_finite_mask(paths)
        group, paths = _apply_delayed_entry_execution_model(
            group,
            paths,
            data_root=str(data_root),
            market_mode=market_mode,
        )
        finite_after = _policy_path_finite_mask(paths)
        finite = finite_before & finite_after
        if not bool(finite.any()):
            idx = replay_rows["source_row_index"].isin(group.get("source_row_index", pd.Series(dtype=object)))
            replay_rows.loc[idx & replay_rows["policy_replay_status"].eq("pending_simulation"), "policy_replay_status"] = "missing_policy_path"
            continue
        finite_group = group.loc[finite].copy().reset_index(drop=True)
        finite_paths = tuple(np.asarray(arr)[finite] for arr in paths)
        candidates = _build_simple_policy_candidate_rows(
            strategy_id=str(strategy_id),
            df_top=finite_group,
            paths=finite_paths,  # type: ignore[arg-type]
            cost_pct=float(DEFAULT_POLICY_PER_SIDE_COST_PCT),
            best_params=dict(strategy_policy["params"]),
            best_size_power=float(strategy_policy["best_size_power"]),
            base_strategy_threshold=float(strategy_policy["deployment_rank_threshold"]),
            market_mode=market_mode,
        )
        if candidates.empty:
            idx = replay_rows["source_row_index"].isin(finite_group.get("source_row_index", pd.Series(dtype=object)))
            replay_rows.loc[idx & replay_rows["policy_replay_status"].eq("pending_simulation"), "policy_replay_status"] = "simulator_filtered_all_rows"
            continue
        keep_cols = [
            "timestamp",
            "symbol",
            "strategy_id",
            "side",
            "entry_price",
            "policy_executable_entry_price",
            "exit_timestamp",
            "exit_price",
            "net_return",
            "net_return_before_spread",
            "gross_return",
            "fees_bps",
            "spread_adjustment_bps",
            "expected_spread_bps",
            "expected_half_spread_bps",
            "spread_cost_bps",
            "exit_spread_cost_bps",
            "holding_bars",
            "simple_policy_exit_reason",
            "entry_execution_source",
            "delayed_entry_ts",
            "delayed_entry_effective_ts",
            "entry_gap_bps",
            "entry_slippage_proxy_bps",
            "barrier_pct",
            "policy_effective_barrier_pct",
        ]
        candidates = candidates[[c for c in keep_cols if c in candidates.columns]].copy()
        candidates["strategy_id"] = str(strategy_id)
        resolved_frames.append(candidates)

    resolved = pd.concat(resolved_frames, axis=0, ignore_index=True) if resolved_frames else pd.DataFrame()
    if not resolved.empty:
        left = replay_rows.copy()
        left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
        right = resolved.copy()
        right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce")
        merge_cols = ["timestamp", "symbol", "strategy_id"]
        enriched = left.merge(right, on=merge_cols, how="left", suffixes=("", "_policy"))
        has_result = pd.to_numeric(enriched.get("net_return"), errors="coerce").notna()
        reason = enriched.get("simple_policy_exit_reason", pd.Series("", index=enriched.index)).astype(str).str.lower()
        holding_bars = pd.to_numeric(enriched.get("holding_bars"), errors="coerce")
        available = pd.to_numeric(enriched.get("policy_replay_available_forward_bars"), errors="coerce")
        full_path = _bool_series(enriched, "policy_replay_full_path_mature")
        early_policy_exit = reason.ne("timeout") & holding_bars.lt(available)
        resolvable = has_result & (full_path | early_policy_exit)
        enriched.loc[
            resolvable & enriched["policy_replay_status"].eq("pending_simulation"),
            "policy_replay_status",
        ] = "resolved"
        enriched.loc[
            has_result
            & ~resolvable
            & enriched["policy_replay_status"].eq("pending_simulation"),
            "policy_replay_status",
        ] = "pending_future_bars"
        replay_rows = enriched
    replay_rows.loc[
        replay_rows["policy_replay_status"].eq("pending_simulation"),
        "policy_replay_status",
    ] = "simulator_filtered"

    replay_rows.to_parquet(output_path, index=False)
    summary = {
        "status": "ok",
        "scope": scope,
        "latest_rows": int(len(latest_rows)),
        "rank_rejected_rows": int(rank_rejects.sum()),
        "selected_rows": int(len(selected)),
        "prepared_rows": int(len(prepared)),
        "full_path_mature_rows": int(full_maturity_mask.sum()),
        "simulation_attempt_rows": int(simulation_mask.sum()),
        "pending_future_bar_rows": int((available_bars <= 1).sum()),
        "policy_params": params_diag,
        "barriers": barrier_diag,
        "path_maturity_path": str(detail_path),
        "output_path": str(output_path),
        "summary": _policy_replay_summary(replay_rows),
        "mtm_proxy_used": False,
        "download_missing_15m": bool(download_missing_15m),
    }
    _write_json(summary_path, summary)
    return summary


def _latest_candle_rows(ledger_path: Path) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    ledger = _read_table(ledger_path)
    if ledger.empty or "signal_bar_ts" not in ledger.columns:
        return pd.DataFrame(), None
    signal_ts = pd.to_datetime(ledger["signal_bar_ts"], utc=True, errors="coerce")
    latest_ts = signal_ts.max()
    if pd.isna(latest_ts):
        return pd.DataFrame(), None
    rows = ledger.loc[signal_ts.eq(latest_ts)].copy()
    return rows, pd.Timestamp(latest_ts)


def _stamp(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    return pd.Timestamp(ts).strftime("%Y%m%dT%H%M%SZ")


def _source_parity_path(live_data_root: Path, run_id: str, ts: pd.Timestamp | None) -> Path:
    return (
        live_data_root
        / "artifacts"
        / str(run_id)
        / "live_source_parity"
        / f"{_stamp(ts)}_model_sources.json"
    )


def _feature_parity_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    accepted = payload.get("accepted_symbols") or []
    rejected = payload.get("rejected_symbols") or []
    errors = payload.get("global_errors") or []
    return {
        "path": str(path),
        "exists": path.exists(),
        "ok": bool(payload.get("ok", False)) if payload else False,
        "mode": payload.get("mode"),
        "end_ts": payload.get("end_ts"),
        "accepted_symbols": int(len(accepted)) if isinstance(accepted, list) else 0,
        "rejected_symbols": int(len(rejected)) if isinstance(rejected, list) else 0,
        "global_error_count": int(len(errors)) if isinstance(errors, list) else 0,
        "source_rejection_summary": payload.get("source_rejection_summary") or {},
        "required_source_groups": payload.get("required_source_groups") or [],
    }


def _threshold_parity_summary(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"rows": 0}
    applied = _bool_series(rows, "dynamic_hr_surprise_applied")
    final_threshold = _num(rows, "final_threshold")
    dynamic_threshold = _num(rows, "dynamic_hr_surprise_threshold")
    final_gate_threshold = _num(rows, "final_gate_threshold")
    threshold_rank = _num(rows, "threshold_rank_score")
    policy_rank = _num(rows, "policy_rank_pct")
    gate_rank = _num(rows, "final_gate_rank_score")
    rank_score = threshold_rank.where(threshold_rank.notna(), gate_rank)
    threshold = final_gate_threshold.where(final_gate_threshold.notna(), final_threshold)
    portfolio_decision = rows.get("portfolio_decision", pd.Series("", index=rows.index)).astype(str)
    reject_reason = rows.get("portfolio_reject_reason", pd.Series("", index=rows.index)).astype(str)
    rank_rejected = portfolio_decision.eq("rank_rejected") | reject_reason.str.contains("rank_below", na=False)
    passed_like = portfolio_decision.isin({"accepted", "traded", "would_trade", "shadow_traded"}) | _bool_series(rows, "was_traded")
    below = rank_score.lt(threshold - 1e-9)
    dynamic_mismatch = applied & (final_threshold.sub(dynamic_threshold).abs() > 1e-9)
    policy_rank_mismatch = threshold_rank.notna() & policy_rank.notna() & threshold_rank.sub(policy_rank).abs().gt(1e-9)
    gate_mismatch = (rank_rejected & ~below) | (passed_like & below)
    by_head = {}
    if "dynamic_hr_surprise_head" in rows.columns:
        for head, group in rows.groupby("dynamic_hr_surprise_head", dropna=False):
            by_head[str(head)] = {
                "rows": int(len(group)),
                "threshold": _summary(_num(group, "final_threshold")),
                "z_eff": _summary(_num(group, "dynamic_hr_surprise_z_eff")),
                "guarded_y": _summary(_num(group, "dynamic_hr_surprise_guarded_y")),
                "w_lower": _summary(_num(group, "dynamic_hr_surprise_w_lower")),
                "w_raise": _summary(_num(group, "dynamic_hr_surprise_w_raise")),
            }
    return {
        "rows": int(len(rows)),
        "dynamic_applied_rows": int(applied.sum()),
        "dynamic_threshold_mismatch_rows": int(dynamic_mismatch.sum()),
        "threshold_rank_policy_mismatch_rows": int(policy_rank_mismatch.sum()),
        "rank_gate_mismatch_rows": int(gate_mismatch.sum()),
        "rank_rejected_rows": int(rank_rejected.sum()),
        "passed_like_rows": int(passed_like.sum()),
        "threshold_rank_score": _summary(threshold_rank),
        "final_threshold": _summary(final_threshold),
        "estimated_ev_net_return": _summary(_num(rows, "estimated_ev_net_return")),
        "estimated_ev_cost_bps": _summary(_num(rows, "estimated_ev_cost_bps")),
        "by_dynamic_head": by_head,
        "reject_reasons": reject_reason.value_counts(dropna=False).head(12).to_dict(),
    }


def _spread_execution_summary(rows: pd.DataFrame) -> dict[str, Any]:
    if rows.empty:
        return {"rows": 0}
    live_spread = _num(rows, "ticker_spread_bps").where(
        _num(rows, "ticker_spread_bps").notna(),
        _num(rows, "spread_bps"),
    )
    realized = _num(rows, "realized_entry_price")
    expected_fill = _num(rows, "expected_fill_price")
    side = rows.get("side", pd.Series("", index=rows.index)).astype(str)
    sign = np.where(side.eq("short"), -1.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        fill_delta = pd.Series(
            sign * (realized.to_numpy(dtype=float) / np.maximum(expected_fill.to_numpy(dtype=float), 1e-12) - 1.0) * 10000.0,
            index=rows.index,
        )
    traded = _bool_series(rows, "was_traded")
    return {
        "rows": int(len(rows)),
        "traded_rows": int(traded.sum()),
        "live_spread_bps": _summary(live_spread),
        "expected_fill_slippage_bps": _summary(_num(rows, "expected_fill_slippage_bps")),
        "expected_total_entry_friction_bps": _summary(_num(rows, "expected_total_entry_friction_bps")),
        "actual_fill_vs_expected_bps_traded": _summary(fill_delta.loc[traded]),
        "signal_to_entry_seconds": _summary(_num(rows, "signal_to_entry_seconds").loc[traded]),
        "decision_to_entry_seconds": _summary(_num(rows, "decision_to_entry_seconds").loc[traded]),
    }


def _drift_uncertainty_summary(rows: pd.DataFrame, live_data_root: Path) -> dict[str, Any]:
    perf_path = live_data_root / "live_state" / "dynamic_strategy_performance.json"
    perf = _read_json(perf_path)
    strategy_perf = {}
    for key, value in (perf.get("strategies") or {}).items():
        if not isinstance(value, Mapping):
            continue
        strategy_perf[str(key)] = {
            "reason": value.get("reason"),
            "threshold_multiplier": value.get("threshold_multiplier"),
            "recent_resolved_n_21d": value.get("recent_resolved_n_21d"),
            "expected_hit_rate_oos_top40": value.get("expected_hit_rate_oos_top40"),
            "inference_drift_score": value.get("inference_drift_score"),
            "uncertainty_score": value.get("uncertainty_score"),
            "prediction_score_psi": value.get("prediction_score_psi"),
            "rank_pct_psi": value.get("rank_pct_psi"),
        }
    return {
        "dynamic_strategy_performance_path": str(perf_path),
        "dynamic_strategy_performance_updated_at": perf.get("updated_at"),
        "hit_rate_surprise_z_eff": _summary(_num(rows, "dynamic_hr_surprise_z_eff")),
        "inference_drift_score": _summary(_num(rows, "inference_drift_score")),
        "uncertainty_score": _summary(_num(rows, "uncertainty_score")),
        "prob_uncertainty": _summary(_num(rows, "prob_uncertainty")),
        "feature_drift_psi_core": _summary(_num(rows, "feature_drift_psi_core")),
        "feature_drift_ks_core": _summary(_num(rows, "feature_drift_ks_core")),
        "feature_drift_cov_shift": _summary(_num(rows, "feature_drift_cov_shift")),
        "strategy_performance": strategy_perf,
    }


def _reconciliation_summary(
    *,
    latest_rows: pd.DataFrame,
    candle_dir: Path,
    policy_config: Path,
    data_root: Path,
    model_run_id: str,
    trade_log: Path,
    prediction_parity_max_rows: int,
    initial_wallet: float,
) -> dict[str, Any]:
    if latest_rows.empty:
        return {"status": "skipped_empty_latest_rows"}
    candle_dir.mkdir(parents=True, exist_ok=True)
    latest_path = candle_dir / "latest_candle_prediction_ledger.parquet"
    latest_rows.to_parquet(latest_path, index=False)
    try:
        result = run_reconciliation(
            prediction_ledger_path=latest_path,
            portfolio_policy_config_path=policy_config,
            output_dir=candle_dir / "reconciliation",
            trade_log_path=trade_log if trade_log.exists() else None,
            data_root=data_root,
            run_id=model_run_id,
            prediction_parity_max_rows=prediction_parity_max_rows,
            initial_wallet=initial_wallet,
        )
        result["status"] = "ok"
        return result
    except Exception as exc:
        return {"status": "failed", "error": str(exc)}


def _render_markdown(report: Mapping[str, Any]) -> str:
    feature = report.get("feature_parity", {})
    threshold = report.get("threshold_parity", {})
    spread = report.get("spread_execution", {})
    drift = report.get("drift_uncertainty", {})
    replay = report.get("counterfactual_policy_replay", {})
    replay_summary = replay.get("summary", {}) if isinstance(replay, Mapping) else {}
    recon = report.get("reconciliation", {})
    return "\n".join(
        [
            f"# Live Dynamic HR Monitor - {report.get('checked_at')}",
            "",
            f"- Mode: `{report.get('mode')}`",
            f"- Latest candle: `{report.get('latest_signal_bar_ts')}`",
            f"- New candle processed: `{report.get('new_candle_processed')}`",
            f"- Inference PID: `{report.get('inference_pid')}` alive=`{report.get('inference_pid_alive')}`",
            "",
            "## Feature Parity",
            f"- Source parity exists: `{feature.get('exists')}` ok=`{feature.get('ok')}`",
            f"- Accepted/rejected symbols: `{feature.get('accepted_symbols')}` / `{feature.get('rejected_symbols')}`",
            f"- Rejection summary: `{feature.get('source_rejection_summary')}`",
            "",
            "## Threshold Parity",
            f"- Dynamic applied rows: `{threshold.get('dynamic_applied_rows')}` / `{threshold.get('rows')}`",
            f"- Dynamic threshold mismatches: `{threshold.get('dynamic_threshold_mismatch_rows')}`",
            f"- Rank-policy mismatches: `{threshold.get('threshold_rank_policy_mismatch_rows')}`",
            f"- Rank gate mismatches: `{threshold.get('rank_gate_mismatch_rows')}`",
            f"- Reject reasons: `{threshold.get('reject_reasons')}`",
            "",
            "## Spread And Execution",
            f"- Traded rows: `{spread.get('traded_rows')}` / `{spread.get('rows')}`",
            f"- Live spread bps: `{spread.get('live_spread_bps')}`",
            f"- Actual fill vs expected bps: `{spread.get('actual_fill_vs_expected_bps_traded')}`",
            "",
            "## HR Surprise / Drift / Uncertainty",
            f"- HR surprise z_eff: `{drift.get('hit_rate_surprise_z_eff')}`",
            f"- Inference drift: `{drift.get('inference_drift_score')}`",
            f"- Uncertainty: `{drift.get('uncertainty_score')}`",
            f"- Prob uncertainty: `{drift.get('prob_uncertainty')}`",
            "",
            "## Counterfactual Policy Replay",
            f"- Status: `{replay.get('status')}`",
            f"- Scope: `{replay.get('scope')}`",
            f"- Selected/replayed rows: `{replay.get('selected_rows')}` / `{replay_summary.get('resolved_rows')}`",
            f"- Pending future bars: `{replay_summary.get('pending_rows')}`",
            f"- Mean net return: `{replay_summary.get('mean_net_return')}`",
            f"- Hit rate: `{replay_summary.get('hit_rate')}`",
            f"- MTM proxy used: `{replay.get('mtm_proxy_used')}`",
            f"- Output: `{replay.get('output_path')}`",
            "",
            "## Reconciliation",
            f"- Status: `{recon.get('status')}`",
            f"- Output dir: `{recon.get('output_dir')}`",
            f"- Spread/slippage: `{recon.get('spread_slippage')}`",
            f"- Decision replay: `{recon.get('decision_replay')}`",
            f"- Prediction/rank parity: `{recon.get('prediction_rank_parity')}`",
            "",
        ]
    )


def _pid_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except ProcessLookupError:
        return False
    except OSError:
        try:
            proc = subprocess.run(
                ["ps", "-p", str(pid)],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return proc.returncode == 0
        except OSError:
            return False


def run_check(args: argparse.Namespace, *, force_reconcile: bool = False) -> dict[str, Any]:
    monitor_root = Path(args.output_dir)
    monitor_root.mkdir(parents=True, exist_ok=True)
    state_path = monitor_root / "monitor_state.json"
    state = _read_json(state_path)
    latest_rows, latest_ts = _latest_candle_rows(Path(args.prediction_ledger))
    latest_iso = latest_ts.isoformat() if latest_ts is not None else None
    previous_iso = state.get("last_reconciled_signal_bar_ts")
    new_candle = bool(latest_iso and latest_iso != previous_iso)
    should_reconcile = bool(force_reconcile or new_candle)
    candle_dir = monitor_root / _stamp(latest_ts)
    pid = None
    if args.pid_file and Path(args.pid_file).exists():
        try:
            pid = int(Path(args.pid_file).read_text(encoding="utf-8").strip())
        except Exception:
            pid = None
    feature_path = _source_parity_path(Path(args.live_data_root), args.model_run_id, latest_ts)
    report: dict[str, Any] = {
        "checked_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "mode": args.mode,
        "latest_signal_bar_ts": latest_iso,
        "latest_rows": int(len(latest_rows)),
        "new_candle_processed": should_reconcile,
        "inference_pid": pid,
        "inference_pid_alive": _pid_alive(pid),
        "feature_parity": _feature_parity_summary(feature_path),
        "threshold_parity": _threshold_parity_summary(latest_rows),
        "spread_execution": _spread_execution_summary(latest_rows),
        "drift_uncertainty": _drift_uncertainty_summary(latest_rows, Path(args.live_data_root)),
    }
    policy_replay_wrote_candle_artifact = False
    if str(args.policy_replay_scope) != "disabled":
        report["counterfactual_policy_replay"] = _counterfactual_policy_replay(
            latest_rows=latest_rows,
            candle_dir=candle_dir,
            data_root=Path(args.data_root),
            policy_run_id=args.policy_run_id,
            market_mode=args.market_mode,
            exchange=args.exchange,
            scope=args.policy_replay_scope,
            deployment_payload_path=_resolve_policy_deployment_payload(
                data_root=Path(args.data_root),
                policy_run_id=args.policy_run_id,
                market_mode=args.market_mode,
                explicit_path=args.policy_deployment_payload,
            ),
            reference_candidates_path=_resolve_policy_reference_candidates(
                data_root=Path(args.data_root),
                policy_run_id=args.policy_run_id,
                explicit_path=args.policy_reference_candidates,
            ),
            path_len=int(args.policy_replay_path_len),
            max_rows=int(args.policy_replay_max_rows),
            download_missing_15m=bool(args.policy_replay_download_missing_15m),
        )
        policy_replay_wrote_candle_artifact = True
    else:
        report["counterfactual_policy_replay"] = {"status": "disabled", "mtm_proxy_used": False}
    if should_reconcile:
        report["reconciliation"] = _reconciliation_summary(
            latest_rows=latest_rows,
            candle_dir=candle_dir,
            policy_config=Path(args.portfolio_policy_config),
            data_root=Path(args.data_root),
            model_run_id=args.model_run_id,
            trade_log=Path(args.trade_log),
            prediction_parity_max_rows=int(args.prediction_parity_max_rows),
            initial_wallet=float(args.initial_wallet),
        )
        if latest_iso:
            state["last_reconciled_signal_bar_ts"] = latest_iso
    else:
        report["reconciliation"] = {
            "status": "skipped_no_new_candle",
            "last_reconciled_signal_bar_ts": previous_iso,
        }
    state["last_checked_at"] = report["checked_at"]
    state["last_signal_bar_ts"] = latest_iso
    _write_json(state_path, state)
    _write_json(monitor_root / "latest_monitor_report.json", report)
    _append_jsonl(monitor_root / "monitor_reports.jsonl", report)
    md_path = monitor_root / "latest_monitor_report.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    if should_reconcile or policy_replay_wrote_candle_artifact:
        _write_json(candle_dir / "monitor_report.json", report)
        (candle_dir / "monitor_report.md").write_text(_render_markdown(report), encoding="utf-8")
    print(json.dumps(_json_safe(report), indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--live-data-root", default="data_perp/exchanges/krakenfutures")
    parser.add_argument("--model-run-id", default=DEFAULT_MODEL_RUN_ID)
    parser.add_argument("--policy-run-id", default=DEFAULT_POLICY_RUN_ID)
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument(
        "--prediction-ledger",
        default=(
            "data_perp/exchanges/krakenfutures/live_state/prediction_ledgers/"
            f"{DEFAULT_MODEL_RUN_ID}/prediction_ledger.parquet"
        ),
    )
    parser.add_argument(
        "--portfolio-policy-config",
        default=(
            "data_perp/artifacts/"
            f"{DEFAULT_POLICY_RUN_ID}/policy_params/optimized_portfolio_policy_config.json"
        ),
    )
    parser.add_argument("--trade-log", default="inference_trades.csv")
    parser.add_argument(
        "--output-dir",
        default="data_perp/exchanges/krakenfutures/live_state/monitoring/dynamic_hr_t16",
    )
    parser.add_argument("--pid-file", default="")
    parser.add_argument("--mode", default="live")
    parser.add_argument("--interval-seconds", type=int, default=1800)
    parser.add_argument("--prediction-parity-max-rows", type=int, default=120)
    parser.add_argument("--initial-wallet", type=float, default=10_000.0)
    parser.add_argument("--policy-deployment-payload", default="")
    parser.add_argument("--policy-reference-candidates", default="")
    parser.add_argument(
        "--policy-replay-scope",
        choices=("capacity_rejects", "non_rank_rejects", "all_rejects", "disabled"),
        default="capacity_rejects",
        help="Rejected latest-candle rows to replay under the simple policy; rank rejects are excluded by default.",
    )
    parser.add_argument("--policy-replay-path-len", type=int, default=DEFAULT_FORWARD_BARS)
    parser.add_argument("--policy-replay-max-rows", type=int, default=250)
    parser.add_argument(
        "--policy-replay-download-missing-15m",
        action="store_true",
        help="Allow policy replay to fill missing 15m bars from the exchange chart API.",
    )
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force-reconcile", action="store_true")
    args = parser.parse_args()
    first = True
    while True:
        run_check(args, force_reconcile=bool(args.force_reconcile and first))
        first = False
        if args.once:
            return 0
        time.sleep(max(int(args.interval_seconds), 60))


if __name__ == "__main__":
    raise SystemExit(main())
