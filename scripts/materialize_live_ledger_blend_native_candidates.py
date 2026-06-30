#!/usr/bin/env python3
"""Materialize native simple-policy candidates from live ledgers and blend scores.

This is intentionally fail-closed.  The active replay goal requires reliability
blend scores, not the live meta score, so this script refuses to use ledger meta
scores unless the caller explicitly passes ``--allow-ledger-score``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    DEFAULT_FORWARD_BARS,
    DEFAULT_POLICY_PER_SIDE_COST_PCT,
    _apply_delayed_entry_execution_model,
    _build_simple_policy_candidate_rows,
    _fetch_policy_paths,
    _make_policy_replay_store,
    _policy_path_coverage,
)
from scripts.run_fixed_tpsl_blend_simple_policy_optimiser import (  # noqa: E402
    SIDES,
    STRATEGY_IDS,
    _file_sha256,
    _json_safe,
)
from scripts.reliability_blend_rank_reference import apply_frozen_policy_rank_reference  # noqa: E402
from scripts.score_market_state_controller_bundle import (  # noqa: E402
    _controller_config_payload as _market_state_controller_config_payload,
    _load_bundle as _load_market_state_controller_bundle,
    score_candidates as _score_market_state_controller_candidates,
)


HEAD_BY_STRATEGY_ID = {strategy_id: head for head, strategy_id in STRATEGY_IDS.items()}
STRATEGY_BY_HEAD = dict(STRATEGY_IDS)
SIDE_VALUE = {"long": 1.0, "short": -1.0}
DEFAULT_REFERENCE_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_longdist050"
    "/simple_policy_optimiser/simple_policy_candidates_broad.parquet"
)
DEFAULT_SCORE_COLUMN = "reliability_blend_score"
EMPTY_CANDIDATE_COLUMNS = (
    "timestamp",
    "symbol",
    "strategy_id",
    "head",
    "side",
    "calibrated_score",
    "reliability_blend_score",
    "policy_rank_pct",
    "strategy_rank_pct",
    "normalized_rank_score",
    "rank_pct",
    "deployment_rank_threshold",
    "base_strategy_threshold",
    "barrier_pct",
    "entry_price",
    "exit_price",
    "exit_timestamp",
    "holding_bars",
    "gross_return",
    "net_return",
    "simple_policy_exit_reason",
)


POLICY_PARAM_COLUMNS = (
    "sl_mult",
    "trailing_activation_mult",
    "trailing_power",
    "trailing_squash_divisor",
    "giveback_beta",
    "capital_protect_mfe_mult",
    "capital_protect_regression_frac",
    "adverse_exit_enabled",
    "adverse_exit_min_mae_atr",
    "adverse_exit_min_speed",
    "adverse_exit_theta_quantile",
    "adverse_exit_theta",
    "adverse_exit_alpha",
    "adverse_exit_beta",
    "adverse_exit_delta",
    "adverse_exit_fast_bars",
    "adverse_exit_max_mfe_atr",
    "atr_power",
    "atr_multiplier",
    "hard_tp_abs_pct",
    "exit_pressure_enabled",
    "exit_pressure_alpha",
    "exit_pressure_beta",
    "exit_pressure_delta",
    "exit_pressure_kappa",
    "exit_pressure_psi",
    "exit_pressure_omega",
    "exit_pressure_min_multiplier",
    "redeploy_scale_bps",
    "target_holding_hours",
    "churn_penalty_bps",
    "policy_median_barrier_frac",
    "median_barrier_frac",
    "exit_quote_half_spread_bps",
)


def _rank_pct(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rank(method="average", pct=True)


def _load_ledgers(paths: list[Path], *, start: str | None, end: str | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_parquet(path)
        df["_ledger_path"] = str(path)
        frames.append(df)
    if not frames:
        raise RuntimeError("No live ledger paths were provided.")
    out = pd.concat(frames, axis=0, ignore_index=True, copy=False)
    ts_col = "signal_bar_ts" if "signal_bar_ts" in out.columns else "timestamp"
    missing = [c for c in (ts_col, "symbol", "strategy_id") if c not in out.columns]
    if missing:
        raise RuntimeError(f"Live ledger is missing required columns: {missing}.")
    out["timestamp"] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    if start:
        out = out[out["timestamp"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        out = out[out["timestamp"] <= pd.Timestamp(end, tz="UTC")]
    out["strategy_id"] = out["strategy_id"].astype(str)
    out["head"] = out["strategy_id"].map(HEAD_BY_STRATEGY_ID)
    out = out[out["head"].notna()].copy()
    out = out.dropna(subset=["timestamp", "symbol", "strategy_id"]).copy()
    out = out.sort_values(["timestamp", "strategy_id", "symbol"], kind="mergesort")
    out = out.drop_duplicates(["timestamp", "strategy_id", "symbol"], keep="last")
    return out.reset_index(drop=True)


def _load_scores(score_path: Path | None, score_column: str) -> pd.DataFrame:
    if score_path is None:
        return pd.DataFrame()
    scores = pd.read_parquet(score_path)
    scores["timestamp"] = pd.to_datetime(scores["timestamp"], utc=True, errors="coerce")
    if score_column not in scores.columns:
        raise RuntimeError(f"Score column {score_column!r} not found in {score_path}.")
    keep = ["timestamp", "symbol", score_column]
    if "head" in scores.columns:
        keep.append("head")
    if "strategy_id" in scores.columns:
        keep.append("strategy_id")
    return scores[[c for c in keep if c in scores.columns]].copy()


def _attach_scores(
    ledgers: pd.DataFrame,
    *,
    score_path: Path | None,
    score_column: str,
    allow_ledger_score: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = ledgers.copy()
    score_source = "missing"
    matched = 0
    if out.empty:
        out["reliability_blend_score"] = pd.Series(dtype="float64")
        return out.reset_index(drop=True), {
            "score_source": "empty_live_ledger",
            "score_column": score_column,
            "matched_score_rows": 0,
            "scored_rows": 0,
        }
    if score_path is not None:
        scores = _load_scores(score_path, score_column)
        join_keys = ["timestamp", "symbol"]
        if "strategy_id" in scores.columns:
            join_keys.append("strategy_id")
        elif "head" in scores.columns:
            join_keys.append("head")
        if scores.duplicated(join_keys).any():
            raise RuntimeError(f"Score file has duplicate join keys: {join_keys}.")
        before_cols = set(out.columns)
        out = out.merge(scores, on=join_keys, how="left", suffixes=("", "_scorefile"))
        score_source = str(score_path)
        matched = int(pd.to_numeric(out.get(score_column), errors="coerce").notna().sum())
        if score_column not in before_cols and score_column not in out.columns:
            raise RuntimeError(f"Score column {score_column!r} was not attached from score file.")
    elif allow_ledger_score:
        if score_column not in out.columns:
            raise RuntimeError(f"Ledger score column {score_column!r} not found.")
        score_source = f"ledger:{score_column}"
        matched = int(pd.to_numeric(out[score_column], errors="coerce").notna().sum())
    else:
        raise RuntimeError(
            "No reliability-blend score file was supplied. Refusing to fall back to "
            "ledger meta scores. Pass --score-path with a materialized blend score "
            "table, or explicitly pass --allow-ledger-score for a non-strict audit."
        )

    out["reliability_blend_score"] = pd.to_numeric(out[score_column], errors="coerce")
    out = out[np.isfinite(out["reliability_blend_score"].to_numpy(dtype=np.float64))].copy()
    if out.empty:
        raise RuntimeError("No rows with finite reliability blend scores.")
    return out.reset_index(drop=True), {
        "score_source": score_source,
        "score_column": score_column,
        "matched_score_rows": matched,
        "scored_rows": int(len(out)),
    }


def _load_reference_policy(reference_candidates: Path) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    ref = pd.read_parquet(reference_candidates)
    ref["strategy_id"] = ref["strategy_id"].astype(str)
    params_by_strategy: dict[str, dict[str, Any]] = {}
    barrier_by_strategy: dict[str, float] = {}
    for strategy_id, group in ref.groupby("strategy_id", sort=True):
        row = group.iloc[0]
        params: dict[str, Any] = {}
        for col in POLICY_PARAM_COLUMNS:
            if col in group.columns:
                val = row[col]
                if isinstance(val, (np.bool_, bool)):
                    params[col] = bool(val)
                elif pd.isna(val):
                    continue
                else:
                    try:
                        params[col] = float(val)
                    except Exception:
                        params[col] = val
        params_by_strategy[str(strategy_id)] = params
        barrier = pd.to_numeric(group.get("barrier_pct"), errors="coerce")
        barrier_by_strategy[str(strategy_id)] = float(barrier.dropna().median()) if barrier.notna().any() else 0.005
    return params_by_strategy, barrier_by_strategy


def _default_threshold_path(reference_candidates: Path) -> Path:
    return reference_candidates.parent / "blend_native_selected_thresholds.csv"


def _load_deployment_thresholds(
    *,
    reference_candidates: Path,
    thresholds_csv: Path | None,
    forced_thresholds: list[str] | None,
) -> tuple[dict[str, float], list[dict[str, Any]], str | None]:
    threshold_path = thresholds_csv or _default_threshold_path(reference_candidates)
    thresholds: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    source: str | None = None
    if threshold_path.exists():
        table = pd.read_csv(threshold_path)
        required = {"strategy_id", "deployment_rank_threshold"}
        missing = sorted(required.difference(table.columns))
        if missing:
            raise RuntimeError(f"Threshold file {threshold_path} is missing columns: {missing}.")
        for row in table.to_dict("records"):
            sid = str(row.get("strategy_id", "")).strip()
            threshold = float(row.get("deployment_rank_threshold"))
            if sid and np.isfinite(threshold):
                thresholds[sid] = float(threshold)
                rows.append(dict(row))
        source = str(threshold_path)
    elif thresholds_csv is not None:
        raise RuntimeError(f"Requested threshold file does not exist: {thresholds_csv}")

    for raw in forced_thresholds or []:
        if "=" not in raw:
            raise ValueError(f"Invalid --force-threshold {raw!r}; expected HEAD_OR_STRATEGY_ID=THRESHOLD.")
        key, value = raw.split("=", 1)
        key = key.strip()
        threshold = float(value)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Invalid forced threshold for {key!r}: {threshold}; expected 0..1.")
        strategy_id = STRATEGY_BY_HEAD.get(key, key)
        if strategy_id not in HEAD_BY_STRATEGY_ID:
            raise ValueError(f"Unknown forced threshold key {key!r}.")
        thresholds[strategy_id] = threshold
        rows = [r for r in rows if str(r.get("strategy_id", "")) != strategy_id]
        rows.append(
            {
                "head": HEAD_BY_STRATEGY_ID[strategy_id],
                "strategy_id": strategy_id,
                "deployment_rank_threshold": threshold,
                "selection_reason": f"forced_live_materializer_threshold:{threshold:.4f}",
            }
        )
        source = (source or "none") + "+forced"
    return thresholds, rows, source


def _prepare_rows(rows: pd.DataFrame, *, barrier_pct: float) -> pd.DataFrame:
    out = rows.copy()
    out["calibrated_score"] = pd.to_numeric(out["reliability_blend_score"], errors="coerce")
    out["rank_pct_debug_not_deployable"] = _rank_pct(out["calibrated_score"])
    out["barrier_pct"] = float(barrier_pct)
    head = str(out["head"].iloc[0])
    out["side_name"] = SIDES.get(head, "")
    out["side"] = float(SIDE_VALUE.get(SIDES.get(head, "long"), 1.0))
    for src, dest in (
        ("spread_bps", "expected_spread_bps"),
        ("half_spread_bps", "expected_half_spread_bps"),
        ("entry_slippage_proxy_bps", "entry_slippage_proxy_bps"),
    ):
        if src in out.columns and dest not in out.columns:
            out[dest] = pd.to_numeric(out[src], errors="coerce")
    if "expected_spread_bps" not in out.columns:
        out["expected_spread_bps"] = 0.0
    if "expected_half_spread_bps" not in out.columns:
        out["expected_half_spread_bps"] = pd.to_numeric(
            out["expected_spread_bps"], errors="coerce"
        ).fillna(0.0) / 2.0
    if "entry_slippage_proxy_bps" not in out.columns:
        out["entry_slippage_proxy_bps"] = 0.0
    if "exit_spread_cost_bps" not in out.columns:
        out["exit_spread_cost_bps"] = pd.to_numeric(
            out["expected_half_spread_bps"], errors="coerce"
        ).fillna(0.0)
    if "exit_quote_half_spread_bps" not in out.columns:
        out["exit_quote_half_spread_bps"] = pd.to_numeric(
            out["exit_spread_cost_bps"], errors="coerce"
        ).fillna(0.0)
    return out.reset_index(drop=True)


def _metric_block(rows: pd.DataFrame) -> dict[str, Any]:
    n = int(len(rows))
    if n == 0:
        return {"rows": 0, "win_rate": np.nan, "mean_net": np.nan, "net_pnl": 0.0}
    net = pd.to_numeric(rows["net_return"], errors="coerce").fillna(0.0)
    gross = pd.to_numeric(rows.get("gross_return", net), errors="coerce").fillna(0.0)
    return {
        "rows": n,
        "timestamps": int(pd.to_datetime(rows["timestamp"], utc=True, errors="coerce").nunique()),
        "symbols": int(rows["symbol"].astype(str).nunique()),
        "win_rate": float((net > 0).mean()),
        "mean_net": float(net.mean()),
        "q05_net": float(net.quantile(0.05)),
        "net_pnl": float(net.sum()),
        "gross_pnl": float(gross.sum()),
        "cost_pnl": float((gross - net).sum()),
    }


def _latest_feature_store_dir(data_root: str | Path) -> Path | None:
    root = Path(data_root) / "features"
    if not root.exists():
        return None
    dirs = [path for path in root.iterdir() if path.is_dir()]
    if not dirs:
        return None
    return sorted(dirs, key=lambda p: p.name)[-1]


def _materialize_market_state_controller_overlay(
    *,
    candidates: pd.DataFrame,
    deployable: pd.DataFrame,
    controller_bundle_path: Path,
    feature_store_dir: Path,
    feature_store_symbol_cap: int,
    allow_candidate_state_fallback: bool,
    portfolio_policy_manifest: Path | None,
    portfolio_policy_variant: str | None,
    ev_reference_candidates: Path | None,
    market_mode: str,
    policy_dir: Path,
) -> dict[str, Any]:
    bundle = _load_market_state_controller_bundle(controller_bundle_path)
    rank_contract = str(bundle.get("rank_contract", "strict"))
    disabled_heads = set(bundle.get("disabled_heads", []))
    controller_input = candidates.copy()
    from scripts import run_market_state_threshold_controller as mstc  # local import keeps default path unchanged

    if "normalized_rank_score" not in controller_input.columns:
        for source in ("policy_rank_pct", "strategy_rank_pct", "rank_pct"):
            if source in controller_input.columns:
                controller_input["normalized_rank_score"] = pd.to_numeric(
                    controller_input[source], errors="coerce"
                )
                break
    controller_input = mstc._apply_rank_contract(controller_input, rank_contract)
    controller_input = mstc._disable_heads(controller_input, disabled_heads)
    scored, predictions, schedule, state, score_report = _score_market_state_controller_candidates(
        bundle=bundle,
        candidates=controller_input,
        feature_store_dir=feature_store_dir,
        feature_store_symbol_cap=int(feature_store_symbol_cap),
        allow_candidate_state_fallback=bool(allow_candidate_state_fallback),
    )
    rank = pd.to_numeric(scored.get("policy_rank_pct"), errors="coerce")
    threshold = pd.to_numeric(scored.get("deployment_rank_threshold"), errors="coerce").fillna(0.0)
    controller_deployable = scored.loc[(rank >= threshold).fillna(False)].copy()
    state_level = str(bundle.get("state_spec", {}).get("state_level", "observed"))
    timestamp_panel = mstc.market_state_timestamp_panel([("live_overlay", state_level, state)])
    feature_coverage = mstc.market_state_feature_coverage(timestamp_panel)
    scored.to_parquet(policy_dir / "market_state_controller_candidates_broad.parquet", index=False)
    controller_deployable.to_parquet(
        policy_dir / "market_state_controller_candidates_deployable.parquet",
        index=False,
    )
    predictions.to_parquet(policy_dir / "market_state_controller_predictions.parquet", index=False)
    state.to_parquet(policy_dir / "market_state_controller_state.parquet", index=False)
    timestamp_panel.to_parquet(policy_dir / "market_state_timestamp_panel.parquet", index=False)
    feature_coverage.to_csv(policy_dir / "market_state_feature_coverage.csv", index=False)
    schedule.to_csv(policy_dir / "market_state_controller_schedule.csv", index=False)
    schedule.to_parquet(policy_dir / "strategy_threshold_schedule.parquet", index=False)
    action_audit = mstc.threshold_action_audit(schedule)
    action_audit.to_csv(policy_dir / "strategy_threshold_action_audit.csv", index=False)
    controller_config = _market_state_controller_config_payload(bundle)
    (policy_dir / "strategy_threshold_controller_config.json").write_text(
        json.dumps(_json_safe(controller_config), indent=2) + "\n",
        encoding="utf-8",
    )
    by_head = scored.groupby("head", sort=True).apply(_metric_block, include_groups=False).apply(pd.Series)
    deployable_by_head = (
        controller_deployable.groupby("head", sort=True).apply(_metric_block, include_groups=False).apply(pd.Series)
        if not controller_deployable.empty
        else pd.DataFrame()
    )
    by_head.to_csv(policy_dir / "market_state_controller_metrics_by_head.csv")
    deployable_by_head.to_csv(policy_dir / "market_state_controller_deployable_metrics_by_head.csv")
    portfolio_replay: dict[str, Any] = {
        "enabled": False,
        "reason": "no_market_state_controller_policy_manifest",
    }
    if portfolio_policy_manifest is not None:
        if ev_reference_candidates is None:
            ev_reference_candidates = policy_dir / "simple_policy_candidates.parquet"
        if not portfolio_policy_manifest.exists():
            raise RuntimeError(f"Portfolio policy manifest does not exist: {portfolio_policy_manifest}")
        if not ev_reference_candidates.exists():
            raise RuntimeError(f"EV reference candidates do not exist: {ev_reference_candidates}")
        from extreme_price_movements.portfolio_policy_replay import (  # noqa: WPS433
            fit_hierarchical_ev_curves,
            normalise_candidate_table,
            replay_candidates,
        )

        params, _ = mstc._load_policy_params(portfolio_policy_manifest, portfolio_policy_variant)
        ev_reference = normalise_candidate_table(pd.read_parquet(ev_reference_candidates))
        ev_curve = fit_hierarchical_ev_curves(ev_reference)
        replay_rows: list[dict[str, Any]] = []
        replay_outputs: dict[str, str] = {}
        for arm, frame in (
            ("static_anchor", candidates),
            ("market_state_controller", scored),
        ):
            replay_candidates_frame = normalise_candidate_table(frame)
            decisions, equity, replay_metrics = replay_candidates(
                replay_candidates_frame,
                params,
                mode="global_auction",
                ev_curve=ev_curve,
                market_mode=str(market_mode),
            )
            accepted = mstc._accepted_trades(replay_candidates_frame, decisions)
            decisions_path = policy_dir / f"{arm}_portfolio_decisions.parquet"
            accepted_path = policy_dir / f"{arm}_accepted_trades.parquet"
            equity_path = policy_dir / f"{arm}_equity_curve.parquet"
            by_head_path = policy_dir / f"{arm}_portfolio_by_head.csv"
            decisions.to_parquet(decisions_path, index=False)
            accepted.to_parquet(accepted_path, index=False)
            equity.to_parquet(equity_path, index=False)
            mstc._by_head(arm, accepted).to_csv(by_head_path, index=False)
            schedule_for_metrics = schedule if arm == "market_state_controller" else pd.DataFrame()
            replay_rows.append(mstc._metrics_row(arm, replay_metrics, accepted, schedule_for_metrics))
            replay_outputs[f"{arm}_decisions"] = str(decisions_path)
            replay_outputs[f"{arm}_accepted_trades"] = str(accepted_path)
            replay_outputs[f"{arm}_equity_curve"] = str(equity_path)
            replay_outputs[f"{arm}_by_head"] = str(by_head_path)
        replay_summary = pd.DataFrame(replay_rows)
        replay_summary_path = policy_dir / "market_state_controller_portfolio_replay_summary.csv"
        replay_summary.to_csv(replay_summary_path, index=False)
        replay_output_paths = {
            "summary": str(replay_summary_path),
            **replay_outputs,
        }
        portfolio_replay = {
            "enabled": True,
            "policy_manifest": str(portfolio_policy_manifest),
            "policy_variant": portfolio_policy_variant,
            "ev_reference_candidates": str(ev_reference_candidates),
            "summary": replay_summary.to_dict("records"),
            "outputs": replay_output_paths,
            "output_sha256": {
                name: _file_sha256(Path(path))
                for name, path in replay_output_paths.items()
                if Path(path).exists()
            },
        }
    base_keys = {
        tuple(row)
        for row in deployable[["timestamp", "symbol", "strategy_id"]]
        .assign(timestamp=lambda x: pd.to_datetime(x["timestamp"], utc=True, errors="coerce"))
        .astype({"symbol": str, "strategy_id": str})
        .drop_duplicates()
        .to_numpy()
    } if not deployable.empty else set()
    controller_keys = {
        tuple(row)
        for row in controller_deployable[["timestamp", "symbol", "strategy_id"]]
        .assign(timestamp=lambda x: pd.to_datetime(x["timestamp"], utc=True, errors="coerce"))
        .astype({"symbol": str, "strategy_id": str})
        .drop_duplicates()
        .to_numpy()
    } if not controller_deployable.empty else set()
    union = base_keys | controller_keys
    output_paths = {
        "candidates_broad": str(policy_dir / "market_state_controller_candidates_broad.parquet"),
        "candidates_deployable": str(policy_dir / "market_state_controller_candidates_deployable.parquet"),
        "predictions": str(policy_dir / "market_state_controller_predictions.parquet"),
        "state": str(policy_dir / "market_state_controller_state.parquet"),
        "market_state_timestamp_panel": str(policy_dir / "market_state_timestamp_panel.parquet"),
        "market_state_feature_coverage": str(policy_dir / "market_state_feature_coverage.csv"),
        "schedule": str(policy_dir / "market_state_controller_schedule.csv"),
        "strategy_threshold_schedule": str(policy_dir / "strategy_threshold_schedule.parquet"),
        "strategy_threshold_action_audit": str(policy_dir / "strategy_threshold_action_audit.csv"),
        "strategy_threshold_controller_config": str(policy_dir / "strategy_threshold_controller_config.json"),
        "metrics_by_head": str(policy_dir / "market_state_controller_metrics_by_head.csv"),
        "deployable_metrics_by_head": str(policy_dir / "market_state_controller_deployable_metrics_by_head.csv"),
    }
    return {
        "enabled": True,
        "bundle": str(controller_bundle_path),
        "feature_store_dir": str(feature_store_dir),
        "rank_contract": rank_contract,
        "disabled_heads": sorted(disabled_heads),
        "selected_arm": bundle.get("selected_arm"),
        "state_level": state_level,
        "controller_enabled_heads": bundle.get("controller_enabled_heads"),
        "input_candidate_rows": int(len(controller_input)),
        "scored_candidate_rows": int(len(scored)),
        "deployable_rows": int(len(controller_deployable)),
        "static_deployable_rows": int(len(deployable)),
        "deployable_overlap": int(len(base_keys & controller_keys)),
        "deployable_added": int(len(controller_keys - base_keys)),
        "deployable_removed": int(len(base_keys - controller_keys)),
        "deployable_jaccard": float(len(base_keys & controller_keys) / max(1, len(union))),
        "score_report": score_report,
        "metrics_by_head": by_head.reset_index().to_dict("records"),
        "deployable_metrics_by_head": deployable_by_head.reset_index().to_dict("records")
        if not deployable_by_head.empty
        else [],
        "portfolio_replay": portfolio_replay,
        "outputs": output_paths,
        "output_sha256": {
            name: _file_sha256(Path(path))
            for name, path in output_paths.items()
            if Path(path).exists()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", action="append", type=Path, required=True)
    parser.add_argument("--score-path", type=Path, default=None)
    parser.add_argument("--score-column", default=DEFAULT_SCORE_COLUMN)
    parser.add_argument("--allow-ledger-score", action="store_true")
    parser.add_argument("--reference-candidates", type=Path, default=DEFAULT_REFERENCE_CANDIDATES)
    parser.add_argument("--thresholds-csv", type=Path, default=None)
    parser.add_argument("--force-threshold", action="append", default=None)
    parser.add_argument("--data-root", default="data_perp")
    parser.add_argument("--exchange", default="krakenfutures")
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--output-run-id", default="reliability_blend_live_ledger_native_candidates_20260624")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--path-len", type=int, default=DEFAULT_FORWARD_BARS)
    parser.add_argument("--rank-reference-run-id", type=str, default=None)
    parser.add_argument(
        "--market-state-controller-bundle",
        type=Path,
        default=None,
        help=(
            "Optional frozen market-state controller bundle. When supplied, "
            "the script writes dual-scored controller candidates/schedule "
            "alongside the native static-threshold outputs without replacing them."
        ),
    )
    parser.add_argument(
        "--market-state-controller-feature-store-dir",
        type=Path,
        default=None,
        help=(
            "Feature-store directory used to build the controller's current "
            "timestamp-level market state. Defaults to the latest data_root/features/* directory."
        ),
    )
    parser.add_argument("--market-state-controller-feature-store-symbol-cap", type=int, default=220)
    parser.add_argument(
        "--market-state-controller-policy-manifest",
        type=Path,
        default=None,
        help=(
            "Optional portfolio policy manifest. When supplied with a controller bundle, "
            "the script also replays static-anchor and controller candidates through "
            "the global auction and persists accepted-trade metrics."
        ),
    )
    parser.add_argument("--market-state-controller-policy-variant", type=str, default=None)
    parser.add_argument(
        "--market-state-controller-ev-reference-candidates",
        type=Path,
        default=None,
        help=(
            "Candidate table used to fit hierarchical EV curves for controller replay. "
            "Defaults to <reference-candidates parent>/simple_policy_candidates.parquet "
            "when present."
        ),
    )
    parser.add_argument(
        "--allow-market-state-controller-candidate-fallback",
        action="store_true",
        help="Allow candidate-population market-state fallback for controller smoke tests only.",
    )
    parser.add_argument(
        "--allow-window-rank-debug",
        action="store_true",
        help="Allow non-deployable current-window rank fallback for audit materialization only.",
    )
    args = parser.parse_args()

    ledgers = _load_ledgers(args.ledger, start=args.start, end=args.end)
    scored, score_diag = _attach_scores(
        ledgers,
        score_path=args.score_path,
        score_column=str(args.score_column),
        allow_ledger_score=bool(args.allow_ledger_score),
    )
    params_by_strategy, barrier_by_strategy = _load_reference_policy(args.reference_candidates)
    deployment_thresholds, threshold_rows, threshold_source = _load_deployment_thresholds(
        reference_candidates=args.reference_candidates,
        thresholds_csv=args.thresholds_csv,
        forced_thresholds=args.force_threshold,
    )
    if scored.empty:
        output_root = Path(args.data_root) / "artifacts" / str(args.output_run_id)
        policy_dir = output_root / "simple_policy_optimiser"
        policy_dir.mkdir(parents=True, exist_ok=True)
        candidates_all = pd.DataFrame(columns=list(EMPTY_CANDIDATE_COLUMNS))
        deployable = candidates_all.copy()
        broad_path = policy_dir / "simple_policy_candidates_broad.parquet"
        candidates_all.to_parquet(broad_path, index=False)
        deployable.to_parquet(policy_dir / "simple_policy_candidates.parquet", index=False)
        deployable.to_parquet(policy_dir / "simple_policy_candidates_deployable.parquet", index=False)
        coverage = pd.DataFrame(
            columns=[
                "strategy_id",
                "head",
                "input_rows",
                "path_finite_rows",
                "path_total_rows",
                "path_coverage",
                "path_finite_rows_after_delayed_entry",
                "path_total_rows_after_delayed_entry",
                "path_coverage_after_delayed_entry",
                "candidate_rows",
                "rank_reference",
            ]
        )
        coverage.to_csv(policy_dir / "live_ledger_native_path_coverage.csv", index=False)
        pd.DataFrame(threshold_rows).to_csv(policy_dir / "live_ledger_selected_thresholds.csv", index=False)
        pd.DataFrame().to_csv(policy_dir / "live_ledger_native_metrics_by_head.csv")
        pd.DataFrame().to_csv(policy_dir / "live_ledger_native_deployable_metrics_by_head.csv")
        manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generated_by": "materialize_live_ledger_blend_native_candidates",
            "score_diagnostics": score_diag,
            "ledger_paths": [str(p) for p in args.ledger],
            "ledger_sha256": {str(p): _file_sha256(p) for p in args.ledger if p.exists()},
            "score_path": str(args.score_path) if args.score_path is not None else None,
            "score_sha256": _file_sha256(args.score_path) if args.score_path is not None and args.score_path.exists() else None,
            "reference_candidates": str(args.reference_candidates),
            "reference_candidates_sha256": _file_sha256(args.reference_candidates),
            "threshold_source": threshold_source,
            "selected_thresholds": threshold_rows,
            "start": args.start,
            "end": args.end,
            "path_len": int(args.path_len),
            "exchange": str(args.exchange),
            "policy_replay_store": "",
            "policy_replay_root": "",
            "candidate_rows": 0,
            "deployable_rows": 0,
            "candidate_timestamp_min": None,
            "candidate_timestamp_max": None,
            "coverage": [],
            "rank_reference_run_id": args.rank_reference_run_id,
            "rank_reference_debug_fallback_allowed": bool(args.allow_window_rank_debug),
            "metrics_by_head": [],
            "deployable_metrics_by_head": [],
            "market_state_controller_overlay": {"enabled": False, "reason": "empty_scored_ledger"},
            "strict_score_requirement": not bool(args.allow_ledger_score),
            "outputs": {
                "candidates_broad": str(broad_path),
                "candidates_deployable": str(policy_dir / "simple_policy_candidates_deployable.parquet"),
                "simple_policy_candidates": str(policy_dir / "simple_policy_candidates.parquet"),
            },
        }
        (output_root / "live_ledger_native_materialization_manifest.json").write_text(
            json.dumps(_json_safe(manifest), indent=2) + "\n"
        )
        print(json.dumps(_json_safe(manifest), indent=2)[:6000])
        return
    # The policy replay store reads the exchange from environment-backed helpers.
    # Do not let a parent shell's stale EPM_EXCHANGE silently override the CLI
    # contract; that can route Kraken perps ledgers into a Binance execution
    # cache and drop otherwise executable candidates.
    os.environ["EPM_EXCHANGE"] = str(args.exchange)
    os.environ["EXCHANGE_NAME"] = str(args.exchange)
    ds = _make_policy_replay_store(args.data_root, args.market_mode)

    frames: list[pd.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for strategy_id, group in scored.groupby("strategy_id", sort=True):
        if strategy_id not in params_by_strategy:
            continue
        prepared = _prepare_rows(group, barrier_pct=barrier_by_strategy.get(strategy_id, 0.005))
        prepared, rank_diag = apply_frozen_policy_rank_reference(
            prepared,
            data_root=args.data_root,
            run_id=args.rank_reference_run_id,
            score_col="calibrated_score",
            allow_window_rank_debug=bool(args.allow_window_rank_debug),
        )
        prepared["rank_pct"] = pd.to_numeric(prepared["policy_rank_pct"], errors="coerce")
        paths = _fetch_policy_paths(prepared, ds, path_len=int(args.path_len))
        paths = tuple(np.asarray(arr, dtype=np.float32) for arr in paths)
        finite, total, coverage = _policy_path_coverage(paths)
        prepared, paths = _apply_delayed_entry_execution_model(
            prepared,
            paths,  # type: ignore[arg-type]
            data_root=str(args.data_root),
            market_mode=str(args.market_mode),
        )
        finite_after, total_after, coverage_after = _policy_path_coverage(paths)  # type: ignore[arg-type]
        candidates = _build_simple_policy_candidate_rows(
            strategy_id=str(strategy_id),
            df_top=prepared,
            paths=paths,  # type: ignore[arg-type]
            cost_pct=float(DEFAULT_POLICY_PER_SIDE_COST_PCT),
            best_params=params_by_strategy[str(strategy_id)],
            best_size_power=1.0,
            base_strategy_threshold=float(deployment_thresholds.get(str(strategy_id), 0.0)),
            market_mode=str(args.market_mode),
        )
        if not candidates.empty:
            candidates["head"] = candidates["strategy_id"].astype(str).map(HEAD_BY_STRATEGY_ID)
            candidates["score_source"] = "reliability_blend_score"
            candidates["reliability_blend_score"] = pd.to_numeric(candidates["calibrated_score"], errors="coerce")
            candidates["deployment_rank_threshold"] = candidates["strategy_id"].astype(str).map(
                deployment_thresholds
            )
            candidates["base_strategy_threshold"] = candidates["deployment_rank_threshold"]
            if "policy_rank_pct" not in candidates.columns:
                candidates["policy_rank_pct"] = pd.to_numeric(
                    candidates.get("strategy_rank_pct"), errors="coerce"
                )
            rank_source_values = (
                prepared.get("threshold_rank_score_source", pd.Series(dtype=object))
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            candidates["threshold_rank_score_source"] = (
                rank_source_values[0]
                if len(rank_source_values) == 1
                else "policy_rank_reference_percentile_mixed"
            )
            frames.append(candidates)
        coverage_rows.append(
            {
                "strategy_id": str(strategy_id),
                "head": HEAD_BY_STRATEGY_ID.get(str(strategy_id), ""),
                "input_rows": int(len(group)),
                "path_finite_rows": int(finite),
                "path_total_rows": int(total),
                "path_coverage": float(coverage),
                "path_finite_rows_after_delayed_entry": int(finite_after),
                "path_total_rows_after_delayed_entry": int(total_after),
                "path_coverage_after_delayed_entry": float(coverage_after),
                "candidate_rows": int(len(candidates)),
                "rank_reference": rank_diag,
            }
        )

    if not frames:
        raise RuntimeError("No native simple-policy candidate rows were materialized.")
    candidates_all = pd.concat(frames, axis=0, ignore_index=True, copy=False)
    candidates_all["timestamp"] = pd.to_datetime(candidates_all["timestamp"], utc=True, errors="coerce")
    candidates_all = candidates_all.sort_values(["timestamp", "strategy_id", "symbol"], kind="mergesort")
    rank = pd.to_numeric(candidates_all.get("policy_rank_pct"), errors="coerce")
    threshold = pd.to_numeric(candidates_all.get("deployment_rank_threshold"), errors="coerce").fillna(0.0)
    deployable = candidates_all.loc[(rank >= threshold).fillna(False)].copy()
    output_root = Path(args.data_root) / "artifacts" / str(args.output_run_id)
    policy_dir = output_root / "simple_policy_optimiser"
    policy_dir.mkdir(parents=True, exist_ok=True)
    broad_path = policy_dir / "simple_policy_candidates_broad.parquet"
    candidates_all.to_parquet(broad_path, index=False)
    deployable.to_parquet(policy_dir / "simple_policy_candidates.parquet", index=False)
    deployable.to_parquet(policy_dir / "simple_policy_candidates_deployable.parquet", index=False)
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(policy_dir / "live_ledger_native_path_coverage.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(policy_dir / "live_ledger_selected_thresholds.csv", index=False)
    by_head = candidates_all.groupby("head", sort=True).apply(_metric_block, include_groups=False).apply(pd.Series)
    deployable_by_head = (
        deployable.groupby("head", sort=True).apply(_metric_block, include_groups=False).apply(pd.Series)
        if not deployable.empty
        else pd.DataFrame()
    )
    by_head.to_csv(policy_dir / "live_ledger_native_metrics_by_head.csv")
    deployable_by_head.to_csv(policy_dir / "live_ledger_native_deployable_metrics_by_head.csv")
    controller_overlay: dict[str, Any] = {"enabled": False}
    if args.market_state_controller_bundle is not None:
        feature_store_dir = args.market_state_controller_feature_store_dir or _latest_feature_store_dir(args.data_root)
        if feature_store_dir is None or not feature_store_dir.exists():
            raise RuntimeError(
                "A market-state controller bundle was supplied, but no usable feature-store "
                "directory was provided or inferred. Pass --market-state-controller-feature-store-dir."
            )
        ev_reference_candidates = args.market_state_controller_ev_reference_candidates
        if ev_reference_candidates is None:
            default_ev_reference = args.reference_candidates.parent / "simple_policy_candidates.parquet"
            ev_reference_candidates = default_ev_reference if default_ev_reference.exists() else None
        controller_overlay = _materialize_market_state_controller_overlay(
            candidates=candidates_all,
            deployable=deployable,
            controller_bundle_path=args.market_state_controller_bundle,
            feature_store_dir=feature_store_dir,
            feature_store_symbol_cap=int(args.market_state_controller_feature_store_symbol_cap),
            allow_candidate_state_fallback=bool(args.allow_market_state_controller_candidate_fallback),
            portfolio_policy_manifest=args.market_state_controller_policy_manifest,
            portfolio_policy_variant=args.market_state_controller_policy_variant,
            ev_reference_candidates=ev_reference_candidates,
            market_mode=str(args.market_mode),
            policy_dir=policy_dir,
        )
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "materialize_live_ledger_blend_native_candidates",
        "score_diagnostics": score_diag,
        "ledger_paths": [str(p) for p in args.ledger],
        "ledger_sha256": {str(p): _file_sha256(p) for p in args.ledger if p.exists()},
        "score_path": str(args.score_path) if args.score_path is not None else None,
        "score_sha256": _file_sha256(args.score_path) if args.score_path is not None and args.score_path.exists() else None,
        "reference_candidates": str(args.reference_candidates),
        "reference_candidates_sha256": _file_sha256(args.reference_candidates),
        "threshold_source": threshold_source,
        "selected_thresholds": threshold_rows,
        "start": args.start,
        "end": args.end,
        "path_len": int(args.path_len),
        "exchange": str(args.exchange),
        "effective_exchange_env": str(os.environ.get("EPM_EXCHANGE", "")),
        "effective_exchange_name_env": str(os.environ.get("EXCHANGE_NAME", "")),
        "policy_replay_store": type(ds).__name__,
        "policy_replay_root": str(getattr(ds, "root_dir", "")),
        "candidate_rows": int(len(candidates_all)),
        "deployable_rows": int(len(deployable)),
        "candidate_timestamp_min": pd.to_datetime(candidates_all["timestamp"], utc=True).min().isoformat(),
        "candidate_timestamp_max": pd.to_datetime(candidates_all["timestamp"], utc=True).max().isoformat(),
        "coverage": coverage_rows,
        "rank_reference_run_id": args.rank_reference_run_id,
        "rank_reference_debug_fallback_allowed": bool(args.allow_window_rank_debug),
        "metrics_by_head": by_head.reset_index().to_dict("records"),
        "deployable_metrics_by_head": deployable_by_head.reset_index().to_dict("records")
        if not deployable_by_head.empty
        else [],
        "market_state_controller_overlay": controller_overlay,
        "strict_score_requirement": not bool(args.allow_ledger_score),
    }
    (output_root / "live_ledger_native_materialization_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n"
    )
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])
    print(f"\nWrote {output_root}")


if __name__ == "__main__":
    main()
