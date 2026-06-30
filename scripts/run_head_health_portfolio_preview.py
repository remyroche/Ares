#!/usr/bin/env python3
"""Apply HeadHealth through the portfolio manager for score-only live rows.

This is a decision preview, not a realised PnL replay.  It converts live
reliability-blend scores into neutral candidate rows, applies the frozen
HeadHealth overlay, then routes the result through the same global-auction
portfolio replay used by the policy optimiser.  The accepted rows therefore
respect portfolio capacity, per-bar caps, per-strategy caps, symbol caps, and
dynamic thresholds.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from extreme_price_movements.portfolio_policy_replay import (
    fit_monotone_ev_curve,
    normalise_candidate_table,
    portfolio_policy_params_from_live_config,
    replay_candidates,
)
from scripts.run_fixed_tpsl_blend_simple_policy_optimiser import _file_sha256
from scripts.run_head_health_portfolio_policy_ablation import (
    HeadHealthState,
    _apply_head_health,
    _read_base_config,
)
from scripts.run_reliability_blend_portfolio_policy_ablation import _accepted_trades


DEFAULT_SCORE_PATH = Path(
    "data_perp/reports/reliability_blend_symbol_live_scores_20260624_jun24_0800_forced_features"
    "/live_reliability_blend_scores.parquet"
)
DEFAULT_TRAIN_CANDIDATES = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_PORTFOLIO_MANIFEST = Path(
    "data_perp/reports/reliability_blend_portfolio_policy_ablation_20260624"
    "/portfolio_policy_ablation_manifest.json"
)
DEFAULT_HEAD_HEALTH_CONFIG = Path(
    "data_perp/reports/head_health_portfolio_policy_frozen_action_20260624"
    "/head_health_policy_freeze_manifest.json"
)
DEFAULT_THRESHOLDS_CSV = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_longdist_floor070"
    "/simple_policy_optimiser/blend_native_selected_thresholds_floor070.csv"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/head_health_portfolio_preview_20260624_jun24_0800"
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _infer_side(strategy_id: Any) -> str:
    text = str(strategy_id or "").lower()
    return "short" if text.startswith("short") or text.startswith("s_") else "long"


def _load_thresholds(path: Path) -> dict[str, float]:
    table = pd.read_csv(path)
    required = {"strategy_id", "deployment_rank_threshold"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise RuntimeError(f"Threshold file {path} is missing columns: {missing}")
    out: dict[str, float] = {}
    for row in table.to_dict("records"):
        sid = str(row.get("strategy_id", "")).strip()
        value = pd.to_numeric(row.get("deployment_rank_threshold"), errors="coerce")
        if sid and np.isfinite(float(value)):
            out[sid] = float(value)
    return out


def _load_portfolio_params(path: Path, variant: str):
    manifest = json.loads(path.read_text(encoding="utf-8"))
    variants = manifest.get("variant_params")
    if isinstance(variants, dict) and variant in variants:
        payload = variants[variant]
    elif isinstance(manifest.get("refit_params"), dict):
        payload = manifest["refit_params"]
    elif isinstance(manifest.get("baseline_policy_params"), dict):
        payload = manifest["baseline_policy_params"]
    else:
        raise RuntimeError(
            f"Could not resolve portfolio params from {path}; "
            f"missing variant_params[{variant!r}], refit_params, and baseline_policy_params."
        )
    params = portfolio_policy_params_from_live_config(payload)
    return replace(params, global_threshold_floor=max(float(params.global_threshold_floor), 0.70))


def _build_score_only_candidates(
    scores: pd.DataFrame,
    *,
    score_column: str,
    thresholds: dict[str, float],
    bar_minutes: int,
) -> pd.DataFrame:
    work = scores.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["symbol"] = work["symbol"].astype(str)
    work["strategy_id"] = work["strategy_id"].astype(str)
    work["side"] = work["strategy_id"].map(_infer_side)
    work["calibrated_score"] = pd.to_numeric(work[score_column], errors="coerce")
    work = work.dropna(subset=["timestamp", "symbol", "strategy_id", "calibrated_score"])
    work = work.sort_values(["timestamp", "strategy_id", "symbol"], kind="mergesort")
    work["normalized_rank_score"] = work.groupby(["timestamp", "strategy_id"])[
        "calibrated_score"
    ].rank(method="average", pct=True)
    work["strategy_rank_pct"] = work["normalized_rank_score"]
    work["base_strategy_threshold"] = (
        work["strategy_id"].map(thresholds).fillna(0.70).astype(float)
    )
    work["deployment_rank_threshold"] = work["base_strategy_threshold"]
    work["entry_price"] = 1.0
    work["exit_price"] = 1.0
    work["exit_timestamp"] = work["timestamp"] + pd.Timedelta(minutes=int(bar_minutes))
    work["net_return"] = 0.0
    work["gross_return"] = 0.0
    work["holding_bars"] = 1.0
    work["simple_policy_exit_reason"] = "score_only_unrealized"
    work["fees_bps"] = 0.0
    work["slippage_bps"] = 0.0
    work["price_gap_bps"] = 0.0
    work["expected_friction_bps"] = 0.0
    work["liquidity_capacity_weight"] = 1.0
    return normalise_candidate_table(work)


def _summarise(
    candidates_before: pd.DataFrame,
    candidates_after: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    accepted = decisions.loc[decisions["accepted"]].copy()
    accepted_counts = (
        accepted.groupby("strategy_id").size().rename("portfolio_accepted").to_dict()
        if not accepted.empty
        else {}
    )
    rows: list[dict[str, Any]] = []
    for strategy_id, before in candidates_before.groupby("strategy_id", sort=True):
        after = candidates_after.loc[candidates_after["strategy_id"].eq(strategy_id)]
        before_sel = (
            pd.to_numeric(before["normalized_rank_score"], errors="coerce")
            >= pd.to_numeric(before["base_strategy_threshold"], errors="coerce")
        )
        after_sel = (
            pd.to_numeric(after["normalized_rank_score"], errors="coerce")
            >= pd.to_numeric(after["base_strategy_threshold"], errors="coerce")
        )
        rows.append(
            {
                "strategy_id": strategy_id,
                "head": str(after["head"].dropna().iloc[0]) if "head" in after.columns and after["head"].notna().any() else "",
                "rows": int(len(before)),
                "threshold_selected_before_head_health": int(before_sel.sum()),
                "threshold_selected_after_head_health": int(after_sel.sum()),
                "threshold_removed_by_head_health": int(max(before_sel.sum() - after_sel.sum(), 0)),
                "portfolio_accepted": int(accepted_counts.get(strategy_id, 0)),
                "head_health_median": float(pd.to_numeric(after.get("head_health"), errors="coerce").median()),
                "rank_shift_median": float(pd.to_numeric(after.get("head_health_score_delta"), errors="coerce").median()),
                "threshold_after_median": float(pd.to_numeric(after["base_strategy_threshold"], errors="coerce").median()),
                "hard_brake_rows": int(
                    pd.Series(after.get("head_health_hard_brake", False), index=after.index)
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
                "size_multiplier_median": float(
                    pd.to_numeric(after.get("portfolio_size_multiplier"), errors="coerce").median()
                ),
                "max_new_entries_bar_median": float(
                    pd.to_numeric(after.get("portfolio_max_new_entries_per_bar"), errors="coerce").median()
                ),
                "max_new_entries_strategy_median": float(
                    pd.to_numeric(
                        after.get("portfolio_max_new_entries_per_strategy_per_bar"),
                        errors="coerce",
                    ).median()
                ),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-deprecated-head-health",
        action="store_true",
        help="Run this deprecated historical audit tool. HeadHealth is disabled from active policy logic.",
    )
    parser.add_argument("--score-path", type=Path, default=DEFAULT_SCORE_PATH)
    parser.add_argument("--score-column", default="reliability_blend_score")
    parser.add_argument("--train-candidates", type=Path, default=DEFAULT_TRAIN_CANDIDATES)
    parser.add_argument("--portfolio-manifest", type=Path, default=DEFAULT_PORTFOLIO_MANIFEST)
    parser.add_argument("--portfolio-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--head-health-config", type=Path, default=DEFAULT_HEAD_HEALTH_CONFIG)
    parser.add_argument("--thresholds-csv", type=Path, default=DEFAULT_THRESHOLDS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bar-minutes", type=int, default=15)
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()
    if not bool(args.allow_deprecated_head_health):
        raise SystemExit(
            "HeadHealth active execution is deprecated and disabled. Use the "
            "reliability-blend parity/portfolio ablation path instead, or pass "
            "--allow-deprecated-head-health for historical audit reproduction."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_parquet(args.score_path)
    thresholds = _load_thresholds(args.thresholds_csv)
    train = normalise_candidate_table(pd.read_parquet(args.train_candidates))
    params = _load_portfolio_params(args.portfolio_manifest, args.portfolio_variant)
    config = _read_base_config(args.head_health_config)
    config["base_max_new_entries_per_bar"] = int(params.max_new_entries_per_bar)
    config["base_max_new_entries_per_strategy_per_bar"] = int(
        params.max_new_entries_per_strategy_per_bar
        if params.max_new_entries_per_strategy_per_bar is not None
        else params.max_new_entries_per_bar
    )
    config["base_max_concurrent_per_strategy"] = int(
        params.max_concurrent_per_strategy
        if params.max_concurrent_per_strategy is not None
        else params.max_concurrent_positions
    )
    candidates = _build_score_only_candidates(
        scores,
        score_column=args.score_column,
        thresholds=thresholds,
        bar_minutes=int(args.bar_minutes),
    )
    candidates["threshold_selected_before_head_health"] = (
        pd.to_numeric(candidates["normalized_rank_score"], errors="coerce")
        >= pd.to_numeric(candidates["base_strategy_threshold"], errors="coerce")
    )
    state = HeadHealthState.fit(train, config)
    transformed = _apply_head_health(
        candidates,
        history=train,
        reference=train,
        config=config,
        state=state,
    )
    transformed["threshold_selected_after_head_health"] = (
        pd.to_numeric(transformed["normalized_rank_score"], errors="coerce")
        >= pd.to_numeric(transformed["base_strategy_threshold"], errors="coerce")
    )
    ev_curve = fit_monotone_ev_curve(train)
    decisions, equity, metrics = replay_candidates(
        transformed,
        params,
        mode="global_auction",
        ev_curve=ev_curve,
        market_mode=args.market_mode,
    )
    accepted = _accepted_trades(transformed, decisions)
    summary = _summarise(candidates, transformed, decisions)

    transformed.to_parquet(args.output_dir / "head_health_portfolio_preview_candidates.parquet", index=False)
    decisions.to_parquet(args.output_dir / "head_health_portfolio_preview_decisions.parquet", index=False)
    equity.to_parquet(args.output_dir / "head_health_portfolio_preview_equity_placeholder.parquet", index=False)
    accepted.to_parquet(args.output_dir / "head_health_portfolio_preview_accepted.parquet", index=False)
    summary.to_csv(args.output_dir / "head_health_portfolio_preview_summary.csv", index=False)

    manifest = {
        "generated_by": "run_head_health_portfolio_preview",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "score_only_decision_preview": True,
        "pnl_metrics_valid": False,
        "reason_pnl_invalid": "Live score rows are neutral outcome placeholders; use this only for portfolio acceptance preview.",
        "score_path": str(args.score_path),
        "score_sha256": _file_sha256(args.score_path),
        "score_column": args.score_column,
        "train_candidates": str(args.train_candidates),
        "train_candidates_sha256": _file_sha256(args.train_candidates),
        "portfolio_manifest": str(args.portfolio_manifest),
        "portfolio_manifest_sha256": _file_sha256(args.portfolio_manifest),
        "portfolio_variant": args.portfolio_variant,
        "head_health_config": str(args.head_health_config),
        "head_health_config_sha256": _file_sha256(args.head_health_config),
        "thresholds_csv": str(args.thresholds_csv),
        "thresholds_csv_sha256": _file_sha256(args.thresholds_csv),
        "candidate_rows": int(len(transformed)),
        "threshold_selected_before_head_health": int(candidates["threshold_selected_before_head_health"].sum()),
        "threshold_selected_after_head_health": int(transformed["threshold_selected_after_head_health"].sum()),
        "portfolio_accepted": int(decisions["accepted"].sum()) if "accepted" in decisions.columns else 0,
        "portfolio_policy_params": params.to_live_config(),
        "head_health_config_payload": config,
        "portfolio_metrics_placeholder": metrics,
        "outputs": {
            "candidates": str(args.output_dir / "head_health_portfolio_preview_candidates.parquet"),
            "decisions": str(args.output_dir / "head_health_portfolio_preview_decisions.parquet"),
            "accepted": str(args.output_dir / "head_health_portfolio_preview_accepted.parquet"),
            "summary": str(args.output_dir / "head_health_portfolio_preview_summary.csv"),
        },
    }
    (args.output_dir / "head_health_portfolio_preview_manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(json.dumps(_json_safe({k: manifest[k] for k in [
        "candidate_rows",
        "threshold_selected_before_head_health",
        "threshold_selected_after_head_health",
        "portfolio_accepted",
    ]}), indent=2))
    print(f"\nWrote {args.output_dir}")


if __name__ == "__main__":
    main()
