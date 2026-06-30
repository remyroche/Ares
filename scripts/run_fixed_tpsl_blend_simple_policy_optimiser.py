#!/usr/bin/env python3
"""Run a simple-policy threshold pass on reliability-blend TP/SL outcomes.

This is intentionally a narrow wrapper around the reliability-blend outcome
ledger.  The native ``simple_policy_optimiser`` builds candidates from model
artifacts and therefore ranks with the original meta score.  This script
materialises a compatible candidate table whose policy rank columns are driven
by the selected reliability-blend score, then runs a transparent rank-threshold
grid over the already computed fixed TP/SL outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.reliability_blend_rank_reference import apply_frozen_policy_rank_reference


DEFAULT_SOURCE_DIR = Path(
    "data_perp/reports/reliability_blend_optuna_20260623_native_lgbm_only_50k"
)
DEFAULT_OUTPUT_RUN_ID = "reliability_blend_volnorm_tpsl_policy_fees_20260624"
DEFAULT_CONFIG_PATH = Path("config/reliability_blend_default_configs.json")
DEFAULT_ROW_OUTCOMES = "reliability_blend_volnorm_tpsl_tp150_sl100_h5_v48_row_outcomes.parquet"

STRATEGY_IDS = {
    "long_bars": (
        "long_bars_in_high_vol_state_log_norm_-0_49417102_loc_range_pos_48_0_22034115"
        "_loc_swing_range_pos_24_1_0002919_atr_percentile_-1_477338_range_24h_pct_0_13988039"
        "_variance_ratio_10_48_0_92117828"
    ),
    "long_dist": (
        "long_dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579"
        "_leverage_build_score_0_45107844_return_autocorr_48_1_18643_rolling_range_20_-0_25967735"
    ),
    "short_asset": (
        "short_asset_minus_mkt_oi_1d_peer_resid_0_34164831"
        "_oi_expansion_compression_balance_24h_0_42287597"
    ),
    "short_boll": (
        "short_bollinger_band_width_-0_0062114433_oi_value_z_90d_0_082444385"
        "_price_rv_15d_robust_z_0_060036644"
    ),
}

SIDES = {
    "long_bars": "long",
    "long_dist": "long",
    "short_asset": "short",
    "short_boll": "short",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rank_pct(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.rank(method="average", pct=True)


def _normalise_outcome(values: pd.Series) -> pd.Series:
    raw = values.astype(str).str.strip().str.lower()
    mapped = raw.replace(
        {
            "0": "sl",
            "0.0": "sl",
            "sl": "sl",
            "stop_loss": "sl",
            "stop": "sl",
            "1": "timeout",
            "1.0": "timeout",
            "timeout": "timeout",
            "none": "timeout",
            "2": "tp",
            "2.0": "tp",
            "tp": "tp",
            "take_profit": "tp",
            "takeprofit": "tp",
        }
    )
    return mapped


def _load_default_variants(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text())
    configs = payload.get("configs") or {}
    variants: dict[str, str] = {}
    for head, cfg in configs.items():
        variant = str((cfg or {}).get("variant") or "").strip()
        if variant:
            variants[str(head)] = variant
    return variants


def _materialise_candidates(
    rows: pd.DataFrame,
    variants: dict[str, str],
    *,
    round_trip_cost_pct: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    work = rows.copy()
    work["head"] = work["head"].astype(str).str.strip()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work["fixed_y_tp"] = pd.to_numeric(work["fixed_y_tp"], errors="coerce").fillna(0.0)
    work["fixed_return"] = pd.to_numeric(work["fixed_return"], errors="coerce")

    missing: list[dict[str, Any]] = []
    blend_scores = np.full(len(work), np.nan, dtype=np.float64)
    blend_variants: list[str | None] = [None] * len(work)
    for head, idx in work.groupby("head", sort=False).groups.items():
        variant = variants.get(str(head))
        col = f"blend_{variant}_score" if variant else ""
        if not variant or col not in work.columns:
            missing.append(
                {
                    "head": head,
                    "variant": variant,
                    "missing_score_column": col,
                    "rows": int(len(idx)),
                }
            )
            continue
        positions = np.asarray(idx, dtype=np.int64)
        blend_scores[positions] = pd.to_numeric(
            work.iloc[positions][col], errors="coerce"
        ).to_numpy(dtype=np.float64)
        for pos in positions:
            blend_variants[int(pos)] = variant

    work["reliability_blend_score"] = blend_scores
    work["reliability_blend_variant"] = blend_variants
    work["anchor_score"] = pd.to_numeric(work["anchor_score"], errors="coerce")
    work = work.dropna(
        subset=["timestamp", "head", "symbol", "reliability_blend_score", "fixed_return"]
    ).copy()

    work["blend_strategy_rank_pct_debug"] = work.groupby("head", group_keys=False)[
        "reliability_blend_score"
    ].apply(_rank_pct)
    work["anchor_strategy_rank_pct"] = work.groupby("head", group_keys=False)[
        "anchor_score"
    ].apply(_rank_pct)
    work["blend_timestamp_rank_pct"] = work.groupby(["head", "timestamp"], group_keys=False)[
        "reliability_blend_score"
    ].apply(_rank_pct)
    work["anchor_timestamp_rank_pct"] = work.groupby(["head", "timestamp"], group_keys=False)[
        "anchor_score"
    ].apply(_rank_pct)
    outcome_name = _normalise_outcome(work["fixed_outcome"])
    gross_return = pd.to_numeric(work["fixed_return"], errors="coerce")
    net_return = gross_return - float(round_trip_cost_pct)
    round_trip_cost_bps = float(round_trip_cost_pct) * 10_000.0

    out = pd.DataFrame(
        {
            "timestamp": work["timestamp"],
            "symbol": work["symbol"].astype(str),
            "side": work["head"].map(SIDES).fillna("unknown"),
            "strategy_id": work["head"].map(STRATEGY_IDS).fillna(work["head"]),
            "head": work["head"],
            "row_id": work.get("row_id"),
            "blend_strategy_rank_pct_debug": work["blend_strategy_rank_pct_debug"],
            "calibrated_score": work["reliability_blend_score"],
            "reliability_blend_score": work["reliability_blend_score"],
            "reliability_blend_variant": work["reliability_blend_variant"],
            "anchor_score": work["anchor_score"],
            "anchor_strategy_rank_pct": work["anchor_strategy_rank_pct"],
            "blend_timestamp_rank_pct": work["blend_timestamp_rank_pct"],
            "anchor_timestamp_rank_pct": work["anchor_timestamp_rank_pct"],
            "base_strategy_threshold": np.nan,
            "deployment_rank_threshold": np.nan,
            "threshold_rank_score_source": "policy_rank_pct",
            "net_return": net_return,
            "net_return_before_spread": net_return,
            "net_return_before_legacy_entry_spread_haircut": net_return,
            "gross_return": gross_return,
            "fees_bps": round_trip_cost_bps,
            "expected_spread_bps": 0.0,
            "expected_half_spread_bps": 0.0,
            "spread_cost_bps": round_trip_cost_bps,
            "slippage_bps": 0.0,
            "holding_bars": 5,
            "simple_policy_exit_reason": outcome_name,
            "barrier_pct": pd.to_numeric(work.get("fixed_barrier_pct", pd.Series(0.03, index=work.index)), errors="coerce").fillna(0.03),
            "policy_effective_barrier_pct": pd.to_numeric(work.get("fixed_barrier_pct", pd.Series(0.03, index=work.index)), errors="coerce").fillna(0.03),
            "policy_sl_mult": (
                pd.to_numeric(work.get("fixed_effective_sl", pd.Series(0.02, index=work.index)), errors="coerce")
                / pd.to_numeric(work.get("fixed_effective_tp", pd.Series(0.03, index=work.index)), errors="coerce").replace(0.0, np.nan)
            ).fillna(2.0 / 3.0),
            "policy_atr_power": 0.0,
            "policy_atr_multiplier": np.nan,
            "policy_hard_tp_abs_pct": pd.to_numeric(work.get("fixed_effective_tp", pd.Series(0.03, index=work.index)), errors="coerce").fillna(0.03),
            "policy_target_holding_hours": 5.0,
            "market_mode": "perps",
            "fixed_outcome": outcome_name,
            "fixed_y_tp": work["fixed_y_tp"],
            "fixed_return": gross_return,
            "fixed_barrier_pct": work.get("fixed_barrier_pct"),
            "fixed_effective_tp": work.get("fixed_effective_tp"),
            "fixed_effective_sl": work.get("fixed_effective_sl"),
            "fixed_barrier_mode": work.get("fixed_barrier_mode"),
            "fixed_return_net_after_cost": net_return,
            "round_trip_cost_pct": float(round_trip_cost_pct),
            "fixed_conflict_same_bar": work["fixed_conflict_same_bar"].astype(bool),
            "score_source": "reliability_blend_default_variant",
            "outcome_source": work.get("fixed_barrier_mode", pd.Series("fixed_or_volnorm_tpsl", index=work.index)),
        }
    )
    return out.reset_index(drop=True), missing


def _threshold_metrics(rows: pd.DataFrame) -> dict[str, Any]:
    n = int(len(rows))
    if n == 0:
        return {
            "n_trades": 0,
            "hit_rate": 0.0,
            "tp_rate": 0.0,
            "sl_rate": 0.0,
            "timeout_rate": 0.0,
            "conflict_rate": 0.0,
            "net_pnl": 0.0,
            "mean_net_trade": 0.0,
            "median_net_trade": 0.0,
            "q05_net_trade": 0.0,
            "q25_net_trade": 0.0,
            "timestamp_count": 0,
            "mean_trades_per_timestamp": 0.0,
            "positive_timestamp_share": 0.0,
        }
    returns = pd.to_numeric(rows["net_return"], errors="coerce").fillna(0.0)
    outcomes = _normalise_outcome(rows["fixed_outcome"])
    by_ts = returns.groupby(pd.to_datetime(rows["timestamp"], utc=True)).sum()
    return {
        "n_trades": n,
        "hit_rate": float(pd.to_numeric(rows["fixed_y_tp"], errors="coerce").mean()),
        "tp_rate": float((outcomes == "tp").mean()),
        "sl_rate": float((outcomes == "sl").mean()),
        "timeout_rate": float((outcomes == "timeout").mean()),
        "conflict_rate": float(pd.Series(rows["fixed_conflict_same_bar"]).astype(bool).mean()),
        "net_pnl": float(returns.sum()),
        "mean_net_trade": float(returns.mean()),
        "median_net_trade": float(returns.median()),
        "q05_net_trade": float(returns.quantile(0.05)),
        "q25_net_trade": float(returns.quantile(0.25)),
        "timestamp_count": int(by_ts.shape[0]),
        "mean_trades_per_timestamp": float(n / max(1, by_ts.shape[0])),
        "positive_timestamp_share": float((by_ts > 0.0).mean()) if len(by_ts) else 0.0,
    }


def _evaluate_threshold_grid(
    candidates: pd.DataFrame,
    *,
    rank_col: str,
    score_source: str,
    threshold_lo: float,
    threshold_hi: float,
    threshold_step: float,
    local_band_width: float,
    confirmation_bands: int,
    confirmation_min_positive: int,
    min_trades: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    thresholds = np.unique(
        np.round(
            np.arange(
                float(threshold_lo),
                float(threshold_hi) + float(threshold_step) / 2.0,
                float(threshold_step),
            ),
            4,
        )
    )
    for (head, strategy_id), group in candidates.groupby(["head", "strategy_id"], sort=True):
        group = group.copy()
        rank = pd.to_numeric(group[rank_col], errors="coerce").fillna(-np.inf)
        threshold_rows: list[dict[str, Any]] = []
        for threshold in thresholds:
            cumulative = group.loc[rank >= threshold]
            local = group.loc[
                (rank >= threshold)
                & (rank < min(1.0 + 1e-9, threshold + local_band_width))
            ]
            item = {
                "head": head,
                "strategy_id": strategy_id,
                "score_source": score_source,
                "rank_col": rank_col,
                "deployment_rank_threshold": float(threshold),
                "local_band_lo": float(threshold),
                "local_band_hi": float(min(1.0, threshold + local_band_width)),
                **{
                    f"cumulative_{k}": v
                    for k, v in _threshold_metrics(cumulative).items()
                },
                **{f"local_{k}": v for k, v in _threshold_metrics(local).items()},
            }
            next_positive_count = 0
            next_band_metrics: list[dict[str, Any]] = []
            for band_no in range(1, confirmation_bands + 1):
                band_lo = float(threshold + band_no * local_band_width)
                band_hi = float(band_lo + local_band_width)
                band = group.loc[(rank >= band_lo) & (rank < min(1.0 + 1e-9, band_hi))]
                metrics = _threshold_metrics(band)
                positive = (
                    int(metrics["n_trades"]) >= max(1, min_trades // 4)
                    and float(metrics["mean_net_trade"]) > 0.0
                )
                next_positive_count += int(positive)
                next_band_metrics.append(
                    {
                        "band_lo": band_lo,
                        "band_hi": min(1.0, band_hi),
                        "n_trades": int(metrics["n_trades"]),
                        "mean_net_trade": float(metrics["mean_net_trade"]),
                        "net_pnl": float(metrics["net_pnl"]),
                        "positive": bool(positive),
                    }
                )
            local_positive = (
                int(item["local_n_trades"]) >= max(1, min_trades // 4)
                and float(item["local_mean_net_trade"]) > 0.0
            )
            cumulative_positive = (
                int(item["cumulative_n_trades"]) >= min_trades
                and float(item["cumulative_mean_net_trade"]) > 0.0
            )
            confirmed = bool(
                local_positive
                and cumulative_positive
                and next_positive_count >= confirmation_min_positive
            )
            item["next_band_positive_count"] = int(next_positive_count)
            item["confirmation_bands"] = int(confirmation_bands)
            item["confirmation_min_positive"] = int(confirmation_min_positive)
            item["local_confirmation_passed"] = confirmed
            item["next_band_metrics_json"] = json.dumps(
                _json_safe(next_band_metrics), separators=(",", ":")
            )
            item["objective"] = float(
                item["cumulative_net_pnl"]
                - 0.25 * abs(min(0.0, item["cumulative_q05_net_trade"]))
                + 0.10 * item["cumulative_positive_timestamp_share"]
            )
            threshold_rows.append(item)
            rows.append(item)

        confirmed_rows = [r for r in threshold_rows if r["local_confirmation_passed"]]
        if confirmed_rows:
            confirmed_thresholds = np.asarray(
                [float(r["deployment_rank_threshold"]) for r in confirmed_rows],
                dtype=np.float64,
            )
            target = float(np.quantile(confirmed_thresholds, 0.20))
            nearest_threshold = min(
                confirmed_thresholds, key=lambda value: abs(float(value) - target)
            )
            candidates_at_threshold = [
                r
                for r in confirmed_rows
                if float(r["deployment_rank_threshold"]) == float(nearest_threshold)
            ]
            best = max(
                candidates_at_threshold,
                key=lambda r: (
                    float(r["cumulative_mean_net_trade"]),
                    float(r["cumulative_net_pnl"]),
                    int(r["cumulative_n_trades"]),
                ),
            )
            reason = "iq20_confirmed_positive_local_and_cumulative"
        else:
            best = max(
                threshold_rows,
                key=lambda r: (
                    int(r["next_band_positive_count"]),
                    float(r["local_mean_net_trade"]),
                    float(r["cumulative_mean_net_trade"]),
                    float(r["cumulative_net_pnl"]),
                ),
            )
            reason = "fallback_best_local_mean_no_confirmed_threshold"

        selected.append(
            {
                **best,
                "selection_reason": reason,
                "profitable_threshold_count": int(len(confirmed_rows)),
                "profitable_threshold_min": (
                    float(min(r["deployment_rank_threshold"] for r in confirmed_rows))
                    if confirmed_rows
                    else None
                ),
                "profitable_threshold_max": (
                    float(max(r["deployment_rank_threshold"] for r in confirmed_rows))
                    if confirmed_rows
                    else None
                ),
            }
        )
    return pd.DataFrame(rows), selected


def _apply_selected_thresholds(
    candidates: pd.DataFrame,
    selected: list[dict[str, Any]],
    *,
    rank_col: str,
) -> pd.DataFrame:
    threshold_by_strategy = {
        str(item["strategy_id"]): float(item["deployment_rank_threshold"])
        for item in selected
    }
    out = candidates.copy()
    out["base_strategy_threshold"] = out["strategy_id"].astype(str).map(threshold_by_strategy)
    out["deployment_rank_threshold"] = out["base_strategy_threshold"]
    keep = pd.to_numeric(out[rank_col], errors="coerce") >= pd.to_numeric(
        out["deployment_rank_threshold"], errors="coerce"
    )
    return out.loc[keep.fillna(False)].reset_index(drop=True)


def _portfolio_summary(rows: pd.DataFrame) -> dict[str, Any]:
    out = _threshold_metrics(rows)
    out["heads"] = sorted(rows["head"].astype(str).unique().tolist()) if len(rows) else []
    out["strategies"] = (
        sorted(rows["strategy_id"].astype(str).unique().tolist()) if len(rows) else []
    )
    out["timestamp_min"] = (
        pd.to_datetime(rows["timestamp"], utc=True).min().isoformat() if len(rows) else None
    )
    out["timestamp_max"] = (
        pd.to_datetime(rows["timestamp"], utc=True).max().isoformat() if len(rows) else None
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--row-outcomes", type=str, default=DEFAULT_ROW_OUTCOMES)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--data-root", type=Path, default=Path("data_perp"))
    parser.add_argument("--output-run-id", type=str, default=DEFAULT_OUTPUT_RUN_ID)
    parser.add_argument("--threshold-lo", type=float, default=0.50)
    parser.add_argument("--threshold-hi", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument(
        "--round-trip-cost-pct",
        type=float,
        default=0.002,
        help=(
            "Round-trip execution cost subtracted from fixed TP/SL gross returns. "
            "Default 0.002 matches simple_policy_optimiser's 20 bps default."
        ),
    )
    parser.add_argument("--local-band-width", type=float, default=0.05)
    parser.add_argument("--confirmation-bands", type=int, default=3)
    parser.add_argument("--confirmation-min-positive", type=int, default=2)
    parser.add_argument("--min-trades", type=int, default=5)
    parser.add_argument("--rank-reference-run-id", type=str, default=None)
    parser.add_argument(
        "--allow-window-rank-debug",
        action="store_true",
        help="Allow non-deployable whole-window rank fallback when no frozen rank reference exists.",
    )
    args = parser.parse_args()

    source_path = args.source_dir / args.row_outcomes
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    variants = _load_default_variants(args.config)
    rows = pd.read_parquet(source_path)
    candidates, missing = _materialise_candidates(
        rows,
        variants,
        round_trip_cost_pct=float(args.round_trip_cost_pct),
    )
    candidates, rank_reference_diag = apply_frozen_policy_rank_reference(
        candidates,
        data_root=args.data_root,
        run_id=args.rank_reference_run_id,
        score_col="calibrated_score",
        allow_window_rank_debug=bool(args.allow_window_rank_debug),
    )
    if candidates.empty:
        raise RuntimeError("No blend-scored candidate rows were materialised.")

    output_root = args.data_root / "artifacts" / args.output_run_id
    policy_dir = output_root / "simple_policy_optimiser"
    deployment_dir = policy_dir / "deployment"
    policy_params_dir = output_root / "policy_params"
    policy_dir.mkdir(parents=True, exist_ok=True)
    deployment_dir.mkdir(parents=True, exist_ok=True)
    policy_params_dir.mkdir(parents=True, exist_ok=True)

    blend_grid, blend_selected = _evaluate_threshold_grid(
        candidates,
        rank_col="policy_rank_pct",
        score_source="reliability_blend",
        threshold_lo=args.threshold_lo,
        threshold_hi=args.threshold_hi,
        threshold_step=args.threshold_step,
        local_band_width=args.local_band_width,
        confirmation_bands=args.confirmation_bands,
        confirmation_min_positive=args.confirmation_min_positive,
        min_trades=args.min_trades,
    )
    anchor_grid, anchor_selected = _evaluate_threshold_grid(
        candidates,
        rank_col="anchor_strategy_rank_pct",
        score_source="anchor_meta",
        threshold_lo=args.threshold_lo,
        threshold_hi=args.threshold_hi,
        threshold_step=args.threshold_step,
        local_band_width=args.local_band_width,
        confirmation_bands=args.confirmation_bands,
        confirmation_min_positive=args.confirmation_min_positive,
        min_trades=args.min_trades,
    )

    broad_path = policy_dir / "simple_policy_candidates_broad.parquet"
    candidates.to_parquet(broad_path, index=False)
    deployable = _apply_selected_thresholds(
        candidates, blend_selected, rank_col="policy_rank_pct"
    )
    candidate_path = policy_dir / "simple_policy_candidates.parquet"
    deployable_path = policy_dir / "simple_policy_candidates_deployable.parquet"
    deployable.to_parquet(candidate_path, index=False)
    deployable.to_parquet(deployable_path, index=False)

    blend_grid_path = policy_dir / "blend_threshold_sensitivity.csv"
    anchor_grid_path = policy_dir / "anchor_threshold_sensitivity.csv"
    comparison_path = policy_dir / "blend_vs_anchor_selected_thresholds.csv"
    blend_grid.to_csv(blend_grid_path, index=False)
    anchor_grid.to_csv(anchor_grid_path, index=False)
    selected_df = pd.DataFrame(blend_selected + anchor_selected)
    selected_df.to_csv(comparison_path, index=False)

    selected_by_strategy = {
        str(item["strategy_id"]): item for item in blend_selected
    }
    strategies = []
    for item in blend_selected:
        head_candidates = candidates.loc[candidates["head"].astype(str).eq(str(item["head"]))]
        median_barrier = pd.to_numeric(head_candidates.get("barrier_pct"), errors="coerce").median()
        median_tp = pd.to_numeric(head_candidates.get("policy_hard_tp_abs_pct"), errors="coerce").median()
        median_sl_mult = pd.to_numeric(head_candidates.get("policy_sl_mult"), errors="coerce").median()
        rank_sources = (
            head_candidates.get("threshold_rank_score_source", pd.Series(dtype=object))
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        strategies.append(
            {
                "strategy_id": item["strategy_id"],
                "head": item["head"],
                "side": SIDES.get(str(item["head"]), "unknown"),
                "deployment_rank_threshold": float(item["deployment_rank_threshold"]),
                "threshold_rank_score_source": rank_sources[0]
                if len(rank_sources) == 1
                else "policy_rank_reference_percentile_mixed",
                "score_source": "reliability_blend_default_variant",
                "blend_variant": variants.get(str(item["head"])),
                "policy_effective_barrier_pct": float(median_barrier) if np.isfinite(median_barrier) else 0.03,
                "policy_sl_mult": float(median_sl_mult) if np.isfinite(median_sl_mult) else 2.0 / 3.0,
                "policy_hard_tp_abs_pct": float(median_tp) if np.isfinite(median_tp) else 0.03,
                "policy_target_holding_hours": 5.0,
                "deployment_threshold_metrics": _json_safe(item),
            }
        )

    deployment_payload = {
        "schema_version": "simple_policy_v1",
        "generated_by": "run_fixed_tpsl_blend_simple_policy_optimiser",
        "run_id": args.output_run_id,
        "market_mode": "perps",
        "selection_rules": {
            "metric_type": "OOF_tpsl_proxy",
            "score_source": "reliability_blend_default_variant",
            "outcome_source": str(candidates.get("fixed_barrier_mode", pd.Series(["fixed_or_volnorm_tpsl"])).dropna().astype(str).iloc[0])
            if "fixed_barrier_mode" in candidates.columns and candidates["fixed_barrier_mode"].notna().any()
            else "fixed_or_volnorm_tpsl",
            "costs_included": bool(float(args.round_trip_cost_pct) != 0.0),
            "round_trip_cost_pct": float(args.round_trip_cost_pct),
            "round_trip_cost_bps": float(args.round_trip_cost_pct) * 10_000.0,
            "threshold_space": "per_strategy_rank_percentile",
            "threshold_rank_score_source": str(candidates["threshold_rank_score_source"].dropna().astype(str).iloc[0])
            if "threshold_rank_score_source" in candidates.columns and candidates["threshold_rank_score_source"].notna().any()
            else "policy_rank_pct",
            "threshold_lo": float(args.threshold_lo),
            "threshold_hi": float(args.threshold_hi),
            "threshold_step": float(args.threshold_step),
            "local_band_width": float(args.local_band_width),
            "confirmation_bands": int(args.confirmation_bands),
            "confirmation_min_positive": int(args.confirmation_min_positive),
            "min_trades": int(args.min_trades),
            "selection_quantile": 0.20,
        },
        "strategies": strategies,
        "rejected_strategies": [],
        "source": {
            "row_outcomes_path": str(source_path),
            "row_outcomes_sha256": _file_sha256(source_path),
            "default_config_path": str(args.config),
            "default_config_sha256": _file_sha256(args.config),
            "default_variants": variants,
            "missing_blend_score_columns": missing,
            "rank_reference": rank_reference_diag,
        },
        "candidate_artifacts": {
            "broad_candidates": str(broad_path),
            "deployable_candidates": str(candidate_path),
            "blend_threshold_sensitivity": str(blend_grid_path),
            "anchor_threshold_sensitivity": str(anchor_grid_path),
            "selected_threshold_comparison": str(comparison_path),
        },
    }
    for path in (
        deployment_dir / "best_policy_params.json",
        policy_params_dir / "best_policy_params.json",
        output_root / "best_policy_params.json",
        output_root / "strategy_for_inference.json",
    ):
        path.write_text(json.dumps(_json_safe(deployment_payload), indent=2))

    blend_deployable_summary = _portfolio_summary(deployable)
    anchor_deployable = _apply_selected_thresholds(
        candidates, anchor_selected, rank_col="anchor_strategy_rank_pct"
    )
    anchor_summary = _portfolio_summary(anchor_deployable)
    optimisation = {
        "schema_version": "tpsl_blend_simple_policy_optimiser_v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_type": "OOF_tpsl_proxy",
        "costs_included": bool(float(args.round_trip_cost_pct) != 0.0),
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "round_trip_cost_bps": float(args.round_trip_cost_pct) * 10_000.0,
        "barrier_mode": str(candidates.get("fixed_barrier_mode", pd.Series([""])).dropna().astype(str).iloc[0])
        if "fixed_barrier_mode" in candidates.columns and candidates["fixed_barrier_mode"].notna().any()
        else None,
        "effective_tp_mean": float(pd.to_numeric(candidates.get("fixed_effective_tp"), errors="coerce").mean())
        if "fixed_effective_tp" in candidates.columns
        else None,
        "effective_sl_mean": float(pd.to_numeric(candidates.get("fixed_effective_sl"), errors="coerce").mean())
        if "fixed_effective_sl" in candidates.columns
        else None,
        "barrier_pct_mean": float(pd.to_numeric(candidates.get("fixed_barrier_pct"), errors="coerce").mean())
        if "fixed_barrier_pct" in candidates.columns
        else None,
        "horizon_hours": 5,
        "source_rows": int(len(rows)),
        "candidate_rows": int(len(candidates)),
        "deployable_rows": int(len(deployable)),
        "source_path": str(source_path),
        "output_root": str(output_root),
        "blend_portfolio_summary": blend_deployable_summary,
        "anchor_selected_threshold_portfolio_summary": anchor_summary,
        "blend_selected_thresholds": _json_safe(blend_selected),
        "anchor_selected_thresholds": _json_safe(anchor_selected),
        "missing_blend_score_columns": missing,
    }
    (output_root / "policy_optimisation.json").write_text(
        json.dumps(_json_safe(optimisation), indent=2)
    )
    (output_root / "policy_optimisation_oos_metrics.json").write_text(
        json.dumps(
            _json_safe(
                {
                    **optimisation,
                    "oos_status": "not_oos",
                    "note": (
                        "This run optimises the reliability-blend TP/SL OOF ledger. "
                        "It is not a fresh chronological OOS policy result."
                    ),
                }
            ),
            indent=2,
        )
    )

    report_lines = [
        "# TP/SL Reliability-Blend Simple Policy Optimiser",
        "",
        (
            "Metric type: OOF TP/SL proxy using persisted row outcome barriers. "
            f"Round-trip cost: {float(args.round_trip_cost_pct) * 10_000.0:.1f} bps."
        ),
        "",
        f"Source rows: {len(rows):,}",
        f"Candidate rows: {len(candidates):,}",
        f"Blend deployable rows: {len(deployable):,}",
        "",
        "## Selected Blend Thresholds",
        "",
        "| head | threshold | trades | hit_rate | mean_net | net_pnl | selection |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in blend_selected:
        report_lines.append(
            "| {head} | {thr:.2f} | {n} | {hr:.3f} | {mean:.5f} | {pnl:.5f} | {reason} |".format(
                head=item["head"],
                thr=float(item["deployment_rank_threshold"]),
                n=int(item["cumulative_n_trades"]),
                hr=float(item["cumulative_hit_rate"]),
                mean=float(item["cumulative_mean_net_trade"]),
                pnl=float(item["cumulative_net_pnl"]),
                reason=item["selection_reason"],
            )
        )
    report_lines.extend(
        [
            "",
            "## Portfolio Summary",
            "",
            "| score_source | trades | hit_rate | mean_net | net_pnl | q05 | positive_ts_share |",
            "|---|---:|---:|---:|---:|---:|---:|",
            "| blend | {n} | {hr:.3f} | {mean:.5f} | {pnl:.5f} | {q05:.5f} | {pos:.3f} |".format(
                n=int(blend_deployable_summary["n_trades"]),
                hr=float(blend_deployable_summary["hit_rate"]),
                mean=float(blend_deployable_summary["mean_net_trade"]),
                pnl=float(blend_deployable_summary["net_pnl"]),
                q05=float(blend_deployable_summary["q05_net_trade"]),
                pos=float(blend_deployable_summary["positive_timestamp_share"]),
            ),
            "| anchor_meta | {n} | {hr:.3f} | {mean:.5f} | {pnl:.5f} | {q05:.5f} | {pos:.3f} |".format(
                n=int(anchor_summary["n_trades"]),
                hr=float(anchor_summary["hit_rate"]),
                mean=float(anchor_summary["mean_net_trade"]),
                pnl=float(anchor_summary["net_pnl"]),
                q05=float(anchor_summary["q05_net_trade"]),
                pos=float(anchor_summary["positive_timestamp_share"]),
            ),
            "",
            "Artifacts are under:",
            "",
            f"`{output_root}`",
            "",
        ]
    )
    report_path = policy_dir / "tpsl_blend_simple_policy_report.md"
    report_path.write_text("\n".join(report_lines))

    print(json.dumps(_json_safe(optimisation), indent=2)[:6000])
    print(f"\nWrote {output_root}")


if __name__ == "__main__":
    main()
