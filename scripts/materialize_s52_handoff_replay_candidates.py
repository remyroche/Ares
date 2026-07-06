#!/usr/bin/env python3
"""Convert a clean S52 meta-threshold handoff into simple-policy replay candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_optimiser import (  # noqa: E402
    _json_safe,
    _with_policy_spread_cost_columns,
)


DEFAULT_HANDOFF = Path(
    "data_perp/reports/s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1/"
    "s52_trailing_regime_meta_handoff_longsplit_v2/"
    "s52_meta_threshold_top10_longaware_sidebad055_v1/"
    "s52_meta_threshold_guarded_candidates.parquet"
)
DEFAULT_OUT_DIR = Path("data_perp/reports/s52_replay_candidates_current_handoff")


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def _side_label(frame: pd.DataFrame) -> pd.Series:
    if "side_name" in frame.columns:
        text = frame["side_name"].astype(str).str.lower()
        return pd.Series(np.where(text.str.startswith("short"), "short", "long"), index=frame.index)
    side = pd.to_numeric(frame.get("side", pd.Series(1.0, index=frame.index)), errors="coerce")
    return pd.Series(np.where(side < 0.0, "short", "long"), index=frame.index)


def _rank_pct(values: pd.Series, month: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(1.0, index=values.index)
    ranked = numeric.groupby(month.astype(str)).rank(method="average", pct=True)
    return ranked.fillna(numeric.rank(method="average", pct=True)).clip(0.0, 1.0)


def _coalesce(frame: pd.DataFrame, *columns: str, default: float = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    for col in columns:
        if col not in frame.columns:
            continue
        values = pd.to_numeric(frame[col], errors="coerce")
        out = out.where(out.notna(), values)
    return out.fillna(default)


def _materialize(
    source: pd.DataFrame,
    *,
    barrier_pct: float,
    base_threshold: float,
    market_mode: str,
) -> pd.DataFrame:
    frame = source.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp", "symbol"]).copy()
    frame = frame.loc[frame.get("accepted", True).astype(bool)].copy()
    side_name = _side_label(frame)
    month = frame.get("month", frame["timestamp"].dt.strftime("%Y-%m")).astype(str)
    score = _coalesce(
        frame,
        "meta_score_oof",
        "exec_guard_score_oof",
        "meta_clean_exec_score_oos",
        "base_score_oof",
        default=0.0,
    )
    base_score = _coalesce(frame, "base_score_oof", "meta_score_oof", default=0.0)
    rank_pct = _rank_pct(score, month)
    policy_archetype = (
        frame["source_tag"].astype(str)
        if "source_tag" in frame.columns
        else side_name.astype(str) + "__" + frame.get("source_semantic_family", "unknown").astype(str)
    )

    out = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "symbol": frame["symbol"].astype(str),
            "side": np.where(side_name.eq("short"), -1.0, 1.0),
            "strategy_id": side_name + "_s52_meta_threshold_handoff",
            "rank_pct": rank_pct.astype(float),
            "calibrated_score": score.astype(float),
            "barrier_pct": float(barrier_pct),
            "base_strategy_threshold": float(base_threshold),
            "policy_archetype": policy_archetype.astype(str),
            "local_side_archetype": policy_archetype.astype(str),
            "policy_archetype_source": "s52_source_tag",
            "side_name": side_name.astype(str),
            "month": month.astype(str),
            "handoff_row_id": frame.get("handoff_row_id", pd.Series("", index=frame.index)).astype(str),
            "scenario_id": frame.get("scenario_id", pd.Series("", index=frame.index)).astype(str),
            "scenario_family": frame.get("scenario_family", pd.Series("", index=frame.index)).astype(str),
            "base_score_oof": base_score.astype(float),
            "meta_score_oof": score.astype(float),
            "exec_guard_score_oof": _coalesce(frame, "exec_guard_score_oof", "meta_score_oof", default=0.0),
            "gmm_cluster_id": _coalesce(frame, "gmm_cluster_id", default=np.nan),
            "gmm_entropy": _coalesce(frame, "gmm_entropy", default=np.nan),
            "mahalanobis_distance": _coalesce(frame, "mahalanobis_distance", default=np.nan),
            "AE_reconstruction_error": _coalesce(frame, "AE_reconstruction_error", default=np.nan),
            "dae_reconstruction_error": _coalesce(frame, "dae_reconstruction_error", default=np.nan),
            "latent_speed": _coalesce(frame, "latent_speed", default=np.nan),
            "latent_acceleration": _coalesce(frame, "latent_acceleration", default=np.nan),
            "source_semantic_family": frame.get("source_semantic_family", pd.Series("unknown", index=frame.index)).astype(str),
            "source_family": frame.get("source_family", pd.Series("unknown", index=frame.index)).astype(str),
            "source_volatility_state": frame.get("source_volatility_state", pd.Series("unknown", index=frame.index)).astype(str),
            "source_pressure_state": frame.get("source_pressure_state", pd.Series("unknown", index=frame.index)).astype(str),
            "source_trend_state": frame.get("source_trend_state", pd.Series("unknown", index=frame.index)).astype(str),
        }
    )
    if "long_source_regime_split" in frame.columns:
        out["long_source_regime_split"] = frame["long_source_regime_split"].astype(str)
    for col in (
        "aegmm_cluster",
        "aegmm_entropy_bin",
        "aegmm_distance_bin",
        "aegmm_expected_distance_bin",
        "reconstruction_bin",
        "dae_reconstruction_bin",
        "latent_speed_bin",
        "side_aegmm_cluster",
        "regime_first_touch_bad_mae_score_bin",
        "regime_timeout_score_bin",
        "regime_dirty_positive_score_bin",
        "regime_clean_exec_score_bin",
        "regime_lgbm_leaf_bad_mae_k4",
        "regime_lgbm_leaf_exec_margin_k4",
    ):
        if col in frame.columns:
            out[col] = frame[col].astype(str)

    stop_mult = _coalesce(frame, "stop_mult", default=0.5).fillna(0.5)
    tp_activation_r = _coalesce(frame, "tp_activation_r", default=0.75).fillna(0.75)
    trail_gap_r = _coalesce(frame, "trail_gap_r", default=0.35).fillna(0.35)
    max_activation_bars = _coalesce(frame, "max_activation_bars", default=12.0).fillna(12.0)
    out["policy_sl_mult"] = stop_mult.astype(float)
    out["policy_sl_abs_cap_pct"] = 0.0
    out["policy_trailing_activation_mult"] = tp_activation_r.astype(float)
    out["policy_trailing_activation_cap_pct"] = 0.0
    out["policy_trailing_activation_decay_half_life_bars"] = 0.0
    out["policy_trailing_activation_decay_start_bars"] = max_activation_bars.astype(int)
    out["policy_trailing_activation_min_mult"] = tp_activation_r.astype(float)
    out["policy_trailing_power"] = 1.5
    out["policy_trailing_squash_divisor"] = 2.0
    out["policy_giveback_beta"] = trail_gap_r.astype(float)
    out["policy_capital_protect_mfe_mult"] = 0.0
    out["policy_capital_protect_regression_frac"] = 0.45
    out["policy_capital_protect_lock_frac"] = np.nan
    out["policy_capital_protect_min_lock_bps"] = 0.0
    out["policy_atr_power"] = 1.0
    out["policy_atr_multiplier"] = 1.0
    out["policy_hard_tp_abs_pct"] = 0.0
    out["policy_median_barrier_frac"] = 1.0
    out["policy_redeploy_scale_bps"] = 100.0
    out["policy_target_holding_hours"] = _coalesce(frame, "horizon_hours", default=7.5).fillna(7.5)
    out["policy_churn_penalty_bps"] = 100.0
    out["stage_a_simple_sl_mult"] = stop_mult.astype(float)
    out["stage_a_simple_tp_mult"] = tp_activation_r.astype(float)
    out["s52_trail_gap_r"] = trail_gap_r.astype(float)
    out["s52_horizon_bars"] = _coalesce(frame, "horizon_bars", default=30.0).fillna(30.0).astype(int)

    out = _with_policy_spread_cost_columns(out, market_mode=market_mode)
    out = out.sort_values(["timestamp", "symbol", "strategy_id"], kind="mergesort").reset_index(drop=True)
    return out


def _summary(frame: pd.DataFrame) -> Dict[str, Any]:
    side_counts = frame["side_name"].value_counts().to_dict() if "side_name" in frame.columns else {}
    arch_counts = frame["policy_archetype"].value_counts().head(50).to_dict()
    return {
        "rows": int(len(frame)),
        "symbols": int(frame["symbol"].nunique()) if "symbol" in frame.columns else 0,
        "months": sorted(frame["month"].astype(str).unique().tolist()) if "month" in frame.columns else [],
        "side_counts": {str(k): int(v) for k, v in side_counts.items()},
        "policy_archetype_count": int(frame["policy_archetype"].nunique()) if "policy_archetype" in frame.columns else 0,
        "policy_archetype_counts": {str(k): int(v) for k, v in arch_counts.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--barrier-pct", type=float, default=0.03)
    parser.add_argument("--base-threshold", type=float, default=0.0)
    parser.add_argument("--market-mode", default="perps")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = _read(args.handoff)
    candidates = _materialize(
        source,
        barrier_pct=float(args.barrier_pct),
        base_threshold=float(args.base_threshold),
        market_mode=str(args.market_mode),
    )
    out_path = args.out_dir / "simple_policy_candidates_with_archetypes.parquet"
    manifest_path = args.out_dir / "manifest.json"
    candidates.to_parquet(out_path, index=False)
    manifest = {
        "generated_by": "materialize_s52_handoff_replay_candidates",
        "source_handoff": str(args.handoff),
        "output_candidates": str(out_path),
        "barrier_pct": float(args.barrier_pct),
        "base_threshold": float(args.base_threshold),
        "market_mode": str(args.market_mode),
        "summary": _summary(candidates),
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    print(json.dumps(_json_safe(manifest), indent=2))


if __name__ == "__main__":
    main()
