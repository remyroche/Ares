#!/usr/bin/env python3
"""Deep-dive S52 late-June decay: uncertainty, drift, support, archetypes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("data_perp/reports")
BASE_DIR = ROOT / "s52_trailing_profit_best_pointwise_scored_ledger_20260705_v1"
META_DIR = BASE_DIR / "s52_trailing_regime_meta_handoff_longsplit_v2"
CURRENT_THRESHOLD_DIR = META_DIR / "s52_meta_threshold_top10_longaware_sidebad055_v1"
REPLAY_METRICS_DIR = ROOT / "s52_side_archetype_policy_metrics_sidefix_cost1pct_20260705_v1"
WF_DIR = ROOT / "simple_policy_exit_side_archetype_s52_current_top10_longaware_sidebad055_sidefix_cost1pct_20260705_v1"
OUT_DIR = ROOT / "s52_june_decay_deep_dive_20260705_v1"

ABLATION_DIRS = {
    "original_promoted_smoke": META_DIR / "train_meta_regime_handoff_smoke_v1",
    "current_code_baseline": META_DIR / "train_meta_regime_handoff_smoke_ablate_baseline_20260705",
    "base_prior_features": META_DIR / "train_meta_regime_handoff_smoke_ablate_baseprior_20260705",
    "path_order_heads": META_DIR / "train_meta_regime_handoff_smoke_ablate_pathheads_20260705",
    "path_order_blend": META_DIR / "train_meta_regime_handoff_smoke_ablate_pathblend_20260705",
    "base_prior_path_order_blend": META_DIR / "train_meta_regime_handoff_smoke_ablate_baseprior_pathblend_20260705",
}


def _num(values: Any, default: float = np.nan) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce")
    return pd.to_numeric(pd.Series(values), errors="coerce").fillna(default)


def _rate(values: Any) -> float:
    vals = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.clip(0.0, 1.0).mean()) if len(vals) else float("nan")


def _mean(values: Any) -> float:
    vals = _num(values).replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def _entropy(prob: pd.Series) -> pd.Series:
    p = _num(prob).clip(1e-6, 1.0 - 1e-6)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def _week_start(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return (ts.dt.normalize() - pd.to_timedelta(ts.dt.weekday, unit="D")).dt.date.astype(str)


def _js_divergence(p: pd.Series, q: pd.Series) -> float:
    labels = sorted(set(p.index.astype(str)) | set(q.index.astype(str)))
    pv = p.reindex(labels, fill_value=0.0).to_numpy(dtype=np.float64)
    qv = q.reindex(labels, fill_value=0.0).to_numpy(dtype=np.float64)
    if pv.sum() <= 0 or qv.sum() <= 0:
        return float("nan")
    pv = pv / pv.sum()
    qv = qv / qv.sum()
    m = 0.5 * (pv + qv)
    mask_p = pv > 0
    mask_q = qv > 0
    kl_pm = float((pv[mask_p] * np.log2(pv[mask_p] / m[mask_p])).sum())
    kl_qm = float((qv[mask_q] * np.log2(qv[mask_q] / m[mask_q])).sum())
    return 0.5 * (kl_pm + kl_qm)


def _weekly_current_handoff(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["week"] = _week_start(df["__ts__"])
    side = df["side_name"].astype(str).str.lower()
    clean = _num(df.get("score_meta_clean_exec")).fillna(0.0)
    long_clean = _num(df.get("score_meta_long_clean_exec"))
    bad = _num(df.get("score_meta_bad_path")).fillna(1.0)
    long_bad = _num(df.get("score_meta_long_bad_path"))
    side_clean = clean.where(~side.eq("long"), long_clean.fillna(clean))
    side_bad = bad.where(~side.eq("long"), long_bad.fillna(bad))
    timeout = _num(df.get("score_meta_timeout")).fillna(1.0)
    df["score_entropy_clean"] = _entropy(side_clean)
    df["score_entropy_bad"] = _entropy(side_bad)
    df["score_entropy_timeout"] = _entropy(timeout)
    df["score_clean_bad_gap"] = side_clean - side_bad
    df["score_risk_pressure"] = side_bad + timeout
    rows: list[dict[str, Any]] = []
    for week, group in df.groupby("week", dropna=False):
        side_g = group["side_name"].astype(str).str.lower()
        rows.append(
            {
                "week": week,
                "rows": int(len(group)),
                "symbols": int(group["__symbol__"].nunique()),
                "long_share": float(side_g.eq("long").mean()),
                "short_share": float(side_g.eq("short").mean()),
                "mean_exec_margin_label": _mean(group.get("exec_margin")),
                "mean_ret_net_label": _mean(group.get("ret_net")),
                "clean_exec_precision_label": _rate(group.get("clean_exec")),
                "full_path_bad_mae_label": _rate(group.get("full_path_bad_mae_1r")),
                "timeout_label": _rate(group.get("timeout")),
                "dirty_positive_label": _rate(group.get("dirty_positive")),
                "mean_base_score": _mean(group.get("score_base")),
                "mean_meta_score": _mean(group.get("score_meta_long_aware_clean_minus_risk")),
                "mean_clean_score": _mean(side_clean.loc[group.index]),
                "mean_bad_score": _mean(side_bad.loc[group.index]),
                "mean_timeout_score": _mean(timeout.loc[group.index]),
                "mean_clean_bad_gap": _mean(group["score_clean_bad_gap"]),
                "mean_risk_pressure": _mean(group["score_risk_pressure"]),
                "mean_clean_entropy": _mean(group["score_entropy_clean"]),
                "mean_bad_entropy": _mean(group["score_entropy_bad"]),
                "mean_timeout_entropy": _mean(group["score_entropy_timeout"]),
            }
        )
    return pd.DataFrame(rows).sort_values("week")


def _replay_weekly() -> pd.DataFrame:
    stage = pd.read_csv(WF_DIR / "walkforward_stage_summary.csv")
    stage = stage[stage["arm"].isin(["A0_baseline", "A6_time_decay"])].copy()
    stage["week"] = pd.to_datetime(stage["validation_start"], utc=True, errors="coerce").dt.date.astype(str)
    keep = [
        "week",
        "arm",
        "accepted_trades",
        "portfolio_net_pnl",
        "portfolio_gross_pnl",
        "portfolio_objective",
        "portfolio_full_sl_rate",
        "portfolio_timeout_rate",
        "portfolio_max_drawdown",
        "portfolio_side_concentration",
    ]
    return stage[[c for c in keep if c in stage.columns]].sort_values(["week", "arm"])


def _numeric_feature_drift(frame: pd.DataFrame, reference_week: str = "2026-05-11") -> pd.DataFrame:
    df = frame.copy()
    df["week"] = _week_start(df["__ts__"])
    feature_cols = [
        "score_base",
        "score_meta_long_aware_clean_minus_risk",
        "score_meta_clean_exec",
        "score_meta_bad_path",
        "score_meta_timeout",
        "gmm_entropy",
        "mahalanobis_distance",
        "AE_reconstruction_error",
        "dae_reconstruction_error",
        "latent_speed",
        "latent_acceleration",
        "meta_context_weight_hint",
        "meta_threshold_adjustment_hint",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]
    ref = df[df["week"].eq(reference_week)]
    rows: list[dict[str, Any]] = []
    for col in feature_cols:
        ref_vals = _num(ref[col]).replace([np.inf, -np.inf], np.nan).dropna()
        ref_mean = float(ref_vals.mean()) if len(ref_vals) else float("nan")
        ref_std = float(ref_vals.std()) if len(ref_vals) > 1 else float("nan")
        if not math.isfinite(ref_std) or ref_std <= 1e-12:
            ref_std = 1.0
        for week, group in df.groupby("week", dropna=False):
            vals = _num(group[col]).replace([np.inf, -np.inf], np.nan).dropna()
            if not len(vals):
                continue
            rows.append(
                {
                    "feature": col,
                    "week": week,
                    "rows": int(len(vals)),
                    "mean": float(vals.mean()),
                    "median": float(vals.median()),
                    "std": float(vals.std()) if len(vals) > 1 else 0.0,
                    "reference_week": reference_week,
                    "reference_mean": ref_mean,
                    "reference_std": ref_std,
                    "mean_z_delta_vs_reference": float((vals.mean() - ref_mean) / ref_std),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "week"])


def _categorical_support_drift(frame: pd.DataFrame, reference_week: str = "2026-05-11") -> pd.DataFrame:
    df = frame.copy()
    df["week"] = _week_start(df["__ts__"])
    cat_cols = [
        "source_tag",
        "source_family",
        "source_volatility_state",
        "source_pressure_state",
        "source_trend_state",
        "aegmm_cluster",
        "side_aegmm_cluster",
        "aegmm_entropy_bin",
        "aegmm_distance_bin",
        "aegmm_expected_distance_bin",
        "reconstruction_bin",
        "regime_first_touch_bad_mae_score_bin",
        "regime_timeout_score_bin",
        "regime_dirty_positive_score_bin",
        "regime_clean_exec_score_bin",
        "regime_lgbm_leaf_bad_mae_k4",
        "regime_lgbm_leaf_exec_margin_k4",
    ]
    cat_cols = [c for c in cat_cols if c in df.columns]
    ref = df[df["week"].eq(reference_week)]
    rows: list[dict[str, Any]] = []
    for col in cat_cols:
        ref_dist = ref[col].astype(str).value_counts(normalize=True, dropna=False)
        ref_support = set(ref[col].astype(str).unique())
        for week, group in df.groupby("week", dropna=False):
            vals = group[col].astype(str)
            dist = vals.value_counts(normalize=True, dropna=False)
            rows.append(
                {
                    "feature": col,
                    "week": week,
                    "rows": int(len(group)),
                    "reference_week": reference_week,
                    "js_divergence_vs_reference": _js_divergence(ref_dist, dist),
                    "unseen_share_vs_reference": float((~vals.isin(ref_support)).mean()) if len(vals) else float("nan"),
                    "top_value": str(vals.value_counts(dropna=False).index[0]) if len(vals) else "",
                    "top_value_share": float(vals.value_counts(normalize=True, dropna=False).iloc[0]) if len(vals) else float("nan"),
                }
            )
    return pd.DataFrame(rows).sort_values(["feature", "week"])


def _archetype_replay_attribution() -> pd.DataFrame:
    detail = pd.read_csv(REPLAY_METRICS_DIR / "side_archetype_week_metrics_aggregated.csv")
    detail = detail[(detail["arm"].eq("A0_baseline")) & (detail["accepted_trades"] > 0)].copy()
    detail["is_weak_week"] = detail["period"].isin(["2026-06-15", "2026-06-22"])
    agg = (
        detail.groupby(["side", "policy_archetype"], dropna=False)
        .agg(
            weeks=("period", "nunique"),
            trades=("accepted_trades", "sum"),
            net_pnl=("accepted_net_pnl", "sum"),
            gross_pnl=("accepted_gross_pnl", "sum"),
            full_sl_rows=("accepted_full_sl_rows", "sum"),
            timeout_rows=("accepted_timeout_rows", "sum"),
            weak_week_trades=("accepted_trades", lambda s: int(s[detail.loc[s.index, "is_weak_week"]].sum())),
            weak_week_net_pnl=("accepted_net_pnl", lambda s: float(s[detail.loc[s.index, "is_weak_week"]].sum())),
        )
        .reset_index()
    )
    agg["full_sl_rate"] = agg["full_sl_rows"] / agg["trades"].clip(lower=1)
    agg["timeout_rate"] = agg["timeout_rows"] / agg["trades"].clip(lower=1)
    agg["weak_week_pnl_share"] = agg["weak_week_net_pnl"] / agg["net_pnl"].replace(0.0, np.nan)
    return agg.sort_values(["weak_week_net_pnl", "net_pnl"], ascending=[True, True])


def _ablation_summary() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm, directory in ABLATION_DIRS.items():
        summary_path = directory / "s52_train_meta_regime_handoff_threshold_policy_summary.csv"
        manifest_path = directory / "manifest.json"
        if not summary_path.exists():
            continue
        df = pd.read_csv(summary_path)
        best = df.iloc[0].to_dict() if not df.empty else {}
        manifest = {}
        if manifest_path.exists():
            with manifest_path.open() as f:
                manifest = json.load(f)
        rows.append(
            {
                "arm": arm,
                "selector": best.get("selector"),
                "policy_id": best.get("policy_id"),
                "budget_frac": best.get("budget_frac"),
                "mean_exec_margin": best.get("mean_exec_margin"),
                "worst_exec_margin": best.get("worst_exec_margin"),
                "mean_clean_exec_precision": best.get("mean_clean_exec_precision"),
                "mean_full_path_bad_mae": best.get("mean_full_path_bad_mae"),
                "mean_timeout": best.get("mean_timeout"),
                "mean_oracle_recall": best.get("mean_oracle_recall"),
                "mean_long_share": best.get("mean_long_share"),
                "status": best.get("threshold_policy_status"),
                "enable_base_prior_features": manifest.get("enable_base_prior_features"),
                "enable_path_order_heads": manifest.get("enable_path_order_heads"),
                "enable_path_order_blends": manifest.get("enable_path_order_blends"),
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(["mean_exec_margin", "mean_full_path_bad_mae"], ascending=[False, True])


def _write_report(
    weekly: pd.DataFrame,
    replay: pd.DataFrame,
    drift: pd.DataFrame,
    support: pd.DataFrame,
    archetypes: pd.DataFrame,
    ablations: pd.DataFrame,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    late = weekly[weekly["week"].isin(["2026-06-15", "2026-06-22"])]
    drift_late = drift[drift["week"].isin(["2026-06-15", "2026-06-22"])].copy()
    drift_late["abs_z"] = drift_late["mean_z_delta_vs_reference"].abs()
    support_late = support[support["week"].isin(["2026-06-15", "2026-06-22"])].copy()
    lines = [
        "# S52 Late-June Decay Deep Dive",
        "",
        "## Read",
        "",
        "The late-June failure is not explained by one scalar model-confidence collapse. It is a mix of lower offline clean precision, high stop-loss realization in replay, sparse candidate count on 2026-06-22, and concentration in a few fragile side/archetype cells.",
        "",
        "## Weekly Selected Meta Candidates",
        "",
        weekly.to_markdown(index=False),
        "",
        "## Replay Weekly Outcome",
        "",
        replay.to_markdown(index=False),
        "",
        "## Largest Late-June Numeric Drifts vs 2026-05-11",
        "",
        drift_late.sort_values("abs_z", ascending=False).head(25).drop(columns=["abs_z"]).to_markdown(index=False),
        "",
        "## Largest Late-June Support / Leaf Drifts vs 2026-05-11",
        "",
        support_late.sort_values("js_divergence_vs_reference", ascending=False).head(25).to_markdown(index=False),
        "",
        "## Archetypes Most Affected",
        "",
        archetypes.head(25).to_markdown(index=False),
        "",
        "## Meta Ablation Decision",
        "",
        ablations.to_markdown(index=False),
        "",
        "## Recommendations",
        "",
        "1. Do not replace the current S52 threshold handoff with the path-order/base-prior ablation yet. None of the new arms beats the promoted handoff on expected margin.",
        "2. Keep path-order heads as diagnostic/exported meta features. The path-order blend improves clean/path metrics in places but gives up too much expected margin and oracle recall.",
        "3. Use base-conditioned diagnostics as meta-layer context, not as a hard base gate. The bad late-June pockets are concentrated in specific side/archetype/base-score bands.",
        "4. Add an execution-regime meta feature family focused on stop-loss propensity: short-term adverse excursion speed, realized volatility burst, spread/liquidity stress, and time-to-first-profit proxies.",
        "5. Add a leaf/support drift guard at meta or execution layer. The report computes categorical support drift for LGBM leaf and AE/GMM bins; high drift should raise thresholds or reduce size, not block base candidates globally.",
        "6. Add archetype-specific uncertainty calibration. Clean/bad score entropy alone is insufficient; use score gap, risk pressure, and historical reliability by side x archetype x base band.",
        "7. Treat 2026-06-22 as sparse evidence. It has only eight replay trades; use it as a warning cell, not as a standalone policy refit target.",
        "",
    ]
    (OUT_DIR / "s52_june_decay_deep_dive.md").write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(CURRENT_THRESHOLD_DIR / "s52_meta_threshold_guarded_offline_eval_candidates.parquet")
    weekly = _weekly_current_handoff(frame)
    replay = _replay_weekly()
    drift = _numeric_feature_drift(frame)
    support = _categorical_support_drift(frame)
    archetypes = _archetype_replay_attribution()
    ablations = _ablation_summary()
    weekly.to_csv(OUT_DIR / "weekly_selected_meta_uncertainty.csv", index=False)
    replay.to_csv(OUT_DIR / "weekly_replay_outcomes.csv", index=False)
    drift.to_csv(OUT_DIR / "numeric_feature_drift_vs_2026_05_11.csv", index=False)
    support.to_csv(OUT_DIR / "categorical_leaf_support_drift_vs_2026_05_11.csv", index=False)
    archetypes.to_csv(OUT_DIR / "archetype_replay_attribution.csv", index=False)
    ablations.to_csv(OUT_DIR / "meta_ablation_promotion_matrix.csv", index=False)
    _write_report(weekly, replay, drift, support, archetypes, ablations)
    manifest = {
        "generated_by": "report_s52_june_decay_deep_dive",
        "current_threshold_dir": str(CURRENT_THRESHOLD_DIR),
        "replay_metrics_dir": str(REPLAY_METRICS_DIR),
        "output_dir": str(OUT_DIR),
        "reference_week": "2026-05-11",
        "weak_weeks": ["2026-06-15", "2026-06-22"],
        "files": {
            "report": str(OUT_DIR / "s52_june_decay_deep_dive.md"),
            "weekly_selected_meta_uncertainty": str(OUT_DIR / "weekly_selected_meta_uncertainty.csv"),
            "numeric_feature_drift": str(OUT_DIR / "numeric_feature_drift_vs_2026_05_11.csv"),
            "categorical_leaf_support_drift": str(OUT_DIR / "categorical_leaf_support_drift_vs_2026_05_11.csv"),
            "archetype_replay_attribution": str(OUT_DIR / "archetype_replay_attribution.csv"),
            "meta_ablation_promotion_matrix": str(OUT_DIR / "meta_ablation_promotion_matrix.csv"),
        },
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
