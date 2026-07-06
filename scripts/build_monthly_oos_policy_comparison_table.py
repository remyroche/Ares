#!/usr/bin/env python3
"""Build month-level OOS policy comparison tables with matching model OOF context."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DATA_ROOT = Path("data_perp")
MONTHLY_WF_ID = os.environ.get(
    "EPM_MONTHLY_WF_ID",
    "20260701_130000_single_head_monthly_walkforward_oos",
)
MONTHLY_REPORT = DATA_ROOT / "reports" / MONTHLY_WF_ID
SOURCE_RUN_ID = os.environ.get("EPM_SOURCE_RUN_ID", "20260629_050000_lgbm_mda")
SOURCE_ARTIFACT = DATA_ROOT / "artifacts" / SOURCE_RUN_ID
STRATEGY_ID = (
    "long_dist_ema20_atr_-0_92271453_loc_bb_channel_pos_48_0_60767579_"
    "leverage_build_score_0_45107844_return_autocorr_48_1_18643_"
    "rolling_range_20_-0_25967735"
)
TOP15_THRESHOLD = 0.85

OVERLAY_POLICIES = [
    {
        "policy": "T16_q42_weighted_guard_hr35_last7_11",
        "path": DATA_ROOT
        / "reports"
        / "20260629_050000_lgbm_mda_dynamic_hr_surprise_t16_6mo_overlay_20260630"
        / "T16_q42_weighted_guard_hr35_last7_11"
        / "calendar_dynamic_hr_surprise_selected_rows.csv",
        "scope": "source_run_four_head_overlay",
        "model_run_id": SOURCE_RUN_ID,
        "notes": "Source-run T16 overlay, no A1 prehead guard.",
    },
    {
        "policy": "A1_l4of5_24h -> T16_q42_weighted_guard_hr35_last7_11",
        "path": DATA_ROOT
        / "reports"
        / "prehead_symbol_guard_threshold_sweep_rel_disp_breadth10_20260630"
        / "A1_l4of5_24h"
        / "T16_recomputed_calendar_replay"
        / "calendar_dynamic_hr_surprise_selected_rows.csv",
        "scope": "source_run_four_head_overlay",
        "model_run_id": SOURCE_RUN_ID,
        "notes": "A1_l4of5_24h materialized prehead guard then T16 calendar replay.",
    },
]


def _monthly_strategy_id() -> str:
    summary_path = MONTHLY_REPORT / "summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            strategy_id = str(payload.get("strategy_id") or "").strip()
            if strategy_id:
                return strategy_id
        except Exception:
            pass
    return STRATEGY_ID


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def _month_label(ts: pd.Series) -> pd.Series:
    return pd.to_datetime(ts, utc=True, errors="coerce").dt.strftime("%Y-%m")


def _drawdown(values: pd.Series) -> float:
    net = pd.to_numeric(values, errors="coerce").fillna(0.0)
    if net.empty:
        return 0.0
    curve = net.cumsum()
    dd = curve - curve.cummax()
    return float(dd.min())


def _sortino(values: pd.Series) -> float:
    net = pd.to_numeric(values, errors="coerce").dropna()
    if net.empty:
        return 0.0
    downside = net[net < 0.0]
    if downside.empty:
        return 0.0
    denom = float(downside.std(ddof=0))
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0
    return float(net.mean() / denom)


def _strategy_to_head(strategy_id: str) -> str:
    if strategy_id.startswith("long_dist"):
        return "long_dist"
    if strategy_id.startswith("long_bars"):
        return "long_bars"
    if strategy_id.startswith("short_asset"):
        return "short_asset"
    if strategy_id.startswith("short_boll"):
        return "short_boll"
    return strategy_id.split("_", 1)[0]


def _oof_stats_from_parquet(path: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    score_col = next(
        (col for col in ("oof_pred", "oof_meta_clf", "clf", "oof_p_move") if col in df.columns),
        None,
    )
    label_col = next((col for col in ("y_bin", "target", "label") if col in df.columns), None)
    ret_col = "return" if "return" in df.columns else None
    ts = pd.to_datetime(df.get("timestamp"), utc=True, errors="coerce")
    out: dict[str, Any] = {
        "oof_n": int(len(df)),
        "oof_period_start": ts.min().isoformat() if len(ts.dropna()) else None,
        "oof_period_end": ts.max().isoformat() if len(ts.dropna()) else None,
        "oof_top15_hit_rate": None,
        "oof_top15_mean_return": None,
        "oof_top15_return_sum": None,
    }
    if score_col and label_col:
        score = pd.to_numeric(df[score_col], errors="coerce")
        rank = score.rank(method="max", pct=True)
        mask = rank >= TOP15_THRESHOLD
        labels = pd.to_numeric(df.loc[mask, label_col], errors="coerce")
        out["oof_top15_rows"] = int(mask.sum())
        out["oof_top15_hit_rate"] = _finite(labels.mean())
        if ret_col:
            rets = pd.to_numeric(df.loc[mask, ret_col], errors="coerce")
            out["oof_top15_mean_return"] = _finite(rets.mean())
            out["oof_top15_return_sum"] = _finite(rets.sum())
    return out


def _load_oof_stats(run_root: Path, strategy_id: str) -> dict[str, Any]:
    path = run_root / "meta_oof" / f"meta_oof_{strategy_id}_tbm_clf.parquet"
    stats = _oof_stats_from_parquet(path) if path.exists() else {}
    metrics_path = run_root / "meta_oof" / "meta_head_metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        item = metrics.get(strategy_id) or metrics.get(f"{strategy_id}_tbm_clf") or {}
        stats.update(
            {
                "oof_auc": _finite(item.get("auc")),
                "oof_pr_auc": _finite(item.get("pr_auc")),
                "oof_ic": _finite(item.get("ic")),
                "oof_base_rate": _finite(item.get("base_rate") or item.get("hit_rate")),
                "oof_precision_20": _finite(item.get("precision_20")),
                "oof_precision_10": _finite(item.get("precision_10")),
                "oof_precision_5": _finite(item.get("precision_5")),
            }
        )
    return stats


def _load_source_oof_by_head() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    metrics_path = SOURCE_ARTIFACT / "meta_oof" / "meta_head_metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    for path in sorted((SOURCE_ARTIFACT / "meta_oof").glob("meta_oof_*_tbm_clf.parquet")):
        strategy_id = path.name.removeprefix("meta_oof_").removesuffix("_tbm_clf.parquet")
        head = _strategy_to_head(strategy_id)
        stats = _oof_stats_from_parquet(path)
        item = metrics.get(strategy_id) or metrics.get(f"{strategy_id}_tbm_clf") or {}
        stats.update(
            {
                "strategy_id": strategy_id,
                "oof_auc": _finite(item.get("auc")),
                "oof_pr_auc": _finite(item.get("pr_auc")),
                "oof_ic": _finite(item.get("ic")),
                "oof_base_rate": _finite(item.get("base_rate") or item.get("hit_rate")),
                "oof_precision_20": _finite(item.get("precision_20")),
                "oof_precision_10": _finite(item.get("precision_10")),
                "oof_precision_5": _finite(item.get("precision_5")),
            }
        )
        out[head] = stats
    return out


def _weighted_oof_for_heads(head_counts: pd.Series, source_oof: dict[str, dict[str, Any]]) -> dict[str, Any]:
    weights = head_counts.astype(float)
    total = float(weights.sum())
    keys = [
        "oof_top15_hit_rate",
        "oof_top15_mean_return",
        "oof_auc",
        "oof_pr_auc",
        "oof_ic",
        "oof_base_rate",
        "oof_precision_20",
        "oof_precision_10",
        "oof_precision_5",
    ]
    out: dict[str, Any] = {
        "oof_n": 0,
        "oof_period_start": None,
        "oof_period_end": None,
    }
    periods_start: list[str] = []
    periods_end: list[str] = []
    for key in keys:
        num = 0.0
        den = 0.0
        for head, weight in weights.items():
            stats = source_oof.get(str(head), {})
            value = stats.get(key)
            if value is None:
                continue
            num += float(value) * float(weight)
            den += float(weight)
        out[key] = float(num / den) if den > 0.0 else None
    for head in weights.index:
        stats = source_oof.get(str(head), {})
        out["oof_n"] += int(stats.get("oof_n", 0) or 0)
        if stats.get("oof_period_start"):
            periods_start.append(str(stats["oof_period_start"]))
        if stats.get("oof_period_end"):
            periods_end.append(str(stats["oof_period_end"]))
    out["oof_period_start"] = min(periods_start) if periods_start else None
    out["oof_period_end"] = max(periods_end) if periods_end else None
    out["oof_head_weighting"] = "selected_trade_count"
    out["oof_selected_head_count"] = int(len(weights[weights > 0]))
    out["oof_selected_trade_count_for_weights"] = int(total)
    return out


def _monthly_single_head_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    strategy_id = _monthly_strategy_id()
    vanilla = pd.read_csv(MONTHLY_REPORT / "vanilla_walkforward_oos_summary.csv")
    vanilla = vanilla[vanilla["rank_slice"].eq("top_15")].copy()
    optuna = pd.read_csv(MONTHLY_REPORT / "summary.csv")
    fold_to_month = {
        "train_through_march_score_april": "2026-04",
        "train_through_april_score_may": "2026-05",
        "train_through_may_score_june": "2026-06",
    }
    for _, row in vanilla.iterrows():
        run_id = str(row["run_id"])
        oof = _load_oof_stats(DATA_ROOT / "artifacts" / run_id, strategy_id)
        rows.append(
            {
                "eval_month": fold_to_month.get(str(row["fold"]), ""),
                "scope": "single_head_monthly_walkforward",
                "model_run_id": run_id,
                "policy": "Vanilla top15 fixed simulate_and_score defaults",
                "oos_window_start": row["validation_start"],
                "oos_window_end": row["validation_end"],
                "candidate_rows": int(row["candidate_rows"]),
                "n_trades": int(row["n_trades"]),
                "net_pnl": _finite(row["net_pnl"]),
                "mean_net_trade": _finite(row["mean_net_trade"]),
                "hit_rate": _finite(row["hit_rate"]),
                "max_drawdown": _finite(row["max_drawdown"]),
                "sortino": _finite(row["sortino"]),
                "rank_slice": "top_15",
                "metric_type": "policy-OOS outer validation; fixed geometry",
                "notes": "No Optuna policy params, no Stage A threshold grid, no portfolio replay.",
                **oof,
            }
        )
    for _, row in optuna.iterrows():
        run_id = str(row["run_id"])
        oof = _load_oof_stats(DATA_ROOT / "artifacts" / run_id, strategy_id)
        rows.append(
            {
                "eval_month": fold_to_month.get(str(row["fold"]), ""),
                "scope": "single_head_monthly_walkforward",
                "model_run_id": run_id,
                "policy": "Optuna simple_policy only",
                "oos_window_start": row["validation_start"],
                "oos_window_end": row["validation_end"],
                "candidate_rows": None,
                "n_trades": int(row["oos_n_trades"]),
                "net_pnl": _finite(row["oos_net_pnl"]),
                "mean_net_trade": _finite(row["oos_mean_net_trade"]),
                "hit_rate": _finite(row["oos_hit_rate"]),
                "max_drawdown": _finite(row["oos_max_drawdown"]),
                "sortino": _finite(row["oos_sortino"]),
                "rank_slice": "policy threshold/export rows",
                "metric_type": "policy-OOS outer validation; Optuna exit geometry only",
                "notes": "No A1/T16 overlay and no portfolio allocation/replay in reported metric.",
                **oof,
            }
        )
    return rows


def _overlay_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_oof = _load_source_oof_by_head()
    for policy in OVERLAY_POLICIES:
        path = policy["path"]
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df = df.dropna(subset=["timestamp"]).copy()
        df["eval_month"] = _month_label(df["timestamp"])
        df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce").fillna(0.0)
        for month, group in df.groupby("eval_month"):
            head_counts = group["head"].astype(str).value_counts()
            oof = _weighted_oof_for_heads(head_counts, source_oof)
            rows.append(
                {
                    "eval_month": str(month),
                    "scope": policy["scope"],
                    "model_run_id": policy["model_run_id"],
                    "policy": policy["policy"],
                    "oos_window_start": group["timestamp"].min().isoformat(),
                    "oos_window_end": group["timestamp"].max().isoformat(),
                    "candidate_rows": None,
                    "n_trades": int(len(group)),
                    "net_pnl": _finite(group["net_return"].sum()),
                    "mean_net_trade": _finite(group["net_return"].mean()),
                    "hit_rate": _finite((group["net_return"] > 0.0).mean()),
                    "max_drawdown": _drawdown(group.sort_values("timestamp")["net_return"]),
                    "sortino": _sortino(group["net_return"]),
                    "rank_slice": "calendar dynamic selected rows",
                    "metric_type": "calendar policy-OOS overlay selected rows",
                    "notes": policy["notes"],
                    **oof,
                }
            )
    return rows


def _format_markdown(df: pd.DataFrame) -> str:
    display_cols = [
        "eval_month",
        "scope",
        "policy",
        "oos_window_start",
        "oos_window_end",
        "n_trades",
        "net_pnl",
        "mean_net_trade",
        "hit_rate",
        "max_drawdown",
        "sortino",
        "oof_period_start",
        "oof_period_end",
        "oof_top15_hit_rate",
        "oof_top15_mean_return",
        "oof_auc",
        "oof_ic",
        "metric_type",
    ]
    work = df[display_cols].copy()
    for col in [
        "net_pnl",
        "mean_net_trade",
        "hit_rate",
        "max_drawdown",
        "sortino",
        "oof_top15_hit_rate",
        "oof_top15_mean_return",
        "oof_auc",
        "oof_ic",
    ]:
        work[col] = work[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
    return work.to_markdown(index=False)


def main() -> int:
    out_dir = MONTHLY_REPORT / "policy_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(_monthly_single_head_rows() + _overlay_rows())
    if df.empty:
        raise RuntimeError("No rows built")
    df = df.sort_values(["eval_month", "scope", "policy"]).reset_index(drop=True)
    csv_path = out_dir / "monthly_oos_policy_comparison.csv"
    md_path = out_dir / "monthly_oos_policy_comparison.md"
    manifest_path = out_dir / "monthly_oos_policy_comparison_manifest.json"
    df.to_csv(csv_path, index=False)
    md = (
        "# Monthly OOS Policy Comparison\n\n"
        "Rows are policy-OOS only. OOF columns are model-OOF context for the same "
        "model run, not a policy-OOS replay metric.\n\n"
        + _format_markdown(df)
        + "\n"
    )
    md_path.write_text(md, encoding="utf-8")
    manifest = {
        "generated_by": Path(__file__).name,
        "monthly_report": str(MONTHLY_REPORT),
        "monthly_walkforward_id": MONTHLY_WF_ID,
        "source_run_id": SOURCE_RUN_ID,
        "strategy_id": _monthly_strategy_id(),
        "top15_threshold": TOP15_THRESHOLD,
        "outputs": {
            "csv": str(csv_path),
            "markdown": str(md_path),
        },
        "notes": [
            "Single-head Vanilla and Optuna rows are comparable across April-May-June monthly retrained models.",
            "A1/T16 rows use the source 20260629_050000_lgbm_mda four-head overlay and are not the same monthly retrained model.",
            "OOF top15 hit/return are label/OOF diagnostics, not identical to OOS policy execution PnL.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(csv_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
