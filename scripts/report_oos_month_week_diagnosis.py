#!/usr/bin/env python3
"""Month/week OOS PnL diagnosis for the four-head OOF gap investigation."""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import scripts.report_single_head_monthly_vanilla_walkforward_oos as vanilla
import scripts.run_single_head_monthly_walkforward_oos as wf
from extreme_price_movements import simple_policy_optimiser as spo


SOURCE_RUN_ID = os.environ.get("EPM_SOURCE_RUN_ID", "20260629_050000_lgbm_mda")
MONTHLY_WF_ID = os.environ.get(
    "EPM_MONTHLY_WF_ID",
    "20260701_130000_single_head_monthly_walkforward_oos",
)
BASE_REPORT_DIR = ROOT / "data_perp" / "reports" / f"{SOURCE_RUN_ID}_four_head_oof_policy_gap"
OUT_DIR = Path(
    os.environ.get(
        "EPM_OOS_DIAG_OUTPUT_DIR",
        str(BASE_REPORT_DIR / "oos_month_week_diagnosis"),
    )
)
SOURCE_ROOT = ROOT / "data_perp" / "artifacts" / SOURCE_RUN_ID
MONTHLY_ROOT = ROOT / "data_perp" / "reports" / MONTHLY_WF_ID

OOS_WINDOWS = {
    "2026-04": (pd.Timestamp("2026-04-16", tz="UTC"), pd.Timestamp("2026-05-01", tz="UTC")),
    "2026-05": (pd.Timestamp("2026-05-16", tz="UTC"), pd.Timestamp("2026-06-01", tz="UTC")),
    "2026-06": (pd.Timestamp("2026-06-16", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")),
}

FOLD_TO_MONTH = {
    "train_through_march_score_april": "2026-04",
    "train_through_april_score_may": "2026-05",
    "train_through_may_score_june": "2026-06",
}

FOCUS_OVERLAY_POLICIES = {
    "T16_q42_weighted_guard_hr35_last7_11",
    "A1_l4of5_24h -> T16_q42_weighted_guard_hr35_last7_11",
    "A1_loss_cooldown_3of4_24h -> T16_q42_weighted_guard_hr35_last7_11",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _sign(value: float) -> str:
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "flat"


def _week_bounds(month: str, ts: pd.Series) -> pd.DataFrame:
    start, end = OOS_WINDOWS[month]
    dt = pd.to_datetime(ts, utc=True)
    idx = np.floor((dt - start).dt.total_seconds() / (7 * 24 * 3600)).astype(int) + 1
    week_start = start + pd.to_timedelta(idx - 1, unit="W")
    week_end = week_start + pd.Timedelta(days=7)
    week_end = pd.Series(week_end).where(pd.Series(week_end) < end, end)
    return pd.DataFrame(
        {
            "week_index": idx,
            "week_start": week_start.to_numpy(),
            "week_end": pd.to_datetime(week_end, utc=True).to_numpy(),
        },
        index=ts.index,
    )


def _summarize_trade_rows(
    frame: pd.DataFrame,
    *,
    group_cols: list[str],
    pnl_col: str,
    return_col: str | None = None,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for keys, group in frame.groupby(group_cols, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        pnl = pd.to_numeric(group[pnl_col], errors="coerce").fillna(0.0)
        ret_source = pd.to_numeric(group[return_col], errors="coerce") if return_col and return_col in group.columns else pnl
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "n_trades": int(len(group)),
                "net_pnl": float(pnl.sum()),
                "mean_net": float(pnl.mean()) if len(pnl) else 0.0,
                "hit_rate": float((ret_source > 0).mean()) if len(ret_source) else float("nan"),
            }
        )
        row["sign"] = _sign(row["net_pnl"])
        rows.append(row)
    return pd.DataFrame(rows)


def reconstruct_vanilla_top15_trades() -> pd.DataFrame:
    os.environ.setdefault("EPM_EXCHANGE", "krakenfutures")
    os.environ.setdefault("EPM_SIMPLE_POLICY_15M_DOWNLOAD", "0")
    os.environ.setdefault("EPM_SIMPLE_POLICY_1M_DOWNLOAD", "0")
    os.environ.setdefault("MPLCONFIGDIR", str(wf.ROOT / ".mplconfig"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

    summary = json.loads((MONTHLY_ROOT / "summary.json").read_text(encoding="utf-8"))
    strategy_id = str(summary["strategy_id"])
    ds = spo._make_policy_replay_store(str(wf.DATA_ROOT), "perps")
    trades: list[pd.DataFrame] = []
    for fold in wf._folds(MONTHLY_WF_ID):
        month = FOLD_TO_MONTH[fold.name]
        df_all, split_info = vanilla._prepare_policy_frame(fold.run_id, strategy_id)
        all_paths = spo._fetch_policy_paths(df_all, ds)
        df_all, all_paths = spo._apply_delayed_entry_execution_model(
            df_all,
            all_paths,
            data_root=str(wf.DATA_ROOT),
            market_mode="perps",
        )
        validation_idx = np.flatnonzero(split_info["validation_mask"])
        validation_df = df_all.iloc[validation_idx].copy().reset_index(drop=True)
        validation_paths = spo._path_take(all_paths, validation_idx)
        rank_idx = np.flatnonzero(validation_df["rank_pct"].to_numpy(dtype=np.float32) >= 0.85)
        rank_rows = validation_df.iloc[rank_idx].copy().reset_index(drop=True)
        rank_paths = spo._path_take(validation_paths, rank_idx)
        metrics = spo.simulate_and_score(
            rank_rows,
            *rank_paths,
            cost_pct=spo.DEFAULT_POLICY_PER_SIDE_COST_PCT,
            size_power=1.0,
            market_mode="perps",
            max_concurrent_trades=spo.MAX_CONCURRENT_TRADES,
            max_concurrent_per_asset=spo.DEPLOYMENT_MAX_CONCURRENT_PER_ASSET,
        )
        selected_mask = np.asarray(metrics.get("selected_mask", []), dtype=bool)
        if selected_mask.size != len(rank_rows):
            raise RuntimeError(f"selected_mask mismatch for {fold.run_id}: {selected_mask.size} vs {len(rank_rows)}")
        selected = rank_rows.iloc[np.flatnonzero(selected_mask)].copy().reset_index(drop=True)
        raw_gains = np.asarray(metrics.get("raw_gains", []), dtype=np.float64)
        if len(raw_gains) != len(selected):
            raise RuntimeError(f"raw_gains mismatch for {fold.run_id}: {len(raw_gains)} vs {len(selected)}")
        selected["net_pnl"] = raw_gains
        selected["exit_reason"] = np.asarray(metrics.get("exit_reason", []), dtype=object)
        selected["eval_month"] = month
        selected["policy"] = "single_head_long_dist_vanilla_top15"
        wb = _week_bounds(month, selected["timestamp"])
        selected = pd.concat([selected, wb.reset_index(drop=True)], axis=1)
        trades.append(selected)
    return pd.concat(trades, ignore_index=True) if trades else pd.DataFrame()


def _policy_name_from_selected_rows_path(path: Path) -> str:
    parts = path.parts
    if "20260629_050000_lgbm_mda_dynamic_hr_surprise_t16_6mo_overlay_20260630" in parts:
        return "T16_q42_weighted_guard_hr35_last7_11"
    parent = path.parent.parent.name
    grandparent = path.parent.parent.parent.name
    if grandparent in {
        "rank_failure_guard_ablation_20260630",
        "prehead_symbol_guard_ablation_20260630",
        "prehead_symbol_guard_threshold_sweep_rel_disp_breadth10_20260630",
    }:
        return f"{parent} -> T16_q42_weighted_guard_hr35_last7_11"
    return " -> ".join(path.parts[-4:-1])


def load_exact_overlay_rows() -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted((ROOT / "data_perp" / "reports").glob("**/calendar_dynamic_hr_surprise_selected_rows.csv")):
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "timestamp" not in df.columns or "net_return" not in df.columns:
            continue
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce")
        df = df.dropna(subset=["timestamp", "net_return"]).copy()
        if df.empty:
            continue
        policy = _policy_name_from_selected_rows_path(path)
        for month, (start, end) in OOS_WINDOWS.items():
            sub = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()
            if sub.empty:
                continue
            sub["eval_month"] = month
            sub["policy"] = policy
            sub["net_pnl"] = sub["net_return"]
            wb = _week_bounds(month, sub["timestamp"])
            sub = pd.concat([sub.reset_index(drop=True), wb.reset_index(drop=True)], axis=1)
            rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def load_source_candidate_rows() -> pd.DataFrame:
    path = SOURCE_ROOT / "simple_policy_optimiser" / "simple_policy_candidates_broad.parquet"
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce")
    df = df.dropna(subset=["timestamp", "net_return"]).copy()
    rows = []
    for month, (start, end) in OOS_WINDOWS.items():
        sub = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()
        if sub.empty:
            continue
        sub["eval_month"] = month
        sub["net_pnl"] = sub["net_return"]
        sub["rank_slice"] = np.where(pd.to_numeric(sub["rank_pct"], errors="coerce") >= 0.85, "top15", "all_only")
        wb = _week_bounds(month, sub["timestamp"])
        sub = pd.concat([sub.reset_index(drop=True), wb.reset_index(drop=True)], axis=1)
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def write_report(
    clean_monthly: pd.DataFrame,
    clean_weekly: pd.DataFrame,
    optuna_monthly: pd.DataFrame,
    overlay_monthly_focus: pd.DataFrame,
    overlay_weekly_focus: pd.DataFrame,
    overlay_weekly_policy_stats: pd.DataFrame,
    candidate_monthly: pd.DataFrame,
    candidate_weekly: pd.DataFrame,
) -> None:
    lines = [
        "# OOS Month/Week PnL Diagnosis",
        "",
        "## Clean Policy-OOS: Single-Head Long Dist",
        "",
        "This is the clean Apr-May-Jun policy-OOS walk-forward artifact, reconstructed from per-trade vanilla top15 simulator rows. It covers only the selected `long_dist` head, not the four-head portfolio.",
        "",
        "### Month By Month: Reconstructed Vanilla Top15",
        "",
        clean_monthly.to_markdown(index=False),
        "",
        "### Week By Week: Reconstructed Vanilla Top15",
        "",
        clean_weekly.to_markdown(index=False),
        "",
        "### Optuna Month-Level Only",
        "",
        "The Optuna deployment validation artifact stores aggregate validation metrics, not per-trade rows, so week-level Optuna PnL is not recoverable from the saved report without rerunning/materializing that replay.",
        "",
        optuna_monthly.to_markdown(index=False),
        "",
        "## Four-Head Source Overlay: Exact Held-Out Windows",
        "",
        "These rows use selected-row overlay artifacts filtered to exact OOS windows. They are useful directional evidence but are not the same protocol as the clean single-head walk-forward.",
        "",
        "### Focus Policies Month By Month",
        "",
        overlay_monthly_focus.to_markdown(index=False),
        "",
        "### Focus Policies Week By Week",
        "",
        overlay_weekly_focus.to_markdown(index=False),
        "",
        "### Overlay Policy Positivity By Week",
        "",
        overlay_weekly_policy_stats.to_markdown(index=False),
        "",
        "## Source Candidate Broad Rows",
        "",
        "These are broad source-run candidates before selected-row overlay policy selection. They are not deployable portfolio PnL, but they test whether raw executable candidate economics are universally broken.",
        "",
        "### Month By Month",
        "",
        candidate_monthly.to_markdown(index=False),
        "",
        "### Week By Week",
        "",
        candidate_weekly.to_markdown(index=False),
        "",
        "## Diagnosis",
        "",
        "- Clean single-head policy-OOS is negative in April, May, and June. That means this is not only a late-June difficult-market issue for the selected `long_dist` policy.",
        "- The four-head source overlay is strongly positive in exact May but negative in exact June across all scanned overlay policies. That supports a recent regime/window sensitivity issue at the overlay/portfolio layer.",
        "- Broad source candidates are positive in exact May and exact June, including top15. That argues against the claim that OOF returns never translate after executable policy simulation at the candidate level.",
        "- The confirmed gap is from OOF/model evidence to selected executable policy-OOS: ranking/label quality remains high, but the deployed selection/exit/concurrency layer can turn it negative, especially for the clean `long_dist` walk-forward and the late-June selected overlays.",
        "- We still do not have a clean four-head Apr-May-Jun walk-forward replay. So the evidence does not prove all four models never fare well OOS; it proves the current clean one-head OOS is negative and the source-run four-head selected overlay is not robust in the exact June holdout.",
        "",
    ]
    (OUT_DIR / "oos_month_week_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly_comparison = pd.read_csv(MONTHLY_ROOT / "policy_comparison" / "monthly_oos_policy_comparison.csv")

    optuna_monthly = monthly_comparison[
        monthly_comparison["scope"].eq("single_head_monthly_walkforward")
        & monthly_comparison["policy"].eq("Optuna simple_policy only")
    ][
        ["eval_month", "policy", "oos_window_start", "oos_window_end", "n_trades", "net_pnl", "mean_net_trade", "hit_rate", "oof_top15_hit_rate", "oof_top15_mean_return"]
    ].copy()
    optuna_monthly["sign"] = optuna_monthly["net_pnl"].map(_sign)

    vanilla_trades = reconstruct_vanilla_top15_trades()
    clean_monthly = _summarize_trade_rows(
        vanilla_trades,
        group_cols=["eval_month", "policy"],
        pnl_col="net_pnl",
    ).sort_values(["eval_month", "policy"])
    clean_monthly["oos_window_start"] = clean_monthly["eval_month"].map(
        {k: v[0].isoformat() for k, v in OOS_WINDOWS.items()}
    )
    clean_monthly["oos_window_end"] = clean_monthly["eval_month"].map(
        {k: v[1].isoformat() for k, v in OOS_WINDOWS.items()}
    )
    clean_monthly = clean_monthly[
        ["eval_month", "policy", "oos_window_start", "oos_window_end", "n_trades", "net_pnl", "mean_net", "hit_rate", "sign"]
    ]
    clean_weekly = _summarize_trade_rows(
        vanilla_trades,
        group_cols=["eval_month", "week_index", "week_start", "week_end", "policy"],
        pnl_col="net_pnl",
    ).sort_values(["eval_month", "week_index"])

    overlay_rows = load_exact_overlay_rows()
    overlay_monthly = _summarize_trade_rows(
        overlay_rows,
        group_cols=["eval_month", "policy"],
        pnl_col="net_pnl",
        return_col="net_return",
    )
    overlay_monthly_focus = overlay_monthly[
        overlay_monthly["policy"].isin(FOCUS_OVERLAY_POLICIES)
    ].sort_values(["eval_month", "policy"])
    overlay_weekly = _summarize_trade_rows(
        overlay_rows,
        group_cols=["eval_month", "week_index", "week_start", "week_end", "policy"],
        pnl_col="net_pnl",
        return_col="net_return",
    )
    overlay_weekly_focus = overlay_weekly[
        overlay_weekly["policy"].isin(FOCUS_OVERLAY_POLICIES)
    ].sort_values(["eval_month", "policy", "week_index"])
    overlay_weekly_policy_stats = overlay_weekly.groupby(
        ["eval_month", "week_index", "week_start", "week_end"], observed=True
    ).agg(
        policies=("policy", "nunique"),
        positive_policies=("net_pnl", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        best_net_pnl=("net_pnl", "max"),
        worst_net_pnl=("net_pnl", "min"),
    ).reset_index()
    overlay_weekly_policy_stats["sign_consensus"] = np.where(
        overlay_weekly_policy_stats["positive_policies"] == overlay_weekly_policy_stats["policies"],
        "all_positive",
        np.where(overlay_weekly_policy_stats["positive_policies"] == 0, "all_negative", "mixed"),
    )

    candidate_rows = load_source_candidate_rows()
    candidate_rows_top15 = candidate_rows[pd.to_numeric(candidate_rows["rank_pct"], errors="coerce") >= 0.85].copy()
    candidate_monthly_all = _summarize_trade_rows(
        candidate_rows,
        group_cols=["eval_month"],
        pnl_col="net_pnl",
        return_col="net_return",
    )
    candidate_monthly_all["rank_slice"] = "all"
    candidate_monthly_top15 = _summarize_trade_rows(
        candidate_rows_top15,
        group_cols=["eval_month"],
        pnl_col="net_pnl",
        return_col="net_return",
    )
    candidate_monthly_top15["rank_slice"] = "top15"
    candidate_monthly = pd.concat([candidate_monthly_all, candidate_monthly_top15], ignore_index=True).sort_values(["eval_month", "rank_slice"])
    candidate_weekly_all = _summarize_trade_rows(
        candidate_rows,
        group_cols=["eval_month", "week_index", "week_start", "week_end"],
        pnl_col="net_pnl",
        return_col="net_return",
    )
    candidate_weekly_all["rank_slice"] = "all"
    candidate_weekly_top15 = _summarize_trade_rows(
        candidate_rows_top15,
        group_cols=["eval_month", "week_index", "week_start", "week_end"],
        pnl_col="net_pnl",
        return_col="net_return",
    )
    candidate_weekly_top15["rank_slice"] = "top15"
    candidate_weekly = pd.concat([candidate_weekly_all, candidate_weekly_top15], ignore_index=True).sort_values(["eval_month", "week_index", "rank_slice"])

    outputs = {
        "clean_single_head_monthly": OUT_DIR / "clean_single_head_monthly.csv",
        "clean_single_head_vanilla_weekly": OUT_DIR / "clean_single_head_vanilla_weekly.csv",
        "clean_single_head_vanilla_trades": OUT_DIR / "clean_single_head_vanilla_trades.csv",
        "optuna_single_head_monthly": OUT_DIR / "optuna_single_head_monthly.csv",
        "overlay_monthly_focus": OUT_DIR / "overlay_monthly_focus.csv",
        "overlay_weekly_focus": OUT_DIR / "overlay_weekly_focus.csv",
        "overlay_weekly_policy_stats": OUT_DIR / "overlay_weekly_policy_stats.csv",
        "source_candidate_monthly": OUT_DIR / "source_candidate_monthly.csv",
        "source_candidate_weekly": OUT_DIR / "source_candidate_weekly.csv",
        "report": OUT_DIR / "oos_month_week_diagnosis.md",
    }
    clean_monthly.to_csv(outputs["clean_single_head_monthly"], index=False)
    clean_weekly.to_csv(outputs["clean_single_head_vanilla_weekly"], index=False)
    vanilla_trades.to_csv(outputs["clean_single_head_vanilla_trades"], index=False)
    optuna_monthly.to_csv(outputs["optuna_single_head_monthly"], index=False)
    overlay_monthly_focus.to_csv(outputs["overlay_monthly_focus"], index=False)
    overlay_weekly_focus.to_csv(outputs["overlay_weekly_focus"], index=False)
    overlay_weekly_policy_stats.to_csv(outputs["overlay_weekly_policy_stats"], index=False)
    candidate_monthly.to_csv(outputs["source_candidate_monthly"], index=False)
    candidate_weekly.to_csv(outputs["source_candidate_weekly"], index=False)
    write_report(
        clean_monthly,
        clean_weekly,
        optuna_monthly,
        overlay_monthly_focus,
        overlay_weekly_focus,
        overlay_weekly_policy_stats,
        candidate_monthly,
        candidate_weekly,
    )
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "generated_by": Path(__file__).name,
                    "source_run_id": SOURCE_RUN_ID,
                    "monthly_walkforward_id": MONTHLY_WF_ID,
                    "outputs": {k: str(v) for k, v in outputs.items()},
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    print(outputs["report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
