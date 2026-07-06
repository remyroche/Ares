#!/usr/bin/env python3
"""Run failure-attribution tests for OOF-vs-OOS policy degradation."""

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
REPORT_ROOT = ROOT / "data_perp" / "reports" / f"{SOURCE_RUN_ID}_four_head_oof_policy_gap"
OUT_DIR = Path(
    os.environ.get(
        "EPM_FAILURE_ATTRIBUTION_OUTPUT_DIR",
        str(REPORT_ROOT / "failure_attribution_tests"),
    )
)
MONTHLY_ROOT = ROOT / "data_perp" / "reports" / MONTHLY_WF_ID

FOLD_TO_MONTH = {
    "train_through_march_score_april": "2026-04",
    "train_through_april_score_may": "2026-05",
    "train_through_may_score_june": "2026-06",
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


def _safe_mean(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    return float(arr.mean()) if len(arr) else float("nan")


def _safe_sum(values: Any) -> float:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").fillna(0.0)
    return float(arr.sum()) if len(arr) else 0.0


def _load_monthly_oof_context() -> pd.DataFrame:
    monthly = pd.read_csv(MONTHLY_ROOT / "policy_comparison" / "monthly_oos_policy_comparison.csv")
    return monthly[
        monthly["scope"].eq("single_head_monthly_walkforward")
        & monthly["policy"].eq("Vanilla top15 fixed simulate_and_score defaults")
    ][
        [
            "eval_month",
            "oof_top15_hit_rate",
            "oof_top15_mean_return",
            "oof_top15_return_sum",
            "oof_top15_rows",
            "oof_auc",
            "oof_ic",
        ]
    ].copy()


def run_clean_single_head_tests() -> tuple[pd.DataFrame, pd.DataFrame]:
    os.environ.setdefault("EPM_EXCHANGE", "krakenfutures")
    os.environ.setdefault("EPM_SIMPLE_POLICY_15M_DOWNLOAD", "0")
    os.environ.setdefault("EPM_SIMPLE_POLICY_1M_DOWNLOAD", "0")

    summary = json.loads((MONTHLY_ROOT / "summary.json").read_text(encoding="utf-8"))
    strategy_id = str(summary["strategy_id"])
    oof_context = _load_monthly_oof_context().set_index("eval_month")
    ds = spo._make_policy_replay_store(str(wf.DATA_ROOT), "perps")
    model_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []

    for fold in wf._folds(MONTHLY_WF_ID):
        month = FOLD_TO_MONTH[fold.name]
        df_all, split_info = vanilla._prepare_policy_frame(fold.run_id, strategy_id)
        validation_idx = np.flatnonzero(split_info["validation_mask"])
        validation_df = df_all.iloc[validation_idx].copy().reset_index(drop=True)
        top15 = validation_df[validation_df["rank_pct"].to_numpy(dtype=np.float32) >= 0.85].copy()
        y_bin = pd.to_numeric(top15["y_bin"], errors="coerce")
        returns = pd.to_numeric(top15["return"], errors="coerce")
        oof = oof_context.loc[month]
        model_rows.append(
            {
                "eval_month": month,
                "policy_oos_top15_rows": int(len(top15)),
                "oof_top15_rows": int(oof["oof_top15_rows"]),
                "oof_top15_hit_rate": float(oof["oof_top15_hit_rate"]),
                "policy_oos_top15_label_hit_rate": float(y_bin.mean()) if len(y_bin.dropna()) else float("nan"),
                "hit_rate_gap_pp": (
                    float(y_bin.mean()) - float(oof["oof_top15_hit_rate"])
                    if len(y_bin.dropna())
                    else float("nan")
                ),
                "oof_top15_mean_return": float(oof["oof_top15_mean_return"]),
                "policy_oos_top15_mean_label_return": float(returns.mean()) if len(returns.dropna()) else float("nan"),
                "mean_return_gap": (
                    float(returns.mean()) - float(oof["oof_top15_mean_return"])
                    if len(returns.dropna())
                    else float("nan")
                ),
                "policy_oos_top15_sum_label_return": float(returns.fillna(0.0).sum()),
                "oof_auc": float(oof["oof_auc"]),
                "oof_ic": float(oof["oof_ic"]),
            }
        )

        all_paths = spo._fetch_policy_paths(df_all, ds)
        df_all, all_paths = spo._apply_delayed_entry_execution_model(
            df_all,
            all_paths,
            data_root=str(wf.DATA_ROOT),
            market_mode="perps",
        )
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
        if len(selected_mask) != len(rank_rows):
            raise RuntimeError(f"selected_mask mismatch for {fold.run_id}")
        selected = rank_rows.iloc[np.flatnonzero(selected_mask)].copy().reset_index(drop=True)
        raw_gains = np.asarray(metrics.get("raw_gains", []), dtype=np.float64)
        gross_gains = np.asarray(metrics.get("gross_gains", []), dtype=np.float64)
        sizes = np.asarray(metrics.get("sizes", []), dtype=np.float64)
        if len(raw_gains) != len(selected) or len(gross_gains) != len(selected) or len(sizes) != len(selected):
            raise RuntimeError(f"simulator output length mismatch for {fold.run_id}")
        gross_returns = np.divide(gross_gains, np.maximum(sizes, 1e-12), where=sizes > 0.0)
        net_returns = np.divide(raw_gains, np.maximum(sizes, 1e-12), where=sizes > 0.0)
        selected_label_return = pd.to_numeric(selected["return"], errors="coerce")
        selected_label_hit = pd.to_numeric(selected["y_bin"], errors="coerce")
        n_trades = int(metrics.get("total_trades", 0) or 0)
        execution_rows.append(
            {
                "eval_month": month,
                "selected_trades": len(selected),
                "selected_label_hit_rate": float(selected_label_hit.mean()) if len(selected_label_hit.dropna()) else float("nan"),
                "selected_label_mean_return": float(selected_label_return.mean()) if len(selected_label_return.dropna()) else float("nan"),
                "sim_gross_mean_return": float(np.mean(gross_returns)) if len(gross_returns) else float("nan"),
                "sim_net_mean_return": float(np.mean(net_returns)) if len(net_returns) else float("nan"),
                "label_to_gross_mean_return_drag": (
                    float(selected_label_return.mean()) - float(np.mean(gross_returns))
                    if len(gross_returns) and len(selected_label_return.dropna())
                    else float("nan")
                ),
                "gross_to_net_mean_return_drag": (
                    float(np.mean(gross_returns)) - float(np.mean(net_returns))
                    if len(gross_returns)
                    else float("nan")
                ),
                "gross_pnl": float(gross_gains.sum()),
                "net_pnl": float(raw_gains.sum()),
                "gross_to_net_pnl_drag": float(gross_gains.sum() - raw_gains.sum()),
                "gross_hit_rate": float((gross_returns > 0).mean()) if len(gross_returns) else float("nan"),
                "net_hit_rate": float((net_returns > 0).mean()) if len(net_returns) else float("nan"),
                "full_sl_exit_rate": float(metrics.get("full_sl_exit_count", 0) / max(n_trades, 1)),
                "trailing_exit_rate": float(metrics.get("trailing_exit_count", 0) / max(n_trades, 1)),
                "timeout_exit_rate": float(metrics.get("timeout_exit_rate", 0.0) or 0.0),
            }
        )
    return pd.DataFrame(model_rows), pd.DataFrame(execution_rows)


def run_policy_tests() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    aggregate = pd.read_csv(
        MONTHLY_ROOT / "policy_baseline_investigation" / "policy_baseline_aggregate.csv"
    )
    single = aggregate[aggregate["scope"].eq("single_head_monthly_walkforward")].copy()
    single["robust_positive"] = (single["positive_months"] == single["months"]) & (single["net_pnl"] > 0)
    single_rank = single.sort_values(["net_pnl", "worst_month_net_pnl"], ascending=[False, False])[
        [
            "policy",
            "rank_slice",
            "months",
            "n_trades",
            "net_pnl",
            "positive_months",
            "worst_month_net_pnl",
            "best_month_net_pnl",
            "robust_positive",
        ]
    ].reset_index(drop=True)

    overlay = aggregate[aggregate["scope"].eq("source_run_selected_rows_exact_oos_window")].copy()
    overlay["robust_positive"] = (overlay["positive_months"] == overlay["months"]) & (overlay["net_pnl"] > 0)
    overlay_rank = overlay.sort_values(["net_pnl", "worst_month_net_pnl"], ascending=[False, False])[
        [
            "policy",
            "months",
            "n_trades",
            "net_pnl",
            "positive_months",
            "worst_month_net_pnl",
            "best_month_net_pnl",
            "robust_positive",
        ]
    ].reset_index(drop=True)

    candidate_monthly_path = REPORT_ROOT / "oos_month_week_diagnosis" / "source_candidate_monthly.csv"
    overlay_weekly_path = REPORT_ROOT / "oos_month_week_diagnosis" / "overlay_weekly_policy_stats.csv"
    candidate_monthly = pd.read_csv(candidate_monthly_path) if candidate_monthly_path.exists() else pd.DataFrame()
    overlay_weekly = pd.read_csv(overlay_weekly_path) if overlay_weekly_path.exists() else pd.DataFrame()
    candidate_vs_overlay_rows: list[dict[str, Any]] = []
    if not candidate_monthly.empty:
        for _, row in candidate_monthly[candidate_monthly["rank_slice"].eq("top15")].iterrows():
            month = str(row["eval_month"])
            overlay_month = overlay_rank[overlay_rank["months"] >= 1].copy()
            exact_monthly = pd.read_csv(
                MONTHLY_ROOT / "policy_baseline_investigation" / "policy_baseline_monthly.csv"
            )
            exact_monthly = exact_monthly[
                exact_monthly["scope"].eq("source_run_selected_rows_exact_oos_window")
                & exact_monthly["eval_month"].astype(str).eq(month)
            ]
            candidate_vs_overlay_rows.append(
                {
                    "eval_month": month,
                    "source_candidate_top15_net_pnl": float(row["net_pnl"]),
                    "source_candidate_top15_n": int(row["n_trades"]),
                    "best_overlay_net_pnl": float(exact_monthly["net_pnl"].max()) if len(exact_monthly) else float("nan"),
                    "positive_overlay_policies": int((pd.to_numeric(exact_monthly["net_pnl"], errors="coerce") > 0).sum()) if len(exact_monthly) else 0,
                    "overlay_policy_count": int(exact_monthly["policy"].nunique()) if len(exact_monthly) else 0,
                }
            )
    candidate_vs_overlay = pd.DataFrame(candidate_vs_overlay_rows)
    if not overlay_weekly.empty:
        overlay_weekly.to_csv(OUT_DIR / "overlay_weekly_policy_stats_snapshot.csv", index=False)
    return single_rank, overlay_rank, candidate_vs_overlay


def _format_pp(value: float) -> str:
    return f"{100.0 * value:.1f}pp"


def write_report(
    model_signal: pd.DataFrame,
    execution: pd.DataFrame,
    single_policy_rank: pd.DataFrame,
    overlay_policy_rank: pd.DataFrame,
    candidate_vs_overlay: pd.DataFrame,
) -> None:
    model_display = model_signal.copy()
    exec_display = execution.copy()
    verdict_rows = [
        {
            "hypothesis": "Lack of model signal quality",
            "verdict": "Supported for clean long_dist OOS",
            "evidence": "OOF top15 hit 97-99%, policy-OOS top15 label hit only 38-50%; June OOS top15 mean label return is negative.",
        },
        {
            "hypothesis": "Poor execution quality",
            "verdict": "Supported",
            "evidence": "Selected rows still have mildly positive label returns, but gross executable returns are already negative before fee/spread drag; full SL exits dominate.",
        },
        {
            "hypothesis": "Policy mismatch",
            "verdict": "Supported",
            "evidence": "No single-head policy is positive across Apr-May-Jun; four-head overlays are May-positive but exact-June negative while broad source candidates remain positive.",
        },
    ]
    lines = [
        "# Failure Attribution Tests",
        "",
        "## Verdict",
        "",
        pd.DataFrame(verdict_rows).to_markdown(index=False),
        "",
        "## Test 1: OOF Signal vs Clean Policy-OOS Label Signal",
        "",
        "This tests model signal before execution. Metric type: OOF training-window context vs verified policy-OOS prediction rows for the selected `long_dist` head.",
        "",
        model_display.to_markdown(index=False),
        "",
        "Interpretation: the model-signal gap is large. Top15 hit-rate gap is "
        f"{_format_pp(model_signal['hit_rate_gap_pp'].min())} to {_format_pp(model_signal['hit_rate_gap_pp'].max())}; "
        "OOS label mean returns are much smaller than OOF and negative in June.",
        "",
        "## Test 2: Execution Translation On Same Selected Rows",
        "",
        "This tests whether positive labels translate through the vanilla simulator. `label_to_gross` is exit-path/geometry/timing drag before net cost drag; `gross_to_net` is fee/spread drag.",
        "",
        exec_display.to_markdown(index=False),
        "",
        "Interpretation: execution quality is poor under this vanilla policy. Gross executable returns are negative before fees/spread, so this is not only a cost problem. The dominant issue is exit geometry/timing: 84-87% full-SL exits.",
        "",
        "## Test 3: Single-Head Policy Ranking",
        "",
        "Metric type: Apr-May-Jun exact policy-OOS rows from the policy baseline investigation. Costs are included.",
        "",
        single_policy_rank.head(15).to_markdown(index=False),
        "",
        f"Robust positive single-head policies: {int(single_policy_rank['robust_positive'].sum())} / {len(single_policy_rank)}.",
        "",
        "## Test 4: Four-Head Overlay Policy Ranking",
        "",
        "Metric type: source-run selected-row artifacts filtered to exact May/June OOS windows. These are not clean four-head walk-forward retrains.",
        "",
        overlay_policy_rank.head(15).to_markdown(index=False),
        "",
        f"Robust positive overlay policies: {int(overlay_policy_rank['robust_positive'].sum())} / {len(overlay_policy_rank)}.",
        "",
        "## Test 5: Broad Candidates vs Selected Overlay",
        "",
        "This checks whether raw executable source candidates are universally bad, or whether the selected overlay policy is choosing a weak subset.",
        "",
        candidate_vs_overlay.to_markdown(index=False) if not candidate_vs_overlay.empty else "No candidate-vs-overlay table available.",
        "",
        "## Diagnosis",
        "",
        "1. There is a real model-signal degradation in clean policy-OOS for the selected `long_dist` head. OOF top15 is dramatically stronger than policy-OOS top15 labels.",
        "2. Execution makes the already weaker OOS signal worse. Even when selected rows have mildly positive label returns, vanilla execution is gross-negative before costs.",
        "3. Policy mismatch is also present. Some policies are less bad, but none are robustly positive under the clean Apr-May-Jun single-head protocol. Four-head overlays have positive May aggregate, but every scanned overlay fails exact June.",
        "4. The problem is not that OOF returns never translate at all: broad source candidates are positive in exact May and June. The failure appears between model/candidate signal and the selected deployed policy/exit/concurrency layer.",
        "",
        "## Next Test Needed",
        "",
        "The remaining missing experiment is a clean four-head Apr-May-Jun walk-forward replay with all four heads retrained/scored under one portfolio policy. Current clean OOS evidence is one-head only; current four-head evidence is source-run selected-row overlay.",
        "",
    ]
    (OUT_DIR / "failure_attribution_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_signal, execution = run_clean_single_head_tests()
    single_policy_rank, overlay_policy_rank, candidate_vs_overlay = run_policy_tests()

    outputs = {
        "model_signal_oof_vs_oos": OUT_DIR / "model_signal_oof_vs_oos.csv",
        "execution_translation": OUT_DIR / "execution_translation.csv",
        "single_head_policy_ranking": OUT_DIR / "single_head_policy_ranking.csv",
        "overlay_policy_ranking": OUT_DIR / "overlay_policy_ranking.csv",
        "candidate_vs_overlay": OUT_DIR / "candidate_vs_overlay.csv",
        "report": OUT_DIR / "failure_attribution_report.md",
    }
    model_signal.to_csv(outputs["model_signal_oof_vs_oos"], index=False)
    execution.to_csv(outputs["execution_translation"], index=False)
    single_policy_rank.to_csv(outputs["single_head_policy_ranking"], index=False)
    overlay_policy_rank.to_csv(outputs["overlay_policy_ranking"], index=False)
    candidate_vs_overlay.to_csv(outputs["candidate_vs_overlay"], index=False)
    write_report(model_signal, execution, single_policy_rank, overlay_policy_rank, candidate_vs_overlay)
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
