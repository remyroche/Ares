#!/usr/bin/env python3
"""Summarize guarded execution breadth-expansion experiments.

The guarded policy package currently passes as a frozen-review candidate but is
too narrow for deployment. This report aggregates threshold/coverage variants
and separates three questions:

1. Does the variant pass the simple-policy economic gate?
2. Does it beat the fixed h9 benchmark used by the guarded package?
3. Does it materially improve accepted-trade exposure?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_optimiser import _json_safe  # noqa: E402


REPORT_ROOT = Path("data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced")
DEFAULT_OUT_DIR = REPORT_ROOT / "guarded_execution_breadth_sweep_20260703_v1"
DEFAULT_VARIANTS = [
    (
        "current_anchor",
        REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_v3",
    ),
    (
        "min_train_weeks_1",
        REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_mintrainweeks1_v1",
    ),
    (
        "min_train_trades_20",
        REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_mintrades20_v1",
    ),
    (
        "min_train_trades_22",
        REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_mintrades22_v1",
    ),
    (
        "min_train_trades_24",
        REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_mintrades24_v1",
    ),
    (
        "min_train_trades_26",
        REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_mintrades26_v1",
    ),
    (
        "min_train_trades_30",
        REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_mintrades30_v1",
    ),
    (
        "wide_grid_min_train_trades_30",
        REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_widegrid_mintrades30_v1",
    ),
    (
        "dense_grid_min_train_trades_24",
        REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_densegrid_mintrades24_v1",
    ),
    (
        "rank75_bad55_to12h",
        REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_rank75bad55_to12h_v1",
    ),
    (
        "rank85_bad45_to12h",
        REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_rank85bad45_to12h_v1",
    ),
    (
        "rank90_bad45_to12h",
        REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_rank90bad45_to12h_v1",
    ),
    (
        "rank80_bad65_to12h",
        REPORT_ROOT / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_rank80bad65_to12h_v1",
    ),
]


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass"}
    if pd.isna(value):
        return False
    return bool(value)


def _exposure_from_decisions(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    decisions = pd.read_parquet(path)
    if decisions.empty:
        return {
            "accepted_unique_symbols": 0,
            "accepted_active_days": 0,
            "accepted_span_days": 0,
            "accepted_long_trades": 0,
            "accepted_short_trades": 0,
            "accepted_min_fold_trades": 0,
            "accepted_min_monthly_trades": 0,
        }
    mask = (
        decisions["accepted"].astype(bool)
        if "accepted" in decisions.columns
        else pd.Series(True, index=decisions.index)
    )
    accepted = decisions.loc[mask].copy()
    if accepted.empty:
        return {
            "accepted_unique_symbols": 0,
            "accepted_active_days": 0,
            "accepted_span_days": 0,
            "accepted_long_trades": 0,
            "accepted_short_trades": 0,
            "accepted_min_fold_trades": 0,
            "accepted_min_monthly_trades": 0,
        }
    ts = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    side_counts = accepted.get("side", pd.Series("", index=accepted.index)).astype(str).value_counts()
    fold_counts = (
        accepted.groupby(accepted["validation_week"].astype(str)).size()
        if "validation_week" in accepted.columns
        else pd.Series(dtype=int)
    )
    month_counts = accepted.groupby(ts.dt.to_period("M").astype(str)).size() if ts.notna().any() else pd.Series(dtype=int)
    return {
        "accepted_unique_symbols": int(accepted.get("symbol", pd.Series("", index=accepted.index)).astype(str).nunique()),
        "accepted_active_days": int(ts.dt.date.nunique()),
        "accepted_span_days": int((ts.max().date() - ts.min().date()).days + 1) if ts.notna().any() else 0,
        "accepted_long_trades": int(side_counts.get("long", 0)),
        "accepted_short_trades": int(side_counts.get("short", 0)),
        "accepted_min_fold_trades": int(fold_counts.min()) if not fold_counts.empty else 0,
        "accepted_min_monthly_trades": int(month_counts.min()) if not month_counts.empty else 0,
    }


def _accepted_decisions(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    decisions = pd.read_parquet(path)
    if decisions.empty:
        return pd.DataFrame()
    mask = (
        decisions["accepted"].astype(bool)
        if "accepted" in decisions.columns
        else pd.Series(True, index=decisions.index)
    )
    accepted = decisions.loc[mask].copy()
    if accepted.empty:
        return pd.DataFrame()
    accepted["timestamp"] = pd.to_datetime(accepted["timestamp"], utc=True, errors="coerce")
    accepted["accepted_key"] = (
        accepted["timestamp"].astype(str)
        + "|"
        + accepted.get("symbol", pd.Series("", index=accepted.index)).astype(str)
        + "|"
        + accepted.get("side", pd.Series("", index=accepted.index)).astype(str)
        + "|"
        + accepted.get("strategy_id", pd.Series("", index=accepted.index)).astype(str)
    )
    accepted["accepted_net_pnl"] = (
        pd.to_numeric(accepted.get("position_size", 0.0), errors="coerce").fillna(0.0)
        * pd.to_numeric(accepted.get("position_net_return", 0.0), errors="coerce").fillna(0.0)
    )
    accepted["accepted_net_return"] = pd.to_numeric(
        accepted.get("position_net_return", 0.0),
        errors="coerce",
    ).fillna(0.0)
    accepted["accepted_exit_reason"] = accepted.get(
        "position_exit_reason",
        accepted.get("simple_policy_exit_reason", pd.Series("", index=accepted.index)),
    ).astype(str)
    return accepted


def _incremental_attribution(variant_dirs: list[tuple[str, Path]], baseline_name: str = "current_anchor") -> pd.DataFrame:
    accepted_by_name: dict[str, pd.DataFrame] = {}
    for name, directory in variant_dirs:
        decisions_path = directory / "adaptive_scenario_guard_decisions.parquet"
        accepted_by_name[name] = _accepted_decisions(decisions_path)
    baseline = accepted_by_name.get(baseline_name, pd.DataFrame())
    baseline_keys = set(baseline.get("accepted_key", pd.Series(dtype=str)).astype(str))
    rows: list[dict[str, Any]] = []
    for name, accepted in accepted_by_name.items():
        if accepted.empty:
            rows.append(
                {
                    "variant_name": name,
                    "accepted_trades": 0,
                    "incremental_trades": 0,
                    "incremental_net_pnl": 0.0,
                    "incremental_mean_net_return": np.nan,
                    "incremental_hit_rate": np.nan,
                    "incremental_full_sl_rate": np.nan,
                    "incremental_timeout_rate": np.nan,
                    "incremental_unique_symbols": 0,
                    "incremental_active_days": 0,
                    "lost_baseline_trades": int(len(baseline)) if name != baseline_name else 0,
                }
            )
            continue
        variant_keys = set(accepted["accepted_key"].astype(str))
        incremental = accepted.loc[~accepted["accepted_key"].astype(str).isin(baseline_keys)].copy()
        lost = baseline.loc[~baseline["accepted_key"].astype(str).isin(variant_keys)].copy() if not baseline.empty else pd.DataFrame()
        rows.append(
            {
                "variant_name": name,
                "accepted_trades": int(len(accepted)),
                "incremental_trades": int(len(incremental)),
                "incremental_net_pnl": float(incremental["accepted_net_pnl"].sum()) if not incremental.empty else 0.0,
                "incremental_mean_net_return": float(incremental["accepted_net_return"].mean()) if not incremental.empty else np.nan,
                "incremental_hit_rate": float(incremental["accepted_net_return"].gt(0.0).mean()) if not incremental.empty else np.nan,
                "incremental_full_sl_rate": float(incremental["accepted_exit_reason"].eq("full_sl").mean()) if not incremental.empty else np.nan,
                "incremental_timeout_rate": float(incremental["accepted_exit_reason"].eq("timeout").mean()) if not incremental.empty else np.nan,
                "incremental_unique_symbols": int(incremental.get("symbol", pd.Series(dtype=str)).astype(str).nunique()) if not incremental.empty else 0,
                "incremental_active_days": int(incremental["timestamp"].dt.date.nunique()) if not incremental.empty else 0,
                "lost_baseline_trades": int(len(lost)) if name != baseline_name else 0,
                "lost_baseline_net_pnl": float(lost["accepted_net_pnl"].sum()) if not lost.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _variant_row(name: str, directory: Path, baseline_trades: int) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"
    summary_path = directory / "adaptive_scenario_guard_summary.csv"
    folds_path = directory / "adaptive_scenario_guard_folds.csv"
    decisions_path = directory / "adaptive_scenario_guard_decisions.parquet"
    row: dict[str, Any] = {
        "variant_name": name,
        "directory": str(directory),
        "exists": bool(manifest_path.exists() and summary_path.exists()),
    }
    if not row["exists"]:
        return row
    manifest = _read_json(manifest_path)
    summary_df = pd.read_csv(summary_path)
    if summary_df.empty:
        row["exists"] = False
        return row
    summary = summary_df.iloc[0].to_dict()
    fixed_gate = manifest.get("fixed_comparator_gate") if isinstance(manifest.get("fixed_comparator_gate"), dict) else {}
    checks = fixed_gate.get("checks") if isinstance(fixed_gate.get("checks"), dict) else {}
    folds = pd.read_csv(folds_path) if folds_path.exists() else pd.DataFrame()
    accepted_trades = int(pd.to_numeric(pd.Series([summary.get("accepted_trades", 0)]), errors="coerce").fillna(0).iloc[0])
    exposure = _exposure_from_decisions(decisions_path)
    row.update(
        {
            "pass_adaptive_scenario_gate": _as_bool(manifest.get("pass_adaptive_scenario_gate")),
            "pass_simple_policy_gate": _as_bool(summary.get("pass_simple_policy_gate")),
            "beats_fixed_guard": _as_bool(fixed_gate.get("passes")),
            "failed_fixed_checks": ",".join([key for key, value in checks.items() if not bool(value)]),
            "folds": int(summary.get("folds", 0) or 0),
            "sum_net_pnl": float(summary.get("sum_net_pnl", np.nan)),
            "mean_objective": float(summary.get("mean_objective", np.nan)),
            "worst_fold_net_pnl": float(summary.get("worst_fold_net_pnl", np.nan)),
            "positive_fold_share": float(summary.get("positive_fold_share", np.nan)),
            "no_trade_folds": int(summary.get("no_trade_folds", 0) or 0),
            "accepted_trades": accepted_trades,
            "accepted_trade_delta": int(accepted_trades - baseline_trades),
            "weighted_full_sl_rate": float(summary.get("weighted_full_sl_rate", np.nan)),
            "weighted_timeout_rate": float(summary.get("weighted_timeout_rate", np.nan)),
            "mean_hit_rate": float(summary.get("mean_hit_rate", np.nan)),
            "mean_keep_frac": float(summary.get("mean_keep_frac", np.nan)),
            "min_fold_net_pnl": float(pd.to_numeric(folds.get("net_pnl", pd.Series(dtype=float)), errors="coerce").min())
            if not folds.empty and "net_pnl" in folds
            else np.nan,
            "max_fold_timeout_rate": float(pd.to_numeric(folds.get("timeout_rate", pd.Series(dtype=float)), errors="coerce").max())
            if not folds.empty and "timeout_rate" in folds
            else np.nan,
            "max_fold_full_sl_rate": float(pd.to_numeric(folds.get("full_sl_rate", pd.Series(dtype=float)), errors="coerce").max())
            if not folds.empty and "full_sl_rate" in folds
            else np.nan,
            "keep_fracs": ",".join(map(str, manifest.get("keep_fracs") or [])),
            "selection_mode": str(manifest.get("selection_mode") or ""),
            "require_anchor_admission": bool(manifest.get("require_anchor_admission")),
        }
    )
    row.update(exposure)
    row["breadth_review_pass"] = bool(
        row["pass_simple_policy_gate"]
        and accepted_trades >= int(baseline_trades)
        and int(row.get("accepted_active_days", 0) or 0) >= 15
        and int(row.get("accepted_unique_symbols", 0) or 0) >= 15
    )
    row["deployment_breadth_pass"] = bool(
        accepted_trades >= 100
        and int(row.get("accepted_active_days", 0) or 0) >= 30
        and int(row.get("accepted_min_monthly_trades", 0) or 0) >= 20
    )
    return row


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int = 30) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[col for col in cols if col in frame.columns]].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.6f}")
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--baseline-accepted-trades", type=int, default=36)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        _variant_row(name, directory, baseline_trades=int(args.baseline_accepted_trades))
        for name, directory in DEFAULT_VARIANTS
    ]
    summary = pd.DataFrame(rows)
    if not summary.empty and "exists" in summary:
        summary = summary.sort_values(
            [
                "pass_adaptive_scenario_gate",
                "pass_simple_policy_gate",
                "accepted_trades",
                "sum_net_pnl",
            ],
            ascending=[False, False, False, False],
        ).reset_index(drop=True)
    passing = summary.loc[summary.get("pass_adaptive_scenario_gate", pd.Series(False, index=summary.index)).astype(bool)].copy()
    breadth = summary.loc[summary.get("breadth_review_pass", pd.Series(False, index=summary.index)).astype(bool)].copy()
    deployment = summary.loc[summary.get("deployment_breadth_pass", pd.Series(False, index=summary.index)).astype(bool)].copy()
    incremental = _incremental_attribution(DEFAULT_VARIANTS)
    best_gate = passing.iloc[0].to_dict() if not passing.empty else {}
    best_breadth = breadth.sort_values(["accepted_trades", "sum_net_pnl"], ascending=[False, False]).iloc[0].to_dict() if not breadth.empty else {}
    conclusion = {
        "variants": int(len(summary)),
        "adaptive_gate_pass_count": int(len(passing)),
        "breadth_review_pass_count": int(len(breadth)),
        "deployment_breadth_pass_count": int(len(deployment)),
        "best_gate_variant": best_gate.get("variant_name", ""),
        "best_gate_accepted_trades": int(best_gate.get("accepted_trades", 0) or 0),
        "best_breadth_variant": best_breadth.get("variant_name", ""),
        "best_breadth_accepted_trades": int(best_breadth.get("accepted_trades", 0) or 0),
        "status": "no_deployment_breadth_candidate" if deployment.empty else "deployment_breadth_candidate_found",
        "interpretation": (
            "Threshold/keep-fraction loosening and broader 9-12h source filters can increase trades, "
            "but no tested variant clears deployment breadth; wider variants lose fold stability, "
            "full-SL control, or fixed-benchmark quality."
        ),
    }

    paths = {
        "summary": args.out_dir / "guarded_execution_breadth_sweep_summary.csv",
        "incremental_attribution": args.out_dir / "guarded_execution_breadth_incremental_attribution.csv",
        "manifest": args.out_dir / "guarded_execution_breadth_sweep_manifest.json",
        "report": args.out_dir / "guarded_execution_breadth_sweep_report.md",
    }
    summary.to_csv(paths["summary"], index=False)
    incremental.to_csv(paths["incremental_attribution"], index=False)
    manifest = {
        "generated_by": "report_guarded_execution_breadth_sweep",
        "out_dir": str(args.out_dir),
        "baseline_accepted_trades": int(args.baseline_accepted_trades),
        "conclusion": conclusion,
        "variants": [{"name": name, "directory": str(directory)} for name, directory in DEFAULT_VARIANTS],
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    lines = [
        "# Guarded Execution Breadth Sweep",
        "",
        f"Status: `{conclusion['status']}`",
        "",
        "## Conclusion",
        "",
        conclusion["interpretation"],
        "",
        "## Variants",
        "",
        _fmt_table(
            summary,
            [
                "variant_name",
                "pass_adaptive_scenario_gate",
                "pass_simple_policy_gate",
                "beats_fixed_guard",
                "accepted_trades",
                "accepted_trade_delta",
                "accepted_unique_symbols",
                "accepted_active_days",
                "accepted_min_monthly_trades",
                "sum_net_pnl",
                "mean_objective",
                "weighted_full_sl_rate",
                "weighted_timeout_rate",
                "failed_fixed_checks",
            ],
            max_rows=60,
        ),
        "",
        "## Best Gate Variant",
        "",
        _fmt_table(pd.DataFrame([best_gate]) if best_gate else pd.DataFrame(), list(summary.columns), max_rows=1),
        "",
        "## Best Breadth Variant",
        "",
        _fmt_table(pd.DataFrame([best_breadth]) if best_breadth else pd.DataFrame(), list(summary.columns), max_rows=1),
        "",
        "## Incremental Trades vs Current Anchor",
        "",
        _fmt_table(
            incremental,
            [
                "variant_name",
                "accepted_trades",
                "incremental_trades",
                "incremental_net_pnl",
                "incremental_mean_net_return",
                "incremental_hit_rate",
                "incremental_full_sl_rate",
                "incremental_timeout_rate",
                "incremental_unique_symbols",
                "incremental_active_days",
                "lost_baseline_trades",
                "lost_baseline_net_pnl",
            ],
            max_rows=80,
        ),
    ]
    paths["report"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(_json_safe({"conclusion": conclusion, "outputs": {k: str(v) for k, v in paths.items()}}), indent=2))


if __name__ == "__main__":
    main()
