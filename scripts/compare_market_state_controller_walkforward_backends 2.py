#!/usr/bin/env python3
"""Compare market-state controller walk-forward backend artifacts.

This report is intentionally artifact-only: it reads completed walk-forward
runs, compares selection-gate outcomes and paired portfolio deltas, and writes a
small CSV/Markdown bundle. It does not refit models, replay portfolios, or alter
controller state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_ARTIFACTS = (
    "market_state_feature_contract.json",
    "market_state_training_reference.joblib",
    "market_state_timestamp_panel.parquet",
    "market_state_feature_coverage.csv",
    "market_state_target_definitions.json",
    "market_state_target_cdfs.joblib",
    "market_state_oof_predictions.parquet",
    "market_state_head_diagnostics.csv",
    "strategy_rank_outcome_curves.joblib",
    "strategy_residual_target_ledger.parquet",
    "strategy_response_ebm_models.joblib",
    "strategy_response_oof_predictions.parquet",
    "strategy_state_effect_matrix.csv",
    "strategy_threshold_schedule.parquet",
    "strategy_threshold_controller_config.json",
    "strategy_threshold_action_audit.csv",
    "portfolio_replay_summary.csv",
    "portfolio_replay_by_head.csv",
    "walkforward_aggregate_delta.csv",
    "walkforward_controller_candidate_selection.csv",
    "walkforward_selected_controller_candidate.json",
    "manifest.json",
)


SUMMARY_ARMS = (
    "S0_baseline_static_thresholds",
    "S1_observed_axes_shared_response",
    "S2_observed_forecast_shared_response",
    "S1_observed_axes_shared_response__post_selection_overlay",
    "S2_observed_forecast_shared_response__post_selection_overlay",
)

SUMMARY_METRIC_COLUMNS = (
    "median_delta_net_pnl",
    "q25_delta_net_pnl",
    "positive_delta_share",
    "mean_delta_net_pnl",
    "median_trade_count",
    "median_trade_retention_share",
    "median_delta_full_sl_rate",
    "median_delta_max_drawdown",
    "median_delta_worst_24h",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _selection_payload(root: Path) -> dict[str, Any]:
    selected = _read_json(root / "walkforward_selected_controller_candidate.json")
    manifest = _read_json(root / "manifest.json")
    return {
        "selected_arm": selected.get("selected_arm"),
        "selected_reason": selected.get("reason"),
        "rank_contract": manifest.get("rank_contract"),
        "forecast_model_kind": manifest.get("forecast_model_kind"),
        "response_model_kind": manifest.get("response_model_kind"),
        "source_contract_audit_passed": bool(
            (manifest.get("source_contract_audit") or {}).get("overall_passed")
        ),
    }


def _artifact_contract(root: Path, backend: str) -> dict[str, Any]:
    model_artifact = (
        "market_state_xgb_models.joblib"
        if backend.lower() in {"xgb", "xgboost"}
        else "market_state_lgbm_models.joblib"
    )
    required = [*REQUIRED_ARTIFACTS, model_artifact]
    present = {name: (root / name).exists() for name in required}
    missing = sorted(name for name, exists in present.items() if not exists)
    oof_rows = None
    oof_cols = None
    oof_split_count = None
    oof_fold_count = None
    oof_path = root / "market_state_oof_predictions.parquet"
    if oof_path.exists():
        oof = pd.read_parquet(oof_path)
        oof_rows = int(len(oof))
        oof_cols = int(len(oof.columns))
        oof_split_count = int(oof["split"].nunique()) if "split" in oof.columns else None
        oof_fold_count = int(oof["fold"].nunique()) if "fold" in oof.columns else None
    manifest = _read_json(root / "manifest.json")
    source_audit = manifest.get("source_contract_audit") or {}
    return {
        "backend": backend,
        "root": str(root),
        "required_artifacts": len(required),
        "missing_artifacts": ";".join(missing),
        "all_required_artifacts_present": not missing,
        "source_contract_audit_passed": bool(source_audit.get("overall_passed")),
        "actual_order_book_features_allowed": bool(
            source_audit.get("actual_order_book_features_allowed")
        ),
        "candidate_population_fallback_allowed_for_production": bool(
            source_audit.get("candidate_population_fallback_allowed_for_production")
        ),
        "oof_rows": oof_rows,
        "oof_cols": oof_cols,
        "oof_split_count": oof_split_count,
        "oof_fold_count": oof_fold_count,
    }


def _aggregate_metrics(root: Path, backend: str) -> pd.DataFrame:
    path = root / "walkforward_aggregate_delta.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if {"median_trade_retention_share", "median_delta_full_sl_rate"} - set(df.columns):
        summary_path = root / "walkforward_summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path)
            required = {"fold", "arm", "trade_count", "full_sl_rate"}
            if required.issubset(summary.columns):
                base = summary.loc[
                    summary["arm"].astype(str).eq("S0_baseline_static_thresholds"),
                    ["fold", "trade_count", "full_sl_rate"],
                ].rename(columns={"trade_count": "base_trade_count", "full_sl_rate": "base_full_sl_rate"})
                merged = summary.merge(base, on="fold", how="left")
                base_trade_count = pd.to_numeric(merged["base_trade_count"], errors="coerce").replace(0.0, float("nan"))
                merged["_trade_retention_share"] = pd.to_numeric(merged["trade_count"], errors="coerce") / base_trade_count
                merged.loc[merged["arm"].astype(str).eq("S0_baseline_static_thresholds"), "_trade_retention_share"] = 1.0
                merged["_delta_full_sl_rate"] = (
                    pd.to_numeric(merged["full_sl_rate"], errors="coerce")
                    - pd.to_numeric(merged["base_full_sl_rate"], errors="coerce")
                )
                safety = (
                    merged.groupby("arm", sort=False)
                    .agg(
                        median_trade_retention_share=("_trade_retention_share", "median"),
                        median_delta_full_sl_rate=("_delta_full_sl_rate", "median"),
                    )
                    .reset_index()
                )
                add_cols = [col for col in safety.columns if col == "arm" or col not in df.columns]
                df = df.merge(safety[add_cols], on="arm", how="left")
    df.insert(0, "backend", backend)
    return df


def _selection_metrics(root: Path, backend: str) -> pd.DataFrame:
    path = root / "walkforward_controller_candidate_selection.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "backend", backend)
    return df


def _prospective_increment(aggregate: pd.DataFrame, backend: str) -> dict[str, Any]:
    if aggregate.empty:
        return {"backend": backend}
    by_arm = aggregate.set_index("arm")
    s1 = by_arm.loc["S1_observed_axes_shared_response"] if "S1_observed_axes_shared_response" in by_arm.index else None
    s2 = by_arm.loc["S2_observed_forecast_shared_response"] if "S2_observed_forecast_shared_response" in by_arm.index else None
    if s1 is None or s2 is None:
        return {"backend": backend}
    return {
        "backend": backend,
        "s2_minus_s1_median_delta_net_pnl": float(
            s2["median_delta_net_pnl"] - s1["median_delta_net_pnl"]
        ),
        "s2_minus_s1_q25_delta_net_pnl": float(
            s2["q25_delta_net_pnl"] - s1["q25_delta_net_pnl"]
        ),
        "s2_minus_s1_positive_delta_share": float(
            s2["positive_delta_share"] - s1["positive_delta_share"]
        ),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _render_markdown(
    *,
    output_dir: Path,
    summary: pd.DataFrame,
    selection: pd.DataFrame,
    contracts: pd.DataFrame,
    increments: pd.DataFrame,
    selected: pd.DataFrame,
) -> str:
    lines: list[str] = [
        "# Market-State Controller Backend Comparison",
        "",
        "This report compares completed walk-forward artifacts. It does not refit models, replay portfolios, or enable controller execution.",
        "",
        "## Selection Verdict",
        "",
    ]
    for _, row in selected.iterrows():
        lines.append(
            f"- `{row['backend']}`: selected arm `{row['selected_arm']}`; reason `{row['selected_reason']}`; "
            f"rank contract `{row['rank_contract']}`; source audit passed `{row['source_contract_audit_passed']}`."
        )
    lines.extend(
        [
            "",
            "## Paired Portfolio Delta Summary",
            "",
            "| backend | arm | median_delta_net_pnl | q25_delta_net_pnl | positive_delta_share | mean_delta_net_pnl | median_trade_count | trade_retention | delta_full_sl | delta_max_dd | delta_worst_24h |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    display = summary.loc[summary["arm"].isin(SUMMARY_ARMS)].copy()
    for _, row in display.iterrows():
        values = {col: row.get(col) for col in SUMMARY_METRIC_COLUMNS}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["backend"]),
                    str(row["arm"]),
                    _fmt(float(values["median_delta_net_pnl"])),
                    _fmt(float(values["q25_delta_net_pnl"])),
                    _fmt(float(values["positive_delta_share"])),
                    _fmt(float(values["mean_delta_net_pnl"])),
                    _fmt(float(values["median_trade_count"])),
                    _fmt(values.get("median_trade_retention_share")),
                    _fmt(values.get("median_delta_full_sl_rate")),
                    _fmt(values.get("median_delta_max_drawdown")),
                    _fmt(values.get("median_delta_worst_24h")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Prospective Forecast Increment",
            "",
            "| backend | S2-S1 median delta | S2-S1 q25 delta | S2-S1 positive share |",
            "|---|---:|---:|---:|",
        ]
    )
    for _, row in increments.iterrows():
        lines.append(
            f"| {row['backend']} | "
            f"{_fmt(row.get('s2_minus_s1_median_delta_net_pnl'))} | "
            f"{_fmt(row.get('s2_minus_s1_q25_delta_net_pnl'))} | "
            f"{_fmt(row.get('s2_minus_s1_positive_delta_share'))} |"
        )
    lines.extend(
        [
            "",
            "## Selection Gate Failures",
            "",
            "| backend | arm | passed | selection_score | failure_reasons |",
            "|---|---|---:|---:|---|",
        ]
    )
    for _, row in selection.iterrows():
        if str(row.get("arm")) not in SUMMARY_ARMS:
            continue
        lines.append(
            f"| {row['backend']} | {row['arm']} | {bool(row.get('passed_selection_gates'))} | "
            f"{_fmt(float(row.get('selection_score', 0.0)))} | {row.get('selection_fail_reasons', '')} |"
        )
    lines.extend(
        [
            "",
            "## Artifact Contract",
            "",
            "| backend | all_required_present | missing_artifacts | source_audit | order_book_allowed | production_fallback_allowed | oof_rows | oof_folds |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in contracts.iterrows():
        lines.append(
            f"| {row['backend']} | {bool(row['all_required_artifacts_present'])} | "
            f"{row.get('missing_artifacts', '')} | {bool(row['source_contract_audit_passed'])} | "
            f"{bool(row['actual_order_book_features_allowed'])} | "
            f"{bool(row['candidate_population_fallback_allowed_for_production'])} | "
            f"{_fmt(row.get('oof_rows'))} | {_fmt(row.get('oof_fold_count'))} |"
        )
    selected_arms = [
        str(row.get("selected_arm"))
        for _, row in selected.iterrows()
        if str(row.get("selected_arm")).strip().lower() not in {"", "none", "nan"}
    ]
    s2 = summary.loc[summary["arm"].astype(str).eq("S2_observed_forecast_shared_response")].copy()
    s2["median_delta_net_pnl"] = pd.to_numeric(s2.get("median_delta_net_pnl"), errors="coerce")
    if s2.empty or s2["median_delta_net_pnl"].dropna().empty:
        prospective_sentence = "The prospective S2 backend comparison is unavailable in these artifacts."
    else:
        s2 = s2.sort_values("median_delta_net_pnl", ascending=False)
        best = s2.iloc[0]
        prospective_sentence = (
            f"The `{best['backend']}` prospective S2 arm has the stronger raw median fold PnL "
            f"delta (`{_fmt(float(best['median_delta_net_pnl']))}`), but promotion still depends "
            "on the post-selection defensive/suppression gates."
        )

    if selected_arms:
        promotion_sentence = (
            "At least one backend selected a controller arm in these artifacts; inspect the "
            "selection-gate table before any deployment decision."
        )
    else:
        promotion_sentence = (
            "No backend produced a promotable controller in these artifacts. The executable "
            "controller should therefore remain disabled/no-op against T1 until a later run "
            "passes the full promotion contract."
        )

    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            f"{promotion_sentence} {prospective_sentence}",
            "",
            "Generated files:",
            f"- `{output_dir / 'backend_metric_comparison.csv'}`",
            f"- `{output_dir / 'controller_selection_comparison.csv'}`",
            f"- `{output_dir / 'artifact_contract_comparison.csv'}`",
            f"- `{output_dir / 'prospective_increment_comparison.csv'}`",
            f"- `{output_dir / 'selected_controller_comparison.csv'}`",
        ]
    )
    return "\n".join(lines) + "\n"


def compare_backends(run_dirs: dict[str, Path], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_frames = []
    selection_frames = []
    contracts = []
    increments = []
    selected_rows = []
    for backend, root in run_dirs.items():
        root = Path(root)
        aggregate = _aggregate_metrics(root, backend)
        selection = _selection_metrics(root, backend)
        aggregate_frames.append(aggregate)
        selection_frames.append(selection)
        contracts.append(_artifact_contract(root, backend))
        increments.append(_prospective_increment(aggregate, backend))
        selected_rows.append({"backend": backend, **_selection_payload(root)})

    aggregate_df = pd.concat(aggregate_frames, ignore_index=True) if aggregate_frames else pd.DataFrame()
    selection_df = pd.concat(selection_frames, ignore_index=True) if selection_frames else pd.DataFrame()
    contract_df = pd.DataFrame(contracts)
    increment_df = pd.DataFrame(increments)
    selected_df = pd.DataFrame(selected_rows)

    paths = {
        "backend_metric_comparison": output_dir / "backend_metric_comparison.csv",
        "controller_selection_comparison": output_dir / "controller_selection_comparison.csv",
        "artifact_contract_comparison": output_dir / "artifact_contract_comparison.csv",
        "prospective_increment_comparison": output_dir / "prospective_increment_comparison.csv",
        "selected_controller_comparison": output_dir / "selected_controller_comparison.csv",
        "report": output_dir / "market_state_backend_comparison_report.md",
    }
    aggregate_df.to_csv(paths["backend_metric_comparison"], index=False)
    selection_df.to_csv(paths["controller_selection_comparison"], index=False)
    contract_df.to_csv(paths["artifact_contract_comparison"], index=False)
    increment_df.to_csv(paths["prospective_increment_comparison"], index=False)
    selected_df.to_csv(paths["selected_controller_comparison"], index=False)
    paths["report"].write_text(
        _render_markdown(
            output_dir=output_dir,
            summary=aggregate_df,
            selection=selection_df,
            contracts=contract_df,
            increments=increment_df,
            selected=selected_df,
        )
    )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lgbm-dir", required=True, type=Path)
    parser.add_argument("--xgb-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = compare_backends(
        {"lgbm": args.lgbm_dir, "xgb": args.xgb_dir},
        args.output_dir,
    )
    print(f"Wrote backend comparison report: {paths['report']}")


if __name__ == "__main__":
    main()
