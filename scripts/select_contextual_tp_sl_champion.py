#!/usr/bin/env python3
"""Select contextual TP/SL challengers from monthly walk-forward metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


PROFILE_RULES: Dict[str, Dict[str, float]] = {
    "pnl_dominant": {
        "min_positive_month_share": 4.0 / 6.0,
        "max_mean_full_sl_delta": 0.0,
        "min_june_drawdown_delta": -0.005,
    },
    "balanced_tail": {
        "min_positive_month_share": 4.0 / 6.0,
        "max_mean_full_sl_delta": 0.0,
        "min_mean_drawdown_delta": 0.0,
        "min_monthly_drawdown_delta": -0.010,
        "max_monthly_full_sl_deterioration": 0.011,
    },
    "strict_tail": {
        "min_positive_month_share": 4.0 / 6.0,
        "max_mean_full_sl_delta": 0.0,
        "min_mean_drawdown_delta": 0.0,
        "min_monthly_drawdown_delta": 0.0,
        "max_monthly_full_sl_deterioration": 0.0,
    },
}


FAMILY_BY_LABEL = {
    "shortasset_uncertainty_only": "uncertainty",
    "shortasset_drift_only": "drift",
    "shortasset_ood_only": "ood",
    "longbars_uncertainty_only": "uncertainty",
    "longbars_drift_only": "drift",
    "longbars_ood_only": "ood",
    "longbars_weekgate_only": "recent_hr_surprise",
    "combined": "uncertainty_plus_recent_hr_surprise",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _summarize_label(group: pd.DataFrame) -> Dict[str, Any]:
    pnl = group["delta_net_pnl"].astype(float)
    full_sl = group["delta_full_sl_rate"].astype(float)
    drawdown = group["delta_max_drawdown"].astype(float)
    june = group.loc[group["month"].astype(str) == "2026-06"]
    june_drawdown_delta = float(june["delta_max_drawdown"].iloc[0]) if len(june) else np.nan
    q35 = float(pnl.quantile(0.35))
    q20 = float(pnl.quantile(0.20))
    objective = float(pnl.mean() + 0.7 * q35 + 0.3 * q20)
    return {
        "label": str(group["label"].iloc[0]),
        "diagnostic_family": FAMILY_BY_LABEL.get(str(group["label"].iloc[0]), "unknown"),
        "months": int(len(group)),
        "objective": objective,
        "mean_delta_net_pnl": float(pnl.mean()),
        "sum_delta_net_pnl": float(pnl.sum()),
        "median_delta_net_pnl": float(pnl.median()),
        "q35_delta_net_pnl": q35,
        "q20_delta_net_pnl": q20,
        "min_monthly_delta_net_pnl": float(pnl.min()),
        "positive_month_share": float((pnl > 0).mean()),
        "positive_month_count": int((pnl > 0).sum()),
        "mean_delta_full_sl_rate": float(full_sl.mean()),
        "max_monthly_full_sl_deterioration": float(full_sl.max()),
        "months_full_sl_improved": int((full_sl < 0).sum()),
        "mean_delta_max_drawdown": float(drawdown.mean()),
        "min_monthly_drawdown_delta": float(drawdown.min()),
        "months_drawdown_improved": int((drawdown > 0).sum()),
        "june_delta_net_pnl": float(june["delta_net_pnl"].iloc[0]) if len(june) else np.nan,
        "june_delta_full_sl_rate": float(june["delta_full_sl_rate"].iloc[0]) if len(june) else np.nan,
        "june_delta_max_drawdown": june_drawdown_delta,
        "sum_delta_trade_count": int(group["delta_trade_count"].sum()),
    }


def _profile_pass(row: pd.Series, rules: Dict[str, float]) -> tuple[bool, List[str]]:
    failures: List[str] = []
    if row["positive_month_share"] < rules.get("min_positive_month_share", -np.inf):
        failures.append("positive_month_share")
    if row["mean_delta_full_sl_rate"] > rules.get("max_mean_full_sl_delta", np.inf):
        failures.append("mean_full_sl_delta")
    if row["mean_delta_max_drawdown"] < rules.get("min_mean_drawdown_delta", -np.inf):
        failures.append("mean_drawdown_delta")
    if row["min_monthly_drawdown_delta"] < rules.get("min_monthly_drawdown_delta", -np.inf):
        failures.append("monthly_drawdown_tail")
    if row["max_monthly_full_sl_deterioration"] > rules.get("max_monthly_full_sl_deterioration", np.inf):
        failures.append("monthly_full_sl_tail")
    if row["june_delta_max_drawdown"] < rules.get("min_june_drawdown_delta", -np.inf):
        failures.append("june_drawdown")
    return not failures, failures


def _markdown_table(df: pd.DataFrame, columns: List[str]) -> str:
    if df.empty:
        return "_No rows._"
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in df[columns].iterrows():
        vals = []
        for value in row:
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--global-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--baseline-label", default="wf_recent")
    args = parser.parse_args()

    global_csv = Path(args.global_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(global_csv)
    challengers = df[df["label"] != args.baseline_label].copy()
    if challengers.empty:
        raise ValueError("No challenger rows found")

    summary = pd.DataFrame([_summarize_label(group) for _, group in challengers.groupby("label")])
    summary = summary.sort_values(["objective", "sum_delta_net_pnl"], ascending=False).reset_index(drop=True)
    summary["objective_rank"] = np.arange(1, len(summary) + 1)

    profile_rows: List[Dict[str, Any]] = []
    champions: Dict[str, Any] = {}
    for profile, rules in PROFILE_RULES.items():
        ranked = summary.copy()
        passes = []
        fail_reasons = []
        for _, row in ranked.iterrows():
            ok, failures = _profile_pass(row, rules)
            passes.append(ok)
            fail_reasons.append(",".join(failures))
        ranked["profile"] = profile
        ranked["passes_profile"] = passes
        ranked["profile_fail_reasons"] = fail_reasons
        ranked = ranked.sort_values(["passes_profile", "objective", "sum_delta_net_pnl"], ascending=[False, False, False])
        profile_rows.extend(ranked.to_dict("records"))
        passed = ranked[ranked["passes_profile"]]
        champions[profile] = passed.iloc[0].to_dict() if len(passed) else None

    profile_df = pd.DataFrame(profile_rows)
    summary.to_csv(out_dir / "champion_summary.csv", index=False)
    profile_df.to_csv(out_dir / "champion_profile_ranking.csv", index=False)
    (out_dir / "champion_selection.json").write_text(
        json.dumps(
            _json_safe(
                {
                    "source_global_csv": str(global_csv),
                    "objective": "mean_delta_net_pnl + 0.7*q35_delta_net_pnl + 0.3*q20_delta_net_pnl",
                    "profiles": PROFILE_RULES,
                    "champions": champions,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    columns = [
        "objective_rank",
        "label",
        "diagnostic_family",
        "objective",
        "sum_delta_net_pnl",
        "positive_month_count",
        "mean_delta_full_sl_rate",
        "mean_delta_max_drawdown",
        "june_delta_max_drawdown",
    ]
    profile_columns = [
        "profile",
        "label",
        "passes_profile",
        "profile_fail_reasons",
        "objective",
        "sum_delta_net_pnl",
        "positive_month_count",
        "mean_delta_full_sl_rate",
        "min_monthly_drawdown_delta",
    ]
    md = [
        "# Contextual TP/SL Champion Selection",
        "",
        "This is a development walk-forward selector, not untouched OOS. It ranks the already-completed monthly replays against `wf_recent`.",
        "",
        "Objective: `mean_delta_net_pnl + 0.7 * q35_delta_net_pnl + 0.3 * q20_delta_net_pnl`.",
        "",
        "## Overall Ranking",
        "",
        _markdown_table(summary, columns),
        "",
        "## Profile Champions",
        "",
    ]
    for profile in PROFILE_RULES:
        champ = champions.get(profile)
        if champ is None:
            md.append(f"- `{profile}`: no challenger passed all guardrails; keep baseline.")
        else:
            md.append(
                f"- `{profile}`: `{champ['label']}` "
                f"(objective={champ['objective']:.2f}, sum_delta_net_pnl={champ['sum_delta_net_pnl']:.2f})."
            )
    md.extend(["", "## Profile Ranking", "", _markdown_table(profile_df, profile_columns), ""])
    (out_dir / "champion_selection_report.md").write_text("\n".join(md), encoding="utf-8")
    print(out_dir / "champion_selection_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
