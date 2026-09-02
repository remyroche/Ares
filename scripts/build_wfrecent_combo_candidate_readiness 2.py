#!/usr/bin/env python3
"""Build a readiness packet for wf_recent smooth-penalty combo challengers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_wfrecent_row_guard_walkforward import _fmt_table, _json_safe  # noqa: E402


DEFAULT_GATES = {
    "min_guarded_weeks": 20,
    "min_sum_delta_net_pnl": 0.0,
    "min_delta_tail_objective": 0.0,
    "min_positive_delta_week_share": 0.50,
    "min_boot_prob_sum_positive": 0.95,
    "min_boot_prob_objective_positive": 0.95,
    "min_boot_sum_q05": 0.0,
    "min_boot_objective_delta_q05": 0.0,
    "max_delta_full_sl_rate": 0.0,
}


def _bool(value: Any) -> bool:
    return bool(value) and not pd.isna(value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _gate_rows(metrics: pd.DataFrame, outputs: pd.DataFrame, gates: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    output_variants = set(outputs["combo"].astype(str)) if not outputs.empty and "combo" in outputs.columns else set()
    for _, row in metrics.iterrows():
        variant = str(row["variant"])
        checks = {
            "enough_guarded_weeks": float(row["weeks"]) >= gates["min_guarded_weeks"],
            "positive_net_pnl": float(row["sum_delta_net_pnl"]) > gates["min_sum_delta_net_pnl"],
            "positive_tail_objective": float(row["delta_tail_objective"]) > gates["min_delta_tail_objective"],
            "positive_week_share": float(row["positive_delta_week_share"]) >= gates["min_positive_delta_week_share"],
            "bootstrap_sum_positive": float(row["boot_prob_sum_positive"]) >= gates["min_boot_prob_sum_positive"],
            "bootstrap_objective_positive": float(row["boot_prob_objective_positive"])
            >= gates["min_boot_prob_objective_positive"],
            "bootstrap_sum_floor": float(row["boot_sum_q05"]) > gates["min_boot_sum_q05"],
            "bootstrap_objective_floor": float(row["boot_objective_delta_q05"])
            > gates["min_boot_objective_delta_q05"],
            "full_sl_not_worse": float(row["delta_full_sl_rate"]) <= gates["max_delta_full_sl_rate"],
            "deployable_output_exists": variant in output_variants,
        }
        rows.append(
            {
                "variant": variant,
                "passed": all(checks.values()),
                "failed_checks": ",".join(name for name, passed in checks.items() if not passed),
                **checks,
            }
        )
    return pd.DataFrame(rows)


def _decision_score(row: pd.Series) -> float:
    # PnL remains dominant, but give explicit credit to tail improvement and
    # bootstrap lower bounds so a fragile high-PnL variant cannot win by itself.
    return float(
        row["sum_delta_net_pnl"]
        + 0.70 * row["delta_tail_objective"]
        + 0.20 * row["boot_sum_q05"]
        + 0.10 * row["boot_objective_delta_q05"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--robustness-dir",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_expanding_robustness_20260701"),
    )
    parser.add_argument(
        "--freeze-apply-dir",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_freeze_v2_apply_smoke_20260701"),
    )
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_freeze_v2_20260701"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_perp/reports/contextual_tp_sl_wfrecent_smooth_penalty_combo_readiness_20260701"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(args.robustness_dir / "combo_expanding_robustness_decision.csv")
    outputs = pd.read_csv(args.freeze_apply_dir / "smooth_penalty_combo_apply_audit.csv")
    apply_manifest = _load_json(args.freeze_apply_dir / "smooth_penalty_combo_apply_manifest.json")
    bundle_manifest = _load_json(args.freeze_dir / "smooth_penalty_combo_bundle_manifest.json")
    gates = dict(DEFAULT_GATES)

    deploy = outputs.rename(columns={"combo": "variant"}).copy()
    decision = metrics.merge(deploy, on="variant", how="left")
    decision["decision_score"] = decision.apply(_decision_score, axis=1)
    gate_table = _gate_rows(metrics, outputs, gates)
    decision = decision.merge(gate_table[["variant", "passed", "failed_checks"]], on="variant", how="left")
    decision = decision.sort_values(["passed", "decision_score"], ascending=[False, False]).reset_index(drop=True)
    passed = decision[decision["passed"].map(_bool)].copy()
    selected = passed.iloc[0].to_dict() if not passed.empty else {}
    deployable = decision[decision["output"].notna()].copy() if "output" in decision.columns else pd.DataFrame()
    pnl_dominant = (
        deployable.sort_values("decision_score", ascending=False).iloc[0].to_dict()
        if not deployable.empty
        else {}
    )

    decision.to_csv(args.output_dir / "candidate_readiness_decision.csv", index=False)
    gate_table.to_csv(args.output_dir / "candidate_readiness_gates.csv", index=False)
    selected_payload = {
        "selected_variant": selected.get("variant"),
        "selected_output": selected.get("output"),
        "selected_output_sha256": selected.get("output_sha256"),
        "bundle_hash": apply_manifest.get("bundle_hash") or bundle_manifest.get("bundle_hash"),
        "bundle_dir": str(args.freeze_dir),
        "freeze_apply_dir": str(args.freeze_apply_dir),
        "robustness_dir": str(args.robustness_dir),
        "decision_score": selected.get("decision_score"),
        "pnl_dominant_variant": pnl_dominant.get("variant"),
        "pnl_dominant_output": pnl_dominant.get("output"),
        "pnl_dominant_output_sha256": pnl_dominant.get("output_sha256"),
        "pnl_dominant_decision_score": pnl_dominant.get("decision_score"),
        "pnl_dominant_passed_strict_gates": pnl_dominant.get("passed"),
        "pnl_dominant_failed_checks": pnl_dominant.get("failed_checks"),
        "gates": gates,
        "status": "ready_for_prospective_dual_scoring" if selected else "no_candidate_passed",
    }
    (args.output_dir / "selected_candidate_manifest.json").write_text(
        json.dumps(_json_safe(selected_payload), indent=2, sort_keys=True) + "\n"
    )

    lines = [
        "# wf_recent Combo Candidate Readiness",
        "",
        "This packet converts the existing long-window A/B replay evidence into a candidate selection for prospective dual scoring. It does not rerun portfolio replay.",
        "",
        f"Robustness source: `{args.robustness_dir}`",
        f"Frozen output source: `{args.freeze_apply_dir}`",
        f"Bundle hash: `{selected_payload['bundle_hash']}`",
        "",
        "## Selected Candidate",
        "",
    ]
    if selected:
        lines.extend(
            [
                f"- Variant: `{selected['variant']}`",
                f"- Output: `{selected['output']}`",
                f"- Output SHA256: `{selected['output_sha256']}`",
                f"- Decision score: `{float(selected['decision_score']):,.3f}`",
                "- Status: `ready_for_prospective_dual_scoring`",
            ]
        )
    else:
        lines.append("- No candidate passed all gates.")
    if pnl_dominant:
        lines.extend(
            [
                "",
                "## PnL-Dominant Challenger",
                "",
                f"- Variant: `{pnl_dominant['variant']}`",
                f"- Output: `{pnl_dominant['output']}`",
                f"- Output SHA256: `{pnl_dominant['output_sha256']}`",
                f"- Decision score: `{float(pnl_dominant['decision_score']):,.3f}`",
                f"- Passed strict gates: `{bool(pnl_dominant.get('passed'))}`",
                f"- Failed checks: `{pnl_dominant.get('failed_checks') or ''}`",
                "",
                "This arm is retained as an aggressive A/B challenger when its decision score is higher than the strict selected candidate but recurrence gates are weaker.",
            ]
        )
    lines.extend(
        [
            "",
            "## Decision Table",
            "",
            _fmt_table(
                decision,
                [
                    "variant",
                    "passed",
                    "decision_score",
                    "sum_delta_net_pnl",
                    "delta_tail_objective",
                    "positive_delta_week_share",
                    "boot_sum_q05",
                    "boot_prob_sum_positive",
                    "boot_objective_delta_q05",
                    "boot_prob_objective_positive",
                    "delta_full_sl_rate",
                    "output_sha256",
                    "failed_checks",
                ],
            ),
            "",
            "## Gates",
            "",
            _fmt_table(
                gate_table,
                [
                    "variant",
                    "passed",
                    "enough_guarded_weeks",
                    "positive_net_pnl",
                    "positive_tail_objective",
                    "positive_week_share",
                    "bootstrap_sum_positive",
                    "bootstrap_objective_positive",
                    "bootstrap_sum_floor",
                    "bootstrap_objective_floor",
                    "full_sl_not_worse",
                    "deployable_output_exists",
                    "failed_checks",
                ],
            ),
            "",
            "## Interpretation",
            "",
            "- The selected candidate is not promoted to production by this packet.",
            "- It is the leading frozen challenger for the next delayed/prospective dual-scoring comparison.",
            "- Promotion still requires live-equivalent scoring on later rows with the same rank/cost/portfolio contract.",
        ]
    )
    (args.output_dir / "candidate_readiness_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
