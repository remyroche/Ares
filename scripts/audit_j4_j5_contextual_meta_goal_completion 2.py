"""Audit completion state for the J4/J5 contextual meta goal.

The goal is only complete after:

J4 contextual capacity HPO -> J5 controlled distillation -> freeze ->
fresh chronological OOS confirmation.

This audit is intentionally strict.  If fresh guarded labels and frozen
candidate scores are not available, the status is blocked_pending_fresh_oos
instead of complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_HEADS = {"long_bars", "long_dist", "short_asset", "short_boll"}
DEFAULT_TOP30_AUDIT = Path(
    "data_perp/reports/top30_aware_contextual_meta_training_ablation_compact_all_heads_20260623/"
    "top30_aware_contextual_meta_training_requirement_audit.json"
)
DEFAULT_FREEZE_MANIFEST = Path(
    "data_perp/reports/j4_j5_contextual_meta_all_head_freeze_20260623/"
    "j4_j5_contextual_meta_all_head_freeze_manifest.csv"
)
DEFAULT_FREEZE_AUDIT = Path(
    "data_perp/reports/j4_j5_contextual_meta_all_head_freeze_20260623/"
    "j4_j5_contextual_meta_all_head_freeze_audit.json"
)
DEFAULT_READINESS = Path(
    "data_perp/reports/j4_j5_contextual_meta_fresh_oos_readiness_20260623/"
    "j4_j5_contextual_meta_fresh_oos_readiness_by_head.csv"
)
DEFAULT_READINESS_AUDIT = Path(
    "data_perp/reports/j4_j5_contextual_meta_fresh_oos_readiness_20260623/"
    "j4_j5_contextual_meta_fresh_oos_readiness_audit.json"
)
DEFAULT_EVAL_AUDIT = Path(
    "data_perp/reports/j4_j5_contextual_meta_fresh_oos_eval_20260623/"
    "j4_j5_contextual_meta_fresh_oos_eval_audit.json"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/j4_j5_contextual_meta_goal_completion_audit_20260623")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return None if not np.isfinite(val) else val
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "missing", "items": []}
    return json.loads(path.read_text())


def _requirement(requirement: str, status: str, evidence: str, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"requirement": requirement, "status": status, "evidence": evidence, "metrics": metrics or {}}


def build_audit(
    *,
    top30_audit_path: Path,
    freeze_manifest_path: Path,
    freeze_audit_path: Path,
    readiness_path: Path,
    readiness_audit_path: Path,
    eval_audit_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    top30_audit = _read_json(top30_audit_path)
    freeze_audit = _read_json(freeze_audit_path)
    readiness_audit = _read_json(readiness_audit_path)
    eval_audit = _read_json(eval_audit_path)

    freeze = pd.read_csv(freeze_manifest_path) if freeze_manifest_path.exists() else pd.DataFrame()
    readiness = pd.read_csv(readiness_path) if readiness_path.exists() else pd.DataFrame()

    rows.append(
        _requirement(
            "Retire J1-J3 as default interventions and keep top30 weighting as negative control",
            "implemented" if top30_audit.get("status") == "passed" else "missing_or_failed",
            str(top30_audit_path),
            {"audit_status": top30_audit.get("status")},
        )
    )
    heads = set(freeze.get("head", pd.Series(dtype=str)).astype(str))
    rows.append(
        _requirement(
            "Run/freeze J4 on the directional contextual champions for all required heads",
            "implemented" if heads == EXPECTED_HEADS and not freeze.empty else "missing_or_failed",
            str(freeze_manifest_path),
            {"expected_heads": sorted(EXPECTED_HEADS), "found_heads": sorted(heads)},
        )
    )
    rows.append(
        _requirement(
            "Use ordinary BCE weights for J4 rather than J2/J3 top30 reweighting",
            "implemented"
            if not freeze.empty and freeze.get("sample_weight_contract", pd.Series(dtype=str)).astype(str).eq("ordinary_bce_no_top30_reweighting").all()
            else "missing_or_failed",
            str(freeze_manifest_path),
            {
                "sample_weight_contracts": sorted(
                    set(freeze.get("sample_weight_contract", pd.Series(dtype=str)).astype(str))
                )
            },
        )
    )
    rows.append(
        _requirement(
            "J4 searches bounded LightGBM capacity/regularization controls with fixed seeds",
            "implemented"
            if not freeze.empty
            and freeze.get("j4_seeds", pd.Series(dtype=str)).astype(str).eq("29,31,37").all()
            and freeze.get("max_j4_configs", pd.Series(dtype=float)).fillna(0).ge(10).all()
            else "missing_or_failed",
            str(freeze_manifest_path),
            {
                "j4_seeds": sorted(set(freeze.get("j4_seeds", pd.Series(dtype=str)).astype(str))),
                "max_j4_configs": sorted(set(freeze.get("max_j4_configs", pd.Series(dtype=float)).dropna().astype(int).tolist())),
            },
        )
    )
    rows.append(
        _requirement(
            "Persist J4 leaf support, context split diagnostics, calibration, and directional metrics",
            "implemented" if freeze_audit.get("status") == "passed" else "missing_or_failed",
            str(freeze_audit_path),
            {"audit_status": freeze_audit.get("status")},
        )
    )
    rows.append(
        _requirement(
            "Select/freeze exact baseline, contextual arm, capacity config, distillation config, and directional thresholds",
            "implemented"
            if not freeze.empty
            and {
                "baseline_artifact_dir",
                "selected_contextual_feature_arm",
                "selected_capacity_config",
                "selected_distillation_variant",
                "rank_threshold",
                "hr10_min_delta",
                "hr20_min_delta",
                "normal_period_hr30_min_delta",
                "ndcg30_min_delta",
            }
            <= set(freeze.columns)
            else "missing_or_failed",
            str(freeze_manifest_path),
            {
                "decisions": sorted(set(freeze.get("decision", pd.Series(dtype=str)).astype(str))),
                "selected_capacity_configs": sorted(set(freeze.get("selected_capacity_config", pd.Series(dtype=str)).astype(str))),
            },
        )
    )
    rows.append(
        _requirement(
            "Run J5 only on exact promoted J4 winners and keep hard-label arm when no J4 winner promotes",
            "implemented"
            if not freeze.empty
            and freeze.get("j5_rows", pd.Series(dtype=float)).fillna(0).eq(0).all()
            and freeze.get("selected_distillation_variant", pd.Series(dtype=str)).astype(str).eq("hard_label_context_arm").all()
            else "missing_or_failed",
            str(freeze_manifest_path),
            {
                "j5_rows_total": int(freeze.get("j5_rows", pd.Series(dtype=float)).fillna(0).sum()) if not freeze.empty else 0,
                "selected_distillation_variants": sorted(set(freeze.get("selected_distillation_variant", pd.Series(dtype=str)).astype(str))),
            },
        )
    )
    ready = (
        not readiness.empty
        and "ready_for_fresh_oos_confirmation" in readiness.columns
        and readiness["ready_for_fresh_oos_confirmation"].astype(bool).all()
        and readiness_audit.get("status") == "ready"
        and eval_audit.get("status") == "passed"
    )
    rows.append(
        _requirement(
            "Fresh chronological OOS confirmation after the frozen boundary",
            "complete" if ready else "blocked_pending_fresh_oos",
            f"{readiness_path}; {eval_audit_path}",
            {
                "readiness_status": readiness_audit.get("status"),
                "eval_status": eval_audit.get("status"),
                "guarded_fresh_oos_start": sorted(set(readiness.get("guarded_fresh_oos_start", pd.Series(dtype=str)).astype(str))) if not readiness.empty else [],
                "label_rows_after_guard": {
                    str(row["head"]): int(row.get("label_rows_after_guard", 0))
                    for row in readiness.to_dict(orient="records")
                }
                if not readiness.empty
                else {},
                "candidate_score_rows_after_guard": {
                    str(row["head"]): int(row.get("candidate_score_rows_after_guard", 0))
                    for row in readiness.to_dict(orient="records")
                }
                if not readiness.empty
                else {},
            },
        )
    )

    table = pd.DataFrame(rows)
    failed = table.loc[table["status"].isin(["missing_or_failed"])]
    blocked = table.loc[table["status"].eq("blocked_pending_fresh_oos")]
    if not failed.empty:
        status = "failed"
    elif not blocked.empty:
        status = "blocked_pending_fresh_oos"
    else:
        status = "complete"
    audit = {
        "status": status,
        "blocking_condition": ""
        if status != "blocked_pending_fresh_oos"
        else "No guarded fresh labels and frozen candidate score rows after the effective fresh-OOS boundary.",
        "requirements": rows,
    }
    return table, audit


def _write_report(out_dir: Path, table: pd.DataFrame, audit: dict[str, Any]) -> None:
    lines = [
        "# J4/J5 Contextual Meta Goal Completion Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
    ]
    if audit.get("blocking_condition"):
        lines.extend(["Blocking condition:", "", audit["blocking_condition"], ""])
    lines.extend(["## Requirements", "", table.to_markdown(index=False), ""])
    (out_dir / "j4_j5_contextual_meta_goal_completion_audit.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top30-audit", type=Path, default=DEFAULT_TOP30_AUDIT)
    parser.add_argument("--freeze-manifest", type=Path, default=DEFAULT_FREEZE_MANIFEST)
    parser.add_argument("--freeze-audit", type=Path, default=DEFAULT_FREEZE_AUDIT)
    parser.add_argument("--readiness", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--readiness-audit", type=Path, default=DEFAULT_READINESS_AUDIT)
    parser.add_argument("--eval-audit", type=Path, default=DEFAULT_EVAL_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fail-if-incomplete", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    table, audit = build_audit(
        top30_audit_path=args.top30_audit,
        freeze_manifest_path=args.freeze_manifest,
        freeze_audit_path=args.freeze_audit,
        readiness_path=args.readiness,
        readiness_audit_path=args.readiness_audit,
        eval_audit_path=args.eval_audit,
    )
    table.to_csv(args.output_dir / "j4_j5_contextual_meta_goal_completion_requirements.csv", index=False)
    (args.output_dir / "j4_j5_contextual_meta_goal_completion_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=_json_default)
    )
    _write_report(args.output_dir, table, audit)
    print(f"[j4_j5_goal_audit] status={audit['status']} wrote {args.output_dir}", flush=True)
    if args.fail_if_incomplete and audit["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
