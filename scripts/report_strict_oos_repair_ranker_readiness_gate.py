#!/usr/bin/env python3
"""Report whether frozen strict-OOS repair profiles are integration-ready.

This is a diagnostic promotion gate. It does not run models or select profiles.
It consumes the frozen-profile workflow manifest and validation aggregate, then
emits a single machine-readable readiness verdict.
"""

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

from scripts.run_label_quality_proxy_diagnostics import _json_safe  # noqa: E402


DEFAULT_RUNNER_MANIFEST = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_repair_ranker_frozen_profile_run/manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/source_quality_label_walkforward_ablation_v1/"
    "strict_oos_repair_ranker_readiness_gate"
)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_validation_aggregate(runner_manifest: dict[str, Any]) -> pd.DataFrame:
    validation_manifest = runner_manifest.get("validation_manifest", {})
    outputs = validation_manifest.get("outputs", {}) if isinstance(validation_manifest, dict) else {}
    aggregate_path = outputs.get("aggregate")
    if not aggregate_path:
        return pd.DataFrame()
    path = Path(aggregate_path)
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def build_readiness_decision(
    runner_manifest: dict[str, Any],
    validation_aggregate: pd.DataFrame,
    *,
    min_promoted_profiles: int = 1,
    require_reference_audit: bool = True,
) -> dict[str, Any]:
    reasons: list[str] = []
    if runner_manifest.get("scope") != "strict_oos_repair_ranker_frozen_profile_run":
        reasons.append("unexpected_runner_manifest_scope")

    reference = runner_manifest.get("reference_consistency")
    if require_reference_audit and not isinstance(reference, dict):
        reasons.append("reference_audit_missing")
    elif isinstance(reference, dict):
        if reference.get("enabled") is not True:
            reasons.append("reference_audit_disabled")
        if reference.get("status") != "checked":
            reasons.append(f"reference_audit_{reference.get('status', 'unknown')}")
        if reference.get("passes") is not True:
            reasons.append("reference_consistency_failed")

    validation_manifest = runner_manifest.get("validation_manifest", {})
    promotion_allowed_count = _safe_int(
        validation_manifest.get("promotion_allowed_count") if isinstance(validation_manifest, dict) else None
    )
    status_counts = (
        validation_manifest.get("status_counts", {})
        if isinstance(validation_manifest, dict) and isinstance(validation_manifest.get("status_counts"), dict)
        else {}
    )
    if validation_aggregate.empty:
        reasons.append("validation_aggregate_empty")
    if promotion_allowed_count < int(min_promoted_profiles):
        reasons.append("no_profiles_passed_frozen_validation")

    if not validation_aggregate.empty and "validation_status" in validation_aggregate.columns:
        statuses = set(validation_aggregate["validation_status"].dropna().astype(str).tolist())
        if "passes_guards_but_retrospective_only" in statuses:
            reasons.append("retrospective_only_profile_present")
        if "fails_frozen_validation" in statuses:
            reasons.append("frozen_validation_failed")
    if not validation_aggregate.empty and "missing_periods" in validation_aggregate.columns:
        missing = validation_aggregate["missing_periods"].fillna("").astype(str).str.strip()
        if missing.ne("").any():
            reasons.append("missing_validation_periods")

    unique_reasons = list(dict.fromkeys(reasons))
    ready = not unique_reasons
    return {
        "ready_for_training_integration": ready,
        "decision": "ready" if ready else "blocked",
        "block_reasons": unique_reasons,
        "promotion_allowed_count": promotion_allowed_count,
        "min_promoted_profiles": int(min_promoted_profiles),
        "validation_status_counts": status_counts,
        "reference_consistency": reference if isinstance(reference, dict) else {},
        "validation_periods": validation_manifest.get("validation_periods", [])
        if isinstance(validation_manifest, dict)
        else [],
        "history_periods": runner_manifest.get("history_periods", []),
        "profile_count": _safe_int(runner_manifest.get("profile_count")),
    }


def _table(frame: pd.DataFrame, cols: list[str]) -> str:
    if frame.empty:
        return "No rows."
    view = frame[[col for col in cols if col in frame.columns]].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{float(value):.4f}" if pd.notna(value) else "")
    return view.to_markdown(index=False)


def _write_markdown(
    path: Path,
    *,
    runner_manifest_path: Path,
    decision: dict[str, Any],
    validation_aggregate: pd.DataFrame,
) -> None:
    cols = [
        "profile_name",
        "validation_status",
        "expected_periods",
        "observed_periods",
        "missing_periods",
        "mean_repair_u",
        "mean_proxy_u",
        "mean_delta_u_vs_proxy",
        "min_oracle_capture",
        "promotion_allowed",
        "failure_reasons",
    ]
    lines = [
        "# Strict OOS Repair Ranker Readiness Gate",
        "",
        f"- Runner manifest: `{runner_manifest_path}`",
        f"- Decision: `{decision['decision']}`",
        f"- Ready for training integration: `{decision['ready_for_training_integration']}`",
        f"- Block reasons: `{', '.join(decision['block_reasons']) if decision['block_reasons'] else 'none'}`",
        f"- Promotion allowed profiles: `{decision['promotion_allowed_count']}`",
        f"- Validation periods: `{', '.join(map(str, decision.get('validation_periods', [])))}`",
        f"- History periods: `{', '.join(map(str, decision.get('history_periods', [])))}`",
        "",
        "## Reference Consistency",
        "",
        f"- Status: `{decision.get('reference_consistency', {}).get('status', 'unknown')}`",
        f"- Passes: `{decision.get('reference_consistency', {}).get('passes')}`",
        f"- Rows checked: `{decision.get('reference_consistency', {}).get('rows_checked')}`",
        "",
        "## Validation Profiles",
        "",
        _table(validation_aggregate, cols),
        "",
        "## Interpretation",
        "",
        "- `ready` means the frozen profile path reproduced reference rows and at least one profile passed non-retrospective frozen validation.",
        "- `blocked` means training integration must remain disabled.",
        "",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_gate(
    *,
    runner_manifest_path: Path,
    output_dir: Path,
    min_promoted_profiles: int,
    require_reference_audit: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    runner_manifest = _load_json(runner_manifest_path)
    validation_aggregate = _load_validation_aggregate(runner_manifest)
    decision = build_readiness_decision(
        runner_manifest,
        validation_aggregate,
        min_promoted_profiles=min_promoted_profiles,
        require_reference_audit=require_reference_audit,
    )
    outputs = {
        "json": output_dir / "strict_oos_repair_ranker_readiness_gate.json",
        "markdown": output_dir / "strict_oos_repair_ranker_readiness_gate.md",
        "validation_profiles": output_dir / "strict_oos_repair_ranker_readiness_validation_profiles.csv",
    }
    validation_aggregate.to_csv(outputs["validation_profiles"], index=False)
    result = {
        "scope": "strict_oos_repair_ranker_readiness_gate",
        "runner_manifest_path": str(runner_manifest_path),
        **decision,
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    outputs["json"].write_text(json.dumps(_json_safe(result), indent=2), encoding="utf-8")
    _write_markdown(
        outputs["markdown"],
        runner_manifest_path=runner_manifest_path,
        decision=result,
        validation_aggregate=validation_aggregate,
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-manifest", type=Path, default=DEFAULT_RUNNER_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-promoted-profiles", type=int, default=1)
    parser.add_argument("--allow-missing-reference-audit", action="store_true")
    parser.add_argument("--fail-on-blocked", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_gate(
        runner_manifest_path=args.runner_manifest,
        output_dir=args.output_dir,
        min_promoted_profiles=args.min_promoted_profiles,
        require_reference_audit=not args.allow_missing_reference_audit,
    )
    print(json.dumps(_json_safe(result), indent=2))
    if args.fail_on_blocked and not result["ready_for_training_integration"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
