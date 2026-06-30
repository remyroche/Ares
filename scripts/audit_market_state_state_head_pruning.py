#!/usr/bin/env python3
"""Audit market-state activation registry pruning decisions.

The threshold controller can only become executable after state heads are
validated as useful, non-redundant, and defensively helpful. This script turns
`market_state_activation_registry.csv` into a compact, reproducible pruning
audit and verifies that recommended statuses follow the documented gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_ARTIFACT_DIR = Path(
    "data_perp/reports/market_state_threshold_controller_walkforward_20260626_t1_lgbm_strict_defensive_v3"
)
DEFAULT_OUTPUT_DIR = Path(
    "data_perp/reports/market_state_state_head_pruning_audit_20260626"
)
REQUIRED_COLUMNS = [
    "state_level",
    "state_head",
    "component_group",
    "aggregate_status",
    "recommended_status",
    "activation_disable_reason",
    "forecast_skill_gate_pass",
    "response_gate_pass",
    "action_gate_pass",
    "leave_one_out_gate_pass",
    "defensive_action_gate_pass",
    "loo_median_increment_net_pnl",
    "loo_q25_increment_net_pnl",
    "loo_positive_increment_share",
    "loo_state_head_defensive_success",
    "loo_state_head_loss_avoided",
    "loo_state_head_winner_pnl_sacrificed",
    "activation_registry_version",
]
ALLOWED_STATUSES = {"active_candidate", "disabled_candidate", "shadow", "shadow_candidate"}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _num(value: Any, default: float = np.nan) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else float(default)


def _expected_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    aggregate_status = str(row.get("aggregate_status", ""))
    if aggregate_status != "active":
        reasons.append(aggregate_status or "not_active")
    if str(row.get("state_level")) == "forecast" and not _bool(row.get("forecast_skill_gate_pass")):
        reasons.append("weak_or_unstable_forecast_skill")
    if _bool(row.get("redundancy_flag")) and not _bool(row.get("response_gate_pass")):
        reasons.append("redundant_without_response_effect")
    if not _bool(row.get("response_gate_pass")):
        reasons.append("weak_response_effect")
    if not _bool(row.get("action_gate_pass")):
        reasons.append("no_material_threshold_action")
    if not _bool(row.get("leave_one_out_gate_pass")):
        reasons.append("no_positive_leave_one_out_increment")
    if not _bool(row.get("defensive_action_gate_pass")):
        reasons.append("state_action_sacrifices_winners")
    return list(dict.fromkeys(reasons))


def _expected_status(row: dict[str, Any], reasons: list[str]) -> str:
    aggregate_status = str(row.get("aggregate_status", ""))
    if aggregate_status != "active":
        return "shadow"
    if (
        "redundant_without_response_effect" in reasons
        or "weak_or_unstable_forecast_skill" in reasons
        or "no_positive_leave_one_out_increment" in reasons
        or "state_action_sacrifices_winners" in reasons
    ):
        return "disabled_candidate"
    if "weak_response_effect" in reasons and "no_material_threshold_action" in reasons:
        return "shadow"
    return "active_candidate"


def _split_reasons(value: Any) -> set[str]:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return set()
    return {part for part in str(value).split(";") if part and part.lower() != "nan"}


def audit_state_head_pruning(
    artifact_dir: Path,
    *,
    min_active_loo_q25: float = 0.0,
) -> tuple[dict[str, Any], pd.DataFrame]:
    registry_path = artifact_dir / "market_state_activation_registry.csv"
    registry = _read_csv(registry_path)
    manifest = _read_json(artifact_dir / "manifest.json")
    failures: list[str] = []
    if registry.empty:
        failures.append(f"{registry_path} is missing or empty")
        return {
            "generated_by": "audit_market_state_state_head_pruning",
            "artifact_dir": str(artifact_dir),
            "passed": False,
            "failures": failures,
        }, registry

    missing = [col for col in REQUIRED_COLUMNS if col not in registry.columns]
    if missing:
        failures.append(f"market_state_activation_registry missing columns: {missing}")

    if "activation_registry_version" in registry.columns:
        versions = set(registry["activation_registry_version"].dropna().astype(str).unique())
        if versions != {"market_state_activation_registry_v1"}:
            failures.append(f"unexpected activation_registry_version values: {sorted(versions)}")

    status_values = set(registry.get("recommended_status", pd.Series(dtype=str)).dropna().astype(str).unique())
    unknown = sorted(status_values - ALLOWED_STATUSES)
    if unknown:
        failures.append(f"unexpected recommended_status values: {unknown}")

    rows: list[dict[str, Any]] = []
    for _, row in registry.iterrows():
        record = row.to_dict()
        expected_reasons = _expected_reasons(record)
        expected_status = _expected_status(record, expected_reasons)
        actual_status = str(record.get("recommended_status", ""))
        actual_reasons = _split_reasons(record.get("activation_disable_reason"))
        reason_mismatch = expected_status != "active_candidate" and set(expected_reasons) != actual_reasons
        status_mismatch = actual_status != expected_status
        if status_mismatch:
            failures.append(
                f"{record.get('state_head')} recommended_status={actual_status} expected {expected_status}"
            )
        if reason_mismatch:
            failures.append(
                f"{record.get('state_head')} activation_disable_reason mismatch: "
                f"{sorted(actual_reasons)} expected {sorted(expected_reasons)}"
            )
        if actual_status == "active_candidate":
            q25 = _num(record.get("loo_q25_increment_net_pnl"), np.nan)
            if not np.isfinite(q25) or q25 < float(min_active_loo_q25):
                failures.append(f"{record.get('state_head')} active_candidate q25 LOO increment below gate")
            if _split_reasons(record.get("activation_disable_reason")):
                failures.append(f"{record.get('state_head')} active_candidate has disable reasons")
        rows.append(
            {
                "state_level": record.get("state_level"),
                "state_head": record.get("state_head"),
                "component_group": record.get("component_group"),
                "recommended_status": actual_status,
                "expected_status": expected_status,
                "status_mismatch": bool(status_mismatch),
                "reason_mismatch": bool(reason_mismatch),
                "activation_disable_reason": record.get("activation_disable_reason"),
                "loo_median_increment_net_pnl": _num(record.get("loo_median_increment_net_pnl"), np.nan),
                "loo_q25_increment_net_pnl": _num(record.get("loo_q25_increment_net_pnl"), np.nan),
                "loo_positive_increment_share": _num(record.get("loo_positive_increment_share"), np.nan),
                "loo_state_head_defensive_success": _num(record.get("loo_state_head_defensive_success"), np.nan),
                "loo_state_head_loss_avoided": _num(record.get("loo_state_head_loss_avoided"), np.nan),
                "loo_state_head_winner_pnl_sacrificed": _num(
                    record.get("loo_state_head_winner_pnl_sacrificed"), np.nan
                ),
                "max_abs_spearman_corr": _num(record.get("max_abs_spearman_corr"), np.nan),
                "redundant_with": record.get("redundant_with"),
                "redundancy_group": record.get("redundancy_group"),
                "forecast_skill_gate_pass": _bool(record.get("forecast_skill_gate_pass")),
                "response_gate_pass": _bool(record.get("response_gate_pass")),
                "action_gate_pass": _bool(record.get("action_gate_pass")),
                "leave_one_out_gate_pass": _bool(record.get("leave_one_out_gate_pass")),
                "defensive_action_gate_pass": _bool(record.get("defensive_action_gate_pass")),
            }
        )
    audit = pd.DataFrame(rows)
    reason_counter: Counter[str] = Counter()
    for value in audit["activation_disable_reason"].dropna().astype(str):
        for reason in value.split(";"):
            if reason and reason.lower() != "nan":
                reason_counter[reason] += 1

    active = audit.loc[audit["recommended_status"].astype(str).eq("active_candidate")].copy()
    disabled = audit.loc[audit["recommended_status"].astype(str).eq("disabled_candidate")].copy()
    shadow = audit.loc[audit["recommended_status"].astype(str).str.contains("shadow", regex=False)].copy()
    payload = {
        "generated_by": "audit_market_state_state_head_pruning",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "market_mode": manifest.get("market_mode"),
        "passed": not failures,
        "failures": failures,
        "registry_rows": int(len(registry)),
        "status_counts": registry["recommended_status"].astype(str).value_counts().to_dict(),
        "active_candidate_count": int(len(active)),
        "disabled_candidate_count": int(len(disabled)),
        "shadow_count": int(len(shadow)),
        "active_candidates": active.to_dict("records"),
        "top_disabled_by_loo_penalty": disabled.sort_values(
            ["loo_q25_increment_net_pnl", "loo_median_increment_net_pnl"],
            ascending=[True, True],
        )
        .head(10)
        .to_dict("records"),
        "disable_reason_counts": dict(reason_counter),
        "selection_context": {
            "controller_promotion_allowed": False,
            "reason": "state-head pruning audit only; controller promotion is handled by controller promotion gates",
        },
    }
    return payload, audit


def _render_report(payload: dict[str, Any], audit: pd.DataFrame) -> str:
    lines = [
        "# Market-State State-Head Pruning Audit",
        "",
        f"Artifact dir: `{payload['artifact_dir']}`",
        f"Passed: `{payload['passed']}`",
        f"Registry rows: `{payload['registry_rows']}`",
        f"Active candidates: `{payload['active_candidate_count']}`",
        f"Disabled candidates: `{payload['disabled_candidate_count']}`",
        f"Shadow heads: `{payload['shadow_count']}`",
        "",
        "## Status Counts",
        "",
    ]
    lines.append(pd.DataFrame([payload["status_counts"]]).to_markdown(index=False))
    lines.extend(["", "## Active Candidates", ""])
    active = audit.loc[audit["recommended_status"].astype(str).eq("active_candidate")]
    if active.empty:
        lines.append("_None._")
    else:
        cols = [
            "state_level",
            "state_head",
            "component_group",
            "loo_median_increment_net_pnl",
            "loo_q25_increment_net_pnl",
            "loo_positive_increment_share",
            "loo_state_head_defensive_success",
            "loo_state_head_loss_avoided",
            "loo_state_head_winner_pnl_sacrificed",
        ]
        lines.append(active[[c for c in cols if c in active.columns]].to_markdown(index=False))
    lines.extend(["", "## Disable Reasons", ""])
    reason_counts = payload.get("disable_reason_counts") or {}
    if reason_counts:
        reason_df = pd.DataFrame(
            [{"reason": key, "count": value} for key, value in sorted(reason_counts.items())]
        )
        lines.append(reason_df.to_markdown(index=False))
    else:
        lines.append("_None._")
    lines.extend(["", "## Worst Disabled Heads By LOO Q25", ""])
    disabled = audit.loc[audit["recommended_status"].astype(str).eq("disabled_candidate")]
    if disabled.empty:
        lines.append("_None._")
    else:
        cols = [
            "state_head",
            "activation_disable_reason",
            "loo_q25_increment_net_pnl",
            "loo_median_increment_net_pnl",
            "loo_state_head_defensive_success",
            "max_abs_spearman_corr",
            "redundant_with",
        ]
        view = disabled.sort_values(
            ["loo_q25_increment_net_pnl", "loo_median_increment_net_pnl"],
            ascending=[True, True],
        ).head(10)
        lines.append(view[[c for c in cols if c in view.columns]].to_markdown(index=False))
    lines.extend(["", "## Failures", ""])
    failures = payload.get("failures") or []
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("_None._")
    return "\n".join(lines) + "\n"


def write_pruning_audit(payload: dict[str, Any], audit: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_dir / "market_state_state_head_pruning_audit.json",
        "csv": output_dir / "market_state_state_head_pruning_audit.csv",
        "report": output_dir / "market_state_state_head_pruning_audit.md",
    }
    paths["json"].write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    audit.to_csv(paths["csv"], index=False)
    paths["report"].write_text(_render_report(payload, audit), encoding="utf-8")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-active-loo-q25", type=float, default=0.0)
    args = parser.parse_args()

    payload, audit = audit_state_head_pruning(
        args.artifact_dir,
        min_active_loo_q25=float(args.min_active_loo_q25),
    )
    paths = write_pruning_audit(payload, audit, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "passed": bool(payload["passed"]),
                "active_candidate_count": int(payload["active_candidate_count"]),
                "disabled_candidate_count": int(payload["disabled_candidate_count"]),
                "report": str(paths["report"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
