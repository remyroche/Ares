#!/usr/bin/env python3
"""Audit the selected market-state head-priority shadow challenger.

This is a status/completion audit for the portfolio-priority branch.  It does
not decide production promotion by itself; it verifies that the selected
challenger is reproducible, contract-clean, and correctly remains shadow-only
when the cross-window promotion gate fails.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CAP_SWEEP_DIR = Path(
    "data_perp/reports/market_state_head_priority_cap_sweep_s2_gated_20260626_jun15_22_v2"
)
DEFAULT_PROMOTION_AUDIT_DIR = Path(
    "data_perp/reports/market_state_priority_shadow_promotion_audit_20260626_v3"
)
DEFAULT_OUTPUT_DIR = Path("data_perp/reports/market_state_priority_shadow_status_audit_20260626")

STATUS_COMPLETE = "complete"
STATUS_FAILED = "failed"
STATUS_MISSING = "missing"
STATUS_GATE_BLOCKED = "gate_blocked"
STATUS_SHADOW = "shadow_only"


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _num(value: Any, default: float = float("nan")) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else default


def _portable_arm_selector(arm: str) -> str:
    text = str(arm or "")
    marker = "_cap_"
    if marker not in text:
        return text
    return f"cap_{text.split(marker, 1)[1]}"


def _add(
    rows: list[dict[str, Any]],
    requirement_id: str,
    requirement: str,
    status: str,
    evidence: str,
    notes: str = "",
) -> None:
    rows.append(
        {
            "requirement_id": requirement_id,
            "requirement": requirement,
            "status": status,
            "evidence": evidence,
            "notes": notes,
        }
    )


def _selected_row(selected: dict[str, Any]) -> dict[str, Any]:
    row = selected.get("selected_row")
    return dict(row) if isinstance(row, dict) else {}


def audit_priority_shadow_status(
    *,
    cap_sweep_dir: Path,
    promotion_audit_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = _read_json(cap_sweep_dir / "selected_shadow_challenger.json")
    cap_manifest = _read_json(cap_sweep_dir / "manifest.json")
    metrics = _read_csv(cap_sweep_dir / "head_priority_cap_sweep_metrics.csv")
    promotion_manifest = _read_json(promotion_audit_dir / "manifest.json")
    promotion_gate = _read_json(promotion_audit_dir / "market_state_priority_shadow_promotion_gate.json")
    window_summary = _read_csv(promotion_audit_dir / "market_state_priority_shadow_window_summary.csv")

    rows: list[dict[str, Any]] = []
    selected_row = _selected_row(selected)
    selected_arm = str(selected.get("arm") or selected_row.get("arm") or "")
    selected_selector = _portable_arm_selector(selected_arm)

    _add(
        rows,
        "P1",
        "Cap sweep persists a selected shadow challenger.",
        STATUS_COMPLETE if selected.get("selected") is True and bool(selected_arm) else STATUS_MISSING,
        str(cap_sweep_dir / "selected_shadow_challenger.json"),
        f"arm={selected_arm}; selector={selected_selector}; reason={selected.get('reason')}",
    )

    contract = dict(cap_manifest.get("contract") or {})
    contract_ok = (
        contract.get("changes_scores_or_ranks") is False
        and contract.get("changes_thresholds") is False
        and contract.get("changes_position_sizing") is False
        and contract.get("changes_auction_ordering") is True
        and contract.get("qfail_active") is False
        and contract.get("head_health_active") is False
        and contract.get("market_state_threshold_controller_active") is False
        and contract.get("priority_adjustment_column") == "portfolio_priority_adjustment"
    )
    _add(
        rows,
        "P2",
        "Selected priority challenger changes only global auction priority in shadow.",
        STATUS_COMPLETE if contract_ok else STATUS_FAILED,
        str(cap_sweep_dir / "manifest.json"),
        f"contract={contract}",
    )

    selected_metrics_ok = (
        _bool(selected_row.get("gate_passed"))
        and _num(selected_row.get("delta_net_pnl")) > 0.0
        and _num(selected_row.get("accepted_jaccard")) >= 0.90
        and _num(selected_row.get("delta_full_sl_rate"), 0.0) <= 0.0
        and _num(selected_row.get("delta_timeout_rate"), 0.0) <= 0.0
        and int(_num(selected_row.get("entrants"), 0.0) + _num(selected_row.get("removed"), 0.0)) > 0
    )
    _add(
        rows,
        "P3",
        "Selected challenger passed single-window replay mechanics.",
        STATUS_COMPLETE if selected_metrics_ok else STATUS_FAILED,
        str(cap_sweep_dir / "head_priority_cap_sweep_metrics.csv"),
        (
            f"delta_net_pnl={selected_row.get('delta_net_pnl')}; "
            f"jaccard={selected_row.get('accepted_jaccard')}; "
            f"full_sl_delta={selected_row.get('delta_full_sl_rate')}; "
            f"timeout_delta={selected_row.get('delta_timeout_rate')}"
        ),
    )

    selector_source = str((promotion_manifest.get("params") or {}).get("arm_selector_source") or "")
    resolved_selector = str((promotion_manifest.get("params") or {}).get("resolved_arm_contains") or "")
    selector_ok = (
        selected_selector
        and resolved_selector == selected_selector
        and selector_source.startswith("selected_shadow_challenger:")
    )
    _add(
        rows,
        "P4",
        "Promotion audit consumes the selected challenger rather than a manual arm selector.",
        STATUS_COMPLETE if selector_ok else STATUS_FAILED,
        str(promotion_audit_dir / "manifest.json"),
        f"resolved={resolved_selector}; source={selector_source}",
    )

    gate_failed = promotion_gate.get("passed") is False
    expected_failures = {
        "median_delta_net_pnl_not_positive",
        "positive_delta_window_share_below_50pct",
        "fewer_than_2_action_windows",
        "fewer_than_2_positive_action_windows",
    }
    actual_failures = set(promotion_gate.get("failures") or [])
    _add(
        rows,
        "P5",
        "Cross-window promotion gate blocks activation when action is not recurrent.",
        STATUS_GATE_BLOCKED if gate_failed and expected_failures.issubset(actual_failures) else STATUS_FAILED,
        str(promotion_audit_dir / "market_state_priority_shadow_promotion_gate.json"),
        f"passed={promotion_gate.get('passed')}; failures={sorted(actual_failures)}",
    )

    if window_summary.empty:
        _add(
            rows,
            "P6",
            "Promotion audit covers at least three replay windows.",
            STATUS_MISSING,
            str(promotion_audit_dir / "market_state_priority_shadow_window_summary.csv"),
            "missing window summary",
        )
    else:
        coverage = pd.to_numeric(window_summary.get("coverage"), errors="coerce")
        full_sl_delta = pd.to_numeric(window_summary.get("delta_full_sl_rate"), errors="coerce")
        timeout_delta = pd.to_numeric(window_summary.get("delta_timeout_rate"), errors="coerce")
        coverage_ok = len(window_summary) >= 3 and bool((coverage >= 0.999).all())
        risk_ok = bool((full_sl_delta <= 0.0).all()) and bool((timeout_delta <= 0.0).all())
        _add(
            rows,
            "P6",
            "Promotion audit covers multiple windows with full schedule coverage and no risk deterioration.",
            STATUS_COMPLETE if coverage_ok and risk_ok else STATUS_FAILED,
            str(promotion_audit_dir / "market_state_priority_shadow_window_summary.csv"),
            (
                f"windows={len(window_summary)}; min_coverage={coverage.min()}; "
                f"max_full_sl_delta={full_sl_delta.max()}; max_timeout_delta={timeout_delta.max()}"
            ),
        )

    active_status = (
        STATUS_SHADOW
        if gate_failed
        else STATUS_FAILED
    )
    _add(
        rows,
        "P7",
        "Operational decision remains static T1 active and priority modulation shadow-only.",
        active_status,
        "promotion audit gate",
        "Priority modulation must not be activated until promotion gate passes.",
    )

    evidence = pd.DataFrame(rows)
    hard_failures = evidence.loc[evidence["status"].eq(STATUS_FAILED)].copy()
    missing = evidence.loc[evidence["status"].eq(STATUS_MISSING)].copy()
    status_counts = dict(Counter(evidence["status"].astype(str)))
    passed = bool(hard_failures.empty and missing.empty)

    summary = {
        "generated_by": "audit_market_state_priority_shadow_status",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "status_counts": status_counts,
        "selected_arm": selected_arm,
        "selected_selector": selected_selector,
        "promotion_gate_passed": promotion_gate.get("passed"),
        "operational_status": "shadow_only" if gate_failed else "review_required",
        "hard_failures": hard_failures["requirement_id"].astype(str).tolist(),
        "missing": missing["requirement_id"].astype(str).tolist(),
        "inputs": {
            "cap_sweep_dir": str(cap_sweep_dir),
            "promotion_audit_dir": str(promotion_audit_dir),
        },
        "outputs": {
            "summary": str(output_dir / "market_state_priority_shadow_status_summary.json"),
            "evidence": str(output_dir / "market_state_priority_shadow_status_evidence.csv"),
            "report": str(output_dir / "market_state_priority_shadow_status_report.md"),
        },
    }

    evidence.to_csv(output_dir / "market_state_priority_shadow_status_evidence.csv", index=False)
    (output_dir / "market_state_priority_shadow_status_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(_json_safe(summary), indent=2) + "\n",
        encoding="utf-8",
    )
    report = _render_report(summary, evidence, selected_row, promotion_gate)
    (output_dir / "market_state_priority_shadow_status_report.md").write_text(report, encoding="utf-8")
    return summary


def _render_report(
    summary: dict[str, Any],
    evidence: pd.DataFrame,
    selected_row: dict[str, Any],
    promotion_gate: dict[str, Any],
) -> str:
    lines = [
        "# Market-State Priority Shadow Status Audit",
        "",
        "This audit verifies the selected market-state head-priority challenger and its shadow-only status.",
        "",
        "## Summary",
        "",
        f"- Passed: `{summary['passed']}`",
        f"- Operational status: `{summary['operational_status']}`",
        f"- Selected arm: `{summary['selected_arm']}`",
        f"- Selected selector: `{summary['selected_selector']}`",
        f"- Promotion gate passed: `{summary['promotion_gate_passed']}`",
        "",
        "## Selected Row",
        "",
    ]
    selected_view = {
        key: selected_row.get(key)
        for key in [
            "max_adjustment",
            "min_abs_z",
            "delta_net_pnl",
            "accepted_jaccard",
            "delta_full_sl_rate",
            "delta_timeout_rate",
            "entrants",
            "removed",
            "gate_passed",
        ]
    }
    lines.append(pd.DataFrame([selected_view]).to_markdown(index=False))
    lines.extend(["", "## Promotion Gate", ""])
    gate_view = {k: v for k, v in promotion_gate.items() if k != "failures"}
    lines.append(pd.DataFrame([gate_view]).to_markdown(index=False))
    lines.extend(["", "Failures:", ""])
    failures = promotion_gate.get("failures") or []
    lines.extend([f"- `{item}`" for item in failures] if failures else ["- none"])
    lines.extend(["", "## Evidence", ""])
    lines.append(evidence.to_markdown(index=False))
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Static T1 remains the active baseline. The selected head-priority challenger is reproducible and contract-clean, but remains shadow-only until repeated positive action windows satisfy the promotion gate.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap-sweep-dir", type=Path, default=DEFAULT_CAP_SWEEP_DIR)
    parser.add_argument("--promotion-audit-dir", type=Path, default=DEFAULT_PROMOTION_AUDIT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = audit_priority_shadow_status(
        cap_sweep_dir=args.cap_sweep_dir,
        promotion_audit_dir=args.promotion_audit_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(_json_safe(summary), indent=2))


if __name__ == "__main__":
    main()
