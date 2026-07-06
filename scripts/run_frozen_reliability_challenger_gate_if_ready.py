#!/usr/bin/env python3
"""Run frozen reliability challengers only when fresh evidence is ready.

The runner reads ``frozen_reliability_challenger_bundle_v1.json``-style
bundles, scans flat candidate sources for post-cutoff coverage, and either
blocks with explicit readiness reasons or runs the frozen conditional-filter
rules through replay, decision-pack reporting, and bootstrap confidence.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from scripts.run_latest_frozen_dual_scoring_gate_if_ready import _scan_one, _select_source


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATES = (
    Path(
        "data_perp/reports/contextual_tp_sl_ablation_workflow_v15_eligible_head_gate_20260701/"
        "cumulative_ledger/cumulative_flat_candidates.parquet"
    ),
    Path(
        "data_perp/reports/contextual_tp_sl_ablation_workflow_v13_reliability_family_staleness_20260701/"
        "cumulative_ledger/cumulative_flat_candidates.parquet"
    ),
    Path(
        "data_perp/reports/contextual_tp_sl_ablation_workflow_v9_policy_evidence_by_head_20260701/"
        "cumulative_ledger/cumulative_flat_candidates.parquet"
    ),
)


def _json_safe(value: Any) -> Any:
    if not isinstance(value, (dict, list, tuple)):
        try:
            missing = pd.isna(value)
        except Exception:
            missing = False
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if np.isfinite(val) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _load_bundle(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload.get("rules"), dict) or not payload["rules"]:
        raise ValueError(f"Bundle `{path}` has no frozen rules")
    return payload


def _scan_args(bundle: Dict[str, Any], args: argparse.Namespace) -> SimpleNamespace:
    req = bundle.get("forward_validation_requirements") or {}
    required_heads = list(req.get("required_matured_outcome_heads") or [])
    return SimpleNamespace(
        min_post_cutoff_rows=int(args.min_post_cutoff_rows or req.get("minimum_post_cutoff_rows", 2000)),
        min_post_cutoff_timestamps=int(
            args.min_post_cutoff_timestamps or req.get("minimum_post_cutoff_timestamps", 40)
        ),
        min_post_cutoff_active_heads=int(args.min_post_cutoff_active_heads or 3),
        min_policy_action_rows=int(args.min_policy_action_rows or req.get("minimum_policy_action_rows", 50)),
        min_policy_action_timestamps=int(args.min_policy_action_timestamps or 10),
        min_policy_outcome_rows=int(args.min_policy_outcome_rows or req.get("minimum_policy_outcome_rows", 50)),
        min_policy_outcome_timestamps=int(args.min_policy_outcome_timestamps or 10),
        min_policy_outcome_rows_per_action_head=int(args.min_policy_outcome_rows_per_action_head or 0),
        required_policy_outcome_head=required_heads,
        min_policy_outcome_rows_per_required_head=int(
            args.min_policy_outcome_rows_per_required_head
            or req.get("minimum_policy_outcome_rows_per_required_head", 3)
        ),
        min_diagnostic_group_features=int(args.min_diagnostic_group_features),
        min_diagnostic_group_finite_rate=float(args.min_diagnostic_group_finite_rate),
    )


def _candidate_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = [Path(item) for item in (args.candidate or [])]
    if not paths:
        paths.extend(DEFAULT_CANDIDATES)
    for root_item in args.root or []:
        root = Path(root_item)
        if not root.exists():
            continue
        paths.extend(root.glob("contextual_tp_sl_ablation_workflow_*/cumulative_ledger/cumulative_flat_candidates.parquet"))

    out: List[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _rule_file(bundle: Dict[str, Any], out_dir: Path) -> Path:
    rules: Dict[str, Dict[str, Any]] = {"none": {}}
    for rule_id, spec in (bundle.get("rules") or {}).items():
        rules[str(rule_id)] = {
            "heads": list(spec.get("heads") or []),
            "condition": str(spec.get("condition")),
            "action": str(spec.get("action")),
            "value": float(spec.get("value")),
        }
    path = out_dir / "frozen_reliability_rules.json"
    path.write_text(json.dumps(_json_safe(rules), indent=2, sort_keys=True) + "\n")
    return path


def _run(cmd: List[str]) -> None:
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def _parse_count_map(value: Any) -> Dict[str, int]:
    if isinstance(value, dict):
        return {str(k): int(v) for k, v in value.items()}
    if value is None:
        return {}
    try:
        payload = json.loads(str(value))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    out: Dict[str, int] = {}
    for key, raw in payload.items():
        try:
            out[str(key)] = int(raw)
        except Exception:
            out[str(key)] = 0
    return out


def _readiness_deficit_rows(bundle: Dict[str, Any], args: argparse.Namespace, scan: pd.DataFrame) -> pd.DataFrame:
    if scan.empty:
        return pd.DataFrame()
    scan_args = _scan_args(bundle, args)
    nearest = _select_source(scan, force=True)
    if nearest is None:
        return pd.DataFrame()

    checks = [
        ("post_cutoff_rows", nearest.get("post_cutoff_rows"), scan_args.min_post_cutoff_rows),
        ("post_cutoff_timestamps", nearest.get("post_cutoff_timestamps"), scan_args.min_post_cutoff_timestamps),
        ("post_cutoff_active_heads", nearest.get("post_cutoff_active_heads"), scan_args.min_post_cutoff_active_heads),
        ("policy_action_rows", nearest.get("policy_action_rows_estimate"), scan_args.min_policy_action_rows),
        (
            "policy_action_timestamps",
            nearest.get("policy_action_timestamps_estimate"),
            scan_args.min_policy_action_timestamps,
        ),
        ("policy_outcome_rows", nearest.get("policy_outcome_rows_estimate"), scan_args.min_policy_outcome_rows),
        (
            "policy_outcome_timestamps",
            nearest.get("policy_outcome_timestamps_estimate"),
            scan_args.min_policy_outcome_timestamps,
        ),
    ]
    rows: List[Dict[str, Any]] = []
    for gate_name, observed_raw, required_raw in checks:
        observed = int(observed_raw or 0)
        required = int(required_raw or 0)
        rows.append(
            {
                "source_path": nearest.get("path"),
                "gate": gate_name,
                "head": "",
                "observed": observed,
                "required": required,
                "deficit": max(required - observed, 0),
                "pass": observed >= required,
            }
        )

    outcome_counts = _parse_count_map(nearest.get("policy_outcome_head_counts"))
    for head in scan_args.required_policy_outcome_head:
        observed = int(outcome_counts.get(str(head), 0))
        required = int(scan_args.min_policy_outcome_rows_per_required_head)
        rows.append(
            {
                "source_path": nearest.get("path"),
                "gate": "policy_outcome_rows_per_required_head",
                "head": str(head),
                "observed": observed,
                "required": required,
                "deficit": max(required - observed, 0),
                "pass": observed >= required,
            }
        )
    return pd.DataFrame(rows).sort_values(["pass", "deficit"], ascending=[True, False])


def _run_reports(args: argparse.Namespace, bundle: Dict[str, Any], source: pd.Series, out_dir: Path) -> Dict[str, Any]:
    rule_path = _rule_file(bundle, out_dir)
    replay_dir = out_dir / "conditional_filter_replay"
    decision_dir = out_dir / "decision_pack"
    bootstrap_dir = out_dir / "bootstrap_confidence"
    rules = [str(rule_id) for rule_id in (bundle.get("rules") or {})]

    _run(
        [
            sys.executable,
            "scripts/ablate_contextual_tp_sl_conditional_head_filters.py",
            "--flat-candidate-table",
            str(source["path"]),
            "--out-dir",
            str(replay_dir),
            "--rule-file",
            str(rule_path),
            "--threshold-mode",
            str(args.threshold_mode),
            "--min-threshold-history",
            str(args.min_threshold_history),
            "--save-accepted-decisions",
        ]
    )
    decision_cmd = [
        sys.executable,
        "scripts/report_conditional_filter_decision_pack.py",
        "--attribution-dir",
        str(replay_dir),
        "--out-dir",
        str(decision_dir),
        "--baseline-rule",
        str(bundle.get("baseline_rule", "none")),
    ]
    for rule in rules:
        decision_cmd.extend(["--rule", rule])
    _run(decision_cmd)

    bootstrap_cmd = [
        sys.executable,
        "scripts/report_conditional_filter_bootstrap_confidence.py",
        "--attribution-dir",
        str(replay_dir),
        "--out-dir",
        str(bootstrap_dir),
        "--baseline-rule",
        str(bundle.get("baseline_rule", "none")),
        "--n-bootstrap",
        str(args.n_bootstrap),
        "--seed",
        str(args.seed),
    ]
    for rule in rules:
        bootstrap_cmd.extend(["--rule", rule])
    _run(bootstrap_cmd)
    return {
        "rule_file": str(rule_path),
        "replay_dir": str(replay_dir),
        "decision_pack_dir": str(decision_dir),
        "bootstrap_confidence_dir": str(bootstrap_dir),
    }


def _write_report(
    out_dir: Path,
    args: argparse.Namespace,
    bundle: Dict[str, Any],
    scan: pd.DataFrame,
    selected: pd.Series | None,
    ran_outputs: Dict[str, Any] | None,
) -> None:
    scan.to_csv(out_dir / "frozen_reliability_source_scan.csv", index=False)
    readiness_deficits = _readiness_deficit_rows(bundle, args, scan)
    if not readiness_deficits.empty:
        readiness_deficits.to_csv(out_dir / "frozen_reliability_readiness_deficits.csv", index=False)
    ready_count = int(scan["ready"].sum()) if "ready" in scan.columns else 0
    nearest = _select_source(scan, force=True)
    payload = {
        "generated_by": Path(__file__).name,
        "bundle": str(args.bundle),
        "bundle_id": bundle.get("bundle_id"),
        "cutoff": str(args.cutoff),
        "ready_sources": ready_count,
        "ran_gate": ran_outputs is not None,
        "selected_source": selected.to_dict() if selected is not None else None,
        "nearest_source": nearest.to_dict() if nearest is not None else None,
        "readiness_deficits": readiness_deficits.to_dict(orient="records"),
        "outputs": ran_outputs or {},
    }
    (out_dir / "frozen_reliability_gate_manifest.json").write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n"
    )
    lines = [
        "# Frozen Reliability Challenger Gate",
        "",
        f"Bundle: `{args.bundle}`",
        f"Bundle id: `{bundle.get('bundle_id')}`",
        f"Cutoff: `{args.cutoff}`",
        f"Ready sources: `{ready_count}`",
        f"Ran gate: `{ran_outputs is not None}`",
        "",
        "## Frozen Rules",
        "",
        pd.DataFrame(
            [
                {
                    "rule_id": rule_id,
                    "role": spec.get("role"),
                    "heads": ",".join(spec.get("heads") or []),
                    "condition": spec.get("condition"),
                    "action": spec.get("action"),
                    "value": spec.get("value"),
                }
                for rule_id, spec in (bundle.get("rules") or {}).items()
            ]
        ).to_markdown(index=False),
        "",
        "## Selected Source",
        "",
    ]
    if selected is None:
        lines.append("_No source met the frozen challenger readiness requirements._")
    else:
        lines.append(pd.DataFrame([selected.to_dict()]).to_markdown(index=False))
    if nearest is not None:
        lines.extend(["", "## Nearest Source", "", pd.DataFrame([nearest.to_dict()]).to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## Readiness Deficits",
            "",
            readiness_deficits.to_markdown(index=False) if not readiness_deficits.empty else "_No deficits._",
        ]
    )
    if ran_outputs:
        lines.extend(["", "## Outputs", "", pd.DataFrame([ran_outputs]).to_markdown(index=False)])
    lines.extend(
        [
            "",
            "## Source Scan",
            "",
            scan.sort_values("post_cutoff_rows", ascending=False).head(30).to_markdown(index=False)
            if not scan.empty
            else "_No sources scanned._",
        ]
    )
    (out_dir / "frozen_reliability_gate_report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="Flat candidate parquet to scan. Repeatable. Defaults to recent cumulative reliability ledgers.",
    )
    parser.add_argument(
        "--root",
        action="append",
        default=None,
        help="Directory containing contextual_tp_sl_ablation_workflow_* reports to discover cumulative ledgers.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-06-27T00:00:00+00:00")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threshold-mode", default="expanding", choices=["expanding", "full_sample"])
    parser.add_argument("--min-threshold-history", type=int, default=500)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--min-post-cutoff-rows", type=int, default=0)
    parser.add_argument("--min-post-cutoff-timestamps", type=int, default=0)
    parser.add_argument("--min-post-cutoff-active-heads", type=int, default=0)
    parser.add_argument("--min-policy-action-rows", type=int, default=0)
    parser.add_argument("--min-policy-action-timestamps", type=int, default=0)
    parser.add_argument("--min-policy-outcome-rows", type=int, default=0)
    parser.add_argument("--min-policy-outcome-timestamps", type=int, default=0)
    parser.add_argument("--min-policy-outcome-rows-per-action-head", type=int, default=0)
    parser.add_argument("--min-policy-outcome-rows-per-required-head", type=int, default=0)
    parser.add_argument("--min-diagnostic-group-features", type=int, default=1)
    parser.add_argument("--min-diagnostic-group-finite-rate", type=float, default=0.25)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bundle = _load_bundle(args.bundle)
    scan_args = _scan_args(bundle, args)
    cutoff = pd.Timestamp(args.cutoff, tz="UTC")
    candidate_paths = _candidate_paths(args)
    scan = pd.DataFrame([_scan_one(path, cutoff, scan_args) for path in candidate_paths])
    selected = _select_source(scan, force=bool(args.force))
    ran_outputs = None
    if selected is not None and (bool(selected.get("ready", False)) or args.force):
        ran_outputs = _run_reports(args, bundle, selected, args.out_dir)
    _write_report(args.out_dir, args, bundle, scan, selected, ran_outputs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
