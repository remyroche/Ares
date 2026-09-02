#!/usr/bin/env python3
"""Create a reproducible freeze/readiness manifest for a size-action arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_ARM = "C3em_bagged_safety_c3ed_or_high_value_zero_classifier_expanded_union_gate"
DEFAULT_REQUIRED_TESTS = [
    "tests/test_exact_state_size_action_nonoverlap.py",
    "tests/test_compare_size_action_learning_runs.py",
]
DEFAULT_SOURCE_FILES = [
    "scripts/run_exact_state_size_action_learning.py",
    "scripts/compare_size_action_learning_runs.py",
    "scripts/audit_size_action_interventions.py",
    "scripts/report_size_action_champion.py",
    "scripts/freeze_size_action_champion.py",
]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": bool(path.exists()),
        "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else None,
        "sha256": _sha256(path),
    }


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _arm_rows(frame: pd.DataFrame, arm: str) -> pd.DataFrame:
    if frame.empty or "arm" not in frame.columns:
        return pd.DataFrame()
    return frame.loc[frame["arm"].astype(str).eq(str(arm))].copy()


def _numeric_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(default)


def _bool_all(frame: pd.DataFrame, column: str) -> bool | None:
    if frame.empty or column not in frame.columns:
        return None
    return bool(frame[column].fillna(False).astype(bool).all())


def _binding_opportunity_groups(panel: pd.DataFrame) -> int | None:
    if panel.empty or not {"fold_id", "timestamp", "strategy_id", "split", "group_can_bind"}.issubset(panel.columns):
        return None
    work = panel.loc[panel["split"].astype(str).eq("eval")].copy()
    work["group_can_bind"] = pd.to_numeric(work["group_can_bind"], errors="coerce").fillna(0.0)
    work = work.loc[work["group_can_bind"] > 0.0]
    if work.empty:
        return 0
    return int(work.drop_duplicates(["fold_id", "timestamp", "strategy_id"]).shape[0])


def _parity_status(local_parity: pd.DataFrame, external_parity: pd.DataFrame | None = None) -> dict[str, Any]:
    for source, frame in (("run_local", local_parity), ("external", external_parity if external_parity is not None else pd.DataFrame())):
        if frame.empty or "noop_decision_signature_equal" not in frame.columns:
            continue
        values = frame["noop_decision_signature_equal"].fillna(False).astype(bool)
        return {
            "noop_decision_signature_all_equal": bool(values.all()),
            "noop_parity_source": source,
            "noop_parity_rows": int(len(frame)),
        }
    return {
        "noop_decision_signature_all_equal": None,
        "noop_parity_source": "missing",
        "noop_parity_rows": 0,
    }


def _summarize_arm(run_dir: Path, arm: str, *, parity_artifact: Path | None = None) -> dict[str, Any]:
    promotion = _arm_rows(_read_csv(run_dir / "size_action_promotion_summary.csv"), arm)
    replay = _arm_rows(_read_csv(run_dir / "size_action_replay_vs_label_audit.csv"), arm)
    quality = _arm_rows(_read_csv(run_dir / "size_action_action_quality.csv"), arm)
    panel = _read_csv(run_dir / "size_action_exact_panel.csv")
    parity = _read_csv(run_dir / "size_action_noop_parity.csv")
    external_parity = _read_csv(parity_artifact) if parity_artifact is not None else None
    parity_info = _parity_status(parity, external_parity)

    metrics: dict[str, Any] = {
        "arm": arm,
        "promotion_rows": int(len(promotion)),
        "replay_rows": int(len(replay)),
        "quality_rows": int(len(quality)),
        "folds": int(replay["fold_id"].nunique()) if "fold_id" in replay.columns else 0,
        "binding_opportunity_groups": _binding_opportunity_groups(panel),
        **parity_info,
    }
    if not promotion.empty:
        row = promotion.iloc[0]
        for col in [
            "median_delta_net_pnl",
            "q25_delta_net_pnl",
            "mean_delta_net_pnl",
            "positive_delta_net_pnl_share",
            "median_delta_cost_pnl",
            "median_exposure_ratio",
            "median_multiplier",
        ]:
            if col in row:
                metrics[col] = float(row[col])
    if not replay.empty:
        metrics.update(
            {
                "interventions": int(_numeric_series(replay, "intervention_count").sum()),
                "positive_actions": int(_numeric_series(replay, "positive_action_count").sum()),
                "delta_net_pnl_sum": float(_numeric_series(replay, "delta_net_pnl").sum()),
                "realized_delta_full_J_sum": float(_numeric_series(replay, "realized_delta_full_J_sum").sum()),
                "realized_delta_full_net_pnl_sum": float(_numeric_series(replay, "realized_delta_full_net_pnl_sum").sum()),
                "positive_replay_folds": int(_numeric_series(replay, "delta_net_pnl").gt(0.0).sum()),
                "sequential_replay_positive_all": _bool_all(replay, "sequential_replay_positive"),
                "independent_label_positive_all": _bool_all(replay, "independent_label_positive"),
                "replay_label_disagreements": int(
                    replay.get("sequential_replay_disagrees_with_label", pd.Series(False, index=replay.index))
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
                "min_fold_delta_net_pnl": float(_numeric_series(replay, "delta_net_pnl").min()),
            }
        )
    if not quality.empty:
        metrics["oracle_gain_capture_ratio_mean"] = float(
            _numeric_series(quality, "oracle_gain_capture_ratio").mean()
        )
        metrics["oracle_positive_group_capture_rate_mean"] = float(
            _numeric_series(quality, "oracle_positive_group_capture_rate").mean()
        )
    if metrics.get("interventions", 0) > 0:
        metrics["precision_total"] = float(metrics.get("positive_actions", 0) / max(metrics["interventions"], 1))
    binding_groups = metrics.get("binding_opportunity_groups")
    if binding_groups:
        metrics["binding_intervention_rate_total"] = float(metrics.get("interventions", 0) / max(int(binding_groups), 1))
    return metrics


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def _gate_status(metrics: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "positive_precision": float(metrics.get("precision_total", 0.0)) >= 1.0,
        "positive_all_folds": bool(metrics.get("sequential_replay_positive_all")) and bool(metrics.get("independent_label_positive_all")),
        "no_replay_label_disagreements": int(metrics.get("replay_label_disagreements", 1)) == 0,
        "positive_q25_delta": float(metrics.get("q25_delta_net_pnl", 0.0)) > 0.0,
        "exposure_retained": float(metrics.get("median_exposure_ratio", 0.0)) >= 0.98,
        "intervention_band": 0.05 <= float(metrics.get("binding_intervention_rate_total", 0.0)) <= 0.15,
        "noop_parity": metrics.get("noop_decision_signature_all_equal") is True,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "checks": checks,
        "research_ready": not failed,
        "production_ready": False,
        "failed_research_gates": failed,
        "production_blockers": [
            "true_prospective_frozen_dual_scoring_not_completed",
            "live_inference_materialization_not_verified",
        ],
    }


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    gates = payload["gate_status"]
    lines = [
        "# Size-Action Freeze Manifest",
        "",
        f"Arm: `{payload['arm']}`",
        f"Run: `{payload['run_dir']}`",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "## Readiness",
        "",
        f"- Research ready: `{gates['research_ready']}`",
        f"- Production ready: `{gates['production_ready']}`",
        f"- Failed research gates: `{', '.join(gates['failed_research_gates']) or 'none'}`",
        f"- Production blockers: `{', '.join(gates['production_blockers'])}`",
        "",
        "## Key Metrics",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in [
        "interventions",
        "positive_actions",
        "precision_total",
        "delta_net_pnl_sum",
        "median_delta_net_pnl",
        "q25_delta_net_pnl",
        "min_fold_delta_net_pnl",
        "positive_replay_folds",
        "median_exposure_ratio",
        "binding_opportunity_groups",
        "binding_intervention_rate_total",
        "oracle_gain_capture_ratio_mean",
        "replay_label_disagreements",
    ]:
        if key in metrics:
            lines.append(f"| {key} | {metrics[key]} |")
    lines.extend(["", "## Gate Checks", "", "| gate | pass |", "|---|---:|"])
    for key, ok in gates["checks"].items():
        lines.append(f"| {key} | {ok} |")
    lines.extend(["", "## Hashed Inputs", "", "| path | sha256 |", "|---|---|"])
    for record in payload["artifacts"]:
        lines.append(f"| `{record['path']}` | `{record['sha256']}` |")
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", default=DEFAULT_ARM)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--extra-artifact", action="append", default=[])
    parser.add_argument(
        "--parity-artifact",
        type=Path,
        default=None,
        help="Optional no-op parity CSV from an equivalent fresh exact-state panel build.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = _read_json(args.run_dir / "manifest.json")
    metrics = _summarize_arm(args.run_dir, args.arm, parity_artifact=args.parity_artifact)
    gate_status = _gate_status(metrics)

    artifact_paths = [
        args.run_dir / "manifest.json",
        args.run_dir / "size_action_promotion_summary.csv",
        args.run_dir / "size_action_replay_vs_label_audit.csv",
        args.run_dir / "size_action_action_quality.csv",
        args.run_dir / "size_action_schedules.csv",
        args.run_dir / "size_action_gate_thresholds.csv",
        args.run_dir / "size_action_selected_features.csv",
        args.run_dir / "size_action_noop_parity.csv",
        args.run_dir / "size_action_exact_panel.csv",
    ]
    if args.parity_artifact is not None:
        artifact_paths.append(args.parity_artifact)
    artifact_paths.extend(Path(p) for p in args.extra_artifact)
    source_paths = [Path(p) for p in DEFAULT_SOURCE_FILES + DEFAULT_REQUIRED_TESTS]

    payload = {
        "generated_by": "freeze_size_action_champion",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "arm": args.arm,
        "run_dir": str(args.run_dir),
        "source_manifest": manifest,
        "metrics": metrics,
        "gate_status": gate_status,
        "artifacts": [_file_record(p) for p in artifact_paths],
        "source_files": [_file_record(p) for p in source_paths],
        "required_tests": DEFAULT_REQUIRED_TESTS,
    }
    (args.out_dir / "size_action_freeze_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    _write_markdown(args.out_dir / "size_action_freeze_manifest.md", payload)
    print(
        {
            "out_dir": str(args.out_dir),
            "arm": args.arm,
            "research_ready": bool(gate_status["research_ready"]),
            "production_ready": bool(gate_status["production_ready"]),
            "failed_research_gates": gate_status["failed_research_gates"],
            "production_blockers": gate_status["production_blockers"],
        }
    )


if __name__ == "__main__":
    main()
