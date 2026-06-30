#!/usr/bin/env python3
"""Run frozen market-state priority shadow replay across labelled windows.

This is an orchestration wrapper.  It keeps the active T1 policy contract fixed,
scores each supplied candidate ledger with the frozen market-state bundle, trains
the fixed S2 head-priority model on the existing walk-forward residual ledger,
applies the bounded priority cap/gate, and refreshes the cross-window promotion
audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_market_state_priority_window_readiness import audit_window_readiness

DEFAULT_BUNDLE = Path(
    "data_perp/reports/market_state_controller_bundle_t1_stage1_shadow_s2_20260626_v1"
    "/market_state_controller_bundle.joblib"
)
DEFAULT_WALKFORWARD_DIR = Path(
    "data_perp/reports/market_state_threshold_controller_walkforward_20260626_t1_lgbm_pruned_shockup_v1"
)
DEFAULT_FEATURE_STORE_DIR = Path("data_perp/features/20260627_120000")
DEFAULT_POLICY_MANIFEST = Path(
    "data_perp/reports/reliability_blend_component_arm_portfolio_ablation_20260625"
    "/A0_anchor_only/portfolio_policy_ablation_manifest.json"
)
DEFAULT_TRAIN_DEPLOYABLE = Path(
    "data_perp/artifacts/reliability_blend_native_simple_policy_replay_20260624_floor070"
    "/simple_policy_optimiser/simple_policy_candidates.parquet"
)
DEFAULT_OUTPUT_ROOT = Path("data_perp/reports/market_state_priority_shadow_windows_20260626")


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


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or path.is_dir():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "window"


def _candidate_manifest_path(path: Path) -> Path | None:
    artifact_root = path.parent.parent if len(path.parents) >= 2 else path.parent
    names = [
        "t1_repaired_static_baseline_manifest.json",
        "t1_anchor_scored_candidate_manifest.json",
        "live_ledger_native_materialization_manifest.json",
    ]
    for name in names:
        candidate = artifact_root / name
        if candidate.exists():
            return candidate
    manifests = sorted(artifact_root.glob("*manifest*.json"))
    return manifests[0] if manifests else None


def _read_candidate_contract(path: Path) -> dict[str, Any]:
    manifest_path = _candidate_manifest_path(path)
    payload = _load_json(manifest_path)
    active_stack = dict(payload.get("active_stack") or {})
    validation = dict(payload.get("validation") or {})
    rank_reference = dict(validation.get("rank_reference_contract") or {})
    return {
        "manifest_path": str(manifest_path) if manifest_path is not None else None,
        "manifest_sha256": _sha256(manifest_path),
        "generated_by": payload.get("generated_by"),
        "rank_contract": active_stack.get("rank_contract"),
        "rank_scope": active_stack.get("rank_scope"),
        "promotion_status": active_stack.get("promotion_status"),
        "promotion_basis": active_stack.get("promotion_basis"),
        "rank_reference_run_id": active_stack.get("rank_reference_run_id")
        or rank_reference.get("eval_rank_reference_run_id"),
        "enabled_heads": sorted(map(str, active_stack.get("enabled_heads") or [])),
        "disabled_heads": sorted(map(str, active_stack.get("disabled_heads") or [])),
        "qfail_active": active_stack.get("qfail_active"),
        "head_health_active": active_stack.get("head_health_active"),
        "market_state_threshold_controller_active": active_stack.get(
            "market_state_threshold_controller_active"
        ),
        "policy_variant": active_stack.get("policy_variant"),
        "auction": active_stack.get("auction"),
    }


def _read_window(path: Path) -> dict[str, Any]:
    try:
        frame = pd.read_parquet(path, columns=["timestamp", "head"])
    except Exception:
        frame = pd.read_parquet(path)
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")
    ts = ts.dropna()
    heads = (
        sorted(frame["head"].dropna().astype(str).unique().tolist())
        if "head" in frame.columns
        else []
    )
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "rows": int(len(frame)),
        "timestamp_count": int(ts.nunique()),
        "start": ts.min().isoformat() if not ts.empty else None,
        "end": ts.max().isoformat() if not ts.empty else None,
        "heads": heads,
        "contract": _read_candidate_contract(path),
    }


def _metric_delta(
    observed: dict[str, Any],
    expected: dict[str, Any],
    key: str,
) -> float | None:
    obs = observed.get(key)
    exp = expected.get(key)
    if obs is None or exp is None:
        return None
    try:
        obs_f = float(obs)
        exp_f = float(exp)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(obs_f) or not np.isfinite(exp_f):
        return None
    return float(obs_f - exp_f)


def _static_baseline_parity(
    *,
    window: dict[str, Any],
    priority_dir: Path,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Compare P0 replay metrics with the materialized baseline manifest.

    Priority-shadow runs are only promotion-grade when their static arm exactly
    reproduces the frozen baseline for the same candidate contract.  A mismatch
    usually means the run is still useful as a diagnostic, but not a clean
    attribution replay against T1.
    """
    contract = dict((window.get("candidate") or {}).get("contract") or {})
    manifest_path = contract.get("manifest_path")
    if not manifest_path:
        return {
            "checked": False,
            "passed": False,
            "reason": "missing_candidate_manifest",
        }
    manifest = _load_json(Path(str(manifest_path)))
    expected = dict(manifest.get("summary") or {})
    if not expected:
        return {
            "checked": False,
            "passed": False,
            "reason": "missing_candidate_manifest_summary",
            "manifest_path": str(manifest_path),
        }
    summary_path = priority_dir / "head_priority_learning_replay_summary.csv"
    if not summary_path.exists():
        return {
            "checked": False,
            "passed": False,
            "reason": "missing_static_priority_summary",
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
        }
    try:
        summary = pd.read_csv(summary_path)
    except Exception as exc:  # pragma: no cover - defensive IO guard
        return {
            "checked": False,
            "passed": False,
            "reason": "unreadable_static_priority_summary",
            "error": str(exc),
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
        }
    if "arm" not in summary.columns:
        return {
            "checked": False,
            "passed": False,
            "reason": "static_priority_summary_missing_arm",
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
        }
    rows = summary.loc[summary["arm"].astype(str).eq("P0_static_priority")]
    if rows.empty:
        return {
            "checked": False,
            "passed": False,
            "reason": "missing_p0_static_priority_row",
            "manifest_path": str(manifest_path),
            "summary_path": str(summary_path),
        }
    observed = rows.iloc[0].to_dict()
    keys = [
        "trade_count",
        "net_pnl",
        "gross_pnl",
        "cost_pnl",
        "full_sl_rate",
        "timeout_rate",
        "worst_24h_net_pnl",
    ]
    deltas = {
        key: _metric_delta(observed, expected, key)
        for key in keys
        if _metric_delta(observed, expected, key) is not None
    }
    failures = [
        key
        for key, value in deltas.items()
        if abs(float(value)) > float(tolerance)
    ]
    return {
        "checked": True,
        "passed": not failures,
        "manifest_path": str(manifest_path),
        "summary_path": str(summary_path),
        "tolerance": float(tolerance),
        "failures": failures,
        "deltas": deltas,
        "observed": {key: _json_safe(observed.get(key)) for key in keys if key in observed},
        "expected": {key: _json_safe(expected.get(key)) for key in keys if key in expected},
    }


def _resolve_run_rank_contract(
    windows: list[dict[str, Any]],
    *,
    allow_mixed_rank_contracts: bool = False,
) -> dict[str, Any]:
    contracts: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str | None, str | None]] = set()
    for window in windows:
        contract = dict((window.get("candidate") or {}).get("contract") or {})
        key = (
            contract.get("rank_contract"),
            contract.get("rank_scope"),
            contract.get("rank_reference_run_id"),
        )
        if key in seen:
            continue
        seen.add(key)
        contracts.append(
            {
                "rank_contract": contract.get("rank_contract"),
                "rank_scope": contract.get("rank_scope"),
                "rank_reference_run_id": contract.get("rank_reference_run_id"),
                "promotion_status": contract.get("promotion_status"),
                "manifest_path": contract.get("manifest_path"),
            }
        )
    known_scopes = sorted({str(c["rank_scope"]) for c in contracts if c.get("rank_scope")})
    known_contracts = sorted(
        {str(c["rank_contract"]) for c in contracts if c.get("rank_contract")}
    )
    if len(known_scopes) > 1 and not allow_mixed_rank_contracts:
        raise ValueError(
            "mixed candidate rank scopes are not allowed in one shadow-window run: "
            + ", ".join(known_scopes)
        )
    if len(known_contracts) > 1 and not allow_mixed_rank_contracts:
        raise ValueError(
            "mixed candidate rank contracts are not allowed in one shadow-window run: "
            + ", ".join(known_contracts)
        )
    if len(known_scopes) == 1:
        preserved = known_scopes[0]
    elif len(known_contracts) == 1:
        preserved = known_contracts[0]
    elif contracts:
        preserved = "mixed" if allow_mixed_rank_contracts else "unknown"
    else:
        preserved = "unknown"
    return {
        "rank_contract_preserved": preserved,
        "candidate_rank_contracts": contracts,
        "candidate_rank_scopes": known_scopes,
        "candidate_rank_contract_names": known_contracts,
        "allow_mixed_rank_contracts": bool(allow_mixed_rank_contracts),
    }


def _run_command(cmd: list[str], *, dry_run: bool) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    payload = {
        "command": cmd,
        "started_at_utc": started.isoformat(),
        "dry_run": bool(dry_run),
    }
    if dry_run:
        payload.update({"returncode": None, "completed_at_utc": None})
        return payload
    proc = subprocess.run(cmd, cwd=ROOT, check=False)
    completed = datetime.now(timezone.utc)
    payload.update(
        {
            "returncode": int(proc.returncode),
            "completed_at_utc": completed.isoformat(),
            "elapsed_seconds": (completed - started).total_seconds(),
        }
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: {' '.join(cmd)}")
    return payload


def _needs_run(required_outputs: list[Path], *, force: bool) -> bool:
    return bool(force) or not all(path.exists() for path in required_outputs)


def _script(name: str) -> str:
    return str(ROOT / "scripts" / name)


def _shadow_priority_contract_fields() -> dict[str, Any]:
    return {
        "qfail_active": False,
        "head_health_active": False,
        "market_state_threshold_controller_active": False,
        "operational_status": "shadow_only",
        "execution_enabled": False,
        "production_eligible": False,
        "requires_promotion_gate": True,
        "priority_action": "portfolio_priority_adjustment_shadow_only",
        "changes_scores_or_thresholds": False,
        "changes_auction_ordering": True,
    }


def _score_command(args: argparse.Namespace, *, candidate: Path, output_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        _script("score_market_state_controller_bundle.py"),
        "--bundle",
        str(args.bundle),
        "--eval-candidates",
        str(candidate),
        "--eval-feature-store-dir",
        str(args.feature_store_dir),
        "--output-dir",
        str(output_dir),
        "--policy-manifest",
        str(args.policy_manifest),
        "--policy-variant",
        str(args.policy_variant),
        "--train-deployable-candidates",
        str(args.train_deployable_candidates),
        "--market-mode",
        str(args.market_mode),
    ]


def _priority_command(
    args: argparse.Namespace,
    *,
    score_dir: Path,
    output_dir: Path,
    static_baseline_manifest: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        _script("run_market_state_head_priority_learning.py"),
        "--walkforward-dir",
        str(args.walkforward_dir),
        "--score-dir",
        str(score_dir),
        "--output-dir",
        str(output_dir),
        "--state-arm",
        str(args.state_arm),
        "--backends",
        str(args.backend),
        "--target-mode",
        str(args.target_mode),
        "--min-rank",
        str(args.min_rank),
        "--frontier-gamma",
        str(args.frontier_gamma),
        "--frontier-bandwidth",
        str(args.frontier_bandwidth),
        "--sl-penalty",
        str(args.sl_penalty),
        "--timeout-penalty",
        str(args.timeout_penalty),
        "--rank-residual-weight",
        str(args.rank_residual_weight),
        "--max-adjustment",
        str(args.cap),
        "--max-priority-multiplier",
        str(args.max_priority_multiplier),
        "--max-rank-adjustment",
        str(args.max_rank_adjustment),
        "--priority-action",
        str(args.priority_action),
        "--validation-mode",
        str(args.validation_mode),
        "--train-deployable-candidates",
        str(args.train_deployable_candidates),
        "--policy-manifest",
        str(args.policy_manifest),
        "--policy-variant",
        str(args.policy_variant),
        "--market-mode",
        str(args.market_mode),
    ]
    if bool(getattr(args, "use_all_state_heads", False)):
        cmd.append("--use-all-state-heads")
    state_head_statuses = str(getattr(args, "state_head_statuses", "") or "").strip()
    if state_head_statuses:
        cmd.extend(["--state-head-statuses", state_head_statuses])
    if static_baseline_manifest:
        cmd.extend(["--static-baseline-manifest", str(static_baseline_manifest)])
    return cmd


def _cap_sweep_command(
    args: argparse.Namespace,
    *,
    priority_dir: Path,
    output_dir: Path,
    static_baseline_manifest: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        _script("replay_market_state_learned_priority_cap_sweep.py"),
        "--priority-dir",
        str(priority_dir),
        "--output-dir",
        str(output_dir),
        "--caps",
        str(args.cap_grid),
        "--min-abs-z-thresholds",
        str(args.min_abs_z_grid),
        "--selection-gate-mode",
        str(args.selection_gate_mode),
        "--selection-min-accepted-jaccard",
        str(args.selection_min_accepted_jaccard),
        "--selection-max-full-sl-delta",
        str(args.selection_max_full_sl_delta),
        "--selection-max-timeout-delta",
        str(args.selection_max_timeout_delta),
        "--policy-variant",
        str(args.policy_variant),
        "--market-mode",
        str(args.market_mode),
    ]
    if static_baseline_manifest:
        cmd.extend(["--static-baseline-manifest", str(static_baseline_manifest)])
    return cmd


def _audit_command(args: argparse.Namespace, *, cap_dirs: list[Path], labels: list[str], output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        _script("audit_market_state_priority_shadow_promotion.py"),
        "--arm-contains",
        str(args.arm_contains),
        "--output-dir",
        str(output_dir),
    ]
    if bool(getattr(args, "use_selected_challenger", False)):
        cmd.append("--use-selected-challenger")
    for cap_dir, label in zip(cap_dirs, labels, strict=True):
        cmd.extend(["--cap-sweep-dir", str(cap_dir), "--window-label", str(label)])
    return cmd


def _run_readiness_preflight(args: argparse.Namespace, candidates: list[Path]) -> dict[str, Any] | None:
    existing_manifest = getattr(args, "readiness_existing_manifest", None)
    if existing_manifest is None:
        return None
    output_dir = getattr(args, "readiness_output_dir", None)
    if output_dir is None:
        output_dir = args.output_root / "window_readiness"
    summary = audit_window_readiness(
        candidates=candidates,
        existing_manifest=Path(existing_manifest),
        output_dir=Path(output_dir),
        min_timestamp_count=int(getattr(args, "readiness_min_timestamp_count", 3)),
        min_rows=int(getattr(args, "readiness_min_rows", 1)),
    )
    if not bool(summary.get("passed")):
        raise RuntimeError(
            "window readiness preflight failed; inspect "
            f"{Path(output_dir) / 'market_state_priority_window_readiness_report.md'}"
        )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", action="append", type=Path, required=True)
    parser.add_argument("--window-label", action="append", default=[])
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--walkforward-dir", type=Path, default=DEFAULT_WALKFORWARD_DIR)
    parser.add_argument("--feature-store-dir", type=Path, default=DEFAULT_FEATURE_STORE_DIR)
    parser.add_argument("--policy-manifest", type=Path, default=DEFAULT_POLICY_MANIFEST)
    parser.add_argument("--train-deployable-candidates", type=Path, default=DEFAULT_TRAIN_DEPLOYABLE)
    parser.add_argument("--policy-variant", default="refit_bar4_strategy_bar2")
    parser.add_argument("--market-mode", default="perps")
    parser.add_argument("--state-arm", default="S2_observed_forecast_shared_response")
    parser.add_argument(
        "--state-head-statuses",
        default="active_candidate",
        help=(
            "Comma-separated activation-registry recommended_status values "
            "passed to run_market_state_head_priority_learning.py. Ignored "
            "when --use-all-state-heads is set."
        ),
    )
    parser.add_argument(
        "--use-all-state-heads",
        action="store_true",
        help=(
            "Pass all deployable state_/forecast_ columns from the selected "
            "state arm to the shadow priority learner. This is useful for "
            "observed-state routing experiments because the threshold-controller "
            "active_candidate registry is intentionally narrower."
        ),
    )
    parser.add_argument("--backend", default="lgbm")
    parser.add_argument("--target-mode", default="head_top_candidate")
    parser.add_argument("--min-rank", type=float, default=0.50)
    parser.add_argument("--frontier-gamma", type=float, default=3.0)
    parser.add_argument("--frontier-bandwidth", type=float, default=0.08)
    parser.add_argument("--sl-penalty", type=float, default=0.0)
    parser.add_argument("--timeout-penalty", type=float, default=0.002)
    parser.add_argument("--rank-residual-weight", type=float, default=1.0)
    parser.add_argument("--validation-mode", default="fold_aware")
    parser.add_argument("--cap", type=float, default=0.15)
    parser.add_argument("--max-priority-multiplier", type=float, default=1.0)
    parser.add_argument("--max-rank-adjustment", type=float, default=0.0)
    parser.add_argument(
        "--priority-action",
        choices=["adjustment", "both", "multiplier"],
        default="adjustment",
    )
    parser.add_argument(
        "--caps",
        default=None,
        help=(
            "Comma-separated priority-adjustment caps for the cap sweep. "
            "Defaults to the legacy single --cap value."
        ),
    )
    parser.add_argument("--min-abs-z", type=float, default=0.50)
    parser.add_argument(
        "--min-abs-z-thresholds",
        default=None,
        help=(
            "Comma-separated minimum absolute priority-z thresholds for the "
            "cap sweep. Defaults to the legacy single --min-abs-z value."
        ),
    )
    parser.add_argument(
        "--selection-gate-mode",
        choices=["defensive", "opportunity"],
        default="opportunity",
        help=(
            "Replay gate used for cap selection. Use opportunity for "
            "cross-head routing; defensive is stricter and suited to "
            "threshold suppression."
        ),
    )
    parser.add_argument("--selection-min-accepted-jaccard", type=float, default=0.95)
    parser.add_argument("--selection-max-full-sl-delta", type=float, default=0.005)
    parser.add_argument("--selection-max-timeout-delta", type=float, default=0.0)
    parser.add_argument("--arm-contains", default="cap_0p15_zge_0p5")
    parser.add_argument("--no-use-selected-challenger", action="store_true")
    parser.add_argument(
        "--readiness-existing-manifest",
        type=Path,
        default=None,
        help=(
            "Existing safe-grid shadow-window manifest used to preflight append "
            "candidate ledgers. When supplied, stale, overlapping or "
            "rank-contract-incompatible windows fail before scoring/replay."
        ),
    )
    parser.add_argument("--readiness-output-dir", type=Path, default=None)
    parser.add_argument("--readiness-min-timestamp-count", type=int, default=3)
    parser.add_argument("--readiness-min-rows", type=int, default=1)
    parser.add_argument(
        "--allow-mixed-rank-contracts",
        action="store_true",
        help=(
            "Allow one shadow-window run to include candidate ledgers with "
            "different rank scopes/contracts. Disabled by default because it "
            "confounds timestamp-vs-global attribution."
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.use_selected_challenger = not bool(args.no_use_selected_challenger)
    args.cap_grid = str(args.caps) if args.caps else str(args.cap)
    args.min_abs_z_grid = (
        str(args.min_abs_z_thresholds) if args.min_abs_z_thresholds else str(args.min_abs_z)
    )
    return args


def main() -> None:
    args = _parse_args()
    candidates = list(args.candidate or [])
    labels = list(args.window_label or [])
    if labels and len(labels) != len(candidates):
        raise ValueError("--window-label must be supplied once per --candidate, or omitted")
    if not labels:
        labels = [path.parent.parent.name for path in candidates]
    for path in [
        args.bundle,
        args.walkforward_dir,
        args.feature_store_dir,
        args.policy_manifest,
        args.train_deployable_candidates,
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
    for path in candidates:
        if not path.exists():
            raise FileNotFoundError(path)
    readiness_summary = _run_readiness_preflight(args, candidates)

    args.output_root.mkdir(parents=True, exist_ok=True)
    windows: list[dict[str, Any]] = []
    cap_dirs: list[Path] = []
    commands: list[dict[str, Any]] = []
    for idx, (candidate, label) in enumerate(zip(candidates, labels, strict=True), start=1):
        slug = f"{idx:02d}_{_slug(label)}"
        score_dir = args.output_root / slug / "score_bundle"
        priority_dir = args.output_root / slug / "priority_learning"
        cap_dir = args.output_root / slug / "cap_sweep"
        score_dir.mkdir(parents=True, exist_ok=True)
        priority_dir.mkdir(parents=True, exist_ok=True)
        cap_dir.mkdir(parents=True, exist_ok=True)

        window = {
            "label": label,
            "slug": slug,
            "candidate": _read_window(candidate),
            "score_dir": str(score_dir),
            "priority_dir": str(priority_dir),
            "cap_sweep_dir": str(cap_dir),
        }
        static_baseline_manifest = (
            dict(window.get("candidate") or {})
            .get("contract", {})
            .get("manifest_path")
        )
        if _needs_run(
            [score_dir / "controller_scored_candidates.parquet", score_dir / "market_state_timestamp_panel.parquet"],
            force=bool(args.force),
        ):
            cmd = _score_command(args, candidate=candidate, output_dir=score_dir)
            commands.append(_run_command(cmd, dry_run=bool(args.dry_run)))
        if _needs_run(
            [priority_dir / "head_priority_learned_schedule.parquet", priority_dir / "head_priority_learning_replay_summary.csv"],
            force=bool(args.force),
        ):
            cmd = _priority_command(
                args,
                score_dir=score_dir,
                output_dir=priority_dir,
                static_baseline_manifest=static_baseline_manifest,
            )
            commands.append(_run_command(cmd, dry_run=bool(args.dry_run)))
        if _needs_run(
            [cap_dir / "head_priority_cap_sweep_metrics.csv", cap_dir / "head_priority_cap_sweep_by_head.csv"],
            force=bool(args.force),
        ):
            cmd = _cap_sweep_command(
                args,
                priority_dir=priority_dir,
                output_dir=cap_dir,
                static_baseline_manifest=static_baseline_manifest,
            )
            commands.append(_run_command(cmd, dry_run=bool(args.dry_run)))
        if not bool(args.dry_run):
            window["static_baseline_parity"] = _static_baseline_parity(
                window=window,
                priority_dir=priority_dir,
            )
        cap_dirs.append(cap_dir)
        windows.append(window)

    rank_contract = _resolve_run_rank_contract(
        windows,
        allow_mixed_rank_contracts=bool(args.allow_mixed_rank_contracts),
    )
    parity_results = [
        dict(window.get("static_baseline_parity") or {})
        for window in windows
        if window.get("static_baseline_parity") is not None
    ]
    parity_checked_count = sum(1 for row in parity_results if bool(row.get("checked")))
    parity_passed_count = sum(1 for row in parity_results if bool(row.get("passed")))
    parity_failures = [
        {
            "label": window.get("label"),
            "slug": window.get("slug"),
            "parity": window.get("static_baseline_parity"),
        }
        for window in windows
        if window.get("static_baseline_parity") is not None
        and not bool(dict(window.get("static_baseline_parity") or {}).get("passed"))
    ]
    audit_dir = args.output_root / "promotion_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    if _needs_run(
        [audit_dir / "market_state_priority_shadow_promotion_gate.json"],
        force=bool(args.force),
    ):
        cmd = _audit_command(args, cap_dirs=cap_dirs, labels=labels, output_dir=audit_dir)
        commands.append(_run_command(cmd, dry_run=bool(args.dry_run)))

    manifest = {
        "generated_by": "run_market_state_priority_shadow_windows",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "fixed_contract_market_state_priority_shadow_window_runner",
        "contract": {
            "active_baseline": "static_T1",
            "rank_contract_preserved": rank_contract["rank_contract_preserved"],
            "candidate_rank_scopes": rank_contract["candidate_rank_scopes"],
            "candidate_rank_contract_names": rank_contract["candidate_rank_contract_names"],
            "candidate_rank_contracts": rank_contract["candidate_rank_contracts"],
            "allow_mixed_rank_contracts": rank_contract["allow_mixed_rank_contracts"],
            **_shadow_priority_contract_fields(),
        },
        "static_baseline_parity": {
            "checked_windows": int(parity_checked_count),
            "passed_windows": int(parity_passed_count),
            "all_checked_windows_passed": bool(parity_results)
            and parity_checked_count == len(parity_results)
            and parity_passed_count == len(parity_results),
            "promotion_grade": bool(parity_results)
            and parity_checked_count == len(parity_results)
            and parity_passed_count == len(parity_results),
            "failures": parity_failures,
        },
        "params": {
            "state_arm": str(args.state_arm),
            "state_head_statuses": str(args.state_head_statuses),
            "use_all_state_heads": bool(args.use_all_state_heads),
            "backend": str(args.backend),
            "target_mode": str(args.target_mode),
            "min_rank": float(args.min_rank),
            "frontier_gamma": float(args.frontier_gamma),
            "frontier_bandwidth": float(args.frontier_bandwidth),
            "sl_penalty": float(args.sl_penalty),
            "timeout_penalty": float(args.timeout_penalty),
            "cap": float(args.cap),
            "cap_grid": str(args.cap_grid),
            "min_abs_z": float(args.min_abs_z),
            "min_abs_z_grid": str(args.min_abs_z_grid),
            "selection_gate_mode": str(args.selection_gate_mode),
            "selection_min_accepted_jaccard": float(args.selection_min_accepted_jaccard),
            "selection_max_full_sl_delta": float(args.selection_max_full_sl_delta),
            "selection_max_timeout_delta": float(args.selection_max_timeout_delta),
            "arm_contains": str(args.arm_contains),
            "use_selected_challenger": bool(args.use_selected_challenger),
            "policy_variant": str(args.policy_variant),
            "market_mode": str(args.market_mode),
        },
        "inputs": {
            "bundle": str(args.bundle),
            "bundle_sha256": _sha256(args.bundle),
            "walkforward_dir": str(args.walkforward_dir),
            "feature_store_dir": str(args.feature_store_dir),
            "policy_manifest": str(args.policy_manifest),
            "policy_manifest_sha256": _sha256(args.policy_manifest),
            "train_deployable_candidates": str(args.train_deployable_candidates),
            "train_deployable_candidates_sha256": _sha256(args.train_deployable_candidates),
        },
        "readiness_preflight": readiness_summary,
        "windows": windows,
        "commands": commands,
        "outputs": {
            "promotion_audit": str(audit_dir),
            "manifest": str(args.output_root / "manifest.json"),
        },
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_safe({"output_root": str(args.output_root), "windows": windows}), indent=2))


if __name__ == "__main__":
    main()
