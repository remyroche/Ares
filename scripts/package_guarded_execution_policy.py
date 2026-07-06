#!/usr/bin/env python3
"""Package guarded execution replay evidence for the next frozen-policy gate.

This script compares the fixed h9 guarded replay against the anchored adaptive
guarded replay, verifies that both are frozen/leakage-safe replay artifacts, and
writes a compact promotion-review package with hashes for every referenced file.

The package is intentionally evidence-only. It does not refit a guard, reselect
thresholds, choose scenarios, or rewrite candidate rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.simple_policy_optimiser import _json_safe  # noqa: E402


REPORT_ROOT = Path("data_perp/reports/gmm_cluster_policy_smoke_20260702_wide_sidebalanced")
DEFAULT_FIXED_REPLAY_DIR = (
    REPORT_ROOT / "meta_handoff_guarded_execution_h9x4_execnet_v1" / "frozen_replay_v1"
)
DEFAULT_ADAPTIVE_REPLAY_DIR = (
    REPORT_ROOT
    / "meta_handoff_adaptive_scenario_guard_anchoradmit_h9_buffer000_materialized_v1"
    / "frozen_replay_v1"
)
DEFAULT_OUT_DIR = REPORT_ROOT / "guarded_execution_policy_package_20260703_v1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "pass"}
    if pd.isna(value):
        return False
    return bool(value)


def _as_float(row: Mapping[str, Any], key: str, default: float = np.nan) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_int(row: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _require_output(manifest: Mapping[str, Any], key: str, *, replay_dir: Path) -> Path:
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    value = outputs.get(key)
    if value:
        path = Path(value)
    else:
        path = replay_dir / {
            "summary": "frozen_replay_summary.csv",
            "audit": "frozen_replay_audit.json",
            "manifest": "frozen_replay_manifest.json",
            "folds": "frozen_replay_fold_summary.csv",
            "parity": "frozen_replay_parity.csv",
            "decisions": "frozen_replay_decisions.parquet",
            "fixed_policy_config": "fixed_portfolio_policy_config.json",
        }[key]
    if not path.exists():
        raise FileNotFoundError(f"Missing replay output {key}: {path}")
    return path


def _metric_thresholds(args: argparse.Namespace) -> dict[str, float | int]:
    return {
        "min_net_pnl": float(args.min_net_pnl),
        "min_mean_objective": float(args.min_mean_objective),
        "min_positive_fold_share": float(args.min_positive_fold_share),
        "max_full_sl_rate": float(args.max_full_sl_rate),
        "max_timeout_rate": float(args.max_timeout_rate),
        "min_worst_fold_net_pnl": float(args.min_worst_fold_net_pnl),
        "max_no_trade_folds": int(args.max_no_trade_folds),
        "min_review_accepted_trades": int(args.min_review_accepted_trades),
        "min_review_active_days": int(args.min_review_active_days),
        "min_review_unique_symbols": int(args.min_review_unique_symbols),
        "min_review_fold_trades": int(args.min_review_fold_trades),
        "min_review_monthly_trades": int(args.min_review_monthly_trades),
        "min_deployment_accepted_trades": int(args.min_deployment_accepted_trades),
        "min_deployment_active_days": int(args.min_deployment_active_days),
        "min_deployment_monthly_trades": int(args.min_deployment_monthly_trades),
    }


def _load_summary(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Empty replay summary: {path}")
    return dict(frame.iloc[0].to_dict())


def _gate_metrics(summary: Mapping[str, Any], thresholds: Mapping[str, float | int]) -> dict[str, bool]:
    return {
        "pass_simple_policy_gate": _as_bool(summary.get("pass_simple_policy_gate")),
        "net_positive": _as_float(summary, "sum_net_pnl") >= float(thresholds["min_net_pnl"]),
        "objective_positive": _as_float(summary, "mean_objective") >= float(thresholds["min_mean_objective"]),
        "positive_fold_share_ok": _as_float(summary, "positive_fold_share") >= float(thresholds["min_positive_fold_share"]),
        "no_trade_folds_ok": _as_int(summary, "no_trade_folds") <= int(thresholds["max_no_trade_folds"]),
        "worst_fold_ok": _as_float(summary, "worst_fold_net_pnl") >= float(thresholds["min_worst_fold_net_pnl"]),
        "full_sl_ok": _as_float(summary, "weighted_full_sl_rate") <= float(thresholds["max_full_sl_rate"]),
        "timeout_ok": _as_float(summary, "weighted_timeout_rate") <= float(thresholds["max_timeout_rate"]),
    }


def _accepted_exposure(decisions_path: Path) -> dict[str, Any]:
    decisions = pd.read_parquet(decisions_path)
    if decisions.empty:
        return {
            "accepted_trades": 0,
            "accepted_unique_symbols": 0,
            "accepted_active_days": 0,
            "accepted_span_days": 0,
            "accepted_first_ts": "",
            "accepted_last_ts": "",
            "accepted_long_trades": 0,
            "accepted_short_trades": 0,
            "accepted_min_fold_trades": 0,
            "accepted_min_monthly_trades": 0,
        }
    accepted_mask = (
        decisions["accepted"].astype(bool)
        if "accepted" in decisions.columns
        else pd.Series(True, index=decisions.index)
    )
    accepted = decisions.loc[accepted_mask].copy()
    if accepted.empty:
        return {
            "accepted_trades": 0,
            "accepted_unique_symbols": 0,
            "accepted_active_days": 0,
            "accepted_span_days": 0,
            "accepted_first_ts": "",
            "accepted_last_ts": "",
            "accepted_long_trades": 0,
            "accepted_short_trades": 0,
            "accepted_min_fold_trades": 0,
            "accepted_min_monthly_trades": 0,
        }
    ts = pd.to_datetime(accepted.get("timestamp"), utc=True, errors="coerce")
    dates = ts.dt.date
    span_days = int((ts.max().date() - ts.min().date()).days + 1) if ts.notna().any() else 0
    if "validation_week" in accepted.columns:
        fold_counts = accepted.groupby(accepted["validation_week"].astype(str)).size()
        min_fold_trades = int(fold_counts.min()) if not fold_counts.empty else 0
    else:
        min_fold_trades = 0
    month_counts = accepted.groupby(ts.dt.to_period("M").astype(str)).size() if ts.notna().any() else pd.Series(dtype=int)
    side_counts = accepted.get("side", pd.Series("", index=accepted.index)).astype(str).value_counts()
    return {
        "accepted_trades": int(len(accepted)),
        "accepted_unique_symbols": int(accepted.get("symbol", pd.Series("", index=accepted.index)).astype(str).nunique()),
        "accepted_active_days": int(dates.nunique()),
        "accepted_span_days": span_days,
        "accepted_first_ts": ts.min().isoformat() if ts.notna().any() else "",
        "accepted_last_ts": ts.max().isoformat() if ts.notna().any() else "",
        "accepted_long_trades": int(side_counts.get("long", 0)),
        "accepted_short_trades": int(side_counts.get("short", 0)),
        "accepted_min_fold_trades": min_fold_trades,
        "accepted_min_monthly_trades": int(month_counts.min()) if not month_counts.empty else 0,
    }


def _file_hash_records(
    *,
    replay_name: str,
    replay_dir: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    paths: dict[str, Path] = {"frozen_replay_manifest": manifest_path}
    outputs = manifest.get("outputs") if isinstance(manifest.get("outputs"), dict) else {}
    for key, value in outputs.items():
        if value:
            paths[f"output:{key}"] = Path(value)
    for key in [
        "clean_handoff_path",
        "offline_replay_candidates_path",
        "ev_curve_train_candidates_path",
        "source_candidates_path",
    ]:
        value = manifest.get(key)
        if value:
            paths[key] = Path(value)

    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for role, path in sorted(paths.items()):
        if path in seen:
            continue
        seen.add(path)
        records.append(
            {
                "replay": replay_name,
                "role": role,
                "path": str(path),
                "exists": path.exists(),
                "sha256": _sha256_path(path) if path.exists() and path.is_file() else "",
                "size_bytes": int(path.stat().st_size) if path.exists() and path.is_file() else 0,
                "replay_dir": str(replay_dir),
            }
        )
    return records


def _evaluate_replay(
    *,
    name: str,
    role: str,
    replay_dir: Path,
    thresholds: Mapping[str, float | int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = replay_dir / "frozen_replay_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing frozen replay manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    summary_path = _require_output(manifest, "summary", replay_dir=replay_dir)
    audit_path = _require_output(manifest, "audit", replay_dir=replay_dir)
    decisions_path = _require_output(manifest, "decisions", replay_dir=replay_dir)
    summary = _load_summary(summary_path)
    audit_payload = _read_json(audit_path)
    exposure = _accepted_exposure(decisions_path)
    leakage = manifest.get("leakage_audit_input")
    if not isinstance(leakage, dict):
        leakage = audit_payload.get("leakage_audit_input")
    if not isinstance(leakage, dict):
        leakage = {}

    metric_checks = _gate_metrics(summary, thresholds)
    parity = manifest.get("summary_parity") if isinstance(manifest.get("summary_parity"), dict) else {}
    handoff_match = manifest.get("handoff_key_match") if isinstance(manifest.get("handoff_key_match"), dict) else {}
    clean_forbidden = manifest.get("clean_handoff_forbidden_columns")
    if not isinstance(clean_forbidden, list):
        clean_forbidden = []
    prohibited_guard_features = leakage.get("prohibited_guard_features")
    if not isinstance(prohibited_guard_features, list):
        prohibited_guard_features = []

    ev_source = str(manifest.get("ev_curve_train_source") or "")
    ev_boundary = manifest.get("ev_curve_train_boundary_violations")
    ev_rows = manifest.get("ev_curve_train_rows_replayed")
    ev_curve_ok = bool(manifest.get("prior_window_ev_curve_by_fold"))
    if ev_source == "frozen_manifest_rows":
        ev_curve_ok = ev_curve_ok and int(ev_boundary or 0) == 0 and int(ev_rows or 0) > 0

    replay_checks = {
        "manifest_status_pass": str(manifest.get("status") or "").lower() == "pass",
        "frozen_replay_gate_pass": _as_bool(manifest.get("pass_frozen_guarded_replay_gate")),
        "source_hash_ok": _as_bool(manifest.get("source_candidates_hash_matches_manifest")),
        "clean_handoff_no_outcomes": _as_bool(manifest.get("clean_handoff_has_no_realized_outcomes")),
        "clean_handoff_no_forbidden_columns": len(clean_forbidden) == 0,
        "handoff_row_count_match": _as_bool(handoff_match.get("row_count_match")),
        "handoff_key_set_match": _as_bool(handoff_match.get("key_set_match")),
        "parity_pass": _as_bool(parity.get("passes")),
        "guard_not_refit": manifest.get("guard_refit") is False,
        "threshold_not_reselected": manifest.get("threshold_reselection") is False,
        "scenario_not_reselected": manifest.get("scenario_reselection") is False,
        "prior_window_ev_curve_ok": ev_curve_ok,
        "leakage_safe_for_frozen_replay": _as_bool(leakage.get("leakage_safe_for_frozen_replay")),
        "decision_time_safe": _as_bool(leakage.get("all_guard_inputs_decision_time_safe")),
        "no_prohibited_guard_features": len(prohibited_guard_features) == 0,
        "no_guard_before_signal": int(leakage.get("guard_decision_before_signal_rows") or 0) == 0,
        "no_missing_required_decision_time": int(leakage.get("missing_required_delayed_entry_timestamp_rows") or 0) == 0,
    }
    review_exposure_checks = {
        "review_accepted_trades_ok": int(exposure["accepted_trades"]) >= int(thresholds["min_review_accepted_trades"]),
        "review_active_days_ok": int(exposure["accepted_active_days"]) >= int(thresholds["min_review_active_days"]),
        "review_unique_symbols_ok": int(exposure["accepted_unique_symbols"]) >= int(thresholds["min_review_unique_symbols"]),
        "review_min_fold_trades_ok": int(exposure["accepted_min_fold_trades"]) >= int(thresholds["min_review_fold_trades"]),
        "review_min_monthly_trades_ok": int(exposure["accepted_min_monthly_trades"])
        >= int(thresholds["min_review_monthly_trades"]),
    }
    deployment_exposure_checks = {
        "deployment_accepted_trades_ok": int(exposure["accepted_trades"])
        >= int(thresholds["min_deployment_accepted_trades"]),
        "deployment_active_days_ok": int(exposure["accepted_active_days"])
        >= int(thresholds["min_deployment_active_days"]),
        "deployment_min_monthly_trades_ok": int(exposure["accepted_min_monthly_trades"])
        >= int(thresholds["min_deployment_monthly_trades"]),
    }
    package_gate_pass = all(metric_checks.values()) and all(replay_checks.values()) and all(review_exposure_checks.values())
    signal_time_safe = _as_bool(leakage.get("all_guard_inputs_signal_time_safe"))
    evidence = {
        "name": name,
        "role": role,
        "replay_dir": str(replay_dir),
        "manifest": str(manifest_path),
        "scenario": str(manifest.get("scenario") or summary.get("scenario") or ""),
        "guard_method": str(manifest.get("guard_method") or ""),
        "status": "pass" if package_gate_pass else "fail",
        "package_gate_pass": package_gate_pass,
        "signal_time_safe": signal_time_safe,
        "decision_time_safe": replay_checks["decision_time_safe"],
        "decision_time_caveat": "" if signal_time_safe else str(leakage.get("leakage_safe_caveat") or ""),
        "sum_net_pnl": _as_float(summary, "sum_net_pnl"),
        "mean_objective": _as_float(summary, "mean_objective"),
        "worst_fold_net_pnl": _as_float(summary, "worst_fold_net_pnl"),
        "positive_folds": _as_int(summary, "positive_folds"),
        "positive_fold_share": _as_float(summary, "positive_fold_share"),
        "no_trade_folds": _as_int(summary, "no_trade_folds"),
        "accepted_trades": _as_int(summary, "accepted_trades"),
        "accepted_unique_symbols": int(exposure["accepted_unique_symbols"]),
        "accepted_active_days": int(exposure["accepted_active_days"]),
        "accepted_span_days": int(exposure["accepted_span_days"]),
        "accepted_first_ts": exposure["accepted_first_ts"],
        "accepted_last_ts": exposure["accepted_last_ts"],
        "accepted_long_trades": int(exposure["accepted_long_trades"]),
        "accepted_short_trades": int(exposure["accepted_short_trades"]),
        "accepted_min_fold_trades": int(exposure["accepted_min_fold_trades"]),
        "accepted_min_monthly_trades": int(exposure["accepted_min_monthly_trades"]),
        "review_exposure_ok": all(review_exposure_checks.values()),
        "deployment_exposure_ok": all(deployment_exposure_checks.values()),
        "mean_keep_frac": _as_float(summary, "mean_keep_frac"),
        "weighted_full_sl_rate": _as_float(summary, "weighted_full_sl_rate"),
        "weighted_timeout_rate": _as_float(summary, "weighted_timeout_rate"),
        "parity_max_abs_diff": _as_float(parity, "max_abs_diff"),
        "ev_curve_train_source": ev_source,
        "ev_curve_train_rows_replayed": int(ev_rows or 0),
        "ev_curve_train_boundary_violations": int(ev_boundary or 0),
        "guarded_candidate_rows": int(leakage.get("guarded_candidate_rows") or 0),
        "portfolio_accepted_rows": int(leakage.get("portfolio_accepted_rows") or 0),
        "delayed_observation_decision_rows": int(leakage.get("delayed_observation_decision_rows") or 0),
        "signal_time_no_delay_rows": int(leakage.get("signal_time_no_delay_rows") or 0),
        "failed_checks": ",".join(
            [
                key
                for key, value in {
                    **metric_checks,
                    **replay_checks,
                    **review_exposure_checks,
                }.items()
                if not bool(value)
            ]
        ),
        "deployment_exposure_failed_checks": ",".join(
            [key for key, value in deployment_exposure_checks.items() if not bool(value)]
        ),
    }
    return evidence, _file_hash_records(
        replay_name=name,
        replay_dir=replay_dir,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def _champion_comparison(champion: Mapping[str, Any], benchmark: Mapping[str, Any]) -> dict[str, Any]:
    checks = {
        "champion_net_gte_benchmark": _as_float(champion, "sum_net_pnl") >= _as_float(benchmark, "sum_net_pnl"),
        "champion_objective_gte_benchmark": _as_float(champion, "mean_objective") >= _as_float(benchmark, "mean_objective"),
        "champion_positive_share_gte_benchmark": _as_float(champion, "positive_fold_share")
        >= _as_float(benchmark, "positive_fold_share"),
        "champion_worst_fold_gte_benchmark": _as_float(champion, "worst_fold_net_pnl")
        >= _as_float(benchmark, "worst_fold_net_pnl"),
        "champion_full_sl_lte_benchmark": _as_float(champion, "weighted_full_sl_rate")
        <= _as_float(benchmark, "weighted_full_sl_rate") + 1e-12,
        "champion_timeout_lte_benchmark": _as_float(champion, "weighted_timeout_rate")
        <= _as_float(benchmark, "weighted_timeout_rate") + 1e-12,
        "champion_trades_gte_benchmark": _as_int(champion, "accepted_trades") >= _as_int(benchmark, "accepted_trades"),
    }
    return {
        "champion": champion.get("name"),
        "benchmark": benchmark.get("name"),
        "pass": all(checks.values()),
        "net_delta": _as_float(champion, "sum_net_pnl") - _as_float(benchmark, "sum_net_pnl"),
        "mean_objective_delta": _as_float(champion, "mean_objective") - _as_float(benchmark, "mean_objective"),
        "worst_fold_delta": _as_float(champion, "worst_fold_net_pnl") - _as_float(benchmark, "worst_fold_net_pnl"),
        "full_sl_delta": _as_float(champion, "weighted_full_sl_rate") - _as_float(benchmark, "weighted_full_sl_rate"),
        "timeout_delta": _as_float(champion, "weighted_timeout_rate") - _as_float(benchmark, "weighted_timeout_rate"),
        "failed_checks": ",".join([key for key, value in checks.items() if not value]),
        **checks,
    }


def _fmt_table(frame: pd.DataFrame, cols: list[str], max_rows: int = 40) -> str:
    if frame.empty:
        return "_No rows._"
    view = frame[[col for col in cols if col in frame.columns]].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:,.6f}")
    return view.to_markdown(index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-replay-dir", type=Path, default=DEFAULT_FIXED_REPLAY_DIR)
    parser.add_argument("--adaptive-replay-dir", type=Path, default=DEFAULT_ADAPTIVE_REPLAY_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--champion", choices=["fixed_h9", "anchored_adaptive"], default="anchored_adaptive")
    parser.add_argument("--min-net-pnl", type=float, default=0.0)
    parser.add_argument("--min-mean-objective", type=float, default=0.0)
    parser.add_argument("--min-positive-fold-share", type=float, default=0.67)
    parser.add_argument("--max-full-sl-rate", type=float, default=0.22)
    parser.add_argument("--max-timeout-rate", type=float, default=0.55)
    parser.add_argument("--min-worst-fold-net-pnl", type=float, default=-250.0)
    parser.add_argument("--max-no-trade-folds", type=int, default=0)
    parser.add_argument("--min-review-accepted-trades", type=int, default=30)
    parser.add_argument("--min-review-active-days", type=int, default=15)
    parser.add_argument("--min-review-unique-symbols", type=int, default=15)
    parser.add_argument("--min-review-fold-trades", type=int, default=2)
    parser.add_argument("--min-review-monthly-trades", type=int, default=5)
    parser.add_argument("--min-deployment-accepted-trades", type=int, default=100)
    parser.add_argument("--min-deployment-active-days", type=int, default=30)
    parser.add_argument("--min-deployment-monthly-trades", type=int, default=20)
    args = parser.parse_args()

    thresholds = _metric_thresholds(args)
    fixed, fixed_hashes = _evaluate_replay(
        name="fixed_h9",
        role="benchmark",
        replay_dir=args.fixed_replay_dir,
        thresholds=thresholds,
    )
    adaptive, adaptive_hashes = _evaluate_replay(
        name="anchored_adaptive",
        role="candidate",
        replay_dir=args.adaptive_replay_dir,
        thresholds=thresholds,
    )
    evidence_rows = [fixed, adaptive]
    evidence = pd.DataFrame(evidence_rows)
    champion = adaptive if args.champion == "anchored_adaptive" else fixed
    benchmark = fixed if args.champion == "anchored_adaptive" else adaptive
    comparison = pd.DataFrame([_champion_comparison(champion, benchmark)])
    hashes = pd.DataFrame(fixed_hashes + adaptive_hashes)

    package_gate_pass = bool(
        evidence["package_gate_pass"].all()
        and bool(comparison["pass"].iloc[0])
    )
    any_signal_time_unsafe = bool((~evidence["signal_time_safe"].astype(bool)).any())
    deployment_exposure_ready = bool(evidence["deployment_exposure_ok"].astype(bool).all())
    package_status = "conditional_pass" if package_gate_pass and any_signal_time_unsafe else "pass" if package_gate_pass else "fail"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "evidence": args.out_dir / "guarded_execution_policy_package_evidence.csv",
        "comparison": args.out_dir / "guarded_execution_policy_comparison.csv",
        "file_hashes": args.out_dir / "guarded_execution_policy_file_hashes.csv",
        "live_contract": args.out_dir / "guarded_execution_policy_live_contract.json",
        "manifest": args.out_dir / "guarded_execution_policy_package_manifest.json",
        "report": args.out_dir / "guarded_execution_policy_package_report.md",
    }
    evidence.to_csv(paths["evidence"], index=False)
    comparison.to_csv(paths["comparison"], index=False)
    hashes.to_csv(paths["file_hashes"], index=False)

    live_contract = {
        "package_status": package_status,
        "champion": champion["name"],
        "benchmark": benchmark["name"],
        "deployment_ready": bool(package_gate_pass and not any_signal_time_unsafe and deployment_exposure_ready),
        "deployment_blocker": (
            "Guard is frozen-replay safe but not signal-time safe; live execution must enforce "
            "guard_decision_timestamp after delayed-entry observation where required. "
            "Exposure is also below the deployment breadth floor."
            if not deployment_exposure_ready
            else "Guard is frozen-replay safe but not signal-time safe; live execution must enforce "
            "guard_decision_timestamp after delayed-entry observation where required."
        )
        if any_signal_time_unsafe
        else "Exposure is below the deployment breadth floor."
        if not deployment_exposure_ready
        else "",
        "candidate_replay_dir": champion["replay_dir"],
        "required_runtime_contract": {
            "do_not_refit_guard": True,
            "do_not_reselect_thresholds": True,
            "do_not_reselect_scenarios": True,
            "use_materialized_scenario_choice": True,
            "use_materialized_guard_threshold": True,
            "enforce_guard_decision_timestamp": True,
            "allow_signal_time_fallback_only_when_no_delayed_entry_observation": True,
        },
        "metrics": {
            "sum_net_pnl": champion["sum_net_pnl"],
            "mean_objective": champion["mean_objective"],
            "worst_fold_net_pnl": champion["worst_fold_net_pnl"],
            "positive_fold_share": champion["positive_fold_share"],
            "accepted_trades": champion["accepted_trades"],
            "accepted_unique_symbols": champion["accepted_unique_symbols"],
            "accepted_active_days": champion["accepted_active_days"],
            "accepted_span_days": champion["accepted_span_days"],
            "accepted_long_trades": champion["accepted_long_trades"],
            "accepted_short_trades": champion["accepted_short_trades"],
            "accepted_min_fold_trades": champion["accepted_min_fold_trades"],
            "accepted_min_monthly_trades": champion["accepted_min_monthly_trades"],
            "weighted_full_sl_rate": champion["weighted_full_sl_rate"],
            "weighted_timeout_rate": champion["weighted_timeout_rate"],
        },
    }
    paths["live_contract"].write_text(json.dumps(_json_safe(live_contract), indent=2), encoding="utf-8")

    manifest: dict[str, Any] = {
        "generated_by": "package_guarded_execution_policy",
        "status": package_status,
        "package_gate_pass": package_gate_pass,
        "champion": champion["name"],
        "benchmark": benchmark["name"],
        "thresholds": thresholds,
        "fixed_replay_dir": str(args.fixed_replay_dir),
        "adaptive_replay_dir": str(args.adaptive_replay_dir),
        "out_dir": str(args.out_dir),
        "evidence": evidence.to_dict(orient="records"),
        "comparison": comparison.to_dict(orient="records"),
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    manifest["package_hash"] = _sha256_json({k: v for k, v in manifest.items() if k != "package_hash"})
    paths["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2, sort_keys=True), encoding="utf-8")

    report_lines = [
        "# Guarded Execution Policy Package",
        "",
        f"Status: `{package_status}`",
        f"Champion: `{champion['name']}`",
        f"Benchmark: `{benchmark['name']}`",
        f"Package hash: `{manifest['package_hash']}`",
        "",
        "## Evidence",
        "",
        _fmt_table(
            evidence,
            [
                "name",
                "status",
                "scenario",
                "sum_net_pnl",
                "mean_objective",
                "worst_fold_net_pnl",
                "positive_fold_share",
                "accepted_trades",
                "accepted_unique_symbols",
                "accepted_active_days",
                "accepted_span_days",
                "accepted_min_fold_trades",
                "accepted_min_monthly_trades",
                "weighted_full_sl_rate",
                "weighted_timeout_rate",
                "parity_max_abs_diff",
                "decision_time_safe",
                "signal_time_safe",
                "review_exposure_ok",
                "deployment_exposure_ok",
            ],
        ),
        "",
        "## Champion vs Benchmark",
        "",
        _fmt_table(
            comparison,
            [
                "champion",
                "benchmark",
                "pass",
                "net_delta",
                "mean_objective_delta",
                "worst_fold_delta",
                "full_sl_delta",
                "timeout_delta",
                "failed_checks",
            ],
        ),
        "",
        "## Runtime Caveat",
        "",
        (
            "The champion is frozen-replay safe but not signal-time safe. The guard must be evaluated "
            "at `guard_decision_timestamp`; rows with delayed-entry observations cannot be admitted at "
            "the original signal timestamp."
            if any_signal_time_unsafe
            else "No signal-time caveat was detected."
        ),
        "",
        "## Exposure Caveat",
        "",
        (
            "The champion passes the review exposure floor, but not the deployment exposure floor. Treat this as "
            "a frozen-policy review candidate and require a broader validation window before live promotion."
            if not deployment_exposure_ready
            else "The champion passes the deployment exposure floor."
        ),
        "",
        "## Outputs",
        "",
        _fmt_table(pd.DataFrame([{"artifact": key, "path": str(value)} for key, value in paths.items()]), ["artifact", "path"]),
    ]
    paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            _json_safe(
                {
                    "status": package_status,
                    "package_gate_pass": package_gate_pass,
                    "champion": champion["name"],
                    "benchmark": benchmark["name"],
                    "package_hash": manifest["package_hash"],
                    "outputs": {key: str(value) for key, value in paths.items()},
                }
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
