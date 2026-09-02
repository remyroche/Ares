#!/usr/bin/env python3
"""Write a compact, read-only operational report for one strict-R3 live candle.

This reporter intentionally never refreshes data, scores candidates, modifies state,
or calls Kraken.  It consumes only immutable producer/run/audit/executor receipts.
That separation makes the xx:10 operational review safe to run beside the live
writer and prevents a diagnostic process from changing an entry decision.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp" / "artifacts"
REPORTS = ROOT / "data_perp" / "reports"


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _reason_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    values = frame[column].fillna("unknown").astype(str)
    return {key: int(value) for key, value in values.value_counts().sort_index().items()}


def _stage(rows: int, **extra: object) -> dict[str, object]:
    return {"rows": int(rows), **extra}


def _producer_for(*, runtime_tag: str, decision: pd.Timestamp) -> Path:
    tag = decision.strftime("%Y%m%dT%H%M%SZ")
    matches = sorted(
        ARTIFACTS.glob(f"strict_r3_live_hourly_producer_{runtime_tag}_{tag}_v*/run_manifest.json"),
    )
    if not matches:
        raise FileNotFoundError(
            f"no immutable producer receipt yet for {runtime_tag} {decision.isoformat()}"
        )
    # A pass is definitive.  Otherwise preserve the most recent fail-closed attempt.
    payloads = [(path, _read_json(path)) for path in matches]
    passed = [(path, data) for path, data in payloads if data.get("status") == "pass"]
    return (passed[-1] if passed else payloads[-1])[0].parent


def _optional_path(value: object) -> Path | None:
    if not value:
        return None
    path = ROOT / str(value)
    return path if path.exists() else None


def _linked_refresh(*, producer: Path, runtime_tag: str, decision: pd.Timestamp) -> Path | None:
    """Recover the deterministic refresh receipt adjacent to a producer attempt.

    The producer stores the refresh *summary* directly in its manifest, not its
    path.  Both artifacts share the exact runtime/decision/attempt namespace.
    """
    try:
        attempt = producer.name.rsplit("_v", 1)[1]
    except IndexError:
        return None
    tag = decision.strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACTS / f"strict_r3_live_15m_refresh_{runtime_tag}_{tag}_v{attempt}"
    return path if path.exists() else None


def _latest_monitor(*, decision: pd.Timestamp) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[pd.Timestamp, Path, dict[str, Any]]] = []
    for path in ARTIFACTS.glob("strict_r3_live_position_monitor_*/monitor_*/run_manifest.json"):
        try:
            payload = _read_json(path)
            observed = _utc(payload["observed_at"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        # Include the monitor state immediately after this entry decision only.
        if decision <= observed < decision + pd.Timedelta(hours=1, minutes=10):
            candidates.append((observed, path, payload))
    if not candidates:
        return None
    _, path, payload = max(candidates, key=lambda item: item[0])
    return path.parent, payload


def _coverage(refresh: Path | None) -> tuple[dict[str, object] | None, list[str]]:
    if refresh is None:
        return None, ["source_refresh_receipt_missing"]
    coverage_path = refresh / "coverage_after_retry.parquet"
    if not coverage_path.exists():
        return None, ["source_coverage_artifact_missing"]
    frame = pd.read_parquet(coverage_path)
    columns = set(frame.columns)
    ready = (
        frame["feature_source_ready"].fillna(False).astype(bool)
        if "feature_source_ready" in columns else pd.Series(False, index=frame.index)
    )
    summary: dict[str, object] = _stage(
        len(frame),
        ready=int(ready.sum()),
        missing_15m_bar=int(frame.get("missing_15m_bar", pd.Series(False, index=frame.index)).fillna(False).sum()),
        synthetic_flat_bar=int(frame.get("synthetic_flat_bar", pd.Series(False, index=frame.index)).fillna(False).sum()),
        missing_decision_open=int(frame.get("missing_decision_open", pd.Series(False, index=frame.index)).fillna(False).sum()),
    )
    issues: list[str] = []
    if int(summary["ready"]) != len(frame):
        issues.append("source_coverage_incomplete")
    return summary, issues


def _as_markdown(report: dict[str, Any]) -> str:
    decision = report["decision_ts"]
    lines = [
        f"# Strict-R3 live candle report — {decision}",
        "",
        f"Status: **{report['status']}**. This is a read-only post-decision report; it did not call Kraken or alter live state.",
        "",
        "## Funnel",
        "",
        "| Stage | Rows | Detail |",
        "|---|---:|---|",
    ]
    funnel = report["funnel"]
    order = (
        ("universe", "target-free universe"),
        ("source", "complete exchange-observed source"),
        ("candidate_eligible", "candidate eligibility"),
        ("feature_complete", "frozen 120-field feature contract"),
        ("base_routed", "base timestamp-local top-30% route"),
        ("mc1_mapped", "Robust-21 + MC1 expected-EV mapped"),
        ("mc1_admitted", "MC1 admission ≥ +50 bps"),
        ("portfolio_accepted", "portfolio auction accepted"),
        ("execution_submitted", "live order submitted"),
    )
    for key, label in order:
        entry = funnel.get(key, {})
        rows = entry.get("rows", "—")
        detail = ", ".join(
            f"{name.replace('_', ' ')}={value}"
            for name, value in entry.items() if name != "rows"
        ) or "—"
        lines.append(f"| {label} | {rows} | {detail} |")

    reasons = report.get("candidate_rejections", {})
    lines.extend(["", "## Candidate rejections", ""])
    if reasons:
        lines.extend(["| Reason | Symbols |", "|---|---:|"])
        lines.extend(f"| {reason} | {count} |" for reason, count in reasons.items())
    else:
        lines.append("None.")

    execution = report.get("execution", {})
    lines.extend(["", "## Execution", ""])
    if execution:
        for key, value in execution.items():
            lines.append(f"- {key.replace('_', ' ')}: `{value}`")
    else:
        lines.append("No execution receipt was available.")

    monitor = report.get("position_monitor", {})
    lines.extend(["", "## Exit monitoring", ""])
    if monitor:
        for key, value in monitor.items():
            lines.append(f"- {key.replace('_', ' ')}: `{value}`")
    else:
        lines.append("No position-monitor receipt was available in the first 70 minutes after this decision.")

    issues = report.get("irregularities", [])
    lines.extend(["", "## Irregularities", ""])
    if issues:
        lines.extend(f"- **{issue}**" for issue in issues)
        lines.append("")
        lines.append(
            "Required response: preserve the receipt, trace the producing data/code "
            "lineage to the root cause, patch and test that root cause in no-submit "
            "mode, reseal/restart the singleton writer, then validate the repaired "
            "chain and restore its live entry authority. The reporter itself never "
            "performs an unsafe blind retry or patch."
        )
    else:
        lines.append("None detected in the immutable receipts.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-tag", required=True)
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument("--out-dir", type=Path, default=REPORTS)
    args = parser.parse_args()

    decision = _utc(args.decision_ts)
    producer = _producer_for(runtime_tag=args.runtime_tag, decision=decision)
    manifest = _read_json(producer / "run_manifest.json")
    if _utc(manifest["decision_ts"]) != decision:
        raise AssertionError("producer receipt decision timestamp does not match requested candle")

    irregularities: list[str] = []
    if manifest.get("status") != "pass":
        irregularities.append(f"producer_{manifest.get('status', 'unknown')}: {manifest.get('error', 'no error detail')}")

    run = _optional_path(manifest.get("hourly_run"))
    refresh = _linked_refresh(
        producer=producer, runtime_tag=args.runtime_tag, decision=decision,
    )
    audit = _optional_path(manifest.get("live_hour_audit"))
    executor = _optional_path(manifest.get("execution_receipt"))
    source, source_issues = _coverage(refresh)
    irregularities.extend(source_issues)

    funnel: dict[str, dict[str, object]] = {
        "universe": _stage(int(manifest.get("population_rows", 0))),
        "source": source or _stage(0),
    }
    if not int(funnel["universe"]["rows"]) and source is not None:
        funnel["universe"] = _stage(int(source["rows"]))
    rejection_counts: dict[str, int] = {}
    current: pd.DataFrame | None = None
    if run is None:
        irregularities.append("hourly_run_missing")
    else:
        candidate_path = run / "candidate_grid/target_free_candidate_population.parquet"
        decisions_path = run / "cycle/shadow_decisions.parquet"
        if not candidate_path.exists() or not decisions_path.exists():
            irregularities.append("hourly_run_stage_artifact_missing")
        else:
            candidates = pd.read_parquet(candidate_path)
            candidate_now = candidates.loc[
                pd.to_datetime(candidates["__decision_ts__"], utc=True).eq(decision)
            ].copy()
            rejection_counts = {
                reason: count for reason, count in _reason_counts(
                    candidate_now.loc[~candidate_now["eligibility_reason"].eq("eligible")],
                    "eligibility_reason",
                ).items()
            }
            candidate_eligible = candidate_now["eligibility_reason"].eq("eligible")
            funnel["universe"] = _stage(len(candidate_now))
            funnel["candidate_eligible"] = _stage(int(candidate_eligible.sum()), rejected=int((~candidate_eligible).sum()))
            decisions = pd.read_parquet(decisions_path)
            current = decisions.loc[
                pd.to_datetime(decisions["__decision_ts__"], utc=True).eq(decision)
            ].copy()
            feature_rows = int(manifest.get("current_feature_parity_rows", len(current)))
            funnel["feature_complete"] = _stage(feature_rows, skipped=int(manifest.get("row_local_feature_skip_audit", {}).get("skipped_rows", 0)))
            if len(current) != feature_rows:
                irregularities.append("feature_complete_rows_do_not_match_decision_rows")
            route = current.get("base_route_timestamp", pd.Series(False, index=current.index)).fillna(False).astype(bool)
            mapped = pd.to_numeric(current.get("mc1_d2_expected_net_bps"), errors="coerce").notna()
            admitted = current.get("mc1_d2_admitted_ge_50bps", pd.Series(False, index=current.index)).fillna(False).astype(bool)
            accepted = current.get("portfolio_accepted", pd.Series(False, index=current.index)).fillna(False).astype(bool)
            funnel["base_routed"] = _stage(int(route.sum()), stopped_after_base=int((~route).sum()))
            funnel["mc1_mapped"] = _stage(int(mapped.sum()))
            funnel["mc1_admitted"] = _stage(int(admitted.sum()), rejected_below_threshold=int((route & ~admitted).sum()))
            funnel["portfolio_accepted"] = _stage(int(accepted.sum()), rejected=int((admitted & ~accepted).sum()), reasons=_reason_counts(current.loc[admitted & ~accepted], "portfolio_rejection_reason"))

            if not candidate_now.empty and len(candidate_now) != 170:
                irregularities.append(f"universe_not_170:{len(candidate_now)}")
            if source is not None and int(source.get("rows", 0)) != len(candidate_now):
                irregularities.append("source_and_candidate_universe_row_count_mismatch")

    execution_summary: dict[str, object] = {}
    submitted = 0
    if executor is not None and (executor / "run_manifest.json").exists():
        exec_manifest = _read_json(executor / "run_manifest.json")
        execution_summary = {
            "status": exec_manifest.get("status"),
            "exchange_write_calls": exec_manifest.get("exchange_write_calls"),
            "orders_submitted": exec_manifest.get("orders_submitted", exec_manifest.get("submitted_orders")),
            "rejected": exec_manifest.get("rejected_orders", exec_manifest.get("rejections")),
            "receipt": _relative(executor),
        }
        submitted = int(execution_summary["orders_submitted"] or 0)
        if str(exec_manifest.get("status", "")).lower() not in {"pass", "ok", "success"}:
            irregularities.append("execution_receipt_nonpass")
    elif manifest.get("status") == "pass":
        irregularities.append("execution_receipt_missing")
    funnel["execution_submitted"] = _stage(submitted)

    if audit is not None and (audit / "run_manifest.json").exists():
        audit_manifest = _read_json(audit / "run_manifest.json")
        false_checks = [key for key, value in audit_manifest.get("checks", {}).items() if not bool(value)]
        if audit_manifest.get("status") != "pass" or false_checks:
            irregularities.append("live_hour_audit_failed:" + ",".join(false_checks or [str(audit_manifest.get("status"))]))
    elif manifest.get("status") == "pass":
        irregularities.append("live_hour_audit_missing")

    monitor_payload = _latest_monitor(decision=decision)
    monitor_summary: dict[str, object] = {}
    if monitor_payload is not None:
        monitor_dir, monitor = monitor_payload
        monitor_summary = {
            "observed_at": monitor.get("observed_at"),
            "positions_before": monitor.get("positions_before"),
            "positions_after": monitor.get("positions_after"),
            "actions": monitor.get("actions"),
            "exchange_write_calls": monitor.get("exchange_write_calls"),
            "receipt": _relative(monitor_dir),
        }
        if monitor.get("mode") != "live":
            irregularities.append("position_monitor_not_live")
    else:
        irregularities.append("position_monitor_receipt_missing")

    status = "action_required" if irregularities else "pass"
    report = {
        "schema": "strict_r3_live_candle_report_v1",
        "status": status,
        "generated_at": pd.Timestamp(datetime.now(timezone.utc)).isoformat(),
        "decision_ts": decision.isoformat(),
        "runtime_tag": args.runtime_tag,
        "producer_receipt": _relative(producer),
        "hourly_run": _relative(run) if run else None,
        "funnel": funnel,
        "candidate_rejections": rejection_counts,
        "execution": execution_summary,
        "position_monitor": monitor_summary,
        "irregularities": irregularities,
        "remediation_protocol": (
            "docs/STRICT_R3_LIVE_OPERATIONS_REMEDIATION_PROTOCOL.md"
            if irregularities else None
        ),
    }
    tag = decision.strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"strict_r3_live_candle_{args.runtime_tag}_{tag}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    md_path.write_text(_as_markdown(report))
    print(json.dumps({"status": status, "json": _relative(json_path), "markdown": _relative(md_path)}))


if __name__ == "__main__":
    main()
