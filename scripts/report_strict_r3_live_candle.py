#!/usr/bin/env python3
"""Emit one immutable, read-only operational result for a live strict-R3 hour.

The report deliberately consumes receipts only.  It never refreshes data,
scores candidates, changes state, or calls Kraken; the live producer remains
the only entry writer and the position monitor remains the only exit writer.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp" / "artifacts"
REPORTS = ROOT / "data_perp" / "reports"
DEFAULT_LIVE_STATE = (
    ROOT / "data_perp" / "live" /
    "strict_r3_kraken_live_state_v32_v52_full_runtime_guard.json"
)


def utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in frame:
        return {}
    return {
        str(key): int(value)
        for key, value in frame[column].fillna("unknown").astype(str).value_counts().sort_index().items()
    }


def _candidate_symbol(candidate_id: object) -> str:
    """Return the immutable exchange symbol prefix from a strict-R3 ID."""
    return str(candidate_id).split("|", 1)[0]


def live_state_entry_consistency(
    execution_payload: dict[str, Any],
    *,
    state_path: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Read-only postcondition for entry/portfolio asset uniqueness.

    The report deliberately does not query Kraken or modify state.  It verifies
    that the canonical state held immediately after the producer has one row
    per exchange symbol, and that every entry action is represented there.
    This makes an asset-limit regression an explicit operational incident even
    if the primary producer otherwise emitted a successful receipt.
    """
    state_path = state_path or DEFAULT_LIVE_STATE
    if not state_path.is_file():
        return {"state_path": rel(state_path), "available": False}, [
            "live_state_missing_for_entry_consistency",
        ]
    try:
        state = load(state_path)
        positions = list(state.get("positions") or [])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"state_path": rel(state_path), "available": False}, [
            f"live_state_unreadable_for_entry_consistency: {exc}",
        ]
    symbols = [str(row.get("exchange_symbol") or "") for row in positions]
    candidate_ids = {str(row.get("candidate_id") or "") for row in positions}
    issues: list[str] = []
    duplicates = sorted(
        symbol for symbol in set(symbols) if symbol and symbols.count(symbol) > 1
    )
    if duplicates:
        issues.append(f"live_state_duplicate_symbols: {','.join(duplicates)}")
    entries = [
        action for action in list(execution_payload.get("actions") or [])
        if str(action.get("action") or "") == "entry"
    ]
    absent_entry_ids = sorted(
        str(action.get("candidate_id") or "")
        for action in entries
        if str(action.get("candidate_id") or "") not in candidate_ids
    )
    if absent_entry_ids:
        issues.append(
            "entry_actions_absent_from_live_state: " + ",".join(absent_entry_ids)
        )
    entry_symbols = [_candidate_symbol(action.get("candidate_id")) for action in entries]
    duplicate_entries = sorted(
        symbol for symbol in set(entry_symbols) if symbol and entry_symbols.count(symbol) > 1
    )
    if duplicate_entries:
        issues.append(f"duplicate_entry_actions_for_symbol: {','.join(duplicate_entries)}")
    return {
        "state_path": rel(state_path),
        "available": True,
        "position_rows": len(positions),
        "unique_symbols": len(set(symbols)),
        "entry_actions": len(entries),
        "entry_actions_present_in_state": len(entries) - len(absent_entry_ids),
    }, issues


def producer(runtime: str, decision: pd.Timestamp) -> Path | None:
    tag = decision.strftime("%Y%m%dT%H%M%SZ")
    runtime_pattern = "*" if runtime in {"*", "any"} else runtime
    receipts = sorted(ARTIFACTS.glob(
        f"strict_r3_live_hourly_producer_{runtime_pattern}_{tag}_v*/run_manifest.json"
    ))
    if not receipts:
        return None
    parsed = [(path, load(path)) for path in receipts]
    passed = [item for item in parsed if item[1].get("status") == "pass"]
    return (passed[-1] if passed else parsed[-1])[0].parent


def producer_runtime(receipt: Path) -> str:
    """Return the receipt's actual runtime namespace, never the observer's."""
    match = re.match(
        r"^strict_r3_live_hourly_producer_(.+)_\d{8}T\d{6}Z_v\d+$",
        receipt.name,
    )
    if not match:
        raise ValueError(f"unrecognised producer receipt name: {receipt.name}")
    return match.group(1)


def path_from(value: object) -> Path | None:
    if not value:
        return None
    path = ROOT / str(value)
    return path if path.exists() else None


def refresh_for(receipt: Path, runtime: str, decision: pd.Timestamp) -> Path | None:
    try:
        attempt = receipt.name.rsplit("_v", 1)[1]
    except IndexError:
        return None
    tag = decision.strftime("%Y%m%dT%H%M%SZ")
    path = ARTIFACTS / f"strict_r3_live_15m_refresh_{runtime}_{tag}_v{attempt}"
    return path if path.exists() else None


def monitor_for(decision: pd.Timestamp) -> tuple[Path, dict[str, Any]] | None:
    values: list[tuple[pd.Timestamp, Path, dict[str, Any]]] = []
    for manifest in ARTIFACTS.glob("strict_r3_live_position_monitor_*/monitor_*/run_manifest.json"):
        try:
            payload = load(manifest)
            observed = utc(payload["observed_at"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if decision <= observed < decision + pd.Timedelta(hours=1, minutes=10):
            values.append((observed, manifest.parent, payload))
    if not values:
        return None
    _, directory, payload = max(values, key=lambda row: row[0])
    return directory, payload


def source_summary(directory: Path | None) -> tuple[dict[str, int], list[str]]:
    if directory is None or not (directory / "coverage_after_retry.parquet").exists():
        return {}, ["source_refresh_receipt_missing"]
    frame = pd.read_parquet(directory / "coverage_after_retry.parquet")
    false = pd.Series(False, index=frame.index)
    result = {
        "universe": int(len(frame)),
        "source_ready": int(frame.get("feature_source_ready", false).fillna(False).sum()),
        "missing_15m_bar": int(frame.get("missing_15m_bar", false).fillna(False).sum()),
        "synthetic_flat_bar": int(frame.get("synthetic_flat_bar", false).fillna(False).sum()),
        "missing_decision_open": int(frame.get("missing_decision_open", false).fillna(False).sum()),
    }
    return result, (["source_coverage_incomplete"] if result["source_ready"] != result["universe"] else [])


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Strict-R3 live candle — {report['decision_ts']}", "",
        f"Status: **{report['status']}**", "",
        "| Funnel stage | Rows |", "|---|---:|",
    ]
    for key in ("universe", "source_ready", "candidate_eligible", "feature_complete", "base_top30", "mc1_mapped", "mc1_admitted", "portfolio_accepted", "orders_submitted"):
        lines.append(f"| {key.replace('_', ' ')} | {report['funnel'].get(key, '—')} |")
    lines.extend(["", "## Candidate rejection reasons", ""])
    reasons = report.get("candidate_rejections", {})
    if reasons:
        lines.extend(["| Reason | Rows |", "|---|---:|"])
        lines.extend(f"| {key} | {value} |" for key, value in reasons.items())
    else:
        lines.append("None or scoring did not reach candidate materialisation.")
    lines.extend(["", "## Position monitor", ""])
    monitor = report.get("position_monitor", {})
    if monitor:
        lines.extend(f"- {key.replace('_', ' ')}: `{value}`" for key, value in monitor.items())
    else:
        lines.append("No monitor receipt in this candle window.")
    lines.extend(["", "## Irregularities / required action", ""])
    irregularities = report.get("irregularities", [])
    if irregularities:
        lines.extend(f"- **{value}**" for value in irregularities)
        lines.append("")
        lines.append("Required workflow: preserve evidence → trace root cause → apply the narrowest causal patch → test no-submit same-candle replay → reseal/restart the singleton services → validate live state → restore live authority. Do not treat a retry as a repair.")
    else:
        lines.append("No irregularity found in the terminal receipts.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-tag", required=True)
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument(
        "--live-state", type=Path, default=DEFAULT_LIVE_STATE,
        help=(
            "Canonical live state to verify against entry actions.  The "
            "reporter is read-only; this explicit input prevents a runtime "
            "successor state from being compared to a retired static path."
        ),
    )
    args = parser.parse_args()
    decision = utc(args.decision_ts)
    receipt = producer(args.runtime_tag, decision)
    if receipt is None:
        # Non-terminal: the companion loop continues observing this candle.
        raise SystemExit(2)
    payload = load(receipt / "run_manifest.json")
    receipt_runtime = producer_runtime(receipt)
    if utc(payload["decision_ts"]) != decision:
        raise AssertionError("producer receipt decision timestamp mismatch")
    issues: list[str] = []
    if payload.get("status") != "pass":
        issues.append(f"producer_{payload.get('status')}: {payload.get('error', 'no detail')}")
    refresh = refresh_for(receipt, receipt_runtime, decision)
    source, source_issues = source_summary(refresh)
    issues.extend(source_issues)
    funnel: dict[str, int] = dict(source)
    rejections: dict[str, int] = {}
    run = path_from(payload.get("hourly_run"))
    if run is None:
        issues.append("hourly_run_missing")
    else:
        candidates_path = run / "candidate_grid/target_free_candidate_population.parquet"
        decisions_path = run / "cycle/shadow_decisions.parquet"
        if not candidates_path.exists() or not decisions_path.exists():
            issues.append("stage_artifact_missing")
        else:
            candidates = pd.read_parquet(candidates_path)
            current_candidates = candidates.loc[pd.to_datetime(candidates["__decision_ts__"], utc=True).eq(decision)]
            eligible = current_candidates["eligibility_reason"].eq("eligible")
            funnel["universe"] = int(len(current_candidates))
            funnel["candidate_eligible"] = int(eligible.sum())
            rejections = counts(current_candidates.loc[~eligible], "eligibility_reason")
            decisions = pd.read_parquet(decisions_path)
            current = decisions.loc[pd.to_datetime(decisions["__decision_ts__"], utc=True).eq(decision)]
            funnel["feature_complete"] = int(len(current))
            routed = current.get("base_route_timestamp", pd.Series(False, index=current.index)).fillna(False).astype(bool)
            mapped = pd.to_numeric(current.get("mc1_d2_expected_net_bps"), errors="coerce").notna()
            admitted = current.get("mc1_d2_admitted_ge_50bps", pd.Series(False, index=current.index)).fillna(False).astype(bool)
            accepted = current.get("portfolio_accepted", pd.Series(False, index=current.index)).fillna(False).astype(bool)
            funnel.update(base_top30=int(routed.sum()), mc1_mapped=int(mapped.sum()), mc1_admitted=int(admitted.sum()), portfolio_accepted=int(accepted.sum()))
    execution = path_from(payload.get("execution_receipt"))
    if execution is None:
        issues.append("execution_receipt_missing")
        funnel["orders_submitted"] = 0
    else:
        target = execution / "run_manifest.json" if execution.is_dir() else execution
        execution_payload = load(target)
        funnel["orders_submitted"] = int(execution_payload.get("orders_submitted", execution_payload.get("execution_selected_rows", 0)) or 0)
        entry_consistency, entry_issues = live_state_entry_consistency(
            execution_payload, state_path=args.live_state,
        )
        issues.extend(entry_issues)
    monitor_payload = monitor_for(decision)
    monitor: dict[str, Any] = {}
    if monitor_payload:
        directory, item = monitor_payload
        monitor = {"observed_at": item.get("observed_at"), "status": item.get("status", "pass"), "positions_before": item.get("positions_before"), "positions_after": item.get("positions_after"), "actions": item.get("actions"), "receipt": rel(directory)}
        if item.get("status") not in {None, "pass"}:
            issues.append(f"position_monitor_{item.get('status')}: {item.get('error', 'no detail')}")
    else:
        issues.append("position_monitor_receipt_missing")
    report = {
        "schema": "strict_r3_live_candle_report_v2",
        "generated_at": pd.Timestamp(datetime.now(timezone.utc)).isoformat(),
        "decision_ts": decision.isoformat(), "runtime_tag": receipt_runtime,
        "status": "action_required" if issues else "pass",
        "producer_receipt": rel(receipt), "hourly_run": rel(run),
        "funnel": funnel, "candidate_rejections": rejections,
        "position_monitor": monitor,
        "entry_state_consistency": entry_consistency if execution is not None else {},
        "irregularities": issues,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    # Reports are immutable *per producer attempt*.  An incident receipt may
    # therefore remain available for root-cause analysis while a later tested
    # same-candle successor writes its own terminal result instead of being
    # hidden by an overwritten report.
    stem = (
        f"strict_r3_live_candle_{receipt_runtime}_"
        f"{decision.strftime('%Y%m%dT%H%M%SZ')}_{receipt.name}_"
        f"state_{re.sub(r'[^A-Za-z0-9]+', '_', args.live_state.name).strip('_')}"
    )
    destinations = []
    for suffix, body in (("json", json.dumps(report, indent=2, default=str) + "\n"), ("md", markdown(report))):
        destination = REPORTS / f"{stem}.{suffix}"
        if not destination.exists():
            destination.write_text(body)
        destinations.append(destination)
    print(json.dumps({
        "status": report["status"], "report": rel(destinations[-1]),
        "producer_attempt": receipt.name,
    }))
    # A producer failure is an incident, not a completed candle.  The
    # persistent observer keeps watching for the tested successor attempt;
    # it has no retry, source-refresh, or exchange-writing authority itself.
    if report["status"] != "pass":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
