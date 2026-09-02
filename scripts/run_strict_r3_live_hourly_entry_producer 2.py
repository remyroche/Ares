#!/usr/bin/env python3
"""Run the strict-R3 live entry producer once per fresh UTC hour.

This is deliberately a thin operational wrapper around the sealed scorer,
live-hour audit, runtime checkpoint and exchange executor.  It never computes
models itself.  A failed source refresh, scorer, audit or checkpoint produces
an immutable fail-closed receipt and no entry attempt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data_perp/artifacts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _retry_schedule(value: str | None, *, first_retry_seconds: float | None) -> tuple[float, ...]:
    """Parse bounded retry checkpoints as elapsed seconds from first refresh."""
    if value:
        parsed = tuple(float(part.strip()) for part in value.split(",") if part.strip())
        if parsed:
            return parsed
    first = 30.0 if first_retry_seconds is None else float(first_retry_seconds)
    return (first, 60.0, 120.0, 180.0)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_tag(inference_bundle: str) -> str:
    """Return a stable receipt namespace from the sealed bundle filename.

    A runtime-only successor must never overwrite or suppress an earlier
    version's immutable receipts at the same decision timestamp.  The legacy
    v45 name is retained for the v45 bundle; successors get their own lineage.
    """
    match = re.search(r"(?:^|_)(v\d+)(?:_|$)", Path(inference_bundle).stem)
    if not match:
        raise ValueError("inference bundle filename lacks a version namespace")
    return match.group(1)


def _next_receipt(prefix: str) -> tuple[Path, int]:
    """Reserve the next immutable attempt after a terminal fail-closed run.

    A successful receipt is never retried: doing so could duplicate an order.
    A failed receipt has no exchange submission and can be safely followed by
    a later attempt for the same fresh decision.
    """
    attempt = 1
    while True:
        candidate = ARTIFACTS / f"{prefix}_v{attempt}"
        if not candidate.exists():
            return candidate, attempt
        manifest = candidate / "run_manifest.json"
        if not manifest.is_file():
            raise FileExistsError(
                f"producer receipt is in progress or incomplete: {candidate}"
            )
        try:
            status = str(json.loads(manifest.read_text()).get("status") or "")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise FileExistsError(
                f"producer receipt cannot be classified safely: {candidate}"
            ) from exc
        if status == "pass":
            raise FileExistsError(f"successful immutable producer receipt exists: {candidate}")
        attempt += 1


def _run(command: list[str], *, log: Path) -> None:
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=ROOT, stdout=handle,
                                   stderr=subprocess.STDOUT, text=True, check=False)
    if completed.returncode:
        raise RuntimeError(f"stage failed ({completed.returncode}); see {log}")


def _successful_predecessor(*, decision: pd.Timestamp, bundle_hash: str,
                            bootstrap: Path) -> Path:
    """Find the latest verified lock-step state, never a stale score.

    A live-window completion is preferred.  A separately labelled v45
    ``backfill`` run is admissible only as a causal *state bridge* after an
    operational interruption: it has zero exchange calls, the same sealed
    bundle, and a complete feature/Geometry/K9/portfolio successor state.  It
    is not executable evidence and is never treated as a live decision.
    """
    candidates: list[tuple[pd.Timestamp, int, Path]] = []
    manifests = list(ARTIFACTS.glob("strict_r3_successor_*_*/run_manifest.json"))
    for path in manifests:
        try:
            payload = json.loads(path.read_text())
            timestamp = _utc(payload["decision_ts"])
            is_live = "_live_" in path.parent.name
            is_backfill = "_backfill_" in path.parent.name
            valid_completion = bool(payload.get("completed_within_live_decision_window")) if is_live else (
                is_backfill
                and str(payload.get("mode")) == "shadow-only"
                and int(payload.get("exchange_calls", -1)) == 0
            )
            if (
                timestamp < decision
                and str(payload.get("hashes", {}).get("inference_bundle")) == bundle_hash
                and valid_completion
                and (path.parent / "feature_state/bundle/state_bundle_manifest.json").is_file()
                and (path.parent / "cycle/next_portfolio_state.json").is_file()
                and (path.parent / "cycle/score/geometry_k9_state/causal_geometry_k9_history.parquet").is_file()
            ):
                # Prefer a true live completion when two receipts share a timestamp.
                candidates.append((timestamp, 1 if is_live else 0, path.parent))
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            continue
    if candidates:
        return max(candidates, key=lambda item: (item[0], item[1]))[2]
    return bootstrap


def _universe_symbols(universe: Path) -> list[str]:
    payload = json.loads(universe.read_text())
    source_map = payload.get("source_map", {}) if isinstance(payload, dict) else {}
    if not isinstance(source_map, dict) or not source_map:
        raise RuntimeError(f"frozen universe has no source_map: {universe}")
    return sorted(str(symbol) for symbol in source_map)


def _read_15m_cache_window(symbol: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Read the canonical raw-first 15-minute cache over a tiny PIT window."""
    name = f"{symbol.lower().replace('/', '')}_15m.parquet"
    columns = ["open", "high", "low", "close", "volume", "exchange_observed"]
    frames: list[pd.DataFrame] = []
    for root in (
        ROOT / "data_perp/exchanges/krakenfutures/raw/ohlcv_15m",
        ROOT / "15m_ohlcv_perp",
    ):
        path = root / name
        if not path.exists():
            continue
        filters = [
            ("__index_level_0__", ">=", pd.Timestamp(index.min()).to_pydatetime()),
            ("__index_level_0__", "<=", pd.Timestamp(index.max()).to_pydatetime()),
        ]
        try:
            frame = pd.read_parquet(path, columns=columns, filters=filters)
        except Exception:
            try:
                frame = pd.read_parquet(path, columns=columns[:-1], filters=filters)
            except Exception:
                continue
        if not isinstance(frame.index, pd.DatetimeIndex):
            continue
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame.loc[~frame.index.isna()]
        if "exchange_observed" not in frame:
            frame["exchange_observed"] = pd.Series(
                pd.NA, index=frame.index, dtype="boolean",
            )
        else:
            frame["exchange_observed"] = frame["exchange_observed"].astype("boolean")
        frames.append(frame.reindex(index))
    if not frames:
        return pd.DataFrame(index=index, columns=columns)
    # The raw exchange cache wins at exact timestamps; the shared cache is a
    # causal fill source only.  This mirrors the frozen feature source.
    values = frames[0]
    for frame in frames[1:]:
        values = values.combine_first(frame)
    return values.reindex(index)


def _assess_15m_coverage(*, decision: pd.Timestamp, symbols: list[str]) -> pd.DataFrame:
    """Classify completed signal-hour readiness without touching future bars."""
    signal = decision - pd.Timedelta(hours=1)
    signal_index = pd.date_range(signal, decision - pd.Timedelta(minutes=15), freq="15min")
    decision_index = pd.DatetimeIndex([decision])
    # Reuse the exact decision-open source adapter so this receipt reports the
    # same executable-open condition as the candidate materialiser.
    from scripts.run_tp6_sl4_exact170_canonical_consensus import _read_downloaded_15m_decision_open

    records: list[dict[str, object]] = []
    for symbol in symbols:
        frame = _read_15m_cache_window(symbol, signal_index)
        finite = frame[["open", "high", "low", "close"]].notna().all(axis=1)
        flat_zero = (
            finite
            & frame["open"].eq(frame["high"])
            & frame["high"].eq(frame["low"])
            & frame["low"].eq(frame["close"])
            & pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0).le(0.0)
        )
        observed = frame["exchange_observed"].astype("boolean")
        # Unknown legacy rows retain the former conservative rule.  A true
        # observed bit makes a flat no-trade candle source-valid; false marks
        # a local fill explicitly.
        synthetic = flat_zero & ~observed.fillna(False)
        try:
            decision_open = _read_downloaded_15m_decision_open(
                symbol, decision_index,
            )
            has_decision_open = bool(pd.to_numeric(decision_open, errors="coerce").notna().iloc[0])
        except Exception:
            has_decision_open = False
        records.append({
            "symbol": symbol,
            "signal_ts": signal.isoformat(),
            "decision_ts": decision.isoformat(),
            "expected_15m_bars": int(len(signal_index)),
            "finite_15m_bars": int(finite.sum()),
            "exchange_observed_15m_bars": int((finite & observed.fillna(False)).sum()),
            "locally_filled_flat_15m_bars": int((finite & observed.eq(False).fillna(False)).sum()),
            "synthetic_flat_15m_bars": int(synthetic.sum()),
            "missing_15m_bar": bool((~finite).any()),
            "synthetic_flat_bar": bool(synthetic.any()),
            "missing_decision_open": not has_decision_open,
            "feature_source_ready": bool(finite.all() and not synthetic.any()),
        })
    return pd.DataFrame.from_records(records)


def _run_15m_refresh_partitions(
    *,
    start: pd.Timestamp,
    decision: pd.Timestamp,
    universe: Path,
    out_dir: Path,
    prefix: str,
    symbols: list[str] | None = None,
) -> list[int]:
    commands: list[tuple[int, Any, subprocess.Popen[str]]] = []
    for partition in range(16):
        command = [
            sys.executable, str(ROOT / "scripts/download_kraken_15m_hf.py"),
            "--target-free-manifest", str(universe),
            "--force-start", start.isoformat(),
            "--force-end", decision.isoformat(),
            "--hf-data-dir", "15m_ohlcv_perp",
            "--partition-count", "16", "--partition-id", str(partition),
            "--sleep-seconds", "0", "--rate-limit-ms", "1000",
        ]
        if symbols:
            for symbol in symbols:
                command.extend(["--symbol", symbol])
        log = out_dir / f"{prefix}_partition_{partition:02d}.log"
        handle = log.open("w", encoding="utf-8")
        commands.append((partition, handle, subprocess.Popen(
            command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True,
        )))
    failed: list[int] = []
    for partition, handle, process in commands:
        result = process.wait()
        handle.close()
        if result:
            failed.append(partition)
    return failed


def _refresh_15m(
    *,
    decision: pd.Timestamp,
    universe: Path,
    out_dir: Path,
    settled_retry_schedule_seconds: tuple[float, ...] = (30.0, 60.0, 120.0, 180.0),
) -> dict[str, object]:
    """Refresh and audit the exact completed signal hour.

    A refresh with process-level success but incomplete signal bars is
    deliberately labelled ``partial``.  It remains row-local fail-closed: the
    scorer may use ready symbols while incomplete symbols cannot be admitted.
    """
    out_dir.mkdir(parents=True, exist_ok=False)
    start = decision - pd.Timedelta(hours=2)
    symbols = _universe_symbols(universe)
    refresh_started = time.monotonic()
    failed_initial = _run_15m_refresh_partitions(
        start=start, decision=decision, universe=universe, out_dir=out_dir,
        prefix="initial",
    )
    before = _assess_15m_coverage(decision=decision, symbols=symbols)
    before.to_parquet(out_dir / "coverage_before_retry.parquet", index=False)
    print(json.dumps({
        "event": "15m_coverage_before_retry",
        "decision_ts": decision.isoformat(),
        "feature_source_ready": int(before["feature_source_ready"].sum()),
        "missing_15m_bar": int(before["missing_15m_bar"].sum()),
        "synthetic_flat_bar": int(before["synthetic_flat_bar"].sum()),
        "missing_decision_open": int(before["missing_decision_open"].sum()),
    }), flush=True)
    after = before
    retry_attempts: list[dict[str, object]] = []
    failed_retry: list[int] = []
    for target_elapsed in sorted({max(0.0, float(value)) for value in settled_retry_schedule_seconds}):
        retry_symbols = after.loc[
            ~after["feature_source_ready"], "symbol",
        ].astype(str).tolist()
        if not retry_symbols:
            break
        wait_seconds = max(0.0, target_elapsed - (time.monotonic() - refresh_started))
        if wait_seconds:
            time.sleep(wait_seconds)
        failed = _run_15m_refresh_partitions(
            start=start, decision=decision, universe=universe, out_dir=out_dir,
            prefix=f"settled_retry_{int(target_elapsed):03d}s",
            symbols=retry_symbols,
        )
        failed_retry.extend(failed)
        after = _assess_15m_coverage(decision=decision, symbols=symbols)
        artifact = f"coverage_after_retry_{int(target_elapsed):03d}s.parquet"
        after.to_parquet(out_dir / artifact, index=False)
        retry_attempts.append({
            "target_elapsed_seconds": target_elapsed,
            "actual_elapsed_seconds": round(time.monotonic() - refresh_started, 3),
            "symbols_retried": retry_symbols,
            "symbol_count": int(len(retry_symbols)),
            "failed_partitions": failed,
            "coverage_artifact": artifact,
            "feature_source_ready": int(after["feature_source_ready"].sum()),
        })
        print(json.dumps({
            "event": "15m_coverage_after_retry",
            "decision_ts": decision.isoformat(),
            "target_elapsed_seconds": target_elapsed,
            "symbols_retried": int(len(retry_symbols)),
            "feature_source_ready": int(after["feature_source_ready"].sum()),
            "missing_15m_bar": int(after["missing_15m_bar"].sum()),
            "synthetic_flat_bar": int(after["synthetic_flat_bar"].sum()),
        }), flush=True)
    after.to_parquet(out_dir / "coverage_after_retry.parquet", index=False)
    summary = {
        "symbols": int(len(symbols)),
        "feature_source_ready": int(after["feature_source_ready"].sum()),
        "missing_15m_bar": int(after["missing_15m_bar"].sum()),
        "synthetic_flat_bar": int(after["synthetic_flat_bar"].sum()),
        "missing_decision_open": int(after["missing_decision_open"].sum()),
    }
    failed = sorted(set(failed_initial + failed_retry))
    receipt: dict[str, object] = {
        "schema": "strict_r3_live_hourly_15m_refresh_v2",
        "decision_ts": decision.isoformat(), "start": start.isoformat(),
        "end": decision.isoformat(), "partitions": 16,
        "initial_failed_partitions": failed_initial,
        "settled_retry_schedule_seconds": list(settled_retry_schedule_seconds),
        "retry_attempts": retry_attempts,
        "retry_failed_partitions": sorted(set(failed_retry)),
        "failed_partitions": failed,
        "coverage_before_retry": {
            "feature_source_ready": int(before["feature_source_ready"].sum()),
            "missing_15m_bar": int(before["missing_15m_bar"].sum()),
            "synthetic_flat_bar": int(before["synthetic_flat_bar"].sum()),
            "missing_decision_open": int(before["missing_decision_open"].sum()),
        },
        "coverage_after_retry": summary,
        "coverage_artifacts": {
            "before_retry": "coverage_before_retry.parquet",
            "after_retry": "coverage_after_retry.parquet",
        },
        "source_contract": (
            "raw-first 15m cache; exchange_observed=true is accepted even for "
            "flat zero-volume candles; locally-filled or legacy-unknown flat "
            "candles remain unavailable; no post-decision bar is read"
        ),
        "status": "fail_closed" if failed else (
            "pass" if summary["feature_source_ready"] == len(symbols) else "partial"
        ),
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if failed:
        raise RuntimeError(f"15m source refresh failed in partitions {failed}")
    return receipt


def _refresh_official_hourly_analytics(
    *,
    decision: pd.Timestamp,
    universe: Path,
    log: Path,
) -> None:
    """Append official mark/OI/L2 analytics for the just-closed signal hour.

    The frozen target-free grid reads these fields from
    ``frozen_contract_backfill_hourly``.  Refreshing only 15-minute OHLCV
    leaves the grid with a missing signal-hour spread and incorrectly rejects
    otherwise valid candidates.  The existing refresh utility preserves prior
    values with ``combine_first`` and only fills the declared one-hour window.
    """
    start = decision - pd.Timedelta(hours=1)
    _run([
        sys.executable,
        str(ROOT / "scripts/backfill_kraken_frozen_contract_inputs.py"),
        "--symbols-json", str(universe),
        "--out-dir", "data_perp/exchanges/krakenfutures/frozen_contract_backfill_hourly",
        "--start", start.isoformat(),
        "--end", decision.isoformat(),
        "--workers", "16",
        "--include-orderbook-analytics",
    ], log=log)


def _refresh_oi_funding_sidecars(
    *,
    decision: pd.Timestamp,
    universe: Path,
    out_dir: Path,
) -> None:
    """Append causal OI/funding observations before live feature materialisation.

    ``post_liquidation_rebound_score`` depends on a funding-change primitive.
    The frozen-contract order-book refresh does not supply that primitive, so
    the live producer must refresh the established OI/funding sidecars as a
    separate source family.  The sidecar utility shifts observations by its
    declared one-hour availability rule; this producer never reads a value
    after ``decision``.  Partitions are independent and bounded so a slow
    product cannot serialize all 170 symbols.

    A product-level API failure is recorded in its partition receipt.  It is
    deliberately not imputed here: the canonical per-row feature gate later
    rejects that product if the frozen field remains unavailable.
    """
    out_dir.mkdir(parents=True, exist_ok=False)
    # Two completed observation hours are sufficient for the one-hour funding
    # change used by the canonical composite; request a small overlap so the
    # sidecar merge remains gap-filling and recovery after a transient outage
    # is deterministic.
    start = decision - pd.Timedelta(hours=3)
    commands: list[tuple[int, Path, Any, subprocess.Popen[str]]] = []
    for partition in range(16):
        log = out_dir / f"partition_{partition:02d}.log"
        handle = log.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(ROOT / "scripts/backfill_kraken_oi_funding_sidecars.py"),
            "--feature-dir", "data_perp/features",
            "--symbols-file", str(universe),
            "--perp-root", "data_perp/exchanges/krakenfutures",
            "--out-dir", str(out_dir / f"partition_{partition:02d}"),
            "--start-ts", start.isoformat(),
            "--end-ts", decision.isoformat(),
            "--workers", "1",
            "--partition-count", "16",
            "--partition-id", str(partition),
        ]
        commands.append((
            partition,
            log,
            handle,
            subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            ),
        ))
    failed: list[int] = []
    partition_audits: list[dict[str, Any]] = []
    for partition, _, handle, process in commands:
        returncode = process.wait()
        handle.close()
        manifest_path = out_dir / f"partition_{partition:02d}" / "backfill_manifest.json"
        payload: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                payload = json.loads(manifest_path.read_text())
            except (OSError, ValueError, json.JSONDecodeError):
                payload = {"manifest_parse_error": True}
        if returncode:
            failed.append(partition)
        partition_audits.append({
            "partition": partition,
            "returncode": int(returncode),
            "manifest_status": payload.get("status"),
            "result_counts": payload.get("result_counts"),
        })
    receipt = {
        "schema": "strict_r3_live_hourly_oi_funding_refresh_v1",
        "decision_ts": decision.isoformat(),
        "start": start.isoformat(),
        "end": decision.isoformat(),
        "partitions": 16,
        "failed_partitions": failed,
        "partition_audits": partition_audits,
        "source_contract": "Kraken observed OI/funding shifted +1h; no imputation",
        "status": "pass" if not failed else "fail_closed",
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    if failed:
        raise RuntimeError(f"OI/funding source refresh failed in partitions {failed}")


def run_once(args: argparse.Namespace, *, decision: pd.Timestamp) -> dict[str, Any]:
    decision = _utc(decision)
    tag = decision.strftime("%Y%m%dT%H%M%SZ")
    runtime_tag = _runtime_tag(args.inference_bundle)
    receipt_prefix = f"strict_r3_live_hourly_producer_{runtime_tag}_{tag}"
    receipt, attempt = _next_receipt(receipt_prefix)
    receipt.mkdir(parents=True)
    started = pd.Timestamp.now(tz="UTC")
    result: dict[str, Any] = {
        "schema": "strict_r3_live_hourly_entry_producer_v1",
        "decision_ts": decision.isoformat(), "started_at": started.isoformat(),
        "mode": "live", "status": "failed_closed", "exchange_order_submission": False,
    }
    try:
        bundle = ROOT / args.inference_bundle
        bundle_hash = _sha(bundle)
        bundle_payload = json.loads(bundle.read_text())
        predecessor = _successful_predecessor(
            decision=decision, bundle_hash=bundle_hash,
            bootstrap=ROOT / args.bootstrap_previous_run,
        )
        result["predecessor"] = str(predecessor.relative_to(ROOT))
        refresh_dir = ARTIFACTS / f"strict_r3_live_15m_refresh_{runtime_tag}_{tag}_v{attempt}"
        refresh_15m = _refresh_15m(
            decision=decision,
            universe=ROOT / str(bundle_payload["paths"]["frozen_universe_manifest"]),
            out_dir=refresh_dir,
            settled_retry_schedule_seconds=_retry_schedule(
                getattr(args, "settled_retry_schedule_seconds", None),
                first_retry_seconds=getattr(args, "settled_retry_seconds", None),
            ),
        )
        result["refresh_15m"] = refresh_15m
        oi_funding_refresh_dir = (
            ARTIFACTS / f"strict_r3_live_oi_funding_refresh_{runtime_tag}_{tag}_v{attempt}"
        )
        _refresh_oi_funding_sidecars(
            decision=decision,
            universe=ROOT / str(bundle_payload["paths"]["frozen_universe_manifest"]),
            out_dir=oi_funding_refresh_dir,
        )
        _refresh_official_hourly_analytics(
            decision=decision,
            universe=ROOT / str(bundle_payload["paths"]["frozen_universe_manifest"]),
            log=receipt / "official_hourly_analytics_refresh.log",
        )
        run_dir = ARTIFACTS / f"strict_r3_successor_{runtime_tag}_live_{tag}_v{attempt}"
        feature_contract = str(bundle_payload["runtime"]["feature_state"]["contract_sha256"])
        feature_tail = str(bundle_payload["runtime"]["feature_state"]["panel_tail_hours"])
        _run([
            sys.executable, str(ROOT / "scripts/run_strict_r3_hourly_shadow_resume_v15.py"),
            "--inference-bundle", str(bundle.relative_to(ROOT)),
            "--portfolio-state-json", str((ROOT / args.live_state).relative_to(ROOT)),
            "--decision-ts", decision.isoformat(), "--out-dir", str(run_dir.relative_to(ROOT)),
            "--previous-shadow-run", str(predecessor.relative_to(ROOT)),
            "--feature-state-bundle", str((predecessor / "feature_state/bundle").relative_to(ROOT)),
            "--feature-state-contract-hash", feature_contract,
            "--feature-state-tail-hours", feature_tail,
            "--portfolio-state-reconciliation", "--enforce-live-wall-clock",
        ], log=receipt / "hourly_shadow.log")
        audit_dir = ARTIFACTS / f"strict_r3_live_hour_audit_{runtime_tag}_{tag}_v{attempt}"
        _run([
            sys.executable, str(ROOT / "scripts/audit_strict_r3_schema_v6_live_hour.py"),
            "--run", str(run_dir.relative_to(ROOT)),
            "--previous-run", str(predecessor.relative_to(ROOT)),
            "--out", str(audit_dir.relative_to(ROOT)), "--enforce-live-wall-clock",
        ], log=receipt / "live_hour_audit.log")
        checkpoint_dir = ARTIFACTS / f"strict_r3_live_runtime_checkpoint_{runtime_tag}_{tag}_v{attempt}"
        _run([
            sys.executable, str(ROOT / "scripts/checkpoint_strict_r3_runtime.py"), "create",
            "--run-dir", str(run_dir.relative_to(ROOT)),
            "--inference-bundle", str(bundle.relative_to(ROOT)),
            "--feature-state-bundle", str((run_dir / "feature_state/bundle").relative_to(ROOT)),
            # The checkpoint binds inputs at the decision boundary.  The next
            # portfolio state is post-auction and therefore newer than that
            # boundary; using it both fails the checkpoint and would be the
            # wrong lineage for an entry decision.
            "--portfolio-state", str((run_dir / "portfolio_reconciliation_state.json").relative_to(ROOT)),
            "--out-dir", str(checkpoint_dir.relative_to(ROOT)),
        ], log=receipt / "runtime_checkpoint.log")
        execution_dir = ARTIFACTS / f"strict_r3_live_execution_{runtime_tag}_{tag}_v{attempt}"
        _run([
            sys.executable, str(ROOT / "scripts/execute_strict_r3_kraken_live.py"),
            "--execution-bundle", args.execution_bundle,
            "--hourly-run", str(run_dir.relative_to(ROOT)),
            "--state", args.live_state, "--out", str(execution_dir.relative_to(ROOT)),
            "--live-hour-audit", str(audit_dir.relative_to(ROOT)),
            "--runtime-checkpoint", str(checkpoint_dir.relative_to(ROOT)),
            "--submit-orders",
        ], log=receipt / "execution.log")
        result.update({
            "status": "pass", "exchange_order_submission": True,
            "hourly_run": str(run_dir.relative_to(ROOT)),
            "live_hour_audit": str(audit_dir.relative_to(ROOT)),
            "runtime_checkpoint": str(checkpoint_dir.relative_to(ROOT)),
            "execution_receipt": str(execution_dir.relative_to(ROOT)),
        })
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    result["completed_at"] = pd.Timestamp.now(tz="UTC").isoformat()
    result["decision_age_at_completion_seconds"] = float(
        (pd.Timestamp.now(tz="UTC") - decision).total_seconds()
    )
    (receipt / "run_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-bundle", required=True)
    parser.add_argument("--execution-bundle", required=True)
    parser.add_argument("--live-state", required=True)
    parser.add_argument("--bootstrap-previous-run", required=True)
    parser.add_argument("--decision-ts")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument(
        "--settled-retry-schedule-seconds", default="30,60,120,180",
        help=(
            "Comma-separated elapsed seconds after the initial source refresh "
            "at which only incomplete completed signal-hour symbols are retried."
        ),
    )
    parser.add_argument(
        "--settled-retry-seconds", type=float, default=None,
        help="Deprecated compatibility override for the first retry checkpoint.",
    )
    args = parser.parse_args()
    if args.loop and args.decision_ts:
        raise ValueError("--loop and --decision-ts are mutually exclusive")
    if args.decision_ts:
        print(json.dumps(run_once(args, decision=_utc(args.decision_ts)), sort_keys=True))
        return
    if not args.loop:
        raise ValueError("provide --decision-ts for one run or --loop for the service")
    last: pd.Timestamp | None = None
    while True:
        now = pd.Timestamp.now(tz="UTC")
        decision = now.floor("h")
        if now.minute == 0 and now.second < 15 and decision != last:
            print(json.dumps(run_once(args, decision=decision), sort_keys=True), flush=True)
            last = decision
        time.sleep(max(1.0, float(args.poll_seconds)))


if __name__ == "__main__":
    main()
