#!/usr/bin/env python3
"""Materialise an immutable exact-1m HPO dataset from score-only candidates.

This is an offline research producer.  Candidate routing is derived solely
from prequential score columns; complete future paths are joined afterwards.
Missing one-minute paths are recorded as invalid supervision and are never
converted into zero-return outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore  # noqa: E402
from extreme_price_movements.exact_1m_policy_contract import (  # noqa: E402
    Exact1mExecutionContract,
)


DEFAULT_LEDGER = ROOT / (
    "data_perp/artifacts/strict_r3_schema_v2_prequential_ledger_targetfree_long_"
    "2024_2026_raw15m_strictfull_20260812_v1/prequential_stack_ledger.parquet"
)
DEFAULT_MINUTE_ROOT = ROOT / "data_perp/exchanges/krakenfutures/execution_1m"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_exact_1m_policy_hpo_dataset_202402_20260817_v1"
DEFAULT_DOWNLOAD_REQUEST = ROOT / "data_perp/artifacts/strict_r3_exact_1m_policy_hpo_download_request_2024_20260817_v1"
# Wilder-14 uses 100 complete hourly true-range observations.  A delayed
# entry at e.g. ``xx:05`` makes a nominal 100-hour source window begin at
# ``xx:05`` too, leaving its first resampled hourly bin incomplete and only
# 99 complete bins before the entry.  Acquire/load one additional hour so the
# causal state always has 100 complete bins.  Keep ``WARMUP_HOURS`` as the
# ATR/statistical contract and use the explicit source lookback below only for
# causal input coverage.
WARMUP_HOURS = 100
ATR_SOURCE_LOOKBACK_HOURS = WARMUP_HOURS + 1
ATR_PERIODS = 14

# The explicit-input mode deliberately reads a narrow, score-only request.  It
# is not a convenience path for a labelled policy panel: policy outcomes must
# only enter below when complete one-minute paths are materialised.
EXPLICIT_REQUIRED_COLUMNS = {
    "candidate_id", "timestamp", "symbol", "side_name", "entry_ts",
}
EXPLICIT_SCORE_COLUMNS = ("priority_bps", "score")
FORBIDDEN_OUTCOME_COLUMN_TOKENS = (
    "path_valid", "outcome", "label", "policy_net", "policy_gross",
    "exit_", "realized", "future_", "mfe", "mae",
)
REQUIRED_TARGET_FREE_MANIFEST_KEYS = {
    "schema", "target_free", "selection_inputs", "forbidden_selection_inputs",
    "candidate_sha256", "contract_hash", "rows",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="coerce")


def _utc_timestamp(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _score_population(
    ledger: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    retained_fraction: float,
    cap_per_month: int,
    side: str = "long",
) -> pd.DataFrame:
    columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "base_contract_complete", "stack_is_prequential", "prequential_upstream",
    ]
    frame = pd.read_parquet(ledger, columns=columns).rename(columns={
        "__decision_ts__": "timestamp", "__symbol__": "symbol",
        "prequential_upstream": "score",
    })
    frame["timestamp"] = _utc(frame["timestamp"])
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    frame = frame.loc[
        frame["side_name"].astype(str).str.lower().eq(side)
        & frame["base_contract_complete"].fillna(False).astype(bool)
        & frame["stack_is_prequential"].fillna(False).astype(bool)
        & frame["timestamp"].ge(start)
        & frame["timestamp"].lt(end)
        & frame["score"].notna()
    ].copy()
    if frame.empty:
        raise RuntimeError(f"no strict-prequential {side} candidates in requested period")
    cutoff = float(frame["score"].quantile(1.0 - float(retained_fraction), interpolation="higher"))
    frame = frame.loc[frame["score"].ge(cutoff)].copy()
    frame["month"] = frame["timestamp"].dt.tz_localize(None).dt.to_period("M").astype(str)
    kept: list[pd.DataFrame] = []
    for _, group in frame.groupby("month", sort=True):
        if len(group) <= cap_per_month:
            kept.append(group)
            continue
        keys = pd.util.hash_pandas_object(group["candidate_id"].astype(str), index=False).to_numpy(np.uint64)
        kept.append(group.iloc[np.argsort(keys, kind="stable")[:cap_per_month]])
    output = pd.concat(kept, ignore_index=True)
    output["candidate_id"] = output["candidate_id"].astype(str)
    output["symbol"] = output["symbol"].astype(str)
    if output["candidate_id"].duplicated().any():
        raise AssertionError("score population has duplicate candidate IDs")
    return output.drop(columns="month").sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True)


def _explicit_candidate_population(
    candidate_input: Path,
    candidate_manifest: Path,
    contract: Exact1mExecutionContract,
    side: str = "long",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a predeclared score-only request without touching outcomes.

    The request must be generated before one-minute paths are queried.  This
    makes the BCF/current dual-admission replay usable by the exact-policy HPO
    while keeping path availability strictly a post-routing supervision fact.
    """
    candidate_input = Path(candidate_input).resolve()
    candidate_manifest = Path(candidate_manifest).resolve()
    if not candidate_input.is_file() or not candidate_manifest.is_file():
        raise FileNotFoundError(
            "explicit candidate mode requires both a candidate parquet and its target-free manifest"
        )
    manifest = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    missing_manifest = REQUIRED_TARGET_FREE_MANIFEST_KEYS.difference(manifest)
    if missing_manifest:
        raise AssertionError(
            f"explicit candidate manifest is incomplete: {sorted(missing_manifest)}"
        )
    if manifest.get("target_free") is not True:
        raise AssertionError("explicit candidate manifest must declare target_free=true")
    manifest_side = str(manifest.get("side") or side).strip().lower()
    if manifest_side != side:
        raise AssertionError("explicit candidate manifest side differs from requested side")
    if str(manifest.get("contract_hash")) != contract.hash:
        raise AssertionError("explicit candidate contract differs from the exact-1m materialiser contract")
    if str(manifest.get("candidate_sha256")) != _sha256(candidate_input):
        raise AssertionError("explicit candidate manifest hash does not bind its parquet")

    selection_inputs = {str(value) for value in manifest.get("selection_inputs", [])}
    forbidden_selection_inputs = {str(value) for value in manifest.get("forbidden_selection_inputs", [])}
    if not selection_inputs:
        raise AssertionError("explicit candidate manifest must declare score-only selection_inputs")
    forbidden_selected = selection_inputs.intersection(forbidden_selection_inputs)
    if forbidden_selected:
        raise AssertionError(
            f"explicit candidate selection used forbidden outcome inputs: {sorted(forbidden_selected)}"
        )

    schema_columns = set(pq.read_schema(candidate_input).names)
    missing_columns = EXPLICIT_REQUIRED_COLUMNS.difference(schema_columns)
    if missing_columns:
        raise AssertionError(f"explicit candidate input lacks columns: {sorted(missing_columns)}")
    score_column = next((name for name in EXPLICIT_SCORE_COLUMNS if name in schema_columns), None)
    if score_column is None:
        raise AssertionError(
            "explicit candidate input requires priority_bps or score for portfolio replay ordering"
        )
    forbidden_columns = sorted(
        column for column in schema_columns
        if any(token in column.lower() for token in FORBIDDEN_OUTCOME_COLUMN_TOKENS)
    )
    if forbidden_columns:
        raise AssertionError(
            f"explicit candidate input must not contain outcome-derived columns: {forbidden_columns}"
        )

    use_columns = sorted(EXPLICIT_REQUIRED_COLUMNS | {score_column})
    frame = pd.read_parquet(candidate_input, columns=use_columns).copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame["entry_ts"] = pd.to_datetime(frame["entry_ts"], utc=True, errors="coerce")
    frame["score"] = pd.to_numeric(frame[score_column], errors="coerce")
    if len(frame) != int(manifest["rows"]):
        raise AssertionError("explicit candidate row count differs from its manifest")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("explicit candidate input has duplicate candidate IDs")
    if frame.loc[:, ["candidate_id", "symbol", "timestamp", "entry_ts", "score"]].isna().any().any():
        raise AssertionError("explicit candidate input has missing identity, time, or priority values")
    if not frame["side_name"].eq(side).all():
        raise AssertionError("exact-1m parent-policy HPO input must be side-local")
    expected_entry = frame["timestamp"] + pd.Timedelta(minutes=contract.entry_delay_minutes)
    if not frame["entry_ts"].eq(expected_entry).all():
        raise AssertionError("explicit candidate entries must equal decision timestamp plus the declared delay")
    frame["priority_source_column"] = score_column
    audit = {
        "mode": "explicit_target_free_candidate_input",
        "candidate_input": str(candidate_input),
        "candidate_input_sha256": _sha256(candidate_input),
        "candidate_manifest": str(candidate_manifest),
        "candidate_manifest_sha256": _sha256(candidate_manifest),
        "candidate_manifest_schema": str(manifest["schema"]),
        "selection_inputs": sorted(selection_inputs),
        "forbidden_selection_inputs": sorted(forbidden_selection_inputs),
        "score_column": score_column,
        "side": side,
        "target_free": True,
    }
    return frame.sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True), audit


def _write_download_request(
    output: Path,
    population: pd.DataFrame,
    contract: Exact1mExecutionContract,
) -> None:
    """Persist the exact source request before any future-path join.

    The established Kraken downloader intentionally accepts signal timestamps,
    so its ``--warmup-minutes`` and ``--horizon-minutes`` arguments can cover
    both the uniform delayed entry and the post-entry H12 path without leaking
    an outcome-derived eligibility decision into the request.
    """
    columns = ["candidate_id", "timestamp", "symbol", "score", "entry_ts"]
    population.loc[:, columns].to_parquet(
        output / "candidate_download_request.parquet", index=False
    )
    request = {
        "schema": "strict_r3_exact_1m_policy_download_request_v1",
        "contract_hash": contract.hash,
        "rows": int(len(population)),
        "entry_delay_minutes": int(contract.entry_delay_minutes),
        "required_downloader_warmup_minutes": int(ATR_SOURCE_LOOKBACK_HOURS * 60),
        "required_downloader_horizon_minutes": int(
            contract.entry_delay_minutes + contract.horizon_minutes
        ),
        "timestamp_semantics": "decision timestamp; downloader window is [timestamp-warmup, timestamp+horizon)",
        "candidate_source": "strict-prequential score-only routing; no future-path qualification",
    }
    (output / "candidate_download_request.json").write_text(
        json.dumps(request, indent=2) + "\n", encoding="utf-8"
    )


def _verify_download_receipts(
    request_dir: Path,
    population: pd.DataFrame,
    contract: Exact1mExecutionContract,
    *,
    allow_incomplete_symbols: bool = False,
) -> dict[str, Any]:
    """Bind materialisation to the complete, exact candidate download request."""
    request_path = request_dir / "candidate_download_request.parquet"
    request_manifest_path = request_dir / "candidate_download_request.json"
    if not request_path.is_file() or not request_manifest_path.is_file():
        raise FileNotFoundError("exact-1m materialisation requires the bound request parquet and manifest")
    request_manifest = json.loads(request_manifest_path.read_text())
    if str(request_manifest.get("contract_hash")) != contract.hash:
        raise AssertionError("download request contract differs from materialiser contract")
    request = pd.read_parquet(request_path, columns=["candidate_id", "timestamp", "symbol", "entry_ts"])
    actual = population.loc[:, ["candidate_id", "timestamp", "symbol", "entry_ts"]].copy()
    for frame in (request, actual):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["symbol"] = frame["symbol"].astype(str)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
        frame["entry_ts"] = pd.to_datetime(frame["entry_ts"], utc=True, errors="raise")
    request = request.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    actual = actual.sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if not request.equals(actual):
        raise AssertionError("materialiser candidate population does not exactly match bound download request")
    receipt_paths = sorted(request_dir.glob("download_partition_*.json"))
    if not receipt_paths:
        raise FileNotFoundError("exact-1m materialisation requires completed download partition receipts")
    request_sha = _sha256(request_path)
    request_manifest_sha = _sha256(request_manifest_path)
    seen: set[int] = set()
    source_contracts: set[str] = set()
    total = {
        "required_minutes": 0,
        "covered_minutes": 0,
        "fetched_rows": 0,
        "incomplete_symbols": 0,
    }
    incomplete_audit: list[dict[str, Any]] = []
    for path in receipt_paths:
        receipt = json.loads(path.read_text())
        partition_count = int(receipt.get("partition_count", 0))
        partition_id = int(receipt.get("partition_id", -1))
        if partition_count != 16 or not 0 <= partition_id < partition_count:
            raise AssertionError(f"invalid download partition receipt: {path.name}")
        if partition_id in seen:
            raise AssertionError(f"duplicate download partition receipt {partition_id}")
        seen.add(partition_id)
        if str(receipt.get("candidate_sha256")) != request_sha:
            raise AssertionError(f"download receipt candidate hash mismatch: {path.name}")
        stage = receipt.get("stage_manifest") or {}
        if str(stage.get("sha256")) != request_manifest_sha:
            raise AssertionError(f"download receipt stage-manifest mismatch: {path.name}")
        summary = dict(receipt.get("summary") or {})
        if int(summary.get("failed_symbols", -1)) != 0:
            raise AssertionError(f"download receipt has failed symbols: {path.name}")
        receipt_incomplete = int(summary.get("incomplete_symbols", -1))
        if receipt_incomplete < 0:
            raise AssertionError(f"download receipt lacks incomplete-symbol count: {path.name}")
        if receipt_incomplete and not allow_incomplete_symbols:
            raise AssertionError(f"download receipt is not complete: {path.name}")
        for row in receipt.get("results") or []:
            status = str(row.get("status"))
            coverage = float(row.get("coverage_after", 0.0))
            if status == "ok" and coverage == 1.0:
                continue
            if allow_incomplete_symbols and status == "incomplete" and 0.0 <= coverage < 1.0:
                incomplete_audit.append({
                    "receipt": path.name,
                    "symbol": str(row.get("symbol")),
                    "coverage_after": coverage,
                    "required_minutes": int(row.get("required_minutes", 0)),
                    "covered_after": int(row.get("covered_after", 0)),
                })
                continue
            raise AssertionError(f"invalid exact-1m symbol receipt: {path.name}:{row.get('symbol')}")
        source_contracts.add(str(receipt.get("product_mapping_contract") or ""))
        for key in ("required_minutes", "covered_minutes", "fetched_rows", "incomplete_symbols"):
            total[key] += int(summary.get(key, 0))
    if seen != set(range(16)):
        raise AssertionError(f"missing exact-1m download partitions: {sorted(set(range(16)).difference(seen))}")
    return {
        "request_dir": str(request_dir),
        "request_sha256": request_sha,
        "request_manifest_sha256": request_manifest_sha,
        "receipts": {path.name: _sha256(path) for path in receipt_paths},
        "summary": total,
        "incomplete_symbol_paths": incomplete_audit,
        "allows_incomplete_symbols": bool(allow_incomplete_symbols),
        "product_mapping_contracts": sorted(source_contracts),
    }


def _causal_atr(frame: pd.DataFrame) -> pd.Series:
    """Wilder-14 ATR from 100 complete preceding hourly windows."""
    hourly = frame.resample("1h", label="left", closed="left").agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), count=("close", "size"),
    )
    finite = np.isfinite(hourly.loc[:, ["open", "high", "low", "close"]].to_numpy(float)).all(axis=1)
    complete = hourly["count"].eq(60) & finite
    prior_close = hourly["close"].shift(1)
    tr = pd.concat([
        hourly["high"] - hourly["low"],
        (hourly["high"] - prior_close).abs(),
        (hourly["low"] - prior_close).abs(),
    ], axis=1).max(axis=1).where(complete)
    atr = tr.ewm(alpha=1.0 / ATR_PERIODS, adjust=False, min_periods=WARMUP_HOURS).mean()
    stable = complete.rolling(WARMUP_HOURS, min_periods=WARMUP_HOURS).sum().eq(WARMUP_HOURS)
    atr = atr.where(stable)
    atr.index = atr.index + pd.Timedelta(hours=1)
    return atr


def _clean_minute(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close"], index=pd.DatetimeIndex([], tz="UTC"))
    out = frame.loc[:, ["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    return out.loc[out.index.notna() & ~out.index.duplicated(keep="last")].sort_index()


def materialize(args: argparse.Namespace) -> Path:
    output = Path(args.out_dir).resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite immutable dataset: {output}")
    output.mkdir(parents=True, exist_ok=True)
    side = str(getattr(args, "side", "long")).strip().lower()
    if side not in {"long", "short"}:
        raise ValueError("exact-1m policy materialisation requires side=long or short")
    contract = Exact1mExecutionContract(entry_delay_minutes=int(args.entry_delay_minutes))
    contract.validate()
    start = _utc_timestamp(args.start)
    end = _utc_timestamp(args.end)
    candidate_input = getattr(args, "candidate_input", None)
    candidate_source: dict[str, Any]
    if candidate_input:
        if args.request_only:
            raise ValueError(
                "--request-only is only for ledger mode; explicit candidate input is already the sealed request"
            )
        candidate_input_path = Path(candidate_input).resolve()
        candidate_manifest_arg = getattr(args, "candidate_manifest", None)
        candidate_manifest_path = (
            Path(candidate_manifest_arg).resolve()
            if candidate_manifest_arg
            else candidate_input_path.with_suffix(".json")
        )
        population, candidate_source = _explicit_candidate_population(
            candidate_input_path, candidate_manifest_path, contract, side,
        )
        # An explicit panel owns its population.  Silently filtering it again
        # would alter a predeclared target-free route.
        if population["timestamp"].lt(start).any() or population["timestamp"].ge(end).any():
            raise AssertionError("explicit candidate input falls outside the declared materialisation period")
        download_request_dir = (
            Path(args.download_request_dir).resolve()
            if args.download_request_dir is not None
            else candidate_input_path.parent
        )
        routing = "explicit target-free score/admission request; future path validity joined after routing"
    else:
        ledger = Path(args.ledger).resolve()
        population = _score_population(
            ledger, start=start, end=end, retained_fraction=float(args.retained_fraction),
            cap_per_month=int(args.cap_per_month), side=side,
        )
        population["entry_ts"] = population["timestamp"] + pd.Timedelta(minutes=contract.entry_delay_minutes)
        _write_download_request(output, population, contract)
        if args.request_only:
            return output
        candidate_source = {
            "mode": "ledger_score_only_request",
            "ledger": str(ledger),
            "ledger_sha256": _sha256(ledger),
            "side": side,
        }
        download_request_dir = (
            Path(args.download_request_dir).resolve()
            if args.download_request_dir is not None
            else DEFAULT_DOWNLOAD_REQUEST.resolve()
        )
        routing = "strict-prequential score-only top-fraction; future path validity joined after routing"
    source_receipts = _verify_download_receipts(
        download_request_dir,
        population,
        contract,
        allow_incomplete_symbols=bool(getattr(args, "allow_incomplete_symbols", False)),
    )
    population["path_valid"] = False
    population["path_invalid_reason"] = "unmaterialized"
    population["signal_atr"] = np.nan
    population["entry_price"] = np.nan
    store = PartitionedOHLCVStore(str(Path(args.minute_root).resolve()), timeframe="1m")
    arrays: dict[str, list[np.ndarray]] = {key: [] for key in ("entry", "atr", "high", "low", "close")}
    valid_rows: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for symbol, group in population.groupby("symbol", sort=True):
        group = group.copy()
        earliest = group["entry_ts"].min() - pd.Timedelta(
            hours=ATR_SOURCE_LOOKBACK_HOURS
        )
        latest = group["entry_ts"].max() + pd.Timedelta(minutes=contract.horizon_minutes - 1)
        minute = _clean_minute(store.load(
            symbol, columns=["ts", "open", "high", "low", "close"], start_ts=earliest, end_ts=latest,
        ))
        if minute.empty:
            population.loc[group.index, "path_invalid_reason"] = "missing_minute_source"
            audit.append({"symbol": symbol, "rows": len(group), "valid_rows": 0, "reason": "missing_minute_source"})
            continue
        atr = _causal_atr(minute)
        entries = pd.DatetimeIndex(group["entry_ts"])
        locations = minute.index.get_indexer(entries)
        offsets = np.arange(contract.horizon_minutes, dtype=np.int64)
        locations_2d = locations[:, None] + offsets[None, :]
        in_range = (locations >= 0) & (locations_2d[:, -1] < len(minute))
        values = {key: minute[key].to_numpy(float) for key in ("open", "high", "low", "close")}
        # ``_causal_atr`` is indexed by the end of each completed hourly bar.
        # Entries intentionally occur five minutes after the decision.  Use
        # only the latest *already completed* ATR, bounded to one hour so a
        # data outage cannot carry a stale volatility estimate into a later
        # candidate.
        atr_values = atr.reindex(
            entries,
            method="ffill",
            tolerance=pd.Timedelta(hours=1),
        ).to_numpy(float)
        complete = np.zeros(len(group), dtype=bool)
        if in_range.any():
            idx = np.flatnonzero(in_range)
            selected = locations_2d[idx]
            finite_paths = np.ones(len(idx), dtype=bool)
            for key in values:
                finite_paths &= np.isfinite(values[key][selected]).all(axis=1)
            complete[idx] = finite_paths & np.isfinite(atr_values[idx]) & (atr_values[idx] > 0.0)
        if complete.any():
            idx = np.flatnonzero(complete)
            selected = locations_2d[idx]
            selected_rows = group.iloc[idx].copy()
            selected_rows["path_valid"] = True
            selected_rows["path_invalid_reason"] = ""
            selected_rows["signal_atr"] = atr_values[idx]
            selected_rows["entry_price"] = values["open"][selected[:, 0]]
            population.loc[selected_rows.index, ["path_valid", "path_invalid_reason", "signal_atr", "entry_price"]] = selected_rows[
                ["path_valid", "path_invalid_reason", "signal_atr", "entry_price"]
            ]
            valid_rows.append(selected_rows)
            arrays["entry"].append(selected_rows["entry_price"].to_numpy(float))
            arrays["atr"].append(selected_rows["signal_atr"].to_numpy(float))
            for key in ("high", "low", "close"):
                arrays[key].append(values[key][selected].astype(np.float32, copy=False))
        invalid = ~complete
        if invalid.any():
            missing_entry = locations < 0
            missing_horizon = ~in_range
            no_atr = ~np.isfinite(atr_values) | (atr_values <= 0.0)
            reason = np.where(missing_entry, "missing_entry_minute", np.where(missing_horizon, "incomplete_h12_minute_path", np.where(no_atr, "missing_causal_atr", "nonfinite_minute_path")))
            population.loc[group.index[invalid], "path_invalid_reason"] = reason[invalid]
        audit.append({"symbol": symbol, "rows": len(group), "valid_rows": int(complete.sum()), "reason": "ok"})
    if not valid_rows:
        population.to_parquet(output / "candidate_path_audit.parquet", index=False)
        pd.DataFrame(audit).to_parquet(output / "symbol_coverage.parquet", index=False)
        raise RuntimeError(
            "no complete exact-1m HPO paths; acquire the requested historical Kraken minute source before retrying"
        )
    # Group iteration is symbol-sorted.  Preserve that exact order because the
    # NPZ path arrays are appended in the same loop; candidate identity, rather
    # than a later sort, is the binding between rows and execution paths.
    array_rows = pd.concat(valid_rows, ignore_index=True)
    identity = array_rows["candidate_id"].astype(str).to_numpy()
    if len(identity) != len(np.concatenate(arrays["entry"])):
        raise AssertionError("path arrays do not match selected candidate identities")
    training = array_rows.reset_index(drop=True)
    np.savez_compressed(
        output / "exact_paths.npz",
        entry=np.concatenate(arrays["entry"]).astype(np.float64),
        atr=np.concatenate(arrays["atr"]).astype(np.float64),
        high=np.concatenate(arrays["high"]).astype(np.float32),
        low=np.concatenate(arrays["low"]).astype(np.float32),
        close=np.concatenate(arrays["close"]).astype(np.float32),
        candidate_id=identity.astype("U"),
    )
    population.to_parquet(output / "candidate_path_audit.parquet", index=False)
    training.to_parquet(output / "training_rows.parquet", index=False)
    pd.DataFrame(audit).to_parquet(output / "symbol_coverage.parquet", index=False)
    manifest = {
        "schema": "strict_r3_exact_1m_policy_hpo_dataset_v1",
        "side": side,
        "research_only": True,
        "contract": contract.to_dict(),
        "contract_hash": contract.hash,
        "candidate_source": candidate_source,
        "minute_root": str(Path(args.minute_root).resolve()),
        "source_receipts": source_receipts,
        "period": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "routing": routing,
        "candidate_rows": int(len(population)), "valid_training_rows": int(len(training)),
        "invalid_rows": int((~population["path_valid"].fillna(False)).sum()),
        "atr": (
            "Wilder14 from 100 complete prior minute-aggregated hourly bars; "
            "source/load lookback is 101 hours so a delayed xx:05 entry still has "
            "100 complete prior hourly bins"
        ),
        "path": "720 complete post-entry Kraken one-minute bars; no interpolation or synthetic flat bars",
    }
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--side", choices=["long", "short"], default="long")
    parser.add_argument(
        "--candidate-input", type=Path, default=None,
        help=(
            "Predeclared target-free candidate request/panel. Requires identity, "
            "side_name, exact entry_ts, and priority_bps or score; bypasses the ledger route."
        ),
    )
    parser.add_argument(
        "--candidate-manifest", type=Path, default=None,
        help="Target-free provenance manifest for --candidate-input (defaults to its sibling JSON).",
    )
    parser.add_argument("--minute-root", type=Path, default=DEFAULT_MINUTE_ROOT)
    parser.add_argument(
        "--download-request-dir", type=Path, default=None,
        help=(
            "Immutable candidate request and all 16 completed downloader receipts. "
            "Defaults to --candidate-input's parent in explicit mode, otherwise the legacy request."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    # The strict-prequential ledger first has compatible long candidates in
    # February 2024.  Do not quietly fill January with a weaker score family.
    parser.add_argument("--start", default="2024-02-01T00:00:00Z")
    parser.add_argument("--end", default="2025-01-01T00:00:00Z")
    parser.add_argument("--entry-delay-minutes", type=int, default=5)
    parser.add_argument("--retained-fraction", type=float, default=0.05)
    parser.add_argument("--cap-per-month", type=int, default=3500)
    parser.add_argument(
        "--request-only",
        action="store_true",
        help="Write only the target-free minute-data request; do not join paths.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--allow-incomplete-symbols",
        action="store_true",
        help=(
            "Accept terminal downloader receipts with explicit per-symbol source gaps; "
            "affected paths remain path_invalid and are excluded after target-free routing."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(materialize(parse_args()))
