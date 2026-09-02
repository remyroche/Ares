#!/usr/bin/env python3
"""Materialise exact-one-minute rich-policy outcomes for sealed P8U scores.

Candidates are loaded and audited before any one-minute path is opened.  A
missing path is an explicit invalid outcome, not a selection input and not a
zero-return pseudo-trade.  This is an offline replay producer: it has no
inference, exchange-account, portfolio-state, or order-submission code.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.data_store import PartitionedOHLCVStore
from scripts.materialize_strict_r3_exact_1m_policy_hpo_dataset import (
    ATR_SOURCE_LOOKBACK_HOURS,
    _causal_atr,
    _clean_minute,
)
from scripts.run_strict_r3_exact_1m_rich_entry_delay_ladder import (
    HORIZON_MINUTES,
    _complete_mask,
    _replay_batch,
)
from scripts.run_strict_r3_exact_1m_rich_matched_attribution import (
    _assert_v2_matches_live_oracle_sample,
    _load_frozen_policy,
)

DEFAULT_CANDIDATES = ROOT / (
    "data_perp/artifacts/strict_r3_p8u_exact1m_targetfree_dual50_aug01_27_"
    "20260829_v3"
)
DEFAULT_DUAL = ROOT / (
    "data_perp/artifacts/strict_r3_p8u_f72_underf120_dual_mc1_nov25_aug27_"
    "20260828_v1/dual_predictions.parquet"
)
DEFAULT_POLICY = ROOT / (
    "data_perp/artifacts/strict_r3_rich_policy_smooth_protection_long_"
    "20260817_v1/frozen_policy.json"
)
DEFAULT_MINUTE_ROOT = ROOT / "data_perp/exchanges/krakenfutures/execution_1m"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(values: Any) -> pd.Series | pd.Timestamp:
    if isinstance(values, pd.Series):
        return pd.to_datetime(values, utc=True, errors="raise")
    value = pd.Timestamp(values)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _write_once(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def _load_candidates(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root.resolve()
    path = root / "candidates.parquet"
    manifest_path = root / "candidate_manifest.json"
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError("sealed P8U target-free candidate request is incomplete")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target_free") is not True:
        raise AssertionError("exact-policy materialiser requires target-free candidates")
    if str(manifest.get("candidate_sha256")) != _sha256(path):
        raise AssertionError("candidate parquet differs from its immutable manifest")
    columns = [
        "candidate_id", "timestamp", "symbol", "side_name", "entry_ts", "priority_bps",
        "product_id",
    ]
    frame = pd.read_parquet(path, columns=columns).copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["symbol"] = frame["symbol"].astype(str)
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame["timestamp"] = _utc(frame["timestamp"])
    frame["entry_ts"] = _utc(frame["entry_ts"])
    frame["priority_bps"] = pd.to_numeric(frame["priority_bps"], errors="coerce")
    # This request is bound to a frozen source identity. A missing identity is
    # an explicit invalid label; never let another local symbol directory make
    # it look like a valid exact one-minute source.
    frame["product_id"] = frame["product_id"].astype("string")
    if frame["candidate_id"].duplicated().any() or not frame["side_name"].eq("long").all():
        raise AssertionError("candidate identities are invalid or not long-only")
    delay = int(manifest["entry_delay_minutes"])
    if not frame["entry_ts"].eq(frame["timestamp"] + pd.Timedelta(minutes=delay)).all():
        raise AssertionError("candidate entry timestamps do not bind to the declared delay")
    if not np.isfinite(frame["priority_bps"].to_numpy(float)).all():
        raise AssertionError("candidate priorities must be finite")
    return frame.sort_values(["timestamp", "candidate_id"], kind="stable").reset_index(drop=True), manifest


def _parent_policy(dual_path: Path, candidates: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_id", "__decision_ts__", "__symbol__", "policy_path_valid",
        "policy_gross_bps", "policy_net_bps", "policy_entry_price", "policy_exit_price",
        "policy_exit_reason", "policy_cost_bps",
    ]
    source = pd.read_parquet(dual_path.resolve(), columns=columns).copy()
    source["candidate_id"] = source["candidate_id"].astype(str)
    source["__decision_ts__"] = _utc(source["__decision_ts__"])
    source = source.loc[source["candidate_id"].isin(set(candidates["candidate_id"]))].copy()
    merged = candidates.loc[:, ["candidate_id", "timestamp", "symbol"]].merge(
        source, on="candidate_id", how="left", validate="one_to_one"
    )
    if len(merged) != len(candidates) or merged["__decision_ts__"].isna().any():
        raise AssertionError("dual-policy source misses target-free P8U candidate identities")
    if not _utc(merged["timestamp"]).equals(_utc(merged["__decision_ts__"])):
        raise AssertionError("parent policy timestamp differs from target-free candidate identity")
    if not merged["symbol"].eq(merged["__symbol__"].astype(str)).all():
        raise AssertionError("parent policy symbol differs from target-free candidate identity")
    return merged.drop(columns=["timestamp", "symbol", "__decision_ts__", "__symbol__"])


def run(args: argparse.Namespace) -> Path:
    output = Path(args.out_dir).resolve()
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")
    candidates, candidate_manifest = _load_candidates(Path(args.candidate_dir))
    requested_candidate_rows = int(len(candidates))
    resolved_before = None
    if args.label_available_before is not None:
        resolved_before = _utc(args.label_available_before)
        # The final observed path minute is decision + 12h04m and the label
        # becomes usable at decision + 12h05m.  Filter before any minute path
        # source is opened: a local cache may contain later data, but it may
        # never make a not-yet-resolved candidate eligible for calibration.
        candidates = candidates.loc[
            (candidates["timestamp"] + pd.Timedelta(hours=12, minutes=5)).lt(resolved_before)
        ].copy()
        if candidates.empty:
            raise RuntimeError("no exact rich-policy labels are resolved before the declared cutoff")
    delay = int(candidate_manifest["entry_delay_minutes"])
    params, median_atr_fraction, policy_audit = _load_frozen_policy(Path(args.frozen_policy))
    store = PartitionedOHLCVStore(str(Path(args.minute_root).resolve()), timeframe="1m")

    # Loading an exact 12-hour path also needs the preceding ATR warm-up.
    # The immutable one-minute cache deliberately consists of append-only
    # micro-parts, so this can involve many local fragments per symbol.  Load
    # independent symbols concurrently, then consume the completed frames in
    # the original deterministic symbol order below.  This is a read-only
    # transport optimisation: no source row, candidate ordering, policy
    # calculation, or oracle audit depends on completion order.
    grouped: list[tuple[str, pd.DataFrame, list[Any]]] = []
    for symbol, raw_group in candidates.groupby("symbol", sort=True):
        group = raw_group.reset_index(drop=True)
        product_ids = group["product_id"].dropna().unique().tolist()
        if len(product_ids) > 1:
            raise AssertionError(f"candidate source identity is not stable for {symbol}")
        grouped.append((str(symbol), group, product_ids))

    def _load_symbol_minute(symbol: str, group: pd.DataFrame) -> tuple[str, pd.DataFrame]:
        earliest = group["timestamp"].min() - pd.Timedelta(hours=ATR_SOURCE_LOOKBACK_HOURS, minutes=1)
        latest = group["entry_ts"].max() + pd.Timedelta(minutes=HORIZON_MINUTES - 1)
        return symbol, _clean_minute(store.load(
            symbol, columns=["ts", "open", "high", "low", "close"],
            start_ts=earliest, end_ts=latest,
        ))

    source_load_started = time.monotonic()
    minute_by_symbol: dict[str, pd.DataFrame] = {}
    source_groups = [(symbol, group) for symbol, group, product_ids in grouped if product_ids]
    if int(args.symbol_load_workers) == 1:
        for symbol, group in source_groups:
            loaded_symbol, minute = _load_symbol_minute(symbol, group)
            minute_by_symbol[loaded_symbol] = minute
    else:
        with ThreadPoolExecutor(max_workers=int(args.symbol_load_workers), thread_name_prefix="p8u-exact1m") as pool:
            futures = {
                pool.submit(_load_symbol_minute, symbol, group): symbol
                for symbol, group in source_groups
            }
            for future in as_completed(futures):
                loaded_symbol, minute = future.result()
                minute_by_symbol[loaded_symbol] = minute
    source_load_seconds = time.monotonic() - source_load_started

    valid_parts: list[pd.DataFrame] = []
    invalid_parts: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    oracle_audits: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    pending_rows = 0

    def flush() -> None:
        nonlocal pending_rows
        if not pending_rows:
            return
        rows = pd.concat([part["rows"] for part in pending], ignore_index=True)
        entry = np.concatenate([part["entry"] for part in pending]).astype(float, copy=False)
        atr = np.concatenate([part["atr"] for part in pending]).astype(float, copy=False)
        highs = np.concatenate([part["high"] for part in pending]).astype(np.float32, copy=False)
        lows = np.concatenate([part["low"] for part in pending]).astype(np.float32, copy=False)
        closes = np.concatenate([part["close"] for part in pending]).astype(np.float32, copy=False)
        # This offline label materialisation is intentionally held to a
        # stronger standard than an acceleration sentinel: every valid path,
        # across every batch, is compared with the scalar live-policy oracle.
        oracle_audits.append(_assert_v2_matches_live_oracle_sample(
            entries=entry, atr=atr, highs=highs, lows=lows, closes=closes,
            entry_timestamps=rows["entry_timestamp"].reset_index(drop=True),
            candidate_ids=rows["candidate_id"].reset_index(drop=True),
            params=params, median_atr_fraction=median_atr_fraction,
            entry_delay_minutes=delay, sample_size=args.oracle_sample_size,
            verify_live_state_machine=True,
        ))
        replay = _replay_batch(
            delay=delay, parts=pending, params=params,
            median_atr_fraction=median_atr_fraction,
        )
        # Under-F120's retained label contract requires the *complete* H12
        # path to have reached the frozen trailing-activation level.  The
        # policy exit may occur earlier, so exit reason is not a valid proxy
        # for this path condition.  Persist the observed full-path MFE/ATR
        # alongside the exact policy outcome.  These are labels only and are
        # never candidate, feature, or score-time inputs.
        path_mfe_atr = np.maximum(highs - entry[:, None], 0.0).max(axis=1) / np.maximum(atr, 1e-12)
        replay["path_mfe_atr_h12"] = np.asarray(path_mfe_atr, dtype=np.float32)
        replay["path_reached_trailing_activation_0p5atr"] = (
            np.asarray(path_mfe_atr, dtype=float) >= 0.5
        )
        valid_parts.append(replay)
        pending.clear()
        pending_rows = 0

    for symbol, group, product_ids in grouped:
        if not product_ids:
            invalid = group.copy()
            invalid["outcome_invalid_reason"] = "missing_frozen_product_id"
            invalid_parts.append(invalid)
            coverage.append({
                "symbol": str(symbol), "candidate_rows": len(group), "valid_rows": 0,
                "reason": "missing_frozen_product_id",
            })
            continue
        minute = minute_by_symbol[str(symbol)]
        if minute.empty:
            invalid = group.copy()
            invalid["outcome_invalid_reason"] = "missing_minute_source"
            invalid_parts.append(invalid)
            coverage.append({"symbol": str(symbol), "candidate_rows": len(group), "valid_rows": 0, "reason": "missing_minute_source"})
            continue
        atr = _causal_atr(minute)
        valid, locations, atr_values, reasons = _complete_mask(minute, atr, pd.DatetimeIndex(group["entry_ts"]))
        invalid = group.loc[~valid].copy()
        invalid["outcome_invalid_reason"] = reasons[~valid]
        if not invalid.empty:
            invalid_parts.append(invalid)
        invalid_reason_counts = (
            pd.Series(reasons[~valid], dtype="string")
            .value_counts(dropna=False)
            .sort_index()
            .to_dict()
        )
        coverage.append({
            "symbol": str(symbol), "candidate_rows": len(group), "valid_rows": int(valid.sum()),
            "invalid_rows": int((~valid).sum()),
            "reason": "ok" if bool(valid.all()) else "partial_outcome_coverage",
            "outcome_invalid_reason_counts": json.dumps(invalid_reason_counts, sort_keys=True),
        })
        if not valid.any():
            continue
        offsets = np.arange(HORIZON_MINUTES, dtype=np.int64)
        selected = locations[valid, None] + offsets[None, :]
        source = {field: minute[field].to_numpy(float) for field in ("open", "high", "low", "close")}
        rows = group.loc[valid, ["candidate_id", "timestamp", "entry_ts"]].copy().rename(
            columns={"timestamp": "decision_timestamp", "entry_ts": "entry_timestamp"}
        )
        pending.append({
            "rows": rows,
            "entry": source["open"][selected[:, 0]],
            "atr": atr_values[valid],
            "high": source["high"][selected].astype(np.float32, copy=False),
            "low": source["low"][selected].astype(np.float32, copy=False),
            "close": source["close"][selected].astype(np.float32, copy=False),
        })
        pending_rows += int(valid.sum())
        if pending_rows >= int(args.batch_rows):
            flush()
    flush()

    exact = pd.concat(valid_parts, ignore_index=True) if valid_parts else pd.DataFrame()
    invalid = pd.concat(invalid_parts, ignore_index=True) if invalid_parts else pd.DataFrame()
    if exact.empty:
        raise RuntimeError("no complete exact one-minute P8U paths")
    oracle_rows = sum(int(a["oracle_equivalence_rows"]) for a in oracle_audits)
    live_state_rows = sum(int(a.get("live_state_machine_equivalence_rows", 0)) for a in oracle_audits)
    if oracle_rows <= 0 or live_state_rows != oracle_rows:
        raise AssertionError("scalar/live-state oracle audit did not cover the same nonzero path population")
    exhaustive_oracle = args.oracle_sample_size is None
    if exhaustive_oracle and oracle_rows != len(exact):
        raise AssertionError("exhaustive scalar-oracle audit did not cover every valid exact one-minute path")
    oracle_audit = {
        "engine": "vectorized_v2_default_extensions",
        "oracle_equivalence_scope": "all_valid_rows",
        "oracle_equivalence_rows": int(oracle_rows),
        "oracle_equivalence_batches": int(len(oracle_audits)),
        "oracle_equivalence_scope": "all_valid_rows" if exhaustive_oracle else "deterministic_candidate_id_sample_per_batch",
        "oracle_sample_size_per_batch": None if exhaustive_oracle else int(args.oracle_sample_size),
        "oracle_equivalence": (
            "every valid path exactly equals scalar-v1 for validity/reason/minute/timestamp and 1e-12 numeric outcomes"
            if exhaustive_oracle else
            "each deterministic candidate-ID sample exactly equals scalar-v1 for validity/reason/minute/timestamp and 1e-12 numeric outcomes"
        ),
        "live_state_machine_equivalence_rows": int(live_state_rows),
        "live_state_machine_equivalence": (
            "every valid path exactly equals the completed-one-minute live policy state machine; historical threshold-fill proxy only, not exchange-fill quality"
            if exhaustive_oracle else
            "each deterministic candidate-ID sample exactly equals the completed-one-minute live policy state machine; historical threshold-fill proxy only, not exchange-fill quality"
        ),
    }
    # Complete source coverage is a normal and desirable case.  ``concat``
    # of an empty invalid-parts list has no columns, so construct the empty
    # schema explicitly rather than indexing it as if an invalid row existed.
    if invalid.empty:
        invalid_outcomes = pd.DataFrame({
            "candidate_id": pd.Series(dtype="string"),
            "decision_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "entry_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "entry_price": pd.Series(dtype=float), "exit_timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "exit_price": pd.Series(dtype=float), "gross_bps": pd.Series(dtype=float), "net_bps": pd.Series(dtype=float),
            "exit_reason": pd.Series(dtype="string"), "exit_minute": pd.Series(dtype=float),
            "outcome_available": pd.Series(dtype=bool), "outcome_invalid_reason": pd.Series(dtype="string"),
            "outcome_source": pd.Series(dtype="string"),
        })
    else:
        invalid_outcomes = pd.DataFrame({
            "candidate_id": invalid["candidate_id"].astype(str),
            "decision_timestamp": _utc(invalid["timestamp"]),
            "entry_timestamp": _utc(invalid["entry_ts"]),
            "entry_price": np.nan, "exit_timestamp": pd.NaT, "exit_price": np.nan,
            "gross_bps": np.nan, "net_bps": np.nan,
            "exit_reason": "OUTCOME_UNAVAILABLE_EXCLUDED_FROM_EVALUATION",
            "exit_minute": np.nan, "outcome_available": False,
            "outcome_invalid_reason": invalid["outcome_invalid_reason"].astype(str),
            "outcome_source": "exact_1m_frozen_rich_p8u_v1",
        })
    outcomes = pd.concat([exact, invalid_outcomes], ignore_index=True)
    if len(outcomes) != len(candidates) or outcomes["candidate_id"].duplicated().any():
        raise AssertionError("exact one-minute outcome identities do not cover the sealed target-free request exactly")
    cost_error = (
        pd.to_numeric(exact["gross_bps"], errors="coerce")
        - pd.to_numeric(exact["net_bps"], errors="coerce") - 100.0
    ).abs()
    if (cost_error > 1e-8).any():
        raise AssertionError("frozen rich exact policy cost was not applied exactly once")

    if args.skip_parent_policy_comparison:
        compare = outcomes.copy()
        summary = pd.DataFrame([{
            "target_free_candidates": int(len(candidates)),
            "exact_valid_rows": int(outcomes["outcome_available"].sum()),
            "exact_coverage": float(outcomes["outcome_available"].mean()),
            "parent_valid_rows": None,
            "shared_valid_rows": None,
            "exact_net_bps_mean_shared": None,
            "parent_15m_net_bps_mean_shared": None,
            "exact_minus_parent_net_bps_mean_shared": None,
            "cost_once_max_abs_error_bps": float(cost_error.max()),
            "entry_delay_minutes": delay,
            "comparison_note": "Parent comparison intentionally omitted: this is a target-free successor label ledger before upstream scores exist.",
        }])
    else:
        parent = _parent_policy(Path(args.dual_predictions), candidates)
        compare = outcomes.merge(parent, on="candidate_id", how="left", validate="one_to_one", suffixes=("_exact", "_parent"))
        parent_valid = compare["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(compare["policy_net_bps"], errors="coerce"))
        both = compare["outcome_available"].fillna(False).astype(bool) & parent_valid
        summary = pd.DataFrame([{
            "target_free_candidates": int(len(candidates)),
            "exact_valid_rows": int(outcomes["outcome_available"].sum()),
            "exact_coverage": float(outcomes["outcome_available"].mean()),
            "parent_valid_rows": int(parent_valid.sum()),
            "shared_valid_rows": int(both.sum()),
            "exact_net_bps_mean_shared": float(pd.to_numeric(compare.loc[both, "net_bps"], errors="coerce").mean()),
            "parent_15m_net_bps_mean_shared": float(pd.to_numeric(compare.loc[both, "policy_net_bps"], errors="coerce").mean()),
            "exact_minus_parent_net_bps_mean_shared": float((pd.to_numeric(compare.loc[both, "net_bps"], errors="coerce") - pd.to_numeric(compare.loc[both, "policy_net_bps"], errors="coerce")).mean()),
            "cost_once_max_abs_error_bps": float(cost_error.max()),
            "entry_delay_minutes": delay,
            "comparison_note": "Parent labels use their historical 15m decision-entry proxy; the exact arm uses a +5m actual-entry proxy, so this difference combines entry delay and exit resolution.",
        }])

    output.mkdir(parents=True)
    outcomes.to_parquet(output / "exact_1m_policy_outcomes.parquet", index=False, compression="zstd")
    compare.to_parquet(output / "exact_vs_parent_policy_comparison.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_parquet(output / "source_coverage.parquet", index=False, compression="zstd")
    summary.to_parquet(output / "summary.parquet", index=False, compression="zstd")
    invalid_reason_counts = (
        outcomes.loc[~outcomes["outcome_available"].fillna(False).astype(bool), "outcome_invalid_reason"]
        .astype(str)
        .value_counts()
        .sort_index()
        .to_dict()
    )
    _write_once(output / "run_manifest.json", {
        "schema": "strict_r3_p8u_exact_1m_rich_policy_v1",
        "status": "complete",
        "scope": "offline exact one-minute policy materialisation; no model scoring, portfolio replay, exchange account IO, or order submission",
        "target_free_candidate_request": str(Path(args.candidate_dir).resolve()),
        "candidate_manifest_sha256": _sha256(Path(args.candidate_dir).resolve() / "candidate_manifest.json"),
        "candidate_rows": int(len(candidates)),
        "requested_candidate_rows": requested_candidate_rows,
        "label_available_before": None if resolved_before is None else resolved_before.isoformat(),
        "unresolved_candidates_not_opened": (
            0 if resolved_before is None else int(requested_candidate_rows - len(candidates))
        ),
        "minute_root": str(Path(args.minute_root).resolve()),
        "source_loading": {
            "symbol_load_workers": int(args.symbol_load_workers),
            "symbols_with_frozen_product_identity": int(len(source_groups)),
            "elapsed_seconds": float(source_load_seconds),
            "contract": "parallel immutable local symbol reads; deterministic candidate consumption",
        },
        "frozen_policy": policy_audit,
        "frozen_policy_sha256": _sha256(Path(args.frozen_policy).resolve()),
        "entry": "observed Kraken one-minute open at decision plus five minutes",
        "path": "720 complete post-entry one-minute bars; no interpolation or synthetic flat bars",
        "cost": "100 bps exactly once",
        "oracle_equivalence": oracle_audit,
        "outcome_handling": "candidate route is target-free; missing paths are explicit invalid supervision and excluded only after routing",
        "invalid_outcome_rows": int((~outcomes["outcome_available"].fillna(False).astype(bool)).sum()),
        "invalid_outcome_reason_counts": invalid_reason_counts,
        "parent_policy_comparison": summary.iloc[0].to_dict(),
        "code_sha256": _sha256(Path(__file__).resolve()),
    })
    print(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--dual-predictions", type=Path, default=DEFAULT_DUAL)
    parser.add_argument("--skip-parent-policy-comparison", action="store_true", help="Create a successor label ledger before upstream scores exist; omit only the diagnostic parent-policy comparison.")
    parser.add_argument("--frozen-policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--label-available-before",
        help="Strict UTC cutoff: materialise only labels whose decision+12h05m is earlier; unresolved paths are never opened.",
    )
    parser.add_argument("--minute-root", type=Path, default=DEFAULT_MINUTE_ROOT)
    parser.add_argument(
        "--symbol-load-workers", type=int, default=1,
        help=(
            "Bounded concurrent immutable one-minute source reads. Default 1 preserves "
            "the historical sequential transport; values 2-8 are a read-only acceleration."
        ),
    )
    parser.add_argument("--batch-rows", type=int, default=1_000)
    parser.add_argument(
        "--oracle-sample-size", type=int, default=None,
        help=(
            "Deterministic candidate-ID scalar/live-state audit rows per replay batch. "
            "Omit for exhaustive audit of every valid path."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= int(args.symbol_load_workers) <= 8:
        parser.error("--symbol-load-workers must be between 1 and 8")
    run(args)


if __name__ == "__main__":
    main()
