#!/usr/bin/env python3
"""Materialise one full-universe, target-free C1-LVA snapshot from append state.

The S/R state must already have been advanced through the exact completed
15-minute decision bar by ``append_causal_sr_c1_state.py``.  This producer
therefore never rebuilds or refits Geometry/K9/S/R state: for every eligible
symbol it reads the already-processed terminal state, verifies source overlap,
and emits the target-free snapshot without saving a modified checkpoint.

Symbols outside the frozen C1 state intersection remain explicit unavailable
rows.  No candidate is removed merely because C1 is unavailable.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_sr_engine import read_symbol_bars
from extreme_price_movements.inference.causal_sr_c1_lva_bundle import (
    CausalSRC1LVABundle,
    OUTPUT_COLUMNS,
)
from extreme_price_movements.inference.causal_sr_c1_state import (
    CausalSRC1AppendState,
    score_c1_lva_target_free,
)


SCHEMA = "p8u-c1-lva-runtime-snapshot-v1"
IDENTITY = ("candidate_id", "__decision_ts__", "__symbol__", "side_name")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(raw: object) -> pd.Timestamp:
    stamp = pd.Timestamp(raw)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _unavailable(candidate: dict[str, object], *, reason: str, state_eligible: bool) -> dict[str, object]:
    output = {key: candidate[key] for key in IDENTITY}
    output["snapshot_ts"] = candidate["__decision_ts__"]
    output["target_kind"] = "entry"
    output["target_id"] = candidate["candidate_id"]
    for column in OUTPUT_COLUMNS:
        output[column] = float("nan")
    output["sr_snapshot_available"] = False
    output["c1_lva_source_state"] = "append_only_completed_bar" if state_eligible else "c1_state_unavailable"
    output["c1_runtime_status"] = reason
    output["c1_state_eligible"] = bool(state_eligible)
    return output


def _worker(payload: tuple[dict[str, object], str, str, str, str, str]) -> dict[str, object]:
    candidate, state_root_raw, source_origin_raw, engine_source_raw, bundle_raw, bars_root_raw = payload
    symbol = str(candidate["__symbol__"])
    decision = _utc(candidate["__decision_ts__"])
    state_root, bundle_root, bars_root = Path(state_root_raw), Path(bundle_raw), Path(bars_root_raw)
    try:
        store = CausalSRC1AppendState(
            state_root, source_origin=_utc(source_origin_raw), engine_source_path=Path(engine_source_raw),
        )
        checkpoint = store.checkpoint_path(symbol)
        if not checkpoint.is_file():
            raise FileNotFoundError("C1 append-state checkpoint is unavailable")
        before = _sha256(checkpoint)
        bars = read_symbol_bars(bars_root, symbol)
        bars.index = pd.to_datetime(bars.index, utc=True, errors="raise")
        bars = bars.loc[bars.index <= decision]
        if bars.empty or bars.index.max() != decision:
            raise ValueError("C1 source lacks the exact completed decision bar")
        target = {
            "target_kind": "entry", "target_id": str(candidate["candidate_id"]),
            "candidate_id": str(candidate["candidate_id"]),
        }
        output = score_c1_lva_target_free(
            state=store,
            bundle=CausalSRC1LVABundle.load(bundle_root),
            symbol=symbol,
            bars=bars,
            # C1 S/R requires the complete source history to verify the
            # append-only chain.  LVA is separately bounded to exactly its
            # declared 30-calendar-day warm-up, so replaying it does not scan
            # the entire pre-history on every inference candle.
            lva_bars=bars.loc[bars.index >= decision.floor("1h") - pd.Timedelta(days=30)],
            decision_ts=decision,
            targets=[target],
        )
        after = _sha256(checkpoint)
        if before != after:
            raise AssertionError("same-bar C1 snapshot rewrote the append-state checkpoint")
        if output.empty:
            return {
                "row": _unavailable(candidate, reason="no_active_sr_zone", state_eligible=True),
                "coverage": {"__symbol__": symbol, "candidate_id": candidate["candidate_id"], "state_eligible": True,
                             "snapshot_available": False, "status": "no_active_sr_zone", "checkpoint_unchanged": True},
            }
        if len(output) != 1 or output["candidate_id"].astype(str).tolist() != [str(candidate["candidate_id"])]:
            raise AssertionError("C1 source emitted an unexpected candidate identity")
        row = output.iloc[0].to_dict()
        row.update({key: candidate[key] for key in IDENTITY})
        row["sr_snapshot_available"] = bool(row.get("sr_snapshot_available", False))
        row["c1_runtime_status"] = "available" if row["sr_snapshot_available"] else "no_active_sr_zone"
        row["c1_state_eligible"] = True
        return {
            "row": row,
            "coverage": {"__symbol__": symbol, "candidate_id": candidate["candidate_id"], "state_eligible": True,
                         "snapshot_available": bool(row["sr_snapshot_available"]), "status": str(row["c1_runtime_status"]),
                         "checkpoint_unchanged": True},
        }
    except Exception as exc:
        # A frozen-state eligible row that cannot produce a source-valid
        # snapshot is made explicit.  The caller decides whether this
        # row-local C1 unavailability is admissible; no identity disappears.
        return {
            "row": _unavailable(candidate, reason=f"runtime_error:{type(exc).__name__}", state_eligible=True),
            "coverage": {"__symbol__": symbol, "candidate_id": candidate["candidate_id"], "state_eligible": True,
                         "snapshot_available": False, "status": f"runtime_error:{type(exc).__name__}",
                         "exception": str(exc), "checkpoint_unchanged": None},
        }


def _load_candidates(path: Path, *, decision: pd.Timestamp) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=list(IDENTITY)).copy()
    missing = set(IDENTITY).difference(frame.columns)
    if missing:
        raise KeyError(f"candidate receipt lacks identities: {sorted(missing)}")
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    frame["side_name"] = frame["side_name"].astype(str).str.lower()
    frame = frame.loc[frame["__decision_ts__"].eq(decision)].copy()
    if frame.empty:
        raise ValueError("candidate receipt has no rows at requested decision timestamp")
    if not frame["side_name"].eq("long").all() or frame.duplicated(list(IDENTITY)).any():
        raise ValueError("runtime C1 candidates violate long-only identity uniqueness")
    return frame.sort_values(["__symbol__", "candidate_id"], kind="stable").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--state-advance-receipt", type=Path, required=True)
    parser.add_argument("--source-map", type=Path, required=True)
    parser.add_argument("--c1-bundle", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--decision-ts", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bars-root", type=Path, default=ROOT / "15m_ohlcv_perp")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    state_root, output = args.state_root.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("runtime C1 snapshot output must be immutable")
    decision = _utc(args.decision_ts)
    if decision != decision.floor("15min"):
        raise ValueError("C1 runtime decision must be a completed 15-minute timestamp")
    state_manifest_path = state_root / "state_manifest.json"
    if not state_manifest_path.is_file():
        raise FileNotFoundError("C1 state root lacks its immutable manifest")
    state_manifest = json.loads(state_manifest_path.read_text(encoding="utf-8"))
    source_origin = _utc(state_manifest.get("source_origin"))
    advance_manifest_path = args.state_advance_receipt.resolve() / "run_manifest.json"
    if not advance_manifest_path.is_file():
        raise FileNotFoundError("C1 state advance receipt is unavailable")
    advance = json.loads(advance_manifest_path.read_text(encoding="utf-8"))
    if advance.get("status") != "pass_append_only_c1_state" or not bool(advance.get("state_root_promoted")):
        raise ValueError("C1 state advance receipt was not atomically promoted")
    if Path(str(advance.get("state_root"))).resolve() != state_root:
        raise ValueError("C1 state advance receipt belongs to another state root")
    if _utc(advance.get("end_inclusive")) != decision:
        raise ValueError("C1 state advance receipt must end at the exact snapshot decision")
    if int(advance.get("failed_symbols", -1)) != 0:
        raise ValueError("C1 state advance receipt has failed symbol coverage")
    if str(advance.get("state_manifest_sha256")) != _sha256(state_manifest_path):
        raise ValueError("C1 state advance manifest hash mismatch")
    source_map_payload = json.loads(args.source_map.resolve().read_text(encoding="utf-8"))
    eligible = set(map(str, source_map_payload.get("c1_state_eligible_symbols") or ()))
    unavailable = set(map(str, source_map_payload.get("c1_unavailable_symbols") or ()))
    if not eligible or eligible.intersection(unavailable):
        raise ValueError("C1 source map has invalid eligibility partition")
    candidates = _load_candidates(args.candidates.resolve(), decision=decision)
    universe = set(candidates["__symbol__"])
    if not universe.issubset(eligible | unavailable):
        raise ValueError("candidate universe contains symbols outside the frozen C1 source map")
    bundle = CausalSRC1LVABundle.load(args.c1_bundle.resolve())
    candidate_rows = candidates.to_dict("records")
    work = [row for row in candidate_rows if str(row["__symbol__"]) in eligible]
    results: list[dict[str, Any]] = []
    payloads = [
        (row, str(state_root), source_origin.isoformat(), str(ROOT / "extreme_price_movements/causal_sr_engine.py"),
         str(args.c1_bundle.resolve()), str(args.bars_root.resolve()))
        for row in work
    ]
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(_worker, payload): payload[0]["candidate_id"] for payload in payloads}
        for future in as_completed(futures):
            results.append(future.result())
    row_by_id = {str(item["row"]["candidate_id"]): item["row"] for item in results}
    coverage = [item["coverage"] for item in results]
    for row in candidate_rows:
        if str(row["__symbol__"]) in unavailable:
            unavail = _unavailable(row, reason="outside_frozen_c1_state_intersection", state_eligible=False)
            row_by_id[str(row["candidate_id"])] = unavail
            coverage.append({"__symbol__": row["__symbol__"], "candidate_id": row["candidate_id"], "state_eligible": False,
                             "snapshot_available": False, "status": "outside_frozen_c1_state_intersection", "checkpoint_unchanged": True})
    if set(row_by_id) != set(candidates["candidate_id"]):
        raise AssertionError("C1 runtime output lost a target-free candidate identity")
    output_rows = pd.DataFrame([row_by_id[str(value)] for value in candidates["candidate_id"]])
    if output_rows.duplicated("candidate_id").any() or len(output_rows) != len(candidates):
        raise AssertionError("C1 runtime output identity multiplicity is invalid")
    output.mkdir(parents=True, exist_ok=False)
    panel_path, coverage_path = output / "entry_sr_oof_features.parquet", output / "source_coverage.parquet"
    output_rows.to_parquet(panel_path, index=False, compression="zstd")
    coverage_frame = pd.DataFrame(coverage).sort_values(["__symbol__", "candidate_id"], kind="stable")
    coverage_frame.to_parquet(coverage_path, index=False, compression="zstd")
    runtime_failures = coverage_frame["status"].astype(str).str.startswith("runtime_error:")
    manifest = {
        "schema": SCHEMA,
        "status": "PASS_TARGET_FREE_C1_RUNTIME_SNAPSHOT" if not runtime_failures.any() else "PARTIAL_TARGET_FREE_C1_RUNTIME_SNAPSHOT",
        "decision_ts": decision.isoformat(),
        "candidate_rows": int(len(candidates)),
        "candidate_sha256": _sha256(args.candidates.resolve()),
        "state_root": str(state_root),
        "state_manifest_sha256": _sha256(state_manifest_path),
        "state_advance_receipt": str(args.state_advance_receipt.resolve()),
        "state_advance_manifest_sha256": _sha256(advance_manifest_path),
        "source_map": str(args.source_map.resolve()),
        "source_map_sha256": _sha256(args.source_map.resolve()),
        "c1_bundle": str(args.c1_bundle.resolve()),
        "c1_bundle_manifest_sha256": bundle.manifest_sha256,
        "output": {"entry_sr_oof_features.parquet": _sha256(panel_path), "source_coverage.parquet": _sha256(coverage_path)},
        "coverage": {
            "state_eligible": int(coverage_frame["state_eligible"].fillna(False).astype(bool).sum()),
            "explicit_state_unavailable": int((~coverage_frame["state_eligible"].fillna(False).astype(bool)).sum()),
            "snapshot_available": int(coverage_frame["snapshot_available"].fillna(False).astype(bool).sum()),
            "runtime_failures": int(runtime_failures.sum()),
            "checkpoint_mutations": int((~coverage_frame["checkpoint_unchanged"].fillna(False).astype(bool)).sum()),
        },
        "causality": {
            "inputs": "only frozen target-free candidates plus completed 15-minute source through exact decision timestamp",
            "state": "append-only checkpoint is read and overlap-verified; same-bar snapshot does not save a checkpoint",
            "outcome_columns_consumed": [],
            "model_refit": False,
            "exchange_order_submission_called": False,
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if runtime_failures.any() or int(manifest["coverage"]["checkpoint_mutations"]) != 0:
        raise RuntimeError("C1 runtime snapshot failed coverage or mutated a checkpoint")
    print(output)


if __name__ == "__main__":
    main()
