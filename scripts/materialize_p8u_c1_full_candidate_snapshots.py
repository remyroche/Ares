#!/usr/bin/env python3
"""Materialise complete target-free C1-LVA snapshots for P8U score candidates.

The archived ``entry_sr_oof_features`` panel was produced only for an earlier
routed subset.  It must not be used to judge a C1 MC1 mapper on a full score
population: absent rows then conflate "no structural zone" with "not scored".

This no-order producer instead starts from the union of BCF/current candidate
identities, builds one causal S/R engine per symbol from the declared source
origin, and emits exactly one candidate-time row for every requested score
identity.  A row that has no available support/resistance zone remains in the
output with an explicit unavailable C1 snapshot; it is never dropped.

Source heads are strictly prequential by calendar month.  The structural
snapshot at a decision never consumes a policy outcome; policy labels are not
read by this producer.  The resulting feature panel can safely be joined to
the separate score/policy producer by ``candidate_id`` and ``snapshot_ts``.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.causal_sr_engine import CausalSREngine, read_symbol_bars
from scripts import run_causal_sr_heads as heads


DEFAULT_SOURCE = ROOT / "data_perp/artifacts/causal_sr_engine_2025_train_2026_score_20260830_v1"
DEFAULT_PROFILE = ROOT / "data_perp/artifacts/causal_profile_geometry_2025_train_2026_score_20260831_v2"
DEFAULT_BARS = ROOT / "15m_ohlcv_perp"
SOURCE_ORIGIN = pd.Timestamp("2025-01-01T00:00:00Z")
IDENTITY = ("candidate_id", "__decision_ts__", "__symbol__")
PROFILE_FIELDS = heads.PROFILE_CONTEXT_GROUPS["levels"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _symbol_snapshot_path(root: Path, symbol: str) -> Path:
    return root / f"{symbol.replace('/', '_').replace(':', '_')}.parquet"


def _read_score_identities(path: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    required = {"candidate_id", "__decision_ts__", "__symbol__", "side_name"}
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"score source {path} lacks {missing}")
    frame = pd.read_parquet(path, columns=sorted(required)).copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__symbol__"] = frame["__symbol__"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if not frame["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError(f"score source {path} is not long-only")
    frame = frame.loc[frame["__decision_ts__"].ge(start) & frame["__decision_ts__"].lt(end), list(IDENTITY)].copy()
    if frame.duplicated("candidate_id").any():
        raise ValueError(f"score source {path} duplicates candidate_id")
    return frame


def _union_identities(bcf: Path, current: Path, *, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    left = _read_score_identities(bcf, start=start, end=end)
    right = _read_score_identities(current, start=start, end=end)
    combined = pd.concat((left, right), ignore_index=True)
    semantic = combined.groupby("candidate_id", sort=False).agg(
        decision_n=("__decision_ts__", "nunique"), symbol_n=("__symbol__", "nunique"),
    )
    bad = semantic.loc[semantic.decision_n.ne(1) | semantic.symbol_n.ne(1)]
    if not bad.empty:
        raise AssertionError("BCF/current candidate identities disagree")
    return combined.drop_duplicates("candidate_id").sort_values(["__symbol__", "__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _symbol_raw_worker(payload: tuple[str, pd.DataFrame, str, str, str, str]) -> dict[str, object]:
    """Materialise one symbol independently; failure remains C1-unavailable."""
    symbol, targets, bars_raw, origin_raw, end_raw, scratch_raw = payload
    bars_root, scratch = Path(bars_raw), Path(scratch_raw)
    origin, end = _utc(origin_raw), _utc(end_raw)
    requested = targets.copy()
    try:
        bars = read_symbol_bars(bars_root, symbol)
        bars.index = pd.to_datetime(bars.index, utc=True, errors="raise")
        # A snapshot at a decision timestamp needs the completed 15-minute
        # source bar at that timestamp, not an additional post-decision bar.
        # ``end`` is intentionally padded by the caller for the engine's
        # internal output range, so it must not be used as a source-coverage
        # requirement here.
        final_target = pd.to_datetime(requested["__decision_ts__"], utc=True, errors="raise").max()
        if bars.empty or bars.index.max() < final_target:
            raise ValueError("local 15-minute source lacks the final requested completed decision bar")
        # A listing cannot have history before it existed.  Requiring every
        # asset to start at the global universe origin would incorrectly turn
        # valid later listings into C1-unavailable rows.  Their causal state
        # instead begins at the first locally observed completed 15-minute
        # bar; the engine still supplies an explicit unavailable snapshot
        # during any insufficient structural warm-up.
        effective_origin = max(origin, pd.Timestamp(bars.index.min()))
        target_map: dict[pd.Timestamp, list[dict[str, object]]] = {}
        for stamp, part in requested.groupby("__decision_ts__", sort=False):
            target_map[pd.Timestamp(stamp)] = [
                {
                    "target_kind": "entry",
                    "target_id": str(row.candidate_id),
                    "candidate_id": str(row.candidate_id),
                }
                for row in part.itertuples(index=False)
            ]
        engine = CausalSREngine(
            symbol, bars, output_start=effective_origin, output_end=end,
            snapshot_targets=target_map, record_tape=False,
        )
        _candidates, _zones, _events, snapshots = engine.run()
        if snapshots.empty:
            snapshots = pd.DataFrame(columns=["candidate_id", "snapshot_ts"])
        snapshots["candidate_id"] = snapshots.get("candidate_id", pd.Series(dtype=str)).astype(str)
        snapshots["snapshot_ts"] = pd.to_datetime(snapshots.get("snapshot_ts"), utc=True, errors="coerce")
        emitted = snapshots.loc[:, [column for column in snapshots.columns if column != "__decision_ts__"]].copy()
        snapshot_path = _symbol_snapshot_path(scratch, symbol)
        emitted.to_parquet(snapshot_path, index=False, compression="zstd")
        emitted_ids = set(emitted.candidate_id.dropna().astype(str)) if "candidate_id" in emitted else set()
        return {
            "__symbol__": symbol, "source_ready": True, "requested_rows": int(len(requested)),
            "emitted_rows": int(len(emitted)), "emitted_candidate_ids": int(len(emitted_ids)),
            "missing_snapshot_ids": int(len(requested) - len(emitted_ids)),
            "effective_source_origin": effective_origin.isoformat(),
            "path": str(snapshot_path),
        }
    except Exception as exc:  # A source failure cannot delete a target-free candidate.
        return {
            "__symbol__": symbol, "source_ready": False, "requested_rows": int(len(requested)),
            "emitted_rows": 0, "emitted_candidate_ids": 0, "missing_snapshot_ids": int(len(requested)),
            "exception_type": type(exc).__name__, "exception": str(exc),
        }


def _score_monthly_prequential(
    raw: pd.DataFrame,
    *, source: Path, profile: Path, start: pd.Timestamp, end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit source heads on prior resolved interactions, then score each month."""
    interactions = pd.read_parquet(source / "interaction_events.parquet")
    interactions["event_ts"] = pd.to_datetime(interactions["event_ts"], utc=True, errors="raise")
    interactions["label_available_ts"] = pd.to_datetime(interactions["label_available_ts"], utc=True, errors="raise")
    states = pd.read_parquet(profile / "profile_hourly_states.parquet")
    required_states = {"__symbol__", "state_ts", *PROFILE_FIELDS}
    missing_states = sorted(required_states.difference(states.columns))
    if missing_states:
        raise ValueError(f"profile state contract lacks {missing_states}")
    interactions = heads._merge_profile_context(
        interactions, states, timestamp="event_ts", fields=PROFILE_FIELDS,
    )
    zones = heads._zone_snapshot_rows(raw, context_features=())
    if zones.empty:
        return raw.iloc[0:0].copy(), pd.DataFrame()
    zones = heads._merge_profile_context(zones, states, timestamp="snapshot_ts", fields=PROFILE_FIELDS)
    features = (*heads.CONDITIONAL_FEATURES, *PROFILE_FIELDS, heads.PROFILE_CONTEXT_AVAILABLE)
    # Score every calendar month which intersects the requested half-open
    # candidate window.  ``end`` can fall within the same month as ``start``
    # for a live/forward partial-month receipt; subtracting a MonthBegin from
    # its normalised date used to yield an empty month list in that case and
    # incorrectly prevented an otherwise valid strictly-prior September fit.
    first_month = start.normalize().replace(day=1)
    last_included = end - pd.Timedelta(nanoseconds=1)
    last_month = last_included.normalize().replace(day=1)
    months = pd.date_range(first_month, last_month, freq="MS", tz="UTC")
    scored: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for held in months:
        held_end = held + pd.offsets.MonthBegin(1)
        train = interactions.loc[
            interactions.label_available_ts.lt(held) & interactions.event_ts.lt(held)
        ].copy()
        test = zones.loc[zones.snapshot_ts.ge(held) & zones.snapshot_ts.lt(held_end)].copy()
        if test.empty:
            continue
        if len(train) < 2_000:
            raise RuntimeError(f"{held:%Y-%m}: insufficient strict-prior C1 interaction support ({len(train)})")
        values = heads._fit_predict(train, test, features)
        test["sr_prior_strength"] = values[0]
        test["sr_conditional_strength"] = values[1]
        test["sr_accepted_break_probability"] = values[2]
        test["sr_reaction_magnitude_q50"] = values[3]
        scored.append(test)
        audit.append({
            "held_month": f"{held:%Y-%m}", "train_rows": int(len(train)),
            "score_zone_rows": int(len(test)), "candidate_ids": int(test.candidate_id.nunique()),
            "train_label_max": str(train.label_available_ts.max()),
            "profile_available_fraction": float(test[heads.PROFILE_CONTEXT_AVAILABLE].mean()),
        })
    if not scored:
        return zones.iloc[0:0].copy(), pd.DataFrame(audit)
    return pd.concat(scored, ignore_index=True), pd.DataFrame(audit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--start", required=True, help="inclusive UTC candidate-decision timestamp")
    parser.add_argument("--end", required=True, help="exclusive UTC candidate-decision timestamp")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--bars-root", type=Path, default=DEFAULT_BARS)
    parser.add_argument("--origin", default=SOURCE_ORIGIN.isoformat())
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--symbol", action="append", help="bounded no-order diagnostic; repeatable exact symbol filter")
    parser.add_argument("--resume", action="store_true", help="resume an interrupted raw-symbol phase before source-head scoring")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(f"immutable output exists: {output}")
    if output.exists() and args.resume and (output / "run_manifest.json").exists():
        raise FileExistsError("completed immutable output cannot be resumed")
    start, end, origin = _utc(args.start), _utc(args.end), _utc(args.origin)
    if end <= start or start < origin:
        raise ValueError("require source origin <= start < end")
    bcf, current = args.bcf.resolve(), args.current.resolve()
    source, profile, bars_root = args.source.resolve(), args.profile.resolve(), args.bars_root.resolve()
    identities = _union_identities(bcf, current, start=start, end=end)
    if args.symbol:
        requested_symbols = {str(value) for value in args.symbol}
        identities = identities.loc[identities["__symbol__"].isin(requested_symbols)].copy()
    if identities.empty:
        raise ValueError("no target-free score identities in requested range")
    output.mkdir(parents=True, exist_ok=bool(args.resume))
    scratch = output / "raw_symbol_snapshots"
    scratch.mkdir(exist_ok=bool(args.resume))
    rows: list[dict[str, object]] = []
    payloads = []
    for symbol, part in identities.groupby("__symbol__", sort=True):
        symbol = str(symbol)
        prior_path = _symbol_snapshot_path(scratch, symbol)
        if args.resume and prior_path.is_file():
            emitted = pd.read_parquet(prior_path, columns=["candidate_id"])
            emitted_ids = emitted.candidate_id.dropna().astype(str)
            requested_ids = set(part.candidate_id.astype(str))
            if emitted_ids.duplicated().any() or not set(emitted_ids).issubset(requested_ids):
                raise AssertionError(f"resume snapshot does not match current target-free identity for {symbol}")
            rows.append({
                "__symbol__": symbol, "source_ready": True, "requested_rows": int(len(part)),
                "emitted_rows": int(len(emitted)), "emitted_candidate_ids": int(emitted_ids.nunique()),
                "missing_snapshot_ids": int(len(part) - emitted_ids.nunique()),
                "status": "resumed_verified_symbol_snapshot", "path": str(prior_path),
            })
            continue
        payloads.append((symbol, part.copy(), str(bars_root), origin.isoformat(), (end - pd.Timedelta(minutes=15)).isoformat(), str(scratch)))
    with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
        futures = {pool.submit(_symbol_raw_worker, item): item[0] for item in payloads}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(json.dumps(row, default=str), flush=True)
    coverage = pd.DataFrame(rows).sort_values("__symbol__", kind="stable")
    coverage.to_parquet(output / "source_coverage.parquet", index=False, compression="zstd")
    # A fully source-unavailable target set is still a valid target-free
    # result: every candidate survives as an explicit C1-unavailable row.
    # Do not turn that fail-closed state into a KeyError merely because no
    # worker emitted a ``path`` field.
    ready_paths = (
        coverage.loc[coverage.source_ready, "path"].dropna()
        if "path" in coverage.columns else pd.Series(dtype=object)
    )
    emitted_frames = [pd.read_parquet(Path(path)) for path in ready_paths]
    raw_emitted = pd.concat(emitted_frames, ignore_index=True) if emitted_frames else pd.DataFrame(columns=["candidate_id", "snapshot_ts"])
    if not raw_emitted.empty and raw_emitted.duplicated(["candidate_id", "snapshot_ts"]).any():
        raise AssertionError("C1 engine emitted duplicate candidate-time snapshot")
    requested = identities.rename(columns={"__decision_ts__": "snapshot_ts"}).copy()
    raw = requested.merge(raw_emitted, on=["candidate_id", "snapshot_ts"], how="left", suffixes=("", "_engine"), validate="one_to_one")
    if len(raw) != len(requested) or raw.duplicated(["candidate_id", "snapshot_ts"]).any():
        raise AssertionError("full C1 candidate snapshot identity changed")
    if "target_kind" not in raw:
        raw["target_kind"] = "entry"
    else:
        raw["target_kind"] = raw["target_kind"].fillna("entry")
    if "target_id" not in raw:
        raw["target_id"] = raw["candidate_id"].astype(str)
    else:
        raw["target_id"] = raw["target_id"].fillna(raw["candidate_id"]).astype(str)
    # A source-local engine failure deliberately retains the candidate without
    # a zone.  Normalise that explicit unavailable state before converting to
    # one-row-per-zone source-head inputs.
    for column in ("support_available", "resistance_available"):
        if column not in raw:
            raw[column] = False
        else:
            raw[column] = raw[column].fillna(False).astype(bool)
    raw.to_parquet(output / "raw_candidate_snapshots.parquet", index=False, compression="zstd")
    zone_predictions, monthly_audit = _score_monthly_prequential(raw, source=source, profile=profile, start=start, end=end)
    if zone_predictions.empty:
        wide = raw.loc[:, ["__symbol__", "snapshot_ts", "target_kind", "target_id", "candidate_id"]].copy()
    else:
        wide_scored = heads._wide_snapshot_predictions(zone_predictions)
        wide = raw.loc[:, ["__symbol__", "snapshot_ts", "target_kind", "target_id", "candidate_id"]].merge(
            wide_scored, on=["__symbol__", "snapshot_ts", "target_kind", "target_id", "candidate_id"], how="left", validate="one_to_one",
        )
    output_columns = (
        "sr_long_support_hold_strength", "sr_long_resistance_break_probability",
        "sr_long_downside_break_probability", "sr_long_resistance_rejection_strength",
        "sr_long_structure_balance", "sr_long_support_distance_atr",
        "sr_long_resistance_distance_atr", "sr_support_prior_strength",
        "sr_resistance_prior_strength", "sr_support_reaction_magnitude_q50",
        "sr_resistance_reaction_magnitude_q50",
    )
    for column in output_columns:
        if column not in wide:
            wide[column] = np.nan
    wide["sr_snapshot_available"] = wide.loc[:, list(output_columns)].notna().any(axis=1).astype("int8")
    wide.to_parquet(output / "entry_sr_oof_features.parquet", index=False, compression="zstd")
    zone_predictions.to_parquet(output / "zone_snapshot_head_oof_predictions.parquet", index=False, compression="zstd")
    monthly_audit.to_parquet(output / "source_head_fold_audit.parquet", index=False, compression="zstd")
    manifest: dict[str, Any] = {
        "schema": "p8u-c1-full-candidate-snapshots-v1",
        "scope": "no-order, target-free complete score-candidate C1-LVA source materialisation",
        "candidate_window": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "source_origin": origin.isoformat(),
        "sources": {
            "bcf": str(bcf), "bcf_sha256": _sha256(bcf),
            "current": str(current), "current_sha256": _sha256(current),
            "sr": str(source), "sr_manifest_sha256": _sha256(source / "run_manifest.json"),
            "profile": str(profile / "profile_hourly_states.parquet"), "profile_sha256": _sha256(profile / "profile_hourly_states.parquet"),
        },
        "counts": {
            "requested_candidate_rows": int(len(requested)),
            "requested_symbols": int(requested.__symbol__.nunique()),
            "source_ready_symbols": int(coverage.source_ready.sum()),
            "engine_snapshot_rows": int(len(raw_emitted)),
            "c1_snapshot_available_rows": int(wide.sr_snapshot_available.sum()),
            "c1_snapshot_available_fraction": float(wide.sr_snapshot_available.mean()),
        },
        "causality": {
            "candidates": "unioned score identities are target-free; no policy fields are read",
            "geometry": "each symbol replays completed 15-minute bars from the fixed origin; parent interaction outcomes resolve only after their own eight-hour horizon",
            "source_heads": "each scored month fits only interactions with label_available_ts strictly before the held month",
            "missingness": "every requested candidate survives; missing source/zone output remains an explicit unavailable C1 snapshot",
            "authority": "no admission, portfolio, exchange, or order authority",
        },
    }
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
