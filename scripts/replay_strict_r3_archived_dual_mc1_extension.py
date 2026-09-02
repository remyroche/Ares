#!/usr/bin/env python3
"""Build an offline strict BCF/current-v5 dual-MC1 live-contract extension.

The producer is intentionally ordered:

  immutable target-free archived scores
  -> each mapper scored only from already-resolved policy labels
  -> BCF >= 30 AND current-v5 >= 30 target-free request
  -> reuse exact 1m paths only after the request is sealed

It never modifies the live producer, its state, or any exchange-facing file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_policy_contract import Exact1mExecutionContract
from extreme_price_movements.strict_r3_bcf_mc1_mapper import BCFMC1D2Bundle, derive_bcf_mc1_features
from extreme_price_movements.strict_r3_mc1_mapper import MC1D2Bundle, _robust_mean, score_bands


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _utc(value: Any) -> pd.Series | pd.Timestamp:
    if isinstance(value, pd.Series):
        return pd.to_datetime(value, utc=True, errors="raise")
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _load_scores(path: Path, *, fields: list[str]) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=fields).copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = _utc(frame["__decision_ts__"])
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"duplicate candidate IDs in {path}")
    return frame


def _label_lookup(prequential: Path, augment: Path, *, candidate_ids: pd.Index) -> pd.DataFrame:
    columns = ["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"]
    ids = pd.Index(candidate_ids, dtype="string").dropna().unique().tolist()
    if not ids:
        raise ValueError("cannot build a causal label lookup without score identities")
    # Reading the full 2.8m-row ledger is both unnecessary and can exhaust the
    # replay worker.  The maps use no label for an identity that is absent from
    # their own reference/held score panels, so predicate-push the exact IDs.
    base = ds.dataset(prequential, format="parquet").to_table(
        columns=columns,
        filter=pc.field("candidate_id").isin(pa.array(ids, type=pa.string())),
    ).to_pandas()
    later = ds.dataset(augment, format="parquet").to_table(
        columns=columns,
        filter=pc.field("candidate_id").isin(pa.array(ids, type=pa.string())),
    ).to_pandas()
    for item in (base, later):
        item["candidate_id"] = item["candidate_id"].astype(str)
        item["policy_label_available_ts"] = _utc(item["policy_label_available_ts"])
        item["policy_path_valid"] = item["policy_path_valid"].fillna(False).astype(bool)
    # Later source gives the repaired August policy outcome where identities overlap.
    output = pd.concat([base, later], ignore_index=True).drop_duplicates("candidate_id", keep="last")
    if output["candidate_id"].duplicated().any():
        raise AssertionError("label lookup did not deduplicate")
    return output


def _with_labels(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    result = scores.merge(labels, on="candidate_id", how="left", validate="one_to_one")
    result["policy_path_valid"] = result["policy_path_valid"].fillna(False).astype(bool)
    return result


def _native_bcf(scores: pd.DataFrame) -> pd.DataFrame:
    need = {"conditional_consensus_rank", "ordinary_shadow_consensus_rank", "correctness_rank"}
    if need.issubset(scores.columns):
        return scores
    native = derive_bcf_mc1_features(scores)
    # The derived table also carries source coordinates such as ``final_score``.
    # Retain those from the immutable score receipt and attach only the fields
    # missing from older BCF archives.
    return scores.merge(
        native.loc[:, ["candidate_id", *sorted(need)]],
        on="candidate_id", how="inner", validate="one_to_one",
    )


def _history(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.loc[:, [
        "candidate_id", "__decision_ts__", "final_score", "policy_path_valid",
        "policy_net_bps", "policy_label_available_ts",
    ]].copy()
    result = result.loc[result["policy_path_valid"] & pd.to_numeric(result["policy_net_bps"], errors="coerce").notna()].copy()
    result["policy_label_available_ts"] = _utc(result["policy_label_available_ts"])
    result["score_band"] = score_bands(result)
    return result


def _map_bcf(held: pd.DataFrame, history: pd.DataFrame, bundle: BCFMC1D2Bundle) -> pd.DataFrame:
    features = list(bundle.manifest["features_ordered"])
    curve = np.asarray(bundle.payload["structural_curve_bps"], dtype=float)
    labels_ns = history["policy_label_available_ts"].astype("int64").to_numpy()
    decisions_ns = history["__decision_ts__"].astype("int64").to_numpy()
    residual = pd.to_numeric(history["policy_net_bps"], errors="raise").to_numpy(float) - curve[history["score_band"].to_numpy(int)]
    window = int(pd.Timedelta(days=21).value)
    parts: list[pd.DataFrame] = []
    for decision, group in held.groupby("__decision_ts__", sort=True):
        now = pd.Timestamp(decision).value
        shift = _robust_mean(residual[(labels_ns < now) & (decisions_ns >= now - window)], trim=0.10)
        x = group.loc[:, features].apply(pd.to_numeric, errors="coerce")
        available = np.isfinite(x.to_numpy(float)).all(axis=1) & np.isfinite(shift)
        prediction = np.full(len(group), np.nan, dtype=float)
        if available.any():
            prediction[available] = bundle.payload["model"].predict(x.loc[available, features]) + shift
        parts.append(pd.DataFrame({
            "candidate_id": group["candidate_id"].astype(str).to_numpy(),
            "bcf_mc1_expected_bps": prediction,
            "bcf_mc1_recent_shift_bps": shift,
            "bcf_mc1_available": available,
        }))
    return pd.concat(parts, ignore_index=True)


def _map_current(held: pd.DataFrame, history: pd.DataFrame, bundle: MC1D2Bundle) -> pd.DataFrame:
    features = list(bundle.manifest["features_ordered"])
    curve = np.asarray(bundle.payload["structural_curve_bps"], dtype=float)
    labels_ns = history["policy_label_available_ts"].astype("int64").to_numpy()
    decisions_ns = history["__decision_ts__"].astype("int64").to_numpy()
    residual = pd.to_numeric(history["policy_net_bps"], errors="raise").to_numpy(float) - curve[history["score_band"].to_numpy(int)]
    window = int(pd.Timedelta(days=21).value)
    parts: list[pd.DataFrame] = []
    for decision, group in held.groupby("__decision_ts__", sort=True):
        now = pd.Timestamp(decision).value
        shift = _robust_mean(residual[(labels_ns < now) & (decisions_ns >= now - window)], trim=0.10)
        x = group.loc[:, features].apply(pd.to_numeric, errors="coerce")
        available = np.isfinite(x.to_numpy(float)).all(axis=1) & np.isfinite(shift)
        prediction = np.full(len(group), np.nan, dtype=float)
        if available.any():
            prediction[available] = bundle.payload["model"].predict(x.loc[available, features]) + shift
        parts.append(pd.DataFrame({
            "candidate_id": group["candidate_id"].astype(str).to_numpy(),
            "current_v5_mc1_expected_bps": prediction,
            "current_v5_mc1_recent_shift_bps": shift,
            "current_v5_mc1_available": available,
        }))
    return pd.concat(parts, ignore_index=True)


def _filter_dataset(source: Path, candidates: pd.DataFrame, out: Path) -> dict[str, int]:
    rows = pd.read_parquet(source / "training_rows.parquet").copy()
    paths = np.load(source / "exact_paths.npz", allow_pickle=False)
    ids = paths["candidate_id"].astype(str)
    if not np.array_equal(ids, rows["candidate_id"].astype(str).to_numpy()):
        raise AssertionError("source rows and paths are not identity aligned")
    wanted = set(candidates["candidate_id"].astype(str))
    keep = rows["candidate_id"].astype(str).isin(wanted).to_numpy()
    rows = rows.loc[keep].copy().reset_index(drop=True)
    arrays = {key: paths[key][keep] for key in paths.files}
    missing = sorted(wanted.difference(set(rows["candidate_id"].astype(str))))
    audit = pd.read_parquet(source / "candidate_path_audit.parquet")
    pending = audit.loc[audit["candidate_id"].astype(str).isin(missing)].copy()
    if len(rows):
        if not rows["path_valid"].fillna(False).all():
            raise AssertionError("source replay rows must be resolved paths only")
    out.mkdir(parents=True, exist_ok=False)
    rows.to_parquet(out / "training_rows.parquet", index=False, compression="zstd")
    np.savez_compressed(out / "exact_paths.npz", **arrays)
    pending.to_parquet(out / "candidate_path_audit.parquet", index=False, compression="zstd")
    source_manifest = json.loads((source / "dataset_manifest.json").read_text())
    source_manifest["candidate_rows"] = int(len(candidates))
    source_manifest["valid_rows"] = int(len(rows))
    source_manifest["invalid_rows"] = int(len(missing))
    source_manifest["candidate_source"] = {
        "target_free": True,
        "score_column": "priority_bps",
        "selection_inputs": ["bcf_mc1_expected_bps", "current_v5_mc1_expected_bps"],
        "forbidden_selection_inputs": ["label", "outcome", "policy_net_bps", "policy_path_valid"],
        "predicate": "BCF MC1 >=30 AND current-v5 MC1 >=30",
    }
    (out / "dataset_manifest.json").write_text(json.dumps(source_manifest, indent=2, sort_keys=True) + "\n")
    return {"resolved": int(len(rows)), "pending_or_missing": int(len(missing))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf-held", type=Path, required=True)
    parser.add_argument("--bcf-reference", type=Path, required=True)
    parser.add_argument("--current-held", type=Path, required=True)
    parser.add_argument("--current-reference", type=Path, required=True)
    parser.add_argument("--prequential-labels", type=Path, required=True)
    parser.add_argument("--august-labels", type=Path, required=True)
    parser.add_argument("--bcf-bundle", type=Path, required=True)
    parser.add_argument("--current-bundle", type=Path, required=True)
    parser.add_argument("--exact-source", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    out.mkdir(parents=True)
    start, end = _utc(args.start), _utc(args.end)
    bcf_fields = [
        # Native BCF archives intentionally omit the redundant symbol field.
        # Symbol identity is taken from the matched current-v5 score receipt.
        "candidate_id", "__decision_ts__", "side_name", "final_score", "base_rank42", "upstream", "consensus_rank",
        *[f"residual_head__cap{cap}_{mode}__rank" for cap in (40, 60, 80, 100, 120) for mode in ("ordinary", "equal_month")],
    ]
    current_fields = ["candidate_id", "__decision_ts__", "__symbol__", "side_name", *list(MC1D2Bundle.load(args.current_bundle).manifest["features_ordered"])]
    bcf_held = _native_bcf(_load_scores(args.bcf_held, fields=bcf_fields))
    bcf_ref = _native_bcf(_load_scores(args.bcf_reference, fields=bcf_fields))
    current_held = _load_scores(args.current_held, fields=current_fields)
    current_ref = _load_scores(args.current_reference, fields=current_fields)
    bcf_held = bcf_held.loc[bcf_held["__decision_ts__"].between(start, end, inclusive="left")].copy()
    current_held = current_held.loc[current_held["__decision_ts__"].between(start, end, inclusive="left")].copy()
    # Both frozen MC1 maps have a 21-calendar-day residual shift.  Older
    # reference rows can never affect this extension's first decision.
    history_start = start - pd.Timedelta(days=21)
    bcf_ref = bcf_ref.loc[bcf_ref["__decision_ts__"].ge(history_start)].copy()
    current_ref = current_ref.loc[current_ref["__decision_ts__"].ge(history_start)].copy()
    ids = pd.Index(pd.concat([
        bcf_held["candidate_id"], bcf_ref["candidate_id"],
        current_held["candidate_id"], current_ref["candidate_id"],
    ], ignore_index=True).astype(str).unique())
    labels = _label_lookup(args.prequential_labels, args.august_labels, candidate_ids=ids)
    bcf_held, bcf_ref = _with_labels(bcf_held, labels), _with_labels(bcf_ref, labels)
    current_held, current_ref = _with_labels(current_held, labels), _with_labels(current_ref, labels)
    bcf_bundle, current_bundle = BCFMC1D2Bundle.load(args.bcf_bundle), MC1D2Bundle.load(args.current_bundle)
    bcf_map = _map_bcf(bcf_held, _history(pd.concat([bcf_ref, bcf_held], ignore_index=True).drop_duplicates("candidate_id", keep="last")), bcf_bundle)
    current_map = _map_current(current_held, _history(pd.concat([current_ref, current_held], ignore_index=True).drop_duplicates("candidate_id", keep="last")), current_bundle)
    bcf = bcf_held.merge(bcf_map, on="candidate_id", how="inner", validate="one_to_one")
    current = current_held.merge(current_map, on="candidate_id", how="inner", validate="one_to_one")
    merged = bcf.merge(current.loc[:, ["candidate_id", "__symbol__", "current_v5_mc1_expected_bps", "current_v5_mc1_recent_shift_bps", "current_v5_mc1_available"]], on="candidate_id", how="inner", validate="one_to_one")
    selected = merged.loc[
        merged["bcf_mc1_available"] & merged["current_v5_mc1_available"]
        & merged["bcf_mc1_expected_bps"].ge(30.0) & merged["current_v5_mc1_expected_bps"].ge(30.0)
    ].copy()
    contract = Exact1mExecutionContract(entry_delay_minutes=5)
    request = pd.DataFrame({
        "candidate_id": selected["candidate_id"].astype(str), "timestamp": selected["__decision_ts__"],
        "symbol": selected["__symbol__"].astype(str), "side_name": selected["side_name"].astype(str),
        "entry_ts": selected["__decision_ts__"] + pd.Timedelta(minutes=5),
        "priority_bps": pd.to_numeric(selected["bcf_mc1_expected_bps"], errors="raise"),
    }).sort_values(["timestamp", "priority_bps", "candidate_id"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
    request_path = out / "candidate_download_request.parquet"
    request.to_parquet(request_path, index=False, compression="zstd")
    request_manifest = {
        "schema": "strict_r3_archived_dual_mc1_live_contract_request_v1", "target_free": True,
        "selection_inputs": ["bcf_mc1_expected_bps", "current_v5_mc1_expected_bps"],
        "selection_predicate": "bcf_mc1_expected_bps >= 30 AND current_v5_mc1_expected_bps >= 30",
        "forbidden_selection_inputs": ["policy_path_valid", "policy_net_bps", "outcome", "label"],
        "candidate_sha256": _sha(request_path), "contract_hash": contract.hash, "rows": int(len(request)),
    }
    (out / "candidate_download_request.json").write_text(json.dumps(request_manifest, indent=2, sort_keys=True) + "\n")
    audit_columns = ["candidate_id", "__decision_ts__", "__symbol__", "bcf_mc1_expected_bps", "current_v5_mc1_expected_bps", "bcf_mc1_recent_shift_bps", "current_v5_mc1_recent_shift_bps", "bcf_mc1_available", "current_v5_mc1_available"]
    merged.loc[:, audit_columns].to_parquet(out / "dual_mapping_audit.parquet", index=False, compression="zstd")
    dataset_audit = _filter_dataset(args.exact_source, request, out / "exact1m_dataset")
    manifest = {
        "schema": "strict_r3_archived_dual_mc1_live_contract_extension_v1", "research_only": True,
        "target_free_before_outcome_join": True, "strict_prior_resolved_labels": "policy_label_available_ts < decision_ts",
        "selection": request_manifest, "coverage": {"start": str(start), "end_exclusive": str(end), "bcf_rows": int(len(bcf)), "current_rows": int(len(current)), "matched_rows": int(len(merged)), "selected": int(len(request)), **dataset_audit},
        "inputs": {str(p): _sha(p) for p in [args.bcf_held,args.bcf_reference,args.current_held,args.current_reference,args.prequential_labels,args.august_labels,args.bcf_bundle/'run_manifest.json',args.current_bundle/'run_manifest.json']},
        "mapper_bundles": {"bcf": bcf_bundle.manifest["bundle_id"], "current": current_bundle.manifest["bundle_id"]},
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"out_dir": str(out), **manifest["coverage"]}, sort_keys=True))


if __name__ == "__main__":
    main()
