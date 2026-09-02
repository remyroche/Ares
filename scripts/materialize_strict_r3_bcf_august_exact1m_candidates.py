#!/usr/bin/env python3
"""Build a target-free August exact-1m request for the frozen live BCF route.

This is deliberately an offline, BCF-only replay utility.  It does not alter
the live trader.  The scorer is the frozen August BCF bundle; its MC1 mapper
is refit nowhere and receives only policy outcomes whose availability time is
strictly earlier than each decision.  The current stack contributes only the
live top-30% base routing gate.  Exact paths are requested after this output
has been sealed, so path availability cannot decide candidate membership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.exact_1m_policy_contract import Exact1mExecutionContract
from extreme_price_movements.strict_r3_bcf_mc1_mapper import (
    BCFMC1D2Bundle,
    derive_bcf_mc1_features,
)
from extreme_price_movements.strict_r3_mc1_mapper import _robust_mean, score_bands


DEFAULT_BCF_HELD = ROOT / "data_perp/artifacts/strict_r3_bcf_august_batch_scores_20260817_v1/predictions.parquet"
DEFAULT_BCF_REFERENCE = ROOT / "data_perp/artifacts/strict_r3_bcf_august_batch_scores_20260817_v1/same_model_prior42_reference_scores.parquet"
DEFAULT_CURRENT = ROOT / "data_perp/artifacts/strict_r3_bcf_current_dual_fullcycle_smoke_20260817T050000Z_v5/score/predictions.parquet"
DEFAULT_PREQUENTIAL = ROOT / "data_perp/artifacts/strict_r3_schema_v2_prequential_ledger_targetfree_long_2024_2026_raw15m_strictfull_20260812_v1/prequential_stack_ledger.parquet"
DEFAULT_AUGUST_LABELS = ROOT / "data_perp/artifacts/strict_r3_august_live_parity_policy_labels_20260816_v1/frozen_policy_labels.parquet"
DEFAULT_BUNDLE = ROOT / "data_perp/artifacts/strict_r3_bcf_mc1_d2_canonical_long_20260801_native_v1"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_bcf_august_exact1m_candidates_20260817_v1"
START = pd.Timestamp("2026-08-01T00:00:00Z")
END = pd.Timestamp("2026-08-18T00:00:00Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def _stamp(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _labels(path: Path, *, source: str) -> pd.DataFrame:
    columns = [
        "candidate_id", "policy_path_valid", "policy_net_bps",
        "policy_label_available_ts",
    ]
    frame = pd.read_parquet(path, columns=columns).copy()
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["policy_label_available_ts"] = _utc(frame["policy_label_available_ts"])
    if frame["candidate_id"].duplicated().any():
        raise ValueError(f"{source} policy labels have duplicate candidate IDs")
    return frame.assign(_policy_source=source)


def _read_bcf_scores(path: Path) -> pd.DataFrame:
    """Read only BCF score fields needed by the native MC1 contract.

    Full scorer receipts also carry hundreds of diagnostic fields.  Loading
    them for the 42-day reserve is needlessly memory-heavy and risks an OOM
    before the target-free request is written.
    """
    names = pq.read_schema(path).names
    rank_fields = sorted(
        field for field in names
        if field.startswith("residual_head__") and field.endswith("__rank")
    )
    if len(rank_fields) != 10:
        raise ValueError(f"BCF score receipt must contain 10 head ranks, found {len(rank_fields)}")
    columns = [
        "candidate_id", "__decision_ts__", "side_name",
        "final_score", "base_rank42", "upstream", "consensus_rank", *rank_fields,
    ]
    if "__symbol__" in names:
        columns.append("__symbol__")
    missing = sorted(set(columns).difference(names))
    if missing:
        raise ValueError(f"BCF score receipt misses required fields: {missing}")
    return pd.read_parquet(path, columns=columns)


def _with_native_and_labels(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    role: str,
) -> pd.DataFrame:
    scores = scores.copy()
    scores["candidate_id"] = scores["candidate_id"].astype(str)
    scores["__decision_ts__"] = _utc(scores["__decision_ts__"])
    derived_fields = [
        "conditional_consensus_rank", "ordinary_shadow_consensus_rank",
        "correctness_rank",
    ]
    # ``score_strict_r3_bcf_forward`` already emits these fields.  Older raw
    # scorer receipts do not, so derive them only in that case.  Avoiding a
    # second merge preserves one unambiguous lineage for every coordinate.
    if set(derived_fields).issubset(scores.columns):
        output = scores
    else:
        native = derive_bcf_mc1_features(scores)
        output = scores.merge(
            native.loc[:, ["candidate_id", *derived_fields]],
            on="candidate_id", how="inner", validate="one_to_one",
        )
    output = output.merge(
        labels.loc[:, [
            "candidate_id", "policy_path_valid", "policy_net_bps",
            "policy_label_available_ts", "_policy_source",
        ]],
        on="candidate_id", how="left", validate="one_to_one",
    )
    output["_score_role"] = role
    return output


def _top30(current: pd.DataFrame) -> pd.DataFrame:
    required = {"candidate_id", "__decision_ts__", "__symbol__", "base_score", "frozen_base_contract_complete"}
    missing = sorted(required.difference(current.columns))
    if missing:
        raise ValueError(f"current score panel misses route fields: {missing}")
    output = current.loc[:, list(required)].copy()
    output["candidate_id"] = output["candidate_id"].astype(str)
    output["__decision_ts__"] = _utc(output["__decision_ts__"])
    output["base_score"] = pd.to_numeric(output["base_score"], errors="coerce")
    output["frozen_base_contract_complete"] = output["frozen_base_contract_complete"].fillna(False).astype(bool)
    output = output.loc[output["base_score"].notna()].copy()
    output["_position"] = np.arange(len(output), dtype=np.int64)
    ordered = output.sort_values(
        ["__decision_ts__", "base_score", "candidate_id"],
        ascending=[True, False, True], kind="stable",
    ).copy()
    ordered["_rank"] = ordered.groupby("__decision_ts__", sort=False).cumcount() + 1
    ordered["_count"] = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    ordered["base_route_top30"] = ordered["_rank"].le(np.ceil(0.30 * ordered["_count"]))
    return ordered.sort_values("_position", kind="stable").drop(columns=["_position"])


def _map_prequential_bcf(
    held: pd.DataFrame,
    history: pd.DataFrame,
    controller: BCFMC1D2Bundle,
) -> pd.DataFrame:
    """Vector-light equivalent of one ``BCFMC1D2Bundle.score`` per hour.

    The library method defensively copies the complete history per decision,
    which is desirable for live one-hour scoring but makes an offline month
    replay allocate hundreds of large frames.  This preserves its exact
    causal calculation while retaining the compact history as arrays.
    """
    features = list(controller.manifest["features_ordered"])
    valid = history.loc[
        history["policy_path_valid"].fillna(False).astype(bool)
        & pd.to_numeric(history["policy_net_bps"], errors="coerce").notna()
    ].copy()
    valid["score_band"] = score_bands(valid)
    curve = np.asarray(controller.payload["structural_curve_bps"], dtype=float)
    residual = (
        pd.to_numeric(valid["policy_net_bps"], errors="coerce").to_numpy(float)
        - curve[valid["score_band"].to_numpy(int)]
    )
    label_ns = _utc(valid["policy_label_available_ts"]).astype("int64").to_numpy()
    decision_ns = _utc(valid["__decision_ts__"]).astype("int64").to_numpy()
    window_ns = int(pd.Timedelta(days=21).value)
    parts: list[pd.DataFrame] = []
    for decision, group in held.groupby("__decision_ts__", sort=True):
        stamp_ns = pd.Timestamp(decision).value
        values = residual[(label_ns < stamp_ns) & (decision_ns >= stamp_ns - window_ns)]
        shift = _robust_mean(values, trim=0.10)
        matrix = group.loc[:, features].apply(pd.to_numeric, errors="coerce")
        available = np.isfinite(matrix.to_numpy(float)).all(axis=1)
        prediction = np.full(len(group), np.nan, dtype=float)
        if available.any() and np.isfinite(shift):
            prediction[available] = np.asarray(
                controller.payload["model"].predict(matrix.loc[available, features]), dtype=float,
            ) + float(shift)
        parts.append(pd.DataFrame({
            "candidate_id": group["candidate_id"].astype(str).to_numpy(),
            "__decision_ts__": decision,
            "bcf_mc1_expected_net_bps": prediction,
            "bcf_mc1_recent_global_shift_bps": float(shift),
            "bcf_mc1_available": available,
        }))
    result = pd.concat(parts, ignore_index=True)
    result["bcf_mc1_admitted_ge_30bps"] = (
        result["bcf_mc1_available"]
        & np.isfinite(result["bcf_mc1_expected_net_bps"])
        & result["bcf_mc1_expected_net_bps"].ge(30.0)
    )
    result["bcf_mc1_bundle_id"] = str(controller.manifest["bundle_id"])
    return result


def materialize(args: argparse.Namespace) -> Path:
    out = Path(args.out_dir).resolve()
    if out.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {out}")
    out.mkdir(parents=True)
    held_path = Path(args.bcf_held).resolve()
    reference_path = Path(args.bcf_reference).resolve()
    current_path = Path(args.current_scores).resolve()
    prequential_path = Path(args.prequential_labels).resolve()
    august_path = Path(args.august_labels).resolve()
    bundle_path = Path(args.bcf_mc1_bundle).resolve()

    prequential_labels = _labels(prequential_path, source="pre_august_prequential")
    august_labels = _labels(august_path, source="august_parent_policy")
    labels = pd.concat([prequential_labels, august_labels], ignore_index=True)
    labels = labels.drop_duplicates("candidate_id", keep="last")

    print("loading compact BCF score receipts", flush=True)
    reference = _with_native_and_labels(
        _read_bcf_scores(reference_path), labels, role="same_model_prior42_reference",
    )
    held = _with_native_and_labels(
        _read_bcf_scores(held_path), labels, role="august_held",
    )
    start, end = _stamp(args.start), _stamp(args.end)
    held = held.loc[
        held["__decision_ts__"].ge(start)
        & held["__decision_ts__"].lt(end)
    ].copy()
    if held.empty:
        raise ValueError("no BCF held scores in requested August interval")

    # The map only uses these fields.  Keeping a compact history prevents a
    # large raw-score panel from being copied once per decision timestamp.
    history_columns = [
        "candidate_id", "__decision_ts__", "policy_label_available_ts",
        "policy_path_valid", "policy_net_bps", "final_score",
    ]
    history = pd.concat([
        reference.loc[:, history_columns], held.loc[:, history_columns],
    ], ignore_index=True)
    controller = BCFMC1D2Bundle.load(bundle_path)
    print(f"mapping {held['__decision_ts__'].nunique()} BCF decision timestamps", flush=True)
    mapped = _map_prequential_bcf(held, history, controller)
    held = held.merge(mapped, on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")

    current = _top30(pd.read_parquet(current_path, columns=[
        "candidate_id", "__decision_ts__", "__symbol__", "base_score", "frozen_base_contract_complete",
    ]))
    current = current.loc[
        current["__decision_ts__"].ge(start)
        & current["__decision_ts__"].lt(end)
    ].copy()
    merged = held.merge(
        current.loc[:, [
            "candidate_id", "__decision_ts__", "__symbol__",
            "frozen_base_contract_complete", "base_route_top30",
        ]],
        on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("no identity overlap between BCF held scores and current route")
    admitted = (
        merged["frozen_base_contract_complete"].fillna(False).astype(bool)
        & merged["base_route_top30"].fillna(False).astype(bool)
        & merged["bcf_mc1_available"].fillna(False).astype(bool)
        & pd.to_numeric(merged["bcf_mc1_expected_net_bps"], errors="coerce").ge(30.0)
    )
    selected = merged.loc[admitted].copy()
    if selected.empty:
        raise ValueError("BCF +30 admission selected no August candidates")
    contract = Exact1mExecutionContract(entry_delay_minutes=5)
    contract.validate()
    request = pd.DataFrame({
        "candidate_id": selected["candidate_id"].astype(str),
        "timestamp": selected["__decision_ts__"],
        "symbol": selected["__symbol__"].astype(str),
        "side_name": selected["side_name"].astype(str),
        "entry_ts": selected["__decision_ts__"] + pd.Timedelta(minutes=5),
        "priority_bps": pd.to_numeric(selected["bcf_mc1_expected_net_bps"], errors="raise"),
    }).sort_values(["timestamp", "priority_bps", "candidate_id"], ascending=[True, False, True], kind="stable").reset_index(drop=True)
    request_path = out / "candidate_download_request.parquet"
    request.to_parquet(request_path, index=False, compression="zstd")
    request_sha = _sha256(request_path)
    request_manifest = {
        "schema": "strict_r3_exact_1m_bcf30_august_download_request_v1",
        "target_free": True,
        "selection_inputs": [
            "current_timestamp_local_base_top30", "bcf_mc1_expected_net_bps",
        ],
        "selection_predicate": (
            "current_base_top30 AND BCF_MC1_expected_net_bps>=30; "
            "BCF MC1 uses only prior-resolved parent-policy outcomes"
        ),
        "forbidden_selection_inputs": ["policy_path_valid", "policy_net_bps", "outcome", "label"],
        "candidate_sha256": request_sha,
        "contract_hash": contract.hash,
        "rows": int(len(request)),
        "entry_delay_minutes": 5,
        "horizon_minutes": 720,
        "score_column": "priority_bps",
        "auction_priority": "priority_bps = BCF_MC1_expected_net_bps",
        "base_route": "timestamp-local current base top 30%; candidate_id ascending tie break",
        "bcf_bundle_id": controller.manifest["bundle_id"],
    }
    (out / "candidate_download_request.json").write_text(json.dumps(request_manifest, indent=2, sort_keys=True) + "\n")

    mapping_fields = [
        "candidate_id", "__decision_ts__", "__symbol__", "final_score", "base_rank42",
        "conditional_consensus_rank", "upstream", "ordinary_shadow_consensus_rank",
        "correctness_rank", "bcf_mc1_expected_net_bps", "bcf_mc1_recent_global_shift_bps",
        "bcf_mc1_available", "frozen_base_contract_complete", "base_route_top30",
    ]
    merged.loc[:, mapping_fields].to_parquet(out / "bcf_scoring_and_mapping_audit.parquet", index=False, compression="zstd")
    summary = {
        "schema": "strict_r3_bcf_august_exact1m_candidate_materialisation_v1",
        "start": str(args.start), "end_exclusive": str(args.end),
        "target_free_candidate_rows": int(len(request)),
        "bcf_held_rows": int(len(held)),
        "current_route_rows": int(len(current)),
        "matched_rows": int(len(merged)),
        "base_routed_rows": int((merged["frozen_base_contract_complete"] & merged["base_route_top30"]).sum()),
        "bcf_admitted_rows": int(len(request)),
        "mapping_history": {
            "same_model_reference_rows": int(len(reference)),
            "august_label_rows_available": int(august_labels["policy_path_valid"].fillna(False).sum()),
            "strictly_prior_resolution_filter": True,
        },
        "inputs": {str(path): _sha256(path) for path in [
            held_path, reference_path, current_path, prequential_path, august_path,
            bundle_path / "run_manifest.json", bundle_path / "bcf_mc1_d2.joblib",
        ]},
        "request": request_manifest,
    }
    (out / "run_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", **summary}))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bcf-held", type=Path, default=DEFAULT_BCF_HELD)
    parser.add_argument("--bcf-reference", type=Path, default=DEFAULT_BCF_REFERENCE)
    parser.add_argument("--current-scores", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--prequential-labels", type=Path, default=DEFAULT_PREQUENTIAL)
    parser.add_argument("--august-labels", type=Path, default=DEFAULT_AUGUST_LABELS)
    parser.add_argument("--bcf-mc1-bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--start", default=str(START))
    parser.add_argument("--end", default=str(END))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    materialize(parser.parse_args())


if __name__ == "__main__":
    main()
