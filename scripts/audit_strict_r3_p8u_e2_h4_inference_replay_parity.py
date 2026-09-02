#!/usr/bin/env python3
"""Independently replay the sealed E2/H4 authorities on target-free inputs.

The audit deliberately reimplements E2's reserve/marginal pairing rather than
calling the inference selector twice.  It loads only the sealed feature fields
and identities from source Parquet panels: no policy result, path label, or
outcome column is read.  H4's direct model prediction is similarly compared
with the inference authority on the same completed-state feature rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_e2_h4_live_parity import (
    CORE_FLOOR_BPS,
    MAX_PER_TIMESTAMP,
    RESERVE_FLOOR_BPS,
    P8UE2H4LiveParityBundle,
    apply_e2_replacement,
    apply_h4_next_interval,
)


ENTRY_OLD = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_ordinal_mc1_threshold_observed25h_20260830_v4_manifested_results/target_free_15m_features.parquet"
ENTRY_VWAP = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_vwap_target_free_20260830_v1/target_free_15m_features.parquet"
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _names(path: Path) -> set[str]:
    return set(pq.ParquetFile(path).schema_arrow.names)


def _read(path: Path, columns: set[str]) -> pd.DataFrame:
    available = _names(path)
    requested = sorted(columns.intersection(available))
    if not requested:
        raise ValueError(f"no requested inference fields found in {path}")
    return pd.read_parquet(path, columns=requested)


def _entry_frame(bundle: P8UE2H4LiveParityBundle, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    fields = bundle.e2_features
    raw = {field.removeprefix("margin__") for field in fields if field != "incumbent_bcf_mc1_expected_bps"}
    ids = {"candidate_id", "__decision_ts__", "__symbol__", "bcf_final_score", "bcf_mc1_expected_bps", "current_mc1_expected_bps", "dual_mc1_min_bps", "feature_source_status"}
    old = _read(ENTRY_OLD, ids | raw)
    vwap = _read(ENTRY_VWAP, {"candidate_id", *raw})
    old["candidate_id"] = old["candidate_id"].astype(str)
    vwap["candidate_id"] = vwap["candidate_id"].astype(str)
    duplicate = sorted(set(old.columns).intersection(vwap.columns).difference({"candidate_id"}))
    if duplicate:
        vwap = vwap.drop(columns=duplicate)
    result = old.merge(vwap, on="candidate_id", how="inner", validate="one_to_one")
    result["__decision_ts__"] = pd.to_datetime(result["__decision_ts__"], utc=True, errors="raise")
    result = result.loc[result["__decision_ts__"].ge(start) & result["__decision_ts__"].lt(end)].copy()
    if "feature_source_status" not in result:
        result["feature_source_status"] = "complete"
    if result.empty:
        raise ValueError("chosen parity range has no target-free E2 rows")
    missing = sorted(raw.difference(result.columns))
    if missing:
        raise ValueError(f"entry target-free panel lacks sealed fields: {missing}")
    return result, {"entry_old": sorted(set(old.columns)), "entry_vwap": sorted(set(vwap.columns))}


def _contract_source(bundle: P8UE2H4LiveParityBundle, role: str) -> Path:
    """Return a hash-bound action-aligned H4 source declared by the bundle."""
    descriptor = bundle.manifest["continuation_training_contract"][role]
    path = Path(str(descriptor["path"]))
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file() or _sha256(path) != descriptor["sha256"]:
        raise ValueError(f"H4 {role} source receipt does not match the sealed bundle")
    return path


def _h4_frame(bundle: P8UE2H4LiveParityBundle, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    fields = set(bundle.h4_features)
    keys = {"candidate_id", "state_decision_ts", "state_bar_15m", "entry_decision_ts"}
    states_path = _contract_source(bundle, "states")
    route_path = _contract_source(bundle, "route")
    states = _read(states_path, keys | fields)
    route = _read(route_path, {"candidate_id", "bcf_mc1_expected_bps"})
    states["candidate_id"] = states["candidate_id"].astype(str)
    states["state_decision_ts"] = pd.to_datetime(states["state_decision_ts"], utc=True, errors="raise")
    route["candidate_id"] = route["candidate_id"].astype(str)
    if states.duplicated(["candidate_id", "state_decision_ts"]).any() or route["candidate_id"].duplicated().any():
        raise ValueError("sealed action-aligned H4 sources duplicate an identity")
    result = states.drop(columns=["MC1_expected_bps"], errors="ignore").merge(
        route,
        on="candidate_id",
        how="inner",
        validate="many_to_one",
    ).rename(columns={"bcf_mc1_expected_bps": "MC1_expected_bps"})
    result = result.loc[result["state_decision_ts"].ge(start) & result["state_decision_ts"].lt(end)].copy()
    if result.empty:
        raise ValueError("chosen parity range has no completed H4 state rows")
    missing = sorted(fields.difference(result.columns))
    if missing:
        raise ValueError(f"H4 target-free state panel lacks sealed fields: {missing}")
    return result, {"h4_states": sorted(set(states.columns)), "h4_route": sorted(set(route.columns))}


def _independent_e2(frame: pd.DataFrame, *, bundle: P8UE2H4LiveParityBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Independent replay of the fixed E2 selection semantics."""
    fields = bundle.e2_features
    selected: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for stamp, group in frame.groupby("__decision_ts__", sort=True):
        group = group.copy()
        dual = pd.to_numeric(group["dual_mc1_min_bps"], errors="coerce")
        core = group.loc[dual.ge(CORE_FLOOR_BPS)].sort_values(
            ["bcf_mc1_expected_bps", "bcf_final_score", "candidate_id"], ascending=[False, False, True], kind="stable"
        )
        initial = core.head(MAX_PER_TIMESTAMP)
        selected.extend({"candidate_id": str(row.candidate_id), "e2_replay_action": "ordinary_bcf_top2"} for _, row in initial.iterrows())
        if core.empty:
            continue
        incumbent = core.iloc[min(len(core), MAX_PER_TIMESTAMP) - 1]
        reserves = group.loc[dual.ge(RESERVE_FLOOR_BPS) & dual.lt(CORE_FLOOR_BPS)]
        for _, reserve in reserves.iterrows():
            if str(reserve.get("feature_source_status", "complete")) not in {"ok", "complete"}:
                continue
            if str(incumbent.get("feature_source_status", "complete")) not in {"ok", "complete"}:
                continue
            row: dict[str, Any] = {
                "__decision_ts__": stamp,
                "reserve_candidate_id": str(reserve.candidate_id),
                "incumbent_candidate_id": str(incumbent.candidate_id),
                "reserve_bcf_mc1_expected_bps": float(reserve.bcf_mc1_expected_bps),
            }
            for field in fields:
                if field == "incumbent_bcf_mc1_expected_bps":
                    row[field] = float(incumbent.bcf_mc1_expected_bps)
                elif field.startswith("margin__"):
                    source = field.removeprefix("margin__")
                    row[field] = float(reserve[source]) - float(incumbent[source])
                else:
                    row[field] = reserve[field]
            pair_rows.append(row)
    pairs = pd.DataFrame(pair_rows)
    if pairs.empty:
        return pd.DataFrame(selected), pairs
    h0, h3 = bundle.e2_models()
    pairs["h0_q50_pair_advantage_bps"] = h0.predict(pairs.loc[:, fields])
    pairs["h3_q50_pair_advantage_bps"] = h3.predict(pairs.loc[:, fields])
    qualified = np.isfinite(pairs["h0_q50_pair_advantage_bps"]) & np.isfinite(pairs["h3_q50_pair_advantage_bps"])
    qualified &= pairs["h0_q50_pair_advantage_bps"].ge(50.0) & pairs["h3_q50_pair_advantage_bps"].ge(50.0)
    for _, proposal in pairs.loc[qualified].groupby("__decision_ts__", sort=True):
        winner = proposal.sort_values(
            ["h0_q50_pair_advantage_bps", "h3_q50_pair_advantage_bps", "reserve_bcf_mc1_expected_bps", "reserve_candidate_id"],
            ascending=[False, False, False, True], kind="stable",
        ).iloc[0]
        selected = [row for row in selected if row["candidate_id"] != str(winner.incumbent_candidate_id)]
        selected.append({"candidate_id": str(winner.reserve_candidate_id), "e2_replay_action": "e2_q50_agreement_replacement"})
    return pd.DataFrame(selected), pairs


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) != len(right):
        raise AssertionError("parity outputs have different row counts")
    return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))) if len(left) else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--start", default="2026-08-20T00:00:00Z")
    parser.add_argument("--end", default="2026-08-21T00:00:00Z")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("parity receipt output must be immutable")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    if end <= start:
        raise ValueError("parity end must be after start")
    bundle = P8UE2H4LiveParityBundle.load(args.bundle.resolve())
    entry, entry_loaded = _entry_frame(bundle, start, end)
    inferred, inference_pairs = apply_e2_replacement(entry, bundle=bundle)
    replay_selected, replay_pairs = _independent_e2(entry, bundle=bundle)
    inferred_selected = inferred.loc[inferred.e2_entry_selected, ["candidate_id", "e2_action"]].rename(columns={"e2_action": "e2_inference_action"})
    compare = inferred_selected.merge(replay_selected, on="candidate_id", how="outer", indicator=True, validate="one_to_one")
    if not compare["_merge"].eq("both").all() or not compare["e2_inference_action"].eq(compare["e2_replay_action"]).all():
        raise AssertionError("independent E2 replay does not match inference selection")
    pair_keys = ["reserve_candidate_id", "incumbent_candidate_id"]
    inferred_pairs = inference_pairs.loc[:, [*pair_keys, "h0_q50_pair_advantage_bps", "h3_q50_pair_advantage_bps"]]
    replay_scores = replay_pairs.loc[:, [*pair_keys, "h0_q50_pair_advantage_bps", "h3_q50_pair_advantage_bps"]]
    pair_compare = inferred_pairs.merge(replay_scores, on=pair_keys, suffixes=("_inference", "_replay"), how="outer", indicator=True, validate="one_to_one")
    if not pair_compare["_merge"].eq("both").all():
        raise AssertionError("independent E2 pair population does not match inference")
    h0_delta = _max_abs(pair_compare["h0_q50_pair_advantage_bps_inference"], pair_compare["h0_q50_pair_advantage_bps_replay"])
    h3_delta = _max_abs(pair_compare["h3_q50_pair_advantage_bps_inference"], pair_compare["h3_q50_pair_advantage_bps_replay"])
    if h0_delta != 0.0 or h3_delta != 0.0:
        raise AssertionError("independent E2 model prediction delta is nonzero")

    h4_state, h4_loaded = _h4_frame(bundle, start, end)
    h4_inferred = apply_h4_next_interval(h4_state, bundle=bundle)
    direct = np.asarray(bundle.h4_model().predict(h4_state.loc[:, bundle.h4_features]), dtype=float)
    h4_delta = _max_abs(h4_inferred["h4_activation50_advantage_bps"], direct)
    direct_active = direct >= 0.0
    if h4_delta != 0.0 or not np.array_equal(h4_inferred["h4_active"].to_numpy(bool), direct_active):
        raise AssertionError("H4 inference authority does not match direct replay")

    output.mkdir(parents=True, exist_ok=False)
    compare.to_parquet(output / "e2_selection_parity.parquet", index=False, compression="zstd")
    pair_compare.to_parquet(output / "e2_pair_prediction_parity.parquet", index=False, compression="zstd")
    h4_inferred.loc[:, ["candidate_id", "state_decision_ts", "state_bar_15m", "h4_activation50_advantage_bps", "h4_active"]].to_parquet(
        output / "h4_prediction_parity.parquet", index=False, compression="zstd"
    )
    receipt = {
        "schema": "strict_r3_p8u_e2_h4_inference_replay_parity_v1",
        "status": "pass_exact_target_free_inference_replay_parity",
        "order_submission": False,
        "bundle": str(args.bundle.resolve()),
        "bundle_manifest_sha256": bundle.manifest_sha256,
        "range": {"start": start.isoformat(), "end_exclusive": end.isoformat()},
        "entry_rows": int(len(entry)),
        "e2_inference_selected": int(len(inferred_selected)),
        "e2_pair_rows": int(len(pair_compare)),
        "e2_h0_max_abs_delta": h0_delta,
        "e2_h3_max_abs_delta": h3_delta,
        "h4_state_rows": int(len(h4_state)),
        "h4_prediction_max_abs_delta": h4_delta,
        "h4_active_rows": int(h4_inferred["h4_active"].sum()),
        "source_hashes": {
            str(ENTRY_OLD.relative_to(ROOT)): _sha256(ENTRY_OLD),
            str(ENTRY_VWAP.relative_to(ROOT)): _sha256(ENTRY_VWAP),
            str(_contract_source(bundle, "states").relative_to(ROOT)): _sha256(_contract_source(bundle, "states")),
            str(_contract_source(bundle, "route").relative_to(ROOT)): _sha256(_contract_source(bundle, "route")),
        },
        "loaded_target_free_columns": {**entry_loaded, **h4_loaded},
        "outcome_columns_consumed": [],
        "exchange_or_order_submission_called": False,
        "parity_tolerance": 0.0,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
