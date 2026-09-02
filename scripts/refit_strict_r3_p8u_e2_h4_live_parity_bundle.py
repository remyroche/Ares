#!/usr/bin/env python3
"""Materialise one current, no-order E2/H4 live-parity model bundle.

This is intentionally a bundle producer, not a scorer and not an execution
entrypoint.  It trains from target-free panels after joining labels only for
rows whose labels resolved before ``--cutoff``.  The current exchange-writing
gateway cannot consume this output until a successor adapter, scorer, monitor,
and parity receipt have been separately sealed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.inference.p8u_e2_h4_live_parity import (
    H4_GIVEBACK20_AUTHORITY,
    H4_GIVEBACK20_TARGET,
    SCHEMA,
    _canonical_json_hash,
    _sha256,
)
from extreme_price_movements.p8u_e2_h4_giveback20_contract import DEFAULT_CONTRACT, P8UE2H4Giveback20Contract
from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS, VWAP_15M_FEATURE_KEYS
from scripts import run_strict_r3_p8u_15m_entry_feature_contract_ablation as entry_study
from scripts import run_strict_r3_p8u_15m_entry_postfs_hpo as entry_hpo
from scripts import run_strict_r3_p8u_15m_entry_pairwise_replacement_ablation as entry_base
from scripts import run_strict_r3_p8u_15m_continuation_feature_contract_ablation as h4_study
from scripts import run_strict_r3_p8u_15m_continuation_postfs_hpo as h4_hpo


ENTRY_FEATURE_STUDY = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_feature_contract_20260830_v2"
H4_FEATURE_STUDY = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_continuation_feature_contract_20260830_v2"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_e2_h4_live_parity_bundle_20260830_v1"


def _utc(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _ordered_features(path: Path, *, arm_col: str, arm: str, held_month: str, count: int) -> tuple[str, ...]:
    selection = pd.read_parquet(path)
    required = {arm_col, "held_month", "feature", "position"}
    missing = sorted(required.difference(selection.columns))
    if missing:
        raise ValueError(f"feature selection receipt lacks {missing}")
    rows = selection.loc[
        selection[arm_col].eq(arm) & selection["held_month"].eq(held_month)
    ].sort_values("position", kind="stable")
    fields = tuple(rows["feature"].astype(str))
    if len(fields) != count or len(set(fields)) != count:
        raise ValueError(f"{arm} does not supply exactly {count} distinct ordered features for {held_month}")
    return fields


def _write_order(path: Path, fields: tuple[str, ...]) -> None:
    payload = {"features": list(fields), "sha256": _canonical_json_hash(list(fields))}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _files(output: Path) -> dict[str, dict[str, str]]:
    mapping = {
        "e2_h0_model": "e2_h0_q50.joblib",
        "e2_h3_model": "e2_h3_q50.joblib",
        "h4_model": "h4_l1_mean.joblib",
        "e2_feature_order": "e2_feature_order.json",
        "h4_feature_order": "h4_feature_order.json",
    }
    return {role: {"name": name, "sha256": _sha256(output / name)} for role, name in mapping.items()}


def _load_entry_labels_fail_closed(root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Read valid immutable label partitions and explicitly exclude bad ones.

    This release has no authority to hydrate, reconstruct, or impute a broken
    outcome partition.  The exact omitted symbols are persisted in the bundle
    manifest so downstream evaluation can preserve the same support boundary.
    """
    columns = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
        "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    ]
    valid: list[pd.DataFrame] = []
    invalid: list[str] = []
    for path in sorted(root.resolve().glob("policy_parts/symbol=*/policy_labels.parquet")):
        try:
            pq.ParquetFile(path)
            valid.append(pd.read_parquet(path, columns=columns))
        except Exception:
            invalid.append(str(path.relative_to(ROOT)))
    if not valid:
        raise RuntimeError("no valid policy-label partitions are available")
    frame = pd.concat(valid, ignore_index=True)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("valid label partitions duplicate candidate identity")
    good = frame["policy_path_valid"].fillna(False).astype(bool)
    gross = pd.to_numeric(frame.loc[good, "policy_gross_bps"], errors="coerce")
    net = pd.to_numeric(frame.loc[good, "policy_net_bps"], errors="coerce")
    if not np.isclose(gross - net, 100.0, rtol=0.0, atol=1e-8).all():
        raise AssertionError("rich policy cost is not exactly 100 bps once")
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="raise")
    return frame, invalid


def _action_aligned_h4_panel(
    *,
    states_path: Path,
    labels_path: Path,
    route_path: Path,
) -> pd.DataFrame:
    """Join the exact Giveback-20 state label to its target-free state row.

    This is deliberately separate from the historical activation-only H4
    study panel.  The runtime action changes both activation and giveback, so
    its learner must consume the exact matching counterfactual label.  The
    state identity and label availability are explicit and duplicated rows are
    rejected rather than silently collapsed.
    """
    states = pd.read_parquet(states_path).copy()
    labels = pd.read_parquet(labels_path).copy()
    route = pd.read_parquet(route_path).copy()
    keys = ["candidate_id", "state_decision_ts"]
    required_states = {*keys, "entry_decision_ts", "MC1_expected_bps"}
    required_labels = {*keys, "activation50_advantage_bps", "policy_label_available_ts"}
    required_route = {"candidate_id", "bcf_mc1_expected_bps"}
    missing_states = sorted(required_states.difference(states.columns))
    missing_labels = sorted(required_labels.difference(labels.columns))
    missing_route = sorted(required_route.difference(route.columns))
    if missing_states:
        raise ValueError(f"action-aligned H4 state panel lacks {missing_states}")
    if missing_labels:
        raise ValueError(f"action-aligned H4 labels lack {missing_labels}")
    if missing_route:
        raise ValueError(f"action-aligned H4 route lacks {missing_route}")
    for frame in (states, labels):
        frame["candidate_id"] = frame["candidate_id"].astype(str)
        frame["state_decision_ts"] = pd.to_datetime(frame["state_decision_ts"], utc=True, errors="raise")
    if states.duplicated(keys).any() or labels.duplicated(keys).any():
        raise ValueError("action-aligned H4 inputs duplicate a state identity")
    route["candidate_id"] = route["candidate_id"].astype(str)
    if route["candidate_id"].duplicated().any():
        raise ValueError("action-aligned H4 route duplicates candidate identity")
    labels = labels.loc[:, [*keys, "activation50_advantage_bps", "policy_label_available_ts"]].rename(
        columns={"activation50_advantage_bps": H4_GIVEBACK20_TARGET}
    )
    result = states.merge(labels, on=keys, how="inner", validate="one_to_one")
    # The exact path materialiser intentionally stores MC1 as a neutral zero:
    # MC1 must not influence an exit counterfactual.  Training eligibility and
    # the MC1 state feature instead come from the immutable target-free route
    # that existed at the entry decision.  Without this join every state would
    # be spuriously excluded by the >=30-bps support gate.
    result = result.drop(columns=["MC1_expected_bps"], errors="ignore").merge(
        route.loc[:, ["candidate_id", "bcf_mc1_expected_bps"]],
        on="candidate_id",
        how="inner",
        validate="many_to_one",
    ).rename(columns={"bcf_mc1_expected_bps": "MC1_expected_bps"})
    result["entry_decision_ts"] = pd.to_datetime(result["entry_decision_ts"], utc=True, errors="raise")
    result["policy_label_available_ts"] = pd.to_datetime(result["policy_label_available_ts"], utc=True, errors="raise")
    if result.empty:
        raise RuntimeError("action-aligned H4 join produced no labelled states")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cutoff", default="2026-08-29T00:00:00Z", help="exclusive resolved-label cutoff, UTC")
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--feature-held-month", default="2026-08", help="frozen selection receipt month")
    parser.add_argument("--research-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--h4-state-panel",
        type=Path,
        required=True,
        help="target-free completed-state panel for the exact activation50+Giveback-20 action",
    )
    parser.add_argument(
        "--h4-label-panel",
        type=Path,
        required=True,
        help="hash-bound exact activation50+Giveback-20 counterfactual label panel",
    )
    parser.add_argument(
        "--h4-route-panel",
        type=Path,
        required=True,
        help="immutable target-free MC1 route used to restore decision-time H4 eligibility and MC1 state",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"live-parity bundle output must be immutable: {output}")
    if args.train_months < 2:
        raise ValueError("live-parity refit requires at least two trailing calendar months")
    cutoff = _utc(args.cutoff)
    start = cutoff - pd.DateOffset(months=int(args.train_months))
    canonical = P8UE2H4Giveback20Contract.load(args.research_contract.resolve(), workspace_root=ROOT)

    entry_selection = (ENTRY_FEATURE_STUDY / "stable_selected_features.parquet").resolve()
    h4_selection = (H4_FEATURE_STUDY / "stable_selected_features.parquet").resolve()
    e2_features = _ordered_features(
        entry_selection, arm_col="variant", arm="E3_vwap_fs",
        held_month=args.feature_held_month, count=30,
    )
    h4_features = _ordered_features(
        h4_selection, arm_col="arm", arm="C4_normalized_vwap_fs",
        held_month=args.feature_held_month, count=45,
    )

    # E2: source feature values are materialised before policy labels are
    # joined.  Pairs then retain only labels available strictly before cutoff.
    entry_panel = entry_study._candidate_frame(entry_study._load_panel(entry_study.OLD_PANEL, entry_study.VWAP_PANEL))
    labels, invalid_label_partitions = _load_entry_labels_fail_closed(entry_study.LABEL_ROOT)
    labelled = entry_panel.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    labelled = labelled.loc[labelled.policy_path_valid.fillna(False)].copy()
    labelled["policy_label_available_ts"] = pd.to_datetime(labelled["policy_label_available_ts"], utc=True, errors="raise")
    train_rows = labelled.loc[
        labelled["__decision_ts__"].ge(start)
        & labelled["__decision_ts__"].lt(cutoff)
        & labelled["policy_label_available_ts"].lt(cutoff)
    ].copy()
    raw = (*FIFTEEN_MINUTE_FEATURE_KEYS, *entry_study.SCORE_FEATURES, *VWAP_15M_FEATURE_KEYS)
    entry_pairs = entry_study._pairs(train_rows, raw, require_labels=True)
    entry_pairs = entry_pairs.loc[pd.to_datetime(entry_pairs["pair_label_available_ts"], utc=True, errors="raise").lt(cutoff)].copy()
    if len(entry_pairs) < 100:
        raise RuntimeError("insufficient prior-resolved E2 pair support")
    missing_e2 = sorted(set(e2_features).difference(entry_pairs.columns))
    if missing_e2:
        raise ValueError(f"E2 feature contract cannot be materialised from training pair panel: {missing_e2}")
    h0 = entry_hpo._fit(entry_pairs, e2_features, entry_hpo.SPECS["H0_q50_d3_l7_baseline"])
    h3 = entry_hpo._fit(entry_pairs, e2_features, entry_hpo.SPECS["H3_q50_d2_l3_strict"])

    # H4: the runtime modifies both trailing activation and giveback.  It must
    # never be trained on the old activation-only label panel.
    h4_state_panel = args.h4_state_panel.resolve()
    h4_label_panel = args.h4_label_panel.resolve()
    h4_route_panel = args.h4_route_panel.resolve()
    h4_panel = _action_aligned_h4_panel(
        states_path=h4_state_panel,
        labels_path=h4_label_panel,
        route_path=h4_route_panel,
    )
    h4_train = h4_panel.loc[
        pd.to_numeric(h4_panel["MC1_expected_bps"], errors="coerce").ge(30.0)
        & h4_panel["entry_decision_ts"].ge(start)
        & h4_panel["entry_decision_ts"].lt(cutoff)
        & h4_panel["policy_label_available_ts"].lt(cutoff)
        & pd.to_numeric(h4_panel[H4_GIVEBACK20_TARGET], errors="coerce").notna()
    ].copy()
    if len(h4_train) < 100:
        raise RuntimeError("insufficient prior-resolved H4 state support")
    missing_h4 = sorted(set(h4_features).difference(h4_train.columns))
    if missing_h4:
        raise ValueError(f"H4 feature contract cannot be materialised from state panel: {missing_h4}")
    h4_fit = h4_hpo.SPECS["H4_l1_d4_l15_leaf5_reg20"]
    h4 = h4_hpo._fit(
        h4_train.rename(columns={H4_GIVEBACK20_TARGET: "activation50_advantage_bps"}),
        h4_features,
        h4_fit,
    )

    output.mkdir(parents=True, exist_ok=False)
    joblib.dump(h0, output / "e2_h0_q50.joblib")
    joblib.dump(h3, output / "e2_h3_q50.joblib")
    joblib.dump(h4, output / "h4_l1_mean.joblib")
    _write_order(output / "e2_feature_order.json", e2_features)
    _write_order(output / "h4_feature_order.json", h4_features)
    manifest = {
        "schema": SCHEMA,
        "status": "SEALED_NO_ORDER_LIVE_PARITY_CANDIDATE",
        "order_submission": False,
        "side": "long",
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "research_contract": {
            "schema": canonical.payload["schema"],
            "path": str(canonical.path.relative_to(ROOT)),
            "sha256": canonical.sha256,
        },
        "cutoff": cutoff.isoformat(),
        "training_window": {"start": start.isoformat(), "end_exclusive": cutoff.isoformat(), "months": int(args.train_months)},
        "features": {
            "e2": {"selection_path": str(entry_selection.relative_to(ROOT)), "selection_sha256": _sha256(entry_selection), "held_month": args.feature_held_month, "fields": len(e2_features)},
            "h4": {"selection_path": str(h4_selection.relative_to(ROOT)), "selection_sha256": _sha256(h4_selection), "held_month": args.feature_held_month, "fields": len(h4_features)},
        },
        "training": {
            "e2_pairs": int(len(entry_pairs)),
            "e2_pair_labels_resolved_before_cutoff": True,
            "h4_states": int(len(h4_train)),
            "h4_labels_resolved_before_cutoff": True,
            "e2_h0": entry_hpo.SPECS["H0_q50_d3_l7_baseline"],
            "e2_h3": entry_hpo.SPECS["H3_q50_d2_l3_strict"],
            "h4": h4_fit,
        },
        "source_hashes": {
            "entry_target_free": _sha256(entry_study.OLD_PANEL),
            "entry_target_free_vwap": _sha256(entry_study.VWAP_PANEL),
            "h4_target_free_states": _sha256(h4_state_panel),
            "h4_action_aligned_labels": _sha256(h4_label_panel),
            "h4_target_free_route": _sha256(h4_route_panel),
        },
        "label_partition_audit": {
            "label_root": str(entry_study.LABEL_ROOT.relative_to(ROOT)),
            "invalid_partitions_excluded_fail_closed": invalid_label_partitions,
            "valid_partitions_used": int(len(list(entry_study.LABEL_ROOT.glob("policy_parts/symbol=*/policy_labels.parquet"))) - len(invalid_label_partitions)),
        },
        "files": _files(output),
        "authority": {
            "entry": "dual20 target-free BCF top-two with one E2 q50 agreement replacement at most; no capacity expansion",
            "continuation": "H4 prediction >=0 applies activation_earlier=.5 and giveback_tighten=.2 to next completed 15m interval only",
        },
        "continuation_training_contract": {
            "target": H4_GIVEBACK20_TARGET,
            "authority": H4_GIVEBACK20_AUTHORITY,
            "labels": {
                "path": str(h4_label_panel.relative_to(ROOT)),
                "sha256": _sha256(h4_label_panel),
            },
            "states": {
                "path": str(h4_state_panel.relative_to(ROOT)),
                "sha256": _sha256(h4_state_panel),
            },
            "route": {
                "path": str(h4_route_panel.relative_to(ROOT)),
                "sha256": _sha256(h4_route_panel),
                "mc1_column": "bcf_mc1_expected_bps",
            },
        },
    }
    (output / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
