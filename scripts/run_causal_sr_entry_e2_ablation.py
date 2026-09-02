#!/usr/bin/env python3
"""Strict-OOS causal S/R augmentation of the E2 q50 agreement entry arm.

The incumbent's target-free universe, reserve/ordinary BCF topology, pairwise
target, 30--50 bps replacement authority and portfolio replay are unchanged.
Only independently OOF S/R snapshot fields are appended to the pre-existing
per-fold E3 feature contract.  This is a challenger-only research script.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_strict_r3_p8u_15m_entry_postfs_hpo as hpo
from scripts import run_strict_r3_p8u_15m_entry_feature_contract_ablation as study
from scripts import run_strict_r3_p8u_15m_entry_pairwise_replacement_ablation as base
from scripts import run_strict_r3_p8u_15m_entry_agreement_ablation as agreement
from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS, VWAP_15M_FEATURE_KEYS


SR_ROOT = ROOT / "data_perp/artifacts/causal_sr_heads_oof_20260830_v1"
FEATURE_STUDY = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_feature_contract_20260830_v2"
INCUMBENT_E2 = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_agreement_ablation_20260830_v1/E2_q50_agreement_selection_target_free.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_sr_entry_e2_agreement_20260830_v1"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")
SR_FEATURES = (
    "sr_long_support_hold_strength", "sr_long_resistance_break_probability",
    "sr_long_downside_break_probability", "sr_long_resistance_rejection_strength",
    "sr_long_structure_balance", "sr_long_support_distance_atr",
    "sr_long_resistance_distance_atr", "sr_support_prior_strength",
    "sr_resistance_prior_strength", "sr_support_reaction_magnitude_q50",
    "sr_resistance_reaction_magnitude_q50",
)
SR_PAIR_FEATURES = (*SR_FEATURES, *(f"margin__{feature}" for feature in SR_FEATURES))
RAW_PAIR_FEATURES = (*FIFTEEN_MINUTE_FEATURE_KEYS, *study.SCORE_FEATURES, *VWAP_15M_FEATURE_KEYS, *SR_FEATURES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _merge_sr(panel: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    sr = pd.read_parquet(path)
    sr["snapshot_ts"] = pd.to_datetime(sr.snapshot_ts, utc=True, errors="raise")
    keys = ["candidate_id", "snapshot_ts"]
    if sr.duplicated(keys).any():
        raise AssertionError("S/R entry snapshots have duplicate candidate identities")
    work = panel.copy()
    work["snapshot_ts"] = pd.to_datetime(work.__decision_ts__, utc=True, errors="raise")
    merged = work.merge(sr.loc[:, [*keys, *SR_FEATURES]], on=keys, how="left", validate="one_to_one")
    coverage = merged.assign(sr_available=merged.loc[:, list(SR_FEATURES)].notna().any(axis=1)).groupby(
        pd.to_datetime(merged.__decision_ts__, utc=True).dt.to_period("M"), observed=True
    ).agg(rows=("candidate_id", "size"), sr_available=("sr_available", "sum")).reset_index(names="decision_month")
    return merged, coverage


def _pairs(frame: pd.DataFrame, *, require_labels: bool) -> pd.DataFrame:
    """Build the incumbent pair contract, then add explicit S/R pair margins.

    The incumbent E3 selection contains ``margin__`` fields.  Its generic
    pair-builder deliberately creates those margins only for the historical
    base/VWAP fields, so new causal S/R fields need the same reserve-minus-
    incumbent projection here.  This changes neither the label nor
    replacement authority.
    """
    pairs = study._pairs(frame, RAW_PAIR_FEATURES, require_labels=require_labels)
    if pairs.empty:
        return pairs
    candidates = frame.loc[:, ["candidate_id", *SR_FEATURES]].copy().set_index("candidate_id", verify_integrity=True)
    for feature in SR_FEATURES:
        reserve = pairs.reserve_candidate_id.map(candidates[feature])
        incumbent = pairs.incumbent_candidate_id.map(candidates[feature])
        pairs[f"margin__{feature}"] = pd.to_numeric(reserve, errors="coerce") - pd.to_numeric(incumbent, errors="coerce")
    return pairs


def _scope(selection: pd.DataFrame, labels: pd.DataFrame, arm: str, output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scope, frame in (
        ("selection_jun_jul", selection.loc[selection.__decision_ts__.lt(SELECTION_END)]),
        ("august_holdout", selection.loc[selection.__decision_ts__.ge(SELECTION_END)]),
        ("all_oos", selection),
    ):
        if frame.empty:
            continue
        metric = base._replay(frame.copy(), labels, f"{arm}__{scope}", output)
        metric["model_arm"], metric["evaluation_scope"] = arm, scope
        rows.append(metric)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr-root", type=Path, default=SR_ROOT)
    parser.add_argument("--feature-study", type=Path, default=FEATURE_STUDY)
    parser.add_argument("--incumbent-e2", type=Path, default=INCUMBENT_E2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-months", type=int, default=4)
    args = parser.parse_args()
    if args.train_months < 2:
        raise ValueError("strict pair training needs at least two prior months")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    held_months = tuple(pd.date_range("2026-06-01", "2026-08-01", freq="MS", tz="UTC"))
    feature_map = hpo._features_by_month(args.feature_study.resolve() / "stable_selected_features.parquet", "E3_vwap_fs", held_months)
    panel = study._candidate_frame(study._load_panel(study.OLD_PANEL, study.VWAP_PANEL))
    panel, coverage = _merge_sr(panel, args.sr_root.resolve() / "entry_sr_oof_features.parquet")
    labels = base._labels(study.LABEL_ROOT)
    labelled = panel.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    labelled = labelled.loc[labelled.policy_path_valid.fillna(False)].copy()
    labelled["policy_label_available_ts"] = pd.to_datetime(labelled.policy_label_available_ts, utc=True, errors="raise")
    choices: dict[str, list[pd.DataFrame]] = {"E2_q50_agreement_control": [], "E2_q50_agreement_plus_sr": []}
    trace: list[pd.DataFrame] = []
    fold_trace: list[dict[str, object]] = []
    for held in held_months:
        end, start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=args.train_months)
        train_raw = labelled.loc[labelled.__decision_ts__.ge(start) & labelled.__decision_ts__.lt(held)].copy()
        base_features = feature_map[held]
        train_pairs_control = _pairs(train_raw, require_labels=True)
        train_pairs_sr = train_pairs_control
        train_pairs_control = train_pairs_control.loc[pd.to_datetime(train_pairs_control.pair_label_available_ts, utc=True).lt(held)].copy()
        train_pairs_sr = train_pairs_sr.loc[pd.to_datetime(train_pairs_sr.pair_label_available_ts, utc=True).lt(held)].copy()
        test = panel.loc[panel.__decision_ts__.ge(held) & panel.__decision_ts__.lt(end)].copy()
        test_pairs_control = _pairs(test, require_labels=False)
        test_pairs_sr = test_pairs_control
        if len(train_pairs_control) < 100 or len(train_pairs_sr) < 100 or test.empty:
            raise RuntimeError(f"incomplete strict OOS E2/SR fold {held:%Y-%m}")
        for arm, train_pairs, test_pairs, features in (
            ("E2_q50_agreement_control", train_pairs_control, test_pairs_control, tuple(base_features)),
            ("E2_q50_agreement_plus_sr", train_pairs_sr, test_pairs_sr, tuple((*base_features, *SR_PAIR_FEATURES))),
        ):
            selected_by_model: list[pd.DataFrame] = []
            for spec_name in ("H0_q50_d3_l7_baseline", "H3_q50_d2_l3_strict"):
                model = hpo._fit(train_pairs, features, hpo.SPECS[spec_name])
                prediction = test_pairs.loc[:, ["reserve_candidate_id", "incumbent_candidate_id", "__decision_ts__", "__symbol__", "reserve_bcf_mc1_expected_bps", "incumbent_bcf_mc1_expected_bps"]].copy()
                prediction["pair_lcb_advantage_bps"] = model.predict(test_pairs.loc[:, features])
                chosen, proposals = base._apply_replacement(test, prediction, 50.0)
                selected_by_model.append(chosen)
                proposals["held_month"], proposals["model_arm"], proposals["pair_model"] = held.strftime("%Y-%m"), arm, spec_name
                trace.append(proposals)
            selected_ids = set(selected_by_model[0].candidate_id).intersection(selected_by_model[1].candidate_id)
            choice = selected_by_model[0].loc[selected_by_model[0].candidate_id.isin(selected_ids)].copy()
            choice["held_month"], choice["model_arm"] = held.strftime("%Y-%m"), arm
            choices[arm].append(choice)
        fold_trace.append({"held_month": held.strftime("%Y-%m"), "train_pairs_control": len(train_pairs_control), "train_pairs_sr": len(train_pairs_sr), "test_candidates": len(test), "sr_available_test": int(test.loc[:, list(SR_FEATURES)].notna().any(axis=1).sum())})
    incumbent = agreement._read(args.incumbent_e2.resolve())
    summary_rows: list[dict[str, object]] = []
    for arm, frames in choices.items():
        selection = pd.concat(frames, ignore_index=True)
        if selection.candidate_id.duplicated().any():
            raise AssertionError(f"{arm} selected duplicate candidate identities")
        selection.to_parquet(output / f"{arm}_selection_target_free.parquet", index=False, compression="zstd")
        summary_rows.extend(_scope(selection, labels, arm, output))
    incumbent.to_parquet(output / "E2_frozen_incumbent_selection_target_free.parquet", index=False, compression="zstd")
    summary_rows.extend(_scope(incumbent, labels, "E2_frozen_incumbent", output))
    summary = pd.DataFrame(summary_rows)
    summary["total_ev_per_abs_drawdown"] = summary.total_policy_net_bps / summary.max_drawdown.abs().replace(0.0, np.nan)
    for scope, group in summary.groupby("evaluation_scope", sort=False):
        reference = group.loc[group.model_arm.eq("E2_frozen_incumbent")].iloc[0]
        for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown", "worst_week", "total_ev_per_abs_drawdown"):
            summary.loc[group.index, f"delta_vs_incumbent_{metric}"] = group[metric] - reference[metric]
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    coverage.to_parquet(output / "sr_merge_coverage.parquet", index=False)
    pd.DataFrame(fold_trace).to_parquet(output / "fold_trace.parquet", index=False)
    pd.concat(trace, ignore_index=True).to_parquet(output / "replacement_proposals.parquet", index=False, compression="zstd")
    (output / "run_manifest.json").write_text(json.dumps({
        "schema": "causal-sr-entry-e2-ablation-v1", "scope": "offline strict-OOS challenger; no live/canonical mutation",
        "sr_root": str(args.sr_root.resolve()), "sr_manifest_sha256": _sha256(args.sr_root.resolve() / "run_manifest.json"),
        "incumbent_e2": str(args.incumbent_e2.resolve()), "incumbent_e2_sha256": _sha256(args.incumbent_e2.resolve()),
        "pair_target": "unchanged pair_advantage_bps", "authority": "unchanged marginal reserve replacement only; E2 is q50 d3/d2 intersection",
        "features": {"base": "per-fold E3_vwap_fs", "added_sr": list(SR_PAIR_FEATURES)}, "fold_trace": fold_trace,
        "held_months": [f"{x:%Y-%m}" for x in held_months],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
