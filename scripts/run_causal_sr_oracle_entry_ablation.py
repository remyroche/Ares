#!/usr/bin/env python3
"""Non-causal S/R ceiling test for the frozen E2 marginal-replacement path.

The oracle arm receives realised outcomes of the next policy-relevant S/R
interaction.  It is intentionally invalid for live/replay use and measures a
ceiling only: if this arm cannot improve E2 under matched portfolio replay,
there is no evidence that more elaborate causal S/R entry heads are worth
developing.  Candidate universe, reserve band, lower-quantile replacement
authority, two-model intersection, policy labels, costs and portfolio auction
remain unchanged.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from pyarrow.lib import ArrowInvalid


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_causal_sr_entry_e2_ablation as causal
from scripts import run_causal_sr_oracle_audit as audit
from scripts import run_strict_r3_p8u_15m_entry_agreement_ablation as agreement
from scripts import run_strict_r3_p8u_15m_entry_feature_contract_ablation as study
from scripts import run_strict_r3_p8u_15m_entry_pairwise_replacement_ablation as base
from scripts import run_strict_r3_p8u_15m_entry_postfs_hpo as hpo


ORACLE_ROOT = ROOT / "data_perp/artifacts/causal_sr_oracle_audit_20260830_v1"
HEADS_ROOT = ROOT / "data_perp/artifacts/causal_sr_heads_oof_20260830_v3_entrypivotfix"
FEATURE_STUDY = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_feature_contract_20260830_v2"
INCUMBENT_E2 = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_agreement_ablation_20260830_v1/E2_q50_agreement_selection_target_free.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_sr_oracle_entry_ceiling_20260830_v1"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")
# GAS and TON have no readable archival policy-label parts in the frozen
# control tree.  They must therefore be excluded from this outcome-joined
# diagnostic, never imputed or substituted.  This keeps the comparison
# target-free until the normal label join and applies exactly the same
# exclusion to every arm (including the frozen incumbent).
UNAVAILABLE_LABEL_SYMBOLS = frozenset({"GAS_USD:USD", "TON_USD:USD"})

ORACLE_PAIR_FEATURES = tuple((*audit.ORACLE_FEATURES, *(f"margin__{item}" for item in audit.ORACLE_FEATURES)))
RAW_PAIR_FEATURES = (*causal.RAW_PAIR_FEATURES, *audit.ORACLE_FEATURES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _labels(root: Path) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Read the immutable rich-policy labels, failing closed per symbol.

    The two known unavailable parts are an archival-data limitation, not a
    licence to use another label source.  Explicitly removing their symbols
    before *every* outcome join preserves a matched candidate universe across
    the control, causal-head and non-causal-oracle arms.
    """
    parts = sorted(root.resolve().glob("policy_parts/symbol=*/policy_labels.parquet"))
    if not parts:
        raise FileNotFoundError(f"no readable labels under {root}")
    columns = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
        "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
    ]
    frames: list[pd.DataFrame] = []
    unavailable: list[dict[str, str]] = []
    for path in parts:
        symbol = path.parent.name.removeprefix("symbol=")
        if symbol in UNAVAILABLE_LABEL_SYMBOLS:
            unavailable.append({"symbol": symbol, "reason": "predeclared_archival_unavailable"})
            continue
        try:
            frames.append(pd.read_parquet(path, columns=columns))
        except (ArrowInvalid, OSError) as exc:
            # An unreadable immutable label part cannot be repaired by a
            # different outcome source.  Record and remove its whole symbol
            # from every matched arm below.
            unavailable.append({"symbol": symbol, "reason": f"unreadable_parquet:{type(exc).__name__}"})
    if not frames:
        raise RuntimeError("all rich-policy label parts are unavailable")
    frame = pd.concat(frames, ignore_index=True)
    frame["candidate_id"] = frame.candidate_id.astype(str)
    if frame.candidate_id.duplicated().any():
        raise AssertionError("policy label identities are not unique")
    valid = frame.policy_path_valid.fillna(False).astype(bool)
    if not np.isclose(
        pd.to_numeric(frame.loc[valid, "policy_gross_bps"], errors="coerce")
        - pd.to_numeric(frame.loc[valid, "policy_net_bps"], errors="coerce"),
        100.0, rtol=0.0, atol=1e-8,
    ).all():
        raise AssertionError("rich policy cost is not exactly 100 bps once")
    frame["policy_label_available_ts"] = pd.to_datetime(frame.policy_label_available_ts, utc=True, errors="raise")
    return frame, unavailable


def _merge(panel: pd.DataFrame, heads_root: Path, oracle_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = panel.copy()
    work["snapshot_ts"] = pd.to_datetime(work.__decision_ts__, utc=True, errors="raise")
    keys = ["candidate_id", "snapshot_ts"]
    predicted = pd.read_parquet(heads_root / "entry_sr_oof_features.parquet")
    predicted["snapshot_ts"] = pd.to_datetime(predicted.snapshot_ts, utc=True, errors="raise")
    if predicted.duplicated(keys).any():
        raise AssertionError("predicted S/R entry identity is duplicated")
    oracle = pd.read_parquet(oracle_root / "entry_oracle_labels_NONCAUSAL_DIAGNOSTIC_ONLY.parquet")
    oracle["snapshot_ts"] = pd.to_datetime(oracle.snapshot_ts, utc=True, errors="raise")
    if oracle.duplicated(keys).any():
        raise AssertionError("oracle S/R entry identity is duplicated")
    prediction_cols = [item for item in causal.SR_FEATURES if item in predicted]
    oracle_cols = [item for item in audit.ORACLE_FEATURES if item in oracle]
    result = work.merge(predicted.loc[:, [*keys, *prediction_cols]], on=keys, how="left", validate="one_to_one")
    result = result.merge(oracle.loc[:, [*keys, *oracle_cols]], on=keys, how="left", validate="one_to_one")
    coverage = result.assign(
        predicted_sr_available=result.loc[:, prediction_cols].notna().any(axis=1),
        oracle_sr_available=result.loc[:, oracle_cols].notna().any(axis=1),
    ).groupby(pd.to_datetime(result.__decision_ts__, utc=True).dt.to_period("M"), observed=True).agg(
        rows=("candidate_id", "size"), predicted_sr_available=("predicted_sr_available", "sum"),
        oracle_sr_available=("oracle_sr_available", "sum"),
    ).reset_index(names="decision_month")
    return result, coverage


def _pairs(frame: pd.DataFrame, *, require_labels: bool) -> pd.DataFrame:
    pairs = study._pairs(frame, RAW_PAIR_FEATURES, require_labels=require_labels)
    if pairs.empty:
        return pairs
    # The legacy generic pair builder only makes margins for its native and
    # VWAP fields.  Project both causal OOF S/R fields and the diagnostic
    # oracle fields explicitly, reserve minus incumbent, under the unchanged
    # pair identity and replacement authority.
    projected = tuple((*causal.SR_FEATURES, *audit.ORACLE_FEATURES))
    candidates = frame.loc[:, ["candidate_id", *projected]].copy().set_index("candidate_id", verify_integrity=True)
    for feature in projected:
        reserve = pairs.reserve_candidate_id.map(candidates[feature])
        incumbent = pairs.incumbent_candidate_id.map(candidates[feature])
        pairs[f"margin__{feature}"] = pd.to_numeric(reserve, errors="coerce") - pd.to_numeric(incumbent, errors="coerce")
    return pairs


def _scope(selection: pd.DataFrame, labels: pd.DataFrame, arm: str, output: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    ts = pd.to_datetime(selection.__decision_ts__, utc=True, errors="raise")
    for scope, frame in (
        ("selection_jun_jul", selection.loc[ts.lt(SELECTION_END)]),
        ("august_holdout", selection.loc[ts.ge(SELECTION_END)]),
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
    parser.add_argument("--oracle-root", type=Path, default=ORACLE_ROOT)
    parser.add_argument("--heads-root", type=Path, default=HEADS_ROOT)
    parser.add_argument("--feature-study", type=Path, default=FEATURE_STUDY)
    parser.add_argument("--incumbent-e2", type=Path, default=INCUMBENT_E2)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; defaults June--August")
    args = parser.parse_args()
    if args.train_months < 2:
        raise ValueError("strict E2 pair training needs at least two prior months")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    held = tuple(pd.Timestamp(f"{item}-01", tz="UTC") for item in args.held_month) if args.held_month else tuple(pd.date_range("2026-06-01", "2026-08-01", freq="MS", tz="UTC"))
    feature_map = hpo._features_by_month(args.feature_study.resolve() / "stable_selected_features.parquet", "E3_vwap_fs", held)
    panel = study._candidate_frame(study._load_panel(study.OLD_PANEL, study.VWAP_PANEL))
    panel, coverage = _merge(panel, args.heads_root.resolve(), args.oracle_root.resolve())
    labels, unavailable_label_parts = _labels(study.LABEL_ROOT)
    unavailable_symbols = frozenset(item["symbol"] for item in unavailable_label_parts)
    panel = panel.loc[~panel.__symbol__.isin(unavailable_symbols)].copy()
    labelled = panel.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    labelled = labelled.loc[labelled.policy_path_valid.fillna(False)].copy()
    labelled["policy_label_available_ts"] = pd.to_datetime(labelled.policy_label_available_ts, utc=True, errors="raise")
    output.mkdir(parents=True, exist_ok=False)
    arms = {
        "E2_q50_agreement_control": tuple(),
        "E2_q50_agreement_plus_predicted_sr": tuple(causal.SR_PAIR_FEATURES),
        "E2_q50_agreement_plus_ORACLE_sr_NONCAUSAL": ORACLE_PAIR_FEATURES,
    }
    choices: dict[str, list[pd.DataFrame]] = {name: [] for name in arms}
    proposals: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    for month in held:
        end, start = month + pd.offsets.MonthBegin(1), month - pd.DateOffset(months=args.train_months)
        train_raw = labelled.loc[labelled.__decision_ts__.ge(start) & labelled.__decision_ts__.lt(month)].copy()
        train_pairs = _pairs(train_raw, require_labels=True)
        train_pairs = train_pairs.loc[pd.to_datetime(train_pairs.pair_label_available_ts, utc=True).lt(month)].copy()
        test = panel.loc[panel.__decision_ts__.ge(month) & panel.__decision_ts__.lt(end)].copy()
        test_pairs = _pairs(test, require_labels=False)
        if len(train_pairs) < 100 or test.empty or test_pairs.empty:
            raise RuntimeError(f"{month:%Y-%m}: incomplete strict E2 oracle fold")
        base_features = tuple(feature_map[month])
        for arm, added in arms.items():
            features = tuple((*base_features, *added))
            missing = set(features).difference(train_pairs.columns) | set(features).difference(test_pairs.columns)
            if missing:
                raise AssertionError(f"{arm}: pair contract missing {sorted(missing)}")
            selected_by_model: list[pd.DataFrame] = []
            for spec_name in ("H0_q50_d3_l7_baseline", "H3_q50_d2_l3_strict"):
                model = hpo._fit(train_pairs, features, hpo.SPECS[spec_name])
                prediction = test_pairs.loc[:, [
                    "reserve_candidate_id", "incumbent_candidate_id", "__decision_ts__", "__symbol__",
                    "reserve_bcf_mc1_expected_bps", "incumbent_bcf_mc1_expected_bps",
                ]].copy()
                prediction["pair_lcb_advantage_bps"] = model.predict(test_pairs.loc[:, features])
                chosen, detail = base._apply_replacement(test, prediction, 50.0)
                selected_by_model.append(chosen)
                detail["held_month"], detail["model_arm"], detail["pair_model"] = month.strftime("%Y-%m"), arm, spec_name
                proposals.append(detail)
            selected_ids = set(selected_by_model[0].candidate_id).intersection(selected_by_model[1].candidate_id)
            choice = selected_by_model[0].loc[selected_by_model[0].candidate_id.isin(selected_ids)].copy()
            choice["held_month"], choice["model_arm"] = month.strftime("%Y-%m"), arm
            choices[arm].append(choice)
        folds.append({
            "held_month": month.strftime("%Y-%m"), "train_pairs": len(train_pairs), "test_candidates": len(test),
            "test_pairs": len(test_pairs), "oracle_entry_interactions": int(pd.to_numeric(test.sr_oracle_any_interaction, errors="coerce").fillna(0.0).sum()),
        })
    metrics: list[dict[str, object]] = []
    for arm, parts in choices.items():
        selection = pd.concat(parts, ignore_index=True)
        if selection.candidate_id.duplicated().any():
            raise AssertionError(f"{arm} duplicated selected candidate identity")
        selection.to_parquet(output / f"{arm}_selection_target_free.parquet", index=False, compression="zstd")
        metrics.extend(_scope(selection, labels, arm, output))
    incumbent = agreement._read(args.incumbent_e2.resolve())
    incumbent = incumbent.loc[~incumbent.__symbol__.isin(unavailable_symbols)].copy()
    incumbent.to_parquet(output / "E2_frozen_incumbent_selection_target_free.parquet", index=False, compression="zstd")
    metrics.extend(_scope(incumbent, labels, "E2_frozen_incumbent", output))
    summary = pd.DataFrame(metrics)
    summary["total_ev_per_abs_drawdown"] = summary.total_policy_net_bps / summary.max_drawdown.abs().replace(0.0, np.nan)
    for scope, group in summary.groupby("evaluation_scope", sort=False):
        reference = group.loc[group.model_arm.eq("E2_frozen_incumbent")].iloc[0]
        for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown", "worst_week", "total_ev_per_abs_drawdown"):
            summary.loc[group.index, f"delta_vs_incumbent_{metric}"] = group[metric] - reference[metric]
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    coverage.to_parquet(output / "sr_merge_coverage.parquet", index=False)
    pd.DataFrame(folds).to_parquet(output / "fold_trace.parquet", index=False)
    pd.concat(proposals, ignore_index=True).to_parquet(output / "replacement_proposals.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "causal-sr-oracle-entry-ceiling-v1",
        "scope": "offline non-causal ceiling diagnostic only; no live/canonical/execution mutation",
        "oracle_root": str(args.oracle_root.resolve()), "oracle_manifest_sha256": _sha256(args.oracle_root.resolve() / "run_manifest.json"),
        "heads_root": str(args.heads_root.resolve()), "heads_manifest_sha256": _sha256(args.heads_root.resolve() / "run_manifest.json"),
        "incumbent_e2": str(args.incumbent_e2.resolve()), "incumbent_e2_sha256": _sha256(args.incumbent_e2.resolve()),
        "authority": "unchanged: 20--30 bps reserve may replace only the marginal ordinary-core incumbent at lower-quantile advantage >= 50 bps; H0/H3 selection intersection and portfolio constraints unchanged",
        "oracle_arm_warning": "E2_q50_agreement_plus_ORACLE_sr_NONCAUSAL receives future S/R interaction outcomes and is forbidden from inference, causal replay, calibration, or live execution",
        "unavailable_label_parts_excluded_from_all_arms": unavailable_label_parts,
        "held_months": [f"{item:%Y-%m}" for item in held], "fold_trace": folds,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
