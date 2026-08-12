#!/usr/bin/env python3
"""Portable MDA for the additive, frozen-schema-v2 K9 reliability surface.

The ledger owns the repaired strict-R3 scores.  This runner uses the full
established causal support/OOD/reliability/committee pool *plus* the new soft
K9-membership-weighted cluster-health fields.  It never imports values from a
legacy score lineage merely because candidate IDs overlap.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.n5_forest_support_sizing import BASELINE_N5_PARAMS  # noqa: E402
from extreme_price_movements.stage_i_causal_admission import (  # noqa: E402
    Causal21dAdmissionSpec,
    apply_causal_21d_side_admission,
)
from extreme_price_movements.trust_sizing_ablation import ParentExpectation  # noqa: E402
import scripts.run_strict_r3_n5_canonical_selection as selection  # noqa: E402


SEED = 20260811
GROUPS = ROOT / "config/strict_r3_n5_feature_groups_schema_v2_additive.json"
CORE = {
    "base_score", "base_rank", "base_anchor_bps", "consensus_rank", "residual_rank",
    "upstream", "severe200_probability", "final_score",
}


def _fields(frame: pd.DataFrame) -> list[str]:
    config = json.loads(GROUPS.read_text())
    accepted: list[str] = []
    for field in frame.columns:
        if field in CORE or field.startswith((
            "leaf_support_", "leaf_ood_", "k9_", "cluster_recent_", "cluster_scorecond_",
            "residual_heads_", "reliability_", "continuous_regime__", "meta_context__",
        )):
            try:
                selection._mda_group(field, config)
            except ValueError:
                continue
            values = pd.to_numeric(frame[field], errors="coerce")
            if values.notna().mean() >= 0.90 and values.var() > 1e-12:
                accepted.append(field)
    return list(dict.fromkeys(accepted))


def _candidate_feature_columns(surface: Path) -> list[str]:
    """Return every schema-declared equal-status MDA candidate without data IO.

    The wide research surface also contains raw residual-head scores and path
    evaluation details that are not legal/model inputs to this LDF.  Reading
    those fields first, then copying the whole frame through admission, made a
    209-field neutral MDA needlessly memory-bound.  This is a projection-only
    optimisation: the accepted field semantics and later coverage/variance
    gate remain exactly `_fields`.
    """

    config = json.loads(GROUPS.read_text())
    accepted: list[str] = []
    for field in pq.ParquetFile(surface).schema.names:
        if field in CORE or field.startswith((
            "leaf_support_", "leaf_ood_", "k9_", "cluster_recent_", "cluster_scorecond_",
            "residual_heads_", "reliability_", "continuous_regime__", "meta_context__",
        )):
            try:
                selection._mda_group(field, config)
            except ValueError:
                continue
            accepted.append(str(field))
    return list(dict.fromkeys(accepted))


def _load_admitted_surface(surface: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply admission before loading only the usable MDA feature projection."""

    candidate_fields = _candidate_feature_columns(surface)
    ledger_columns = list(dict.fromkeys([
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "geometry_bundle_sha256", "policy_path_valid", "policy_label_available_ts",
        "policy_net_bps", "policy_gross_bps", "policy_exit_reason", "final_score",
    ]))
    ledger = pd.read_parquet(surface, columns=ledger_columns)
    ledger["__decision_ts__"] = pd.to_datetime(ledger["__decision_ts__"], utc=True)
    ledger["policy_label_available_ts"] = pd.to_datetime(
        ledger["policy_label_available_ts"], utc=True, errors="coerce"
    )
    if ledger["candidate_id"].duplicated().any() or ledger["geometry_bundle_sha256"].nunique() != 1:
        raise AssertionError("MDA requires unique candidates and one frozen Geometry/K9 identity")
    admitted, admission = apply_causal_21d_side_admission(
        ledger, score_column="final_score", net_column="policy_net_bps",
        decision_column="__decision_ts__", label_available_column="policy_label_available_ts",
        identity_column="candidate_id", spec=Causal21dAdmissionSpec(mode="hierarchical_tail_side_shrinkage_v2"),
    )
    admitted["raw_expected_bps"] = pd.to_numeric(
        admitted["causal_21d_side_expected_net_bps"], errors="coerce"
    )
    admitted["mapped_ev_available"] = admitted["raw_expected_bps"].notna()
    # Explicitly release the pre-admission ledger before materialising wide
    # feature data.  On the large target-free population this removes a full
    # duplicate of the candidate state from the memory peak.
    del ledger
    feature_columns = ["candidate_id", *[field for field in candidate_fields if field != "final_score"]]
    features = pd.read_parquet(surface, columns=feature_columns)
    if features["candidate_id"].duplicated().any():
        raise AssertionError("MDA feature projection contains duplicate candidate IDs")
    admitted = admitted.merge(features, on="candidate_id", how="left", validate="one_to_one")
    if admitted[candidate_fields].isna().all(axis=1).any():
        raise AssertionError("MDA feature projection does not cover every admitted candidate")
    return admitted, admission


def _equal_month_sample(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    month = frame["__decision_ts__"].dt.to_period("M").astype(str)
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    quota = max(1, cap // month.nunique())
    for token in sorted(month.unique()):
        block = frame.loc[month.eq(token)]
        if len(block) > quota:
            block = block.iloc[np.sort(rng.choice(len(block), quota, replace=False))]
        parts.append(block)
    return pd.concat(parts, ignore_index=True).sort_values(
        ["__decision_ts__", "candidate_id"], kind="stable",
    )


def _folds(frame: pd.DataFrame, *, train_cap: int, held_cap: int) -> list[dict[str, object]]:
    folds: list[dict[str, object]] = []
    # Jan--Mar establish the first resolved same-contract history.  Apr--Jul
    # are four strictly chronological one-month MDA environments.
    for ordinal, cutoff in enumerate(pd.date_range("2025-04-01", "2025-07-01", freq="MS", tz="UTC")):
        held_end = cutoff + pd.offsets.MonthBegin(1)
        train_start = cutoff - pd.DateOffset(months=3)
        train_all = frame.loc[
            frame["__decision_ts__"].ge(train_start)
            & frame["__decision_ts__"].lt(cutoff)
            & frame["policy_label_available_ts"].lt(cutoff)
            & frame["policy_path_valid"].fillna(False).astype(bool)
            & frame["mapped_ev_available"].astype(bool)
            & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        ].copy()
        if len(train_all) < 10_000:
            raise ValueError(f"{cutoff:%Y-%m} has insufficient resolved training support")
        parent = ParentExpectation.fit(train_all["final_score"], train_all["policy_net_bps"])
        train_all["parent_expected_bps"] = parent.predict(train_all["final_score"])
        floor = float(pd.to_numeric(train_all["final_score"], errors="coerce").quantile(0.70))
        train = train_all.loc[pd.to_numeric(train_all["final_score"], errors="coerce").ge(floor)].copy()
        train = _equal_month_sample(train, train_cap, SEED + ordinal)
        held = frame.loc[
            frame["__decision_ts__"].ge(cutoff) & frame["__decision_ts__"].lt(held_end)
        ].copy()
        held["parent_expected_bps"] = parent.predict(held["final_score"])
        held["trust_gate_active"] = held["mapped_ev_available"].astype(bool) & pd.to_numeric(
            held["final_score"], errors="coerce"
        ).ge(floor)
        if len(held) > held_cap:
            held = selection._sampling_frame(held, cap=held_cap, seed=SEED + 100 + ordinal)
        folds.append({
            "fold": ordinal, "train_start": train_start, "cutoff": cutoff,
            "held_end": held_end, "train": train, "held": held, "train_floor": floor,
        })
    return folds


def _streaming_full_evaluation(
    *,
    surface: Path,
    contract: dict[str, object],
    out_dir: Path,
    train_cap: int,
    fold_start: str,
    fold_end: str,
) -> None:
    """Evaluate full held months without retaining the seven-month panel."""

    all_fields = list(contract["fields"])
    compact_fields = list(contract["compact_fields"])
    required = list(dict.fromkeys([
        "candidate_id", "__decision_ts__", "__symbol__", "side_name",
        "geometry_bundle_sha256", "policy_path_valid", "policy_label_available_ts",
        "policy_net_bps", "policy_gross_bps", "policy_exit_reason", *all_fields,
    ]))
    arms = (("full_additive", all_fields), ("compact_additive", compact_fields))
    outputs: dict[str, list[Path]] = {arm: [] for arm, _ in arms}
    fold_audit: list[dict[str, object]] = []
    cutoffs = pd.date_range(fold_start, fold_end, freq="MS", tz="UTC")
    if cutoffs.empty:
        raise ValueError("full-population evaluation requires at least one monthly fold")
    for ordinal, cutoff in enumerate(cutoffs):
        held_end = cutoff + pd.offsets.MonthBegin(1)
        window_start = cutoff - pd.DateOffset(months=3) - pd.Timedelta(days=21)
        window = pd.read_parquet(
            surface, columns=required,
            filters=[("__decision_ts__", ">=", window_start), ("__decision_ts__", "<", held_end)],
        )
        window["__decision_ts__"] = pd.to_datetime(window["__decision_ts__"], utc=True)
        window["policy_label_available_ts"] = pd.to_datetime(
            window["policy_label_available_ts"], utc=True, errors="coerce"
        )
        admitted, admission = apply_causal_21d_side_admission(
            window, score_column="final_score", net_column="policy_net_bps",
            decision_column="__decision_ts__", label_available_column="policy_label_available_ts",
            identity_column="candidate_id", spec=Causal21dAdmissionSpec(mode="hierarchical_tail_side_shrinkage_v2"),
        )
        admitted["raw_expected_bps"] = pd.to_numeric(
            admitted["causal_21d_side_expected_net_bps"], errors="coerce"
        )
        admitted["mapped_ev_available"] = admitted["raw_expected_bps"].notna()
        train_start = cutoff - pd.DateOffset(months=3)
        train_all = admitted.loc[
            admitted["__decision_ts__"].ge(train_start)
            & admitted["__decision_ts__"].lt(cutoff)
            & admitted["policy_label_available_ts"].lt(cutoff)
            & admitted["policy_path_valid"].fillna(False).astype(bool)
            & admitted["mapped_ev_available"].astype(bool)
            & np.isfinite(pd.to_numeric(admitted["policy_net_bps"], errors="coerce"))
        ].copy()
        parent = ParentExpectation.fit(train_all["final_score"], train_all["policy_net_bps"])
        train_all["parent_expected_bps"] = parent.predict(train_all["final_score"])
        floor = float(pd.to_numeric(train_all["final_score"], errors="coerce").quantile(0.70))
        train = train_all.loc[pd.to_numeric(train_all["final_score"], errors="coerce").ge(floor)].copy()
        train = _equal_month_sample(train, int(train_cap), SEED + ordinal)
        held = admitted.loc[
            admitted["__decision_ts__"].ge(cutoff) & admitted["__decision_ts__"].lt(held_end)
        ].copy()
        held["parent_expected_bps"] = parent.predict(held["final_score"])
        held["trust_gate_active"] = held["mapped_ev_available"].astype(bool) & pd.to_numeric(
            held["final_score"], errors="coerce"
        ).ge(floor)
        fold = {"fold": ordinal, "train": train, "held": held}
        for arm, fields in arms:
            output, metrics = selection._evaluate_contract(
                [fold], fields, params=BASELINE_N5_PARAMS, arm=arm,
            )
            path = out_dir / f"{arm}_fold{ordinal:02d}.parquet"
            output.to_parquet(path, index=False, compression="zstd")
            outputs[arm].append(path)
            fold_audit.append({
                "arm": arm, "fold": ordinal, "cutoff": cutoff,
                "train_rows": len(train), "held_rows": len(held),
                "field_count": len(fields), "admission_rows": len(admission), **metrics,
            })
            del output
        del held, train, train_all, admitted, window
        gc.collect()
    metrics: list[pd.DataFrame] = []
    for arm, _fields in arms:
        output = pd.concat([pd.read_parquet(path) for path in outputs[arm]], ignore_index=True)
        global_metrics = selection._period_tail_metrics(output, arm=arm, period_kind="global")
        monthly_metrics = selection._period_tail_metrics(output, arm=arm, period_kind="month")
        global_metrics.to_parquet(out_dir / f"{arm}_global_metrics.parquet", index=False)
        monthly_metrics.to_parquet(out_dir / f"{arm}_monthly_metrics.parquet", index=False)
        metrics.extend([global_metrics.assign(metric_kind="global"), monthly_metrics.assign(metric_kind="month")])
        del output
    pd.concat(metrics, ignore_index=True).to_parquet(out_dir / "full_population_metrics.parquet", index=False)
    pd.DataFrame(fold_audit).to_parquet(out_dir / "full_population_fold_audit.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-cap", type=int, default=40_000)
    parser.add_argument("--held-cap", type=int, default=12_000)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--diagnose-equal-size-only", action="store_true",
        help="Write the no-overlay control on the same folds, then exit before MDA.",
    )
    parser.add_argument(
        "--full-evaluate-contract", type=Path,
        help="Evaluate the full additive and selected compact contracts on unsampled held populations, then exit.",
    )
    parser.add_argument(
        "--full-evaluate-fold-start", default="2025-04-01",
        help="First held month (inclusive) for --full-evaluate-contract; defaults to the historical 2025 control.",
    )
    parser.add_argument(
        "--full-evaluate-fold-end", default="2025-07-01",
        help="Last held month (inclusive) for --full-evaluate-contract; defaults to the historical 2025 control.",
    )
    parser.add_argument(
        "--reuse-mda-dir", type=Path,
        help="Reuse a completed immutable MDA pass, then run retrained backward elimination.",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    selection.FEATURE_GROUP_CONFIG = GROUPS
    if args.full_evaluate_contract is not None:
        args.out_dir.mkdir(parents=True)
        _streaming_full_evaluation(
            surface=args.surface,
            contract=json.loads(args.full_evaluate_contract.read_text()),
            out_dir=args.out_dir, train_cap=int(args.train_cap),
            fold_start=str(args.full_evaluate_fold_start),
            fold_end=str(args.full_evaluate_fold_end),
        )
        print(json.dumps({"event": "streaming_full_population_evaluation_complete"}))
        return
    admitted, admission = _load_admitted_surface(args.surface)
    fields = _fields(admitted)
    if len(fields) < 50:
        raise ValueError(f"additive MDA feature pool unexpectedly small: {len(fields)}")
    folds = _folds(admitted, train_cap=int(args.train_cap), held_cap=int(args.held_cap))
    frozen_geometry_bundle_sha256 = str(admitted["geometry_bundle_sha256"].iloc[0])
    admission_rows = int(len(admission))
    # `_folds` owns isolated sampled train/held copies.  Retaining the full
    # admitted 200+ field population while fit_n5_forest holds four bundles
    # only inflates peak RSS; it cannot affect any fold or result.
    del admitted
    gc.collect()
    args.out_dir.mkdir(parents=True)
    equal = pd.concat(
        [
            fold["held"].loc[:, [
                "candidate_id", "__decision_ts__", "__symbol__", "final_score",
                "policy_path_valid", "policy_gross_bps", "policy_net_bps",
                "policy_exit_reason", "geometry_bundle_sha256", "raw_expected_bps",
                "parent_expected_bps", "trust_gate_active",
            ]].assign(trust_size_multiplier=1.0, arm="equal_size_control")
            for fold in folds
        ],
        ignore_index=True,
    )
    equal_score, equal_metrics = selection._objective(equal, arm="equal_size_control")
    selection._period_tail_metrics(equal, arm="equal_size_control", period_kind="global").to_parquet(
        args.out_dir / "equal_size_global_metrics.parquet", index=False,
    )
    selection._period_tail_metrics(equal, arm="equal_size_control", period_kind="month").to_parquet(
        args.out_dir / "equal_size_monthly_metrics.parquet", index=False,
    )
    (args.out_dir / "equal_size_control.json").write_text(json.dumps({
        "selection_score": equal_score, **equal_metrics,
        "folds": [str(fold["cutoff"]) for fold in folds],
        "rows": len(equal),
    }, indent=2) + "\n")
    if args.diagnose_equal_size_only:
        print(json.dumps({"event": "equal_size_diagnosis_complete", **equal_metrics}))
        return
    if args.reuse_mda_dir is None:
        detail, group_detail, proposed = selection._mda(
            folds, fields, params=BASELINE_N5_PARAMS, repeats=int(args.repeats),
            checkpoint_dir=args.out_dir / "mda_checkpoints",
        )
    else:
        detail = pd.read_parquet(args.reuse_mda_dir / "portable_mda_detail.parquet")
        group_detail = pd.read_parquet(args.reuse_mda_dir / "portable_group_mda_detail.parquet")
        previous = json.loads((args.reuse_mda_dir / "mda_feature_contract.json").read_text())
        if list(previous["fields"]) != fields:
            raise ValueError("cannot reuse MDA from a different additive feature contract")
        proposed = list(previous["mda_proposed_fields"])
    summary = detail.drop_duplicates("field").loc[:, [
        "field", "family", "group", "mda_median", "mda_mad", "mda_worst_fold",
        "positive_fold_recurrence", "portable_mda_score",
    ]].sort_values("portable_mda_score", ascending=False, kind="stable")
    detail.to_parquet(args.out_dir / "portable_mda_detail.parquet", index=False)
    group_detail.to_parquet(args.out_dir / "portable_group_mda_detail.parquet", index=False)
    summary.to_parquet(args.out_dir / "portable_mda_summary.parquet", index=False)
    compact, elimination = selection._backward_grouped_elimination(
        folds, fields, params=BASELINE_N5_PARAMS,
        group_mda=group_detail, feature_mda=detail,
        checkpoint_path=args.out_dir / "backward_elimination_progress.parquet",
    )
    elimination.to_parquet(args.out_dir / "backward_elimination_path.parquet", index=False)
    (args.out_dir / "mda_feature_contract.json").write_text(json.dumps({
        "schema": "strict_r3_schema_v2_additive_k9weighted_mda_v3_neutral_all_fields",
        "fields": fields, "field_count": len(fields), "mda_proposed_fields": proposed,
        "mda_proposed_field_count": len(proposed),
        "compact_fields": compact, "compact_field_count": len(compact),
        "selection_rule": (
            "all existing and newly-derived fields are equal MDA and retrained "
            "elimination candidates; no protected feature tier"
        ),
        "frozen_geometry_bundle_sha256": frozen_geometry_bundle_sha256,
        "history_rule": "cluster correctness/residuals use only resolved labels strictly before decision timestamp",
        "raw_k9_memberships_used": False,
        "folds": [str(fold["cutoff"]) for fold in folds],
        "admission_rows": admission_rows,
    }, indent=2) + "\n")
    print(json.dumps({"event": "complete", "rows": admission_rows, "features": len(fields), "mda_proposed": len(proposed)}))


if __name__ == "__main__":
    main()
