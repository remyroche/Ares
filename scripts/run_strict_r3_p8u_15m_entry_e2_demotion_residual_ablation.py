#!/usr/bin/env python3
"""Strict-OOS E2 demotion-only and residual-EV entry-head comparison.

This study has three independent, target-free entry authorities:

1. E2 demotion-only controls.  Begin with ordinary BCF top-two candidates and
   remove incumbents that the existing q50/d3 and q50/d2 replacement heads
   would both (or either) displace.  They never add a reserve candidate.
2. A candidate-level residual head for rich-policy EV relative to the mean of
   the two pre-existing MC1 EV estimates.  Its output is added equally back to
   both MC1 estimates before the normal dual 30-bps admission and top-two
   timestamp route.
3. A candidate-level residual head relative to a same-fold, prior-resolved
   isotonic map of the base timestamp rank.  Its output is fed into both MC1
   estimates under the same admission route.

All held selections are built only from target-free rows and prior-trained
models.  Policy labels are used only in prior-resolved folds and after
selection for portfolio-constrained outcome replay.  June--July is the
selection window; August is untouched until selection is frozen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_strict_r3_p8u_15m_entry_agreement_ablation as agreement
from scripts import run_strict_r3_p8u_15m_entry_feature_contract_ablation as feature_study
from scripts import run_strict_r3_p8u_15m_entry_pairwise_replacement_ablation as base
from scripts import run_strict_r3_p8u_15m_entry_postfs_hpo as h0


DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_e2_demotion_residual_20260830_v1"
HPO_ROOT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_postfs_hpo_20260830_v1"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")
CORE_FLOOR = 30.0
RESERVE_FLOOR = 20.0
SEED = 1729
RESIDUAL_WEIGHTS = (0.50, 1.00)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _candidate_features(path: Path, held_months: tuple[pd.Timestamp, ...]) -> dict[pd.Timestamp, tuple[str, ...]]:
    selected = h0._features_by_month(path, "E3_vwap_fs", held_months)
    contracts: dict[pd.Timestamp, tuple[str, ...]] = {}
    for held, fields in selected.items():
        result: list[str] = ["base_timestamp_rank"]
        for field in fields:
            raw = str(field).removeprefix("margin__")
            if raw == "incumbent_bcf_mc1_expected_bps":
                raw = "bcf_mc1_expected_bps"
            if raw not in result:
                result.append(raw)
        if not 25 <= len(result) <= 35:
            raise AssertionError(f"{held:%Y-%m}: invalid E3 candidate projection size {len(result)}")
        contracts[held] = tuple(result)
    return contracts


def _fit_residual(train: pd.DataFrame, features: tuple[str, ...], target: pd.Series) -> lgb.LGBMRegressor:
    spec = h0.SPECS["H0_q50_d3_l7_baseline"]
    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=int(spec["n_estimators"]), learning_rate=float(spec["learning_rate"]),
        max_depth=int(spec["max_depth"]), num_leaves=int(spec["num_leaves"]),
        min_child_samples=max(8, int(np.ceil(len(train) * float(spec["min_child_fraction"])))),
        subsample=.80, colsample_bytree=.80, reg_lambda=float(spec["reg_lambda"]),
        random_state=SEED, n_jobs=2, verbosity=-1,
    )
    # Preserve more support near the dual admission boundary without allowing
    # a candidate's held label to influence its own training weight.
    dual = pd.to_numeric(train.dual_mc1_min_bps, errors="raise").to_numpy(float)
    weights = 1.0 + np.clip((40.0 - dual) / 20.0, 0.0, 1.0)
    model.fit(train.loc[:, features], target.to_numpy(float), sample_weight=weights)
    return model


def _fit_base_calibrator(train: pd.DataFrame) -> IsotonicRegression:
    rank = pd.to_numeric(train.base_timestamp_rank, errors="coerce")
    target = pd.to_numeric(train.policy_net_bps, errors="coerce")
    valid = np.isfinite(rank.to_numpy(float)) & np.isfinite(target.to_numpy(float))
    if valid.sum() < 100:
        raise RuntimeError("base residual map lacks prior resolved support")
    # Lower rank is stronger, hence the monotone input is its negative.  This
    # map is fit only on the fold's prior resolved outcomes.
    mapper = IsotonicRegression(increasing=True, out_of_bounds="clip")
    mapper.fit(-rank.to_numpy(float)[valid], target.to_numpy(float)[valid])
    return mapper


def _select_adjusted(rows: pd.DataFrame, *, arm: str, prediction: np.ndarray, weight: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = rows.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "base_timestamp_rank", "bcf_final_score", "bcf_mc1_expected_bps", "current_mc1_expected_bps", "dual_mc1_min_bps"]].copy()
    work["residual_ev_bps"] = prediction
    work["bcf_mc1_adjusted_bps"] = pd.to_numeric(work.bcf_mc1_expected_bps, errors="raise").to_numpy(float) + float(weight) * prediction
    work["current_mc1_adjusted_bps"] = pd.to_numeric(work.current_mc1_expected_bps, errors="raise").to_numpy(float) + float(weight) * prediction
    work["dual_mc1_adjusted_bps"] = np.minimum(work.bcf_mc1_adjusted_bps, work.current_mc1_adjusted_bps)
    work["eligible_after_residual"] = work.dual_mc1_adjusted_bps.ge(CORE_FLOOR)
    chosen: list[pd.DataFrame] = []
    audit: list[pd.DataFrame] = []
    for _, group in work.groupby("__decision_ts__", sort=True):
        base_ids = set(base._incumbent_top2(group).candidate_id.astype(str))
        ranked = group.loc[group.eligible_after_residual].sort_values(
            ["bcf_mc1_adjusted_bps", "bcf_final_score", "candidate_id"], ascending=[False, False, True], kind="stable",
        ).head(base.MAX_NEW_ENTRIES).copy()
        selected_ids = set(ranked.candidate_id.astype(str))
        group = group.copy()
        group["selected"] = group.candidate_id.astype(str).isin(selected_ids)
        group["baseline_bcf_top2"] = group.candidate_id.astype(str).isin(base_ids)
        group["action"] = "not_selected"
        group.loc[group.baseline_bcf_top2 & ~group.eligible_after_residual, "action"] = "demoted_by_residual_gate"
        group.loc[group.baseline_bcf_top2 & group.eligible_after_residual & ~group.selected, "action"] = "demoted_by_residual_priority"
        group.loc[group.selected & ~group.baseline_bcf_top2, "action"] = "promoted_by_residual"
        group.loc[group.selected & group.baseline_bcf_top2, "action"] = "kept_bcf_core"
        group["arm"] = arm
        audit.append(group)
        chosen.append(ranked)
    selection = pd.concat(chosen, ignore_index=True)
    trace = pd.concat(audit, ignore_index=True)
    if selection.candidate_id.duplicated().any() or selection.groupby("__decision_ts__").size().gt(base.MAX_NEW_ENTRIES).any():
        raise AssertionError(f"{arm}: residual authority violated target identity/capacity")
    forbidden = [name for name in selection if name.startswith("policy_") or "label_available" in name]
    if forbidden:
        raise AssertionError(f"{arm}: target-free selection contains outcome fields {forbidden}")
    return selection, trace


def _demotion_only(bcf: pd.DataFrame, head_a: pd.DataFrame, head_b: pd.DataFrame, *, both: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove only original BCF incumbents; reserve candidates can never enter."""
    a_ids, b_ids = set(head_a.candidate_id.astype(str)), set(head_b.candidate_id.astype(str))
    work = bcf.copy()
    ids = work.candidate_id.astype(str)
    absent_a, absent_b = ~ids.isin(a_ids), ~ids.isin(b_ids)
    demote = absent_a & absent_b if both else absent_a | absent_b
    work["e2_head_a_displaces"] = absent_a
    work["e2_head_b_displaces"] = absent_b
    work["demoted"] = demote
    work["action"] = np.where(demote, "demoted_no_promotion", "kept_bcf_core")
    selection = work.loc[~work.demoted].copy()
    if not selection.candidate_id.isin(bcf.candidate_id).all():
        raise AssertionError("demotion-only control manufactured a candidate")
    return selection, work


def _scope_replay(selection: pd.DataFrame, labels: pd.DataFrame, arm: str, output: Path) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for scope, subset in (
        ("selection_jun_jul", selection.loc[pd.to_datetime(selection.__decision_ts__, utc=True).lt(SELECTION_END)].copy()),
        ("august_holdout", selection.loc[pd.to_datetime(selection.__decision_ts__, utc=True).ge(SELECTION_END)].copy()),
        ("all_oos", selection),
    ):
        if subset.empty:
            continue
        metrics = base._replay(subset, labels, f"{arm}__{scope}", output)
        metrics["model_arm"], metrics["evaluation_scope"] = arm, scope
        summaries.append(metrics)
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-study", type=Path, default=h0.FEATURE_STUDY)
    parser.add_argument("--hpo-root", type=Path, default=HPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; default Jun--Aug 2026")
    args = parser.parse_args()
    if args.train_months < 2:
        raise ValueError("strict residual training needs at least two preceding calendar months")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    held_months = tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in args.held_month) if args.held_month else tuple(pd.date_range("2026-06-01", "2026-08-01", freq="MS", tz="UTC"))
    feature_file = args.feature_study.resolve() / "stable_selected_features.parquet"
    feature_map = _candidate_features(feature_file, held_months)
    target_free = feature_study._candidate_frame(feature_study._load_panel(feature_study.OLD_PANEL, feature_study.VWAP_PANEL))
    labels = base._labels(base.LABEL_ROOT)
    labelled = target_free.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    labelled = labelled.loc[labelled.policy_path_valid.fillna(False)].copy()
    labelled["policy_label_available_ts"] = pd.to_datetime(labelled.policy_label_available_ts, utc=True, errors="raise")
    hpo_root = args.hpo_root.resolve()
    tables = {
        "B0_bcf_top2": agreement._read(hpo_root / "B0_bcf_top2_selection_target_free.parquet"),
        "H0_q50_d3": agreement._read(hpo_root / "H0_q50_d3_l7_baseline_selection_target_free.parquet"),
        "H3_q50_d2": agreement._read(hpo_root / "H3_q50_d2_l3_strict_selection_target_free.parquet"),
    }
    selections: dict[str, list[pd.DataFrame]] = {"B0_bcf_top2": []}
    traces: list[pd.DataFrame] = []
    scored: list[pd.DataFrame] = []
    # E2 controls are generated once from already target-free selections.
    bcf = tables["B0_bcf_top2"]
    for name, both in (("D1_e2_both_heads_demote_only", True), ("D2_e2_either_head_demote_only", False)):
        selection, trace = _demotion_only(bcf, tables["H0_q50_d3"], tables["H3_q50_d2"], both=both)
        selections[name] = [selection]
        trace["arm"] = name
        traces.append(trace)
    for held in held_months:
        end, start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=args.train_months)
        train = labelled.loc[labelled.__decision_ts__.ge(start) & labelled.__decision_ts__.lt(held) & labelled.policy_label_available_ts.lt(held)].copy()
        test = target_free.loc[target_free.__decision_ts__.ge(held) & target_free.__decision_ts__.lt(end)].copy()
        features = feature_map[held]
        missing = set(features).difference(train.columns) | set(features).difference(test.columns)
        observed = set(pd.to_datetime(train.__decision_ts__, utc=True).dt.strftime("%Y-%m"))
        required = {(held - pd.DateOffset(months=n)).strftime("%Y-%m") for n in range(1, args.train_months + 1)}
        if missing or len(train) < 500 or test.empty or not required.issubset(observed):
            raise RuntimeError(f"{held:%Y-%m}: strict residual fold lacks support; missing={sorted(missing)} rows={len(train)}")
        mc1_prior = (pd.to_numeric(train.bcf_mc1_expected_bps, errors="raise") + pd.to_numeric(train.current_mc1_expected_bps, errors="raise")) / 2.0
        base_mapper = _fit_base_calibrator(train)
        train_base_prior = base_mapper.predict(-pd.to_numeric(train.base_timestamp_rank, errors="raise").to_numpy(float))
        test_base_prior = base_mapper.predict(-pd.to_numeric(test.base_timestamp_rank, errors="raise").to_numpy(float))
        targets = {
            "M_mc1_pair_residual": pd.to_numeric(train.policy_net_bps, errors="raise") - mc1_prior,
            "B_base_rank_residual_to_mc1": pd.to_numeric(train.policy_net_bps, errors="raise") - train_base_prior,
        }
        for family, target in targets.items():
            model = _fit_residual(train, features, target)
            prediction = model.predict(test.loc[:, features])
            lo, hi = np.quantile(target.to_numpy(float), [.02, .98])
            prediction = np.clip(prediction, lo, hi)
            if not np.isfinite(prediction).all():
                raise AssertionError(f"{held:%Y-%m}/{family}: non-finite residual prediction")
            for weight in RESIDUAL_WEIGHTS:
                arm = f"{family}_w{int(weight * 100):03d}"
                selection, trace = _select_adjusted(test, arm=arm, prediction=prediction, weight=weight)
                selection["held_month"] = held.strftime("%Y-%m")
                selections.setdefault(arm, []).append(selection)
                trace["held_month"] = held.strftime("%Y-%m")
                traces.append(trace)
            scored.append(pd.DataFrame({
                "candidate_id": test.candidate_id.astype(str), "__decision_ts__": test.__decision_ts__, "family": family,
                "base_prequential_expected_bps": test_base_prior, "residual_ev_bps": prediction,
                "residual_clip_low_bps": lo, "residual_clip_high_bps": hi, "held_month": held.strftime("%Y-%m"),
            }))
    output.mkdir(parents=True, exist_ok=False)
    summaries: list[dict[str, object]] = []
    # The ordinary BCF control is copied from its immutable target-free source.
    selections["B0_bcf_top2"] = [tables["B0_bcf_top2"]]
    # Keep E2 itself in the comparison, with its existing no-new-candidate
    # intersection authority.
    e2_ids = set(tables["H0_q50_d3"].candidate_id).intersection(tables["H3_q50_d2"].candidate_id)
    selections["E2_q50_agreement"] = [tables["H0_q50_d3"].loc[tables["H0_q50_d3"].candidate_id.isin(e2_ids)].copy()]
    for arm, frames in selections.items():
        selection = pd.concat(frames, ignore_index=True)
        selection["candidate_id"] = selection.candidate_id.astype(str)
        if selection.candidate_id.duplicated().any():
            raise AssertionError(f"{arm}: duplicated selected candidate")
        selection.to_parquet(output / f"{arm}_selection_target_free.parquet", index=False, compression="zstd")
        summaries.extend(_scope_replay(selection, labels, arm, output))
    summary = pd.DataFrame(summaries)
    summary["total_ev_per_abs_drawdown"] = summary.total_policy_net_bps / summary.max_drawdown.abs().replace(0.0, np.nan)
    for scope, group in summary.groupby("evaluation_scope", sort=False):
        reference = group.loc[group.model_arm.eq("E2_q50_agreement")]
        if len(reference) != 1:
            raise AssertionError(f"{scope}: E2 reference missing")
        for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown", "worst_week", "sortino", "total_ev_per_abs_drawdown"):
            summary.loc[group.index, f"delta_vs_E2_{metric}"] = group[metric] - reference.iloc[0][metric]
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    ranking = summary.loc[summary.evaluation_scope.eq("selection_jun_jul") & ~summary.model_arm.eq("B0_bcf_top2")].sort_values(
        ["total_ev_per_abs_drawdown", "total_policy_net_bps", "policy_net_bps_per_trade", "worst_week"], ascending=[False, False, False, False], kind="stable"
    )
    ranking.to_parquet(output / "selection_ranking_jun_jul.parquet", index=False)
    pd.concat(traces, ignore_index=True).to_parquet(output / "action_trace_target_free.parquet", index=False, compression="zstd")
    pd.concat(scored, ignore_index=True).to_parquet(output / "residual_scores_target_free.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict-r3-p8u-entry-e2-demotion-residual-v1",
        "scope": "offline strict-OOS research only; no live/canonical mutation, exchange IO, or order submission",
        "selection_period": "2026-06 through 2026-07 only; August untouched until selection is frozen",
        "held_months": [f"{value:%Y-%m}" for value in held_months],
        "e2_demotion_control": "BCF top-two incumbents only; remove only rows displaced by both/either pre-existing target-free q50 heads; never add a reserve",
        "mc1_residual_target": "rich-policy net bps minus arithmetic mean of contemporaneous BCF and current MC1 expected bps",
        "base_residual_target": "rich-policy net bps minus preceding-fold isotonic expected bps map of -base_timestamp_rank",
        "residual_application": "add clipped residual prediction equally to both MC1 expected bps; require both adjusted MC1 values >=30 bps; route top two by adjusted BCF MC1",
        "residual_weights": RESIDUAL_WEIGHTS,
        "residual_model": {"objective": "L1 mean", "source_geometry": "H0_q50_d3_l7", "seed": SEED},
        "feature_contract": {"projection": "candidate-level roots of E3_vwap_fs selected fields plus base_timestamp_rank", "source": str(feature_file), "sha256": _sha256(feature_file)},
        "training": f"up to {args.train_months} preceding complete calendar months; labels resolve before held boundary",
        "outcome_contract": "all selection/score receipts are target-free; policy labels join only after selection; cost embedded once",
        "inputs": {"hpo_root": str(hpo_root), "old_panel": str(feature_study.OLD_PANEL), "vwap_panel": str(feature_study.VWAP_PANEL), "label_root": str(base.LABEL_ROOT)},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
