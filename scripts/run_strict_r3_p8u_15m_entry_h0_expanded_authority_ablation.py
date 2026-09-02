#!/usr/bin/env python3
"""Strict-OOS H0 expanded-authority entry-head ablation.

The frozen H0 q50/d3/l7 model originally saw only 20--30 bps reserve rows and
could replace the marginal BCF incumbent.  This research-only successor keeps
the same model geometry and E3 feature contracts, but trains a *relative
advantage* model for every eligible candidate against the BCF marginal core
candidate at the same timestamp.  Consequently its authority can be tested in
three mutually visible ways, without looking at held outcomes:

* demote a weak BCF core candidate;
* replace a marginal core candidate with a strong reserve candidate; and
* promote a strong reserve candidate into capacity made available by a
  demotion.

All decisions are made on target-free held rows.  Policy labels are used only
in prior-resolved fold training and after selection for the normal portfolio
outcome replay.  June--July select an authority arm; August is held out.
This is deliberately an offline challenger and never changes live/canonical
artifacts.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS, VWAP_15M_FEATURE_KEYS
from scripts import run_strict_r3_p8u_15m_entry_feature_contract_ablation as feature_study
from scripts import run_strict_r3_p8u_15m_entry_pairwise_replacement_ablation as base
from scripts import run_strict_r3_p8u_15m_entry_postfs_hpo as h0


DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_h0_expanded_authority_20260830_v1"
FROZEN_H0_SELECTION = ROOT / "data_perp/artifacts/strict_r3_p8u_15m_entry_postfs_hpo_20260830_v1/H0_q50_d3_l7_baseline_selection_target_free.parquet"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")
SEED = 1729

# All arms keep a maximum of two candidate rows per timestamp.  The two
# weights act only on the candidate-vs-anchor H0 advantage in bps: positive
# evidence promotes, negative evidence demotes.  Floors are predeclared in
# predicted advantage bps.  A core candidate below -demotion_floor is removed;
# a reserve candidate needs promotion_floor before it may enter the top two.
AUTHORITY_ARMS: dict[str, dict[str, float]] = {
    "A1_sym025_p50_d50": {"promote_weight": .25, "demote_weight": .25, "promotion_floor": 50., "demotion_floor": 50.},
    "A2_sym050_p50_d50": {"promote_weight": .50, "demote_weight": .50, "promotion_floor": 50., "demotion_floor": 50.},
    "A3_sym100_p50_d50": {"promote_weight": 1.00, "demote_weight": 1.00, "promotion_floor": 50., "demotion_floor": 50.},
    "A4_demoteheavy_p25_d25": {"promote_weight": .50, "demote_weight": 1.00, "promotion_floor": 25., "demotion_floor": 25.},
    "A5_promoteheavy_p25_d25": {"promote_weight": 1.00, "demote_weight": .50, "promotion_floor": 25., "demotion_floor": 25.},
    "A6_sym100_p0_d0": {"promote_weight": 1.00, "demote_weight": 1.00, "promotion_floor": 0., "demotion_floor": 0.},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _feature_map(path: Path, held_months: tuple[pd.Timestamp, ...]) -> dict[pd.Timestamp, tuple[str, ...]]:
    return h0._features_by_month(path, "E3_vwap_fs", held_months)


def _numeric(value: object) -> float:
    result = float(pd.to_numeric(value, errors="coerce"))
    if not np.isfinite(result):
        raise ValueError("non-finite input to H0 authority pair")
    return result


def _feature_value(value: object) -> float:
    """Match the existing H0 contract: LightGBM receives feature NaNs natively."""
    return float(pd.to_numeric(value, errors="coerce"))


def _anchor_pairs(frame: pd.DataFrame, raw_features: tuple[str, ...], *, require_labels: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build candidate-vs-marginal-core rows and a target-free coverage trace."""
    records: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    for timestamp, group in frame.groupby("__decision_ts__", sort=True):
        anchor = base._marginal_incumbent(group)
        if anchor is None:
            coverage.append({"__decision_ts__": timestamp, "status": "no_core_anchor", "rows": len(group)})
            continue
        anchor_id = str(anchor.candidate_id)
        emitted = 0
        for _, candidate in group.iterrows():
            try:
                candidate_id = str(candidate.candidate_id)
                role = "anchor" if candidate_id == anchor_id else (
                    "core" if _numeric(candidate.dual_mc1_min_bps) >= base.CORE_FLOOR else "reserve"
                )
                row: dict[str, object] = {
                    "candidate_id": candidate_id,
                    "anchor_candidate_id": anchor_id,
                    "__decision_ts__": timestamp,
                    "__symbol__": str(candidate.__symbol__),
                    "candidate_role": role,
                    "candidate_bcf_mc1_expected_bps": _numeric(candidate.bcf_mc1_expected_bps),
                    "candidate_bcf_final_score": _numeric(candidate.bcf_final_score),
                    "candidate_dual_mc1_min_bps": _numeric(candidate.dual_mc1_min_bps),
                    "anchor_bcf_mc1_expected_bps": _numeric(anchor.bcf_mc1_expected_bps),
                }
                for feature in raw_features:
                    candidate_value = _feature_value(candidate[feature])
                    anchor_value = _feature_value(anchor[feature])
                    row[feature] = candidate_value
                    row[f"margin__{feature}"] = candidate_value - anchor_value
                # Historical E3 calls this input incumbent_*.  Here it is
                # exactly the contemporaneous marginal BCF anchor.
                row["incumbent_bcf_mc1_expected_bps"] = _numeric(anchor.bcf_mc1_expected_bps)
                if require_labels:
                    row["relative_advantage_bps"] = _numeric(candidate.policy_net_bps) - _numeric(anchor.policy_net_bps)
                    row["relative_label_available_ts"] = max(
                        pd.Timestamp(candidate.policy_label_available_ts),
                        pd.Timestamp(anchor.policy_label_available_ts),
                    )
                records.append(row)
                emitted += 1
            except (TypeError, ValueError, KeyError):
                continue
        coverage.append({"__decision_ts__": timestamp, "status": "ok" if emitted else "no_finite_pair_rows", "rows": len(group), "emitted": emitted})
    return pd.DataFrame(records), pd.DataFrame(coverage)


def _fit(train: pd.DataFrame, features: tuple[str, ...]) -> lgb.LGBMRegressor:
    spec = h0.SPECS["H0_q50_d3_l7_baseline"]
    child = max(8, int(np.ceil(len(train) * float(spec["min_child_fraction"]))))
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=float(spec["alpha"]), n_estimators=int(spec["n_estimators"]),
        learning_rate=float(spec["learning_rate"]), max_depth=int(spec["max_depth"]),
        num_leaves=int(spec["num_leaves"]), min_child_samples=child, subsample=.80,
        colsample_bytree=.80, reg_lambda=float(spec["reg_lambda"]), random_state=SEED,
        n_jobs=2, verbosity=-1,
    )
    dual = pd.to_numeric(train.candidate_dual_mc1_min_bps, errors="raise").to_numpy(float)
    reserve_position = np.clip((dual - base.RESERVE_FLOOR) / (base.CORE_FLOOR - base.RESERVE_FLOOR), 0.0, 1.0)
    # Preserve H0's greater attention to the upper part of the reserve band;
    # core rows remain unit weighted so they supply demotion calibration rather
    # than dominating the fit.
    sample_weight = np.where(dual < base.CORE_FLOOR, 1.0 + reserve_position, 1.0)
    model.fit(train.loc[:, features], pd.to_numeric(train.relative_advantage_bps, errors="raise"), sample_weight=sample_weight)
    return model


def _select(frame: pd.DataFrame, *, arm: str, config: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose at most two target-free rows per timestamp under declared authority."""
    selected: list[pd.DataFrame] = []
    trace: list[pd.DataFrame] = []
    for timestamp, group in frame.groupby("__decision_ts__", sort=True):
        work = group.copy()
        head = pd.to_numeric(work.h0_relative_advantage_bps, errors="raise").to_numpy(float)
        # The anchor's defined advantage over itself is exactly zero.  Pinning
        # it avoids an extrapolating model prediction turning the reference
        # candidate into a spurious promotion/demotion signal.
        head[work.candidate_role.eq("anchor").to_numpy()] = 0.0
        work["h0_relative_advantage_bps"] = head
        is_reserve = work.candidate_role.eq("reserve").to_numpy()
        gate = np.where(
            is_reserve,
            head >= float(config["promotion_floor"]),
            head >= -float(config["demotion_floor"]),
        )
        work["authority_eligible"] = gate
        adjustment = (
            float(config["promote_weight"]) * np.maximum(head, 0.0)
            - float(config["demote_weight"]) * np.maximum(-head, 0.0)
        )
        work["authority_adjustment_bps"] = adjustment
        work["authority_priority_bps"] = pd.to_numeric(work.candidate_bcf_mc1_expected_bps, errors="raise").to_numpy(float) + adjustment
        original = work.loc[work.candidate_role.ne("reserve")].sort_values(
            ["candidate_bcf_mc1_expected_bps", "candidate_bcf_final_score", "candidate_id"],
            ascending=[False, False, True], kind="stable",
        ).head(base.MAX_NEW_ENTRIES).copy()
        original_ids = set(original.candidate_id.astype(str))
        eligible = work.loc[work.authority_eligible].sort_values(
            ["authority_priority_bps", "candidate_bcf_mc1_expected_bps", "candidate_id"],
            ascending=[False, False, True], kind="stable",
        ).head(base.MAX_NEW_ENTRIES).copy()
        selected_ids = set(eligible.candidate_id.astype(str))
        work["selected"] = work.candidate_id.astype(str).isin(selected_ids)
        work["original_bcf_top2"] = work.candidate_id.astype(str).isin(original_ids)
        work["action"] = "not_selected"
        work.loc[work.original_bcf_top2 & ~work.authority_eligible, "action"] = "demoted_by_head_gate"
        work.loc[work.original_bcf_top2 & work.authority_eligible & ~work.selected, "action"] = "demoted_by_weighted_priority"
        work.loc[work.selected & work.candidate_role.eq("reserve"), "action"] = "promoted_reserve"
        work.loc[work.selected & work.candidate_role.ne("reserve"), "action"] = "kept_core"
        if not eligible.empty:
            reserve_selected = eligible.candidate_role.eq("reserve").any()
            core_omitted = bool(original_ids.difference(selected_ids))
            if reserve_selected and core_omitted:
                work.loc[work.selected & work.candidate_role.eq("reserve"), "action"] = "replacement_reserve"
            elif reserve_selected:
                work.loc[work.selected & work.candidate_role.eq("reserve"), "action"] = "promotion_into_freed_capacity"
            eligible["authority_arm"] = arm
            selected.append(eligible)
        work["authority_arm"] = arm
        trace.append(work)
    chosen = pd.concat(selected, ignore_index=True) if selected else frame.iloc[0:0].copy()
    audit = pd.concat(trace, ignore_index=True) if trace else frame.iloc[0:0].copy()
    if chosen.candidate_id.duplicated().any() or chosen.groupby("__decision_ts__").size().gt(base.MAX_NEW_ENTRIES).any():
        raise AssertionError(f"{arm}: authority selection changed target identity/capacity contract")
    # No outcome field may leak into the held selection receipt.
    forbidden = [column for column in chosen if column.startswith("policy_") or "label_available" in column or "advantage_bps" in column and column != "h0_relative_advantage_bps"]
    if forbidden:
        raise AssertionError(f"{arm}: target-free selection retained outcome fields: {forbidden}")
    return chosen, audit


def _replay_scopes(selection: pd.DataFrame, labels: pd.DataFrame, arm: str, output: Path) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    decision_ts = pd.to_datetime(selection.__decision_ts__, utc=True, errors="raise")
    for scope, subset in (
        ("selection_jun_jul", selection.loc[decision_ts.lt(SELECTION_END)].copy()),
        ("august_holdout", selection.loc[decision_ts.ge(SELECTION_END)].copy()),
        ("all_oos", selection),
    ):
        if subset.empty:
            continue
        metrics = base._replay(subset, labels, f"{arm}__{scope}", output)
        metrics["model_arm"] = arm
        metrics["evaluation_scope"] = scope
        results.append(metrics)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-study", type=Path, default=h0.FEATURE_STUDY)
    parser.add_argument("--feature-panel", type=Path, default=feature_study.OLD_PANEL, help="frozen legacy target-free feature panel")
    parser.add_argument("--vwap-panel", type=Path, default=feature_study.VWAP_PANEL, help="target-free VWAP overlay with identical legacy values")
    parser.add_argument("--labels-root", type=Path, default=base.LABEL_ROOT)
    parser.add_argument("--frozen-h0-selection", type=Path, default=FROZEN_H0_SELECTION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; default Jun--Aug 2026")
    args = parser.parse_args()
    if args.train_months < 2:
        raise ValueError("strict training needs at least two complete preceding months")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    held_months = tuple(pd.Timestamp(f"{value}-01", tz="UTC") for value in args.held_month) if args.held_month else tuple(pd.date_range("2026-06-01", "2026-08-01", freq="MS", tz="UTC"))
    raw_features = (*FIFTEEN_MINUTE_FEATURE_KEYS, *feature_study.SCORE_FEATURES, *VWAP_15M_FEATURE_KEYS)
    selected_features = _feature_map(args.feature_study.resolve() / "stable_selected_features.parquet", held_months)
    target_free = feature_study._candidate_frame(
        feature_study._load_panel(args.feature_panel.resolve(), args.vwap_panel.resolve())
    )
    labels = base._labels(args.labels_root.resolve())
    labelled = target_free.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    labelled = labelled.loc[labelled.policy_path_valid.fillna(False)].copy()
    labelled["policy_label_available_ts"] = pd.to_datetime(labelled.policy_label_available_ts, utc=True, errors="raise")
    output.mkdir(parents=True, exist_ok=False)
    selected_by_arm: dict[str, list[pd.DataFrame]] = {name: [] for name in AUTHORITY_ARMS}
    scored_rows: list[pd.DataFrame] = []
    traces: list[pd.DataFrame] = []
    coverage_rows: list[pd.DataFrame] = []
    controls: list[pd.DataFrame] = []
    for held in held_months:
        end = held + pd.offsets.MonthBegin(1)
        start = held - pd.DateOffset(months=args.train_months)
        training = labelled.loc[labelled.__decision_ts__.ge(start) & labelled.__decision_ts__.lt(held)].copy()
        train_pairs, train_coverage = _anchor_pairs(training, raw_features, require_labels=True)
        train_pairs = train_pairs.loc[pd.to_datetime(train_pairs.relative_label_available_ts, utc=True, errors="raise").lt(held)].copy()
        observed = set(pd.to_datetime(train_pairs.__decision_ts__, utc=True).dt.strftime("%Y-%m")) if not train_pairs.empty else set()
        required = {(held - pd.DateOffset(months=offset)).strftime("%Y-%m") for offset in range(1, args.train_months + 1)}
        test = target_free.loc[target_free.__decision_ts__.ge(held) & target_free.__decision_ts__.lt(end)].copy()
        test_pairs, test_coverage = _anchor_pairs(test, raw_features, require_labels=False)
        features = selected_features[held]
        missing = set(features).difference(train_pairs.columns) | set(features).difference(test_pairs.columns)
        if missing or len(train_pairs) < 200 or test_pairs.empty or not required.issubset(observed):
            raise RuntimeError(f"{held:%Y-%m}: strict H0 authority support missing; fields={sorted(missing)} rows={len(train_pairs)}")
        model = _fit(train_pairs, features)
        scored = test_pairs.loc[:, ["candidate_id", "anchor_candidate_id", "__decision_ts__", "__symbol__", "candidate_role", "candidate_bcf_mc1_expected_bps", "candidate_bcf_final_score", "candidate_dual_mc1_min_bps", "anchor_bcf_mc1_expected_bps"]].copy()
        # Preserve the ordinary target-free field names expected by the common
        # BCF-priority portfolio replay; these are aliases, not recalculated
        # scores or outcome-bearing values.
        scored["bcf_mc1_expected_bps"] = scored.candidate_bcf_mc1_expected_bps
        scored["bcf_final_score"] = scored.candidate_bcf_final_score
        scored["dual_mc1_min_bps"] = scored.candidate_dual_mc1_min_bps
        scored["h0_relative_advantage_bps"] = model.predict(test_pairs.loc[:, features])
        if not np.isfinite(scored.h0_relative_advantage_bps).all():
            raise AssertionError(f"{held:%Y-%m}: non-finite H0 prediction")
        scored["held_month"] = held.strftime("%Y-%m")
        scored_rows.append(scored)
        for name, config in AUTHORITY_ARMS.items():
            chosen, audit = _select(scored, arm=name, config=config)
            chosen["held_month"] = held.strftime("%Y-%m")
            selected_by_arm[name].append(chosen)
            audit["held_month"] = held.strftime("%Y-%m")
            traces.append(audit)
        controls.append(base._incumbent_top2(test).assign(held_month=held.strftime("%Y-%m")))
        train_coverage["held_month"], train_coverage["phase"] = held.strftime("%Y-%m"), "train"
        test_coverage["held_month"], test_coverage["phase"] = held.strftime("%Y-%m"), "held"
        coverage_rows.extend([train_coverage, test_coverage])
    frozen_h0 = pd.read_parquet(args.frozen_h0_selection.resolve())
    frozen_h0["candidate_id"] = frozen_h0.candidate_id.astype(str)
    held_ids = set(pd.concat(controls, ignore_index=True).candidate_id.astype(str)) | set(pd.concat(scored_rows, ignore_index=True).candidate_id.astype(str))
    frozen_h0 = frozen_h0.loc[frozen_h0.candidate_id.isin(held_ids)].copy()
    # The control must only contain held months and must remain target-free.
    frozen_h0["__decision_ts__"] = pd.to_datetime(frozen_h0.__decision_ts__, utc=True, errors="raise")
    controls.append(frozen_h0.assign(_frozen_control="H0_replace_only"))
    results: list[dict[str, object]] = []
    bcf_control = pd.concat(controls[:-1], ignore_index=True)
    if bcf_control.candidate_id.duplicated().any():
        raise AssertionError("BCF control candidate identity duplicated")
    bcf_control.to_parquet(output / "B0_bcf_top2_selection_target_free.parquet", index=False, compression="zstd")
    results.extend(_replay_scopes(bcf_control, labels, "B0_bcf_top2", output))
    frozen_h0.to_parquet(output / "R0_frozen_h0_replace_only_selection_target_free.parquet", index=False, compression="zstd")
    results.extend(_replay_scopes(frozen_h0, labels, "R0_frozen_h0_replace_only", output))
    for name, frames in selected_by_arm.items():
        selection = pd.concat(frames, ignore_index=True)
        selection.to_parquet(output / f"{name}_selection_target_free.parquet", index=False, compression="zstd")
        results.extend(_replay_scopes(selection, labels, name, output))
    summary = pd.DataFrame(results)
    summary["total_ev_per_abs_drawdown"] = summary.total_policy_net_bps / summary.max_drawdown.abs().replace(0.0, np.nan)
    for scope, group in summary.groupby("evaluation_scope", sort=False):
        baseline = group.loc[group.model_arm.eq("R0_frozen_h0_replace_only")]
        if len(baseline) != 1:
            raise AssertionError(f"{scope}: exact frozen H0 control missing")
        for metric in ("portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown", "worst_week", "sortino", "total_ev_per_abs_drawdown"):
            summary.loc[group.index, f"delta_vs_R0_{metric}"] = group[metric] - baseline.iloc[0][metric]
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    ranking = summary.loc[summary.evaluation_scope.eq("selection_jun_jul") & summary.model_arm.isin(AUTHORITY_ARMS)].sort_values(
        ["total_ev_per_abs_drawdown", "total_policy_net_bps", "policy_net_bps_per_trade", "worst_week"], ascending=[False, False, False, False], kind="stable"
    )
    ranking.to_parquet(output / "selection_ranking_jun_jul.parquet", index=False)
    pd.concat(scored_rows, ignore_index=True).to_parquet(output / "scored_target_free.parquet", index=False, compression="zstd")
    pd.concat(traces, ignore_index=True).to_parquet(output / "authority_action_trace_target_free.parquet", index=False, compression="zstd")
    pd.concat(coverage_rows, ignore_index=True).to_parquet(output / "anchor_pair_coverage.parquet", index=False)
    manifest = {
        "schema": "strict-r3-p8u-15m-entry-h0-expanded-authority-v1",
        "scope": "offline strict-OOS research challenger only; no live/canonical mutation, exchange IO, or order submission",
        "selection_period": "2026-06 through 2026-07 only; August untouched until selection is frozen",
        "held_months": [f"{held:%Y-%m}" for held in held_months],
        "model": {"source": "H0_q50_d3_l7", "spec": h0.SPECS["H0_q50_d3_l7_baseline"], "seed": SEED},
        "training": f"up to {args.train_months} trailing complete calendar months; every candidate and anchor label resolves before held boundary",
        "target": "candidate rich-policy net bps minus contemporaneous BCF marginal-core candidate rich-policy net bps",
        "authority": "target-free score chooses at most two rows/timestamp; positive score may promote/replacement reserve rows, negative score may demote core rows",
        "authority_arms": AUTHORITY_ARMS,
        "feature_selection": {"variant": "E3_vwap_fs", "path": str((args.feature_study.resolve() / "stable_selected_features.parquet")), "sha256": _sha256(args.feature_study.resolve() / "stable_selected_features.parquet")},
        "inputs": {
            "legacy_feature_panel": str(args.feature_panel.resolve()),
            "legacy_feature_panel_sha256": _sha256(args.feature_panel.resolve()),
            "vwap_feature_panel": str(args.vwap_panel.resolve()),
            "vwap_feature_panel_sha256": _sha256(args.vwap_panel.resolve()),
            "labels_root": str(args.labels_root.resolve()),
            "frozen_h0_selection": str(args.frozen_h0_selection.resolve()),
            "frozen_h0_selection_sha256": _sha256(args.frozen_h0_selection.resolve()),
        },
        "outcome_contract": "held selection receipts remain target-free; labels join only in normal portfolio replay; 100 bps embedded exactly once",
        "portfolio_priority": "unchanged BCF MC1 expected bps in normal global auction after timestamp authority selection",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
