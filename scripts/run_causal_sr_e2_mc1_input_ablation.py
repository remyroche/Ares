#!/usr/bin/env python3
"""Strict-OOS MC1 input ablation: causal S/R, all-candidate 15m E2, and both.

This is an offline challenger.  It holds fixed the target-free candidate
stream, source-aligned rich-policy labels, paired BCF/current MC1 residual
target, residual model, dual-admission rule and global portfolio auction.

``E2_15m_direct`` is deliberately *not* the canonical pairwise E2 entry
authority.  It is a fresh all-candidate 70-feature 15-minute L1 policy-EV
head.  Within each outer MC1 fold it is fitted on prior-resolved labels, and
its training-row inputs are strictly prequential raw predictions.  It is
supplied to MC1 as an input only: it cannot route, promote, veto or
otherwise alter candidates outside the residual mapper.
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

from extreme_price_movements.p8u_15m_features import FIFTEEN_MINUTE_FEATURE_KEYS
from scripts import run_causal_sr_mc1_residual_ablation as sr
from scripts import run_strict_r3_p8u_15m_entry_e2_demotion_residual_ablation as control
from scripts import run_strict_r3_p8u_15m_entry_feature_contract_ablation as feature_study
from scripts import run_strict_r3_p8u_15m_entry_pairwise_replacement_ablation as base


DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/causal_sr_e2_mc1_input_ablation_20260831_v1"
SELECTION_END = pd.Timestamp("2026-08-01", tz="UTC")
E2_FEATURES = tuple(FIFTEEN_MINUTE_FEATURE_KEYS)
E2_OUTPUT = "e2_15m_prequential_raw_policy_bps"
E2_AVAILABLE = "e2_15m_prequential_available"
E2_MIN_OUTER_ROWS = 500
E2_MIN_INNER_ROWS = 200
E2_MIN_PREQUENTIAL_ROWS = 200
SEED = 1729


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _month_set(frame: pd.DataFrame) -> set[str]:
    return set(pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise").dt.strftime("%Y-%m"))


def _require_prior_months(frame: pd.DataFrame, held: pd.Timestamp, months: int) -> None:
    required = {(held - pd.DateOffset(months=n)).strftime("%Y-%m") for n in range(1, months + 1)}
    if not required.issubset(_month_set(frame)):
        raise RuntimeError(f"{held:%Y-%m}: required prior months absent: {sorted(required - _month_set(frame))}")


def _fit_e2_direct(train: pd.DataFrame) -> lgb.LGBMRegressor:
    """Fit the retrained all-candidate E2 15m head on prior labels only."""
    if len(train) < E2_MIN_INNER_ROWS:
        raise RuntimeError("E2 direct head has insufficient prior-resolved support")
    model = lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=350, learning_rate=.03,
        max_depth=4, num_leaves=15,
        min_child_samples=max(8, int(np.ceil(len(train) * .02))),
        subsample=.80, colsample_bytree=.80, reg_lambda=4.0,
        random_state=SEED, n_jobs=2, verbosity=-1,
    )
    target = pd.to_numeric(train["policy_net_bps"], errors="raise").to_numpy(float)
    model.fit(train.loc[:, E2_FEATURES], target)
    return model


def _prequential_e2_head(
    train: pd.DataFrame, test: pd.DataFrame, held: pd.Timestamp,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    """Return a held score plus train-row prequential scores for MC1 fitting.

    A training row may receive the E2 input only when its direct 15m prediction
    was fitted strictly before that row's month.  The raw L1 output is used
    rather than a same-row label calibration: this keeps the feature's training
    and held inference semantics identical and prevents target leakage through
    an in-sample calibration transform.
    """
    if len(train) < E2_MIN_OUTER_ROWS:
        raise RuntimeError(f"{held:%Y-%m}: E2 direct outer train has only {len(train)} rows")
    _require_prior_months(train, held, 4)
    first = held - pd.DateOffset(months=4)
    inner_months = pd.date_range(first + pd.DateOffset(months=2), held - pd.offsets.MonthBegin(1), freq="MS", tz="UTC")
    oof_rows: list[pd.DataFrame] = []
    for inner in inner_months:
        inner_train = train.loc[
            train["__decision_ts__"].lt(inner)
            & train["policy_label_available_ts"].lt(inner)
        ].copy()
        inner_test = train.loc[
            train["__decision_ts__"].ge(inner)
            & train["__decision_ts__"].lt(inner + pd.offsets.MonthBegin(1))
        ].copy()
        if inner_test.empty or len(inner_train) < E2_MIN_INNER_ROWS:
            continue
        _require_prior_months(inner_train, inner, 2)
        inner_model = _fit_e2_direct(inner_train)
        oof_rows.append(pd.DataFrame({
            "candidate_id": inner_test["candidate_id"].astype(str).to_numpy(),
            E2_OUTPUT: inner_model.predict(inner_test.loc[:, E2_FEATURES]),
            "__decision_ts__": inner_test["__decision_ts__"].to_numpy(),
        }))
    if not oof_rows:
        raise RuntimeError(f"{held:%Y-%m}: no prequential E2 calibration rows")
    oof = pd.concat(oof_rows, ignore_index=True)
    if len(oof) < E2_MIN_PREQUENTIAL_ROWS:
        raise RuntimeError(f"{held:%Y-%m}: insufficient prequential E2 feature rows")
    if oof["candidate_id"].duplicated().any() or not pd.to_datetime(oof["__decision_ts__"], utc=True, errors="raise").lt(held).all():
        raise AssertionError("E2 calibration contains an outer-held timestamp")
    final_model = _fit_e2_direct(train)
    held_raw = final_model.predict(test.loc[:, E2_FEATURES])
    if not np.isfinite(held_raw).all() or not np.isfinite(oof[E2_OUTPUT].to_numpy(float)).all():
        raise AssertionError(f"{held:%Y-%m}: non-finite all-candidate E2 head output")
    trace = {
        "e2_prequential_train_feature_rows": int(len(oof)),
        "e2_prequential_train_feature_months": sorted(pd.to_datetime(oof["__decision_ts__"], utc=True).dt.strftime("%Y-%m").unique().tolist()),
        "e2_prequential_latest_score_ts": str(pd.to_datetime(oof["__decision_ts__"], utc=True).max()),
    }
    return held_raw, oof.loc[:, ["candidate_id", E2_OUTPUT]], trace


def _scope_replay(selection: pd.DataFrame, labels: pd.DataFrame, arm: str, output: Path) -> list[dict[str, object]]:
    output_rows: list[dict[str, object]] = []
    timestamp = pd.to_datetime(selection["__decision_ts__"], utc=True, errors="raise")
    for scope, subset in (
        ("selection_jun_jul", selection.loc[timestamp.lt(SELECTION_END)].copy()),
        ("august_holdout", selection.loc[timestamp.ge(SELECTION_END)].copy()),
        ("all_oos", selection),
    ):
        if subset.empty:
            continue
        metric = base._replay(subset, labels, f"{arm}__{scope}", output)
        metric["model_arm"], metric["evaluation_scope"] = arm, scope
        output_rows.append(metric)
    return output_rows


def _assert_target_free_selection(selection: pd.DataFrame, arm: str) -> None:
    if selection["candidate_id"].duplicated().any():
        raise AssertionError(f"{arm}: selected candidate identity is duplicated")
    if selection.groupby("__decision_ts__", sort=False).size().gt(base.MAX_NEW_ENTRIES).any():
        raise AssertionError(f"{arm}: selection breached the two-entry timestamp cap")
    forbidden = [name for name in selection if name.startswith("policy_") or "label_available" in name]
    if forbidden:
        raise AssertionError(f"{arm}: target-free selection contains outcome fields {forbidden}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sr-root", type=Path, default=sr.SR_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--label-read-workers", type=int, default=8)
    parser.add_argument("--residual-weight", type=float, action="append", default=[], help="repeatable mapper authority; default 1.0")
    parser.add_argument("--held-month", action="append", help="repeatable YYYY-MM; default Jun--Aug 2026")
    args = parser.parse_args()
    if args.train_months != 4:
        raise ValueError("this matched E2 input study fixes the four-month MC1 train contract")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"immutable output exists: {output}")
    weights = tuple(args.residual_weight) if args.residual_weight else (1.0,)
    if not weights or any(weight <= 0.0 or weight > 1.0 for weight in weights):
        raise ValueError("residual authority must be in (0, 1]")
    held_months = (
        tuple(pd.Timestamp(f"{token}-01", tz="UTC") for token in args.held_month)
        if args.held_month else tuple(pd.date_range("2026-06-01", "2026-08-01", freq="MS", tz="UTC"))
    )
    base_features = control._candidate_features(control.h0.FEATURE_STUDY / "stable_selected_features.parquet", held_months)
    raw_target_free = feature_study._candidate_frame(
        feature_study._load_panel(feature_study.OLD_PANEL, feature_study.VWAP_PANEL)
    )
    target_free, sr_coverage = sr._merge_causal_sr(raw_target_free, args.sr_root)
    labels, unavailable_label_parts = sr._source_aligned_labels(base.LABEL_ROOT, workers=args.label_read_workers)
    unavailable_symbols = frozenset(item["symbol"] for item in unavailable_label_parts)
    if unavailable_symbols:
        target_free = target_free.loc[~target_free["__symbol__"].isin(unavailable_symbols)].copy()
    labelled = target_free.merge(labels, on="candidate_id", how="inner", validate="one_to_one")
    labelled = labelled.loc[labelled["policy_path_valid"].fillna(False)].copy()
    labelled["policy_label_available_ts"] = pd.to_datetime(labelled["policy_label_available_ts"], utc=True, errors="raise")
    missing_15m = set(E2_FEATURES).difference(target_free.columns)
    if missing_15m:
        raise AssertionError(f"target-free candidate panel lacks all-candidate E2 features: {sorted(missing_15m)}")
    output.mkdir(parents=True, exist_ok=False)

    arm_extras = {
        "M_mc1_pair_residual_control": (),
        "M_mc1_pair_residual_plus_causal_sr": (*sr.SR_FEATURES, "sr_snapshot_available"),
        "M_mc1_pair_residual_plus_e2_15m": (E2_OUTPUT, E2_AVAILABLE),
        "M_mc1_pair_residual_plus_causal_sr_e2_15m": (*sr.SR_FEATURES, "sr_snapshot_available", E2_OUTPUT, E2_AVAILABLE),
    }
    selections: dict[str, list[pd.DataFrame]] = {name: [] for name in arm_extras}
    traces: list[pd.DataFrame] = []
    residual_scores: list[pd.DataFrame] = []
    e2_scores: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []

    for held in held_months:
        end, start = held + pd.offsets.MonthBegin(1), held - pd.DateOffset(months=args.train_months)
        train = labelled.loc[
            labelled["__decision_ts__"].ge(start)
            & labelled["__decision_ts__"].lt(held)
            & labelled["policy_label_available_ts"].lt(held)
        ].copy()
        test = target_free.loc[
            target_free["__decision_ts__"].ge(held) & target_free["__decision_ts__"].lt(end)
        ].copy()
        _require_prior_months(train, held, args.train_months)
        if len(train) < E2_MIN_OUTER_ROWS or test.empty:
            raise RuntimeError(f"{held:%Y-%m}: strict MC1 fold lacks support")
        base_fields = base_features[held]
        missing_base = set(base_fields).difference(train.columns) | set(base_fields).difference(test.columns)
        if missing_base:
            raise AssertionError(f"{held:%Y-%m}: base MC1 features absent: {sorted(missing_base)}")
        train_ids = train["candidate_id"].astype(str).to_numpy()
        e2_raw, train_e2_oof, e2_trace = _prequential_e2_head(train, test, held)
        train = train.merge(train_e2_oof, on="candidate_id", how="left", validate="one_to_one", sort=False)
        if len(train) != len(train_ids) or not np.array_equal(train["candidate_id"].astype(str).to_numpy(), train_ids):
            raise AssertionError("E2 OOF train merge changed MC1 train identity")
        train[E2_AVAILABLE] = train[E2_OUTPUT].notna().astype(np.int8)
        test[E2_OUTPUT] = e2_raw
        test[E2_AVAILABLE] = np.int8(1)
        target = pd.to_numeric(train["policy_net_bps"], errors="raise") - (
            pd.to_numeric(train["bcf_mc1_expected_bps"], errors="raise")
            + pd.to_numeric(train["current_mc1_expected_bps"], errors="raise")
        ) / 2.0
        clip_low, clip_high = np.quantile(target.to_numpy(float), [.02, .98])
        for arm, extras in arm_extras.items():
            fields = (*base_fields, *extras)
            missing = set(fields).difference(train.columns) | set(fields).difference(test.columns)
            if missing:
                raise AssertionError(f"{held:%Y-%m}/{arm}: MC1 input fields absent: {sorted(missing)}")
            model = control._fit_residual(train, fields, target)
            residual = np.clip(model.predict(test.loc[:, fields]), clip_low, clip_high)
            if not np.isfinite(residual).all():
                raise AssertionError(f"{held:%Y-%m}/{arm}: non-finite MC1 residual output")
            for weight in weights:
                name = f"{arm}_w{int(round(weight * 100)):03d}"
                selection, trace = control._select_adjusted(test, arm=name, prediction=residual, weight=weight)
                _assert_target_free_selection(selection, name)
                selection["held_month"] = held.strftime("%Y-%m")
                selections[arm].append(selection.assign(arm=name))
                trace["held_month"], trace["mapper_arm"], trace["weight"] = held.strftime("%Y-%m"), arm, weight
                traces.append(trace)
                residual_scores.append(pd.DataFrame({
                    "candidate_id": test["candidate_id"].astype(str), "__decision_ts__": test["__decision_ts__"],
                    "mapper_arm": arm, "weight": weight, "mc1_residual_bps": residual,
                    "residual_clip_low_bps": clip_low, "residual_clip_high_bps": clip_high,
                    "held_month": held.strftime("%Y-%m"),
                }))
        e2_scores.append(pd.DataFrame({
            "candidate_id": test["candidate_id"].astype(str), "__decision_ts__": test["__decision_ts__"],
            "held_month": held.strftime("%Y-%m"), E2_OUTPUT: e2_raw, E2_AVAILABLE: np.int8(1),
        }))
        fold_rows.append({
            "held_month": held.strftime("%Y-%m"), "train_start": str(start), "train_rows": int(len(train)),
            "test_rows": int(len(test)), "train_sr_available": int(train["sr_snapshot_available"].sum()),
            "test_sr_available": int(test["sr_snapshot_available"].sum()),
            "residual_clip_low_bps": float(clip_low), "residual_clip_high_bps": float(clip_high), **e2_trace,
        })

    summary_rows: list[dict[str, object]] = []
    for arm, frames in selections.items():
        selected = pd.concat(frames, ignore_index=True)
        for weight in weights:
            name = f"{arm}_w{int(round(weight * 100)):03d}"
            subset = selected.loc[selected["arm"].eq(name)].copy()
            _assert_target_free_selection(subset, name)
            subset.to_parquet(output / f"{name}_selection_target_free.parquet", index=False, compression="zstd")
            summary_rows.extend(_scope_replay(subset, labels, name, output))
    summary = pd.DataFrame(summary_rows)
    summary["total_ev_per_abs_drawdown"] = summary["total_policy_net_bps"] / summary["max_drawdown"].abs().replace(0.0, np.nan)
    for scope, rows in summary.groupby("evaluation_scope", sort=False):
        reference = rows.loc[rows["model_arm"].eq("M_mc1_pair_residual_control_w100")]
        if len(reference) != 1:
            raise AssertionError(f"{scope}: control w100 missing")
        for metric in (
            "portfolio_accepted", "policy_net_bps_per_trade", "total_policy_net_bps", "max_drawdown",
            "worst_week", "sortino", "total_ev_per_abs_drawdown",
        ):
            summary.loc[rows.index, f"delta_vs_control_w100_{metric}"] = rows[metric] - reference.iloc[0][metric]
    summary.to_parquet(output / "portfolio_summary.parquet", index=False)
    sr_coverage.to_parquet(output / "sr_merge_coverage.parquet", index=False)
    pd.DataFrame(fold_rows).to_parquet(output / "fold_trace.parquet", index=False)
    pd.concat(traces, ignore_index=True).to_parquet(output / "admission_trace_target_free.parquet", index=False, compression="zstd")
    pd.concat(residual_scores, ignore_index=True).to_parquet(output / "mc1_residual_scores_target_free.parquet", index=False, compression="zstd")
    pd.concat(e2_scores, ignore_index=True).to_parquet(output / "e2_15m_scores_target_free.parquet", index=False, compression="zstd")
    sr_snapshot = sr._assert_causal_sr_root(args.sr_root)
    manifest = {
        "schema": "causal-sr-e2-mc1-input-ablation-v1",
        "scope": "offline strict-OOS challenger; no live/canonical/execution mutation",
        "isolated_change": "append causal S/R outputs, a retrained all-candidate 15m E2 output, or both to the frozen MC1 residual input projection",
        "control": "paired BCF/current MC1 residual target, L1 model geometry, residual clipping, dual >=30 admission and BCF-priority top-two route",
        "residual_target": "rich-policy net bps minus mean contemporaneous BCF/current MC1 expected bps",
        "e2_15m_head": {
            "kind": "all-candidate direct policy-EV input; not the pairwise E2 replacement authority",
            "features": list(E2_FEATURES), "target": "rich-policy net bps with its embedded 100-bps cost exactly once",
            "model": {"family": "LightGBM", "loss": "L1", "max_depth": 4, "num_leaves": 15, "n_estimators": 350, "learning_rate": .03, "reg_lambda": 4.0, "seed": SEED},
            "train_feature_provenance": "each non-missing training feature is an inner prequential score fitted strictly before that row's month; earlier rows remain unavailable/missing",
            "authority": "MC1 input only; no direct selection, routing, veto or capacity authority",
        },
        "s_r_source": str(args.sr_root.resolve()), "s_r_snapshot": str(sr_snapshot),
        "s_r_manifest_sha256": _sha256(args.sr_root.resolve() / "run_manifest.json"),
        "s_r_fields": list(sr.SR_FEATURES),
        "missing_s_r": "LightGBM missing input plus sr_snapshot_available; never candidate eligibility",
        "unavailable_source_aligned_label_parts_excluded_from_all_arms": unavailable_label_parts,
        "held_months": [f"{item:%Y-%m}" for item in held_months],
        "selection_period": "June--July 2026; August is holdout and not used to choose an arm",
        "training": "four complete prior calendar months; outer labels resolve before held boundary; E2 calibration uses inner prequential predictions only",
        "residual_weights": list(weights), "folds": fold_rows,
        "outcome_contract": "candidate selection is target-free; outcomes enter only prior training and post-selection portfolio replay; no exchange calls",
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
