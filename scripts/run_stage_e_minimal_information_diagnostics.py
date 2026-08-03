#!/usr/bin/env python3
"""Stage-E E2/E3 minimal-information and conditional-permutation diagnostics.

This runner deliberately stops at the already-consumed Stage-D development OOF
window.  It never reads, scores, tunes on, or otherwise opens a second-OOS
period.  All arms use the canonical Stage-D row population, purged folds,
seeds, target, side-local calibration procedure, zero-bps absolute action rule,
and evaluator.  Feature deletion is declared before each fold; preprocessing
and any feature selection are fit on that fold's training rows only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_stage_d_compact_action_model import (  # noqa: E402
    DEV_START,
    FEATURES,
    FINAL_START,
    GROUPS,
    HORIZON,
    MIN_TRAIN,
    SEED,
    SIDES,
    apply_preprocess,
    calibrate,
    diagnostics,
    fit_calibration,
    fit_model,
    fit_preprocess,
    idhash,
    load_frame,
    replay,
    sha,
    train_mask,
)

ART = ROOT / "data_perp/artifacts"
CANONICAL = ART / "stage_d_compact_action_model_20260731_v9"
CANONICAL_REPLAY = CANONICAL / "stage_d_action_policy_replay.parquet"
CANONICAL_MANIFEST = CANONICAL / "run_manifest.json"
DEFAULT_OUTPUT = ART / "stage_e_minimal_information_diagnostics_20260731_v1"
SCHEMA = "stage_e_minimal_information_diagnostics_v1"
MARGIN_BPS = 0.0
PERMUTATION_REPEATS = 5

ACTION_STATE = [
    "time_to_clear_minutes",
    "gross_return_at_action_bps",
    "estimated_net_if_exit_now_bps",
]
COST = [
    "known_row_cost_bps",
    "estimated_spread_bps",
    "entry_half_spread_bps",
    "exit_half_spread_bps",
]
POLICY_GEOMETRY = ["barrier_pct"]
IDENTITY = ["side_long"]
UPSTREAM_OUTPUTS: list[str] = []


def _sha(path: Path) -> str:
    return sha(path)


def _dump(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def feature_families(a0: list[str]) -> dict[str, list[str]]:
    """Return an exhaustive, disjoint Stage-E grouping of the sealed A0 list."""
    reserved = set(ACTION_STATE + COST + POLICY_GEOMETRY + IDENTITY + UPSTREAM_OUTPUTS)
    groups = {
        "action_state": [c for c in ACTION_STATE if c in a0],
        "entry_static": [c for c in a0 if c not in reserved],
        "cost": [c for c in COST if c in a0],
        "policy_geometry": [c for c in POLICY_GEOMETRY if c in a0],
        "upstream_model_outputs": [c for c in UPSTREAM_OUTPUTS if c in a0],
        "symbol_or_side_identity": [c for c in IDENTITY if c in a0],
    }
    flat = [c for values in groups.values() for c in values]
    if len(flat) != len(set(flat)) or set(flat) != set(a0):
        raise ValueError("Stage-E feature-family partition is not exhaustive/disjoint")
    return groups


def arm_contracts(a0: list[str], families: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    m2 = list(dict.fromkeys(ACTION_STATE + COST + POLICY_GEOMETRY))
    return {
        "M0_full_frozen_A0": {"features": a0, "model": "canonical_lgbm_huber"},
        "M1_three_action_state": {"features": ACTION_STATE, "model": "canonical_lgbm_huber"},
        "M2_action_state_cost_geometry": {"features": m2, "model": "canonical_lgbm_huber"},
        "M3_entry_static_only": {"features": families["entry_static"], "model": "canonical_lgbm_huber"},
        "M4_action_state_without_exit_net": {"features": ACTION_STATE[:2], "model": "canonical_lgbm_huber"},
        "M5_estimated_exit_net_only": {"features": ["estimated_net_if_exit_now_bps"], "model": "canonical_lgbm_huber"},
        "M6_time_to_clear_only": {"features": ["time_to_clear_minutes"], "model": "canonical_lgbm_huber"},
        "M7_gross_return_at_action_only": {"features": ["gross_return_at_action_bps"], "model": "canonical_lgbm_huber"},
        "M8a_ridge_M2": {"features": m2, "model": "ridge"},
        "M8b_logistic_M2": {"features": m2, "model": "logistic_continue_better"},
        "M8c_tree_depth2_M2": {"features": m2, "model": "tree_depth2"},
        "M8d_tree_depth3_M2": {"features": m2, "model": "tree_depth3"},
    }


def _fit_predict(model_kind: str, x_train: pd.DataFrame, y: pd.Series,
                 binary: pd.Series, x_test: pd.DataFrame, seed: int) -> np.ndarray:
    if model_kind == "canonical_lgbm_huber":
        return fit_model(x_train, y, seed).predict(x_test)
    if model_kind == "ridge":
        return Ridge(alpha=10.0).fit(x_train, y).predict(x_test)
    if model_kind == "logistic_continue_better":
        return LogisticRegression(C=1.0, max_iter=500, random_state=seed).fit(x_train, binary).predict_proba(x_test)[:, 1]
    if model_kind.startswith("tree_depth"):
        depth = int(model_kind.removeprefix("tree_depth"))
        return DecisionTreeRegressor(max_depth=depth, min_samples_leaf=80, random_state=seed).fit(x_train, y).predict(x_test)
    raise ValueError(f"unknown model kind: {model_kind}")


@dataclass
class FoldBundle:
    arm: str
    side: str
    fold: str
    test_index: np.ndarray
    selected_features: list[str]
    preprocess: dict[str, Any]
    predict: Callable[[pd.DataFrame], np.ndarray]


BASE_COLUMNS = [
    "candidate_id", "source_symbol", "side", "action_decision_ts",
    "time_to_clear_bucket", "volatility_bucket", "net_exit_now_gross_bps",
    "net_exit_now_cost_bps", "net_exit_now_bps", "net_continue_gross_bps",
    "net_continue_cost_bps", "net_continue_bps", "delta_continue_bps",
    "continue_better",
]


def score_arm(frame: pd.DataFrame, arm: str, features: list[str], model_kind: str,
              keep_bundles: bool = False) -> tuple[pd.DataFrame, list[dict[str, Any]], list[FoldBundle]]:
    """Score canonical development OOF folds; never access FINAL_START or later."""
    outputs: list[pd.DataFrame] = []
    states: list[dict[str, Any]] = []
    bundles: list[FoldBundle] = []
    months = pd.date_range(DEV_START, FINAL_START, freq="MS", inclusive="left")
    for side_index, side in enumerate(SIDES):
        side_frame = frame[frame.side.eq(side)]
        for fold_index, start in enumerate(months):
            end = start + pd.offsets.MonthBegin(1)
            train = side_frame[train_mask(side_frame, start)]
            test = side_frame[side_frame.action_decision_ts.ge(start) & side_frame.action_decision_ts.lt(end)]
            if len(train) < MIN_TRAIN or test.empty:
                continue
            seed = SEED + side_index * 100 + fold_index
            state = fit_preprocess(train, features, train.delta_continue_bps.to_numpy(), seed)
            if not state["selected"]:
                raise ValueError(f"{arm}/{side}/{start:%Y-%m}: no usable features")
            x_train = apply_preprocess(train, state)
            x_test = apply_preprocess(test, state)
            if model_kind == "canonical_lgbm_huber":
                model = fit_model(x_train, train.delta_continue_bps, seed)
            elif model_kind == "ridge":
                model = Ridge(alpha=10.0).fit(x_train, train.delta_continue_bps)
            elif model_kind == "logistic_continue_better":
                model = LogisticRegression(C=1.0, max_iter=500, random_state=seed).fit(x_train, train.continue_better)
            elif model_kind.startswith("tree_depth"):
                depth = int(model_kind.removeprefix("tree_depth"))
                model = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=80, random_state=seed).fit(x_train, train.delta_continue_bps)
            else:
                raise ValueError(model_kind)
            if model_kind == "logistic_continue_better":
                raw = model.predict_proba(x_test)[:, 1]
                predictor = lambda x, m=model: m.predict_proba(x)[:, 1]
            else:
                raw = model.predict(x_test)
                predictor = model.predict
            scored = test[BASE_COLUMNS].copy()
            scored["arm"] = arm
            scored["split"] = "development_oof"
            scored["fold"] = start.strftime("%Y-%m")
            scored["raw_predicted_delta_bps"] = raw
            outputs.append(scored)
            states.append({
                "arm": arm, "model_kind": model_kind, "side": side,
                "fold": start.strftime("%Y-%m"), "train_rows": len(train),
                "test_rows": len(test), "test_candidate_id_sha256": idhash(test.candidate_id),
                "train_max_action_decision_ts": str(train.action_decision_ts.max()),
                "train_max_label_available_ts": str(train.label_available_ts.max()),
                "heldout_start": str(start), "purge_horizon_hours": HORIZON.total_seconds() / 3600,
                "requested_features": list(features), "preprocessing": state,
                "feature_deletion_declared_before_fold_fit": True,
            })
            if keep_bundles:
                bundles.append(FoldBundle(arm, side, start.strftime("%Y-%m"), test.index.to_numpy(), list(state["selected"]), state, predictor))
    predictions = pd.concat(outputs, ignore_index=True)
    calibrated: list[pd.DataFrame] = []
    for side in SIDES:
        part = predictions[predictions.side.eq(side)].copy()
        state = fit_calibration(part)
        mapped, probability = calibrate(part.raw_predicted_delta_bps.to_numpy(), state)
        part["predicted_delta_continue_bps"] = mapped
        part["predicted_continue_probability"] = probability
        calibrated.append(part)
    return pd.concat(calibrated, ignore_index=True), states, bundles


def metric_rows(policy: pd.DataFrame, arm: str, m0_uplift: float) -> list[dict[str, Any]]:
    month = policy.action_decision_ts.dt.strftime("%Y-%m")
    parts = [("overall", "ALL", policy)]
    parts += [("side", str(value), part) for value, part in policy.groupby("side", sort=True)]
    parts += [("month", str(value), part) for value, part in policy.assign(month=month).groupby("month", sort=True)]
    rows: list[dict[str, Any]] = []
    for dimension, value, part in parts:
        row = {"arm": arm, "split": "development_oof", "dimension": dimension, "value": value}
        row.update(diagnostics(part))
        row["fraction_m0_uplift_retained"] = (
            float(row["incremental_vs_continue_bps"] / m0_uplift)
            if abs(m0_uplift) > 1e-12 else np.nan
        )
        rows.append(row)
    return rows


def assert_m0_reproduction(m0: pd.DataFrame) -> dict[str, Any]:
    canonical = pd.read_parquet(CANONICAL_REPLAY)
    canonical = canonical[
        canonical.split.eq("development_oof")
        & canonical.selected_margin_from_development.eq(True)
    ].copy()
    left = m0.sort_values("candidate_id").reset_index(drop=True)
    right = canonical.sort_values("candidate_id").reset_index(drop=True)
    if left.candidate_id.tolist() != right.candidate_id.tolist():
        raise AssertionError("M0 does not reproduce canonical development candidate rows")
    numeric = ["raw_predicted_delta_bps", "predicted_delta_continue_bps", "predicted_continue_probability"]
    max_error = {c: float(np.max(np.abs(left[c].to_numpy() - right[c].to_numpy()))) for c in numeric}
    action_equal = bool(left.action.eq(right.action).all())
    if any(value > 1e-9 for value in max_error.values()) or not action_equal:
        raise AssertionError(f"M0 canonical reproduction failed: {max_error}, action_equal={action_equal}")
    return {
        "canonical_rows": len(right), "candidate_id_sha256": idhash(right.candidate_id),
        "max_absolute_error": max_error, "actions_exactly_equal": action_equal,
        "canonical_v9_manifest_sha256": _sha(CANONICAL_MANIFEST),
    }


def _conditional_permute(frame: pd.DataFrame, columns: list[str], strata: list[str], seed: int) -> pd.DataFrame:
    """Jointly permute a family inside every declared stratum."""
    result = frame.copy()
    rng = np.random.default_rng(seed)
    positions = pd.Series(np.arange(len(frame)), index=frame.index)
    for _, index in frame.groupby(strata, observed=True, sort=True).groups.items():
        destination = positions.loc[index].to_numpy()
        if len(destination) < 2:
            continue
        source = rng.permutation(destination)
        result.iloc[destination, result.columns.get_indexer(columns)] = frame.iloc[source][columns].to_numpy()
    return result


def conditional_permutation(
    frame: pd.DataFrame, baseline: pd.DataFrame, bundles: list[FoldBundle],
    families: dict[str, list[str]], baseline_metrics: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    calibrators = {side: fit_calibration(baseline[baseline.side.eq(side)]) for side in SIDES}
    schemes = {
        "within_utc_day_and_side": ["utc_day", "side"],
        "within_utc_day_side_time_to_clear": ["utc_day", "side", "time_to_clear_bucket"],
    }
    for scheme_index, (scheme, strata) in enumerate(schemes.items()):
        for family_index, (family, family_features) in enumerate(families.items()):
            if not family_features:
                rows.append({"scheme": scheme, "feature_family": family, "repeat": -1, "status": "NOT_APPLICABLE_EMPTY_FAMILY"})
                continue
            for repeat in range(PERMUTATION_REPEATS):
                predictions: list[pd.DataFrame] = []
                for bundle_index, bundle in enumerate(bundles):
                    test = frame.loc[bundle.test_index].copy()
                    test["utc_day"] = test.action_decision_ts.dt.floor("D")
                    selected = [c for c in family_features if c in bundle.selected_features]
                    if selected:
                        test = _conditional_permute(
                            test, selected, strata,
                            SEED + 10_000 + scheme_index * 1_000 + family_index * 100 + repeat * 10 + bundle_index,
                        )
                    raw = bundle.predict(apply_preprocess(test, bundle.preprocess))
                    part = test[BASE_COLUMNS].copy()
                    part["raw_predicted_delta_bps"] = raw
                    predictions.append(part)
                pred = pd.concat(predictions, ignore_index=True)
                calibrated = []
                for side in SIDES:
                    part = pred[pred.side.eq(side)].copy()
                    mapped, probability = calibrate(part.raw_predicted_delta_bps.to_numpy(), calibrators[side])
                    part["predicted_delta_continue_bps"] = mapped
                    part["predicted_continue_probability"] = probability
                    calibrated.append(part)
                policy = replay(pd.concat(calibrated, ignore_index=True), MARGIN_BPS)
                metric = diagnostics(policy)
                rows.append({
                    "scheme": scheme, "feature_family": family, "repeat": repeat,
                    "status": "RUN", "rows": len(policy),
                    "candidate_id_sha256": idhash(sorted(policy.candidate_id)),
                    "delta_mae_bps": metric["mae_bps"] - baseline_metrics["mae_bps"],
                    "delta_spearman_ic": metric["spearman_ic"] - baseline_metrics["spearman_ic"],
                    "delta_policy_uplift_bps": metric["incremental_vs_continue_bps"] - baseline_metrics["incremental_vs_continue_bps"],
                    "delta_giveback_capture": metric["giveback_cases_exited_pct"] - baseline_metrics["giveback_cases_exited_pct"],
                    "delta_false_exit_cost_bps": metric["false_exit_opportunity_cost_bps"] - baseline_metrics["false_exit_opportunity_cost_bps"],
                    "permuted_selected_feature_count": sum(c in b.selected_features for b in bundles for c in family_features),
                    "strata": " x ".join(strata),
                })
    return pd.DataFrame(rows)


def run(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    if not all(path.exists() for path in (FEATURES, GROUPS, CANONICAL_REPLAY, CANONICAL_MANIFEST)):
        raise FileNotFoundError("canonical Stage-D inputs missing")
    frame, sealed_groups = load_frame()
    # Strong fail-closed boundary: E2/E3 may hold historical rows for training,
    # but score only Apr-Jul development OOF and never inspect any later period.
    if not frame.action_decision_ts.ge(FINAL_START).any():
        raise ValueError("canonical frame unexpectedly lacks sealed final rows")
    a0 = list(sealed_groups["A0_minimal_action_state_control"])
    families = feature_families(a0)
    contracts = arm_contracts(a0, families)

    predictions: dict[str, pd.DataFrame] = {}
    all_states: list[dict[str, Any]] = []
    m0_bundles: list[FoldBundle] = []
    for arm, contract in contracts.items():
        pred, states, bundles = score_arm(
            frame, arm, contract["features"], contract["model"],
            keep_bundles=arm == "M0_full_frozen_A0",
        )
        predictions[arm] = pred
        all_states.extend(states)
        m0_bundles.extend(bundles)

    m0_policy = replay(predictions["M0_full_frozen_A0"], MARGIN_BPS)
    reproduction = assert_m0_reproduction(m0_policy)
    m0_metrics = diagnostics(m0_policy)
    results: list[dict[str, Any]] = []
    for arm, pred in predictions.items():
        policy = replay(pred, MARGIN_BPS)
        if set(policy.candidate_id) != set(m0_policy.candidate_id):
            raise AssertionError(f"minimal arm row mismatch: {arm}")
        if sorted(policy.fold.unique()) != sorted(m0_policy.fold.unique()):
            raise AssertionError(f"minimal arm fold mismatch: {arm}")
        results.extend(metric_rows(policy, arm, m0_metrics["incremental_vs_continue_bps"]))

    deletion_predictions: dict[str, pd.DataFrame] = {}
    deletion_states: list[dict[str, Any]] = []
    for family, deleted in families.items():
        if not deleted:
            continue
        kept = [c for c in a0 if c not in set(deleted)]
        arm = f"leave_out__{family}"
        pred, states, _ = score_arm(frame, arm, kept, "canonical_lgbm_huber")
        deletion_predictions[arm] = pred
        deletion_states.extend(states)
    deletion_rows: list[dict[str, Any]] = []
    for arm, pred in deletion_predictions.items():
        policy = replay(pred, MARGIN_BPS)
        if set(policy.candidate_id) != set(m0_policy.candidate_id):
            raise AssertionError(f"group-deletion row mismatch: {arm}")
        deletion_rows.extend(metric_rows(policy, arm, m0_metrics["incremental_vs_continue_bps"]))
    # Explicitly record the structurally empty upstream-output group.
    deletion_rows.append({
        "arm": "leave_out__upstream_model_outputs", "split": "development_oof",
        "dimension": "overall", "value": "ALL", "rows": len(m0_policy),
        "status": "NOT_APPLICABLE_EMPTY_FAMILY", "fraction_m0_uplift_retained": 1.0,
    })

    permutation = conditional_permutation(frame, predictions["M0_full_frozen_A0"], m0_bundles, families, m0_metrics)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        pd.DataFrame(results).to_parquet(stage / "stage_e_minimal_ablation_results.parquet", index=False, compression="zstd")
        pd.DataFrame(deletion_rows).to_parquet(stage / "stage_e_leave_group_out_results.parquet", index=False, compression="zstd")
        permutation.to_parquet(stage / "stage_e_conditional_permutation_results.parquet", index=False, compression="zstd")
        _dump(stage / "stage_e_e2_e3_model_contract.json", {
            "arms": contracts, "feature_families": families,
            "development_scoring_period": [str(DEV_START), str(FINAL_START)],
            "target": "delta_continue_bps = net_continue_bps - net_exit_now_bps",
            "margin_bps": MARGIN_BPS, "m0_reproduction": reproduction,
            "fold_states": all_states, "deletion_fold_states": deletion_states,
            "second_oos_access": "PROHIBITED_AND_NOT_RUN",
        })
        outputs = {path.name: _sha(path) for path in stage.iterdir()}
        manifest = {
            "schema": SCHEMA, "status": "RESEARCH_ONLY_E2_E3_COMPLETE",
            "canonical_inputs": {
                str(path): _sha(path) for path in (FEATURES, GROUPS, CANONICAL_REPLAY, CANONICAL_MANIFEST)
            },
            "development_rows": len(m0_policy),
            "development_candidate_id_sha256": idhash(sorted(m0_policy.candidate_id)),
            "development_folds": sorted(m0_policy.fold.unique().tolist()),
            "side_local": True, "purge_horizon_hours": HORIZON.total_seconds() / 3600,
            "seed": SEED, "permutation_repeats": PERMUTATION_REPEATS,
            "conditional_permutation_schemes": ["utc_day x side", "utc_day x side x time_to_clear_bucket"],
            "m0_reproduction": reproduction, "second_oos_access": "PROHIBITED_AND_NOT_RUN",
            "outputs_sha256": outputs, "runner_sha256": _sha(Path(__file__)),
            "tests_sha256": _sha(ROOT / "tests/test_stage_e_minimal_information_diagnostics.py"),
        }
        _dump(stage / "run_manifest.json", manifest)
        (stage / "manifest.sha256").write_text(f"{_sha(stage / 'run_manifest.json')}  run_manifest.json\n")
        os.replace(stage, output)
        return manifest
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, default=str))
