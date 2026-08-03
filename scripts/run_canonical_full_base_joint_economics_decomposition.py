#!/usr/bin/env python3
"""Chronological joint-economics decomposition on canonical panel v2.

This is deliberately a bounded *diagnostic* experiment.  It uses three
expanding, calendar-fixed OOF folds (Feb-15--Mar-01, Mar-01--Mar-16, and
Mar-16--Apr-01) and an exact 12-hour label-resolution purge.  Both frozen
feature arms (S0 and S1+B) use a fixed depth-5 side-local model.  April is
scored only after this contract is fixed; it is a reused diagnostic period and
is explicitly not promotion evidence.

The component heads all see identical rows: direct net primary, gross-opportunity
probabilities, favorable/adverse conditional payoffs, and four mutually
exclusive exit probability/payoff pairs.  Scores are composed in net-return
units.  In particular, the exit mixture is a partitioned expectation and is
never additionally penalised by the adverse-loss component.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL_ROOT = ROOT / "data_perp/artifacts/canonical_opportunity_payoff_trust_panel_20260729_v2"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/canonical_full_base_joint_economics_decomposition_20260729_v1"
SCHEMA = "canonical_full_base_joint_economics_decomposition_v1"
SIDES = ("long", "short")
ARMS = ("S0", "S1+B")
FRACTIONS = (0.01, 0.05, 0.10, 0.20)
PURGE_HOURS = 12
SEED = 20260729
APRIL_FREEZE = pd.Timestamp("2025-04-01T00:00:00Z")
FOLD_BOUNDARIES = (
    ("2025-02-15T00:00:00Z", "2025-03-01T00:00:00Z"),
    ("2025-03-01T00:00:00Z", "2025-03-16T00:00:00Z"),
    ("2025-03-16T00:00:00Z", "2025-04-01T00:00:00Z"),
)
IDENTITY = ("candidate_id", "side_name", "__symbol__", "__ts__")
EXIT_CLASSES = ("trailing", "timeout", "full_stop", "adverse_exit")

# Exact, retained frozen base contracts from the panel materializer.
BASE_LONG = (
    "base_input__climax_decay", "base_input__cross_asset_corr_1h",
    "base_input__delta_stall_6", "base_input__dow_cos", "base_input__dow_sin",
    "base_input__eig_effective_rank__breakout_all",
    "base_input__eig_participation_ratio__breakout_all", "base_input__eth_btc_ret_1h",
    "base_input__fragmented_flush_recovery", "base_input__giveback_vol_units",
    "base_input__hour_cos", "base_input__hour_sin", "base_input__liquidation_onset_score",
    "base_input__mark_perp_dislocation", "base_input__mark_vs_perp_bps",
    "base_input__market_breadth_1h", "base_input__median_volume_z",
    "base_input__mkt_atr_expansion_1h", "base_input__pct_assets_above_ema_fast",
    "base_input__pct_assets_above_vwap", "base_input__prog_eff_12",
    "base_input__prog_eff_24", "base_input__q_iqr__amihud_z_peer_resid",
    "base_input__qv", "base_input__range_12h_pct",
    "base_input__regime_transition_entropy_48h", "base_input__rejection_proxy",
    "base_input__rvol_z_peer_resid", "base_input__z_r_24", "base_input__dae_b16_02",
    "base_input__gmm_ood_score",
)
BASE_SHORT = (
    "base_input__mark_perp_dislocation", "base_input__mark_vs_perp_bps",
    "base_input__climax_decay", "base_input__impact_12", "base_input__post_flush_leverage_rebuild",
    "base_input__shock_12h", "base_input__bb_pos_12", "base_input__liquidation_onset_score",
)
SCORE_CONTEXT = (
    "base_rank_pct_timestamp_side", "base_score_z_timestamp_side",
    "base_group_rows_timestamp_side", "base_margin_to_top40_cutoff_z",
    "base_rank_pct_timestamp_global", "base_score_z_timestamp_global",
    "base_group_rows_timestamp_global",
)
FORBIDDEN_PREFIXES = ("mapped_", "causal_score_", "opportunity_", "execution_", "exit_", "__first_touch_")
FORBIDDEN_TOKENS = ("target_price", "wait_action", "timing", "mfe", "mae", "label_resolution", "future", "realized")


@dataclass(frozen=True)
class Geometry:
    name: str = "fixed_d5"
    iterations: int = 300
    depth: int = 5
    learning_rate: float = 0.04
    l2_leaf_reg: float = 12.0


GEOMETRY = Geometry()


@dataclass(frozen=True)
class Fold:
    fold_id: int
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def arm_features(arm: str, side: str) -> tuple[str, ...]:
    if arm not in ARMS or side not in SIDES:
        raise ValueError("unknown arm or side")
    features: tuple[str, ...] = ("base_oof_score",)
    if arm == "S1+B":
        features += SCORE_CONTEXT + (BASE_LONG if side == "long" else BASE_SHORT)
    validate_feature_names(features)
    return features


def validate_feature_names(features: Sequence[str]) -> None:
    for name in features:
        lower = str(name).lower()
        if str(name).startswith(FORBIDDEN_PREFIXES) or any(token in lower for token in FORBIDDEN_TOKENS):
            raise ValueError(f"forbidden model feature: {name}")


def required_columns() -> tuple[str, ...]:
    features = [feature for arm in ARMS for side in SIDES for feature in arm_features(arm, side)]
    labels = [
        "__decision_ts__", "execution_label_end_utc",
        "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return",
        "opportunity_gross_above_cost_0bps", "opportunity_gross_above_cost_25bps",
        "execution_exit_class", *[f"exit_is_{name}" for name in EXIT_CLASSES],
    ]
    return tuple(dict.fromkeys((*IDENTITY, *labels, *features)))


def load_panel(panel_root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path, sidecar, panel_path = (panel_root / "manifest.json", panel_root / "manifest.sha256", panel_root / "panel.parquet")
    if not all(path.exists() for path in (manifest_path, sidecar, panel_path)):
        raise FileNotFoundError("panel root lacks panel.parquet, manifest.json, or manifest.sha256")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "canonical_opportunity_payoff_trust_panel_v2":
        raise ValueError("canonical panel v2 is required")
    if sidecar.read_text().split()[0] != sha256_file(manifest_path):
        raise ValueError("panel manifest SHA256 mismatch")
    if manifest.get("outputs_sha256", {}).get("panel.parquet") != sha256_file(panel_path):
        raise ValueError("panel SHA256 mismatch")
    frame = pd.read_parquet(panel_path, columns=list(required_columns()))
    for name in ("__ts__", "__decision_ts__", "execution_label_end_utc"):
        frame[name] = pd.to_datetime(frame[name], utc=True, errors="raise")
    validate_panel(frame)
    return frame.sort_values(list(IDENTITY), kind="stable").reset_index(drop=True), manifest


def validate_panel(frame: pd.DataFrame) -> None:
    if frame["candidate_id"].duplicated().any() or not frame["side_name"].isin(SIDES).all():
        raise ValueError("canonical identity/side contract failed")
    if not frame["__decision_ts__"].eq(frame["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("decision timestamp is not signal + 1 hour")
    if not frame["execution_label_end_utc"].eq(frame["__decision_ts__"] + pd.Timedelta(hours=PURGE_HOURS)).all():
        raise ValueError("exact 12-hour execution label contract failed")
    if not np.allclose(frame["execution_gross_ev_12h"] - frame["execution_cost_return"], frame["execution_net_ev_12h"], atol=1e-12, rtol=0.0):
        raise ValueError("gross - cost != net")
    flags = frame[[f"exit_is_{name}" for name in EXIT_CLASSES]].astype(int)
    if not flags.sum(axis=1).eq(1).all() or set(frame["execution_exit_class"].astype(str)) != set(EXIT_CLASSES):
        raise ValueError("four exit classes must be mutually exclusive and exhaustive")


def split_development_april(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = frame.loc[
        frame["__ts__"].ge(pd.Timestamp("2025-02-01", tz="UTC"))
        & frame["__ts__"].lt(APRIL_FREEZE)
        & frame["execution_label_end_utc"].lt(APRIL_FREEZE)
    ].copy()
    april = frame.loc[frame["__ts__"].dt.strftime("%Y-%m").eq("2025-04")].copy()
    if not development["execution_label_end_utc"].lt(APRIL_FREEZE).all():
        raise AssertionError("unresolved development label")
    # These are immutable canonical-v2 population checks, not a requirement for
    # small unit-test frames.
    if len(frame) == 509_868 and len(development) != 334_298:
        raise ValueError(f"expected 334298 resolved February--March rows, got {len(development)}")
    return development.reset_index(drop=True), april.reset_index(drop=True)


def make_expanding_folds() -> tuple[Fold, ...]:
    return tuple(Fold(index, pd.Timestamp(start), pd.Timestamp(end)) for index, (start, end) in enumerate(FOLD_BOUNDARIES))


def fold_masks(frame: pd.DataFrame, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    """Return strict expanding train and OOF masks with an exact path purge."""
    validation = frame["__ts__"].ge(fold.validation_start).to_numpy() & frame["__ts__"].lt(fold.validation_end).to_numpy()
    resolution = pd.to_datetime(frame["execution_label_end_utc"], utc=True)
    label_end = pd.to_datetime(frame["execution_label_end_utc"], utc=True)
    training = resolution.lt(fold.validation_start).to_numpy()
    if not validation.any() or not training.any():
        raise ValueError(f"empty strict expanding fold {fold.fold_id}")
    if not label_end.loc[training].lt(fold.validation_start).all():
        raise AssertionError("training row violates the exact 12-hour label-resolution purge")
    return training, validation


def numeric_features(frame: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    result = frame.loc[:, list(features)].apply(pd.to_numeric, errors="raise")
    if np.isinf(result.to_numpy(dtype=float)).any():
        raise ValueError("infinite model feature")
    return result


def _fit_predict(train_x: pd.DataFrame, train_y: np.ndarray, test_x: pd.DataFrame, *, classifier: bool, seed: int, threads: int) -> np.ndarray:
    # A constant is the correct conditional fallback when a historical side has
    # no examples of a rare exit/outcome; it is not an in-sample substitute.
    valid = np.isfinite(train_y)
    if valid.sum() == 0:
        return np.zeros(len(test_x), dtype=float)
    y = train_y[valid]
    if np.unique(y).size < 2:
        return np.full(len(test_x), float(np.mean(y)), dtype=float)
    common = dict(iterations=GEOMETRY.iterations, depth=GEOMETRY.depth, learning_rate=GEOMETRY.learning_rate,
                  l2_leaf_reg=GEOMETRY.l2_leaf_reg, random_seed=seed, thread_count=threads,
                  verbose=False, allow_writing_files=False, bootstrap_type="Bayesian")
    if classifier:
        from catboost import CatBoostClassifier
        model = CatBoostClassifier(loss_function="CrossEntropy", **common)
        model.fit(train_x.loc[valid], y)
        return np.asarray(model.predict_proba(test_x)[:, 1], dtype=float)
    from catboost import CatBoostRegressor
    model = CatBoostRegressor(loss_function="RMSE", **common)
    model.fit(train_x.loc[valid], y)
    return np.asarray(model.predict(test_x), dtype=float)


def head_targets(frame: pd.DataFrame) -> dict[str, tuple[np.ndarray, bool]]:
    net = frame["execution_net_ev_12h"].to_numpy(float)
    positive = net > 0.0
    adverse = net < 0.0
    result: dict[str, tuple[np.ndarray, bool]] = {
        "direct_net": (net, False),
        "p_gross_gt_cost": (frame["opportunity_gross_above_cost_0bps"].to_numpy(float), True),
        "p_gross_gt_cost_25bps": (frame["opportunity_gross_above_cost_25bps"].to_numpy(float), True),
        "conditional_favorable_payoff": (np.where(positive, net, np.nan), False),
        "conditional_adverse_loss_severity": (np.where(adverse, -net, np.nan), False),
    }
    exit_class = frame["execution_exit_class"].astype(str).to_numpy()
    for name in EXIT_CLASSES:
        mask = exit_class == name
        result[f"p_exit_{name}"] = (mask.astype(float), True)
        result[f"conditional_net_{name}"] = (np.where(mask, net, np.nan), False)
    return result


def generate_side_local_oof(development: pd.DataFrame, folds: Sequence[Fold], *, arm: str, threads: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    heads = head_targets(development)
    predictions = pd.DataFrame(index=development.index, columns=list(heads), dtype=float)
    fold_id = np.full(len(development), -1, dtype=np.int16)
    for fold in folds:
        training, validation = fold_masks(development, fold)
        for side_index, side in enumerate(SIDES):
            train_side = training & development["side_name"].eq(side).to_numpy()
            valid_side = validation & development["side_name"].eq(side).to_numpy()
            if not valid_side.any():
                continue
            features = arm_features(arm, side)
            train_x, valid_x = numeric_features(development.loc[train_side], features), numeric_features(development.loc[valid_side], features)
            for head_index, (head, (target, classifier)) in enumerate(heads.items()):
                predictions.loc[valid_side, head] = _fit_predict(train_x, target[train_side], valid_x, classifier=classifier, seed=seed + fold.fold_id * 1000 + side_index * 100 + head_index, threads=threads)
        fold_id[validation] = fold.fold_id
    eligible = fold_id >= 0
    if not np.isfinite(predictions.loc[eligible].to_numpy(float)).all():
        raise ValueError("side-local OOF heads left a missing prediction")
    return predictions, fold_id


def compose_component_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compose partitioned expectations; adverse loss is absent from exit mixture."""
    result = predictions.copy()
    p = np.clip(result["p_gross_gt_cost"].to_numpy(float), 0.0, 1.0)
    favorable = np.maximum(result["conditional_favorable_payoff"].to_numpy(float), 0.0)
    adverse = np.maximum(result["conditional_adverse_loss_severity"].to_numpy(float), 0.0)
    result["opportunity_score"] = p * favorable - (1.0 - p) * adverse
    exit_mixture = np.zeros(len(result), dtype=float)
    probability_sum = np.zeros(len(result), dtype=float)
    for name in EXIT_CLASSES:
        probability = np.clip(result[f"p_exit_{name}"].to_numpy(float), 0.0, 1.0)
        exit_mixture += probability * result[f"conditional_net_{name}"].to_numpy(float)
        probability_sum += probability
    # Independently fitted binary heads need not sum exactly to one.  Renormalise
    # only their mutually exclusive mixture, preserving its common net unit.
    result["exit_probability_sum"] = probability_sum
    result["exit_mixture_score"] = exit_mixture / np.maximum(probability_sum, 1e-12)
    return result


def component_columns() -> tuple[str, ...]:
    return ("direct_net", "p_gross_gt_cost", "p_gross_gt_cost_25bps", "conditional_favorable_payoff", "conditional_adverse_loss_severity", "opportunity_score", "exit_mixture_score") + tuple(f"p_exit_{name}" for name in EXIT_CLASSES) + tuple(f"conditional_net_{name}" for name in EXIT_CLASSES)


def _ridge_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray) -> np.ndarray:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    model = Ridge(alpha=25.0, fit_intercept=True)
    return model.fit(scaler.fit_transform(train_x), train_y).predict(scaler.transform(test_x))


def compose_residual_oof(frame: pd.DataFrame, components: pd.DataFrame, fold_id: np.ndarray, folds: Sequence[Fold]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Pool only *prior resolved OOF* component predictions for each fold."""
    output = np.full(len(frame), np.nan, dtype=float)
    audit: list[dict[str, Any]] = []
    columns = list(component_columns())
    net = frame["execution_net_ev_12h"].to_numpy(float)
    direct = components["direct_net"].to_numpy(float)
    for fold in folds:
        held = fold_id == fold.fold_id
        prior = (fold_id >= 0) & (fold_id < fold.fold_id) & frame["execution_label_end_utc"].lt(fold.validation_start).to_numpy()
        # The first eligible fold has no prior OOF components.  Its direct head
        # is the pooled common-unit anchor, not a same-fold residual fit.
        if prior.sum() < 100:
            output[held] = direct[held]
            audit.append({"fold_id": fold.fold_id, "composer": "direct_common_anchor", "prior_resolved_oof_rows": int(prior.sum())})
            continue
        x_train = components.loc[prior, columns].to_numpy(float)
        x_held = components.loc[held, columns].to_numpy(float)
        residual = net[prior] - direct[prior]
        output[held] = direct[held] + _ridge_predict(x_train, residual, x_held)
        audit.append({"fold_id": fold.fold_id, "composer": "pooled_ridge_residual", "prior_resolved_oof_rows": int(prior.sum())})
    if not np.isfinite(output[fold_id >= 0]).all():
        raise ValueError("residual composer left OOF gaps")
    return output, audit


def stable_global_top_mask(frame: pd.DataFrame, score: Sequence[float], fraction: float) -> np.ndarray:
    if fraction not in FRACTIONS:
        raise ValueError("only predeclared global top-1/5/10/20% cuts are allowed")
    ranking = pd.DataFrame({"position": np.arange(len(frame)), "candidate_id": frame["candidate_id"].astype(str), "score": np.asarray(score, dtype=float)}).sort_values(["score", "candidate_id"], ascending=[False, True], kind="stable")
    mask = np.zeros(len(frame), dtype=bool)
    mask[ranking["position"].to_numpy()[: max(1, int(math.ceil(len(frame) * fraction)))]] = True
    return mask


def side_balance_gate(frame: pd.DataFrame, composed: Sequence[float], direct_anchor: Sequence[float], fraction: float, *, min_side_rows: int = 1, min_share: float = 0.05) -> tuple[np.ndarray, str]:
    """Use a pooled direct-anchor fallback, then abstain if a side still collapses."""
    for score, mode in ((composed, "composed"), (direct_anchor, "direct_anchor_fallback")):
        mask = stable_global_top_mask(frame, score, fraction)
        counts = frame.loc[mask, "side_name"].value_counts()
        total = int(mask.sum())
        if all(int(counts.get(side, 0)) >= min_side_rows and float(counts.get(side, 0)) / total >= min_share for side in SIDES):
            return mask, mode
    return np.zeros(len(frame), dtype=bool), "abstain_side_collapse"


def _clip_probability(value: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(value, dtype=float), 1.0e-6, 1.0 - 1.0e-6)


def _rank_ic(actual: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.isfinite(actual) & np.isfinite(prediction)
    if valid.sum() < 3:
        return float("nan")
    left = pd.Series(actual[valid]).rank(method="average").to_numpy()
    right = pd.Series(prediction[valid]).rank(method="average").to_numpy()
    if np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _binary_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    probability = _clip_probability(prediction)
    y = np.asarray(actual, dtype=float)
    result = {
        "brier": float(np.mean((probability - y) ** 2)),
        "logloss": float(-np.mean(y * np.log(probability) + (1.0 - y) * np.log(1.0 - probability))),
    }
    if np.unique(y).size > 1:
        from sklearn.metrics import average_precision_score, roc_auc_score
        result["auc"] = float(roc_auc_score(y, probability))
        result["average_precision"] = float(average_precision_score(y, probability))
    else:
        result["auc"] = float("nan")
        result["average_precision"] = float("nan")
    buckets = np.minimum((pd.Series(probability).rank(method="first", pct=True).to_numpy() * 10).astype(int), 9)
    result["calibration_ece_10"] = float(sum(abs(probability[buckets == item].mean() - y[buckets == item].mean()) * (buckets == item).mean() for item in range(10) if (buckets == item).any()))
    return result


def head_metrics(frame: pd.DataFrame, predicted: pd.DataFrame, *, arm: str, split: str) -> list[dict[str, Any]]:
    """Head diagnostics use realised subsets only for conditional targets."""
    rows: list[dict[str, Any]] = []
    direct_truth = frame["execution_net_ev_12h"].to_numpy(float)
    direct_estimate = predicted["direct_net"].to_numpy(float)
    rows.append({"arm": arm, "split": split, "head": "direct_net", "kind": "direct_regression", "rows": len(frame), "mae": float(np.mean(np.abs(direct_estimate - direct_truth))), "rank_ic": _rank_ic(direct_truth, direct_estimate)})
    for head, target in (("p_gross_gt_cost", "opportunity_gross_above_cost_0bps"), ("p_gross_gt_cost_25bps", "opportunity_gross_above_cost_25bps")):
        rows.append({"arm": arm, "split": split, "head": head, "kind": "binary", "rows": len(frame), **_binary_metrics(frame[target].to_numpy(float), predicted[head].to_numpy(float))})
    net = direct_truth
    for head, allowed, realised in (("conditional_favorable_payoff", net > 0.0, net), ("conditional_adverse_loss_severity", net < 0.0, -net)):
        truth, estimate = realised[allowed], predicted.loc[allowed, head].to_numpy(float)
        rows.append({"arm": arm, "split": split, "head": head, "kind": "conditional_regression", "rows": int(allowed.sum()), "mae": float(np.mean(np.abs(estimate - truth))) if len(truth) else np.nan, "rank_ic": _rank_ic(truth, estimate)})
    exit_class = frame["execution_exit_class"].astype(str).to_numpy()
    probability_sum = np.zeros(len(frame), dtype=float)
    for name in EXIT_CLASSES:
        probability = predicted[f"p_exit_{name}"].to_numpy(float)
        probability_sum += probability
        actual = (exit_class == name).astype(float)
        rows.append({"arm": arm, "split": split, "head": f"p_exit_{name}", "kind": "independent_binary_exit_probability", "rows": len(frame), **_binary_metrics(actual, probability)})
        allowed = exit_class == name
        truth, estimate = net[allowed], predicted.loc[allowed, f"conditional_net_{name}"].to_numpy(float)
        rows.append({"arm": arm, "split": split, "head": f"conditional_net_{name}", "kind": "conditional_exit_payoff", "rows": int(allowed.sum()), "mae": float(np.mean(np.abs(estimate - truth))) if len(truth) else np.nan, "rank_ic": _rank_ic(truth, estimate)})
    rows.append({"arm": arm, "split": split, "head": "exit_probability_sum", "kind": "independent_binary_exit_probability_sum", "rows": len(frame), "mean_probability_sum": float(probability_sum.mean()), "mae_to_one": float(np.mean(np.abs(probability_sum - 1.0)))})
    return rows


def tail_metrics(frame: pd.DataFrame, score: np.ndarray, direct: np.ndarray, *, arm: str, split: str, score_name: str) -> list[dict[str, Any]]:
    """Raw pooled ranking is retained even when a later balance gate abstains."""
    rows: list[dict[str, Any]] = []
    for fraction in FRACTIONS:
        raw = stable_global_top_mask(frame, score, fraction)
        selected, gate = side_balance_gate(frame, score, direct, fraction)
        raw_subset, subset = frame.loc[raw], frame.loc[selected]
        rows.append({"arm": arm, "split": split, "score_name": score_name, "fraction": fraction, "selection_gate": gate,
                     "raw_global_selected_rows": int(raw.sum()), "raw_global_mean_net_bps": float(raw_subset["execution_net_ev_12h"].mean() * 10000.0), "raw_global_sum_net": float(raw_subset["execution_net_ev_12h"].sum()), "raw_global_long_share": float(raw_subset["side_name"].eq("long").mean()),
                     "gate_selected_rows": int(selected.sum()), "gate_mean_net_bps": float(subset["execution_net_ev_12h"].mean() * 10000.0) if len(subset) else np.nan, "gate_sum_net": float(subset["execution_net_ev_12h"].sum()), "gate_long_share": float(subset["side_name"].eq("long").mean()) if len(subset) else np.nan})
    return rows


def fit_budget() -> dict[str, int]:
    heads = 5 + 2 * len(EXIT_CLASSES)
    return {"frozen_feature_arms": len(ARMS), "heads_per_arm_side": heads, "strict_expanding_oof_model_fits": len(ARMS) * len(SIDES) * len(make_expanding_folds()) * heads,
            "april_diagnostic_final_model_fits": len(ARMS) * len(SIDES) * heads, "hpo_model_fits": 0, "feature_selection_fits": 0,
            "maximum_catboost_model_fits": len(ARMS) * len(SIDES) * heads * (len(make_expanding_folds()) + 1)}


def _fit_final_components(development: pd.DataFrame, april: pd.DataFrame, *, arm: str, threads: int, seed: int) -> pd.DataFrame:
    targets = head_targets(development)
    output = pd.DataFrame(index=april.index, columns=list(targets), dtype=float)
    for side_index, side in enumerate(SIDES):
        train = development["side_name"].eq(side).to_numpy()
        evaluate = april["side_name"].eq(side).to_numpy()
        if not evaluate.any():
            continue
        x_train, x_eval = numeric_features(development.loc[train], arm_features(arm, side)), numeric_features(april.loc[evaluate], arm_features(arm, side))
        for head_index, (head, (target, classifier)) in enumerate(targets.items()):
            output.loc[evaluate, head] = _fit_predict(x_train, target[train], x_eval, classifier=classifier, seed=seed + side_index * 100 + head_index, threads=threads)
    return output


def _write_outputs(output: Path, temporary: Path, manifest: dict[str, Any]) -> None:
    manifest["outputs_sha256"] = {str(path.relative_to(temporary)): sha256_file(path) for path in sorted(temporary.rglob("*")) if path.is_file() and path.name not in {"manifest.json", "manifest.sha256"}}
    manifest_path = temporary / "manifest.json"
    manifest_path.write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n")
    (temporary / "manifest.sha256").write_text(f"{sha256_file(manifest_path)}  manifest.json\n")
    os.replace(temporary, output)


def run(args: argparse.Namespace) -> Path:
    frame, panel_manifest = load_panel(args.panel_root)
    development, april = split_development_april(frame)
    folds = make_expanding_folds()
    # Validate all boundaries up front, before any fits or April inspection.
    oof_coverage = np.zeros(len(development), dtype=bool)
    for fold in folds:
        _, validation = fold_masks(development, fold)
        if (oof_coverage & validation).any():
            raise AssertionError("chronological OOF fold overlap")
        oof_coverage |= validation
    # The nominal three validation windows contain 258,014 rows, but 3,120
    # late-March rows resolve on/after the April freeze.  They must not enter
    # pre-April model/composer selection, leaving 254,894 eligible OOF rows.
    if len(development) == 334_298 and int(oof_coverage.sum()) != 254_894:
        raise ValueError(f"expected 254894 pre-April-resolved strict OOF rows, got {int(oof_coverage.sum())}")
    if args.plan_only:
        print(json.dumps({"development_rows": len(development), "strict_expanding_oof_rows": int(oof_coverage.sum()), "folds": [asdict(fold) for fold in folds], "fit_budget": fit_budget(), "april_status": "reused_diagnostic_not_promotion_evidence"}, default=str, indent=2))
        return args.output
    if args.output.exists():
        raise FileExistsError(f"immutable output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(dir=args.output.parent, prefix=f".{args.output.name}."))
    try:
        all_oof: list[pd.DataFrame] = []
        all_april: list[pd.DataFrame] = []
        tails: list[dict[str, Any]] = []
        statistics: list[dict[str, Any]] = []
        composer_audit: dict[str, list[dict[str, Any]]] = {}
        for arm_index, arm in enumerate(ARMS):
            heads, fold_id = generate_side_local_oof(development, folds, arm=arm, threads=args.threads, seed=args.seed + arm_index * 10_000)
            composed = compose_component_scores(heads)
            residual, audit = compose_residual_oof(development, composed, fold_id, folds)
            composer_audit[arm] = audit
            eligible = fold_id >= 0
            if len(development) == 334_298 and int(eligible.sum()) != 254_894:
                raise ValueError(f"expected 254894 pre-April-resolved strict OOF rows, got {int(eligible.sum())}")
            oof = development.loc[:, [*IDENTITY, "__decision_ts__", "execution_label_end_utc", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", "opportunity_gross_above_cost_0bps", "opportunity_gross_above_cost_25bps", "execution_exit_class", *[f"exit_is_{name}" for name in EXIT_CLASSES]]].copy()
            oof["fold_id"] = fold_id
            oof["arm"] = arm
            for name in composed.columns:
                oof[f"prediction__{name}"] = composed[name].to_numpy(float)
            oof["direct_primary_score"] = composed["direct_net"].to_numpy(float)
            oof["joint_score"] = residual
            all_oof.append(oof.loc[eligible])
            eligible_frame, eligible_components = development.loc[eligible].reset_index(drop=True), composed.loc[eligible].reset_index(drop=True)
            statistics.extend(head_metrics(eligible_frame, eligible_components, arm=arm, split="development_strict_expanding_oof"))
            scores = {"direct_primary": composed.loc[eligible, "direct_net"].to_numpy(float), "opportunity": composed.loc[eligible, "opportunity_score"].to_numpy(float), "exit_mixture": composed.loc[eligible, "exit_mixture_score"].to_numpy(float), "joint": residual[eligible]}
            for score_name, score in scores.items():
                tails.extend(tail_metrics(eligible_frame, score, scores["direct_primary"], arm=arm, split="development_strict_expanding_oof", score_name=score_name))

            # April has no bearing on arm, features, geometry, or composer choice.
            april_heads = compose_component_scores(_fit_final_components(development, april, arm=arm, threads=args.threads, seed=args.seed + arm_index * 10_000 + 5_000))
            prior = composed.loc[eligible, list(component_columns())].to_numpy(float)
            prior_residual = development.loc[eligible, "execution_net_ev_12h"].to_numpy(float) - composed.loc[eligible, "direct_net"].to_numpy(float)
            april_joint = april_heads["direct_net"].to_numpy(float) + _ridge_predict(prior, prior_residual, april_heads.loc[:, list(component_columns())].to_numpy(float))
            diagnostic = april.loc[:, [*IDENTITY, "__decision_ts__", "execution_label_end_utc", "execution_net_ev_12h", "execution_gross_ev_12h", "execution_cost_return", "opportunity_gross_above_cost_0bps", "opportunity_gross_above_cost_25bps", "execution_exit_class", *[f"exit_is_{name}" for name in EXIT_CLASSES]]].copy()
            diagnostic["arm"] = arm
            for name in april_heads.columns:
                diagnostic[f"prediction__{name}"] = april_heads[name].to_numpy(float)
            diagnostic["direct_primary_score"] = april_heads["direct_net"].to_numpy(float)
            diagnostic["joint_score"] = april_joint
            all_april.append(diagnostic)
            statistics.extend(head_metrics(april, april_heads, arm=arm, split="april_reused_diagnostic"))
            april_scores = {"direct_primary": diagnostic["direct_primary_score"].to_numpy(float), "opportunity": april_heads["opportunity_score"].to_numpy(float), "exit_mixture": april_heads["exit_mixture_score"].to_numpy(float), "joint": april_joint}
            for score_name, score in april_scores.items():
                tails.extend(tail_metrics(april, score, april_scores["direct_primary"], arm=arm, split="april_reused_diagnostic", score_name=score_name))

        pd.concat(all_oof, ignore_index=True).to_parquet(temporary / "development_strict_expanding_oof_predictions.parquet", index=False, compression="zstd")
        pd.concat(all_april, ignore_index=True).to_parquet(temporary / "april_reused_diagnostic_predictions.parquet", index=False, compression="zstd")
        pd.DataFrame(tails).to_parquet(temporary / "tail_metrics.parquet", index=False)
        pd.DataFrame(statistics).to_parquet(temporary / "head_statistics.parquet", index=False)
        manifest = {
            "schema": SCHEMA, "status": "COMPLETED_REUSED_APRIL_DIAGNOSTIC_NOT_PROMOTION_EVIDENCE",
            "source": {"panel_root": str(args.panel_root), "panel_sha256": panel_manifest["outputs_sha256"]["panel.parquet"], "panel_manifest_sha256": sha256_file(args.panel_root / "manifest.json"), "runner_sha256": sha256_file(Path(__file__).resolve())},
            "validation": {"kind": "strict_chronological_expanding", "folds": [asdict(item) for item in folds], "label_availability_column": "execution_label_end_utc", "exact_label_resolution_purge_hours": PURGE_HOURS, "nominal_validation_rows": 258014, "pre_april_resolved_oof_rows": 254894, "late_march_rows_excluded_before_april_freeze": 3120, "legacy_base_effective_label_resolution_not_used": True, "april_reused_diagnostic_not_untouched_promotion_evidence": True},
            "features": {"frozen_arms": {arm: {side: list(arm_features(arm, side)) for side in SIDES} for arm in ARMS}, "feature_selection": "none", "hpo": "none", "forbidden_action_layer_fields": True},
            "heads": {"identical_rows": True, "direct_primary": "execution_net_ev_12h", "opportunity": ["P(gross>cost)", "P(gross>cost+25bps)", "conditional_favorable_payoff", "conditional_adverse_loss_severity"], "exit": {"classes": list(EXIT_CLASSES), "mutually_exclusive_labels": True, "probability_models": "four independent binary heads; probability sum and per-class Brier/logloss are reported", "conditional_payoff": "net payoff"}},
            "composition": {"opportunity": "p_gross_gt_cost*favorable_payoff - (1-p_gross_gt_cost)*adverse_loss_severity", "exit_mixture": "sum(normalized p(exit)*conditional_net(exit))", "no_adverse_double_count": True, "residual_composer": "pooled Ridge residual trained only on prior resolved OOF component predictions; fold 0 direct/common anchor", "side_intercept": "disabled; pooled common-unit composer has no unconstrained side adjustment"},
            "selection": {"scope": "pooled global", "fractions": list(FRACTIONS), "tie_break": "candidate_id ascending", "side_balance": "composed, then direct-anchor fallback, else abstain"},
            "composer_audit": composer_audit, "fit_budget": fit_budget(), "seed": args.seed, "threads": args.threads,
        }
        _write_outputs(args.output, temporary, manifest)
        return args.output
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
