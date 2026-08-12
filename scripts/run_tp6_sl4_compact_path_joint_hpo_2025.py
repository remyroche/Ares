#!/usr/bin/env python3
"""Sequential compact path-health / joint reliability ablation on TP6/SL4.

The reference score is the exact long-only ``base_plus_consensus25`` score
from ``TP6_SL4_BASE_CONSENSUS_CANONICAL_20260807``.  This script neither
replaces that stack nor retrains its base/consensus heads.  It only learns a
bounded, causal reliability probability and applies it by a predeclared
shrinkage or multiplier transform.

The funnel is deliberately staged:

1. Build feature-only and prior-resolved leaf/prototype state.
2. Materialise individual active recurrent-path health variables.  Before each
   held month, choose a compact subset using a training-only binned conditional
   MI proxy (conditional on base-score state, with a selected-state update and
   redundancy penalty).
3. Compare predeclared feature-block combinations.  Each combination receives
   a 2024-only, subsampled Optuna/MedianPruner HPO; outer 2025 months are never
   used to select its parameters.
4. Trim low-gain fields inside the broad joint arm using only its 2024 refit,
   then run the same HPO process for that trimmed arm.
5. Emit the full shrink/multiply grid, all global and monthly tails, feature
   usage, and a gate for a possible *later* residual-path experiment.

Outcome-bearing path health and correctness states are indexed by
``label_available_ts`` and joined strictly before the candidate decision.  The
frozen K=9 soft memberships are explicit inputs to every structural challenger.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import zlib
from pathlib import Path
from typing import Any, Sequence

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, mutual_info_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_tp6_sl4_rule_state_reliability_ablation_2025 as state  # noqa: E402
from scripts.run_tp6_sl4_downstream_retrain_2025 import MONTHS, _load  # noqa: E402
from scripts.run_tp6_sl4_prototype_cluster_use_ablation_2025 import _causal_dynamic_state  # noqa: E402


SEED = 20260812
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
OUT = ROOT / "data_perp/artifacts/tp6_sl4_compact_path_joint_hpo_20260809_v1"
STRUCTURE = state.STRUCTURE
CONTROL = state.CONTROL
MIN_PATH_EFFECTIVE_SUPPORT = state.MIN_PATH_EFFECTIVE_SUPPORT
HPO_CUTOFF = pd.Timestamp("2024-12-01", tz="UTC")
HPO_TRAIN_END = pd.Timestamp("2024-10-01", tz="UTC")
DEVELOPMENT_MONTHS = tuple(f"2025-{month:02d}" for month in range(1, 10))
CONFIRMATION_MONTHS = tuple(f"2025-{month:02d}" for month in range(10, 13))


def _safe(frame: pd.DataFrame, fields: Sequence[str], med: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    return state._safe(frame, fields, med)


def _map_base(train: pd.DataFrame, held: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return state._map_base(train, held)


def _metric_table(prediction: pd.DataFrame, arms: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Global, month, and month-stability metrics for every requested tail."""
    glob: list[dict[str, object]] = []
    month_rows: list[dict[str, object]] = []
    stability: list[dict[str, object]] = []
    for arm in arms:
        values = pd.to_numeric(prediction[arm], errors="coerce").fillna(0.0).to_numpy(float)
        overall_ic = float(spearmanr(values, prediction.net_bps.to_numpy(float)).statistic)
        for tail in TAILS:
            n = max(1, int(math.ceil(len(prediction) * tail)))
            top = prediction.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(n)
            glob.append({
                "arm": arm, "scope": "all_2025_global_rank", "tail": tail, "trades": len(top),
                "gross_bps_per_trade": float(top.gross_bps.mean()),
                "net_bps_per_trade": float(top.net_bps.mean()),
                "net_pnl_bps": float(top.net_bps.sum()),
                "positive_net_rate": float((top.net_bps > 0.0).mean()), "rank_ic": overall_ic,
            })
            rows = []
            for month, block in prediction.groupby("month", sort=True, observed=True):
                m_n = max(1, int(math.ceil(len(block) * tail)))
                selected = block.sort_values([arm, "candidate_id"], ascending=[False, True], kind="stable").head(m_n)
                value = float(selected.net_bps.mean())
                ic = float(spearmanr(pd.to_numeric(block[arm], errors="coerce").fillna(0.0), block.net_bps).statistic)
                item = {
                    "arm": arm, "month": str(month), "tail": tail, "trades": len(selected),
                    "gross_bps_per_trade": float(selected.gross_bps.mean()), "net_bps_per_trade": value,
                    "net_pnl_bps": float(selected.net_bps.sum()), "positive_net_rate": float((selected.net_bps > 0.0).mean()),
                    "rank_ic": ic,
                }
                month_rows.append(item)
                rows.append(item)
            value = np.asarray([row["net_bps_per_trade"] for row in rows], dtype=float)
            med = float(np.median(value))
            mad = float(np.median(np.abs(value - med)))
            stability.append({
                "arm": arm, "tail": tail, "months": len(value), "mean_net_bps": float(value.mean()),
                "median_net_bps": med, "mad_net_bps": mad, "std_net_bps": float(value.std(ddof=0)),
                "worst_month_net_bps": float(value.min()), "positive_months": int((value > 0.0).sum()),
                "mean_month_rank_ic": float(np.nanmean([row["rank_ic"] for row in rows])),
                "portability_score_bps": med - 0.5 * mad - max(0.0, -float(value.min())),
            })
    return pd.DataFrame(glob), pd.DataFrame(month_rows), pd.DataFrame(stability)


def _build_panel() -> tuple[pd.DataFrame, dict[str, list[str]], pd.DataFrame, pd.DataFrame]:
    """Recreate the matched canonical panel and all causal state blocks."""
    source, context, context_hash = _load()
    source = source.loc[source.side_name.eq("long")].copy()
    base = source[[
        "candidate_id", "__ts__", "month", "label_available_ts", "base_score",
        "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", "exact_net_bps",
        "exact_gross_bps", *context,
    ]].copy().rename(columns={"exact_net_bps": "net_bps", "exact_gross_bps": "gross_bps"})
    structural = pd.read_parquet(STRUCTURE / "prototype_cluster_row_features.parquet")
    sizes = pd.read_parquet(STRUCTURE / "prototype_cluster_size_sweep_features.parquet")
    structural = structural.merge(sizes, on=["candidate_id", "__ts__", "month"], how="left", validate="one_to_one")
    keep = [
        "candidate_id", "__ts__", "month", "base_expected_bps",
        *[c for c in structural if c.startswith("prototype__")],
        *[c for c in structural if c.startswith("k09__")],
        "prototype_matched_mass", "prototype_unmatched_mass", "prototype_match_similarity",
        "prototype_top2_margin", "prototype_entropy", "prototype_exposure_top2_margin",
        "prototype_assignment_count",
    ]
    panel = base.merge(structural.loc[:, list(dict.fromkeys(keep))], on=["candidate_id", "__ts__", "month"], how="inner", validate="one_to_one")
    panel = panel.rename(columns={"base_expected_bps": "frozen_base_expected_bps"})
    control = pd.read_parquet(CONTROL)
    control = control.loc[control.side_name.eq("long") & control.month.astype(str).isin(MONTHS), ["candidate_id", "month", "base_plus_consensus25"]]
    panel = panel.merge(control, on=["candidate_id", "month"], how="left", validate="one_to_one")
    valid_months = [*MONTHS, *[f"2024-{month:02d}" for month in range(4, 12)]]
    panel = panel.loc[panel.month.astype(str).isin(valid_months)].copy()
    panel["__ts__"] = pd.to_datetime(panel["__ts__"], utc=True)
    panel["label_available_ts"] = pd.to_datetime(panel["label_available_ts"], utc=True)
    if panel.loc[panel.month.astype(str).isin(MONTHS), "base_plus_consensus25"].isna().any():
        raise RuntimeError("the canonical Base+Consensus control is incomplete")
    panel = panel.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    proto_abs = [c for c in panel if c.startswith("prototype__") and c.endswith("__abs_contribution")]
    k9_memberships = [c for c in panel if c.startswith("k09__cluster__") and c.endswith("__membership")]
    k9_raw = [c for c in panel if c.startswith("k09__cluster__")]
    leaf, leaf_audit = state._leaf_support_features(panel)
    proto = state._prototype_state_features(panel, proto_abs)
    recent, recent_audit = state._recent_correctness_features(panel, proto_abs)
    covariance_inputs = [
        "base_score", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak",
        "prototype_matched_mass", "prototype_entropy", "prototype_top2_margin", *k9_memberships[:3],
    ]
    activation = panel.loc[:, k9_memberships].max(axis=1).to_numpy(float)
    cov_activation = state._covariance_break(panel, event_ts="__ts__", target=activation, fields=covariance_inputs, prefix="activation")
    cov_success = state._covariance_break(panel, event_ts="label_available_ts", target=(panel.net_bps.to_numpy(float) > 0.0).astype(float), fields=covariance_inputs, prefix="success")
    dynamic = _causal_dynamic_state(panel, k9_memberships)
    path_health, path_health_audit = _individual_path_health(panel, proto_abs)
    panel = pd.concat([panel, leaf, proto, recent, cov_activation, cov_success, dynamic, path_health], axis=1)
    base_anchor = ["base_anchor", "base_score", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak"]
    blocks = {
        "market_context": [*base_anchor, *context],
        "soft_membership": k9_raw,
        "activated_leaf_support": list(leaf.columns),
        "rule_path_ood_drift": list(proto.columns),
        "covariance_correlation_break": [*cov_activation.columns, *cov_success.columns],
        "model_state": [c for c in recent if c.startswith("model_recent_cross_")],
        "recent_correctness": [c for c in recent if c.startswith(("model_recent_", "path_recent_")) and "cross_" not in c],
        "individual_path_health": list(path_health.columns),
        "incumbent_support": [c for c in dynamic if c.startswith("archetype_support__")],
        "incumbent_uncertainty": [c for c in ["prototype_entropy", "prototype_top2_margin", "prototype_exposure_top2_margin", "prototype_assignment_count", "prototype_match_similarity"] if c in panel],
    }
    # ``base_anchor`` is deliberately materialised inside each fold from the
    # train-only isotonic mapping, so it cannot exist in the raw panel yet.
    unavailable = [
        field for fields in blocks.values() for field in fields
        if field != "base_anchor" and (field not in panel.columns or not panel[field].notna().any())
    ]
    if unavailable:
        raise RuntimeError(f"requested state fields unavailable: {unavailable[:12]}")
    lineage = pd.DataFrame([
        {"context_sha256": context_hash, "context_fields": len(context), "rows": len(panel), "prototypes": len(proto_abs), "k9_memberships": len(k9_memberships)}
    ])
    audits = pd.concat([
        leaf_audit.assign(audit="leaf_support"), recent_audit.assign(audit="aggregate_path_correctness"),
        path_health_audit.assign(audit="individual_path_health"),
    ], ignore_index=True, sort=False)
    return panel, blocks, lineage, audits


def _individual_path_health(panel: pd.DataFrame, proto_fields: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Causal, active per-prototype health candidates.

    A field is zero when its prototype is not active or has insufficient prior
    resolved support.  It therefore represents *this candidate's activated
    recurrent path health*, rather than a future outcome signature.
    """
    expected = pd.to_numeric(panel["frozen_base_expected_bps"], errors="coerce").fillna(0.0).to_numpy(float)
    net = panel.net_bps.to_numpy(float)
    residual = net - expected
    win = net > 0.0
    metrics = {
        "directional_correct": ((panel.base_score.to_numpy(float) > 0.0) == win).astype(float),
        "approximately_correct": (np.abs(residual) <= 50.0).astype(float),
        "adverse_residual_rate": (residual <= -50.0).astype(float),
        "strong_adverse_residual_rate": (residual <= -100.0).astype(float),
    }
    weights = np.maximum(panel.loc[:, list(proto_fields)].to_numpy(float), 0.0)
    values: dict[str, np.ndarray] = {}
    audit: list[dict[str, object]] = []
    for idx, raw_name in enumerate(proto_fields):
        proto = raw_name.rsplit("__", 2)[1]
        rolling = state._rolling_rates(panel.label_available_ts, weights[:, idx], metrics, prefix="")
        mapped = state._asof_features(panel, rolling)
        for days in (3, 7, 14):
            support = mapped[f"support_{days}d"].to_numpy(float)
            active = weights[:, idx] * (support >= MIN_PATH_EFFECTIVE_SUPPORT)
            for metric in metrics:
                name = f"path_health__{proto}__{metric}__{days}d"
                values[name] = active * mapped[f"{metric}_{days}d"].to_numpy(float)
                audit.append({
                    "prototype": proto, "metric": metric, "window": f"{days}d",
                    "median_prior_support": float(np.median(support)),
                    "adequate_active_fraction": float(np.mean(active > 0.0)),
                    "feature": name,
                })
    out = pd.DataFrame(values, index=panel.index)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32"), pd.DataFrame(audit)


def _bin_from_train(value: pd.Series, reference: pd.Series | None = None, bins: int = 5) -> np.ndarray:
    value = pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    ref = value if reference is None else pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) <= 2:
        return np.zeros(len(value), dtype=np.int16)
    return np.digitize(value, edges[1:-1], right=True).astype(np.int16)


def _conditional_mi(feature: np.ndarray, target: np.ndarray, condition: np.ndarray) -> float:
    total = len(target)
    score = 0.0
    for key in np.unique(condition):
        keep = condition == key
        if int(keep.sum()) < 40 or np.unique(target[keep]).size < 2 or np.unique(feature[keep]).size < 2:
            continue
        score += float(keep.mean()) * float(mutual_info_score(feature[keep], target[keep]))
    return score


def _select_path_health(train: pd.DataFrame, candidates: Sequence[str], target: np.ndarray, *, limit: int = 8) -> tuple[list[str], pd.DataFrame]:
    """Greedy binned CMI proxy, updated by selected-path state.

    The selection's condition begins with a train-only base-score quintile.  At
    later stages it additionally contains a quintile of the mean binned health
    state of the already selected fields.  A small redundancy penalty prevents
    the compact set from being several encodings of one recurrent path.
    """
    base_condition = _bin_from_train(train.base_score, bins=5)
    cache = {field: _bin_from_train(train[field], bins=5) for field in candidates}
    selected: list[str] = []
    records: list[dict[str, object]] = []
    remaining = list(candidates)
    for stage_no in range(limit):
        if selected:
            selected_state = np.mean(np.column_stack([cache[field] for field in selected]), axis=1)
            condition = base_condition * 5 + _bin_from_train(pd.Series(selected_state), bins=5)
        else:
            condition = base_condition
        best: tuple[float, str, float, float] | None = None
        stage_rows: list[dict[str, object]] = []
        for field in remaining:
            score = _conditional_mi(cache[field], target, condition)
            redundancy = float(np.mean([mutual_info_score(cache[field], cache[prior]) for prior in selected])) if selected else 0.0
            incremental = score - 0.15 * redundancy
            stage_rows.append({
                "selection_stage": stage_no + 1, "field": field, "incremental_cmi_proxy": incremental,
                "conditional_mi_proxy": score, "redundancy_mi": redundancy,
                "condition": "base_score_quintile" if stage_no == 0 else "base_score_quintile_x_selected_health_state",
            })
            option = (incremental, field, score, redundancy)
            if best is None or option[0] > best[0] or (option[0] == best[0] and option[1] < best[1]):
                best = option
        if best is None or best[0] <= 0.0:
            records.extend(stage_rows)
            break
        incremental, field, raw_cmi, redundancy = best
        for row in stage_rows:
            row["selected"] = row["field"] == field
        records.extend(stage_rows)
        selected.append(field)
        remaining.remove(field)
    return selected, pd.DataFrame(records)


def _uniform_subsample(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    index = np.linspace(0, len(frame) - 1, maximum, dtype=np.int64)
    return frame.iloc[np.unique(index)].copy()


def _params_from_trial(trial: optuna.Trial, n_rows: int) -> dict[str, Any]:
    depth = trial.suggest_int("max_depth", 3, 5)
    # Keep Optuna's categorical distribution stable across trials.  The
    # requested capacity is then safely capped by the chosen maximum depth.
    leaf_depth = trial.suggest_int("leaf_depth", 3, 5)
    return {
        "n_estimators": 500,
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.07, log=True),
        "max_depth": depth,
        "num_leaves": min(2 ** depth - 1, 2 ** leaf_depth - 1),
        "min_child_samples": max(80, int(math.ceil(n_rows * trial.suggest_float("min_child_fraction", 0.015, 0.08)))),
        "colsample_bytree": trial.suggest_float("feature_fraction", 0.55, 0.90),
        "subsample": trial.suggest_float("bagging_fraction", 0.55, 0.90),
        "subsample_freq": 1,
        "reg_alpha": trial.suggest_float("lambda_l1", 1e-4, 5.0, log=True),
        "reg_lambda": trial.suggest_float("lambda_l2", 0.3, 35.0, log=True),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127]),
    }


def _fit_probability(train: pd.DataFrame, held: pd.DataFrame, fields: Sequence[str], target: np.ndarray, params: dict[str, Any], *, seed: int, eval_set: tuple[pd.DataFrame, np.ndarray] | None = None, return_model: bool = False) -> tuple[np.ndarray, dict[str, float], lgb.LGBMClassifier | None, pd.Series]:
    x_train, med = _safe(train, fields)
    x_held, _ = _safe(held, fields, med)
    y = np.asarray(target, dtype=np.int8)
    if y.min() == y.max():
        return np.full(len(held), float(y.mean()), np.float32), {"feature_count": len(fields), "best_iteration": 0.0}, None, med
    # Hundreds of sequential ablation fits are run in one process.  A single
    # LightGBM worker avoids retaining several native thread arenas per fit,
    # which is materially safer than a small speed gain here.
    model = lgb.LGBMClassifier(objective="binary", random_state=seed, n_jobs=1, verbosity=-1, **params)
    fit_args: dict[str, Any] = {}
    if eval_set is not None:
        val, val_target = eval_set
        x_val, _ = _safe(val, fields, med)
        fit_args = {"eval_set": [(x_val, val_target)], "callbacks": [lgb.early_stopping(30, verbose=False)]}
    model.fit(x_train, y, **fit_args)
    probability = np.asarray(model.predict_proba(x_held)[:, 1], dtype=np.float32)
    info = {"feature_count": float(len(fields)), "best_iteration": float(model.best_iteration_ or params["n_estimators"])}
    if return_model:
        return probability, info, model, med
    del model, x_train, x_held
    gc.collect()
    return probability, info, None, med


def _inner_hpo(train: pd.DataFrame, validation: pd.DataFrame, fields: Sequence[str], *, trials: int, seed: int) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    train = _uniform_subsample(train, 6500)
    validation = _uniform_subsample(validation, 2800)
    map_train, map_validation = _map_base(train, validation)
    train = train.copy(); validation = validation.copy()
    train["base_anchor"] = map_train; validation["base_anchor"] = map_validation
    target = (train.net_bps.to_numpy(float) - map_train > 0.0).astype(np.int8)
    target_val = (validation.net_bps.to_numpy(float) - map_validation > 0.0).astype(np.int8)
    fields = [field for field in fields if field in train.columns]
    if not fields:
        raise ValueError("empty HPO feature set")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True), pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=0))

    def objective(trial: optuna.Trial) -> float:
        params = _params_from_trial(trial, len(train))
        probability, info, _, _ = _fit_probability(train, validation, fields, target, params, seed=seed + trial.number * 97, eval_set=(validation, target_val))
        auc = float(roc_auc_score(target_val, probability)) if np.unique(target_val).size > 1 else 0.5
        brier = float(brier_score_loss(target_val, probability)) if np.unique(target_val).size > 1 else 0.25
        n = max(1, int(math.ceil(len(validation) * 0.05)))
        idx = np.argsort(-probability, kind="stable")[:n]
        top5_net = float(validation.net_bps.to_numpy(float)[idx].mean())
        complexity = 0.001 * params["max_depth"] + 0.0002 * params["num_leaves"]
        score = auc - 0.10 * brier + 0.0002 * top5_net - complexity
        trial.set_user_attr("auc", auc); trial.set_user_attr("brier", brier); trial.set_user_attr("top5_net_bps", top5_net)
        trial.set_user_attr("best_iteration", info["best_iteration"])
        trial.report(score, 0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best = study.best_trial
    params = _params_from_best(best.params, len(train), int(best.user_attrs.get("best_iteration", 120)))
    trial_rows = []
    for trial in study.trials:
        trial_rows.append({"trial": trial.number, "state": trial.state.name, "value": trial.value, **trial.params, **{f"metric_{key}": value for key, value in trial.user_attrs.items()}})
    # Refit only the HPO training period to create a leakage-safe gain audit.
    _, _, model, _ = _fit_probability(train, validation, fields, target, params, seed=seed + 777, return_model=True)
    importance = pd.DataFrame({"field": fields, "gain": model.booster_.feature_importance(importance_type="gain") if model is not None else np.zeros(len(fields))})
    del model
    gc.collect()
    return params, pd.DataFrame(trial_rows), importance


def _params_from_best(raw: dict[str, Any], n_rows: int, n_estimators: int) -> dict[str, Any]:
    depth = int(raw["max_depth"])
    leaves = min(2 ** depth - 1, 2 ** int(raw["leaf_depth"]) - 1)
    return {
        "n_estimators": max(35, min(500, int(n_estimators))),
        "learning_rate": float(raw["learning_rate"]), "max_depth": depth, "num_leaves": leaves,
        "min_child_samples": max(80, int(math.ceil(n_rows * float(raw["min_child_fraction"])))),
        "colsample_bytree": float(raw["feature_fraction"]), "subsample": float(raw["bagging_fraction"]), "subsample_freq": 1,
        "reg_alpha": float(raw["lambda_l1"]), "reg_lambda": float(raw["lambda_l2"]), "max_bin": int(raw["max_bin"]),
    }


def _hpo_reference(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = panel.loc[panel.__ts__.lt(HPO_TRAIN_END) & panel.label_available_ts.lt(HPO_TRAIN_END)].copy()
    validation = panel.loc[panel.__ts__.ge(HPO_TRAIN_END) & panel.__ts__.lt(HPO_CUTOFF)].copy()
    if len(train) < 500 or len(validation) < 250:
        raise RuntimeError("insufficient 2024 HPO reference support")
    return train, validation


def _field_groups(blocks: dict[str, list[str]], compact: Sequence[str], *, trimmed: dict[str, list[str]] | None = None) -> dict[str, list[str]]:
    """Predeclared joint ablation arms.  ``trimmed`` alters only broad joint."""
    core = ["base_anchor", "base_score", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak"]
    context = blocks["market_context"]
    membership = blocks["soft_membership"]
    support = blocks["activated_leaf_support"]
    ood = blocks["rule_path_ood_drift"]
    covariance = blocks["covariance_correlation_break"]
    model_state = blocks["model_state"]
    correctness = blocks["recent_correctness"]
    broad = [*context, *membership, *support, *ood, *covariance, *model_state, *correctness, *compact]
    arm = {
        "market_context": context,
        "compact_path_cmi": [*context, *membership, *compact],
        "support_ood": [*context, *membership, *support, *ood],
        "support_ood_covariance": [*context, *membership, *support, *ood, *covariance],
        "support_ood_state_correctness": [*context, *membership, *support, *ood, *model_state, *correctness],
        "joint_all_compact": broad,
        "joint_no_market_context": [*core, *membership, *support, *ood, *covariance, *model_state, *correctness, *compact],
    }
    if trimmed:
        arm["joint_all_compact_trimmed"] = [*trimmed["market_context"], *trimmed["soft_membership"], *trimmed["activated_leaf_support"], *trimmed["rule_path_ood_drift"], *trimmed["covariance_correlation_break"], *trimmed["model_state"], *trimmed["recent_correctness"], *compact[:4]]
        arm["joint_all_compact_trimmed"] = list(dict.fromkeys([*core, *arm["joint_all_compact_trimmed"]]))
    return {name: list(dict.fromkeys(fields)) for name, fields in arm.items()}


def _trim_fields(importance: pd.DataFrame, blocks: dict[str, list[str]], core: Sequence[str]) -> tuple[dict[str, list[str]], pd.DataFrame]:
    gain = importance.set_index("field")["gain"].to_dict()
    out: dict[str, list[str]] = {}
    audit: list[dict[str, object]] = []
    for block in ("market_context", "soft_membership", "activated_leaf_support", "rule_path_ood_drift", "covariance_correlation_break", "model_state", "recent_correctness"):
        fields = [field for field in blocks[block] if field not in core]
        values = sorted(((float(gain.get(field, 0.0)), field) for field in fields), key=lambda item: (-item[0], item[1]))
        keep_n = max(1, int(math.ceil(len(values) * 0.5)))
        keep = [field for _, field in values[:keep_n]]
        out[block] = keep
        for order, (value, field) in enumerate(values, 1):
            audit.append({"block": block, "field": field, "gain": value, "gain_rank": order, "kept_trim50": field in set(keep)})
    return out, pd.DataFrame(audit)


def _score_one_fold(train: pd.DataFrame, held: pd.DataFrame, configs: dict[str, list[str]], params: dict[str, dict[str, Any]], blocks: dict[str, list[str]], *, seed: int) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame, pd.DataFrame]:
    tr_anchor, te_anchor = _map_base(train, held)
    train = train.copy(); held = held.copy()
    train["base_anchor"] = tr_anchor; held["base_anchor"] = te_anchor
    target = (train.net_bps.to_numpy(float) - tr_anchor > 0.0).astype(np.int8)
    held_target = (held.net_bps.to_numpy(float) - te_anchor > 0.0).astype(np.int8)
    compact, cmi = _select_path_health(train, blocks["individual_path_health"], target, limit=8)
    result = held[["candidate_id", "__ts__", "month", "net_bps", "gross_bps", "base_plus_consensus25"]].copy()
    result["canonical_control"] = held.base_plus_consensus25.to_numpy(float)
    audits: list[dict[str, object]] = []
    imports: list[pd.DataFrame] = []
    contracts: list[dict[str, object]] = []
    for name, template in configs.items():
        fields = [field for field in template if field not in blocks["individual_path_health"]]
        # Dynamic CMI fields are injected by the symbolic compact arm; the
        # broad joint arms have their static part plus each fold's selection.
        if name in {"compact_path_cmi", "joint_all_compact", "joint_no_market_context"}:
            fields.extend(compact)
        if name == "joint_all_compact_trimmed":
            fields.extend(compact[:4])
        fields = list(dict.fromkeys(field for field in fields if field in train.columns))
        probability, info, model, _ = _fit_probability(train, held, fields, target, params[name], seed=seed + (zlib.adler32(name.encode()) % 100000), return_model=True)
        auc = float(roc_auc_score(held_target, probability)) if np.unique(held_target).size > 1 else float("nan")
        brier = float(brier_score_loss(held_target, probability)) if np.unique(held_target).size > 1 else float("nan")
        audits.append({"month": str(held.month.iloc[0]), "arm": name, "held_auc": auc, "held_brier": brier, "train_rows": len(train), "held_rows": len(held), **info, "compact_fields": len(compact)})
        contracts.extend({"month": str(held.month.iloc[0]), "arm": name, "field": field} for field in fields)
        if model is not None:
            importance = model.booster_.feature_importance(importance_type="gain")
            imports.append(pd.DataFrame({"month": str(held.month.iloc[0]), "arm": name, "field": fields, "gain": importance}))
            del model
            gc.collect()
        for lower in (0.00, 0.25, 0.50, 0.75):
            result[f"shrink__{name}__lo{int(lower * 100):02d}"] = 0.5 + (result.canonical_control.to_numpy(float) - 0.5) * (lower + (1.0 - lower) * probability)
        for alpha in (0.25, 0.50, 0.75, 1.00, 1.25):
            multiplier = np.clip(1.0 + alpha * (probability - 0.5), 1.0 - 0.5 * alpha, 1.0 + 0.5 * alpha)
            result[f"multiply__{name}__a{int(alpha * 100):03d}"] = result.canonical_control.to_numpy(float) * multiplier
    for field in [field for field in result if field.startswith(("shrink__", "multiply__"))]:
        result[field] = result[field].rank(pct=True, method="average").astype("float32")
    cmi["month"] = str(held.month.iloc[0])
    return result, audits, pd.concat(imports, ignore_index=True), pd.concat([cmi, pd.DataFrame()], ignore_index=True).assign(_kind="cmi"), pd.DataFrame(contracts)


def _development_rank(global_metrics: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    dev = monthly.loc[monthly.month.isin(DEVELOPMENT_MONTHS) & monthly["tail"].eq(0.05)].copy()
    rows: list[dict[str, object]] = []
    for arm, group in dev.groupby("arm", sort=True):
        values = group.net_bps_per_trade.to_numpy(float)
        med = float(np.median(values)); mad = float(np.median(np.abs(values - med))); worst = float(values.min())
        row = {"arm": arm, "development_top5_mean": float(values.mean()), "development_top5_median": med, "development_top5_mad": mad, "development_top5_worst": worst, "development_portability": med - .5 * mad - max(0.0, -worst)}
        all_top1 = global_metrics.loc[(global_metrics.arm == arm) & global_metrics["tail"].eq(0.01), "net_bps_per_trade"]
        row["all_2025_top1_net"] = float(all_top1.iloc[0]) if len(all_top1) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["development_portability", "development_top5_mean", "all_2025_top1_net"], ascending=False, kind="stable")


def run(*, out: Path = OUT, hpo_trials: int = 12, seed: int = SEED) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    panel, blocks, lineage, state_audits = _build_panel()
    hpo_train, hpo_validation = _hpo_reference(panel)
    # HPO's compact contract is selected solely from its 2024 training slice.
    hpo_anchor, _ = _map_base(hpo_train, hpo_validation)
    hpo_compact, hpo_cmi = _select_path_health(hpo_train.assign(base_anchor=hpo_anchor), blocks["individual_path_health"], (hpo_train.net_bps.to_numpy(float) - hpo_anchor > 0.0).astype(np.int8), limit=8)
    configs = _field_groups(blocks, hpo_compact)
    params: dict[str, dict[str, Any]] = {}
    trial_frames: list[pd.DataFrame] = []
    hpo_usage: list[pd.DataFrame] = []
    for ordinal, (name, fields) in enumerate(configs.items()):
        print(f"HPO_START {name}", flush=True)
        actual = [field for field in fields if field in hpo_train.columns or field == "base_anchor"]
        hp, trials, gain = _inner_hpo(hpo_train, hpo_validation, actual, trials=hpo_trials, seed=seed + ordinal * 1009)
        params[name] = hp
        trials["arm"] = name; trial_frames.append(trials)
        gain["arm"] = name; hpo_usage.append(gain)
        gc.collect()
        print(f"HPO_DONE {name}", flush=True)
    hpo_gain = pd.concat(hpo_usage, ignore_index=True)
    core = ["base_anchor", "base_score", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak"]
    broad_gain = hpo_gain.loc[hpo_gain.arm.eq("joint_all_compact")].copy()
    trimmed, trim_audit = _trim_fields(broad_gain, blocks, core)
    configs = _field_groups(blocks, hpo_compact, trimmed=trimmed)
    trim_fields = configs["joint_all_compact_trimmed"]
    print("HPO_START joint_all_compact_trimmed", flush=True)
    hp, trials, gain = _inner_hpo(hpo_train, hpo_validation, trim_fields, trials=hpo_trials, seed=seed + 99991)
    params["joint_all_compact_trimmed"] = hp
    trials["arm"] = "joint_all_compact_trimmed"; trial_frames.append(trials)
    gain["arm"] = "joint_all_compact_trimmed"; hpo_gain = pd.concat([hpo_gain, gain], ignore_index=True)
    gc.collect()
    print("HPO_DONE joint_all_compact_trimmed", flush=True)
    parts: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    importance_parts: list[pd.DataFrame] = []
    cmi_parts: list[pd.DataFrame] = []
    contract_parts: list[pd.DataFrame] = []
    for month_no, month in enumerate(MONTHS):
        print(f"FOLD_START {month}", flush=True)
        cutoff = pd.Timestamp(month, tz="UTC")
        train = panel.loc[panel.__ts__.lt(cutoff) & panel.label_available_ts.lt(cutoff)].copy()
        held = panel.loc[panel.month.astype(str).eq(month)].copy()
        if len(train) < 500 or held.empty:
            continue
        result, audit, importance, cmi, contract = _score_one_fold(train, held, configs, params, blocks, seed=seed + month_no * 10000)
        parts.append(result); audits.extend(audit); importance_parts.append(importance); cmi_parts.append(cmi); contract_parts.append(contract)
        print(f"FOLD_DONE {month}", flush=True)
    prediction = pd.concat(parts, ignore_index=True)
    arms = ["canonical_control", *[field for field in prediction if field.startswith(("shrink__", "multiply__"))]]
    glob, monthly, stability = _metric_table(prediction, arms)
    ranked = _development_rank(glob, monthly)
    top5 = ranked.head(5).arm.tolist()
    all_importance = pd.concat(importance_parts, ignore_index=True)
    field_to_block = {field: block for block, fields in blocks.items() for field in fields}
    field_to_block.update({field: "base_core" for field in core})
    all_importance["block"] = all_importance.field.map(field_to_block).fillna("fold_compact_path")
    usage = all_importance.groupby(["arm", "block", "field"], observed=True).agg(mean_gain=("gain", "mean"), median_gain=("gain", "median"), used_months=("gain", lambda v: int((v > 0.0).sum())), folds=("month", "nunique")).reset_index()
    block_usage = usage.groupby(["arm", "block"], observed=True).agg(fields=("field", "nunique"), total_gain=("mean_gain", "sum"), nonzero_fields=("used_months", lambda v: int((v > 0).sum()))).reset_index()
    top_detail = {
        "global": glob.loc[glob.arm.isin(top5)].copy(), "monthly": monthly.loc[monthly.arm.isin(top5)].copy(),
        "stability": stability.loc[stability.arm.isin(top5)].copy(), "usage": usage.loc[usage.arm.isin([name.split("__")[1] if "__" in name else name for name in top5])].copy(),
    }
    prediction.to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    glob.to_parquet(out / "metrics_global.parquet", index=False); monthly.to_parquet(out / "metrics_monthly.parquet", index=False); stability.to_parquet(out / "metrics_stability.parquet", index=False)
    ranked.to_parquet(out / "development_selection.parquet", index=False)
    pd.DataFrame(audits).to_parquet(out / "model_fold_audit.parquet", index=False)
    pd.concat(trial_frames, ignore_index=True).to_parquet(out / "hpo_trials.parquet", index=False)
    pd.DataFrame([{"arm": name, "params_json": json.dumps(value, sort_keys=True)} for name, value in params.items()]).to_parquet(out / "hpo_winners.parquet", index=False)
    hpo_gain.to_parquet(out / "hpo_feature_gain.parquet", index=False); trim_audit.to_parquet(out / "trim_feature_audit.parquet", index=False)
    all_importance.to_parquet(out / "feature_usage_by_fold.parquet", index=False); usage.to_parquet(out / "feature_usage_summary.parquet", index=False); block_usage.to_parquet(out / "feature_usage_by_block.parquet", index=False)
    pd.concat(cmi_parts, ignore_index=True).to_parquet(out / "path_health_cmi_selection.parquet", index=False)
    hpo_cmi.assign(month="2024_hpo_train").to_parquet(out / "path_health_hpo_cmi_selection.parquet", index=False)
    pd.concat(contract_parts, ignore_index=True).to_parquet(out / "fold_feature_contract.parquet", index=False)
    lineage.to_parquet(out / "lineage.parquet", index=False); state_audits.to_parquet(out / "path_state_audits.parquet", index=False)
    for name, table in top_detail.items():
        table.to_parquet(out / f"top5_configs_{name}.parquet", index=False)
    correctness = {
        "schema": "tp6_sl4_compact_path_joint_hpo_correctness_v1",
        "canonical_control": "exact TP6_SL4_BASE_CONSENSUS_CANONICAL_20260807 base_plus_consensus25 on identical long candidate IDs",
        "target": "P(exact TP6/SL4/H12 net bps > train-only isotonic base-map expected bps)",
        "individual_path_health_is_asof_label_available_ts": True,
        "individual_path_health_is_active_and_support_gated": True,
        "cmi_selection_is_outer_train_only": True,
        "hpo_period": "2024-04 through 2024-11; train before 2024-10, validation 2024-10 through 2024-11",
        "hpo_uses_subsample_and_median_pruner": True,
        "outer_2025_months_not_used_for_hpo_or_cmi_selection": True,
        "k9_soft_memberships_explicit_in_structural_challengers": True,
        "modulation_grid_predeclared": {"shrink_lower": [0.0, 0.25, 0.5, 0.75], "multiplier_alpha": [0.25, 0.5, 0.75, 1.0, 1.25]},
        "all_prediction_scores_finite": bool(np.isfinite(prediction[arms].to_numpy(float)).all()),
        "candidate_month_unique": bool(not prediction.duplicated(["candidate_id", "month"]).any()),
        "scope": "2025 long-only matched canonical development replay; Oct-Dec confirmation is kept separate in monthly outputs. No residual-path layer has been opened.",
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    selected_arm = str(ranked.iloc[0]["arm"])
    canonical_top5 = float(glob.loc[(glob.arm.eq("canonical_control")) & glob["tail"].eq(0.05), "net_bps_per_trade"].iloc[0])
    selected_top5 = float(glob.loc[(glob.arm.eq(selected_arm)) & glob["tail"].eq(0.05), "net_bps_per_trade"].iloc[0])
    confirm = monthly.loc[monthly.month.isin(CONFIRMATION_MONTHS) & monthly["tail"].eq(0.05)]
    canonical_confirmation = float(confirm.loc[confirm.arm.eq("canonical_control"), "net_bps_per_trade"].mean())
    selected_confirmation = float(confirm.loc[confirm.arm.eq(selected_arm), "net_bps_per_trade"].mean())
    gate_pass = (selected_top5 - canonical_top5 >= 5.0) and (selected_confirmation >= canonical_confirmation)
    gate = {
        "gate": "BASE_LEAF_RELIABILITY_CONCLUSIVE",
        "criterion": "A selected arm must beat canonical control by >=5 bps at global Top-5 and have no worse Oct-Dec mean Top-5 result before residual-path modeling opens.",
        "selected_by_development": selected_arm,
        "development_top5_configs": top5,
        "canonical_global_top5_net_bps": canonical_top5,
        "selected_global_top5_net_bps": selected_top5,
        "global_top5_uplift_bps": selected_top5 - canonical_top5,
        "canonical_confirmation_top5_mean_bps": canonical_confirmation,
        "selected_confirmation_top5_mean_bps": selected_confirmation,
        "confirmation_top5_uplift_bps": selected_confirmation - canonical_confirmation,
        "status": "PASS_OPEN_RESIDUAL_PATH_STAGE" if gate_pass else "FAIL_HOLD_RESIDUAL_PATH_STAGE",
        "next_stage": "Residual-leaf/path health remains closed until a selected base-leaf arm clears both the global and confirmation criteria." if not gate_pass else "A residual-leaf/path bounded-reliability ablation may now be opened on the same contract.",
    }
    (out / "residual_path_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    manifest = {
        "schema": "tp6_sl4_compact_path_joint_hpo_20260809_v1", "status": "COMPLETE", "seed": seed, "rows": len(prediction), "months": list(MONTHS),
        "hpo": {"trials_per_arm": hpo_trials, "train_end": "2024-10-01", "validation": "2024-10-01 to 2024-12-01", "max_train_subsample": 6500, "median_pruner": True},
        "selection": {"development_months": list(DEVELOPMENT_MONTHS), "confirmation_months": list(CONFIRMATION_MONTHS), "tie_break": "development portability, then top-5 mean, then all-2025 top-1"},
        "contracts": {"label": "exact TP6/SL4/H12 net cost encoded once", "base": "frozen canonical base+consensus", "path_health": "prior resolved as-of state", "ranking": "pooled global ranking after held-month transform normalization"},
        "blocks": {name: len(fields) for name, fields in blocks.items()}, "artifacts": sorted(path.name for path in out.iterdir()),
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    top = glob.loc[glob["tail"].eq(0.05)].merge(ranked[["arm", "development_portability"]], on="arm", how="left").sort_values("net_bps_per_trade", ascending=False).head(30)
    report = [
        "# Compact recurrent-path health and joint reliability ablation — TP6/SL4/H12", "",
        "This is a long-only, matched 2025 canonical-development replay. The reference is the exact `base_plus_consensus25` score from the canonical TP6/SL4 stack. 2024-only HPO and fold-local, training-only CMI selection are frozen before every 2025 held month; no result is an untouched final test.", "",
        "## Global Top-5 grid", "", top.round(3).to_string(index=False), "",
        "## Development selection and confirmation discipline", "", ranked.head(12).round(3).to_string(index=False), "",
        "## Feature use by block", "", block_usage.sort_values(["arm", "total_gain"], ascending=[True, False]).round(3).to_string(index=False), "",
        "## Residual-path gate", "", json.dumps(gate, indent=2), "",
        "## Correctness", "", json.dumps(correctness, indent=2), "",
    ]
    (out / "COMPACT_PATH_JOINT_HPO_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"out": str(out), "rows": len(prediction), "arms": len(arms), "top5_configs": top5}, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--hpo-trials", type=int, default=12)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(out=args.out, hpo_trials=args.hpo_trials, seed=args.seed)
