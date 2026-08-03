#!/usr/bin/env python3
"""Non-walk-forward OOF diagnostic on the existing 2022--23 transition panel.

This is intentionally a *learnability* study, not a deployment backtest.  Each
calendar-block holdout is scored by models that may use labels from later
blocks.  That answers whether the historical causal feature substrate contains
the information required by the target; it must not be used as causal or live
performance evidence.

Every score is in exact frozen-policy H12 net-return units and all tail metrics
are one pooled global selection across both sides and timestamps.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
CONTEXT = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_transition_context_continuation_20260730_v1/context.parquet"
LABELS = ROOT / "data_perp/artifacts/failure_2022_2023_pf_exact1m_multitask_labels_20260730_v1/joined_multitask_labels.parquet"
SCORES = ROOT / "data_perp/artifacts/reconstructed_base_residual_stack_2022_2024_20260730_v4/oof_scores.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/historical_transition_target_learnability_20260731_v1"
IDENTITY = ["candidate_id", "__ts__", "__symbol__", "side_name"]
TARGET = "execution_net_ev_12h"
SCORE_FEATURES = ["score_base_alpha", "score_residual_alpha", "score_base_expected_ev", "score_residual_expected_ev"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    temp = path.with_name(f".{path.name}.{os.getpid()}.partial")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temp, path)


def rank_ic(left: pd.Series, right: pd.Series) -> float:
    valid = pd.to_numeric(left, errors="coerce").notna() & pd.to_numeric(right, errors="coerce").notna()
    if valid.sum() < 3:
        return float("nan")
    return float(left.loc[valid].rank().corr(right.loc[valid].rank()))


def matrix(train: pd.DataFrame, evaluate: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    source = train.loc[:, columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    median = source.median().fillna(0.0)
    x_train = source.fillna(median).to_numpy(np.float32)
    x_eval = evaluate.loc[:, columns].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(median).to_numpy(np.float32)
    return x_train, x_eval


def panel() -> tuple[pd.DataFrame, list[str]]:
    context = pd.read_parquet(CONTEXT)
    labels = pd.read_parquet(LABELS, columns=[*IDENTITY, TARGET, "execution_gross_ev_12h", "execution_cost_return", "__opportunity_occurred_12h__"])
    scores = pd.read_parquet(SCORES, columns=[*IDENTITY, *SCORE_FEATURES, "residual_is_oof"])
    for frame in (context, labels, scores):
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        if frame.duplicated(IDENTITY).any():
            raise ValueError("source does not have a unique candidate identity")
    scores = scores.loc[scores["__ts__"].between(context["__ts__"].min(), context["__ts__"].max())].copy()
    if not scores["residual_is_oof"].astype(bool).all():
        raise ValueError("historical score source includes non-OOF rows")
    result = context.merge(labels, on=IDENTITY, how="inner", validate="one_to_one").merge(scores.drop(columns="residual_is_oof"), on=IDENTITY, how="inner", validate="one_to_one")
    if len(result) != len(context) or len(result) != len(labels) or len(result) != len(scores):
        raise ValueError("context, labels, and score identities are not exactly matched")
    if not np.allclose(result["execution_gross_ev_12h"] - result["execution_cost_return"], result[TARGET], atol=1e-12, rtol=0.0):
        raise ValueError("gross minus cost does not reproduce net target")
    excluded = set(IDENTITY + ["__decision_ts__", "source_family", "transition_context_available"])
    transition = [name for name in context.columns if name not in excluded and pd.api.types.is_numeric_dtype(context[name])]
    if not transition or result[transition].isna().all(axis=None):
        raise ValueError("no numeric decision-time transition features")
    result["calendar_block"] = result["__ts__"].dt.strftime("%Y-%m")
    return result.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True), transition


def feature_sets(transition: list[str], *, compact: bool) -> dict[str, list[str]]:
    """Return only predeclared causal mechanism groups.

    The compact arms are deliberately non-overlapping except for the common
    score control.  They answer whether one mechanism is incremental; they do
    not search arbitrary feature subsets or constitute feature selection.
    """
    output = {"score_only": SCORE_FEATURES}
    if not compact:
        return {**output, "score_plus_transition": [*SCORE_FEATURES, *transition]}
    state = [c for c in transition if c.startswith("state_context__")]
    regime_change = [c for c in transition if c.startswith("mkt_regime_change__")]
    dynamic = [c for c in transition if c.startswith("transition_new__")]
    breadth = [c for c in dynamic if any(token in c for token in ("breadth", "dispersion", "correlation", "fragmented", "relative_strength"))]
    flow = [c for c in dynamic if c not in breadth]
    static = [c for c in transition if c not in set(state + regime_change + dynamic)]
    arms = {
        "state_geometry": state,
        "regime_change_dynamics": regime_change,
        "transition_breadth_correlation": breadth,
        "transition_flow_recovery": flow,
        "static_transition_state": static,
        "all_transition_control": transition,
    }
    if any(not values for values in arms.values()):
        raise ValueError("a compact mechanism group is empty")
    return {**output, **{name: [*SCORE_FEATURES, *values] for name, values in arms.items()}}


def fit_direct(train: pd.DataFrame, evaluate: pd.DataFrame, features: list[str], seed: int) -> np.ndarray:
    x_train, x_eval = matrix(train, evaluate, features)
    model = lgb.LGBMRegressor(
        n_estimators=220, learning_rate=0.035, num_leaves=23, min_child_samples=240,
        colsample_bytree=0.75, reg_lambda=5.0, random_state=seed, n_jobs=4, verbosity=-1,
    )
    model.fit(x_train, train[TARGET].to_numpy(float))
    return np.asarray(model.predict(x_eval), dtype=float)


def fit_hurdle(train: pd.DataFrame, evaluate: pd.DataFrame, features: list[str], seed: int) -> tuple[np.ndarray, np.ndarray]:
    x_train, x_eval = matrix(train, evaluate, features)
    y = train[TARGET].gt(0.0).astype(np.int8).to_numpy()
    classifier = lgb.LGBMClassifier(
        n_estimators=180, learning_rate=0.04, num_leaves=19, min_child_samples=220,
        colsample_bytree=0.75, reg_lambda=5.0, random_state=seed, n_jobs=4, verbosity=-1,
    ).fit(x_train, y)
    probability = np.asarray(classifier.predict_proba(x_eval)[:, 1], dtype=float)
    values: list[np.ndarray] = []
    for value, condition in ((1, y == 1), (0, y == 0)):
        sample = np.flatnonzero(condition)
        if len(sample) < 500:
            values.append(np.full(len(evaluate), float(train.loc[condition, TARGET].mean())))
            continue
        regressor = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.04, num_leaves=15, min_child_samples=180,
            colsample_bytree=0.75, reg_lambda=6.0, random_state=seed + value + 31, n_jobs=4, verbosity=-1,
        ).fit(x_train[sample], train.iloc[sample][TARGET].to_numpy(float))
        values.append(np.asarray(regressor.predict(x_eval), dtype=float))
    return probability * values[0] + (1.0 - probability) * values[1], probability


def selected_metrics(frame: pd.DataFrame, score_name: str, fraction: float) -> dict[str, float | int]:
    count = int(np.ceil(len(frame) * fraction))
    chosen = frame.sort_values([score_name, "candidate_id"], ascending=[False, True], kind="stable").head(count)
    return {
        "selected_rows": int(len(chosen)),
        "mean_net_bps": float(chosen[TARGET].mean() * 1e4),
        "mean_gross_bps": float(chosen["execution_gross_ev_12h"].mean() * 1e4),
        "mean_cost_bps": float(chosen["execution_cost_return"].mean() * 1e4),
        "positive_net_fraction": float(chosen[TARGET].gt(0).mean()),
        "long_share": float(chosen["side_name"].eq("long").mean()),
    }


def run(output: Path, *, compact_groups: bool = False) -> Path:
    if output.exists():
        raise FileExistsError(output)
    data, transition_features = panel()
    feature_groups = feature_sets(transition_features, compact=compact_groups)
    groups = data["calendar_block"].to_numpy()
    splits = list(GroupKFold(n_splits=5).split(data, groups=groups))
    output_scores: list[pd.DataFrame] = []
    folds: list[dict[str, object]] = []
    for feature_name, features in feature_groups.items():
        per_arm = data.loc[:, [*IDENTITY, TARGET, "execution_gross_ev_12h", "execution_cost_return", "__opportunity_occurred_12h__"]].copy()
        per_arm["direct_score"] = np.nan
        per_arm["hurdle_score"] = np.nan
        per_arm["p_net_positive"] = np.nan
        for fold, (train_index, evaluate_index) in enumerate(splits):
            for side in ("long", "short"):
                local_train = data.iloc[train_index].loc[lambda x: x.side_name.eq(side)]
                local_eval = data.iloc[evaluate_index].loc[lambda x: x.side_name.eq(side)]
                direct = fit_direct(local_train, local_eval, features, 1729 + fold)
                hurdle, probability = fit_hurdle(local_train, local_eval, features, 8191 + fold)
                per_arm.loc[local_eval.index, "direct_score"] = direct
                per_arm.loc[local_eval.index, "hurdle_score"] = hurdle
                per_arm.loc[local_eval.index, "p_net_positive"] = probability
            folds.append({
                "feature_set": feature_name, "fold": fold,
                "train_blocks": sorted(set(data.iloc[train_index]["calendar_block"])),
                "evaluation_blocks": sorted(set(data.iloc[evaluate_index]["calendar_block"])),
                "train_rows": int(len(train_index)), "evaluation_rows": int(len(evaluate_index)),
                "validation": "symmetric_calendar_block_oof_not_walk_forward",
            })
        if per_arm[["direct_score", "hurdle_score", "p_net_positive"]].isna().any().any():
            raise ValueError("OOF scoring incomplete")
        per_arm["feature_set"] = feature_name
        output_scores.append(per_arm)
    predictions = pd.concat(output_scores, ignore_index=True)
    metrics: list[dict[str, object]] = []
    periods: list[dict[str, object]] = []
    for feature_name, local in predictions.groupby("feature_set", sort=True):
        for target_name in ("direct_score", "hurdle_score"):
            intercept, slope = np.polyfit(local[target_name].to_numpy(float), local[TARGET].to_numpy(float), 1)[1], np.polyfit(local[target_name].to_numpy(float), local[TARGET].to_numpy(float), 1)[0]
            record: dict[str, object] = {
                "feature_set": feature_name, "target_arm": target_name.removesuffix("_score"), "rows": int(len(local)),
                "execution_rank_ic": rank_ic(local[target_name], local[TARGET]),
                "alpha_rank_ic": rank_ic(local[target_name], local["__opportunity_occurred_12h__"]),
                "calibration_intercept_bps": float(intercept * 1e4), "calibration_slope": float(slope),
                "threshold_rows": int(local[target_name].gt(0.0).sum()),
                "threshold_net_bps": float(local.loc[local[target_name].gt(0.0), TARGET].mean() * 1e4),
            }
            for fraction in (0.01, 0.05, 0.10):
                record.update({f"top_{int(fraction * 100)}_{k}": v for k, v in selected_metrics(local, target_name, fraction).items()})
            metrics.append(record)
            for month, month_rows in local.groupby(local["__ts__"].dt.strftime("%Y-%m"), sort=True):
                periods.append({"feature_set": feature_name, "target_arm": target_name.removesuffix("_score"), "month": month, **selected_metrics(month_rows, target_name, .10)})
        p = local["p_net_positive"].to_numpy(float)
        y = local[TARGET].gt(0.0).astype(int).to_numpy()
        metrics.append({"feature_set": feature_name, "target_arm": "hurdle_probability_diagnostic", "rows": int(len(local)), "net_positive_prevalence": float(y.mean()), "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p))})
    temporary = Path(tempfile.mkdtemp(dir=output.parent, prefix=f".{output.name}."))
    try:
        predictions.to_parquet(temporary / "oof_predictions.parquet", index=False)
        pd.DataFrame(folds).to_parquet(temporary / "fold_provenance.parquet", index=False)
        pd.DataFrame(metrics).to_csv(temporary / "aggregate_metrics.csv", index=False)
        pd.DataFrame(periods).to_csv(temporary / "monthly_global_top10_metrics.csv", index=False)
        contract = {
            "evidence_scope": "non_walk_forward_calendar_block_oof_learnability_diagnostic",
            "promotion_eligible": False,
            "execution": "exact 1m frozen-policy H12 gross minus frozen/current-spread cost",
            "selection": "one pooled global top-k over both sides and timestamps; deterministic candidate_id tie break",
            "feature_availability": "existing transition context is decision-known; no action/timing/MAE/target-price or realised-path labels are features",
            "feature_sets": {key: len(value) for key, value in feature_groups.items()},
            "targets": {"direct": "exact execution_net_ev_12h", "hurdle": "P(net>0) plus conditional positive/negative net magnitudes"},
            "important_limit": "calendar block models see labels from later calendar blocks; results assess information availability, not causal transport or deployability",
        }
        write_json(temporary / "contract.json", contract)
        files = [temporary / name for name in ("oof_predictions.parquet", "fold_provenance.parquet", "aggregate_metrics.csv", "monthly_global_top10_metrics.csv", "contract.json")]
        manifest = {"schema": "historical_transition_target_learnability_v1", "status": "COMPLETE_RESEARCH_ONLY", "sources": {str(path): sha256(path) for path in (CONTEXT, LABELS, SCORES)}, "rows": int(len(data)), "coverage": [str(data["__ts__"].min()), str(data["__ts__"].max())], "compact_mechanism_groups": compact_groups, "outputs_sha256": {path.name: sha256(path) for path in files}, **contract}
        write_json(temporary / "manifest.json", manifest)
        (temporary / "manifest.sha256").write_text(f"{sha256(temporary / 'manifest.json')}  manifest.json\n")
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--compact-mechanism-groups", action="store_true", help="Test predeclared transition mechanisms individually, not a bulk context block.")
    args = parser.parse_args()
    print(run(args.output, compact_groups=args.compact_mechanism_groups))
