#!/usr/bin/env python3
"""Compare conservative local residual-state learners on daily top-tail states.

The input is the leakage-audited daily state output from
``run_daily_observable_failure_state_screen.py``.  Every model is fit per
side x archetype on earlier daily rows only.  It compares a robust linear
baseline, a sparse interaction-logistic challenger, and a RuleFit-style
shallow-tree leaf model.  This is discovery evidence, not a trading overlay.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, mutual_info_score
from sklearn.preprocessing import OneHotEncoder, RobustScaler


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATE = ROOT / (
    "data_perp/reports/daily_observable_failure_state_screen_frozen_meta_2025_20260720_v3_relative_tail/"
    "daily_observable_state.parquet"
)
DEFAULT_OUTPUT = ROOT / "data_perp/reports/local_residual_state_challengers_20260720_v1"
TARGETS = (
    "target__negative_relative_ev_day",
    "target__negative_hit_surprise_day",
    "target__positive_hit_surprise_day",
)
CANONICAL_RESIDUAL_EVENT_CONTEXT = (
    "resid_event_aegmm_local_support_log1p",
    "resid_event_aegmm_gmm_entropy",
    "resid_event_aegmm_expected_market_peer_surprise",
    "resid_event_aegmm_expected_ev_timestamp_neutral_surprise",
)


def _screen(frame: pd.DataFrame, target: np.ndarray, columns: list[str], maximum: int) -> list[str]:
    """Train-only binned MI screen with robust handling for thin daily cells."""

    rows: list[tuple[float, str]] = []
    for name in columns:
        values = pd.to_numeric(frame[name], errors="coerce")
        valid = values.notna().to_numpy() & np.isfinite(target)
        if valid.sum() < 60 or np.unique(target[valid]).size < 2:
            continue
        try:
            bins = pd.qcut(values.loc[valid], q=8, labels=False, duplicates="drop")
        except ValueError:
            continue
        if bins.nunique(dropna=True) < 2:
            continue
        rows.append((float(mutual_info_score(target[valid], bins)), name))
    return [name for _, name in sorted(rows, reverse=True)[:maximum]]


def _matrix(train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    x_train = train[columns].apply(pd.to_numeric, errors="coerce")
    medians = x_train.median().fillna(0.0)
    x_train = x_train.fillna(medians).clip(-1e6, 1e6)
    x_test = test[columns].apply(pd.to_numeric, errors="coerce").fillna(medians).clip(-1e6, 1e6)
    return scaler.fit_transform(x_train).astype(np.float32), scaler.transform(x_test).astype(np.float32)


def _interaction_matrix(
    train: pd.DataFrame, test: pd.DataFrame, columns: list[str], target: np.ndarray,
    maximum_rules: int = 20,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate sparse, interpretable high/low pair rules from train only."""

    if len(columns) < 2:
        return np.zeros((len(train), 0), np.float32), np.zeros((len(test), 0), np.float32), []
    cuts: dict[str, tuple[float, float]] = {}
    indicators_train: dict[str, np.ndarray] = {}
    indicators_test: dict[str, np.ndarray] = {}
    for name in columns:
        values = pd.to_numeric(train[name], errors="coerce")
        lo, hi = values.quantile([0.2, 0.8]).to_numpy(dtype=np.float64)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
            continue
        cuts[name] = (float(lo), float(hi))
        for suffix, threshold, above in (("low", lo, False), ("high", hi, True)):
            key = f"{name}__{suffix}"
            a = pd.to_numeric(train[name], errors="coerce").to_numpy(dtype=np.float64)
            b = pd.to_numeric(test[name], errors="coerce").to_numpy(dtype=np.float64)
            indicators_train[key] = (a >= threshold if above else a <= threshold).astype(np.float32)
            indicators_test[key] = (b >= threshold if above else b <= threshold).astype(np.float32)
    keys = sorted(indicators_train)
    candidates: list[tuple[float, str, np.ndarray, np.ndarray]] = []
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            if left.rsplit("__", 1)[0] == right.rsplit("__", 1)[0]:
                continue
            a = indicators_train[left] * indicators_train[right]
            if a.sum() < 12 or a.sum() > len(a) - 12:
                continue
            relevance = float(mutual_info_score(target, a.astype(np.int8)))
            candidates.append((relevance, f"{left} & {right}", a, indicators_test[left] * indicators_test[right]))
    # The sparse logistic model decides which rules survive jointly, after this
    # train-only nonlinear relevance screen limits the combinatorial search.
    candidates.sort(key=lambda row: (-row[0], row[1]))
    selected = candidates[:maximum_rules]
    if not selected:
        return np.zeros((len(train), 0), np.float32), np.zeros((len(test), 0), np.float32), []
    return (
        np.column_stack([row[2] for row in selected]).astype(np.float32),
        np.column_stack([row[3] for row in selected]).astype(np.float32),
        [row[1] for row in selected],
    )


def _metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float]:
    if len(y) == 0 or np.unique(y).size < 2:
        return {"average_precision": np.nan, "top10_precision": np.nan, "top10_lift": np.nan}
    cutoff = float(np.quantile(score, 0.90))
    selected = score >= cutoff
    precision = float(y[selected].mean()) if selected.any() else np.nan
    prevalence = float(y.mean())
    return {
        "average_precision": float(average_precision_score(y, score)),
        "top10_precision": precision,
        "top10_lift": precision / prevalence if prevalence > 0 and np.isfinite(precision) else np.nan,
    }


def _fit_predict(method: str, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, seed: int) -> np.ndarray:
    if method == "ridge_logit":
        model = LogisticRegression(C=0.15, class_weight="balanced", max_iter=600, random_state=seed)
        return model.fit(x_train, y_train).predict_proba(x_test)[:, 1]
    if method == "interaction_logit":
        model = LogisticRegression(C=0.08, class_weight="balanced", max_iter=800, random_state=seed)
        return model.fit(x_train, y_train).predict_proba(x_test)[:, 1]
    forest = RandomForestClassifier(
        n_estimators=48, max_depth=2, min_samples_leaf=14, max_features=0.75,
        class_weight="balanced_subsample", n_jobs=1, random_state=seed,
    ).fit(x_train, y_train)
    leaves_train = forest.apply(x_train).reshape(len(x_train), -1)
    leaves_test = forest.apply(x_test).reshape(len(x_test), -1)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    rule_train = encoder.fit_transform(leaves_train)
    rule_test = encoder.transform(leaves_test)
    model = LogisticRegression(
        C=0.06, penalty="l1", solver="liblinear", class_weight="balanced", max_iter=400,
        random_state=seed,
    )
    return model.fit(rule_train, y_train).predict_proba(rule_test)[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-train-days", type=int, default=90)
    parser.add_argument("--eval-days", type=int, default=30)
    parser.add_argument("--max-features", type=int, default=10)
    parser.add_argument("--side", default="")
    parser.add_argument("--archetype", default="")
    args = parser.parse_args()

    state = pd.read_parquet(args.state)
    state["day"] = pd.to_datetime(state["day"], utc=True)
    features = [name for name in state if name.startswith(("state_med__", "state_p90__"))]
    residual_context = [
        name for name in features
        if name.removeprefix("state_med__").removeprefix("state_p90__")
        in CANONICAL_RESIDUAL_EVENT_CONTEXT
    ]
    if any(name in state for name in CANONICAL_RESIDUAL_EVENT_CONTEXT) and not residual_context:
        raise ValueError("Canonical residual-event context was present but not admitted to daily features")
    rows: list[dict[str, object]] = []
    rule_rows: list[dict[str, object]] = []
    for (side, archetype), local in state.groupby(["side_name", "archetype_policy_key"], observed=True, sort=True):
        if args.side and str(side) != str(args.side).lower():
            continue
        if args.archetype and str(archetype) != str(args.archetype):
            continue
        local = local.sort_values("day", kind="stable").reset_index(drop=True)
        for target_name in TARGETS:
            if target_name not in local:
                continue
            first_eval = local["day"].min() + pd.Timedelta(days=int(args.min_train_days))
            fold = 0
            print(f"local={side}__{archetype} target={target_name}", flush=True)
            while first_eval < local["day"].max():
                end = first_eval + pd.Timedelta(days=int(args.eval_days))
                train = local.loc[local["day"].lt(first_eval)].copy()
                test = local.loc[local["day"].ge(first_eval) & local["day"].lt(end)].copy()
                # A missing local target means the side/archetype did not have
                # enough comparable tail support that day. It is not an easy
                # negative and must never be coerced to False.
                train = train.loc[train[target_name].notna()].copy()
                test = test.loc[test[target_name].notna()].copy()
                y_train = train[target_name].astype(bool).to_numpy(dtype=np.int8)
                y_test = test[target_name].astype(bool).to_numpy(dtype=np.int8)
                if len(test) < 8 or y_train.sum() < 5 or y_train.sum() == len(y_train) or np.unique(y_test).size < 2:
                    first_eval = end; fold += 1; continue
                selected = _screen(train, y_train, features, int(args.max_features))
                if len(selected) < 2:
                    first_eval = end; fold += 1; continue
                x_train, x_test = _matrix(train, test, selected)
                x_rule_train, x_rule_test, rules = _interaction_matrix(train, test, selected, y_train)
                matrices = {
                    "ridge_logit": (x_train, x_test),
                    "interaction_logit": (np.column_stack([x_train, x_rule_train]), np.column_stack([x_test, x_rule_test])),
                    "rulefit_style": (x_train, x_test),
                }
                for method, (x_fit, x_score) in matrices.items():
                    score = _fit_predict(method, x_fit, y_train, x_score, seed=20260720 + fold)
                    rows.append({
                        "side_name": side, "archetype_policy_key": archetype,
                        "target": target_name, "fold": fold, "train_end": first_eval,
                        "eval_end": end, "train_days": len(train), "test_days": len(test),
                        "train_events": int(y_train.sum()), "test_events": int(y_test.sum()),
                        "method": method, "selected_features": "|".join(selected),
                        **_metrics(y_test, score),
                    })
                for rule in rules:
                    rule_rows.append({"side_name": side, "archetype_policy_key": archetype, "target": target_name, "fold": fold, "rule": rule})
                first_eval = end; fold += 1
    args.output.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    report.to_csv(args.output / "local_challenger_fold_metrics.csv", index=False)
    summary_columns = [
        "side_name", "archetype_policy_key", "target", "method", "folds",
        "test_days", "events", "average_precision", "top10_precision", "top10_lift",
    ]
    if report.empty:
        summary = pd.DataFrame(columns=summary_columns)
    else:
        summary = report.groupby(["side_name", "archetype_policy_key", "target", "method"], observed=True, as_index=False).agg(
            folds=("fold", "size"), test_days=("test_days", "sum"), events=("test_events", "sum"),
            average_precision=("average_precision", "mean"), top10_precision=("top10_precision", "mean"), top10_lift=("top10_lift", "mean"),
        ).sort_values(["target", "average_precision"], ascending=[True, False])
    summary.to_csv(args.output / "local_challenger_summary.csv", index=False)
    pd.DataFrame(rule_rows).to_csv(args.output / "selected_interaction_rules.csv", index=False)
    availability = {
        name: bool(
            f"state_med__{name}" in state.columns
            or f"state_p90__{name}" in state.columns
        )
        for name in CANONICAL_RESIDUAL_EVENT_CONTEXT
    }
    (args.output / "manifest.json").write_text(json.dumps({
        "schema": "local_residual_state_challengers_v1",
        "state": str(args.state), "methods": ["ridge_logit", "interaction_logit", "rulefit_style"],
        "canonical_residual_event_context_available_in_input": availability,
        "canonical_residual_event_context_note": (
            "Fields are generated from the frozen residual-event state using "
            "pre-entry inputs, frozen meta score, and archetype routing only."
        ),
        "feature_count": len(features), "targets": list(TARGETS),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
