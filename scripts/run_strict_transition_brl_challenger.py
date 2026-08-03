#!/usr/bin/env python3
"""Strict native Beta-Binomial MAP rule-list transition challenger.

Every arm is fitted, selected and calibrated with resolved 2022--2025 rows
only.  The 2026 panel is read once, after the winning configuration has been
frozen.  This is deliberately an interpretable challenger, not a policy gate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.residual_rule_models import BayesianRuleListArm
from scripts.run_strict_forward_transition_evaluation import (
    ART, CATALOGUE, CURRENT, TRAIN_END, causal_feature_columns, ece,
    global_top10, label_available, safe, sha256,
)

OUT = ART / "strict_transition_brl_challenger_20260730_v1"
FOLDS = (
    pd.Timestamp("2024-01-01", tz="UTC"),
    pd.Timestamp("2024-07-01", tz="UTC"),
    pd.Timestamp("2025-01-01", tz="UTC"),
    pd.Timestamp("2025-07-01", tz="UTC"),
)
HEADS: tuple[tuple[str, str], ...] = (
    ("stable_vs_transition", "target__transition_active"),
    ("onset_h1", "target__onset_within_1h"),
    ("onset_h3", "target__onset_within_3h"),
    ("onset_h6", "target__onset_within_6h"),
    ("onset_h12", "target__onset_within_12h"),
)
# A compact pre-registered search: increasing expressiveness, but still a
# human-readable bounded rule list.  No 2026 row is involved in this choice.
CONFIGS = (
    {"max_input_features": 6, "max_rules": 3, "max_rule_width": 1},
    {"max_input_features": 10, "max_rules": 3, "max_rule_width": 1},
    {"max_input_features": 10, "max_rules": 5, "max_rule_width": 2},
)
WEIGHTS = (1.0, 5.0)


def metric_values(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    return {
        "ap": float(average_precision_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else np.nan,
        "brier": float(brier_score_loss(y, p)),
        "ece10": float(ece(pd.Series(y), pd.Series(p))),
    }


def _arm(config: dict[str, int], *, seed: int) -> BayesianRuleListArm:
    return BayesianRuleListArm(
        seed=seed,
        max_rows=2_500,
        max_input_features=int(config["max_input_features"]),
        max_rules=int(config["max_rules"]),
        max_rule_width=int(config["max_rule_width"]),
        list_length_prior=3.0,
        list_width_prior=1.0,
        beta_alpha=1.0,
        beta_beta=1.0,
        min_rule_support=8,
    )


def _fit_predict(
    train: pd.DataFrame, test: pd.DataFrame, *, features: list[str], target: str,
    config: dict[str, int], positive_weight: float, seed: int,
) -> tuple[np.ndarray, BayesianRuleListArm]:
    y = pd.to_numeric(train[target], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    weights = np.where(y == 1, float(positive_weight), 1.0)
    arm = _arm(config, seed=seed).fit(
        train.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32),
        y, weights, features,
    )
    probability = arm.predict_proba(
        test.loc[:, features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    )
    return np.asarray(probability, dtype=float), arm


def _platt(prior: pd.DataFrame, raw: np.ndarray) -> np.ndarray:
    if len(prior) < 20 or prior["y"].nunique() < 2:
        return np.asarray(raw, dtype=float)
    model = LogisticRegression(max_iter=200, random_state=20260730)
    return model.fit(prior[["raw"]], prior["y"]).predict_proba(
        pd.DataFrame({"raw": raw})
    )[:, 1]


def _oof_predictions(
    frame: pd.DataFrame, *, features: list[str], target: str,
    config: dict[str, int], positive_weight: float,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold, start in enumerate(FOLDS):
        stop = start + pd.DateOffset(months=6)
        train = frame.loc[frame.source_utc.lt(start)]
        test = frame.loc[frame.source_utc.ge(start) & frame.source_utc.lt(stop)]
        if train.empty or test.empty or train[target].nunique() < 2:
            continue
        raw, _ = _fit_predict(
            train, test, features=features, target=target, config=config,
            positive_weight=positive_weight, seed=20260730 + fold,
        )
        rows.append(pd.DataFrame({
            "fold": fold, "source_utc": test.source_utc.to_numpy(),
            "y": pd.to_numeric(test[target], errors="coerce").fillna(0).astype(int).to_numpy(),
            "raw": raw,
        }))
    if not rows:
        raise ValueError(f"no blocked OOF folds for {target}")
    return pd.concat(rows, ignore_index=True)


def _selection_rows(
    frame: pd.DataFrame, *, head: str, target: str, features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    stored: dict[tuple[int, float], pd.DataFrame] = {}
    for config_id, config in enumerate(CONFIGS):
        for positive_weight in WEIGHTS:
            raw = _oof_predictions(
                frame, features=features, target=target, config=config,
                positive_weight=positive_weight,
            )
            stored[(config_id, positive_weight)] = raw
            for calibration in ("none", "platt"):
                calibrated: list[pd.DataFrame] = []
                for fold, group in raw.groupby("fold", sort=True):
                    local = group.copy()
                    local["probability"] = local["raw"] if calibration == "none" else _platt(
                        raw.loc[raw.fold.lt(fold)], local.raw.to_numpy()
                    )
                    calibrated.append(local)
                prediction = pd.concat(calibrated, ignore_index=True)
                fold_scores = [metric_values(g.y.to_numpy(), g.probability.to_numpy()) for _, g in prediction.groupby("fold", sort=True)]
                scores = pd.DataFrame(fold_scores)
                rows.append({
                    "head": head, "target": target, "config_id": config_id,
                    **config, "positive_weight": positive_weight, "calibration": calibration,
                    "oof_rows": len(prediction), "oof_positive_rate": float(prediction.y.mean()),
                    "mean_ap": float(scores.ap.mean()), "mean_brier": float(scores.brier.mean()),
                    "mean_ece10": float(scores.ece10.mean()),
                    "mean_composite": float((scores.ap - scores.brier).mean()),
                    "min_fold_composite": float((scores.ap - scores.brier).min()),
                    "mean_auc": float(scores.auc.mean()),
                })
    table = pd.DataFrame(rows)
    winner = table.sort_values(
        ["mean_composite", "min_fold_composite", "mean_ap", "config_id"],
        ascending=[False, False, False, True], kind="stable",
    ).iloc[0].to_dict()
    return table, winner, stored[(int(winner["config_id"]), float(winner["positive_weight"]))]


def _candidate_economics(
    candidates: pd.DataFrame, forward: pd.DataFrame, *, head: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = candidates.loc[candidates.selected_global_top10].merge(
        forward.loc[forward["head"].eq(head), ["source_utc", "probability", "target"]],
        left_on="__ts__", right_on="source_utc", how="inner", validate="many_to_one",
    )
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged["risk_decile"] = merged.groupby(["month", "side_name"], sort=True)["probability"].transform(
        lambda value: pd.qcut(value.rank(method="first"), q=10, labels=False, duplicates="drop")
    )
    group = ["head", "month", "side_name", "risk_decile"]
    merged["head"] = head
    attribution = merged.groupby(group, dropna=False, as_index=False).agg(
        selected_rows=("candidate_id", "size"),
        mean_net_bps=("execution_net_ev_12h", lambda value: float(value.mean() * 1e4)),
        mean_probability=("probability", "mean"), observed_target=("target", "mean"),
    )
    support = merged.groupby(["head", "month", "side_name"], as_index=False).agg(
        selected_rows=("candidate_id", "size"),
        exact_economic_rows=("execution_net_ev_12h", lambda value: int(value.notna().sum())),
        mean_net_bps=("execution_net_ev_12h", lambda value: float(value.mean() * 1e4)),
    )
    return attribution, support


def run(*, catalogue: Path = CATALOGUE, current: Path = CURRENT, output: Path = OUT) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    frame = pd.read_parquet(catalogue).copy()
    frame["source_utc"] = pd.to_datetime(frame.source_utc, utc=True, errors="raise")
    resolved = label_available(frame)
    latest = pd.to_datetime(pd.read_parquet(current, columns=["__ts__"])["__ts__"].max(), utc=True)
    train = frame.loc[frame.source_utc.lt(TRAIN_END) & resolved.lt(TRAIN_END)].copy()
    test = frame.loc[frame.source_utc.ge(TRAIN_END) & resolved.le(latest)].copy()
    for _, target in HEADS:
        frame[target] = pd.to_numeric(frame[target], errors="coerce").fillna(0).astype(np.int8)
        train[target] = frame.loc[train.index, target]
        test[target] = frame.loc[test.index, target]
    features = causal_feature_columns(frame, train)[:32]
    if not features:
        raise ValueError("no causal hourly features")
    selection, winners, forward_rows, rule_lists = [], [], [], []
    for head, target in HEADS:
        table, winner, raw_oof = _selection_rows(train, head=head, target=target, features=features)
        selection.append(table)
        config = CONFIGS[int(winner["config_id"])]
        raw, arm = _fit_predict(
            train, test, features=features, target=target, config=config,
            positive_weight=float(winner["positive_weight"]), seed=20260801 + len(winners),
        )
        probability = raw if winner["calibration"] == "none" else _platt(raw_oof, raw)
        forward_rows.append(pd.DataFrame({
            "source_utc": test.source_utc.to_numpy(), "head": head, "target": test[target].to_numpy(),
            "raw_probability": raw, "probability": probability,
        }))
        winner["backend"] = arm.backend
        winners.append(winner)
        rule_lists.append({
            "head": head, "target": target, "backend": arm.backend,
            "selected_causal_features": [features[int(i)] for i in np.asarray(arm.selected_indices, dtype=int)],
            "config": {**config, "positive_weight": float(winner["positive_weight"]), "calibration": winner["calibration"]},
            "rules": arm.describe(),
        })
    hpo = pd.concat(selection, ignore_index=True)
    winner_frame = pd.DataFrame(winners)
    forward = pd.concat(forward_rows, ignore_index=True)
    metric_rows, support_rows = [], []
    for head, group in forward.groupby("head", sort=True):
        for scope, local in [("all_2026", group), *[(f"month::{month}", part) for month, part in group.assign(month=group.source_utc.dt.strftime("%Y-%m")).groupby("month", sort=True)]]:
            metric_rows.append({"head": head, "scope": scope, **metric_values(local.target.to_numpy(), local.probability.to_numpy())})
            support_rows.append({"head": head, "scope": scope, "rows": len(local), "positives": int(local.target.sum()), "prevalence": float(local.target.mean())})
    candidates = pd.read_parquet(current, columns=["candidate_id", "__ts__", "side_name", "execution_net_ev_12h", "catboost__residual__without_hpo__all_features"])
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True)
    candidates = candidates.loc[candidates.__ts__.le(test.source_utc.max())].copy()
    candidates["month"] = candidates.__ts__.dt.strftime("%Y-%m")
    candidates["side_name"] = candidates.side_name.astype(str).str.lower()
    candidates["selected_global_top10"] = False
    for _, group in candidates.groupby("month", sort=True):
        candidates.loc[group.index, "selected_global_top10"] = global_top10(group, "catboost__residual__without_hpo__all_features")
    econ_rows, side_rows = [], []
    for head, _ in HEADS:
        econ, support = _candidate_economics(candidates, forward, head=head)
        econ_rows.append(econ); side_rows.append(support)
    economics = pd.concat(econ_rows, ignore_index=True)
    candidate_support = pd.concat(side_rows, ignore_index=True)
    expected_sides = set(candidates.loc[candidates.selected_global_top10, "side_name"].unique())
    observed_sides = set(candidate_support.side_name.unique())
    if not expected_sides.issubset(observed_sides):
        raise AssertionError(f"missing candidate-side attribution: {expected_sides - observed_sides}")
    stage = output.parent / f".{output.name}.{uuid.uuid4().hex}.stage"
    stage.mkdir(parents=True, exist_ok=False)
    try:
        hpo.to_csv(stage / "train_only_hpo.csv", index=False)
        winner_frame.to_csv(stage / "frozen_head_winners.csv", index=False)
        forward.to_parquet(stage / "forward_brl_predictions.parquet", index=False, compression="zstd")
        pd.DataFrame(metric_rows).to_csv(stage / "untouched_2026_discrimination_calibration.csv", index=False)
        pd.DataFrame(support_rows).to_csv(stage / "untouched_2026_monthly_support.csv", index=False)
        economics.to_csv(stage / "global_top10_economic_attribution.csv", index=False)
        candidate_support.to_csv(stage / "global_top10_candidate_side_support.csv", index=False)
        (stage / "frozen_rule_lists.json").write_text(json.dumps(safe(rule_lists), indent=2, sort_keys=True) + "\n")
        pd.DataFrame({"feature": features}).to_csv(stage / "causal_hourly_feature_contract.csv", index=False)
        manifest = {
            "schema": "strict_transition_brl_challenger_v1", "research_only": True,
            "promotion_eligible": False,
            "train_contract": "all feature selection, native-MAP list geometry, positive weight and calibration selected by blocked 2022-2025 OOF only; labels resolved before 2026-01-01",
            "test_contract": f"one untouched 2026 assessment through {test.source_utc.max()}",
            "method": "native_beta_binomial_map ordered low-cardinality rule list; explicit non-MCMC fallback provenance in frozen_rule_lists.json",
            "separation": "same causal hourly feature panel as strict LGBM onset heads; phase/state/identity/future/outcome fields excluded",
            "economics_contract": "global top10 selected once per UTC month before side and BRL-score attribution; diagnostic only",
            "inputs_sha256": {"catalogue": sha256(catalogue), "current_candidates": sha256(current)},
            "outputs_sha256": {path.name: sha256(path) for path in stage.iterdir() if path.is_file()},
            "counts": {"train": len(train), "test": len(test), "features": len(features), "heads": len(HEADS), "candidate_selected_rows": int(candidates.selected_global_top10.sum())},
        }
        (stage / "manifest.json").write_text(json.dumps(safe(manifest), indent=2, sort_keys=True) + "\n")
        (stage / "manifest.sha256").write_text(f"{sha256(stage / 'manifest.json')}  manifest.json\n")
        os.replace(stage, output)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--current", type=Path, default=CURRENT)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args(argv)
    print(json.dumps(safe(run(catalogue=args.catalogue, current=args.current, output=args.output)), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
