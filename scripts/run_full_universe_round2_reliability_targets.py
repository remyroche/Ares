#!/usr/bin/env python3
"""Round-2 reliability-target ablations on the frozen causal B2 representation.

The runner deliberately changes only the reliability supervision.  The event
base, daily-prequential side-shrunk payoff map, candidate admission rule, and
the separately trained residual-value head are all left untouched.  Scores are
emitted for the causal high-base population only; callers must retain the
frozen residual/base value outside that population when constructing a book.

Targets:
* net_gt_{0,25,50}: hard robust cost-clear controls;
* soft_cost_clear_{50,100}: sigmoid(net / tau), fitted as a bounded regressor;
* ordinal_margin: four cost-margin classes, reconstructed into cost-clear
  probability and a strongly-shrunk expected-net diagnostic.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, mean_squared_error, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_full_universe_round_b_meta_targets import (  # noqa: E402
    PROBS,
    _attach_prequential_expected,
    _attach_prequential_population,
    _base_predictions,
    _state_features,
)
from scripts.run_full_universe_residual_meta import select  # noqa: E402


TARGETS = (
    "net_gt_0", "net_gt_25", "net_gt_50",
    "soft_cost_clear_50", "soft_cost_clear_100", "ordinal_margin",
)
ORDINAL_EDGES = np.array([-np.inf, -200.0, 0.0, 100.0, np.inf])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -35.0, 35.0)))


def _kind(target: str) -> str:
    if target.startswith("net_gt_"):
        return "binary"
    if target.startswith("soft_cost_clear_"):
        return "soft"
    return "ordinal"


def _target(frame: pd.DataFrame, target: str) -> np.ndarray:
    net = frame.net_bps.to_numpy(float)
    if target.startswith("net_gt_"):
        return net > float(target.rsplit("_", 1)[1])
    if target.startswith("soft_cost_clear_"):
        tau = float(target.rsplit("_", 1)[1])
        return _sigmoid(net / tau)
    # 0: <=-200, 1: (-200,0], 2: (0,100], 3: >100.
    return np.digitize(net, ORDINAL_EDGES[1:-1], right=True)


def _model(kind: str):
    common = dict(
        n_estimators=180, learning_rate=.05, num_leaves=24,
        min_child_samples=400, colsample_bytree=.8, subsample=.8,
        reg_lambda=10., random_state=20260804, n_jobs=1, verbosity=-1,
    )
    if kind == "binary":
        return lgb.LGBMClassifier(objective="binary", **common)
    if kind == "ordinal":
        return lgb.LGBMClassifier(objective="multiclass", num_class=4, **common)
    return lgb.LGBMRegressor(objective="huber", alpha=.9, **common)


def _soft_label_description(target: str) -> str:
    if target.startswith("net_gt_"):
        return f"I(realised net > {target.rsplit('_', 1)[1]} bps)"
    if target.startswith("soft_cost_clear_"):
        return f"sigmoid(realised net / {target.rsplit('_', 1)[1]} bps)"
    return "ordinal realised-net bins: <=-200, (-200,0], (0,100], >100 bps"


def _metrics(frame: pd.DataFrame, score: np.ndarray) -> list[dict]:
    ranked = frame.assign(_score=score).sort_values(["_score", "candidate_id"], ascending=[False, True])
    rows = []
    for q in (.01, .05, .10, .20):
        z = ranked.head(int(np.ceil(len(ranked) * q)))
        rows.append({"top_fraction": q, "n": len(z), "gross_bps": float(z.gross_bps.mean()),
                     "net_bps": float(z.net_bps.mean()), "long_n": int(z.side_name.eq("long").sum()),
                     "short_n": int(z.side_name.eq("short").sum())})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--base-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--target", choices=TARGETS, required=True)
    p.add_argument("--geometry", default="tp3_sl2")
    p.add_argument("--train-start", default="2024-05-01")
    p.add_argument("--oos-start", default="2024-06-15")
    p.add_argument("--oos-end", default="2024-08-01")
    p.add_argument("--population", type=float, default=.30,
                   help="Causal high-base fraction; Round 2 control is 0.30.")
    a = p.parse_args()
    if not 0.0 < a.population <= 1.0:
        raise ValueError("population must be in (0,1]")
    train_start = pd.Timestamp(a.train_start, tz="UTC")
    oos_start = pd.Timestamp(a.oos_start, tz="UTC")
    oos_end = pd.Timestamp(a.oos_end, tz="UTC")
    net, gross, event = (f"t4_{a.geometry}_net_bps", f"t4_{a.geometry}_gross_bps",
                         f"t2_{a.geometry}_event")
    context = json.loads(a.audit.read_text())["meta"]["coverage_ge_90pct"]
    columns = ["candidate_id", "__ts__", "__label_available_at__", "side_name", net, gross, event, *context]
    raw = pd.concat([pd.read_parquet(x, columns=columns) for x in sorted((a.panel / "parts").glob("*.parquet"))], ignore_index=True)
    raw["__ts__"] = pd.to_datetime(raw["__ts__"], utc=True)
    raw["__label_available_at__"] = pd.to_datetime(raw["__label_available_at__"], utc=True)
    raw = raw.rename(columns={net: "net_bps", gross: "gross_bps", event: "event"})
    data = raw.merge(_base_predictions(a.base_root, a.geometry), on="candidate_id", validate="one_to_one")
    mapped = _attach_prequential_expected(data, pd.Timestamp("2024-04-15", tz="UTC"), oos_start)
    mapped = _attach_prequential_population(mapped, a.population, train_start, oos_start)
    train = mapped[mapped.__ts__.ge(train_start) & mapped.__ts__.lt(oos_start)
                   & mapped.__label_available_at__.lt(oos_start) & mapped.high_base_eligible].copy()
    evaluation = mapped[mapped.__ts__.ge(oos_start) & mapped.__ts__.lt(oos_end)].copy()
    eligible = evaluation[evaluation.high_base_eligible].copy()
    if train.empty or eligible.empty:
        raise RuntimeError("causal population selection produced no train/evaluation rows")

    kind = _kind(a.target)
    y = _target(train, a.target)
    # Select context only within the reliability population; base-state inputs
    # remain mandatory in _state_features and do not compete for this cap.
    chosen = select(train, context, y, n=30)
    model = _model(kind).fit(_state_features(train, chosen), y)
    xev = _state_features(eligible, chosen)
    if kind == "binary":
        probability = model.predict_proba(xev)[:, 1]
        score = probability
        actual_cost_clear = eligible.net_bps.gt(0).to_numpy(int)
        target_actual = _target(eligible, a.target).astype(int)
        diagnostics = {"target_auc": float(roc_auc_score(target_actual, probability)),
                       "target_brier": float(brier_score_loss(target_actual, probability)),
                       "target_prevalence": float(target_actual.mean()),
                       "cost_clear_auc": float(roc_auc_score(actual_cost_clear, probability)),
                       "cost_clear_brier": float(brier_score_loss(actual_cost_clear, probability))}
        expected_net = np.full(len(eligible), np.nan)
    elif kind == "soft":
        score = np.clip(model.predict(xev), 0., 1.)
        target_actual = _target(eligible, a.target)
        actual_cost_clear = eligible.net_bps.gt(0).to_numpy(int)
        diagnostics = {"soft_label_mse": float(mean_squared_error(target_actual, score)),
                       "soft_label_spearman": float(pd.Series(score).corr(pd.Series(target_actual), method="spearman")),
                       "cost_clear_auc": float(roc_auc_score(actual_cost_clear, score)),
                       "cost_clear_brier": float(brier_score_loss(actual_cost_clear, score))}
        probability, expected_net = score, np.full(len(eligible), np.nan)
    else:
        posterior = model.predict_proba(xev)
        # Train-only conditional means make the ordinal output economically
        # interpretable without using evaluation outcomes.  Strong shrinkage
        # makes sparse outer bins stable.
        cls = _target(train, a.target).astype(int)
        global_mean = float(train.net_bps.mean())
        means = np.array([(train.net_bps.to_numpy(float)[cls == k].sum() + 2000. * global_mean) /
                          (max(int((cls == k).sum()), 0) + 2000.) for k in range(4)])
        probability = posterior[:, 2:].sum(axis=1)
        expected_net = posterior @ means
        score = probability
        target_actual = _target(eligible, a.target).astype(int)
        actual_cost_clear = eligible.net_bps.gt(0).to_numpy(int)
        diagnostics = {"ordinal_accuracy": float((posterior.argmax(axis=1) == target_actual).mean()),
                       "cost_clear_auc": float(roc_auc_score(actual_cost_clear, probability)),
                       "cost_clear_brier": float(brier_score_loss(actual_cost_clear, probability)),
                       "ordinal_bin_expected_net_bps": means.tolist()}

    out = evaluation[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "base_expected_net_bps",
                      "base_expected_gross_bps", "base_payoff_mixture_sd_bps", "high_base_eligible", "high_base_cutoff"]].copy()
    out["reliability_score"] = np.nan
    # Compatibility alias for the existing rank-blend/replay helpers.  It is
    # deliberately the probability-like trust coordinate, never the ordinal
    # expected-net diagnostic.
    out["meta_score"] = np.nan
    out["cost_clear_probability"] = np.nan
    out["ordinal_expected_net_bps"] = np.nan
    out.loc[eligible.index, "reliability_score"] = score
    out.loc[eligible.index, "meta_score"] = score
    out.loc[eligible.index, "cost_clear_probability"] = probability
    out.loc[eligible.index, "ordinal_expected_net_bps"] = expected_net
    # This diagnostic ranks only the admitted population; it must not be
    # confused with a global stack replay.
    diagnostic_metrics = _metrics(eligible, score)
    a.out.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out / "predictions.parquet", index=False)
    pd.DataFrame(diagnostic_metrics).to_parquet(a.out / "eligible_score_metrics.parquet", index=False)
    manifest = {"schema": "full_universe_round2_reliability_target_v1", "target": a.target,
                "target_definition": _soft_label_description(a.target), "target_kind": kind,
                "population_fraction": a.population, "feature_count": len(chosen), "meta_features": chosen,
                "base_contract": "frozen event base; daily-prequential side-shrunk B2 expected net, frozen at OOS boundary",
                "residual_contract": "not retrained or used as reliability training input; downstream replay must join frozen residual value",
                "train_window": [str(train_start), str(oos_start)], "eval_window": [str(oos_start), str(oos_end)],
                "eligible_train_rows": len(train), "eligible_eval_rows": len(eligible), "diagnostics": diagnostics,
                "eligible_only_score_metrics": diagnostic_metrics}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"target": a.target, "diagnostics": diagnostics, "eligible_top10": diagnostic_metrics[2]}))


if __name__ == "__main__":
    main()
