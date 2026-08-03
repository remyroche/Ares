#!/usr/bin/env python3
"""Stage-7 high-base pairwise trust-reranker ablation.

The selected B2 mapping and residual-value layer are frozen inputs.  This
script changes only the reliability overlay inside the causal P30 population.
It evaluates a pointwise cost-clear control, a pairwise realised-net reranker,
and their equal-rank hybrid.  Pairwise groups deliberately restrict each
comparison to candidates with the same side, calendar week, and a training-set
base-value bucket.  Thus the rank loss answers a local trust question rather
than attempting to replace the base opportunity model.

There are two chronological experiments in one invocation:

* development: train 2024-05-01--06-15, evaluate 06-15--08-01 and select one
  of {pointwise, pairwise, hybrid};
* untouched OOS: refit that fixed recipe through 2024-08-01, then evaluate
  08-01--12-01 once.

All final books remain pooled globally.  No outcome, path, or label-availability
field is included in the model feature matrix.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_full_universe_residual_meta import select  # noqa: E402
from scripts.run_full_universe_round_b_meta_targets import (  # noqa: E402
    _attach_prequential_expected, _attach_prequential_population, _base_predictions,
    _state_features,
)


def _rank_pct(values: pd.Series) -> np.ndarray:
    """Stable increasing rank coordinate; only called inside eligible P30."""
    return values.rank(method="first", pct=True).to_numpy(float)


def _metric(frame: pd.DataFrame, score: np.ndarray, global_rows: int, fraction: float = .10) -> dict:
    z = frame.assign(_score=score).sort_values(["_score", "candidate_id"], ascending=[False, True])
    # P30 admission only determines which rows get an overlay; top-k itself is
    # always a fraction of the *full* long/short/timestamp candidate pool.
    z = z.head(int(np.ceil(global_rows * fraction)))
    return {"n": int(len(z)), "gross_bps": float(z.gross_bps.mean()),
            "net_bps": float(z.net_bps.mean()), "long_n": int(z.side_name.eq("long").sum()),
            "short_n": int(z.side_name.eq("short").sum())}


def _context_matrix(frame: pd.DataFrame, chosen: list[str]) -> np.ndarray:
    # State inputs are all decision-time algebraic functions of frozen base B2.
    return _state_features(frame, chosen)


def _point_model() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=180, learning_rate=.05, num_leaves=24,
        min_child_samples=400, colsample_bytree=.8, subsample=.8, reg_lambda=10.,
        random_state=20260807, n_jobs=1, verbosity=-1,
    )


def _pair_model() -> lgb.LGBMRanker:
    return lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", eval_at=(10,), n_estimators=180,
        learning_rate=.05, num_leaves=24, min_child_samples=400, colsample_bytree=.8,
        subsample=.8, reg_lambda=10., lambdarank_truncation_level=50,
        random_state=20260807, n_jobs=1, verbosity=-1,
    )


def _fit_value_bins(train: pd.DataFrame, n_bins: int = 5) -> np.ndarray:
    # Fixed once from the train period.  Duplicate edges are harmless: digitize
    # still yields a deterministic coarser grouping instead of peeking at eval.
    return np.unique(np.quantile(train.base_expected_net_bps.to_numpy(float), np.linspace(0., 1., n_bins + 1)))


def _groups(frame: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    out = frame.copy()
    # Daily buckets are deliberately stricter than "same week or nearby
    # dates": candidates are compared on the same decision day, avoiding a
    # large cross-week pair set and keeping the local-market contract exact.
    out["_week"] = out.__ts__.dt.floor("D")
    out["_value_bucket"] = np.digitize(out.base_expected_net_bps.to_numpy(float), edges[1:-1], right=True)
    out["_group"] = (out.side_name.astype(str) + "|" + out._week.astype(str) + "|" + out._value_bucket.astype(str))
    return out


def _pair_labels(frame: pd.DataFrame) -> np.ndarray:
    # LambdaRank labels must be non-negative.  Within-group ordinal realised
    # net retains the requested order but caps gain range and makes ties stable.
    pct = frame.groupby("_group", observed=True).net_bps.rank(method="average", pct=True)
    # LightGBM's LambdaRank requires labels strictly below its number of
    # observed label levels.  Keep thirty-one ordinal gains (0..30).
    return np.minimum(np.floor(pct.to_numpy(float) * 31.).astype(int), 30)


def _fit_predict(train: pd.DataFrame, evaluation: pd.DataFrame, context: list[str]) -> tuple[pd.DataFrame, dict]:
    """Fit both heads only on prior resolved rows and score eligible eval rows."""
    point_y = train.net_bps.gt(0).to_numpy(int)
    point_features = select(train, context, point_y, n=30)
    point = _point_model().fit(_context_matrix(train, point_features), point_y)
    pair_features = select(train, context, train.net_bps.to_numpy(float), n=30)
    edges = _fit_value_bins(train)
    pair_train = _groups(train, edges).sort_values(["_group", "candidate_id"], kind="mergesort")
    counts = pair_train.groupby("_group", observed=True).size()
    # A group with one row contains no pair.  It is removed rather than silently
    # pooling across weeks/sides, preserving the comparable-context contract.
    keep_group = counts[counts.ge(2)].index
    pair_train = pair_train[pair_train._group.isin(keep_group)].copy()
    group_sizes = pair_train.groupby("_group", observed=True, sort=False).size().to_numpy(int)
    pair = _pair_model().fit(_context_matrix(pair_train, pair_features), _pair_labels(pair_train), group=group_sizes)
    out = evaluation.copy()
    out["pointwise_score"] = point.predict_proba(_context_matrix(out, point_features))[:, 1]
    out["pairwise_score"] = pair.predict(_context_matrix(out, pair_features))
    # Internal calibration is rank-only: common units are not claimed for a
    # LambdaRank score.  It is deliberately calibrated only by its eligible
    # evaluation population rank before the predeclared final blend.
    out["pointwise_rank"] = _rank_pct(out.pointwise_score)
    out["pairwise_rank"] = _rank_pct(out.pairwise_score)
    out["hybrid_rank"] = .5 * out.pointwise_rank + .5 * out.pairwise_rank
    diag = {
        "pointwise_feature_count": len(point_features), "pointwise_features": point_features,
        "pairwise_feature_count": len(pair_features), "pairwise_features": pair_features,
        "pair_groups": int(len(group_sizes)), "pair_train_rows": int(len(pair_train)),
        "pair_group_min_size": int(group_sizes.min()), "pair_group_median_size": float(np.median(group_sizes)),
        "pair_group_max_size": int(group_sizes.max()), "value_bucket_edges_bps": edges.tolist(),
    }
    return out, diag


def _final_scores(full: pd.DataFrame) -> dict[str, np.ndarray]:
    """Global pooled blend; non-P30 rows retain their frozen value rank."""
    value_rank = _rank_pct(full.residual_value_score)
    admitted = full.high_base_eligible.to_numpy(bool)
    def overlay(column: str) -> np.ndarray:
        # A P30 model is intentionally unavailable outside its admission
        # population.  Setting its rank coordinate to the frozen value rank
        # gives exactly zero overlay correction there:
        # .75 * value_rank + .25 * value_rank == value_rank.
        out = value_rank.copy()
        out[admitted] = _rank_pct(full.loc[admitted, column])
        return out
    return {
        "residual_only": value_rank,
        "pointwise": .75 * value_rank + .25 * overlay("pointwise_score"),
        "pairwise": .75 * value_rank + .25 * overlay("pairwise_score"),
        "hybrid": .75 * value_rank + .25 * overlay("hybrid_rank"),
    }


def _load_raw(panel: Path, audit: Path, base_root: Path, geometry: str) -> tuple[pd.DataFrame, list[str]]:
    meta = json.loads(audit.read_text())["meta"]["coverage_ge_90pct"]
    net, gross, event = (f"t4_{geometry}_net_bps", f"t4_{geometry}_gross_bps", f"t2_{geometry}_event")
    columns = ["candidate_id", "__ts__", "__label_available_at__", "side_name", net, gross, event, *meta]
    raw = pd.concat([pd.read_parquet(p, columns=columns) for p in sorted((panel / "parts").glob("*.parquet"))], ignore_index=True)
    raw["__ts__"] = pd.to_datetime(raw["__ts__"], utc=True)
    raw["__label_available_at__"] = pd.to_datetime(raw["__label_available_at__"], utc=True)
    raw = raw.rename(columns={net: "net_bps", gross: "gross_bps", event: "event"})
    return raw.merge(_base_predictions(base_root, geometry), on="candidate_id", validate="one_to_one"), meta


def _experiment(data: pd.DataFrame, context: list[str], residual_path: Path, train_start: pd.Timestamp,
                split: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    mapped = _attach_prequential_expected(data, pd.Timestamp("2024-04-15", tz="UTC"), split)
    mapped = _attach_prequential_population(mapped, .30, train_start, split)
    train = mapped[mapped.__ts__.ge(train_start) & mapped.__ts__.lt(split) & mapped.__label_available_at__.lt(split) & mapped.high_base_eligible].copy()
    full_evaluation = mapped[mapped.__ts__.ge(split) & mapped.__ts__.lt(end)].copy()
    evaluation = full_evaluation[full_evaluation.high_base_eligible].copy()
    residual = pd.read_parquet(residual_path, columns=["candidate_id", "final_score"])
    full_evaluation = full_evaluation.merge(residual.rename(columns={"final_score": "residual_value_score"}), on="candidate_id", validate="one_to_one")
    evaluation = evaluation.merge(full_evaluation[["candidate_id", "residual_value_score"]], on="candidate_id", validate="one_to_one")
    if len(evaluation) == 0 or len(train) == 0:
        raise RuntimeError("no eligible causal P30 rows")
    scored, diag = _fit_predict(train, evaluation, context)
    # Score calibration and target diagnostics are descriptive; not used to
    # choose cross-period configuration beyond the final economic decision.
    actual = scored.net_bps.gt(0).to_numpy(int)
    diag["pointwise_oos_cost_clear_auc"] = float(roc_auc_score(actual, scored.pointwise_score))
    diag["pointwise_oos_cost_clear_brier"] = float(brier_score_loss(actual, scored.pointwise_score))
    diag["pairwise_oos_net_spearman"] = float(scored.pairwise_score.corr(scored.net_bps, method="spearman"))
    diag["full_global_evaluation_rows"] = int(len(full_evaluation))
    # Merge the admitted overlay back to the full global book.  No overlay is
    # manufactured for non-admitted rows; _final_scores preserves their value
    # rank exactly.
    score_cols = ["candidate_id", "pointwise_score", "pairwise_score", "pointwise_rank", "pairwise_rank", "hybrid_rank"]
    full = full_evaluation.merge(scored[score_cols], on="candidate_id", how="left", validate="one_to_one")
    return full, diag


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--panel", type=Path, required=True); p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--base-root", type=Path, required=True); p.add_argument("--residual-dev", type=Path, required=True)
    p.add_argument("--residual-oos", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--geometry", default="tp3_sl2")
    p.add_argument("--phase", choices=("development", "oos", "full"), default="full",
                   help="Run separately when memory is constrained; full is only a convenience.")
    p.add_argument("--selected", choices=("pointwise", "pairwise", "hybrid"),
                   help="Required for --phase oos; must be the saved development selection.")
    a = p.parse_args()
    data, context = _load_raw(a.panel, a.audit, a.base_root, a.geometry)
    print("loaded causal panel", flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    if a.phase in ("development", "full"):
        dev, dev_diag = _experiment(data, context, a.residual_dev, pd.Timestamp("2024-05-01", tz="UTC"), pd.Timestamp("2024-06-15", tz="UTC"), pd.Timestamp("2024-08-01", tz="UTC"))
        print("scored development", flush=True)
        dev_scores = _final_scores(dev)
        development = {name: _metric(dev, score, dev_diag["full_global_evaluation_rows"]) for name, score in dev_scores.items()}
        # Deterministic and predeclared selection: pooled global top-10 net,
        # then gross, then retain the simpler pointwise control on exact tie.
        order = {"pointwise": 0, "pairwise": 1, "hybrid": 2}
        selected = min(order, key=lambda n: (-development[n]["net_bps"], -development[n]["gross_bps"], order[n]))
        dev_saved = dev[["candidate_id", "__ts__", "side_name", "gross_bps", "net_bps", "base_expected_net_bps", "residual_value_score", "pointwise_score", "pairwise_score", "pointwise_rank", "pairwise_rank", "hybrid_rank"]].copy()
        for name, score in dev_scores.items(): dev_saved[f"score_{name}"] = score
        dev_saved.to_parquet(a.out / "development_predictions.parquet", index=False)
        development_manifest = {"development_diagnostics": dev_diag, "development_global_top10": development,
                                "selection": {"rule": "max development pooled-global top10 net, tie gross then pointwise", "selected": selected}}
        (a.out / "development_manifest.json").write_text(json.dumps(development_manifest, indent=2))
        if a.phase == "development":
            print(json.dumps({"selected": selected, "development": development, "out": str(a.out)})); return
        del dev, dev_scores, dev_saved; gc.collect()
    else:
        saved = json.loads((a.out / "development_manifest.json").read_text())
        development, dev_diag = saved["development_global_top10"], saved["development_diagnostics"]
        selected = a.selected or saved["selection"]["selected"]
    # Match the development arm's fixed 45-day fit horizon.  This is a
    # predeclared rolling refit, not a post-OOS choice, and prevents a much
    # larger stale pair set from changing the ranking objective's geometry.
    oos, oos_diag = _experiment(data, context, a.residual_oos, pd.Timestamp("2024-06-15", tz="UTC"), pd.Timestamp("2024-08-01", tz="UTC"), pd.Timestamp("2024-12-01", tz="UTC"))
    print("scored untouched OOS", flush=True)
    oos_scores = _final_scores(oos)
    oos_metrics = {name: _metric(oos, score, oos_diag["full_global_evaluation_rows"]) for name, score in oos_scores.items()}
    oos["selected_score"] = oos_scores[selected]
    oos.sort_values(["selected_score", "candidate_id"], ascending=[False, True]).to_parquet(a.out / "oos_selected_predictions.parquet", index=False)
    manifest = {
        "schema": "full_universe_stage7_pairwise_trust_v1",
        "contract": "frozen B2 expected-payoff map and frozen residual value; only P30 reliability overlay is refit",
        "admission": "daily-prequential pooled top-30% B2 expected-net threshold; frozen at each evaluation boundary",
        "pairwise_groups": "same side + same decision day (a stricter nearby-date rule) + fixed train-period B2 expected-net quintile; fixed cost means no cost bucket is required",
        "pairwise_target": "within-group ordinal rank of realised TP3/SL2 H12 net bps",
        "pointwise_control_target": "I(realised TP3/SL2 H12 net bps > 0)",
        "combination": "within eligible P30: .75 rank(frozen residual value) + .25 rank(overlay); all final selections pooled globally",
        "development_window": {"fit": ["2024-05-01", "2024-06-15"], "eval": ["2024-06-15", "2024-08-01"]},
        "oos_window": {"fit": ["2024-06-15", "2024-08-01"], "eval": ["2024-08-01", "2024-12-01"]},
        "development_diagnostics": dev_diag, "oos_diagnostics": oos_diag,
        "development_global_top10": development, "selection": {"rule": "max development pooled-global top10 net, tie gross then pointwise", "selected": selected},
        "oos_global_top10": oos_metrics, "selected_oos_global_top10": oos_metrics[selected],
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report = ["# Stage 7 — high-base pairwise trust reranker", "", "## Design", "", manifest["contract"], "", "- Groups: " + manifest["pairwise_groups"], "- Pairwise target: " + manifest["pairwise_target"], "- Combination: " + manifest["combination"], "", "## Pooled global top-10 results", "", "| Variant | Development net | Development gross | OOS net | OOS gross |", "|---|---:|---:|---:|---:|"]
    for name in ("residual_only", "pointwise", "pairwise", "hybrid"):
        d, q = development[name], oos_metrics[name]
        report.append(f"| {name} | {d['net_bps']:.2f} | {d['gross_bps']:.2f} | {q['net_bps']:.2f} | {q['gross_bps']:.2f} |")
    report.extend(["", f"Development-selected recipe: **{selected}**.", "", "The OOS column is a single later replay; it was not used for recipe selection."])
    (a.out / "REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"selected": selected, "development": development, "oos": oos_metrics, "out": str(a.out)}))


if __name__ == "__main__":
    main()
