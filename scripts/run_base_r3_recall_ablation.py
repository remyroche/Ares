#!/usr/bin/env python3
"""Run the narrow, chronological R3 base-recall funnel.

Usage is deliberately phase-specific.  Run ``score`` on the existing strict
OOF B25 simplex, select one score definition on R3 ranking metrics, then run
``labels`` (uniform only), then ``weights`` for just the selected label.  The
``ranker`` phase is intentionally not automatic: it is available only when
the predecessor record passes its explicit R3 no-regression gate.

The input is an already materialised valid-path panel.  It must contain the
F0 side-local fields from ``--features-json`` and the resolved pre-adverse
label primitives.  It never reads a raw feature store or constructs features.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.base_r3_recall_ablation import (
    R3_CLEAR_DEFINITIONS,
    R3_SCORE_DEFINITIONS,
    R3_WEIGHT_DEFINITIONS,
    ScoreDefinition,
    WeightDefinition,
    build_r3_sample_weight,
    materialize_r3_classes,
    query_group_sizes,
    query_r3_ranking_metrics,
    ranker_may_advance,
    score_r3_simplex,
)
from extreme_price_movements.feature_portability_f4_panel import FROZEN_R3_BASE_PARAMS


SCHEMA = "base_r3_recall_ablation_v1"
FORBIDDEN_FEATURE_TOKENS = (
    "label", "target", "outcome", "future", "mfe", "mae", "gross", "net", "pnl", "event", "path", "exit",
)


def _load_features(path: Path) -> dict[str, list[str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or set(raw) != {"long", "short"}:
        raise ValueError("--features-json must be an exact {'long': [...], 'short': [...]} mapping")
    output: dict[str, list[str]] = {}
    for side in ("long", "short"):
        values = list(map(str, raw[side]))
        if not values or len(values) != len(set(values)):
            raise ValueError(f"{side} needs a non-empty unique feature contract")
        bad = [field for field in values if any(token in field.lower() for token in FORBIDDEN_FEATURE_TOKENS)]
        if bad:
            raise ValueError(f"{side} feature contract includes outcome/path-like fields: {bad[:5]}")
        output[side] = values
    return output


def _strict_blocks(frame: pd.DataFrame, *, folds: int, min_train_rows: int) -> list[np.ndarray]:
    decision = pd.to_datetime(frame.decision_ts, utc=True, errors="raise")
    available = pd.to_datetime(frame.label_available_ts, utc=True, errors="raise")
    ordered = frame.assign(_decision=decision).sort_values(["_decision", "candidate_id"], kind="stable")
    grouped = list(ordered.groupby("_decision", sort=True, observed=True))
    groups = [x.index.to_numpy(dtype=np.int64) for _, x in grouped]
    # Count label-available rows before every candidate test timestamp with a
    # single sorted search.  The earlier implementation aligned a full-series
    # comparison to every timestamp group, producing a quadratic temporary
    # matrix on the 1.6m-row panel before any model was fit.
    group_times = np.asarray([pd.Timestamp(key).value for key, _ in grouped], dtype=np.int64)
    available_ns = np.sort(available.view("int64").to_numpy(copy=False))
    resolved_counts = np.searchsorted(available_ns, group_times, side="left")
    eligible = np.flatnonzero(resolved_counts >= int(min_train_rows))
    first = int(eligible[0]) if len(eligible) else None
    if first is None:
        raise ValueError("no strict prior-resolved base training support")
    positions = np.array_split(np.arange(first, len(groups)), min(folds, len(groups) - first))
    return [np.concatenate([groups[int(i)] for i in block]) for block in positions if len(block)]


def _fit_classifier(train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str], y: np.ndarray, weight: np.ndarray, seed: int) -> np.ndarray:
    from lightgbm import LGBMClassifier
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("each strict base train fold must retain all R3 classes")
    model = LGBMClassifier(random_state=int(seed), **dict(FROZEN_R3_BASE_PARAMS))
    model.fit(train.loc[:, list(features)].astype("float32"), y, sample_weight=weight)
    p = model.predict_proba(test.loc[:, list(features)].astype("float32"))
    if p.shape != (len(test), 3) or not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("classifier did not emit an R3 probability simplex")
    return p


def _fit_ranker(train: pd.DataFrame, test: pd.DataFrame, features: Sequence[str], y_clear: np.ndarray, weight: np.ndarray, seed: int) -> np.ndarray:
    from lightgbm import LGBMRanker
    ordered = train.assign(_y=y_clear).sort_values(["decision_ts", "side_name", "candidate_id"], kind="stable")
    groups = query_group_sizes(ordered)
    model = LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_at=[5, 30, 40],
        n_estimators=140, learning_rate=.05, num_leaves=31, min_child_samples=350,
        subsample=.80, colsample_bytree=.80, reg_lambda=8.0, n_jobs=1,
        random_state=int(seed), verbosity=-1,
    )
    ordered_weight = pd.Series(weight, index=train.index).loc[ordered.index].to_numpy(float)
    model.fit(
        ordered.loc[:, list(features)].astype("float32"), ordered._y.to_numpy(np.int8),
        group=groups, sample_weight=ordered_weight,
    )
    return model.predict(test.loc[:, list(features)].astype("float32"))


def _metrics(scored: pd.DataFrame, *, score_column: str, own_target: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scope, part in [("pooled", scored), *[(f"side:{side}", x) for side, x in scored.groupby("side_name", observed=True, sort=True)], *[(f"month:{month}", x) for month, x in scored.groupby(scored.decision_ts.dt.strftime("%Y-%m"), observed=True, sort=True)]]:
        own = query_r3_ranking_metrics(part, score_column=score_column, target_column=own_target)
        check = part.copy()
        canonical_metrics = query_r3_ranking_metrics(check, score_column=score_column, target_column="canonical_b25_class")
        output.append({"scope": scope, "target_metric": "own", **own})
        output.append({"scope": scope, "target_metric": "canonical_b25", **canonical_metrics})
    return output


def _month_stability_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarise the predeclared month-level base-quality gates per arm.

    The raw metrics file retains every month.  This companion summary makes
    the promotion-relevant worst-period and dispersion checks explicit, so a
    pooled score cannot conceal a weak month.  It is purely a reporting
    transform of strict-OOF metrics; no score, label, or model input changes.
    """
    months = metrics.loc[metrics["scope"].astype(str).str.startswith("month:")].copy()
    if months.empty:
        return pd.DataFrame()
    arm_columns = [name for name in ("phase", "score_definition", "target_definition", "weight_definition", "target_metric") if name in months]
    result: list[dict[str, Any]] = []
    for keys, part in months.groupby(arm_columns, dropna=False, observed=True, sort=True):
        values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(arm_columns, values, strict=True))
        top30 = part["top30_winner_recall"]
        top40 = part["top40_winner_recall"]
        top5_clear = part["top5_clear_uplift"]
        top5_net = part["top5_net_uplift_bps"]
        ic = part["within_query_rank_ic"]
        row.update({
            "months": int(len(part)),
            "min_month_ic": float(ic.min()),
            "max_month_ic": float(ic.max()),
            "positive_month_ic_count": int(ic.gt(0.0).sum()),
            "min_top30_winner_recall": float(top30.min()),
            "max_top30_winner_recall": float(top30.max()),
            "top30_winner_recall_range": float(top30.max() - top30.min()),
            "min_top40_winner_recall": float(top40.min()),
            "max_top40_winner_recall": float(top40.max()),
            "top40_winner_recall_range": float(top40.max() - top40.min()),
            "min_top5_clear_uplift": float(top5_clear.min()),
            "min_top5_net_uplift_bps": float(top5_net.min()),
            "mean_top5_net_uplift_bps": float(top5_net.mean()),
            "positive_top5_net_month_count": int(top5_net.gt(0.0).sum()),
            "monthly_decile_adjacent_violations": int(part["target_decile_adjacent_violations"].sum()),
            "all_month_ic_positive": bool(ic.gt(0.0).all()),
            "all_month_top5_clear_uplift_positive": bool(top5_clear.gt(0.0).all()),
            "all_month_top30_recall_ge_40pct": bool(top30.ge(0.40).all()),
            "all_month_top40_recall_ge_50pct": bool(top40.ge(0.50).all()),
            "all_month_deciles_monotonic": bool(part["target_decile_adjacent_violations"].eq(0.0).all()),
        })
        result.append(row)
    return pd.DataFrame(result)


def _strict_oof(frame: pd.DataFrame, *, features_by_side: Mapping[str, Sequence[str]], classes: pd.Series, weight_definition: WeightDefinition, score_definition: ScoreDefinition, ranker: bool, folds: int, min_train_rows: int, seed: int) -> pd.DataFrame:
    output: list[pd.DataFrame] = []
    for side, source in frame.groupby("side_name", observed=True, sort=True):
        local = source.copy()
        local["__source_index"] = local.index
        local["r3_class"] = classes.loc[local.index].to_numpy(np.int8)
        score = np.full(len(local), np.nan, dtype=float)
        for fold_id, test_indices in enumerate(_strict_blocks(local, folds=folds, min_train_rows=min_train_rows)):
            start = pd.to_datetime(local.loc[test_indices, "decision_ts"], utc=True).min()
            train_indices = local.index[pd.to_datetime(local.label_available_ts, utc=True).lt(start)]
            train = local.loc[train_indices]
            test = local.loc[test_indices]
            y = train.r3_class.to_numpy(np.int8)
            weights = build_r3_sample_weight(train, y, weight_definition)
            if ranker:
                score[np.isin(local.index, test_indices)] = _fit_ranker(train, test, features_by_side[str(side)], y.eq(2).astype(np.int8), weights, seed + fold_id)
            else:
                p = _fit_classifier(train, test, features_by_side[str(side)], y, weights, seed + fold_id)
                score[np.isin(local.index, test_indices)] = score_r3_simplex(p, score_definition)
        local["score"] = score
        output.append(local.loc[np.isfinite(local.score)].copy())
    result = pd.concat(output, ignore_index=True)
    if result.empty:
        raise ValueError("strict OOF result is empty")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--features-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phase", choices=("score", "labels", "weights", "ranker"), required=True)
    parser.add_argument("--score", default="contrast_l1p0")
    parser.add_argument("--target", default="b25_current")
    parser.add_argument("--weight", default="uniform")
    parser.add_argument("--control-metrics", type=Path)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-train-rows", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--side", choices=("long", "short"), help="run one side only; useful for side-local target repair")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    features = _load_features(args.features_json)
    frame = pd.read_parquet(args.input)
    if args.side is not None:
        frame = frame.loc[frame.side_name.astype(str).str.lower().eq(args.side)].copy()
        if frame.empty:
            raise ValueError(f"requested side has no rows: {args.side}")
    required = {"candidate_id", "decision_ts", "label_available_ts", "side_name", "net_bps"}
    if args.phase != "score":
        required.update({*sum((list(x) for x in features.values()), []), "pre_adverse_mfe_bps", "atr_bps", "lower_touch_minute", "robust_clear_event_b0", "robust_clear_event_b25", "robust_clear_event_b50"})
    else:
        required.update({"r3_class", "p_adverse", "p_weak", "p_clear"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"input lacks required causal/model fields: {missing[:12]}")
    frame.decision_ts = pd.to_datetime(frame.decision_ts, utc=True, errors="raise")
    frame.label_available_ts = pd.to_datetime(frame.label_available_ts, utc=True, errors="raise")
    if frame.candidate_id.duplicated().any() or not frame.side_name.astype(str).isin(("long", "short")).all():
        raise ValueError("input must have globally unique candidate IDs and canonical sides")
    score_definition = next((x for x in R3_SCORE_DEFINITIONS if x.name == args.score), None)
    if score_definition is None:
        raise ValueError("unknown predeclared score")
    target_definition = next((x for x in R3_CLEAR_DEFINITIONS if x.name == args.target), None)
    if target_definition is None:
        raise ValueError("unknown predeclared target")
    weight_definition = next((x for x in R3_WEIGHT_DEFINITIONS if x.name == args.weight), None)
    if weight_definition is None:
        raise ValueError("unknown predeclared weight")
    if args.phase == "score":
        canonical = pd.to_numeric(frame["r3_class"], errors="raise").astype(np.int8)
    else:
        canonical = materialize_r3_classes(frame, R3_CLEAR_DEFINITIONS[0]).r3_class
    if args.phase == "score":
        simplex = frame.loc[:, ["p_adverse", "p_weak", "p_clear"]].to_numpy(float)
        rows: list[dict[str, Any]] = []
        for definition in R3_SCORE_DEFINITIONS:
            local = frame.copy()
            local["score"] = score_r3_simplex(simplex, definition)
            local["canonical_b25_class"] = canonical.to_numpy(np.int8)
            rows.extend({"phase": "score", "score_definition": definition.name, **x} for x in _metrics(local, score_column="score", own_target="r3_class"))
        scored = frame.loc[:, ["candidate_id", "decision_ts", "side_name"]].copy()
    else:
        labels = materialize_r3_classes(frame, target_definition)
        ranker = args.phase == "ranker"
        if ranker:
            if args.control_metrics is None:
                raise ValueError("ranker requires --control-metrics from the selected classifier")
            control = json.loads(args.control_metrics.read_text(encoding="utf-8"))
            # The caller must pass only the pooled/own record, preventing a
            # month-specific gate from silently advancing a global ranker.
            precheck = control.get("control")
            candidate = control.get("candidate")
            if not isinstance(precheck, Mapping) or not isinstance(candidate, Mapping):
                raise ValueError("ranker control file must contain {control: {...}, candidate: {...}} pooled own-target metrics")
            if not ranker_may_advance(precheck, candidate):
                raise ValueError("ranker predecessor failed the R3 no-regression gate")
        scored = _strict_oof(frame, features_by_side=features, classes=labels.r3_class, weight_definition=weight_definition, score_definition=score_definition, ranker=ranker, folds=args.folds, min_train_rows=args.min_train_rows, seed=args.seed)
        scored["canonical_b25_class"] = canonical.loc[scored["__source_index"].to_numpy()].to_numpy(np.int8)
        rows = [{"phase": args.phase, "score_definition": score_definition.name, "target_definition": target_definition.name, "weight_definition": weight_definition.name, **x} for x in _metrics(scored, score_column="score", own_target="r3_class")]
    args.out.mkdir(parents=True)
    metrics = pd.DataFrame(rows)
    metrics.to_parquet(args.out / "metrics.parquet", index=False, compression="zstd")
    _month_stability_summary(metrics).to_parquet(args.out / "month_stability_summary.parquet", index=False, compression="zstd")
    scored.to_parquet(args.out / "oof_predictions.parquet", index=False, compression="zstd")
    (args.out / "manifest.json").write_text(json.dumps({
        "schema": SCHEMA, "status": "complete", "phase": args.phase,
        "input": str(args.input), "features_json": str(args.features_json),
        "frozen_model_params": dict(FROZEN_R3_BASE_PARAMS), "score": args.score,
        "target": args.target, "weight": args.weight,
        "strictness": "label_available_ts < held_out_fold_start; per-side features; next-entry H12 path labels",
        "plan": "score -> labels_uniform -> selected_target_weights -> gated_query_ranker",
        "side_scope": [args.side] if args.side is not None else ["long", "short"],
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
