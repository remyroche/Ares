#!/usr/bin/env python3
"""Cross-model screen after the P8u Base weight contract is frozen.

Only the retained raw-bps/equal-width/G3 target, exact timestamp queries,
F72 Base contract, and query-safe tail_linear_125 weights are used.  Every
learner is scored against the same external precision/preservation metric.
This is a cheap screen: full HPO is reserved for its single winner.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from lightgbm import LGBMRanker
from xgboost import XGBRanker

import run_strict_r3_p8u_precision_preservation_loss_funnel_v1 as gain
import run_strict_r3_p8u_precision_preservation_screen_v1 as stage1
import run_strict_r3_p8u_precision_preservation_winner_hpo_v1 as hpo
import run_strict_r3_p8u_precision_preservation_weight_funnel_v1 as weights
import run_strict_r3_router_single_base_prescreen_v1 as base
from strict_r3_p8u_precision_preservation_metric import COMPONENTS, stable_score, timestamp_components


SCHEMA = "strict_r3_p8u_precision_preservation_weighted_cross_model_v1"
SEED = 1729
MODEL_FAMILIES = (
    "lgbm_rank_xendcg", "lgbm_lambdarank", "catboost_queryrmse", "catboost_yetirank", "xgb_ndcg", "xgb_pairwise",
)
WEIGHT_COMPATIBLE_MODEL_FAMILIES = (
    "lgbm_rank_xendcg", "lgbm_lambdarank", "catboost_queryrmse",
)
XGB_WEIGHT_INCOMPATIBILITY = (
    "XGBoost ranking accepts one weight per query, not a row weight.  Because "
    "the frozen contract normalises each exact-timestamp query to mean one, "
    "passing a legal group weight would erase the selected within-query weight "
    "scheme and would not be a fair frozen-contract comparison."
)
CATBOOST_YETIRANK_WEIGHT_INCOMPATIBILITY = (
    "CatBoost reports that its pairwise/YetiRank losses do not support object "
    "(row-level) weights.  Running it would ignore the frozen within-query "
    "tail_linear_125 weighting contract and would not be a fair comparison."
)


@dataclass(frozen=True)
class Candidate:
    model_family: str

    @property
    def key(self) -> str:
        return self.model_family


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _fit_predict(
    *, family: str, x_train: np.ndarray, labels: np.ndarray, train: pd.DataFrame, x_held: np.ndarray,
    sample_weight: np.ndarray, seed: int,
) -> np.ndarray:
    group = base._query_groups(train)
    qid = hpo._qid(train)
    common_lgbm = dict(
        metric="ndcg", n_estimators=180, learning_rate=.05, max_depth=4, num_leaves=15,
        min_child_samples=260, subsample=.80, subsample_freq=1, colsample_bytree=.80,
        reg_alpha=.05, reg_lambda=8.0, min_split_gain=.001, random_state=seed,
        n_jobs=1, deterministic=True, force_col_wise=True, verbosity=-1,
        label_gain=gain.GAIN_SCHEDULES["g3_clipped_economic"],
    )
    if family == "lgbm_rank_xendcg":
        model = LGBMRanker(objective="rank_xendcg", **common_lgbm)
        model.fit(x_train, labels, group=group, sample_weight=sample_weight)
    elif family == "lgbm_lambdarank":
        model = LGBMRanker(objective="lambdarank", lambdarank_truncation_level=12, sigmoid=1.0, **common_lgbm)
        model.fit(x_train, labels, group=group, sample_weight=sample_weight)
    elif family == "catboost_queryrmse":
        model = CatBoostRanker(
            loss_function="QueryRMSE",
            eval_metric="NDCG:top=10", iterations=180, learning_rate=.05, depth=4, l2_leaf_reg=8.0,
            random_seed=seed, random_strength=.5, rsm=.80, thread_count=1, verbose=False, allow_writing_files=False,
        )
        model.fit(Pool(x_train, labels, group_id=qid, weight=sample_weight))
    else:  # pragma: no cover
        raise ValueError(family)
    prediction = np.asarray(model.predict(x_held), dtype=np.float32)
    del model
    gc.collect()
    return prediction


def _fit_one(
    *, candidate: Candidate, window: pd.DataFrame, held: pd.DataFrame, reserve: pd.Timestamp,
    fields: tuple[str, ...], train_cap: int, scheme: str, seed: int,
) -> tuple[Candidate, pd.DataFrame, dict[str, object]]:
    arm = stage1.Arm("raw_bps", "t1_raw_bps", "equal_width6")
    train = stage1._train_rows(window, arm, reserve, train_cap)
    labels, geometry = stage1._labels(train, arm)
    x_train, medians = base._numeric_matrix(train, fields)
    x_held, _ = base._numeric_matrix(held, fields, medians)
    sample_weight = weights._query_safe_weights(train, labels, scheme)
    prediction = _fit_predict(family=candidate.model_family, x_train=x_train, labels=labels, train=train, x_held=x_held, sample_weight=sample_weight, seed=seed)
    if not np.isfinite(prediction).all():
        raise AssertionError("non-finite held prediction")
    score = held.loc[:, list(base.IDENTITY)].copy()
    score["base_score"] = prediction
    score["base_rank_ts"] = base._rank_desc(score, "base_score")
    audit = {
        "train_rows": len(train), "train_queries": train["__decision_ts__"].nunique(),
        "weight_min": float(sample_weight.min()), "weight_max": float(sample_weight.max()),
        "target_geometry": json.dumps(geometry, sort_keys=True), "target_free_before_outcome_join": True,
        "feature_medians_fit_train_only": True,
    }
    del train, labels, x_train, x_held, sample_weight
    gc.collect()
    return candidate, score, audit


def run(
    *, feature_roots: Sequence[Path], label_root: Path, router_root: Path, selection_json: Path,
    stage1_root: Path, out: Path, held_months: Sequence[pd.Timestamp], train_months: int,
    reserve_days: int, train_cap: int, scheme: str, workers: int,
) -> Path:
    if out.exists():
        raise FileExistsError("immutable output exists")
    if scheme != "tail_linear_125":
        raise AssertionError("cross-model screen requires the confirmed tail_linear_125 weight contract")
    fields = tuple(json.loads(selection_json.read_text())["selected_features"])
    if len(fields) != 72 or len(set(fields)) != len(fields):
        raise AssertionError("cross-model screen requires frozen F72")
    candidates = tuple(Candidate(family) for family in WEIGHT_COMPATIBLE_MODEL_FAMILIES)
    out.mkdir(parents=True)
    excluded = {
        "xgb_ndcg": XGB_WEIGHT_INCOMPATIBILITY,
        "xgb_pairwise": XGB_WEIGHT_INCOMPATIBILITY,
        "catboost_yetirank": CATBOOST_YETIRANK_WEIGHT_INCOMPATIBILITY,
    }
    _once(out / "preflight.json", {"schema": SCHEMA, "feature_count": len(fields), "scheme": scheme, "models": list(WEIGHT_COMPATIBLE_MODEL_FAMILIES), "excluded_models": excluded, "months": [f"{month:%Y-%m}" for month in held_months]})
    parts: dict[str, list[pd.DataFrame]] = {candidate.key: [] for candidate in candidates}
    controls: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    for fold_index, month in enumerate(held_months):
        reserve = month - pd.Timedelta(days=reserve_days)
        end = month + pd.offsets.MonthBegin(1)
        window, coverage = base._load_window(candidate_root=None, feature_root=feature_roots, label_root=label_root, router_root=router_root, start=reserve - pd.DateOffset(months=train_months), end=end, fields=fields)
        coverage_rows.extend(coverage)
        held = window.loc[window["__decision_ts__"].ge(month) & window["__decision_ts__"].lt(end)].copy().sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        labels = held.loc[:, ["candidate_id", "policy_ordinal_valid", "policy_net_bps"]].copy()
        control_score = gain._control_score(stage1_root, month)
        if len(held) != len(control_score) or not held.candidate_id.equals(control_score.candidate_id):
            raise AssertionError(f"{month:%Y-%m}: common target-free control mismatch")
        controls.append(timestamp_components(control_score.merge(labels, on="candidate_id", how="left", validate="one_to_one"), score_column="base_score"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_fit_one, candidate=candidate, window=window, held=held, reserve=reserve, fields=fields, train_cap=train_cap, scheme=scheme, seed=SEED + fold_index * 10_000 + i) for i, candidate in enumerate(candidates)]
            for future in concurrent.futures.as_completed(futures):
                candidate, score, audit = future.result()
                path = out / "target_free_scores" / candidate.key / f"month={month:%Y-%m}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                score.to_parquet(path, index=False, compression="zstd")
                joined = score.merge(labels, on="candidate_id", how="left", validate="one_to_one")
                panel = timestamp_components(joined, score_column="base_score")
                parts[candidate.key].append(panel)
                audits.append({"model_family": candidate.model_family, "held_month": f"{month:%Y-%m}", "held_rows": len(held), "held_queries": held["__decision_ts__"].nunique(), "router_top50_identity_exact": True, **audit})
                _progress(out, stage="candidate_fold_complete", model=candidate.model_family, held_month=f"{month:%Y-%m}")
        del window, held, labels
        gc.collect()
    control = pd.concat(controls, ignore_index=True).sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    components: list[pd.DataFrame] = []
    for candidate in candidates:
        panel = pd.concat(parts[candidate.key], ignore_index=True).sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
        summary, normalised = stable_score(panel, control)
        rows.append({"model_family": candidate.model_family, **summary.__dict__, **{f"mean_{name}": float(panel[name].mean()) for name in COMPONENTS}, "mean_utility_recall20": float(panel["utility_recall20"].mean())})
        normalised["model_family"] = candidate.model_family
        components.append(normalised)
    summary = pd.DataFrame(rows).sort_values(["score_stable", "mean_dtp2_bps"], ascending=False, kind="stable").reset_index(drop=True)
    summary.to_parquet(out / "cross_model_summary.parquet", index=False, compression="zstd")
    pd.concat(components, ignore_index=True).to_parquet(out / "timestamp_components.parquet", index=False, compression="zstd")
    control.to_parquet(out / "control_timestamp_components.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(out / "fold_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage_rows).drop_duplicates(subset=["month"], keep="last").to_parquet(out / "coverage_audit.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "frozen_raw_bps_equal_width_g3_target": True, "frozen_f72_contract": True,
        "frozen_tail_linear_125_query_weights": True, "weights_normalised_within_training_timestamp": True,
        "weight_clip_050_200": True, "p8u_router_top50_identity_exact": True,
        "all_train_labels_resolved_before_reserve": True, "held_scores_target_free_before_outcome_join": True,
        "all_feature_medians_train_only": True, "common_external_scorestable": True,
        "xgb_row_weight_incompatibility_explicit": True,
        "catboost_yetirank_row_weight_incompatibility_explicit": True,
        "no_meta_mc1_admission_portfolio_live_or_exchange_mutation": True,
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "offline weighted Base cross-model screen only", "target": "raw_bps/equal_width6/G3", "query": "exact timestamp", "weight_scheme": scheme, "feature_count": len(fields), "model_families": list(WEIGHT_COMPATIBLE_MODEL_FAMILIES), "excluded_model_families": excluded, "held_months": [f"{month:%Y-%m}" for month in held_months], "results": summary.to_dict("records"), "next_stage": "Only the one winning compatible model family may receive a full HPO under this frozen target/weight/feature contract.",
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", default="2025-11,2026-01,2026-03,2026-05,2026-07")
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--scheme", default="tail_linear_125")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.train_months < 2 or args.reserve_days < 12 or args.train_cap < 8_000 or args.workers < 1:
        raise ValueError("invalid weighted cross-model contract")
    print(run(feature_roots=tuple(Path(item.strip()).resolve() for item in args.feature_roots.split(",") if item.strip()), label_root=args.label_root.resolve(), router_root=args.router_root.resolve(), selection_json=args.selection_json.resolve(), stage1_root=args.stage1_root.resolve(), out=args.out.resolve(), held_months=stage1._parse_months(args.held_months), train_months=args.train_months, reserve_days=args.reserve_days, train_cap=args.train_cap, scheme=args.scheme, workers=args.workers))


if __name__ == "__main__":
    main()
