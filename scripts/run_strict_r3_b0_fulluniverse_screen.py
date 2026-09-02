#!/usr/bin/env python3
"""Full-universe, strict-OOF feature screen for the selected B0 challenger.

This is a research-only implementation of stages 3--10 of the routed-base
feature-selection protocol.  The candidate B0 label is the preselected
policy-ordinal G3 LambdaRank label.  E/T predictions are loaded only for a
downstream complementarity diagnostic and never enter the B0 feature matrix.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from run_strict_r3_b0_replacement_ranker_screen import (
    GAIN_SCHEDULES, SEED, TARGETS, _groups, _rank, _sample_queries,
)
from run_strict_r3_routed_et_fulluniverse_screen import (
    IDENTITY, ROUTER_FIELD, _correlation_sample, _coverage, _feature_family,
    _metric_suite, _month_range, _numeric_fields, _redundancy,
    _selected_feature_matrix, _stratified_index, _univariate, _utc,
)

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def _exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(out: Path, **payload: object) -> None:
    with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _route(frame: pd.DataFrame, fraction: float) -> pd.Series:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", ROUTER_FIELD]].copy()
    work["position"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", ROUTER_FIELD, "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size")
    output = pd.Series(False, index=np.arange(len(frame)))
    output.iloc[work.position.to_numpy(np.int64)] = ordinal.le(np.ceil(count.to_numpy(float) * fraction)).to_numpy(bool)
    return output


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _read_window(
    feature_root: Path, score_root: Path, router_root: Path, label_root: Path,
    start: pd.Timestamp, end: pd.Timestamp, target: str,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    valid = target.replace("_grade", "_valid")
    for month in _month_range(start, end):
        token = f"{month:%Y-%m}"
        feature_path = feature_root / f"month={token}" / "causal_feature_universe.parquet"
        score_path = score_root / "target_free_monthly" / f"month={token}" / "scores_features.parquet"
        router_path = router_root / "target_free_scores" / f"month={token}.parquet"
        label_path = label_root / f"month={token}" / "b0_replacement_targets.parquet"
        if not all(path.exists() for path in (feature_path, score_path, router_path, label_path)):
            raise FileNotFoundError(f"missing target-free source for {token}")
        identities = pd.read_parquet(feature_path, columns=list(IDENTITY))
        scores = pd.read_parquet(score_path, columns=[*IDENTITY, "efficiency_bps", "timing_bps"])
        router = pd.read_parquet(router_path, columns=[*IDENTITY, ROUTER_FIELD])
        labels = pd.read_parquet(label_path, columns=["candidate_id", "label_available_ts", valid, target, "policy_net_bps"])
        for frame in (identities, scores, router):
            frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        labels["label_available_ts"] = pd.to_datetime(labels["label_available_ts"], utc=True, errors="coerce")
        work = identities.merge(scores, on=list(IDENTITY), how="inner", validate="one_to_one")
        work = work.merge(router, on=list(IDENTITY), how="inner", validate="one_to_one")
        if len(work) != len(identities):
            raise AssertionError(f"{token}: target-free score/router identity mismatch")
        work = work.merge(labels, on="candidate_id", how="left", validate="one_to_one")
        work = work.loc[work.__decision_ts__.ge(start) & work.__decision_ts__.lt(end)].copy()
        work["label_joined"] = work[valid].notna()
        work[valid] = work[valid].fillna(False).astype(bool)
        work["router_selected"] = _route(work, .50).to_numpy(bool)
        parts.append(work)
    return pd.concat(parts, ignore_index=True)


def _valid_train(frame: pd.DataFrame, valid: str, target: str, reserve: pd.Timestamp) -> pd.DataFrame:
    mask = (
        frame.router_selected
        & frame[valid].fillna(False).astype(bool)
        & frame.label_available_ts.lt(reserve)
        & np.isfinite(pd.to_numeric(frame[target], errors="coerce"))
    )
    return frame.loc[mask].copy()


def _valid_held(frame: pd.DataFrame, valid: str) -> pd.DataFrame:
    mask = (
        frame.router_selected
        & frame[valid].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(frame.policy_net_bps, errors="coerce"))
    )
    return frame.loc[mask].copy()


def _model_params(seed: int, n_jobs: int, *, cheap: bool = False, feature_fraction: float = .80) -> dict[str, object]:
    return {
        "objective": "lambdarank", "metric": "ndcg", "n_estimators": 110 if cheap else 140,
        "learning_rate": .05, "max_depth": 4 if not cheap else 3, "num_leaves": 15,
        "min_child_samples": 260, "subsample": .80, "subsample_freq": 1,
        "colsample_bytree": feature_fraction, "reg_alpha": .05, "reg_lambda": 8.0,
        "min_split_gain": .001, "lambdarank_truncation_level": 12,
        "label_gain": GAIN_SCHEDULES["g3_clipped_economic"], "lambdarank_norm": True,
        "random_state": seed, "n_jobs": n_jobs, "deterministic": True,
        "force_col_wise": True, "verbosity": -1,
    }


def _blend_metrics(held: pd.DataFrame, x_score: np.ndarray) -> dict[str, float]:
    work = held.loc[:, ["candidate_id", "__decision_ts__", "policy_net_bps", "efficiency_bps", "timing_bps"]].copy()
    work["x_score"] = x_score
    work["x_rank"] = _rank(work, "x_score")
    work["e_rank"] = _rank(work, "efficiency_bps")
    work["t_rank"] = _rank(work, "timing_bps")
    work["etx_rank"] = (work.x_rank + work.e_rank + work.t_rank) / 3.0
    return {f"x_{key}": value for key, value in _metric_suite(work, "x_score").items()} | {
        f"etx_{key}": value for key, value in _metric_suite(work, "etx_rank").items()
    }


def _screen(
    args: argparse.Namespace, fields: list[str], target: str, valid: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gain_rows: list[dict[str, object]] = []
    shap_rows: list[dict[str, object]] = []
    univariate_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for ordinal, held_text in enumerate(args.held_months):
        held_month = _utc(held_text)
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        window = _read_window(
            args.feature_root, args.score_root, args.router_root, args.label_root,
            reserve - pd.DateOffset(months=args.train_months), _month_end(held_month), target,
        )
        train = _sample_queries(_valid_train(window.loc[window.__decision_ts__.lt(reserve)], valid, target, reserve), args.train_cap)
        train = train.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
        held = _valid_held(window.loc[window.__decision_ts__.ge(held_month)], valid)
        held = _sample_queries(held, args.held_cap).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        if len(train) < 8_000 or len(held) < 1_000:
            raise AssertionError(f"{held_month:%Y-%m}: insufficient strict routed support")
        selected = pd.concat([train, held], ignore_index=True)
        values = _selected_feature_matrix(args.feature_root, selected, fields)
        medians = np.nanmedian(values[:len(train)], axis=0).astype(np.float32)
        medians[~np.isfinite(medians)] = 0.0
        for begin in range(0, values.shape[1], 48):
            end = min(values.shape[1], begin + 48)
            block = values[:, begin:end]
            missing = ~np.isfinite(block)
            if missing.any():
                block[missing] = np.broadcast_to(medians[begin:end], block.shape)[missing]
        x_train, x_held = values[:len(train)], values[len(train):]
        model = LGBMRanker(**_model_params(SEED + ordinal, args.n_jobs))
        model.fit(x_train, pd.to_numeric(train[target], errors="coerce").to_numpy(np.int32), group=_groups(train))
        x_score = model.predict(x_held)
        metrics = _blend_metrics(held, x_score)
        fold_rows.append({
            "held_month": f"{held_month:%Y-%m}", "train_rows": len(train), "held_rows": len(held),
            "unlabelled_candidate_rows": int((~window.label_joined).sum()), **metrics,
        })
        gain = model.booster_.feature_importance(importance_type="gain")
        split = model.booster_.feature_importance(importance_type="split")
        total = max(float(gain.sum()), 1e-12)
        for field, raw_gain, raw_split in zip(fields, gain, split, strict=True):
            gain_rows.append({"held_month": f"{held_month:%Y-%m}", "feature": field, "gain": float(raw_gain), "gain_norm": float(raw_gain / total), "split": int(raw_split), "used": bool(raw_split > 0)})
        global_ix = _stratified_index(held, args.shap_cap, seed=SEED + ordinal)
        held_rank = pd.Series(_rank(held.assign(__score__=x_score), "__score__"))
        precision_ix = _stratified_index(held, args.shap_cap, rank=held_rank, seed=SEED + 1000 + ordinal)
        for sample, index in (("general", global_ix), ("precision_region_p70_100", precision_ix)):
            contribution = np.asarray(model.predict(x_held[index], pred_contrib=True), dtype=np.float64)[:, :-1]
            for field, mean_abs, median_abs in zip(fields, np.abs(contribution).mean(axis=0), np.median(np.abs(contribution), axis=0), strict=True):
                shap_rows.append({"held_month": f"{held_month:%Y-%m}", "sample": sample, "feature": field, "mean_abs_shap": float(mean_abs), "median_abs_shap": float(median_abs), "rows": len(index)})
        _univariate(held, x_held, fields, univariate_rows, held=f"{held_month:%Y-%m}")
        positions = pd.Series(np.arange(len(train), dtype=np.int64), index=train.index)
        for repeat in range(args.randomized_models_per_fold):
            seed = SEED + 20_000 + 1000 * ordinal + repeat
            random_train = _sample_queries(train, min(len(train), int(.85 * len(train))))
            random_train = random_train.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
            row_positions = positions.reindex(random_train.index).to_numpy(np.int64)
            random_model = LGBMRanker(**_model_params(seed, args.n_jobs, cheap=True, feature_fraction=.5 + .1 * (repeat % 3)))
            random_model.fit(x_train[row_positions], pd.to_numeric(random_train[target], errors="coerce").to_numpy(np.int32), group=_groups(random_train))
            random_gain = random_model.booster_.feature_importance(importance_type="gain")
            random_split = random_model.booster_.feature_importance(importance_type="split")
            random_total = max(float(random_gain.sum()), 1e-12)
            for field, raw_gain, raw_split in zip(fields, random_gain, random_split, strict=True):
                stability_rows.append({"held_month": f"{held_month:%Y-%m}", "seed": seed, "feature": field, "gain_norm": float(raw_gain / random_total), "used": bool(raw_split > 0)})
        _progress(args.out, stage="screen_fold_complete", held_month=f"{held_month:%Y-%m}", train_rows=len(train), held_rows=len(held), **metrics)
        del model, values, x_train, x_held, selected, train, held, window
        gc.collect()
    return (pd.DataFrame(gain_rows), pd.DataFrame(shap_rows), pd.DataFrame(univariate_rows), pd.DataFrame(stability_rows), pd.DataFrame(fold_rows))


def _shortlist(fields: list[str], correlation: pd.DataFrame, gain: pd.DataFrame, shap: pd.DataFrame, univariate: pd.DataFrame, stability: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    gain_summary = gain.groupby("feature", sort=False).agg(gain_median=("gain_norm", "median"), fold_use=("used", "mean"), split_median=("split", "median"))
    gain_summary["stable_gain"] = gain_summary.gain_median * gain_summary.fold_use
    shap_summary = shap.pivot_table(index="feature", columns="sample", values="mean_abs_shap", aggfunc="median").rename(columns={"general": "global_shap", "precision_region_p70_100": "precision_shap"}).fillna(0.0)
    uni_summary = univariate.groupby("feature", sort=False).univariate_ts_top10_ev.agg(["mean", "min"]).rename(columns={"mean": "univariate_top10_ev", "min": "univariate_worst_fold_ev"})
    stability_summary = stability.groupby("feature", sort=False).agg(random_use=("used", "mean"), random_gain=("gain_norm", "median"))
    summary = pd.DataFrame(index=fields).join(gain_summary).join(shap_summary).join(uni_summary).join(stability_summary).fillna(0.0)
    summary["family"] = [_feature_family(field) for field in summary.index]
    for source, name in (("stable_gain", "stable_gain_rank"), ("global_shap", "global_shap_rank"), ("precision_shap", "precision_shap_rank"), ("univariate_top10_ev", "univariate_rank"), ("random_use", "stability_rank")):
        summary[name] = summary[source].rank(method="average", pct=True, ascending=True).fillna(0.0)
    summary["screen_score"] = (.30 * summary.stable_gain_rank + .20 * summary.global_shap_rank + .25 * summary.precision_shap_rank + .15 * summary.univariate_rank + .10 * summary.stability_rank)
    union: set[str] = set()
    for column, count in (("stable_gain", 80), ("global_shap", 60), ("precision_shap", 60), ("univariate_top10_ev", 40), ("random_use", 60)):
        union.update(summary.nlargest(min(count, len(summary)), column).index)
    for _, family in summary.groupby("family", sort=False):
        union.add(str(family.nlargest(1, "screen_score").index[0]))
    eligible = summary.loc[summary.index.isin(union)].copy()
    representative = correlation.set_index("feature").cluster_representative
    eligible["correlation_representative"] = representative.reindex(eligible.index).eq(eligible.index).fillna(True)
    eligible = eligible.loc[eligible.correlation_representative]
    chosen = list(eligible.groupby("family", sort=False).screen_score.idxmax())
    chosen.extend(field for field in eligible.sort_values("screen_score", ascending=False, kind="stable").index if field not in set(chosen))
    chosen = chosen[:120]
    summary["in_shortlist_union"] = summary.index.isin(union)
    summary["selected_screen120"] = summary.index.isin(chosen)
    return summary.reset_index(names="feature").sort_values("screen_score", ascending=False, kind="stable"), chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--target", choices=("policy_ordinal_base",), default="policy_ordinal_base")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", nargs="+", default=("2026-02-01", "2026-03-01", "2026-04-01"))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--held-cap", type=int, default=15_000)
    parser.add_argument("--shap-cap", type=int, default=5_000)
    parser.add_argument("--randomized-models-per-fold", type=int, default=4)
    parser.add_argument("--correlation-sample-rows", type=int, default=4_096)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    target = TARGETS[args.target]
    valid = target.replace("_grade", "_valid")
    folds = tuple(_utc(value) for value in args.held_months)
    coverage_months = tuple(sorted({*folds, *(_utc(folds[0] - pd.DateOffset(months=4) + pd.DateOffset(months=i)) for i in range(4))}))
    all_fields = _numeric_fields(args.feature_root, f"{coverage_months[-1]:%Y-%m}")
    coverage = _coverage(args.feature_root, all_fields, coverage_months)
    eligible = coverage.loc[coverage.coverage_pass, "feature"].tolist()
    if len(eligible) < 700:
        raise AssertionError(f"hygiene left only {len(eligible)} full-universe fields")
    sample = _correlation_sample(args.feature_root, eligible, folds[0], args.correlation_sample_rows)
    survivors, clusters = _redundancy(eligible, coverage, sample, .995)
    if len(survivors) < 650:
        raise AssertionError(f"near-duplicate veto too aggressive: {len(survivors)}")
    coverage.to_parquet(args.out / "feature_hygiene_coverage.parquet", index=False, compression="zstd")
    clusters.to_parquet(args.out / "correlation_clusters.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "strict_r3_b0_fulluniverse_screen_v1", "scope": "offline B0 candidate only; E/T/B0 live contracts unchanged",
        "target": args.target, "target_column": target, "valid_column": valid, "gain_schedule": "g3_clipped_economic",
        "router": "frozen strict-OOF top50", "route_fraction": .50, "strict_oof": True,
        "feature_root": str(args.feature_root), "score_root": str(args.score_root), "router_root": str(args.router_root), "label_root": str(args.label_root),
        "hygiene": {"full_numeric": len(all_fields), "coverage_eligible": len(eligible), "post_near_duplicate": len(survivors), "near_duplicate_abs_spearman": .995},
        "screen": {"folds": len(folds), "seeds": 1, "randomized_models_per_fold": args.randomized_models_per_fold, "shap": "OOF TreeSHAP general + precision-region"},
        "outcomes_or_targets_in_features": False,
    }
    _exclusive(args.out / "run_manifest.json", manifest)
    _progress(args.out, stage="hygiene_complete", **manifest["hygiene"])
    gain, shap, univariate, stability, fold_metrics = _screen(args, survivors, target, valid)
    summary, selected = _shortlist(survivors, clusters, gain, shap, univariate, stability)
    contract = {"head": "B0_challenger", "target": args.target, "feature_contract": selected, "feature_count": len(selected), "feature_contract_sha256": hashlib.sha256("\n".join(selected).encode()).hexdigest(), "selection": "full-universe screen union: stable gain, global/precision OOF TreeSHAP, univariate economic rescue, randomized stability, semantic rescue, near-duplicate veto"}
    _exclusive(args.out / "b0_screen120_contract.json", contract)
    gain.to_parquet(args.out / "screen_gain.parquet", index=False, compression="zstd")
    shap.to_parquet(args.out / "screen_shap.parquet", index=False, compression="zstd")
    univariate.to_parquet(args.out / "screen_univariate.parquet", index=False, compression="zstd")
    stability.to_parquet(args.out / "screen_randomized_stability.parquet", index=False, compression="zstd")
    fold_metrics.to_parquet(args.out / "screen_fold_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(args.out / "b0_screen_feature_summary.parquet", index=False, compression="zstd")
    _progress(args.out, stage="screen_complete", screen120=len(selected), feature_contract_sha256=contract["feature_contract_sha256"])


if __name__ == "__main__":
    main()
