#!/usr/bin/env python3
"""Strict-OOF HPO screen for one selected incumbent meta-head family.

This is deliberately a *development-only* screen.  The retained upstream is
always the frozen 50/50 efficiency/timing blend.  It uses four separated
earlier calendar months for timestamp-local ranking/stability selection and
does not fit MC1, run admission, replay a portfolio, or change inference.

The later Apr--Jul 2026 span remains untouched by this HPO and is reserved for
the downstream MC1 + dual-admission portfolio comparison of its finalists.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_strict_r3_incumbent_meta_target_query_grid_v1 as grid  # noqa: E402


SCHEMA = "strict_r3_incumbent_meta_family_hpo_v1"
SEED = 1729
DEFAULT_CONTRACT = (
    ROOT
    / "data_perp/artifacts/strict_r3_incumbent_meta_fullfeatures_selection_20260827_v3"
    / "contracts/under/under_f50.json"
)
DEFAULT_FEATURE_ROOTS = (
    ROOT / "data_perp/artifacts/strict_r3_incumbent_meta_causal_features_20260827_preaug_v1",
    ROOT / "data_perp/artifacts/strict_r3_incumbent_meta_causal_features_20260827_v1",
)
UNDER_ARM = grid.Arm(
    name="under_atr100__timestamp__gain_small",
    family="under",
    scale="atr",
    query="timestamp",
    threshold=1.0,
    classes=7,
    gain_schedule="small",
    truncation_level=None,
)


def _load_arm(config_path: Path | None, arm_name: str | None) -> grid.Arm:
    """Load one predeclared target/query arm, or preserve the U default.

    Target/query selection must remain upstream of HPO.  Callers may therefore
    choose only an exact arm already frozen in a candidate configuration.
    Omitting both arguments preserves the historical U configuration.
    """
    if config_path is None and arm_name is None:
        return UNDER_ARM
    if config_path is None or not arm_name:
        raise ValueError("--arm-config and --arm-name must be supplied together")
    payload = json.loads(config_path.read_text())
    choices = payload.get("arms")
    if not isinstance(choices, list):
        raise ValueError(f"{config_path}: requires arms[]")
    matches = [item for item in choices if isinstance(item, dict) and item.get("name") == arm_name]
    if len(matches) != 1:
        raise ValueError(f"{config_path}: expected exactly one arm named {arm_name!r}")
    item = matches[0]
    try:
        family = str(item["family"])
        scale = str(item["scale"])
        query = str(item["query"])
        if family not in {"magnitude", "under", "over", "state"}:
            raise ValueError(f"unsupported family {family!r}")
        if scale not in {"bps", "atr", "sqrt_atr"}:
            raise ValueError(f"unsupported scale {scale!r}")
        if query not in {"base_band", "timestamp", "base_band_block28"}:
            raise ValueError(f"unsupported query {query!r}")
        threshold = item.get("threshold")
        if family in {"under", "over"} and threshold is None:
            raise ValueError("under/over requires threshold")
        edges = item.get("state_edges")
        return grid.Arm(
            name=str(item["name"]), family=family, scale=scale, query=query,
            threshold=None if threshold is None else float(threshold),
            classes=int(item.get("classes", 7)),
            state_edges=None if edges is None else tuple(float(value) for value in edges),
            gain_schedule=str(item.get("gain_schedule", "medium")),
            truncation_level=(None if item.get("truncation_level") is None else int(item["truncation_level"])),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{config_path}: invalid arm {arm_name!r}") from exc


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _parse_months(raw: str) -> tuple[pd.Timestamp, ...]:
    output = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in raw.split(",") if item.strip())
    if len(output) < 3 or tuple(sorted(set(output))) != output:
        raise ValueError("--held-months must contain >=3 unique chronological calendar months")
    return output


def _fold_seed(month: pd.Timestamp) -> int:
    """Stable seed for a calendar fold, independent of caller month order."""
    return SEED + 12 * (int(month.year) - 2000) + int(month.month)


@dataclass(frozen=True)
class PreparedFold:
    month: pd.Timestamp
    train_x: np.ndarray
    train_y: np.ndarray
    fit_mask: np.ndarray
    tune_mask: np.ndarray
    fit_groups: list[int]
    tune_groups: list[int]
    held_x: np.ndarray
    held: pd.DataFrame
    held_residual_bps: np.ndarray


def _prepare_fold(fold: grid.Fold, *, arm: grid.Arm, seed: int) -> PreparedFold:
    sampled = grid._sample_queries(fold.train, grid.MAX_TRAIN_ROWS, seed)
    labels, _residual, _info = grid._target(sampled, arm, train=True)
    valid = labels >= 0
    train = sampled.loc[valid].reset_index(drop=True)
    labels = labels[valid]
    if len(train) < 20_000 or len(np.unique(labels)) < 2:
        raise AssertionError(f"{fold.held_month:%Y-%m}: insufficient under-target support")
    held_anchor = grid._fit_anchor(train)
    _held_labels, held_residual, _held_info = grid._target(
        fold.held, arm, train=False, held_anchor=held_anchor
    )
    train_x, held_x = grid._impute(
        grid._matrix(train, fold.source_features),
        grid._matrix(fold.held, fold.source_features),
    )
    order, ordered_query_ids, _groups = grid._ordered_query(train, grid._query_ids(train, arm.query))
    unique = pd.Index(ordered_query_ids).drop_duplicates()
    cut = max(1, int(math.floor(.80 * len(unique))))
    fit_queries, tune_queries = set(unique[:cut]), set(unique[cut:])
    fit_mask = np.asarray([value in fit_queries for value in ordered_query_ids], dtype=bool)
    tune_mask = np.asarray([value in tune_queries for value in ordered_query_ids], dtype=bool)
    if not fit_mask.any() or not tune_mask.any():
        raise AssertionError(f"{fold.held_month:%Y-%m}: insufficient causal early-stop groups")
    fit_groups = pd.Series(ordered_query_ids[fit_mask]).groupby(pd.Series(ordered_query_ids[fit_mask]), sort=False).size().astype(int).tolist()
    tune_groups = pd.Series(ordered_query_ids[tune_mask]).groupby(pd.Series(ordered_query_ids[tune_mask]), sort=False).size().astype(int).tolist()
    return PreparedFold(
        month=fold.held_month,
        train_x=train_x[order],
        train_y=labels[order],
        fit_mask=fit_mask,
        tune_mask=tune_mask,
        fit_groups=fit_groups,
        tune_groups=tune_groups,
        held_x=held_x,
        held=fold.held.copy(),
        held_residual_bps=np.asarray(held_residual, dtype=np.float32),
    )


def _params(trial: optuna.Trial, *, train_rows: int) -> dict[str, Any]:
    # Encode valid depth/leaf pairs as one categorical hyperparameter.  A
    # dynamic ``suggest_int`` range is invalid for depth=2 and also changes
    # Optuna's distribution across trials.
    geometry = trial.suggest_categorical(
        "tree_geometry", ("d2_l3", "d2_l4", "d3_l7", "d3_l8", "d4_l15", "d4_l16", "d5_l31")
    )
    depth, leaves = ((int(part[1:]) for part in geometry.split("_")))
    min_fraction = trial.suggest_float("min_data_fraction", .003, .02, log=True)
    trunc = trial.suggest_categorical("truncation", ("none", 5, 10, 20))
    return {
        "n_estimators": 1_200,
        "learning_rate": trial.suggest_float("learning_rate", .02, .08, log=True),
        "max_depth": depth,
        "num_leaves": leaves,
        "min_child_samples": max(80, int(round(train_rows * min_fraction))),
        "subsample": trial.suggest_float("bagging_fraction", .70, .92),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("feature_fraction", .70, .95),
        "reg_alpha": trial.suggest_float("lambda_l1", 1e-5, 1.0, log=True),
        "reg_lambda": trial.suggest_float("lambda_l2", 1.0, 30.0, log=True),
        "min_split_gain": trial.suggest_float("min_gain_to_split", 1e-5, .01, log=True),
        "sigmoid": trial.suggest_float("sigmoid", .5, 1.5),
        "lambdarank_truncation_level": None if trunc == "none" else int(trunc),
    }


def _serialise_trial_params(params: dict[str, Any]) -> dict[str, Any]:
    """Keep Parquet trial columns type-stable across categorical choices."""
    output = dict(params)
    if "truncation" in output:
        output["truncation"] = str(output["truncation"])
    return output


def _rank(frame: pd.DataFrame, score: np.ndarray) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__"]].copy()
    work["meta_score"] = score
    return grid._rank_desc(work, "meta_score")


def _tail_bps(frame: pd.DataFrame, score: np.ndarray, *, k: int) -> float:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "policy_path_valid", "policy_net_bps"]].copy()
    work["score"] = score
    valid = work.policy_path_valid.fillna(False).astype(bool) & np.isfinite(pd.to_numeric(work.policy_net_bps, errors="coerce"))
    work = work.loc[valid].copy()
    ordered = work.sort_values(["__decision_ts__", "score", "candidate_id"], ascending=[True, False, True], kind="stable")
    return float(ordered.groupby("__decision_ts__", sort=False).head(k).groupby("__decision_ts__", sort=False).policy_net_bps.mean().mean())


def _metrics(item: PreparedFold, raw: np.ndarray, *, best_iteration: int) -> dict[str, float | int | str]:
    meta_rank = _rank(item.held, raw)
    base_rank = pd.to_numeric(item.held.inc_base_rank_ts, errors="coerce").to_numpy(float)
    policy = pd.to_numeric(item.held.policy_net_bps, errors="coerce").to_numpy(float)
    valid = item.held.policy_path_valid.fillna(False).to_numpy(bool) & np.isfinite(policy)
    values: dict[str, float | int | str] = {
        "held_month": f"{item.month:%Y-%m}",
        "best_iteration": int(best_iteration),
        "valid_policy_rows": int(valid.sum()),
        "residual_spearman_ic": float(spearmanr(meta_rank[valid], item.held_residual_bps[valid]).statistic) if int(valid.sum()) >= 20 else float("nan"),
        "conditional_mi_meta_policy_given_base": grid._conditional_mi(meta_rank[valid], base_rank[valid], policy[valid]),
    }
    combined = .75 * base_rank + .25 * meta_rank
    for k in (1, 2, 5, 10):
        values[f"base_top{k}_bps"] = _tail_bps(item.held, base_rank, k=k)
        values[f"blend_top{k}_bps"] = _tail_bps(item.held, combined, k=k)
        values[f"blend_delta_top{k}_bps"] = float(values[f"blend_top{k}_bps"]) - float(values[f"base_top{k}_bps"])
    return values


def _selection(rows: list[dict[str, float | int | str]]) -> float:
    values = pd.DataFrame(rows)
    top10 = pd.to_numeric(values.blend_top10_bps, errors="coerce")
    # Timestamp-local Top-10 is primary; use both lower-tail summaries to
    # favour broad portability over a one-month windfall.  The Top-2 term is
    # deliberately subordinate and only rewards a useful extreme tail.
    return float(
        .45 * top10.mean()
        + .25 * top10.quantile(.25)
        + .20 * top10.min()
        + .10 * pd.to_numeric(values.blend_top2_bps, errors="coerce").mean()
    )


def _fit(
    item: PreparedFold,
    params: dict[str, Any],
    *,
    arm: grid.Arm,
    seed: int,
    model_jobs: int,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    cleaned = {key: value for key, value in params.items() if value is not None}
    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=grid._gain(item.train_y, arm.gain_schedule),
        lambdarank_norm=True,
        random_state=seed,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        n_jobs=model_jobs,
        **cleaned,
    )
    model.fit(
        item.train_x[item.fit_mask], item.train_y[item.fit_mask], group=item.fit_groups,
        eval_set=[(item.train_x[item.tune_mask], item.train_y[item.tune_mask])],
        eval_group=[item.tune_groups],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )
    raw = np.asarray(model.predict(item.held_x), dtype=np.float32)
    # The over target learns likelihood of an adverse surprise.  Its score
    # must be reversed before every rank-based development metric, exactly as
    # the final strict-OOF scorer does, so HPO chooses the same orientation
    # that MC1 will consume.
    if arm.family == "over":
        raw *= -1.0
    return raw, _metrics(item, raw, best_iteration=int(model.best_iteration_ or model.n_estimators))


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(f"{args.out}: immutable HPO output already exists")
    arm = _load_arm(args.arm_config, args.arm_name)
    months = _parse_months(args.held_months)
    fields = grid._load_feature_contract(args.feature_contract)
    roots = tuple(Path(item.strip()) for item in args.feature_roots.split(",") if item.strip())
    if not 30 <= len(fields) <= 70 or len(roots) < 2:
        raise ValueError("requires a 30..70-field contract and predecessor/current feature roots")
    policy = grid._read_policy(args.policy)
    folds = grid._prepare_folds(
        source_root=args.source_root,
        policy=policy,
        path_root=args.path_root,
        held_months=months,
        full_feature_roots=roots,
        full_feature_fields=fields,
    )
    prepared = [_prepare_fold(fold, arm=arm, seed=_fold_seed(fold.held_month)) for fold in folds]
    args.out.mkdir(parents=True)
    _exclusive_json(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline strict-OOF development HPO for one predeclared meta family; no MC1/admission/portfolio/inference/live/exchange mutation",
        "incumbent_upstream": "0.50 * efficiency_bps + 0.50 * timing_bps",
        "arm": vars(arm),
        "arm_config": None if args.arm_config is None else str(args.arm_config),
        "feature_contract": str(args.feature_contract),
        "feature_count": len(fields),
        "feature_roots": [str(root) for root in roots],
        "development_held_months": [f"{month:%Y-%m}" for month in months],
        "untouched_downstream_validation": "2026-04 through 2026-07",
        "objective": "timestamp-local blend Top-10 policy bps with cross-month mean/q25/min stability and subordinate Top-2 term",
        "hpo": {"trials": args.trials, "pruner": "MedianPruner(startup=6,warmup_folds=2)", "early_stopping_rounds": 30},
    })
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=6, n_warmup_steps=2, interval_steps=1),
    )
    trial_rows: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        rows: list[dict[str, float | int | str]] = []
        for fold_index, item in enumerate(prepared):
            params = _params(trial, train_rows=int(item.fit_mask.sum()))
            _raw, metric = _fit(item, params, arm=arm, seed=_fold_seed(item.month), model_jobs=args.model_jobs)
            rows.append(metric)
            partial = _selection(rows)
            trial.report(partial, step=fold_index)
            if trial.should_prune():
                raise optuna.TrialPruned()
        value = _selection(rows)
        trial_rows.append({
            "trial": trial.number, "state": "complete", "selection_score": value,
            **{key: float(pd.to_numeric(pd.DataFrame(rows)[key], errors="coerce").mean()) for key in ("blend_top1_bps", "blend_top2_bps", "blend_top5_bps", "blend_top10_bps", "blend_delta_top10_bps", "residual_spearman_ic", "conditional_mi_meta_policy_given_base")},
            **_serialise_trial_params(trial.params),
        })
        return value

    study.optimize(objective, n_trials=args.trials, n_jobs=1, gc_after_trial=True)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.PRUNED:
            trial_rows.append({"trial": trial.number, "state": "pruned", **_serialise_trial_params(trial.params)})
    complete = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        raise RuntimeError("no complete HPO trials")
    trials = pd.DataFrame(trial_rows).sort_values(["state", "selection_score"], ascending=[True, False], na_position="last")
    trials.to_parquet(args.out / "trials.parquet", index=False, compression="zstd")
    # Refit the winning parameterization across the same blocked OOF folds so
    # its development receipt can be audited without retaining target fields.
    best = study.best_trial
    oof: list[pd.DataFrame] = []
    fold_metrics: list[dict[str, float | int | str]] = []
    for fold_index, item in enumerate(prepared):
        params = _params(best, train_rows=int(item.fit_mask.sum()))
        raw, metric = _fit(item, params, arm=arm, seed=_fold_seed(item.month), model_jobs=args.model_jobs)
        fold_metrics.append(metric)
        score = item.held.loc[:, list(grid.IDENTITY)].copy()
        score["meta_raw_score"] = raw
        score["meta_rank_ts"] = _rank(item.held, raw)
        score["arm"] = arm.name
        oof.append(score)
    pd.DataFrame(fold_metrics).to_parquet(args.out / "best_fold_metrics.parquet", index=False, compression="zstd")
    pd.concat(oof, ignore_index=True).to_parquet(args.out / "best_oof_target_free_predictions.parquet", index=False, compression="zstd")
    chosen = _params(best, train_rows=int(np.median([item.fit_mask.sum() for item in prepared])))
    _exclusive_json(args.out / "winner.json", {
        "trial": best.number,
        "selection_score": best.value,
        "optuna_params": best.params,
        "representative_model_params": chosen,
        "arm": vars(arm),
        "feature_contract": str(args.feature_contract),
        "development_only": True,
        "next_step": "strictly rescore selected candidates, then assess them through separate prequential Current/BCF MC1 maps and the unchanged dual-admission portfolio replay",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--arm-config", type=Path, default=None, help="predeclared candidate config containing arms[]")
    parser.add_argument("--arm-name", default=None, help="exact arm name selected from --arm-config")
    parser.add_argument("--held-months", default="2025-09,2025-11,2026-01,2026-03")
    parser.add_argument("--trials", type=int, default=24)
    parser.add_argument("--model-jobs", type=int, default=4)
    parser.add_argument("--source-root", type=Path, default=grid.DEFAULT_SOURCE_ROOT)
    parser.add_argument("--policy", type=Path, default=grid.DEFAULT_POLICY)
    parser.add_argument("--path-root", type=Path, default=grid.DEFAULT_PATH_ROOT)
    parser.add_argument("--feature-roots", default=",".join(str(root) for root in DEFAULT_FEATURE_ROOTS))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
