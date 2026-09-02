#!/usr/bin/env python3
"""Chronological CatBoost YetiRank HPO for the sealed P8u Meta winner.

The runner refuses arbitrary learner selection: the supplied cross-model
receipt must nominate CatBoost YetiRank.  HPO uses only the declared screen
months, chronological inner early stopping, an Optuna median pruner, and
query-safe training samples.  The selected configuration is then scored on
separate confirmation months with target-free receipts written before policy
outcomes are consulted.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRanker, Pool

import run_strict_r3_p8u_meta_crossmodel_v1 as cross
import run_strict_r3_p8u_meta_lgbm_objective_screen_v1 as objective
import run_strict_r3_p8u_meta_target_query_grid_v1 as screen


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "strict_r3_p8u_meta_catboost_hpo_v1"
IDENTITY = screen.IDENTITY
SEED = 1729


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(path: Path, payload: Mapping[str, Any]) -> None:
    with (path / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, default=str) + "\n")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    members = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    for member in members:
        digest.update(str(member).encode())
        with member.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _parse_months(values: Sequence[str]) -> tuple[pd.Timestamp, ...]:
    months = tuple(screen._utc_month(value) for value in values)
    if len(months) < 3 or tuple(sorted(months)) != months:
        raise ValueError("need at least three strictly chronological months")
    return months


def _inner_masks(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    timestamps = train.loc[:, ["__decision_ts__"]].drop_duplicates().sort_values("__decision_ts__", kind="stable")
    cut = max(1, min(len(timestamps) - 1, int(np.floor(.80 * len(timestamps)))))
    fit_timestamps = set(timestamps.iloc[:cut]["__decision_ts__"])
    fit = train["__decision_ts__"].isin(fit_timestamps).to_numpy(bool)
    if not fit.any() or fit.all():
        raise AssertionError("inner chronological split lacks fit or validation support")
    starts = np.flatnonzero(np.r_[True, train.__decision_ts__.to_numpy()[1:] != train.__decision_ts__.to_numpy()[:-1]])
    ends = np.r_[starts[1:], len(train)]
    if any(not (fit[start:end].all() or (~fit[start:end]).all()) for start, end in zip(starts, ends, strict=True)):
        raise AssertionError("inner early-stop split cut an exact timestamp query")
    return fit, ~fit


def _qid(frame: pd.DataFrame) -> np.ndarray:
    codes, _ = pd.factorize(frame["__decision_ts__"], sort=True)
    if np.any(codes < 0):
        raise AssertionError("invalid ranking query ID")
    return codes.astype(np.int64)


def _suggest(trial: optuna.Trial) -> dict[str, float | int]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.10, log=True),
        "depth": trial.suggest_int("depth", 3, 7),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.10, 30.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.01, 2.0, log=True),
        "rsm": trial.suggest_float("rsm", 0.70, 0.95),
        "subsample": trial.suggest_float("subsample", 0.70, 0.95),
    }


def _fit_predict(
    *, train_x: np.ndarray, labels: np.ndarray, train: pd.DataFrame,
    held_x: np.ndarray, params: Mapping[str, float | int], seed: int,
) -> tuple[np.ndarray, int]:
    fit, valid = _inner_masks(train)
    train_fit, train_valid = train.loc[fit].reset_index(drop=True), train.loc[valid].reset_index(drop=True)
    model = CatBoostRanker(
        loss_function="YetiRank:mode=NDCG;top=12",
        eval_metric="NDCG:top=10",
        iterations=2000,
        learning_rate=float(params["learning_rate"]),
        depth=int(params["depth"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        random_strength=float(params["random_strength"]),
        rsm=float(params["rsm"]),
        bootstrap_type="Bernoulli",
        subsample=float(params["subsample"]),
        random_seed=int(seed),
        thread_count=1,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=30,
    )
    model.fit(
        Pool(train_x[fit], labels[fit], group_id=_qid(train_fit)),
        eval_set=Pool(train_x[valid], labels[valid], group_id=_qid(train_valid)),
        use_best_model=True,
        verbose=False,
    )
    output = np.asarray(model.predict(held_x), dtype=np.float32)
    if not np.isfinite(output).all():
        raise AssertionError("CatBoost HPO produced a non-finite score")
    return output, int(model.get_best_iteration())


def _score(
    *, prepared: objective.PreparedFold, arm: screen.Arm,
    params: Mapping[str, float | int], seed: int,
) -> tuple[pd.DataFrame, int]:
    raw, best_iteration = _fit_predict(
        train_x=prepared.train_x,
        labels=prepared.labels,
        train=prepared.train_frame,
        held_x=prepared.held_x,
        params=params,
        seed=seed,
    )
    if arm.family == "over":
        raw *= -1.0
    score = prepared.held_target_free.loc[:, list(IDENTITY) + ["base_score", "base_rank_ts"]].copy().reset_index(drop=True)
    score["meta_raw_score"] = raw
    rank = score.loc[:, ["candidate_id", "__decision_ts__"]].copy()
    rank["value"] = raw
    score["meta_rank_ts"] = screen._rank_desc(rank, "value")
    score["arm"] = arm.name
    score["family"] = arm.family
    score["scale"] = arm.scale
    score["query_contract"] = arm.query
    score["trial"] = "catboost_yetirank_hpo_winner"
    score["held_month"] = f"{prepared.held_month:%Y-%m}"
    score["target_free"] = True
    score["fit_weight_profile"] = "unweighted"
    return score, best_iteration


def _prepare(
    *, config: Path, arm: screen.Arm, fields: Sequence[str], continuous_fields: Sequence[str],
    sidecar: Path | None, months: Sequence[pd.Timestamp], seed_offset: int,
    raw_override: Mapping[str, Any] | None = None,
) -> tuple[list[objective.PreparedFold], screen.Spec, pd.DataFrame, Path, Path, Path, tuple[Path, ...]]:
    raw = dict(raw_override) if raw_override is not None else json.loads(config.read_text())
    spec = screen.Spec(raw=raw, config_path=config)
    panel_fields = tuple(field for field in fields if field not in set(continuous_fields))
    base_root = ROOT / str(spec.source["base_target_free_root"])
    feature_roots = tuple(ROOT / str(value) for value in spec.source["full_feature_roots"])
    policy_path = ROOT / str(spec.source["policy_labels"])
    path_root = ROOT / str(spec.source["path_labels"])
    policy = screen._read_policy(policy_path)
    prepared = [objective._prepare_fold(
        base_root=base_root,
        feature_roots=feature_roots,
        policy=policy,
        path_root=path_root,
        arm=arm,
        fields=fields,
        panel_fields=panel_fields,
        continuous_sidecar=sidecar,
        continuous_fields=continuous_fields,
        spec=spec,
        held_month=month,
        seed=int(spec.folds["seed"]) + seed_offset + index,
    ) for index, month in enumerate(months)]
    return prepared, spec, policy, base_root, policy_path, path_root, feature_roots


def _evaluate(
    *, folds: Sequence[objective.PreparedFold], arm: screen.Arm,
    spec: screen.Spec, params: Mapping[str, float | int], seed_offset: int, trial: optuna.Trial | None = None,
    persist: Path | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    metrics: list[dict[str, Any]] = []
    weekly_parts: list[pd.DataFrame] = []
    bands_parts: list[pd.DataFrame] = []
    audit: list[dict[str, Any]] = []
    for index, prepared in enumerate(folds):
        score, best_iteration = _score(prepared=prepared, arm=arm, params=params, seed=SEED + seed_offset + index)
        if persist is not None:
            path = persist / "target_free_scores" / "catboost_yetirank_hpo_winner" / f"month={prepared.held_month:%Y-%m}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            score.to_parquet(path, index=False, compression="zstd")
        weekly, bands, fold_metrics = screen._metrics(
            score=score, held_labelled=prepared.held_labelled, held_anchor=prepared.held_anchor,
            spec=spec,
        )
        weekly["held_month"] = f"{prepared.held_month:%Y-%m}"
        weekly_parts.append(weekly)
        if not bands.empty:
            bands["held_month"] = f"{prepared.held_month:%Y-%m}"
            bands_parts.append(bands)
        metrics.append(fold_metrics)
        audit.append({**prepared.audit, **fold_metrics, "best_iteration": best_iteration})
        if trial is not None:
            value = float(pd.DataFrame(metrics).sstable_meta.mean())
            trial.report(value, index)
            if index >= 1 and trial.should_prune():
                raise optuna.TrialPruned()
    aggregate = {key: float(value) for key, value in pd.DataFrame(metrics).mean(numeric_only=True).to_dict().items()}
    return aggregate, audit, weekly_parts, bands_parts


def run(
    *, config: Path, cross_root: Path, feature_contract: Path, out: Path,
    screen_month_values: Sequence[str], confirmation_month_values: Sequence[str], trials: int,
    source_override: Path | None = None,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    if trials < 8:
        raise ValueError("HPO needs at least eight trials")
    required = [cross_root / "run_manifest.json", cross_root / "correctness_report.json", cross_root / "cross_model_winner.parquet"]
    if any(not path.exists() for path in required):
        raise FileNotFoundError("cross-model receipt is incomplete")
    if not all(json.loads((cross_root / "correctness_report.json").read_text()).values()):
        raise AssertionError("cross-model receipt failed correctness")
    winner = pd.read_parquet(cross_root / "cross_model_winner.parquet")
    if len(winner) != 1 or str(winner.iloc[0].model_family) != "catboost_yetirank":
        raise AssertionError("this HPO contract only accepts the sealed CatBoost YetiRank winner")
    raw, applied_source_override = screen._apply_source_override(json.loads(config.read_text()), source_override)
    arm_name = str(winner.iloc[0].arm)
    arm = {item.name: item for item in screen._arm_specs(raw, None)}.get(arm_name)
    if arm is None or arm.query != "timestamp":
        raise AssertionError("cross-model winner target/query no longer matches the frozen timestamp contract")
    fields, continuous_fields, sidecar = objective._read_contract(feature_contract)
    screen_months = _parse_months(screen_month_values)
    confirmation_months = _parse_months(confirmation_month_values)
    if screen_months[-1] >= confirmation_months[0]:
        raise AssertionError("confirmation months must follow every HPO screen month")
    out.mkdir(parents=True)
    spec = screen.Spec(raw=raw, config_path=config)
    panel_fields = tuple(field for field in fields if field not in set(continuous_fields))
    coverage_months = tuple(screen._month_range(screen_months[0] - pd.DateOffset(months=5), screen._month_end(confirmation_months[-1])))
    coverage = screen._preflight(spec, panel_fields, coverage_months)
    coverage.to_parquet(out / "source_coverage_audit.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline strict-OOF CatBoost YetiRank Meta HPO; no MC1/admission/portfolio/live/exchange mutation",
        "cross_model_root": str(cross_root),
        "cross_model_winner": winner.to_dict("records"),
        "arm": dataclasses.asdict(arm),
        "feature_contract": str(feature_contract),
        "feature_contract_sha256": _sha(feature_contract),
        "feature_count": len(fields),
        "source_override": str(source_override) if source_override else None,
        "source_override_sha256": _sha(source_override) if source_override else None,
        "source_override_payload": applied_source_override,
        "screen_months": [f"{month:%Y-%m}" for month in screen_months],
        "confirmation_months": [f"{month:%Y-%m}" for month in confirmation_months],
        "trials": int(trials),
        "selection": "mean SStableMeta on screen folds; median pruning after two folds; confirmation is not HPO input",
        "hpo_space": {
            "learning_rate": [0.015, 0.10, "log"], "depth": [3, 7],
            "l2_leaf_reg": [0.10, 30.0, "log"], "random_strength": [0.01, 2.0, "log"],
            "rsm": [0.70, 0.95], "subsample": [0.70, 0.95],
            "iterations_ceiling": 2000, "early_stopping_rounds": 30,
        },
        "external_hpo_reference": "https://chatgpt.com/s/t_6a905139226881918b63983b8c5c0a16 (unavailable to this environment at run time; no parameters were imported from it)",
        "causality": raw["causality"],
    })
    screen_folds, *_unused = _prepare(
        config=config, arm=arm, fields=fields, continuous_fields=continuous_fields, sidecar=sidecar,
        months=screen_months, seed_offset=0, raw_override=raw,
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1, interval_steps=1),
    )
    trial_rows: list[dict[str, Any]] = []

    def objective_fn(trial: optuna.Trial) -> float:
        params = _suggest(trial)
        try:
            metrics, audit, _weekly, _bands = _evaluate(
                folds=screen_folds, arm=arm, spec=spec, params=params, seed_offset=100_000 * trial.number, trial=trial,
            )
        except optuna.TrialPruned:
            trial_rows.append({"trial": trial.number, "status": "pruned", **params})
            raise
        row = {"trial": trial.number, "status": "complete", **params, **metrics,
               "mean_best_iteration": float(np.mean([item["best_iteration"] for item in audit]))}
        trial_rows.append(row)
        _progress(out, {"event": "hpo_trial_complete", **row})
        return float(metrics["sstable_meta"])

    study.optimize(objective_fn, n_trials=int(trials), n_jobs=1, gc_after_trial=True)
    results = pd.DataFrame(trial_rows).sort_values(["status", "sstable_meta", "trial"], ascending=[True, False, True], kind="stable")
    results.to_parquet(out / "hpo_trials.parquet", index=False, compression="zstd")
    complete = results.loc[results.status.eq("complete")].copy()
    if complete.empty:
        raise AssertionError("all HPO trials were pruned or failed")
    winner_row = complete.sort_values(["sstable_meta", "conditional_mi_meta_policy_given_base", "trial"], ascending=[False, False, True], kind="stable").iloc[0]
    winner_params = {key: winner_row[key].item() if hasattr(winner_row[key], "item") else winner_row[key] for key in _suggest(study.best_trial)}
    _once(out / "selected_params.json", {
        "schema": SCHEMA,
        "model_family": "catboost_yetirank",
        "selection_metric": "mean SStableMeta over frozen chronological screen months",
        "trial": int(winner_row.trial),
        "params": winner_params,
        "screen_metrics": {key: float(winner_row[key]) for key in complete.columns if key.startswith(("sstable_", "conditional_", "mean_", "worst_", "residual_")) and pd.notna(winner_row[key])},
    })
    confirm_folds, *_unused = _prepare(
        config=config, arm=arm, fields=fields, continuous_fields=continuous_fields, sidecar=sidecar,
        months=confirmation_months, seed_offset=50_000, raw_override=raw,
    )
    confirmation, audit, weekly, bands = _evaluate(
        folds=confirm_folds, arm=arm, spec=spec, params=winner_params, seed_offset=2_000_000, persist=out,
    )
    pd.DataFrame(audit).to_parquet(out / "confirmation_fold_metrics.parquet", index=False, compression="zstd")
    pd.concat(weekly, ignore_index=True).to_parquet(out / "confirmation_weekly_sstable_meta.parquet", index=False, compression="zstd")
    (pd.concat(bands, ignore_index=True) if bands else pd.DataFrame()).to_parquet(
        out / "confirmation_base_band_conversion_metrics.parquet", index=False, compression="zstd"
    )
    _once(out / "confirmation_metrics.json", confirmation)
    _once(out / "correctness_report.json", {
        "cross_model_winner_is_sealed_catboost_yetirank": True,
        "all_train_labels_resolved_before_reserve": True,
        "inner_early_stop_is_chronological_and_query_intact": True,
        "hpo_screen_precedes_confirmation_months": True,
        "held_confirmation_scores_persisted_before_outcome_metrics": True,
        "feature_medians_fit_train_only": True,
        "no_policy_or_path_field_in_target_free_inputs": True,
        "no_mc1_admission_portfolio_live_or_exchange_mutation": True,
    })
    _progress(out, {"event": "complete", "selected_trial": int(winner_row.trial), "confirmation_sstable_meta": confirmation["sstable_meta"]})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cross-root", type=Path, required=True)
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--screen-months", nargs="+", required=True)
    parser.add_argument("--confirmation-months", nargs="+", required=True)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--source-override", type=Path, help="immutable source-only binding receipt")
    args = parser.parse_args()
    print(run(
        config=args.config.resolve(),
        cross_root=args.cross_root.resolve(),
        feature_contract=args.feature_contract.resolve(),
        out=args.out.resolve(),
        screen_month_values=args.screen_months,
        confirmation_month_values=args.confirmation_months,
        trials=int(args.trials),
        source_override=args.source_override.resolve() if args.source_override else None,
    ))


if __name__ == "__main__":
    main()
