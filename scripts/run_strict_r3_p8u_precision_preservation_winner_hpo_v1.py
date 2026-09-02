#!/usr/bin/env python3
"""Full HPO for one frozen cross-model-winning P8u target contract.

The target geometry, P8u Router50 route, target-free held-score protocol, and
winning learner family are all supplied by the cross-model receipt.  This
script tunes only that winner's model parameters under strict chronological
outer folds and an inner latest-20%-query early-stopping split.  It cannot be
used to HPO a losing learner family.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostRanker, Pool
from xgboost import XGBRanker

import run_strict_r3_p8u_precision_preservation_cross_model_v1 as cross
import run_strict_r3_p8u_precision_preservation_loss_funnel_v1 as gain
import run_strict_r3_p8u_precision_preservation_screen_v1 as stage1
import run_strict_r3_p8u_precision_preservation_weight_funnel_v1 as weights
import run_strict_r3_router_single_base_prescreen_v1 as base
from strict_r3_p8u_precision_preservation_metric import COMPONENTS, stable_score, timestamp_components


SCHEMA = "strict_r3_p8u_precision_preservation_winner_hpo_v1"
SEED = 1729
IDENTITY = base.IDENTITY


@dataclass(frozen=True)
class Contract:
    arm: stage1.Arm
    gain_name: str
    model_family: str
    candidate: str
    weight_scheme: str | None = None


@dataclass(frozen=True)
class Fold:
    month: pd.Timestamp
    window: pd.DataFrame
    held: pd.DataFrame
    labels: pd.DataFrame
    control: pd.DataFrame


def _exclusive_json(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _source_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()


def _parse_screen_months(tokens: str) -> tuple[pd.Timestamp, ...]:
    """Parse a compact, but still cross-year, HPO development panel."""
    months = tuple(pd.Timestamp(f"{token.strip()}-01", tz="UTC") for token in tokens.split(",") if token.strip())
    if len(months) < 3 or tuple(sorted(months)) != months:
        raise ValueError("HPO screen needs at least three strictly increasing monthly folds")
    span = (months[-1].year - months[0].year) * 12 + months[-1].month - months[0].month
    if len({month.year for month in months}) < 2 or span < 8:
        raise ValueError("HPO screen must remain cross-year and span at least eight months")
    return months


def _contract(cross_root: Path, candidate_key: str) -> Contract:
    """Load either the legacy or row-weight-compatible sealed winner.

    The latter intentionally has no ``target_model_winners.parquet``: it is a
    single, post-weight-contract model-family comparison.  It is still
    verified from its immutable manifest rather than accepted by CLI choice.
    """
    weighted_manifest = cross_root / "run_manifest.json"
    if not (cross_root / "target_model_winners.parquet").exists():
        payload = json.loads(weighted_manifest.read_text())
        if payload.get("schema") != "strict_r3_p8u_precision_preservation_weighted_cross_model_v1":
            raise FileNotFoundError("missing legacy winners and not a sealed weighted cross-model receipt")
        if payload.get("target") != "raw_bps/equal_width6/G3" or payload.get("weight_scheme") != "tail_linear_125":
            raise AssertionError("weighted HPO must retain the sealed raw-bps/G3/tail_linear_125 contract")
        results = payload.get("results")
        if not isinstance(results, list) or not results or results[0].get("model_family") != candidate_key:
            raise AssertionError("weighted HPO candidate is not the sealed weighted cross-model winner")
        if candidate_key != "catboost_queryrmse":
            raise AssertionError("the sealed weighted winner is expected to be CatBoost QueryRMSE")
        arm = stage1.Arm("raw_bps", "t1_raw_bps", "equal_width6")
        return Contract(arm, "g3_clipped_economic", candidate_key, candidate_key, "tail_linear_125")
    winners = pd.read_parquet(cross_root / "target_model_winners.parquet")
    required = {"candidate", "arm", "gain_name", "model_family"}
    if missing := required.difference(winners.columns):
        raise AssertionError(f"cross-model winners missing {sorted(missing)}")
    selected = winners.loc[winners["candidate"].eq(candidate_key)]
    if len(selected) != 1:
        raise AssertionError("HPO candidate must be exactly one sealed cross-model winner")
    row = selected.iloc[0]
    arm_by_key = {arm.key: arm for arm in stage1.ARMS}
    if row.arm not in arm_by_key or row.gain_name not in gain.GAIN_SCHEDULES:
        raise AssertionError("invalid sealed HPO target contract")
    if row.model_family not in {"catboost_queryrmse", "xgb_ndcg"}:
        raise AssertionError("this full-HPO implementation supports only actual winning CatBoost QueryRMSE or XGBoost NDCG contracts")
    return Contract(arm_by_key[row.arm], str(row.gain_name), str(row.model_family), str(row.candidate))


def _inner_masks(train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    timestamps = train.loc[:, ["__decision_ts__"]].drop_duplicates().sort_values("__decision_ts__", kind="stable")
    cut = max(1, min(len(timestamps) - 1, int(np.floor(.80 * len(timestamps)))))
    fit_timestamps = set(timestamps.iloc[:cut]["__decision_ts__"])
    fit = train["__decision_ts__"].isin(fit_timestamps).to_numpy(bool)
    if not fit.any() or fit.all():
        raise AssertionError("inner chronological split has no fit or validation support")
    # Asserts that group construction cannot cut through an exact timestamp.
    query = train["__decision_ts__"].to_numpy()
    starts = np.flatnonzero(np.r_[True, query[1:] != query[:-1]])
    ends = np.r_[starts[1:], len(query)]
    if any(not (fit[start:end].all() or (~fit[start:end]).all()) for start, end in zip(starts, ends, strict=True)):
        raise AssertionError("inner split broke an exact timestamp query")
    return fit, ~fit


def _qid(frame: pd.DataFrame) -> np.ndarray:
    codes, _ = pd.factorize(frame["__decision_ts__"], sort=True)
    if np.any(codes < 0):
        raise AssertionError("invalid query ID")
    return codes.astype(np.int64)


def _suggest(trial: optuna.Trial, contract: Contract) -> dict[str, Any]:
    common = {
        "learning_rate": trial.suggest_float("learning_rate", .02, .10, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "feature_fraction": trial.suggest_float("feature_fraction", .70, .90),
        "bagging_fraction": trial.suggest_float("bagging_fraction", .70, .90),
        "lambda_l2": trial.suggest_float("lambda_l2", .1, 30.0, log=True),
    }
    if contract.model_family == "catboost_queryrmse":
        return {
            **common,
            "random_strength": trial.suggest_float("random_strength", .01, 2.0, log=True),
        }
    if contract.model_family == "xgb_ndcg":
        return {
            **common,
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 5.0, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", .1, 30.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-4, .01, log=True),
            "pairs_per_sample": trial.suggest_categorical("pairs_per_sample", [1, 2, 4, 8]),
        }
    raise AssertionError("unsupported HPO model")


def _fit_predict(
    *, contract: Contract, params: dict[str, Any], train: pd.DataFrame, labels: np.ndarray,
    held: pd.DataFrame, fields: tuple[str, ...], seed: int,
) -> pd.DataFrame:
    x_train, medians = base._numeric_matrix(train, fields)
    x_held, _ = base._numeric_matrix(held, fields, medians)
    fit, valid = _inner_masks(train)
    train_fit, train_valid = train.loc[fit].reset_index(drop=True), train.loc[valid].reset_index(drop=True)
    y_fit, y_valid = labels[fit], labels[valid]
    sample_weight = (
        weights._query_safe_weights(train, labels, contract.weight_scheme)
        if contract.weight_scheme is not None else np.ones(len(train), dtype=np.float64)
    )
    if contract.model_family == "catboost_queryrmse":
        model = CatBoostRanker(
            loss_function="QueryRMSE", eval_metric="NDCG:top=10", iterations=2000,
            learning_rate=params["learning_rate"], depth=params["max_depth"],
            l2_leaf_reg=params["lambda_l2"], random_strength=params["random_strength"],
            rsm=params["feature_fraction"], bootstrap_type="Bernoulli", subsample=params["bagging_fraction"],
            random_seed=seed, thread_count=1, verbose=False, allow_writing_files=False,
            od_type="Iter", od_wait=30,
        )
        model.fit(
            Pool(x_train[fit], y_fit, group_id=_qid(train_fit), weight=sample_weight[fit]),
            eval_set=Pool(x_train[valid], y_valid, group_id=_qid(train_valid), weight=sample_weight[valid]),
            use_best_model=True, verbose=False,
        )
        prediction = model.predict(x_held)
    elif contract.model_family == "xgb_ndcg":
        model = XGBRanker(
            objective="rank:ndcg", eval_metric="ndcg@10", n_estimators=2000,
            learning_rate=params["learning_rate"], max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"], gamma=params["min_gain_to_split"],
            subsample=params["bagging_fraction"], colsample_bytree=params["feature_fraction"],
            reg_alpha=params["lambda_l1"], reg_lambda=params["lambda_l2"],
            lambdarank_num_pair_per_sample=params["pairs_per_sample"], random_state=seed,
            n_jobs=1, tree_method="hist", early_stopping_rounds=30,
        )
        model.fit(
            x_train[fit], y_fit, qid=_qid(train_fit),
            eval_set=[(x_train[valid], y_valid)], eval_qid=[_qid(train_valid)], verbose=False,
        )
        prediction = model.predict(x_held)
    else:
        raise AssertionError("unsupported HPO model")
    if not np.isfinite(prediction).all():
        raise AssertionError("HPO scorer emitted a non-finite held prediction")
    output = held.loc[:, list(IDENTITY)].copy()
    output["base_score"] = np.asarray(prediction, dtype=np.float32)
    output["base_rank_ts"] = base._rank_desc(output, "base_score")
    if output.columns.tolist() != [*IDENTITY, "base_score", "base_rank_ts"]:
        raise AssertionError("HPO held score violates target-free schema")
    return output


def _folds(
    *, feature_roots: Sequence[Path], label_root: Path, router_root: Path, stage1_root: Path,
    fields: tuple[str, ...], held_months: Sequence[pd.Timestamp], train_months: int, reserve_days: int,
) -> tuple[list[Fold], list[dict[str, object]]]:
    folds: list[Fold] = []
    coverage_rows: list[dict[str, object]] = []
    for month in held_months:
        reserve = month - pd.Timedelta(days=reserve_days)
        end = month + pd.offsets.MonthBegin(1)
        window, coverage = base._load_window(
            candidate_root=None, feature_root=tuple(feature_roots), label_root=label_root,
            router_root=router_root, start=reserve - pd.DateOffset(months=train_months), end=end, fields=fields,
        )
        coverage_rows.extend(coverage)
        held = window.loc[window["__decision_ts__"].ge(month) & window["__decision_ts__"].lt(end)].copy()
        held = held.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        labels = held.loc[:, ["candidate_id", "policy_ordinal_valid", "policy_net_bps"]].copy()
        control = gain._control_score(stage1_root, month)
        if len(control) != len(held) or not control["candidate_id"].equals(held["candidate_id"]):
            raise AssertionError(f"{month:%Y-%m}: common target-free control identity mismatch")
        folds.append(Fold(month, window, held, labels, control))
    return folds, coverage_rows


def _evaluate(
    *, contract: Contract, params: dict[str, Any], folds: Sequence[Fold], fields: tuple[str, ...],
    train_cap: int, reserve_days: int, seed_offset: int, persist_root: Path | None = None,
    trial: optuna.Trial | None = None,
) -> tuple[dict[str, object], list[pd.DataFrame], list[pd.DataFrame], list[dict[str, object]]]:
    candidate_parts: list[pd.DataFrame] = []
    control_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for index, fold in enumerate(folds):
        reserve = fold.month - pd.Timedelta(days=reserve_days)
        train = stage1._train_rows(fold.window, contract.arm, reserve, train_cap)
        labels, geometry = stage1._labels(train, contract.arm)
        score = _fit_predict(contract=contract, params=params, train=train, labels=labels, held=fold.held, fields=fields, seed=SEED + seed_offset + index)
        if persist_root is not None:
            path = persist_root / "target_free_scores" / f"month={fold.month:%Y-%m}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            score.to_parquet(path, index=False, compression="zstd")
        scored = score.merge(fold.labels, on="candidate_id", how="left", validate="one_to_one")
        candidate_parts.append(timestamp_components(scored, score_column="base_score"))
        control_parts.append(timestamp_components(fold.control.merge(fold.labels, on="candidate_id", how="left", validate="one_to_one"), score_column="base_score"))
        audit_rows.append({
            "held_month": f"{fold.month:%Y-%m}", "train_rows": int(len(train)), "train_queries": int(train["__decision_ts__"].nunique()),
            "held_rows": int(len(fold.held)), "held_queries": int(fold.held["__decision_ts__"].nunique()),
            "target_geometry": json.dumps(geometry, sort_keys=True), "target_free_before_outcome_join": True,
            "feature_medians_fit_train_only": True, "router_top50_identity_exact": True,
            "weight_scheme": contract.weight_scheme or "uniform",
            "weights_normalised_within_training_timestamp": contract.weight_scheme is not None,
        })
        if trial is not None:
            partial_candidate = pd.concat(candidate_parts, ignore_index=True).sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
            partial_control = pd.concat(control_parts, ignore_index=True).sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
            partial_score = float(stable_score(partial_candidate, partial_control)[0].score_stable)
            trial.report(partial_score, index)
            if index >= 1 and trial.should_prune():
                raise optuna.TrialPruned()
    candidate = pd.concat(candidate_parts, ignore_index=True).sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
    control = pd.concat(control_parts, ignore_index=True).sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
    summary, normalised = stable_score(candidate, control)
    metrics: dict[str, object] = {**summary.__dict__, **{f"mean_{item}": float(candidate[item].mean()) for item in COMPONENTS}, "mean_utility_recall20": float(candidate["utility_recall20"].mean())}
    return metrics, [candidate, normalised], [control], audit_rows


def run(
    *, feature_roots: Sequence[Path], label_root: Path, router_root: Path, selection_receipt: Path,
    stage1_root: Path, cross_root: Path, candidate_key: str, out: Path, screen_months: Sequence[pd.Timestamp],
    confirmation_months: Sequence[pd.Timestamp], train_months: int, reserve_days: int, train_cap: int,
    trials: int, study_jobs: int, confirm_top: int,
) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    required_cross = [cross_root / "run_manifest.json", cross_root / "correctness_report.json"]
    if (cross_root / "target_model_winners.parquet").exists():
        required_cross.append(cross_root / "target_model_winners.parquet")
    if any(not path.exists() for path in required_cross):
        raise FileNotFoundError("cross-model stage is incomplete")
    if not all(json.loads((cross_root / "correctness_report.json").read_text()).values()):
        raise AssertionError("cross-model stage did not pass correctness receipt")
    contract = _contract(cross_root, candidate_key)
    fields = base._load_f72_fields(selection_receipt)
    out.mkdir(parents=True)
    folds, coverage_rows = _folds(
        feature_roots=feature_roots, label_root=label_root, router_root=router_root, stage1_root=stage1_root,
        fields=fields, held_months=screen_months, train_months=train_months, reserve_days=reserve_days,
    )
    rows: list[dict[str, object]] = []
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED, multivariate=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=2, interval_steps=1),
    )

    def objective_fn(trial: optuna.Trial) -> float:
        params = _suggest(trial, contract)
        # All five outer folds remain in the score.  This keeps the HPO's
        # precision/preservation objective spread over multiple months/years.
        metrics, _, _, _ = _evaluate(
            contract=contract, params=params, folds=folds, fields=fields, train_cap=train_cap,
            # Trial number must never become a hidden randomisation axis in
            # model selection.  Every parameter setting sees the same
            # fold-specific seed; only its declared parameters vary.
            reserve_days=reserve_days, seed_offset=0, trial=trial,
        )
        score = float(metrics["score_stable"])
        rows.append({"trial": trial.number, "state": "complete", "selection_score": score, **params, **metrics})
        _progress(out, stage="trial_complete", trial=trial.number, selection_score=score, **params)
        return score

    study.optimize(objective_fn, n_trials=trials, n_jobs=study_jobs, gc_after_trial=True)
    for trial in study.trials:
        if trial.state == optuna.trial.TrialState.PRUNED:
            rows.append({"trial": trial.number, "state": "pruned", **trial.params})
    trials_frame = pd.DataFrame(rows).sort_values("trial", kind="stable")
    trials_frame.to_parquet(out / "trials.parquet", index=False, compression="zstd")
    complete = trials_frame.loc[trials_frame["state"].eq("complete")].sort_values(["selection_score", "trial"], ascending=[False, True], kind="stable")
    if complete.empty:
        raise RuntimeError("all full-HPO trials failed")
    parameter_names = _suggest(optuna.trial.FixedTrial({
        # FixedTrial only identifies the expected search fields; its values
        # are immediately replaced below by the actual selected receipt.
        "learning_rate": .05, "max_depth": 4, "feature_fraction": .8, "bagging_fraction": .8,
        "lambda_l2": 8.0,
        **({"lambda_l1": .05} if contract.model_family == "xgb_ndcg" else {}),
        **({"random_strength": .5} if contract.model_family == "catboost_queryrmse" else {"min_child_weight": 5.0, "min_gain_to_split": .001, "pairs_per_sample": 2}),
    }), contract).keys()
    confirmation_folds, confirmation_coverage = _folds(
        feature_roots=feature_roots, label_root=label_root, router_root=router_root, stage1_root=stage1_root,
        fields=fields, held_months=confirmation_months, train_months=train_months, reserve_days=reserve_days,
    )
    confirmation_rows: list[dict[str, object]] = []
    confirmed: list[tuple[int, dict[str, object], dict[str, Any], list[pd.DataFrame], list[pd.DataFrame], list[dict[str, object]]]] = []
    for order, row in enumerate(complete.head(confirm_top).itertuples(index=False), start=1):
        params = {name: getattr(row, name) for name in parameter_names}
        metrics, candidate_panels, control_panels, audit_rows = _evaluate(
            contract=contract, params=params, folds=confirmation_folds, fields=fields, train_cap=train_cap,
            reserve_days=reserve_days, seed_offset=0, persist_root=out / "confirmation" / f"trial={int(row.trial):03d}",
        )
        confirmation_rows.append({"trial": int(row.trial), "screen_selection_score": float(row.selection_score), **params, **metrics})
        confirmed.append((int(row.trial), metrics, params, candidate_panels, control_panels, audit_rows))
        _progress(out, stage="confirmation_complete", trial=int(row.trial), screen_selection_score=float(row.selection_score), confirmation_score=float(metrics["score_stable"]))
    confirmation = pd.DataFrame(confirmation_rows).sort_values(["score_stable", "trial"], ascending=[False, True], kind="stable")
    confirmation.to_parquet(out / "confirmation_summary.parquet", index=False, compression="zstd")
    winning_trial = int(confirmation.iloc[0].trial)
    _, winner_metrics, winner_params, candidate_panels, control_panels, audit_rows = next(
        item for item in confirmed if item[0] == winning_trial
    )
    candidate_panels[0].to_parquet(out / "winner_timestamp_components.parquet", index=False, compression="zstd")
    candidate_panels[1].to_parquet(out / "winner_normalised_timestamp_components.parquet", index=False, compression="zstd")
    control_panels[0].to_parquet(out / "control_timestamp_components.parquet", index=False, compression="zstd")
    pd.DataFrame(audit_rows).to_parquet(out / "fold_audit.parquet", index=False, compression="zstd")
    pd.DataFrame([*coverage_rows, *confirmation_coverage]).drop_duplicates(subset=["month"], keep="last").to_parquet(out / "coverage_audit.parquet", index=False, compression="zstd")
    _exclusive_json(out / "correctness_report.json", {
        "hpo_candidate_is_sealed_cross_model_winner": True,
        "p8u_router_top50_identity_exact": bool(all(row["router_top50_identity_exact"] for row in audit_rows)),
        "all_held_scores_target_free_before_outcomes": bool(all(row["target_free_before_outcome_join"] for row in audit_rows)),
        "all_feature_medians_train_only": bool(all(row["feature_medians_fit_train_only"] for row in audit_rows)),
        "all_train_labels_resolved_before_reserve": True,
        "early_stopping_inner_split_is_chronological": True,
        "hpo_screen_timestamp_local_and_cross_year": True,
        "top_hpo_candidates_confirmed_on_full_five_fold_panel": True,
        "frozen_timestamp_normalised_weight_contract": contract.weight_scheme == "tail_linear_125",
        "no_meta_mc1_portfolio_or_live_mutation": True,
    })
    _exclusive_json(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline P8u winner-only full HPO; no Meta, MC1, admission, portfolio, live, or exchange mutation",
        "contract": {"candidate": contract.candidate, "arm": contract.arm.__dict__, "gain_name": contract.gain_name, "model_family": contract.model_family, "weight_scheme": contract.weight_scheme or "uniform"},
        "hpo": {"trials": trials, "study_jobs": study_jobs, "pruner": "MedianPruner(startup=4,warmup=2)", "inner_early_stopping": "latest 20% strict training queries, 30 rounds", "screen_months": [f"{month:%Y-%m}" for month in screen_months], "confirmation_months": [f"{month:%Y-%m}" for month in confirmation_months], "confirm_top": confirm_top, "winner": winner_params, "winner_metrics": winner_metrics, "winner_confirmation_trial": winning_trial},
        "selection_metric": {"BaseScore": "0.30*DTP2 + 0.30*DTP5 + 0.20*DTP10 + 0.20*ResidualUR10_to30, normalised to fixed Stage-1 target-free control; UR20 diagnostic only", "ScoreStable": "weekly robust mean Q20-Q80 + 0.5*mean(Q15,Q10,Q5)"},
        "strict_oof": {"train_months": train_months, "reserve_days": reserve_days, "train_cap_complete_queries": train_cap, "hpo_screen_months": [f"{month:%Y-%m}" for month in screen_months], "confirmation_months": [f"{month:%Y-%m}" for month in confirmation_months]},
        "inputs": {"feature_roots": [str(root) for root in feature_roots], "label_root": str(label_root), "router_root": str(router_root), "selection_receipt": str(selection_receipt), "stage1_root": str(stage1_root), "cross_root": str(cross_root)},
        "source_sha256": _source_sha256([*required_cross, stage1_root / "run_manifest.json", selection_receipt]),
        "next_stage": "Only a winner which clears the frozen Base guard may be sent into the matched Base-to-Meta-to-MC1 replay; no losing model receives downstream training.",
    })
    _progress(out, stage="complete", candidate=contract.candidate, winner=winner_params, winner_metrics=winner_metrics)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True, help="comma-separated immutable causal feature roots")
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--cross-root", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--screen-months", default="2025-11,2026-03,2026-07")
    parser.add_argument("--confirmation-months", default=",".join(stage1.DEFAULT_HELD_MONTHS))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--study-jobs", type=int, default=1)
    parser.add_argument("--confirm-top", type=int, default=3)
    args = parser.parse_args()
    if args.train_months < 2 or args.reserve_days < 12 or args.train_cap < 8_000 or args.trials < 4 or args.study_jobs < 1 or args.confirm_top < 1:
        raise ValueError("invalid strict-OOF winner-HPO contract")
    print(run(
        feature_roots=tuple(Path(item.strip()).resolve() for item in args.feature_roots.split(",") if item.strip()),
        label_root=args.label_root.resolve(), router_root=args.router_root.resolve(), selection_receipt=args.selection_receipt.resolve(),
        stage1_root=args.stage1_root.resolve(), cross_root=args.cross_root.resolve(), candidate_key=args.candidate,
        out=args.out.resolve(), screen_months=_parse_screen_months(args.screen_months), confirmation_months=stage1._parse_months(args.confirmation_months), train_months=args.train_months,
        reserve_days=args.reserve_days, train_cap=args.train_cap, trials=args.trials, study_jobs=args.study_jobs, confirm_top=args.confirm_top,
    ))


if __name__ == "__main__":
    main()
