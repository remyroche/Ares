#!/usr/bin/env python3
"""Query-safe target-weight screen for a frozen raw-bps P8u Base contract.

All candidates retain the frozen Router50 identity, raw-bps ordinal target,
declared causal feature contract, CatBoost model family, and fixed HPO parameters.  A weight
can only reallocate emphasis *within* a resolved training timestamp: query
mean is one, values are clipped to [0.5, 2.0], and held scoring is target-free.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

import run_strict_r3_p8u_precision_preservation_group_mda_beam_v1 as beam
import run_strict_r3_p8u_precision_preservation_winner_hpo_v1 as hpo
import run_strict_r3_p8u_precision_preservation_loss_funnel_v1 as gain
import run_strict_r3_router_single_base_prescreen_v1 as base
from strict_r3_p8u_precision_preservation_metric import COMPONENTS, stable_score, timestamp_components


SCHEMA = "strict_r3_p8u_precision_preservation_weight_funnel_v1"
SEED = 1729
SCHEMES = ("uniform", "positive_125", "positive_150", "tail_linear_125", "tail_linear_250", "tail_convex_500")
MIN_FEATURES = 16
MAX_FEATURES = 160


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _months(text: str, *, allow_early_history_extension: bool = False) -> tuple[pd.Timestamp, ...]:
    result = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in text.split(",") if item.strip())
    minimum_months = 1 if allow_early_history_extension else 3
    if len(result) < minimum_months or tuple(sorted(result)) != result:
        raise ValueError(
            "early history extension needs at least one increasing held month"
            if allow_early_history_extension
            else "need at least three increasing held months"
        )
    if allow_early_history_extension:
        expected = tuple(pd.date_range(result[0], result[-1], freq="MS", tz="UTC"))
        if result != expected:
            raise ValueError("early history extension must use one contiguous monthly block")
    else:
        span = (result[-1].year - result[0].year) * 12 + result[-1].month - result[0].month
        if len({item.year for item in result}) < 2 or span < 8:
            raise ValueError("held months must remain cross-year and span at least eight months")
    return result


def _schemes(text: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in text.split(",") if item.strip())
    if not result or len(set(result)) != len(result) or any(item not in SCHEMES for item in result):
        raise ValueError(f"unknown/duplicate schemes; allowed {SCHEMES}")
    return result


def _load_fields(selection_json: Path) -> tuple[str, ...]:
    """Load a frozen Base feature contract without changing its semantics."""
    payload = json.loads(selection_json.read_text())
    fields = tuple(payload.get("selected_features", payload.get("features", ())))
    if not (MIN_FEATURES <= len(fields) <= MAX_FEATURES) or len(set(fields)) != len(fields):
        raise AssertionError(
            f"feature contract must contain {MIN_FEATURES}..{MAX_FEATURES} unique causal fields"
        )
    return fields


def _raw_weight(labels: np.ndarray, scheme: str) -> np.ndarray:
    grade = np.asarray(labels, dtype=float)
    if scheme == "uniform":
        return np.ones(len(grade), dtype=np.float64)
    if scheme == "positive_125":
        return 1.0 + .25 * (grade >= 3.0)
    if scheme == "positive_150":
        return 1.0 + .50 * (grade >= 3.0)
    if scheme == "tail_linear_125":
        return 1.0 + .125 * grade
    if scheme == "tail_linear_250":
        return 1.0 + .25 * grade
    if scheme == "tail_convex_500":
        return 1.0 + .50 * (grade / 5.0) ** 2
    raise AssertionError("unreachable scheme")


def _query_safe_weights(train: pd.DataFrame, labels: np.ndarray, scheme: str) -> np.ndarray:
    """Normalise each resolved timestamp after bounded target emphasis."""
    raw = _raw_weight(labels, scheme)
    result = np.empty(len(raw), dtype=np.float64)
    for _, index in train.groupby("__decision_ts__", sort=False).groups.items():
        position = np.asarray(list(index), dtype=np.int64)
        values = raw[position]
        # Solve mean(clip(values / scale, .5, 2.0)) == 1.0.  A simple second
        # division after clipping can violate the upper cap for concentrated
        # labels; this monotone projection preserves both constraints exactly.
        lower, upper = 1e-12, max(float(values.max()) * 4.0, 1.0)
        for _ in range(64):
            scale = .5 * (lower + upper)
            projected = np.clip(values / scale, .5, 2.0)
            if float(projected.mean()) > 1.0:
                lower = scale
            else:
                upper = scale
        values = np.clip(values / upper, .5, 2.0)
        if (values < .5 - 1e-10).any() or (values > 2.0 + 1e-10).any() or not np.isclose(values.mean(), 1.0):
            raise AssertionError("timestamp-safe weight contract violated")
        result[position] = values
    return result.astype(np.float32)


def _fit_predict(
    *, train: pd.DataFrame, labels: np.ndarray, held: pd.DataFrame, fields: tuple[str, ...],
    params: dict[str, float], scheme: str, seed: int,
) -> pd.DataFrame:
    x_train, medians = base._numeric_matrix(train, fields)
    x_held, _ = base._numeric_matrix(held, fields, medians)
    fit, valid = hpo._inner_masks(train)
    weights = _query_safe_weights(train, labels, scheme)
    fit_frame, valid_frame = train.loc[fit].reset_index(drop=True), train.loc[valid].reset_index(drop=True)
    model = CatBoostRanker(
        loss_function="QueryRMSE", eval_metric="NDCG:top=10", iterations=2000,
        learning_rate=params["learning_rate"], depth=int(params["max_depth"]), l2_leaf_reg=params["lambda_l2"],
        random_strength=params["random_strength"], rsm=params["feature_fraction"], bootstrap_type="Bernoulli",
        subsample=params["bagging_fraction"], random_seed=seed, thread_count=1, verbose=False,
        allow_writing_files=False, od_type="Iter", od_wait=30,
    )
    model.fit(
        Pool(x_train[fit], labels[fit], group_id=hpo._qid(fit_frame), weight=weights[fit]),
        eval_set=Pool(x_train[valid], labels[valid], group_id=hpo._qid(valid_frame), weight=weights[valid]),
        use_best_model=True, verbose=False,
    )
    prediction = np.asarray(model.predict(x_held), dtype=np.float32)
    # CatBoost's native buffers can otherwise accumulate across a sequential
    # scheme screen.  The input panel is deliberately shared, but each fitted
    # booster and its matrices must be released before the next fold.
    del model, x_train, x_held
    gc.collect()
    output = held.loc[:, list(base.IDENTITY)].copy()
    output["base_score"] = prediction
    output["base_rank_ts"] = base._rank_desc(output, "base_score")
    if not np.isfinite(output["base_score"]).all():
        raise AssertionError("non-finite target-free score")
    return output


def _evaluate(
    *, folds: Sequence[hpo.Fold], fields: tuple[str, ...], arm, params: dict[str, float], scheme: str,
    train_cap: int, reserve_days: int, persist_root: Path | None,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    candidates: list[pd.DataFrame] = []
    controls: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for index, fold in enumerate(folds):
        reserve = fold.month - pd.Timedelta(days=reserve_days)
        train = hpo.stage1._train_rows(fold.window, arm, reserve, train_cap)
        labels, geometry = hpo.stage1._labels(train, arm)
        score = _fit_predict(train=train, labels=labels, held=fold.held, fields=fields, params=params, scheme=scheme, seed=SEED + index)
        if persist_root is not None:
            path = persist_root / "target_free_scores" / f"month={fold.month:%Y-%m}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            score.to_parquet(path, index=False, compression="zstd")
        scored = score.merge(fold.labels, on="candidate_id", how="left", validate="one_to_one")
        candidates.append(timestamp_components(scored, score_column="base_score"))
        controls.append(timestamp_components(fold.control.merge(fold.labels, on="candidate_id", how="left", validate="one_to_one"), score_column="base_score"))
        weights = _query_safe_weights(train, labels, scheme)
        audit.append({
            "held_month": f"{fold.month:%Y-%m}", "scheme": scheme, "train_rows": len(train),
            "train_queries": train["__decision_ts__"].nunique(), "weight_min": float(weights.min()),
            "weight_max": float(weights.max()), "weight_query_mean_min": float(pd.Series(weights, index=train["__decision_ts__"]).groupby(level=0).mean().min()),
            "weight_query_mean_max": float(pd.Series(weights, index=train["__decision_ts__"]).groupby(level=0).mean().max()),
            "target_geometry": json.dumps(geometry, sort_keys=True), "target_free_before_outcome_join": True,
        })
        del score, scored, train, labels, weights
        gc.collect()
    candidate = pd.concat(candidates, ignore_index=True).sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
    control = pd.concat(controls, ignore_index=True).sort_values("__decision_ts__", kind="stable").reset_index(drop=True)
    summary, _ = stable_score(candidate, control)
    metrics = {**summary.__dict__, **{f"mean_{key}": float(candidate[key].mean()) for key in COMPONENTS}, "mean_utility_recall20": float(candidate["utility_recall20"].mean())}
    return metrics, audit


def run(
    *, feature_roots: Sequence[Path], label_root: Path, router_root: Path, stage1_root: Path, hpo_root: Path,
    selection_json: Path, out: Path, months: Sequence[pd.Timestamp], schemes: Sequence[str], train_months: int,
    reserve_days: int, train_cap: int,
) -> Path:
    if out.exists():
        raise FileExistsError("immutable output exists")
    arm, params = beam._load_contract(hpo_root)
    fields = _load_fields(selection_json)
    out.mkdir(parents=True)
    _once(out / "preflight.json", {"schema": SCHEMA, "scope": "offline Base weighting screen only", "schemes": list(schemes), "months": [f"{item:%Y-%m}" for item in months], "feature_count": len(fields)})
    folds, coverage = hpo._folds(feature_roots=feature_roots, label_root=label_root, router_root=router_root, stage1_root=stage1_root, fields=fields, held_months=months, train_months=train_months, reserve_days=reserve_days)
    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for scheme in schemes:
        _progress(out, stage="scheme_start", scheme=scheme)
        root = out / f"scheme={scheme}"
        metrics, audit = _evaluate(folds=folds, fields=fields, arm=arm, params=params, scheme=scheme, train_cap=train_cap, reserve_days=reserve_days, persist_root=root)
        rows.append({"scheme": scheme, **metrics})
        audit_rows.extend(audit)
        _progress(out, stage="scheme_complete", scheme=scheme, score_stable=float(metrics["score_stable"]))
    result = pd.DataFrame(rows).sort_values(["score_stable", "mean_dtp2_bps"], ascending=False, kind="stable").reset_index(drop=True)
    result.to_parquet(out / "scheme_metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(audit_rows).to_parquet(out / "weight_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(coverage).to_parquet(out / "feature_coverage.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "frozen_declared_feature_contract": True, "p8u_router_top50_identity_exact": True,
        "all_train_labels_resolved_before_reserve": True, "weights_normalised_within_training_timestamp": True,
        "weight_clip_050_200": True, "held_scores_target_free_before_outcome_join": True,
        "feature_medians_fit_train_only": True, "no_meta_mc1_admission_portfolio_live_or_exchange_mutation": True,
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "offline query-safe Base weighting screen only", "schemes": list(schemes),
        "fixed_hpo_root": str(hpo_root), "selection_json": str(selection_json), "feature_count": len(fields),
        "strict_oof": {"months": [f"{item:%Y-%m}" for item in months], "train_months": train_months, "reserve_days": reserve_days, "train_cap_complete_queries": train_cap},
        "results": result.to_dict("records"), "next_stage": "Only a weighting scheme that improves the fixed Base control may receive five-fold confirmation; Meta/MC1 remain untouched.",
    })
    return out


def _target_free_held(
    *, feature_roots: Sequence[Path], router_root: Path, month: pd.Timestamp,
    fields: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Read a held Router50 panel without opening any held outcome source.

    ``base._load_window`` intentionally serves supervised Base experiments and
    therefore joins the label panel for every calendar month in its window.
    This history-only producer is different: its held score must be entirely
    target-free.  Read the same immutable feature and Router receipts directly
    and retain the exact timestamp-local Router50 identity, while leaving the
    supervised helper in its default mode for the preceding training window.
    """
    feature_path = base._feature_path(feature_roots, month)
    router_path = base._router_path(router_root, month)
    features = pd.read_parquet(feature_path, columns=[*base.IDENTITY, *fields]).copy()
    router = pd.read_parquet(
        router_path, columns=[*base.IDENTITY, "router_primary_rank"],
    ).copy()
    for frame in (features, router):
        frame["__decision_ts__"] = pd.to_datetime(
            frame["__decision_ts__"], utc=True, errors="raise",
        )
        if frame["candidate_id"].duplicated().any():
            raise AssertionError(f"{month:%Y-%m}: duplicate target-free candidate identity")
    expected = base._top_half_identities(router)
    held = expected.merge(features, on=list(base.IDENTITY), how="left", validate="one_to_one")
    if len(held) != len(expected) or held.loc[:, list(fields)].columns.tolist() != list(fields):
        raise AssertionError(f"{month:%Y-%m}: target-free held feature join failed")
    if held.loc[:, list(fields)].isna().all(axis=None):
        raise AssertionError(f"{month:%Y-%m}: target-free held feature coverage is empty")
    held = held.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    return held, {
        "month": f"{month:%Y-%m}",
        "candidate_rows": int(len(held)),
        "router_top50_identity_exact": True,
        "feature_columns": int(len(fields)),
        "feature_complete_rows": int(held.loc[:, list(fields)].notna().all(axis=1).sum()),
        "held_labels_opened": False,
    }


def materialize_history(
    *, feature_roots: Sequence[Path], label_root: Path, router_root: Path, hpo_root: Path,
    selection_json: Path, out: Path, months: Sequence[pd.Timestamp], scheme: str, train_months: int,
    reserve_days: int, train_cap: int, early_history_extension: bool = False,
) -> Path:
    """Write an OOF score history without requiring held outcome/control data.

    This is used solely to provide a target-free Base ledger to later Meta
    training.  The training side remains identical to the screen: only labels
    resolved before the 28-day reserve can enter the fit.  Unlike the screen,
    the held rows never join a policy label or fixed Stage-1 control.
    """
    if out.exists():
        raise FileExistsError("immutable output exists")
    arm, params = beam._load_contract(hpo_root)
    fields = _load_fields(selection_json)
    out.mkdir(parents=True)
    _once(out / "preflight.json", {"schema": SCHEMA, "scope": "offline strict-prequential target-free Base history only", "scheme": scheme, "months": [f"{item:%Y-%m}" for item in months], "feature_count": len(fields)})
    coverage_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    for index, month in enumerate(months):
        reserve = month - pd.Timedelta(days=reserve_days)
        end = month + pd.offsets.MonthBegin(1)
        # Training keeps the original supervised loader and strict availability
        # filter.  Crucially, it ends at the reserve: the held calendar month is
        # materialised through ``_target_free_held`` below and never causes an
        # outcome/label file to be opened.
        window, coverage = base._load_window(
            candidate_root=None, feature_root=feature_roots, label_root=label_root, router_root=router_root,
            start=reserve - pd.DateOffset(months=train_months), end=reserve, fields=fields,
        )
        held, held_coverage = _target_free_held(
            feature_roots=feature_roots, router_root=router_root, month=month, fields=fields,
        )
        coverage_rows.extend([*coverage, held_coverage])
        if held.empty or held["candidate_id"].duplicated().any():
            raise AssertionError(f"{month:%Y-%m}: invalid target-free held identity")
        train = hpo.stage1._train_rows(window, arm, reserve, train_cap)
        labels, geometry = hpo.stage1._labels(train, arm)
        score = _fit_predict(train=train, labels=labels, held=held, fields=fields, params=params, scheme=scheme, seed=SEED + index)
        path = out / f"scheme={scheme}" / "target_free_scores" / f"month={month:%Y-%m}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        score.to_parquet(path, index=False, compression="zstd")
        weights = _query_safe_weights(train, labels, scheme)
        audit_rows.append({
            "held_month": f"{month:%Y-%m}", "held_rows": len(held), "held_queries": held["__decision_ts__"].nunique(),
            "train_rows": len(train), "train_queries": train["__decision_ts__"].nunique(), "scheme": scheme,
            "weight_min": float(weights.min()), "weight_max": float(weights.max()),
            "target_geometry": json.dumps(geometry, sort_keys=True), "held_score_target_free": True,
            "train_labels_resolved_before_reserve": True,
        })
        _progress(out, stage="month_complete", held_month=f"{month:%Y-%m}", rows=len(held))
        del window, held, train, labels, score, weights
        gc.collect()
    pd.DataFrame(coverage_rows).to_parquet(out / "feature_coverage.parquet", index=False, compression="zstd")
    pd.DataFrame(audit_rows).to_parquet(out / "history_fold_audit.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "frozen_declared_feature_contract": True, "p8u_router_top50_identity_exact": True,
        "all_train_labels_resolved_before_reserve": True, "weights_normalised_within_training_timestamp": True,
        "weight_clip_050_200": True, "held_scores_target_free_without_policy_join": True,
        "feature_medians_fit_train_only": True, "no_meta_mc1_admission_portfolio_live_or_exchange_mutation": True,
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA, "scope": "offline strict-prequential target-free Base-score history only",
        "scheme": scheme, "feature_count": len(fields), "hpo_root": str(hpo_root),
        "strict_oof": {"months": [f"{item:%Y-%m}" for item in months], "train_months": train_months, "reserve_days": reserve_days, "train_cap_complete_queries": train_cap},
        "historical_startup_extension": bool(early_history_extension),
        "next_stage": "This target-free history may feed a separately trained Meta evaluation; no outcome-joined selection occurred here.",
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument(
        "--stage1-root",
        type=Path,
        help=(
            "required only for the outcome-joined weight screen; history-only "
            "target-free score materialisation never opens this control artifact"
        ),
    )
    parser.add_argument("--hpo-root", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", default="2025-11,2026-03,2026-07")
    parser.add_argument("--schemes", default=",".join(SCHEMES))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--history-only", action="store_true", help="materialise only strict-prequential target-free monthly scores; never join held outcomes or Stage-1 controls")
    parser.add_argument(
        "--early-history-extension", action="store_true",
        help=("permit one contiguous short historical OOS block only with --history-only; "
              "this is for causal Meta warm-up and is not a model-selection result"),
    )
    args = parser.parse_args()
    schemes = _schemes(args.schemes)
    if args.early_history_extension and not args.history_only:
        raise ValueError("--early-history-extension is valid only with --history-only")
    if not args.history_only and args.stage1_root is None:
        parser.error("--stage1-root is required unless --history-only is used")
    if args.history_only:
        if len(schemes) != 1:
            raise ValueError("history-only mode requires exactly one frozen scheme")
        result = materialize_history(feature_roots=tuple(Path(item.strip()).resolve() for item in args.feature_roots.split(",") if item.strip()), label_root=args.label_root.resolve(), router_root=args.router_root.resolve(), hpo_root=args.hpo_root.resolve(), selection_json=args.selection_json.resolve(), out=args.out.resolve(), months=_months(args.months, allow_early_history_extension=args.early_history_extension), scheme=schemes[0], train_months=args.train_months, reserve_days=args.reserve_days, train_cap=args.train_cap, early_history_extension=args.early_history_extension)
    else:
        result = run(feature_roots=tuple(Path(item.strip()).resolve() for item in args.feature_roots.split(",") if item.strip()), label_root=args.label_root.resolve(), router_root=args.router_root.resolve(), stage1_root=args.stage1_root.resolve(), hpo_root=args.hpo_root.resolve(), selection_json=args.selection_json.resolve(), out=args.out.resolve(), months=_months(args.months), schemes=schemes, train_months=args.train_months, reserve_days=args.reserve_days, train_cap=args.train_cap)
    print(result)


if __name__ == "__main__":
    main()
