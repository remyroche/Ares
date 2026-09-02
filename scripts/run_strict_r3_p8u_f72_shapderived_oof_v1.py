#!/usr/bin/env python3
"""Strict-OOF F72 SHAP-derived feature audit.

This is a deliberately narrow, offline challenger built around the v6
Router50 -> F72 -> Under F120 research control.  It does *not* rescreen raw
causal inputs and it does not alter Under, MC1, admission, the portfolio, or
any live artifact.

For every outer month the exact F72 CatBoost contract is fit only on labels
resolved before the 28-day reserve.  It then writes a target-free held receipt
containing F72 score/rank and per-row SHAP-derived explanations.  Only after
every such receipt is immutable do we join canonical policy outcomes to assess
the *new SHAP-derived fields*.  CMI and IC are intentionally never run on the
raw 72 F72 inputs in this producer.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.feature_selection import mutual_info_regression

import run_strict_r3_p8u_precision_preservation_group_mda_beam_v1 as beam
import run_strict_r3_p8u_precision_preservation_weight_funnel_v1 as weights
import run_strict_r3_p8u_precision_preservation_winner_hpo_v1 as hpo
import run_strict_r3_p8u_precision_preservation_screen_v1 as stage1
import run_strict_r3_router_single_base_prescreen_v1 as base


SCHEMA = "strict_r3_p8u_f72_shapderived_oof_v1"
SEED = 1729
IDENTITY = tuple(base.IDENTITY)
DEFAULT_MONTHS = "2025-11,2026-03,2026-07"


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _hash(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _months(text: str, *, prehistory_extension: bool = False) -> tuple[pd.Timestamp, ...]:
    result = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in text.split(",") if item.strip())
    if len(result) < 3 or tuple(sorted(result)) != result:
        raise ValueError("need at least three ordered strict-OOF held months")
    span = (result[-1].year - result[0].year) * 12 + result[-1].month - result[0].month
    if not prehistory_extension and (len({item.year for item in result}) < 2 or span < 8):
        raise ValueError("held months must span at least eight months and two calendar years")
    if prehistory_extension:
        expected = tuple(pd.date_range(result[0], result[-1], freq="MS", tz="UTC"))
        if result != expected:
            raise ValueError("prehistory extension must cover every contiguous calendar month")
    return result


def _canonical_month_seed(month: pd.Timestamp, origin: pd.Timestamp, base_seed: int = SEED) -> int:
    """Match an uninterrupted canonical F72 history's monthly seed index.

    The parent ledger increments CatBoost's seed for every calendar month in
    its continuous history.  A sparse audit must therefore retain that
    absolute index rather than restart it at zero for its selected folds.
    """
    offset = (month.year - origin.year) * 12 + month.month - origin.month
    if offset < 0:
        raise ValueError(f"F72 canonical seed schedule begins in {origin:%Y-%m}")
    return int(base_seed) + int(offset)


def _fit_f72_with_shap(
    *, train: pd.DataFrame, labels: np.ndarray, held: pd.DataFrame,
    fields: tuple[str, ...], params: dict[str, float], seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fit the frozen F72 learner and return target-free score/explanations."""
    x_train, medians = base._numeric_matrix(train, fields)
    x_held, _ = base._numeric_matrix(held, fields, medians)
    fit, valid = hpo._inner_masks(train)
    weights_by_row = weights._query_safe_weights(train, labels, "tail_linear_125")
    train_fit = train.loc[fit].reset_index(drop=True)
    train_valid = train.loc[valid].reset_index(drop=True)
    model = CatBoostRanker(
        loss_function="QueryRMSE", eval_metric="NDCG:top=10", iterations=2000,
        learning_rate=float(params["learning_rate"]), depth=int(params["max_depth"]),
        l2_leaf_reg=float(params["lambda_l2"]), random_strength=float(params["random_strength"]),
        rsm=float(params["feature_fraction"]), bootstrap_type="Bernoulli",
        subsample=float(params["bagging_fraction"]), random_seed=int(seed), thread_count=1,
        verbose=False, allow_writing_files=False, od_type="Iter", od_wait=30,
    )
    model.fit(
        Pool(x_train[fit], labels[fit], group_id=hpo._qid(train_fit), weight=weights_by_row[fit]),
        eval_set=Pool(x_train[valid], labels[valid], group_id=hpo._qid(train_valid), weight=weights_by_row[valid]),
        use_best_model=True, verbose=False,
    )
    score = np.asarray(model.predict(x_held), dtype=np.float32)
    shap_values = np.asarray(
        model.get_feature_importance(Pool(x_held), type="ShapValues", prettified=False), dtype=np.float32,
    )
    if shap_values.shape != (len(held), len(fields) + 1):
        raise AssertionError(f"unexpected CatBoost SHAP shape {shap_values.shape}")
    contribution = shap_values[:, :-1]
    reconstructed = contribution.sum(axis=1) + shap_values[:, -1]
    if not np.allclose(reconstructed, score, rtol=3e-4, atol=3e-4):
        raise AssertionError("F72 SHAP values do not reconstruct the held prediction")
    result = held.loc[:, list(IDENTITY)].copy()
    result["f72_base_score"] = score
    # The canonical helper already returns a NumPy array (rather than a
    # Series), so preserve it directly without a second conversion.
    result["f72_base_rank_ts"] = np.asarray(
        base._rank_desc(result.rename(columns={"f72_base_score": "base_score"}), "base_score"), dtype=np.float32,
    )
    # Exact aliases make the strict prequential ledger consumable by an
    # isolated Under challenger without translating the canonical F72 score
    # coordinate.  The explicit ``f72_*`` copies remain for audit readability.
    result["base_score"] = result["f72_base_score"]
    result["base_rank_ts"] = result["f72_base_rank_ts"]
    absolute = np.abs(contribution)
    total = absolute.sum(axis=1)
    probability = np.divide(absolute, total[:, None], out=np.zeros_like(absolute), where=total[:, None] > 1e-12)
    order = np.sort(absolute, axis=1)
    result["shap_f72_abs_total"] = total.astype(np.float32)
    result["shap_f72_positive_total"] = np.maximum(contribution, 0).sum(axis=1).astype(np.float32)
    result["shap_f72_negative_abs_total"] = np.maximum(-contribution, 0).sum(axis=1).astype(np.float32)
    result["shap_f72_signed_balance"] = (result["shap_f72_positive_total"] - result["shap_f72_negative_abs_total"]).astype(np.float32)
    result["shap_f72_top1_share"] = np.divide(order[:, -1], total, out=np.zeros(len(total), dtype=np.float32), where=total > 1e-12)
    result["shap_f72_top3_share"] = np.divide(order[:, -min(3, len(fields)):].sum(axis=1), total, out=np.zeros(len(total), dtype=np.float32), where=total > 1e-12)
    entropy = -(np.where(probability > 0, probability * np.log(np.maximum(probability, 1e-12)), 0.0)).sum(axis=1)
    result["shap_f72_entropy"] = (entropy / math.log(len(fields))).astype(np.float32)
    for index, field in enumerate(fields):
        result[f"shap_f72_contrib__{field}"] = contribution[:, index]
    if not np.isfinite(result.drop(columns=list(IDENTITY)).to_numpy(np.float32)).all():
        raise AssertionError("non-finite SHAP-derived value")
    audit = {
        "train_rows": int(len(train)), "held_rows": int(len(held)), "trees": int(model.tree_count_),
        "weight_min": float(weights_by_row.min()), "weight_max": float(weights_by_row.max()),
        "shap_reconstruction_max_abs_error": float(np.abs(reconstructed - score).max()),
    }
    del model, x_train, x_held, contribution, shap_values, absolute, probability
    gc.collect()
    return result, audit


def _labels(label_root: Path, month: pd.Timestamp) -> pd.DataFrame:
    path = label_root / f"month={month:%Y-%m}" / "target_labels.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    result = pd.read_parquet(path, columns=["candidate_id", "policy_ordinal_valid", "policy_net_bps"])
    if result.candidate_id.duplicated().any():
        raise AssertionError(f"{month:%Y-%m}: duplicate policy label identity")
    result["policy_net_bps"] = pd.to_numeric(result["policy_net_bps"], errors="coerce")
    result["policy_ordinal_valid"] = result["policy_ordinal_valid"].fillna(False).astype(bool)
    return result


def _timestamp_ic(frame: pd.DataFrame, field: str) -> float:
    values: list[float] = []
    for _, part in frame.loc[:, ["__decision_ts__", field, "policy_net_bps"]].groupby("__decision_ts__", sort=False):
        if len(part) < 8 or part[field].nunique() < 3 or part.policy_net_bps.nunique() < 3:
            continue
        value = float(part[field].rank().corr(part.policy_net_bps.rank()))
        if np.isfinite(value):
            values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _timestamp_top10(frame: pd.DataFrame, field: str, ascending: bool) -> float:
    selected: list[float] = []
    for _, part in frame.loc[:, ["__decision_ts__", field, "policy_net_bps"]].groupby("__decision_ts__", sort=False):
        size = max(1, int(math.ceil(.10 * len(part))))
        selected.extend(part.sort_values([field], ascending=ascending, kind="stable").head(size).policy_net_bps.tolist())
    return float(np.mean(selected)) if selected else float("nan")


def _conditional_mi(frame: pd.DataFrame, field: str, seed: int) -> float:
    """Binned CMI proxy, conditional on the canonical Base-rank coordinate."""
    work = frame.loc[:, [field, "f72_base_rank_ts", "policy_net_bps"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(work) < 2_000 or work[field].nunique() < 10:
        return float("nan")
    # Global deciles of the rank coordinate are a causal conditioning surface;
    # score/outcome residualisation occurs only after held receipt persistence.
    work["__band__"] = pd.qcut(work.f72_base_rank_ts.rank(method="first"), q=min(10, work.f72_base_rank_ts.nunique()), duplicates="drop")
    work["__x__"] = work[field] - work.groupby("__band__", observed=True)[field].transform("median")
    work["__y__"] = work.policy_net_bps - work.groupby("__band__", observed=True).policy_net_bps.transform("median")
    work = work.loc[work.__x__.notna() & work.__y__.notna()].copy()
    if len(work) > 8_000:
        rng = np.random.default_rng(seed)
        index = rng.choice(len(work), size=8_000, replace=False)
        work = work.iloc[np.sort(index)]
    value = mutual_info_regression(work[["__x__"]].to_numpy(float), work.__y__.to_numpy(float), random_state=seed, n_neighbors=5)
    return float(value[0])


def _metrics_for_month(frame: pd.DataFrame, month: pd.Timestamp) -> pd.DataFrame:
    fields = [column for column in frame.columns if column.startswith("shap_f72_")]
    rows: list[dict[str, object]] = []
    for index, field in enumerate(fields):
        high = _timestamp_top10(frame, field, ascending=False)
        low = _timestamp_top10(frame, field, ascending=True)
        rows.append({
            "held_month": f"{month:%Y-%m}", "feature": field,
            "conditional_mi_binned_given_f72_rank": _conditional_mi(frame, field, SEED + index + month.month * 1000),
            "timestamp_rank_ic": _timestamp_ic(frame, field),
            "ts_top10_ev_high": high, "ts_top10_ev_low": low,
            "ts_top10_ev_best_orientation": max(high, low),
            "best_orientation": "high" if high >= low else "low",
            "rows": int(len(frame)), "timestamps": int(frame.__decision_ts__.nunique()),
        })
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    months = _months(args.held_months, prehistory_extension=bool(args.prehistory_extension))
    seed_origin = pd.Timestamp(f"{args.canonical_seed_origin}-01", tz="UTC")
    seed_base = int(args.canonical_seed_base)
    if seed_origin.day != 1 or not all(month >= seed_origin for month in months):
        raise ValueError("--canonical-seed-origin must be no later than every held month")
    fields = weights._load_fields(args.selection_json)
    arm, params = beam._load_contract(args.hpo_root)
    if arm.key != "raw_bps__equal_width6" or len(fields) != 72:
        raise AssertionError("this audit is fixed to the canonical F72 raw-bps / 72-field contract")
    args.out.mkdir(parents=True)
    _once(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline strict-OOF F72 SHAP-derived diagnostic; no Under/MC1/admission/portfolio/live mutation",
        "canonical_parent": "Router50 -> F72 Raw-bps CatBoost QueryRMSE tail_linear_125",
        "held_months": [f"{month:%Y-%m}" for month in months], "feature_count": len(fields),
        "canonical_seed_origin": f"{seed_origin:%Y-%m}", "canonical_seed_base": seed_base,
        "prehistory_extension": bool(args.prehistory_extension),
        "selection_authority": False,
        "feature_contract_sha256": _hash(fields), "raw_feature_cmi_or_ic": False,
        "derived_feature_cmi_or_ic": True,
    })
    score_paths: list[tuple[pd.Timestamp, Path]] = []
    receipts: list[dict[str, object]] = []
    # Entire target-free pass first.  ``_target_free_held`` only opens
    # contemporaneous feature and Router receipts; policy labels cannot reach
    # the held matrix or parent model prediction.
    for month in months:
        reserve = month - pd.Timedelta(days=args.reserve_days)
        window, coverage = base._load_window(
            candidate_root=None, feature_root=tuple(args.feature_roots), label_root=args.label_root,
            router_root=args.router_root, start=reserve - pd.DateOffset(months=args.train_months), end=reserve,
            fields=fields,
        )
        held, held_coverage = weights._target_free_held(
            feature_roots=tuple(args.feature_roots), router_root=args.router_root, month=month, fields=fields,
        )
        train = stage1._train_rows(window, arm, reserve, args.train_cap)
        labels, geometry = stage1._labels(train, arm)
        target_free, audit = _fit_f72_with_shap(
            train=train, labels=labels, held=held, fields=fields, params=params,
            seed=_canonical_month_seed(month, seed_origin, seed_base),
        )
        path = args.out / "target_free_shap_features" / f"month={month:%Y-%m}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        target_free.to_parquet(path, index=False, compression="zstd")
        score_paths.append((month, path))
        receipt = {
            "held_month": f"{month:%Y-%m}", "reserve": str(reserve),
            "canonical_month_seed": _canonical_month_seed(month, seed_origin, seed_base),
            "train_labels_resolved_before_reserve": True,
            "target_free_persisted_before_metrics": True, "target_free_path": str(path),
            "target_free_feature_columns": int(len(target_free.columns) - len(IDENTITY)),
            "router_top50_identity_exact": True, "target_geometry": geometry,
            "coverage_records": int(len(coverage)), "held_coverage": held_coverage, **audit,
        }
        receipts.append(receipt)
        _progress(args.out, stage="target_free_month_complete", **receipt)
        del window, held, train, labels, target_free
        gc.collect()
    # The sole outcome-joined pass follows all immutable target-free receipts.
    evidence: list[pd.DataFrame] = []
    for month, path in score_paths:
        target_free = pd.read_parquet(path)
        target_free["__decision_ts__"] = pd.to_datetime(target_free["__decision_ts__"], utc=True, errors="raise")
        labelled = target_free.merge(_labels(args.label_root, month), on="candidate_id", how="left", validate="one_to_one")
        valid = labelled.policy_ordinal_valid & np.isfinite(labelled.policy_net_bps)
        if int(valid.sum()) < 2_000:
            raise AssertionError(f"{month:%Y-%m}: insufficient valid policy outcome support")
        evidence.append(_metrics_for_month(labelled.loc[valid].copy(), month))
    detail = pd.concat(evidence, ignore_index=True)
    summary = detail.groupby("feature", sort=True).agg(
        cmi_median=("conditional_mi_binned_given_f72_rank", "median"),
        cmi_min=("conditional_mi_binned_given_f72_rank", "min"),
        ts_ic_mean=("timestamp_rank_ic", "mean"), ts_ic_min=("timestamp_rank_ic", "min"),
        ts_top10_best_median=("ts_top10_ev_best_orientation", "median"),
        ts_top10_best_min=("ts_top10_ev_best_orientation", "min"),
        positive_ic_folds=("timestamp_rank_ic", lambda value: int((value > 0).sum())),
        folds=("held_month", "nunique"),
    ).reset_index()
    summary["stable_shap_signal"] = summary.cmi_median + .25 * summary.ts_ic_mean + .001 * summary.ts_top10_best_min
    summary = summary.sort_values(["stable_shap_signal", "feature"], ascending=[False, True], kind="stable")
    detail.to_parquet(args.out / "shap_derived_oof_evidence.parquet", index=False, compression="zstd")
    summary.to_parquet(args.out / "shap_derived_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(receipts).to_parquet(args.out / "target_free_receipts.parquet", index=False, compression="zstd")
    _once(args.out / "correctness_report.json", {
        "canonical_f72_contract_exact": True,
        "all_parent_training_labels_before_reserve": True,
        "all_held_shap_receipts_persisted_before_outcome_join": True,
        "raw_feature_cmi_or_ic_not_run": True,
        "cmi_ic_are_restricted_to_new_shap_derived_features": True,
        "held_router_and_feature_inputs_target_free": True,
        "catboost_shap_reconstructs_parent_score": True,
        "no_under_mc1_admission_portfolio_live_or_exchange_mutation": True,
        "prehistory_extension_has_no_selection_authority": bool(args.prehistory_extension),
    })
    _progress(args.out, stage="complete", derived_features=int(len(summary)), held_months=len(months))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--hpo-root", type=Path, required=True)
    parser.add_argument("--selection-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", default=DEFAULT_MONTHS)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--canonical-seed-origin", default="2025-11", help="first YYYY-MM in the parent uninterrupted F72 seed schedule")
    parser.add_argument("--canonical-seed-base", type=int, default=SEED, help="seed assigned to --canonical-seed-origin; use to extend a predeclared uninterrupted monthly schedule")
    parser.add_argument("--prehistory-extension", action="store_true", help="allow a contiguous same-era sequence solely to materialise target-free model prehistory; never use its outcome diagnostics for selection")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
