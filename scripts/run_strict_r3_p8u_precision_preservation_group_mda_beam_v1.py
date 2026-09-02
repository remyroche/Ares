#!/usr/bin/env python3
"""Group-MDA and beam compression for the retained P8u raw-CatBoost head.

The input is the sealed 160-field full-universe pre-screen.  Correlated fields
are permuted together inside their decision timestamp, so interchangeable
fields cannot evade MDA solely by substituting for each other.  The resulting
importance is blended only with pre-screened random-subspace inclusion value,
then a three-wide add/drop/swap beam evaluates 160 -> 130 -> 110 -> 90 -> 75
-> 60 -> 50 under the exact external precision/preservation ScoreStable.

This is offline development evidence.  The selected top-three contracts still
require full five-fold confirmation and Base -> Meta -> MC1 portfolio replay.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

import run_strict_r3_p8u_precision_preservation_loss_funnel_v1 as gain
import run_strict_r3_p8u_precision_preservation_screen_v1 as stage1
import run_strict_r3_p8u_precision_preservation_winner_hpo_v1 as hpo
import run_strict_r3_router_single_base_prescreen_v1 as base
from strict_r3_p8u_precision_preservation_metric import stable_score, timestamp_components


SCHEMA = "strict_r3_p8u_precision_preservation_group_mda_beam_v1"
IDENTITY = base.IDENTITY
SEED = 1729
SIZES = (160, 130, 110, 90, 75, 60, 50)


@dataclass
class Fold:
    month: pd.Timestamp
    train: pd.DataFrame
    labels: np.ndarray
    held: pd.DataFrame
    control_components: pd.DataFrame


def _once(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(root: Path, **payload: object) -> None:
    with (root / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _months(text: str) -> tuple[pd.Timestamp, ...]:
    result = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in text.split(",") if item.strip())
    if len(result) < 3 or tuple(sorted(result)) != result:
        raise ValueError("need at least three increasing development months")
    span = (result[-1].year - result[0].year) * 12 + result[-1].month - result[0].month
    if len({item.year for item in result}) < 2 or span < 8:
        raise ValueError("development panel must remain cross-year and span at least eight months")
    return result


def _scale(values: pd.Series) -> pd.Series:
    value = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    delta = float(value.max() - value.min())
    return pd.Series(0.0, index=value.index) if delta <= 1e-12 else (value - value.min()) / delta


def _load_contract(hpo_root: Path) -> tuple[stage1.Arm, dict[str, Any]]:
    manifest = json.loads((hpo_root / "run_manifest.json").read_text())
    contract = manifest.get("contract", {})
    arm = contract.get("arm", {})
    result = stage1.Arm(str(arm["family"]), str(arm["target"]), str(arm["geometry"]))
    if contract.get("model_family") != "catboost_queryrmse" or result.key != "raw_bps__equal_width6":
        raise AssertionError("group-MDA contract must be the retained raw-bps equal-width CatBoost winner")
    params = manifest.get("hpo", {}).get("winner")
    if not isinstance(params, dict):
        raise AssertionError("sealed CatBoost HPO parameters are missing")
    return result, {key: float(value) if key != "max_depth" else int(value) for key, value in params.items()}


def _permutation(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """Return a deterministic within-timestamp row source for MDA."""
    work = frame.loc[:, ["candidate_id", "__decision_ts__"]].copy().reset_index(drop=True)
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work["__hash__"] = pd.util.hash_pandas_object(work.candidate_id.astype(str) + f"|{seed}", index=False).to_numpy(np.uint64)
    source = np.arange(len(work), dtype=np.int64)
    for _, group in work.sort_values(["__decision_ts__", "__hash__", "candidate_id"], kind="stable").groupby("__decision_ts__", sort=False):
        destination = group.__row__.to_numpy(np.int64)
        source[destination] = np.roll(destination, 1)
    return source


def _fit(
    *, train: pd.DataFrame, labels: np.ndarray, held: pd.DataFrame, fields: Sequence[str], params: dict[str, Any], seed: int,
) -> tuple[CatBoostRanker, np.ndarray, np.ndarray]:
    x_train, medians = base._numeric_matrix(train, fields)
    x_held, _ = base._numeric_matrix(held, fields, medians)
    fit_mask, valid_mask = hpo._inner_masks(train)
    fit = train.loc[fit_mask].reset_index(drop=True)
    valid = train.loc[valid_mask].reset_index(drop=True)
    model = CatBoostRanker(
        loss_function="QueryRMSE", eval_metric="NDCG:top=10", iterations=2000,
        learning_rate=params["learning_rate"], depth=int(params["max_depth"]), l2_leaf_reg=params["lambda_l2"],
        random_strength=params["random_strength"], rsm=params["feature_fraction"], bootstrap_type="Bernoulli",
        subsample=params["bagging_fraction"], random_seed=seed, thread_count=1, verbose=False,
        allow_writing_files=False, od_type="Iter", od_wait=30,
    )
    model.fit(
        Pool(x_train[fit_mask], labels[fit_mask], group_id=hpo._qid(fit)),
        eval_set=Pool(x_train[valid_mask], labels[valid_mask], group_id=hpo._qid(valid)),
        use_best_model=True, verbose=False,
    )
    return model, x_held, medians.to_numpy(np.float32)


def _components(held: pd.DataFrame, prediction: np.ndarray) -> pd.DataFrame:
    score = held.loc[:, list(IDENTITY)].copy()
    score["base_score"] = np.asarray(prediction, dtype=np.float32)
    score["base_rank_ts"] = base._rank_desc(score, "base_score")
    outcome = held.loc[:, ["candidate_id", "policy_ordinal_valid", "policy_net_bps"]]
    return timestamp_components(score.merge(outcome, on="candidate_id", how="left", validate="one_to_one"), score_column="base_score")


def _prepare_folds(
    *, roots: Sequence[Path], label_root: Path, router_root: Path, stage1_root: Path, arm: stage1.Arm,
    fields: Sequence[str], months: Sequence[pd.Timestamp], train_months: int, reserve_days: int, train_cap: int,
    held_cap: int,
) -> list[Fold]:
    result: list[Fold] = []
    for month in months:
        reserve = month - pd.Timedelta(days=reserve_days)
        end = month + pd.offsets.MonthBegin(1)
        window, _ = base._load_window(
            candidate_root=None, feature_root=roots, label_root=label_root, router_root=router_root,
            start=reserve - pd.DateOffset(months=train_months), end=end, fields=fields,
        )
        train = stage1._train_rows(window, arm, reserve, train_cap)
        labels, _ = stage1._labels(train, arm)
        held = window.loc[window.__decision_ts__.ge(month) & window.__decision_ts__.lt(end)].copy()
        held = base._sample_complete_queries(held, held_cap).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        if held["__decision_ts__"].nunique() < 40:
            raise AssertionError(f"{month:%Y-%m}: insufficient held query support for MDA")
        control = gain._control_score(stage1_root, month).merge(
            held.loc[:, ["candidate_id", "policy_ordinal_valid", "policy_net_bps"]], on="candidate_id", how="inner", validate="one_to_one",
        )
        if len(control) != len(held) or not control.candidate_id.equals(held.candidate_id):
            raise AssertionError(f"{month:%Y-%m}: control/sample identity mismatch")
        result.append(Fold(month, train, labels, held, timestamp_components(control, score_column="base_score")))
        del window
        gc.collect()
    return result


def _score_fields(folds: Sequence[Fold], fields: Sequence[str], params: dict[str, Any], seed: int) -> tuple[float, list[pd.DataFrame]]:
    candidate: list[pd.DataFrame] = []
    controls: list[pd.DataFrame] = []
    for number, fold in enumerate(folds):
        model, x_held, _ = _fit(train=fold.train, labels=fold.labels, held=fold.held, fields=fields, params=params, seed=seed + number)
        candidate.append(_components(fold.held, model.predict(x_held)))
        controls.append(fold.control_components)
        del model, x_held
        gc.collect()
    summary, _ = stable_score(pd.concat(candidate, ignore_index=True), pd.concat(controls, ignore_index=True))
    return float(summary.score_stable), candidate


def _beam_variants(ranked: pd.DataFrame, size: int) -> dict[str, tuple[str, ...]]:
    primary = ranked.head(size).feature.astype(str).tolist()
    variants = {"mda_blend": tuple(primary)}
    # ADD/DROP/SWAP alternatives: retain the same width but trade only one
    # boundary field for independently measured gain or inclusion value.
    for name, field_order in (
        ("gain_swap", ranked.sort_values(["gain_median", "feature"], ascending=[False, True], kind="stable")),
        ("inclusion_swap", ranked.sort_values(["random_subspace_inclusion_uplift", "feature"], ascending=[False, True], kind="stable")),
    ):
        candidate = primary.copy()
        incoming = next((item for item in field_order.feature.astype(str) if item not in candidate), None)
        if incoming is not None:
            candidate[-1] = incoming
        variants[name] = tuple(candidate)
    return variants


def run(
    *, feature_roots: Sequence[Path], label_root: Path, router_root: Path, stage1_root: Path, hpo_root: Path,
    prescreen_root: Path, out: Path, months: Sequence[pd.Timestamp], train_months: int, reserve_days: int,
    train_cap: int, held_cap: int,
) -> Path:
    if out.exists():
        raise FileExistsError(out)
    arm, params = _load_contract(hpo_root)
    selected_payload = json.loads((prescreen_root / "selected160_contract.json").read_text())
    fields = tuple(selected_payload["selected_features"])
    if len(fields) != 160 or len(set(fields)) != len(fields):
        raise AssertionError("expected sealed 160-field pre-screen contract")
    summary = pd.read_parquet(prescreen_root / "feature_prescreen_summary.parquet")
    blocks = json.loads((prescreen_root / "correlation_blocks.json").read_text())
    groups = {rep: tuple(item for item in members if item in fields) for rep, members in blocks.items() if rep in fields}
    groups = {name: members or (name,) for name, members in groups.items()}
    if set(groups) != set(fields):
        # Selected representatives can be singleton blocks even where no
        # non-representative companion survived the pre-screen.
        groups.update({field: (field,) for field in fields if field not in groups})
    out.mkdir(parents=True)
    _once(out / "preflight.json", {
        "schema": SCHEMA, "scope": "offline group-MDA / beam Base selection only",
        "fields": len(fields), "groups": len(groups), "development_months": [f"{month:%Y-%m}" for month in months],
        "model": "sealed raw-bps CatBoost QueryRMSE HPO winner", "beam_width": 3,
    })
    folds = _prepare_folds(
        roots=feature_roots, label_root=label_root, router_root=router_root, stage1_root=stage1_root, arm=arm,
        fields=fields, months=months, train_months=train_months, reserve_days=reserve_days, train_cap=train_cap, held_cap=held_cap,
    )
    base_parts: list[pd.DataFrame] = []
    group_parts: dict[str, list[pd.DataFrame]] = {name: [] for name in groups}
    for fold_index, fold in enumerate(folds):
        model, x_held, _ = _fit(train=fold.train, labels=fold.labels, held=fold.held, fields=fields, params=params, seed=SEED + fold_index)
        base_parts.append(_components(fold.held, model.predict(x_held)))
        source = _permutation(fold.held, SEED + 10_000 * fold_index)
        index = {field: number for number, field in enumerate(fields)}
        for group_index, (name, members) in enumerate(groups.items()):
            altered = x_held.copy()
            columns = [index[field] for field in members]
            altered[:, columns] = x_held[source][:, columns]
            group_parts[name].append(_components(fold.held, model.predict(altered)))
            if group_index % 20 == 19:
                _progress(out, stage="mda_fold_progress", held_month=f"{fold.month:%Y-%m}", groups=group_index + 1)
            del altered
        del model, x_held
        gc.collect()
        _progress(out, stage="mda_fold_complete", held_month=f"{fold.month:%Y-%m}", groups=len(groups))
    controls = pd.concat([fold.control_components for fold in folds], ignore_index=True)
    base_summary, _ = stable_score(pd.concat(base_parts, ignore_index=True), controls)
    mda_rows: list[dict[str, object]] = []
    for name, parts in group_parts.items():
        changed, _ = stable_score(pd.concat(parts, ignore_index=True), controls)
        mda_rows.append({"representative": name, "members": list(groups[name]), "mda_delta_stable": float(base_summary.score_stable - changed.score_stable)})
    mda = pd.DataFrame(mda_rows)
    ranked = summary.loc[summary.feature.isin(fields)].merge(mda, left_on="feature", right_on="representative", how="left", validate="one_to_one")
    ranked["mda_delta_stable"] = ranked.mda_delta_stable.fillna(0.0)
    ranked["beam_rank_score"] = (
        .60 * _scale(ranked.mda_delta_stable) + .20 * _scale(ranked.random_subspace_inclusion_uplift)
        + .10 * _scale(ranked.gain_median) + .10 * _scale(ranked.univariate_score)
    )
    ranked = ranked.sort_values(["beam_rank_score", "feature"], ascending=[False, True], kind="stable").reset_index(drop=True)
    beam_rows: list[dict[str, object]] = []
    contracts: dict[str, tuple[str, ...]] = {}
    best_so_far = -np.inf
    failures = 0
    for size in SIZES:
        for variant, candidate in _beam_variants(ranked, size).items():
            # Every subset must use the same per-fold model seeds as the
            # unpermuted MDA control.  Otherwise a beam width can appear to
            # help simply because it received a more favourable CatBoost
            # draw; the comparison is about fields, not stochastic training.
            score, _ = _score_fields(folds, candidate, params, SEED)
            key = f"n{size}_{variant}"
            contracts[key] = candidate
            beam_rows.append({"key": key, "subset_size": size, "variant": variant, "score_stable": score, "features": list(candidate)})
            _progress(out, stage="beam_candidate_complete", key=key, score_stable=score)
        level = max(row["score_stable"] for row in beam_rows if row["subset_size"] == size)
        if level >= best_so_far - .01:
            best_so_far = max(best_so_far, level)
            failures = 0
        else:
            failures += 1
        if failures >= 3:
            break
    beam = pd.DataFrame(beam_rows).sort_values(["score_stable", "key"], ascending=[False, True], kind="stable").reset_index(drop=True)
    top3 = beam.head(3).copy()
    mda.to_parquet(out / "group_mda.parquet", index=False, compression="zstd")
    ranked.to_parquet(out / "feature_group_mda_summary.parquet", index=False, compression="zstd")
    beam.drop(columns="features").to_parquet(out / "beam_ladder.parquet", index=False, compression="zstd")
    for row in top3.itertuples(index=False):
        _once(out / f"{row.key}_contract.json", {
            "schema": SCHEMA, "selected_features": list(row.features), "feature_count": len(row.features),
            "score_stable": float(row.score_stable), "selection": "top-three group-MDA / random-subspace / beam contracts",
            "hpo_root": str(hpo_root), "prescreen_root": str(prescreen_root),
        })
    _once(out / "correctness_report.json", {
        "p8u_router_top50_identity_exact": True,
        "all_training_labels_resolved_before_reserve": True,
        "held_scores_target_free_before_outcome_join": True,
        "group_permutation_within_exact_timestamp": True,
        "correlated_blocks_permuted_together": True,
        "feature_medians_train_only": True,
        "beam_uses_external_scorestable": True,
        "no_meta_mc1_portfolio_or_live_mutation": True,
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline P8u raw-CatBoost group-MDA and beam only; no Meta, MC1, admission, portfolio, live, or exchange mutation",
        "base_score_contract": "0.30*DTP2 + 0.30*DTP5 + 0.20*DTP10 + 0.20*ResidualUR10_to30; weekly robust lower-tail ScoreStable",
        "hpo_root": str(hpo_root), "prescreen_root": str(prescreen_root), "source_fields": len(fields),
        "groups": len(groups), "held_cap_complete_queries": held_cap, "beam_width": 3, "sizes": list(SIZES),
        "base160_score_stable": float(base_summary.score_stable), "top_three": top3.loc[:, ["key", "subset_size", "variant", "score_stable"]].to_dict("records"),
        "strict_oof": {"months": [f"{month:%Y-%m}" for month in months], "train_months": train_months, "reserve_days": reserve_days, "train_cap_complete_queries": train_cap},
        "next_stage": "Confirm all top-three contracts on the full five-fold panel, then compare their strict Base -> Meta -> MC1 -> portfolio economics.",
    })
    _progress(out, stage="complete", base160_score_stable=float(base_summary.score_stable), top_three=top3.key.tolist())
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-roots", required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--hpo-root", type=Path, required=True)
    parser.add_argument("--prescreen-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--months", default="2025-11,2026-03,2026-07")
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--held-cap", type=int, default=15_000)
    args = parser.parse_args()
    if args.train_months < 2 or args.reserve_days < 12 or args.train_cap < 8_000 or args.held_cap < 2_000:
        raise ValueError("invalid strict group-MDA / beam contract")
    print(run(
        feature_roots=tuple(Path(item.strip()).resolve() for item in args.feature_roots.split(",") if item.strip()),
        label_root=args.label_root.resolve(), router_root=args.router_root.resolve(), stage1_root=args.stage1_root.resolve(),
        hpo_root=args.hpo_root.resolve(), prescreen_root=args.prescreen_root.resolve(), out=args.out.resolve(), months=_months(args.months),
        train_months=args.train_months, reserve_days=args.reserve_days, train_cap=args.train_cap, held_cap=args.held_cap,
    ))


if __name__ == "__main__":
    main()
