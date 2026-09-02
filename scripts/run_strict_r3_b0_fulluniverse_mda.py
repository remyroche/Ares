#!/usr/bin/env python3
"""Strict OOF B0 MDA, Top-10-boundary MDA, and family MDA.

The selected B0 Screen120 contract is evaluated with two deterministic seeds
per blocked fold.  Permutations are within decision timestamp, preserving
market-wide state while destroying candidate discrimination.  Each feature is
measured both standalone and through the E+T+X blend, so redundant standalone
B0 features do not advance merely because they make X look good in isolation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from run_strict_r3_b0_fulluniverse_screen import (
    _blend_metrics, _model_params, _read_window, _valid_held, _valid_train,
)
from run_strict_r3_b0_replacement_ranker_screen import SEED, _groups, _rank, _sample_queries
from run_strict_r3_routed_et_fulluniverse_screen import _feature_family, _metric_suite, _selected_feature_matrix, _utc


def _exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(out: Path, **payload: object) -> None:
    with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _within_timestamp_permutation(frame: pd.DataFrame, seed: int) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__"]].reset_index(drop=True).copy()
    work["position"] = np.arange(len(work), dtype=np.int64)
    work["hash"] = pd.util.hash_pandas_object(work.candidate_id.astype(str) + f"|{seed}", index=False).to_numpy(np.uint64)
    output = np.arange(len(work), dtype=np.int64)
    for _, group in work.sort_values(["__decision_ts__", "hash", "candidate_id"], kind="stable").groupby("__decision_ts__", sort=False):
        positions = group.position.to_numpy(np.int64)
        if len(positions) > 1:
            output[positions] = np.roll(positions, 1)
    return output


def _top_set(frame: pd.DataFrame, score: np.ndarray) -> tuple[set[tuple[pd.Timestamp, str]], np.ndarray]:
    work = frame.loc[:, ["__decision_ts__", "candidate_id"]].reset_index(drop=True).copy()
    work["score"] = np.asarray(score, dtype=float)
    work["position"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "score", "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size")
    selected = ordinal.le(np.ceil(count.to_numpy(float) * .10))
    membership = np.zeros(len(work), dtype=bool)
    membership[work.position.to_numpy(np.int64)] = selected.to_numpy(bool)
    ranks = np.empty(len(work), dtype=np.float32)
    ranks[work.position.to_numpy(np.int64)] = (1.0 - (ordinal.to_numpy(float) - .5) / count.to_numpy(float)).astype(np.float32)
    pairs = set(zip(work.loc[selected, "__decision_ts__"], work.loc[selected, "candidate_id"], strict=True))
    return pairs, ranks


def _boundary_delta(frame: pd.DataFrame, baseline: np.ndarray, perturbed: np.ndarray) -> tuple[float, float]:
    base_set, base_rank = _top_set(frame, baseline)
    pert_set, pert_rank = _top_set(frame, perturbed)
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "policy_net_bps"]].copy()
    work["base"] = [pair in base_set for pair in zip(work.__decision_ts__, work.candidate_id, strict=True)]
    work["perturbed"] = [pair in pert_set for pair in zip(work.__decision_ts__, work.candidate_id, strict=True)]
    work["near_boundary"] = (base_rank >= .75) | (pert_rank >= .75)
    deltas: list[float] = []
    for _, group in work.loc[work.near_boundary].groupby("__decision_ts__", sort=False):
        removed = group.loc[group.base & ~group.perturbed, "policy_net_bps"]
        added = group.loc[group.perturbed & ~group.base, "policy_net_bps"]
        if len(removed) and len(added):
            deltas.append(float(removed.mean() - added.mean()))
    jaccard = len(base_set & pert_set) / max(1, len(base_set | pert_set))
    return (float(np.mean(deltas)) if deltas else 0.0, float(jaccard))


def _evaluate(frame: pd.DataFrame, x_score: np.ndarray) -> dict[str, float]:
    standalone = _metric_suite(frame.assign(__score__=x_score), "__score__")
    blended = _blend_metrics(frame, x_score)
    return {
        "x_top01_ev": standalone["ts_top01_ev"], "x_top05_ev": standalone["ts_top05_ev"], "x_top10_ev": standalone["ts_top10_ev"],
        "x_stable_p10": standalone["base_stable_p10"],
        "x_precision50": standalone["ts_top10_precision50"],
        "blend_top01_ev": blended["etx_ts_top01_ev"], "blend_top05_ev": blended["etx_ts_top05_ev"],
        "blend_top10_ev": blended["etx_ts_top10_ev"], "blend_stable_p10": blended["etx_base_stable_p10"],
        "blend_precision50": blended["etx_ts_top10_precision50"],
    }


def _score(model: LGBMRanker, values: np.ndarray, perturb: list[int] | None, source: np.ndarray) -> np.ndarray:
    if not perturb:
        return model.predict(values)
    work = values.copy()
    for index in perturb:
        work[:, index] = work[source, index]
    return model.predict(work)


def _family_indices(fields: list[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {}
    for index, field in enumerate(fields):
        groups.setdefault(_feature_family(field), []).append(index)
    return groups


def _summarize(rows: pd.DataFrame, group_column: str) -> pd.DataFrame:
    result: list[dict[str, object]] = []
    for value, data in rows.groupby(group_column, sort=False):
        item: dict[str, object] = {group_column: value, "observations": len(data)}
        for field in ("mda_x_top10_ev", "mda_x_stable_p10", "mda_blend_top05_ev", "mda_blend_top10_ev", "mda_blend_stable_p10", "mda_blend_precision50", "boundary_mda_ev", "top10_jaccard"):
            values = pd.to_numeric(data[field], errors="coerce")
            item[f"median_{field}"] = float(values.median())
            item[f"iqr_{field}"] = float(values.quantile(.75) - values.quantile(.25))
            item[f"worst_{field}"] = float(values.min())
            item[f"positive_{field}_count"] = int(values.gt(0).sum())
        item["stable_blend_mda"] = float(item["median_mda_blend_stable_p10"] - .5 * item["iqr_mda_blend_stable_p10"])
        item["stable_boundary_mda"] = float(item["median_boundary_mda_ev"] - .5 * item["iqr_boundary_mda_ev"])
        item["selection_score"] = float(
            .50 * item["stable_blend_mda"] + .20 * (item["median_mda_blend_top10_ev"] - .5 * item["iqr_mda_blend_top10_ev"])
            + .20 * item["stable_boundary_mda"] + .10 * (item["median_mda_blend_top05_ev"] - .5 * item["iqr_mda_blend_top05_ev"])
        )
        result.append(item)
    return pd.DataFrame(result).sort_values("selection_score", ascending=False, kind="stable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", nargs="+", default=("2026-02-01", "2026-03-01", "2026-04-01"))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--held-cap", type=int, default=15_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    contract = json.loads((args.screen_root / "b0_screen120_contract.json").read_text())
    fields = list(contract["feature_contract"])
    if not 1 <= len(fields) <= 120:
        raise AssertionError("MDA requires the frozen <=120 Screen120 contract")
    target = "policy_ordinal_base_grade"
    valid = "policy_ordinal_base_valid"
    args.out.mkdir(parents=True)
    _exclusive(args.out / "run_manifest.json", {
        "schema": "strict_r3_b0_fulluniverse_mda_v1", "scope": "offline B0 candidate only; live unchanged",
        "screen_contract": str(args.screen_root / "b0_screen120_contract.json"), "feature_count": len(fields),
        "target": "policy_ordinal_base", "strict_oof": True, "seeds": (SEED, SEED + 70_000),
        "permutation": "deterministic within-timestamp derangement", "outcomes_or_targets_in_features": False,
        "blend_measure": "equal E+T+X; E/T are downstream metrics only",
    })
    feature_rows: list[dict[str, object]] = []
    family_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    families = _family_indices(fields)
    for fold_index, held_text in enumerate(args.held_months):
        held_month = _utc(held_text)
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        window = _read_window(
            args.feature_root, args.score_root, args.router_root, args.label_root,
            reserve - pd.DateOffset(months=args.train_months), held_month + pd.offsets.MonthBegin(1), target,
        )
        train = _sample_queries(_valid_train(window.loc[window.__decision_ts__.lt(reserve)], valid, target, reserve), args.train_cap)
        train = train.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
        held = _sample_queries(_valid_held(window.loc[window.__decision_ts__.ge(held_month)], valid), args.held_cap)
        held = held.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        selected = pd.concat([train, held], ignore_index=True)
        values = _selected_feature_matrix(args.feature_root, selected, fields)
        medians = np.nanmedian(values[:len(train)], axis=0).astype(np.float32)
        medians[~np.isfinite(medians)] = 0.0
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.broadcast_to(medians, values.shape)[missing]
        x_train, x_held = values[:len(train)], values[len(train):]
        for seed_index, seed in enumerate((SEED, SEED + 70_000)):
            model = LGBMRanker(**_model_params(seed + fold_index, args.n_jobs))
            model.fit(x_train, pd.to_numeric(train[target], errors="coerce").to_numpy(np.int32), group=_groups(train))
            source = _within_timestamp_permutation(held, seed + 4000 * fold_index)
            baseline_score = _score(model, x_held, None, source)
            baseline = _evaluate(held, baseline_score)
            baseline_rows.append({"held_month": f"{held_month:%Y-%m}", "seed": seed, **baseline})
            for index, field in enumerate(fields):
                perturbed_score = _score(model, x_held, [index], source)
                perturbed = _evaluate(held, perturbed_score)
                boundary, jaccard = _boundary_delta(held, baseline_score, perturbed_score)
                feature_rows.append({
                    "feature": field, "family": _feature_family(field), "held_month": f"{held_month:%Y-%m}", "seed": seed,
                    "mda_x_top10_ev": baseline["x_top10_ev"] - perturbed["x_top10_ev"],
                    "mda_x_stable_p10": baseline["x_stable_p10"] - perturbed["x_stable_p10"],
                    "mda_blend_top05_ev": baseline["blend_top05_ev"] - perturbed["blend_top05_ev"],
                    "mda_blend_top10_ev": baseline["blend_top10_ev"] - perturbed["blend_top10_ev"],
                    "mda_blend_stable_p10": baseline["blend_stable_p10"] - perturbed["blend_stable_p10"],
                    "mda_blend_precision50": baseline["blend_precision50"] - perturbed["blend_precision50"],
                    "boundary_mda_ev": boundary, "top10_jaccard": jaccard,
                })
            for family, indices in families.items():
                perturbed_score = _score(model, x_held, indices, source)
                perturbed = _evaluate(held, perturbed_score)
                boundary, jaccard = _boundary_delta(held, baseline_score, perturbed_score)
                family_rows.append({
                    "family": family, "fields": len(indices), "held_month": f"{held_month:%Y-%m}", "seed": seed,
                    "mda_x_top10_ev": baseline["x_top10_ev"] - perturbed["x_top10_ev"],
                    "mda_x_stable_p10": baseline["x_stable_p10"] - perturbed["x_stable_p10"],
                    "mda_blend_top05_ev": baseline["blend_top05_ev"] - perturbed["blend_top05_ev"],
                    "mda_blend_top10_ev": baseline["blend_top10_ev"] - perturbed["blend_top10_ev"],
                    "mda_blend_stable_p10": baseline["blend_stable_p10"] - perturbed["blend_stable_p10"],
                    "mda_blend_precision50": baseline["blend_precision50"] - perturbed["blend_precision50"],
                    "boundary_mda_ev": boundary, "top10_jaccard": jaccard,
                })
            _progress(args.out, stage="mda_fold_seed_complete", held_month=f"{held_month:%Y-%m}", seed=seed, features=len(fields))
            del model
        del values, x_train, x_held, selected, train, held, window
        gc.collect()
    feature = pd.DataFrame(feature_rows)
    family = pd.DataFrame(family_rows)
    feature.to_parquet(args.out / "b0_mda_observations.parquet", index=False, compression="zstd")
    family.to_parquet(args.out / "b0_family_mda_observations.parquet", index=False, compression="zstd")
    _summarize(feature, "feature").merge(pd.DataFrame({"feature": fields, "family": [_feature_family(field) for field in fields]}), on="feature", how="left").to_parquet(args.out / "b0_economic_boundary_mda.parquet", index=False, compression="zstd")
    _summarize(family, "family").to_parquet(args.out / "b0_family_mda.parquet", index=False, compression="zstd")
    pd.DataFrame(baseline_rows).to_parquet(args.out / "b0_mda_baseline_metrics.parquet", index=False, compression="zstd")
    _progress(args.out, stage="mda_complete", features=len(fields), families=len(families))


if __name__ == "__main__":
    main()
