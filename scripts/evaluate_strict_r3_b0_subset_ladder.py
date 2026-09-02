#!/usr/bin/env python3
"""Evaluate compact B0 feature subsets from blend-aware OOF MDA evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRanker

from run_strict_r3_b0_fulluniverse_mda import _evaluate
from run_strict_r3_b0_fulluniverse_screen import _model_params, _read_window, _valid_held, _valid_train
from run_strict_r3_b0_replacement_ranker_screen import SEED, _groups, _sample_queries
from run_strict_r3_routed_et_fulluniverse_screen import _feature_family, _selected_feature_matrix, _utc


SIZES = (120, 90, 70, 50, 35, 25)


def _exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(out: Path, **payload: object) -> None:
    with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _feature_sets(mda: pd.DataFrame, family_mda: pd.DataFrame, sizes: tuple[int, ...]) -> dict[int, list[str]]:
    work = mda.copy().reset_index(drop=True)
    work["family"] = work["family"].fillna(work.feature.map(_feature_family))
    primary = work.sort_values(["selection_score", "stable_blend_mda", "feature"], ascending=[False, False, True], kind="stable")
    boundary = work.sort_values(["stable_boundary_mda", "selection_score", "feature"], ascending=[False, False, True], kind="stable")
    positive_families = set(family_mda.loc[family_mda.stable_blend_mda.gt(0), "family"])
    rescue: list[str] = []
    for family in sorted(positive_families):
        choices = primary.loc[primary.family.eq(family), "feature"]
        if len(choices):
            rescue.append(str(choices.iloc[0]))
    output: dict[int, list[str]] = {}
    for size in sizes:
        core_n = max(1, int(round(size * .75)))
        boundary_n = max(1, int(round(size * .15)))
        selected: list[str] = []
        for field in primary.feature:
            if len(selected) >= core_n:
                break
            selected.append(str(field))
        for field in boundary.feature:
            if len(selected) >= core_n + boundary_n:
                break
            if field not in selected:
                selected.append(str(field))
        for field in rescue:
            if len(selected) >= size:
                break
            if field not in selected:
                selected.append(field)
        for field in primary.feature:
            if len(selected) >= size:
                break
            if field not in selected:
                selected.append(str(field))
        if len(selected) != size:
            raise AssertionError(f"could not form F{size}")
        output[size] = selected
    return output


def _aggregate(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for size, data in metrics.groupby("feature_count", sort=False):
        item: dict[str, object] = {"feature_count": int(size), "observations": len(data)}
        for prefix in ("x", "blend"):
            for name in ("top01_ev", "top05_ev", "top10_ev", "stable_p10", "precision50"):
                field = f"{prefix}_{name}"
                values = pd.to_numeric(data[field], errors="coerce")
                item[f"mean_{field}"] = float(values.mean())
                item[f"median_{field}"] = float(values.median())
                item[f"q10_{field}"] = float(values.quantile(.10))
                item[f"q25_{field}"] = float(values.quantile(.25))
                item[f"worst_{field}"] = float(values.min())
        rows.append(item)
    return pd.DataFrame(rows).sort_values("feature_count", ascending=False)


def _select(summary: pd.DataFrame) -> tuple[int, str]:
    best = summary.loc[summary.mean_blend_stable_p10.idxmax()]
    top10_floor = .99 * float(best.mean_blend_top10_ev)
    stable_floor = .99 * float(best.mean_blend_stable_p10)
    qualifying = summary.loc[
        summary.mean_blend_top10_ev.ge(top10_floor)
        & summary.mean_blend_stable_p10.ge(stable_floor)
        & summary.q10_blend_top10_ev.ge(float(best.q10_blend_top10_ev) - 1e-9)
        & summary.q25_blend_top10_ev.ge(float(best.q25_blend_top10_ev) - 1e-9)
    ].sort_values("feature_count")
    if len(qualifying):
        return int(qualifying.iloc[0].feature_count), "smallest within 1% of best blend Top10/stability without q10/q25 deterioration"
    return int(best.feature_count), "no smaller subset satisfied the predeclared stability guardrails"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--mda-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", nargs="+", default=("2026-02-01", "2026-03-01", "2026-04-01"))
    parser.add_argument("--sizes", nargs="+", type=int, choices=SIZES, default=SIZES)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--held-cap", type=int, default=15_000)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    mda = pd.read_parquet(args.mda_root / "b0_economic_boundary_mda.parquet")
    family_mda = pd.read_parquet(args.mda_root / "b0_family_mda.parquet")
    sizes = tuple(sorted(set(args.sizes), reverse=True))
    feature_sets = _feature_sets(mda, family_mda, sizes)
    args.out.mkdir(parents=True)
    _exclusive(args.out / "run_manifest.json", {
        "schema": "strict_r3_b0_subset_ladder_v1", "scope": "offline B0 candidate only; live unchanged",
        "mda_root": str(args.mda_root), "subsets": list(sizes), "seeds": (SEED, SEED + 70_000),
        "target": "policy_ordinal_base", "gain_schedule": "g3_clipped_economic", "strict_oof": True,
        "selection_rule": "smallest subset within 1% of best blend Top10/stable and no q10/q25 deterioration",
    })
    for size, fields in feature_sets.items():
        _exclusive(args.out / f"f{size}_contract.json", {"feature_count": size, "features": fields, "families": {_feature_family(field) for field in fields}})
    target, valid = "policy_ordinal_base_grade", "policy_ordinal_base_valid"
    observations: list[dict[str, object]] = []
    for fold_index, held_text in enumerate(args.held_months):
        held_month = _utc(held_text)
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        window = _read_window(args.feature_root, args.score_root, args.router_root, args.label_root, reserve - pd.DateOffset(months=args.train_months), held_month + pd.offsets.MonthBegin(1), target)
        train = _sample_queries(_valid_train(window.loc[window.__decision_ts__.lt(reserve)], valid, target, reserve), args.train_cap)
        train = train.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
        held = _sample_queries(_valid_held(window.loc[window.__decision_ts__.ge(held_month)], valid), args.held_cap)
        held = held.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        selected = pd.concat([train, held], ignore_index=True)
        union = list(dict.fromkeys(field for fields in feature_sets.values() for field in fields))
        all_values = _selected_feature_matrix(args.feature_root, selected, union)
        medians = np.nanmedian(all_values[:len(train)], axis=0).astype(np.float32)
        medians[~np.isfinite(medians)] = 0.0
        missing = ~np.isfinite(all_values)
        if missing.any():
            all_values[missing] = np.broadcast_to(medians, all_values.shape)[missing]
        positions = {field: index for index, field in enumerate(union)}
        for size, fields in feature_sets.items():
            index = np.asarray([positions[field] for field in fields], dtype=np.int64)
            x_train, x_held = all_values[:len(train), index], all_values[len(train):, index]
            for seed in (SEED, SEED + 70_000):
                model = LGBMRanker(**_model_params(seed + fold_index, args.n_jobs))
                model.fit(x_train, pd.to_numeric(train[target], errors="coerce").to_numpy(np.int32), group=_groups(train))
                result = _evaluate(held, model.predict(x_held))
                observations.append({"feature_count": size, "held_month": f"{held_month:%Y-%m}", "seed": seed, **result})
                _progress(
                    args.out, stage="subset_fold_seed_complete", feature_count=size,
                    held_month=f"{held_month:%Y-%m}", seed=seed,
                    blend_top10_ev=result["blend_top10_ev"], blend_stable_p10=result["blend_stable_p10"],
                )
                del model
        del all_values, selected, train, held, window
    metric = pd.DataFrame(observations)
    summary = _aggregate(metric)
    selected_size, rationale = _select(summary)
    metric.to_parquet(args.out / "subset_fold_seed_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(args.out / "subset_summary.parquet", index=False, compression="zstd")
    _exclusive(args.out / "selection.json", {"selected_feature_count": selected_size, "rationale": rationale, "contract": str(args.out / f"f{selected_size}_contract.json")})


if __name__ == "__main__":
    main()
