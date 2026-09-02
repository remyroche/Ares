#!/usr/bin/env python3
"""One-at-a-time semantic-family add-back validation for frozen B0 F70."""

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
from run_strict_r3_routed_et_fulluniverse_screen import _selected_feature_matrix, _utc


FAMILIES = (
    "structure_location", "volatility_tails", "other_structural", "returns_trend",
    "open_interest_flow", "liquidity_microstructure",
)


def _exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for arm, data in frame.groupby("arm", sort=False):
        item: dict[str, object] = {"arm": arm, "feature_count": int(data.feature_count.iloc[0]), "observations": len(data)}
        for name in ("blend_top01_ev", "blend_top05_ev", "blend_top10_ev", "blend_stable_p10", "blend_precision50", "x_top10_ev", "x_stable_p10"):
            values = pd.to_numeric(data[name], errors="coerce")
            item[f"mean_{name}"] = float(values.mean())
            item[f"q10_{name}"] = float(values.quantile(.10))
            item[f"q25_{name}"] = float(values.quantile(.25))
            item[f"worst_{name}"] = float(values.min())
        rows.append(item)
    return pd.DataFrame(rows).sort_values("mean_blend_stable_p10", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--base-contract", type=Path, required=True)
    parser.add_argument("--screen-summary", type=Path, required=True)
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
    base = list(json.loads(args.base_contract.read_text())["features"])
    screen = pd.read_parquet(args.screen_summary).set_index("feature")
    screen120 = screen.loc[screen.selected_screen120].copy()
    arms: dict[str, list[str]] = {"f70_control": base}
    for family in FAMILIES:
        extras = [field for field in screen120.loc[screen120.family.eq(family)].sort_values("screen_score", ascending=False, kind="stable").index if field not in set(base)]
        if extras:
            arm = list(dict.fromkeys(base + extras))
            if len(arm) > 120:
                raise AssertionError(f"{family} arm breaches 120 features")
            arms[f"f70_plus_{family}"] = arm
    union = list(dict.fromkeys(field for fields in arms.values() for field in fields))
    args.out.mkdir(parents=True)
    _exclusive(args.out / "run_manifest.json", {"schema": "strict_r3_b0_family_addback_v1", "scope": "offline B0 candidate only; live unchanged", "base_contract": str(args.base_contract), "families": list(FAMILIES), "arms": {name: len(fields) for name, fields in arms.items()}, "target": "policy_ordinal_base", "strict_oof": True, "seeds": (SEED, SEED + 70_000), "excluded": {"funding": "negative/unstable family MDA; not a useful add-back"}})
    target, valid = "policy_ordinal_base_grade", "policy_ordinal_base_valid"
    rows: list[dict[str, object]] = []
    for fold_index, held_text in enumerate(args.held_months):
        held_month = _utc(held_text)
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        window = _read_window(args.feature_root, args.score_root, args.router_root, args.label_root, reserve - pd.DateOffset(months=args.train_months), held_month + pd.offsets.MonthBegin(1), target)
        train = _sample_queries(_valid_train(window.loc[window.__decision_ts__.lt(reserve)], valid, target, reserve), args.train_cap).sort_values(["__decision_ts__", "candidate_id"], kind="stable")
        held = _sample_queries(_valid_held(window.loc[window.__decision_ts__.ge(held_month)], valid), args.held_cap).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
        selected = pd.concat([train, held], ignore_index=True)
        values = _selected_feature_matrix(args.feature_root, selected, union)
        medians = np.nanmedian(values[:len(train)], axis=0).astype(np.float32)
        medians[~np.isfinite(medians)] = 0.0
        missing = ~np.isfinite(values)
        if missing.any():
            values[missing] = np.broadcast_to(medians, values.shape)[missing]
        positions = {field: index for index, field in enumerate(union)}
        for arm, fields in arms.items():
            indices = np.asarray([positions[field] for field in fields], dtype=np.int64)
            for seed in (SEED, SEED + 70_000):
                model = LGBMRanker(**_model_params(seed + fold_index, args.n_jobs))
                model.fit(values[:len(train), indices], pd.to_numeric(train[target], errors="coerce").to_numpy(np.int32), group=_groups(train))
                result = _evaluate(held, model.predict(values[len(train):, indices]))
                rows.append({"arm": arm, "feature_count": len(fields), "held_month": f"{held_month:%Y-%m}", "seed": seed, **result})
                del model
        del values, selected, train, held, window
    observations = pd.DataFrame(rows)
    summary = _aggregate(observations)
    control = summary.loc[summary.arm.eq("f70_control")].iloc[0]
    summary["delta_blend_top10_ev"] = summary.mean_blend_top10_ev - float(control.mean_blend_top10_ev)
    summary["delta_blend_stable_p10"] = summary.mean_blend_stable_p10 - float(control.mean_blend_stable_p10)
    accepted = summary.loc[
        summary.arm.ne("f70_control") & summary.delta_blend_top10_ev.gt(0)
        & summary.delta_blend_stable_p10.gt(0)
        & summary.q10_blend_top10_ev.ge(float(control.q10_blend_top10_ev))
        & summary.q25_blend_top10_ev.ge(float(control.q25_blend_top10_ev))
    ].sort_values(["delta_blend_stable_p10", "delta_blend_top10_ev"], ascending=False)
    winner = str(accepted.iloc[0].arm) if len(accepted) else "f70_control"
    observations.to_parquet(args.out / "addback_fold_seed_metrics.parquet", index=False, compression="zstd")
    summary.to_parquet(args.out / "addback_summary.parquet", index=False, compression="zstd")
    _exclusive(args.out / "selection.json", {"selected_arm": winner, "selected_features": arms[winner], "rationale": "must improve both blend Top10/stable and not lower q10/q25; otherwise F70 control", "arms": {name: fields for name, fields in arms.items()}})


if __name__ == "__main__":
    main()
