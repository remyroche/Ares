#!/usr/bin/env python3
"""Two-seed, complementarity-aware OOF MDA for routed E or T.

This is deliberately a feature-selection diagnostic, not a new downstream
stack.  It trains only one physical E/T head on its existing supportive target
and measures a feature both on that head and after equal timestamp-rank blending
with the immutable target-free B0 and counterpart-head scores.  Labels are
joined only after the target-free feature/score identity panel is fixed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_strict_r3_routed_et_fulluniverse_screen import (  # noqa: E402
    HEADS, IDENTITY, SEED, _feature_family, _held_eval, _impute_from_train,
    _joined, _metric_suite, _params, _selected_feature_matrix, _strict_train,
    _time_balanced_sample, _utc,
)


def _exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(path: Path, **payload: object) -> None:
    with (path / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _months(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    first = _utc(start).normalize().replace(day=1)
    final = (_utc(end) - pd.Timedelta(nanoseconds=1)).normalize().replace(day=1)
    return tuple(pd.date_range(first, final, freq="MS", tz="UTC"))


def _scores(score_root: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for month in _months(start, end):
        path = score_root / "target_free_monthly" / f"month={month:%Y-%m}" / "scores_features.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        part = pd.read_parquet(path, columns=[*IDENTITY, "base_bps", "efficiency_bps", "timing_bps"])
        part["__decision_ts__"] = pd.to_datetime(part["__decision_ts__"], utc=True, errors="raise")
        if part.candidate_id.duplicated().any():
            raise AssertionError(f"duplicate target-free score identity in {path}")
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    if result.candidate_id.duplicated().any():
        raise AssertionError("duplicate target-free score identity across months")
    for field in ("base_bps", "efficiency_bps", "timing_bps"):
        result[field] = pd.to_numeric(result[field], errors="coerce")
    return result


def _window(args: argparse.Namespace, start: pd.Timestamp, end: pd.Timestamp, policy: pd.DataFrame) -> pd.DataFrame:
    frame = _joined(
        feature_root=args.feature_root, router_root=args.router_root,
        labels_root=args.labels_root, policy=policy, start=start, end=end,
        fields=(), route_fraction=.50,
    )
    scores = _scores(args.score_root, start, end)
    frame = frame.merge(scores, on=list(IDENTITY), how="left", validate="one_to_one")
    if frame[["base_bps", "efficiency_bps", "timing_bps"]].isna().any().any():
        raise AssertionError("target-free score identity/coverage failure")
    return frame


def _rank(frame: pd.DataFrame, column: str) -> np.ndarray:
    work = frame.loc[:, ["__decision_ts__", "candidate_id", column]].copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", column, "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    output = np.empty(len(work), dtype=np.float32)
    output[work["__row__"].to_numpy(np.int64)] = (1.0 - (ordinal - .5) / count).astype(np.float32)
    return output


def _blend_score(held: pd.DataFrame, head: str, score: np.ndarray) -> np.ndarray:
    work = held.loc[:, ["__decision_ts__", "candidate_id", "base_bps", "efficiency_bps", "timing_bps"]].copy()
    work["candidate_score"] = np.asarray(score, dtype=float)
    if head == "E":
        work["efficiency_bps"] = work["candidate_score"]
    else:
        work["timing_bps"] = work["candidate_score"]
    return (_rank(work, "base_bps") + _rank(work, "efficiency_bps") + _rank(work, "timing_bps")) / 3.0


def _permutation(frame: pd.DataFrame, seed: int) -> np.ndarray:
    work = frame.loc[:, ["__decision_ts__", "candidate_id"]].reset_index(drop=True).copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work["__hash__"] = pd.util.hash_pandas_object(work.candidate_id.astype(str) + f"|{seed}", index=False).to_numpy(np.uint64)
    output = np.arange(len(work), dtype=np.int64)
    for _, group in work.sort_values(["__decision_ts__", "__hash__", "candidate_id"], kind="stable").groupby("__decision_ts__", sort=False):
        rows = group["__row__"].to_numpy(np.int64)
        if len(rows) > 1:
            output[rows] = np.roll(rows, 1)
    return output


def _top10(frame: pd.DataFrame, score: np.ndarray) -> tuple[set[tuple[pd.Timestamp, str]], np.ndarray]:
    work = frame.loc[:, ["__decision_ts__", "candidate_id"]].reset_index(drop=True).copy()
    work["score"] = np.asarray(score, dtype=float)
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "score", "candidate_id"], ascending=[True, False, True], kind="stable")
    order = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float) + 1.0
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    chosen = order <= np.ceil(count * .10)
    ranks = np.empty(len(work), dtype=np.float32)
    ranks[work["__row__"].to_numpy(np.int64)] = (1.0 - (order - .5) / count).astype(np.float32)
    pairs = set(zip(work.loc[chosen, "__decision_ts__"], work.loc[chosen, "candidate_id"], strict=True))
    return pairs, ranks


def _boundary(frame: pd.DataFrame, baseline: np.ndarray, perturbed: np.ndarray) -> tuple[float, float]:
    base_set, base_rank = _top10(frame, baseline)
    pert_set, pert_rank = _top10(frame, perturbed)
    work = frame.loc[:, ["__decision_ts__", "candidate_id", "policy_net_bps"]].copy()
    pair = list(zip(work.__decision_ts__, work.candidate_id, strict=True))
    work["base"] = [item in base_set for item in pair]
    work["perturbed"] = [item in pert_set for item in pair]
    work["near"] = (base_rank >= .75) | (pert_rank >= .75)
    deltas: list[float] = []
    for _, group in work.loc[work.near].groupby("__decision_ts__", sort=False):
        removed = group.loc[group.base & ~group.perturbed, "policy_net_bps"]
        added = group.loc[group.perturbed & ~group.base, "policy_net_bps"]
        if len(removed) and len(added):
            deltas.append(float(removed.mean() - added.mean()))
    return float(np.mean(deltas)) if deltas else 0.0, len(base_set & pert_set) / max(1, len(base_set | pert_set))


def _metrics(held: pd.DataFrame, head: str, score: np.ndarray) -> dict[str, float]:
    standalone = _metric_suite(held.assign(__score__=score), "__score__")
    blended = _metric_suite(held.assign(__blend__=_blend_score(held, head, score)), "__blend__")
    return {
        "x_top10_ev": standalone["ts_top10_ev"], "x_top05_ev": standalone["ts_top05_ev"],
        "x_stable": standalone["base_stable_p10"], "blend_top10_ev": blended["ts_top10_ev"],
        "blend_top05_ev": blended["ts_top05_ev"], "blend_stable": blended["base_stable_p10"],
        "blend_precision50": blended["ts_top10_precision50"],
    }


def _summary(observations: pd.DataFrame, key: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for value, data in observations.groupby(key, sort=False):
        row: dict[str, object] = {key: value, "observations": len(data)}
        for metric in ("mda_x_top10_ev", "mda_x_stable", "mda_blend_top10_ev", "mda_blend_top05_ev", "mda_blend_stable", "boundary_mda_ev"):
            sample = pd.to_numeric(data[metric], errors="coerce")
            row[f"median_{metric}"] = float(sample.median())
            row[f"iqr_{metric}"] = float(sample.quantile(.75) - sample.quantile(.25))
            row[f"worst_{metric}"] = float(sample.min())
            row[f"positive_{metric}"] = int(sample.gt(0).sum())
        row["selection_score"] = float(
            .50 * (row["median_mda_blend_stable"] - .5 * row["iqr_mda_blend_stable"])
            + .20 * (row["median_mda_blend_top10_ev"] - .5 * row["iqr_mda_blend_top10_ev"])
            + .20 * (row["median_boundary_mda_ev"] - .5 * row["iqr_boundary_mda_ev"])
            + .10 * (row["median_mda_blend_top05_ev"] - .5 * row["iqr_mda_blend_top05_ev"])
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False, kind="stable")


def _subset_contracts(out: Path, head: str, fields: list[str], feature_summary: pd.DataFrame, family_summary: pd.DataFrame) -> None:
    work = feature_summary.set_index("feature").reindex(fields).copy()
    work["family"] = [_feature_family(field) for field in work.index]
    rescue = set()
    positive_family = set(family_summary.loc[family_summary.median_mda_blend_stable.gt(0), "family"])
    for family, group in work.loc[work.family.isin(positive_family)].groupby("family", sort=False):
        rescue.add(str(group.sort_values("selection_score", ascending=False, kind="stable").index[0]))
    ordered = list(rescue)
    ordered.extend(field for field in work.sort_values("selection_score", ascending=False, kind="stable").index if field not in rescue)
    for size in (120, 90, 70, 50, 35, 25):
        chosen = ordered[:min(size, len(ordered))]
        _exclusive(out / f"{head.lower()}_conditional_subset{size}_contract.json", {
            "head": head, "feature_count": len(chosen), "features": chosen,
            "sha256": hashlib.sha256("\n".join(chosen).encode()).hexdigest(),
            "selection": "two-seed strict-OOF conditional E/T MDA plus boundary/family rescue",
        })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--score-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--head", choices=("E", "T"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", nargs="+", default=("2026-02-01", "2026-03-01", "2026-04-01"))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=35000)
    parser.add_argument("--held-cap", type=int, default=15000)
    parser.add_argument("--n-jobs", type=int, default=3)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    contract = json.loads((args.screen_root / f"{args.head.lower()}_screen120_contract.json").read_text())
    fields = list(contract["feature_contract"])
    if not 1 <= len(fields) <= 120:
        raise AssertionError("conditional MDA requires a frozen <=120 Screen120 contract")
    args.out.mkdir(parents=True)
    _exclusive(args.out / "run_manifest.json", {
        "schema": "strict_r3_routed_et_conditional_mda_v1", "scope": "offline E/T research only; B0/live unchanged",
        "head": args.head, "target": HEADS[args.head], "screen_contract": str(args.screen_root / f"{args.head.lower()}_screen120_contract.json"),
        "score_root": str(args.score_root), "route": "strict-OOF timestamp-local top50", "seeds": [SEED, SEED + 70000],
        "permutation": "within timestamp", "conditional_metric": "equal B0/E/T timestamp-rank blend", "outcomes_or_targets_in_features": False,
    })
    policy = pd.read_parquet(args.policy_path, columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"])
    policy["policy_path_valid"] = policy["policy_path_valid"].fillna(False).astype(bool)
    policy["policy_label_available_ts"] = pd.to_datetime(policy["policy_label_available_ts"], utc=True, errors="coerce")
    target, direction = str(HEADS[args.head]["target"]), float(HEADS[args.head]["direction"])
    observations: list[dict[str, object]] = []
    family_observations: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    family_indices: dict[str, list[int]] = {}
    for index, field in enumerate(fields):
        family_indices.setdefault(_feature_family(field), []).append(index)
    for fold, held_value in enumerate(args.held_months):
        held_month = _utc(held_value)
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        window = _window(args, reserve - pd.DateOffset(months=args.train_months), held_month + pd.offsets.MonthBegin(1), policy)
        train = _strict_train(window.loc[window.__decision_ts__.lt(reserve)].copy(), reserve, target, args.train_cap)
        held = _time_balanced_sample(_held_eval(window.loc[window.__decision_ts__.ge(held_month)].copy()), args.held_cap, seed=SEED + fold)
        if len(train) < 8000 or len(held) < 1000:
            raise AssertionError(f"{args.head}/{held_month:%Y-%m}: insufficient strict support")
        selected = pd.concat([train, held], ignore_index=True)
        values = _selected_feature_matrix(args.feature_root, selected, fields)
        values, _ = _impute_from_train(values, len(train))
        x_train, x_held = values[:len(train)], values[len(train):]
        held = held.reset_index(drop=True)
        source = _permutation(held, SEED + 4000 * fold)
        for seed in (SEED, SEED + 70000):
            model = LGBMRegressor(**_params(seed=seed + fold + (0 if args.head == "E" else 10000), n_jobs=args.n_jobs, cheap=False))
            model.fit(x_train, pd.to_numeric(train[target], errors="coerce").to_numpy(float))
            baseline_score = direction * model.predict(x_held)
            baseline = _metrics(held, args.head, baseline_score)
            blend_baseline = _blend_score(held, args.head, baseline_score)
            baseline_rows.append({"head": args.head, "held_month": f"{held_month:%Y-%m}", "seed": seed, **baseline})
            for index, field in enumerate(fields):
                work = x_held.copy(); work[:, index] = work[source, index]
                altered_score = direction * model.predict(work)
                altered = _metrics(held, args.head, altered_score)
                boundary, jaccard = _boundary(held, blend_baseline, _blend_score(held, args.head, altered_score))
                observations.append({"feature": field, "family": _feature_family(field), "held_month": f"{held_month:%Y-%m}", "seed": seed,
                    "mda_x_top10_ev": baseline["x_top10_ev"] - altered["x_top10_ev"], "mda_x_stable": baseline["x_stable"] - altered["x_stable"],
                    "mda_blend_top10_ev": baseline["blend_top10_ev"] - altered["blend_top10_ev"], "mda_blend_top05_ev": baseline["blend_top05_ev"] - altered["blend_top05_ev"],
                    "mda_blend_stable": baseline["blend_stable"] - altered["blend_stable"], "boundary_mda_ev": boundary, "top10_jaccard": jaccard})
            for family, indices in family_indices.items():
                work = x_held.copy(); work[:, indices] = work[source][:, indices]
                altered_score = direction * model.predict(work)
                altered = _metrics(held, args.head, altered_score)
                boundary, jaccard = _boundary(held, blend_baseline, _blend_score(held, args.head, altered_score))
                family_observations.append({"family": family, "fields": len(indices), "held_month": f"{held_month:%Y-%m}", "seed": seed,
                    "mda_x_top10_ev": baseline["x_top10_ev"] - altered["x_top10_ev"], "mda_x_stable": baseline["x_stable"] - altered["x_stable"],
                    "mda_blend_top10_ev": baseline["blend_top10_ev"] - altered["blend_top10_ev"], "mda_blend_top05_ev": baseline["blend_top05_ev"] - altered["blend_top05_ev"],
                    "mda_blend_stable": baseline["blend_stable"] - altered["blend_stable"], "boundary_mda_ev": boundary, "top10_jaccard": jaccard})
            _progress(args.out, stage="fold_seed_complete", head=args.head, held_month=f"{held_month:%Y-%m}", seed=seed, features=len(fields))
            del model
        del values, x_train, x_held, selected, train, held, window
        gc.collect()
    feature = pd.DataFrame(observations)
    family = pd.DataFrame(family_observations)
    feature_summary = _summary(feature, "feature")
    feature_summary["family"] = feature_summary.feature.map({field: _feature_family(field) for field in fields})
    family_summary = _summary(family, "family")
    feature.to_parquet(args.out / f"{args.head.lower()}_conditional_mda_observations.parquet", index=False, compression="zstd")
    family.to_parquet(args.out / f"{args.head.lower()}_conditional_family_mda_observations.parquet", index=False, compression="zstd")
    feature_summary.to_parquet(args.out / f"{args.head.lower()}_conditional_mda_summary.parquet", index=False, compression="zstd")
    family_summary.to_parquet(args.out / f"{args.head.lower()}_conditional_family_mda_summary.parquet", index=False, compression="zstd")
    pd.DataFrame(baseline_rows).to_parquet(args.out / f"{args.head.lower()}_conditional_mda_baseline.parquet", index=False, compression="zstd")
    _subset_contracts(args.out, args.head, fields, feature_summary, family_summary)
    _progress(args.out, stage="complete", head=args.head, features=len(fields))


if __name__ == "__main__":
    main()
