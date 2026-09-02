#!/usr/bin/env python3
"""OOF timestamp-local economic MDA for the routed E/T feature contracts.

Features are permuted within decision timestamps on held rows.  That preserves
market-wide state and tests the incremental cross-sectional information needed
by the routed base layer.  No outcome/label is fed into a feature matrix.
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
    HEADS, SEED, _feature_family, _held_eval, _impute_from_train, _joined,
    _metric_suite, _params, _selected_feature_matrix, _strict_train,
    _time_balanced_sample, _utc,
)


def _exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _progress(path: Path, **payload: object) -> None:
    with (path / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _within_timestamp_permutation(frame: pd.DataFrame, *, seed: int) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__"]].reset_index(drop=True).copy()
    work["__pos__"] = np.arange(len(work), dtype=np.int64)
    hashed = pd.util.hash_pandas_object(work.candidate_id.astype(str) + f"|{seed}", index=False).to_numpy(np.uint64)
    work["__hash__"] = hashed
    source = np.arange(len(work), dtype=np.int64)
    for _, group in work.sort_values(["__decision_ts__", "__hash__", "candidate_id"], kind="stable").groupby("__decision_ts__", sort=False):
        position = group["__pos__"].to_numpy(np.int64)
        if len(position) > 1:
            source[position] = np.roll(position, 1)
    return source


def _top10_membership(frame: pd.DataFrame, score: np.ndarray) -> set[tuple[pd.Timestamp, str]]:
    work = frame.loc[:, ["__decision_ts__", "candidate_id"]].reset_index(drop=True).copy()
    work["score"] = score
    work = work.sort_values(["__decision_ts__", "score", "candidate_id"], ascending=[True, False, True], kind="stable")
    order = work.groupby("__decision_ts__", sort=False).cumcount() + 1
    count = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size")
    chosen = work.loc[order.le(np.ceil(count.to_numpy(float) * .10))]
    return set(zip(chosen.__decision_ts__, chosen.candidate_id, strict=True))


def _family_sets(fields: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for field in fields:
        result.setdefault(_feature_family(field), []).append(field)
    return result


def _evaluate(frame: pd.DataFrame, score: np.ndarray) -> dict[str, float]:
    work = frame.loc[:, ["__decision_ts__", "candidate_id", "policy_net_bps"]].reset_index(drop=True).copy()
    work["score"] = score
    metrics = _metric_suite(work, "score")
    top10 = _top10_membership(work, score)
    return {
        "top01_ev": metrics["ts_top01_ev"],
        "top05_ev": metrics["ts_top05_ev"],
        "top10_ev": metrics["ts_top10_ev"],
        "top10_precision50": metrics["ts_top10_precision50"],
        "stable_p10": metrics["base_stable_p10"],
        "top10_members": float(len(top10)),
        "_members": top10,
    }


def _permuted_score(model: LGBMRegressor, x: np.ndarray, indices: list[int], source: np.ndarray) -> np.ndarray:
    work = x.copy()
    for index in indices:
        work[:, index] = work[source, index]
    return model.predict(work)


def _head(args: argparse.Namespace, head: str, fields: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = str(HEADS[head]["target"])
    direction = float(HEADS[head]["direction"])
    policy = pd.read_parquet(args.policy_path, columns=["candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts"])
    policy.policy_label_available_ts = pd.to_datetime(policy.policy_label_available_ts, utc=True, errors="coerce")
    policy.policy_path_valid = policy.policy_path_valid.fillna(False).astype(bool)
    folds = tuple(_utc(x) for x in args.held_months)
    aggregate: dict[str, list[dict[str, object]]] = {field: [] for field in fields}
    family_aggregate: dict[str, list[dict[str, object]]] = {family: [] for family in _family_sets(fields)}
    fold_rows: list[dict[str, object]] = []
    for fold_index, held_month in enumerate(folds):
        reserve = held_month - pd.Timedelta(days=args.reserve_days)
        start = reserve - pd.DateOffset(months=args.train_months)
        end = held_month + pd.offsets.MonthBegin(1)
        window = _joined(
            feature_root=args.feature_root, router_root=args.router_root, labels_root=args.labels_root,
            policy=policy, start=start, end=end, fields=(), route_fraction=.50,
        )
        train = _strict_train(window.loc[window.__decision_ts__.lt(reserve)].copy(), reserve, target, args.train_cap)
        held = _time_balanced_sample(_held_eval(window.loc[window.__decision_ts__.ge(held_month)].copy()), args.held_cap, seed=SEED + fold_index)
        if len(train) < 8000 or len(held) < 1000:
            raise AssertionError(f"{head}/{held_month:%Y-%m}: insufficient strict support")
        selected = pd.concat([train, held], ignore_index=True)
        values = _selected_feature_matrix(args.feature_root, selected, fields)
        values, _ = _impute_from_train(values, len(train))
        x_train, x_held = values[:len(train)], values[len(train):]
        model = LGBMRegressor(**_params(seed=SEED + fold_index + (0 if head == "E" else 10000), n_jobs=args.n_jobs, cheap=False))
        model.fit(x_train, pd.to_numeric(train[target], errors="coerce").to_numpy(float))
        baseline_score = direction * model.predict(x_held)
        baseline = _evaluate(held, baseline_score)
        source = _within_timestamp_permutation(held, seed=SEED + 100 * fold_index + (0 if head == "E" else 5000))
        baseline_members = baseline.pop("_members")
        fold_rows.append({"head": head, "held_month": f"{held_month:%Y-%m}", "kind": "baseline", "field": "__all__", **baseline})
        _progress(args.out, stage="mda_fold_baseline", head=head, held_month=f"{held_month:%Y-%m}", fields=len(fields), **baseline)
        for index, field in enumerate(fields):
            perturbed = _evaluate(held, direction * _permuted_score(model, x_held, [index], source))
            members = perturbed.pop("_members")
            aggregate[field].append({
                "held_month": f"{held_month:%Y-%m}", "delta_top10_ev": baseline["top10_ev"] - perturbed["top10_ev"],
                "delta_top01_ev": baseline["top01_ev"] - perturbed["top01_ev"],
                "delta_stable_p10": baseline["stable_p10"] - perturbed["stable_p10"],
                "delta_precision50": baseline["top10_precision50"] - perturbed["top10_precision50"],
                "top10_jaccard": len(baseline_members & members) / max(1, len(baseline_members | members)),
            })
            if (index + 1) % args.progress_every == 0:
                _progress(args.out, stage="mda_feature_progress", head=head, held_month=f"{held_month:%Y-%m}", complete=index + 1, fields=len(fields))
                gc.collect()
        for family, group in _family_sets(fields).items():
            indices = [fields.index(field) for field in group]
            perturbed = _evaluate(held, direction * _permuted_score(model, x_held, indices, source))
            members = perturbed.pop("_members")
            family_aggregate[family].append({
                "held_month": f"{held_month:%Y-%m}", "family": family, "fields": len(group),
                "delta_top10_ev": baseline["top10_ev"] - perturbed["top10_ev"],
                "delta_top01_ev": baseline["top01_ev"] - perturbed["top01_ev"],
                "delta_stable_p10": baseline["stable_p10"] - perturbed["stable_p10"],
                "delta_precision50": baseline["top10_precision50"] - perturbed["top10_precision50"],
                "top10_jaccard": len(baseline_members & members) / max(1, len(baseline_members | members)),
            })
        _progress(args.out, stage="mda_fold_complete", head=head, held_month=f"{held_month:%Y-%m}", fields=len(fields))
        del model, values, x_train, x_held, selected, train, held, window
        gc.collect()
    feature_rows: list[dict[str, object]] = []
    for field, rows in aggregate.items():
        data = pd.DataFrame(rows)
        feature_rows.append({
            "head": head, "feature": field, "family": _feature_family(field), "folds": len(data),
            "mda_top10_ev": data.delta_top10_ev.mean(), "mda_top01_ev": data.delta_top01_ev.mean(),
            "mda_stable_p10": data.delta_stable_p10.mean(), "mda_precision50": data.delta_precision50.mean(),
            "mda_top10_jaccard": data.top10_jaccard.mean(),
            "mda_worst_month_top10_ev": data.delta_top10_ev.min(),
        })
    family_rows = pd.concat([pd.DataFrame(rows) for rows in family_aggregate.values()], ignore_index=True)
    return pd.DataFrame(feature_rows), family_rows, pd.DataFrame(fold_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--heads", nargs="+", choices=("E", "T"), default=("E", "T"))
    parser.add_argument("--held-months", nargs="+", default=("2026-02-01", "2026-03-01", "2026-04-01"))
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=35000)
    parser.add_argument("--held-cap", type=int, default=15000)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=None, help="smoke-test only; never used for a final selection contract")
    parser.add_argument("--progress-every", type=int, default=16)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    manifest = {"schema": "strict_r3_routed_et_mda_v1", "scope": "offline E/T only; B0/live unchanged", "strict_oof": True, "permutation": "within-timestamp deterministic derangement", "target_or_outcome_in_features": False}
    _exclusive(args.out / "run_manifest.json", manifest)
    for head in args.heads:
        contract = json.loads((args.screen_root / f"{head.lower()}_screen120_contract.json").read_text())
        fields = list(contract["feature_contract"])
        if args.max_features is not None:
            fields = fields[:args.max_features]
        feature, family, folds = _head(args, head, fields)
        feature.to_parquet(args.out / f"{head.lower()}_economic_mda.parquet", index=False, compression="zstd")
        family.to_parquet(args.out / f"{head.lower()}_family_mda.parquet", index=False, compression="zstd")
        folds.to_parquet(args.out / f"{head.lower()}_mda_baseline_metrics.parquet", index=False, compression="zstd")


if __name__ == "__main__":
    main()
