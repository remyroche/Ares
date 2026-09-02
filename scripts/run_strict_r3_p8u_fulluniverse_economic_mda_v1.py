#!/usr/bin/env python3
"""Strict-OOF economic / Top-10-boundary MDA for Screen120 Base heads.

Raw causal fields are assessed only by timestamp-local economic permutation
loss.  In particular, this stage intentionally performs no raw-field CMI or
IC: those diagnostics belong exclusively to the separate SHAP-derived feature
receipt.  All outcome use is post-score evaluation on outer held folds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import run_strict_r3_routed_et_fulluniverse_screen as screen  # noqa: E402


SCHEMA = "strict_r3_p8u_fulluniverse_economic_mda_v1"
SEED = 1729
IDENTITY = screen.IDENTITY


def _months(value: str) -> tuple[pd.Timestamp, ...]:
    result = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in value.split(",") if item.strip())
    if len(result) != 3 or tuple(sorted(result)) != result:
        raise ValueError("require exactly three chronological held months")
    return result


def _once(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _hash(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode()).hexdigest()


def _progress(out: Path, **payload: object) -> None:
    with (out / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str, sort_keys=True) + "\n")


def _fields(screen_root: Path, head: str) -> tuple[str, ...]:
    payload = json.loads((screen_root / f"{head.lower()}_screen120_contract.json").read_text())
    fields = tuple(map(str, payload["feature_contract"]))
    if not 20 <= len(fields) <= 120 or len(fields) != len(set(fields)):
        raise AssertionError(f"{head}: invalid screen contract")
    if payload.get("feature_contract_sha256") != _hash(fields):
        raise AssertionError(f"{head}: feature-contract hash mismatch")
    return fields


def _within_timestamp_permutation(frame: pd.DataFrame, seed: int) -> np.ndarray:
    """Deterministic query-local permutation; does not alter query state."""
    work = frame.loc[:, ["candidate_id", "__decision_ts__"]].reset_index(drop=True).copy()
    work["__row__"] = np.arange(len(work), dtype=np.int64)
    work["__hash__"] = pd.util.hash_pandas_object(work.candidate_id.astype(str) + f"|{seed}", index=False).to_numpy(np.uint64)
    output = np.arange(len(work), dtype=np.int64)
    for _, part in work.sort_values(["__decision_ts__", "__hash__", "candidate_id"], kind="stable").groupby("__decision_ts__", sort=False):
        rows = part.__row__.to_numpy(np.int64)
        if len(rows) > 1:
            output[rows] = np.roll(rows, 1)
    return output


def _boundary_delta(frame: pd.DataFrame, base: np.ndarray, altered: np.ndarray) -> float:
    """Loss of realised policy quality due to membership changes at Top-10%."""
    work = frame.loc[:, ["__decision_ts__", "candidate_id", "policy_net_bps"]].reset_index(drop=True).copy()
    work["base"] = np.asarray(base, dtype=float)
    work["altered"] = np.asarray(altered, dtype=float)
    deltas: list[float] = []
    for _, part in work.groupby("__decision_ts__", sort=False):
        if len(part) < 2:
            continue
        k = max(1, int(np.ceil(len(part) * .10)))
        left = set(part.sort_values(["base", "candidate_id"], ascending=[False, True], kind="stable").head(k).candidate_id)
        right = set(part.sort_values(["altered", "candidate_id"], ascending=[False, True], kind="stable").head(k).candidate_id)
        lost = part.loc[part.candidate_id.isin(left - right), "policy_net_bps"]
        gained = part.loc[part.candidate_id.isin(right - left), "policy_net_bps"]
        if len(lost) and len(gained):
            deltas.append(float(lost.mean() - gained.mean()))
        elif len(lost) != len(gained):
            raise AssertionError("top10 replacement membership is not balanced")
    return float(np.mean(deltas)) if deltas else 0.0


def _prepare(
    *, head: str, fields: Sequence[str], held_month: pd.Timestamp, feature_root: Path, router_root: Path,
    labels_root: Path, base_labels_root: Path | None, policy: pd.DataFrame, train_months: int,
    reserve_days: int, train_cap: int, held_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    target = str(screen.HEADS[head]["target"])
    reserve = held_month - pd.Timedelta(days=reserve_days)
    start = reserve - pd.DateOffset(months=train_months)
    window = screen._joined(
        feature_root=feature_root, router_root=router_root, labels_root=labels_root, base_labels_root=base_labels_root,
        policy=policy, start=start, end=screen._month_end(held_month), fields=(), route_fraction=.50,
    )
    train = screen._strict_train(window.loc[window.__decision_ts__.lt(reserve)].copy(), reserve, target, train_cap)
    held = screen._time_balanced_sample(screen._held_eval(window.loc[window.__decision_ts__.ge(held_month)].copy(), target), held_cap, seed=SEED + held_month.month)
    if len(train) < 8_000 or len(held) < 1_000:
        raise AssertionError(f"{head}/{held_month:%Y-%m}: insufficient strict support")
    selected = pd.concat([train, held], ignore_index=True)
    matrix = screen._selected_feature_matrix(feature_root, selected, fields)
    matrix, _ = screen._impute_from_train(matrix, len(train))
    return train, held.reset_index(drop=True), matrix[:len(train)], matrix[len(train):]


def _fold(
    *, head: str, fields: Sequence[str], held_month: pd.Timestamp, feature_root: Path, router_root: Path,
    labels_root: Path, base_labels_root: Path | None, policy: pd.DataFrame, train_months: int,
    reserve_days: int, train_cap: int, held_cap: int, n_jobs: int, out: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    target, direction = str(screen.HEADS[head]["target"]), float(screen.HEADS[head]["direction"])
    train, held, x_train, x_held = _prepare(
        head=head, fields=fields, held_month=held_month, feature_root=feature_root, router_root=router_root,
        labels_root=labels_root, base_labels_root=base_labels_root, policy=policy, train_months=train_months,
        reserve_days=reserve_days, train_cap=train_cap, held_cap=held_cap,
    )
    model = LGBMRegressor(**screen._params(seed=SEED + held_month.month * 1000 + (0 if head == "B0" else 100_000 if head == "E" else 200_000), n_jobs=n_jobs))
    model.fit(x_train, pd.to_numeric(train[target], errors="coerce").to_numpy(float))
    base = direction * model.predict(x_held)
    target_free = held.loc[:, list(IDENTITY)].copy()
    target_free["parent_score"] = np.asarray(base, dtype=np.float32)
    score_path = out / "target_free_scores" / f"head={head}" / f"month={held_month:%Y-%m}.parquet"
    score_path.parent.mkdir(parents=True, exist_ok=True)
    target_free.to_parquet(score_path, index=False, compression="zstd")
    baseline = screen._metric_suite(held.assign(__score__=base), "__score__")
    rows: list[dict[str, object]] = []
    permutation = _within_timestamp_permutation(held, seed=SEED + held_month.month)
    for column, field in enumerate(fields):
        altered_matrix = x_held.copy()
        altered_matrix[:, column] = x_held[permutation, column]
        altered = direction * model.predict(altered_matrix)
        metric = screen._metric_suite(held.assign(__score__=altered), "__score__")
        rows.append({
            "head": head, "held_month": f"{held_month:%Y-%m}", "kind": "individual", "name": field,
            "delta_ts_top10_ev": float(baseline["ts_top10_ev"] - metric["ts_top10_ev"]),
            "delta_ts_top5_ev": float(baseline["ts_top05_ev"] - metric["ts_top05_ev"]),
            "delta_base_stable_p10": float(baseline["base_stable_p10"] - metric["base_stable_p10"]),
            "boundary_delta_top10_ev": _boundary_delta(held, base, altered),
            "baseline_ts_top10_ev": float(baseline["ts_top10_ev"]),
        })
    families: dict[str, list[int]] = {}
    for index, field in enumerate(fields):
        families.setdefault(screen._feature_family(field), []).append(index)
    for family, indices in families.items():
        altered_matrix = x_held.copy()
        for column in indices:
            altered_matrix[:, column] = x_held[permutation, column]
        altered = direction * model.predict(altered_matrix)
        metric = screen._metric_suite(held.assign(__score__=altered), "__score__")
        rows.append({
            "head": head, "held_month": f"{held_month:%Y-%m}", "kind": "semantic_family", "name": family,
            "delta_ts_top10_ev": float(baseline["ts_top10_ev"] - metric["ts_top10_ev"]),
            "delta_ts_top5_ev": float(baseline["ts_top05_ev"] - metric["ts_top05_ev"]),
            "delta_base_stable_p10": float(baseline["base_stable_p10"] - metric["base_stable_p10"]),
            "boundary_delta_top10_ev": _boundary_delta(held, base, altered),
            "baseline_ts_top10_ev": float(baseline["ts_top10_ev"]),
        })
    receipt = {
        "head": head, "held_month": f"{held_month:%Y-%m}", "train_rows": len(train), "held_rows": len(held),
        "target_free_score": str(score_path), "target_free_persisted_before_metrics": True,
        "strict_labels_before_reserve": True,
    }
    return pd.DataFrame(rows), pd.DataFrame([{"head": head, "held_month": f"{held_month:%Y-%m}", **baseline}]), receipt


def run(args: argparse.Namespace) -> None:
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    months = _months(args.held_months)
    policy = screen._read_policy(args.policy_path, months)
    _once(args.out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline strict-OOF economic MDA; no live/MC1/admission/portfolio/exchange mutation",
        "heads": args.heads, "screen_roots": {head: str(root) for head, root in zip(args.heads, args.screen_roots, strict=True)},
        "held_months": [f"{month:%Y-%m}" for month in months], "train_months": args.train_months, "reserve_days": args.reserve_days,
        "raw_feature_cmi_or_ic": False, "raw_feature_selection": "economic permutation only",
        "permutation": "candidate fields rotate within exact timestamp queries; query-constant distributions remain intact",
    })
    all_rows: list[pd.DataFrame] = []
    base_rows: list[pd.DataFrame] = []
    provenance: list[dict[str, object]] = []
    contracts: dict[str, list[str]] = {}
    for head, root in zip(args.heads, args.screen_roots, strict=True):
        fields = _fields(root, head)
        contracts[head] = list(fields)
        for month in months:
            detail, baseline, receipt = _fold(
                head=head, fields=fields, held_month=month, feature_root=args.feature_root, router_root=args.router_root,
                labels_root=args.labels_root, base_labels_root=args.base_labels_root, policy=policy,
                train_months=args.train_months, reserve_days=args.reserve_days, train_cap=args.train_cap,
                held_cap=args.held_cap, n_jobs=args.n_jobs, out=args.out,
            )
            all_rows.append(detail); base_rows.append(baseline); provenance.append(receipt)
            _progress(args.out, stage="fold_complete", **receipt)
    detail = pd.concat(all_rows, ignore_index=True)
    summary = detail.loc[detail.kind.eq("individual")].groupby(["head", "name"], sort=False).agg(
        mda_top10_median=("delta_ts_top10_ev", "median"),
        mda_top10_iqr=("delta_ts_top10_ev", lambda x: float(x.quantile(.75) - x.quantile(.25))),
        mda_top10_worst=("delta_ts_top10_ev", "min"),
        mda_top5_median=("delta_ts_top5_ev", "median"),
        mda_stable_median=("delta_base_stable_p10", "median"),
        boundary_mda_median=("boundary_delta_top10_ev", "median"),
        positive_fold_count=("delta_ts_top10_ev", lambda x: int((x > 0).sum())),
        folds=("held_month", "nunique"),
    ).reset_index().rename(columns={"name": "feature"})
    summary["stable_mda"] = summary.mda_top10_median - .5 * summary.mda_top10_iqr
    group = detail.loc[detail.kind.eq("semantic_family")].groupby(["head", "name"], sort=False).agg(
        family_mda_top10_median=("delta_ts_top10_ev", "median"),
        family_boundary_mda_median=("boundary_delta_top10_ev", "median"),
        family_positive_fold_count=("delta_ts_top10_ev", lambda x: int((x > 0).sum())),
    ).reset_index().rename(columns={"name": "semantic_family"})
    detail.to_parquet(args.out / "economic_mda_detail.parquet", index=False, compression="zstd")
    summary.sort_values(["head", "stable_mda", "boundary_mda_median", "feature"], ascending=[True, False, False, True], kind="stable").to_parquet(args.out / "economic_mda_summary.parquet", index=False, compression="zstd")
    group.to_parquet(args.out / "economic_family_mda.parquet", index=False, compression="zstd")
    pd.concat(base_rows, ignore_index=True).to_parquet(args.out / "baseline_fold_metrics.parquet", index=False, compression="zstd")
    _once(args.out / "feature_contracts.json", {"schema": SCHEMA, "heads": contracts, "provenance": provenance})
    _once(args.out / "correctness_report.json", {
        "all_held_scores_persisted_before_metrics": True,
        "all_training_labels_before_reserve": True,
        "raw_feature_cmi_or_ic_not_run": True,
        "individual_permutations_are_timestamp_local": True,
        "semantic_family_permutations_are_timestamp_local": True,
        "live_or_exchange_mutation": False,
    })
    _progress(args.out, stage="complete", heads=args.heads, individual_features=int(len(summary)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--heads", nargs="+", choices=("B0", "E", "T"), required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--base-labels-root", type=Path)
    parser.add_argument("--policy-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--held-months", default="2025-05,2025-06,2025-07")
    parser.add_argument("--train-months", type=int, default=4)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=120_000)
    parser.add_argument("--held-cap", type=int, default=25_000)
    parser.add_argument("--n-jobs", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()
    if len(args.heads) != len(args.screen_roots) or args.out.exists() or args.train_months < 2 or args.reserve_days < 12:
        raise ValueError("invalid immutable economic-MDA contract")
    if "B0" in args.heads and args.base_labels_root is None:
        raise ValueError("--base-labels-root required for B0")
    run(args)


if __name__ == "__main__":
    main()
