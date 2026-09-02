#!/usr/bin/env python3
"""Score one frozen B/E/T head on a target-free, strict chronological ledger.

The input winner is the immutable result of
``run_strict_r3_direct_head_crossyear_hpo_v1.py``.  For every held month this
producer fits only on labels resolved before the preceding reserve and emits
scores for *every* point-in-time router-selected candidate.  Policy outcomes,
path validity, and supportive labels are excluded from the emitted panel and
cannot determine whether a held candidate is scored.

Research only: this producer does not change live inference or any downstream
consensus, MC1, admission, portfolio, execution, or exchange contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from run_strict_r3_base_stability_selector_v2 import (
    HEADS, IDENTITY, SEED, _impute, _materialize, _next_month, _read_policy,
    _months as _source_months, _read_head_labels, _read_router,
    _route_top_fraction, _train_rows, _utc, _window,
)
from run_strict_r3_direct_head_crossyear_hpo_v1 import _feature_contract, _model, _params, _split_early


def _exclusive(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _months(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Timestamp, ...]:
    values = tuple(pd.date_range(start.normalize(), end.normalize(), freq="MS", tz="UTC"))
    if not values or values[0] != start.normalize() or values[-1] != end.normalize():
        raise ValueError("start/end must be UTC calendar-month starts")
    return values


def _winner(path: Path, head: str) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if str(data.get("head")) != head:
        raise AssertionError(f"winner {path} is not a {head}-head winner")
    params = data.get("best_params")
    fields = data.get("features")
    if not isinstance(params, dict) or not isinstance(fields, list) or not fields:
        raise AssertionError(f"winner {path} lacks parameters or frozen features")
    if len(fields) > 120 or len(set(fields)) != len(fields):
        raise AssertionError("frozen head contract must contain 1..120 unique fields")
    return {"params": params, "features": [str(value) for value in fields], "trial": int(data["best_trial"])}


def _fit_and_score(
    *, train: pd.DataFrame, held: pd.DataFrame, fields: list[str], head: str,
    trial_params: dict[str, Any], feature_root: Path, seed: int,
) -> tuple[np.ndarray, int, float]:
    fit, valid = _split_early(train.sort_values(["__decision_ts__", "candidate_id"], kind="stable"))
    selected = pd.concat([fit, valid, held], ignore_index=True)
    matrix = _impute(_materialize(feature_root, selected, fields), len(fit))
    fit_end = len(fit)
    valid_end = fit_end + len(valid)
    fixed_trial = optuna.trial.FixedTrial(trial_params)
    params = _params(fixed_trial, head=head, rows=len(fit))
    model = _model(params, head=head, seed=seed, jobs=2)
    callbacks = []
    import lightgbm as lgb
    callbacks.append(lgb.early_stopping(30, verbose=False))
    target = str(HEADS[head]["target"])
    if head == "B":
        group_fit = fit.groupby("__decision_ts__", sort=False).size().to_numpy(np.int32)
        group_valid = valid.groupby("__decision_ts__", sort=False).size().to_numpy(np.int32)
        model.fit(
            matrix[:fit_end], pd.to_numeric(fit[target], errors="raise").to_numpy(np.int32), group=group_fit,
            eval_set=[(matrix[fit_end:valid_end], pd.to_numeric(valid[target], errors="raise").to_numpy(np.int32))],
            eval_group=[group_valid], callbacks=callbacks,
        )
    else:
        model.fit(
            matrix[:fit_end], pd.to_numeric(fit[target], errors="raise").to_numpy(float),
            eval_set=[(matrix[fit_end:valid_end], pd.to_numeric(valid[target], errors="raise").to_numpy(float))], callbacks=callbacks,
        )
    raw = np.asarray(model.predict(matrix[valid_end:]), dtype=float)
    score = float(HEADS[head]["direction"]) * raw
    complete_fraction = float(np.isfinite(matrix[valid_end:]).all(axis=1).mean())
    return score, int(model.best_iteration_ or 0), complete_fraction


def _b_window_without_score_geometry(
    *, feature_root: Path, router_root: Path, label_root: Path,
    policy: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
    route_fraction: float,
) -> pd.DataFrame:
    """Build the B-head fit window without irrelevant incumbent coordinates.

    The B ranker consumes only its frozen causal features, B labels, policy
    validity/availability gates, and the causal router.  Generic
    ``_window`` additionally joins B/E/T raw coordinates for cross-head
    metrics, which did not exist before April 2025 and are not inputs to this
    single-head score.  Omitting them lets the strict B reconstruction retain
    the same causal training rule instead of manufacturing a fake coordinate.
    """
    pieces: list[pd.DataFrame] = []
    for month in _source_months(start, end):
        feature_path = feature_root / f"month={month:%Y-%m}" / "causal_feature_universe.parquet"
        identities = pd.read_parquet(feature_path, columns=list(IDENTITY))
        identities["__decision_ts__"] = pd.to_datetime(identities["__decision_ts__"], utc=True, errors="raise")
        router = _read_router(router_root, month)
        labels = _read_head_labels("B", label_root, (month,))
        data = identities.merge(router, on=list(IDENTITY), how="inner", validate="one_to_one")
        if len(data) != len(identities):
            raise AssertionError(f"{month:%Y-%m}: target-free feature/router identity mismatch")
        data = data.merge(labels, on="candidate_id", how="left", validate="one_to_one")
        data = data.merge(policy, on="candidate_id", how="left", validate="one_to_one")
        data["router_selected"] = _route_top_fraction(data, route_fraction).to_numpy(bool)
        pieces.append(data)
    result = pd.concat(pieces, ignore_index=True)
    result = result.loc[result.__decision_ts__.ge(start) & result.__decision_ts__.lt(end)].copy()
    valid = str(HEADS["B"]["valid"])
    result[valid] = result[valid].fillna(False).astype(bool)
    result.policy_path_valid = result.policy_path_valid.fillna(False).astype(bool)
    result["label_joined"] = result[str(HEADS["B"]["target"])].notna()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", choices=tuple(HEADS), required=True)
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--router-root", type=Path, required=True)
    parser.add_argument("--score-root", type=Path)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path, required=True)
    # The frozen 1,407-field causal universe begins in April 2025 and the
    # frozen full-universe router begins in July.  November is therefore the
    # first held month that can use the declared July--September three-month
    # strict training window and the 28-day reserve without shortening either
    # source history.
    parser.add_argument("--start", default="2025-11-01")
    parser.add_argument("--end", default="2026-07-01")
    parser.add_argument("--route-fraction", type=float, default=.50)
    parser.add_argument("--train-months", type=int, default=3)
    parser.add_argument("--reserve-days", type=int, default=28)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--min-train-rows", type=int, default=8_000)
    parser.add_argument(
        "--b-head-no-score-geometry", action="store_true",
        help=(
            "for head B only, omit the generic raw B/E/T coordinate join; "
            "those columns are not B-model inputs and may be unavailable in "
            "otherwise valid early history"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if not 0.0 < args.route_fraction <= 1.0:
        raise ValueError("route fraction must be in (0, 1]")
    if args.b_head_no_score_geometry and args.head != "B":
        raise ValueError("--b-head-no-score-geometry is valid only for head B")
    if args.score_root is None and not args.b_head_no_score_geometry:
        raise ValueError("--score-root is required unless B uses --b-head-no-score-geometry")
    start, end = _utc(args.start), _utc(args.end)
    if end < start:
        raise ValueError("end must not precede start")
    winner = _winner(args.winner, args.head)
    policy = _read_policy(args.policy_path)
    args.out.mkdir(parents=True)
    _exclusive(args.out / "run_manifest.json", {
        "schema": "strict_r3_frozen_head_oof_v1",
        "scope": "offline research-only target-free frozen-head scoring; no live/inference/consensus/MC1/admission/portfolio/execution mutation",
        "head": args.head,
        "winner": str(args.winner), "winner_sha256": _sha(args.winner),
        "features": winner["features"], "feature_count": len(winner["features"]),
        "feature_contract_sha256": hashlib.sha256("\n".join(winner["features"]).encode()).hexdigest(),
        "score_months": [f"{month:%Y-%m}" for month in _months(start, end)],
        "strict_train": {"train_months": args.train_months, "reserve_days": args.reserve_days, "labels_resolved_before_reserve": True, "train_cap": args.train_cap},
        "held_population": "every point-in-time router-selected candidate; no policy path or label-validity eligibility filter",
        "held_output": "identity, decision timestamp, side, strict-OOF score, training cutoff and feature-coverage audit only",
        "target_fields_in_output": False,
        "score_geometry_mode": "none_for_single_B_head" if args.b_head_no_score_geometry else "generic_BET_coordinate_join",
    })
    audit: list[dict[str, Any]] = []
    for index, month in enumerate(_months(start, end)):
        reserve = month - pd.Timedelta(days=args.reserve_days)
        source_start = reserve - pd.DateOffset(months=args.train_months)
        window = (
            _b_window_without_score_geometry(
                feature_root=args.feature_root, router_root=args.router_root,
                label_root=args.label_root, policy=policy, start=source_start,
                end=_next_month(month), route_fraction=args.route_fraction,
            )
            if args.b_head_no_score_geometry
            else _window(
                head=args.head, feature_root=args.feature_root, router_root=args.router_root,
                score_root=args.score_root, label_root=args.label_root, policy=policy,
                start=source_start, end=_next_month(month), route_fraction=args.route_fraction,
            )
        )
        train = _train_rows(window.loc[window.__decision_ts__.lt(reserve)].copy(), args.head, reserve, args.train_cap)
        held = window.loc[
            window.__decision_ts__.ge(month)
            & window.__decision_ts__.lt(_next_month(month))
            & window.router_selected.fillna(False).astype(bool)
        ].copy()
        if len(train) < args.min_train_rows:
            raise AssertionError(f"{month:%Y-%m}: insufficient strict train rows {len(train)}")
        if held.empty:
            raise AssertionError(f"{month:%Y-%m}: no target-free router-selected candidates")
        score, iteration, finite_fraction = _fit_and_score(
            train=train, held=held, fields=winner["features"], head=args.head,
            trial_params=winner["params"], feature_root=args.feature_root, seed=SEED + index,
        )
        output = held.loc[:, [*IDENTITY]].copy()
        output["held_month"] = f"{month:%Y-%m}"
        output["head_score"] = score.astype(np.float32)
        output["train_reserve_end"] = reserve
        output["best_iteration"] = np.int16(iteration)
        target = args.out / f"month={month:%Y-%m}"
        target.mkdir()
        output.to_parquet(target / "target_free_scores.parquet", index=False, compression="zstd")
        audit.append({
            "head": args.head, "month": f"{month:%Y-%m}", "train_rows": int(len(train)),
            "held_rows": int(len(held)), "held_timestamps": int(held.__decision_ts__.nunique()),
            "best_iteration": iteration, "held_matrix_finite_fraction_after_train_imputation": finite_fraction,
            "reserve_end": reserve.isoformat(), "held_target_free": True,
        })
        print(json.dumps({"event": "month_complete", **audit[-1]}), flush=True)
    pd.DataFrame(audit).to_parquet(args.out / "score_audit.parquet", index=False, compression="zstd")


if __name__ == "__main__":
    main()
