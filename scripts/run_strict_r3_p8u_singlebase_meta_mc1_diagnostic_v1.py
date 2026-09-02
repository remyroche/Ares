#!/usr/bin/env python3
"""Strict short-warm-up Base -> Meta -> MC1 diagnostic for the P8u challenger.

This is intentionally *not* a promotion or live consumer.  The new P8u Base
ledger begins in November 2025 and its first four-month Meta histories begin
in April 2026, so a normal six-month MC1 fit is not yet possible on a common
new-Base representation.  This diagnostic uses only two strictly earlier
months to test whether the retained under-confidence Meta head has any
conservative mapping value over the new Base.  It cannot manufacture rank
authority: the auction is still ranked by the frozen Base score.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from run_strict_r3_incumbent_meta_mc1_screen_v1 import (
    IDENTITY,
    POLICY_COLUMNS,
    _portfolio,
    _read_policy,
    _robust_mean,
)


SCHEMA = "strict_r3_p8u_singlebase_meta_mc1_diagnostic_v1"
SEED = 1729
ADMISSION_BPS = 50.0
SHIFT_DAYS = 21
PROHIBITED = frozenset({
    "policy_path_valid", "policy_gross_bps", "policy_net_bps",
    "policy_exit_bar_15m", "policy_entry_price", "policy_exit_price",
    "policy_exit_reason", "policy_label_available_ts", "policy_cost_bps",
})


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _months(raw: str) -> tuple[pd.Timestamp, ...]:
    result = tuple(pd.Timestamp(f"{item.strip()}-01", tz="UTC") for item in raw.split(",") if item.strip())
    if len(result) < 2 or tuple(sorted(set(result))) != result:
        raise ValueError("need at least two unique, chronological held months")
    return result


def _month_end(month: pd.Timestamp) -> pd.Timestamp:
    return month + pd.offsets.MonthBegin(1)


def _path(root: Path, month: pd.Timestamp) -> Path:
    return root / f"month={month:%Y-%m}" / "scores_features.parquet"


def _meta_path(root: Path, arm: str, month: pd.Timestamp) -> Path:
    return root / "target_free_scores" / arm / f"month={month:%Y-%m}.parquet"


def _assert_target_free(path: Path) -> None:
    leaked = sorted(PROHIBITED.intersection(pq.ParquetFile(path).schema_arrow.names))
    if leaked:
        raise AssertionError(f"{path}: target-free source leaks policy fields {leaked}")


def _rank_bands(frame: pd.DataFrame) -> np.ndarray:
    work = frame.loc[:, ["candidate_id", "__decision_ts__", "base_rank_ts"]].copy()
    work["row"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["__decision_ts__", "base_rank_ts", "candidate_id"], ascending=[True, False, True], kind="stable")
    ordinal = work.groupby("__decision_ts__", sort=False).cumcount().to_numpy(float)
    size = work.groupby("__decision_ts__", sort=False).candidate_id.transform("size").to_numpy(float)
    work["score_band"] = np.minimum(9, (10.0 * (ordinal + .5) / size).astype(np.int8))
    return work.sort_values("row", kind="stable")["score_band"].to_numpy(np.int8)


def _read_target_free(base_root: Path, meta_root: Path, arm: str, month: pd.Timestamp) -> pd.DataFrame:
    source, meta = _path(base_root, month), _meta_path(meta_root, arm, month)
    _assert_target_free(source)
    _assert_target_free(meta)
    base = pd.read_parquet(source, columns=[*IDENTITY, "enhanced_base_bps", "base_rank_ts", "enhanced_base_routed"])
    head = pd.read_parquet(meta, columns=[*IDENTITY, "meta_rank_ts", "meta_raw_score", "arm"])
    for frame in (base, head):
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        if frame.duplicated(IDENTITY).any() or not frame.side_name.eq("long").all():
            raise AssertionError(f"{month:%Y-%m}: invalid target-free identity")
    if not base.enhanced_base_routed.fillna(False).astype(bool).all():
        raise AssertionError(f"{month:%Y-%m}: Base source reintroduced a post-router cutoff")
    out = base.merge(head, on=list(IDENTITY), how="inner", validate="one_to_one")
    if len(out) != len(base) or not out.arm.eq(arm).all():
        raise AssertionError(f"{month:%Y-%m}: Meta source does not exactly cover Base identities")
    return out.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)


def _fit(frame: pd.DataFrame, features: tuple[str, ...]) -> tuple[HistGradientBoostingRegressor, pd.Series, np.ndarray]:
    work = frame.copy()
    work["score_band"] = _rank_bands(work)
    work["day"] = work.__decision_ts__.dt.normalize()
    samples: list[pd.DataFrame] = []
    for _day, group in work.groupby("day", sort=True):
        ordered = group.sort_values(["__decision_ts__", "base_rank_ts", "candidate_id"], ascending=[True, False, True], kind="stable")
        samples.append(pd.concat([ordered.head(50), ordered.iloc[50:].sample(min(250, max(0, len(ordered) - 50)), random_state=SEED)]))
    work = pd.concat(samples, ignore_index=True)
    raw = pd.to_numeric(work.policy_net_bps, errors="coerce")
    low, high = raw.quantile([.02, .98])
    work["target"] = raw.clip(low, high)
    if len(work) > 50_000:
        work = work.sample(50_000, random_state=SEED)
    medians = work.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").median().fillna(0.0)
    x = work.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").fillna(medians)
    model = HistGradientBoostingRegressor(
        max_depth=2, max_iter=80, learning_rate=.04, l2_regularization=20.0,
        min_samples_leaf=100, random_state=SEED,
    ).fit(x, work.target)
    global_mean = _robust_mean(work.target)
    curve = np.full(10, global_mean, dtype=float)
    for band, group in work.groupby("score_band", sort=True):
        mean, sd, n = float(group.target.mean()), max(float(group.target.std(ddof=0)), 1.0), len(group)
        precision, prior = n / (sd * sd + 1.0), 80.0 / (250.0 ** 2)
        curve[int(band)] = (precision * mean + prior * global_mean) / (precision + prior)
    curve = -IsotonicRegression(increasing=True).fit_transform(np.arange(10), -curve)
    return model, medians, curve


def _score_arm(
    *, target_free: pd.DataFrame, policy: pd.DataFrame, features: tuple[str, ...], held_months: tuple[pd.Timestamp, ...], history_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labelled = target_free.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    if len(labelled) != len(target_free):
        raise AssertionError("outcome join changed target-free identities")
    labelled["policy_path_valid"] = labelled.policy_path_valid.fillna(False).astype(bool)
    labelled["score_band"] = _rank_bands(labelled)
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for month in held_months:
        start, end = month - pd.DateOffset(months=history_months), _month_end(month)
        train = labelled.loc[
            labelled.__decision_ts__.ge(start) & labelled.__decision_ts__.lt(month)
            & labelled.policy_path_valid & labelled.policy_label_available_ts.lt(month)
            & np.isfinite(pd.to_numeric(labelled.policy_net_bps, errors="coerce"))
        ].copy()
        held = labelled.loc[labelled.__decision_ts__.ge(month) & labelled.__decision_ts__.lt(end)].copy()
        if len(train) < 30_000 or held.empty:
            audits.append({"month": f"{month:%Y-%m}", "status": "insufficient", "train_rows": len(train), "held_rows": len(held)})
            continue
        model, medians, curve = _fit(train, features)
        x = held.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").fillna(medians)
        held["static_expected_bps"] = model.predict(x)
        shifts: dict[pd.Timestamp, float] = {}
        for day in pd.date_range(month.normalize(), (end - pd.Timedelta(days=1)).normalize(), freq="D", tz="UTC"):
            history = labelled.loc[
                labelled.__decision_ts__.ge(day - pd.Timedelta(days=SHIFT_DAYS)) & labelled.__decision_ts__.lt(day)
                & labelled.policy_path_valid & labelled.policy_label_available_ts.lt(day)
                & np.isfinite(pd.to_numeric(labelled.policy_net_bps, errors="coerce"))
            ]
            residual = pd.to_numeric(history.policy_net_bps, errors="coerce").to_numpy(float) - curve[history.score_band.to_numpy(int)]
            shifts[day] = _robust_mean(residual, trim=.10) if len(residual) else 0.0
        held["recent_shift_bps"] = held.__decision_ts__.dt.normalize().map(shifts).fillna(0.0)
        held["mc1_expected_bps"] = held.static_expected_bps + held.recent_shift_bps
        predictions.append(held)
        audits.append({"month": f"{month:%Y-%m}", "status": "scored", "train_rows": len(train), "held_rows": len(held), "features": list(features)})
    if not predictions:
        raise AssertionError("no short-warm-up MC1 held month could be scored")
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audits)


def _persist_target_free(out: Path, name: str, frame: pd.DataFrame) -> None:
    retained = frame.loc[:, [*IDENTITY, "enhanced_base_bps", "base_rank_ts", "meta_rank_ts", "meta_raw_score", "arm"]].copy()
    if PROHIBITED.intersection(retained.columns):
        raise AssertionError("target-free panel leak")
    retained.to_parquet(out / f"target_free_{name}.parquet", index=False, compression="zstd")


def _run_portfolio(out: Path, name: str, prediction: pd.DataFrame) -> tuple[dict[str, object], pd.DataFrame]:
    work = prediction.copy()
    # The inherited constrained replay requires two mapper columns.  They are
    # deliberately identical here: this is one conservative new-Base mapper,
    # not a claim of separate BCF/current confirmations.
    work["current_mc1_expected_bps"] = work.mc1_expected_bps
    work["bcf_mc1_expected_bps"] = work.mc1_expected_bps
    metrics, decisions = _portfolio(work, arm=name, out=out)
    return metrics, decisions


def run(*, base_root: Path, meta_root: Path, policy_path: Path, arm: str, out: Path, held_months: tuple[pd.Timestamp, ...], history_months: int) -> Path:
    if out.exists():
        raise FileExistsError("immutable output exists")
    if history_months < 2 or history_months > 3:
        raise ValueError("short diagnostic must use exactly two or three prior months")
    source_months = tuple(pd.date_range(held_months[0] - pd.DateOffset(months=history_months), held_months[-1], freq="MS", tz="UTC"))
    out.mkdir(parents=True)
    sources = [_read_target_free(base_root, meta_root, arm, month) for month in source_months]
    target_free = pd.concat(sources, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    _persist_target_free(out, "base_plus_under", target_free)
    base_only = target_free.copy()
    base_only["meta_rank_ts"] = .5
    base_only["meta_raw_score"] = 0.0
    _persist_target_free(out, "base_only", base_only)
    policy = _read_policy(policy_path)
    variants = {
        "base_only": (base_only, ("base_rank_ts", "enhanced_base_bps")),
        "base_plus_under": (target_free, ("base_rank_ts", "enhanced_base_bps", "meta_rank_ts")),
    }
    summaries: list[dict[str, object]] = []
    for name, (panel, features) in variants.items():
        prediction, audit = _score_arm(target_free=panel, policy=policy, features=features, held_months=held_months, history_months=history_months)
        target_free_prediction = prediction.loc[:, [*IDENTITY, "base_rank_ts", "mc1_expected_bps", "static_expected_bps", "recent_shift_bps"]]
        target_free_prediction.to_parquet(out / f"target_free_mc1_{name}.parquet", index=False, compression="zstd")
        metrics, decisions = _run_portfolio(out, name, prediction)
        audit.to_parquet(out / f"mc1_{name}_audit.parquet", index=False, compression="zstd")
        decisions.to_parquet(out / f"portfolio_{name}.parquet", index=False, compression="zstd")
        summaries.append({"variant": name, "features": list(features), **metrics})
    pd.DataFrame(summaries).to_parquet(out / "portfolio_summary.parquet", index=False, compression="zstd")
    _once(out / "correctness_report.json", {
        "all_source_panels_target_free_before_policy_join": True,
        "base_meta_identity_exact": True,
        "base_rank_has_only_router50_rows": True,
        "all_mc1_train_labels_resolved_before_held_month": True,
        "daily_shift_uses_only_prior_resolved_labels": True,
        "mc1_has_demotion_only_authority": True,
        "short_warmup_is_explicitly_nonpromotion": True,
        "no_live_exchange_or_canonical_mutation": True,
    })
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline, short-warm-up new-Base Meta/MC1 diagnostic only; never a live promotion input",
        "limitation": "normal six-month new-Base Meta history is unavailable before later data; this diagnostic uses only two prior months and is not comparable to the live dual-MC1 contract",
        "base_root": str(base_root), "meta_root": str(meta_root), "policy_path": str(policy_path),
        "meta_arm": arm, "held_months": [f"{month:%Y-%m}" for month in held_months],
        "history_months": history_months, "admission_bps": ADMISSION_BPS,
        "authority": "Meta rank enters MC1 expected-value mapping only; auction remains Base-rank ordered",
        "variants": ["base_only", "base_plus_under"],
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--meta-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--arm", default="under_atr1__timestamp")
    parser.add_argument("--held-months", default="2026-06,2026-07")
    parser.add_argument("--history-months", type=int, default=2)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(base_root=args.base_root.resolve(), meta_root=args.meta_root.resolve(), policy_path=args.policy.resolve(), arm=args.arm, out=args.out.resolve(), held_months=_months(args.held_months), history_months=args.history_months))


if __name__ == "__main__":
    main()
