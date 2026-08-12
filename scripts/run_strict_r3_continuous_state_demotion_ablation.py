#!/usr/bin/env python3
"""Causal continuous-market-state shrinker/demotion ablation.

This deliberately does *not* add another general ranker.  Market-state fields
are largely common to all candidates at a decision time, so their plausible
role is to demote a score during an adverse conversion environment.  Each
monthly fold therefore fits a regularised state-only residual model on prior,
resolved high-score rows and subtracts only its negative prediction from the
score's train-only expected-policy value.

All output rankings are cross-sectional/global diagnostics.  The model has no
access to a held outcome, a held-month percentile, or a fold-local latent state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


SEED = 20260811
TAILS = (0.005, 0.01, 0.02, 0.05)
TRAIN_FRACTIONS = (0.20, 0.30)
DEMO_ALPHAS = (0.25, 0.50, 1.00)


def _fields(frame: pd.DataFrame) -> list[str]:
    fields = [column for column in frame if column.startswith("continuous_regime__")]
    if len(fields) < 40:
        raise ValueError(f"expected continuous state contract, found only {len(fields)} fields")
    bad = [
        field for field in fields
        if frame[field].notna().mean() < 0.90
        or not np.isfinite(pd.to_numeric(frame[field], errors="coerce").var())
        or pd.to_numeric(frame[field], errors="coerce").var() <= 1e-12
    ]
    if bad:
        raise ValueError(f"continuous state coverage/variance gate failed: {bad}")
    return fields


def _parent_expected(train: pd.DataFrame, apply: pd.DataFrame) -> np.ndarray:
    """Train-only monotone score-to-policy map, robust to sparse score bins."""

    score = pd.to_numeric(train["final_score"], errors="coerce").to_numpy(float)
    target = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float)
    usable = np.isfinite(score) & np.isfinite(target)
    if usable.sum() < 1_000:
        raise ValueError("insufficient prior rows for train-only parent map")
    # Equal-frequency bins avoid fitting a flexible response curve on the held
    # month.  Linear interpolation is intentionally conservative and uses only
    # previous resolved scores/outcomes.
    edges = np.unique(np.quantile(score[usable], np.linspace(0.0, 1.0, 21)))
    if len(edges) < 3:
        return np.full(len(apply), float(np.nanmean(target[usable])), dtype=float)
    bins = np.clip(np.searchsorted(edges, score[usable], side="right") - 1, 0, len(edges) - 2)
    table = pd.DataFrame({"bin": bins, "target": target[usable]}).groupby("bin", sort=True)["target"].mean()
    centers = 0.5 * (edges[:-1] + edges[1:])
    values = np.asarray([table.get(index, np.nan) for index in range(len(centers))], dtype=float)
    fallback = float(np.nanmean(target[usable]))
    values = pd.Series(values).interpolate(limit_direction="both").fillna(fallback).to_numpy(float)
    # PAVA-like cumulative maximum keeps the score interpretation monotone.
    values = np.maximum.accumulate(values)
    apply_score = pd.to_numeric(apply["final_score"], errors="coerce").to_numpy(float)
    return np.interp(apply_score, centers, values, left=values[0], right=values[-1])


def _folds(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp):
    cutoff = start
    while cutoff <= end:
        held_end = cutoff + pd.offsets.MonthBegin(1)
        train_start = cutoff - pd.DateOffset(months=3)
        train = frame.loc[
            frame["__decision_ts__"].ge(train_start)
            & frame["__decision_ts__"].lt(cutoff)
            & frame["policy_label_available_ts"].lt(cutoff)
            & frame["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(frame["policy_net_bps"], errors="coerce"))
        ].copy()
        held = frame.loc[
            frame["__decision_ts__"].ge(cutoff) & frame["__decision_ts__"].lt(held_end)
        ].copy()
        if len(train) >= 10_000 and len(held):
            yield cutoff, train, held
        cutoff = held_end


def _sample_equal_month(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame
    months = frame["__decision_ts__"].dt.to_period("M").astype(str)
    rng = np.random.default_rng(seed)
    quota = max(1, cap // months.nunique())
    pieces = []
    for month in sorted(months.unique()):
        block = frame.loc[months.eq(month)]
        if len(block) > quota:
            block = block.iloc[np.sort(rng.choice(len(block), quota, replace=False))]
        pieces.append(block)
    return pd.concat(pieces, ignore_index=True)


def _metrics(frame: pd.DataFrame, score_column: str, arm: str, period_kind: str) -> pd.DataFrame:
    groups = [("global", frame)] if period_kind == "global" else list(frame.groupby(frame["__decision_ts__"].dt.strftime("%Y-%m"), sort=True))
    rows = []
    for period, work in groups:
        ordered = work.sort_values([score_column, "candidate_id"], ascending=[False, True], kind="stable")
        for tail in TAILS:
            selected = ordered.iloc[:max(1, int(np.ceil(len(ordered) * tail)))].copy()
            valid = selected.loc[selected["policy_path_valid"].fillna(False).astype(bool) & np.isfinite(pd.to_numeric(selected["policy_net_bps"], errors="coerce"))]
            rows.append({
                "arm": arm, "period_kind": period_kind, "period": str(period), "tail": tail,
                "population_rows": len(work), "selected_score_rows": len(selected),
                "valid_outcomes": len(valid), "outcome_coverage": float(len(valid) / len(selected)),
                "net_bps_per_trade": float(pd.to_numeric(valid["policy_net_bps"], errors="coerce").mean()) if len(valid) else np.nan,
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fold-start", required=True)
    parser.add_argument("--fold-end", required=True)
    parser.add_argument("--train-cap", type=int, default=80_000)
    parser.add_argument("--ridge-alpha", type=float, default=250.0)
    parser.add_argument(
        "--equal-timestamp-weighting", action="store_true",
        help="Give each decision timestamp equal total weight in the state fit.",
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    keep = [
        "candidate_id", "__decision_ts__", "policy_label_available_ts", "policy_path_valid",
        "policy_net_bps", "final_score", "geometry_bundle_sha256",
    ]
    schema = pq.ParquetFile(args.surface).schema.names
    continuous = [column for column in schema if column.startswith("continuous_regime__")]
    frame = pd.read_parquet(args.surface, columns=keep + continuous)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="coerce")
    if frame["candidate_id"].duplicated().any() or frame["geometry_bundle_sha256"].nunique() != 1:
        raise AssertionError("demotion ablation requires unique candidates and one frozen geometry bundle")
    fields = _fields(frame)
    start = pd.Timestamp(args.fold_start, tz="UTC")
    end = pd.Timestamp(args.fold_end, tz="UTC")
    predictions: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for ordinal, (cutoff, train_all, held) in enumerate(_folds(frame, start, end)):
        parent_train = _parent_expected(train_all, train_all)
        parent_held = _parent_expected(train_all, held)
        train_all["_parent"] = parent_train
        held["_parent"] = parent_held
        threshold = float(pd.to_numeric(train_all["final_score"], errors="coerce").quantile(0.70))
        for fraction in TRAIN_FRACTIONS:
            floor = float(pd.to_numeric(train_all["final_score"], errors="coerce").quantile(1.0 - fraction))
            train = train_all.loc[pd.to_numeric(train_all["final_score"], errors="coerce").ge(floor)].copy()
            train = _sample_equal_month(train, int(args.train_cap), SEED + ordinal)
            target = (pd.to_numeric(train["policy_net_bps"], errors="coerce") - train["_parent"]).clip(-800.0, 800.0).to_numpy(float)
            # State values are principally shared by candidates at the same
            # timestamp.  Equal timestamp mass prevents a busy cross-section
            # from being mistaken for repeated independent regime evidence.
            model = make_pipeline(
                SimpleImputer(strategy="median"), RobustScaler(),
                Ridge(alpha=float(args.ridge_alpha), solver="lsqr", max_iter=4_000),
            )
            if args.equal_timestamp_weighting:
                counts = train.groupby("__decision_ts__", observed=True)["candidate_id"].transform("size")
                sample_weight = 1.0 / counts.to_numpy(float)
                sample_weight *= len(sample_weight) / sample_weight.sum()
                model.fit(train.loc[:, fields], target, ridge__sample_weight=sample_weight)
            else:
                model.fit(train.loc[:, fields], target)
            residual = model.predict(held.loc[:, fields])
            output = held.loc[:, ["candidate_id", "__decision_ts__", "policy_path_valid", "policy_net_bps", "final_score"]].copy()
            output["parent_expected_bps"] = parent_held
            output["state_residual_bps"] = residual
            output["fold"] = ordinal
            output["train_fraction"] = fraction
            output["trust_gate_floor"] = threshold
            output["score__control"] = output["parent_expected_bps"]
            for alpha in DEMO_ALPHAS:
                # Only adverse state predictions have authority: a favourable
                # state never promotes a candidate above its score-derived EV.
                output[f"score__demote_{fraction:.2f}_a{alpha:.2f}"] = (
                    output["parent_expected_bps"] - alpha * np.maximum(-residual, 0.0)
                )
            predictions.append(output)
            audit.append({
                "fold": ordinal, "cutoff": cutoff.isoformat(), "train_fraction": fraction,
                "train_rows": len(train), "held_rows": len(held), "state_fields": len(fields),
                "negative_state_prediction_rate": float((residual < 0.0).mean()),
                "state_prediction_sd_bps": float(np.std(residual)),
                "equal_timestamp_weighting": bool(args.equal_timestamp_weighting),
            })
    if not predictions:
        raise ValueError("no evaluable folds")
    output = pd.concat(predictions, ignore_index=True)
    output.to_parquet(args.out_dir / "oof_predictions.parquet", index=False, compression="zstd")
    metrics = []
    for score_column in [column for column in output if column.startswith("score__")]:
        arm = score_column.removeprefix("score__")
        metrics.extend([_metrics(output, score_column, arm, "global"), _metrics(output, score_column, arm, "month")])
    all_metrics = pd.concat(metrics, ignore_index=True)
    all_metrics.to_parquet(args.out_dir / "metrics.parquet", index=False, compression="zstd")
    pd.DataFrame(audit).to_parquet(args.out_dir / "fold_audit.parquet", index=False, compression="zstd")
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "schema": "strict_r3_continuous_state_demotion_v1",
        "surface": str(args.surface), "continuous_fields": fields,
        "target": "winsorized policy_net_bps minus train-only monotone final_score expectation",
        "authority": "demotion only; score expected value minus alpha * max(-state_residual, 0)",
        "train_fractions": list(TRAIN_FRACTIONS), "alphas": list(DEMO_ALPHAS),
        "fold_start": args.fold_start, "fold_end": args.fold_end,
        "equal_timestamp_weighting": bool(args.equal_timestamp_weighting),
        "geometry_bundle_sha256": str(frame["geometry_bundle_sha256"].iloc[0]),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
