#!/usr/bin/env python3
"""Offline strict-prequential B3/B4/B5 base-utility funnel for long Strict-R3.

This runner consumes the immutable B0 reconstruction produced by
``run_strict_r3_base_recall_funnel.py``.  It never changes or imports a live
bundle.  Every utility model is fit only to policy labels whose availability
precedes its block cutoff, with the preceding 28 calendar days held out from
all supervised fitting.  It scores target-free held candidates first; policy
outcomes are joined only for diagnostics after timestamp-local route
membership is fixed.

It implements the remaining Funnel-A arms:

* B3: U50/U150 shallow classifiers, day-balanced and equal-month objectives,
  fused with the frozen D2 score at four predeclared weights.
* B4: absolute and hybrid policy-net ordinal LambdaRank heads, same fusions.
* B5: 20/10 and 15/15 D2/utility quota routes using the development-selected
  B3 utility model without changing the D2 coordinate.

The producer is a screening step only.  It does not rebuild downstream
residuals, MC1, portfolio admission, exits, or any inference artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_r3_base_recall_funnel import (  # noqa: E402
    BASE_ROUTE_FRACTION,
    DEFAULT_CONTROL,
    DEFAULT_POLICY,
    DEFAULT_SOURCE,
    PERIODS,
    _utc,
    timestamp_route,
)


DEFAULT_B0_ROOT = ROOT / (
    "data_perp/artifacts/strict_r3_long_base_recall_funnel_2025dev_holdout_"
    "2026oos_20260822_v1"
)
SEED = 1729
MAX_TRAIN_ROWS = 240_000
FUSION_WEIGHTS = (.10, .25, .40, .50)


@dataclass(frozen=True)
class TrainWeight:
    name: str
    description: str


WEIGHTS = (
    TrainWeight("day_balanced", "Each resolved policy day has equal total loss weight."),
    TrainWeight("equal_month", "Each resolved calendar month has equal total loss weight."),
)


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def stable_feature_contract(control_root: Path) -> tuple[str, ...]:
    bundles = sorted(control_root.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib"))
    if not bundles:
        raise FileNotFoundError("no frozen current-v5 upstream bundles")
    contracts = {tuple(joblib.load(path).base_fields) for path in bundles}
    if len(contracts) != 1:
        raise AssertionError("the B3/B4 funnel requires one stable frozen base feature order")
    fields = next(iter(contracts))
    if len(fields) != 120:
        raise AssertionError(f"expected frozen 120-field contract, found {len(fields)}")
    return fields


def canonical_policy_panel(policy: Path) -> pd.DataFrame:
    """Load the sole permitted rich-parent supervision substrate.

    The feature ledger contains historical policy-like columns from an older
    materialisation.  They are deliberately not trusted for B3/B4/B5 targets:
    only this versioned source-aligned parent-policy panel defines the target.
    """

    labels = pd.read_parquet(policy, columns=[
        "candidate_id", "policy_path_valid", "policy_net_bps", "policy_label_available_ts",
    ])
    if labels["candidate_id"].duplicated().any():
        raise AssertionError("canonical policy candidate IDs must be unique")
    labels["policy_path_valid"] = labels["policy_path_valid"].fillna(False).astype(bool)
    labels["policy_net_bps"] = pd.to_numeric(labels["policy_net_bps"], errors="coerce")
    labels["policy_label_available_ts"] = pd.to_datetime(
        labels["policy_label_available_ts"], utc=True, errors="coerce",
    )
    return labels


def load_raw_panel(source: Path, policy: Path, features: tuple[str, ...]) -> pd.DataFrame:
    columns = ["candidate_id", "__decision_ts__", *features]
    frame = pd.read_parquet(source, columns=columns)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    phase = frame["__decision_ts__"]
    if ((phase.dt.minute != 0) | (phase.dt.second != 0) | (phase.dt.microsecond != 0)).any():
        raise AssertionError("B3/B4/B5 source contains a non-:00 research decision")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError("raw source candidate IDs must be unique")
    return frame.merge(canonical_policy_panel(policy), on="candidate_id", how="left", validate="one_to_one")


def deterministic_day_sample(frame: pd.DataFrame, *, max_rows: int) -> pd.DataFrame:
    """Cap fitting rows evenly across resolved days without changing prevalence."""
    if len(frame) <= max_rows:
        return frame.copy()
    work = frame.copy()
    work["__day__"] = work["__decision_ts__"].dt.floor("D")
    days = int(work["__day__"].nunique())
    per_day = max(1, int(math.ceil(max_rows / days)))
    hashes = pd.util.hash_pandas_object(work["candidate_id"], index=False).to_numpy(np.uint64)
    work["__hash__"] = hashes
    sampled = (
        work.sort_values(["__day__", "__hash__", "candidate_id"], kind="stable")
        .groupby("__day__", sort=False, group_keys=False)
        .head(per_day)
        .drop(columns=["__day__", "__hash__"])
    )
    return sampled.iloc[:max_rows].copy()


def supervised_rows(raw: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    reserve_start = cutoff - pd.Timedelta(days=28)
    eligible = raw.loc[
        raw["__decision_ts__"].lt(reserve_start)
        & raw["policy_path_valid"]
        & raw["policy_net_bps"].notna()
        # The preceding 28 days are the bundle's calibration reserve.  A
        # policy outcome resolved during that window cannot enter a utility
        # head fit, even when its decision timestamp predates the window.
        & raw["policy_label_available_ts"].lt(reserve_start),
    ].copy()
    if eligible.empty:
        raise ValueError(f"no strict-prequential policy rows before {cutoff.isoformat()}")
    return deterministic_day_sample(eligible, max_rows=MAX_TRAIN_ROWS).reset_index(drop=True)


def training_weights(frame: pd.DataFrame, mode: TrainWeight) -> np.ndarray:
    if mode.name == "day_balanced":
        bucket = frame["__decision_ts__"].dt.floor("D")
    elif mode.name == "equal_month":
        bucket = frame["__decision_ts__"].dt.to_period("M").astype(str)
    else:
        raise ValueError(mode.name)
    counts = bucket.value_counts(dropna=False)
    weights = bucket.map(1.0 / counts).to_numpy(float)
    return weights / weights.mean()


def classifier_params() -> dict[str, object]:
    return {
        "objective": "binary",
        "n_estimators": 180,
        "learning_rate": .05,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 1_000,
        "subsample": .80,
        "subsample_freq": 1,
        "colsample_bytree": .80,
        "reg_lambda": 8.0,
        "random_state": SEED,
        "n_jobs": 4,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
    }


def ranker_params() -> dict[str, object]:
    return {
        "objective": "lambdarank",
        "n_estimators": 180,
        "learning_rate": .05,
        "max_depth": 4,
        "num_leaves": 15,
        "min_child_samples": 1_000,
        "subsample": .80,
        "subsample_freq": 1,
        "colsample_bytree": .80,
        "reg_lambda": 8.0,
        "random_state": SEED,
        "n_jobs": 4,
        "deterministic": True,
        "force_col_wise": True,
        "verbosity": -1,
        "lambdarank_norm": True,
        "lambdarank_truncation_level": 20,
        "label_gain": [0.0, 1.0, 2.0, 4.0, 7.0, 12.0],
    }


def utility_targets(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    net = frame["policy_net_bps"].to_numpy(float)
    absolute = np.digitize(net, [-200.0, -50.0, 50.0, 150.0, 300.0], right=False)
    # The relative grade is derived within the training population's decision
    # timestamp only.  It is an outcome label, never an inference feature.
    rel_rank = frame.groupby("__decision_ts__", sort=False)["policy_net_bps"].rank(
        method="average", pct=True,
    ).to_numpy(float)
    relative = np.minimum(4, np.floor(rel_rank * 5.0).astype(int))
    hybrid = np.rint((absolute + relative) / 2.0).astype(int)
    hybrid = np.minimum(5, hybrid)
    hybrid[net <= 0.0] = np.minimum(3, hybrid[net <= 0.0])
    return net >= 50.0, net >= 150.0, absolute, hybrid


def fit_block_predictions(
    train: pd.DataFrame,
    held: pd.DataFrame,
    features: tuple[str, ...],
    weight: TrainWeight,
) -> pd.DataFrame:
    """Fit the four predeclared utility heads and score target-free held rows."""
    forbidden = {"policy_path_valid", "policy_net_bps", "policy_label_available_ts"}.intersection(held.columns)
    if forbidden:
        raise AssertionError(f"held utility scoring must be target-free, found {sorted(forbidden)}")
    x_train = train.loc[:, features]
    x_held = held.loc[:, features]
    sample_weight = training_weights(train, weight)
    u50, u150, absolute, hybrid = utility_targets(train)
    if min(u50.sum(), (~u50).sum(), u150.sum(), (~u150).sum()) < 100:
        raise ValueError("utility target lacks minimum class support")
    out = held.loc[:, ["candidate_id"]].copy()
    p50 = lgb.LGBMClassifier(**classifier_params()).fit(x_train, u50, sample_weight=sample_weight).predict_proba(x_held)[:, 1]
    p150 = lgb.LGBMClassifier(**classifier_params()).fit(x_train, u150, sample_weight=sample_weight).predict_proba(x_held)[:, 1]
    out[f"u_{weight.name}"] = .60 * p50 + .40 * p150
    order = train.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    group = order.groupby("__decision_ts__", sort=False)["candidate_id"].size().to_numpy(int)
    for target, name in ((absolute, "absolute"), (hybrid, "hybrid")):
        ordered_target = target[order.index.to_numpy()]
        ordered_weight = sample_weight[order.index.to_numpy()]
        model = lgb.LGBMRanker(**ranker_params()).fit(
            order.loc[:, features], ordered_target, group=group, sample_weight=ordered_weight,
        )
        out[f"r_{name}_{weight.name}"] = model.predict(x_held)
    return out


def timestamp_rank(frame: pd.DataFrame, column: str) -> pd.Series:
    rank = frame.groupby("__decision_ts__", sort=False)[column].rank(method="average")
    count = frame.groupby("__decision_ts__", sort=False)[column].transform("count")
    return (rank - .5) / count.clip(lower=1)


def quota_route(frame: pd.DataFrame, primary: str, secondary: str, primary_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-30% primary-plus-secondary route and source code."""
    if primary_fraction not in (.15, .20):
        raise ValueError("only predeclared B5 primary fractions are allowed")
    work = frame.loc[:, ["candidate_id", "__decision_ts__", primary, secondary]].copy()
    work["__row__"] = np.arange(len(work), dtype=int)
    work = work.sort_values(["__decision_ts__", primary, "candidate_id"], ascending=[True, False, True], kind="stable")
    work["__primary_rank__"] = work.groupby("__decision_ts__", sort=False).cumcount()
    count = work.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    primary_n = np.ceil(primary_fraction * count).astype(int)
    total_n = np.ceil(BASE_ROUTE_FRACTION * count).astype(int)
    work["__primary__"] = work["__primary_rank__"].lt(primary_n)
    rem = work.loc[~work["__primary__"]].copy().sort_values(
        ["__decision_ts__", secondary, "candidate_id"], ascending=[True, False, True], kind="stable",
    )
    rem["__secondary_rank__"] = rem.groupby("__decision_ts__", sort=False).cumcount()
    rem_need = (total_n - primary_n).groupby(work["__decision_ts__"], sort=False).first()
    rem["__secondary__"] = rem["__secondary_rank__"].lt(rem["__decision_ts__"].map(rem_need).astype(int))
    work = work.merge(rem.loc[:, ["__row__", "__secondary__"]], on="__row__", how="left", validate="one_to_one")
    work["__secondary__"] = work["__secondary__"].eq(True)
    work["__selected__"] = work["__primary__"] | work["__secondary__"]
    # Secondary selection excludes the primary D2 quota.  Keep the separate
    # utility-quota membership for the primary rows too, so later residual
    # analysis can distinguish D2-only from corroborated D2+utility routing
    # without changing the prescribed 30% selected set.
    full_utility = work.sort_values(
        ["__decision_ts__", secondary, "candidate_id"],
        ascending=[True, False, True], kind="stable",
    ).copy()
    full_utility["__utility_rank__"] = full_utility.groupby("__decision_ts__", sort=False).cumcount()
    secondary_n = (total_n - primary_n).groupby(work["__decision_ts__"], sort=False).first()
    full_utility["__utility_candidate__"] = full_utility["__utility_rank__"].lt(
        full_utility["__decision_ts__"].map(secondary_n).astype(int)
    )
    work = work.merge(
        full_utility.loc[:, ["__row__", "__utility_candidate__"]],
        on="__row__", how="left", validate="one_to_one",
    )
    work["__source__"] = np.select(
        [work["__primary__"] & work["__utility_candidate__"], work["__primary__"]],
        ["both", "D2"],
        default="utility",
    )
    work = work.sort_values("__row__", kind="stable")
    return work["__selected__"].to_numpy(bool), work["__source__"].to_numpy(object)


def add_outcome_labels(scores: pd.DataFrame, source: Path, policy: Path) -> pd.DataFrame:
    r3 = pd.read_parquet(source, columns=["candidate_id", "r3_class", "r3_label_available_ts"])
    if r3["candidate_id"].duplicated().any():
        raise AssertionError("source R3 candidate IDs must be unique")
    r3["r3_label_available_ts"] = pd.to_datetime(r3["r3_label_available_ts"], utc=True, errors="coerce")
    output = scores.merge(r3, on="candidate_id", how="left", validate="one_to_one")
    output = output.merge(canonical_policy_panel(policy), on="candidate_id", how="left", validate="one_to_one")
    output["policy_path_valid"] = output["policy_path_valid"].fillna(False).astype(bool)
    output["policy_net_bps"] = pd.to_numeric(output["policy_net_bps"], errors="coerce")
    valid = output["policy_path_valid"] & output["policy_net_bps"].notna()
    output["is_r3_clear"] = output["r3_class"].eq(2) & output["r3_label_available_ts"].notna()
    for level in (30, 50, 100, 200):
        output[f"policy_ge_{level}"] = valid & output["policy_net_bps"].ge(level)
    ordered = output.loc[valid, ["candidate_id", "__decision_ts__", "policy_net_bps"]].copy().sort_values(
        ["__decision_ts__", "policy_net_bps", "candidate_id"], ascending=[True, False, True], kind="stable",
    )
    rank = ordered.groupby("__decision_ts__", sort=False).cumcount()
    count = ordered.groupby("__decision_ts__", sort=False)["candidate_id"].transform("size")
    for fraction, name in ((.20, "positive_top20"), (.10, "positive_top10")):
        selected = ordered["policy_net_bps"].gt(0) & rank.lt(np.ceil(fraction * count).astype(int))
        output = output.merge(
            ordered.loc[:, ["candidate_id"]].assign(**{name: selected.to_numpy(bool)}),
            on="candidate_id", how="left", validate="one_to_one",
        )
        output[name] = output[name].fillna(False).astype(bool)
    return output


def rank_ic(frame: pd.DataFrame, score: str) -> float:
    valid = frame.loc[
        frame["policy_path_valid"] & frame["policy_net_bps"].notna(),
        ["__decision_ts__", score, "policy_net_bps"],
    ].copy()
    if valid.empty:
        return float("nan")
    valid["x"] = valid.groupby("__decision_ts__", sort=False)[score].rank()
    valid["y"] = valid.groupby("__decision_ts__", sort=False)["policy_net_bps"].rank()
    result = valid.groupby("__decision_ts__", sort=False)[["x", "y"]].corr().iloc[0::2, -1]
    return float(result.mean())


def recall(frame: pd.DataFrame, selected: np.ndarray, field: str) -> float:
    label = frame[field].fillna(False).to_numpy(bool)
    return float((selected & label).sum() / label.sum()) if label.any() else float("nan")


def metrics_for_routes(
    scored: pd.DataFrame,
    arms: Iterable[tuple[str, str, Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray | None]]]],
    periods: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    memberships: list[pd.DataFrame] = []
    for arm, family, make_route in arms:
        selected, source = make_route(scored)
        member = scored.loc[:, ["candidate_id", "__decision_ts__"]].copy()
        member["arm"] = arm
        member["routed"] = selected
        if source is not None:
            member["route_source"] = source
        memberships.append(member)
        score = f"score__{arm}"
        for period in periods:
            start, end = PERIODS[period]
            mask = scored["__decision_ts__"].ge(_utc(start)) & scored["__decision_ts__"].lt(_utc(end))
            subset = scored.loc[mask].copy()
            take = selected[mask.to_numpy()]
            valid = subset["policy_path_valid"] & subset["policy_net_bps"].notna()
            fields = ("policy_ge_50", "policy_ge_100", "policy_ge_200", "positive_top20", "positive_top10")
            result: dict[str, object] = {
                "arm": arm, "family": family, "period": period,
                "candidate_rows": int(len(subset)), "routed_rows": int(take.sum()),
                "route_fraction": float(take.mean()),
                "routed_policy_net_mean_bps": float(subset.loc[take & valid, "policy_net_bps"].mean()) if (take & valid).any() else float("nan"),
                "routed_policy_net_median_bps": float(subset.loc[take & valid, "policy_net_bps"].median()) if (take & valid).any() else float("nan"),
                "equal_timestamp_rank_ic": rank_ic(subset, score),
            }
            for field in fields:
                result[f"row_recall__{field}"] = recall(subset, take, field)
            result["recall_composite"] = (
                .20 * result["row_recall__policy_ge_50"] + .30 * result["row_recall__policy_ge_100"]
                + .25 * result["row_recall__policy_ge_200"] + .15 * result["row_recall__positive_top20"]
                + .10 * result["row_recall__positive_top10"]
            )
            rows.append(result)
    return pd.DataFrame(rows), pd.concat(memberships, ignore_index=True)


def choose_b3_winner(metrics: pd.DataFrame) -> str:
    dev = metrics.loc[metrics["period"].eq("development_2025q1q3")].copy()
    b3 = dev.loc[dev["family"].eq("B3")].sort_values(
        ["recall_composite", "routed_policy_net_mean_bps", "arm"], ascending=[False, False, True], kind="stable",
    )
    if b3.empty:
        raise AssertionError("no B3 metrics available")
    return str(b3.iloc[0]["arm"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--control-root", type=Path, default=DEFAULT_CONTROL)
    parser.add_argument("--b0-root", type=Path, default=DEFAULT_B0_ROOT)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    b0_path = args.b0_root / "b0_target_free_reconstruction.parquet"
    if not b0_path.is_file():
        raise FileNotFoundError(b0_path)
    fields = stable_feature_contract(args.control_root)
    b0 = pd.read_parquet(b0_path)
    b0["__decision_ts__"] = pd.to_datetime(b0["__decision_ts__"], utc=True, errors="raise")
    raw = load_raw_panel(args.source, args.policy, fields)
    raw_index = raw.set_index("candidate_id", drop=False)
    blocks = sorted(args.control_root.glob("bundles/block=*/upstream/monthly_upstream_bundle.joblib"))
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for path in blocks:
        block = path.parents[1].name
        bundle = joblib.load(path)
        cutoff = _utc(bundle.cutoff)
        held_ids = b0.loc[b0["control_block"].eq(block), "candidate_id"]
        if held_ids.empty:
            continue
        held = raw_index.loc[held_ids.to_numpy()].copy().reset_index(drop=True)
        if len(held) != len(held_ids):
            raise AssertionError(f"B0 held identity loss in {block}")
        held = held.merge(b0.loc[:, ["candidate_id", "__decision_ts__", "base_score"]], on=["candidate_id", "__decision_ts__"], how="inner", validate="one_to_one")
        if len(held) != len(held_ids):
            raise AssertionError(f"B0/raw timestamp identity mismatch in {block}")
        train = supervised_rows(raw, cutoff)
        held_target_free = held.drop(
            columns=["policy_path_valid", "policy_net_bps", "policy_label_available_ts"],
        )
        record = held_target_free.loc[:, ["candidate_id", "__decision_ts__", "base_score"]].copy()
        for weight in WEIGHTS:
            model_scores = fit_block_predictions(train, held_target_free, fields, weight)
            record = record.merge(model_scores, on="candidate_id", how="left", validate="one_to_one")
        record["control_block"] = block
        predictions.append(record)
        audits.append({
            "block": block, "cutoff": cutoff.isoformat(), "held_rows": int(len(held)),
            "train_rows": int(len(train)), "train_max_ts": train["__decision_ts__"].max().isoformat(),
            "reserve_start": (cutoff - pd.Timedelta(days=28)).isoformat(),
            "all_labels_before_reserve_start": bool(
                train["policy_label_available_ts"].lt(cutoff - pd.Timedelta(days=28)).all()
            ),
        })
        print(json.dumps({
            "event": "utility_block_complete",
            "block": block,
            "held_rows": int(len(held)),
            "train_rows": int(len(train)),
        }, sort_keys=True), flush=True)
    predicted = pd.concat(predictions, ignore_index=True).sort_values(["__decision_ts__", "candidate_id"], kind="stable")
    if predicted["candidate_id"].duplicated().any() or len(predicted) != len(b0):
        raise AssertionError("utility prediction identities must exactly preserve frozen B0 population")
    for column in [c for c in predicted.columns if c.startswith(("u_", "r_"))]:
        predicted[f"rank__{column}"] = timestamp_rank(predicted, column)
    predicted["rank__base_score"] = timestamp_rank(predicted, "base_score")
    utility_arms: list[tuple[str, str, Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray | None]]]] = []
    for column in ("u_day_balanced", "u_equal_month"):
        for eta in FUSION_WEIGHTS:
            arm = f"B3_{column}_eta{eta:g}"
            score = (1.0 - eta) * predicted["rank__base_score"] + eta * predicted[f"rank__{column}"]
            predicted[f"score__{arm}"] = score
            utility_arms.append((arm, "B3", lambda x, name=arm: (timestamp_route(x, f"score__{name}", fraction=.30), None)))
    for column in ("r_absolute_day_balanced", "r_absolute_equal_month", "r_hybrid_day_balanced", "r_hybrid_equal_month"):
        for eta in FUSION_WEIGHTS:
            arm = f"B4_{column}_eta{eta:g}"
            predicted[f"score__{arm}"] = (1.0 - eta) * predicted["rank__base_score"] + eta * predicted[f"rank__{column}"]
            utility_arms.append((arm, "B4", lambda x, name=arm: (timestamp_route(x, f"score__{name}", fraction=.30), None)))
    predicted = add_outcome_labels(predicted, args.source, args.policy)
    b3_b4_metrics, b3_b4_membership = metrics_for_routes(
        predicted, utility_arms, ("development_2025q1q3", "frozen_holdout_2025q4", "frozen_oos_2026jan_jul"),
    )
    winner = choose_b3_winner(b3_b4_metrics)
    winner_utility = winner.split("_eta", 1)[0].removeprefix("B3_")
    if winner_utility not in predicted.columns:
        raise AssertionError(f"development-selected B3 utility column is unavailable: {winner_utility}")
    b5_arms: list[tuple[str, str, Callable[[pd.DataFrame], tuple[np.ndarray, np.ndarray | None]]]] = []
    for fraction in (.20, .15):
        arm = f"B5_D2_{int(round(fraction * 100))}_utility_{int(round((.30 - fraction) * 100))}_{winner_utility}"
        predicted[f"score__{arm}"] = predicted["base_score"]
        b5_arms.append((
            arm, "B5",
            lambda x, primary=fraction, utility=winner_utility: quota_route(x, "base_score", utility, primary),
        ))
    b5_metrics, b5_membership = metrics_for_routes(
        predicted, b5_arms, ("development_2025q1q3", "frozen_holdout_2025q4", "frozen_oos_2026jan_jul"),
    )
    args.out_dir.mkdir(parents=True)
    predicted.to_parquet(args.out_dir / "utility_target_free_scores_and_outcome_audit.parquet", index=False, compression="zstd")
    pd.DataFrame(audits).to_parquet(args.out_dir / "utility_block_training_audit.parquet", index=False)
    pd.concat([b3_b4_metrics, b5_metrics], ignore_index=True).to_parquet(args.out_dir / "utility_base_recall_metrics.parquet", index=False)
    pd.concat([b3_b4_membership, b5_membership], ignore_index=True).to_parquet(args.out_dir / "utility_route_membership.parquet", index=False, compression="zstd")
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_long_base_utility_funnel_v2_strict_reserve",
        "scope": "offline B3/B4/B5 base funnel only; no downstream or live artifact modified",
        "source": {"path": str(args.source), "sha256": sha256(args.source)},
        "canonical_policy": {"path": str(args.policy), "sha256": sha256(args.policy)},
        "b0_input": {"path": str(b0_path), "candidate_rows": int(len(b0))},
        "feature_contract": {"count": len(fields), "ordered_fields": list(fields)},
        "strict_prequential": {
            "reserve_days": 28,
            "label_rule": "policy_label_available_ts < calibration_reserve_start",
            "training_cap": MAX_TRAIN_ROWS,
            "target_free_held_scoring": True,
        },
        "b3": {"targets": ["policy_net_bps >= 50", "policy_net_bps >= 150"], "weights": [asdict(x) for x in WEIGHTS], "fusion_eta": list(FUSION_WEIGHTS)},
        "b4": {"targets": ["absolute_policy_net_grades", "hybrid_absolute_relative_grades"], "edges_bps": [-200, -50, 50, 150, 300], "fusion_eta": list(FUSION_WEIGHTS)},
        "b5": {"development_selected_b3_arm": winner, "quota_routes": ["20%D2+10%utility", "15%D2+15%utility"]},
    }, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"event": "complete", "rows": int(len(predicted)), "b3_dev_selected": winner, "metrics": int(len(b3_b4_metrics) + len(b5_metrics))}, sort_keys=True))


if __name__ == "__main__":
    main()
