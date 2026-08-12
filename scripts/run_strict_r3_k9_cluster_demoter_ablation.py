#!/usr/bin/env python3
"""Strict-prequential K9-cluster downside-demoter ablation.

The LDF sizing test established that the additive K9 cluster-history fields do
not materially improve a *relative size* multiplier.  This script tests the
more appropriate alternative: use the same fields to demote a high-ranked
candidate when its frozen-cluster, prior-resolved history predicts a severe
policy residual.  It never refits Geometry/K9, uses no raw membership slot and
never uses held outcomes to train or calibrate a held score.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SEED = 20260811
TAILS = (0.005, 0.01, 0.02, 0.05)
# These are a K9-history candidate family.  Their slots are already
# soft-membership weighted and strictly prior-resolved by the materialiser;
# raw membership columns are never admitted here.  The comparison reference is
# supplied explicitly at the CLI, rather than treated as a protected tier.
K9_HISTORY_FIELDS = tuple(
    f"cluster_recent_{days}d_{metric}"
    for days in (3, 7, 14)
    for metric in (
        "support", "mean_residual_bps", "directional_rate", "positive_rate",
        "positive100_rate", "approx_rate", "adverse100_rate", "adverse200_rate",
    )
)
K9_SUPPORT_FIELDS = tuple(f"cluster_recent_{days}d_support" for days in (3, 7, 14))
CORE = ("final_score", "base_rank", "base_anchor_bps", "consensus_rank")


@dataclass(frozen=True)
class Arm:
    name: str
    kind: str  # control/logistic/lgbm
    threshold_bps: float = -200.0
    alpha: float = 0.0
    feature_set: str = "baseline_k9_all"


ARMS = (
    Arm("control", "control"),
    # Established contract control: it is the correct comparison for every
    # new K9 field, not the older four-score shortcut.
    Arm("logistic_baseline_risk200_a005", "logistic", -200.0, 0.05, "baseline"),
    Arm("logistic_baseline_risk200_a010", "logistic", -200.0, 0.10, "baseline"),
    Arm("logistic_baseline_risk200_a020", "logistic", -200.0, 0.20, "baseline"),
    Arm("logistic_baseline_risk200_a025", "logistic", -200.0, 0.25, "baseline"),
    # Full 24-field additive K9 history contract.
    Arm("logistic_baseline_plus_k9all_risk200_a005", "logistic", -200.0, 0.05, "baseline_k9_all"),
    Arm("logistic_baseline_plus_k9all_risk200_a010", "logistic", -200.0, 0.10, "baseline_k9_all"),
    Arm("logistic_baseline_plus_k9all_risk200_a020", "logistic", -200.0, 0.20, "baseline_k9_all"),
    Arm("logistic_baseline_plus_k9all_risk200_a025", "logistic", -200.0, 0.25, "baseline_k9_all"),
    # Small support-only probe distinguishes whether instability comes from
    # noisy path correctness or merely insufficient activated-cluster support.
    Arm("logistic_baseline_plus_k9support_risk200_a010", "logistic", -200.0, 0.10, "baseline_k9_support"),
    Arm("logistic_baseline_plus_k9support_risk200_a020", "logistic", -200.0, 0.20, "baseline_k9_support"),
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", required=True, type=Path)
    parser.add_argument(
        "--history-surface", type=Path,
        help="Optional earlier compatible surface used only for resolved chronological training support.",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--reference-feature-contract", required=True, type=Path,
        help="JSON feature contract whose compact_fields are the explicit comparison reference; it is not a protected selection tier.",
    )
    parser.add_argument("--train-cap", type=int, default=80_000)
    parser.add_argument("--first-held-month", type=str, default="2025-04-01")
    parser.add_argument("--last-held-month", type=str, default="2025-07-01")
    return parser.parse_args()


def _equal_month_cap(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame
    month = frame["__decision_ts__"].dt.to_period("M").astype(str)
    rng = np.random.default_rng(seed)
    quota = max(1, cap // month.nunique())
    pieces = []
    for token in sorted(month.unique()):
        part = frame.loc[month.eq(token)]
        if len(part) > quota:
            part = part.iloc[np.sort(rng.choice(len(part), quota, replace=False))]
        pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def _matrix(frame: pd.DataFrame, fields: tuple[str, ...]) -> np.ndarray:
    return (
        frame.loc[:, list(fields)].apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
    )


def _fields_for_arm(
    frame: pd.DataFrame, arm: Arm, reference_fields: tuple[str, ...],
) -> tuple[str, ...]:
    available = set(frame.columns)
    reference = tuple(field for field in reference_fields if field in available)
    missing_reference = sorted(set(reference_fields).difference(reference))
    if missing_reference:
        raise KeyError(f"surface lacks reference contract fields: {missing_reference}")
    if arm.feature_set == "core":
        return CORE
    if arm.feature_set == "baseline":
        return reference
    if arm.feature_set == "baseline_k9_all":
        return reference + K9_HISTORY_FIELDS
    if arm.feature_set == "baseline_k9_support":
        return reference + K9_SUPPORT_FIELDS
    raise ValueError(f"unknown feature set {arm.feature_set}")


def _fit_predict(
    train: pd.DataFrame, held: pd.DataFrame, arm: Arm, reference_fields: tuple[str, ...],
) -> np.ndarray:
    if arm.kind == "control":
        return np.zeros(len(held), dtype=np.float32)
    fields = _fields_for_arm(train, arm, reference_fields)
    x_train, x_held = _matrix(train, fields), _matrix(held, fields)
    y = pd.to_numeric(train["policy_net_bps"], errors="coerce").le(arm.threshold_bps).to_numpy(np.int8)
    if y.min() == y.max():
        return np.full(len(held), float(y[0]), dtype=np.float32)
    if arm.kind == "logistic":
        scaler = StandardScaler()
        model = LogisticRegression(
            C=0.25, class_weight="balanced", max_iter=300, random_state=SEED,
        )
        model.fit(scaler.fit_transform(x_train), y)
        return model.predict_proba(scaler.transform(x_held))[:, 1].astype(np.float32)
    model = LGBMClassifier(
        objective="binary", n_estimators=180, learning_rate=0.04,
        num_leaves=8, max_depth=3, min_child_samples=500,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=8.0,
        class_weight="balanced", random_state=SEED, verbosity=-1,
    )
    model.fit(x_train, y)
    return model.predict_proba(x_held)[:, 1].astype(np.float32)


def _metrics(frame: pd.DataFrame, arm: str, period: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    population = frame.loc[np.isfinite(pd.to_numeric(frame["demoted_score"], errors="coerce"))]
    for tail in TAILS:
        count = max(1, int(np.ceil(tail * len(population))))
        selected = population.nlargest(count, "demoted_score", keep="first")
        valid = selected.loc[selected["policy_path_valid"].fillna(False).astype(bool)]
        rows.append({
            "arm": arm, "period": period, "tail": tail, "population_rows": len(population),
            "selected_score_rows": len(selected), "valid_outcomes": len(valid),
            "outcome_coverage": len(valid) / max(len(selected), 1),
            "net_bps_per_trade": float(pd.to_numeric(valid["policy_net_bps"], errors="coerce").mean()),
            "gross_bps_per_trade": float(pd.to_numeric(valid["policy_gross_bps"], errors="coerce").mean()),
            "positive_rate": float(pd.to_numeric(valid["policy_net_bps"], errors="coerce").gt(0).mean()),
        })
    return rows


def main() -> None:
    args = _args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    contract = json.loads(args.reference_feature_contract.read_text())
    reference_fields = tuple(contract["compact_fields"])
    required = [
        "candidate_id", "__decision_ts__", "geometry_bundle_sha256", "policy_path_valid",
        "policy_label_available_ts", "policy_net_bps", "policy_gross_bps",
        *reference_fields, *K9_HISTORY_FIELDS,
    ]
    frame = pd.read_parquet(args.surface, columns=required)
    if args.history_surface is not None:
        history = pd.read_parquet(args.history_surface, columns=required)
        frame = pd.concat([history, frame], ignore_index=True)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True)
    frame["policy_label_available_ts"] = pd.to_datetime(frame["policy_label_available_ts"], utc=True, errors="coerce")
    if frame["candidate_id"].duplicated().any() or frame["geometry_bundle_sha256"].nunique() != 1:
        raise AssertionError("demoter requires unique rows and one frozen Geometry/K9 bundle")
    first = pd.Timestamp(args.first_held_month, tz="UTC")
    last = pd.Timestamp(args.last_held_month, tz="UTC")
    if first.day != 1 or last.day != 1 or first > last:
        raise ValueError("held months must be ascending calendar-month starts")
    args.out_dir.mkdir(parents=True)
    predictions: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for fold, cutoff in enumerate(pd.date_range(first, last, freq="MS", tz="UTC")):
        held_end = cutoff + pd.offsets.MonthBegin(1)
        train_start = cutoff - pd.DateOffset(months=3)
        train = frame.loc[
            frame["__decision_ts__"].ge(train_start)
            & frame["__decision_ts__"].lt(cutoff)
            & frame["policy_label_available_ts"].lt(cutoff)
            & frame["policy_path_valid"].fillna(False).astype(bool)
        ].copy()
        # The training domain is defined solely by the prior score distribution.
        floor = float(pd.to_numeric(train["final_score"], errors="coerce").quantile(0.70))
        train = train.loc[pd.to_numeric(train["final_score"], errors="coerce").ge(floor)]
        train = _equal_month_cap(train, int(args.train_cap), SEED + fold)
        if train.empty:
            raise ValueError(f"no valid resolved prequential training rows for held month {cutoff:%Y-%m}")
        held = frame.loc[frame["__decision_ts__"].ge(cutoff) & frame["__decision_ts__"].lt(held_end)].copy()
        for arm in ARMS:
            risk = _fit_predict(train, held, arm, reference_fields)
            output = held.loc[:, [
                "candidate_id", "__decision_ts__", "policy_path_valid", "policy_net_bps", "policy_gross_bps", "final_score",
            ]].copy()
            output["risk_probability"] = risk
            output["demoted_score"] = pd.to_numeric(output["final_score"], errors="coerce").to_numpy(float) * (1.0 - arm.alpha * risk)
            output["arm"] = arm.name
            output["fold"] = fold
            predictions.append(output)
            audit.append({
                "fold": fold, "cutoff": cutoff, "arm": arm.name, "kind": arm.kind,
                "risk_threshold_bps": arm.threshold_bps, "alpha": arm.alpha,
                "feature_set": arm.feature_set,
                "train_rows": len(train), "held_rows": len(held), "score_floor": floor,
                "train_severe_rate": float(pd.to_numeric(train["policy_net_bps"], errors="coerce").le(arm.threshold_bps).mean()),
            })
        gc.collect()
        print(json.dumps({"event": "fold_complete", "fold": fold, "cutoff": str(cutoff), "rows": len(held)}), flush=True)
    scored = pd.concat(predictions, ignore_index=True)
    scored.to_parquet(args.out_dir / "oof_predictions.parquet", index=False, compression="zstd")
    metrics: list[dict[str, object]] = []
    for arm, block in scored.groupby("arm", sort=True):
        metrics.extend(_metrics(block, str(arm), "global"))
        for month, month_block in block.groupby(block["__decision_ts__"].dt.to_period("M").astype(str), sort=True):
            metrics.extend(_metrics(month_block, str(arm), str(month)))
    pd.DataFrame(metrics).to_parquet(args.out_dir / "metrics.parquet", index=False)
    pd.DataFrame(audit).to_parquet(args.out_dir / "fold_audit.parquet", index=False)
    (args.out_dir / "run_manifest.json").write_text(json.dumps({
        "schema": "strict_r3_frozen_k9_cluster_demoter_ablation_v1",
        "surface": str(args.surface), "history_surface": str(args.history_surface) if args.history_surface else None,
        "geometry_bundle_sha256": str(frame["geometry_bundle_sha256"].iloc[0]),
        "reference_feature_contract": str(args.reference_feature_contract),
        "reference_fields": list(reference_fields),
        "additive_k9_history_fields": list(K9_HISTORY_FIELDS), "raw_k9_memberships": False,
        "training": "3 prior calendar months; policy labels resolved strictly before held cutoff; prior-score top 30%",
        "held_months": {"first": str(first), "last": str(last)},
        "integration": "demoted_score = final_score * (1 - alpha * P(severe policy loss))",
        "arms": [arm.__dict__ for arm in ARMS],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
