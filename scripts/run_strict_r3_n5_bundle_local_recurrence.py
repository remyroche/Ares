#!/usr/bin/env python3
"""Bundle-local path/archetype recurrence audit for canonical N5.

Raw K9 fields are used only inside one exact geometry bundle.  Cluster slots
are aligned with a train-only residual-value ordering before their held-period
use, yielding stable role names instead of pretending cluster_00 has the same
meaning across bundles.  2025 selects recurring roles; 2026 is confirmation.

This is a diagnostic half-bundle replay.  It does not yet make bundle-local
fields live at the first timestamp of a new bundle; production promotion needs
the bundle encoder replayed over its pre-cutoff history.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.n5_forest_support_sizing import (  # noqa: E402
    CANONICAL_N5_SPEC,
    fit_canonical_n5_bundle,
)
from extreme_price_movements.trust_sizing_ablation import ParentExpectation, trust_feature_family  # noqa: E402
from scripts.run_strict_r3_trust_sizing_ablation import INPUTS, _load, _sample_equal_month  # noqa: E402


SCHEMA = "strict_r3_n5_bundle_local_recurrence_v2"
SEED = 20260810
KINDS = ("membership", "negative_distance", "confidence")
ROLE_OUTPUTS = (
    "expected_residual_bps",
    "downside_risk_bps",
    "effective_support",
    "confidence",
)
TAILS = (0.05, 0.10, 0.20)


def _raw_fields() -> list[str]:
    return [f"k09__cluster_{cluster:02d}__{kind}" for cluster in range(9) for kind in KINDS]


def _load_with_raw(year: int) -> tuple[pd.DataFrame, list[str]]:
    stable, fields, _audit = _load(INPUTS[year])
    raw = pd.read_parquet(INPUTS[year], columns=["candidate_id", *_raw_fields()])
    result = stable.merge(raw, on="candidate_id", how="left", validate="one_to_one")
    return result, fields


def _cluster_statistics(
    train: pd.DataFrame,
    parent: ParentExpectation,
) -> dict[str, list[float]]:
    residual = pd.to_numeric(train["policy_net_bps"], errors="coerce").to_numpy(float) - parent.predict(train["final_score"])
    expected: list[float] = []
    downside: list[float] = []
    support_fraction: list[float] = []
    train_confidence: list[float] = []
    for cluster in range(9):
        membership = pd.to_numeric(
            train[f"k09__cluster_{cluster:02d}__membership"], errors="coerce",
        ).fillna(0.0).clip(lower=0.0).to_numpy(float)
        confidence = pd.to_numeric(
            train[f"k09__cluster_{cluster:02d}__confidence"], errors="coerce",
        ).fillna(0.0).clip(lower=0.0).to_numpy(float)
        support = float(membership.sum())
        expected.append(float(np.sum(membership * residual) / support) if support > 1e-9 else 0.0)
        adverse = np.minimum(residual, 0.0)
        downside.append(float(np.sqrt(np.sum(membership * adverse**2) / support)) if support > 1e-9 else 0.0)
        support_fraction.append(float(support / max(len(train), 1)))
        train_confidence.append(float(np.sum(membership * confidence) / support) if support > 1e-9 else 0.0)
    return {
        "expected_residual_bps": expected,
        "downside_risk_bps": downside,
        "effective_support_fraction": support_fraction,
        "mean_confidence": train_confidence,
    }


def _role_map(cluster_statistics: dict[str, list[float]]) -> list[int]:
    expected = cluster_statistics["expected_residual_bps"]
    return sorted(range(9), key=lambda cluster: (expected[cluster], -cluster), reverse=True)


def _normalized_activation(frame: pd.DataFrame) -> np.ndarray:
    activation = np.column_stack(
        [
            pd.to_numeric(frame[f"k09__cluster_{cluster:02d}__membership"], errors="coerce")
            .fillna(0.0).clip(lower=0.0).to_numpy(float)
            for cluster in range(9)
        ]
    )
    total = activation.sum(axis=1, keepdims=True)
    return np.divide(
        activation,
        total,
        out=np.full_like(activation, 1.0 / activation.shape[1]),
        where=total > 1e-12,
    )


def _materialize_roles(
    frame: pd.DataFrame,
    ordering: Sequence[int],
    cluster_statistics: dict[str, list[float]],
) -> tuple[pd.DataFrame, list[str]]:
    """Expose bundle-local clusters through stable, economic role outputs.

    Roles are ordered on training-only expected policy residual.  Every field
    is an activation-scaled contribution, so downstream models never consume
    a raw cluster slot whose semantics change with the Geometry/K9 bundle.
    """

    result = frame.copy()
    weights = _normalized_activation(frame)
    fields: list[str] = []
    for role, cluster in enumerate(ordering):
        activation = weights[:, cluster]
        confidence = pd.to_numeric(
            frame[f"k09__cluster_{cluster:02d}__confidence"], errors="coerce",
        ).fillna(cluster_statistics["mean_confidence"][cluster]).clip(lower=0.0).to_numpy(float)
        values = {
            "expected_residual_bps": activation * cluster_statistics["expected_residual_bps"][cluster],
            "downside_risk_bps": activation * cluster_statistics["downside_risk_bps"][cluster],
            "effective_support": activation * cluster_statistics["effective_support_fraction"][cluster],
            "confidence": activation * confidence,
        }
        for category in ROLE_OUTPUTS:
            target = f"bundle_role_{role:02d}_{category}"
            result[target] = values[category]
            fields.append(target)
    return result, fields


def _activation_aggregates(
    train: pd.DataFrame,
    frame: pd.DataFrame,
    parent: ParentExpectation,
    *,
    cluster_statistics: dict[str, list[float]] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, list[float]]]:
    """Collapse cluster-local states using current per-row activation weights."""

    stats = cluster_statistics or _cluster_statistics(train, parent)
    expected = stats["expected_residual_bps"]
    downside = stats["downside_risk_bps"]
    support_fraction = stats["effective_support_fraction"]
    train_confidence = stats["mean_confidence"]
    weights = _normalized_activation(frame)
    current_confidence = np.column_stack(
        [
            pd.to_numeric(frame[f"k09__cluster_{cluster:02d}__confidence"], errors="coerce")
            .fillna(0.0).clip(lower=0.0).to_numpy(float)
            for cluster in range(9)
        ]
    )
    output = frame.copy()
    fields = [
        "bundle_activation_expected_residual_bps",
        "bundle_activation_downside_risk_bps",
        "bundle_activation_effective_support",
        "bundle_activation_confidence",
    ]
    output[fields[0]] = weights @ np.asarray(expected, dtype=float)
    output[fields[1]] = weights @ np.asarray(downside, dtype=float)
    output[fields[2]] = weights @ np.asarray(support_fraction, dtype=float)
    # Current confidence is the primary row-level quantity.  The train-local
    # confidence prior keeps missing/degenerate cluster outputs conservative.
    confidence_matrix = np.where(
        np.isfinite(current_confidence),
        current_confidence,
        np.asarray(train_confidence, dtype=float)[None, :],
    )
    output[fields[3]] = np.sum(weights * confidence_matrix, axis=1)
    return output, fields, stats


def _role_family(field: str) -> str:
    for category in ROLE_OUTPUTS:
        if field.endswith(category):
            return "bundle_role_" + category
    return "bundle_role_other"


def _sample(frame: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame.copy()
    rng = np.random.default_rng(seed)
    score = pd.to_numeric(frame["final_score"], errors="coerce")
    decile = pd.qcut(score.rank(method="first"), 10, labels=False, duplicates="drop")
    quota = max(1, cap // max(int(decile.nunique()), 1))
    chosen: list[np.ndarray] = []
    for value in sorted(decile.dropna().unique()):
        index = np.flatnonzero(decile.eq(value).to_numpy())
        if len(index) > quota:
            index = np.sort(rng.choice(index, quota, replace=False))
        chosen.append(index)
    selected = np.concatenate(chosen)
    return frame.iloc[selected].sort_values(["__decision_ts__", "candidate_id"], kind="stable")


def _metric(frame: pd.DataFrame, multiplier: np.ndarray) -> tuple[float, dict[str, float]]:
    score = pd.to_numeric(frame["final_score"], errors="coerce")
    net = pd.to_numeric(frame["policy_net_bps"], errors="coerce")
    weight = np.asarray(multiplier, dtype=float)
    values: dict[str, float] = {}
    objective = 0.0
    coefficients = {0.05: 1.0, 0.10: 0.5, 0.20: 0.2}
    for tail in TAILS:
        count = max(1, int(math.ceil(tail * len(frame))))
        index = score.nlargest(count, keep="first").index
        valid = net.loc[index].notna()
        positions = frame.index.get_indexer(index[valid])
        value = float(np.average(net.loc[index[valid]], weights=weight[positions]))
        values[f"top{int(tail * 100)}_weighted_net_bps"] = value
        objective += coefficients[tail] * value
    return objective, values


def _bundle_runs(
    frame: pd.DataFrame,
    stable_fields: Sequence[str],
    *,
    year: int,
    selected_2025: Sequence[str] | None,
    train_cap: int,
    held_cap: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    comparison: list[dict[str, Any]] = []
    importance: list[dict[str, Any]] = []
    outputs: list[pd.DataFrame] = []
    rng = np.random.default_rng(SEED + year)
    period_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
    period_end = pd.Timestamp(f"{year}-08-01", tz="UTC")
    work = frame.loc[frame["__decision_ts__"].ge(period_start) & frame["__decision_ts__"].lt(period_end)].copy()
    for bundle_index, (bundle_id, block) in enumerate(work.groupby("geometry_bundle_id", sort=True)):
        block = block.sort_values(["__decision_ts__", "candidate_id"], kind="stable")
        timestamps = block["__decision_ts__"].drop_duplicates().sort_values()
        if len(timestamps) < 24 * 10:
            continue
        split = timestamps.iloc[len(timestamps) // 2]
        train_all = block.loc[
            block["__decision_ts__"].lt(split)
            & block["policy_label_available_ts"].lt(split)
            & block["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(block["policy_net_bps"], errors="coerce"))
            & block["mapped_ev_available"].astype(bool)
        ].copy()
        held = block.loc[
            block["__decision_ts__"].ge(split)
            & block["policy_path_valid"].fillna(False).astype(bool)
            & np.isfinite(pd.to_numeric(block["policy_net_bps"], errors="coerce"))
        ].copy()
        if len(train_all) < 2_000 or len(held) < 2_000:
            continue
        parent = ParentExpectation.fit(train_all["final_score"], train_all["policy_net_bps"])
        cluster_statistics = _cluster_statistics(train_all, parent)
        ordering = _role_map(cluster_statistics)
        train_all, role_fields = _materialize_roles(train_all, ordering, cluster_statistics)
        held, _ = _materialize_roles(held, ordering, cluster_statistics)
        train_all, activation_fields, activation_stats = _activation_aggregates(
            train_all, train_all, parent, cluster_statistics=cluster_statistics,
        )
        held, _, _ = _activation_aggregates(
            train_all, held, parent, cluster_statistics=cluster_statistics,
        )
        floor = float(train_all["final_score"].quantile(0.70))
        train = train_all.loc[train_all["final_score"].ge(floor)].copy()
        train = _sample_equal_month(train, train_cap)
        train["parent_expected_bps"] = parent.predict(train["final_score"])
        held = _sample(held, held_cap, SEED + 1000 + bundle_index)
        candidate_fields = list(stable_fields) + list(role_fields)
        role_selected_fields = candidate_fields if selected_2025 is None else [
            field for field in candidate_fields if field in set(stable_fields) | set(selected_2025)
        ]
        models = {
            "stable_only": list(stable_fields),
            "bundle_role": role_selected_fields,
            "activation_aggregate": [*stable_fields, *activation_fields],
            "bundle_role_plus_activation": [*role_selected_fields, *activation_fields],
        }
        scored: dict[str, tuple[Any, np.ndarray]] = {}
        for arm, fields in models.items():
            model = fit_canonical_n5_bundle(
                train,
                fields,
                [],
                parent_expectation=parent,
                cutoff=split,
                training_score_floor=floor,
            )
            prediction, multiplier = model.size_multiplier(held)
            score, metrics = _metric(held, multiplier)
            scored[arm] = (model, multiplier)
            comparison.append(
                {
                    "year": year,
                    "geometry_bundle_id": bundle_id,
                    "geometry_bundle_sha256": str(block["geometry_bundle_sha256"].iloc[0]),
                    "split": split,
                    "arm": arm,
                    "train_rows": len(train),
                    "held_rows": len(held),
                    "feature_count": len(fields),
                    "objective": score,
                    "activation_stats": json.dumps(activation_stats, sort_keys=True),
                    **metrics,
                }
            )
        rich_fields = models["bundle_role_plus_activation"]
        rich_model, rich_multiplier = scored["bundle_role_plus_activation"]
        baseline_score, _ = _metric(held, rich_multiplier)
        month = held["__decision_ts__"].dt.to_period("M").astype(str).to_numpy()
        decile = pd.qcut(held["final_score"].rank(method="first"), 10, labels=False, duplicates="drop").to_numpy()
        strata = np.asarray([f"{m}|{d}" for m, d in zip(month, decile)], dtype=object)
        for field in rich_fields:
            permuted = held.copy()
            values = permuted[field].to_numpy(copy=True)
            shuffled = values.copy()
            for stratum in np.unique(strata):
                index = np.flatnonzero(strata == stratum)
                shuffled[index] = values[rng.permutation(index)]
            permuted[field] = shuffled
            _prediction, multiplier = rich_model.size_multiplier(permuted)
            permuted_score, _ = _metric(held, multiplier)
            importance.append(
                {
                    "year": year,
                    "geometry_bundle_id": bundle_id,
                    "field": field,
                    "family": (
                        _role_family(field)
                        if field.startswith("bundle_role_") else trust_feature_family(field)
                    ),
                    "mda_loss": baseline_score - permuted_score,
                }
            )
        output = held.loc[:, ["candidate_id", "__decision_ts__", "final_score", "policy_net_bps"]].copy()
        output["stable_size_multiplier"] = scored["stable_only"][1]
        output["bundle_role_size_multiplier"] = scored["bundle_role"][1]
        output["activation_aggregate_size_multiplier"] = scored["activation_aggregate"][1]
        output["bundle_role_plus_activation_size_multiplier"] = scored["bundle_role_plus_activation"][1]
        for field in activation_fields:
            output[field] = held[field].to_numpy(float)
        output["geometry_bundle_id"] = bundle_id
        output["year"] = year
        outputs.append(output)
    return pd.DataFrame(comparison), pd.DataFrame(importance), pd.concat(outputs, ignore_index=True)


def _recurrence(importance: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (field, family), block in importance.groupby(["field", "family"], sort=True):
        values = block["mda_loss"].to_numpy(float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        worst = float(np.min(values))
        rows.append(
            {
                "field": field,
                "family": family,
                "bundles": len(values),
                "positive_bundle_fraction": float(np.mean(values > 0.0)),
                "mda_median": median,
                "mda_mad": mad,
                "mda_worst_bundle": worst,
                "portable_mda_score": median - 0.5 * mad - max(0.0, -worst),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["portable_mda_score", "positive_bundle_fraction"], ascending=False, kind="stable",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-cap", type=int, default=15_000)
    parser.add_argument("--held-cap", type=int, default=6_000)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    args.out_dir.mkdir(parents=True)
    frame25, stable_fields = _load_with_raw(2025)
    comparison25, importance25, output25 = _bundle_runs(
        frame25,
        stable_fields,
        year=2025,
        selected_2025=None,
        train_cap=args.train_cap,
        held_cap=args.held_cap,
    )
    recurrence25 = _recurrence(importance25)
    selected_roles = recurrence25.loc[
        recurrence25["field"].str.startswith("bundle_role_")
        & recurrence25["positive_bundle_fraction"].ge(0.60)
        & recurrence25["mda_median"].gt(0.0)
    ].head(12)["field"].tolist()
    frame26, stable26 = _load_with_raw(2026)
    if stable26 != stable_fields:
        raise ValueError("2025/2026 stable N5 contracts differ")
    comparison26, importance26, output26 = _bundle_runs(
        frame26,
        stable_fields,
        year=2026,
        selected_2025=selected_roles,
        train_cap=args.train_cap,
        held_cap=args.held_cap,
    )
    recurrence26 = _recurrence(importance26)
    arm_selection = (
        comparison25.groupby("arm", as_index=False)
        .agg(
            mean_objective=("objective", "mean"),
            median_objective=("objective", "median"),
            worst_bundle_objective=("objective", "min"),
            positive_bundles=("objective", lambda value: int((value > 0.0).sum())),
            bundles=("objective", "size"),
        )
        .sort_values(
            ["mean_objective", "worst_bundle_objective"],
            ascending=False,
            kind="stable",
        )
    )
    winner_arm = str(arm_selection.iloc[0]["arm"])
    comparison = pd.concat([comparison25, comparison26], ignore_index=True)
    importance = pd.concat([importance25, importance26], ignore_index=True)
    outputs = pd.concat([output25, output26], ignore_index=True)
    comparison.to_parquet(args.out_dir / "bundle_comparison.parquet", index=False)
    importance.to_parquet(args.out_dir / "bundle_mda_detail.parquet", index=False)
    recurrence25.assign(year=2025).to_parquet(args.out_dir / "feature_recurrence_2025.parquet", index=False)
    recurrence26.assign(year=2026).to_parquet(args.out_dir / "feature_recurrence_2026.parquet", index=False)
    outputs.to_parquet(args.out_dir / "bundle_local_predictions.parquet", index=False, compression="zstd")
    arm_selection.to_parquet(args.out_dir / "arm_selection_2025.parquet", index=False)
    (args.out_dir / "selected_bundle_roles.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "selection_year": 2025,
                "selected_roles": selected_roles,
                "selected_arm": winner_arm,
                "2026_used_for_selection": False,
            },
            indent=2,
        )
        + "\n"
    )
    manifest = {
        "schema": SCHEMA,
        "status": "diagnostic_not_live_ready",
        "split": "first half of each exact geometry bundle trains; second half evaluates",
        "raw_k9_scope": "one identical geometry_bundle_id only",
        "role_alignment": "train-only membership-weighted policy residual ordering",
        "selected_roles": selected_roles,
        "selected_arm": winner_arm,
        "activation_weighting": "per-row normalized K9 memberships across all nine clusters",
        "activation_outputs": [
            "expected residual", "downside risk", "effective support", "confidence"
        ],
        "selection_year": 2025,
        "confirmation_year": 2026,
        "limitation": "cold-start unresolved; production requires transforming pre-cutoff history with the new bundle encoder",
        "train_cap": args.train_cap,
        "held_cap": args.held_cap,
        "seed": SEED,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"event": "complete", "out_dir": str(args.out_dir), "selected_roles": selected_roles, "selected_arm": winner_arm}))


if __name__ == "__main__":
    main()
