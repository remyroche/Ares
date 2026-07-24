#!/usr/bin/env python3
"""Nested forward validation and compact family export for residual states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.features_negative_residuals import (
    NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS,
    NEGATIVE_RESIDUAL_META_FEATURE_KEYS,
    negative_residual_feature_contract,
)
from extreme_price_movements.residual_state_family_features import (
    ResidualStateFamilyContract,
    fit_definition,
)


def _daily(path: Path, start: str, end: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=NEGATIVE_RESIDUAL_META_FEATURE_KEYS)
    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.loc[(frame.index >= start) & (frame.index < end)]
    composites = set(NEGATIVE_RESIDUAL_COMPOSITE_FEATURE_KEYS)
    return pd.concat(
        [
            (
                frame[key].groupby(frame.index.floor("D")).max()
                if key in composites
                else frame[key].groupby(frame.index.floor("D")).mean()
            ).rename(key)
            for key in NEGATIVE_RESIDUAL_META_FEATURE_KEYS
        ],
        axis=1,
    ).astype(np.float32)


def _metric(score: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, float]:
    valid = np.isfinite(score)
    score, target = score[valid], target[valid].astype(bool)
    selected = score >= threshold
    prevalence = float(target.mean()) if len(target) else np.nan
    precision = float(target[selected].mean()) if selected.any() else 0.0
    return {
        "precision": precision,
        "lift": precision / max(prevalence, 1e-8),
        "fpr": float((selected & ~target).sum() / max((~target).sum(), 1)),
        "support": int((selected & target).sum()),
    }


def _quarters() -> list[pd.Period]:
    return list(pd.period_range("2025Q3", "2026Q2", freq="Q"))


def run(args: argparse.Namespace) -> dict[str, object]:
    args.output.mkdir(parents=True, exist_ok=True)
    daily = _daily(args.feature_file, args.start, args.end)
    calendar = pd.read_csv(args.calendar)
    calendar["day"] = pd.to_datetime(calendar["day"], utc=True).dt.floor("D")
    calendar = calendar.drop_duplicates(["day", "side_name", "archetype_policy_key"])
    candidates = pd.read_csv(args.candidates).drop_duplicates(
        ["side_name", "archetype_policy_key", "base_feature", "form", "gate_feature"]
    )
    fold_rows: list[dict[str, object]] = []
    eligibility_rows: list[dict[str, object]] = []
    candidate_audit_rows: list[dict[str, object]] = []
    for evaluation in _quarters():
        selection = evaluation - 1
        selection_start = pd.Timestamp(selection.start_time, tz="UTC")
        selection_end = pd.Timestamp(selection.end_time + pd.Timedelta(days=1), tz="UTC")
        evaluation_start = pd.Timestamp(evaluation.start_time, tz="UTC")
        evaluation_end = pd.Timestamp(evaluation.end_time + pd.Timedelta(days=1), tz="UTC")
        discovery_mask = np.asarray(daily.index < selection_start)
        selection_mask = np.asarray((daily.index >= selection_start) & (daily.index < selection_end))
        evaluation_mask = np.asarray((daily.index >= evaluation_start) & (daily.index < evaluation_end))
        for (side, archetype), local in candidates.groupby(
            ["side_name", "archetype_policy_key"], observed=True
        ):
            events = set(
                calendar.loc[
                    (calendar["side_name"] == side)
                    & (calendar["archetype_policy_key"] == archetype),
                    "day",
                ]
            )
            target = np.asarray(daily.index.isin(events), dtype=bool)
            discovery_adverse = int(target[discovery_mask].sum())
            selection_adverse = int(target[selection_mask].sum())
            evaluation_adverse = int(target[evaluation_mask].sum())
            evaluation_benign = int(evaluation_mask.sum() - evaluation_adverse)
            skipped_reason = ""
            if discovery_adverse < 3:
                skipped_reason = "insufficient_discovery_adverse_support"
            elif selection_adverse < 1:
                skipped_reason = "no_selection_quarter_adverse_events"
            elif evaluation_adverse < 1:
                skipped_reason = "no_evaluation_quarter_adverse_events"
            eligible = not skipped_reason
            eligibility_rows.append(
                {
                    "evaluation_quarter": str(evaluation),
                    "selection_quarter": str(selection),
                    "side_name": side,
                    "archetype_policy_key": archetype,
                    "eligible": eligible,
                    "evaluated": eligible,
                    "skipped_reason": skipped_reason,
                    "discovery_adverse_support": discovery_adverse,
                    "selection_adverse_support": selection_adverse,
                    "evaluation_adverse_support": evaluation_adverse,
                    "evaluation_benign_support": evaluation_benign,
                }
            )
            if not eligible:
                continue
            discovery = daily.loc[discovery_mask]
            discovery_target = target[discovery_mask]
            for _, row in local.iterrows():
                definition = fit_definition(discovery, discovery_target, row)
                score = definition.transform(daily)
                discovery_threshold = float(np.nanquantile(score[discovery_mask], 0.90))
                base = pd.to_numeric(daily[row["base_feature"]], errors="coerce").to_numpy(float)
                base_threshold = float(np.nanquantile(base[discovery_mask], 0.90))
                selected = _metric(score[selection_mask], target[selection_mask], discovery_threshold)
                selected_base = _metric(base[selection_mask], target[selection_mask], base_threshold)
                selection_precision_gain = selected["precision"] - selected_base["precision"]
                admitted = (
                    selected["lift"] > 1.0
                    and selected["fpr"] < 0.20
                    and selection_precision_gain > 0
                )
                candidate_audit_rows.append(
                    {
                        "evaluation_quarter": str(evaluation),
                        "selection_quarter": str(selection),
                        "side_name": side,
                        "archetype_policy_key": archetype,
                        "base_feature": row["base_feature"],
                        "form": row["form"],
                        "gate_feature": row["gate_feature"],
                        "eligible": True,
                        "evaluated": admitted,
                        "skipped_reason": "" if admitted else "selection_quarter_rejected",
                        "selection_lift": selected["lift"],
                        "selection_fpr": selected["fpr"],
                        "selection_precision_gain": selection_precision_gain,
                        "evaluation_adverse_support": evaluation_adverse,
                        "evaluation_benign_support": evaluation_benign,
                    }
                )
                if not admitted:
                    continue
                evaluated = _metric(score[evaluation_mask], target[evaluation_mask], discovery_threshold)
                evaluated_base = _metric(base[evaluation_mask], target[evaluation_mask], base_threshold)
                fold_rows.append(
                    {
                        "evaluation_quarter": str(evaluation),
                        "selection_quarter": str(selection),
                        "discovery_end": selection_start.isoformat(),
                        "side_name": side,
                        "archetype_policy_key": archetype,
                        "base_feature": row["base_feature"],
                        "form": row["form"],
                        "gate_feature": row["gate_feature"],
                        "evaluation_lift": evaluated["lift"],
                        "evaluation_fpr": evaluated["fpr"],
                        "evaluation_precision": evaluated["precision"],
                        "evaluation_precision_gain": evaluated["precision"] - evaluated_base["precision"],
                        "evaluation_support": evaluated["support"],
                    }
                )
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(args.output / "nested_fold_metrics.csv", index=False)
    pd.DataFrame(eligibility_rows).to_csv(
        args.output / "quarter_eligibility.csv", index=False
    )
    pd.DataFrame(candidate_audit_rows).to_csv(
        args.output / "candidate_quarter_audit.csv", index=False
    )
    keys = ["side_name", "archetype_policy_key", "base_feature", "form", "gate_feature"]
    summary = (
        folds.groupby(keys, observed=True)
        .agg(
            lift_q25=("evaluation_lift", lambda value: value.quantile(0.25)),
            mean_lift=("evaluation_lift", "mean"),
            fpr_q75=("evaluation_fpr", lambda value: value.quantile(0.75)),
            mean_fpr=("evaluation_fpr", "mean"),
            median_precision_gain=("evaluation_precision_gain", "median"),
            positive_folds=("evaluation_precision_gain", lambda value: int((value > 0).sum())),
            evaluated_folds=("evaluation_quarter", "nunique"),
            adverse_support=("evaluation_support", "sum"),
        )
        .reset_index()
    )
    summary["fold_stability"] = np.clip(
        summary["lift_q25"] / summary["mean_lift"].clip(lower=1e-8), 0.0, 1.0
    )
    summary["conservative_score"] = (
        summary["lift_q25"]
        * (1.0 - summary["fpr_q75"])
        * np.sqrt(summary["adverse_support"] / (summary["adverse_support"] + 10.0))
    )
    summary["nested_promoted"] = (
        (summary["lift_q25"] > 1.0)
        & (summary["fpr_q75"] < 0.20)
        & (summary["median_precision_gain"] > 0)
        & (summary["positive_folds"] >= 3)
        & (summary["adverse_support"] >= 5)
    )
    summary = summary.sort_values("conservative_score", ascending=False, kind="stable")
    summary.to_csv(args.output / "nested_definition_summary.csv", index=False)
    promoted = summary.loc[summary["nested_promoted"]].copy()
    promoted["status"] = "validated_production_candidate"
    definitions = []
    for (side, archetype), local in promoted.groupby(
        ["side_name", "archetype_policy_key"], observed=True
    ):
        target = np.asarray(
            daily.index.isin(
                set(
                    calendar.loc[
                        (calendar["side_name"] == side)
                        & (calendar["archetype_policy_key"] == archetype),
                        "day",
                    ]
                )
            ),
            dtype=bool,
        )
        for _, row in local.iterrows():
            definitions.append(fit_definition(daily, target, row))
    contract = ResidualStateFamilyContract(
        schema_version=1,
        definitions=tuple(definitions),
        source_feature_contract_hash=str(negative_residual_feature_contract()["contract_hash"]),
        fit_end=str(daily.index.max()),
    ).with_hash()
    (args.output / "residual_state_family_contract.json").write_text(
        json.dumps(contract.to_dict(), indent=2) + "\n"
    )
    discovery_definitions = []
    for (side, archetype), local in candidates.groupby(
        ["side_name", "archetype_policy_key"], observed=True
    ):
        target = np.asarray(
            daily.index.isin(
                set(
                    calendar.loc[
                        (calendar["side_name"] == side)
                        & (calendar["archetype_policy_key"] == archetype),
                        "day",
                    ]
                )
            ),
            dtype=bool,
        )
        for _, row in local.iterrows():
            row = row.copy()
            row["status"] = "discovery_only"
            discovery_definitions.append(fit_definition(daily, target, row))
    discovery_contract = ResidualStateFamilyContract(
        schema_version=1,
        definitions=tuple(discovery_definitions),
        source_feature_contract_hash=str(negative_residual_feature_contract()["contract_hash"]),
        fit_end=str(daily.index.max()),
    ).with_hash()
    (args.output / "residual_state_discovery_family_contract.json").write_text(
        json.dumps(discovery_contract.to_dict(), indent=2) + "\n"
    )
    registry = candidates.merge(summary[keys + ["nested_promoted"]], on=keys, how="left")
    registry["status"] = np.where(
        registry["nested_promoted"].eq(True),
        "validated_production_candidate",
        "discovery_only",
    )
    registry.to_csv(args.output / "definition_registry.csv", index=False)
    manifest = {
        "schema": "nested_residual_state_family_validation_v1",
        "calendar_cells": int(len(calendar)),
        "frozen_candidate_definitions": int(len(candidates)),
        "nested_evaluation_rows": int(len(folds)),
        "nested_promoted_definitions": int(len(promoted)),
        "family_contract_hash": contract.contract_hash,
        "discovery_family_contract_hash": discovery_contract.contract_hash,
        "evaluation_quarters": [str(value) for value in _quarters()],
        "interpretation": "auxiliary incremental layer; not total residual-state coverage",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path, default=Path("data_perp/features/20260712_185800/symbol=BTC_USD:USD.parquet"))
    parser.add_argument("--calendar", type=Path, default=Path("data_perp/reports/residual_calendar_feature_matches_20260712_v1/calendar_cells_with_feature_matches.csv"))
    parser.add_argument("--candidates", type=Path, default=Path("data_perp/reports/joined_gated_residual_composites_20260712_v1/promoted_gated_composites.csv"))
    parser.add_argument("--output", type=Path, default=Path("data_perp/reports/nested_residual_state_families_20260712_v1"))
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default="2026-07-10")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
