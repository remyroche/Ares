#!/usr/bin/env python3
"""Align transport-supervised tree leaves into conservative soft-rule definitions.

This is deliberately a *discovery* stage.  It operates only on the compact
rule-candidate table and never scores candidates or performs a membership
join.  Consequently it can establish fold recurrence and robust condition
thresholds, but cannot establish out-of-sample support, conditional economic
effect, transport, or promotion readiness.

The discovery runner fits trees on fold-local robust-standardised features.
The threshold coordinate system below is therefore also fold-local robust
standard units.  A later membership materialisation must supply a frozen
feature-centre/scale lineage contract before any definition can be used as an
inference feature.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "data_perp/artifacts/transport_supervised_archetypes_20260803_v1"


def _direction(value: Any) -> int:
    direction = int(value)
    if direction not in (-1, 1):
        raise ValueError(f"Tree-condition direction must be -1 or 1, got {value!r}")
    return direction


def parse_conditions(encoded: str) -> tuple[tuple[str, int, float], ...]:
    """Parse and canonicalise one extracted tree leaf path.

    Trees can theoretically split a feature more than once.  For the same
    feature/direction we keep its tightest condition; contradictory bounds are
    left as separate directions rather than silently discarded.
    """
    value = json.loads(encoded)
    if not isinstance(value, list) or not value:
        raise ValueError("A rule candidate must contain a non-empty JSON condition list")
    bounds: dict[tuple[str, int], float] = {}
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"Invalid tree condition: {item!r}")
        feature, direction_raw, threshold_raw = item
        if not isinstance(feature, str) or not feature:
            raise ValueError(f"Invalid feature in tree condition: {item!r}")
        direction = _direction(direction_raw)
        threshold = float(threshold_raw)
        if not math.isfinite(threshold):
            raise ValueError(f"Non-finite tree threshold: {item!r}")
        key = (feature, direction)
        # x < threshold: lower threshold is tighter.  x > threshold: higher
        # threshold is tighter.
        if key not in bounds or (direction < 0 and threshold < bounds[key]) or (direction > 0 and threshold > bounds[key]):
            bounds[key] = threshold
    return tuple(sorted((feature, direction, threshold) for (feature, direction), threshold in bounds.items()))


def _family_subsets(conditions: tuple[tuple[str, int, float], ...], minimum: int, maximum: int) -> Iterable[tuple[tuple[str, int], ...]]:
    components = tuple((feature, direction) for feature, direction, _threshold in conditions)
    for size in range(minimum, min(maximum, len(components)) + 1):
        yield from itertools.combinations(components, size)


def _median_iqr(values: Iterable[float]) -> tuple[float, float]:
    x = np.asarray(list(values), dtype=float)
    return float(np.median(x)), float(np.quantile(x, 0.75) - np.quantile(x, 0.25))


def _signature(items: tuple[tuple[str, int], ...]) -> str:
    return " & ".join(f"{feature}{'>' if direction > 0 else '<'}" for feature, direction in items)


def build_consensus(
    candidates: pd.DataFrame,
    *,
    minimum_fold_coverage: float = 0.70,
    minimum_conditions: int = 2,
    maximum_conditions: int = 4,
    sigmoid_temperature: float = 0.35,
    maximum_definitions_per_group: int = 12,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return fold-recurrent aligned rule families and discovery-only definitions.

    Alignment is intentionally side/head/effect-direction local.  It uses
    frequent *feature + direction* constituents, not threshold equality;
    thresholds are then aggregated with a median across the recurring leaves.
    """
    required = {"fold", "side_name", "head", "conditions", "leaf_value", "leaf_rows"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Missing candidate columns: {sorted(missing)}")
    if not 0 < minimum_fold_coverage <= 1:
        raise ValueError("minimum_fold_coverage must be in (0, 1]")
    if not 1 <= minimum_conditions <= maximum_conditions <= 5:
        raise ValueError("conditions must satisfy 1 <= minimum <= maximum <= 5")
    if sigmoid_temperature <= 0:
        raise ValueError("sigmoid_temperature must be positive")

    frame = candidates.copy()
    frame["_parsed"] = frame["conditions"].map(parse_conditions)
    frame["_effect_sign"] = np.where(frame["leaf_value"].to_numpy(float) >= 0.0, "favourable", "unfavourable")
    alignments: list[dict[str, Any]] = []
    definitions_by_group: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for (side, head, effect_sign), group in frame.groupby(["side_name", "head", "_effect_sign"], sort=True):
        available_folds = sorted(int(value) for value in group["fold"].unique())
        required_folds = int(math.ceil(len(available_folds) * minimum_fold_coverage))
        family_rows: dict[tuple[tuple[str, int], ...], list[pd.Series]] = defaultdict(list)
        for _index, row in group.iterrows():
            for family in _family_subsets(row["_parsed"], minimum_conditions, maximum_conditions):
                family_rows[family].append(row)

        # Rank after recurrence.  We preserve all rows in the alignment audit,
        # then choose a small non-redundant discovery catalogue.
        group_records: list[dict[str, Any]] = []
        for family, rows in family_rows.items():
            folds = sorted({int(row["fold"]) for row in rows})
            occurrences: dict[tuple[str, int], list[float]] = defaultdict(list)
            for row in rows:
                by_component = {(feature, direction): threshold for feature, direction, threshold in row["_parsed"]}
                for component in family:
                    occurrences[component].append(float(by_component[component]))
            condition_rows = []
            for feature, direction in family:
                threshold, threshold_iqr = _median_iqr(occurrences[(feature, direction)])
                condition_rows.append(
                    {
                        "feature": feature,
                        "direction": int(direction),
                        "operator": ">" if direction > 0 else "<",
                        "threshold_robust_standard_units": threshold,
                        "threshold_iqr_robust_standard_units": threshold_iqr,
                        "sigmoid_temperature_robust_standard_units": float(sigmoid_temperature),
                    }
                )
            leaf_values = [float(row["leaf_value"]) for row in rows]
            leaf_rows = [int(row["leaf_rows"]) for row in rows]
            fold_coverage = len(folds) / max(len(available_folds), 1)
            robust_fold_eligible = len(folds) >= required_folds
            # Candidate leaves do not contain an out-of-fold population nor
            # memberships.  Label this honestly rather than infer support from
            # in-tree node counts.
            support_status = "PENDING_MEMBERSHIP_OOF_EVALUATION"
            record = {
                "side_name": side,
                "head": head,
                "effect_sign": effect_sign,
                "family_signature": _signature(family),
                "n_conditions": len(family),
                "available_folds": json.dumps(available_folds),
                "recurring_folds": json.dumps(folds),
                "n_available_folds": len(available_folds),
                "n_recurring_folds": len(folds),
                "required_recurring_folds": required_folds,
                "fold_coverage": fold_coverage,
                "robust_fold_eligible": robust_fold_eligible,
                "support_status": support_status,
                "support_eligible": False,
                "candidate_leaf_count": len(rows),
                "median_leaf_rows": float(np.median(leaf_rows)),
                "median_leaf_value_bps": float(np.median(leaf_values)),
                "leaf_value_iqr_bps": float(np.quantile(leaf_values, .75) - np.quantile(leaf_values, .25)),
                "conditions_json": json.dumps(condition_rows, sort_keys=True),
                "selection_score": float(fold_coverage * abs(np.median(leaf_values)) * np.log1p(np.median(leaf_rows))),
            }
            group_records.append(record)

        group_records.sort(key=lambda row: (-int(row["robust_fold_eligible"]), -row["selection_score"], row["family_signature"]))
        selected: list[set[str]] = []
        rank = 0
        for record in group_records:
            component_set = set(record["family_signature"].split(" & "))
            redundant = any(len(component_set & previous) / max(len(component_set | previous), 1) >= 0.80 for previous in selected)
            choose = bool(record["robust_fold_eligible"] and not redundant and rank < maximum_definitions_per_group)
            if choose:
                rank += 1
                selected.append(component_set)
                record["archetype_id"] = f"transport_{side}_{head}_{effect_sign}_{rank:02d}"
                record["selected_for_definition"] = True
                definition = {
                    "archetype_id": record["archetype_id"],
                    "side_name": side,
                    "event_head": head,
                    "effect_direction": effect_sign,
                    "conditions": json.loads(record["conditions_json"]),
                    "fold_recurrence": {
                        "available_folds": available_folds,
                        "recurring_folds": json.loads(record["recurring_folds"]),
                        "coverage": record["fold_coverage"],
                        "minimum_required": minimum_fold_coverage,
                    },
                    "discovery_statistics": {
                        "candidate_leaf_count": record["candidate_leaf_count"],
                        "median_leaf_rows": record["median_leaf_rows"],
                        "median_leaf_value_bps": record["median_leaf_value_bps"],
                        "leaf_value_iqr_bps": record["leaf_value_iqr_bps"],
                    },
                    "promotion_status": "DISCOVERY_ONLY_PENDING_OOF_MEMBERSHIP_SUPPORT_EFFECT_AND_TRANSPORT_GATES",
                }
                definitions_by_group[side][head].append(definition)
            else:
                record["archetype_id"] = None
                record["selected_for_definition"] = False
            alignments.append(record)

    alignment = pd.DataFrame(alignments)
    if not alignment.empty:
        alignment = alignment.sort_values(["side_name", "head", "effect_sign", "selected_for_definition", "selection_score"], ascending=[True, True, True, False, False]).reset_index(drop=True)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "promotion_status": "DISCOVERY_ONLY_NO_PROMOTION_CLAIM",
        "coordinate_system": {
            "feature_values": "fold_training_robust_standard_units",
            "thresholds": "median of recurring leaf thresholds in those units",
            "important_limitation": "No frozen raw feature centre/scale contract is present in candidate artifacts. Definitions must not be used for inference until a later stage materialises this lineage.",
        },
        "membership": {
            "form": "geometric_mean_of_directional_sigmoids",
            "overlap": "memberships are independent; they are not a simplex",
            "temperature_robust_standard_units": float(sigmoid_temperature),
        },
        "alignment": {
            "separate_groups": ["side_name", "event_head", "effect_direction"],
            "constituent": "feature + direction",
            "minimum_conditions": minimum_conditions,
            "maximum_conditions": maximum_conditions,
            "minimum_fold_coverage": minimum_fold_coverage,
            "selection": "fold-recurrent families ranked by robust recurrence/effect/support proxy and greedily de-duplicated by >=0.80 condition Jaccard",
        },
        "support_gate": "PENDING_MEMBERSHIP_OOF_EVALUATION; leaf-node rows are not treated as out-of-sample support",
        "definitions": {side: dict(heads) for side, heads in definitions_by_group.items()},
    }
    return alignment, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--minimum-fold-coverage", type=float, default=0.70)
    parser.add_argument("--minimum-conditions", type=int, default=2)
    parser.add_argument("--maximum-conditions", type=int, default=4)
    parser.add_argument("--sigmoid-temperature", type=float, default=0.35)
    parser.add_argument("--maximum-definitions-per-group", type=int, default=12)
    args = parser.parse_args()
    source = args.artifact_root / "archetype_rule_candidates.parquet"
    if not source.exists():
        raise SystemExit(f"Missing rule candidates: {source}")
    candidates = pd.read_parquet(source)
    scaler_source = args.artifact_root / "archetype_feature_scalers.parquet"
    if not scaler_source.exists():
        raise SystemExit(
            "Missing fold-specific feature scaler lineage; do not align rules "
            "for inference membership without archetype_feature_scalers.parquet"
        )
    scalers = pd.read_parquet(scaler_source)
    required_scaler_columns = {"fold", "side_name", "head", "feature", "center", "scale"}
    if required_scaler_columns.difference(scalers.columns):
        raise SystemExit("feature scaler lineage is incomplete")
    alignment, definitions = build_consensus(
        candidates,
        minimum_fold_coverage=args.minimum_fold_coverage,
        minimum_conditions=args.minimum_conditions,
        maximum_conditions=args.maximum_conditions,
        sigmoid_temperature=args.sigmoid_temperature,
        maximum_definitions_per_group=args.maximum_definitions_per_group,
    )
    definitions["coordinate_system"] = {
        "feature_values": "fold_training_robust_standard_units",
        "thresholds": "median of recurring leaf thresholds in those units",
        "inference_conversion": "for each scored outer fold, use only that fold's earlier-only center/scale from archetype_feature_scalers.parquet before applying the soft rule",
        "scaler_lineage": "fold × side × event-head × feature; earlier-only strict-OOF setup-residual rows",
        "scaler_rows": int(len(scalers)),
    }
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    alignment.to_parquet(args.artifact_root / "archetype_rule_alignment.parquet", index=False)
    (args.artifact_root / "archetype_consensus_definitions.json").write_text(json.dumps(definitions, indent=2, sort_keys=True) + "\n")
    selected = int(alignment["selected_for_definition"].sum()) if not alignment.empty else 0
    eligible = int(alignment["robust_fold_eligible"].sum()) if not alignment.empty else 0
    print(json.dumps({"alignment_rows": len(alignment), "robust_fold_eligible_families": eligible, "discovery_definitions": selected, "promotion_status": definitions["promotion_status"]}))


if __name__ == "__main__":
    main()
