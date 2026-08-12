#!/usr/bin/env python3
"""Materialise the bounded Stage-VI causal/path archetype artifact bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG
from extreme_price_movements.stage_vi_archetype_runner import (
    StageVIRunnerSpec,
    materialize_stage_vi_view_contract,
    run_stage_vi_archetype_funnel,
)


def _json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _feature_list(path: Path) -> list[str]:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, Mapping):
            value = value.get("selected_features", value.get("features"))
        if not isinstance(value, list):
            raise ValueError("selected-feature JSON requires features/selected_features list")
        return list(map(str, value))
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _score_contract(
    path: Path,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    """Load candidate-bound score/OOF mappings; generic attribution is forbidden."""

    value = _json(path)
    scores: dict[str, dict[str, str]] = {}
    flags: dict[str, dict[str, str]] = {}
    arms = {"control", "base", "meta", "both"}
    for candidate_id, contract in value.items():
        if not isinstance(contract, Mapping):
            raise ValueError(f"score contract for {candidate_id} must be an object")
        candidate_scores = contract.get("scores")
        candidate_flags = contract.get("oof_flags")
        if not isinstance(candidate_scores, Mapping) or set(candidate_scores) != arms:
            raise ValueError(f"{candidate_id} requires four candidate-specific scores")
        if not isinstance(candidate_flags, Mapping) or set(candidate_flags) != arms:
            raise ValueError(f"{candidate_id} requires four candidate-specific OOF flags")
        scores[str(candidate_id)] = {
            str(arm): str(column) for arm, column in candidate_scores.items()
        }
        flags[str(candidate_id)] = {
            str(arm): str(column) for arm, column in candidate_flags.items()
        }
    if not scores:
        raise ValueError("score contract must contain at least one candidate")
    return scores, flags


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--selected-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--score-contract", type=Path, required=True,
        help="JSON candidate -> {scores, oof_flags}; each arm is candidate-bound",
    )
    parser.add_argument("--path-views", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument(
        "--full-grid", action="store_true",
        help="explicitly opt into all 925 fits instead of the bounded default funnel",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--min-side-rows", type=int, default=250)
    parser.add_argument("--min-component-rows", type=int, default=20)
    args = parser.parse_args()

    ledger = pd.read_parquet(args.ledger)
    path_views = _json(args.path_views) if args.path_views else None
    views = materialize_stage_vi_view_contract(
        ledger,
        config=CFG,
        selected_causal_columns=_feature_list(args.selected_features),
        explicit_path_views=path_views,
    )
    candidates = None
    if args.candidates:
        value = json.loads(args.candidates.read_text(encoding="utf-8"))
        if not isinstance(value, list):
            raise ValueError("candidate filter must be a JSON list of predeclared IDs")
        candidates = list(map(str, value))
    score_columns, oof_columns = _score_contract(args.score_contract)
    result = run_stage_vi_archetype_funnel(
        ledger,
        views=views,
        output_directory=args.output,
        spec=StageVIRunnerSpec(
            folds=args.folds,
            min_side_rows=args.min_side_rows,
            min_component_rows=args.min_component_rows,
            full_grid=args.full_grid,
            arm_score_columns_by_candidate=score_columns,
            arm_oof_flag_columns_by_candidate=oof_columns,
        ),
        candidate_ids=candidates,
    )
    print(json.dumps({
        "output": str(result.output_directory),
        "candidate_count": int(len(result.candidate_audit)),
        "decision_rows": int(len(result.decision_matrix)),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
