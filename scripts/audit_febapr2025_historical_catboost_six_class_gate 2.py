#!/usr/bin/env python3
"""Gate a predeclared historical-only six-class CatBoost taxonomy.

This leaves the global production seven-class contract unchanged.  The two
merges are fixed before this audit: the established fast pair and the semantic
pair ``early_mfe_full_reversal`` + ``noisy_timeout_usable_mfe``.  Economics is
reported for audit only and does not select a taxonomy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.path_archetype_support import (  # noqa: E402
    PathArchetypeSupportConfig,
    validate_path_archetype_support,
)
from scripts.audit_febapr2025_historical_catboost_merge_gate import (  # noqa: E402
    IDENTITY, _identity_hash, _sha256, _write_json,
)

SCHEMA = "febapr2025_historical_catboost_six_class_gate_v1"
CLASS_ORDER = (
    "immediate_adverse_path", "fast_realization_winner", "late_breakout",
    "slow_grinder", "mfe_reversal_or_timeout", "dead_timeout",
)
TAXONOMY = {
    "version": "historical_execution_12h_six_class_v1",
    "scope": "historical_only; does_not_modify_global_production_7class_contract",
    "class_order": list(CLASS_ORDER),
    "predeclared_merges": {
        "fast_realization_winner": ["fast_clean_winner", "fast_winner_early_drawdown"],
        "mfe_reversal_or_timeout": ["early_mfe_full_reversal", "noisy_timeout_usable_mfe"],
    },
    "selection_rule": "No held-out EV selection: mapping is fixed before audit.",
}


def _labels(root: Path) -> pd.DataFrame:
    index = json.loads((root / "index.json").read_text())
    if not index.get("coverage", {}).get("complete"):
        raise ValueError("exact label index is incomplete")
    columns = [
        *IDENTITY, "path_shape_archetype", "path_arch_complete_12h",
        "path_arch_peak_mfe_r", "path_arch_final_return_r",
        "path_arch_time_to_first_meaningful_mfe_h", "path_arch_peak_mfe_atr",
    ]
    frames = []
    for shard in index["shards"]:
        path = Path(shard["labels"])
        if _sha256(path) != shard["sha256"]:
            raise ValueError(f"shard digest mismatch: {path}")
        frames.append(pd.read_parquet(path, columns=columns))
    result = pd.concat(frames, ignore_index=True)
    if len(result) != int(index["coverage"]["expected_rows"]) or result.duplicated(IDENTITY).any():
        raise ValueError("sealed label identity coverage is invalid")
    if not result["path_arch_complete_12h"].astype(bool).all():
        raise ValueError("incomplete paths cannot enter a taxonomy gate")
    return result


def _six(values: pd.Series) -> pd.Series:
    mapped = values.astype("string").replace({
        "fast_clean_winner": "fast_realization_winner",
        "fast_winner_early_drawdown": "fast_realization_winner",
        "early_mfe_full_reversal": "mfe_reversal_or_timeout",
        "noisy_timeout_usable_mfe": "mfe_reversal_or_timeout",
    })
    invalid = sorted(set(mapped.dropna().astype(str)).difference(CLASS_ORDER))
    if invalid:
        raise ValueError(f"unexpected raw path shape(s): {invalid}")
    return mapped


def _source_audit(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = ["early_mfe_full_reversal", "noisy_timeout_usable_mfe"]
    source = frame.loc[frame["path_shape_archetype"].isin(raw)].copy()
    # This is a semantic guard, not an EV-based merge selection: both classes
    # must have a usable favorable R excursion and a nonpositive raw timeout.
    semantic = source["path_arch_peak_mfe_r"].ge(0.5) & source["path_arch_final_return_r"].le(0.0)
    if not semantic.all():
        raise ValueError("predeclared reversal/timeout source labels violate their path semantics")
    aggregate = source.groupby(["side_name", "path_shape_archetype"], sort=True).agg(
        rows=("candidate_id", "size"),
        mean_peak_mfe_r=("path_arch_peak_mfe_r", "mean"), median_peak_mfe_r=("path_arch_peak_mfe_r", "median"),
        mean_peak_mfe_atr=("path_arch_peak_mfe_atr", "mean"),
        mean_final_return_r=("path_arch_final_return_r", "mean"), median_final_return_r=("path_arch_final_return_r", "median"),
        mean_time_to_usable_mfe_h=("path_arch_time_to_first_meaningful_mfe_h", "mean"),
        mean_execution_net_ev_12h=("execution_net_ev_12h", "mean"), median_execution_net_ev_12h=("execution_net_ev_12h", "median"),
    ).reset_index()
    evidence = {
        "semantic_contract": "each source row has peak_mfe_r >= 0.5 and final_return_r <= 0; execution EV is audit-only",
        "semantic_rows": int(len(source)), "semantic_passed": True,
        "outcome_not_used_for_mapping": True,
    }
    return aggregate, evidence


def run(*, labels_root: Path, context_index: Path, population: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = _labels(labels_root)
    context = pd.read_parquet(context_index, columns=IDENTITY)
    if _identity_hash(labels, feature_store_symbol=True) != _identity_hash(context):
        raise ValueError("labels and PIT context identities do not match")
    economics = pd.read_parquet(population, columns=["candidate_id", "execution_net_ev_12h"])
    frame = labels.merge(economics, on="candidate_id", validate="one_to_one")
    frame["path_geometry_label"] = _six(frame["path_shape_archetype"])
    audit, semantic = _source_audit(frame)
    audit.to_csv(output_dir / "mfe_reversal_or_timeout_source_audit.csv", index=False)
    side_reports: dict[str, Any] = {}
    passed = True
    for side in ("long", "short"):
        cohort = frame.loc[frame["side_name"].eq(side)]
        result = validate_path_archetype_support(
            cohort,
            PathArchetypeSupportConfig(
                label_column="path_geometry_label", timestamp_column="__ts__", side_column="side_name",
                classes=CLASS_ORDER, min_global_class_share=0.01, min_month_side_class_share=0.005,
            ),
        )
        for name, table in (("global", result.global_support), ("month", result.month_support), ("month_side", result.month_side_support), ("violations", result.violations)):
            table.to_csv(output_dir / f"{side}_{name}_support.csv", index=False)
        side_reports[side] = {"accepted": result.accepted, "rows": result.rows, "violations": result.violations.to_dict(orient="records")}
        passed &= result.accepted
    manifest = {
        "schema": SCHEMA,
        "status": "PASS_READY_FOR_SIDE_LOCAL_FS_HPO_GEOMETRY" if passed else "BLOCKED_CLASS_SUPPORT",
        "passed": bool(passed), "taxonomy": TAXONOMY,
        "labels": {"root": str(labels_root), "index_sha256": _sha256(labels_root / "index.json"), "rows": int(len(labels))},
        "pit_context": {"index": str(context_index), "sha256": _sha256(context_index), "identity_exact_match": True},
        "source_path_and_economic_audit": {"csv": "mfe_reversal_or_timeout_source_audit.csv", **semantic},
        "side_local_support": side_reports,
        "competing_risk_challenger": "separate 3-class model; never a six-class taxonomy alias",
        "12h_vs_frozen_v6_24h": "12h execution-compatible derivative only; frozen v6 uses 24h paths.",
        "models_trained": False,
        "next_action": "side-local FS/HPO/geometry/strict OOF with this explicit class-order manifest" if passed else "do not train",
    }
    _write_json(output_dir / "gate.json", manifest)
    _write_json(output_dir / "class_order_manifest.json", TAXONOMY)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels-root", type=Path, required=True)
    parser.add_argument("--context-index", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(labels_root=args.labels_root, context_index=args.context_index, population=args.population, output_dir=args.output_dir), sort_keys=True))


if __name__ == "__main__":
    main()
