#!/usr/bin/env python3
"""Materialize a train_meta feature set from direct cross-asset context evidence.

The family ablation tells us which feature families help at least one side x
archetype cell.  This script turns that into a concrete handoff:

* include base score features plus features from accepted families;
* keep accepted-cell evidence in the manifest/report only;
* do not add future-derived cell pass/fail flags to model inputs;
* preserve outcome columns for training/evaluation, not as features.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_direct_cross_asset_meta_ablation import BASELINE_VARIANT, TARGET_COL, _json_safe  # noqa: E402


DEFAULT_CONTEXT_DIR = Path(
    "data_perp/reports/contextual_tp_sl_ablation_workflow_v14_runtime_health_20260701/"
    "direct_cross_asset_meta_context_v1"
)
DEFAULT_HANDOFF = DEFAULT_CONTEXT_DIR / "direct_cross_asset_meta_context_handoff.parquet"
DEFAULT_FAMILY_ABLATION_DIR = DEFAULT_CONTEXT_DIR / "direct_cross_asset_family_ablation_v1"
DEFAULT_OUT_DIR = DEFAULT_CONTEXT_DIR / "train_meta_direct_context_feature_set_v1"

KEY_COLUMNS = ("__ts__", "__symbol__", "month", "side_name", "source_archetype", "strategy_id")
OUTCOME_COLUMNS = (
    TARGET_COL,
    "exec_net_return",
    "positive_ev_after_1pct",
    "full_sl",
    "timeout",
    "clean_exec_proxy",
    "net_return",
    "gross_return",
    "simple_policy_exit_reason",
)
SAFE_METADATA_COLUMNS = (
    "timestamp",
    "symbol",
    "side",
    "head",
    "policy_target_holding_hours",
    "contextual_tp_sl_arm",
    "simple_grid_selected_sl_mult",
    "simple_grid_selected_tp_mult",
)


def _load_family_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _accepted_family_summary(
    accepted: pd.DataFrame,
    *,
    min_months: int,
    min_cells: int,
    min_mean_delta_ev: float,
) -> pd.DataFrame:
    if accepted.empty:
        return pd.DataFrame(
            columns=[
                "family",
                "accepted_rows",
                "accepted_months",
                "accepted_side_archetype_cells",
                "mean_delta_ev",
                "mean_delta_precision",
                "mean_delta_full_sl",
                "mean_delta_timeout",
                "promotion_tier",
            ]
        )
    summary = accepted.groupby("variant", as_index=False).agg(
        accepted_rows=("variant", "size"),
        accepted_months=("month", "nunique"),
    )
    # pandas named agg cannot count unique tuples directly portably.
    tuple_counts = (
        accepted.assign(_cell=accepted["side_name"].astype(str) + "|" + accepted["source_archetype"].astype(str))
        .groupby("variant")["_cell"]
        .nunique()
    )
    summary["accepted_side_archetype_cells"] = summary["variant"].map(tuple_counts).astype(int)
    metrics = accepted.groupby("variant").agg(
        mean_delta_ev=("delta_mean_ev_after_1pct", "mean"),
        mean_delta_precision=("delta_precision_positive_ev", "mean"),
        mean_delta_full_sl=("delta_full_sl_rate", "mean"),
        mean_delta_timeout=("delta_timeout_rate", "mean"),
    )
    summary = summary.join(metrics, on="variant")
    summary = summary.rename(columns={"variant": "family"})
    conditions_primary = (
        (summary["accepted_months"] >= int(min_months))
        & (summary["accepted_side_archetype_cells"] >= int(min_cells))
        & (summary["mean_delta_ev"] > float(min_mean_delta_ev))
    )
    conditions_cell = summary["accepted_rows"].gt(0) & summary["mean_delta_ev"].gt(0)
    summary["promotion_tier"] = np.where(
        conditions_primary,
        "primary_context",
        np.where(conditions_cell, "cell_context", "diagnostic_only"),
    )
    return summary.sort_values(["promotion_tier", "mean_delta_ev"], ascending=[True, False])


def _family_features(manifest: dict[str, Any], family: str) -> list[str]:
    contract = manifest.get("family_contract", {}).get(family, {})
    features = contract.get("features", [])
    if not isinstance(features, list):
        return []
    return [str(col) for col in features]


def _build_feature_columns(
    family_manifest: dict[str, Any],
    accepted_summary: pd.DataFrame,
    *,
    include_tiers: set[str],
) -> tuple[list[str], dict[str, list[str]]]:
    selected_families = accepted_summary[accepted_summary["promotion_tier"].isin(include_tiers)]["family"].tolist()
    if "f00_score_only" not in selected_families:
        selected_families = ["f00_score_only"] + selected_families
    family_to_features: dict[str, list[str]] = {}
    ordered: list[str] = []
    seen: set[str] = set()
    for family in selected_families:
        cols = _family_features(family_manifest, family)
        family_to_features[family] = cols
        for col in cols:
            if col not in seen:
                seen.add(col)
                ordered.append(col)
    return ordered, family_to_features


def _write_report(
    path: Path,
    manifest: dict[str, Any],
    accepted_summary: pd.DataFrame,
    selected_features: list[str],
) -> None:
    lines = [
        "# Train Meta Direct Context Feature Set",
        "",
        "## Status",
        "",
        f"- Rows: `{manifest['rows']}`",
        f"- Feature columns: `{manifest['feature_count']}`",
        f"- Selected families: `{', '.join(manifest['selected_families'])}`",
        "- The handoff contains raw/OOS-style context features only; accepted-cell labels are report metadata, not model inputs.",
        "- No stability-prior features are used.",
        "",
        "## Accepted Family Summary",
        "",
        accepted_summary.to_markdown(index=False) if not accepted_summary.empty else "No accepted families.",
        "",
        "## Feature Columns",
        "",
        "\n".join(f"- `{col}`" for col in selected_features[:200]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    handoff_path: Path,
    family_ablation_dir: Path,
    output_dir: Path,
    min_months: int,
    min_cells: int,
    min_mean_delta_ev: float,
    include_cell_context: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    family_manifest = _load_family_manifest(family_ablation_dir / "manifest.json")
    accepted_path = family_ablation_dir / "direct_cross_asset_family_ablation_accepted_cells.csv"
    if not accepted_path.exists():
        raise FileNotFoundError(accepted_path)
    accepted = pd.read_csv(accepted_path)
    accepted_summary = _accepted_family_summary(
        accepted,
        min_months=min_months,
        min_cells=min_cells,
        min_mean_delta_ev=min_mean_delta_ev,
    )
    include_tiers = {"primary_context"}
    if include_cell_context:
        include_tiers.add("cell_context")
    feature_columns, family_to_features = _build_feature_columns(
        family_manifest,
        accepted_summary,
        include_tiers=include_tiers,
    )

    frame = pd.read_parquet(handoff_path)
    existing_features = [col for col in feature_columns if col in frame.columns]
    missing_features = sorted(set(feature_columns).difference(frame.columns))
    keep_cols: list[str] = []
    for col in list(KEY_COLUMNS) + list(SAFE_METADATA_COLUMNS) + list(OUTCOME_COLUMNS) + existing_features:
        if col in frame.columns and col not in keep_cols:
            keep_cols.append(col)
    out = frame[keep_cols].copy()
    # Downcast feature-like numeric columns for handoff size.  Outcome columns
    # are also numeric, but preserving float32 is enough for training/eval.
    for col in out.columns:
        if pd.api.types.is_float_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
        elif pd.api.types.is_integer_dtype(out[col]) and col not in {"month"}:
            out[col] = pd.to_numeric(out[col], errors="coerce", downcast="integer")

    outputs = {
        "handoff": output_dir / "train_meta_direct_context_handoff.parquet",
        "feature_manifest": output_dir / "train_meta_direct_context_feature_manifest.json",
        "accepted_summary": output_dir / "train_meta_direct_context_accepted_family_summary.csv",
        "accepted_cells": output_dir / "train_meta_direct_context_accepted_cells.csv",
        "report": output_dir / "train_meta_direct_context_feature_set.md",
        "manifest": output_dir / "manifest.json",
    }
    out.to_parquet(outputs["handoff"], index=False)
    accepted_summary.to_csv(outputs["accepted_summary"], index=False)
    accepted.to_csv(outputs["accepted_cells"], index=False)
    selected_families = [family for family in family_to_features if family != "f00_score_only"]
    feature_manifest = {
        "scope": "train_meta_direct_context_feature_set",
        "feature_columns": existing_features,
        "feature_count": len(existing_features),
        "missing_features": missing_features,
        "families": family_to_features,
        "selected_families": selected_families,
        "include_cell_context": bool(include_cell_context),
        "outcome_columns": [col for col in OUTCOME_COLUMNS if col in out.columns],
        "key_columns": [col for col in KEY_COLUMNS if col in out.columns],
        "no_leakage_contract": {
            "accepted_cells": "audit metadata only; not joined as model input flags",
            "feature_columns": "raw live-predictable context, base score features, OOF/prior-fold context scores",
            "stability_features": "excluded",
        },
    }
    outputs["feature_manifest"].write_text(json.dumps(_json_safe(feature_manifest), indent=2), encoding="utf-8")
    manifest = {
        "scope": "train_meta_direct_context_feature_set",
        "handoff_path": str(handoff_path),
        "family_ablation_dir": str(family_ablation_dir),
        "output_dir": str(output_dir),
        "rows": int(len(out)),
        "columns": int(len(out.columns)),
        "feature_count": int(len(existing_features)),
        "selected_families": selected_families,
        "selected_family_count": int(len(selected_families)),
        "accepted_family_count": int(len(accepted_summary[accepted_summary["promotion_tier"].isin(include_tiers)])),
        "missing_feature_count": int(len(missing_features)),
        "outputs": {key: str(path) for key, path in outputs.items()},
        "no_leakage_contract": feature_manifest["no_leakage_contract"],
    }
    outputs["manifest"].write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    _write_report(outputs["report"], manifest, accepted_summary, existing_features)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-path", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--family-ablation-dir", type=Path, default=DEFAULT_FAMILY_ABLATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-months", type=int, default=4)
    parser.add_argument("--min-cells", type=int, default=3)
    parser.add_argument("--min-mean-delta-ev", type=float, default=0.0)
    parser.add_argument("--primary-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = run(
        handoff_path=args.handoff_path,
        family_ablation_dir=args.family_ablation_dir,
        output_dir=args.output_dir,
        min_months=int(args.min_months),
        min_cells=int(args.min_cells),
        min_mean_delta_ev=float(args.min_mean_delta_ev),
        include_cell_context=not bool(args.primary_only),
    )
    print(json.dumps(_json_safe(manifest), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
