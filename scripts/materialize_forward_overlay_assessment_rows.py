#!/usr/bin/env python3
"""Attach shadow-only forward overlay diagnostics to a scored candidate ledger.

This script is intentionally observational.  It materializes the frozen
short-default leverage-rebuild family columns and preserves a precomputed
short-default uncertainty score only when the caller supplied one.  It never
changes the source rank or creates an admission decision.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pandas as pd

from extreme_price_movements.residual_state_family_features import (
    ResidualStateFamilyContract,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "data_perp/reports/forward_overlay_assessment_bundle_20260714_v1"
SIDE = "short"
ARCHETYPE = "short_default_clean_path"


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_column(frame: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in frame:
            return name
    raise KeyError(f"Expected one of {names}, got no matching column")


def run(args: argparse.Namespace) -> dict[str, object]:
    manifest = _load_json(args.bundle / "manifest.json")
    if manifest.get("status") != "shadow_forward_assessment_only":
        raise ValueError("Only shadow-forward assessment bundles may be materialized here.")
    family_source = manifest["source_artifacts"]["residual_state_family_contract"]
    family_path = ROOT / str(family_source["path"])
    contract = ResidualStateFamilyContract.from_dict(_load_json(family_path))
    expected_hash = str(family_source["sha256"])
    if _sha256(family_path) != expected_hash:
        raise ValueError("Residual-state family contract hash mismatch; refusing shadow materialization.")

    frame = pd.read_parquet(args.input)
    side_column = _resolve_column(frame, "side_name")
    archetype_column = _resolve_column(
        frame,
        "archetype_policy_key",
        "policy_archetype",
        "local_side_archetype",
        "source_archetype",
    )
    primitive_columns = ["short_covering_score_market", "funding_confirmed_long_flush"]
    missing_primitives = [name for name in primitive_columns if name not in frame]
    if missing_primitives:
        raise KeyError(
            "The forward ledger cannot materialize the validated leverage-rebuild "
            f"diagnostic without {missing_primitives}."
        )
    family = contract.transform(frame, frame[side_column], frame[archetype_column])
    for column in family:
        frame[column] = family[column].to_numpy(np.float32, copy=False)

    local = (
        frame[side_column].astype(str).eq(SIDE)
        & frame[archetype_column].astype(str).eq(ARCHETYPE)
    )
    leverage = frame["residual_state_family_leverage_rebuild_pct"].to_numpy(np.float32)
    active = frame["residual_state_family_leverage_rebuild_active"].to_numpy(bool)
    computable = frame["residual_state_family_leverage_rebuild_computable"].to_numpy(bool)
    # The bundle intentionally has no inferred threshold.  This marker merely
    # identifies rows eligible for later resolved-outcome analysis.
    frame["leverage_rebuild_shadow_eligible"] = (local.to_numpy() & active & computable).astype(np.int8)
    frame["leverage_rebuild_shadow_score"] = np.where(local.to_numpy(), leverage, 0.0).astype(np.float32)
    uncertainty_ready = "short_default_uncertainty_score" in frame
    frame["short_default_uncertainty_score_available"] = np.int8(uncertainty_ready)
    if not uncertainty_ready:
        frame["short_default_uncertainty_score"] = np.nan
    if "short_default_uncertainty_rank" not in frame:
        frame["short_default_uncertainty_rank"] = np.nan
    frame["forward_overlay_bundle_id"] = args.bundle.name
    frame["forward_overlay_bundle_shadow_only"] = np.int8(1)
    frame["forward_overlay_family_contract_sha256"] = expected_hash
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output, index=False, compression="zstd")
    result = {
        "rows": int(len(frame)),
        "short_default_rows": int(local.sum()),
        "leverage_rebuild_eligible_rows": int(frame["leverage_rebuild_shadow_eligible"].sum()),
        "uncertainty_score_available": uncertainty_ready,
        "rank_or_policy_changed": False,
        "output": str(args.output),
    }
    (args.output.with_suffix(".manifest.json")).write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))
