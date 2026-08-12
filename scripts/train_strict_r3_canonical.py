#!/usr/bin/env python3
"""Train one current monthly-upstream or frozen-geometry conversion bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    persist_four_week_conversion_bundle,
    persist_monthly_upstream_bundle,
    train_four_week_conversion_bundle,
    train_monthly_upstream_bundle,
)
from extreme_price_movements.strict_r3_canonical_v2 import load_geometry_bundle  # noqa: E402


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _base_fields(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    fields = payload.get("base_fields_by_side", {}).get("long", payload.get("base_fields", []))
    return [str(field) for field in fields]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", choices=("upstream", "conversion"), required=True)
    parser.add_argument("--prequential-ledger", type=Path, required=True)
    parser.add_argument("--geometry-bundle", type=Path)
    parser.add_argument("--prior42-features", type=Path)
    parser.add_argument(
        "--feature-patch", type=Path,
        help=(
            "optional target-free feature patch keyed by candidate_id; finite "
            "frozen base fields replace only matching pre-cutoff ledger rows"
        ),
    )
    parser.add_argument("--feature-contract", type=Path, required=True)
    parser.add_argument("--cutoff", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    ledger = pd.read_parquet(args.prequential_ledger)
    ledger = ledger.loc[ledger["side_name"].astype(str).str.lower().eq("long")].copy()
    source_hashes = {
        "prequential_ledger": _sha(args.prequential_ledger),
        "feature_contract": _sha(args.feature_contract),
    }
    patch_audit = pd.DataFrame()
    if args.feature_patch is not None:
        patch = pd.read_parquet(args.feature_patch)
        patch_fields = _base_fields(args.feature_contract)
        required_patch = {"candidate_id", *patch_fields}
        missing_patch = sorted(required_patch.difference(patch.columns))
        if missing_patch:
            parser.error(f"--feature-patch lacks required columns: {missing_patch}")
        if patch["candidate_id"].duplicated().any():
            parser.error("--feature-patch has duplicate candidate IDs")
        prohibited = [
            column for column in patch.columns
            if any(token in column.lower() for token in (
                "label", "outcome", "policy_net", "gross_bps", "h12_", "target_invalid", "path_valid",
            ))
        ]
        if prohibited:
            parser.error(f"--feature-patch is not target-free: {prohibited}")
        if "__decision_ts__" in patch.columns:
            patch_ts = pd.to_datetime(patch["__decision_ts__"], utc=True, errors="coerce")
            cutoff = pd.to_datetime(args.cutoff, utc=True).normalize()
            if patch_ts.isna().any() or patch_ts.ge(cutoff).any():
                parser.error("--feature-patch must contain only valid pre-cutoff rows")
        merged = ledger.loc[:, ["candidate_id", *patch_fields]].merge(
            patch.loc[:, ["candidate_id", *patch_fields]],
            on="candidate_id", how="left", validate="one_to_one", suffixes=("", "__patch"),
        )
        audit_rows = []
        for field in patch_fields:
            replacement = pd.to_numeric(merged[f"{field}__patch"], errors="coerce")
            mask = replacement.notna()
            before = pd.to_numeric(ledger[field], errors="coerce")
            changed = mask & ~before.eq(replacement)
            ledger.loc[mask, field] = replacement.loc[mask].to_numpy()
            audit_rows.append({
                "feature": field,
                "patched_rows": int(mask.sum()),
                "changed_rows": int(changed.sum()),
                "remaining_null_rows": int(pd.to_numeric(ledger[field], errors="coerce").isna().sum()),
            })
        patch_audit = pd.DataFrame(audit_rows)
        source_hashes["feature_patch"] = _sha(args.feature_patch)
        source_hashes["feature_patch_target_free"] = "true"
    if args.layer == "upstream":
        if args.prior42_features is None:
            parser.error("--prior42-features is required for --layer upstream")
        if "teacher_base_rank42" not in ledger:
            if "prequential_base_rank42" not in ledger:
                parser.error(
                    "upstream training requires teacher_base_rank42 or its explicit "
                    "strict-prequential warm-start alias prequential_base_rank42"
                )
            ledger["teacher_base_rank42"] = ledger["prequential_base_rank42"]
            source_hashes["teacher_rank_alias"] = "prequential_base_rank42"
        prior42 = pd.read_parquet(args.prior42_features)
        source_hashes["prior42_features"] = _sha(args.prior42_features)
        bundle = train_monthly_upstream_bundle(
            cutoff=args.cutoff,
            training_ledger=ledger,
            prior42_features=prior42,
            base_fields=_base_fields(args.feature_contract),
            source_hashes=source_hashes,
        )
        manifest = persist_monthly_upstream_bundle(bundle, args.out_dir)
        if not patch_audit.empty:
            patch_audit.to_parquet(args.out_dir / "feature_patch_audit.parquet", index=False)
    else:
        if args.geometry_bundle is None:
            parser.error("--geometry-bundle is required for --layer conversion")
        geometry = load_geometry_bundle(args.geometry_bundle)
        audit = geometry.fit_audit
        if (
            audit.get("definition_start") != "2024-10-01T00:00:00+00:00"
            or audit.get("definition_end_exclusive") != "2025-01-01T00:00:00+00:00"
        ):
            parser.error("--geometry-bundle must be the frozen Oct-Dec 2024 definition")
        source_hashes.update({
            "geometry_manifest": _sha(args.geometry_bundle / "run_manifest.json"),
        })
        bundle = train_four_week_conversion_bundle(
            cutoff=args.cutoff,
            upstream_ledger=ledger,
            frozen_geometry=geometry,
            base_fields=_base_fields(args.feature_contract),
            source_hashes=source_hashes,
        )
        manifest = persist_four_week_conversion_bundle(bundle, args.out_dir)
    print(json.dumps({"event": "complete", "output": str(args.out_dir), **manifest}, default=str))


if __name__ == "__main__":
    main()
