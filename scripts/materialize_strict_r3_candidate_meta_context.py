#!/usr/bin/env python3
"""Join the reusable exact170 candidate-specific meta panel to an MDA surface.

The exact170 panel is target-free and was generated from decision-time price,
OI/funding and authoritative historical book primitives.  This utility exposes
only fields whose ownership is a meta/context family in ``config.py``.  It
never carries labels, policy results, or future-path eligibility into the
sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.config import CFG  # noqa: E402


IDENTITY = ("candidate_id", "__ts__", "__symbol__", "side_name")


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _meta_owned_fields(manifest: dict[str, object]) -> list[str]:
    """Return exact170 fields owned by a declared meta/context family."""

    names: set[str] = set()
    group_names = [
        "PERP_META_PRIMARY_FEATURE_KEYS", "RESIDUAL_META_FEATURE_KEYS",
        "ORDERBOOK_META_FEATURE_KEYS", "FUNDING_META_FEATURE_KEYS",
        "CROSS_ASSET_META_FEATURE_KEYS", "INTERACTION_META_FEATURE_KEYS",
        "OI_FUNDING_META_CANDIDATE_FEATURE_KEYS", "MARKET_CROSS_SECTIONAL_META_FEATURE_KEYS",
        "MODEL_REGIME_CONTEXT_META_FEATURE_KEYS", "MODEL_REGIME_XS_META_FEATURE_KEYS",
        "MODEL_REGIME_TAIL_META_FEATURE_KEYS", "MODEL_REGIME_EIGEN_META_FEATURE_KEYS",
        "MARKET_SPECTRAL_POSITION_META_FEATURE_KEYS", "OI_WEIGHTED_LOCATION_META_FEATURE_KEYS",
        "WEEKLY_SR_META_FEATURE_KEYS", "VOLUME_FREE_PERP_META_FEATURE_KEYS",
    ]
    for group in group_names:
        value = CFG.get(group, ())
        if isinstance(value, (list, tuple)):
            names.update(map(str, value))
    requested = [str(field) for field in manifest.get("requested_fields", [])]
    selected = [field for field in requested if field in names]
    if len(selected) < 40:
        raise ValueError(f"exact170 meta ownership overlap unexpectedly small: {len(selected)}")
    return list(dict.fromkeys(selected))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate-surface", type=Path, action="append", required=True,
        help="One or more target-free candidate surfaces; concatenate only by immutable candidate ID.",
    )
    parser.add_argument("--exact170-panel", type=Path, required=True)
    parser.add_argument("--exact170-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output exists: {args.out_dir}")
    manifest = json.loads(args.exact170_manifest.read_text())
    fields = _meta_owned_fields(manifest)
    panel_schema = set(pq.ParquetFile(args.exact170_panel).schema.names)
    missing = sorted(set(("__ts__", "__symbol__", *fields)).difference(panel_schema))
    if missing:
        raise KeyError(f"exact170 panel lacks requested meta fields: {missing}")
    candidate_parts: list[pd.DataFrame] = []
    for surface_path in args.candidate_surface:
        surface_schema = set(pq.ParquetFile(surface_path).schema.names)
        missing = sorted(set(IDENTITY).difference(surface_schema))
        if missing:
            raise KeyError(f"candidate surface lacks identity fields: {missing}")
        candidate_parts.append(pd.read_parquet(surface_path, columns=list(IDENTITY)))
    candidates = pd.concat(candidate_parts, ignore_index=True)
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True, errors="raise")
    if candidates["candidate_id"].duplicated().any():
        raise AssertionError("candidate surface has duplicate IDs")
    start, end = candidates["__ts__"].min(), candidates["__ts__"].max() + pd.Timedelta(nanoseconds=1)
    source = pd.read_parquet(
        args.exact170_panel, columns=["__ts__", "__symbol__", *fields],
        filters=[("__ts__", ">=", start), ("__ts__", "<", end)],
    )
    source["__ts__"] = pd.to_datetime(source["__ts__"], utc=True, errors="raise")
    if source.duplicated(["__ts__", "__symbol__"]).any():
        raise AssertionError("exact170 feature panel has duplicate timestamp/symbol identities")
    joined = candidates.merge(
        source, on=["__ts__", "__symbol__"], how="left", validate="one_to_one",
        indicator="__exact170_source_matched",
    )
    if len(joined) != len(candidates) or joined["candidate_id"].duplicated().any():
        raise AssertionError("exact170 join changed candidate identity")
    # The one-hour boundary immediately before the exact170 panel begins is a
    # legitimate causal feature warm-up gap, not evidence that a candidate was
    # unavailable.  Preserve its identity and null values; each individual
    # field subsequently faces the same >=90% MDA coverage gate and train-only
    # imputation as every other candidate input.  Do not encode this source
    # status as a model feature.
    source_identity_gaps = int(joined["__exact170_source_matched"].eq("left_only").sum())
    all_feature_missing = int(joined[fields].isna().all(axis=1).sum())
    output = joined.loc[:, list(IDENTITY)].copy()
    output["exact170_context_source_utc"] = output["__ts__"]
    output["exact170_context_available_utc"] = output["__ts__"]
    values = pd.DataFrame({
        f"meta_context__{field}": pd.to_numeric(joined[field], errors="coerce").astype("float32")
        for field in fields
    })
    output = pd.concat([output.reset_index(drop=True), values.reset_index(drop=True)], axis=1)
    if output["candidate_id"].duplicated().any():
        raise AssertionError("exact170 sidecar identity is not unique")
    coverage = pd.DataFrame({
        "feature": fields,
        "coverage": [float(output[f"meta_context__{field}"].notna().mean()) for field in fields],
        "n_unique": [int(output[f"meta_context__{field}"].nunique(dropna=True)) for field in fields],
    })
    args.out_dir.mkdir(parents=True)
    output.to_parquet(args.out_dir / "candidate_meta_context.parquet", index=False, compression="zstd")
    coverage.to_parquet(args.out_dir / "feature_coverage.parquet", index=False)
    (args.out_dir / "manifest.json").write_text(json.dumps({
        "schema": "strict_r3_exact170_candidate_meta_context_v1",
        "candidate_surfaces": [str(path) for path in args.candidate_surface],
        "candidate_surface_sha256": {str(path): _sha(path) for path in args.candidate_surface},
        "exact170_panel": str(args.exact170_panel),
        "exact170_panel_sha256": _sha(args.exact170_panel),
        "exact170_manifest": str(args.exact170_manifest),
        "exact170_manifest_sha256": _sha(args.exact170_manifest),
        "rows": len(output), "meta_owned_fields": fields,
        "field_count": len(fields),
        "source_identity_gaps": source_identity_gaps,
        "all_feature_missing_rows": all_feature_missing,
        "missingness_policy": "preserve causal source warm-up gaps; do not filter candidates; each field faces ordinary MDA coverage/variance gating and train-only imputation",
        "causality": "exact target-free timestamp/symbol join; feature source and availability timestamps equal candidate decision timestamp",
        "labels_or_outcomes_persisted": [],
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
