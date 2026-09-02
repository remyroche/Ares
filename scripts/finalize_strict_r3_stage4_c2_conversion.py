#!/usr/bin/env python3
"""Finalize completed Stage-4 C2 shards without re-materialising their union.

The conversion runner writes immutable held-block artifacts.  This utility
creates a separate manifest-backed dataset for downstream Stage-4 MC1 work,
avoiding a high-memory concatenation of every held panel.  The special final
coverage block is explicitly allowed to supersede the overlapping tail of the
preceding regular four-week block, based on actual decision timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data_perp/artifacts/strict_r3_stage4_c2_conversion_20260823_v3"
DEFAULT_OUT = ROOT / "data_perp/artifacts/strict_r3_stage4_c2_conversion_finalized_20260823_v1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(source: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for directory in sorted((source / "c2_conversion_bundles").glob("block=*")):
        score_path = directory / "held_target_free_scores.parquet"
        manifest_path = directory / "run_manifest.json"
        if not score_path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"incomplete immutable Stage-4 shard: {directory}")
        frame = pd.read_parquet(score_path, columns=["candidate_id", "__decision_ts__"])
        frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        if frame.empty or frame["candidate_id"].duplicated().any():
            raise ValueError(f"invalid Stage-4 shard identities: {directory}")
        manifest = json.loads(manifest_path.read_text())
        rows.append({
            "block": directory.name.removeprefix("block="),
            "path": str(score_path),
            "sha256": _sha(score_path),
            "rows_raw": int(len(frame)),
            "decision_start": frame["__decision_ts__"].min(),
            "decision_end_exclusive": frame["__decision_ts__"].max() + pd.Timedelta(hours=1),
            "conversion_cutoff": manifest["cutoff"],
            "geometry_bundle_sha256": manifest["geometry_bundle_sha256"],
            "stage3_arm": manifest["stage3_arm"],
            "geometry_refit_cadence": manifest["geometry_refit_cadence"],
        })
    if not rows:
        raise ValueError("no completed Stage-4 C2 shards found")
    return pd.DataFrame(rows).sort_values("decision_start", kind="stable").reset_index(drop=True)


def _nonoverlap(inventory: pd.DataFrame) -> pd.DataFrame:
    result = inventory.copy()
    final_mask = result["block"].str.endswith("_finalcoverage")
    if final_mask.sum() > 1:
        raise ValueError("expected at most one Stage-4 final-coverage shard")
    if final_mask.any():
        boundary = result.loc[final_mask, "decision_start"].iloc[0]
        # The final coverage vintage owns its actual July 4 onward interval.
        # Earlier block shards remain valid only strictly before this boundary.
        result.loc[~final_mask & result["decision_end_exclusive"].gt(boundary), "decision_end_exclusive"] = boundary
    result = result.loc[result["decision_start"].lt(result["decision_end_exclusive"])].copy()
    result["rows_selected"] = pd.NA
    selected_ranges: list[pd.DataFrame] = []
    for row in result.itertuples(index=False):
        frame = pd.read_parquet(row.path, columns=["candidate_id", "__decision_ts__"])
        timestamp = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
        selected = frame.loc[(timestamp >= row.decision_start) & (timestamp < row.decision_end_exclusive)].copy()
        if selected.empty or selected["candidate_id"].duplicated().any():
            raise ValueError(f"invalid selected Stage-4 range: {row.block}")
        selected_ranges.append(selected)
        result.loc[result["block"].eq(row.block), "rows_selected"] = len(selected)
    identities = pd.concat(selected_ranges, ignore_index=True)
    if identities["candidate_id"].duplicated().any():
        raise AssertionError("Stage-4 finalized dataset still has overlapping candidate identities")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    inventory = _nonoverlap(_inventory(args.source))
    if inventory["geometry_refit_cadence"].ne("never").any():
        raise AssertionError("Stage-4 finalization found a refit Geometry/K9 shard")
    if inventory["stage3_arm"].ne("consensus__c2_r50_t05").any():
        raise AssertionError("Stage-4 finalization found a non-C2 consensus shard")
    if inventory["geometry_bundle_sha256"].nunique() != 1:
        raise AssertionError("Stage-4 finalization found mixed Geometry/K9 semantics")
    args.out_dir.mkdir(parents=True)
    inventory.to_parquet(args.out_dir / "c2_target_free_shards.parquet", index=False)
    manifest = {
        "schema": "strict_r3_stage4_c2_conversion_dataset_v1",
        "source": str(args.source),
        "first_shard_sha256": str(inventory.iloc[0]["sha256"]),
        "representation": "manifest-backed immutable target-free parquet shards; do not concatenate",
        "stage3_arm": "consensus__c2_r50_t05",
        "geometry": "one frozen Oct-Dec 2024 Geometry/K9 bundle; never refit",
        "overlap_rule": "finalcoverage owns its actual decision-start onward; preceding regular shard truncated strictly before it",
        "rows_selected": int(inventory["rows_selected"].astype(int).sum()),
        "decision_start": str(inventory["decision_start"].min()),
        "decision_end_exclusive": str(inventory["decision_end_exclusive"].max()),
        "shards": inventory.to_dict(orient="records"),
        "next_required": "strict current-native and separately BCF-native Stage-4 MC1 adaptation; unchanged dual admission and shared constrained portfolio",
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"event": "stage4_c2_dataset_finalized", "rows": manifest["rows_selected"], "shards": len(inventory)}))


if __name__ == "__main__":
    main()
