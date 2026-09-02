#!/usr/bin/env python3
"""Prepare score-complete target-free August inputs for the MC1 input study.

The archived live extension contains the complete BCF surface and a current-v5
surface whose incomplete-base rows deliberately carry null downstream scores.
This producer persists both the full target-free availability receipt and the
strict score-complete intersection used by the mapper.  It never opens policy
labels or execution outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data_perp/artifacts/strict_r3_live_contract_dual30_august_source_20260819_v1"
CORE = (
    "final_score", "base_rank42", "conditional_consensus_rank", "upstream",
    "ordinary_shadow_consensus_rank", "correctness_rank",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path, *, family: str) -> pd.DataFrame:
    columns = ["candidate_id", "__decision_ts__", "side_name", *CORE]
    if family == "current_v5":
        columns.insert(2, "__symbol__")
    frame = pd.read_parquet(path, columns=columns)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    if frame["candidate_id"].duplicated().any():
        raise AssertionError(f"{family} source duplicates candidate identity")
    if not frame["side_name"].astype(str).str.lower().eq("long").all():
        raise AssertionError(f"{family} source is not long-only")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.out_dir.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    bcf_path, current_path = source / "bcf_scores.parquet", source / "current_scores.parquet"
    bcf, current = _read(bcf_path, family="bcf"), _read(current_path, family="current_v5")
    bcf = bcf.merge(
        current.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]],
        on="candidate_id", how="inner", suffixes=("_bcf", "_current"), validate="one_to_one",
    )
    for field in ("__decision_ts__", "side_name"):
        if not bcf[f"{field}_bcf"].astype(str).equals(bcf[f"{field}_current"].astype(str)):
            raise AssertionError(f"BCF/current identity disagrees on {field}")
    bcf = bcf.rename(columns={
        "__decision_ts___bcf": "__decision_ts__", "side_name_bcf": "side_name",
    }).drop(columns=["__decision_ts___current", "side_name_current"])
    # ``current`` is the routed surface.  BCF may contain additional rows, but
    # they are not eligible for a dual-map admission and must not become a
    # hidden substitute candidate universe.
    current = current.loc[current["candidate_id"].isin(bcf["candidate_id"])].copy()
    bcf = bcf.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name", *CORE]].copy()
    current = current.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name", *CORE]].copy()
    current = current.sort_values(["__decision_ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    bcf = bcf.set_index("candidate_id").loc[current["candidate_id"]].reset_index()
    complete_current = np.isfinite(current.loc[:, list(CORE)].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1)
    complete_bcf = np.isfinite(bcf.loc[:, list(CORE)].apply(pd.to_numeric, errors="coerce").to_numpy(float)).all(axis=1)
    complete = complete_current & complete_bcf
    availability = current.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]].copy()
    availability["current_mc1_core_complete"] = complete_current
    availability["bcf_mc1_core_complete"] = complete_bcf
    availability["score_complete_dual_map"] = complete
    availability["mapper_unavailable_reason"] = np.where(
        complete, "", np.where(~complete_current, "current_base_or_consensus_incomplete", "bcf_core_incomplete"),
    )
    selected_current = current.loc[complete].reset_index(drop=True)
    selected_bcf = bcf.loc[complete].reset_index(drop=True)
    candidates = selected_current.loc[:, ["candidate_id", "__decision_ts__", "__symbol__", "side_name"]].copy()
    candidates["__ts__"] = candidates["__decision_ts__"] - pd.Timedelta(hours=1)
    # Candidate IDs encode the preceding signal hour.  The assertion catches a
    # silent change in the historical live identity convention.
    encoded = candidates["candidate_id"].str.rsplit("|", n=1).str[-1]
    expected = candidates["__ts__"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if not encoded.equals(expected):
        raise AssertionError("August candidate IDs do not encode decision_ts - 1h")
    out.mkdir(parents=True)
    availability.to_parquet(out / "target_free_availability.parquet", index=False, compression="zstd")
    candidates.to_parquet(out / "score_complete_candidates.parquet", index=False, compression="zstd")
    selected_current.to_parquet(out / "current_scores_core_complete.parquet", index=False, compression="zstd")
    selected_bcf.to_parquet(out / "bcf_scores_core_complete.parquet", index=False, compression="zstd")
    manifest = {
        "schema": "canonical_sr_e2_august_targetfree_extension_inputs_v1",
        "scope": "offline target-free August score preparation; no outcomes or exchange calls",
        "source": {
            "bcf_scores": {"path": str(bcf_path), "sha256": _sha256(bcf_path)},
            "current_scores": {"path": str(current_path), "sha256": _sha256(current_path)},
        },
        "core_features": list(CORE),
        "total_current_routed_rows": int(len(current)),
        "matched_bcf_current_rows": int(len(bcf)),
        "score_complete_rows": int(complete.sum()),
        "current_core_incomplete_rows": int((~complete_current).sum()),
        "bcf_core_incomplete_rows": int((~complete_bcf).sum()),
        "score_period": {
            "start": current["__decision_ts__"].min().isoformat(),
            "end_inclusive": current["__decision_ts__"].max().isoformat(),
        },
        "selection": "matched current route AND finite six-field BCF/current MC1 core; unavailable rows are retained in target_free_availability only",
        "status": "complete",
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
