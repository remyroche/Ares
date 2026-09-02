#!/usr/bin/env python3
"""Aggregate immutable P8u Meta target/query screen shards.

The target/query screen is deliberately sharded only for parallel offline
compute.  This utility does not reopen Base features, policy labels, path
labels, MC1, portfolio, or exchange state.  It verifies the sealed shard
receipts, combines their already-computed strict-OOF diagnostics, and emits
one family-level ``MetaDecisionStable`` decision receipt plus exact arm-to-score
root locations for the later feature-selection and GateProxy stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


SCHEMA = "strict_r3_p8u_meta_target_query_grid_aggregate_v1"
SHARD_SCHEMA = "strict_r3_p8u_meta_target_query_grid_v1"


def _once(path: Path, payload: object) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _truthy_receipt(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    flags = [value for value in payload.values() if isinstance(value, bool)]
    if not flags or not all(flags):
        raise AssertionError(f"{path}: incomplete correctness receipt")
    return payload


def _read(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path, correctness_path = root / "run_manifest.json", root / "correctness_report.json"
    if not manifest_path.exists() or not correctness_path.exists():
        raise FileNotFoundError(f"missing immutable shard receipt in {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SHARD_SCHEMA:
        raise AssertionError(f"{root}: unexpected shard schema")
    _truthy_receipt(correctness_path)
    required = (
        "target_query_summary.parquet", "target_query_fold_metrics.parquet",
        "weekly_sstable_meta.parquet", "base_band_conversion_metrics.parquet",
    )
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{root}: incomplete target/query shard {missing}")
    summary = pd.read_parquet(root / required[0]); fold = pd.read_parquet(root / required[1])
    weekly = pd.read_parquet(root / required[2]); bands = pd.read_parquet(root / required[3])
    if summary.empty or fold.empty or weekly.empty:
        raise AssertionError(f"{root}: completed shard has an empty diagnostic table")
    arms = {str(value["name"]) for value in manifest.get("arms", ())}
    if set(summary.arm.astype(str)) != arms or not set(fold.arm.astype(str)).issubset(arms):
        raise AssertionError(f"{root}: summary/manifest arm mismatch")
    for item in (summary, fold, weekly, bands):
        item["score_root"] = str(root)
        item["score_root_name"] = root.name
    return summary, fold, weekly, bands, manifest


def run(*, roots: Sequence[Path], out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    if len(roots) < 2:
        raise ValueError("need at least two completed shards")
    summaries: list[pd.DataFrame] = []; folds: list[pd.DataFrame] = []
    weeks: list[pd.DataFrame] = []; bands: list[pd.DataFrame] = []
    manifests: dict[str, dict[str, Any]] = {}
    for root in roots:
        summary, fold, weekly, band, manifest = _read(root)
        summaries.append(summary); folds.append(fold); weeks.append(weekly); bands.append(band)
        manifests[str(root)] = {
            "run_manifest_sha256": _sha(root / "run_manifest.json"),
            "correctness_report_sha256": _sha(root / "correctness_report.json"),
            "arms": [str(item["name"]) for item in manifest["arms"]],
        }
    summary = pd.concat(summaries, ignore_index=True)
    fold = pd.concat(folds, ignore_index=True)
    weekly = pd.concat(weeks, ignore_index=True)
    band = pd.concat(bands, ignore_index=True) if any(not item.empty for item in bands) else pd.DataFrame()
    if summary.arm.duplicated().any() or fold.duplicated(["arm", "held_month"]).any():
        raise AssertionError("arm appears in more than one shard or fold receipt")
    # The frozen source specification makes head-specific MetaDecisionStable
    # the family selector.  Its score is ``sstable_meta``; CMI and Top-2
    # substitution are explicit deterministic tie-breakers, not a new hand
    # weighted objective.  GateProxy later screens only the costly candidates.
    summary = summary.sort_values(
        ["family", "sstable_meta", "conditional_mi_meta_policy_given_base", "mean_top2_substitution_ev_bps", "arm"],
        ascending=[True, False, False, False, True], kind="stable",
    ).reset_index(drop=True)
    summary["family_rank"] = summary.groupby("family", sort=False).cumcount().add(1)
    summary["meta_decisionstable_selected"] = summary.family_rank.eq(1)
    summary["secondary_diagnostic_qualified"] = (
        pd.to_numeric(summary.conditional_mi_meta_policy_given_base, errors="coerce").gt(0.0)
        & pd.to_numeric(summary.mean_iccond, errors="coerce").gt(0.0)
    )
    winners = summary.loc[summary.meta_decisionstable_selected].copy()
    if set(winners.family.astype(str)) != {"magnitude", "under", "over", "state"}:
        raise AssertionError("expected exactly one MetaDecisionStable winner for every target family")
    locations = summary.loc[:, ["arm", "family", "scale", "query", "score_root", "score_root_name"]].copy()
    out.mkdir(parents=True)
    summary.to_parquet(out / "target_query_summary.parquet", index=False, compression="zstd")
    winners.to_parquet(out / "family_winners_pre_feature_selection.parquet", index=False, compression="zstd")
    locations.to_parquet(out / "arm_score_locations.parquet", index=False, compression="zstd")
    fold.to_parquet(out / "target_query_fold_metrics.parquet", index=False, compression="zstd")
    weekly.to_parquet(out / "weekly_sstable_meta.parquet", index=False, compression="zstd")
    band.to_parquet(out / "base_band_conversion_metrics.parquet", index=False, compression="zstd")
    _once(out / "run_manifest.json", {
        "schema": SCHEMA,
        "scope": "offline aggregation of completed strict-OOF target/query diagnostics; no labels, MC1, admission, portfolio, live, or exchange input",
        "shards": manifests,
        "selection": {
            "primary": "head-specific MetaDecisionStable (sstable_meta)",
            "tie_breakers": ["conditional_mi_meta_policy_given_base", "mean_top2_substitution_ev_bps"],
            "secondary": "conditional MI and IC remain diagnostics; GateProxy is reserved for costly feature/model/HPO proposals",
        },
        "arms": int(len(summary)),
        "families": sorted(summary.family.astype(str).unique().tolist()),
    })
    _once(out / "correctness_report.json", {
        "all_input_shards_have_complete_correctness_receipts": True,
        "every_arm_has_one_completed_strict_oof_summary_and_fold_receipt": True,
        "arm_to_target_free_score_root_is_preserved_exactly": True,
        "family_winners_use_only_precomputed_metadecisionstable_diagnostics": True,
        "no_policy_path_mc1_admission_portfolio_live_or_exchange_input_opened": True,
        "gateproxy_and_mc1_remain_later_separate_authorities": True,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", required=True, help="completed target/query shard; repeat")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(roots=tuple(path.resolve() for path in args.root), out=args.out.resolve()))


if __name__ == "__main__":
    main()
