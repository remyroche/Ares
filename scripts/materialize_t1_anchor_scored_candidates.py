#!/usr/bin/env python3
"""Create anchor/meta-scored T1 candidate rows from a concrete OOS ledger.

This is a narrow bridge from an already-matured candidate/outcome ledger to the
T1 rank-contract replay scripts.  It preserves the candidate universe and
execution outcomes, but replaces native reliability-blend scores/ranks with
live final-fit anchor/meta scores and frozen policy-rank-reference lookups.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ACTIVE_HEADS = ("short_asset", "short_boll")
JOIN_KEYS = ("timestamp", "strategy_id", "symbol")
SCORE_COLUMNS = (
    "calibrated_score",
    "policy_rank_pct",
    "auction_rank_pct",
    "base_pred",
    "meta_pred",
    "raw_prediction_score",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    if "timestamp" not in frame.columns:
        raise RuntimeError(f"{path} missing timestamp")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    for col in ("strategy_id", "symbol"):
        if col not in frame.columns:
            raise RuntimeError(f"{path} missing {col}")
        frame[col] = frame[col].astype(str)
    return frame


def _deployable_candidates(frame: pd.DataFrame) -> pd.DataFrame:
    threshold_col = (
        "deployment_rank_threshold"
        if "deployment_rank_threshold" in frame.columns
        else "base_strategy_threshold"
    )
    rank = pd.to_numeric(frame.get("policy_rank_pct"), errors="coerce")
    threshold = pd.to_numeric(frame.get(threshold_col), errors="coerce").fillna(np.inf)
    out = frame.loc[(rank >= threshold).fillna(False)].copy()
    out["active_rank_column"] = "policy_rank_pct"
    out["active_threshold_column"] = threshold_col
    return out.reset_index(drop=True)


def _anchor_col(frame: pd.DataFrame, name: str) -> str:
    suffixed = f"{name}__anchor"
    if suffixed in frame.columns:
        return suffixed
    if name in frame.columns:
        return name
    raise KeyError(suffixed)


def materialize(
    *,
    candidates_path: Path,
    score_ledger_path: Path,
    output_dir: Path,
    active_heads: tuple[str, ...] = ACTIVE_HEADS,
) -> dict[str, Any]:
    candidates = _read_frame(candidates_path)
    scores = _read_frame(score_ledger_path)
    if "head" not in candidates.columns:
        raise RuntimeError("candidate ledger missing head")
    candidates["head"] = candidates["head"].astype(str)
    candidates = candidates.loc[candidates["head"].isin(active_heads)].copy()
    missing_score_cols = [col for col in SCORE_COLUMNS if col not in scores.columns]
    if missing_score_cols:
        raise RuntimeError(f"score ledger missing columns: {missing_score_cols}")
    if candidates.empty:
        output_dir.mkdir(parents=True, exist_ok=True)
        policy_dir = output_dir / "simple_policy_optimiser"
        policy_dir.mkdir(parents=True, exist_ok=True)
        for col in ("calibrated_score", "policy_rank_pct", "normalized_rank_score", "base_pred", "meta_pred"):
            if col not in candidates.columns:
                candidates[col] = pd.Series(dtype="float64")
        broad_path = policy_dir / "simple_policy_candidates_broad.parquet"
        deployable_path = policy_dir / "simple_policy_candidates_deployable.parquet"
        simple_path = policy_dir / "simple_policy_candidates.parquet"
        manifest_path = output_dir / "t1_anchor_scored_candidate_manifest.json"
        broad = candidates.reset_index(drop=True)
        deployable = broad.iloc[0:0].copy()
        broad.to_parquet(broad_path, index=False)
        deployable.to_parquet(deployable_path, index=False)
        deployable.to_parquet(simple_path, index=False)
        manifest = {
            "generated_by": "materialize_t1_anchor_scored_candidates",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "candidates_path": str(candidates_path),
            "candidates_sha256": _sha256(candidates_path),
            "score_ledger_path": str(score_ledger_path),
            "score_ledger_sha256": _sha256(score_ledger_path),
            "active_heads": list(active_heads),
            "rows": 0,
            "deployable_rows": 0,
            "timestamp_min": None,
            "timestamp_max": None,
            "timestamp_count": 0,
            "heads": [],
            "score_contract": {
                "score_column": "calibrated_score",
                "score_source": "live_finalfit_anchor_meta_score",
                "rank_column": "policy_rank_pct",
                "rank_source": "policy_rank_reference_lookup_from_live_finalfit_anchor_score",
                "native_reliability_blend_active": False,
                "qfail_active": False,
                "market_state_threshold_controller_active": False,
            },
            "empty_candidate_ledger": True,
            "outputs": {
                "candidates_broad": str(broad_path),
                "candidates_deployable": str(deployable_path),
                "simple_policy_candidates": str(simple_path),
                "manifest": str(manifest_path),
            },
        }
        manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
        return manifest
    for frame, label in ((candidates, "candidate"), (scores, "score")):
        dup = frame.duplicated(list(JOIN_KEYS), keep=False)
        if bool(dup.any()):
            sample = frame.loc[dup, list(JOIN_KEYS)].head(10).to_dict("records")
            raise RuntimeError(f"{label} ledger has duplicate join keys: {sample}")

    score_keep = list(JOIN_KEYS) + list(SCORE_COLUMNS)
    merged = candidates.merge(
        scores.loc[:, score_keep],
        on=list(JOIN_KEYS),
        how="left",
        suffixes=("", "__anchor"),
        validate="one_to_one",
    )
    anchor_score_col = _anchor_col(merged, "calibrated_score")
    missing = merged[anchor_score_col].isna()
    if bool(missing.any()):
        sample = merged.loc[missing, list(JOIN_KEYS) + ["head"]].head(20).to_dict("records")
        raise RuntimeError(f"Missing anchor scores for {int(missing.sum())} rows: {sample}")

    for col in ("calibrated_score", "reliability_blend_score", "policy_rank_pct"):
        if col in merged.columns:
            merged[f"source_{col}"] = pd.to_numeric(merged[col], errors="coerce")
    merged["anchor_score"] = pd.to_numeric(merged[anchor_score_col], errors="coerce")
    merged["reliability_anchor_only_score"] = merged["anchor_score"]
    merged["calibrated_score"] = merged["anchor_score"]
    merged["raw_prediction_score"] = pd.to_numeric(
        merged[_anchor_col(merged, "raw_prediction_score")],
        errors="coerce",
    )
    merged["base_pred"] = pd.to_numeric(merged[_anchor_col(merged, "base_pred")], errors="coerce")
    merged["meta_pred"] = pd.to_numeric(merged[_anchor_col(merged, "meta_pred")], errors="coerce")
    merged["anchor_policy_rank_pct"] = pd.to_numeric(
        merged[_anchor_col(merged, "policy_rank_pct")],
        errors="coerce",
    )
    merged["anchor_auction_rank_pct"] = pd.to_numeric(
        merged[_anchor_col(merged, "auction_rank_pct")],
        errors="coerce",
    )
    for col in ("policy_rank_pct", "strategy_rank_pct", "normalized_rank_score", "rank_pct"):
        merged[col] = merged["anchor_policy_rank_pct"]
    merged["score_source"] = "live_finalfit_anchor_meta_score"
    merged["threshold_rank_score_source"] = "policy_rank_pct"
    merged["rank_contract_source"] = "anchor_policy_rank_reference_lookup"
    drop_cols = [c for c in merged.columns if c.endswith("__anchor")]
    merged = merged.drop(columns=drop_cols)
    finite_required = [
        "calibrated_score",
        "policy_rank_pct",
        "normalized_rank_score",
        "base_pred",
        "meta_pred",
    ]
    bad = ~np.isfinite(merged[finite_required].apply(pd.to_numeric, errors="coerce")).all(axis=1)
    if bool(bad.any()):
        sample = merged.loc[bad, list(JOIN_KEYS) + finite_required].head(20).to_dict("records")
        raise RuntimeError(f"Non-finite anchor-scored rows: {sample}")

    output_dir.mkdir(parents=True, exist_ok=True)
    policy_dir = output_dir / "simple_policy_optimiser"
    policy_dir.mkdir(parents=True, exist_ok=True)
    broad_path = policy_dir / "simple_policy_candidates_broad.parquet"
    deployable_path = policy_dir / "simple_policy_candidates_deployable.parquet"
    simple_path = policy_dir / "simple_policy_candidates.parquet"
    manifest_path = output_dir / "t1_anchor_scored_candidate_manifest.json"
    broad = merged.sort_values(["timestamp", "strategy_id", "symbol"], kind="mergesort").reset_index(drop=True)
    deployable = _deployable_candidates(broad)
    broad.to_parquet(broad_path, index=False)
    deployable.to_parquet(deployable_path, index=False)
    deployable.to_parquet(simple_path, index=False)
    ts = pd.to_datetime(broad["timestamp"], utc=True, errors="coerce")
    manifest = {
        "generated_by": "materialize_t1_anchor_scored_candidates",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates_path": str(candidates_path),
        "candidates_sha256": _sha256(candidates_path),
        "score_ledger_path": str(score_ledger_path),
        "score_ledger_sha256": _sha256(score_ledger_path),
        "active_heads": list(active_heads),
        "rows": int(len(broad)),
        "deployable_rows": int(len(deployable)),
        "timestamp_min": ts.min(),
        "timestamp_max": ts.max(),
        "timestamp_count": int(ts.nunique()),
        "heads": sorted(broad["head"].dropna().astype(str).unique().tolist()),
        "score_contract": {
            "score_column": "calibrated_score",
            "score_source": "live_finalfit_anchor_meta_score",
            "rank_column": "policy_rank_pct",
            "rank_source": "policy_rank_reference_lookup_from_live_finalfit_anchor_score",
            "native_reliability_blend_active": False,
            "qfail_active": False,
            "market_state_threshold_controller_active": False,
        },
        "outputs": {
            "candidates_broad": str(broad_path),
            "candidates_deployable": str(deployable_path),
            "simple_policy_candidates": str(simple_path),
            "manifest": str(manifest_path),
        },
    }
    manifest_path.write_text(json.dumps(_json_safe(manifest), indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--score-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--active-head", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    active_heads = tuple(args.active_head or ACTIVE_HEADS)
    manifest = materialize(
        candidates_path=args.candidates,
        score_ledger_path=args.score_ledger,
        output_dir=args.output_dir,
        active_heads=active_heads,
    )
    print(json.dumps(_json_safe(manifest), indent=2)[:6000])


if __name__ == "__main__":
    main()
