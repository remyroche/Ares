#!/usr/bin/env python3
"""Materialise exact 90-field hourly transition geometry for frozen forward rows.

The raw rows are recovered only by an exact ``(__symbol__, __ts__)`` lookup in
the frozen feature store.  A candidate feature timestamp is its signal time;
the corresponding execution decision is exactly signal plus one hour.  This
runner deliberately does not use as-of matching, resampling, interpolation or
any outcome/action-layer input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    from scripts.materialize_historical_current_common_transition_geometry import (
        CANONICAL_FEATURES, RAW_FIELDS, build_historical_hourly_state,
    )
except ModuleNotFoundError:
    from materialize_historical_current_common_transition_geometry import CANONICAL_FEATURES, RAW_FIELDS, build_historical_hourly_state


ROOT = Path(__file__).resolve().parents[1]
FORWARD_ROOT = ROOT / "data_perp/artifacts/exact_strict_oof_hurdle_distributional_ablation_20260730_v3"
FEATURE_ROOT = ROOT / "data_perp/features/20260711_070000"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/forward_exact_transition_geometry_20260730_v1"
SCHEMA = "forward_exact_transition_geometry_v1"
WINDOWS = ("may_to_june_forward_control", "later_july_forward")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if value is pd.NaT or (not isinstance(value, (Mapping, list, tuple)) and pd.isna(value)):
        return None
    if isinstance(value, (Path, pd.Timestamp)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_safe(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _bound_forward(root: Path) -> tuple[Path, Path]:
    path, manifest, seal = root / "forward_predictions.parquet", root / "manifest.json", root / "manifest.sha256"
    if not all(item.is_file() for item in (path, manifest, seal)):
        raise FileNotFoundError("frozen hurdle forward artifact is incomplete")
    if seal.read_text(encoding="utf-8").split()[0] != sha256(manifest):
        raise ValueError("frozen hurdle manifest seal fails")
    return path, manifest


def feature_path(feature_root: Path, symbol: str) -> Path:
    return feature_root / f"symbol={symbol.replace('/', '_')}.parquet"


def recover_exact_raw_rows(forward: pd.DataFrame, feature_root: Path) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Recover only requested forward candidate rows by exact timestamp key."""

    needed = {"candidate_id", "__ts__", "__symbol__", "side_name", "execution_decision_utc", "window"}
    if missing := sorted(needed.difference(forward.columns)):
        raise ValueError(f"forward source lacks {missing}")
    work = forward.loc[forward.window.isin(WINDOWS), list(needed)].copy()
    if work.candidate_id.duplicated().any():
        raise ValueError("forward candidate id is duplicate")
    for column in ("__ts__", "execution_decision_utc"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    if not work.execution_decision_utc.eq(work.__ts__ + pd.Timedelta(hours=1)).all():
        raise ValueError("execution decision is not signal plus one hour")
    rows: list[pd.DataFrame] = []
    source_hashes: list[dict[str, Any]] = []
    for symbol, wanted in work.groupby("__symbol__", sort=True, observed=True):
        path = feature_path(feature_root, str(symbol))
        if not path.is_file():
            raise FileNotFoundError(f"frozen feature shard is absent for {symbol}: {path}")
        source_hashes.append({"symbol": str(symbol), "path": str(path), "sha256": sha256(path)})
        raw = pd.read_parquet(path, columns=[*RAW_FIELDS, "__symbol__"])
        raw = raw.reset_index().rename(columns={raw.index.name or "index": "__ts__"})
        raw["__ts__"] = pd.to_datetime(raw["__ts__"], utc=True, errors="raise")
        raw["__symbol__"] = raw["__symbol__"].astype(str)
        if raw.duplicated(["__ts__", "__symbol__"]).any():
            raise ValueError(f"frozen feature shard has duplicate timestamp/symbol: {path}")
        local = wanted.merge(raw, on=["__ts__", "__symbol__"], how="left", validate="many_to_one", indicator=True)
        if not local["_merge"].eq("both").all():
            absent = local.loc[local["_merge"].ne("both"), "__ts__"].min()
            raise ValueError(f"exact raw feature timestamp absent for {symbol}: {absent}")
        rows.append(local.drop(columns="_merge"))
    result = pd.concat(rows, ignore_index=True).sort_values("candidate_id", kind="stable").reset_index(drop=True)
    if len(result) != len(work) or result.candidate_id.nunique() != len(work):
        raise ValueError("exact raw recovery changed forward candidate identity")
    for field in RAW_FIELDS:
        result[field] = pd.to_numeric(result[field], errors="coerce")
    return result, source_hashes


def run(*, forward_root: Path, feature_root: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output_dir}")
    forward_path, forward_manifest = _bound_forward(forward_root)
    raw, source_hashes = recover_exact_raw_rows(pd.read_parquet(forward_path), feature_root)
    hourly = build_historical_hourly_state(raw)
    # This matches the frozen common-geometry availability rule: a timestamp
    # is usable when the exact state exists.  A missing side or an exact
    # 1/3/12h predecessor leaves only the affected canonical fields null;
    # median-imputation is part of the retained logistic recipe.  Requiring
    # all 90 fields here would silently discard legitimate sparse states and
    # change the candidate universe.
    hourly["common_transition_context_available"] = hourly.loc[:, list(CANONICAL_FEATURES)].notna().any(axis=1)
    if not hourly.common_transition_context_available.all():
        raise ValueError("exact raw aggregation has no usable canonical state at a forward timestamp")
    checked = raw.loc[:, ["candidate_id", "__ts__", "window", "execution_decision_utc"]].merge(hourly.loc[:, ["signal_context_utc", "common_transition_context_available"]], left_on="__ts__", right_on="signal_context_utc", how="left", validate="many_to_one")
    checked["context_joined"] = checked["common_transition_context_available"].astype("boolean").fillna(False).astype(bool)
    coverage = checked.groupby("window", sort=True).agg(candidate_rows=("candidate_id", "size"), joined_rows=("context_joined", "sum"), first_signal_utc=("__ts__", "min"), last_signal_utc=("__ts__", "max")).reset_index()
    coverage["coverage"] = coverage.joined_rows / coverage.candidate_rows
    coverage["full_window_coverage"] = coverage.joined_rows.eq(coverage.candidate_rows)
    if set(coverage.window) != set(WINDOWS) or not coverage.full_window_coverage.all():
        raise ValueError("exact forward geometry does not cover both frozen windows")
    stage = Path(tempfile.mkdtemp(dir=output_dir.parent, prefix=f".{output_dir.name}."))
    try:
        raw.to_parquet(stage / "candidate_raw_rows.parquet", index=False, compression="zstd")
        hourly.to_parquet(stage / "hourly_geometry.parquet", index=False, compression="zstd")
        coverage.to_csv(stage / "coverage.csv", index=False)
        manifest = {
            "schema": SCHEMA, "status": "MATERIALIZED_EXACT_FORWARD_90_FIELD_GEOMETRY", "promotion_eligible": False,
            "feature_columns": list(CANONICAL_FEATURES), "raw_fields": list(RAW_FIELDS),
            "contracts": {"raw_recovery": "exact feature-store lookup on (__symbol__, signal __ts__) for every frozen forward candidate; no as-of/resample/interpolation/fill", "geometry": "identical nine raw concepts, per-side median/IQR, long-short gap, and exact 1/3/12h state-mean deltas used by the strict common geometry constructor", "availability": "signal-time features are available no later than execution decision at signal+1h", "prohibited": "targets, outcomes, DAE/GMM, raw model scores, action-layer fields and side quotas are excluded"},
            "sources": {"forward_predictions": {"path": str(forward_path), "sha256": sha256(forward_path), "manifest_sha256": sha256(forward_manifest)}, "feature_store": {"root": str(feature_root), "used_shards": source_hashes}},
            "coverage": coverage.to_dict("records"), "outputs_sha256": {name: sha256(stage / name) for name in ("candidate_raw_rows.parquet", "hourly_geometry.parquet", "coverage.csv")}, "runner": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        }
        _write_json(stage / "manifest.json", manifest)
        (stage / "manifest.sha256").write_text(sha256(stage / "manifest.json") + "\n", encoding="utf-8")
        os.replace(stage, output_dir)
    except Exception:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-root", type=Path, default=FORWARD_ROOT)
    parser.add_argument("--feature-root", type=Path, default=FEATURE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(_safe(run(forward_root=args.forward_root, feature_root=args.feature_root, output_dir=args.output_dir)), sort_keys=True))
