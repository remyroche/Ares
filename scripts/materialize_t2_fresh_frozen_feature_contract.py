#!/usr/bin/env python3
"""Materialise the frozen T2 361-field contract for the untouched 2025 rows.

The 351 ordinary fields come from the historical causal feature store at each
signal-close timestamp.  The remaining ten latent fields are reconstructed
from the explicitly approved, pre-existing per-side AE/GMM states.  Those
states are row-independent, so this utility derives their temporal summary
fields over each complete per-symbol time series before selecting candidates.
No target, path, realised cost, score, or selection field is an input.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.features_gmm_ae import (  # noqa: E402
    load_ae_gmm_state_artifact,
    transform_ae_gmm_features,
)


DEFAULT_POPULATION = ROOT / (
    "data_perp/artifacts/febapr2025_canonical_exact_policy_base_population_20260727_v1/"
    "population.parquet"
)
DEFAULT_FEATURE_STORE = ROOT / "data_perp/features/20260711_070000"
DEFAULT_CONTRACT = ROOT / (
    "data_perp/artifacts/controlled_target_supportive_prepared_ledger_20260801_v5/"
    "frozen_raw_causal_features.json"
)
DEFAULT_LONG_STATE = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1/long/ae_gmm/ae_gmm_state.pkl"
DEFAULT_SHORT_STATE = ROOT / "data_perp/artifacts/packb_side_local_ae_20260724_v1/short/ae_gmm/ae_gmm_state.pkl"
LATENT_FIELDS = (
    "AE_reconstruction_error",
    "mahalanobis_distance",
    "cluster_acceleration",
    "cluster_entropy",
    "cluster_entropy_accel_1",
    "cluster_entropy_delta_1",
    "cluster_entropy_norm",
    "cluster_flip_count_20",
    "cluster_speed",
    "cluster_t",
)


def _temporal_latent_fields(point: pd.DataFrame) -> pd.DataFrame:
    """Recreate the ordinary causal sequence summaries from row-wise outputs."""

    out = point.copy()
    probs = [f"gmm_prob_{idx}" for idx in range(12) if f"gmm_prob_{idx}" in out]
    ts = pd.DatetimeIndex(out.index)
    contiguous = np.r_[False, np.diff(ts.asi8) == pd.Timedelta(hours=1).value]
    entropy = out["cluster_entropy"].to_numpy(dtype=np.float32, copy=False)
    labels = out["cluster_t"].to_numpy(dtype=np.float32, copy=False)
    delta = np.zeros(len(out), dtype=np.float32)
    delta[1:] = entropy[1:] - entropy[:-1]
    delta[~contiguous] = 0.0
    accel = np.zeros(len(out), dtype=np.float32)
    accel[1:] = delta[1:] - delta[:-1]
    accel[~contiguous] = 0.0
    if probs:
        probability = out.loc[:, probs].to_numpy(dtype=np.float32, copy=False)
        diff = np.zeros_like(probability)
        diff[1:] = probability[1:] - probability[:-1]
        speed = np.sqrt(np.sum(diff * diff, axis=1)).astype(np.float32)
    else:
        speed = np.zeros(len(out), dtype=np.float32)
    speed[~contiguous] = 0.0
    speed_accel = np.zeros(len(out), dtype=np.float32)
    speed_accel[1:] = speed[1:] - speed[:-1]
    speed_accel[~contiguous] = 0.0
    changed = np.zeros(len(out), dtype=np.int32)
    changed[1:] = (labels[1:] != labels[:-1]) & contiguous[1:]
    starts = np.maximum(0, np.arange(len(out)) - 19)
    csum = np.r_[0, np.cumsum(changed, dtype=np.int64)]
    flips = (csum[np.arange(len(out)) + 1] - csum[starts + 1]).astype(np.float32)
    out["cluster_entropy_delta_1"] = delta
    out["cluster_entropy_accel_1"] = accel
    out["cluster_speed"] = speed
    out["cluster_acceleration"] = speed_accel
    out["cluster_flip_count_20"] = flips
    return out


def _static_path(store: Path, symbol: str) -> Path:
    return store / f"symbol={symbol.replace('/', '_')}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, default=DEFAULT_POPULATION)
    parser.add_argument("--feature-store", type=Path, default=DEFAULT_FEATURE_STORE)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--long-state", type=Path, default=DEFAULT_LONG_STATE)
    parser.add_argument("--short-state", type=Path, default=DEFAULT_SHORT_STATE)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--history-hours", type=int, default=24)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    fields = [str(x) for x in contract["raw_feature_columns"]]
    static_fields = [field for field in fields if field not in LATENT_FIELDS]
    if len(static_fields) != 351:
        raise ValueError(f"expected 351 static fields, got {len(static_fields)}")
    states = {
        "long": load_ae_gmm_state_artifact(args.long_state),
        "short": load_ae_gmm_state_artifact(args.short_state),
    }
    for side, state in states.items():
        if str(state.get("temporal_feature_contract")) != "row_independent_v1":
            raise ValueError(f"{side} state is not a row-independent frozen state")
    source_fields = set(static_fields)
    for state in states.values():
        source_fields.update(
            str(column) for column in state.get("feature_columns", []) if str(column) != "side"
        )
    candidates = pd.read_parquet(
        args.population, columns=["candidate_id", "__ts__", "__symbol__", "side_name"]
    )
    candidates["__ts__"] = pd.to_datetime(candidates["__ts__"], utc=True)
    start = candidates["__ts__"].min() - pd.Timedelta(hours=int(args.history_hours))
    end = candidates["__ts__"].max()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = args.out_dir / "parts"
    parts_dir.mkdir(exist_ok=True)
    reports: list[dict[str, object]] = []
    for symbol, group in candidates.groupby("__symbol__", sort=True, observed=True):
        part_path = parts_dir / f"symbol={str(symbol).replace('/', '_')}.parquet"
        if part_path.exists():
            try:
                existing = pd.read_parquet(part_path, columns=["candidate_id"])
                if len(existing) == len(group):
                    reports.append({"symbol": str(symbol), "candidate_rows": int(len(group)), "status": "reused"})
                    continue
            except Exception:
                # An interrupted parquet write is not a valid materialized part.
                part_path.unlink()
        path = _static_path(args.feature_store, str(symbol))
        if not path.exists():
            raise FileNotFoundError(f"missing static store for {symbol}: {path}")
        source = pd.read_parquet(
            path,
            columns=sorted(source_fields),
            filters=[("ts", ">=", start), ("ts", "<=", end)],
        )
        source.index = pd.to_datetime(source.index, utc=True)
        source = source.sort_index()
        if not source.index.is_unique:
            raise ValueError(f"non-unique static timestamps for {symbol}")
        output = group.loc[:, ["candidate_id", "__ts__", "__symbol__", "side_name"]].copy()
        for side, side_group in group.groupby("side_name", sort=False, observed=True):
            if side not in states:
                raise ValueError(f"unexpected side {side!r}")
            state = states[str(side)]
            inputs = [str(col) for col in state.get("feature_columns", [])]
            x = source.reindex(columns=[col for col in inputs if col != "side"]).copy()
            if "side" in inputs:
                x["side"] = np.float32(1.0 if side == "long" else -1.0)
            x = x.reindex(columns=inputs).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            transformed = transform_ae_gmm_features(x, state, index=x.index)
            point = pd.DataFrame(index=x.index)
            point["AE_reconstruction_error"] = transformed["AE_reconstruction_error"]
            point["mahalanobis_distance"] = transformed["mahalanobis_distance"]
            point["cluster_entropy"] = transformed["cluster_entropy"]
            point["cluster_entropy_norm"] = transformed["cluster_entropy_norm"]
            point["cluster_t"] = transformed["cluster_t"]
            point.loc[:, [f"gmm_prob_{idx}" for idx in range(12)]] = transformed.loc[:, [f"gmm_prob_{idx}" for idx in range(12)]]
            point = _temporal_latent_fields(point)
            desired = side_group["__ts__"].to_numpy()
            aligned_static = source.reindex(pd.DatetimeIndex(desired)).reindex(columns=static_fields)
            aligned_latent = point.reindex(pd.DatetimeIndex(desired)).reindex(columns=LATENT_FIELDS)
            selected = pd.concat([aligned_static.reset_index(drop=True), aligned_latent.reset_index(drop=True)], axis=1)
            selected.index = side_group.index
            output.loc[side_group.index, fields] = selected.loc[side_group.index, fields]
        output = output.reindex(columns=["candidate_id", "__ts__", "__symbol__", "side_name", *fields])
        output.to_parquet(part_path, index=False)
        finite = np.isfinite(output.loc[:, fields].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32))
        reports.append({
            "symbol": str(symbol), "candidate_rows": int(len(output)),
            "all_fields_finite_rate": float(finite.all(axis=1).mean()),
            "feature_min_finite_rate": float(finite.mean(axis=0).min()),
        })
        if len(reports) % 10 == 0:
            print(json.dumps({"event": "progress", "symbols_complete": len(reports)}), flush=True)
    manifest = {
        "schema": "t2_fresh_frozen_feature_contract_v1",
        "purpose": "untouched 2025 T2 feature contract; no realised outcomes as inputs",
        "population": str(args.population.resolve()),
        "feature_store": str(args.feature_store.resolve()),
        "contract": str(args.contract.resolve()),
        "raw_feature_count": len(fields),
        "static_feature_count": len(static_fields),
        "latent_feature_count": len(LATENT_FIELDS),
        "long_state": str(args.long_state.resolve()),
        "short_state": str(args.short_state.resolve()),
        "state_use_approval": "user-approved later-row pre-existing AE/GMM states",
        "state_input_columns": {side: [str(x) for x in state.get("feature_columns", [])] for side, state in states.items()},
        "history_hours": int(args.history_hours),
        "parts_dir": str(parts_dir.resolve()),
        "rows": int(len(candidates)),
        "symbols": int(candidates["__symbol__"].nunique()),
        "per_symbol": reports,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "rows": len(candidates), "symbols": len(reports)}))


if __name__ == "__main__":
    main()
