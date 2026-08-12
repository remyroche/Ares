#!/usr/bin/env python3
"""Add the strict OOF regime/transition contract to the canonical meta pool.

The canonical TP6/SL4 rows are candidate keyed but the hourly regime sidecar
is keyed by ``source_utc``.  This boundary therefore performs a backward
as-of join (sidecar timestamp <= candidate decision timestamp), preserves the
regime provenance columns, and fails closed on missing or future state data.
No labels, outcomes, or model outputs are used to construct the join.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POOL = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_pool_20260808_v1.parquet"
DEFAULT_SIDECAR = ROOT / "data_perp/artifacts/oof_causal_market_regime_systems_2023q3_2025_20260811_v1/hourly_oof_market_regimes.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_canonical_meta_pool_regime_20260811_v1.parquet"

# Stable state/transition fields are explicitly routed to the meta layer.
# Numeric geometry/continuous fields are included by their sidecar prefixes,
# while IDs and provenance remain audit-only columns.
REGIME_PREFIXES = (
    "regime_state_p__",
    "transition_state_p__",
    "continuous_regime__",
    "geometry_regime__",
)
REGIME_SCALARS = {
    "regime_state_ood_score",
    "regime_state_entropy",
    "regime_state_margin",
    "regime_state_uncertainty",
    "transition_state_ood_score",
    "transition_state_entropy",
    "transition_state_margin",
    "transition_state_uncertainty",
    "regime_top2_margin",
    "state_age_hours",
    "state_age",
    "state_switch_probability",
    "transition_stable_probability",
    "transition_onset_probability",
    "transition_active_probability",
    "transition_settling_probability",
}
AUDIT_FIELDS = {
    "source_utc",
    "regime_fold_id",
    "regime_train_end_utc",
    "regime_available_utc",
    "transition_fold_id",
    "transition_train_end_utc",
    "transition_available_utc",
}


def _is_regime_field(name: str) -> bool:
    return str(name) in REGIME_SCALARS or str(name).startswith(REGIME_PREFIXES)


def run(
    *,
    pool_path: Path = DEFAULT_POOL,
    sidecar_path: Path = DEFAULT_SIDECAR,
    output_path: Path = DEFAULT_OUT,
    max_lag_hours: int = 2,
) -> Path:
    pool_path, sidecar_path, output_path = map(Path, (pool_path, sidecar_path, output_path))
    pool = pd.read_parquet(pool_path)
    sidecar_schema = pq.ParquetFile(sidecar_path).schema.names
    candidate_regime_fields = [
        name for name in sidecar_schema
        if _is_regime_field(name) and name not in AUDIT_FIELDS
    ]
    if "source_utc" not in sidecar_schema:
        raise ValueError("regime sidecar has no source_utc key")
    pool["__ts__"] = pd.to_datetime(pool["__ts__"], utc=True, errors="raise").astype("datetime64[ns, UTC]")
    # Geometry systems can have a fold-local state count.  Their padded
    # centroid-distance coordinates are deliberately unavailable rather than
    # zero; exclude those columns from the numeric meta pool instead of
    # inventing imputation semantics.  Stable posterior coordinates and all
    # continuous context fields remain when finite across the sidecar.
    probe = pd.read_parquet(
        sidecar_path,
        columns=["source_utc", *AUDIT_FIELDS.difference({"source_utc"}), *candidate_regime_fields],
    ).copy()
    usable_regime_fields = [
        name for name in candidate_regime_fields
        if pd.api.types.is_numeric_dtype(probe[name])
        and np.isfinite(pd.to_numeric(probe[name], errors="coerce")).all()
    ]
    dropped_nonfinite_regime_fields = sorted(set(candidate_regime_fields).difference(usable_regime_fields))
    sidecar_fields = ["source_utc", *AUDIT_FIELDS.difference({"source_utc"}), *usable_regime_fields]
    sidecar = probe.loc[:, list(dict.fromkeys(sidecar_fields))].copy()
    sidecar["source_utc"] = pd.to_datetime(sidecar["source_utc"], utc=True, errors="raise").astype("datetime64[ns, UTC]")
    sidecar = sidecar.sort_values("source_utc", kind="stable").drop_duplicates("source_utc", keep="last")
    # The sidecar itself must be strictly prior to the candidate at the join.
    if not (pd.to_datetime(sidecar["regime_train_end_utc"], utc=True) < sidecar["regime_available_utc"]).all():
        raise ValueError("sidecar regime provenance is not strictly prior")
    left = pool.sort_values("__ts__", kind="stable").reset_index(drop=False).rename(columns={"index": "__pool_row"})
    right = sidecar.sort_values("source_utc", kind="stable")
    joined = pd.merge_asof(
        left,
        right,
        left_on="__ts__",
        right_on="source_utc",
        direction="backward",
        tolerance=pd.Timedelta(hours=int(max_lag_hours)),
        allow_exact_matches=True,
    )
    if joined["source_utc"].isna().any():
        missing = int(joined["source_utc"].isna().sum())
        raise ValueError(f"{missing} canonical rows have no causal regime state within the lag bound")
    if (pd.to_datetime(joined["source_utc"], utc=True) > joined["__ts__"]).any():
        raise ValueError("regime as-of join contains future state information")
    if (pd.to_datetime(joined["regime_available_utc"], utc=True) > joined["__ts__"]).any():
        raise ValueError("regime availability is after the candidate decision time")
    # Preserve the original pool order and avoid ambiguous duplicate field
    # names on reruns.
    joined = joined.sort_values("__pool_row", kind="stable")
    joined = joined.drop(columns=["__pool_row", "source_utc"])
    # Candidate pool already has no regime fields; fail closed if that changes.
    overlap = sorted(set(pool.columns).intersection(sidecar_fields).difference({"source_utc"}))
    if overlap:
        raise ValueError(f"pool already contains regime fields; refusing overwrite: {overlap[:8]}")
    numeric_regime = [
        name for name in sidecar_fields
        if name not in AUDIT_FIELDS and name in joined.columns
        and pd.api.types.is_numeric_dtype(joined[name])
    ]
    finite = np.isfinite(joined[numeric_regime].to_numpy(float)).all(axis=1)
    if not finite.all():
        raise ValueError(f"joined regime fields contain non-finite values on {int((~finite).sum())} rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_parquet(output_path, index=False, compression="zstd")
    manifest = {
        "schema": "tp6_sl4_canonical_meta_pool_with_oof_regime_v1",
        "pool": str(pool_path.resolve()),
        "sidecar": str(sidecar_path.resolve()),
        "output": str(output_path.resolve()),
        "rows": int(len(joined)),
        "base_pool_fields": int(len(pool.columns)),
        "added_regime_numeric_fields": numeric_regime,
        "added_regime_numeric_count": int(len(numeric_regime)),
        "dropped_nonfinite_regime_fields": dropped_nonfinite_regime_fields,
        "provenance_fields": [f for f in AUDIT_FIELDS if f in joined.columns and f != "source_utc"],
        "max_lag_hours": int(max_lag_hours),
        "join": "backward_asof(source_utc <= candidate __ts__)",
        "coverage": float(len(joined) / max(len(pool), 1)),
        "finite_fraction": {f: float(np.isfinite(pd.to_numeric(joined[f], errors="coerce")).mean()) for f in numeric_regime},
    }
    output_path.with_suffix(output_path.suffix + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-lag-hours", type=int, default=2)
    args = parser.parse_args()
    run(pool_path=args.pool, sidecar_path=args.sidecar, output_path=args.out, max_lag_hours=args.max_lag_hours)
