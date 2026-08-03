#!/usr/bin/env python3
"""Audit Pack-B target validity, label collisions, and fixed-cost geometry.

This is a diagnostic-only materialisation.  It never changes labels or fits a
model.  Every regime in this artifact is explicitly marked descriptive rather
than an inference input; a later target repair must create its regime state
with prequential lineage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


SOURCE = Path("data_perp/artifacts/20260720_s59_h5_fullthroughjul10_candleclose_trailing_cost100bps_labels/labels")
COST_BPS = 100.0


def _read(source: Path) -> pd.DataFrame:
    cols = [
        "candidate_id", "__ts__", "__symbol__", "side_name", "__barrier_pct__",
        "__r_policy_net__", "__first_touch_target_soft__", "__first_touch_policy_soft__",
        "__first_touch_valid_path__", "__first_touch_eligible__", "__first_touch_hit__",
        "__first_touch_stop__", "__first_touch_timeout__", "__first_touch_same_bar_both__",
        "__first_touch_mfe_norm__", "__first_touch_mae_norm__",
    ]
    frames = [pd.read_parquet(path, columns=cols) for path in sorted(source.glob("train_global_*_5_*.parquet"))]
    if len(frames) != 38:
        raise ValueError(f"expected 38 label shards, found {len(frames)}")
    out = pd.concat(frames, ignore_index=True)
    out["__ts__"] = pd.to_datetime(out["__ts__"], utc=True)
    if out.candidate_id.duplicated().any():
        raise ValueError("candidate identity is not unique")
    return out


def _reason(frame: pd.DataFrame) -> pd.Series:
    valid = frame["__first_touch_valid_path__"].fillna(0).gt(.5)
    eligible = frame["__first_touch_eligible__"].fillna(0).gt(.5)
    hit = frame["__first_touch_hit__"].fillna(0).gt(.5)
    stop = frame["__first_touch_stop__"].fillna(0).gt(.5)
    timeout = frame["__first_touch_timeout__"].fillna(0).gt(.5)
    result = pd.Series("invalid_or_incomplete_path", index=frame.index, dtype="string")
    result.loc[valid & ~eligible] = "policy_geometry_ineligible"
    result.loc[eligible & timeout] = "valid_timeout"
    result.loc[eligible & stop] = "valid_lower_first"
    result.loc[eligible & hit] = "valid_upper_first"
    result.loc[eligible & ~(hit | stop | timeout)] = "valid_unresolved"
    return result


def _summarise(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(groups, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(groups, key, strict=True))
        rec.update({
            "rows": int(len(part)),
            "path_complete_rate": float(part.label_valid.mean()),
            "entry_executable_rate": float(part.entry_executable.mean()),
            "gross_bps": float(part.gross_bps.mean()),
            "net_bps": float(part.net_bps.mean()),
            "target_mean": float(part["__first_touch_target_soft__"].mean()),
            "target_zero_rate": float(part["__first_touch_target_soft__"].eq(0).mean()),
            "tp_first_rate": float(part.event_upper.mean()),
            "sl_first_rate": float(part.event_lower.mean()),
            "timeout_rate": float(part.event_timeout.mean()),
            "median_atr_bps": float(part.atr_bps.median()),
            "median_tp_bps": float(part.effective_tp_bps.median()),
            "median_sl_bps": float(part.effective_sl_bps.median()),
            "median_tp_net_margin_bps": float(part.tp_net_margin_bps.median()),
            "tp_margin_le_0_rate": float(part.tp_net_margin_bps.le(0).mean()),
            "tp_margin_le_25_rate": float(part.tp_net_margin_bps.le(25).mean()),
            "tp_margin_le_50_rate": float(part.tp_net_margin_bps.le(50).mean()),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    columns = [
        "candidate_id", "__ts__", "__symbol__", "side_name", "month", "label_valid",
        "entry_executable", "target_invalid", "ineligible_reason", "event_upper",
        "event_lower", "event_timeout", "atr_bps", "effective_tp_bps", "effective_sl_bps",
        "tp_net_margin_bps", "cost_to_atr", "cost_to_tp", "gross_bps", "net_bps",
        "__first_touch_target_soft__", "__first_touch_policy_soft__",
        "executable_margin_bps", "executable_cost_floor_bps",
        "diagnostic_noncausal_vol_regime",
    ]
    populations, geometries, collisions = [], [], []
    part_root = args.out / "target_population_validity_parts"
    part_root.mkdir()
    paths = sorted(args.source.glob("train_global_*_5_*.parquet"))
    for path in paths:
        part_path = part_root / f"{path.stem}.parquet"
        if part_path.exists():
            frame = pd.read_parquet(part_path)
            populations.append(_summarise(frame, ["side_name", "month", "ineligible_reason"]))
            geometries.append(_summarise(frame[frame.label_valid], ["side_name", "month", "diagnostic_noncausal_vol_regime"]))
            frame["target_bin"] = np.floor(frame["__first_touch_target_soft__"] * 20).clip(upper=19).astype(int)
            collisions.append(_summarise(frame, ["side_name", "month", "target_bin", "ineligible_reason"]))
            continue
        frame = pd.read_parquet(path)
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True)
        frame["month"] = frame["__ts__"].dt.strftime("%Y-%m")
        frame["label_valid"] = frame["__first_touch_valid_path__"].fillna(0).gt(.5)
        frame["entry_executable"] = frame["__first_touch_eligible__"].fillna(0).gt(.5)
        frame["event_upper"] = frame["__first_touch_hit__"].fillna(0).gt(.5)
        frame["event_lower"] = frame["__first_touch_stop__"].fillna(0).gt(.5)
        frame["event_timeout"] = frame["__first_touch_timeout__"].fillna(0).gt(.5)
        frame["ineligible_reason"] = _reason(frame); frame["target_invalid"] = ~frame.label_valid
        frame["atr_bps"] = pd.to_numeric(frame["__barrier_pct__"], errors="coerce") * 1e4
        # These are the actual materialised contract distances.  The prior audit
        # incorrectly inferred 3/2 ATR from the research proposal.  This Pack-B
        # label population instead uses its archetype-specific trailing contract.
        frame["effective_tp_bps"] = pd.to_numeric(frame["__first_touch_effective_tp_abs__"], errors="coerce") * 1e4
        frame["effective_sl_bps"] = pd.to_numeric(frame["__first_touch_effective_sl_abs__"], errors="coerce") * 1e4
        frame["tp_net_margin_bps"] = frame.effective_tp_bps - pd.to_numeric(frame["__first_touch_round_trip_cost__"], errors="coerce") * 1e4
        frame["cost_to_atr"] = pd.to_numeric(frame["__first_touch_round_trip_cost__"], errors="coerce") / frame["__barrier_pct__"].replace(0, np.nan)
        frame["cost_to_tp"] = pd.to_numeric(frame["__first_touch_round_trip_cost__"], errors="coerce") / pd.to_numeric(frame["__first_touch_effective_tp_abs__"], errors="coerce").replace(0, np.nan)
        frame["net_bps"] = pd.to_numeric(frame["__first_touch_capture_net__"], errors="coerce") * 1e4
        frame["gross_bps"] = frame.net_bps + pd.to_numeric(frame["__first_touch_round_trip_cost__"], errors="coerce") * 1e4
        frame["executable_cost_floor_bps"] = pd.to_numeric(frame["__first_touch_round_trip_cost__"], errors="coerce") * 1e4
        frame["executable_margin_bps"] = frame.gross_bps - frame.executable_cost_floor_bps
        frame["diagnostic_noncausal_vol_regime"] = pd.qcut(frame.atr_bps.rank(method="first"), 3, labels=["low", "medium", "high"]).astype("string")
        frame.loc[:, columns].to_parquet(part_path, index=False, compression="zstd")
        populations.append(_summarise(frame, ["side_name", "month", "ineligible_reason"]))
        geometries.append(_summarise(frame[frame.label_valid], ["side_name", "month", "diagnostic_noncausal_vol_regime"]))
        frame["target_bin"] = np.floor(frame["__first_touch_target_soft__"] * 20).clip(upper=19).astype(int)
        collisions.append(_summarise(frame, ["side_name", "month", "target_bin", "ineligible_reason"]))
        print(json.dumps({"shard": path.name, "rows": len(frame)}), flush=True)
    # Assemble from independently-valid shard parts.  This avoids retaining the
    # 4.5m-row audit population in memory and makes interrupted work resumable.
    population_path = args.out / "target_population_validity.parquet"
    writer = None
    for batch in ds.dataset(part_root, format="parquet").scanner().to_batches():
        if writer is None:
            writer = pq.ParquetWriter(population_path, batch.schema, compression="zstd")
        writer.write_batch(batch)
    if writer is None:
        raise RuntimeError("no valid audit parts were materialised")
    writer.close()
    population = pd.concat(populations, ignore_index=True); geometry = pd.concat(geometries, ignore_index=True); collision = pd.concat(collisions, ignore_index=True)
    population.to_parquet(args.out / "target_ineligible_reason_audit.parquet", index=False)
    geometry.to_parquet(args.out / "target_cost_atr_regime_audit.parquet", index=False)
    geometry.to_parquet(args.out / "target_barrier_economics.parquet", index=False)
    collision.to_parquet(args.out / "target_value_outcome_composition.parquet", index=False)
    manifest = {
        "schema": "packb_target_substrate_geometry_audit_v1",
        "status": "materialized",
        "source": str(args.source),
        "rows": int(population["rows"].sum()),
        "geometry": {"source": "per-row materialised effective TP/SL distances", "fixed_round_trip_cost_bps": COST_BPS},
        "regime": "diagnostic_noncausal_vol_regime is descriptive only and must never be an inference feature",
        "label_valid": "__first_touch_valid_path__ > 0.5",
        "target_invalid": "not label_valid; retained only for coverage diagnostics",
    }
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
