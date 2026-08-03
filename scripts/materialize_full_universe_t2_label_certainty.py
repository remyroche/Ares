#!/usr/bin/env python3
"""Build training-only T2 label-stability diagnostics from the full panel.

The full-universe panel retains four exact H12 TP/SL first-touch contracts and
H12 MFE/MAE.  This materializer deliberately uses only those observable
neighbours; contracts requiring a delayed entry, a longer path, or a second
ATR estimate are recorded as unavailable rather than approximated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


GEOMETRIES = ("tp2_sl1", "tp2_sl2", "tp3_sl1", "tp3_sl2")
REFERENCE = "tp3_sl2"


def _state(frame: pd.DataFrame, geometry: str) -> np.ndarray:
    # 0=upper first, 1=lower first, 2=timeout; this is the stored exact
    # first-touch outcome, with the panel's adverse same-minute tie rule.
    return frame[f"t2_{geometry}_event"].to_numpy(np.int8)


def _soft(frame: pd.DataFrame, tau: float) -> np.ndarray:
    """Same soft H12 construction as the base target, for stability only."""
    event = _state(frame, REFERENCE)
    exit_minute = frame[f"t2_{REFERENCE}_exit_minute"].to_numpy(float)
    mfe = frame["t2_path_mfe_atr"].to_numpy(float)
    mae = frame["t2_path_mae_atr"].to_numpy(float)
    upper = (mfe - 3.0) / tau
    lower = (mae - 2.0) / tau
    timeout = np.minimum((3.0 - mfe) / tau, (2.0 - mae) / tau)
    bonus = 2.0 + 0.75 * (1.0 - np.minimum(exit_minute, 720.0) / 720.0)
    upper[event == 0] += bonus[event == 0]
    lower[event == 1] += bonus[event == 1]
    timeout[event == 2] += 2.0
    logits = np.column_stack([upper, lower, timeout])
    logits -= logits.max(axis=1, keepdims=True)
    prob = np.exp(np.clip(logits, -40.0, 0.0))
    return prob / prob.sum(axis=1, keepdims=True)


def _diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    states = np.column_stack([_state(frame, geometry) for geometry in GEOMETRIES])
    reference = states[:, -1]
    event_agreement = (states == reference[:, None]).mean(axis=1)
    sign = np.where(states == 0, 1, np.where(states == 1, -1, 0))
    reference_sign = sign[:, -1]
    sign_agreement = (sign == reference_sign[:, None]).mean(axis=1)
    values = np.column_stack([frame[f"t4_{geometry}_exit_pnl_atr"].to_numpy(float) for geometry in GEOMETRIES])
    value_dispersion = values.std(axis=1)
    soft = np.stack([_soft(frame, tau) for tau in (0.10, 0.25, 0.50)], axis=1)
    soft_dispersion = soft.std(axis=1).mean(axis=1)
    # Near either reference barrier means a small path perturbation can change
    # the label.  Large margins are consequently more stable.
    # Extremely distant paths are all equivalently stable for a nearby-barrier
    # diagnostic.  Capping avoids a handful of tiny-ATR episodes dominating
    # the reported distribution without changing the certainty interpretation.
    distance = np.clip(np.minimum(np.abs(frame["t2_path_mfe_atr"].to_numpy(float) - 3.0), np.abs(frame["t2_path_mae_atr"].to_numpy(float) - 2.0)), 0.0, 10.0)
    value_stability = 1.0 / (1.0 + value_dispersion)
    soft_stability = 1.0 / (1.0 + 4.0 * soft_dispersion)
    margin_stability = np.tanh(np.maximum(distance, 0.0))
    certainty = np.clip(.30 * event_agreement + .20 * sign_agreement + .20 * value_stability + .15 * soft_stability + .15 * margin_stability, 0.0, 1.0)
    return pd.DataFrame({
        "candidate_id": frame["candidate_id"].astype(str),
        "__ts__": frame["__ts__"],
        "side_name": frame["side_name"].astype(str),
        "event_agreement_rate": event_agreement.astype("float32"),
        "target_sign_agreement_rate": sign_agreement.astype("float32"),
        "target_value_dispersion_atr": value_dispersion.astype("float32"),
        "softness_value_dispersion": soft_dispersion.astype("float32"),
        "top_bottom_state_agreement": event_agreement.astype("float32"),
        "distance_nearest_barrier_atr": distance.astype("float32"),
        "same_bar_conflict_flag": np.full(len(frame), np.nan, dtype="float32"),
        "path_completeness": np.ones(len(frame), dtype="float32"),
        "sensitivity_to_entry_delay": np.full(len(frame), np.nan, dtype="float32"),
        "sensitivity_to_atr_definition": np.full(len(frame), np.nan, dtype="float32"),
        "label_certainty": certainty.astype("float32"),
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    columns = ["candidate_id", "__ts__", "side_name", "t2_path_mfe_atr", "t2_path_mae_atr"]
    for geometry in GEOMETRIES:
        columns += [f"t2_{geometry}_event", f"t2_{geometry}_exit_minute", f"t4_{geometry}_exit_pnl_atr"]
    parts = sorted((args.panel / "parts").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no panel parts in {args.panel}")
    args.out.mkdir(parents=True)
    destination = args.out / "label_certainty_diagnostics.parquet"
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for part in parts:
            result = _diagnostics(pd.read_parquet(part, columns=columns))
            table = pa.Table.from_pandas(result, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(result)
    finally:
        if writer is not None:
            writer.close()
    manifest = {
        "schema": "full_universe_t2_label_certainty_v1",
        "status": "COMPLETED_TRAINING_ONLY",
        "panel": str(args.panel),
        "rows": rows,
        "reference": {"geometry": REFERENCE, "horizon_hours": 12, "softness_tau": 0.25},
        "materialized_neighbourhood": {"geometries": list(GEOMETRIES), "softness_tau": [0.10, 0.25, 0.50]},
        "unavailable_without_rematerialising_1m_paths": ["entry delays +1m/+5m", "H16", "short/long alternative pre-entry ATR", "same-minute conflict flag"],
        "inference_use": "forbidden; future-path-derived training weights and diagnostics only",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"rows": rows, "output": str(destination)}))


if __name__ == "__main__":
    main()
