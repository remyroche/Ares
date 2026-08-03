#!/usr/bin/env python3
"""Audit the exact TP6/SL4 H12 target population before target repair.

This is deliberately anchored to the declared winning base-target contract:
exact next-minute entry, TP +6 ATR, SL -4 ATR, adverse tie precedence, H12
timeout, and a fixed 100-bps round-trip cost.  Incomplete minute paths remain
explicitly invalid supervision; they are never converted to zero-value labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data_perp/artifacts/full_universe_t2_t4_panel_20260801_v3"
SIDECAR = ROOT / "data_perp/artifacts/full_universe_tp6_sl4_h12_sidecar_20260802_v1"
COST_BPS = 100.0
TP_ATR, SL_ATR, HORIZON_MINUTES = 6.0, 4.0, 720.0


def _b3_upper(event: np.ndarray, exit_minute: np.ndarray) -> np.ndarray:
    """The selected B3 winner's upper-state membership, without labels as inputs."""
    confidence = .75 + .25 * np.exp(-np.minimum(exit_minute, HORIZON_MINUTES) / 60.0 / 8.0)
    return np.where(event == 0, confidence, (1.0 - confidence) / 2.0)


def _reason(complete: pd.Series, candidate_id: pd.Series) -> pd.Series:
    result = pd.Series("incomplete_h12_path", index=complete.index, dtype="string")
    result.loc[candidate_id.isna() | candidate_id.astype("string").str.len().eq(0)] = "missing_candidate_identity"
    result.loc[complete] = "complete_executable_path"
    return result


def _summary(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for key, part in frame.groupby(groups, dropna=False, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        valid = part.loc[part.label_valid]
        rec: dict[str, object] = dict(zip(groups, key, strict=True))
        rec.update({
            "rows": int(len(part)), "valid_rows": int(len(valid)),
            "label_valid_rate": float(part.label_valid.mean()),
            "entry_executable_rate": float(part.entry_executable.mean()),
            "gross_bps": float(valid.gross_bps.mean()) if len(valid) else float("nan"),
            "net_bps": float(valid.net_bps.mean()) if len(valid) else float("nan"),
            "target_mean": float(valid.target_b3_upper.mean()) if len(valid) else float("nan"),
            "target_zero_rate": float(valid.target_b3_upper.eq(0).mean()) if len(valid) else float("nan"),
            "upper_first_rate": float(valid.event_upper.mean()) if len(valid) else float("nan"),
            "lower_first_rate": float(valid.event_lower.mean()) if len(valid) else float("nan"),
            "timeout_rate": float(valid.event_timeout.mean()) if len(valid) else float("nan"),
            "median_atr_bps": float(valid.atr_bps.median()) if len(valid) else float("nan"),
            "median_tp_bps": float(valid.tp_bps.median()) if len(valid) else float("nan"),
            "median_tp_net_margin_bps": float(valid.tp_net_margin_bps.median()) if len(valid) else float("nan"),
            "tp_margin_le_0_rate": float(valid.tp_net_margin_bps.le(0).mean()) if len(valid) else float("nan"),
            "tp_margin_le_25_rate": float(valid.tp_net_margin_bps.le(25).mean()) if len(valid) else float("nan"),
            "tp_margin_le_50_rate": float(valid.tp_net_margin_bps.le(50).mean()) if len(valid) else float("nan"),
        })
        rows.append(rec)
    return pd.DataFrame(rows)


def _oracle(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.loc[frame.label_valid].copy()
    records: list[dict[str, object]] = []
    for groups in ([], ["side_name"], ["diagnostic_cost_atr_regime"]):
        grouped = [((), work)] if not groups else work.groupby(groups, observed=True, sort=True)
        for key, part in grouped:
            key = key if isinstance(key, tuple) else (key,)
            ordered = part.sort_values(["target_b3_upper", "candidate_id"], ascending=[False, True], kind="mergesort")
            for fraction in (.01, .05, .10, .20):
                selected = ordered.head(max(1, int(np.ceil(len(ordered) * fraction))))
                record = {"scope": "global" if not groups else "+".join(groups), "top_fraction": fraction,
                          "rows": len(part), "selected_rows": len(selected),
                          "oracle_gross_bps": float(selected.gross_bps.mean()), "oracle_net_bps": float(selected.net_bps.mean())}
                record.update(dict(zip(groups, key, strict=True)))
                records.append(record)
    return pd.DataFrame(records)


def _write_combined(part_root: Path, destination: Path) -> None:
    writer = None
    for batch in ds.dataset(part_root, format="parquet").scanner().to_batches():
        if writer is None:
            writer = pq.ParquetWriter(destination, batch.schema, compression="zstd")
        writer.write_batch(batch)
    if writer is None:
        raise RuntimeError("no audit parts materialised")
    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--sidecar", type=Path, default=SIDECAR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-parts", type=int, default=None, help="test-only bounded materialisation")
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    panel_parts = sorted((args.panel / "parts").glob("*.parquet"))
    if not panel_parts:
        raise FileNotFoundError(args.panel / "parts")
    if args.max_parts is not None:
        panel_parts = panel_parts[:args.max_parts]
    args.out.mkdir(parents=True)
    part_root = args.out / "target_population_validity_parts"; part_root.mkdir()
    population, geometry, composition = [], [], []
    panel_cols = ["candidate_id", "__ts__", "__symbol__", "side_name", "decision_price", "atr_1h"]
    sidecar_cols = ["candidate_id", "t2_tp6_sl4_event", "t2_tp6_sl4_exit_minute", "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps"]
    for panel_path in panel_parts:
        sidecar_path = args.sidecar / "parts" / panel_path.name
        base = pd.read_parquet(panel_path, columns=panel_cols)
        labels = pd.read_parquet(sidecar_path, columns=sidecar_cols) if sidecar_path.exists() else pd.DataFrame(columns=sidecar_cols)
        frame = base.merge(labels, on="candidate_id", how="left", validate="one_to_one")
        frame["__ts__"] = pd.to_datetime(frame["__ts__"], utc=True, errors="raise")
        frame["month"] = frame["__ts__"].dt.strftime("%Y-%m")
        frame["label_valid"] = frame["t2_tp6_sl4_event"].notna()
        frame["entry_executable"] = frame.label_valid
        frame["target_invalid"] = ~frame.label_valid
        frame["ineligible_reason"] = _reason(frame.label_valid, frame.candidate_id)
        frame["atr_bps"] = pd.to_numeric(frame.atr_1h, errors="coerce") / pd.to_numeric(frame.decision_price, errors="coerce") * 1e4
        frame["tp_bps"] = TP_ATR * frame.atr_bps; frame["sl_bps"] = SL_ATR * frame.atr_bps
        frame["tp_net_margin_bps"] = frame.tp_bps - COST_BPS
        frame["cost_to_atr"] = COST_BPS / frame.atr_bps.replace(0, np.nan); frame["cost_to_tp"] = COST_BPS / frame.tp_bps.replace(0, np.nan)
        frame["diagnostic_cost_atr_regime"] = pd.cut(frame.cost_to_tp, [-np.inf, .5, 1., np.inf], labels=["headroom", "thin_margin", "cost_dominated"]).astype("string")
        event = pd.to_numeric(frame.t2_tp6_sl4_event, errors="coerce").fillna(2).astype(int).to_numpy()
        exit_minute = pd.to_numeric(frame.t2_tp6_sl4_exit_minute, errors="coerce").fillna(HORIZON_MINUTES).to_numpy(float)
        frame["event_upper"] = frame.label_valid & (event == 0); frame["event_lower"] = frame.label_valid & (event == 1); frame["event_timeout"] = frame.label_valid & (event == 2)
        frame["target_b3_upper"] = np.where(frame.label_valid, _b3_upper(event, exit_minute), np.nan)
        frame["target_bin"] = np.where(frame.label_valid, np.floor(frame.target_b3_upper * 20).clip(0, 19), -1).astype(int)
        frame["gross_bps"] = pd.to_numeric(frame.t4_tp6_sl4_gross_bps, errors="coerce")
        frame["net_bps"] = pd.to_numeric(frame.t4_tp6_sl4_net_bps, errors="coerce")
        keep = ["candidate_id", "__ts__", "__symbol__", "side_name", "month", "label_valid", "entry_executable", "target_invalid", "ineligible_reason", "event_upper", "event_lower", "event_timeout", "atr_bps", "tp_bps", "sl_bps", "tp_net_margin_bps", "cost_to_atr", "cost_to_tp", "diagnostic_cost_atr_regime", "gross_bps", "net_bps", "target_b3_upper", "target_bin"]
        frame.loc[:, keep].to_parquet(part_root / panel_path.name, index=False, compression="zstd")
        population.append(_summary(frame, ["side_name", "month", "ineligible_reason"]))
        geometry.append(_summary(frame, ["side_name", "month", "diagnostic_cost_atr_regime"]))
        composition.append(_summary(frame, ["side_name", "target_bin", "ineligible_reason"]))
        print(json.dumps({"part": panel_path.name, "rows": len(frame), "valid_rows": int(frame.label_valid.sum())}), flush=True)
    _write_combined(part_root, args.out / "target_population_validity.parquet")
    population_frame = pd.concat(population, ignore_index=True)
    geometry_frame = pd.concat(geometry, ignore_index=True)
    composition_frame = pd.concat(composition, ignore_index=True)
    population_frame.to_parquet(args.out / "target_ineligible_reason_audit.parquet", index=False)
    geometry_frame.to_parquet(args.out / "target_cost_atr_regime_audit.parquet", index=False)
    geometry_frame.to_parquet(args.out / "target_barrier_economics.parquet", index=False)
    composition_frame.to_parquet(args.out / "target_value_outcome_composition.parquet", index=False)
    all_rows = ds.dataset(part_root, format="parquet").to_table().to_pandas()
    _oracle(all_rows).to_parquet(args.out / "target_regime_oracle_results.parquet", index=False)
    manifest = {"schema": "tp6_sl4_target_substrate_audit_v1", "status": "COMPLETED" if args.max_parts is None else "PARTIAL_TEST_ONLY", "rows": int(len(all_rows)), "contract": {"entry": "signal close + 1h then exact next-minute open", "geometry": "TP +6 ATR / SL -4 ATR / H12", "same_minute_conflict": "adverse lower first", "cost_bps": COST_BPS, "label_availability": "entry + H12"}, "invalid_semantics": "sidecar absence is incomplete H12 path and target_invalid=true; invalid rows have no numerical target", "target_control": "B3 winner: hard floor .75, time-decay 8h", "diagnostic_regime": "cost_to_tp bands are decision-time geometry diagnostics, not model features"}
    (args.out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
