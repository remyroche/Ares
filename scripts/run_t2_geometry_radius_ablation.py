#!/usr/bin/env python3
"""Leakage-safe T2 ±0.5 ATR geometry screen on the development split only.

This deliberately does not open or reuse the previously inspected final OOS
period.  It tests TP {1.5, 2.0, 2.5} x SL {0.5, 1.0, 1.5}, using the frozen
361 causal raw features plus side only.  Row-level execution cost is a target
component and is expressly excluded until an entry-time cost feature is
materialised with its own availability contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from extreme_price_movements.feature_provenance_gate import validate_feature_columns
from extreme_price_movements.t2_atr_funnel import BarrierGeometry, materialize_geometry_events_bulk, soft_event_targets, top_book_metrics
from scripts.run_t2_atr_sequential_funnel import _add_causal_context, _fit_base, _read_paths, _resolved_before, _score_frame


GEOMETRIES = tuple(BarrierGeometry(tp, sl) for tp in (1.5, 2.0, 2.5) for sl in (0.5, 1.0, 1.5))
TEMPERATURE = 0.25


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _with_labels(frame: pd.DataFrame, events: pd.DataFrame, geometry: BarrierGeometry) -> pd.DataFrame:
    result = frame.merge(events, on="candidate_id", how="left", validate="one_to_one")
    if result.geometry.isna().any():
        raise ValueError(f"{geometry.name} does not cover every candidate")
    result[["t2_upper_soft", "t2_lower_soft", "t2_timeout_soft"]] = soft_event_targets(result, geometry, temperature_atr=TEMPERATURE)
    return result


def _selected_detail(scored: pd.DataFrame, geometry: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    ranked = scored.sort_values(["score_bps", "candidate_id"], ascending=[False, True], kind="mergesort")
    book = ranked.head(int(np.ceil(len(ranked) * 0.10))).copy()
    book["week"] = pd.to_datetime(book["__ts__"], utc=True).dt.to_period("W-SUN").astype(str)
    book["decision_hour_utc"] = pd.to_datetime(book["__ts__"], utc=True).dt.hour
    weekly = book.groupby(["week", "side_name"], observed=True).agg(
        selected_trades=("candidate_id", "size"),
        gross_bps_per_trade=("execution_gross_ev_12h", lambda x: float(x.mean() * 10_000.0)),
        cost_bps_per_trade=("execution_cost_return", lambda x: float(x.mean() * 10_000.0)),
        net_bps_per_trade=("execution_net_ev_12h", lambda x: float(x.mean() * 10_000.0)),
        positive_net_rate=("execution_net_ev_12h", lambda x: float((x > 0.0).mean())),
    ).reset_index()
    hours = book.groupby(["decision_hour_utc", "side_name"], observed=True).agg(
        selected_trades=("candidate_id", "size"),
        net_bps_per_trade=("execution_net_ev_12h", lambda x: float(x.mean() * 10_000.0)),
    ).reset_index()
    shares = weekly.groupby("week", observed=True).selected_trades.sum() / len(book)
    concentration = {
        "geometry": geometry,
        "global_top10_selected_rows": int(len(book)),
        "selected_long_rows": int(book.side_name.eq("long").sum()),
        "selected_short_rows": int(book.side_name.eq("short").sum()),
        "selected_week_count": int(shares.size),
        "week_hhi": float((shares**2).sum()),
        "largest_week_share": float(shares.max()),
        "top_three_weeks_share": float(shares.nlargest(3).sum()),
    }
    return weekly, hours, concentration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--features-json", type=Path, required=True)
    parser.add_argument("--paths", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    raw = list(validate_feature_columns(json.loads(args.features_json.read_text())["raw_feature_columns"]))
    required = {
        "candidate_id", "__ts__", "__decision_ts__", "__label_available_at__", "__symbol__", "side_name",
        "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "oof_fold", *raw,
    }
    ledger = pd.read_parquet(args.ledger, columns=sorted(required))
    ledger = _add_causal_context(ledger)
    for name in ("__ts__", "__decision_ts__", "__label_available_at__"):
        ledger[name] = pd.to_datetime(ledger[name], utc=True, errors="raise")
    if not ledger["__decision_ts__"].eq(ledger["__ts__"] + pd.Timedelta(hours=1)).all():
        raise ValueError("entry is not one complete hourly bar after the feature cutoff")
    if not ledger["__label_available_at__"].eq(ledger["__decision_ts__"] + pd.Timedelta(hours=12)).all():
        raise ValueError("labels are not H12-resolved")
    base = ledger.loc[ledger.oof_fold.eq("base_train")].copy()
    development = ledger.loc[ledger.oof_fold.eq("meta_train")].copy()
    train = _resolved_before(base, development)
    paths = _read_paths(set(ledger.candidate_id.astype(str)), list(args.paths))
    events = materialize_geometry_events_bulk(paths, GEOMETRIES)

    metrics, weekly_rows, hour_rows, concentration_rows = [], [], [], []
    for geometry in GEOMETRIES:
        train_labeled = _with_labels(train, events[geometry.name], geometry)
        development_labeled = _with_labels(development, events[geometry.name], geometry)
        score, probabilities = _fit_base(train_labeled, development_labeled, raw)
        scored = _score_frame(development_labeled, score, "base_only_no_cost_context", "development_geometry_radius", geometry.name, TEMPERATURE)
        result = top_book_metrics(scored, score_column="score_bps")
        result["geometry"] = geometry.name
        result["tp_atr"] = geometry.tp_atr
        result["sl_atr"] = geometry.sl_atr
        result["temperature_atr"] = TEMPERATURE
        result["training_rows_after_resolution_purge"] = len(train_labeled)
        metrics.append(result)
        weekly, hours, concentration = _selected_detail(scored, geometry.name)
        weekly_rows.append(weekly.assign(geometry=geometry.name))
        hour_rows.append(hours.assign(geometry=geometry.name))
        concentration_rows.append(concentration)
        probability = pd.DataFrame(probabilities, columns=["p_upper", "p_lower", "p_timeout"])
        if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-7):
            raise ValueError("base event probabilities do not sum to one")

    stage = Path(tempfile.mkdtemp(prefix=f".{args.output.name}.", dir=args.output.parent))
    try:
        pd.concat(metrics, ignore_index=True).to_parquet(stage / "geometry_radius_metrics.parquet", index=False)
        pd.concat(weekly_rows, ignore_index=True).to_parquet(stage / "top10_weekly_side_metrics.parquet", index=False)
        pd.concat(hour_rows, ignore_index=True).to_parquet(stage / "top10_time_concentration.parquet", index=False)
        pd.DataFrame(concentration_rows).to_parquet(stage / "top10_concentration_summary.parquet", index=False)
        (stage / "feature_contract.json").write_text(json.dumps({
            "raw_feature_count": len(raw), "raw_features": raw,
            "derived_context": ["side_is_long"],
            "excluded_from_model": ["execution_cost_return", "execution_gross_ev_12h", "execution_net_ev_12h", "all path labels"],
        }, indent=2) + "\n")
        manifest = {
            "schema": "t2_geometry_radius_ablation_v1",
            "status": "COMPLETED_DEVELOPMENT_ONLY_FINAL_OOS_NOT_OPENED",
            "geometry_grid": [{"name": g.name, "tp_atr": g.tp_atr, "sl_atr": g.sl_atr} for g in GEOMETRIES],
            "temperature_atr": TEMPERATURE,
            "training": {"raw_base_window": "2023-04-01..2024-03-31 feature cutoffs", "strict_label_rule": "label_available_at < first development execution decision", "rows_after_purge": len(train)},
            "evaluation": {"development_window": "2024-04-01..2024-07-31 feature cutoffs", "entry": "feature cutoff + 1h after completed bar close", "final_oos": "not opened by this ablation"},
            "selection": "pooled global top-k across both sides and all timestamps; no portfolio constraints",
            "feature_contract": "361 frozen causal raw fields plus side_is_long; no realised row cost context",
            "inputs": {str(p): _sha(p) for p in [args.ledger, args.features_json, *args.paths]},
        }
        (stage / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(stage, args.output)
    except Exception:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
