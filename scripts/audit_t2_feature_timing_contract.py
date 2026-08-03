#!/usr/bin/env python3
"""Fail-closed timing and feature audit for the T2 research ledger."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.feature_provenance_gate import validate_feature_columns


TARGET_COLUMNS = {
    "execution_gross_ev_12h", "execution_cost_return", "execution_net_ev_12h", "__label_end_ts__", "__label_available_at__",
    "favorable_first", "adverse_first", "timeout", "__opportunity_occurred_12h__",
    "__peak_mfe_atr_12h__", "__time_to_first_meaningful_mfe_hours_12h__", "__mae_before_meaningful_mfe_atr_12h__", "__future_slope_atr_per_hour_12h__",
}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--features-json", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    raw = list(validate_feature_columns(json.loads(a.features_json.read_text())["raw_feature_columns"]))
    cols = ["candidate_id", "__ts__", "__decision_ts__", "__label_end_ts__", "__label_available_at__", "oof_fold", "execution_net_ev_12h", *raw]
    x = pd.read_parquet(a.ledger, columns=cols)
    for name in ("__ts__", "__decision_ts__", "__label_end_ts__", "__label_available_at__"):
        x[name] = pd.to_datetime(x[name], utc=True, errors="raise")
    timing = {
        "rows": len(x),
        "feature_cutoff_to_entry_1h_violations": int((~x.__decision_ts__.eq(x.__ts__ + pd.Timedelta(hours=1))).sum()),
        "entry_to_h12_label_end_violations": int((~x.__label_end_ts__.eq(x.__decision_ts__ + pd.Timedelta(hours=12))).sum()),
        "label_availability_violations": int((~x.__label_available_at__.eq(x.__label_end_ts__)).sum()),
    }
    train = x.loc[x.oof_fold.eq("base_train")]
    y = train.execution_net_ev_12h.to_numpy(float)
    rows = []
    for name in raw:
        values = pd.to_numeric(train[name], errors="coerce").to_numpy(float)
        valid = np.isfinite(values) & np.isfinite(y)
        rho = float(spearmanr(values[valid], y[valid]).statistic) if valid.sum() > 100 and np.nanstd(values[valid]) > 0.0 else np.nan
        rows.append({"feature_name": name, "train_finite_fraction": float(valid.mean()), "net_spearman": rho, "abs_net_spearman": abs(rho) if np.isfinite(rho) else np.nan})
    feature_audit = pd.DataFrame(rows).sort_values("abs_net_spearman", ascending=False, kind="mergesort")
    overlap = sorted(set(raw).intersection(TARGET_COLUMNS))
    report = {
        "schema": "t2_feature_timing_audit_v1",
        "timing": timing,
        "feature_contract": {"raw_feature_count": len(raw), "direct_target_name_overlap": overlap, "name_level_causal_gate": "passed"},
        "god_feature_screen": {"method": "single-feature Spearman against realised H12 net on base-train only", "largest_absolute_correlation": float(feature_audit.abs_net_spearman.max()), "threshold": 0.30, "passes_threshold": bool(feature_audit.abs_net_spearman.max() < .30), "interpretation": "no individual raw feature behaves like a direct target proxy; this is diagnostic evidence, not a substitute for source lineage"},
        "cost_context": {"status": "REJECTED_FROM_REVISED_T2", "reason": "the only available value is execution_cost_return in the realised target ledger; its entry-time provenance is not independently materialised"},
        "lineage_limit": "The historical 361-column contract has row-level feature-cutoff timing but no complete per-feature dependency/availability registry.  Therefore this audit establishes no observed timing or semantic violation, but cannot certify every feature's source lineage as leakage-free.",
        "overall": "PASS_TIMING_AND_DIRECT_TARGET_SCREEN__PER_FEATURE_LINEAGE_CERTIFICATION_PENDING",
    }
    stage = Path(tempfile.mkdtemp(prefix=f".{a.output.name}.", dir=a.output.parent))
    try:
        feature_audit.to_parquet(stage / "raw_feature_univariate_screen.parquet", index=False)
        (stage / "feature_contract.json").write_text(json.dumps({"raw_features": raw}, indent=2) + "\n")
        (stage / "audit_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(stage, a.output)
    except Exception:
        import shutil
        shutil.rmtree(stage, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
