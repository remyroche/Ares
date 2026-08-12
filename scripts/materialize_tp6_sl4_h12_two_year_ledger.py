#!/usr/bin/env python3
"""Materialize the contiguous two-year TP6/SL4/H12 research ledger.

The Stage-I input already contains the causal feature surface and strict-OOF
base outputs.  This utility joins the authoritative selector labels back in,
restricts to an explicit 24-month window, and writes a self-describing ledger
for the subsequent base/specialist/residual retraining run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data_perp/artifacts/correctness_leaf_regime_two_year_input_20260803_v3/input.parquet"
DEFAULT_LABELS = ROOT / "data_perp/artifacts/stage_i_selector_sample_20260803_v5/selector_ledger.parquet"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_h12_two_year_ledger_20260806_v1"
# Earliest contiguous 24-month window in the compatible Stage-I contract.
# December 2024 is absent from the source, so a later July-2024--June-2026
# window would not be a complete monthly evaluation panel.
START = pd.Timestamp("2022-09-01", tz="UTC")
END = pd.Timestamp("2024-09-01", tz="UTC")

ID = ("candidate_id", "__ts__", "__symbol__", "side_name")
LABELS = (
    "label_valid", "exact_gross_bps", "exact_net_bps", "label_available_ts",
    "t2_tp6_sl4_event", "robust_clear_event_b25", "robust_clear_soft_b25_t50",
    "r3_class", "r3_metric_target", "decision_ts",
)
MODEL_EXCLUDE = set(ID) | {
    "label_valid", "exact_gross_bps", "exact_net_bps", "label_available_ts",
    "t2_tp6_sl4_event", "robust_clear_event_b25", "robust_clear_soft_b25_t50",
    "r3_class", "r3_metric_target", "decision_ts", "gross_bps", "net_bps",
    "era",
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(out: Path = DEFAULT_OUT, *, input_path: Path = DEFAULT_INPUT, labels_path: Path = DEFAULT_LABELS) -> Path:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(input_path)
    labels = pd.read_parquet(labels_path, columns=["candidate_id", *LABELS])
    if labels.candidate_id.duplicated().any():
        raise ValueError("label ledger contains duplicate candidate IDs")
    panel["__ts__"] = pd.to_datetime(panel["__ts__"], utc=True)
    labels["label_available_ts"] = pd.to_datetime(labels["label_available_ts"], utc=True)
    labels["decision_ts"] = pd.to_datetime(labels["decision_ts"], utc=True)
    joined = panel.merge(labels, on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "_label"))
    if not np.allclose(joined["net_bps"], joined["exact_net_bps"], atol=0.01, rtol=0.0):
        raise ValueError("panel and authoritative exact-net labels disagree")
    if not np.allclose(joined["gross_bps"], joined["exact_gross_bps"], atol=0.01, rtol=0.0):
        raise ValueError("panel and authoritative exact-gross labels disagree")
    joined = joined[(joined["__ts__"] >= START) & (joined["__ts__"] < END)].copy()
    joined = joined[joined["label_valid"].fillna(False)].copy()
    joined["month"] = joined["__ts__"].dt.strftime("%Y-%m")
    joined["cost_bps"] = (joined["exact_gross_bps"] - joined["exact_net_bps"]).astype("float32")
    delta = joined["label_available_ts"] - joined["decision_ts"]
    if not (delta == pd.Timedelta(hours=12)).all():
        raise ValueError("TP6/SL4/H12 label availability is not exactly decision + 12h")
    if not np.allclose(joined["cost_bps"], 100.0, atol=0.05, rtol=0.0):
        raise ValueError("fixed 100-bps cost contract failed")
    months = sorted(joined["month"].unique())
    expected = pd.period_range("2022-09", "2024-08", freq="M").astype(str).tolist()
    if months != expected:
        raise ValueError(f"window is not a complete 24-month panel: {months}")
    model_features = [c for c in joined.columns if c not in MODEL_EXCLUDE and pd.api.types.is_numeric_dtype(joined[c])]
    coverage = joined[model_features].replace([np.inf, -np.inf], np.nan).notna().mean()
    variation = joined[model_features].replace([np.inf, -np.inf], np.nan).std(ddof=0) > 1e-12
    availability = pd.DataFrame({
        "feature": model_features,
        "coverage": coverage.to_numpy(float),
        "nonconstant": variation.to_numpy(bool),
        "usable_90pct_nonconstant": (coverage >= 0.90).to_numpy(bool) & variation.to_numpy(bool),
    })
    joined = joined.sort_values(["__ts__", "candidate_id"], kind="stable").reset_index(drop=True)
    joined.to_parquet(out / "ledger.parquet", index=False, compression="zstd")
    availability.to_parquet(out / "feature_availability.parquet", index=False)
    monthly = joined.groupby(["month", "side_name"], observed=True).agg(
        rows=("candidate_id", "size"),
        label_valid=("label_valid", "mean"),
        median_net_bps=("exact_net_bps", "median"),
        mean_net_bps=("exact_net_bps", "mean"),
        mean_cost_bps=("cost_bps", "mean"),
    ).reset_index()
    monthly.to_parquet(out / "monthly_population_audit.parquet", index=False)
    manifest = {
        "schema": "unified_tp6_sl4_h12_feature_label_ledger_v1",
        "status": "COMPLETED",
        "window_start_utc": START.isoformat(),
        "window_end_exclusive_utc": END.isoformat(),
        "months": months,
        "rows": int(len(joined)),
        "rows_by_side": {str(k): int(v) for k, v in joined.side_name.value_counts().items()},
        "geometry": "TP6/SL4/H12; exact 12-hour label horizon; 100 bps fixed cost",
        "entry_contract": "Stage-I candidate decision_ts and label_available_ts = decision_ts + 12h",
        "base_outputs": "strict-OOF R3 probabilities and prequential same-side EV map retained for audit; retraining uses labels below",
        "feature_count_numeric": len(model_features),
        "feature_count_usable_90pct_nonconstant": int(availability.usable_90pct_nonconstant.sum()),
        "label_contract_checks": {
            "all_label_valid": bool(joined.label_valid.all()),
            "all_cost_100_bps": bool(np.allclose(joined.cost_bps, 100.0, atol=0.05)),
            "all_h12_availability": bool((delta == pd.Timedelta(hours=12)).all()) if len(delta) else False,
            "all_months_present": months == expected,
        },
        "sources": {"input": str(input_path), "input_sha256": _sha(input_path), "labels": str(labels_path), "labels_sha256": _sha(labels_path)},
        "outputs": ["ledger.parquet", "feature_availability.parquet", "monthly_population_audit.parquet"],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    args = parser.parse_args()
    print(run(args.out, input_path=args.input, labels_path=args.labels))
