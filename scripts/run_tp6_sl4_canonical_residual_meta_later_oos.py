#!/usr/bin/env python3
"""Apply the selected canonical residual meta arms to the later July window.

The residual model is fit on the 2025 development OOF population only.  The
later 20--23 July 2026 rows are scored once with the same TP6/SL4 exits.  This
is a robustness check, not a new HPO opportunity: no later outcome is used in
feature normalization, the canonical net map, or the meta fit.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_canonical_residual_meta_block_ablation import (  # noqa: E402
    BLOCKS,
    DEFAULT_HEADS,
    _feature_frame,
    _fit_meta,
    _load,
    _map_canonical,
)
from scripts.run_tp6_sl4_downstream_retrain_2025 import _map_base  # noqa: E402
from scripts.run_tp6_sl4_rolling_gam_residual_integration import _pct, _rank_fit  # noqa: E402
from scripts.run_tp6_sl4_gam_canonical_later_oos import _context_fields, _load_panel  # noqa: E402

DEV_PANEL = ROOT / "data_perp/artifacts/tp6_sl4_downstream_retrain_2025_20260807_v1/predictions_2025.parquet"
DEV_HEADS = DEFAULT_HEADS
SELECTION = ROOT / "data_perp/artifacts/tp6_sl4_canonical_residual_feature_selection_20260808_v1/selected_features.json"
LATER_OUT = ROOT / "data_perp/artifacts/tp6_sl4_gam_canonical_later_oos_20260808_v1"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_canonical_residual_meta_later_oos_20260808_v1"
SEED = 20260808
CANONICAL_HEAD_SEED = 20260815


def _head_health(train: pd.DataFrame, held: pd.DataFrame, context: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    tr = train.copy(); te = held.copy()
    tr_anchor, te_anchor = _map_base(tr, te)
    tr["base_anchor"] = tr_anchor; te["base_anchor"] = te_anchor
    residual = tr.exact_net_bps.to_numpy(float) - tr_anchor
    grade = np.digitize(residual, [-150.0, -50.0, 50.0, 150.0]).astype(np.int32)
    base_rank_tr = _pct(tr.base_score.to_numpy(float), tr.base_score.to_numpy(float))
    base_rank_te = _pct(te.base_score.to_numpy(float), tr.base_score.to_numpy(float))
    ranks_tr, ranks_te = [], []
    for cap in (25, 40, 60, min(73, len(context))):
        fields = ["base_anchor", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", *context[:cap]]
        for equal_month in (False, True):
            raw_tr, raw_te = _rank_fit(tr, te, fields, grade, equal_month=equal_month, seed=CANONICAL_HEAD_SEED + 7 * 100 + cap + int(equal_month), feature_fraction=0.82)
            ranks_tr.append(_pct(raw_tr, raw_tr)); ranks_te.append(_pct(raw_te, raw_tr))
    rt = np.column_stack(ranks_tr); re = np.column_stack(ranks_te)
    for out, values in ((tr, rt), (te, re)):
        out["consensus_head_rank_std"] = values.std(axis=1)
        out["consensus_head_rank_mad"] = np.median(np.abs(values - np.median(values, axis=1, keepdims=True)), axis=1)
        out["consensus_head_rank_iqr"] = np.percentile(values, 75, axis=1) - np.percentile(values, 25, axis=1)
        out["consensus_head_rank_min"] = values.min(axis=1)
        out["consensus_head_rank_max"] = values.max(axis=1)
        out["consensus_head_agreement_fraction"] = np.mean(np.abs(values - np.median(values, axis=1, keepdims=True)) <= .10, axis=1)
        out["consensus_head_raw_std"] = values.std(axis=1)
        base_rank = base_rank_tr if out is tr else base_rank_te
        out["base_consensus_disagreement"] = base_rank - np.median(values, axis=1)
        out["base_plus_consensus25"] = 0.75 * base_rank + 0.25 * np.median(values, axis=1)
    return tr, te, float(np.nanmean(np.abs(rt.mean(axis=0) - re.mean(axis=0))))


def _metric(frame: pd.DataFrame, score: str, tail: float) -> dict[str, object]:
    n = max(1, int(math.ceil(len(frame) * tail)))
    top = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
    return {"tail": tail, "trades": len(top), "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": float(top.exact_net_bps.mean()), "rank_ic": float(frame[[score, "exact_net_bps"]].corr(method="spearman").iloc[0, 1])}


def run(*, output_dir: Path = DEFAULT_OUT) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    dev, context, context_hash = _load()
    heads = pd.read_parquet(DEV_HEADS)
    dev = dev.merge(heads, on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "_head"))
    selected_doc = json.loads(SELECTION.read_text())
    selected = {k: list(v) for k, v in selected_doc["selected_features"].items()}
    # Later panel is loaded from the same frozen F0/context contract used by
    # the canonical later replay.  The target is long-only, matching that
    # untouched OOS artifact.
    later_context = _context_fields()
    full_panel, later, later_hash = _load_panel(later_context)
    historical = full_panel.loc[
        full_panel.__ts__.dt.strftime("%Y-%m").isin([f"2025-{m:02d}" for m in range(1, 13)])
        & full_panel.label_valid.fillna(False)
        & full_panel.side_name.eq("long")
    ].copy()
    later = later.copy()
    later["exact_net_bps"] = pd.to_numeric(later["exact_net_bps"], errors="coerce")
    later["exact_gross_bps"] = pd.to_numeric(later["exact_gross_bps"], errors="coerce")
    # Fit the canonical heads before the later window and generate raw health
    # summaries.  No later outcome is read by this function.
    historical, later_health, _ = _head_health(full_panel.loc[full_panel.__ts__.lt(later.__ts__.min()) & full_panel.label_valid.fillna(False) & full_panel.side_name.eq("long")].copy(), later, later_context)
    # Restore the 2025 OOF training substrate for the residual meta fit.
    dev = dev.loc[dev.month.astype(str).isin([f"2025-{m:02d}" for m in range(1, 13)]) & dev.side_name.eq("long")].copy()
    dev["canonical_expected_net_bps"], _ = _map_canonical(dev, dev)
    dev["meta_residual_bps"] = dev.exact_net_bps.to_numpy(float) - dev.canonical_expected_net_bps.to_numpy(float)
    # Use the all-2025 rows as the fit-normalization reference, then apply the
    # same statistics to later rows.  The helper itself uses train-only stats.
    dev_x, later_x, feature_audit = _feature_frame(dev, later_health, context)
    dev_x["canonical_expected_net_bps"] = dev.canonical_expected_net_bps.to_numpy(float)
    # The canonical map for later is fitted solely on the 2025 development rows.
    model_map = IsotonicRegression(out_of_bounds="clip", y_min=-1000.0, y_max=1000.0)
    model_map.fit(dev.base_plus_consensus25, dev.exact_net_bps)
    later["canonical_expected_net_bps"] = model_map.predict(later_health.base_plus_consensus25).astype(np.float32)
    later["base_plus_consensus25"] = later_health.base_plus_consensus25.to_numpy(float)
    results: list[pd.DataFrame] = []
    arms = {"control": (), "B_uncertainty": ("uncertainty",), "H_full": ("uncertainty", "support_ood", "drift", "market_state")}
    for arm, block_names in arms.items():
        if arm == "control":
            rank = np.full(len(later), .5, dtype=np.float32)
        else:
            features = [c for b in block_names for c in selected.get(b, BLOCKS[b]) if c not in {"canonical_expected_net_bps", "base_plus_consensus25"}]
            features = list(dict.fromkeys(["canonical_expected_net_bps", "base_plus_consensus25", *features]))
            xtr = dev_x.copy(); xte = later_x.copy()
            xte["canonical_expected_net_bps"] = later.canonical_expected_net_bps.to_numpy(float)
            xte["base_plus_consensus25"] = later.base_plus_consensus25.to_numpy(float)
            xtr["base_plus_consensus25"] = dev.base_plus_consensus25.to_numpy(float)
            grade = np.digitize(dev.meta_residual_bps.to_numpy(float), [-150.0, -50.0, 50.0, 150.0]).astype(np.int32)
            _, _, rank, _ = _fit_meta(dev, later, xtr.reindex(columns=features).fillna(0.0), xte.reindex(columns=features).fillna(0.0), grade, seed=SEED + len(features))
        out = later[["candidate_id", "__ts__", "side_name", "exact_net_bps", "exact_gross_bps", "label_valid"]].copy()
        out["arm"] = arm
        out["canonical_control"] = later.base_plus_consensus25.to_numpy(float)
        out["meta_score"] = 0.75 * out.canonical_control.to_numpy(float) + 0.25 * rank
        results.append(out)
    pred = pd.concat(results, ignore_index=True)
    metrics = []
    daily = []
    for arm, block in pred.groupby("arm", sort=True):
        for tail in (0.005, 0.01, 0.02, 0.05, 0.10):
            row = _metric(block, "meta_score", tail); row.update({"arm": arm, "scope": "global_later_long"}); metrics.append(row)
        for day, g in block.groupby(block.__ts__.dt.strftime("%Y-%m-%d"), sort=True):
            row = _metric(g, "meta_score", .05); row.update({"arm": arm, "day": day}); daily.append(row)
    stability = []
    for arm, g in pred.groupby("arm", sort=True):
        vals = [float(_metric(x, "meta_score", .05)["net_bps_per_trade"]) for _, x in g.groupby(g.__ts__.dt.strftime("%Y-%m-%d"), sort=True)]
        med = float(np.median(vals)); stability.append({"arm": arm, "periods": len(vals), "mean_top5_net_bps": float(np.mean(vals)), "median_top5_net_bps": med, "mad_top5_net_bps": float(np.median(np.abs(np.asarray(vals)-med))), "worst_period_top5_net_bps": float(np.min(vals)), "positive_periods_top5": int(np.sum(np.asarray(vals)>0))})
    output_dir.mkdir(parents=True)
    pred.to_parquet(output_dir/"predictions.parquet",index=False,compression="zstd")
    pd.DataFrame(metrics).to_parquet(output_dir/"metrics_global.parquet",index=False)
    pd.DataFrame(daily).to_parquet(output_dir/"metrics_daily.parquet",index=False)
    pd.DataFrame(stability).to_parquet(output_dir/"metrics_stability.parquet",index=False)
    correctness={"schema":"tp6_sl4_canonical_residual_meta_later_oos_correctness_v1","target_period":"2026-07-20 through 2026-07-23 UTC","dev_months":"2025-01 through 2025-12","target_outcomes_used_in_fit":False,"canonical_exits":"TP +6 ATR / SL -4 ATR / H12 / 100 bps once","canonical_context_sha256":context_hash,"later_context_sha256":later_hash,"arms":["control","B_uncertainty","H_full"],"selection":str(SELECTION),"feature_audit":feature_audit}
    (output_dir/"correctness_test_report.json").write_text(json.dumps(correctness,indent=2,default=str)+"\n")
    manifest={"schema":"tp6_sl4_canonical_residual_meta_later_oos_v1","status":"COMPLETE","target_rows":int(len(later)),"dev_rows":int(len(dev)),"target_period":"2026-07-20 through 2026-07-23 UTC","target_outcomes_used_in_fit":False,"base_contract":"canonical TP6/SL4 Base+Consensus","selection":str(SELECTION),"artifacts":["predictions.parquet","metrics_global.parquet","metrics_daily.parquet","metrics_stability.parquet","correctness_test_report.json"]}
    (output_dir/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    report="# Canonical residual meta later OOS\n\n"+pd.DataFrame(metrics).round(3).to_string(index=False)+"\n\n## Daily\n\n"+pd.DataFrame(daily).round(3).to_string(index=False)+"\n\n## Stability\n\n"+pd.DataFrame(stability).round(3).to_string(index=False)+"\n"
    (output_dir/"TP6_SL4_CANONICAL_RESIDUAL_META_LATER_OOS_REPORT.md").write_text(report)
    return output_dir


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT); args=parser.parse_args(); print(run(output_dir=args.output_dir))
