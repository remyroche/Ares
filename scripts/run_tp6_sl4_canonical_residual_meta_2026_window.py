#!/usr/bin/env python3
"""Frozen-2025 residual meta validation on the longer 2026 Jan--Jul 10 window."""
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

DEV_HEADS = DEFAULT_HEADS
EVAL_HEADS = ROOT / "data_perp/artifacts/tp6_sl4_canonical_head_health_2026_v1/canonical_head_health_2026.parquet"
SELECTION = ROOT / "data_perp/artifacts/tp6_sl4_canonical_residual_feature_selection_20260808_v1/selected_features.json"
DEFAULT_OUT = ROOT / "data_perp/artifacts/tp6_sl4_canonical_residual_meta_2026_window_20260808_v1"
SEED = 20260808


def _metric(frame: pd.DataFrame, score: str, tail: float) -> dict[str, object]:
    n = max(1, int(math.ceil(len(frame) * tail)))
    top = frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
    return {"tail": tail, "trades": int(len(top)), "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": float(top.exact_net_bps.mean()), "rank_ic": float(frame[[score, "exact_net_bps"]].corr(method="spearman").iloc[0, 1])}


def run(*, output_dir: Path = DEFAULT_OUT) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    dev, context, context_hash = _load()
    dev = dev.merge(pd.read_parquet(DEV_HEADS), on="candidate_id", how="inner", validate="one_to_one", suffixes=("", "_head"))
    eval26 = pd.read_parquet(EVAL_HEADS)
    selection = json.loads(SELECTION.read_text())["selected_features"]
    # Fit each side once on 2025 only.  2026 outcomes are never read before
    # final scoring, so the window is genuinely untouched by this layer.
    outputs: list[pd.DataFrame] = []
    fit_audit: list[dict[str, object]] = []
    for side in ("long", "short"):
        tr = dev.loc[dev.side_name.eq(side)].copy()
        te = eval26.loc[eval26.side_name.eq(side)].copy()
        if len(tr) < 500 or te.empty:
            continue
        tr["canonical_expected_net_bps"], _ = _map_canonical(tr, tr)
        residual = tr.exact_net_bps.to_numpy(float) - tr.canonical_expected_net_bps.to_numpy(float)
        tr["meta_residual_bps"] = residual
        te["canonical_expected_net_bps"] = np.nan
        # Fit the causal score-to-bps map on 2025 only, then apply it to all
        # 2026 target months.
        mapper = IsotonicRegression(out_of_bounds="clip", y_min=-1000.0, y_max=1000.0)
        mapper.fit(tr.base_plus_consensus25, tr.exact_net_bps)
        te["canonical_expected_net_bps"] = mapper.predict(te.base_plus_consensus25).astype(np.float32)
        xtr, xte, fa = _feature_frame(tr, te, context)
        xtr["canonical_expected_net_bps"] = tr.canonical_expected_net_bps.to_numpy(float)
        xte["canonical_expected_net_bps"] = te.canonical_expected_net_bps.to_numpy(float)
        xtr["base_plus_consensus25"] = tr.base_plus_consensus25.to_numpy(float)
        xte["base_plus_consensus25"] = te.base_plus_consensus25.to_numpy(float)
        fit_audit.append({"side": side, "train_rows": int(len(tr)), "eval_rows": int(len(te)), **fa})
        arms = {"control": (), "B_uncertainty": ("uncertainty",), "H_full": ("uncertainty", "support_ood", "drift", "market_state")}
        for arm, block_names in arms.items():
            if arm == "control":
                rank = np.full(len(te), .5, dtype=np.float32)
            else:
                features = [c for b in block_names for c in selection.get(b, BLOCKS[b]) if c not in {"canonical_expected_net_bps", "base_plus_consensus25"}]
                features = list(dict.fromkeys(["canonical_expected_net_bps", "base_plus_consensus25", *features]))
                grade = np.digitize(residual, [-150.0, -50.0, 50.0, 150.0]).astype(np.int32)
                _, _, rank, _ = _fit_meta(tr, te, xtr.reindex(columns=features).fillna(0.0), xte.reindex(columns=features).fillna(0.0), grade, seed=SEED + len(features))
            out = te[["candidate_id", "__ts__", "month", "side_name", "exact_net_bps", "exact_gross_bps"]].copy()
            out["arm"] = arm
            out["canonical_control"] = te.base_plus_consensus25.to_numpy(float)
            out["meta_score"] = 0.75 * out.canonical_control.to_numpy(float) + 0.25 * rank
            out["meta_rank"] = rank
            outputs.append(out)
    pred = pd.concat(outputs, ignore_index=True)
    metrics: list[dict[str, object]] = []
    monthly: list[dict[str, object]] = []
    side_metrics: list[dict[str, object]] = []
    for arm, block in pred.groupby("arm", sort=True):
        for tail in (0.005, 0.01, 0.02, 0.05, 0.10):
            row = _metric(block, "meta_score", tail); row.update({"arm": arm, "scope": "global_2026_jan_jul10"}); metrics.append(row)
        for month, m in block.groupby("month", sort=True):
            row = _metric(m, "meta_score", .05); row.update({"arm": arm, "month": month, "scope": "monthly_2026"}); monthly.append(row)
        for side, s in block.groupby("side_name", sort=True):
            row = _metric(s, "meta_score", .05); row.update({"arm": arm, "side": side, "scope": "side_2026"}); side_metrics.append(row)
    stability=[]
    for arm, g in pred.groupby("arm", sort=True):
        vals=np.asarray([float(_metric(x,"meta_score",.05)["net_bps_per_trade"]) for _,x in g.groupby("month",sort=True)],float); med=float(np.median(vals)); stability.append({"arm":arm,"periods":len(vals),"mean_top5_net_bps":float(vals.mean()),"median_top5_net_bps":med,"mad_top5_net_bps":float(np.median(np.abs(vals-med))),"worst_period_top5_net_bps":float(vals.min()),"positive_periods_top5":int((vals>0).sum())})
    output_dir.mkdir(parents=True)
    pred.to_parquet(output_dir/"predictions.parquet",index=False,compression="zstd")
    pd.DataFrame(metrics).to_parquet(output_dir/"metrics_global.parquet",index=False)
    pd.DataFrame(monthly).to_parquet(output_dir/"metrics_monthly.parquet",index=False)
    pd.DataFrame(side_metrics).to_parquet(output_dir/"metrics_side.parquet",index=False)
    pd.DataFrame(stability).to_parquet(output_dir/"metrics_stability.parquet",index=False)
    pd.DataFrame(fit_audit).to_parquet(output_dir/"fit_audit.parquet",index=False)
    correctness={"schema":"tp6_sl4_canonical_residual_meta_2026_window_correctness_v1","dev_period":"2025-01 through 2025-12","evaluation_period":"2026-01 through 2026-07-10","target_rows":int(pred.candidate_id.nunique()),"target_outcomes_used_in_fit":False,"canonical_exits":"TP +6 ATR / SL -4 ATR / H12 / 100 bps once","canonical_context_sha256":context_hash,"selection":str(SELECTION),"arms":["control","B_uncertainty","H_full"]}
    (output_dir/"correctness_test_report.json").write_text(json.dumps(correctness,indent=2)+"\n")
    manifest={"schema":"tp6_sl4_canonical_residual_meta_2026_window_v1","status":"COMPLETE","dev_period":"2025-01 through 2025-12","evaluation_period":"2026-01 through 2026-07-10","target_rows":int(pred.candidate_id.nunique()),"target_outcomes_used_in_fit":False,"base_contract":"canonical TP6/SL4 Base+Consensus","selection":str(SELECTION),"artifacts":["predictions.parquet","metrics_global.parquet","metrics_monthly.parquet","metrics_side.parquet","metrics_stability.parquet","fit_audit.parquet","correctness_test_report.json"]}
    (output_dir/"run_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    report="# Canonical residual meta frozen-2025 validation\n\n"+pd.DataFrame(metrics).round(3).to_string(index=False)+"\n\n## Monthly\n\n"+pd.DataFrame(monthly).round(3).to_string(index=False)+"\n\n## Side\n\n"+pd.DataFrame(side_metrics).round(3).to_string(index=False)+"\n\n## Stability\n\n"+pd.DataFrame(stability).round(3).to_string(index=False)+"\n"
    (output_dir/"TP6_SL4_CANONICAL_RESIDUAL_META_2026_WINDOW_REPORT.md").write_text(report)
    return output_dir


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=DEFAULT_OUT); args=parser.parse_args(); print(run(output_dir=args.output_dir))
