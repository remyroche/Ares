#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extreme_price_movements.config import CFG
from extreme_price_movements.offline_optimisers import compare_tbm_parameters as cmp
from extreme_price_movements.barrier_geometry import apply_horizon_scaling, make_effective_tp

EPS = 1e-12


def _real_tp_floor(cfg: Dict[str, Any]) -> float:
    return max(float(cfg.get("tp_abs_lo_pct", 0.005)), float(cfg.get("tp_min_abs_pct", 0.005)), float(cfg.get("tp_min_bps", 50)) / 10000.0)


def _tp_components(artifacts: cmp.RunArtifacts, cfg: Dict[str, Any], h: int, side: str):
    atr = artifacts.features["atr_pct"].shift(1).clip(lower=1e-6).bfill(limit=1)
    tp_regime = cmp._regime_multiplier(atr, cfg.get("tp_regime_model", "none"), cfg.get("mix_weight", 0.5))
    sl_regime = cmp._regime_multiplier(atr, cfg.get("sl_regime_model", cfg.get("tp_regime_model", "none")), cfg.get("mix_weight", 0.5))
    side_skew = float(cfg.get("tp_side_skew", 0.0))
    side_mult = (1 + side_skew) if side == "long" else (1 - side_skew)

    tp_method = cfg["tp_method"]
    if tp_method == "atr_mult":
        tp_raw = cfg["k_tp"] * side_mult * atr * tp_regime
    elif tp_method == "semi_atr_mult":
        atr_tp = cfg["k_tp"] * side_mult * atr * tp_regime
        abs_tp = pd.DataFrame(float(cfg.get("tp_abs_pct", cfg.get("tp_base_pct", 0.02))), index=atr.index, columns=atr.columns)
        tp_raw = 0.5 * atr_tp + 0.5 * abs_tp
    elif tp_method == "absolute":
        tp_raw = pd.DataFrame(float(cfg["tp_abs_pct"]), index=atr.index, columns=atr.columns)
    elif tp_method == "atr_norm":
        med = atr.rolling(int(cfg.get("base_atr_window", 168)), min_periods=24).median().fillna(atr.median())
        tp_raw = cfg["k_tp"] * (atr / (med + EPS)) * float(cfg["tp_base_pct"])
    elif tp_method == "semi_atr_norm":
        med = atr.rolling(int(cfg.get("base_atr_window", 168)), min_periods=24).median().fillna(atr.median())
        atrn_tp = cfg["k_tp"] * (atr / (med + EPS)) * float(cfg.get("tp_base_pct", 0.02))
        abs_tp = pd.DataFrame(float(cfg.get("tp_abs_pct", cfg.get("tp_base_pct", 0.02))), index=atr.index, columns=atr.columns)
        tp_raw = 0.5 * atrn_tp + 0.5 * abs_tp
    else:
        raise ValueError(tp_method)

    tp_scaled = apply_horizon_scaling(
        tp_raw,
        horizon=h,
        scaling=cfg.get("horizon_scaling", "none"),
        alpha=float(cfg.get("horizon_alpha", 0.5)),
        base=float(cfg.get("horizon_base", 4.0)),
    )
    tp_eff = make_effective_tp(
        tp_raw,
        horizon=h,
        horizon_scaling=cfg.get("horizon_scaling", "none"),
        lo=float(cfg.get("tp_abs_lo_pct", 0.005)),
        hi=float(cfg.get("tp_abs_hi_pct", 0.08)),
        horizon_alpha=float(cfg.get("horizon_alpha", 0.5)),
        horizon_base=float(cfg.get("horizon_base", 4.0)),
    )
    tp_eff = tp_eff.clip(lower=_real_tp_floor(cfg))

    if cfg["sl_method"] == "tp_pct":
        sl_eff = float(cfg["sl_as_tp_pct"]) * tp_eff
    elif cfg["sl_method"] == "atr_mult":
        sl_eff = float(cfg["k_sl"]) * atr * sl_regime
    else:
        sl_eff = pd.DataFrame(float(cfg["sl_abs_pct"]), index=atr.index, columns=atr.columns)
    sl_eff = sl_eff.clip(lower=float(cfg.get("sl_abs_lo_pct", 0.005)), upper=float(cfg.get("sl_abs_hi_pct", 0.08)))

    return atr, tp_raw, tp_scaled, tp_eff, sl_eff


def _stack_frame(df: pd.DataFrame, name: str, side: str, h: int) -> pd.DataFrame:
    s = df.stack().rename(name)
    out = s.reset_index()
    out.columns = ["ts", "symbol", name]
    out["side"] = side
    out["horizon"] = h
    return out


def analyze_config(artifacts, bucket_masks, cfg, cfg_id: str):
    layer1, layer2, eval_cache = cmp.LRUCache(max_size=2), cmp.LRUCache(max_size=2), cmp.BoundedEvalCache(max_size=4)
    summary, detail, weights = cmp.evaluate_config(
        artifacts=artifacts,
        cfg=cfg,
        horizons=[2, 4, 8],
        bucket_masks=bucket_masks,
        layer1_cache=layer1,
        layer2_cache=layer2,
        eval_cache=eval_cache,
        detailed_slices=False,
        collect_weights=True,
    )
    if weights is None or weights.empty:
        return {"config_id": cfg_id, "summary": summary, "detail": detail, "cell_table": pd.DataFrame(), "rank_penalty": None}

    comp_rows = []
    for h in [2, 4, 8]:
        for side in ["long", "short"]:
            atr, tp_raw, tp_scaled, tp_eff, sl_eff = _tp_components(artifacts, cfg, h, side)
            comp_rows.append(_stack_frame(atr, "atr_entry", side, h))
            comp_rows.append(_stack_frame(tp_raw, "tp_raw", side, h))
            comp_rows.append(_stack_frame(tp_scaled, "tp_scaled", side, h))
            comp_rows.append(_stack_frame(tp_eff, "tp_eff", side, h))
            comp_rows.append(_stack_frame(sl_eff, "sl_eff", side, h))

    from functools import reduce
    per_name = {}
    for df in comp_rows:
        col = [c for c in df.columns if c not in {"ts", "symbol", "side", "horizon"}][0]
        per_name[col] = df
    merged = reduce(lambda l, r: l.merge(r, on=["ts", "symbol", "side", "horizon"], how="inner"), [weights[["ts", "symbol", "side", "horizon", "bucket", "label", "payoff"]]] + [per_name[k] for k in ["atr_entry", "tp_raw", "tp_scaled", "tp_eff", "sl_eff"]])

    fee = float(cfg.get("fee_pct", 0.5)) / 100.0
    slip = float(cfg.get("slip_buffer", 0.1)) / 100.0
    merged["net_tp"] = merged["tp_eff"] - fee - slip
    floor = _real_tp_floor(cfg)
    ceil = float(cfg.get("tp_abs_hi_pct", 0.08))

    rows = []
    for (bucket, h), g in merged.groupby(["bucket", "horizon"], observed=True):
        tp_fb = float(np.mean(g["tp_eff"].values <= floor + 1e-9))
        ceil_bind = float(np.mean(g["tp_eff"].values >= ceil - 1e-9))
        corr = float(np.corrcoef(g["tp_eff"].values, g["atr_entry"].values)[0, 1]) if g["tp_eff"].std() > 0 and g["atr_entry"].std() > 0 else float("nan")
        rows.append({
            "cell": f"{bucket}_H{int(h)}",
            "n": len(g),
            "tp_floor_bind_prod": tp_fb,
            "tp_ceil_bind": ceil_bind,
            "tp_raw_q10": float(np.nanquantile(g["tp_raw"], 0.1)),
            "tp_raw_q50": float(np.nanquantile(g["tp_raw"], 0.5)),
            "tp_scaled_q50": float(np.nanquantile(g["tp_scaled"], 0.5)),
            "tp_eff_q50": float(np.nanquantile(g["tp_eff"], 0.5)),
            "atr_q10": float(np.nanquantile(g["atr_entry"], 0.1)),
            "atr_q50": float(np.nanquantile(g["atr_entry"], 0.5)),
            "net_tp_le_0": float(np.mean(g["net_tp"] <= 0.0)),
            "net_tp_le_20bps": float(np.mean(g["net_tp"] <= 0.002)),
            "net_tp_q10": float(np.nanquantile(g["net_tp"], 0.1)),
            "net_tp_q50": float(np.nanquantile(g["net_tp"], 0.5)),
            "corr_tp_eff_atr": corr,
            "atr_scaling_inactive": bool(np.isfinite(corr) and corr < 0.1 and tp_fb > 0.6),
        })

    cell_table = pd.DataFrame(rows).sort_values("tp_floor_bind_prod", ascending=False)
    agg = {
        "tp_floor_bind_prod_agg": float(np.mean(merged["tp_eff"] <= floor + 1e-9)),
        "max_cell_tp_floor_bind_prod": float(cell_table["tp_floor_bind_prod"].max()) if not cell_table.empty else float("nan"),
        "real_tp_floor": floor,
        "tp_lo_cfg": float(cfg.get("tp_abs_lo_pct", 0.005)),
        "tp_min_abs_pct": float(cfg.get("tp_min_abs_pct", 0.005)),
        "tp_min_bps_floor": float(cfg.get("tp_min_bps", 50))/10000.0,
    }
    return {"config_id": cfg_id, "summary": summary, "detail": detail, "cell_table": cell_table, "aggregate": agg}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=3)
    ap.add_argument("--report-csv", default="extreme_price_movements/offline_optimisers/reports/tbm_parameter_comparison.csv")
    ap.add_argument("--details-json", default="extreme_price_movements/offline_optimisers/reports/tbm_parameter_comparison.json")
    ap.add_argument("--out", default="analysis_output/tbm_floor_dominance_report.md")
    args = ap.parse_args()

    out_df = pd.read_csv(args.report_csv)
    details = json.loads(Path(args.details_json).read_text())
    out_df = out_df.sort_values(["hard_gate", "stage2_score"], ascending=[False, False])
    top = out_df.head(args.top_n)

    runtime_cfg = cmp.apply_offline_optimizer_best_params(dict(CFG))
    feats = cmp._load_features_from_data_root(runtime_cfg)
    panel = cmp._load_panel_from_store(runtime_cfg)
    if panel is None or feats is None:
        raise RuntimeError("Could not load panel/features from local data_root; provide local artifacts or ensure offline store is available.")
    artifacts = cmp.align_artifacts(panel, feats, lookback_years=2)
    bucket_masks = cmp.build_bucket_masks(artifacts, cfg_runtime=runtime_cfg)

    reports = []
    for _, r in top.iterrows():
        cid = str(r["config_id"])
        cfg = details.get(cid, {}).get("config")
        if not cfg:
            continue
        reports.append(analyze_config(artifacts, bucket_masks, cfg, cid))

    lines = ["# TBM Floor-Dominance Diagnostics", "", "## Static callsite checks"]
    lines.append("- `make_effective_tp` callsites: compare `build_barriers`, training `compute_barrier_factory` (raw → scale → clip).")
    lines.append("- `sl_method == tp_pct` derives SL from effective TP in compare path.")
    lines.append("")

    for rep in reports:
        lines.append(f"## Config {rep['config_id']}")
        agg = rep["aggregate"]
        lines.append(f"- tp_floor_bind_prod_agg={agg['tp_floor_bind_prod_agg']:.3f}, max_cell_tp_floor_bind_prod={agg['max_cell_tp_floor_bind_prod']:.3f}")
        lines.append(f"- real_tp_floor=max(tp_abs_lo_pct={agg['tp_lo_cfg']:.4f}, tp_min_abs_pct={agg['tp_min_abs_pct']:.4f}, tp_min_bps_floor={agg['tp_min_bps_floor']:.4f})={agg['real_tp_floor']:.4f}")
        lines.append("")
        ct = rep["cell_table"].copy()
        lines.append("Worst cells by tp_floor_bind_prod:")
        lines.append(ct[["cell", "n", "tp_floor_bind_prod", "tp_ceil_bind", "tp_raw_q50", "tp_scaled_q50", "tp_eff_q50", "atr_q50", "net_tp_le_0", "net_tp_le_20bps", "corr_tp_eff_atr", "atr_scaling_inactive"]].to_markdown(index=False))
        lines.append("")

    # ranking what-if from existing df without rerun
    if "stage2_score" in out_df.columns:
        det = details
        penalties = []
        for _, r in out_df.iterrows():
            cid = str(r["config_id"])
            pa = det.get(cid, {}).get("production_admissibility", {}).get("aggregates", {})
            agg_bind = pa.get("tp_floor_bind_prod_agg", np.nan)
            max_bind = pa.get("max_cell_tp_floor_bind_prod", np.nan)
            pen = 0.0
            if np.isfinite(agg_bind):
                pen += 0.25 * float(agg_bind)
            if np.isfinite(max_bind):
                pen += 0.25 * float(max_bind)
            penalties.append(pen)
        if penalties:
            what = out_df[["config_id", "stage2_score"]].copy()
            what["floor_penalty"] = penalties
            what["stage2_score_floor_adj"] = what["stage2_score"] - what["floor_penalty"]
            what["rank_old"] = what["stage2_score"].rank(ascending=False, method="min")
            what["rank_new"] = what["stage2_score_floor_adj"].rank(ascending=False, method="min")
            what = what.sort_values("rank_new").head(15)
            lines.append("## Ranking change (no re-run; from existing results dataframe)")
            lines.append(what.to_markdown(index=False))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
