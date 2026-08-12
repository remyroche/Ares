#!/usr/bin/env python3
"""Attribute one-month GAM residual/meta uplift with feature-fraction=1.

This is a diagnostic replay, not a new production stack.  All arms use the
same base anchor and existing consensus/residual LambdaRank targets.  The
only ablated inputs are the one-month GAM/transport fields; direct GAM anchor
modulation is intentionally excluded.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_downstream_retrain_2025 import MONTHS, _load, _map_base, _pct
from scripts.run_tp6_sl4_rolling_gam_residual_integration import (
    GAM_INPUT_FIELDS,
    _fill_gam_history,
    _fit_heads,
    _group,
    _join_gam,
    _rank_ic,
)


DEFAULT_ROLLING = ROOT / "data_perp/artifacts/tp6_sl4_rolling_archetype_gam_oos_20260815_v5/rolling_oof_predictions.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/tp6_sl4_rolling_gam_attribution_ff1_20260815_v1"
SEED = 20260815
SIDE = "long"
DENSE_TAILS = (0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10)
MONTHLY_TAILS = (0.005, 0.01, 0.02, 0.05)

FIELD_GROUPS: dict[str, list[str]] = {
    "control": [],
    "gam_residual_only": ["gam_delta_bps", "gam_residual_bps"],
    "transport_only": ["gam_transport_valid", "gam_matched_mass", "gam_unmatched_mass", "gam_archetype_count", "gam_cluster_count"],
    "gam_plus_valid": ["gam_delta_bps", "gam_residual_bps", "gam_transport_valid"],
    "gam_plus_transport": ["gam_delta_bps", "gam_residual_bps", "gam_transport_valid", "gam_matched_mass", "gam_unmatched_mass", "gam_archetype_count", "gam_cluster_count"],
    "full_current": list(GAM_INPUT_FIELDS),
    "placebo_gam_plus_transport": ["gam_delta_bps", "gam_residual_bps", "gam_transport_valid", "gam_matched_mass", "gam_unmatched_mass", "gam_archetype_count", "gam_cluster_count"],
}


def _permute_within_month(train: pd.DataFrame, held: pd.DataFrame, fields: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Destroy row-level signal while preserving per-month values/missingness."""
    tr, te = train.copy(), held.copy()
    rng = np.random.default_rng(seed)
    for frame in (tr, te):
        for month, idx in frame.groupby("month", sort=False).groups.items():
            positions = np.asarray(idx)
            for field in fields:
                values = frame.loc[positions, field].to_numpy(copy=True)
                frame.loc[positions, field] = values[rng.permutation(len(values))]
    return tr, te


def _dense_metrics(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    glob, monthly, stability = [], [], []
    for arm, block in pred.groupby("arm", sort=True):
        for tail in DENSE_TAILS:
            n = max(1, int(math.ceil(len(block) * tail)))
            top = block.sort_values(["stack_score", "candidate_id"], ascending=[False, True], kind="stable").head(n)
            glob.append({"arm": arm, "tail": tail, "trades": n, "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": float(top.exact_net_bps.mean()), "rank_ic": _rank_ic(block.stack_score.to_numpy(float), block.exact_net_bps.to_numpy(float))})
        vals, ics = [], []
        for month, month_block in block.groupby("month", sort=True):
            for tail in MONTHLY_TAILS:
                n = max(1, int(math.ceil(len(month_block) * tail)))
                top = month_block.sort_values(["stack_score", "candidate_id"], ascending=[False, True], kind="stable").head(n)
                monthly.append({"arm": arm, "month": month, "tail": tail, "trades": n, "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": float(top.exact_net_bps.mean()), "rank_ic": _rank_ic(month_block.stack_score.to_numpy(float), month_block.exact_net_bps.to_numpy(float))})
            top = month_block.sort_values(["stack_score", "candidate_id"], ascending=[False, True], kind="stable").head(max(1, int(math.ceil(len(month_block) * 0.05))))
            vals.append(float(top.exact_net_bps.mean())); ics.append(_rank_ic(month_block.stack_score.to_numpy(float), month_block.exact_net_bps.to_numpy(float)))
        arr = np.asarray(vals, float); med = float(np.median(arr)); mad = float(np.median(np.abs(arr - med)))
        stability.append({"arm": arm, "months": len(arr), "mean_top5_net_bps": float(np.mean(arr)), "median_top5_net_bps": med, "mad_top5_net_bps": mad, "worst_month_top5_net_bps": float(np.min(arr)), "positive_months_top5": int(np.sum(arr > 0)), "mean_month_rank_ic": float(np.nanmean(ics))})
    return pd.DataFrame(glob), pd.DataFrame(monthly), pd.DataFrame(stability)


def _split_metrics(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pred = pred.copy()
    for scope, months in (("valid_months", pred.loc[pred.gam_month_valid.eq(1), "month"].unique()), ("invalid_months", pred.loc[pred.gam_month_valid.eq(0), "month"].unique())):
        block_all = pred.loc[pred.month.isin(months)].copy()
        for arm, block in block_all.groupby("arm", sort=True):
            for tail in (0.005, 0.01, 0.02, 0.05):
                n = max(1, int(math.ceil(len(block) * tail)))
                top = block.sort_values(["stack_score", "candidate_id"], ascending=[False, True], kind="stable").head(n)
                rows.append({"scope": scope, "arm": arm, "months": len(months), "tail": tail, "trades": n, "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": float(top.exact_net_bps.mean()), "rank_ic": _rank_ic(block.stack_score.to_numpy(float), block.exact_net_bps.to_numpy(float))})
    return pd.DataFrame(rows)


def _transition(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    control = pred.loc[pred.arm.eq("control"), ["candidate_id", "month", "stack_score", "exact_net_bps", "exact_gross_bps", "base_expected_bps", "gam_delta_bps", "gam_residual_bps", "gam_transport_valid", "gam_matched_mass", "gam_unmatched_mass", "gam_archetype_count", "gam_cluster_count", "consensus_rank", "residual_rank"]].rename(columns={"stack_score": "control_score", "consensus_rank": "control_consensus", "residual_rank": "control_residual"})
    full = pred.loc[pred.arm.eq("full_current"), ["candidate_id", "month", "stack_score", "consensus_rank", "residual_rank"]].rename(columns={"stack_score": "full_score", "consensus_rank": "full_consensus", "residual_rank": "full_residual"})
    m = control.merge(full, on=["candidate_id", "month"], validate="one_to_one")
    n = max(1, int(math.ceil(len(m) * 0.01)))
    c_ids = set(m.sort_values(["control_score", "candidate_id"], ascending=[False, True]).head(n).candidate_id)
    f_ids = set(m.sort_values(["full_score", "candidate_id"], ascending=[False, True]).head(n).candidate_id)
    m["transition"] = np.where(m.candidate_id.isin(f_ids) & ~m.candidate_id.isin(c_ids), "entered_top1", np.where(m.candidate_id.isin(c_ids) & ~m.candidate_id.isin(f_ids), "exited_top1", np.where(m.candidate_id.isin(c_ids) & m.candidate_id.isin(f_ids), "stayed_top1", "outside_top1")))
    summary = m.groupby("transition", sort=True).agg(rows=("candidate_id", "size"), mean_net_bps=("exact_net_bps", "mean"), median_net_bps=("exact_net_bps", "median"), mean_base_expected_bps=("base_expected_bps", "mean"), mean_gam_delta_bps=("gam_delta_bps", "mean"), mean_gam_residual_bps=("gam_residual_bps", "mean"), valid_fraction=("gam_transport_valid", "mean"), mean_matched_mass=("gam_matched_mass", "mean"), mean_unmatched_mass=("gam_unmatched_mass", "mean"), mean_archetype_count=("gam_archetype_count", "mean"), mean_cluster_count=("gam_cluster_count", "mean"), mean_control_score=("control_score", "mean"), mean_full_score=("full_score", "mean")).reset_index()
    monthly = m.loc[m.transition.isin(["entered_top1", "exited_top1"])].groupby(["month", "transition"], sort=True).agg(rows=("candidate_id", "size"), mean_net_bps=("exact_net_bps", "mean"), mean_gam_delta_bps=("gam_delta_bps", "mean"), valid_fraction=("gam_transport_valid", "mean")).reset_index()
    return summary, monthly


def run(*, rolling_path: Path = DEFAULT_ROLLING, output_dir: Path = DEFAULT_OUTPUT) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    x, context, context_hash = _load()
    x = _join_gam(x.loc[x.side_name.eq(SIDE)].copy(), rolling_path)
    parts, audits = [], []
    for month in MONTHS:
        held = x.loc[x.month.astype(str).eq(month)].copy()
        train = x.loc[(x.__ts__ < pd.Timestamp(month, tz="UTC")) & (x.label_available_ts < pd.Timestamp(month, tz="UTC"))].copy()
        if held.empty or len(train) < 300:
            continue
        base_train, base_held = _map_base(train, held)
        _fill_gam_history(train, base_train); _fill_gam_history(held, base_held)
        held["gam_month_valid"] = int(bool(held.gam_transport_valid.mean() > 0.5))
        for arm, fields in FIELD_GROUPS.items():
            tr_model, te_model = train.copy(), held.copy()
            if arm.startswith("placebo"):
                tr_model, te_model = _permute_within_month(tr_model, te_model, fields, SEED + int(month[-2:]) * 100 + len(fields))
            tr_model.attrs["context_fields"] = context; te_model.attrs["context_fields"] = context
            consensus, residual_rank, _, _ = _fit_heads(tr_model, te_model, base_train, base_held, use_gam_inputs=False, extra_fields=fields, feature_fraction=1.0, month=month)
            base_rank = _pct(held.base_score.to_numpy(float), train.base_score.to_numpy(float))
            stack = (0.50 * base_rank + 0.25 * consensus + 0.25 * residual_rank).astype(np.float32)
            held["base_expected_bps"] = np.asarray(base_held, dtype=np.float32)
            out = held[["candidate_id", "__ts__", "month", "exact_net_bps", "exact_gross_bps", "base_score", "base_expected_bps", "gam_expected_bps", "gam_delta_bps", "gam_residual_bps", "gam_transport_valid", "gam_matched_mass", "gam_unmatched_mass", "gam_archetype_count", "gam_cluster_count", "gam_month_valid"]].copy()
            out["arm"] = arm; out["anchor_rank"] = base_rank; out["consensus_rank"] = consensus; out["residual_rank"] = residual_rank; out["stack_rank"] = stack; out["stack_score"] = stack
            parts.append(out)
            audits.append({"month": month, "arm": arm, "feature_fraction": 1.0, "fields": fields, "train_rows": len(train), "held_rows": len(held), "query_groups": int(_group(train)[1].size), "gam_valid_train_fraction": float(train.gam_transport_valid.mean()), "gam_valid_held_fraction": float(held.gam_transport_valid.mean()), "placebo": arm.startswith("placebo")})
    pred = pd.concat(parts, ignore_index=True)
    pred["stack_score"] = pred.groupby(["arm", "month"], sort=False)["stack_rank"].transform(lambda z: z.rank(pct=True, method="average")).astype("float32")
    glob, monthly, stability = _dense_metrics(pred)
    split = _split_metrics(pred)
    transition_summary, transition_monthly = _transition(pred)
    output_dir.mkdir(parents=True)
    pred.to_parquet(output_dir / "predictions.parquet", index=False, compression="zstd")
    glob.to_parquet(output_dir / "metrics_global_dense.parquet", index=False); monthly.to_parquet(output_dir / "metrics_monthly_dense.parquet", index=False); stability.to_parquet(output_dir / "metrics_stability.parquet", index=False); split.to_parquet(output_dir / "metrics_valid_invalid.parquet", index=False); transition_summary.to_parquet(output_dir / "top1_transition_summary.parquet", index=False); transition_monthly.to_parquet(output_dir / "top1_transition_monthly.parquet", index=False); pd.DataFrame(audits).to_parquet(output_dir / "fit_audit.parquet", index=False)
    manifest = {"schema": "tp6_sl4_rolling_gam_attribution_ff1_v1", "status": "COMPLETE", "side": SIDE, "held_months": list(MONTHS), "feature_fraction": 1.0, "arms": FIELD_GROUPS, "rolling_gam": str(rolling_path), "direct_modulation": False, "placebo": "fields permuted independently within training and held months", "no_held_outcomes_in_gam": True, "context_sha256": context_hash, "artifacts": sorted(p.name for p in output_dir.iterdir())}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    lines = ["# TP6/SL4 one-month GAM attribution (feature_fraction=1.0)", "", "## Dense global tails", "", glob.round(3).to_string(index=False), "", "## Valid versus invalid target months", "", split.round(3).to_string(index=False), "", "## Top-1 transitions", "", transition_summary.round(3).to_string(index=False)]
    (output_dir / "TP6_SL4_ROLLING_GAM_ATTRIBUTION_FF1_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output_dir), "rows": len(pred), "metric_rows": len(glob)}, indent=2))
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling", type=Path, default=DEFAULT_ROLLING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(rolling_path=args.rolling, output_dir=args.output_dir)
