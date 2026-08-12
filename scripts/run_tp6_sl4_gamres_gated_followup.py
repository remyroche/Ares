#!/usr/bin/env python3
"""Follow-up decomposition and hard-gated one-month GAM replay."""
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
    _fill_gam_history,
    _fit_heads,
    _group,
    _join_gam,
    _rank_ic,
)


DEFAULT_ROLLING = ROOT / "data_perp/artifacts/tp6_sl4_rolling_archetype_gam_oos_20260815_v5/rolling_oof_predictions.parquet"
DEFAULT_CURRENT = ROOT / "data_perp/artifacts/tp6_sl4_downstream_retrain_2025_20260807_v1/predictions_2025.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/tp6_sl4_gamres_gated_followup_20260815_v1"
SEED = 20260815
SIDE = "long"
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)
FIELD_GROUPS = {
    "control": [],
    "gam_delta_only": ["gam_delta_bps"],
    "gam_residual_only": ["gam_residual_bps"],
    "gam_delta_residual": ["gam_delta_bps", "gam_residual_bps"],
    "gam_delta_residual_valid": ["gam_delta_bps", "gam_residual_bps", "gam_transport_valid"],
}


def _metric(block: pd.DataFrame, score: str, tail: float) -> dict[str, object]:
    n = max(1, int(math.ceil(len(block) * tail)))
    top = block.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
    return {"tail": tail, "trades": n, "gross_bps_per_trade": float(top.exact_gross_bps.mean()), "net_bps_per_trade": float(top.exact_net_bps.mean()), "rank_ic": _rank_ic(block[score].to_numpy(float), block.exact_net_bps.to_numpy(float))}


def _metrics(frame: pd.DataFrame, scores: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    glob, monthly, stability = [], [], []
    for score in scores:
        for tail in TAILS:
            row = _metric(frame, score, tail); row.update({"arm": score, "scope": "global_long_2025"}); glob.append(row)
        vals, ics = [], []
        for month, block in frame.groupby("month", sort=True):
            for tail in (0.01, 0.05):
                row = _metric(block, score, tail); row.update({"arm": score, "month": month}); monthly.append(row)
            vals.append(_metric(block, score, 0.05)["net_bps_per_trade"]); ics.append(_rank_ic(block[score].to_numpy(float), block.exact_net_bps.to_numpy(float)))
        arr = np.asarray(vals, float); med = float(np.median(arr)); mad = float(np.median(np.abs(arr - med)))
        stability.append({"arm": score, "months": len(arr), "mean_top5_net_bps": float(np.mean(arr)), "median_top5_net_bps": med, "mad_top5_net_bps": mad, "worst_month_top5_net_bps": float(np.min(arr)), "positive_months_top5": int(np.sum(arr > 0)), "mean_month_rank_ic": float(np.nanmean(ics))})
    return pd.DataFrame(glob), pd.DataFrame(monthly), pd.DataFrame(stability)


def _valid_invalid(frame: pd.DataFrame, scores: list[str]) -> pd.DataFrame:
    rows = []
    for scope, subset in (("valid_months", frame.loc[frame.gam_month_valid.eq(1)]), ("invalid_months", frame.loc[frame.gam_month_valid.eq(0)])):
        for score in scores:
            for tail in (0.01, 0.05):
                row = _metric(subset, score, tail); row.update({"scope": scope, "arm": score, "months": int(subset.month.nunique())}); rows.append(row)
    return pd.DataFrame(rows)


def _loo(frame: pd.DataFrame, scores: list[str]) -> pd.DataFrame:
    rows = []
    for held_out in sorted(frame.month.unique()):
        block = frame.loc[frame.month.ne(held_out)]
        for score in scores:
            for tail in (0.01, 0.05):
                row = _metric(block, score, tail); row.update({"held_out_month": held_out, "arm": score}); rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap(frame: pd.DataFrame, score_a: str, score_b: str, n_boot: int = 10000) -> pd.DataFrame:
    monthly = []
    for month, block in frame.groupby("month", sort=True):
        a = _metric(block, score_a, 0.01)["net_bps_per_trade"] - _metric(block, score_b, 0.01)["net_bps_per_trade"]
        b = _metric(block, score_a, 0.05)["net_bps_per_trade"] - _metric(block, score_b, 0.05)["net_bps_per_trade"]
        monthly.append({"month": month, "delta_top1": a, "delta_top5": b})
    m = pd.DataFrame(monthly); rng = np.random.default_rng(SEED); values = []
    for tail in ("delta_top1", "delta_top5"):
        x = m[tail].to_numpy(float); draws = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
        values.append({"comparison": f"{score_a}_minus_{score_b}", "metric": tail, "months": len(x), "mean_month_delta": float(x.mean()), "median_month_delta": float(np.median(x)), "ci025": float(np.quantile(draws, 0.025)), "ci975": float(np.quantile(draws, 0.975)), "positive_months": int(np.sum(x > 0))})
    return pd.DataFrame(values)


def _transitions(frame: pd.DataFrame, score: str, control: str = "control") -> pd.DataFrame:
    n = max(1, int(math.ceil(len(frame) * 0.01)))
    cids = set(frame.sort_values([control, "candidate_id"], ascending=[False, True]).head(n).candidate_id)
    sids = set(frame.sort_values([score, "candidate_id"], ascending=[False, True]).head(n).candidate_id)
    x = frame.copy(); x["transition"] = np.where(x.candidate_id.isin(sids) & ~x.candidate_id.isin(cids), "entered_top1", np.where(x.candidate_id.isin(cids) & ~x.candidate_id.isin(sids), "exited_top1", np.where(x.candidate_id.isin(cids) & x.candidate_id.isin(sids), "stayed_top1", "outside_top1")))
    out = x.groupby("transition", sort=True).agg(rows=("candidate_id", "size"), mean_net_bps=("exact_net_bps", "mean"), median_net_bps=("exact_net_bps", "median"), valid_fraction=("gam_month_valid", "mean"), mean_gam_delta=("gam_delta_bps", "mean"), mean_gam_residual=("gam_residual_bps", "mean")).reset_index()
    out["arm"] = score
    return out


def run(*, rolling_path: Path = DEFAULT_ROLLING, current_path: Path = DEFAULT_CURRENT, output_dir: Path = DEFAULT_OUTPUT) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    x, context, context_hash = _load(); x = _join_gam(x.loc[x.side_name.eq(SIDE)].copy(), rolling_path)
    parts, audits, corrs = [], [], []
    for month in MONTHS:
        held = x.loc[x.month.astype(str).eq(month)].copy(); train = x.loc[(x.__ts__ < pd.Timestamp(month, tz="UTC")) & (x.label_available_ts < pd.Timestamp(month, tz="UTC"))].copy()
        if held.empty or len(train) < 300:
            continue
        base_train, base_held = _map_base(train, held); _fill_gam_history(train, base_train); _fill_gam_history(held, base_held)
        held["gam_month_valid"] = int(bool(held.gam_transport_valid.mean() > 0.5)); held["base_expected_bps"] = np.asarray(base_held, dtype=np.float32)
        for period, block in (("train", train), ("held", held)):
            corrs.append({"month": month, "period": period, "rows": len(block), "spearman_delta_residual": float(block.gam_delta_bps.corr(block.gam_residual_bps, method="spearman")), "pearson_delta_residual": float(block.gam_delta_bps.corr(block.gam_residual_bps, method="pearson")), "delta_std": float(block.gam_delta_bps.std()), "residual_std": float(block.gam_residual_bps.std())})
        for arm, fields in FIELD_GROUPS.items():
            tr_model, te_model = train.copy(), held.copy(); tr_model.attrs["context_fields"] = context; te_model.attrs["context_fields"] = context
            consensus, residual_rank, _, _ = _fit_heads(tr_model, te_model, base_train, base_held, use_gam_inputs=False, extra_fields=fields, feature_fraction=1.0, month=month)
            base_rank = _pct(held.base_score.to_numpy(float), train.base_score.to_numpy(float)); stack = (0.50 * base_rank + 0.25 * consensus + 0.25 * residual_rank).astype(np.float32)
            out = held[["candidate_id", "__ts__", "month", "exact_net_bps", "exact_gross_bps", "base_score", "base_expected_bps", "gam_delta_bps", "gam_residual_bps", "gam_transport_valid", "gam_matched_mass", "gam_unmatched_mass", "gam_archetype_count", "gam_cluster_count", "gam_month_valid"]].copy()
            out["arm"] = arm; out["stack_score"] = stack; out["consensus_rank"] = consensus; out["residual_rank"] = residual_rank; out["feature_fields"] = json.dumps(fields)
            parts.append(out); audits.append({"month": month, "arm": arm, "fields": fields, "feature_fraction": 1.0, "train_rows": len(train), "held_rows": len(held), "query_groups": int(_group(train)[1].size)})
    pred = pd.concat(parts, ignore_index=True); pred["stack_score"] = pred.groupby(["arm", "month"], sort=False)["stack_score"].transform(lambda z: z.rank(pct=True, method="average")).astype("float32")
    pivot = pred.pivot(index=["candidate_id", "month"], columns="arm", values="stack_score").reset_index(); meta = pred.loc[pred.arm.eq("control"), ["candidate_id", "month", "exact_net_bps", "exact_gross_bps", "base_expected_bps", "gam_delta_bps", "gam_residual_bps", "gam_month_valid"]].drop_duplicates(["candidate_id", "month"]); frame = meta.merge(pivot, on=["candidate_id", "month"], validate="one_to_one")
    scores = ["control", "gam_delta_only", "gam_residual_only", "gam_delta_residual", "gam_delta_residual_valid"]
    # Hard gate the GAM-residual specialist: valid target month uses the
    # enhanced head, invalid target month is exactly the control score.
    frame["gated_gam_residual"] = np.where(frame.gam_month_valid.eq(1), frame.gam_delta_residual, frame.control)
    current = pd.read_parquet(current_path); current = current.loc[current.side_name.eq(SIDE) & current.month.astype(str).isin(MONTHS), ["candidate_id", "month", "full_base_consensus_residual"]].rename(columns={"full_base_consensus_residual": "current_stack"})
    frame = frame.merge(current, on=["candidate_id", "month"], how="left", validate="one_to_one")
    metrics_global, metrics_monthly, stability = _metrics(frame, [*scores, "gated_gam_residual", "current_stack"])
    split = _valid_invalid(frame, ["control", "gam_residual_only", "gated_gam_residual", "current_stack"])
    loo = _loo(frame, ["control", "gated_gam_residual", "current_stack"])
    bootstrap = _bootstrap(frame, "gated_gam_residual", "control")
    transition_parts = [_transitions(frame, s) for s in ("gated_gam_residual", "gam_delta_residual")]
    transitions = pd.concat(transition_parts, ignore_index=True)
    output_dir.mkdir(parents=True); pred.to_parquet(output_dir / "predictions_decomposition.parquet", index=False, compression="zstd"); frame.to_parquet(output_dir / "predictions_gated.parquet", index=False, compression="zstd"); metrics_global.to_parquet(output_dir / "metrics_global.parquet", index=False); metrics_monthly.to_parquet(output_dir / "metrics_monthly.parquet", index=False); stability.to_parquet(output_dir / "metrics_stability.parquet", index=False); split.to_parquet(output_dir / "metrics_valid_invalid.parquet", index=False); loo.to_parquet(output_dir / "metrics_leave_one_month_out.parquet", index=False); bootstrap.to_parquet(output_dir / "metrics_bootstrap_ci.parquet", index=False); transitions.to_parquet(output_dir / "top1_transitions.parquet", index=False); pd.DataFrame(corrs).to_parquet(output_dir / "gam_field_correlations.parquet", index=False); pd.DataFrame(audits).to_parquet(output_dir / "fit_audit.parquet", index=False)
    manifest = {"schema": "tp6_sl4_gamres_gated_followup_v1", "status": "COMPLETE", "side": SIDE, "held_months": list(MONTHS), "feature_fraction": 1.0, "field_arms": FIELD_GROUPS, "hard_gate": "gated_gam_residual = gam_delta_residual when transport-valid target month, else exact control", "current_stack": str(current_path), "bootstrap_resamples": 10000, "no_held_outcomes_in_gam": True, "context_sha256": context_hash, "artifacts": sorted(p.name for p in output_dir.iterdir())}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    lines = ["# GAM residual decomposition and hard transport gate", "", "## Global metrics", "", metrics_global.round(3).to_string(index=False), "", "## Valid/invalid", "", split.round(3).to_string(index=False), "", "## Leave-one-month-out", "", loo.round(3).to_string(index=False), "", "## Bootstrap", "", bootstrap.round(3).to_string(index=False), "", "## Transitions", "", transitions.round(3).to_string(index=False)]
    (output_dir / "TP6_SL4_GAMRES_GATED_FOLLOWUP_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output_dir), "rows": len(frame), "metric_rows": len(metrics_global)}, indent=2)); return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--rolling", type=Path, default=DEFAULT_ROLLING); parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); args = parser.parse_args(); run(rolling_path=args.rolling, current_path=args.current, output_dir=args.output_dir)
