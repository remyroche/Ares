#!/usr/bin/env python3
"""Repeated within-month placebo distribution for GAM residual fields."""
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
    _join_gam,
    _rank_fit,
)


DEFAULT_ROLLING = ROOT / "data_perp/artifacts/tp6_sl4_rolling_archetype_gam_oos_20260815_v5/rolling_oof_predictions.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/tp6_sl4_gamres_placebo_distribution_20260815_v1"
SEED = 20260815
N_SEEDS = 50
SIDE = "long"


def _permute(frame: pd.DataFrame, fields: list[str], seed: int) -> pd.DataFrame:
    out = frame.copy(); rng = np.random.default_rng(seed)
    for _, idx in out.groupby("month", sort=False).groups.items():
        positions = np.asarray(idx)
        for field in fields:
            values = out.loc[positions, field].to_numpy(copy=True)
            out.loc[positions, field] = values[rng.permutation(len(values))]
    return out


def _top(frame: pd.DataFrame, score: str, tail: float) -> float:
    n = max(1, int(math.ceil(len(frame) * tail)))
    return float(frame.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n).exact_net_bps.mean())


def _score_month(train: pd.DataFrame, held: pd.DataFrame, base_train: np.ndarray, base_held: np.ndarray, fields: list[str], month: str, seed: int) -> np.ndarray:
    residual_target = train.exact_net_bps.to_numpy(float) - base_train
    grade = np.digitize(residual_target, [-100.0, -25.0, 25.0, 100.0]).astype(np.int32)
    model_fields = ["base_anchor", "r3_meta_p_clear", "r3_meta_p_adverse", "r3_meta_p_weak", *train.attrs["context_fields"], *fields]
    train = train.copy(); held = held.copy(); train["base_anchor"] = base_train; held["base_anchor"] = base_held
    tr_raw, te_raw = _rank_fit(train, held, list(dict.fromkeys(model_fields)), grade, equal_month=True, seed=seed, feature_fraction=1.0)
    base_rank = _pct(held.base_score.to_numpy(float), train.base_score.to_numpy(float))
    return (0.75 * base_rank + 0.25 * _pct(te_raw, tr_raw)).astype("float32")


def _transition_delta(real: pd.DataFrame, placebo: pd.DataFrame) -> float:
    n = max(1, int(math.ceil(len(real) * 0.01)))
    real_ids = set(real.sort_values(["real_score", "candidate_id"], ascending=[False, True]).head(n).candidate_id)
    placebo_ids = set(placebo.sort_values(["placebo_score", "candidate_id"], ascending=[False, True]).head(n).candidate_id)
    entered = placebo.loc[placebo.candidate_id.isin(placebo_ids - real_ids), "exact_net_bps"]
    exited = placebo.loc[placebo.candidate_id.isin(real_ids - placebo_ids), "exact_net_bps"]
    return float(entered.mean() - exited.mean()) if len(entered) and len(exited) else float("nan")


def run(*, rolling_path: Path = DEFAULT_ROLLING, output_dir: Path = DEFAULT_OUTPUT, n_seeds: int = N_SEEDS) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    x, context, context_hash = _load(); x = _join_gam(x.loc[x.side_name.eq(SIDE)].copy(), rolling_path)
    month_data: list[tuple[str, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]] = []
    real_parts = []
    for month in MONTHS:
        held = x.loc[x.month.astype(str).eq(month)].copy(); train = x.loc[(x.__ts__ < pd.Timestamp(month, tz="UTC")) & (x.label_available_ts < pd.Timestamp(month, tz="UTC"))].copy()
        if held.empty or len(train) < 300:
            continue
        base_train, base_held = _map_base(train, held); _fill_gam_history(train, base_train); _fill_gam_history(held, base_held)
        train.attrs["context_fields"] = context; held.attrs["context_fields"] = context
        real = _score_month(train, held, base_train, base_held, ["gam_delta_bps", "gam_residual_bps"], month, SEED + int(month[-2:]) * 1000)
        real_parts.append(held[["candidate_id", "month", "exact_net_bps", "exact_gross_bps"]].assign(real_score=real))
        month_data.append((month, train, held, base_train, base_held))
    real_frame = pd.concat(real_parts, ignore_index=True)
    real_top1 = _top(real_frame, "real_score", 0.01); real_top5 = _top(real_frame, "real_score", 0.05)
    real_monthly = pd.Series({m: _top(g, "real_score", 0.05) for m, g in real_frame.groupby("month", sort=True)})
    rows = []
    for placebo_seed in range(n_seeds):
        parts = []
        for month, train, held, base_train, base_held in month_data:
            tr = _permute(train, ["gam_delta_bps", "gam_residual_bps"], SEED + placebo_seed * 100000 + int(month[-2:]))
            te = _permute(held, ["gam_delta_bps", "gam_residual_bps"], SEED + placebo_seed * 100000 + 50000 + int(month[-2:]))
            tr.attrs["context_fields"] = context; te.attrs["context_fields"] = context
            score = _score_month(tr, te, base_train, base_held, ["gam_delta_bps", "gam_residual_bps"], month, SEED + placebo_seed * 10000 + int(month[-2:]) * 1000)
            parts.append(te[["candidate_id", "month", "exact_net_bps", "exact_gross_bps"]].assign(placebo_score=score))
        placebo = pd.concat(parts, ignore_index=True)
        monthly = pd.Series({m: _top(g, "placebo_score", 0.05) for m, g in placebo.groupby("month", sort=True)})
        rows.append({"seed": placebo_seed, "top1_net_bps": _top(placebo, "placebo_score", 0.01), "top5_net_bps": _top(placebo, "placebo_score", 0.05), "mean_month_top5_net_bps": float(monthly.mean()), "q25_month_top5_net_bps": float(monthly.quantile(0.25)), "entered_minus_exited_top1": _transition_delta(real_frame, placebo)})
    distribution = pd.DataFrame(rows)
    real = pd.DataFrame([{"seed": "real", "top1_net_bps": real_top1, "top5_net_bps": real_top5, "mean_month_top5_net_bps": float(real_monthly.mean()), "q25_month_top5_net_bps": float(real_monthly.quantile(0.25)), "entered_minus_exited_top1": np.nan}])
    stats = []
    for metric in ["top1_net_bps", "top5_net_bps", "mean_month_top5_net_bps", "q25_month_top5_net_bps"]:
        value = float(real.iloc[0][metric]); null = distribution[metric].to_numpy(float); stats.append({"metric": metric, "real_value": value, "placebo_mean": float(null.mean()), "placebo_median": float(np.median(null)), "placebo_q05": float(np.quantile(null, .05)), "placebo_q95": float(np.quantile(null, .95)), "empirical_p_ge_real": float((1 + np.sum(null >= value)) / (len(null) + 1)), "placebo_seeds": len(null)})
    output_dir.mkdir(parents=True); distribution.to_parquet(output_dir / "placebo_distribution.parquet", index=False); real.to_parquet(output_dir / "real_reference.parquet", index=False); pd.DataFrame(stats).to_parquet(output_dir / "placebo_empirical_pvalues.parquet", index=False)
    manifest = {"schema": "tp6_sl4_gamres_placebo_distribution_v1", "status": "COMPLETE", "seeds": n_seeds, "side": SIDE, "placebo_fields": ["gam_delta_bps", "gam_residual_bps"], "permutation": "independent within training and held month; marginal values and missingness preserved", "model": "residual LambdaRank head, feature_fraction=1.0, 4-hour x side queries", "no_held_outcomes_in_fit": True, "context_sha256": context_hash, "artifacts": sorted(p.name for p in output_dir.iterdir())}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    lines = ["# GAM residual placebo distribution", "", "## Empirical p-values", "", pd.DataFrame(stats).round(3).to_string(index=False), "", "## Real reference", "", real.round(3).to_string(index=False)]
    (output_dir / "TP6_SL4_GAMRES_PLACEBO_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"output": str(output_dir), "seeds": n_seeds, "real_top1": real_top1, "real_top5": real_top5}, indent=2)); return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--rolling", type=Path, default=DEFAULT_ROLLING); parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT); parser.add_argument("--seeds", type=int, default=N_SEEDS); args = parser.parse_args(); run(rolling_path=args.rolling, output_dir=args.output_dir, n_seeds=args.seeds)
