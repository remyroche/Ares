#!/usr/bin/env python3
"""Robustness checks for the frozen one-field month-ahead GAM contract.

This is deliberately a narrow confirmation replay, not another architecture
search.  The structural branch is already prequential in
``rolling_archetype_gam_oos``.  Here we compare the exact matched residual/meta
stack with:

* the single canonical field ``gam_disagreement = gam_delta_bps``;
* the mathematically duplicate two-field representation;
* the single-field representation with reversed feature-column order.

The one-field arm is hard-gated by the target month's transport validity.  An
invalid month is exactly the control.  Seed sweeps are run on the full matched
stack; the saved artifact also contains an abstain-on-invalid diagnostic.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
import sys
from typing import Sequence

import lightgbm as lgb  # noqa: F401  (keeps dependency explicit for manifests)
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_tp6_sl4_downstream_retrain_2025 import MONTHS, _load, _map_base, _pct  # noqa: E402
from scripts.run_tp6_sl4_rolling_gam_residual_integration import (  # noqa: E402
    _fill_gam_history,
    _fit_heads,
    _join_gam,
)


DEFAULT_ROLLING = ROOT / "data_perp/artifacts/tp6_sl4_rolling_archetype_gam_oos_20260815_v5/rolling_oof_predictions.parquet"
DEFAULT_OUTPUT = ROOT / "data_perp/artifacts/tp6_sl4_gam_onefield_robustness_20260815_v1"
SIDE = "long"
BASE_SEED = 20260815
SEEDS = tuple(BASE_SEED + 1009 * i for i in range(10))
TAILS = (0.005, 0.01, 0.02, 0.05, 0.10)


def _metric(frame: pd.DataFrame, score: str, tail: float) -> dict[str, object]:
    finite = frame.loc[np.isfinite(frame[score].to_numpy(float))].copy()
    n = max(1, int(math.ceil(len(finite) * tail)))
    top = finite.sort_values([score, "candidate_id"], ascending=[False, True], kind="stable").head(n)
    return {
        "tail": float(tail),
        "trades": int(len(top)),
        "gross_bps_per_trade": float(top.exact_gross_bps.mean()) if len(top) else np.nan,
        "net_bps_per_trade": float(top.exact_net_bps.mean()) if len(top) else np.nan,
        "rows_scored": int(len(finite)),
        "exposure_fraction": float(len(finite) / max(len(frame), 1)),
    }


def _monthly(frame: pd.DataFrame, score: str) -> pd.DataFrame:
    rows = []
    for month, block in frame.groupby("month", sort=True):
        row = _metric(block, score, 0.05)
        row.update({"month": str(month), "score": score})
        rows.append(row)
    return pd.DataFrame(rows)


def _build_month_frames(x: pd.DataFrame, context: Sequence[str], month: str):
    held = x.loc[x.month.astype(str).eq(month)].copy()
    train = x.loc[(x.__ts__ < pd.Timestamp(month, tz="UTC")) & (x.label_available_ts < pd.Timestamp(month, tz="UTC"))].copy()
    if held.empty or len(train) < 300:
        return None
    base_train, base_held = _map_base(train, held)
    _fill_gam_history(train, base_train)
    _fill_gam_history(held, base_held)
    train.attrs["context_fields"] = list(context)
    held.attrs["context_fields"] = list(context)
    return train, held, base_train, base_held


def _score_arm(train: pd.DataFrame, held: pd.DataFrame, base_train: np.ndarray, base_held: np.ndarray, *, fields: Sequence[str], seed: int, reverse: bool) -> np.ndarray:
    consensus, residual_rank, _, _ = _fit_heads(
        train.copy(),
        held.copy(),
        base_train,
        base_held,
        use_gam_inputs=False,
        extra_fields=list(fields),
        feature_fraction=1.0,
        month=str(held.month.iloc[0]),
        seed_base=seed,
        reverse_feature_order=reverse,
    )
    base_rank = _pct(held.base_score.to_numpy(float), train.base_score.to_numpy(float))
    return (0.50 * base_rank + 0.25 * consensus + 0.25 * residual_rank).astype(np.float32)


def _apply_gate(score: np.ndarray, control: np.ndarray, held: pd.DataFrame) -> np.ndarray:
    valid = bool(float(held.gam_transport_valid.mean()) > 0.5)
    return score if valid else control


def run(*, rolling_path: Path = DEFAULT_ROLLING, output_dir: Path = DEFAULT_OUTPUT, seeds: Sequence[int] = SEEDS) -> Path:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    x, context, context_hash = _load()
    x = _join_gam(x.loc[x.side_name.eq(SIDE)].copy(), rolling_path)
    parts: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    months = [str(m) for m in MONTHS]
    for seed in seeds:
        for month in months:
            built = _build_month_frames(x, context, month)
            if built is None:
                continue
            train, held, base_train, base_held = built
            control = _score_arm(train, held, base_train, base_held, fields=(), seed=int(seed), reverse=False)
            one = _score_arm(train, held, base_train, base_held, fields=("gam_delta_bps",), seed=int(seed), reverse=False)
            one_rev = _score_arm(train, held, base_train, base_held, fields=("gam_delta_bps",), seed=int(seed), reverse=True)
            for arm, raw in (("control", control), ("one_field", one), ("one_field_reversed", one_rev)):
                gated = raw if arm != "control" else raw
                if arm != "control":
                    gated = _apply_gate(raw, control, held)
                out = held[["candidate_id", "month", "exact_net_bps", "exact_gross_bps", "gam_transport_valid", "gam_delta_bps", "gam_residual_bps"]].copy()
                out["seed"] = int(seed)
                out["arm"] = arm
                out["score"] = gated
                out["raw_score"] = raw
                out["target_month_valid"] = int(float(held.gam_transport_valid.mean()) > 0.5)
                out["field_contract"] = json.dumps(["gam_disagreement"] if arm.startswith("one_field") else [])
                parts.append(out)
            audit.append({"seed": int(seed), "month": month, "train_rows": int(len(train)), "held_rows": int(len(held)), "target_month_valid": int(float(held.gam_transport_valid.mean()) > 0.5), "train_valid_fraction": float(train.gam_transport_valid.mean()), "held_valid_fraction": float(held.gam_transport_valid.mean()), "query_groups": int(len(pd.to_datetime(train.__ts__, utc=True).dt.floor("4h").unique()))})
            del train, held
            gc.collect()

    pred = pd.concat(parts, ignore_index=True)
    # The production stack ranks globally only after each held month has been
    # normalized within month/side.  Keep the raw score for diagnostics but
    # use the month-normalized score for every economic metric below.
    pred["raw_score"] = pred["score"].astype(float)
    pred["score"] = pred.groupby(["seed", "arm", "month"], sort=False)["score"].transform(
        lambda z: z.rank(pct=True, method="average")
    ).astype("float32")
    metrics: list[dict[str, object]] = []
    for (seed, arm), block in pred.groupby(["seed", "arm"], sort=True):
        for tail in TAILS:
            row = _metric(block, "score", tail)
            row.update({"seed": int(seed), "arm": arm})
            metrics.append(row)
    metrics_df = pd.DataFrame(metrics)
    monthly_parts = []
    for (seed, arm), block in pred.groupby(["seed", "arm"], sort=True):
        monthly_parts.append(_monthly(block, "score").assign(seed=int(seed), arm=arm))
    monthly_df = pd.concat(monthly_parts, ignore_index=True)
    # Per-seed matched deltas; primary endpoint is global Top-5.
    top5 = metrics_df.loc[metrics_df["tail"].eq(0.05)].pivot(index="seed", columns="arm", values="net_bps_per_trade").reset_index()
    for arm in ("one_field", "one_field_reversed"):
        top5[f"delta_{arm}_minus_control"] = top5[arm] - top5["control"]
    summary_rows = []
    for arm in ("control", "one_field", "one_field_reversed"):
        vals = top5[arm].to_numpy(float)
        delta = top5[f"delta_{arm}_minus_control"].to_numpy(float) if arm != "control" else np.zeros_like(vals)
        summary_rows.append({"arm": arm, "seeds": int(len(vals)), "top5_mean": float(np.mean(vals)), "top5_median": float(np.median(vals)), "top5_mad": float(np.median(np.abs(vals - np.median(vals)))), "top5_min": float(np.min(vals)), "top5_max": float(np.max(vals)), "positive_seed_fraction": float(np.mean(delta > 0)) if arm != "control" else np.nan, "delta_mean": float(np.mean(delta)) if arm != "control" else 0.0, "delta_median": float(np.median(delta)) if arm != "control" else 0.0})
    summary_df = pd.DataFrame(summary_rows)

    # A cheap and interpretable diagnostic: exact control with no exposure in
    # invalid target months.  This is not promoted as a production score.
    control = pred.loc[pred.arm.eq("control") & pred.seed.eq(int(seeds[0]))].copy()
    control["abstain_invalid_score"] = np.where(control.target_month_valid.eq(1), control.score, -np.inf)
    abstain_rows = []
    for tail in TAILS:
        row = _metric(control, "abstain_invalid_score", tail)
        row.update({"arm": "control_abstain_invalid", "tail": tail})
        abstain_rows.append(row)
    abstain_df = pd.DataFrame(abstain_rows)

    output_dir.mkdir(parents=True)
    pred.to_parquet(output_dir / "predictions_seed_order.parquet", index=False, compression="zstd")
    metrics_df.to_parquet(output_dir / "metrics_seed_order.parquet", index=False)
    monthly_df.to_parquet(output_dir / "metrics_monthly_seed_order.parquet", index=False)
    top5.to_parquet(output_dir / "top5_seed_deltas.parquet", index=False)
    summary_df.to_parquet(output_dir / "seed_order_summary.parquet", index=False)
    abstain_df.to_parquet(output_dir / "abstain_invalid_metrics.parquet", index=False)
    pd.DataFrame(audit).to_parquet(output_dir / "fit_audit.parquet", index=False)
    manifest = {
        "schema": "tp6_sl4_gam_onefield_robustness_v1",
        "status": "COMPLETE",
        "side": SIDE,
        "seeds": [int(s) for s in seeds],
        "frozen_signal": "gam_disagreement = gam_delta_bps",
        "duplicate_reference": "gam_residual_bps = 4 * gam_delta_bps",
        "hard_gate": "target-month transport-valid => one-field arm; invalid => exact control",
        "reversed_order": "all residual/meta feature columns reversed for one_field_reversed",
        "primary_metric": "global top-5 net bps/trade",
        "rolling_gam": str(rolling_path),
        "context_sha256": context_hash,
        "abstention": "control score set to -inf in invalid target months; diagnostic only",
        "artifacts": sorted(p.name for p in output_dir.iterdir()),
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = [
        "# Frozen one-field GAM robustness",
        "",
        "Each target month was scored once using the pre-existing month-ahead rolling GAM artifact. No target-month outcomes enter archetype, cluster, context, or GAM fitting.",
        "",
        "## Seed/order summary",
        "",
        summary_df.round(3).to_string(index=False),
        "",
        "## Abstain-invalid diagnostic",
        "",
        abstain_df.round(3).to_string(index=False),
        "",
        "## Contract",
        "",
        "`gam_residual_bps` is exactly `4 * gam_delta_bps`; the production candidate is therefore the single `gam_disagreement` field.",
    ]
    (output_dir / "TP6_SL4_GAM_ONEFIELD_ROBUSTNESS_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"output": str(output_dir), "rows": int(len(pred)), "seeds": len(seeds)}, indent=2))
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rolling", type=Path, default=DEFAULT_ROLLING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", type=int, default=len(SEEDS))
    args = parser.parse_args()
    run(rolling_path=args.rolling, output_dir=args.output_dir, seeds=tuple(BASE_SEED + 1009 * i for i in range(args.seeds)))
