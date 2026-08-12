#!/usr/bin/env python3
"""Frozen HPO and modulation validation for Support+OOD + global correctness.

Only the globally pooled, prior-resolved base-correctness block is added to
Support+OOD.  The sequence is intentionally strict:

1. Tune the reliability classifier on 2024 only (train Apr--Sep, validate
   Oct--Nov) with a subsampled Optuna/MedianPruner study.
2. Freeze that parameter set, generate expanding monthly 2025 scores, and
   choose a predeclared multiplier/shrink transform on Jan--Sep only.
3. Report Oct--Dec exactly once as confirmation against frozen Support+OOD
   alpha=1.0 on identical candidate IDs.

The October--December rows have appeared in earlier diagnostic experiments, so
this is a disciplined *confirmation reuse*, not an untouched final test.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_tp6_sl4_compact_path_joint_hpo_2025 as compact  # noqa: E402
from scripts import run_tp6_sl4_support_ood_health_blocks_2025 as health  # noqa: E402
from scripts.run_tp6_sl4_downstream_retrain_2025 import MONTHS  # noqa: E402


SEED = 20260813
OUT = ROOT / "data_perp/artifacts/tp6_sl4_support_ood_global_correctness_hpo_20260809_v1"
HPO_TRIALS = 48
DEVELOPMENT_MONTHS = tuple(f"2025-{month:02d}" for month in range(1, 10))
CONFIRMATION_MONTHS = tuple(f"2025-{month:02d}" for month in range(10, 13))
SHRINK_LOWER = (0.0, 0.25, 0.50, 0.75)
MULTIPLY_ALPHA = (0.25, 0.50, 0.75, 1.0, 1.25)


def _selection_table(monthly: pd.DataFrame, global_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dev = monthly.loc[monthly.month.isin(DEVELOPMENT_MONTHS) & monthly["tail"].eq(0.05)]
    for arm, value in dev.groupby("arm", observed=True, sort=True):
        x = value.net_bps_per_trade.to_numpy(float)
        median = float(np.median(x)); mad = float(np.median(np.abs(x - median))); worst = float(x.min())
        top1 = global_metrics.loc[(global_metrics.arm.eq(arm)) & global_metrics["tail"].eq(0.01), "net_bps_per_trade"]
        rows.append({
            "arm": arm, "dev_mean_top5": float(x.mean()), "dev_median_top5": median,
            "dev_mad_top5": mad, "dev_worst_top5": worst,
            "development_portability": median - 0.5 * mad - max(0.0, -worst),
            "all_2025_top1_net": float(top1.iloc[0]) if len(top1) else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["development_portability", "dev_mean_top5", "all_2025_top1_net"], ascending=False, kind="stable")


def _confirmation(monthly: pd.DataFrame, arm: str) -> tuple[float, float]:
    x = monthly.loc[monthly.arm.eq(arm) & monthly.month.isin(CONFIRMATION_MONTHS) & monthly["tail"].eq(0.05), "net_bps_per_trade"].to_numpy(float)
    return float(x.mean()), float(x.min())


def _fold(train: pd.DataFrame, held: pd.DataFrame, fields: list[str], global_params: dict[str, object], baseline_params: dict[str, object], baseline_fields: list[str], month_no: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr_anchor, te_anchor = compact._map_base(train, held)
    train = train.copy(); held = held.copy()
    train["base_anchor"] = tr_anchor; held["base_anchor"] = te_anchor
    target = (train.net_bps.to_numpy(float) - tr_anchor > 0.0).astype(np.int8)
    out = held[["candidate_id", "__ts__", "month", "net_bps", "gross_bps", "base_plus_consensus25"]].copy()
    out["canonical_control"] = held.base_plus_consensus25.to_numpy(float)
    # Exact frozen Support+OOD reference for an apples-to-apples confirmation.
    base_probability, _, base_model, _ = compact._fit_probability(
        train, held, baseline_fields, target, baseline_params,
        seed=compact.SEED + month_no * 10000 + (zlib.adler32(b"support_ood") % 100000), return_model=True,
    )
    out["frozen_support_ood_a100"] = pd.Series(out.canonical_control.to_numpy(float) * np.clip(1.0 + (base_probability - .5), .5, 1.5)).rank(pct=True, method="average").to_numpy("float32")
    del base_model
    probability, info, model, _ = compact._fit_probability(
        train, held, fields, target, global_params,
        seed=SEED + month_no * 10000 + (zlib.adler32(b"support_ood_global_recent") % 100000), return_model=True,
    )
    usage = pd.DataFrame({"month": str(held.month.iloc[0]), "field": fields, "gain": model.booster_.feature_importance(importance_type="gain")}) if model is not None else pd.DataFrame(columns=["month", "field", "gain"])
    del model
    gc.collect()
    for lower in SHRINK_LOWER:
        raw = .5 + (out.canonical_control.to_numpy(float) - .5) * (lower + (1.0 - lower) * probability)
        out[f"shrink__global_recent__lo{int(lower*100):02d}"] = pd.Series(raw).rank(pct=True, method="average").to_numpy("float32")
    for alpha in MULTIPLY_ALPHA:
        raw = out.canonical_control.to_numpy(float) * np.clip(1.0 + alpha * (probability - .5), 1.0 - .5 * alpha, 1.0 + .5 * alpha)
        out[f"multiply__global_recent__a{int(alpha*100):03d}"] = pd.Series(raw).rank(pct=True, method="average").to_numpy("float32")
    return out, usage.assign(best_iteration=info["best_iteration"], feature_count=info["feature_count"])


def run(*, out: Path = OUT, trials: int = HPO_TRIALS) -> Path:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    panel, blocks, lineage, state_audit = compact._build_panel()
    baseline_fields = list(dict.fromkeys([*blocks["market_context"], *blocks["soft_membership"], *blocks["activated_leaf_support"], *blocks["rule_path_ood_drift"]]))
    global_fields = list(dict.fromkeys([*baseline_fields, *health._global_recent(panel)]))
    hpo_train, hpo_validation = compact._hpo_reference(panel)
    print(f"HPO global correctness: {trials} trials", flush=True)
    params, trial_table, hpo_gain = compact._inner_hpo(hpo_train, hpo_validation, global_fields, trials=trials, seed=SEED)
    frozen_baseline = health._baseline_params()
    parts: list[pd.DataFrame] = []
    usage_parts: list[pd.DataFrame] = []
    for month_no, month in enumerate(MONTHS):
        cutoff = pd.Timestamp(month, tz="UTC")
        train = panel.loc[panel.__ts__.lt(cutoff) & panel.label_available_ts.lt(cutoff)].copy()
        held = panel.loc[panel.month.astype(str).eq(month)].copy()
        print(f"FOLD {month}", flush=True)
        score, usage = _fold(train, held, global_fields, params, frozen_baseline, baseline_fields, month_no)
        parts.append(score); usage_parts.append(usage)
    prediction = pd.concat(parts, ignore_index=True)
    arms = ["canonical_control", "frozen_support_ood_a100", *[field for field in prediction if field.startswith(("shrink__", "multiply__"))]]
    global_metrics, monthly, stability = compact._metric_table(prediction, arms)
    selection = _selection_table(monthly, global_metrics)
    selected = str(selection.iloc[0].arm)
    reference_confirm, reference_worst = _confirmation(monthly, "frozen_support_ood_a100")
    selected_confirm, selected_worst = _confirmation(monthly, selected)
    selected_global = float(global_metrics.loc[(global_metrics.arm.eq(selected)) & global_metrics["tail"].eq(.05), "net_bps_per_trade"].iloc[0])
    reference_global = float(global_metrics.loc[(global_metrics.arm.eq("frozen_support_ood_a100")) & global_metrics["tail"].eq(.05), "net_bps_per_trade"].iloc[0])
    promote = selected_confirm > reference_confirm and selected_global > reference_global
    usage = pd.concat(usage_parts, ignore_index=True)
    usage["block"] = np.where(usage.field.str.startswith("model_recent"), "global_recent", "support_ood")
    usage_summary = usage.groupby(["block", "field"], observed=True).agg(mean_gain=("gain", "mean"), used_months=("gain", lambda x: int((x > 0).sum()))).reset_index()
    prediction.to_parquet(out / "predictions.parquet", index=False, compression="zstd")
    global_metrics.to_parquet(out / "metrics_global.parquet", index=False); monthly.to_parquet(out / "metrics_monthly.parquet", index=False); stability.to_parquet(out / "metrics_stability.parquet", index=False)
    trial_table.to_parquet(out / "hpo_trials.parquet", index=False); hpo_gain.to_parquet(out / "hpo_feature_gain.parquet", index=False); pd.DataFrame([{"params_json": json.dumps(params, sort_keys=True)}]).to_parquet(out / "hpo_winner.parquet", index=False)
    selection.to_parquet(out / "modulation_development_selection.parquet", index=False); usage.to_parquet(out / "feature_usage_by_fold.parquet", index=False); usage_summary.to_parquet(out / "feature_usage_summary.parquet", index=False); lineage.to_parquet(out / "lineage.parquet", index=False); state_audit.to_parquet(out / "inherited_state_audit.parquet", index=False)
    gate = {
        "selected_from_jan_sep_only": selected,
        "reference": "frozen_support_ood_a100",
        "selected_global_top5_net_bps": selected_global,
        "reference_global_top5_net_bps": reference_global,
        "selected_confirmation_oct_dec_top5_mean_bps": selected_confirm,
        "reference_confirmation_oct_dec_top5_mean_bps": reference_confirm,
        "selected_confirmation_oct_dec_top5_worst_bps": selected_worst,
        "reference_confirmation_oct_dec_top5_worst_bps": reference_worst,
        "status": "PASS_PROMOTION_CANDIDATE" if promote else "FAIL_HOLD_GLOBAL_CORRECTNESS",
        "note": "Oct-Dec was previously inspected in related diagnostics; this is confirmation reuse, not untouched final OOS.",
    }
    (out / "promotion_gate.json").write_text(json.dumps(gate, indent=2) + "\n")
    correctness = {
        "hpo_train": "2024-04 through 2024-09", "hpo_validation": "2024-10 through 2024-11", "hpo_trials": trials,
        "modulation_development": list(DEVELOPMENT_MONTHS), "confirmation": list(CONFIRMATION_MONTHS),
        "global_correctness_is_strict_prior_label_available_ts": True,
        "base_and_canonical_rows_identical": True, "all_scores_finite": bool(np.isfinite(prediction[arms].to_numpy(float)).all()),
        "scope": "long-only matched canonical development replay; no untouched final-test claim",
    }
    (out / "correctness_test_report.json").write_text(json.dumps(correctness, indent=2) + "\n")
    manifest = {"schema": "tp6_sl4_support_ood_global_correctness_hpo_20260809_v1", "status": "COMPLETE", "rows": len(prediction), "hpo_params": params, "selected_modulation": selected, "artifacts": sorted(path.name for path in out.iterdir())}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    report = ["# Support+OOD + global correctness — frozen HPO/modulation validation", "", "HPO uses 2024 only.  Multiplier/shrink choice is made on Jan--Sep; Oct--Dec is reported separately as confirmation reuse.", "", "## Development modulation selection", "", selection.round(3).to_string(index=False), "", "## Global Top-5", "", global_metrics.loc[global_metrics["tail"].eq(.05)].sort_values("net_bps_per_trade", ascending=False).round(3).to_string(index=False), "", "## Promotion gate", "", json.dumps(gate, indent=2), "", "## Correctness", "", json.dumps(correctness, indent=2)]
    (out / "SUPPORT_OOD_GLOBAL_CORRECTNESS_HPO_REPORT.md").write_text("\n".join(report) + "\n")
    print(json.dumps({"out": str(out), "selected": selected, "gate": gate}, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--trials", type=int, default=HPO_TRIALS)
    args = parser.parse_args()
    run(out=args.out, trials=args.trials)
