#!/usr/bin/env python3
"""Bounded 2025-OOS geometry robustness screen for the DS specialist MC1 arm.

This is not a new model-family search.  DS head is the sole individual
dynamics arm with a positive 2025 EV/trade and total-net increment.  The
four predeclared shallow HGB geometries below test whether that result is a
fragile parameter accident.  2025 strict OOS chooses the geometry; 2026 is
reported only after that choice is frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_causal_exotic_dynamics_mc1_ablation as base
from scripts.ablate_strict_r3_bcf_current_v5_agreement_blend import POLICY_COLUMNS
from scripts.run_strict_r3_mc1_d2_controlled_ablation import SEED, _structural_curve


OUT = base.ROOT / "data_perp/artifacts/causal_exotic_dynamics_ds_head_geometry_hpo_2025oof_2026confirm_20260831_v1"

# Fixed before the run.  All are deliberately shallow/high-support variants.
GEOMETRIES: dict[str, dict[str, float | int]] = {
    "H0_frozen_d2": {"max_depth": 2, "max_iter": 80, "learning_rate": .04, "l2_regularization": 20.0, "min_samples_leaf": 100},
    "H1_shallow_d1": {"max_depth": 1, "max_iter": 100, "learning_rate": .04, "l2_regularization": 20.0, "min_samples_leaf": 100},
    "H2_smooth_d2": {"max_depth": 2, "max_iter": 120, "learning_rate": .03, "l2_regularization": 50.0, "min_samples_leaf": 200},
    "H3_regular_d2": {"max_depth": 2, "max_iter": 80, "learning_rate": .04, "l2_regularization": 80.0, "min_samples_leaf": 250},
}


def _fit_hgb_geometry(
    train: pd.DataFrame, features: list[str], config: dict[str, float | int],
):
    clean = train.dropna(subset=["policy_net_bps"]).copy()
    medians = clean.loc[:, features].apply(pd.to_numeric, errors="coerce").median(numeric_only=True)
    x = clean.loc[:, features].apply(pd.to_numeric, errors="coerce").fillna(medians)
    y = pd.to_numeric(clean["policy_net_bps"], errors="coerce")
    low, high = y.quantile([.02, .98]).to_numpy(float)
    y = y.clip(low, high)
    if len(x) > 50_000:
        take = x.sample(50_000, random_state=SEED).index
        x, y = x.loc[take], y.loc[take]
    model = HistGradientBoostingRegressor(random_state=SEED, **config).fit(x, y)
    return model, medians, _structural_curve(clean), (float(low), float(high))


def _append_pair(
    summaries: list[pd.DataFrame], *, policy: pd.DataFrame, frozen_control: pd.DataFrame,
    arm: str, mapped: pd.DataFrame, out: Path,
) -> None:
    start = mapped["__decision_ts__"].min().floor("1h")
    control_arm = f"M0_frozen_pair_control_{arm}_matched"
    control, _ = base._replay(frozen_control.loc[frozen_control.__decision_ts__.ge(start)].copy(), policy, control_arm, out)
    control["comparison_control_arm"], control["evaluation_start"] = control_arm, start
    summaries.append(control)
    metrics, _ = base._replay(mapped, policy, arm, out)
    metrics["comparison_control_arm"], metrics["evaluation_start"] = control_arm, start
    summaries.append(metrics)


def run(args: argparse.Namespace) -> Path:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    full, target_free = base._load_panel(args.dynamic.resolve())
    frozen, folds = base._contracts(args.assessment.resolve())
    out.mkdir(parents=True, exist_ok=False)
    policy = full.loc[:, ["candidate_id", *POLICY_COLUMNS]].copy()
    control = target_free.loc[target_free.__decision_ts__.ge(base.START)].copy()
    control["bcf_mapped_expected_bps"] = control.bcf_mc1_expected_bps
    control["current_mapped_expected_bps"] = control.current_mc1_expected_bps

    # Specialist is fixed/prequential before geometry screening begins.
    head, specialist_audit = base._specialist_oof(full, "DS", frozen, folds)
    if head.empty:
        raise AssertionError("DS specialist has no prequential output")
    head_field = "ds_specialist_residual_bps"
    full = full.merge(head.loc[:, ["candidate_id", head_field]], on="candidate_id", how="left", validate="one_to_one")
    target_free = target_free.merge(head.loc[:, ["candidate_id", head_field]], on="candidate_id", how="left", validate="one_to_one")

    summaries, mapper_audits = [], []
    baseline, _ = base._replay(control, policy, "M0_frozen_pair_control", out)
    summaries.append(baseline)
    original_fit = base._fit_hgb
    try:
        for arm, config in GEOMETRIES.items():
            base._fit_hgb = lambda train, features, config=config: _fit_hgb_geometry(train, list(features), config)
            mapped, audit = base._map_one_family(
                full, target_free, family="DS", representation="head", frozen=frozen, folds=folds,
            )
            mapper_audits.append(audit.assign(arm=arm))
            if mapped.empty:
                raise AssertionError(f"{arm}: no strict-OOS rows")
            mapped.to_parquet(out / f"{arm}_target_free_scores.parquet", index=False, compression="zstd")
            _append_pair(summaries, policy=policy, frozen_control=control, arm=arm, mapped=mapped, out=out)
    finally:
        base._fit_hgb = original_fit

    result = pd.concat(summaries, ignore_index=True)
    result["comparison_control_arm"] = result.get("comparison_control_arm", "M0_frozen_pair_control").fillna("M0_frozen_pair_control")
    for index, row in result.iterrows():
        reference = result.loc[result.arm.eq(row.comparison_control_arm) & result.period.eq(row.period)]
        if len(reference) != 1:
            continue
        for field in (
            "accepted_rows", "net_ev_bps_per_realised_trade", "net_sum_bps_realised",
            "worst_month_bps", "worst_week_bps", "max_drawdown", "ulcer_index",
            "daily_cvar5", "time_underwater_fraction",
        ):
            result.loc[index, f"delta_vs_m0_{field}"] = row[field] - reference.iloc[0][field]
    result.to_parquet(out / "portfolio_summary.parquet", index=False)
    result.to_csv(out / "portfolio_summary.csv", index=False)
    specialist_audit.to_parquet(out / "specialist_oof_audit.parquet", index=False)
    pd.concat(mapper_audits, ignore_index=True).to_parquet(out / "mapper_oof_audit.parquet", index=False)
    manifest = {
        "schema": "causal-exotic-dynamics-ds-head-geometry-hpo-v1",
        "scope": "offline strict-OOS geometry robustness screen; no live/canonical mutation",
        "selection": "select only from 2025 strict-OOS results; 2026 is frozen confirmation",
        "family": "DS specialist head", "geometries": GEOMETRIES,
        "target": "source-aligned parent-policy net bps; fixed prequential DS residual head",
        "admission": "dual BCF/current >= +50 bps; BCF mapped EV priority",
        "portfolio": "fixed global 7x/10%-slot, 2-new, 8-concurrent, 80%-wallet auction",
        "dynamic_manifest_sha256": base._sha256(args.dynamic.resolve() / "run_manifest.json"),
        "assessment_manifest_sha256": base._sha256(args.assessment.resolve() / "run_manifest.json"),
        "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", type=Path, default=base.DYNAMIC)
    parser.add_argument("--assessment", type=Path, default=base.ASSESSMENT)
    parser.add_argument("--out", type=Path, default=OUT)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
