#!/usr/bin/env python3
"""Run a leakage-safe short-only meta ablation on a fixed candidate universe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_s52_train_meta_regime_handoff_smoke import (
    _projected_handoff_columns_for_selected,
    run_smoke,
)


DEFAULT_HANDOFF = Path(
    "data_perp/reports/s59_h5_signalclose_causal_stagec_packb_wf30_20260721_v1/meta_handoff_top30"
)
DEFAULT_CONTRACT = Path(
    "data_perp/reports/s59_h5_signalclose_causal_stagec_packb_meta_hpo150_wf30_20260721_v1/s52_train_meta_hpo_best.json"
)
DEFAULT_OUTPUT = Path("data_perp/reports/meta_short_only_packb_fixedparams_20260721_v1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff-dir", type=Path, default=DEFAULT_HANDOFF)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-months", default="2026-04,2026-05,2026-06")
    parser.add_argument("--max-oos-model-age-days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--save-fold-models", action="store_true")
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    selected = list(contract.get("selected_feature_union", []) or [])
    if not selected:
        raise ValueError(f"No selected features in {args.contract}")
    params = {
        "classifier": dict(contract.get("classifier_params", {}) or {}),
        "regressor": dict(contract.get("regressor_params", {}) or {}),
    }
    if not params["classifier"]:
        raise ValueError(f"No classifier params in {args.contract}")
    projected_columns = _projected_handoff_columns_for_selected(
        args.handoff_dir / "train_meta_regime_handoff.parquet", selected
    )
    manifest = run_smoke(
        handoff_dir=args.handoff_dir,
        ledger_path=args.handoff_dir / "s52_trailing_regime_scored_ledger.parquet",
        out_dir=args.out_dir,
        frontier="top30",
        seed=int(args.seed),
        train_scope="selected",
        enable_base_prior_features=True,
        enable_reliability_features=True,
        enable_support_drift_features=True,
        enable_hit_surprise_features=True,
        feature_selection_method="fixed_pack_b_short_only",
        max_oos_model_age_days=int(args.max_oos_model_age_days),
        model_train_max_rows=0,
        model_params=params,
        model_profile_name="weighted_pack_b_short_only_fixed_params_v1",
        meta_head_mode="single_base_soft_label",
        fixed_selected_features=selected,
        # One model is intentionally fit on short rows only.  No long model is
        # trained or scored in this diagnostic.
        side_specific_single_head=False,
        eval_months=[m.strip() for m in args.eval_months.split(",") if m.strip()],
        save_fold_models=bool(args.save_fold_models),
        handoff_columns=projected_columns,
        # The historical Pack B handoff predates the forward-label-resolution
        # column. Keep this relative ablation on its exact original protocol;
        # the emitted manifest makes the non-strict status explicit.
        strict_handoff_contract=False,
        active_sides=("short",),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
