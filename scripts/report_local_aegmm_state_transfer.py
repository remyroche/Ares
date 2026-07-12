#!/usr/bin/env python3
"""Report train-prior to OOS transfer for frozen local economic AE/GMM states."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_train_meta_residual_archetype_enhancement import (  # noqa: E402
    local_aegmm_state_transfer_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-dir", type=Path, required=True)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--catalog", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arm_dir = Path(args.arm_dir)
    arm = str(args.arm or arm_dir.name)
    predictions_path = args.predictions or arm_dir / "oos_predictions.parquet"
    default_catalogs = sorted(arm_dir.glob("state/**/local_economic_aegmm_catalog.csv"))
    catalog_path = args.catalog or (default_catalogs[-1] if default_catalogs else None)
    output_path = args.output or arm_dir / "local_aegmm_state_transfer_metrics.csv"
    if catalog_path is None:
        raise FileNotFoundError(f"No local AE/GMM catalog under {arm_dir / 'state'}")
    predictions = pd.read_parquet(predictions_path)
    catalog = pd.read_csv(catalog_path)
    report = local_aegmm_state_transfer_metrics(predictions, arm, catalog)
    if report.empty:
        raise ValueError("No local AE/GMM state transfer rows were produced")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)

    top10 = report.loc[report["scope"].astype(str).str.endswith("top10")].copy()
    summary = (
        top10.groupby(["state_block", "scope"], observed=True)
        .agg(
            state_rows=("selected_rows", "sum"),
            mean_state_ev=("mean_ev_after_1pct", "mean"),
            weighted_state_ev=(
                "mean_ev_after_1pct",
                lambda values: float(
                    (values * top10.loc[values.index, "selected_rows"]).sum()
                    / max(float(top10.loc[values.index, "selected_rows"].sum()), 1.0)
                ),
            ),
            weighted_prior_ev_error=(
                "prior_ev_error",
                lambda values: float(
                    (values * top10.loc[values.index, "selected_rows"]).sum()
                    / max(float(top10.loc[values.index, "selected_rows"].sum()), 1.0)
                ),
            ),
            sign_agreement_rate=("posterior_prior_ev_lift_sign_agrees", "mean"),
            mean_bad_mae=("first_touch_bad_mae_rate", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(
        output_path.with_name("local_aegmm_state_transfer_summary.csv"), index=False
    )

    failures = top10.loc[
        top10["selected_rows"].ge(30)
        & (
            top10["posterior_prior_ev_lift_sign_agrees"].fillna(1.0).lt(0.5)
            | top10["prior_ev_error"].abs().ge(0.003)
        )
    ].sort_values(
        ["scope", "prior_ev_error", "selected_rows"],
        ascending=[True, True, False],
        kind="stable",
    )
    failures.to_csv(
        output_path.with_name("local_aegmm_state_transfer_failures.csv"), index=False
    )
    print(
        {
            "rows": int(len(report)),
            "top10_rows": int(len(top10)),
            "output": str(output_path),
            "catalog": str(catalog_path),
        }
    )


if __name__ == "__main__":
    main()
