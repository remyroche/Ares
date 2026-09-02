#!/usr/bin/env python3
"""Small, predeclared strict-OOS addition funnel for causal dynamics MC1 maps.

The individual-family screen is already sealed in
``causal_exotic_dynamics_mc1_ablation_2025oof_2026confirm_20260831_v1``.
This runner deliberately does not search combinations.  It tests only the
2025-selected distribution-specialist base plus the next two predeclared
2025 additions, then reports 2026 as confirmation only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_causal_exotic_dynamics_mc1_ablation as base
from scripts.ablate_strict_r3_bcf_current_v5_agreement_blend import POLICY_COLUMNS


OUT = base.ROOT / "data_perp/artifacts/causal_exotic_dynamics_mc1_combo_ablation_2025oof_2026confirm_20260831_v1"

# Chosen exclusively from the completed 2025 strict-OOS individual-arm table:
# DS head supplied the only positive EV/trade and total-PnL increment; SP head
# and WV raw+head were the next two total-PnL additions.  This list is frozen
# here before any 2026 confirmation result is read.
COMBOS: dict[str, tuple[tuple[str, str], ...]] = {
    "C1_DS_head_SP_head": (("DS", "head"), ("SP", "head")),
    "C2_DS_head_WV_raw_head": (("DS", "head"), ("WV", "raw_head")),
    "C3_DS_head_SP_head_WV_raw_head": (("DS", "head"), ("SP", "head"), ("WV", "raw_head")),
}


def _append_pair(
    summaries: list[pd.DataFrame], *, policy: pd.DataFrame, frozen_control: pd.DataFrame,
    arm: str, mapped: pd.DataFrame, out: Path,
) -> None:
    arm_start = mapped["__decision_ts__"].min().floor("1h")
    control_arm = f"M0_frozen_pair_control_{arm}_matched"
    control = frozen_control.loc[frozen_control.__decision_ts__.ge(arm_start)].copy()
    control_summary, _ = base._replay(control, policy, control_arm, out)
    control_summary["comparison_control_arm"] = control_arm
    control_summary["evaluation_start"] = arm_start
    summaries.append(control_summary)
    arm_summary, _ = base._replay(mapped, policy, arm, out)
    arm_summary["comparison_control_arm"] = control_arm
    arm_summary["evaluation_start"] = arm_start
    summaries.append(arm_summary)


def run(args: argparse.Namespace) -> Path:
    out = args.out.resolve()
    if out.exists():
        raise FileExistsError(f"immutable output exists: {out}")
    full, target_free = base._load_panel(args.dynamic.resolve())
    frozen, folds = base._contracts(args.assessment.resolve())
    out.mkdir(parents=True, exist_ok=False)
    policy = full.loc[:, ["candidate_id", *POLICY_COLUMNS]].copy()
    frozen_control = target_free.loc[target_free.__decision_ts__.ge(base.START)].copy()
    frozen_control["bcf_mapped_expected_bps"] = frozen_control.bcf_mc1_expected_bps
    frozen_control["current_mapped_expected_bps"] = frozen_control.current_mc1_expected_bps
    summaries: list[pd.DataFrame] = []
    baseline, _ = base._replay(frozen_control, policy, "M0_frozen_pair_control", out)
    summaries.append(baseline)

    specialist_audits = []
    for family in base.FAMILIES:
        head, audit = base._specialist_oof(full, family, frozen, folds)
        specialist_audits.append(audit)
        if head.empty:
            raise AssertionError(f"{family}: no prequential specialist output")
        field = f"{family.lower()}_specialist_residual_bps"
        full = full.merge(head.loc[:, ["candidate_id", field]], on="candidate_id", how="left", validate="one_to_one")
        target_free = target_free.merge(head.loc[:, ["candidate_id", field]], on="candidate_id", how="left", validate="one_to_one")

    combo_audits = []
    for arm, specification in COMBOS.items():
        mapped, audit = base._map_combo(
            full, target_free, arm=arm, specification=specification,
            frozen=frozen, folds=folds,
        )
        combo_audits.append(audit)
        if mapped.empty:
            raise AssertionError(f"{arm}: no strict-OOS mapped rows")
        mapped.to_parquet(out / f"{arm}_target_free_scores.parquet", index=False, compression="zstd")
        _append_pair(summaries, policy=policy, frozen_control=frozen_control, arm=arm, mapped=mapped, out=out)

    result = pd.concat(summaries, ignore_index=True)
    result["comparison_control_arm"] = result.get("comparison_control_arm", "M0_frozen_pair_control").fillna("M0_frozen_pair_control")
    for index, row in result.iterrows():
        reference = result.loc[
            result.arm.eq(row.comparison_control_arm) & result.period.eq(row.period)
        ]
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
    pd.concat(specialist_audits, ignore_index=True).to_parquet(out / "specialist_oof_audit.parquet", index=False)
    pd.concat(combo_audits, ignore_index=True).to_parquet(out / "combo_oof_audit.parquet", index=False)
    manifest = {
        "schema": "causal-exotic-dynamics-mc1-combo-ablation-v1",
        "scope": "offline strict temporal addition funnel; no live/canonical/policy/execution mutation",
        "selection": "all combination specifications predeclared from 2025 strict-OOS individual-arm results; 2026 is confirmation only",
        "combos": {arm: list(spec) for arm, spec in COMBOS.items()},
        "target": "source-aligned parent-policy net bps; specialists predict net-bps residual versus paired frozen MC1 mean",
        "admission": "dual BCF/current >= +50 bps; BCF mapped EV priority",
        "portfolio": "fixed global 7x/10%-slot, 2-new, 8-concurrent, 80%-wallet auction; invalid outcomes excluded before capacity",
        "dynamic_manifest_sha256": base._sha256(args.dynamic.resolve() / "run_manifest.json"),
        "assessment_manifest_sha256": base._sha256(args.assessment.resolve() / "run_manifest.json"),
        "individual_screen_manifest_sha256": base._sha256(args.individual.resolve() / "run_manifest.json"),
        "no_exchange_calls": True,
    }
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamic", type=Path, default=base.DYNAMIC)
    parser.add_argument("--assessment", type=Path, default=base.ASSESSMENT)
    parser.add_argument(
        "--individual", type=Path,
        default=base.ROOT / "data_perp/artifacts/causal_exotic_dynamics_mc1_ablation_2025oof_2026confirm_20260831_v1",
    )
    parser.add_argument("--out", type=Path, default=OUT)
    print(run(parser.parse_args()))


if __name__ == "__main__":
    main()
