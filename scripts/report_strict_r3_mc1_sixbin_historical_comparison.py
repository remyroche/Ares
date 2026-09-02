#!/usr/bin/env python3
"""Matched MC1 six-bin historical portability and validation report.

This is an offline reporting producer.  It never reads or changes any live
state, model bundle, admission setting, or executor.  It compares the fixed
six-field MC1 control with the fixed nine-field agreement-geometry contract,
using the exact six-bin ordinal HPO receipt that reproduces the stored 2025
checkpoint bit-for-bit.

2024 is a backwards portability diagnostic: the model specification was
selected in 2025, but every 2024 prediction remains strict-prequential.  The
first usable month is June because earlier months lack a 28-day calibration
reserve plus the minimum supervised fit support.  April--July 2026 uses the
already-created strict OOF checkpoints; August is deliberately reported as
unavailable because the immutable source ledger ends on 31 July.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_strict_r3_mc1_admission_ablation_v2.py"
PREPARED = ROOT / "data_perp/artifacts/strict_r3_mc1_admission_ablation_v2_prepared_20260816_v1"
HPO = ROOT / "data_perp/artifacts/strict_r3_mc1_admission_ablation_v2_hpo_20260816_v4/hpo_winners.json"
FORWARD = ROOT / "data_perp/artifacts/strict_r3_mc1_admission_ablation_v3_forward_selection_20260816_v1"
CORE_FEATURES = (
    "final_score", "base_rank42", "conditional_consensus_rank", "upstream",
    "ordinary_shadow_consensus_rank", "correctness_rank",
)
UPDATED_FEATURES = (*CORE_FEATURES, "agr_rank_iqr", "agr_frac_far_10sd", "agr_head_mean")


def load_runner():
    spec = importlib.util.spec_from_file_location("mc1_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load MC1 runner")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves postponed annotations through sys.modules during
    # module execution; register this reusable producer before executing it.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_checkpoint_blocks(root: Path, arm: str, starts: Iterable[str]) -> pd.DataFrame:
    frames = []
    for start in starts:
        path = root / arm / f"fold_{start}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def scalar_or_nan(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if len(clean) else float("nan")


def period_detail(module, frame: pd.DataFrame, arm: str, period: str) -> tuple[dict[str, object], pd.DataFrame]:
    """Metrics with one auction run over the whole supplied chronology."""
    frame = frame.loc[frame.pool_consensus30].copy()
    frame["month"] = frame["__decision_ts__"].dt.strftime("%Y-%m")
    frame["week"] = frame["__decision_ts__"].dt.strftime("%G-W%V")
    replay = module.auction(frame, "mapper_expected_bps", "final_score")
    replay["month"] = replay["__decision_ts__"].dt.strftime("%Y-%m")
    replay["week"] = replay["__decision_ts__"].dt.strftime("%G-W%V")
    valid = frame.policy_path_valid.fillna(False).astype(bool) & frame.policy_net_bps.notna()
    admitted = pd.to_numeric(frame.mapper_expected_bps, errors="coerce").ge(50.0)
    rows: list[dict[str, object]] = []
    for month, group in frame.groupby("month", sort=True):
        valid_group = valid.loc[group.index]
        admitted_group = admitted.loc[group.index] & valid_group
        accepted_rows = replay.loc[replay.month.eq(month) & replay.portfolio_accepted.astype(bool)]
        realised = accepted_rows.loc[
            accepted_rows.policy_path_valid.fillna(False).astype(bool) & accepted_rows.policy_net_bps.notna()
        ]
        admission_rows = group.loc[admitted_group]
        ic_values = admission_rows.groupby("__decision_ts__", sort=False).apply(
            lambda x: module._safe_spearman(x.mapper_expected_bps, x.policy_net_bps), include_groups=False,
        ).dropna()
        if len(admission_rows) >= 8 and admission_rows.mapper_expected_bps.nunique() > 1:
            slope, intercept = np.polyfit(
                admission_rows.mapper_expected_bps.to_numpy(float),
                admission_rows.policy_net_bps.to_numpy(float), 1,
            )
        else:
            slope = intercept = float("nan")
        rows.append({
            "period": period, "arm": arm, "month": month,
            "candidate_rows": int(len(group)), "valid_rows": int(valid_group.sum()),
            "admitted_valid_rows": int(admitted_group.sum()), "portfolio_selected_rows": int(len(accepted_rows)),
            "realised_selected_rows": int(len(realised)),
            "selected_label_coverage": float(len(realised) / len(accepted_rows)) if len(accepted_rows) else float("nan"),
            "portfolio_net_ev_bps": scalar_or_nan(realised.policy_net_bps),
            "portfolio_net_sum_bps": float(realised.policy_net_bps.sum()) if len(realised) else float("nan"),
            "within_admission_ic": scalar_or_nan(ic_values),
            "calibration_slope": float(slope), "calibration_intercept": float(intercept),
            "positive_week_fraction": float(
                (realised.groupby("week").policy_net_bps.mean() > 0.0).mean()
            ) if len(realised) else float("nan"),
            "worst_week_bps": float(realised.groupby("week").policy_net_bps.mean().min()) if len(realised) else float("nan"),
        })
    aggregate = {"period": period, "arm": arm, "scope": "all", **module.metric_row(frame, "mapper_expected_bps", "final_score")}
    return aggregate, pd.DataFrame(rows)


def weekly_detail(module, frame: pd.DataFrame, arm: str, period: str) -> pd.DataFrame:
    frame = frame.loc[frame.pool_consensus30].copy()
    frame["week"] = frame["__decision_ts__"].dt.strftime("%G-W%V")
    replay = module.auction(frame, "mapper_expected_bps", "final_score")
    replay["week"] = replay["__decision_ts__"].dt.strftime("%G-W%V")
    valid = frame.policy_path_valid.fillna(False).astype(bool) & frame.policy_net_bps.notna()
    admitted = pd.to_numeric(frame.mapper_expected_bps, errors="coerce").ge(50.0)
    rows: list[dict[str, object]] = []
    for week, group in frame.groupby("week", sort=True):
        valid_group = valid.loc[group.index]
        accepted_rows = replay.loc[replay.week.eq(week) & replay.portfolio_accepted.astype(bool)]
        realised = accepted_rows.loc[
            accepted_rows.policy_path_valid.fillna(False).astype(bool) & accepted_rows.policy_net_bps.notna()
        ]
        rows.append({
            "period": period, "arm": arm, "week": week,
            "candidate_rows": int(len(group)), "admitted_valid_rows": int((admitted.loc[group.index] & valid_group).sum()),
            "portfolio_selected_rows": int(len(accepted_rows)), "realised_selected_rows": int(len(realised)),
            "portfolio_net_ev_bps": scalar_or_nan(realised.policy_net_bps),
            "portfolio_net_sum_bps": float(realised.policy_net_bps.sum()) if len(realised) else float("nan"),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, default=None,
                        help="Existing immutable 2024 block receipt used by --stage report.")
    parser.add_argument("--stage", choices=("prediction", "report", "all"), default="all")
    parser.add_argument("--arm", choices=("core6", "updated9", "both"), default="both")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=args.stage != "all")
    m = load_runner()
    params = json.loads(HPO.read_text())["pool_consensus30|ordinal"]
    starts_2024 = [pd.Timestamp(value, tz="UTC") for value in (
        "2024-06-01", "2024-07-01", "2024-08-01", "2024-09-01",
        "2024-10-01", "2024-11-01", "2024-12-01",
    )]

    if args.stage in ("prediction", "all"):
        # Cached panel is target-free for scoring; labels become eligible only
        # in chronological_split via policy_label_available_ts.  Process the
        # arms independently and persist individual monthly blocks, avoiding
        # a second large model matrix coexisting with the first arm's output.
        panel = pd.read_parquet(PREPARED / "candidate_static_panel.parquet")
        panel["__decision_ts__"] = pd.to_datetime(panel["__decision_ts__"], utc=True)
        panel["policy_label_available_ts"] = pd.to_datetime(panel["policy_label_available_ts"], utc=True)
        requested = (("core6", CORE_FEATURES), ("updated9", UPDATED_FEATURES))
        for arm, features in requested:
            if args.arm not in ("both", arm):
                continue
            block_dir = args.out_dir / f"{arm}_2024_blocks"
            if block_dir.exists():
                raise FileExistsError(f"refuse to overwrite immutable block receipt: {block_dir}")
            paths = m.prequential_prediction_blocks(
                panel, features, "ordinal", params, "pool_consensus30", starts_2024,
                1, 180_000, False, (), state=None, block_output_dir=block_dir,
            )
            (args.out_dir / f"{arm}_2024_blocks.json").write_text(json.dumps([str(path) for path in paths], indent=2) + "\n")
            gc.collect()
        del panel
        gc.collect()
        if args.stage == "prediction":
            return

    # These are already strict OOF checkpoints, generated once during the
    # selected-contract test.  Read only April--July; the immutable ledger
    # contains no August decision rows to evaluate.
    starts_2026 = ("20260401T000000Z", "20260701T000000Z")
    aggregate: list[dict[str, object]] = []
    monthly: list[pd.DataFrame] = []
    weekly: list[pd.DataFrame] = []
    prediction_dir = args.prediction_dir or args.out_dir
    sources = (
        ("2024", "core6", prediction_dir / "core6_2024_blocks"),
        ("2024", "updated9", prediction_dir / "updated9_2024_blocks"),
        ("2026", "core6", FORWARD / "core6_2026_validation"),
        ("2026", "updated9", FORWARD / "d11_2026_validation"),
    )
    for period, arm, root in sources:
        if period == "2024":
            paths = sorted(root.glob("fold_*.parquet"))
            if len(paths) != len(starts_2024):
                raise FileNotFoundError(f"incomplete 2024 block receipt for {arm}: {root}")
            prediction = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
        else:
            prediction = read_checkpoint_blocks(root.parent, root.name, starts_2026)
        overall, per_month = period_detail(m, prediction, arm, period)
        aggregate.append(overall)
        monthly.append(per_month)
        weekly.append(weekly_detail(m, prediction, arm, period))
        del prediction
        gc.collect()

    aggregate_table = pd.DataFrame(aggregate)
    monthly_table = pd.concat(monthly, ignore_index=True)
    weekly_table = pd.concat(weekly, ignore_index=True)
    # Explicitly record unavailable August rather than letting a partial or
    # unresolved label set imply a negative/zero result.
    monthly_table = pd.concat([monthly_table, pd.DataFrame([
        {"period": "2026", "arm": arm, "month": "2026-08", "source_status": "unavailable: immutable source ends 2026-07-31; no resolved historical ledger"}
        for arm in ("core6", "updated9")
    ])], ignore_index=True)

    controls = aggregate_table.loc[aggregate_table.arm.eq("core6")].set_index("period")
    delta_rows: list[dict[str, object]] = []
    for row in aggregate_table.loc[aggregate_table.arm.eq("updated9")].itertuples(index=False):
        control = controls.loc[row.period]
        values = {"period": row.period, "arm": "updated9_minus_core6"}
        for field in ("portfolio_selected_rows", "portfolio_net_ev_bps", "portfolio_net_sum_bps", "within_admission_ic", "calibration_slope", "worst_month_bps", "worst_week_bps", "positive_month_fraction"):
            values[f"delta_{field}"] = float(getattr(row, field) - control[field])
        delta_rows.append(values)
    deltas = pd.DataFrame(delta_rows)

    aggregate_table.to_parquet(args.out_dir / "aggregate_metrics.parquet", index=False)
    monthly_table.to_parquet(args.out_dir / "monthly_metrics.parquet", index=False)
    weekly_table.to_parquet(args.out_dir / "weekly_metrics.parquet", index=False)
    deltas.to_parquet(args.out_dir / "delta_vs_core6.parquet", index=False)
    aggregate_table.to_csv(args.out_dir / "aggregate_metrics.csv", index=False)
    monthly_table.to_csv(args.out_dir / "monthly_metrics.csv", index=False)
    weekly_table.to_csv(args.out_dir / "weekly_metrics.csv", index=False)
    deltas.to_csv(args.out_dir / "delta_vs_core6.csv", index=False)
    manifest = {
        "schema": "strict_r3_mc1_sixbin_historical_comparison_v1",
        "purpose": "offline comparison only; no live bundle or inference semantics changed",
        "target": "six-bin ordinal canonical policy_net_bps; entry next hour, 15m SL3/trailing, H12 timeout, cost once",
        "admission": "mapper expected policy net >= +50 bps; final_score auction; long-only two-new / eight-concurrent proxy",
        "baseline": list(CORE_FEATURES), "updated_features": list(UPDATED_FEATURES),
        "hpo_params": params,
        "2024": "June--December, strict prequential backwards portability diagnostic; specification selected later in 2025",
        "2026": "April--July strict OOF validation checkpoints; August unavailable because source ledger ends 2026-07-31",
        "inputs": {"prepared_panel": str(PREPARED), "hpo_receipt": str(HPO), "forward_receipt": str(FORWARD), "prediction_receipt": str(prediction_dir)},
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(aggregate_table.to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
