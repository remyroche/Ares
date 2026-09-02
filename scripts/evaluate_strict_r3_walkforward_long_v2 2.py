#!/usr/bin/env python3
"""Join outcomes after scoring and report long-only global-tail economics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extreme_price_movements.strict_r3_canonical_current import (  # noqa: E402
    SCHEMA,
)
from extreme_price_movements.strict_r3_canonical_v2 import (  # noqa: E402
    require_single_geometry_hash,
)


OUTCOMES = {
    "tp6_sl4_h12": {
        "valid": "h12_label_valid",
        "gross": "h12_tp6_sl4_gross_bps",
        "net": "h12_tp6_sl4_net_bps",
    },
    "frozen_15m_trailing": {
        "valid": "policy_path_valid",
        "gross": "policy_gross_bps",
        "net": "policy_net_bps",
    },
}

SCORE_STAGES = ("base_rank42", "upstream", "raw_correctness_demote", "final_score")


def _summary(block: pd.DataFrame, *, gross: str, net: str) -> dict[str, float | int]:
    value = pd.to_numeric(block[net], errors="coerce")
    gross_value = pd.to_numeric(block[gross], errors="coerce")
    return {
        "valid_outcome_rows": int(value.notna().sum()),
        "gross_bps_per_trade": float(gross_value.mean()) if gross_value.notna().any() else np.nan,
        "net_bps_per_trade": float(value.mean()) if value.notna().any() else np.nan,
        "net_sum_bps": float(value.sum(min_count=1)) if value.notna().any() else np.nan,
        "positive_trade_rate": float(value.gt(0).mean()) if value.notna().any() else np.nan,
    }


def _period_rows(
    selected: pd.DataFrame,
    *,
    tail: float,
    outcome: str,
    valid: str,
    gross: str,
    net: str,
    period: str,
) -> list[dict[str, object]]:
    if period == "month":
        key = selected["__decision_ts__"].dt.strftime("%Y-%m")
    else:
        key = selected["__decision_ts__"].dt.to_period("W-SUN").astype(str)
    rows: list[dict[str, object]] = []
    for name, block in selected.assign(__period__=key).groupby("__period__", sort=True):
        eligible = block.loc[block[valid].fillna(False).astype(bool)].copy()
        rows.append({
            "period_type": period, "period": str(name),
            "tail_fraction": tail, "outcome": outcome,
            "selected_score_rows": int(len(block)),
            "valid_coverage": float(len(eligible) / len(block)) if len(block) else np.nan,
            **_summary(eligible, gross=gross, net=net),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--source-panel", type=Path, required=True)
    parser.add_argument(
        "--policy-outcomes", type=Path,
        help="Optional independently replayed fixed-policy outcomes overriding source-panel policy columns.",
    )
    parser.add_argument(
        "--exact-root", type=Path,
        help="Optional exact-label artifact whose long parts override matching TP6/SL4 rows.",
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--evaluation-start", default="2025-01-01")
    parser.add_argument("--evaluation-end", default="2026-08-01")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(f"immutable output already exists: {args.out_dir}")
    predictions = pd.read_parquet(args.predictions)
    geometry_hash = require_single_geometry_hash(predictions)
    h12_columns = [
        "candidate_id",
        "h12_label_valid", "h12_label_available_ts",
        "h12_tp6_sl4_gross_bps", "h12_tp6_sl4_net_bps",
    ]
    policy_columns = [
        "candidate_id", "policy_path_valid", "policy_gross_bps", "policy_net_bps",
        "policy_exit_bar_15m", "policy_exit_reason", "policy_entry_price", "policy_exit_price",
    ]
    outcomes = pd.read_parquet(args.source_panel, columns=h12_columns)
    if args.exact_root is not None:
        parts = sorted(args.exact_root.glob("parts/month=*/side=long.parquet"))
        if not parts:
            raise FileNotFoundError(f"no exact long parts under {args.exact_root}")
        exact = pd.concat([
            pd.read_parquet(path, columns=[
                "candidate_id", "label_valid", "__label_available_at__",
                "t4_tp6_sl4_gross_bps", "t4_tp6_sl4_net_bps",
            ]) for path in parts
        ], ignore_index=True).rename(columns={
            "label_valid": "h12_label_valid",
            "__label_available_at__": "h12_label_available_ts",
            "t4_tp6_sl4_gross_bps": "h12_tp6_sl4_gross_bps",
            "t4_tp6_sl4_net_bps": "h12_tp6_sl4_net_bps",
        })
        if exact["candidate_id"].duplicated().any():
            raise ValueError("exact override has duplicate candidate IDs")
        base = outcomes.set_index("candidate_id")
        override = exact.set_index("candidate_id")
        common = base.index.intersection(override.index)
        base.loc[common, override.columns] = override.loc[common, override.columns]
        outcomes = base.reset_index()
    if args.policy_outcomes is not None:
        policy = pd.read_parquet(args.policy_outcomes, columns=policy_columns)
        # The independent replay label is resolved 12 hours after decision.
        identity_time = pd.read_parquet(
            args.policy_outcomes, columns=["candidate_id", "__decision_ts__"],
        )
        policy = policy.merge(identity_time, on="candidate_id", validate="one_to_one")
        policy["policy_label_available_ts"] = pd.to_datetime(policy["__decision_ts__"], utc=True) + pd.Timedelta(hours=12)
        policy = policy.drop(columns="__decision_ts__")
    else:
        policy = pd.read_parquet(
            args.source_panel,
            columns=["candidate_id", "policy_label_available_ts", *policy_columns[1:]],
        )
    if policy["candidate_id"].duplicated().any():
        raise ValueError("policy outcome source has duplicate candidate IDs")
    outcomes = outcomes.merge(policy, on="candidate_id", how="left", validate="one_to_one")
    if outcomes["candidate_id"].duplicated().any():
        raise ValueError("source panel has duplicate candidate IDs")
    ledger = predictions.merge(outcomes, on="candidate_id", how="left", validate="one_to_one")
    ledger["__decision_ts__"] = pd.to_datetime(ledger["__decision_ts__"], utc=True)
    for column in ("policy_label_available_ts", "h12_label_available_ts"):
        ledger[column] = pd.to_datetime(ledger[column], utc=True)
    start = pd.to_datetime(args.evaluation_start, utc=True)
    end = pd.to_datetime(args.evaluation_end, utc=True)
    ledger = ledger.loc[
        ledger["__decision_ts__"].ge(start) & ledger["__decision_ts__"].lt(end)
    ].copy()
    if not ledger["side_name"].astype(str).str.lower().eq("long").all():
        raise ValueError("long-only evaluator received another side")
    for name, spec in OUTCOMES.items():
        valid = ledger[spec["valid"]].fillna(False).astype(bool)
        difference = (
            pd.to_numeric(ledger.loc[valid, spec["gross"]], errors="coerce")
            - pd.to_numeric(ledger.loc[valid, spec["net"]], errors="coerce")
        )
        if len(difference) and not np.allclose(difference, 100.0, atol=1e-4):
            raise AssertionError(f"{name} does not apply the declared 100-bps cost exactly once")
    ranked = ledger.sort_values(["final_score", "candidate_id"], ascending=[False, True], kind="stable")
    global_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    for tail in (0.01, 0.02, 0.03):
        count = max(1, int(math.ceil(tail * len(ranked))))
        selected = ranked.head(count).copy()
        for outcome, spec in OUTCOMES.items():
            eligible = selected.loc[selected[spec["valid"]].fillna(False).astype(bool)].copy()
            global_rows.append({
                "tail_fraction": tail, "outcome": outcome,
                "population_score_rows": int(len(ranked)),
                "selected_score_rows": int(len(selected)),
                "valid_coverage": float(len(eligible) / len(selected)),
                "trades_per_calendar_day": float(len(eligible) / max((end - start).days, 1)),
                **_summary(eligible, gross=spec["gross"], net=spec["net"]),
            })
            for period in ("month", "week"):
                period_rows.extend(_period_rows(
                    selected, tail=tail, outcome=outcome,
                    valid=spec["valid"], gross=spec["gross"], net=spec["net"],
                    period=period,
                ))
    args.out_dir.mkdir(parents=True)
    ledger.to_parquet(args.out_dir / "scored_label_ledger.parquet", index=False, compression="zstd")
    global_metrics = pd.DataFrame(global_rows)
    periods = pd.DataFrame(period_rows)
    global_metrics.to_parquet(args.out_dir / "global_tail_metrics.parquet", index=False)
    periods.loc[periods["period_type"].eq("month")].to_parquet(
        args.out_dir / "monthly_tail_metrics.parquet", index=False,
    )
    periods.loc[periods["period_type"].eq("week")].to_parquet(
        args.out_dir / "weekly_tail_metrics.parquet", index=False,
    )
    stage_rows: list[dict[str, object]] = []
    stage_month_rows: list[dict[str, object]] = []
    for stage in SCORE_STAGES:
        if stage not in ledger:
            continue
        stage_ranked = ledger.loc[np.isfinite(pd.to_numeric(ledger[stage], errors="coerce"))].sort_values(
            [stage, "candidate_id"], ascending=[False, True], kind="stable",
        )
        for tail in (0.01, 0.02, 0.03):
            count = max(1, int(math.ceil(tail * len(stage_ranked))))
            selected = stage_ranked.head(count).copy()
            for outcome, spec in OUTCOMES.items():
                eligible = selected.loc[selected[spec["valid"]].fillna(False).astype(bool)].copy()
                stage_rows.append({
                    "score_stage": stage, "tail_fraction": tail, "outcome": outcome,
                    "population_score_rows": int(len(stage_ranked)),
                    "selected_score_rows": int(len(selected)),
                    "valid_coverage": float(len(eligible) / len(selected)),
                    **_summary(eligible, gross=spec["gross"], net=spec["net"]),
                })
                for row in _period_rows(
                    selected, tail=tail, outcome=outcome,
                    valid=spec["valid"], gross=spec["gross"], net=spec["net"],
                    period="month",
                ):
                    stage_month_rows.append({"score_stage": stage, **row})
    pd.DataFrame(stage_rows).to_parquet(
        args.out_dir / "score_stage_global_tail_metrics.parquet", index=False,
    )
    pd.DataFrame(stage_month_rows).to_parquet(
        args.out_dir / "score_stage_monthly_tail_metrics.parquet", index=False,
    )
    manifest = {
        "schema": f"{SCHEMA}_long_global_tail_evaluation",
        "side_name": "long", "score": "final policy-correctness same-model CDF42",
        "ranking": "one pooled global ranking; not per timestamp and not an executable threshold",
        "tails": [0.01, 0.02, 0.03],
        "evaluation_start": start.isoformat(), "evaluation_end_exclusive": end.isoformat(),
        "score_population_rows": len(ledger),
        "diagnostic_score_stages": list(SCORE_STAGES),
        "future_paths_joined_after_scoring": True,
        "invalid_outcomes_encoded_as_failures": False,
        "cost_bps_once": 100.0,
        "policy_outcome_source": str(args.policy_outcomes or args.source_panel),
        "exact_outcome_source": str(args.exact_root or args.source_panel),
        "geometry_bundle_sha256": geometry_hash,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(global_metrics.to_json(orient="records"))


if __name__ == "__main__":
    main()
