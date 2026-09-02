#!/usr/bin/env python3
"""Audit a compact liquidity-transition panel against its causal contract.

The audit is intentionally schema/coverage based: it confirms which fields
are physically available for inference, distinguishes L2-event proxies from
executed-trade data, and verifies that the requested future supervision labels
are present without ever reading a target into a feature selection rule.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


INFERENCE_FAMILIES: dict[str, tuple[str, ...]] = {
    "l2_snapshot_and_actual_cost": (
        "spread_bps", "book_cost_for_actual_position_bps",
        "bid_depth_10bps", "ask_depth_10bps", "bid_depth_25bps", "ask_depth_25bps",
        "bid_depth_50bps", "ask_depth_50bps", "bid_depth_100bps", "ask_depth_100bps",
        "book_imbalance_25bps", "book_imbalance_50bps", "microprice_minus_mid_bps",
    ),
    "l2_transition": (
        "spread_bps_change_1m", "spread_bps_change_3m", "spread_bps_change_5m",
        "bid_depth_50bps_change_1m", "bid_depth_50bps_change_3m", "bid_depth_50bps_change_5m",
        "sell_book_cost_bps_n500_change_1m", "sell_book_cost_bps_n500_change_3m", "sell_book_cost_bps_n500_change_5m",
        "bid_cancel_rate_50bps", "ask_cancel_rate_50bps",
        "bid_replenishment_rate_50bps", "ask_replenishment_rate_50bps",
        "bid_replenishment_failure_rate_50bps", "book_flow_imbalance_50bps",
        "book_update_intensity_50bps",
    ),
    "asset_state": (
        "ret_1m", "ret_3m", "ret_5m", "ret_10m", "ret_15m", "ret_30m",
        "drawdown_5m", "drawdown_15m", "drawdown_30m",
        "realized_vol_5m", "realized_vol_15m", "realized_vol_30m",
        "realized_vol_acceleration_5v30",
    ),
    "market_state": (
        "btc_ret_1m", "btc_ret_5m", "btc_ret_15m",
        "market_return_1m", "market_return_5m", "market_return_15m",
        "asset_minus_market_ret_5m", "market_downside_vol_5m", "market_breadth_positive_5m",
        "rolling_median_spread_bps_30m", "rolling_median_bid_depth_50bps_30m", "liquidity_rank",
    ),
    "existing_causal_oi_volume_volatility": (
        "mkt_oi_chg_accel_1h", "mkt_oi_flush_z_30d", "mkt_oi_dispersion_1h",
        "volume_percentile", "volume_trend_48", "volume_entropy_24",
        "prior_volatility", "q_tail_width__volatility_zscore", "mkt_rv_4h",
    ),
    "per_minute_trade_recap": (
        "trade_quote_volume", "trade_intensity", "sell_order_flow_imbalance",
        "volume_ratio_5m", "volume_ratio_15m",
        "position_to_quote_volume_1m", "position_to_quote_volume_5m", "position_to_quote_volume_15m",
    ),
}

TRADE_RECAP_FIELDS = (
    "trade_quote_volume", "sell_order_flow_imbalance", "trade_intensity", "volume_ratio_5m", "volume_ratio_15m",
    "position_to_quote_volume_1m", "position_to_quote_volume_5m", "position_to_quote_volume_15m",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--retention-profile",
        choices=("full_recap", "orderbook_only"),
        default="full_recap",
        help="Whether per-minute executed-trade aggregates are contractually retained.",
    )
    args = parser.parse_args()
    paths = sorted(args.panel_root.rglob("surface.parquet"))
    if not paths:
        raise FileNotFoundError(f"no panel partitions below {args.panel_root}")
    frames = [pd.read_parquet(path) for path in paths]
    panel = pd.concat(frames, ignore_index=True, copy=False)
    available = set(panel.columns)
    coverage_rows: list[dict[str, object]] = []
    families: dict[str, dict[str, object]] = {}
    for family, fields in INFERENCE_FAMILIES.items():
        present = [field for field in fields if field in available]
        missing = [field for field in fields if field not in available]
        below_coverage: list[str] = []
        families[family] = {"requested": list(fields), "present": present, "missing": missing}
        for field in present:
            non_null = float(panel[field].notna().mean())
            if non_null < .80:
                below_coverage.append(field)
            coverage_rows.append({
                "family": family,
                "field": field,
                "non_null_fraction": non_null,
                "finite_fraction": float(pd.to_numeric(panel[field], errors="coerce").replace([float("inf"), float("-inf")], pd.NA).notna().mean()),
            })
        families[family]["below_minimum_coverage"] = below_coverage
    labels: dict[str, bool] = {}
    for horizon in (1, 5, 10, 15, 30):
        labels[f"spread_delta_{horizon}m"] = f"spread_widening_{horizon}m" in available
        # The declared $500 actual position equals the n500 order-book grid.
        labels[f"book_cost_delta_{horizon}m_at_static_position_500"] = f"deterioration_sell_{horizon}m_n500" in available
    required_families = [
        family for family in INFERENCE_FAMILIES
        if args.retention_profile != "orderbook_only" or family != "per_minute_trade_recap"
    ]
    retained_contract_complete = all(
        not families[family]["missing"] and not families[family]["below_minimum_coverage"]
        for family in required_families
    )
    report = {
        "schema": "ares.liquidity_transition_contract_audit.v1",
        "panel_root": str(args.panel_root),
        "rows": int(len(panel)),
        "partitions": [str(path) for path in paths],
        "inference_feature_families": families,
        "retention_profile": args.retention_profile,
        "retained_required_families": required_families,
        "retained_contract_complete": retained_contract_complete,
        "policy_unavailable_families": (
            ["per_minute_trade_recap"] if args.retention_profile == "orderbook_only" else []
        ),
        "supervision_labels": labels,
        "raw_trade_retention": {
            "aggregate_fields": list(TRADE_RECAP_FIELDS),
            "aggregate_fields_present": [field for field in TRADE_RECAP_FIELDS if field in available],
            "individual_prints_retained": False,
            "reason": (
                "order-book-only retention: individual prints and per-minute executed-trade aggregates are not retained"
                if args.retention_profile == "orderbook_only"
                else "only per-minute trade aggregates are permitted; raw individual trade prints remain pruned"
            ),
        },
        "causality": {
            "features": "completed L2 minute plus trailing/history-only context; cross-asset values are complete observed universe at the same decision minute",
            "labels": "future L2 terminal states are present only for offline supervision and excluded from evaluator feature groups",
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(coverage_rows).sort_values(["family", "field"]).to_parquet(args.out_dir / "feature_coverage.parquet", index=False)
    (args.out_dir / "feature_contract_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "rows": report["rows"],
        "all_inference_families_complete": all(
            not value["missing"] and not value["below_minimum_coverage"] for value in families.values()
        ),
        "retained_contract_complete": retained_contract_complete,
        "all_supervision_labels_present": all(labels.values()),
        "per_minute_trade_recap_present": bool(report["raw_trade_retention"]["aggregate_fields_present"]),
    }, indent=2))


if __name__ == "__main__":
    main()
