#!/usr/bin/env python3
"""Audit causal predictor support for the O3-v2 market-dynamics label families.

This is deliberately a *feature-inventory* producer, not a label or model
trainer.  It freezes the candidate-side inputs that may accompany each future
market-dynamics target.  Every input comes from the existing target-free
prequential ledger; future labels, outcomes, barrier state, and path fields
are prohibited.  A family is eligible for downstream label testing only when
at least five distinct causal inputs have >=90% coverage and non-zero robust
variation on the audit population.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEDGER = ROOT / "data_perp/artifacts/strict_r3_schema_v2_prequential_ledger_targetfree_long_2024_2026_raw15m_strictfull_20260812_v1/prequential_stack_ledger.parquet"
SCHEMA = "strict_r3_o3v2_market_dynamics_input_audit_v1"
MIN_COVERAGE = 0.90
MAX_PER_FAMILY = 10

# These are intentionally groups of decision-time observables rather than
# labels.  Inputs are drawn from the frozen 120-field causal contract used by
# the auxiliary heads, and not inferred from their names at runtime.
FAMILY_CANDIDATES: dict[str, tuple[str, ...]] = {
    "trend_persistence": (
        "mkt_ret_15m", "mkt_ret_4h", "mkt_return_accel_1h", "mkt_close_location_1h",
        "median_alt_minus_btc", "ret4h_peer_resid", "excess_6h_ts_resid",
        "pct_assets_up_15m", "pct_assets_up_4h", "mkt_lower_wick_ratio_1h",
    ),
    "stretch_reversion": (
        "mkt_close_location_1h", "mkt_lower_wick_ratio_1h", "distance_to_resistance_atr",
        "bars_to_resistance_daily_donchian", "bars_to_resistance_daily_vwap",
        "market_breadth_drawdown_from_6h_max", "market_breadth_recovery_from_24h_min",
        "breadth_recovery_from_6h_min", "pct_assets_bullish_reversal_candle",
        "pct_assets_large_lower_wick",
    ),
    "volatility_regime": (
        "mkt_rv_4h", "prior_volatility", "price_rv_15d_robust_z", "xs_dispersion__rvol_z",
        "xs_dispersion__volatility_zscore", "xs_dispersion__vol_z", "q_tail_asym__vol_z_4h",
        "q_upper_tail__bars_in_high_vol_state_log_norm", "q_tail_width__volatility_zscore",
        "rvol_z_peer_resid",
    ),
    "breadth_participation": (
        "negative_breadth_pct", "breadth_chg_15m", "pct_assets_up_15m", "pct_assets_up_4h",
        "pct_assets_new_low_24h", "pct_assets_bullish_reversal_candle", "pct_assets_large_lower_wick",
        "market_breadth_drawdown_from_6h_max", "market_breadth_recovery_from_24h_min",
        "breadth_recovery_from_6h_min",
    ),
    "cross_sectional_dispersion": (
        "xs_dispersion__vol_z", "xs_dispersion__rvol_z", "xs_dispersion__efficiency_ratio_20",
        "xs_dispersion__amihud_illiq", "xs_dispersion__ffd_amihud_04",
        "xs_dispersion__volume_zscore_48h", "xs_dispersion__asset_minus_mkt_oi_1d",
        "mkt_oi_dispersion_1h", "mkt_oi_dispersion_24h", "state_spectral_eig_gap_1_2",
    ),
    "dependence_common_factor": (
        "cross_asset_corr_1h", "cross_asset_downside_corr_4h", "state_spectral_eig_condition",
        "state_spectral_eig_gap_1_2", "state_spectral_eig_top3_share", "eig_effective_rank__open_interest",
        "xs_dispersion__rvol_z_peer_resid", "xs_dispersion__vol_z_peer_resid",
        "median_alt_minus_btc", "volume_price_corr_ts_resid",
    ),
    "leadership_rotation": (
        "median_alt_minus_btc", "ret4h_peer_resid", "excess_6h_ts_resid", "pct_assets_up_4h",
        "pct_assets_up_15m", "xs_dispersion__efficiency_ratio_20", "xs_dispersion__volume_zscore_48h",
        "q_iqr__ret48h_bench_resid", "q_lower_tail__vol_z_peer_resid",
        "q_lower_tail__xasset_ob_liquidity_peer_resid",
    ),
    "tail_stress": (
        "negative_breadth_pct", "cross_asset_downside_corr_4h", "liquidation_climax_score",
        "asset_minus_mkt_long_flush_intensity_4h", "mkt_oi_flush_z_30d",
        "mkt_pct_oi_drawdown_24h_lt_minus5pct", "pct_assets_price_down_oi_down_1h",
        "pct_assets_extreme_oi_drop_1h", "post_liquidation_rebound_score",
        "q_lower_tail__xasset_mkt_spread_bps",
    ),
    "volume_flow": (
        "volume_percentile", "q_lower_tail__volume_z_24", "q_tail_width__volume_z_12",
        "xs_dispersion__volume_zscore_48h", "volume_price_corr_ts_resid",
        "xs_dispersion__amihud_illiq", "xs_dispersion__ffd_amihud_04",
        "xs_dispersion__oi_to_volume_7d_z_180d", "mkt_ret_per_oi_change_1h",
        "mkt_ret_per_oi_change_4h",
    ),
    "structural_transition": (
        "state_spectral_eig_condition", "state_spectral_eig_gap_1_2", "state_spectral_eig_top3_share",
        "eig_effective_rank__open_interest", "mkt_return_accel_1h", "mkt_oi_chg_accel_1h",
        "breadth_chg_15m", "market_breadth_drawdown_from_6h_max", "mkt_rv_4h",
        "cross_asset_corr_1h",
    ),
    "leverage_positioning": (
        "mkt_oi_chg_15m", "mkt_oi_chg_accel_1h", "mkt_oi_flush_z_30d",
        "mkt_median_oi_recovery_fraction_24h", "mkt_pct_oi_chg_1h_rz_lt_minus1",
        "mkt_pct_oi_chg_4h_rz_lt_minus2", "mkt_pct_oi_drawdown_24h_lt_minus5pct",
        "q_lower_tail__oi_3d_x_funding", "q_lower_tail__oi_7d_x_funding",
        "xs_dispersion__funding_per_hour",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_exclusive(path: Path, payload: object) -> None:
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)


def run(*, ledger: Path, out: Path) -> Path:
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    candidates = tuple(dict.fromkeys(field for fields in FAMILY_CANDIDATES.values() for field in fields))
    available = set(pq.ParquetFile(ledger).schema_arrow.names)
    missing = sorted(set(candidates) - available)
    observed = sorted(set(candidates) & available)
    # The audited ledger is itself target-free.  The time filter is solely for
    # coverage estimation, not a model fit or any target selection.
    frame = pd.read_parquet(ledger, columns=["__decision_ts__", *observed])
    frame["__decision_ts__"] = pd.to_datetime(frame["__decision_ts__"], utc=True, errors="raise")
    frame = frame.loc[frame["__decision_ts__"].ge(pd.Timestamp("2025-01-01T00:00:00Z"))].copy()
    rows: list[dict[str, object]] = []
    eligible: dict[str, list[str]] = {}
    for family, fields in FAMILY_CANDIDATES.items():
        kept: list[str] = []
        for field in fields:
            if field not in frame:
                rows.append({"family": family, "feature": field, "present": False, "coverage": 0.0, "q05": np.nan, "q95": np.nan, "varies": False, "kept": False})
                continue
            values = pd.to_numeric(frame[field], errors="coerce")
            finite = values[np.isfinite(values)]
            coverage = float(len(finite) / len(frame)) if len(frame) else 0.0
            q05, q95 = (float(finite.quantile(.05)), float(finite.quantile(.95))) if len(finite) else (np.nan, np.nan)
            varies = bool(np.isfinite(q05) and np.isfinite(q95) and abs(q95 - q05) > 1e-12)
            keep = coverage >= MIN_COVERAGE and varies and len(kept) < MAX_PER_FAMILY
            if keep:
                kept.append(field)
            rows.append({"family": family, "feature": field, "present": True, "coverage": coverage, "q05": q05, "q95": q95, "varies": varies, "kept": keep})
        eligible[family] = kept
    audit = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=False)
    audit.to_parquet(out / "market_dynamics_feature_audit.parquet", index=False, compression="zstd")
    _write_json_exclusive(out / "market_dynamics_feature_blocks.json", {
        "schema": SCHEMA,
        "ledger": str(ledger.resolve()),
        "ledger_sha256": _sha256(ledger),
        "scope": "target-free causal inputs only; future market labels are not included",
        "minimum_coverage": MIN_COVERAGE,
        "max_features_per_family": MAX_PER_FAMILY,
        "eligible_feature_blocks": eligible,
        "families_with_adequate_support": sorted(name for name, fields in eligible.items() if len(fields) >= 5),
        "families_insufficient": sorted(name for name, fields in eligible.items() if len(fields) < 5),
        "missing_contract_fields": missing,
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(run(ledger=args.ledger.resolve(), out=args.out.resolve()))


if __name__ == "__main__":
    main()
