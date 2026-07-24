"""Import-light contract for causal market-regime transition features."""

from __future__ import annotations


MARKET_REGIME_CHANGE_SCHEMA_VERSION = 1
MARKET_REGIME_CHANGE_2D_HOURS = 48

# These are market-wide, pre-entry state variables.  Each source is already
# normalized or dimensionless before its transition geometry is calculated.
MARKET_REGIME_CHANGE_SOURCES: dict[str, str] = {
    "funding": "funding_1d_chg_ts_resid",
    "oi_contraction": "mkt_median_oi_chg_4h_rz",
    "negative_breadth": "negative_breadth_pct",
    "eth_correlation": "corr_eth_24h",
    "btc_alt_relative_strength": "median_alt_minus_btc",
    "short_covering": "short_covering_score_market",
    "flush_recovery": "flush_recovery_state",
}

MARKET_REGIME_CHANGE_OPERATORS: tuple[str, ...] = (
    "delta_1h",
    "acceleration_1h",
    "cumulative_change_2d",
    "sign_flip_1h",
)


def market_regime_change_feature_name(source_name: str, operator: str) -> str:
    return f"mkt_regime_change__{source_name}__{operator}"


MARKET_REGIME_CHANGE_FEATURE_KEYS = [
    market_regime_change_feature_name(source_name, operator)
    for source_name in MARKET_REGIME_CHANGE_SOURCES
    for operator in MARKET_REGIME_CHANGE_OPERATORS
]


__all__ = [
    "MARKET_REGIME_CHANGE_2D_HOURS",
    "MARKET_REGIME_CHANGE_FEATURE_KEYS",
    "MARKET_REGIME_CHANGE_OPERATORS",
    "MARKET_REGIME_CHANGE_SCHEMA_VERSION",
    "MARKET_REGIME_CHANGE_SOURCES",
    "market_regime_change_feature_name",
]
