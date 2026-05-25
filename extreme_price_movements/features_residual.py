from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


EPS = 1e-12


RESIDUAL_FEATURE_KEYS = [
    "ret4h_bench_resid",
    "ret24h_bench_resid",
    "ret48h_bench_resid",
    "ret4h_peer_resid",
    "ret24h_peer_resid",
    "rv_24h_peer_resid",
    "vol_z_peer_resid",
    "rvol_z_peer_resid",
    "amihud_z_peer_resid",
    "liquidity_ratio_peer_resid",
    "dist_vwap_norm_mkt_resid",
    "dist_ema_fast_mkt_resid",
    "trend_pct_mkt_resid",
    "dist_vwap_norm_ts_resid",
    "dist_ema_fast_ts_resid",
    "rsi_ts_resid",
    "flow_persistence_ts_resid",
    "excess_6h_ts_resid",
    "atr_expansion_ts_resid",
    "coherence_24_ts_resid",
    "overext_surprise",
    "blowoff_risk_surprise",
    "exh_qual_surprise",
    "spike_score_surprise",
    "grind_score_surprise",
    "chop_score_surprise",
    "basis_pct_mkt_resid",
    "funding_per_hour_mkt_resid",
    "fund_abs_z_mkt_resid",
    "basis_fund_div_mkt_resid",
    "xasset_funding_ts_resid",
    "xasset_funding_peer_resid",
    "funding_1d_chg_ts_resid",
    "funding_1d_chg_peer_resid",
    "oi_chg_8h_mkt_resid",
    "oi_rel_vol_8h_peer_resid",
    "oi_chg_8h_robust_z_peer_resid",
    "asset_minus_mkt_oi_1d_ts_resid",
    "asset_minus_mkt_oi_7d_ts_resid",
    "asset_minus_mkt_oi_1d_peer_resid",
    "asset_minus_mkt_oi_7d_peer_resid",
    "squeeze_prob_mkt_resid",
    "ob_pressure_mkt_resid",
    "ob_spread_mkt_resid",
    "ob_depth_mkt_resid",
    "ob_imbalance_mkt_resid",
    "xasset_ob_pressure_ts_resid",
    "xasset_ob_pressure_peer_resid",
    "xasset_ob_liquidity_ts_resid",
    "xasset_ob_liquidity_peer_resid",
    "volume_price_corr_ts_resid",
    "path_efficiency_24_ts_resid",
    "entry_quality_composite_ts_resid",
]


RESIDUAL_LOOKBACKS = {
    "ret4h_bench_resid": 96,
    "ret24h_bench_resid": 240,
    "ret48h_bench_resid": 480,
    "ret4h_peer_resid": 1,
    "ret24h_peer_resid": 1,
    "rv_24h_peer_resid": 1,
    "vol_z_peer_resid": 1,
    "rvol_z_peer_resid": 1,
    "amihud_z_peer_resid": 1,
    "liquidity_ratio_peer_resid": 1,
    "dist_vwap_norm_mkt_resid": 96,
    "dist_ema_fast_mkt_resid": 96,
    "trend_pct_mkt_resid": 96,
    "dist_vwap_norm_ts_resid": 96,
    "dist_ema_fast_ts_resid": 96,
    "rsi_ts_resid": 96,
    "flow_persistence_ts_resid": 96,
    "excess_6h_ts_resid": 96,
    "atr_expansion_ts_resid": 96,
    "coherence_24_ts_resid": 96,
    "overext_surprise": 96,
    "blowoff_risk_surprise": 96,
    "exh_qual_surprise": 96,
    "spike_score_surprise": 96,
    "grind_score_surprise": 96,
    "chop_score_surprise": 96,
    "basis_pct_mkt_resid": 240,
    "funding_per_hour_mkt_resid": 240,
    "fund_abs_z_mkt_resid": 240,
    "basis_fund_div_mkt_resid": 240,
    "xasset_funding_ts_resid": 240,
    "xasset_funding_peer_resid": 1,
    "funding_1d_chg_ts_resid": 240,
    "funding_1d_chg_peer_resid": 1,
    "oi_chg_8h_mkt_resid": 240,
    "oi_rel_vol_8h_peer_resid": 1,
    "oi_chg_8h_robust_z_peer_resid": 1,
    "asset_minus_mkt_oi_1d_ts_resid": 240,
    "asset_minus_mkt_oi_7d_ts_resid": 480,
    "asset_minus_mkt_oi_1d_peer_resid": 1,
    "asset_minus_mkt_oi_7d_peer_resid": 1,
    "squeeze_prob_mkt_resid": 240,
    "ob_pressure_mkt_resid": 168,
    "ob_spread_mkt_resid": 168,
    "ob_depth_mkt_resid": 168,
    "ob_imbalance_mkt_resid": 168,
    "xasset_ob_pressure_ts_resid": 168,
    "xasset_ob_pressure_peer_resid": 1,
    "xasset_ob_liquidity_ts_resid": 168,
    "xasset_ob_liquidity_peer_resid": 1,
    "volume_price_corr_ts_resid": 96,
    "path_efficiency_24_ts_resid": 480,
    "entry_quality_composite_ts_resid": 96,
}


LEGACY_RESIDUAL_ALIASES = {
    "rsi_z": "rsi_ts_resid",
    "dist_ema_fast_z": "dist_ema_fast_ts_resid",
    "dist_vwap_norm_z": "dist_vwap_norm_ts_resid",
    "flow_persistence_z": "flow_persistence_ts_resid",
    "excess_6h_z": "excess_6h_ts_resid",
    "atr_expansion_z": "atr_expansion_ts_resid",
    "coherence_24_z": "coherence_24_ts_resid",
    "dist_vwap_resid": "dist_vwap_norm_mkt_resid",
    "dist_ema_fast_resid": "dist_ema_fast_mkt_resid",
    "trend_pct_resid": "trend_pct_mkt_resid",
}


def residual_feature_names(include_legacy_aliases: bool = True) -> list[str]:
    names = list(RESIDUAL_FEATURE_KEYS)
    if include_legacy_aliases:
        names.extend(LEGACY_RESIDUAL_ALIASES)
    return list(dict.fromkeys(names))


def _as_frame(value: object) -> pd.DataFrame | None:
    if isinstance(value, pd.DataFrame) and not value.empty:
        return value.astype(np.float32)
    return None


def _robust_ts_resid(x: pd.DataFrame, window: int) -> pd.DataFrame:
    min_periods = max(8, min(window // 4, window))
    median = x.rolling(window, min_periods=min_periods).median().shift(1)
    mad = (x - median).abs().rolling(window, min_periods=min_periods).median().shift(1)
    scale = (mad * 1.4826).clip(lower=1e-6)
    return ((x - median) / scale).clip(-5.0, 5.0).astype(np.float32)


def _rolling_beta_resid(
    x: pd.DataFrame,
    factor: pd.Series,
    window: int,
    *,
    standardize: bool,
) -> pd.DataFrame:
    factor = factor.reindex(x.index).astype(float)
    min_periods = max(8, min(window // 4, window))
    f_mean = factor.rolling(window, min_periods=min_periods).mean().shift(1)
    f_dm = factor - f_mean
    f_var = (f_dm * f_dm).rolling(window, min_periods=min_periods).mean().shift(1)

    out = pd.DataFrame(index=x.index, columns=x.columns, dtype=np.float32)
    for col in x.columns:
        s = x[col].astype(float)
        s_mean = s.rolling(window, min_periods=min_periods).mean().shift(1)
        cov = ((s - s_mean) * f_dm).rolling(window, min_periods=min_periods).mean().shift(1)
        beta = cov / (f_var + EPS)
        resid = s - beta * factor
        if standardize:
            rv = resid.rolling(window, min_periods=min_periods).std(ddof=0).shift(1)
            resid = resid / (rv.clip(lower=1e-6) + EPS)
        out[col] = resid.clip(-5.0, 5.0).astype(np.float32)
    return out


def _peer_resid(x: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    peer_median = x.median(axis=1, skipna=True)
    peer_mad = x.sub(peer_median, axis=0).abs().median(axis=1, skipna=True)
    coverage = x.notna().sum(axis=1)
    scale = (peer_mad * 1.4826).where(coverage >= min_n).clip(lower=1e-6)
    out = x.sub(peer_median, axis=0).div(scale, axis=0)
    return out.clip(-5.0, 5.0).astype(np.float32)


def _rolling_rank_surprise(x: pd.DataFrame, window: int) -> pd.DataFrame:
    min_periods = max(8, min(window // 4, window))

    def rank_last(values: np.ndarray) -> float:
        cur = values[-1]
        hist = values[:-1]
        hist = hist[np.isfinite(hist)]
        if not np.isfinite(cur) or len(hist) < min_periods:
            return np.nan
        return float((hist <= cur).mean() * 2.0 - 1.0)

    out = x.rolling(window + 1, min_periods=min_periods + 1).apply(
        rank_last, raw=True
    )
    return out.clip(-5.0, 5.0).astype(np.float32)


def _basket_median_factor(x: pd.DataFrame, basket: list[str] | None = None) -> pd.Series:
    by_base: dict[str, str] = {}
    for col in x.columns:
        text = str(col)
        base = text.split("/", 1)[0] if "/" in text else text
        by_base.setdefault(base.upper(), text)
    cols: list[str] = []
    for raw in basket or []:
        text = str(raw)
        if text in x.columns:
            cols.append(text)
            continue
        base = text.split("/", 1)[0] if "/" in text else text
        mapped = by_base.get(base.upper())
        if mapped:
            cols.append(mapped)
    cols = list(dict.fromkeys(cols))
    if not cols:
        cols = list(x.columns)
    return x[cols].median(axis=1, skipna=True).astype(np.float32)


def _first_frame(feats: Mapping[str, object], names: tuple[str, ...]) -> pd.DataFrame | None:
    for name in names:
        frame = _as_frame(feats.get(name))
        if frame is not None:
            return frame
    return None


def _frame_has_tail_variation(frame: pd.DataFrame, min_std: float = 1e-8) -> bool:
    if frame.empty:
        return False
    tail = frame.iloc[max(0, len(frame) - 512) :]
    if tail.empty:
        tail = frame
    std = tail.std(axis=0, skipna=True).to_numpy(dtype=float)
    return bool(np.isfinite(std).any() and np.nanmax(std) > min_std)


def _first_informative_frame(
    feats: Mapping[str, object], names: tuple[str, ...]
) -> pd.DataFrame | None:
    first: pd.DataFrame | None = None
    for name in names:
        frame = _as_frame(feats.get(name))
        if frame is None:
            continue
        if first is None:
            first = frame
        if _frame_has_tail_variation(frame):
            return frame
    return first


def _resolve_benchmark_column(columns: pd.Index, cfg: Mapping[str, object]) -> str | None:
    candidates: list[str] = []
    for key in ("primary_benchmark", "benchmark_1", "bench1_symbol"):
        val = cfg.get(key)
        if val:
            candidates.append(str(val))
    candidates.extend(["BTC/USDC", "BTCUSDC", "BTC/USDT", "BTCUSDT", "BTC/USD", "BTCUSD"])

    def normalize_symbol(value: object) -> str:
        text = str(value).upper()
        # Kraken USD-settled swaps are represented as BTC/USD:USD. For benchmark
        # matching, the settlement suffix should not distinguish the instrument
        # from the BTC/USD return series itself.
        if ":" in text:
            text = text.split(":", 1)[0]
        return text.replace("/", "").replace("-", "").replace("_", "")

    normalized = {normalize_symbol(c): str(c) for c in columns}
    direct = {str(c): str(c) for c in columns}
    for candidate in candidates:
        if candidate in direct:
            return direct[candidate]
        key = normalize_symbol(candidate)
        if key in normalized:
            return normalized[key]
    for quote in ("USDC", "USDT", "USD"):
        key = f"BTC{quote}"
        if key in normalized:
            return normalized[key]
    return None


def _market_trend_factor(
    feats: Mapping[str, pd.DataFrame],
    mkt_gates: pd.DataFrame | None,
    index: pd.Index,
) -> pd.Series | None:
    if isinstance(mkt_gates, pd.DataFrame) and {"mkt_trend", "mkt_rv"}.issubset(
        mkt_gates.columns
    ):
        trend = pd.to_numeric(mkt_gates["mkt_trend"].reindex(index), errors="coerce")
        rv = pd.to_numeric(mkt_gates["mkt_rv"].reindex(index), errors="coerce")
        return (trend / (rv * np.sqrt(24.0) + EPS)).replace([np.inf, -np.inf], np.nan)
    ret = _as_frame(feats.get("ret24h"))
    if ret is not None:
        return ret.median(axis=1, skipna=True)
    return None


def add_residual_features(
    feats: dict[str, pd.DataFrame],
    mkt_gates: pd.DataFrame | None,
    cfg: Mapping[str, object] | None = None,
) -> set[str]:
    """Add causal residualised features in-place and return keys to skip transform."""
    cfg = cfg or {}
    skip: set[str] = set()
    if not feats:
        return skip

    first = next((v for v in feats.values() if isinstance(v, pd.DataFrame)), None)
    if first is None:
        return skip
    index = first.index
    columns = first.columns
    basket = [str(x) for x in cfg.get("market_basket", [])] if cfg else []

    def add(name: str, value: pd.DataFrame | None, *, replace: bool = False) -> None:
        if value is None:
            return
        if name in feats and not replace:
            existing = _as_frame(feats.get(name))
            if existing is not None and _frame_has_tail_variation(existing):
                skip.add(name)
                return
        feats[name] = value.reindex(index=index, columns=columns).astype(np.float32)
        skip.add(name)

    # Benchmark beta-neutral returns.
    for name, base, horizon in (
        ("ret4h_bench_resid", "ret4h", 4),
        ("ret24h_bench_resid", "ret24h", 24),
        ("ret48h_bench_resid", "ret48h", 48),
    ):
        x = _as_frame(feats.get(base))
        bench_col = _resolve_benchmark_column(x.columns, cfg) if x is not None else None
        if x is not None and bench_col in x.columns:
            factor = x[bench_col]
            add(
                name,
                _rolling_beta_resid(
                    x,
                    factor,
                    RESIDUAL_LOOKBACKS[name],
                    standardize=True,
                ),
            )

    # Peer-neutral local state.
    for name, base in (
        ("ret4h_peer_resid", "ret4h"),
        ("ret24h_peer_resid", "ret24h"),
        ("rv_24h_peer_resid", "rv_24h"),
        ("vol_z_peer_resid", "vol_z"),
        ("rvol_z_peer_resid", "rvol_z"),
        ("amihud_z_peer_resid", "amihud_z"),
        ("liquidity_ratio_peer_resid", "liquidity_ratio"),
        ("oi_rel_vol_8h_peer_resid", "oi_rel_vol_8h"),
        ("oi_chg_8h_robust_z_peer_resid", "oi_chg_8h_robust_z"),
        ("xasset_funding_peer_resid", "xasset_asset_minus_mkt_funding"),
        ("funding_1d_chg_peer_resid", "funding_1d_chg_z_90d"),
        ("asset_minus_mkt_oi_1d_peer_resid", "asset_minus_mkt_oi_1d_z_90d"),
        ("asset_minus_mkt_oi_7d_peer_resid", "asset_minus_mkt_oi_7d_z_180d"),
        ("xasset_ob_pressure_peer_resid", "xasset_asset_minus_mkt_ob_pressure_z_24h"),
        ("xasset_ob_liquidity_peer_resid", "xasset_ob_liquidity_divergence_z_24h"),
    ):
        x = _as_frame(feats.get(base))
        if x is not None:
            add(name, _peer_resid(x))

    # Market/basket beta-neutral residuals.
    mkt_factor = _market_trend_factor(feats, mkt_gates, index)
    for name, base in (
        ("dist_vwap_norm_mkt_resid", "dist_vwap_norm"),
        ("dist_ema_fast_mkt_resid", "dist_ema_fast"),
        ("trend_pct_mkt_resid", "trend_pct"),
    ):
        x = _as_frame(feats.get(base))
        if x is not None and mkt_factor is not None:
            add(name, _rolling_beta_resid(x, mkt_factor, RESIDUAL_LOOKBACKS[name], standardize=False))

    for name, base in (
        ("basis_pct_mkt_resid", "basis_pct_z"),
        ("funding_per_hour_mkt_resid", "funding_per_hour_z"),
        ("fund_abs_z_mkt_resid", "fund_abs_z_14d"),
        ("basis_fund_div_mkt_resid", "basis_fund_div_z"),
        ("oi_chg_8h_mkt_resid", "oi_chg_8h"),
        ("squeeze_prob_mkt_resid", "squeeze_prob"),
        ("ob_spread_mkt_resid", "ob_spread_z_24h"),
    ):
        x = _as_frame(feats.get(base))
        if x is not None:
            factor = _basket_median_factor(x, basket)
            add(name, _rolling_beta_resid(x, factor, RESIDUAL_LOOKBACKS[name], standardize=False))

    ob_pressure = _first_informative_frame(
        feats,
        (
            "ob_microprice_premium_bps",
            "ob_flow_notional_imbalance_1h",
            "ob_book_pressure_l10",
            "ob_imb_10bps",
            "ob_imb_l10",
        ),
    )
    if ob_pressure is not None:
        add(
            "ob_pressure_mkt_resid",
            _rolling_beta_resid(
                ob_pressure,
                _basket_median_factor(ob_pressure, basket),
                RESIDUAL_LOOKBACKS["ob_pressure_mkt_resid"],
                standardize=False,
            ),
        )

    ob_depth = _first_frame(
        feats, ("ob_depth_z_25bps", "ob_depth_usd_l20_z", "ob_depth_l20_to_qv_z_7d")
    )
    if ob_depth is not None:
        add(
            "ob_depth_mkt_resid",
            _rolling_beta_resid(
                ob_depth,
                _basket_median_factor(ob_depth, basket),
                RESIDUAL_LOOKBACKS["ob_depth_mkt_resid"],
                standardize=False,
            ),
        )

    ob_imbalance = _first_informative_frame(
        feats,
        (
            "ob_imb_10bps",
            "ob_microprice_premium_bps",
            "ob_flow_notional_imbalance_1h",
            "ob_l10_imbalance",
            "ob_imb_l10",
        ),
    )
    if ob_imbalance is not None:
        add(
            "ob_imbalance_mkt_resid",
            _rolling_beta_resid(
                ob_imbalance,
                _basket_median_factor(ob_imbalance, basket),
                RESIDUAL_LOOKBACKS["ob_imbalance_mkt_resid"],
                standardize=False,
            ),
        )

    # Own-history surprises.
    for name, base in (
        ("dist_vwap_norm_ts_resid", "dist_vwap_norm"),
        ("dist_ema_fast_ts_resid", "dist_ema_fast"),
        ("rsi_ts_resid", "rsi"),
        ("flow_persistence_ts_resid", "flow_persistence"),
        ("excess_6h_ts_resid", "excess_6h"),
        ("atr_expansion_ts_resid", "atr_expansion"),
        ("coherence_24_ts_resid", "coherence_24"),
        ("overext_surprise", "overext"),
        ("blowoff_risk_surprise", "blowoff_risk"),
        ("exh_qual_surprise", "exh_qual"),
        ("spike_score_surprise", "spike_score"),
        ("grind_score_surprise", "grind_score"),
        ("chop_score_surprise", "chop_score"),
        ("volume_price_corr_ts_resid", "volume_price_corr_10h"),
        ("xasset_funding_ts_resid", "xasset_asset_minus_mkt_funding"),
        ("funding_1d_chg_ts_resid", "funding_1d_chg_z_90d"),
        ("asset_minus_mkt_oi_1d_ts_resid", "asset_minus_mkt_oi_1d_z_90d"),
        ("asset_minus_mkt_oi_7d_ts_resid", "asset_minus_mkt_oi_7d_z_180d"),
        ("xasset_ob_pressure_ts_resid", "xasset_asset_minus_mkt_ob_pressure_z_24h"),
        ("xasset_ob_liquidity_ts_resid", "xasset_ob_liquidity_divergence_z_24h"),
    ):
        x = _as_frame(feats.get(base))
        if x is not None:
            add(name, _robust_ts_resid(x, RESIDUAL_LOOKBACKS[name]))

    entry_quality = _as_frame(feats.get("entry_quality_composite"))
    if entry_quality is not None:
        add(
            "entry_quality_composite_ts_resid",
            _rolling_rank_surprise(
                entry_quality, RESIDUAL_LOOKBACKS["entry_quality_composite_ts_resid"]
            ),
        )

    path_base = _as_frame(feats.get("path_efficiency_24"))
    ret1 = _as_frame(feats.get("ret1h"))
    ret24 = _as_frame(feats.get("ret24h"))
    if ret1 is not None and ret24 is not None:
        abs_path = ret1.abs().rolling(24, min_periods=8).sum()
        path_candidate = (ret24.abs() / (abs_path + EPS)).clip(0.0, 1.0).astype(np.float32)
        if path_base is None:
            path_base = path_candidate
        else:
            path_base = path_base.copy()
            tail = path_base.iloc[max(0, len(path_base) - 512) :]
            flat_cols = tail.std(axis=0, skipna=True).fillna(0.0) <= 1e-8
            if flat_cols.any():
                cols = [c for c, is_flat in flat_cols.items() if bool(is_flat)]
                path_base.loc[:, cols] = path_candidate.reindex(
                    index=path_base.index, columns=cols
                )
    if path_base is not None:
        add(
            "path_efficiency_24_ts_resid",
            _robust_ts_resid(path_base, RESIDUAL_LOOKBACKS["path_efficiency_24_ts_resid"]),
            replace=True,
        )

    for alias, canonical in LEGACY_RESIDUAL_ALIASES.items():
        if canonical in feats:
            feats[alias] = feats[canonical].astype(np.float32)
            skip.add(alias)

    return skip
