from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from extreme_price_movements.config import (
    REGIME_ADAPTOR_ASSET_FEATURE_KEYS,
    REGIME_ADAPTOR_COMBINATION_GRID,
    REGIME_ADAPTOR_EBM_CONSOLIDATED_FEATURE_KEYS,
    REGIME_ADAPTOR_FEATURE_ORDER,
    REGIME_ADAPTOR_FUNDING_FEATURE_KEYS,
    REGIME_ADAPTOR_GLOBAL_BAD_RATE_THRESHOLD,
    REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS,
    REGIME_ADAPTOR_LGBM_CLASSIFIER_PARAMS,
    REGIME_ADAPTOR_ASSET_BAD_RATE_THRESHOLD,
    REGIME_ADAPTOR_OBJECTIVE_WEIGHTS,
    REGIME_ADAPTOR_ORDERBOOK_FEATURE_KEYS,
    REGIME_ADAPTOR_RATIO_CLIPS,
    REGIME_ADAPTOR_ROLLING_PRIOR_FEATURE_KEYS,
    REGIME_ADAPTOR_STRATEGY_ASSET_FEATURE_KEYS,
)

try:
    from lightgbm import LGBMClassifier, early_stopping
except Exception:  # pragma: no cover - optional runtime dependency fallback.
    LGBMClassifier = None  # type: ignore[assignment]
    early_stopping = None  # type: ignore[assignment]

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
except Exception:  # pragma: no cover - optional runtime dependency fallback.
    optuna = None  # type: ignore[assignment]
    MedianPruner = None  # type: ignore[assignment]
    TPESampler = None  # type: ignore[assignment]

EPS = 1e-9
PREDICTION_DISAGREEMENT_THRESHOLD = 0.20
SYMBOL_SHRINK_K = 200.0
PSI_RARITY_THRESHOLD = 1.50
MAX_DISTRIBUTION_FEATURES = 256
REGIME_ADAPTOR_INFERENCE_DISABLED_MODES = {
    "",
    "0",
    "false",
    "no",
    "off",
    "none",
    "disabled",
}
NEW_REGIME_FEATURES = (
    "base_model_prediction",
    "meta_model_prediction",
    "ebm_raw_prediction",
    "ebm_en_prediction",
    "ebm_uncertainty_weighted_prediction",
    "ebm_en_adjustment_abs",
    "ebm_uncertainty_adjustment_abs",
    "ebm_prediction_dispersion",
    "ebm_prediction_brittleness",
    "meta_recent_brier_shrunk",
    "meta_recent_brier_global",
    "abs_base_meta_diff",
    "signed_base_meta_diff",
    "abs_base_meta_diff_3d",
    "abs_base_meta_diff_7d",
    "abs_base_meta_diff_15d",
    "signed_base_meta_diff_3d",
    "signed_base_meta_diff_7d",
    "signed_base_meta_diff_15d",
    "base_meta_diff_ewm_3d",
    "base_meta_disagreement_rate_3d",
    "base_meta_disagreement_rate_7d",
    "base_models_pred_std",
    "base_models_pred_range",
    "base_models_pred_iqr",
    "base_models_disagreement_rate_3d",
    "base_models_disagreement_rate_7d",
    "base_models_disagreement_rate_15d",
    "max_abs_zscore",
    "mean_abs_zscore",
    "num_features_outside_p01_p99",
    "num_features_outside_p05_p95",
    "p95_PSI",
    "mean_PSI",
    "num_features_PSI_above_threshold",
    "pca_reconstruction_error",
    "missing_count",
    "stale_feature_count",
    "symbol_recent_utility_shrunk",
    "symbol_recent_bad_rate_shrunk",
    "symbol_sample_count_log",
    "symbol_liquidity_rank",
    "symbol_vol_rank",
)

ROLLING_REGIME_HORIZONS_DAYS = (3, 5)
ROLLING_REGIME_DEFAULT_BLEND_WEIGHTS = {3: 0.6, 5: 0.4}
ROLLING_REGIME_BLEND_GRID = ((0.8, 0.2), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6))
REGIME_FEATURE_ORDER = list(
    dict.fromkeys(list(REGIME_ADAPTOR_FEATURE_ORDER) + list(NEW_REGIME_FEATURES))
)
GLOBAL_REGIME_FEATURES = tuple(REGIME_ADAPTOR_GLOBAL_FEATURE_KEYS)
CROSS_ASSET_REGIME_FEATURES = GLOBAL_REGIME_FEATURES
ASSET_REGIME_FEATURES = tuple(REGIME_ADAPTOR_ASSET_FEATURE_KEYS)
STRATEGY_ASSET_REGIME_FEATURES = tuple(REGIME_ADAPTOR_STRATEGY_ASSET_FEATURE_KEYS)
FUNDING_REGIME_FEATURES = tuple(REGIME_ADAPTOR_FUNDING_FEATURE_KEYS)
ORDERBOOK_REGIME_FEATURES = tuple(REGIME_ADAPTOR_ORDERBOOK_FEATURE_KEYS)
EBM_CONSOLIDATED_REGIME_FEATURES = tuple(REGIME_ADAPTOR_EBM_CONSOLIDATED_FEATURE_KEYS)
ROLLING_PRIOR_REGIME_FEATURES = tuple(REGIME_ADAPTOR_ROLLING_PRIOR_FEATURE_KEYS)
REQUIRED_LIVE_BAD_REGIME_COLUMNS = (
    "p_bad_regime_global_3d",
    "p_bad_regime_global_5d",
    "p_bad_regime_asset_3d",
    "p_bad_regime_asset_5d",
)
REGIME_OBJECTIVE_WEIGHTS = dict(REGIME_ADAPTOR_OBJECTIVE_WEIGHTS)
GLOBAL_BAD_RATE_THRESHOLD = float(REGIME_ADAPTOR_GLOBAL_BAD_RATE_THRESHOLD)
ASSET_BAD_RATE_THRESHOLD = float(REGIME_ADAPTOR_ASSET_BAD_RATE_THRESHOLD)
REGIME_RATIO_CLIPS = {k: tuple(v) for k, v in REGIME_ADAPTOR_RATIO_CLIPS.items()}
ROLLING_REGIME_LGBM_PARAMS = dict(REGIME_ADAPTOR_LGBM_CLASSIFIER_PARAMS)


def normalize_market_mode(market_mode: str | None = None) -> str:
    import os

    mode = str(market_mode or os.environ.get("EPM_MARKET_MODE", "spot")).strip().lower()
    if mode in {"perp", "perps", "future", "futures"}:
        return "perps"
    return "spot"


def market_file_path(path: Path, market_mode: str | None = None) -> Path:
    mode = normalize_market_mode(market_mode)
    stem = path.stem
    for suffix in ("_spot", "_perps", "_perp"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return path.with_name(f"{stem}_{mode}{path.suffix}")


def _allow_legacy_market_fallback() -> bool:
    import os

    return str(
        os.environ.get("EPM_ALLOW_LEGACY_MARKET_FALLBACK", "")
    ).strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def resolve_market_file_path(path: Path, market_mode: str | None = None) -> Path:
    mode_path = market_file_path(path, market_mode)
    if mode_path.exists():
        return mode_path
    if _allow_legacy_market_fallback():
        return path
    return mode_path

FEATURE_CANDIDATES: Dict[str, Tuple[str, ...]] = {
    "rv_24h": (
        "rv_24h",
        "realized_volatility_24h",
        "ffd_rv_24h_04",
        "ffd_rv_24_04",
        "range_24h_pct",
        "range_norm_24",
        "vol_regime_z",
        "vol_z24",
        "vol_z",
        "volatility_zscore",
        "atr_12_15m",
        "atr_pct",
        "atr_pct_base",
    ),
    "rv1": (
        "rv_1h",
        "rv_2h",
        "realized_volatility_1h",
        "realized_vol_15m_realized_vol_2h",
        "rv_1h_proxy",
        "ret1h_abs",
        "range_norm_12",
        "range_12h_pct",
        "z_r_12",
    ),
    "rv4": (
        "rv_4h",
        "rv_6h",
        "ffd_rv_6h_04",
        "realized_volatility_4h",
        "rv_4h_proxy",
        "range_norm_24",
        "z_r_24",
    ),
    "signed_adx": (
        "signed_adx",
        "adx_zscore",
        "adx_14",
        "adx_10",
        "adx_7",
        "trend_slope_48h",
        "trend_slope_120h",
        "regime_trend_score",
        "trend_regime",
    ),
    "trend_sign": ("trend_regime", "trend_24h", "ret24h", "slope"),
    "dist_ema_fast": (
        "dist_ema_fast",
        "dist_ema_fast_base",
        "dist_ema_fast_z",
        "ffd_dist_ema_fast_04",
        "dist_ema20_atr",
        "z_dist_ema_24",
    ),
    "dist_ema_slow": (
        "dist_ema_slow",
        "dist_ema_slow_base",
        "ffd_dist_ema_slow_04",
        "dist_ema50_atr",
        "dist_ema200_atr",
    ),
    "dist_vwap": (
        "loc_vwap_dev_z_24",
        "dist_vwap_norm",
        "dist_vwap_norm_z",
        "loc_vwap_dev_z_48",
        "z_vwap_24",
        "z_vwap_12",
        "z_dist_vwap_24",
        "dist_vwap_24_atr",
        "dist_vwap_12_atr",
        "dist_weekly_vwap",
    ),
    "prior_day_low": (
        "dist_prior_day_low",
        "loc_prev_day_low",
        "loc_prev_day_range_pos_24",
    ),
    "prior_day_high": (
        "dist_prior_day_high",
        "loc_prev_day_high",
        "loc_prev_day_range_pos_24",
    ),
    "rvol_z": ("rvol_z", "volume_zscore_48h", "volume_z_24", "regime_liquidity_score"),
    "entropy_24h": (
        "spectral_entropy_ret_24",
        "perm_entropy_ret_24",
        "shannon_entropy_ret_16",
        "direction_entropy_20",
        "regime_transition_entropy_48h",
    ),
    "asset_volume_30d": (
        "asset_volume_30d",
        "asset_vol_level",
        "volume",
        "quote_volume",
        "dollar_volume",
        "volume_24h",
        "volume_percentile",
    ),
    "asset_atr_30d": (
        "asset_atr_30d",
        "asset_atr_level",
        "atr_pct",
        "atr_pct_base",
        "atr_12_15m",
        "rv_24h",
        "realized_volatility_24h",
    ),
    "ebm_unc_logodds_var": ("oof_ebm_unc_logodds_var", "ebm_unc_logodds_var"),
    "ebm_unc_pi_width": ("oof_ebm_unc_pi_width", "ebm_unc_pi_width"),
    "ebm_unc_entropy_mean": ("oof_ebm_unc_entropy_mean", "ebm_unc_entropy_mean"),
    "ebm_unc_entropy_std": ("oof_ebm_unc_entropy_std", "ebm_unc_entropy_std"),
    "ebm_unc_conflict_norm": (
        "oof_ebm_unc_conflict_norm",
        "ebm_unc_conflict_norm",
    ),
    "ebm_unc_proximity_min": (
        "oof_ebm_unc_proximity_min",
        "ebm_unc_proximity_min",
    ),
    "ebm_unc_support_mean": ("oof_ebm_unc_support_mean", "ebm_unc_support_mean"),
    "ebm_unc_support_min": ("oof_ebm_unc_support_min", "ebm_unc_support_min"),
    "ebm_unc_concentration": (
        "oof_ebm_unc_concentration",
        "ebm_unc_concentration",
    ),
    "ebm_unc_sign_ratio": ("oof_ebm_unc_sign_ratio", "ebm_unc_sign_ratio"),
    "ebm_unc_interaction_share": (
        "oof_ebm_unc_interaction_share",
        "ebm_unc_interaction_share",
    ),
    "ebm_unc_gap50rel": ("oof_ebm_unc_gap50rel", "ebm_unc_gap50rel"),
    "ebm_unc_support_adjusted_uncertainty": (
        "oof_ebm_unc_support_adjusted_uncertainty",
        "ebm_unc_support_adjusted_uncertainty",
    ),
    "ebm_unc_uncertainty_weight": (
        "oof_ebm_unc_uncertainty_weight",
        "ebm_unc_uncertainty_weight",
    ),
    "ebm_unc_friction_weight": (
        "oof_ebm_unc_friction_weight",
        "ebm_unc_friction_weight",
    ),
}


# Canonical next-few-days bad-regime candidate mappings.  The canonical names
# are stable artifact keys; aliases let existing feature generators feed the
# layer without requiring a parallel subsystem.
FEATURE_CANDIDATES.update(
    {
        "market_breadth_24h": ("market_breadth_24h", "mkt_breadth_24h"),
        "market_breadth_7d": ("market_breadth_7d", "market_breadth_168h"),
        "market_breadth_15d": ("market_breadth_15d", "market_breadth_360h"),
        "cross_asset_return_dispersion_24h": (
            "cross_asset_return_dispersion_24h",
            "market_dispersion_24h",
            "cross_asset_return_dispersion",
            "xasset_return_dispersion",
        ),
        "cross_asset_return_dispersion_7d": (
            "cross_asset_return_dispersion_7d",
            "market_dispersion_7d",
            "xasset_return_dispersion_7d",
        ),
        "cross_asset_vol_dispersion_24h": (
            "cross_asset_vol_dispersion_24h",
            "cross_asset_vol_dispersion",
            "xasset_vol_dispersion",
            "rv_cross_asset_dispersion",
        ),
        "cross_asset_vol_dispersion_7d": (
            "cross_asset_vol_dispersion_7d",
            "xasset_vol_dispersion_7d",
        ),
        "cross_asset_vol_dispersion_15d": (
            "cross_asset_vol_dispersion_15d",
            "xasset_vol_dispersion_15d",
        ),
        "median_asset_rv_24h": ("median_asset_rv_24h", "xasset_median_rv_24h"),
        "median_asset_rv_7d": ("median_asset_rv_7d", "xasset_median_rv_7d"),
        "top_decile_asset_rv_24h": (
            "top_decile_asset_rv_24h",
            "xasset_top_decile_rv_24h",
        ),
        "top_decile_asset_rv_7d": (
            "top_decile_asset_rv_7d",
            "xasset_top_decile_rv_7d",
        ),
        "btc_eth_trend_proxy": (
            "btc_eth_trend_proxy",
            "btc_ret_24h",
            "eth_btc_ret_24h",
        ),
        "btc_eth_vol_proxy": ("btc_eth_vol_proxy", "btc_eth_rv_24h"),
        "cross_asset_correlation_7d": ("cross_asset_correlation_7d", "xasset_corr_7d"),
        "cross_asset_correlation_30d": (
            "cross_asset_correlation_30d",
            "xasset_corr_30d",
        ),
        "funding_rate_cross_asset_dispersion": (
            "funding_rate_cross_asset_dispersion",
            "funding_cross_asset_dispersion",
        ),
        "asset_funding_z": ("asset_funding_z", "funding_z", "funding_rate_z"),
        "asset_funding_side_alignment": (
            "asset_funding_side_alignment",
        ),
        "asset_funding_trend_alignment": (
            "asset_funding_trend_alignment",
            "funding_trend_alignment",
        ),
        "asset_funding_rate_abs_mean_7d": (
            "asset_funding_rate_abs_mean_7d",
            "funding_abs_z",
            "funding_rate_abs_mean",
        ),
        "asset_spread_proxy_p90_24h": (
            "asset_spread_proxy_p90_24h",
            "spread_proxy_p90_24h",
            "ob_spread_z_24h",
            "ob_spread_bps",
        ),
        "asset_spread_proxy_p90_96h": (
            "asset_spread_proxy_p90_96h",
            "spread_proxy_p90_96h",
            "ob_spread_z_24h",
            "ob_spread_bps",
        ),
        "asset_spread_proxy_p90_7d": (
            "asset_spread_proxy_p90_7d",
            "spread_proxy_p90_7d",
            "ob_spread_z_24h",
            "ob_spread_bps",
        ),
        "asset_spread_proxy_p90_15d": (
            "asset_spread_proxy_p90_15d",
            "spread_proxy_p90_15d",
            "ob_spread_z_24h",
            "ob_spread_bps",
        ),
        "asset_volume_depth_risk_p90_24h": (
            "asset_volume_depth_risk_p90_24h",
            "volume_depth_risk_p90_24h",
            "ob_depth_usd_l20_z",
            "ob_depth_usd_l20",
            "ob_top_liquidity_usd",
        ),
        "asset_volume_depth_risk_p90_96h": (
            "asset_volume_depth_risk_p90_96h",
            "volume_depth_risk_p90_96h",
            "ob_depth_usd_l20_z",
            "ob_depth_usd_l20",
            "ob_top_liquidity_usd",
        ),
        "asset_volume_depth_risk_p90_7d": (
            "asset_volume_depth_risk_p90_7d",
            "volume_depth_risk_p90_7d",
            "ob_depth_usd_l20_z",
            "ob_depth_usd_l20",
            "ob_top_liquidity_usd",
        ),
        "asset_volume_depth_risk_p90_15d": (
            "asset_volume_depth_risk_p90_15d",
            "volume_depth_risk_p90_15d",
            "ob_depth_usd_l20_z",
            "ob_depth_usd_l20",
            "ob_top_liquidity_usd",
        ),
        "asset_orderbook_imbalance_abs_mean_24h": (
            "asset_orderbook_imbalance_abs_mean_24h",
            "orderbook_imbalance_abs_mean_24h",
            "ob_imb_l1",
            "ob_imb_l10",
            "ob_wimb_l10",
            "ob_book_pressure_l10",
        ),
        "asset_orderbook_imbalance_abs_mean_96h": (
            "asset_orderbook_imbalance_abs_mean_96h",
            "orderbook_imbalance_abs_mean_96h",
            "ob_imb_l1",
            "ob_imb_l10",
            "ob_wimb_l10",
            "ob_book_pressure_l10",
        ),
        "asset_orderbook_imbalance_abs_mean_7d": (
            "asset_orderbook_imbalance_abs_mean_7d",
            "orderbook_imbalance_abs_mean_7d",
            "ob_imb_l1",
            "ob_imb_l10",
            "ob_wimb_l10",
            "ob_book_pressure_l10",
        ),
        "asset_orderbook_imbalance_abs_mean_15d": (
            "asset_orderbook_imbalance_abs_mean_15d",
            "orderbook_imbalance_abs_mean_15d",
            "ob_imb_l1",
            "ob_imb_l10",
            "ob_wimb_l10",
            "ob_book_pressure_l10",
        ),
        "asset_liquidity_stress_score_7d": (
            "asset_liquidity_stress_score_7d",
            "xasset_mkt_ob_stress",
        ),
    }
)
for _feat in REGIME_FEATURE_ORDER:
    FEATURE_CANDIDATES.setdefault(_feat, (_feat,))


@dataclass
class RegimeAdaptorFit:
    artifact: Dict[str, Any]
    fixed_diagnostics: pd.DataFrame
    adaptive_diagnostics: pd.DataFrame
    asset_diagnostics: pd.DataFrame
    metrics: pd.DataFrame
    regime_weight_oof: np.ndarray
    eligible_oof: np.ndarray
    deployment_score_oof: np.ndarray
    deployment_score_rank_oof: np.ndarray
    regime_utility_pred_15d_oof: Optional[np.ndarray] = None
    regime_utility_pred_30d_oof: Optional[np.ndarray] = None
    combined_regime_utility_oof: Optional[np.ndarray] = None
    regime_logit_offset_oof: Optional[np.ndarray] = None
    trust_score_oof: Optional[np.ndarray] = None
    trust_proba_oof: Optional[np.ndarray] = None


def safe_strategy_slug(strategy_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(strategy_id or "")).strip("_")
    return slug[:180] or "strategy"


def _coerce_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and np.isfinite(float(value)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enabled"}:
        return True
    if text in REGIME_ADAPTOR_INFERENCE_DISABLED_MODES:
        return False
    return None


def regime_adaptor_inference_enabled(
    runtime_cfg: Optional[Dict[str, Any]] = None,
    artifact: Optional[Dict[str, Any]] = None,
) -> bool:
    """Return whether a trained regime adaptor should be applied at inference.

    Research/training enablement and live inference enablement are intentionally
    separate so artifacts can keep diagnostics while deployment stays opt-in.
    """
    artifact = artifact or {}
    if not bool(artifact.get("enable_regime_adaptor", False)):
        return False

    cfg = runtime_cfg or {}
    nested = cfg.get("regime_adaptor", {}) if isinstance(cfg, dict) else {}
    if not isinstance(nested, dict):
        nested = {}

    explicit_cfg = None
    if isinstance(cfg, dict):
        explicit_cfg = _coerce_optional_bool(
            cfg.get("regime_adaptor.inference_enabled")
        )
        if explicit_cfg is None:
            explicit_cfg = _coerce_optional_bool(
                cfg.get("regime_adaptor_inference_enabled")
            )
    if explicit_cfg is None:
        explicit_cfg = _coerce_optional_bool(nested.get("inference_enabled"))
    if explicit_cfg is not None:
        return bool(explicit_cfg)

    mode = None
    if isinstance(cfg, dict):
        mode = cfg.get("regime_adaptor.inference_integration_mode")
    if mode is None:
        mode = nested.get("inference_integration_mode")
    if mode is not None:
        return str(mode).strip().lower() not in REGIME_ADAPTOR_INFERENCE_DISABLED_MODES

    explicit_artifact = _coerce_optional_bool(
        artifact.get("enable_regime_adaptor_inference")
    )
    if explicit_artifact is not None:
        return bool(explicit_artifact)

    if mode is None:
        mode = artifact.get("inference_integration_mode", "disabled")
    return str(mode).strip().lower() not in REGIME_ADAPTOR_INFERENCE_DISABLED_MODES


def _as_float_array(values: Any, n: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(arr) < n:
        out = np.full(n, np.nan, dtype=np.float64)
        out[: len(arr)] = arr
        return out
    return arr[:n].astype(np.float64, copy=False)


def _first_col(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def _col(df: pd.DataFrame, names: Sequence[str], n: int) -> Optional[np.ndarray]:
    name = _first_col(df, names)
    if name is None:
        return None
    return _as_float_array(df[name].values, n)


def _fill_numeric(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float64).copy()
    finite = np.isfinite(x)
    if finite.any():
        med = float(np.nanmedian(x[finite]))
    else:
        med = float(fill)
    x[~finite] = med
    return x.astype(np.float32)


def build_regime_feature_frame(
    feature_frame: pd.DataFrame,
    timestamps: Optional[Sequence[Any]] = None,
    symbols: Optional[Sequence[Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Map available training/live features to the regime adaptor contract."""
    n = len(feature_frame)
    out: Dict[str, np.ndarray] = {}
    mapping: Dict[str, Any] = {}

    rv24 = _col(feature_frame, FEATURE_CANDIDATES["rv_24h"], n)
    if rv24 is not None:
        out["rv_24h"] = _fill_numeric(rv24)
        mapping["rv_24h"] = _first_col(feature_frame, FEATURE_CANDIDATES["rv_24h"])

    rv1 = _col(feature_frame, FEATURE_CANDIDATES["rv1"], n)
    if rv1 is None and "ret1h" in feature_frame.columns:
        rv1 = np.abs(_as_float_array(feature_frame["ret1h"].values, n))
    if rv1 is not None and rv24 is not None:
        out["rv1_rv24"] = _fill_numeric(rv1 / (np.abs(rv24) + EPS))
        mapping["rv1_rv24"] = {
            "numerator": _first_col(feature_frame, FEATURE_CANDIDATES["rv1"])
            or "abs(ret1h)",
            "denominator": mapping.get("rv_24h"),
        }

    rv4 = _col(feature_frame, FEATURE_CANDIDATES["rv4"], n)
    if rv4 is not None and rv24 is not None:
        out["rv4_rv24"] = _fill_numeric(rv4 / (np.abs(rv24) + EPS))
        mapping["rv4_rv24"] = {
            "numerator": _first_col(feature_frame, FEATURE_CANDIDATES["rv4"]),
            "denominator": mapping.get("rv_24h"),
        }

    adx = _col(feature_frame, FEATURE_CANDIDATES["signed_adx"], n)
    if adx is not None:
        sign_src = _col(feature_frame, FEATURE_CANDIDATES["trend_sign"], n)
        if sign_src is not None and (
            _first_col(feature_frame, FEATURE_CANDIDATES["signed_adx"]) or ""
        ).startswith("adx"):
            adx = adx * np.sign(sign_src)
        out["signed_adx"] = _fill_numeric(adx)
        mapping["signed_adx"] = _first_col(
            feature_frame, FEATURE_CANDIDATES["signed_adx"]
        )

    for key in (
        "dist_ema_fast",
        "dist_ema_slow",
        "dist_vwap",
        "prior_day_low",
        "prior_day_high",
        "rvol_z",
        "entropy_24h",
    ):
        arr = _col(feature_frame, FEATURE_CANDIDATES[key], n)
        if arr is not None:
            out[key] = _fill_numeric(arr)
            mapping[key] = _first_col(feature_frame, FEATURE_CANDIDATES[key])

    for key in (
        "ebm_unc_logodds_var",
        "ebm_unc_pi_width",
        "ebm_unc_entropy_mean",
        "ebm_unc_entropy_std",
        "ebm_unc_conflict_norm",
        "ebm_unc_proximity_min",
        "ebm_unc_support_mean",
        "ebm_unc_support_min",
        "ebm_unc_concentration",
        "ebm_unc_sign_ratio",
        "ebm_unc_interaction_share",
        "ebm_unc_gap50rel",
        "ebm_unc_support_adjusted_uncertainty",
        "ebm_unc_uncertainty_weight",
        "ebm_unc_friction_weight",
    ):
        arr = _col(feature_frame, FEATURE_CANDIDATES[key], n)
        if arr is not None:
            out[key] = _fill_numeric(arr)
            mapping[key] = _first_col(feature_frame, FEATURE_CANDIDATES[key])

    for key in CROSS_ASSET_REGIME_FEATURES + FUNDING_REGIME_FEATURES:
        if key in out:
            continue
        arr = _col(feature_frame, FEATURE_CANDIDATES.get(key, (key,)), n)
        if arr is not None:
            out[key] = _fill_numeric(arr, fill=0.0)
            mapping[key] = _first_col(
                feature_frame, FEATURE_CANDIDATES.get(key, (key,))
            )

    if "asset_funding_side_alignment" not in out and "asset_funding_z" in out:
        side = _col(feature_frame, ("trade_side", "side_sign"), n)
        if side is not None:
            out["asset_funding_side_alignment"] = _fill_numeric(
                side * out["asset_funding_z"], fill=0.0
            )
            mapping["asset_funding_side_alignment"] = {
                "trade_side": "trade_side/side_sign",
                "funding": mapping.get("asset_funding_z"),
            }

    for key in ORDERBOOK_REGIME_FEATURES:
        if key in out:
            continue
        arr = _col(feature_frame, FEATURE_CANDIDATES.get(key, (key,)), n)
        if arr is not None:
            if "orderbook_imbalance_abs" in key:
                arr = np.abs(arr)
            if "volume_depth_risk" in key and "risk" not in str(
                _first_col(feature_frame, FEATURE_CANDIDATES.get(key, (key,)))
            ):
                arr = 1.0 / np.sqrt(1.0 + np.maximum(arr, 0.0))
            out[key] = _fill_numeric(arr, fill=0.0)
            mapping[key] = _first_col(
                feature_frame, FEATURE_CANDIDATES.get(key, (key,))
            )

    if "liquidity_stress_score" not in out:
        parts = [
            out[k]
            for k in ("spread_proxy", "volume_depth_proxy", "orderbook_imbalance_proxy")
            if k in out
        ]
        if parts:
            stress_parts = []
            if "spread_proxy" in out:
                stress_parts.append(_zscore(out["spread_proxy"]))
            if "volume_depth_proxy" in out:
                stress_parts.append(
                    _zscore(-np.asarray(out["volume_depth_proxy"], dtype=np.float64))
                )
            if "orderbook_imbalance_proxy" in out:
                stress_parts.append(_zscore(np.abs(out["orderbook_imbalance_proxy"])))
            out["liquidity_stress_score"] = _fill_numeric(
                np.nanmean(np.column_stack(stress_parts), axis=1), fill=0.0
            )
            mapping["liquidity_stress_score"] = "mean_z(spread, -depth, abs(imbalance))"

    for key in EBM_CONSOLIDATED_REGIME_FEATURES + ROLLING_PRIOR_REGIME_FEATURES:
        if key in out:
            continue
        arr = _col(feature_frame, FEATURE_CANDIDATES.get(key, (key,)), n)
        if arr is not None:
            out[key] = _fill_numeric(arr, fill=0.0)
            mapping[key] = _first_col(
                feature_frame, FEATURE_CANDIDATES.get(key, (key,))
            )

    if timestamps is not None and len(timestamps) >= n:
        ts = pd.to_datetime(np.asarray(timestamps)[:n], utc=True, errors="coerce")
        out["is_weekend"] = (pd.DatetimeIndex(ts).dayofweek >= 5).astype(np.float32)
        mapping["is_weekend"] = "timestamp.dayofweek>=5"

    sym_arr = (
        np.asarray(symbols).astype(str)[:n]
        if symbols is not None and len(symbols) >= n
        else np.repeat("all", n).astype(str)
    )
    rolling_30d_periods = 30 * 24
    rolling_30d_min_periods = 24
    if timestamps is not None and len(timestamps) >= n:
        ts_for_window = pd.to_datetime(
            np.asarray(timestamps)[:n], utc=True, errors="coerce"
        )
        finite_ts = pd.Series(ts_for_window).dropna().sort_values()
        if len(finite_ts) > 2:
            median_step = finite_ts.diff().dropna().median()
            if pd.notna(median_step) and median_step >= pd.Timedelta(hours=18):
                rolling_30d_periods = 30
                rolling_30d_min_periods = 3
    for key in ("asset_volume_30d", "asset_atr_30d"):
        if key in out:
            continue
        direct = key in feature_frame.columns
        arr = _col(feature_frame, FEATURE_CANDIDATES[key], n)
        if arr is None:
            continue
        if direct:
            out[key] = _fill_numeric(arr)
            mapping[key] = key
            continue
        series = pd.Series(_fill_numeric(arr), index=np.arange(n))
        group = pd.Series(sym_arr, index=np.arange(n))
        roll = (
            series.groupby(group, sort=False)
            .transform(
                lambda s: s.shift(1)
                .rolling(rolling_30d_periods, min_periods=rolling_30d_min_periods)
                .mean()
            )
            .to_numpy(dtype=np.float64)
        )
        fallback = (
            series.groupby(group, sort=False)
            .transform("median")
            .to_numpy(dtype=np.float64)
        )
        out[key] = _fill_numeric(np.where(np.isfinite(roll), roll, fallback))
        mapping[
            key
        ] = f"rolling30d({_first_col(feature_frame, FEATURE_CANDIDATES[key])})"

    ordered = {key: out[key] for key in REGIME_FEATURE_ORDER if key in out}
    return pd.DataFrame(ordered), mapping


def _candidate_model_feature_columns(
    feature_frame: pd.DataFrame,
    used_feature_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    if used_feature_columns:
        cols = [str(c) for c in used_feature_columns if str(c) in feature_frame.columns]
    else:
        blocked_exact = {
            "timestamp",
            "ts",
            "symbol",
            "side",
            "trade_side",
            "label",
            "target",
            "return",
            "returns",
            "net_gain",
            "gross_gain",
            "mae_ret",
            "mfe_ret",
            "bars_to_mfe",
            "bars_to_mae",
            "t_mfe",
            "t_mae",
        }
        blocked_substrings = ("future", "forward", "outcome", "realized")
        cols = []
        for c in feature_frame.columns:
            lc = str(c).lower()
            if lc in blocked_exact or any(s in lc for s in blocked_substrings):
                continue
            if pd.api.types.is_numeric_dtype(feature_frame[c]):
                cols.append(str(c))
    leaf_cols = [c for c in cols if re.search(r"(?:_leaf|leaf_|_soft$)", c)]
    non_leaf = [c for c in cols if c not in set(leaf_cols)]
    ordered = leaf_cols + non_leaf
    return ordered[:MAX_DISTRIBUTION_FEATURES]


def _fit_distribution_feature_spec(
    feature_frame: pd.DataFrame,
    used_feature_columns: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    cols = _candidate_model_feature_columns(feature_frame, used_feature_columns)
    if not cols:
        return {"enabled": False, "columns": []}
    x = feature_frame[cols].apply(pd.to_numeric, errors="coerce")
    med = x.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    q01 = x.quantile(0.01).replace([np.inf, -np.inf], np.nan).fillna(med)
    q05 = x.quantile(0.05).replace([np.inf, -np.inf], np.nan).fillna(med)
    q25 = x.quantile(0.25).replace([np.inf, -np.inf], np.nan).fillna(med)
    q75 = x.quantile(0.75).replace([np.inf, -np.inf], np.nan).fillna(med)
    q95 = x.quantile(0.95).replace([np.inf, -np.inf], np.nan).fillna(med)
    q99 = x.quantile(0.99).replace([np.inf, -np.inf], np.nan).fillna(med)
    scale = (q75 - q25).abs().replace(0.0, np.nan).fillna(
        x.std(axis=0, skipna=True).replace(0.0, np.nan)
    ).fillna(1.0)
    filled = x.fillna(med)
    scaler = RobustScaler()
    pca_payload: Dict[str, Any] = {"enabled": False}
    if len(filled) >= 20 and len(cols) >= 2:
        try:
            xs = scaler.fit_transform(filled.to_numpy(dtype=np.float64))
            max_components = min(xs.shape[0], xs.shape[1], 64)
            pca = PCA(n_components=max_components, random_state=42)
            pca.fit(xs)
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            n_comp = int(np.searchsorted(cumulative, 0.90) + 1)
            n_comp = max(1, min(n_comp, max_components))
            pca = PCA(n_components=n_comp, random_state=42)
            pca.fit(xs)
            pca_payload = {
                "enabled": True,
                "center": np.asarray(scaler.center_, dtype=float).tolist(),
                "scale": np.asarray(scaler.scale_, dtype=float).tolist(),
                "mean": np.asarray(pca.mean_, dtype=float).tolist(),
                "components": np.asarray(pca.components_, dtype=float).tolist(),
                "explained_variance_ratio": np.asarray(
                    pca.explained_variance_ratio_, dtype=float
                ).tolist(),
            }
        except Exception:
            pca_payload = {"enabled": False}
    bin_edges: Dict[str, List[float]] = {}
    bin_probs: Dict[str, List[float]] = {}
    for c in cols:
        vals = pd.to_numeric(x[c], errors="coerce").to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 20:
            continue
        edges = np.unique(np.quantile(vals, np.linspace(0.0, 1.0, 11)))
        if len(edges) < 3:
            continue
        ids = np.searchsorted(edges[1:-1], vals, side="right")
        counts = np.bincount(ids, minlength=len(edges) - 1).astype(np.float64)
        probs = (counts + 1.0) / (float(np.sum(counts)) + len(counts))
        bin_edges[c] = edges.astype(float).tolist()
        bin_probs[c] = probs.astype(float).tolist()
    return {
        "enabled": True,
        "columns": cols,
        "median": med.astype(float).to_dict(),
        "scale": scale.astype(float).to_dict(),
        "q01": q01.astype(float).to_dict(),
        "q05": q05.astype(float).to_dict(),
        "q95": q95.astype(float).to_dict(),
        "q99": q99.astype(float).to_dict(),
        "bin_edges": bin_edges,
        "bin_probs": bin_probs,
        "pca": pca_payload,
    }


def _append_distribution_features(
    regime_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    spec: Dict[str, Any],
    timestamps: Optional[Sequence[Any]],
    symbols: Optional[Sequence[Any]],
) -> pd.DataFrame:
    if not bool(spec.get("enabled", False)):
        return regime_df
    out = regime_df.copy()
    cols = [c for c in spec.get("columns", []) if c in feature_frame.columns]
    n = len(feature_frame)
    if not cols:
        for name in (
            "max_abs_zscore",
            "mean_abs_zscore",
            "num_features_outside_p01_p99",
            "num_features_outside_p05_p95",
            "p95_PSI",
            "mean_PSI",
            "num_features_PSI_above_threshold",
            "pca_reconstruction_error",
            "missing_count",
            "stale_feature_count",
        ):
            out[name] = np.zeros(n, dtype=np.float32)
        return out
    x = feature_frame[cols].apply(pd.to_numeric, errors="coerce")
    med = pd.Series({c: float(spec.get("median", {}).get(c, 0.0)) for c in cols})
    scale = pd.Series({c: float(spec.get("scale", {}).get(c, 1.0)) for c in cols})
    scale = scale.mask(scale.abs() < EPS, 1.0)
    z = (x - med) / scale
    abs_z = z.abs().to_numpy(dtype=np.float64)
    out["max_abs_zscore"] = np.nanmax(abs_z, axis=1).astype(np.float32)
    out["mean_abs_zscore"] = np.nanmean(abs_z, axis=1).astype(np.float32)
    q01 = pd.Series({c: float(spec.get("q01", {}).get(c, -np.inf)) for c in cols})
    q05 = pd.Series({c: float(spec.get("q05", {}).get(c, -np.inf)) for c in cols})
    q95 = pd.Series({c: float(spec.get("q95", {}).get(c, np.inf)) for c in cols})
    q99 = pd.Series({c: float(spec.get("q99", {}).get(c, np.inf)) for c in cols})
    out["num_features_outside_p01_p99"] = (
        ((x.lt(q01)) | (x.gt(q99))).sum(axis=1).to_numpy(dtype=np.float32)
    )
    out["num_features_outside_p05_p95"] = (
        ((x.lt(q05)) | (x.gt(q95))).sum(axis=1).to_numpy(dtype=np.float32)
    )
    rarity = np.zeros((n, len(cols)), dtype=np.float64)
    for j, c in enumerate(cols):
        edges = np.asarray(spec.get("bin_edges", {}).get(c, []), dtype=np.float64)
        probs = np.asarray(spec.get("bin_probs", {}).get(c, []), dtype=np.float64)
        if len(edges) < 3 or len(probs) != len(edges) - 1:
            continue
        vals = pd.to_numeric(x[c], errors="coerce").to_numpy(dtype=np.float64)
        ids = np.searchsorted(edges[1:-1], vals, side="right")
        ids = np.clip(ids, 0, len(probs) - 1)
        rarity[:, j] = -np.log(np.clip(probs[ids], 1e-6, 1.0))
        rarity[~np.isfinite(vals), j] = 0.0
    out["p95_PSI"] = np.nanpercentile(rarity, 95, axis=1).astype(np.float32)
    out["mean_PSI"] = np.nanmean(rarity, axis=1).astype(np.float32)
    out["num_features_PSI_above_threshold"] = (
        rarity > PSI_RARITY_THRESHOLD
    ).sum(axis=1).astype(np.float32)
    pca = spec.get("pca", {})
    if bool(pca.get("enabled", False)):
        center = np.asarray(pca.get("center", []), dtype=np.float64)
        sc = np.asarray(pca.get("scale", []), dtype=np.float64)
        mean = np.asarray(pca.get("mean", []), dtype=np.float64)
        comps = np.asarray(pca.get("components", []), dtype=np.float64)
        xf = x.fillna(med).to_numpy(dtype=np.float64)
        if len(center) == xf.shape[1] and len(sc) == xf.shape[1] and comps.ndim == 2:
            xs = (xf - center) / np.where(np.abs(sc) > EPS, sc, 1.0)
            scores = (xs - mean) @ comps.T
            recon = scores @ comps + mean
            out["pca_reconstruction_error"] = np.mean(
                np.square(xs - recon), axis=1
            ).astype(np.float32)
        else:
            out["pca_reconstruction_error"] = np.zeros(n, dtype=np.float32)
    else:
        out["pca_reconstruction_error"] = np.zeros(n, dtype=np.float32)
    out["missing_count"] = x.isna().sum(axis=1).to_numpy(dtype=np.float32)
    stale = np.zeros(n, dtype=np.float32)
    if n > 1:
        sy = (
            np.asarray(symbols).astype(str)[:n]
            if symbols is not None and len(symbols) >= n
            else np.repeat("all", n)
        )
        filled = x.fillna(med)
        for _sym, idx in pd.Series(np.arange(n)).groupby(sy, sort=False):
            pos = idx.to_numpy(dtype=np.int64)
            if len(pos) <= 1:
                continue
            delta = np.abs(np.diff(filled.iloc[pos].to_numpy(dtype=np.float64), axis=0))
            stale[pos[1:]] = np.sum(delta <= 1e-12, axis=1)
    out["stale_feature_count"] = stale
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _rolling_by_symbol(
    values: np.ndarray,
    timestamps: Optional[Sequence[Any]],
    symbols: np.ndarray,
    window_days: int,
    *,
    min_periods: int = 5,
    shift: bool = True,
) -> np.ndarray:
    n = len(values)
    if timestamps is None or len(timestamps) < n:
        s = pd.Series(values)
        rolled = s.shift(1 if shift else 0).rolling(
            max(min_periods, window_days), min_periods=min_periods
        ).mean()
        return rolled.to_numpy(dtype=np.float64)
    ts = pd.to_datetime(np.asarray(timestamps)[:n], utc=True, errors="coerce")
    df = pd.DataFrame(
        {"value": np.asarray(values, dtype=np.float64), "symbol": symbols, "ts": ts}
    )
    out = np.full(n, np.nan, dtype=np.float64)
    for _sym, grp in df.groupby("symbol", sort=False):
        g = grp.dropna(subset=["ts"]).sort_values("ts")
        if g.empty:
            continue
        series = pd.Series(g["value"].to_numpy(dtype=np.float64), index=g["ts"])
        if shift:
            series = series.shift(1)
        rolled = series.rolling(f"{int(window_days)}D", min_periods=min_periods).mean()
        out[g.index.to_numpy(dtype=np.int64)] = rolled.to_numpy(dtype=np.float64)
    return out


def _rolling_global(
    values: np.ndarray,
    timestamps: Optional[Sequence[Any]],
    window_days: int,
    *,
    min_periods: int = 1,
    shift: bool = True,
) -> np.ndarray:
    n = len(values)
    if timestamps is None or len(timestamps) < n:
        s = pd.Series(np.asarray(values, dtype=np.float64))
        rolled = s.shift(1 if shift else 0).rolling(
            max(min_periods, int(window_days)), min_periods=min_periods
        ).mean()
        return rolled.to_numpy(dtype=np.float64)
    ts = pd.to_datetime(np.asarray(timestamps)[:n], utc=True, errors="coerce")
    df = pd.DataFrame({"value": np.asarray(values, dtype=np.float64), "ts": ts})
    out = np.full(n, np.nan, dtype=np.float64)
    g = df.dropna(subset=["ts"]).sort_values("ts")
    if g.empty:
        return out
    series = pd.Series(g["value"].to_numpy(dtype=np.float64), index=g["ts"])
    if shift:
        series = series.shift(1)
    rolled = series.rolling(f"{int(window_days)}D", min_periods=min_periods).mean()
    out[g.index.to_numpy(dtype=np.int64)] = rolled.to_numpy(dtype=np.float64)
    return out


def _find_prediction_column(feature_frame: pd.DataFrame, names: Sequence[str]) -> Optional[np.ndarray]:
    col = _first_col(feature_frame, names)
    if col is None:
        return None
    return _as_float_array(feature_frame[col].values, len(feature_frame))


def _prediction_columns(feature_frame: pd.DataFrame) -> List[str]:
    exact = {
        "ridge_score",
        "et_score",
        "lgbm_score",
        "lgbm_clf_score",
        "total_confidence",
        "sizer_score_oof",
        "oof_base_clf",
        "oof_meta_clf",
        "oof_pred",
        "oof_p_tp",
        "oof_p_move",
        "clf",
        "raw_meta_prediction",
        "calibrated_score",
        "oof_ebm_raw",
        "oof_ebm_en",
        "oof_ebm_uncertainty_weighted",
    }
    out: List[str] = []
    for c in feature_frame.columns:
        name = str(c)
        if name in exact:
            out.append(name)
            continue
        if re.match(r"^(?:base_H\d+|base_H\d+_(?:tight|wide|balanced)|pred_H\d+)$", name):
            out.append(name)
            continue
        if re.search(
            r"(?:^|_)(?:base|meta|ridge|lgbm|et|ebm).*(?:pred|score|prob|clf|raw|en|weighted)",
            name,
        ):
            out.append(name)
    seen: List[str] = []
    for c in out:
        if c in feature_frame.columns and c not in seen and pd.api.types.is_numeric_dtype(feature_frame[c]):
            seen.append(c)
    return seen


def _prediction_matrix(feature_frame: pd.DataFrame) -> Optional[np.ndarray]:
    seen = _prediction_columns(feature_frame)
    if len(seen) < 2:
        return None
    mat = feature_frame[seen].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64
    )
    finite_cols = np.sum(np.isfinite(mat), axis=0) >= max(5, int(0.05 * len(mat)))
    mat = mat[:, finite_cols]
    if mat.shape[1] < 2:
        return None
    return mat


def _base_prediction_matrix(feature_frame: pd.DataFrame) -> Optional[np.ndarray]:
    horizon_candidates: List[str] = []
    fallback_candidates: List[str] = []
    for c in feature_frame.columns:
        name = str(c)
        if re.match(r"^base_H\d+(?:_(?:tight|wide|balanced))?$", name):
            horizon_candidates.append(name)
            continue
        if name == "oof_base_clf":
            fallback_candidates.append(name)
            continue
        if re.search(r"(?:^|_)base.*(?:pred|score|prob|clf)", name):
            fallback_candidates.append(name)
    candidates = horizon_candidates if horizon_candidates else fallback_candidates
    seen: List[str] = []
    for c in candidates:
        if c not in seen and c in feature_frame.columns and pd.api.types.is_numeric_dtype(feature_frame[c]):
            seen.append(c)
    if len(seen) < 2:
        return None
    mat = feature_frame[seen].apply(pd.to_numeric, errors="coerce").to_numpy(
        dtype=np.float64
    )
    finite_cols = np.sum(np.isfinite(mat), axis=0) >= max(5, int(0.05 * len(mat)))
    mat = mat[:, finite_cols]
    nonconstant_cols = np.nanstd(mat, axis=0) > 1e-8 if mat.size else np.array([], dtype=bool)
    mat = mat[:, nonconstant_cols]
    if mat.shape[1] < 2:
        return None
    return mat


def _base_prediction_proxy(feature_frame: pd.DataFrame, n: int) -> Optional[np.ndarray]:
    cols = [
        c
        for c in feature_frame.columns
        if re.match(r"^base_H\d+(?:_(?:tight|wide|balanced))?$", str(c))
        and pd.api.types.is_numeric_dtype(feature_frame[c])
    ]
    if cols:
        vals = feature_frame.iloc[:n][cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if vals.shape[1]:
            nonconstant = np.nanstd(vals, axis=0) > 1e-8
            if np.any(nonconstant):
                return np.nanmean(vals[:, nonconstant], axis=1)
    base = _find_prediction_column(
        feature_frame.iloc[:n],
        (
            "base_pred",
            "base_prediction",
            "base_score",
            "base_rank_pct",
            "base_H10",
            "base_H5",
            "base_H4",
            "base_H2",
            "base_H1",
            "oof_base_clf",
            "oof_p_move_base",
            "oof_p_move",
            "oof_p_tp_base",
            "oof_p_tp",
        ),
    )
    if base is not None:
        return base
    return None


def _meta_prediction_proxy(
    feature_frame: pd.DataFrame,
    scores: np.ndarray,
    n: int,
) -> np.ndarray:
    meta = _find_prediction_column(
        feature_frame.iloc[:n],
        (
            "oof_meta_clf",
            "oof_ebm_en",
            "oof_ebm_uncertainty_weighted",
            "clf",
            "oof_p_move",
            "oof_p_tp",
            "raw_meta_prediction",
            "calibrated_score",
        ),
    )
    if meta is None:
        meta = np.asarray(scores, dtype=np.float64)[:n]
    return meta


def _append_prediction_reliability_features(
    regime_df: pd.DataFrame,
    feature_frame: pd.DataFrame,
    scores: np.ndarray,
    returns: Optional[np.ndarray],
    timestamps: Optional[Sequence[Any]],
    symbols: np.ndarray,
    artifact: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    n = len(regime_df)
    out = regime_df.copy()
    meta = np.clip(np.asarray(scores, dtype=np.float64)[:n], 1e-6, 1.0 - 1e-6)
    raw_meta = _meta_prediction_proxy(feature_frame, meta, n)
    out["meta_model_prediction"] = np.clip(
        _fill_numeric(raw_meta, 0.5), 1e-6, 1.0 - 1e-6
    ).astype(np.float32)
    base = _base_prediction_proxy(feature_frame, n)
    if base is None:
        base = meta.copy()
    base = np.clip(_fill_numeric(base, 0.5), 1e-6, 1.0 - 1e-6)
    out["base_model_prediction"] = base.astype(np.float32)
    diff = meta - base
    abs_diff = np.abs(diff)
    out["abs_base_meta_diff"] = abs_diff.astype(np.float32)
    out["signed_base_meta_diff"] = diff.astype(np.float32)
    for days in (3, 7, 15):
        out[f"abs_base_meta_diff_{days}d"] = _fill_numeric(
            _rolling_by_symbol(abs_diff, timestamps, symbols, days), 0.0
        )
        out[f"signed_base_meta_diff_{days}d"] = _fill_numeric(
            _rolling_by_symbol(diff, timestamps, symbols, days), 0.0
        )
    out["base_meta_diff_ewm_3d"] = (
        pd.Series(diff).shift(1).ewm(span=3, min_periods=2, adjust=False).mean()
    ).fillna(0.0).to_numpy(dtype=np.float32)
    disagreement = (abs_diff > PREDICTION_DISAGREEMENT_THRESHOLD).astype(np.float64)
    out["base_meta_disagreement_rate_3d"] = _fill_numeric(
        _rolling_by_symbol(disagreement, timestamps, symbols, 3), 0.0
    )
    out["base_meta_disagreement_rate_7d"] = _fill_numeric(
        _rolling_by_symbol(disagreement, timestamps, symbols, 7), 0.0
    )
    mat = _base_prediction_matrix(feature_frame.iloc[:n])
    if mat is not None:
        out["base_models_pred_std"] = np.nanstd(mat, axis=1).astype(np.float32)
        out["base_models_pred_range"] = (
            np.nanmax(mat, axis=1) - np.nanmin(mat, axis=1)
        ).astype(np.float32)
        out["base_models_pred_iqr"] = (
            np.nanpercentile(mat, 75, axis=1) - np.nanpercentile(mat, 25, axis=1)
        ).astype(np.float32)
        internal_disagree = (
            out["base_models_pred_range"].to_numpy(dtype=np.float64)
            > PREDICTION_DISAGREEMENT_THRESHOLD
        ).astype(np.float64)
        out["base_models_disagreement_rate_3d"] = _fill_numeric(
            _rolling_by_symbol(internal_disagree, timestamps, symbols, 3), 0.0
        )
        out["base_models_disagreement_rate_7d"] = _fill_numeric(
            _rolling_by_symbol(internal_disagree, timestamps, symbols, 7), 0.0
        )
        out["base_models_disagreement_rate_15d"] = _fill_numeric(
            _rolling_by_symbol(internal_disagree, timestamps, symbols, 15), 0.0
        )
    else:
        for c in (
            "base_models_pred_std",
            "base_models_pred_range",
            "base_models_pred_iqr",
            "base_models_disagreement_rate_3d",
            "base_models_disagreement_rate_7d",
            "base_models_disagreement_rate_15d",
        ):
            out[c] = np.zeros(n, dtype=np.float32)
    ebm_cols = [
        c
        for c in ("oof_ebm_raw", "oof_ebm_en", "oof_ebm_uncertainty_weighted")
        if c in feature_frame.columns and pd.api.types.is_numeric_dtype(feature_frame[c])
    ]
    if ebm_cols:
        ebm = feature_frame.iloc[:n][ebm_cols].apply(pd.to_numeric, errors="coerce")
        raw = (
            ebm["oof_ebm_raw"].to_numpy(dtype=np.float64)
            if "oof_ebm_raw" in ebm
            else meta.copy()
        )
        en = (
            ebm["oof_ebm_en"].to_numpy(dtype=np.float64)
            if "oof_ebm_en" in ebm
            else raw.copy()
        )
        weighted = (
            ebm["oof_ebm_uncertainty_weighted"].to_numpy(dtype=np.float64)
            if "oof_ebm_uncertainty_weighted" in ebm
            else en.copy()
        )
        ebm_mat = np.column_stack(
            [_fill_numeric(raw, 0.5), _fill_numeric(en, 0.5), _fill_numeric(weighted, 0.5)]
        )
        dispersion = np.nanmax(ebm_mat, axis=1) - np.nanmin(ebm_mat, axis=1)
        en_adj = np.abs(ebm_mat[:, 1] - ebm_mat[:, 0])
        weighted_adj = np.abs(ebm_mat[:, 2] - ebm_mat[:, 0])
        brittleness = dispersion + 0.5 * (en_adj + weighted_adj)
        out["ebm_raw_prediction"] = ebm_mat[:, 0].astype(np.float32)
        out["ebm_en_prediction"] = ebm_mat[:, 1].astype(np.float32)
        out["ebm_uncertainty_weighted_prediction"] = ebm_mat[:, 2].astype(np.float32)
        out["ebm_en_adjustment_abs"] = en_adj.astype(np.float32)
        out["ebm_uncertainty_adjustment_abs"] = weighted_adj.astype(np.float32)
        out["ebm_prediction_dispersion"] = dispersion.astype(np.float32)
        out["ebm_prediction_brittleness"] = brittleness.astype(np.float32)
        if "global_ebm_unc_dispersion_mean_7d" not in out:
            out["global_ebm_unc_dispersion_mean_7d"] = _fill_numeric(
                _rolling_global(dispersion, timestamps, 7), 0.0
            )
        for days in (3, 7, 15):
            disp_col = f"asset_ebm_unc_dispersion_mean_{days}d"
            conflict_col = f"asset_ebm_conflict_mean_{days}d"
            support_col = f"asset_ebm_support_risk_mean_{days}d"
            brittle_col = f"asset_ebm_brittleness_mean_{days}d"
            if disp_col not in out:
                out[disp_col] = _fill_numeric(
                    _rolling_by_symbol(dispersion, timestamps, symbols, days), 0.0
                )
            if conflict_col not in out:
                out[conflict_col] = _fill_numeric(
                    _rolling_by_symbol(en_adj, timestamps, symbols, days), 0.0
                )
            if support_col not in out:
                out[support_col] = _fill_numeric(
                    _rolling_by_symbol(weighted_adj, timestamps, symbols, days), 0.0
                )
            if brittle_col not in out:
                out[brittle_col] = _fill_numeric(
                    _rolling_by_symbol(brittleness, timestamps, symbols, days), 0.0
                )
    else:
        for c in (
            "ebm_raw_prediction",
            "ebm_en_prediction",
            "ebm_uncertainty_weighted_prediction",
            "ebm_en_adjustment_abs",
            "ebm_uncertainty_adjustment_abs",
            "ebm_prediction_dispersion",
            "ebm_prediction_brittleness",
        ):
            out[c] = np.zeros(n, dtype=np.float32)
    spec: Dict[str, Any] = {}
    if returns is not None and len(returns) >= n:
        y = (np.asarray(returns[:n], dtype=np.float64) > 0.0).astype(np.float64)
        brier = np.square(meta - y)
        global_brier = float(np.nanmean(brier[np.isfinite(brier)])) if np.isfinite(brier).any() else 0.25
        local_brier = _rolling_by_symbol(brier, timestamps, symbols, 15, min_periods=5)
        counts = _rolling_by_symbol(
            np.isfinite(brier).astype(np.float64), timestamps, symbols, 15, min_periods=1
        )
        w = counts / (counts + SYMBOL_SHRINK_K)
        shrunk = w * local_brier + (1.0 - w) * global_brier
        out["meta_recent_brier_shrunk"] = _fill_numeric(shrunk, global_brier)
        out["meta_recent_brier_global"] = np.full(n, global_brier, dtype=np.float32)
        spec["global_brier"] = global_brier
    else:
        global_brier = float(
            (artifact or {}).get("reliability_feature_spec", {}).get("global_brier", 0.25)
        )
        out["meta_recent_brier_shrunk"] = np.full(n, global_brier, dtype=np.float32)
        out["meta_recent_brier_global"] = np.full(n, global_brier, dtype=np.float32)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0), spec


def _append_symbol_features(
    regime_df: pd.DataFrame,
    returns: Optional[np.ndarray],
    timestamps: Optional[Sequence[Any]],
    symbols: np.ndarray,
    artifact: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    n = len(regime_df)
    out = regime_df.copy()
    spec = dict((artifact or {}).get("symbol_feature_spec", {}) or {})
    global_utility = float(spec.get("global_utility", 0.0))
    global_bad = float(spec.get("global_bad_rate", 0.5))
    symbol_stats = spec.get("symbols", {}) if isinstance(spec.get("symbols", {}), dict) else {}
    if returns is not None and len(returns) >= n:
        r = np.asarray(returns[:n], dtype=np.float64)
        finite = np.isfinite(r)
        global_utility = float(np.nanmean(r[finite])) if finite.any() else 0.0
        bad = (r < 0.0).astype(np.float64)
        global_bad = float(np.nanmean(bad[finite])) if finite.any() else 0.5
        local_utility = _rolling_by_symbol(r, timestamps, symbols, 15, min_periods=5)
        local_bad = _rolling_by_symbol(bad, timestamps, symbols, 15, min_periods=5)
        counts = _rolling_by_symbol(
            finite.astype(np.float64), timestamps, symbols, 15, min_periods=1
        )
        w = counts / (counts + SYMBOL_SHRINK_K)
        out["symbol_recent_utility_shrunk"] = _fill_numeric(
            w * local_utility + (1.0 - w) * global_utility, global_utility
        )
        out["symbol_recent_bad_rate_shrunk"] = _fill_numeric(
            w * local_bad + (1.0 - w) * global_bad, global_bad
        )
        stats: Dict[str, Any] = {}
        for sym in sorted(set(str(s) for s in symbols)):
            m = symbols == sym
            rr = r[m]
            ff = np.isfinite(rr)
            stats[str(sym)] = {
                "n": int(np.sum(ff)),
                "utility": float(np.nanmean(rr[ff])) if ff.any() else global_utility,
                "bad_rate": float(np.nanmean(rr[ff] < 0.0)) if ff.any() else global_bad,
            }
        spec = {"global_utility": global_utility, "global_bad_rate": global_bad, "symbols": stats}
    else:
        util = np.full(n, global_utility, dtype=np.float64)
        bad = np.full(n, global_bad, dtype=np.float64)
        for i, sym in enumerate(symbols[:n]):
            row = symbol_stats.get(str(sym), {})
            cnt = float(row.get("n", 0.0))
            w = cnt / (cnt + SYMBOL_SHRINK_K)
            util[i] = w * float(row.get("utility", global_utility)) + (1.0 - w) * global_utility
            bad[i] = w * float(row.get("bad_rate", global_bad)) + (1.0 - w) * global_bad
        out["symbol_recent_utility_shrunk"] = util.astype(np.float32)
        out["symbol_recent_bad_rate_shrunk"] = bad.astype(np.float32)
    counts_map = pd.Series(symbols).value_counts()
    out["symbol_sample_count_log"] = np.log1p(
        pd.Series(symbols).map(counts_map).fillna(0.0).to_numpy(dtype=np.float64)
    ).astype(np.float32)
    if "asset_volume_30d" in out.columns:
        vol_med = out.groupby(pd.Series(symbols, index=out.index), sort=False)[
            "asset_volume_30d"
        ].transform("median")
        out["symbol_liquidity_rank"] = vol_med.rank(method="average", pct=True).to_numpy(
            dtype=np.float32
        )
    else:
        out["symbol_liquidity_rank"] = np.full(n, 0.5, dtype=np.float32)
    if "asset_atr_30d" in out.columns:
        atr_med = out.groupby(pd.Series(symbols, index=out.index), sort=False)[
            "asset_atr_30d"
        ].transform("median")
        out["symbol_vol_rank"] = atr_med.rank(method="average", pct=True).to_numpy(
            dtype=np.float32
        )
    else:
        out["symbol_vol_rank"] = np.full(n, 0.5, dtype=np.float32)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0), spec


def _rank_pct(scores: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(scores, dtype=np.float64))
    return s.rank(method="average", pct=True).to_numpy(dtype=np.float64)


def _top_mask(scores: np.ndarray, frac: float) -> np.ndarray:
    s = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(s)
    mask = np.zeros(len(s), dtype=bool)
    if not finite.any():
        return mask
    n_top = max(1, int(math.ceil(float(np.sum(finite)) * frac)))
    finite_idx = np.where(finite)[0]
    order = finite_idx[np.argsort(s[finite_idx])[-n_top:]]
    mask[order] = True
    return mask


def _drawdown(rets: np.ndarray) -> float:
    r = np.asarray(rets, dtype=np.float64)
    if len(r) == 0:
        return 0.0
    eq = np.cumsum(np.nan_to_num(r, nan=0.0))
    peak = np.maximum.accumulate(eq)
    return float(np.nanmax(peak - eq)) if len(eq) else 0.0


def _period_std(rets: np.ndarray, timestamps: Optional[np.ndarray], freq: str) -> float:
    if timestamps is None or len(timestamps) != len(rets):
        return float(np.nanstd(rets)) if len(rets) > 1 else 0.0
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    df = pd.DataFrame({"ret": rets, "ts": ts}).dropna()
    if df.empty:
        return 0.0
    ts_naive = df["ts"].dt.tz_convert(None)
    grouped = df.groupby(ts_naive.dt.to_period(freq))["ret"].sum()
    return float(grouped.std(ddof=0)) if len(grouped) > 1 else 0.0


def _period_count(rets: np.ndarray, timestamps: Optional[np.ndarray], freq: str) -> int:
    if timestamps is None or len(timestamps) != len(rets):
        return 0
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    df = pd.DataFrame({"ret": rets, "ts": ts}).dropna()
    if df.empty:
        return 0
    ts_naive = df["ts"].dt.tz_convert(None)
    return int(df.groupby(ts_naive.dt.to_period(freq))["ret"].sum().shape[0])


def score_metrics(
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[Sequence[Any]] = None,
    *,
    top_fracs: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
    cost_pct: float = 0.003,
) -> pd.DataFrame:
    s = np.asarray(scores, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    n = min(len(s), len(r))
    s, r = s[:n], r[:n]
    ts = (
        np.asarray(timestamps)[:n]
        if timestamps is not None and len(timestamps) >= n
        else None
    )
    finite = np.isfinite(s) & np.isfinite(r)
    overall_hit = float(np.mean(r[finite] > 0.0)) if finite.any() else 0.0
    rows: List[Dict[str, Any]] = []
    for frac in top_fracs:
        mask = _top_mask(np.where(finite, s, np.nan), frac)
        sel = mask & finite
        sr = r[sel]
        sts = ts[sel] if ts is not None else None
        net = sr - cost_pct
        hit = float(np.mean(sr > 0.0)) if len(sr) else 0.0
        n_sel = int(len(sr))
        gross_std = float(np.std(sr)) if n_sel > 1 else 0.0
        net_std = float(np.std(net)) if n_sel > 1 else 0.0
        hit_se = float(math.sqrt(max(hit * (1.0 - hit), 0.0) / n_sel)) if n_sel else 0.0
        gross_se = float(gross_std / math.sqrt(n_sel)) if n_sel > 1 else 0.0
        net_se = float(net_std / math.sqrt(n_sel)) if n_sel > 1 else 0.0
        ds = net[net < 0.0]
        downside_std = float(np.std(ds)) if len(ds) > 1 else 0.0
        sortino = float(np.mean(net) / (downside_std + EPS)) if len(net) else 0.0
        equity = np.cumsum(net)
        stability = 0.0
        if len(equity) > 5:
            try:
                _, _, r_val, _, _ = linregress(np.arange(len(equity)), equity)
                stability = float(r_val**2) if np.isfinite(r_val) else 0.0
            except Exception:
                stability = 0.0
        rows.append(
            {
                "top_frac": float(frac),
                "lift": float(hit / (overall_hit + EPS)),
                "hit_rate": hit,
                "hit_rate_se": hit_se,
                "lift_se_approx": float(hit_se / (overall_hit + EPS)),
                "mean_gross_return": float(np.mean(sr)) if len(sr) else 0.0,
                "mean_gross_return_se": gross_se,
                "mean_net_return": float(np.mean(net)) if len(net) else 0.0,
                "mean_net_return_se": net_se,
                "net_ret": float(np.sum(net)) if len(net) else 0.0,
                "return_std": net_std,
                "std_weekly": _period_std(net, sts, "W"),
                "std_monthly": _period_std(net, sts, "M"),
                "weekly_periods": _period_count(net, sts, "W"),
                "monthly_periods": _period_count(net, sts, "M"),
                "worst_week_loss": _worst_period_loss(net, sts, "W"),
                "worst_month_loss": _worst_period_loss(net, sts, "M"),
                "sortino": sortino,
                "max_drawdown": _drawdown(net),
                "trades": n_sel,
                "lift_gt_1": bool((hit / (overall_hit + EPS)) > 1.0),
            }
        )
    return pd.DataFrame(rows)


def _worst_period_loss(
    rets: np.ndarray, timestamps: Optional[np.ndarray], freq: str
) -> float:
    if timestamps is None or len(timestamps) != len(rets):
        return float(abs(min(float(np.nansum(rets)), 0.0)))
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    df = pd.DataFrame({"ret": rets, "ts": ts}).dropna()
    if df.empty:
        return 0.0
    ts_naive = df["ts"].dt.tz_convert(None)
    grouped = df.groupby(ts_naive.dt.to_period(freq))["ret"].sum()
    return float(abs(min(float(grouped.min()), 0.0))) if len(grouped) else 0.0


def _safe_ratio(num: float, den: float, neutral: float = 1.0) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < EPS:
        return float(neutral)
    return float(num / den)


def _edge_score(bucket: Dict[str, float], strategy: Dict[str, float]) -> float:
    lift_ratio = _safe_ratio(bucket.get("lift", 0.0), strategy.get("lift", 0.0))
    gross_ratio = _safe_ratio(
        bucket.get("mean_gross_return", 0.0), strategy.get("mean_gross_return", 0.0)
    )
    hit_ratio = _safe_ratio(bucket.get("hit_rate", 0.0), strategy.get("hit_rate", 0.0))
    std_ratio = _safe_ratio(
        bucket.get("return_std", 0.0), strategy.get("return_std", 0.0)
    )
    dd_ratio = _safe_ratio(
        bucket.get("max_drawdown", 0.0), strategy.get("max_drawdown", 0.0)
    )
    return float(
        0.20 * math.log(max(lift_ratio, EPS))
        + 0.25 * math.log(max(gross_ratio, EPS))
        + 0.15 * math.log(max(hit_ratio, EPS))
        - 0.20 * math.log(max(std_ratio, EPS))
        - 0.20 * math.log(max(dd_ratio, EPS))
    )


def _fit_percentile(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([0.0], dtype=np.float64)
    if len(x) > 5000:
        qs = np.linspace(0.0, 1.0, 1001)
        return np.quantile(x, qs).astype(np.float64)
    return np.sort(x).astype(np.float64)


def _apply_percentile(values: np.ndarray, ref: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    if len(r) == 0:
        return np.full(len(x), 0.5, dtype=np.float64)
    pct = np.searchsorted(r, x, side="right") / max(len(r), 1)
    pct = np.where(np.isfinite(x), pct, 0.5)
    return np.clip(pct, 0.01, 0.99).astype(np.float64)


def _walk_forward_splits(
    timestamps: Sequence[Any], n: int, n_splits: int = 5
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Time-only walk-forward splits that keep equal timestamps together."""
    if n < 20:
        return [(np.arange(0, max(1, n // 2)), np.arange(max(1, n // 2), n))]
    ts = pd.to_datetime(np.asarray(timestamps)[:n], utc=True, errors="coerce")
    ts_int = np.where(pd.isna(ts), np.arange(n, dtype=np.int64), ts.view("int64"))
    unique_times = np.array(sorted(pd.unique(ts_int)))
    time_folds = np.array_split(unique_times, min(int(n_splits), len(unique_times)))
    out: List[Tuple[np.ndarray, np.ndarray]] = []
    for i in range(1, len(time_folds)):
        train_times = set(np.concatenate(time_folds[:i]).tolist())
        valid_times = set(time_folds[i].tolist())
        tr = np.asarray(
            [j for j, t in enumerate(ts_int) if int(t) in train_times], dtype=int
        )
        te = np.asarray(
            [j for j, t in enumerate(ts_int) if int(t) in valid_times], dtype=int
        )
        if len(tr) and len(te):
            out.append((np.sort(tr), np.sort(te)))
    if not out:
        split = max(1, n // 2)
        out.append((np.arange(split), np.arange(split, n)))
    return out


def _rank_weight(scores: np.ndarray) -> np.ndarray:
    ranked = _rank_pct(scores)
    rank_in_top10 = np.clip((ranked - 0.90) / 0.10, 0.0, 1.0)
    return (1.0 + 0.5 * rank_in_top10).astype(np.float32)


def _feature_effect_from_stats(pct: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    centers = np.asarray(stats.get("spline_x", []), dtype=np.float64)
    values = np.asarray(stats.get("spline_y", []), dtype=np.float64)
    if len(centers) < 2 or len(values) < 2:
        return np.zeros(len(pct), dtype=np.float32)
    order = np.argsort(centers)
    y = np.interp(np.asarray(pct, dtype=np.float64), centers[order], values[order])
    clip = stats.get("log_effect_clip", [-0.10, 0.10])
    lo, hi = float(clip[0]), float(clip[1])
    return np.clip(y, lo, hi).astype(np.float32)


def _fit_feature_stats(
    pct: np.ndarray,
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    *,
    min_bucket_n: int = 300,
    shrink_k: float = 1500.0,
    tree_min_leaf_frac: float = 0.05,
    max_leaf_nodes: int = 4,
    max_depth: int = 3,
    ccp_alpha: float = 0.001,
    max_bin_share: float = 0.72,
    rank_weight: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    top = _top_mask(scores, 0.05)
    n_top = int(np.sum(top & np.isfinite(pct) & np.isfinite(returns)))
    min_leaf = max(int(min_bucket_n), int(float(tree_min_leaf_frac) * n_top))
    if n_top < max(2 * min_leaf, 2 * min_bucket_n):
        return {
            "enabled": False,
            "reason": "insufficient_top5",
            "spline_x": [],
            "spline_y": [],
        }
    target = returns - float(np.nanmean(returns[top]))
    w = rank_weight if rank_weight is not None else np.ones(len(pct), dtype=np.float32)
    top_idx = np.where(top & np.isfinite(pct) & np.isfinite(target))[0]
    tree_target = target[top_idx] * 100.0
    thresholds: List[float] = []
    tree_alpha = float(ccp_alpha)
    alpha_candidates = [
        float(ccp_alpha),
        min(float(ccp_alpha), 0.0003),
        min(float(ccp_alpha), 0.0001),
        min(float(ccp_alpha), 0.00003),
        0.0,
    ]
    alpha_candidates = list(dict.fromkeys(alpha_candidates))
    balance_counts: List[int] = []
    for alpha in alpha_candidates:
        tree = DecisionTreeRegressor(
            max_leaf_nodes=int(max_leaf_nodes),
            max_depth=int(max_depth),
            min_samples_leaf=min_leaf,
            ccp_alpha=alpha,
            random_state=42,
        )
        tree.fit(
            pct[top_idx].reshape(-1, 1),
            tree_target,
            sample_weight=w[top_idx],
        )
        thresholds = sorted(
            float(t) for t in tree.tree_.threshold if np.isfinite(t) and 0.01 < t < 0.99
        )
        tree_alpha = float(alpha)
        if thresholds:
            candidate_edges = np.array([0.0] + thresholds + [1.0], dtype=np.float64)
            bin_ids = np.searchsorted(candidate_edges[1:-1], pct[top_idx], side="right")
            counts = np.bincount(bin_ids, minlength=len(candidate_edges) - 1)
            non_zero = counts[counts > 0]
            balance_counts = [int(x) for x in counts.tolist()]
            if (
                len(non_zero) >= 2
                and int(np.min(non_zero)) >= min_leaf
                and float(np.max(non_zero) / max(1, np.sum(non_zero)))
                <= float(max_bin_share)
            ):
                break
            thresholds = []
            balance_counts = []
        if thresholds:
            break
    edges = np.array([0.0] + thresholds + [1.0], dtype=np.float64)
    if len(edges) < 3:
        return {
            "enabled": False,
            "reason": "too_few_balanced_bins",
            "spline_x": [],
            "spline_y": [],
        }
    strategy_top = (
        score_metrics(scores, returns, timestamps, top_fracs=(0.05,)).iloc[0].to_dict()
    )
    rows: List[Dict[str, Any]] = []
    centers: List[float] = []
    ys: List[float] = []
    for b in range(len(edges) - 1):
        lo, hi = float(edges[b]), float(edges[b + 1])
        mask = (pct >= lo) & (pct < hi if b < len(edges) - 2 else pct <= hi)
        weighted_n = float(np.sum(w[mask & top]))
        if weighted_n < min_bucket_n:
            continue
        local_scores = np.where(mask, scores, np.nan)
        bm = (
            score_metrics(local_scores, returns, timestamps, top_fracs=(0.05,))
            .iloc[0]
            .to_dict()
        )
        edge = _edge_score(bm, strategy_top)
        reliability = weighted_n / (weighted_n + shrink_k)
        shrunk = float(np.clip(edge * reliability, -0.12, 0.12))
        center = float(np.clip((lo + hi) * 0.5, 0.01, 0.99))
        centers.append(center)
        ys.append(shrunk)
        rows.append(
            {
                "lo": lo,
                "hi": hi,
                "center": center,
                "weighted_n": weighted_n,
                "edge_score": edge,
                "shrunk_edge_score": shrunk,
                **bm,
            }
        )
    if len(centers) < 2:
        return {
            "enabled": False,
            "reason": "too_few_valid_bins",
            "spline_x": [],
            "spline_y": [],
            "bins": rows,
        }
    return {
        "enabled": True,
        "edges": edges.tolist(),
        "tree_ccp_alpha": tree_alpha,
        "tree_min_leaf_frac": float(tree_min_leaf_frac),
        "tree_min_samples_leaf": int(min_leaf),
        "tree_max_leaf_nodes": int(max_leaf_nodes),
        "tree_max_depth": int(max_depth),
        "tree_max_bin_share": float(max_bin_share),
        "tree_top_bin_counts": balance_counts,
        "min_bucket_n": int(min_bucket_n),
        "shrink_k": float(shrink_k),
        "spline_x": centers,
        "spline_y": ys,
        "bins": rows,
    }


def _fit_feature_stats_for_params(
    regime_df: pd.DataFrame,
    features: Sequence[str],
    percentile_refs: Dict[str, np.ndarray],
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    rank_weight: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[Dict[str, Any], np.ndarray]:
    stats: Dict[str, Any] = {}
    for feat in features:
        pct = _apply_percentile(regime_df[feat].values, percentile_refs[feat])
        feat_stats = _fit_feature_stats(
            pct,
            scores,
            returns,
            timestamps,
            min_bucket_n=int(params.get("min_bucket_n", 300)),
            shrink_k=float(params.get("shrink_k", 1500.0)),
            tree_min_leaf_frac=float(params.get("tree_min_leaf_frac", 0.05)),
            max_leaf_nodes=int(params.get("max_leaf_nodes", 4)),
            max_depth=int(params.get("max_depth", 3)),
            ccp_alpha=float(params.get("ccp_alpha", 0.001)),
            max_bin_share=float(params.get("max_bin_share", 0.72)),
            rank_weight=rank_weight,
        )
        feat_stats["log_effect_clip"] = list(
            params.get("log_effect_clip", [-0.10, 0.10])
        )
        stats[feat] = feat_stats
    effects = _effects_from_artifact(
        regime_df,
        {
            "features": list(features),
            "percentile_refs": {k: v.tolist() for k, v in percentile_refs.items()},
            "feature_splines": stats,
        },
    )
    return stats, effects


def _select_spline_hyperparams(
    regime_df: pd.DataFrame,
    features: Sequence[str],
    percentile_refs: Dict[str, np.ndarray],
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    rank_weight: np.ndarray,
) -> Tuple[Dict[str, Any], Dict[str, Any], np.ndarray]:
    base_params: Dict[str, Any] = {
        "min_bucket_n": 300,
        "shrink_k": 1500.0,
        "tree_min_leaf_frac": 0.05,
        "max_leaf_nodes": 4,
        "max_depth": 3,
        "ccp_alpha": 0.001,
        "max_bin_share": 0.72,
        "log_effect_clip": [-0.10, 0.10],
    }

    def score_params(
        params: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any], np.ndarray]:
        stats, effects = _fit_feature_stats_for_params(
            regime_df,
            features,
            percentile_refs,
            scores,
            returns,
            timestamps,
            rank_weight,
            params,
        )
        enabled = sum(1 for s in stats.values() if bool(s.get("enabled", False)))
        if enabled <= 0:
            return float("-inf"), stats, effects
        log_effect = np.clip(
            np.sum(effects.astype(np.float64), axis=1),
            -0.35,
            0.22,
        )
        adjusted = scores * np.clip(np.exp(log_effect), 0.70, 1.25)
        value = _objective(scores, adjusted, returns, timestamps)
        value += 0.002 * min(enabled, 8)
        return float(value), stats, effects

    best_score, best_stats, best_effects = score_params(base_params)
    best_params = dict(base_params)
    trials: List[Dict[str, Any]] = [
        {
            "trial": -1,
            "value": float(best_score),
            "params": dict(base_params),
            "enabled_features": int(
                sum(1 for s in best_stats.values() if bool(s.get("enabled", False)))
            ),
        }
    ]
    if optuna is not None:
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42) if TPESampler is not None else None,
            pruner=(
                MedianPruner(n_startup_trials=4, n_min_trials=2)
                if MedianPruner is not None
                else None
            ),
        )

        def objective(trial: Any) -> float:
            clip_hi = float(trial.suggest_float("log_effect_clip_hi", 0.06, 0.14))
            params = {
                "min_bucket_n": int(
                    trial.suggest_categorical("min_bucket_n", [200, 300, 450])
                ),
                "shrink_k": float(
                    trial.suggest_categorical("shrink_k", [800.0, 1500.0, 2500.0])
                ),
                "tree_min_leaf_frac": float(
                    trial.suggest_float("tree_min_leaf_frac", 0.035, 0.09)
                ),
                "max_leaf_nodes": int(
                    trial.suggest_categorical("max_leaf_nodes", [3, 4])
                ),
                "max_depth": int(trial.suggest_categorical("max_depth", [2, 3])),
                "ccp_alpha": float(
                    trial.suggest_categorical(
                        "ccp_alpha", [0.003, 0.001, 0.0003, 0.0001]
                    )
                ),
                "max_bin_share": float(
                    trial.suggest_float("max_bin_share", 0.60, 0.78)
                ),
                "log_effect_clip": [-clip_hi, clip_hi],
            }
            value, stats, _effects = score_params(params)
            enabled = int(
                sum(1 for s in stats.values() if bool(s.get("enabled", False)))
            )
            trial.set_user_attr("enabled_features", enabled)
            trial.report(float(value), step=0)
            if trial.should_prune():
                raise optuna.TrialPruned()
            return float(value)

        study.optimize(
            objective, n_trials=12, gc_after_trial=True, show_progress_bar=False
        )
        for tr in study.trials:
            if tr.value is None:
                continue
            params = {
                "min_bucket_n": int(tr.params.get("min_bucket_n", 300)),
                "shrink_k": float(tr.params.get("shrink_k", 1500.0)),
                "tree_min_leaf_frac": float(tr.params.get("tree_min_leaf_frac", 0.05)),
                "max_leaf_nodes": int(tr.params.get("max_leaf_nodes", 4)),
                "max_depth": int(tr.params.get("max_depth", 3)),
                "ccp_alpha": float(tr.params.get("ccp_alpha", 0.001)),
                "max_bin_share": float(tr.params.get("max_bin_share", 0.72)),
                "log_effect_clip": [
                    -float(tr.params.get("log_effect_clip_hi", 0.10)),
                    float(tr.params.get("log_effect_clip_hi", 0.10)),
                ],
            }
            trials.append(
                {
                    "trial": int(tr.number),
                    "value": float(tr.value),
                    "params": params,
                    "enabled_features": int(tr.user_attrs.get("enabled_features", 0)),
                }
            )
        if study.best_trial is not None and study.best_value > best_score:
            bp = study.best_trial.params
            clip_hi = float(bp.get("log_effect_clip_hi", 0.10))
            best_params = {
                "min_bucket_n": int(bp.get("min_bucket_n", 300)),
                "shrink_k": float(bp.get("shrink_k", 1500.0)),
                "tree_min_leaf_frac": float(bp.get("tree_min_leaf_frac", 0.05)),
                "max_leaf_nodes": int(bp.get("max_leaf_nodes", 4)),
                "max_depth": int(bp.get("max_depth", 3)),
                "ccp_alpha": float(bp.get("ccp_alpha", 0.001)),
                "max_bin_share": float(bp.get("max_bin_share", 0.72)),
                "log_effect_clip": [-clip_hi, clip_hi],
            }
            best_score, best_stats, best_effects = score_params(best_params)

    best_params["objective"] = float(best_score)
    best_params["trials"] = _jsonify(trials)
    return best_params, best_stats, best_effects


def _effects_from_artifact(
    regime_df: pd.DataFrame, artifact: Dict[str, Any]
) -> np.ndarray:
    features = list(artifact.get("features", []))
    n = len(regime_df)
    effects = np.zeros((n, len(features)), dtype=np.float32)
    refs = artifact.get("percentile_refs", {})
    stats = artifact.get("feature_splines", {})
    for j, feat in enumerate(features):
        if feat not in regime_df.columns:
            continue
        pct = _apply_percentile(
            regime_df[feat].to_numpy(dtype=np.float64),
            np.asarray(refs.get(feat, [0.0]), dtype=np.float64),
        )
        effects[:, j] = _feature_effect_from_stats(pct, stats.get(feat, {}))
    return effects


def _model_matrix_from_regime_features(
    regime_df: pd.DataFrame, features: Sequence[str]
) -> np.ndarray:
    cols = [f for f in features if f in regime_df.columns]
    if not cols:
        return np.empty((len(regime_df), 0), dtype=np.float64)
    return (
        regime_df[cols]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )


def _path_quality_ratio(
    feature_frame: pd.DataFrame,
    n: int,
    mfe: Optional[Sequence[float]] = None,
    mae: Optional[Sequence[float]] = None,
) -> np.ndarray:
    mfe_arr = _as_float_array(mfe, n) if mfe is not None else None
    mae_arr = _as_float_array(mae, n) if mae is not None else None
    if mfe_arr is None:
        mfe_arr = _col(feature_frame, ("mfe_ret", "mfe", "mfe_mean", "oof_mfe_hat"), n)
    if mae_arr is None:
        mae_arr = _col(feature_frame, ("mae_ret", "mae", "mae_mean", "oof_mae_q70_hat"), n)
    if mfe_arr is None or mae_arr is None:
        return np.full(n, np.nan, dtype=np.float64)
    return np.asarray(mfe_arr, dtype=np.float64) / (np.abs(np.asarray(mae_arr, dtype=np.float64)) + EPS)


def _path_quality_components(
    feature_frame: pd.DataFrame,
    n: int,
    *,
    mfe: Optional[Sequence[float]] = None,
    mae: Optional[Sequence[float]] = None,
    t_mfe: Optional[Sequence[float]] = None,
    t_mae: Optional[Sequence[float]] = None,
) -> Dict[str, np.ndarray]:
    mfe_arr = _as_float_array(mfe, n) if mfe is not None else None
    mae_arr = _as_float_array(mae, n) if mae is not None else None
    if mfe_arr is None:
        mfe_arr = _col(feature_frame, ("mfe_ret", "mfe", "mfe_mean", "oof_mfe_hat"), n)
    if mae_arr is None:
        mae_arr = _col(feature_frame, ("mae_ret", "mae", "mae_mean", "oof_mae_q70_hat"), n)
    if mfe_arr is None:
        mfe_arr = np.full(n, np.nan, dtype=np.float64)
    if mae_arr is None:
        mae_arr = np.full(n, np.nan, dtype=np.float64)
    tm = _as_float_array(t_mfe, n) if t_mfe is not None else None
    ta = _as_float_array(t_mae, n) if t_mae is not None else None
    if tm is None:
        tm = _col(feature_frame, ("t_mfe", "time_to_mfe", "bars_to_mfe"), n)
    if ta is None:
        ta = _col(feature_frame, ("t_mae", "time_to_mae", "bars_to_mae"), n)
    if tm is None:
        tm = np.full(n, np.nan, dtype=np.float64)
    if ta is None:
        ta = np.full(n, np.nan, dtype=np.float64)
    tm_filled = np.where(np.isfinite(tm), tm, np.nan)
    ta_filled = np.where(np.isfinite(ta), ta, np.nan)
    horizon = np.where(
        np.isfinite(tm_filled) & np.isfinite(ta_filled),
        np.maximum(tm_filled, ta_filled),
        np.where(np.isfinite(tm_filled), tm_filled, ta_filled),
    )
    horizon = np.where(np.isfinite(horizon) & (horizon > 0.0), horizon, np.nan)
    ratio = np.asarray(mfe_arr, dtype=np.float64) / (
        np.abs(np.asarray(mae_arr, dtype=np.float64)) + EPS
    )
    t_mfe_frac = tm / (horizon + EPS)
    t_mae_frac = ta / (horizon + EPS)
    timing_advantage = (ta - tm) / (horizon + EPS)
    ratio_log = np.log((np.maximum(mfe_arr, 0.0) + 1e-6) / (np.maximum(mae_arr, 0.0) + 1e-6))
    path_score = ratio_log + 0.50 * np.where(np.isfinite(timing_advantage), timing_advantage, 0.0)
    return {
        "mfe": np.asarray(mfe_arr, dtype=np.float64),
        "mae": np.asarray(mae_arr, dtype=np.float64),
        "ratio": ratio,
        "t_mfe": tm,
        "t_mae": ta,
        "t_mfe_frac": t_mfe_frac,
        "t_mae_frac": t_mae_frac,
        "timing_advantage": timing_advantage,
        "path_score": path_score,
    }


def _trust_volatility_scale(
    feature_frame: pd.DataFrame,
    utility: np.ndarray,
) -> np.ndarray:
    n = len(utility)
    for col in (
        "atr_12_15m",
        "atr_pct",
        "atr_pct_base",
        "asset_atr_level",
        "realized_volatility_24h",
        "rv_24h",
    ):
        if col in feature_frame.columns:
            vals = pd.to_numeric(feature_frame[col], errors="coerce").to_numpy(dtype=np.float64)[:n]
            vals = np.abs(vals)
            finite = np.isfinite(vals) & (vals > 1e-6)
            if int(np.sum(finite)) >= max(20, int(0.10 * n)):
                fill = float(np.nanmedian(vals[finite]))
                return np.where(finite, vals, fill)
    finite_u = np.isfinite(utility)
    fallback = float(np.nanmedian(np.abs(utility[finite_u]))) if finite_u.any() else 0.01
    return np.full(n, max(fallback, 1e-4), dtype=np.float64)


def _finite_quantile(values: np.ndarray, q: float, default: float) -> float:
    vals = np.asarray(values, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float(default)
    return float(np.nanquantile(vals, q))


def _trust_targets(
    returns: np.ndarray,
    feature_frame: pd.DataFrame,
    cost_pct: float,
    *,
    gross_returns: Optional[Sequence[float]] = None,
    policy_returns: Optional[Sequence[float]] = None,
    mfe: Optional[Sequence[float]] = None,
    mae: Optional[Sequence[float]] = None,
    t_mfe: Optional[Sequence[float]] = None,
    t_mae: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    n = len(returns)
    net = np.asarray(returns, dtype=np.float64)
    utility = (
        _as_float_array(policy_returns, n)
        if policy_returns is not None
        else net
    )
    gross = (
        _as_float_array(gross_returns, n)
        if gross_returns is not None
        else net + float(cost_pct)
    )
    path = _path_quality_components(
        feature_frame.iloc[:n],
        n,
        mfe=mfe,
        mae=mae,
        t_mfe=t_mfe,
        t_mae=t_mae,
    )
    ratio = path["ratio"]
    path_score = path["path_score"]
    scale = _trust_volatility_scale(feature_frame.iloc[:n], utility)
    utility_norm = utility / np.maximum(scale, 1e-4)
    finite = np.isfinite(utility) & np.isfinite(utility_norm)
    util_med = _finite_quantile(utility, 0.50, 0.0)
    norm_good = _finite_quantile(utility_norm[finite], 0.67, 0.0)
    norm_bad = _finite_quantile(utility_norm[finite], 0.33, 0.0)
    util_good = max(0.0, _finite_quantile(utility[finite], 0.58, util_med))
    util_bad = min(0.0, _finite_quantile(utility[finite], 0.42, util_med))
    path_good = _finite_quantile(path_score, 0.40, 0.0)
    path_bad = _finite_quantile(path_score, 0.25, -0.5)
    clean_path = np.isfinite(path_score) & (path_score >= path_good)
    adverse_path = np.isfinite(path_score) & (path_score <= path_bad)
    y = np.zeros(n, dtype=np.int64)
    good = (
        finite
        & (utility > util_good)
        & (utility_norm >= norm_good)
        & clean_path
    )
    bad = (
        finite
        & (
            (utility < util_bad)
            | (utility_norm <= norm_bad)
            | ((gross < -float(cost_pct)) & adverse_path)
            | ((utility <= util_med) & adverse_path)
        )
    )
    y[good] = 1
    y[bad] = -1
    for good_q, bad_q in ((0.62, 0.38), (0.58, 0.42), (0.55, 0.45)):
        counts = {k: int(np.sum(y == k)) for k in (-1, 0, 1)}
        if min(counts.values()) >= max(10, int(0.02 * n)):
            break
        y[:] = 0
        norm_good = _finite_quantile(utility_norm[finite], good_q, 0.0)
        norm_bad = _finite_quantile(utility_norm[finite], bad_q, 0.0)
        path_good = _finite_quantile(path_score, 0.35, 0.0)
        path_bad = _finite_quantile(path_score, 0.30, -0.5)
        clean_path = np.isfinite(path_score) & (path_score >= path_good)
        adverse_path = np.isfinite(path_score) & (path_score <= path_bad)
        y[
            finite
            & (utility > max(0.0, _finite_quantile(utility[finite], 0.52, util_med)))
            & (utility_norm >= norm_good)
            & clean_path
        ] = 1
        y[
            finite
            & (
                (utility < min(0.0, _finite_quantile(utility[finite], 0.48, util_med)))
                | (utility_norm <= norm_bad)
                | ((utility <= util_med) & adverse_path)
            )
        ] = -1
    spec = {
        "target_source": "policy_realized_utility" if policy_returns is not None else "raw_realized_return",
        "utility_good_threshold": util_good,
        "utility_bad_threshold": util_bad,
        "utility_norm_good_threshold": norm_good,
        "utility_norm_bad_threshold": norm_bad,
        "path_score_good_threshold": path_good,
        "path_score_bad_threshold": path_bad,
        "path_features_used_for_target_only": True,
        "class_counts": {str(k): int(np.sum(y == k)) for k in (-1, 0, 1)},
        "median_policy_utility": _finite_quantile(utility, 0.50, 0.0),
        "median_path_ratio": _finite_quantile(ratio, 0.50, 1.0),
        "median_timing_advantage": _finite_quantile(path["timing_advantage"], 0.50, 0.0),
    }
    return y, spec


def _serialize_logistic_model(
    model: LogisticRegression,
    scaler: RobustScaler,
    classes: np.ndarray,
    features: Sequence[str],
) -> Dict[str, Any]:
    return {
        "enabled": True,
        "model_type": "multinomial_logistic_regression",
        "features": list(features),
        "classes": np.asarray(classes, dtype=int).tolist(),
        "coef": np.asarray(model.coef_, dtype=float).tolist(),
        "intercept": np.asarray(model.intercept_, dtype=float).tolist(),
        "scaler": {
            "center": np.asarray(getattr(scaler, "center_", []), dtype=float).tolist(),
            "scale": np.asarray(getattr(scaler, "scale_", []), dtype=float).tolist(),
        },
        "trust_score_formula": "P_good - 1.2 * P_bad",
    }


def _serialize_binary_logistic_model(
    model: LogisticRegression,
    scaler: RobustScaler,
) -> Dict[str, Any]:
    return {
        "coef": np.asarray(model.coef_, dtype=float).tolist(),
        "intercept": np.asarray(model.intercept_, dtype=float).tolist(),
        "classes": np.asarray(model.classes_, dtype=int).tolist(),
        "scaler": {
            "center": np.asarray(getattr(scaler, "center_", []), dtype=float).tolist(),
            "scale": np.asarray(getattr(scaler, "scale_", []), dtype=float).tolist(),
        },
    }


def _predict_binary_logistic_model(x: np.ndarray, spec: Dict[str, Any]) -> np.ndarray:
    n = x.shape[0]
    coef = np.asarray(spec.get("coef", []), dtype=np.float64)
    intercept = np.asarray(spec.get("intercept", []), dtype=np.float64)
    classes = np.asarray(spec.get("classes", [0, 1]), dtype=int)
    scaler = spec.get("scaler", {})
    center = np.asarray(scaler.get("center", np.zeros(x.shape[1])), dtype=np.float64)
    scale = np.asarray(scaler.get("scale", np.ones(x.shape[1])), dtype=np.float64)
    if coef.ndim != 2 or coef.shape[1] != x.shape[1] or coef.shape[0] < 1:
        return np.full(n, 0.5, dtype=np.float64)
    if len(center) != x.shape[1]:
        center = np.zeros(x.shape[1], dtype=np.float64)
    if len(scale) != x.shape[1]:
        scale = np.ones(x.shape[1], dtype=np.float64)
    xs = (x - center) / np.where(np.abs(scale) > EPS, scale, 1.0)
    if coef.shape[0] == 1:
        logits = xs @ coef[0] + float(intercept[0] if len(intercept) else 0.0)
        p_pos = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
        if len(classes) >= 2 and classes[-1] == 0:
            p_pos = 1.0 - p_pos
        return p_pos.astype(np.float64)
    logits = xs @ coef.T + intercept
    logits -= np.nanmax(logits, axis=1, keepdims=True)
    exp_logits = np.exp(np.clip(logits, -50, 50))
    proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    pos_idx = int(np.where(classes == 1)[0][0]) if np.any(classes == 1) else proba.shape[1] - 1
    return proba[:, pos_idx].astype(np.float64)


def _ordinal_proba_from_thresholds(
    p_gt_bad: np.ndarray,
    p_good: np.ndarray,
) -> np.ndarray:
    p_gt_bad = np.clip(np.asarray(p_gt_bad, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    p_good = np.clip(np.asarray(p_good, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    p_good = np.minimum(p_good, p_gt_bad)
    p_bad = 1.0 - p_gt_bad
    p_neutral = np.maximum(p_gt_bad - p_good, 1e-6)
    proba = np.column_stack([p_good, p_neutral, p_bad])
    proba = proba / np.sum(proba, axis=1, keepdims=True)
    return proba


def _predict_serialized_logistic(
    x: np.ndarray, model_spec: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    n = x.shape[0]
    classes = np.asarray(model_spec.get("classes", [-1, 0, 1]), dtype=int)
    coef = np.asarray(model_spec.get("coef", []), dtype=np.float64)
    intercept = np.asarray(model_spec.get("intercept", []), dtype=np.float64)
    scaler = model_spec.get("scaler", {})
    center = np.asarray(scaler.get("center", np.zeros(x.shape[1])), dtype=np.float64)
    scale = np.asarray(scaler.get("scale", np.ones(x.shape[1])), dtype=np.float64)
    if coef.ndim != 2 or coef.shape[1] != x.shape[1] or len(intercept) != coef.shape[0]:
        proba = np.full((n, 3), 1.0 / 3.0, dtype=np.float64)
        return proba, np.zeros(n, dtype=np.float64)
    if len(center) != x.shape[1]:
        center = np.zeros(x.shape[1], dtype=np.float64)
    if len(scale) != x.shape[1]:
        scale = np.ones(x.shape[1], dtype=np.float64)
    xs = (x - center) / np.where(np.abs(scale) > EPS, scale, 1.0)
    logits = xs @ coef.T + intercept
    if coef.shape[0] == 1 and len(classes) == 2:
        p1 = 1.0 / (1.0 + np.exp(-np.clip(logits[:, 0], -50, 50)))
        raw_proba = np.column_stack([1.0 - p1, p1])
    else:
        logits -= np.nanmax(logits, axis=1, keepdims=True)
        exp_logits = np.exp(np.clip(logits, -50, 50))
        raw_proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    proba = np.zeros((n, 3), dtype=np.float64)
    for j, cls in enumerate(classes):
        if cls == -1:
            proba[:, 2] = raw_proba[:, j]
        elif cls == 0:
            proba[:, 1] = raw_proba[:, j]
        elif cls == 1:
            proba[:, 0] = raw_proba[:, j]
    missing = np.sum(proba, axis=1) <= 0
    proba[missing, :] = 1.0 / 3.0
    trust_score = proba[:, 0] - 1.2 * proba[:, 2]
    return proba, trust_score


def _predict_serialized_trust_model(
    x: np.ndarray, model_spec: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    if str(model_spec.get("model_type", "")) != "ordinal_logistic_plus_multinomial":
        return _predict_serialized_logistic(x, model_spec)
    ordinal = model_spec.get("ordinal_models", {})
    p_gt_bad = _predict_binary_logistic_model(x, ordinal.get("gt_bad", {}))
    p_good = _predict_binary_logistic_model(x, ordinal.get("gt_neutral", {}))
    ordinal_proba = _ordinal_proba_from_thresholds(p_gt_bad, p_good)
    secondary = model_spec.get("secondary_classifier", {})
    if bool(secondary.get("enabled", False)):
        secondary_proba, _ = _predict_serialized_logistic(x, secondary)
        blend = float(model_spec.get("secondary_blend_weight", 0.25))
        blend = float(np.clip(blend, 0.0, 0.5))
        proba = (1.0 - blend) * ordinal_proba + blend * secondary_proba
        proba = proba / np.sum(proba, axis=1, keepdims=True)
    else:
        proba = ordinal_proba
    trust_score = proba[:, 0] - 1.2 * proba[:, 2]
    return proba, trust_score


def _fit_trust_model(
    regime_df: pd.DataFrame,
    features: Sequence[str],
    returns: np.ndarray,
    feature_frame: pd.DataFrame,
    timestamps: Optional[np.ndarray],
    cost_pct: float,
    *,
    gross_returns: Optional[Sequence[float]] = None,
    policy_returns: Optional[Sequence[float]] = None,
    mfe: Optional[Sequence[float]] = None,
    mae: Optional[Sequence[float]] = None,
    t_mfe: Optional[Sequence[float]] = None,
    t_mae: Optional[Sequence[float]] = None,
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    n = len(regime_df)
    y, target_spec = _trust_targets(
        returns,
        feature_frame.iloc[:n],
        cost_pct,
        gross_returns=gross_returns,
        policy_returns=policy_returns,
        mfe=mfe,
        mae=mae,
        t_mfe=t_mfe,
        t_mae=t_mae,
    )
    x = _model_matrix_from_regime_features(regime_df, features)
    neutral_proba = np.full((n, 3), 1.0 / 3.0, dtype=np.float64)
    if x.shape[1] == 0 or n < 80 or len(np.unique(y)) < 2:
        return (
            {
                "enabled": False,
                "reason": "insufficient_classes_or_features",
                "target_spec": target_spec,
            },
            neutral_proba,
            np.zeros(n, dtype=np.float64),
            y,
        )
    oof = np.full((n, 3), 1.0 / 3.0, dtype=np.float64)
    splits = _walk_forward_splits(timestamps if timestamps is not None else np.arange(n), n, 5)
    for tr, va in splits:
        if len(np.unique(y[tr])) < 2:
            continue
        ordinal_fold = np.full((len(va), 3), 1.0 / 3.0, dtype=np.float64)
        scaler = RobustScaler()
        xtr = scaler.fit_transform(x[tr])
        xva = scaler.transform(x[va])
        try:
            gt_bad = (y[tr] > -1).astype(int)
            if len(np.unique(gt_bad)) == 2:
                clf_gt_bad = LogisticRegression(
                    C=0.5,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                )
                clf_gt_bad.fit(xtr, gt_bad)
                p_gt_bad = clf_gt_bad.predict_proba(xva)[:, list(clf_gt_bad.classes_).index(1)]
            else:
                p_gt_bad = np.full(len(va), float(np.mean(gt_bad)), dtype=np.float64)
            gt_neutral = (y[tr] > 0).astype(int)
            if len(np.unique(gt_neutral)) == 2:
                clf_gt_neutral = LogisticRegression(
                    C=0.5,
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                )
                clf_gt_neutral.fit(xtr, gt_neutral)
                p_good = clf_gt_neutral.predict_proba(xva)[:, list(clf_gt_neutral.classes_).index(1)]
            else:
                p_good = np.full(len(va), float(np.mean(gt_neutral)), dtype=np.float64)
            ordinal_fold = _ordinal_proba_from_thresholds(p_gt_bad, p_good)
        except Exception:
            ordinal_fold = np.full((len(va), 3), 1.0 / 3.0, dtype=np.float64)
        clf = LogisticRegression(
            C=0.5,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        try:
            clf.fit(xtr, y[tr])
            pred = clf.predict_proba(xva)
            secondary_fold = np.full((len(va), 3), 1.0 / 3.0, dtype=np.float64)
            for j, cls in enumerate(clf.classes_):
                if cls == 1:
                    secondary_fold[:, 0] = pred[:, j]
                elif cls == 0:
                    secondary_fold[:, 1] = pred[:, j]
                elif cls == -1:
                    secondary_fold[:, 2] = pred[:, j]
            oof[va] = 0.75 * ordinal_fold + 0.25 * secondary_fold
            oof[va] = oof[va] / np.sum(oof[va], axis=1, keepdims=True)
        except Exception:
            oof[va] = ordinal_fold
            continue
    trust_score = oof[:, 0] - 1.2 * oof[:, 2]
    scaler = RobustScaler()
    xs = scaler.fit_transform(x)
    try:
        gt_bad_final = LogisticRegression(
            C=0.5,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        gt_bad_final.fit(xs, (y > -1).astype(int))
        gt_neutral_final = LogisticRegression(
            C=0.5,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        gt_neutral_final.fit(xs, (y > 0).astype(int))
        secondary = LogisticRegression(
            C=0.5,
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        secondary.fit(xs, y)
        spec = {
            "enabled": True,
            "model_type": "ordinal_logistic_plus_multinomial",
            "features": list(features),
            "classes": [-1, 0, 1],
            "ordinal_models": {
                "gt_bad": _serialize_binary_logistic_model(gt_bad_final, scaler),
                "gt_neutral": _serialize_binary_logistic_model(gt_neutral_final, scaler),
            },
            "secondary_classifier": _serialize_logistic_model(
                secondary, scaler, secondary.classes_, features
            ),
            "secondary_blend_weight": 0.25,
            "trust_score_formula": "P_good - 1.2 * P_bad",
            "target_spec": target_spec,
        }
        spec["target_counts"] = {str(k): int(np.sum(y == k)) for k in (-1, 0, 1)}
    except Exception as exc:
        spec = {"enabled": False, "reason": f"fit_failed:{exc}", "target_spec": target_spec}
    return spec, oof, trust_score, y


def apply_regime_adaptor(
    feature_frame: pd.DataFrame,
    pred_calibrated: Sequence[float],
    artifact: Dict[str, Any],
    timestamps: Optional[Sequence[Any]] = None,
    symbols: Optional[Sequence[Any]] = None,
) -> Dict[str, np.ndarray]:
    n = len(pred_calibrated)
    score = _as_float_array(pred_calibrated, n)
    if (
        artifact.get("schema_version") in {"rolling_bad_regime_v2", "rolling_regime_v1"}
        or "selected_combination_params" in artifact
    ):
        regime_df, mapping = build_regime_feature_frame(
            feature_frame.iloc[:n], timestamps, symbols
        )
        required = REQUIRED_LIVE_BAD_REGIME_COLUMNS
        missing_live_columns = [c for c in required if c not in feature_frame.columns]
        available = not missing_live_columns
        requested_enabled = bool(artifact.get("enable_regime_adaptor", False))
        live_enabled = bool(requested_enabled and available)
        p_global_3d = _col(feature_frame, ("p_bad_regime_global_3d",), n)
        p_global_5d = _col(feature_frame, ("p_bad_regime_global_5d",), n)
        p_asset_3d = _col(feature_frame, ("p_bad_regime_asset_3d",), n)
        p_asset_5d = _col(feature_frame, ("p_bad_regime_asset_5d",), n)
        if not live_enabled:
            p_global_3d = p_global_5d = p_asset_3d = p_asset_5d = np.full(n, 0.5)
        else:
            p_global_3d = np.clip(_fill_numeric(p_global_3d, 0.5), 0.0, 1.0)
            p_global_5d = np.clip(_fill_numeric(p_global_5d, 0.5), 0.0, 1.0)
            p_asset_3d = np.clip(_fill_numeric(p_asset_3d, 0.5), 0.0, 1.0)
            p_asset_5d = np.clip(_fill_numeric(p_asset_5d, 0.5), 0.0, 1.0)
        blend = artifact.get("selected_3d_5d_blend", {"3d": 0.6, "5d": 0.4})
        w3 = float(blend.get("3d", 0.6))
        w5 = float(blend.get("5d", 0.4))
        combined_global = w3 * np.asarray(
            p_global_3d, dtype=np.float64
        ) + w5 * np.asarray(p_global_5d, dtype=np.float64)
        combined_asset = w3 * np.asarray(
            p_asset_3d, dtype=np.float64
        ) + w5 * np.asarray(p_asset_5d, dtype=np.float64)
        params = (
            artifact.get("selected_combination_params", {})
            if isinstance(artifact.get("selected_combination_params", {}), dict)
            else {}
        )
        combined = combine_meta_bad_regime_scores(
            score, combined_global, combined_asset, params=params
        )
        deployment_pre_rank = (
            combined["final_score"] if live_enabled else score.copy()
        )
        local_rank = _global_rank(deployment_pre_rank)
        return {
            "regime_weight": np.ones(n, dtype=np.float64),
            "eligible": np.ones(n, dtype=bool),
            "deployment_score_pre_rank": deployment_pre_rank.astype(np.float64),
            "local_batch_rank": local_rank.astype(np.float64),
            "p_bad_regime_global_3d": np.asarray(p_global_3d, dtype=np.float64),
            "p_bad_regime_global_5d": np.asarray(p_global_5d, dtype=np.float64),
            "p_bad_regime_asset_3d": np.asarray(p_asset_3d, dtype=np.float64),
            "p_bad_regime_asset_5d": np.asarray(p_asset_5d, dtype=np.float64),
            "combined_global_bad_regime_score": combined_global.astype(np.float64),
            "combined_asset_bad_regime_score": combined_asset.astype(np.float64),
            "bad_regime_offset": combined["bad_regime_offset"].astype(np.float64),
            "combined_score": combined["final_score"].astype(np.float64),
            "rank_scope": np.repeat("local_batch", n),
            "live_required_columns_available": np.repeat(bool(available), n),
            "missing_live_p_bad_regime_columns": np.repeat(
                ",".join(missing_live_columns), n
            ),
            "score_delta_from_regime_adjustment": combined[
                "score_delta_from_regime_adjustment"
            ].astype(np.float64),
            "regime_adjustment_enabled": np.repeat(live_enabled, n),
            "regime_disabled_reason": np.repeat(
                ""
                if live_enabled
                else (
                    "missing_live_p_bad_regime_columns"
                    if requested_enabled and missing_live_columns
                    else "artifact_disabled"
                ),
                n,
            ),
            "selected_combination_params": np.repeat(
                json.dumps(params, sort_keys=True), n
            ),
            "feature_mapping": mapping,
        }
    regime_df, _ = build_regime_feature_frame(
        feature_frame.iloc[:n], timestamps, symbols
    )
    sym_arr = (
        np.asarray(symbols).astype(str)[:n]
        if symbols is not None and len(symbols) >= n
        else np.repeat("all", n)
    )
    regime_df = _append_distribution_features(
        regime_df,
        feature_frame.iloc[:n],
        artifact.get("distribution_feature_spec", {}),
        timestamps,
        sym_arr,
    )
    regime_df, _ = _append_prediction_reliability_features(
        regime_df,
        feature_frame.iloc[:n],
        score,
        None,
        timestamps,
        sym_arr,
        artifact,
    )
    regime_df, _ = _append_symbol_features(
        regime_df,
        None,
        timestamps,
        sym_arr,
        artifact,
    )
    effects = _effects_from_artifact(regime_df, artifact)
    scaler = artifact.get("elastic_net", {}).get("scaler", {})
    center = np.asarray(
        scaler.get("center", np.zeros(effects.shape[1])), dtype=np.float64
    )
    scale = np.asarray(scaler.get("scale", np.ones(effects.shape[1])), dtype=np.float64)
    if len(center) != effects.shape[1]:
        center = np.zeros(effects.shape[1], dtype=np.float64)
    if len(scale) != effects.shape[1]:
        scale = np.ones(effects.shape[1], dtype=np.float64)
    x_scaled = (effects.astype(np.float64) - center) / np.where(
        np.abs(scale) > EPS, scale, 1.0
    )
    coefs = np.asarray(
        artifact.get("elastic_net", {}).get("coef", np.zeros(effects.shape[1])),
        dtype=np.float64,
    )
    if len(coefs) != effects.shape[1]:
        coefs = np.zeros(effects.shape[1], dtype=np.float64)
    intercept = float(artifact.get("elastic_net", {}).get("intercept", 0.0))
    train_mean = float(
        artifact.get("elastic_net", {}).get("train_prediction_mean", 0.0)
    )
    log_weight = x_scaled @ coefs + intercept - train_mean
    clips = artifact.get("clips", {})
    log_lo, log_hi = clips.get("total_log_weight_clip", [-0.35, 0.22])
    wt_lo, wt_hi = clips.get("regime_weight_clip", [0.70, 1.25])
    log_weight = np.clip(log_weight, float(log_lo), float(log_hi))
    weight = np.clip(np.exp(log_weight), float(wt_lo), float(wt_hi)).astype(np.float64)
    eligible = np.ones(n, dtype=bool)
    if bool(artifact.get("enable_regime_adaptor", False)):
        eligible &= ~_apply_bucket_gates(regime_df, artifact)
        eligible &= ~_apply_asset_gates(
            (
                np.asarray(symbols).astype(str)[:n]
                if symbols is not None and len(symbols) >= n
                else np.repeat("all", n)
            ),
            artifact,
        )
    else:
        weight[:] = 1.0
    trust_model = artifact.get("trust_model", {})
    trust_proba = np.full((n, 3), 1.0 / 3.0, dtype=np.float64)
    trust_score = np.zeros(n, dtype=np.float64)
    trust_multiplier = np.ones(n, dtype=np.float64)
    if bool(trust_model.get("enabled", False)):
        trust_x = _model_matrix_from_regime_features(
            regime_df, trust_model.get("features", artifact.get("features", []))
        )
        trust_proba, trust_score = _predict_serialized_trust_model(trust_x, trust_model)
        trust_multiplier = np.clip(1.0 + 0.25 * trust_score, 0.65, 1.15)
        if bool(artifact.get("enable_regime_adaptor", False)):
            eligible &= trust_score >= float(
                artifact.get("trust_gate_threshold", -0.35)
            )
    deployment = score * weight * trust_multiplier
    deployment[~eligible] = -np.inf
    rank = _rank_pct(np.where(np.isfinite(deployment), deployment, np.nan)).copy()
    rank[~np.isfinite(deployment)] = 0.0
    return {
        "regime_weight": weight.astype(np.float64),
        "eligible": eligible,
        "deployment_score": deployment.astype(np.float64),
        "deployment_score_rank": rank.astype(np.float64),
        "spline_effects": effects.astype(np.float32),
        "trust_score": trust_score.astype(np.float64),
        "trust_proba_good_neutral_bad": trust_proba.astype(np.float64),
        "trust_multiplier": trust_multiplier.astype(np.float64),
    }


def _apply_bucket_gates(
    regime_df: pd.DataFrame, artifact: Dict[str, Any]
) -> np.ndarray:
    n = len(regime_df)
    gated = np.zeros(n, dtype=bool)
    for gate in artifact.get("bucket_gates", []):
        feat = gate.get("feature")
        if feat not in regime_df.columns:
            continue
        ref = np.asarray(
            artifact.get("percentile_refs", {}).get(feat, [0.0]), dtype=np.float64
        )
        pct = _apply_percentile(regime_df[feat].to_numpy(dtype=np.float64), ref)
        lo, hi = float(gate.get("lo", 0.0)), float(gate.get("hi", 1.0))
        gated |= (pct >= lo) & (pct < hi if hi < 1.0 else pct <= hi)
    return gated


def _apply_asset_gates(symbols: np.ndarray, artifact: Dict[str, Any]) -> np.ndarray:
    gated_assets = {str(x) for x in artifact.get("asset_gates", [])}
    if not gated_assets:
        return np.zeros(len(symbols), dtype=bool)
    return np.asarray([str(s) in gated_assets for s in symbols], dtype=bool)


def _fit_elastic_net(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
) -> Tuple[ElasticNet, RobustScaler, float, Dict[str, float]]:
    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x)
    candidates: List[Tuple[float, float]] = [
        (0.00003, 0.05),
        (0.0001, 0.05),
        (0.0003, 0.10),
        (0.001, 0.10),
        (0.003, 0.20),
        (0.01, 0.20),
        (0.1, 0.5),
        (0.3, 0.5),
        (1.0, 0.5),
        (0.3, 0.2),
        (0.3, 0.8),
    ]
    if optuna is not None:
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42) if TPESampler is not None else None,
            pruner=(
                MedianPruner(n_startup_trials=8, n_min_trials=4)
                if MedianPruner is not None
                else None
            ),
        )
        best_seen = {"trial": -1, "value": -np.inf}

        def objective(trial: Any) -> float:
            alpha = float(trial.suggest_float("alpha", 1e-5, 10.0, log=True))
            l1_ratio = float(trial.suggest_float("l1_ratio", 0.1, 0.9))
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=5000,
                random_state=42,
            )
            model.fit(x_scaled, y, sample_weight=weights)
            pred = np.asarray(model.predict(x_scaled), dtype=np.float64)
            pred -= float(np.nanmean(pred))
            adjusted = scores * np.clip(np.exp(np.clip(pred, -0.35, 0.22)), 0.70, 1.25)
            val = _objective(
                raw_scores=scores,
                adjusted_scores=adjusted,
                returns=returns,
                timestamps=timestamps,
            )
            if val > best_seen["value"]:
                best_seen.update({"trial": int(trial.number), "value": float(val)})
            elif int(trial.number) - int(best_seen["trial"]) >= 25:
                study.stop()
            return float(val)

        study.optimize(
            objective, n_trials=50, gc_after_trial=True, show_progress_bar=False
        )
        if study.best_trial is not None:
            candidates.insert(
                0,
                (
                    float(study.best_trial.params["alpha"]),
                    float(study.best_trial.params["l1_ratio"]),
                ),
            )
    best_score = -np.inf
    best_pair = candidates[0]
    for alpha, l1_ratio in candidates:
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=5000,
            random_state=42,
        )
        model.fit(x_scaled, y, sample_weight=weights)
        pred = np.asarray(model.predict(x_scaled), dtype=np.float64)
        pred -= float(np.nanmean(pred))
        adjusted = scores * np.clip(np.exp(np.clip(pred, -0.35, 0.22)), 0.70, 1.25)
        val = _objective(
            raw_scores=scores,
            adjusted_scores=adjusted,
            returns=returns,
            timestamps=timestamps,
        )
        if val > best_score:
            best_score = val
            best_pair = (alpha, l1_ratio)
    final = ElasticNet(
        alpha=best_pair[0],
        l1_ratio=best_pair[1],
        fit_intercept=True,
        max_iter=5000,
        random_state=42,
    )
    final.fit(x_scaled, y, sample_weight=weights)
    train_mean = float(np.nanmean(final.predict(x_scaled)))
    return (
        final,
        scaler,
        train_mean,
        {
            "alpha": float(best_pair[0]),
            "l1_ratio": float(best_pair[1]),
            "objective": float(best_score),
        },
    )


def _objective(
    raw_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
) -> float:
    raw_df = score_metrics(raw_scores, returns, timestamps, top_fracs=(0.01, 0.05))
    adj_df = score_metrics(adjusted_scores, returns, timestamps, top_fracs=(0.01, 0.05))
    weights = {0.01: 0.35, 0.05: 0.65}
    objective = 0.0
    weight_sum = 0.0
    for frac, weight in weights.items():
        raw_rows = raw_df[np.isclose(raw_df["top_frac"].astype(float), frac)]
        adj_rows = adj_df[np.isclose(adj_df["top_frac"].astype(float), frac)]
        if raw_rows.empty or adj_rows.empty:
            continue
        raw = raw_rows.iloc[0].to_dict()
        adj = adj_rows.iloc[0].to_dict()
        std_weekly_ratio = _safe_ratio(adj["std_weekly"], raw["std_weekly"])
        std_monthly_ratio = _safe_ratio(adj["std_monthly"], raw["std_monthly"])
        worst_week_loss_ratio = _safe_ratio(
            adj["worst_week_loss"], raw["worst_week_loss"]
        )
        worst_month_loss_ratio = _safe_ratio(
            adj["worst_month_loss"], raw["worst_month_loss"]
        )
        net_ret_ratio = _safe_ratio(adj["net_ret"], raw["net_ret"])
        lift_ratio = _safe_ratio(adj["lift"], raw["lift"])
        frac_score = float(
            -0.15 * math.log(max(std_weekly_ratio, EPS))
            - 0.15 * math.log(max(std_monthly_ratio, EPS))
            - 0.15 * math.log(max(worst_week_loss_ratio, EPS))
            - 0.15 * math.log(max(worst_month_loss_ratio, EPS))
            + 0.20 * math.log(max(net_ret_ratio, EPS))
            + 0.10 * math.log(max(lift_ratio, EPS))
            - 0.30 * max(0.0, 0.98 - net_ret_ratio)
            - 0.20 * max(0.0, 0.98 - lift_ratio)
        )
        objective += weight * frac_score
        weight_sum += weight
    if weight_sum <= 0.0:
        return float("-inf")
    return float(objective / weight_sum)


def _regime_enable_decision(summary: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    improve_eps = 0.001
    worse_tol = 0.015
    top_priority = {0.05: 0, 0.01: 1}
    best_any_score = -np.inf
    best_pass_key: Tuple[int, float] = (999, np.inf)
    best_any_decision: Dict[str, Any] = {
        "enabled": False,
        "reason": "no_candidate_passed_loose_gate",
        "improve_eps": improve_eps,
        "worse_tol": worse_tol,
        "top_priority": top_priority,
    }
    best_pass_decision: Dict[str, Any] = {}
    for _, r in summary.iterrows():
        top_frac = float(r.get("top_frac", np.nan))
        if round(top_frac, 2) not in top_priority:
            continue
        lift_ratio = float(r.get("lift_ratio", 1.0))
        net_ret_ratio = float(r.get("net_ret_ratio", 1.0))
        gross_ret_ratio = float(r.get("gross_ret_ratio", 1.0))
        std_ratio = float(r.get("std_ratio", 1.0))
        dd_ratio = float(r.get("dd_ratio", 1.0))
        improvements = {
            "lift": lift_ratio > 1.0 + improve_eps,
            "net_ret": net_ret_ratio > 1.0 + improve_eps,
            "gross_ret": gross_ret_ratio > 1.0 + improve_eps,
            "return_std": std_ratio < 1.0 - improve_eps,
            "max_drawdown": dd_ratio < 1.0 - improve_eps,
        }
        no_material_worse = (
            lift_ratio >= 1.0 - worse_tol
            and net_ret_ratio >= 1.0 - worse_tol
            and gross_ret_ratio >= 1.0 - worse_tol
            and std_ratio <= 1.0 + worse_tol
            and dd_ratio <= 1.0 + worse_tol
        )
        score = float(
            0.35 * math.log(max(lift_ratio, EPS))
            + 0.25 * math.log(max(net_ret_ratio, EPS))
            + 0.15 * math.log(max(gross_ret_ratio, EPS))
            - 0.15 * math.log(max(std_ratio, EPS))
            - 0.10 * math.log(max(dd_ratio, EPS))
        )
        enabled = bool(no_material_worse and any(improvements.values()) and score > 0.0)
        decision = {
            "enabled": enabled,
            "reason": (
                "loose_gate_passed" if enabled else "best_candidate_failed_loose_gate"
            ),
            "top_frac": top_frac,
            "selection_score": score,
            "improvements": improvements,
            "no_material_worse": bool(no_material_worse),
            "improve_eps": improve_eps,
            "worse_tol": worse_tol,
            "top_priority": top_priority,
            "ratios": {
                "lift_ratio": lift_ratio,
                "net_ret_ratio": net_ret_ratio,
                "gross_ret_ratio": gross_ret_ratio,
                "std_ratio": std_ratio,
                "dd_ratio": dd_ratio,
            },
        }
        if score > best_any_score:
            best_any_score = score
            best_any_decision = decision
        pass_key = (
            int(top_priority.get(round(top_frac, 2), 999)),
            -score,
        )
        if enabled and pass_key < best_pass_key:
            best_pass_key = pass_key
            best_pass_decision = {
                **decision,
                "enabled": enabled,
            }
    if best_pass_decision:
        return True, best_pass_decision
    return False, best_any_decision


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    arr = np.asarray(x, dtype=np.float64)
    out = 1.0 / (1.0 + np.exp(-np.clip(arr, -50.0, 50.0)))
    return float(out) if np.ndim(x) == 0 else out


def _logit(p: np.ndarray | float) -> np.ndarray | float:
    arr = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    out = np.log(arr / (1.0 - arr))
    return float(out) if np.ndim(p) == 0 else out


def _zscore(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    out = np.zeros(len(x), dtype=np.float64)
    if finite.any():
        med = float(np.nanmedian(x[finite]))
        std = float(np.nanstd(x[finite]))
        out[finite] = (x[finite] - med) / (std if std > EPS else 1.0)
    return out


def compute_hit_rate_surprise(
    net_pnl_per_trade: Sequence[float],
    meta_pred_calibrated: Sequence[float],
    *,
    eps: float = EPS,
) -> Dict[str, float]:
    pnl = np.asarray(net_pnl_per_trade, dtype=np.float64)
    pred = np.clip(np.asarray(meta_pred_calibrated, dtype=np.float64), 0.0, 1.0)
    mask = np.isfinite(pnl) & np.isfinite(pred)
    pnl = pnl[mask]
    pred = pred[mask]
    trade_count = int(len(pnl))
    wins = int(np.sum(pnl > 0.0))
    expected_wins = float(np.sum(pred))
    variance = float(np.sum(pred * (1.0 - pred)))
    return {
        "wins": wins,
        "expected_wins": expected_wins,
        "variance": variance,
        "trade_count": trade_count,
        "expected_hit_rate": expected_wins / max(trade_count, 1),
        "hit_rate_surprise_z": (wins - expected_wins) / math.sqrt(variance + eps),
    }


def _resolve_time_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _point_in_time_rolling(
    df: pd.DataFrame,
    value_col: str,
    anchor_dates: pd.DatetimeIndex,
    window: str,
    *,
    group_col: Optional[str] = None,
    agg: str = "mean",
    abs_value: bool = False,
    higher_worse_depth_risk: bool = False,
) -> np.ndarray:
    if value_col not in df.columns or "_ts" not in df.columns:
        return np.full(len(anchor_dates), np.nan, dtype=np.float64)
    work = df[
        ["_ts", value_col]
        + ([group_col] if group_col and group_col in df.columns else [])
    ].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    if abs_value:
        work[value_col] = work[value_col].abs()
    if higher_worse_depth_risk:
        nonneg = np.maximum(work[value_col].to_numpy(dtype=np.float64), 0.0)
        work[value_col] = 1.0 / np.sqrt(1.0 + nonneg)
    out = []
    delta = pd.Timedelta(window)
    for anchor in anchor_dates:
        hist = work[(work["_ts"] < anchor) & (work["_ts"] >= anchor - delta)]
        vals = hist[value_col].to_numpy(dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            out.append(np.nan)
        elif agg == "p90":
            out.append(float(np.nanquantile(vals, 0.90)))
        elif agg == "max":
            out.append(float(np.nanmax(vals)))
        elif agg == "sum":
            out.append(float(np.nansum(vals)))
        else:
            out.append(float(np.nanmean(vals)))
    return np.asarray(out, dtype=np.float64)


def _prepare_trade_frame(
    trades: pd.DataFrame,
    *,
    timestamp_col: Optional[str],
    net_pnl_col: str,
    wallet_return_col: Optional[str],
    meta_pred_col: str,
    symbol_col: str,
    strategy_id: Optional[str],
) -> pd.DataFrame:
    tcol = timestamp_col or _resolve_time_col(
        trades, ("timestamp", "entry_time", "exit_time", "date")
    )
    if tcol is None:
        raise ValueError(
            "trades must contain a timestamp/entry_time/exit_time/date column"
        )
    df = trades.copy()
    df["_ts"] = pd.to_datetime(df[tcol], utc=True, errors="coerce")
    df = df[df["_ts"].notna()].sort_values("_ts").reset_index(drop=True)
    if symbol_col not in df.columns:
        df[symbol_col] = "all"
    if "strategy_id" not in df.columns:
        df["strategy_id"] = str(strategy_id or "strategy")
    pnl_col = wallet_return_col or (
        "wallet_return" if "wallet_return" in df.columns else net_pnl_col
    )
    if pnl_col not in df.columns:
        raise ValueError(f"missing pnl column {pnl_col!r}")
    df["_wallet_pnl"] = pd.to_numeric(df[pnl_col], errors="coerce").fillna(0.0)
    df["_net_pnl"] = pd.to_numeric(
        df.get(net_pnl_col, df["_wallet_pnl"]), errors="coerce"
    ).fillna(0.0)
    df["_meta_p"] = np.clip(
        pd.to_numeric(df.get(meta_pred_col, 0.5), errors="coerce").fillna(0.5), 0.0, 1.0
    )
    df["_symbol"] = df[symbol_col].astype(str)
    df["_strategy_id"] = df["strategy_id"].astype(str)
    return df


def _worst_day_loss(rets: np.ndarray, timestamps: Optional[np.ndarray]) -> float:
    return _worst_period_loss(rets, timestamps, "D")


def compute_bad_regime_label(
    *,
    future_horizon_wallet_pnl: float,
    future_horizon_maxDD: float,
    future_horizon_hit_rate_surprise_z: float,
    horizon_days: int,
    dd_thresholds: Optional[Dict[int, float]] = None,
) -> bool:
    dd_thresholds = dd_thresholds or {3: 0.05, 5: 0.08}
    threshold = float(dd_thresholds.get(int(horizon_days), 0.05))
    return bool(
        future_horizon_wallet_pnl < 0.0
        or future_horizon_maxDD > threshold
        or future_horizon_hit_rate_surprise_z < -1.0
    )


def add_consolidated_ebm_regime_features(
    frame: pd.DataFrame,
    timestamps: Optional[Sequence[Any]] = None,
    *,
    symbols: Optional[Sequence[Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create higher-is-worse asset and global EBM aggregates point-in-time."""
    out = frame.copy()
    n = len(out)
    ts = pd.to_datetime(
        np.asarray(timestamps)[:n]
        if timestamps is not None and len(timestamps) >= n
        else out.index,
        utc=True,
        errors="coerce",
    )
    out["_ts"] = ts
    if symbols is not None and len(symbols) >= n:
        out["_symbol"] = np.asarray(symbols).astype(str)[:n]
    elif "symbol" in out.columns:
        out["_symbol"] = out["symbol"].astype(str)
    else:
        out["_symbol"] = "all"
    mapping: Dict[str, Any] = {}

    def available(cols: Sequence[str]) -> List[str]:
        return [c for c in cols if c in out.columns]

    primitive_specs = {
        "ebm_unc_dispersion": available(
            [
                "ebm_unc_logodds_var",
                "ebm_unc_pi_width",
                "ebm_unc_entropy_mean",
                "ebm_unc_entropy_std",
                "ebm_unc_gap50rel",
                "ebm_unc_support_adjusted_uncertainty",
            ]
        ),
        "ebm_conflict": available(["ebm_unc_conflict_norm"]),
        "ebm_brittleness": available(
            ["ebm_unc_concentration", "ebm_unc_interaction_share"]
        ),
    }
    primitives: Dict[str, np.ndarray] = {}
    for name, cols in primitive_specs.items():
        if cols:
            primitives[name] = np.nanmean(
                np.column_stack([pd.to_numeric(out[c], errors="coerce") for c in cols]),
                axis=1,
            )
            mapping[name] = cols
    if "ebm_unc_friction_weight" in out.columns:
        friction_risk = 1.0 - pd.to_numeric(
            out["ebm_unc_friction_weight"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        primitives["ebm_conflict"] = np.nanmean(
            np.column_stack(
                [primitives.get("ebm_conflict", friction_risk), friction_risk]
            ),
            axis=1,
        )
        mapping["ebm_conflict"] = mapping.get("ebm_conflict", []) + [
            "1.0-ebm_unc_friction_weight"
        ]
    support_cols = available(
        ["ebm_unc_support_min", "ebm_unc_support_mean", "ebm_unc_proximity_min"]
    )
    if support_cols:
        support_raw = np.nanmin(
            np.column_stack(
                [pd.to_numeric(out[c], errors="coerce") for c in support_cols]
            ),
            axis=1,
        )
        primitives["ebm_support_risk"] = 1.0 / np.sqrt(
            1.0 + np.maximum(support_raw, 0.0)
        )
        mapping["ebm_support_risk"] = [f"risk({c})" for c in support_cols]

    tmp = pd.DataFrame({"_ts": ts, "_symbol": out["_symbol"]})
    for family, values in primitives.items():
        tmp[family] = values
        for days in (3, 7, 15):
            window = f"{days}D"
            asset_mean = []
            global_mean = []
            for sym, anchor in zip(tmp["_symbol"].to_numpy(), pd.DatetimeIndex(ts)):
                hist = tmp[
                    (tmp["_ts"] < anchor)
                    & (tmp["_ts"] >= anchor - pd.Timedelta(window))
                ]
                global_vals = hist[family].to_numpy(dtype=np.float64)
                asset_vals = hist.loc[
                    hist["_symbol"].astype(str) == str(sym), family
                ].to_numpy(dtype=np.float64)
                global_mean.append(
                    float(np.nanmean(global_vals[np.isfinite(global_vals)]))
                    if np.isfinite(global_vals).any()
                    else np.nan
                )
                asset_mean.append(
                    float(np.nanmean(asset_vals[np.isfinite(asset_vals)]))
                    if np.isfinite(asset_vals).any()
                    else np.nan
                )
            out[f"asset_{family}_mean_{days}d"] = asset_mean
            out[f"global_{family}_mean_{days}d"] = global_mean
        if family == "ebm_unc_dispersion":
            out[f"asset_{family}_trend_3d_to_15d"] = out[
                f"asset_{family}_mean_3d"
            ] / (np.abs(out[f"asset_{family}_mean_15d"]) + EPS)
    return out.drop(columns=["_ts", "_symbol"], errors="ignore"), mapping


def build_rolling_bad_regime_panel(
    trades: pd.DataFrame,
    feature_frame: Optional[pd.DataFrame] = None,
    *,
    strategy_id: Optional[str] = None,
    timestamp_col: Optional[str] = None,
    net_pnl_col: str = "net_pnl",
    wallet_return_col: Optional[str] = None,
    meta_pred_col: str = "meta_pred_calibrated",
    symbol_col: str = "symbol",
    anchor_freq: str = "D",
    horizons_days: Sequence[int] = ROLLING_REGIME_HORIZONS_DAYS,
    horizon_dd_thresholds: Optional[Dict[int, float]] = None,
    min_future_trades: int = 1,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Build strategy_id × symbol × anchor_date × horizon bad-regime panel."""
    if trades.empty:
        return pd.DataFrame(), {"missing_features": [], "feature_mapping": {}}
    df = _prepare_trade_frame(
        trades,
        timestamp_col=timestamp_col,
        net_pnl_col=net_pnl_col,
        wallet_return_col=wallet_return_col,
        meta_pred_col=meta_pred_col,
        symbol_col=symbol_col,
        strategy_id=strategy_id,
    )
    start = df["_ts"].min().floor("D") + pd.Timedelta(days=30)
    end = df["_ts"].max().floor("D") - pd.Timedelta(days=max(horizons_days))
    anchors = (
        pd.date_range(start, end, freq=anchor_freq, tz="UTC")
        if end >= start
        else pd.DatetimeIndex([df["_ts"].min().floor("D")])
    )

    base_source = feature_frame.copy() if feature_frame is not None else df.copy()
    if "_ts" not in base_source.columns:
        if timestamp_col and timestamp_col in base_source.columns:
            base_source["_ts"] = pd.to_datetime(
                base_source[timestamp_col], utc=True, errors="coerce"
            )
        else:
            base_source["_ts"] = pd.to_datetime(
                base_source.index, utc=True, errors="coerce"
            )
    if symbol_col not in base_source.columns:
        base_source[symbol_col] = base_source.get("_symbol", "all")
    src_ts = base_source["_ts"].copy()
    base_source, ebm_mapping = add_consolidated_ebm_regime_features(
        base_source, src_ts, symbols=base_source[symbol_col]
    )
    base_source["_ts"] = pd.to_datetime(src_ts, utc=True, errors="coerce")
    base_source["_symbol"] = base_source[symbol_col].astype(str)

    rows: List[Dict[str, Any]] = []
    keys = sorted(set(zip(df["_strategy_id"], df["_symbol"])))
    for sid, sym in keys:
        sdf = df[(df["_strategy_id"] == sid) & (df["_symbol"] == sym)]
        for anchor in anchors:
            hist = sdf[sdf["_ts"] < anchor]
            base_row: Dict[str, Any] = {
                "strategy_id": sid,
                "symbol": sym,
                "anchor_date": anchor,
            }
            for days in (1, 3, 5, 7, 15, 30):
                h = hist[hist["_ts"] >= anchor - pd.Timedelta(days=days)]
                if days in (1, 3, 5, 7, 15, 30):
                    base_row[f"prior_{days}d_strategy_asset_pnl"] = float(
                        np.nansum(h["_wallet_pnl"])
                    )
                if days in (3, 5, 7, 15, 30):
                    base_row[f"prior_{days}d_strategy_asset_maxDD"] = _drawdown(
                        h["_wallet_pnl"].to_numpy(dtype=np.float64)
                    )
                    base_row[f"prior_{days}d_strategy_asset_trade_count"] = int(len(h))
                    surprise = compute_hit_rate_surprise(h["_net_pnl"], h["_meta_p"])
                    base_row[f"prior_{days}d_expected_hit_rate"] = surprise[
                        "expected_hit_rate"
                    ]
                    base_row[f"prior_{days}d_hit_rate_surprise_z"] = surprise[
                        "hit_rate_surprise_z"
                    ]
            hist_features = base_source[
                (base_source["_ts"] < anchor)
                & (base_source["_symbol"].astype(str) == str(sym))
            ].tail(1)
            if hist_features.empty:
                hist_features = base_source[base_source["_ts"] < anchor].tail(1)
            if not hist_features.empty:
                for col in hist_features.columns:
                    if (
                        col.startswith("future_")
                        or col.startswith("target_")
                        or col in {"_ts", "_symbol"}
                    ):
                        continue
                    val = hist_features.iloc[-1][col]
                    if np.isscalar(val) and not isinstance(val, str):
                        base_row[col] = val
            side_hist = hist.tail(1)
            if not side_hist.empty:
                side_val = float(
                    side_hist.iloc[-1].get(
                        "trade_side", side_hist.iloc[-1].get("side_sign", 0.0)
                    )
                    or 0.0
                )
                funding_z_val = base_row.get("asset_funding_z", np.nan)
                if np.isfinite(funding_z_val):
                    base_row["asset_funding_side_alignment"] = side_val * float(
                        funding_z_val
                    )

            for horizon in horizons_days:
                fut = sdf[
                    (sdf["_ts"] >= anchor)
                    & (sdf["_ts"] < anchor + pd.Timedelta(days=int(horizon)))
                ]
                wallet = fut["_wallet_pnl"].to_numpy(dtype=np.float64)
                net = fut["_net_pnl"].to_numpy(dtype=np.float64)
                future_surprise = compute_hit_rate_surprise(
                    net, fut["_meta_p"] if len(fut) else []
                )
                maxdd = _drawdown(wallet)
                worst_day = _worst_day_loss(
                    wallet, fut["_ts"].to_numpy() if len(fut) else None
                )
                std = float(np.nanstd(wallet if len(wallet) else [0.0]))
                wallet_pnl = float(np.nansum(wallet))
                row = dict(base_row)
                future_trade_count = int(len(fut))
                row.update(
                    {
                        "horizon_days": int(horizon),
                        "future_horizon_trade_count": future_trade_count,
                        "future_horizon_wallet_pnl": wallet_pnl,
                        "future_horizon_net_pnl": float(np.nansum(net)),
                        "future_horizon_maxDD": maxdd,
                        "future_horizon_return_std": std,
                        "future_horizon_worst_day_loss": worst_day,
                        "future_horizon_hit_rate_surprise_z": future_surprise[
                            "hit_rate_surprise_z"
                        ],
                        "future_horizon_badness": -wallet_pnl
                        + 2.0 * maxdd
                        + worst_day
                        + 0.5 * std,
                    }
                )
                row["bad_regime_label"] = (
                    np.nan
                    if future_trade_count < int(min_future_trades)
                    else compute_bad_regime_label(
                        future_horizon_wallet_pnl=wallet_pnl,
                        future_horizon_maxDD=maxdd,
                        future_horizon_hit_rate_surprise_z=row[
                            "future_horizon_hit_rate_surprise_z"
                        ],
                        horizon_days=int(horizon),
                        dd_thresholds=horizon_dd_thresholds,
                    )
                )
                rows.append(row)
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel, {
            "missing_features": list(REGIME_FEATURE_ORDER),
            "feature_mapping": {},
        }
    regime_features, mapping = build_regime_feature_frame(
        panel, panel["anchor_date"], panel["symbol"]
    )
    for c in regime_features.columns:
        panel[c] = regime_features[c].values
    missing_features = [f for f in REGIME_FEATURE_ORDER if f not in panel.columns]
    neutral_missing_cols: Dict[str, float] = {}
    for f in missing_features:
        if f in set(
            GLOBAL_REGIME_FEATURES
            + ASSET_REGIME_FEATURES
            + STRATEGY_ASSET_REGIME_FEATURES
            + EBM_CONSOLIDATED_REGIME_FEATURES
        ):
            neutral_missing_cols[f] = 0.0
            neutral_missing_cols[f"{f}_missing"] = 1.0
    if neutral_missing_cols:
        panel = pd.concat(
            [panel, pd.DataFrame(neutral_missing_cols, index=panel.index)], axis=1
        )
    return panel, {
        "schema_version": "rolling_bad_regime_v2",
        "panel_schema": list(panel.columns),
        "feature_columns": [c for c in REGIME_FEATURE_ORDER if c in panel.columns],
        "global_feature_columns": [
            c for c in GLOBAL_REGIME_FEATURES if c in panel.columns
        ],
        "asset_feature_columns": [
            c
            for c in ASSET_REGIME_FEATURES + STRATEGY_ASSET_REGIME_FEATURES
            if c in panel.columns
        ],
        "feature_mapping": {**mapping, "ebm_consolidated": ebm_mapping},
        "missing_features": missing_features,
        "no_leakage_statement": "all rolling realised features and feature_frame joins use timestamp strictly before anchor_date; labels use [anchor_date, anchor_date+horizon_days).",
        "min_future_trades": int(min_future_trades),
        "no_trade_label_policy": "bad_regime_label is NaN/excluded when future_horizon_trade_count < min_future_trades",
        "horizon_dd_threshold_source": "configured defaults or caller supplied horizon_dd_thresholds",
    }


def build_monthly_regime_panel(
    *args: Any, **kwargs: Any
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Backward-compatible alias for the rolling bad-regime panel builder."""
    return build_rolling_bad_regime_panel(*args, **kwargs)


def regime_acceptance_objective(
    candidate: Dict[str, float],
    baseline: Dict[str, float],
    *,
    weights: Optional[Dict[str, float]] = None,
    clips: Optional[Dict[str, Tuple[float, float]]] = None,
    eps: float = EPS,
) -> Dict[str, Any]:
    weights = weights or REGIME_OBJECTIVE_WEIGHTS
    clips = clips or REGIME_RATIO_CLIPS
    diagnostics: Dict[str, Any] = {"valid": True, "fallback_reason": ""}
    b_net = float(baseline.get("net_pnl", np.nan))
    c_net = float(candidate.get("net_pnl", np.nan))
    if not np.isfinite(b_net) or b_net <= 0:
        diagnostics.update(
            {"valid": False, "fallback_reason": "baseline_net_pnl_non_positive"}
        )
    ratios = {"pnl_ratio": c_net / b_net if b_net > eps else np.nan}
    b_sort = float(baseline.get("sortino", np.nan))
    c_sort = float(candidate.get("sortino", np.nan))
    if not np.isfinite(b_sort) or b_sort <= eps:
        ratios["sortino_ratio"] = 1.0 + np.clip((c_sort - max(b_sort, 0.0)), -1.0, 1.0)
        diagnostics["sortino_denominator_handling"] = "signed_delta_floor"
    else:
        ratios["sortino_ratio"] = c_sort / b_sort
    b_dd = float(baseline.get("maxDD", baseline.get("max_drawdown", np.nan)))
    c_dd = float(candidate.get("maxDD", candidate.get("max_drawdown", np.nan)))
    ratios["dd_ratio"] = (
        1.0 if b_dd <= eps else (clips["dd_ratio"][1] if c_dd <= eps else b_dd / c_dd)
    )
    b_std = float(baseline.get("period_std", baseline.get("return_std", np.nan)))
    c_std = float(candidate.get("period_std", candidate.get("return_std", np.nan)))
    ratios["period_std_ratio"] = (
        1.0
        if b_std <= eps
        else (clips["period_std_ratio"][1] if c_std <= eps else b_std / c_std)
    )
    b_wl = abs(
        float(baseline.get("worst_loss", baseline.get("worst_day_loss", np.nan)))
    )
    c_wl = abs(
        float(candidate.get("worst_loss", candidate.get("worst_day_loss", np.nan)))
    )
    ratios["worst_loss_ratio"] = (
        1.0
        if b_wl <= eps
        else (clips["worst_loss_ratio"][1] if c_wl <= eps else b_wl / c_wl)
    )
    log_objective = 0.0
    components: Dict[str, float] = {}
    for k, w in weights.items():
        lo, hi = clips[k]
        r = float(ratios.get(k, np.nan))
        clipped = float(np.clip(r if np.isfinite(r) and r > 0 else lo, lo, hi))
        components[k] = clipped
        log_objective += float(w) * math.log(max(clipped, eps))
    diagnostics.update(
        {
            "ratios": ratios,
            "clipped_components": components,
            "log_objective": log_objective,
            "objective": float(math.exp(log_objective)),
        }
    )
    return diagnostics


def _global_rank(values: Sequence[float]) -> np.ndarray:
    return _rank_pct(np.asarray(values, dtype=np.float64))


def _mean_zscore(values: Sequence[float]) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(x)
    out = np.zeros(len(x), dtype=np.float64)
    if finite.any():
        mean = float(np.nanmean(x[finite]))
        std = float(np.nanstd(x[finite]))
        out[finite] = (x[finite] - mean) / (std if std > EPS else 1.0)
    return out


def combine_meta_bad_regime_scores(
    meta_p_calibrated: Sequence[float],
    p_bad_regime_global: Sequence[float],
    p_bad_regime_asset: Sequence[float],
    *,
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    params = params or {}
    meta = np.clip(np.asarray(meta_p_calibrated, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    p_global = np.clip(
        np.asarray(p_bad_regime_global, dtype=np.float64), 1e-6, 1.0 - 1e-6
    )
    p_asset = np.clip(
        np.asarray(p_bad_regime_asset, dtype=np.float64), 1e-6, 1.0 - 1e-6
    )

    global_weight = max(float(params.get("global_weight", 0.6)), 0.0)
    asset_weight = max(float(params.get("asset_weight", 0.4)), 0.0)
    weight_sum = global_weight + asset_weight
    if weight_sum <= EPS:
        w_global = 0.0
        w_asset = 0.0
    else:
        w_global = global_weight / weight_sum
        w_asset = asset_weight / weight_sum

    gamma_global = max(float(params.get("gamma_global", 1.0)), EPS)
    gamma_asset = max(float(params.get("gamma_asset", 1.0)), EPS)
    w_interaction = max(float(params.get("interaction_weight", 0.0)), 0.0)

    meta_logit = _logit(meta)
    g_raw = _mean_zscore(_logit(p_global))
    a_raw = _mean_zscore(_logit(p_asset))
    g = np.maximum(g_raw, 0.0)
    a = np.maximum(a_raw, 0.0)
    bad_offset = (
        w_global * np.power(g, gamma_global)
        + w_asset * np.power(a, gamma_asset)
        + w_interaction * g * a
    )
    raw = meta_logit - float(params.get("lambda_regime", 1.0)) * bad_offset
    score = _sigmoid(raw)
    rank = _global_rank(score)
    return {
        "final_score_raw": np.asarray(raw, dtype=np.float64),
        "final_score": np.asarray(score, dtype=np.float64),
        "final_global_rank": rank,
        "bad_regime_offset": np.asarray(bad_offset, dtype=np.float64),
        "global_bad_regime_zscore_raw": np.asarray(g_raw, dtype=np.float64),
        "asset_bad_regime_zscore_raw": np.asarray(a_raw, dtype=np.float64),
        "global_bad_regime_positive_zscore": np.asarray(g, dtype=np.float64),
        "asset_bad_regime_positive_zscore": np.asarray(a, dtype=np.float64),
        "score_delta_from_regime_adjustment": np.asarray(
            score - meta, dtype=np.float64
        ),
    }


def combine_meta_regime_scores(
    meta_p_calibrated: Sequence[float],
    regime_utility_calibrated: Sequence[float],
    regime_logit_offset: Sequence[float],
    *,
    family: str = "logit_offset",
    params: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Compatibility wrapper: treats legacy regime input as bad-regime probability."""
    return combine_meta_bad_regime_scores(
        meta_p_calibrated,
        regime_utility_calibrated,
        regime_logit_offset,
        params={"global_weight": 1.0, "asset_weight": 0.0, **(params or {})},
    )


def select_by_final_rank(
    scores: Sequence[float], *, top_frac: float = 0.30
) -> np.ndarray:
    """Threshold/sizing helper: intentionally uses final global rank."""
    rank = _global_rank(np.asarray(scores, dtype=np.float64))
    return rank >= 1.0 - float(top_frac)


def _fit_lgbm_classifier(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict[str, Any]] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> Any:
    params = {
        k: v
        for k, v in {**ROLLING_REGIME_LGBM_PARAMS, **(params or {})}.items()
        if not str(k).startswith("_")
    }
    if len(np.unique(y_train.astype(int))) < 2:
        return float(np.nanmean(y_train))
    if LGBMClassifier is None:
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        )
        model.fit(x_train, y_train.astype(int), sample_weight=sample_weight)
        return model
    model = LGBMClassifier(**params)
    fit_kwargs: Dict[str, Any] = {}
    if (
        x_val is not None
        and y_val is not None
        and len(x_val) > 0
        and len(np.unique(y_val.astype(int))) > 1
        and early_stopping is not None
    ):
        fit_kwargs["eval_set"] = [(x_val, y_val.astype(int))]
        fit_kwargs["callbacks"] = [early_stopping(stopping_rounds=25, verbose=False)]
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight
    model.fit(x_train, y_train.astype(int), **fit_kwargs)
    return model


def _predict_classifier(model: Any, x: pd.DataFrame) -> np.ndarray:
    if isinstance(model, (float, int, np.floating)):
        return np.full(len(x), float(model), dtype=np.float64)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if np.ndim(proba) == 2 and proba.shape[1] > 1:
            return np.asarray(proba[:, 1], dtype=np.float64)
    return np.clip(np.asarray(model.predict(x), dtype=np.float64), 0.0, 1.0)


def _tune_lgbm_classifier_params(
    frame: pd.DataFrame,
    features: Sequence[str],
    label_col: str,
    time_col: str,
    *,
    optuna_trials: int,
    no_improvement_trials: int,
) -> Dict[str, Any]:
    """Small walk-forward Optuna search for the bad-regime classifier."""
    base = dict(ROLLING_REGIME_LGBM_PARAMS)
    if (
        optuna is None
        or LGBMClassifier is None
        or optuna_trials <= 0
        or len(frame) < 40
    ):
        return base
    y_all = frame[label_col].astype(int).to_numpy()
    if len(np.unique(y_all)) < 2:
        return base
    splits = _walk_forward_splits(frame[time_col].to_numpy(), len(frame), n_splits=4)
    if not splits:
        return base
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42) if TPESampler is not None else None,
        pruner=MedianPruner(n_startup_trials=8, n_min_trials=4)
        if MedianPruner is not None
        else None,
    )
    best_seen = {"trial": -1, "value": np.inf}

    def objective(trial: Any) -> float:
        params = {
            **base,
            "max_depth": trial.suggest_int("max_depth", 2, 3),
            "num_leaves": trial.suggest_int("num_leaves", 4, 8),
            "n_estimators": 50,
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 500),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 4.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 5.0, 40.0),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.08),
            "subsample": trial.suggest_float("subsample", 0.6, 0.9),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        }
        losses: List[float] = []
        for tr, te in splits:
            y_tr = y_all[tr]
            y_te = y_all[te]
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                continue
            model = _fit_lgbm_classifier(
                frame.iloc[tr][list(features)],
                y_tr,
                frame.iloc[te][list(features)],
                y_te,
                params,
                sample_weight=frame.iloc[tr]
                .get("label_weight", pd.Series(1.0, index=frame.index[tr]))
                .to_numpy(dtype=np.float64),
            )
            pred = np.clip(
                _predict_classifier(model, frame.iloc[te][list(features)]),
                1e-6,
                1 - 1e-6,
            )
            losses.append(
                float(-np.mean(y_te * np.log(pred) + (1 - y_te) * np.log(1 - pred)))
            )
        val = float(np.nanmean(losses)) if losses else float("inf")
        if val < best_seen["value"]:
            best_seen.update({"trial": int(trial.number), "value": val})
        elif int(trial.number) - int(best_seen["trial"]) >= int(no_improvement_trials):
            study.stop()
        return val

    study.optimize(
        objective,
        n_trials=int(optuna_trials),
        gc_after_trial=True,
        show_progress_bar=False,
    )
    if study.best_trial is not None:
        base.update(study.best_trial.params)
        base["n_estimators"] = 50
        base["_optuna_best_trial"] = {
            "number": int(study.best_trial.number),
            "value": float(study.best_trial.value),
            "params": _jsonify(study.best_trial.params),
        }
    return base


def _selection_metrics(
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[Sequence[Any]] = None,
    *,
    top_frac: float = 0.30,
) -> Dict[str, float]:
    mask = _top_mask(scores, top_frac)
    r = np.asarray(returns, dtype=np.float64)[mask]
    ts = (
        np.asarray(timestamps)[mask]
        if timestamps is not None and len(timestamps) == len(scores)
        else None
    )
    r = r[np.isfinite(r)]
    if len(r) == 0:
        r = np.array([0.0])
        ts = None
    downside = r[r < 0]
    return {
        "net_pnl": float(np.nansum(r)),
        "sortino": float(
            np.nanmean(r) / ((np.nanstd(downside) if len(downside) else 0.0) + EPS)
        ),
        "maxDD": _drawdown(r),
        "period_std": float(np.nanstd(r)),
        "worst_day_loss": _worst_day_loss(r, ts),
        "worst_loss": _worst_day_loss(r, ts),
    }


def compare_regime_combination_families(
    panel: pd.DataFrame,
    *,
    meta_score_col: str = "meta_pred_calibrated",
    return_col: str = "future_horizon_wallet_pnl",
) -> Dict[str, Any]:
    if meta_score_col in panel.columns:
        meta_series = pd.to_numeric(panel[meta_score_col], errors="coerce").fillna(0.5)
    else:
        meta_series = pd.Series(np.full(len(panel), 0.5), index=panel.index)
    meta = np.clip(meta_series.to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6)
    returns = (
        pd.to_numeric(panel[return_col], errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=np.float64)
    )
    if "anchor_date" in panel.columns:
        timestamps = panel["anchor_date"].to_numpy()
    elif "timestamp" in panel.columns:
        timestamps = panel["timestamp"].to_numpy()
    else:
        timestamps = None
    p_global = (
        pd.to_numeric(
            panel.get(
                "combined_global_bad_regime_oof",
                panel.get("combined_global_bad_regime_score", 0.5),
            ),
            errors="coerce",
        )
        .fillna(0.5)
        .to_numpy(dtype=np.float64)
    )
    p_asset = (
        pd.to_numeric(
            panel.get(
                "combined_asset_bad_regime_oof",
                panel.get("combined_asset_bad_regime_score", 0.5),
            ),
            errors="coerce",
        )
        .fillna(0.5)
        .to_numpy(dtype=np.float64)
    )
    baseline_metrics = _selection_metrics(meta, returns, timestamps, top_frac=0.30)
    rows: List[Dict[str, Any]] = []
    best = {"objective": -np.inf, "params": {}}
    grid = REGIME_ADAPTOR_COMBINATION_GRID
    for wg in grid["global_weight"]:
        for wa in grid["asset_weight"]:
            if float(wg) + float(wa) <= 0.0:
                continue
            for gamma_global in grid.get("gamma_global", [1.0]):
                for gamma_asset in grid.get("gamma_asset", [1.0]):
                    for lam in grid["lambda_regime"]:
                        params = {
                            "global_weight": float(wg),
                            "asset_weight": float(wa),
                            "lambda_regime": float(lam),
                            "gamma_global": float(gamma_global),
                            "gamma_asset": float(gamma_asset),
                        }
                        combined = combine_meta_bad_regime_scores(
                            meta, p_global, p_asset, params=params
                        )
                        metrics = _selection_metrics(
                            combined["final_score"],
                            returns,
                            timestamps,
                            top_frac=0.30,
                        )
                        obj = regime_acceptance_objective(metrics, baseline_metrics)
                        row = {"params": params, "metrics": metrics, **obj}
                        rows.append(row)
                        if (
                            obj["valid"]
                            and metrics["net_pnl"]
                            >= 0.90 * baseline_metrics["net_pnl"]
                            and obj["objective"] > best["objective"]
                        ):
                            best = {
                                "objective": obj["objective"],
                                "params": params,
                                "metrics": metrics,
                                "objective_components": obj,
                            }
    accepted = bool(best["objective"] > 1.05)
    return {
        "baseline_top30_metrics": baseline_metrics,
        "candidate_top30_metrics": best.get("metrics", {}),
        "comparison_table": _jsonify(rows),
        "accepted": accepted,
        "selected_params": best["params"] if accepted else {},
        "selected_objective": None
        if not np.isfinite(best["objective"])
        else float(best["objective"]),
        "objective_components": best.get("objective_components", {}),
    }


def join_regime_oof_to_trade_candidates(
    trade_candidates: pd.DataFrame,
    wide_oof_panel: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    strategy_col: str = "strategy_id",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Point-in-time asof join of wide OOF bad-regime scores onto trade candidates."""
    if trade_candidates.empty or wide_oof_panel.empty:
        return trade_candidates.copy()
    left = trade_candidates.copy()
    right = wide_oof_panel.copy()
    left["_ts"] = pd.to_datetime(left[timestamp_col], utc=True, errors="coerce")
    right["_anchor_ts"] = pd.to_datetime(
        right["anchor_date"], utc=True, errors="coerce"
    )
    out_parts: List[pd.DataFrame] = []
    for (sid, sym), chunk in left.groupby([strategy_col, symbol_col], sort=False):
        r = right[
            (right[strategy_col].astype(str) == str(sid))
            & (right[symbol_col].astype(str) == str(sym))
        ]
        if r.empty:
            out_parts.append(chunk)
            continue
        joined = pd.merge_asof(
            chunk.sort_values("_ts"),
            r.sort_values("_anchor_ts"),
            left_on="_ts",
            right_on="_anchor_ts",
            direction="backward",
            suffixes=("", "_regime"),
        )
        out_parts.append(joined)
    return pd.concat(out_parts, ignore_index=True).drop(
        columns=["_ts", "_anchor_ts"], errors="ignore"
    )


def compare_regime_on_trade_candidates(
    trade_candidates: pd.DataFrame,
    *,
    meta_score_col: str = "meta_pred_calibrated",
    return_col: str = "wallet_return",
    timestamp_col: str = "timestamp",
) -> Dict[str, Any]:
    """Trade-candidate top-30 comparison after OOF bad-regime scores are joined."""
    return compare_regime_combination_families(
        trade_candidates,
        meta_score_col=meta_score_col,
        return_col=return_col,
    ) | {"evaluation_universe": "trade_candidates"}


def fit_rolling_regime_adaptor(
    panel: pd.DataFrame,
    *,
    strategy_id: str = "pooled",
    model_name: str = "bad_regime_classifier",
    global_feature_columns: Optional[Sequence[str]] = None,
    asset_feature_columns: Optional[Sequence[str]] = None,
    trade_candidate_oof: Optional[pd.DataFrame] = None,
    optuna_trials: int = 50,
    no_improvement_trials: int = 25,
    global_bad_rate_threshold: float = GLOBAL_BAD_RATE_THRESHOLD,
    asset_bad_rate_threshold: float = ASSET_BAD_RATE_THRESHOLD,
) -> Dict[str, Any]:
    """Train walk-forward pooled global/asset bad-regime classifiers."""
    if panel.empty:
        return {
            "schema_version": "rolling_bad_regime_v2",
            "enabled": False,
            "reason": "empty_panel",
        }
    work = (
        panel.sort_values(["anchor_date", "strategy_id", "symbol", "horizon_days"])
        .reset_index(drop=True)
        .copy()
    )
    global_features = [
        c
        for c in (global_feature_columns or GLOBAL_REGIME_FEATURES)
        if c in work.columns
    ]
    asset_features = [
        c
        for c in (
            asset_feature_columns
            or (ASSET_REGIME_FEATURES + STRATEGY_ASSET_REGIME_FEATURES)
        )
        if c in work.columns
    ]
    for c in set(global_features + asset_features):
        work[c] = pd.to_numeric(work[c], errors="coerce").fillna(0.0)
    if (
        "bad_regime_label" not in work.columns
        or not global_features
        or not asset_features
    ):
        return {
            "schema_version": "rolling_bad_regime_v2",
            "enabled": False,
            "reason": "insufficient_features_or_label",
            "global_features": global_features,
            "asset_features": asset_features,
        }
    valid_label = work["bad_regime_label"].notna()
    no_trade_rows = int((~valid_label).sum())
    work_trainable = work.loc[valid_label].copy()
    oof_cols: Dict[str, np.ndarray] = {}
    split_defs: List[Dict[str, Any]] = []
    params_by_model: Dict[str, Any] = {}
    oof_metrics: Dict[str, Any] = {}

    def _metrics(y_true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
        mask = np.isfinite(y_true) & np.isfinite(pred)
        if not mask.any():
            return {"auc": np.nan, "logloss": np.nan, "brier": np.nan}
        y = y_true[mask].astype(int)
        p = np.clip(pred[mask], 1e-6, 1.0 - 1e-6)
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan
        logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        brier = float(np.mean((p - y) ** 2))
        return {"auc": auc, "logloss": logloss, "brier": brier}

    for horizon in ROLLING_REGIME_HORIZONS_DAYS:
        hwork = work_trainable[
            work_trainable["horizon_days"].astype(int) == int(horizon)
        ].copy()
        if hwork.empty:
            continue
        # Global labels are one row per anchor/horizon to avoid contradictory labels
        # for identical global features.
        global_df = hwork.groupby("anchor_date", as_index=False).agg(
            {
                **{c: "mean" for c in global_features},
                "bad_regime_label": ["mean", "count"],
            }
        )
        global_df.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c
            for c in global_df.columns
        ]
        global_df = global_df.rename(
            columns={
                **{f"{c}_mean": c for c in global_features},
                "bad_regime_label_mean": "bad_rate",
                "bad_regime_label_count": "global_label_weight",
            }
        )
        global_df["label_weight"] = global_df["global_label_weight"].astype(
            np.float64
        )
        global_df["global_bad_regime_label"] = (
            global_df["bad_rate"] >= float(global_bad_rate_threshold)
        ).astype(int)
        # Asset labels are one row per symbol/anchor/horizon, pooled across strategies.
        asset_df = hwork.groupby(["symbol", "anchor_date"], as_index=False).agg(
            {
                **{c: "mean" for c in asset_features},
                "bad_regime_label": ["mean", "count"],
            }
        )
        asset_df.columns = [
            "_".join(c).strip("_") if isinstance(c, tuple) else c
            for c in asset_df.columns
        ]
        asset_df = asset_df.rename(
            columns={
                **{f"{c}_mean": c for c in asset_features},
                "bad_regime_label_mean": "bad_rate",
                "bad_regime_label_count": "asset_label_weight",
            }
        )
        asset_df["label_weight"] = asset_df["asset_label_weight"].astype(np.float64)
        asset_df["asset_bad_regime_label"] = (
            asset_df["bad_rate"] >= float(asset_bad_rate_threshold)
        ).astype(int)

        for scope, model_df, features, label_col, join_cols in (
            (
                "global",
                global_df,
                global_features,
                "global_bad_regime_label",
                ["anchor_date"],
            ),
            (
                "asset",
                asset_df,
                asset_features,
                "asset_bad_regime_label",
                ["symbol", "anchor_date"],
            ),
        ):
            col = f"p_bad_regime_{scope}_{horizon}d_oof"
            pred_frame = model_df[join_cols].copy()
            pred_frame[col] = np.nan
            best_params = _tune_lgbm_classifier_params(
                model_df,
                features,
                label_col,
                "anchor_date",
                optuna_trials=optuna_trials,
                no_improvement_trials=no_improvement_trials,
            )
            splits = _walk_forward_splits(
                model_df["anchor_date"].to_numpy(), len(model_df), n_splits=5
            )
            y = model_df[label_col].astype(int).to_numpy()
            for tr, te in splits:
                if len(tr) == 0 or len(te) == 0:
                    continue
                model = _fit_lgbm_classifier(
                    model_df.iloc[tr][list(features)],
                    y[tr],
                    model_df.iloc[te][list(features)],
                    y[te],
                    best_params,
                    sample_weight=model_df.iloc[tr]["label_weight"].to_numpy(
                        dtype=np.float64
                    )
                    if "label_weight" in model_df.columns
                    else None,
                )
                pred_frame.loc[pred_frame.index[te], col] = _predict_classifier(
                    model, model_df.iloc[te][list(features)]
                )
                split_defs.append(
                    {
                        "horizon_days": int(horizon),
                        "scope": scope,
                        "train_start": str(model_df.iloc[tr]["anchor_date"].min()),
                        "train_end": str(model_df.iloc[tr]["anchor_date"].max()),
                        "valid_start": str(model_df.iloc[te]["anchor_date"].min()),
                        "valid_end": str(model_df.iloc[te]["anchor_date"].max()),
                        "train_rows": int(len(tr)),
                        "valid_rows": int(len(te)),
                    }
                )
            oof_metrics[f"{scope}_{horizon}d"] = _metrics(
                y, pred_frame[col].to_numpy(dtype=np.float64)
            )
            params_by_model[f"{scope}_{horizon}d"] = best_params
            if scope == "global":
                work = work.merge(pred_frame, on=["anchor_date"], how="left")
            else:
                work = work.merge(pred_frame, on=["symbol", "anchor_date"], how="left")
            oof_cols[col] = work[col].to_numpy(dtype=np.float64)

    key_cols = ["strategy_id", "symbol", "anchor_date"]
    wide = work.groupby(key_cols, as_index=False).agg(
        {
            **{
                c: "first"
                for c in work.columns
                if c.startswith("p_bad_regime_") and c.endswith("_oof")
            },
            "future_horizon_wallet_pnl": "mean",
            "future_horizon_trade_count": "sum"
            if "future_horizon_trade_count" in work.columns
            else "size",
            **(
                {"meta_pred_calibrated": "first"}
                if "meta_pred_calibrated" in work.columns
                else {}
            ),
            **(
                {"meta_p_calibrated": "first"}
                if "meta_p_calibrated" in work.columns
                else {}
            ),
        }
    )
    w3, w5 = (
        ROLLING_REGIME_DEFAULT_BLEND_WEIGHTS[3],
        ROLLING_REGIME_DEFAULT_BLEND_WEIGHTS[5],
    )
    for col in (
        "p_bad_regime_global_3d_oof",
        "p_bad_regime_global_5d_oof",
        "p_bad_regime_asset_3d_oof",
        "p_bad_regime_asset_5d_oof",
    ):
        if col not in wide.columns:
            wide[col] = 0.5
    wide["combined_global_bad_regime_oof"] = (
        w3 * wide["p_bad_regime_global_3d_oof"]
        + w5 * wide["p_bad_regime_global_5d_oof"]
    )
    wide["combined_asset_bad_regime_oof"] = (
        w3 * wide["p_bad_regime_asset_3d_oof"] + w5 * wide["p_bad_regime_asset_5d_oof"]
    )
    anchor_diagnostic_comparison = compare_regime_combination_families(
        wide,
        meta_score_col="meta_p_calibrated"
        if "meta_p_calibrated" in wide.columns
        else "meta_pred_calibrated",
    )
    trade_candidate_eval_available = (
        trade_candidate_oof is not None and not trade_candidate_oof.empty
    )
    if trade_candidate_eval_available:
        joined_candidates = join_regime_oof_to_trade_candidates(
            trade_candidate_oof, wide
        )
        comparison = compare_regime_on_trade_candidates(
            joined_candidates,
            meta_score_col="meta_p_calibrated"
            if "meta_p_calibrated" in joined_candidates.columns
            else "meta_pred_calibrated",
            return_col="wallet_return"
            if "wallet_return" in joined_candidates.columns
            else "future_horizon_wallet_pnl",
        )
        accepted = bool(comparison.get("accepted", False))
        selected_params = comparison.get("selected_params", {}) if accepted else {}
        decision_reason = (
            "accepted" if accepted else "failed_trade_candidate_oof_acceptance"
        )
    else:
        comparison = anchor_diagnostic_comparison
        accepted = False
        selected_params = {}
        decision_reason = "missing_trade_candidate_oof_evaluation"
    meta_col = (
        "meta_pred_calibrated"
        if "meta_pred_calibrated" in wide.columns
        else "meta_p_calibrated"
    )
    meta_values = (
        pd.to_numeric(wide[meta_col], errors="coerce")
        .fillna(0.5)
        .to_numpy(dtype=np.float64)
        if meta_col in wide.columns
        else np.full(len(wide), 0.5, dtype=np.float64)
    )
    combined = combine_meta_bad_regime_scores(
        np.clip(meta_values, 1e-6, 1 - 1e-6),
        wide["combined_global_bad_regime_oof"],
        wide["combined_asset_bad_regime_oof"],
        params=selected_params
        or {
            "global_weight": 0.6,
            "asset_weight": 0.4,
            "lambda_regime": 1.0,
            "gamma_global": 1.0,
            "gamma_asset": 1.0,
        },
    )
    wide["combined_bad_regime_offset_oof"] = combined["bad_regime_offset"]
    optuna_best_trial_diagnostics = {
        k: v.get("_optuna_best_trial")
        for k, v in params_by_model.items()
        if isinstance(v, dict) and v.get("_optuna_best_trial") is not None
    }
    return {
        "schema_version": "rolling_bad_regime_v2",
        "strategy_id": str(strategy_id),
        "model_name": str(model_name),
        "model_type": "bad_regime_classifier",
        "enabled": accepted,
        "enable_regime_adaptor": accepted,
        "reason": decision_reason,
        "config_snapshot": {
            "horizons_days": list(ROLLING_REGIME_HORIZONS_DAYS),
            "global_bad_rate_threshold": float(global_bad_rate_threshold),
            "asset_bad_rate_threshold": float(asset_bad_rate_threshold),
            "combination_grid": REGIME_ADAPTOR_COMBINATION_GRID,
            "objective_weights": REGIME_OBJECTIVE_WEIGHTS,
            "ratio_clips": REGIME_RATIO_CLIPS,
        },
        "feature_key_lists": {
            "global": list(global_features),
            "asset": list(asset_features),
        },
        "missing_feature_list": [
            f for f in REGIME_FEATURE_ORDER if f not in work.columns
        ],
        "panel_schema": list(work.columns),
        "wide_oof_panel_schema": list(wide.columns),
        "no_leakage_statement": "walk-forward split by anchor_date only; OOF predictions use train anchors strictly before validation anchors.",
        "live_scoring_mode": "mode_b_precomputed_p_bad_regime_columns",
        "rank_scope": "local_batch",
        "rank_requirement": "portfolio_global_or_per_side_rank_required_downstream_before_thresholding_or_sizing",
        "trade_candidate_eval_available": bool(trade_candidate_eval_available),
        "global_bad_rate_threshold": float(global_bad_rate_threshold),
        "asset_bad_rate_threshold": float(asset_bad_rate_threshold),
        "optuna_best_trial_diagnostics": optuna_best_trial_diagnostics,
        "no_trade_rows_excluded": no_trade_rows,
        "global_classifier_label_definition": f"mean(strategy-symbol bad_regime_label at anchor/horizon) >= {float(global_bad_rate_threshold):.3f}",
        "asset_classifier_label_definition": f"mean(strategy bad_regime_label for symbol at anchor/horizon) >= {float(asset_bad_rate_threshold):.3f}",
        "walk_forward_splits": split_defs,
        "global_classifier_params": {
            k: v for k, v in params_by_model.items() if k.startswith("global")
        },
        "asset_classifier_params": {
            k: v for k, v in params_by_model.items() if k.startswith("asset")
        },
        "oof_p_bad_regime_predictions": wide[
            ["strategy_id", "symbol", "anchor_date"]
            + [
                c
                for c in wide.columns
                if c.startswith("p_bad_regime_") and c.endswith("_oof")
            ]
            + [
                "combined_global_bad_regime_oof",
                "combined_asset_bad_regime_oof",
                "combined_bad_regime_offset_oof",
            ]
        ].to_dict(orient="records"),
        "oof_classifier_metrics": oof_metrics,
        "evaluation_universe": comparison.get(
            "evaluation_universe",
            "trade_candidates" if trade_candidate_eval_available else "unavailable",
        ),
        "selected_3d_5d_blend": {"3d": w3, "5d": w5},
        "selected_combination_params": selected_params,
        "baseline_top30_metrics": comparison.get("baseline_top30_metrics", {}),
        "candidate_top30_metrics": comparison.get("candidate_top30_metrics", {}),
        "multiplicative_objective_components": comparison.get(
            "objective_components", {}
        ),
        "final_enable_disable_decision": {
            "enabled": accepted,
            "reason": decision_reason,
        },
        "inference_contract": {
            "required_live_columns_if_no_serialized_model": [
                "p_bad_regime_global_3d",
                "p_bad_regime_global_5d",
                "p_bad_regime_asset_3d",
                "p_bad_regime_asset_5d",
            ],
            "combination": (
                "sigmoid(logit(meta_p_calibrated) - lambda_regime * bad_offset), "
                "where bad_offset uses positive z-scored bad-regime logits with "
                "global/asset weights and gammas"
            ),
        },
        "combination_family_comparison": comparison,
        "anchor_panel_diagnostic_comparison": anchor_diagnostic_comparison,
    }


def fit_regime_adaptor(
    feature_frame: pd.DataFrame,
    pred_calibrated: Sequence[float],
    returns: Sequence[float],
    timestamps: Optional[Sequence[Any]],
    symbols: Optional[Sequence[Any]],
    *,
    strategy_id: str,
    model_name: str,
    cost_pct: float = 0.003,
    used_feature_columns: Optional[Sequence[str]] = None,
    policy_candidate_mask: Optional[Sequence[bool]] = None,
    gross_returns: Optional[Sequence[float]] = None,
    policy_returns: Optional[Sequence[float]] = None,
    mfe: Optional[Sequence[float]] = None,
    mae: Optional[Sequence[float]] = None,
    t_mfe: Optional[Sequence[float]] = None,
    t_mae: Optional[Sequence[float]] = None,
) -> RegimeAdaptorFit:
    n = min(len(feature_frame), len(pred_calibrated), len(returns))
    frame_n = feature_frame.iloc[:n].copy()
    scores = _as_float_array(pred_calibrated, n)
    rets = _as_float_array(returns, n)
    gross_arr = _as_float_array(gross_returns, n) if gross_returns is not None else None
    policy_arr = _as_float_array(policy_returns, n) if policy_returns is not None else None
    mfe_arr = _as_float_array(mfe, n) if mfe is not None else None
    mae_arr = _as_float_array(mae, n) if mae is not None else None
    t_mfe_arr = _as_float_array(t_mfe, n) if t_mfe is not None else None
    t_mae_arr = _as_float_array(t_mae, n) if t_mae is not None else None
    ts = (
        np.asarray(timestamps)[:n]
        if timestamps is not None and len(timestamps) >= n
        else None
    )
    sy = (
        np.asarray(symbols).astype(str)[:n]
        if symbols is not None and len(symbols) >= n
        else np.repeat("all", n)
    )
    if policy_candidate_mask is not None and len(policy_candidate_mask) >= n:
        keep = np.asarray(policy_candidate_mask, dtype=bool)[:n]
        frame_n = frame_n.iloc[keep].reset_index(drop=True)
        scores = scores[keep]
        rets = rets[keep]
        gross_arr = gross_arr[keep] if gross_arr is not None else None
        policy_arr = policy_arr[keep] if policy_arr is not None else None
        mfe_arr = mfe_arr[keep] if mfe_arr is not None else None
        mae_arr = mae_arr[keep] if mae_arr is not None else None
        t_mfe_arr = t_mfe_arr[keep] if t_mfe_arr is not None else None
        t_mae_arr = t_mae_arr[keep] if t_mae_arr is not None else None
        ts = ts[keep] if ts is not None else None
        sy = sy[keep]
        n = len(scores)
    outcome = policy_arr if policy_arr is not None else rets
    outcome_cost_pct = 0.0 if policy_arr is not None else float(cost_pct)
    outcome_source = (
        "policy_realized_utility"
        if policy_arr is not None
        else "raw_realized_return"
    )
    distribution_spec = _fit_distribution_feature_spec(frame_n, used_feature_columns)
    regime_df, mapping = build_regime_feature_frame(frame_n, ts, sy)
    regime_df = _append_distribution_features(
        regime_df, frame_n, distribution_spec, ts, sy
    )
    regime_df, reliability_spec = _append_prediction_reliability_features(
        regime_df, frame_n, scores, outcome, ts, sy
    )
    regime_df, symbol_feature_spec = _append_symbol_features(
        regime_df, outcome, ts, sy
    )
    features = [f for f in REGIME_FEATURE_ORDER if f in regime_df.columns]
    if not features or n < 50:
        artifact = _empty_artifact(strategy_id, model_name, features, mapping)
        artifact["distribution_feature_spec"] = _jsonify(distribution_spec)
        artifact["reliability_feature_spec"] = _jsonify(reliability_spec)
        artifact["symbol_feature_spec"] = _jsonify(symbol_feature_spec)
        applied = apply_regime_adaptor(frame_n, scores, artifact, ts, sy)
        empty = pd.DataFrame()
        return RegimeAdaptorFit(
            artifact,
            empty,
            empty,
            empty,
            score_metrics(scores, outcome, ts, cost_pct=outcome_cost_pct),
            applied["regime_weight"],
            applied["eligible"],
            applied["deployment_score"],
            applied["deployment_score_rank"],
        )

    rank_weight = _rank_weight(scores)
    percentile_refs = {
        feat: _fit_percentile(regime_df[feat].values) for feat in features
    }
    spline_hpo, full_stats, effects = _select_spline_hyperparams(
        regime_df,
        features,
        percentile_refs,
        scores,
        outcome,
        ts,
        rank_weight,
    )
    adaptive_rows: List[Dict[str, Any]] = []
    for feat in features:
        stats = full_stats.get(feat, {})
        for row in stats.get("bins", []):
            adaptive_rows.append(
                {
                    "strategy_id": strategy_id,
                    "model": model_name,
                    "feature": feat,
                    **row,
                }
            )

    fixed = fixed_bucket_diagnostics(
        regime_df, scores, outcome, ts, sy, strategy_id, model_name, percentile_refs
    )
    asset_diag = asset_diagnostics(scores, outcome, ts, sy, strategy_id, model_name)
    bucket_gates = _bucket_gates(fixed)
    asset_gates = _asset_gates(asset_diag)

    top = _top_mask(scores, 0.10)
    target_center = (
        float(np.nanmean(outcome[top])) if top.any() else float(np.nanmean(outcome))
    )
    target = (outcome - target_center).astype(np.float64)
    model, scaler, train_mean, params = _fit_elastic_net(
        effects, target, rank_weight, scores, outcome, ts
    )
    trust_model, trust_proba_oof, trust_score_oof, trust_target = _fit_trust_model(
        regime_df,
        features,
        rets,
        frame_n,
        ts,
        cost_pct,
        gross_returns=gross_arr,
        policy_returns=policy_arr,
        mfe=mfe_arr,
        mae=mae_arr,
        t_mfe=t_mfe_arr,
        t_mae=t_mae_arr,
    )
    artifact = {
        "schema_version": "v1",
        "strategy_id": str(strategy_id),
        "model_name": str(model_name),
        "features": features,
        "feature_mapping": mapping,
        "training_universe": (
            "policy_candidate_mask_after_simple_policy_optimiser"
            if policy_candidate_mask is not None
            else "provided_oos_rows"
        ),
        "distribution_feature_spec": _jsonify(distribution_spec),
        "reliability_feature_spec": _jsonify(reliability_spec),
        "symbol_feature_spec": _jsonify(symbol_feature_spec),
        "percentile_refs": {
            k: v.astype(float).tolist() for k, v in percentile_refs.items()
        },
        "feature_splines": _jsonify(full_stats),
        "spline_hpo": _jsonify(spline_hpo),
        "elastic_net": {
            "coef": np.asarray(model.coef_, dtype=float).tolist(),
            "intercept": float(model.intercept_),
            "train_prediction_mean": train_mean,
            "params": params,
            "scaler": {
                "center": np.asarray(
                    getattr(scaler, "center_", np.zeros(effects.shape[1])), dtype=float
                ).tolist(),
                "scale": np.asarray(
                    getattr(scaler, "scale_", np.ones(effects.shape[1])), dtype=float
                ).tolist(),
            },
        },
        "clips": {
            "log_effect_clip": [-0.10, 0.10],
            "group_clip": [-0.12, 0.12],
            "total_log_weight_clip": [-0.35, 0.22],
            "regime_weight_clip": [0.70, 1.25],
        },
        "bucket_gates": bucket_gates,
        "asset_gates": [],
        "retired_asset_gates": asset_gates,
        "trust_model": _jsonify(trust_model),
        "trust_gate_threshold": -0.35,
        "outcome_source": outcome_source,
        "outcome_cost_pct": outcome_cost_pct,
        "rank_normalization": {
            "method": "pandas_rank_pct_average",
            "score": "deployment_score",
        },
        "enable_regime_adaptor": False,
        "enable_regime_adaptor_inference": False,
        "inference_integration_mode": "disabled",
    }
    candidate_applied = apply_regime_adaptor(
        frame_n,
        scores,
        artifact | {"enable_regime_adaptor": True},
        ts,
        sy,
    )
    raw_m = score_metrics(
        scores,
        outcome,
        ts,
        top_fracs=(0.01, 0.05, 0.10, 0.20),
        cost_pct=outcome_cost_pct,
    )
    candidate_m = score_metrics(
        candidate_applied["deployment_score_rank"],
        outcome,
        ts,
        top_fracs=(0.01, 0.05, 0.10, 0.20),
        cost_pct=outcome_cost_pct,
    )
    summary = _compare_metrics(raw_m, candidate_m)
    enabled, enable_decision = _regime_enable_decision(summary)
    artifact["enable_regime_adaptor"] = bool(enabled)
    artifact["selection_score"] = float(enable_decision.get("selection_score", 0.0))
    artifact["enable_gate"] = _jsonify(enable_decision)
    final_applied = apply_regime_adaptor(
        frame_n, scores, artifact, ts, sy
    )
    deployed_m = score_metrics(
        final_applied["deployment_score_rank"],
        outcome,
        ts,
        top_fracs=(0.01, 0.05, 0.10, 0.20),
        cost_pct=outcome_cost_pct,
    )
    deployed_summary = _compare_metrics(raw_m, deployed_m)
    model_quality = _model_quality_metrics_table(
        strategy_id=strategy_id,
        model_name=model_name,
        training_universe=artifact["training_universe"],
        raw_scores=scores,
        adjusted_scores=final_applied["deployment_score_rank"],
        returns=outcome,
        timestamps=ts,
        symbols=sy,
        trust_proba=trust_proba_oof,
        trust_score=trust_score_oof,
        trust_target=trust_target,
    )
    top30_comparison = _regime_top30_comparison(
        scores,
        final_applied["deployment_score_rank"],
        outcome,
        ts,
        sy,
        cost_pct=outcome_cost_pct,
    )
    metrics = pd.concat(
        [
            raw_m.assign(stage="raw"),
            candidate_m.assign(stage="regime_adjusted_candidate"),
            deployed_m.assign(stage="regime_adjusted_deployed"),
            top30_comparison.assign(stage="top30_before_after"),
            model_quality.assign(stage="model_quality"),
        ],
        ignore_index=True,
        sort=False,
    )
    metrics["strategy_id"] = strategy_id
    metrics["model"] = model_name
    metrics["regime_adaptor_enabled"] = bool(artifact["enable_regime_adaptor"])
    artifact["candidate_before_after"] = _jsonify(summary.to_dict(orient="records"))
    artifact["deployed_before_after"] = _jsonify(
        deployed_summary.to_dict(orient="records")
    )
    artifact["before_after_top10"] = artifact["candidate_before_after"]
    artifact["metrics"] = _jsonify(metrics.to_dict(orient="records"))
    artifact["model_quality_metrics"] = _jsonify(
        model_quality.to_dict(orient="records")
    )
    artifact["top30_regime_comparison"] = _jsonify(
        top30_comparison.to_dict(orient="records")
    )
    return RegimeAdaptorFit(
        artifact=artifact,
        fixed_diagnostics=fixed,
        adaptive_diagnostics=pd.DataFrame(adaptive_rows),
        asset_diagnostics=asset_diag,
        metrics=metrics,
        regime_weight_oof=final_applied["regime_weight"],
        eligible_oof=final_applied["eligible"],
        deployment_score_oof=final_applied["deployment_score"],
        deployment_score_rank_oof=final_applied["deployment_score_rank"],
        trust_score_oof=trust_score_oof,
        trust_proba_oof=trust_proba_oof,
    )


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    m = np.isfinite(aa) & np.isfinite(bb)
    if int(np.sum(m)) < 3 or float(np.nanstd(aa[m])) <= EPS or float(np.nanstd(bb[m])) <= EPS:
        return float("nan")
    return float(np.corrcoef(aa[m], bb[m])[0, 1])


def _group_corr_stats(
    scores: np.ndarray, returns: np.ndarray, groups: Sequence[Any]
) -> Tuple[float, float]:
    rows = []
    df = pd.DataFrame({"score": scores, "ret": returns, "group": groups})
    for _g, grp in df.groupby("group", sort=False):
        if len(grp) < 3:
            continue
        rows.append(_safe_corr(grp["score"].to_numpy(), grp["ret"].to_numpy()))
    vals = np.asarray([x for x in rows if np.isfinite(x)], dtype=np.float64)
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals))


def _ndcg_at_frac(scores: np.ndarray, returns: np.ndarray, frac: float) -> float:
    s = np.asarray(scores, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    finite = np.isfinite(s) & np.isfinite(r)
    if not finite.any():
        return float("nan")
    k = max(1, int(math.ceil(np.sum(finite) * frac)))
    gain = np.maximum(r[finite], 0.0)
    order = np.argsort(s[finite])[::-1][:k]
    ideal = np.argsort(gain)[::-1][:k]
    denom = np.log2(np.arange(2, k + 2, dtype=np.float64))
    dcg = float(np.sum(gain[order] / denom))
    idcg = float(np.sum(gain[ideal] / denom))
    return dcg / idcg if idcg > EPS else float("nan")


def _ece_binary(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    yy = np.asarray(y, dtype=np.float64)
    pp = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    finite = np.isfinite(yy) & np.isfinite(pp)
    if not finite.any():
        return float("nan")
    yy, pp = yy[finite], pp[finite]
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (pp >= edges[i]) & (pp < edges[i + 1] if i < bins - 1 else pp <= edges[i + 1])
        if not m.any():
            continue
        ece += float(np.mean(m)) * abs(float(np.mean(yy[m])) - float(np.mean(pp[m])))
    return float(ece)


def _decile_utility(scores: np.ndarray, returns: np.ndarray) -> Tuple[List[float], float]:
    df = pd.DataFrame({"score": scores, "ret": returns}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(df) < 10:
        return [], float("nan")
    df["decile"] = pd.qcut(df["score"].rank(method="first"), 10, labels=False, duplicates="drop")
    vals = df.groupby("decile")["ret"].mean().to_list()
    if len(vals) < 2:
        return [float(v) for v in vals], float("nan")
    mono = float(np.corrcoef(np.arange(len(vals)), np.asarray(vals, dtype=float))[0, 1])
    return [float(v) for v in vals], mono


def _quality_row(
    *,
    model_name: str,
    target_name: str,
    training_universe: str,
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    symbols: Optional[np.ndarray],
    y_binary: Optional[np.ndarray] = None,
    proba_binary: Optional[np.ndarray] = None,
    y_multiclass: Optional[np.ndarray] = None,
    proba_multiclass: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    s = np.asarray(scores, dtype=np.float64)
    r = np.asarray(returns, dtype=np.float64)
    finite = np.isfinite(s) & np.isfinite(r)
    n = len(s)
    deciles, mono = _decile_utility(s, r)
    symbol_mean = symbol_std = float("nan")
    if symbols is not None and len(symbols) >= n:
        symbol_mean, symbol_std = _group_corr_stats(s, r, np.asarray(symbols)[:n])
    week_mean = week_std = month_mean = month_std = float("nan")
    if timestamps is not None and len(timestamps) >= n:
        ts = pd.to_datetime(np.asarray(timestamps)[:n], utc=True, errors="coerce")
        ts_naive = pd.Series(ts).dt.tz_convert(None)
        week_mean, week_std = _group_corr_stats(s, r, ts_naive.dt.to_period("W").astype(str))
        month_mean, month_std = _group_corr_stats(s, r, ts_naive.dt.to_period("M").astype(str))
    auc_good = auc_bad = auc_ovr = brier = ece = ll = float("nan")
    if y_binary is not None and proba_binary is not None:
        yy = np.asarray(y_binary, dtype=int)
        pp = np.clip(np.asarray(proba_binary, dtype=np.float64), 1e-6, 1.0 - 1e-6)
        m = np.isfinite(pp) & np.isfinite(yy)
        if len(np.unique(yy[m])) == 2:
            auc_good = float(roc_auc_score(yy[m], pp[m]))
            brier = float(np.mean(np.square(pp[m] - yy[m])))
            ece = _ece_binary(yy[m], pp[m])
            ll = float(log_loss(yy[m], pp[m], labels=[0, 1]))
    if y_multiclass is not None and proba_multiclass is not None:
        yy = np.asarray(y_multiclass, dtype=int)
        labels = np.array([-1, 0, 1], dtype=int)
        pp = np.clip(np.asarray(proba_multiclass, dtype=np.float64), 1e-6, 1.0)
        pp = pp / np.sum(pp, axis=1, keepdims=True)
        pp_ordered = pp[:, [2, 1, 0]]
        m = np.isfinite(pp).all(axis=1)
        unique_classes = np.unique(yy[m])
        if len(unique_classes) >= 2:
            try:
                if set(labels.tolist()).issubset(set(unique_classes.tolist())):
                    auc_ovr = float(
                        roc_auc_score(
                            yy[m], pp_ordered[m], labels=labels, multi_class="ovr"
                        )
                    )
            except Exception:
                auc_ovr = float("nan")
            if np.any(yy[m] == 1) and np.any(yy[m] != 1):
                auc_good = float(roc_auc_score((yy[m] == 1).astype(int), pp[m, 0]))
            if np.any(yy[m] == -1) and np.any(yy[m] != -1):
                auc_bad = float(roc_auc_score((yy[m] == -1).astype(int), pp[m, 2]))
            ll = float(log_loss(yy[m], pp_ordered[m], labels=labels))
            brier = float(np.mean(np.sum((pp[m] - np.column_stack([yy[m] == 1, yy[m] == 0, yy[m] == -1])) ** 2, axis=1)))
    return {
        "model_name": model_name,
        "target_name": target_name,
        "training_universe": training_universe,
        "n_train": int(np.sum(finite)),
        "n_valid": int(np.sum(finite)),
        "coverage": float(np.mean(finite)) if n else 0.0,
        "IC_all": _safe_corr(s, r),
        "IC_symbol_mean": symbol_mean,
        "IC_symbol_std": symbol_std,
        "IC_week_mean": week_mean,
        "IC_week_std": week_std,
        "IC_month_mean": month_mean,
        "IC_month_std": month_std,
        "AUC_ovr": auc_ovr,
        "AUC_good_vs_rest": auc_good,
        "AUC_bad_vs_rest": auc_bad,
        "NDCG@1%": _ndcg_at_frac(s, r, 0.01),
        "NDCG@5%": _ndcg_at_frac(s, r, 0.05),
        "NDCG@10%": _ndcg_at_frac(s, r, 0.10),
        "Brier": brier,
        "ECE": ece,
        "logloss": ll,
        "mean_realized_utility_by_score_decile": deciles,
        "top_decile_utility": float(deciles[-1]) if deciles else float("nan"),
        "bottom_decile_utility": float(deciles[0]) if deciles else float("nan"),
        "monotonicity_score": mono,
    }


def _model_quality_metrics_table(
    *,
    strategy_id: str,
    model_name: str,
    training_universe: str,
    raw_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    symbols: np.ndarray,
    trust_proba: np.ndarray,
    trust_score: np.ndarray,
    trust_target: np.ndarray,
) -> pd.DataFrame:
    y_win = (np.asarray(returns, dtype=np.float64) > 0.0).astype(int)
    rows = [
        _quality_row(
            model_name=f"{model_name}:correctedness_classifier",
            target_name="realized_utility_positive",
            training_universe=training_universe,
            scores=raw_scores,
            returns=returns,
            timestamps=timestamps,
            symbols=symbols,
            y_binary=y_win,
            proba_binary=np.clip(raw_scores, 1e-6, 1.0 - 1e-6),
        ),
        _quality_row(
            model_name=f"{model_name}:utility_global_regressor",
            target_name="realized_net_utility",
            training_universe=training_universe,
            scores=adjusted_scores,
            returns=returns,
            timestamps=timestamps,
            symbols=symbols,
        ),
        _quality_row(
            model_name=f"{model_name}:trustworthiness_classifier",
            target_name="trust_target_-1_0_1",
            training_universe=training_universe,
            scores=trust_score,
            returns=returns,
            timestamps=timestamps,
            symbols=symbols,
            y_multiclass=trust_target,
            proba_multiclass=trust_proba,
        ),
    ]
    df = pd.DataFrame(rows)
    df.insert(0, "strategy_id", strategy_id)
    return df


def _worst_rolling_days(net: np.ndarray, timestamps: Optional[np.ndarray], days: int) -> float:
    if timestamps is None or len(timestamps) != len(net):
        if len(net) == 0:
            return 0.0
        window = max(1, min(days, len(net)))
        vals = pd.Series(net).rolling(window, min_periods=1).sum()
        return float(vals.min())
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    df = pd.DataFrame({"net": net, "ts": ts}).dropna()
    if df.empty:
        return 0.0
    daily = df.set_index("ts")["net"].resample("D").sum().fillna(0.0)
    return float(daily.rolling(days, min_periods=1).sum().min())


def _period_sortino(net: np.ndarray, timestamps: Optional[np.ndarray], freq: str) -> float:
    if timestamps is None or len(timestamps) != len(net):
        vals = np.asarray(net, dtype=np.float64)
    else:
        ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
        df = pd.DataFrame({"net": net, "ts": ts}).dropna()
        if df.empty:
            return 0.0
        vals = df.set_index("ts")["net"].resample(freq).sum().to_numpy(dtype=np.float64)
    down = vals[vals < 0.0]
    if len(down) == 0:
        return 100.0 if len(vals) and float(np.mean(vals)) > 0 else 0.0
    return float(np.mean(vals) / (np.sqrt(np.mean(np.square(down))) + EPS))


def _top30_row(
    label: str,
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    symbols: np.ndarray,
    cost_pct: float,
) -> Dict[str, Any]:
    mask = _top_mask(scores, 0.30)
    net = np.asarray(returns, dtype=np.float64)[mask] - float(cost_pct)
    ts_sel = np.asarray(timestamps)[mask] if timestamps is not None and len(timestamps) == len(scores) else None
    sym_sel = symbols[mask] if len(symbols) == len(scores) else np.repeat("all", len(net))
    asset_means = pd.DataFrame({"sym": sym_sel, "net": net}).groupby("sym")["net"].mean()
    return {
        "comparison": label,
        "top_frac": 0.30,
        "avg_pnl_per_trade": float(np.mean(net)) if len(net) else 0.0,
        "maxdrawdown": _drawdown(net),
        "std_dev_weekly": _period_std(net, ts_sel, "W"),
        "std_dev_monthly": _period_std(net, ts_sel, "M"),
        "worst_7_rolling_days": _worst_rolling_days(net, ts_sel, 7),
        "worst_15_rolling_days": _worst_rolling_days(net, ts_sel, 15),
        "worst_30_rolling_days": _worst_rolling_days(net, ts_sel, 30),
        "sortino_weekly": _period_sortino(net, ts_sel, "W"),
        "sortino_monthly": _period_sortino(net, ts_sel, "ME"),
        "between_asset_mean_utility_std": float(asset_means.std(ddof=0)) if len(asset_means) > 1 else 0.0,
        "trades": int(len(net)),
    }


def _regime_top30_comparison(
    raw_scores: np.ndarray,
    adjusted_scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    symbols: np.ndarray,
    *,
    cost_pct: float,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            _top30_row("without_regime_adaptation", raw_scores, returns, timestamps, symbols, cost_pct),
            _top30_row("with_regime_adaptation", adjusted_scores, returns, timestamps, symbols, cost_pct),
        ]
    )


def _compare_metrics(raw: pd.DataFrame, adj: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for frac in sorted(set(raw["top_frac"]).intersection(set(adj["top_frac"]))):
        r = raw[raw["top_frac"] == frac].iloc[0]
        a = adj[adj["top_frac"] == frac].iloc[0]
        rows.append(
            {
                "top_frac": float(frac),
                "lift_ratio": _safe_ratio(float(a["lift"]), float(r["lift"])),
                "net_ret_ratio": _safe_ratio(float(a["net_ret"]), float(r["net_ret"])),
                "gross_ret_ratio": _safe_ratio(
                    float(a["mean_gross_return"]), float(r["mean_gross_return"])
                ),
                "std_ratio": _safe_ratio(
                    float(a["return_std"]), float(r["return_std"])
                ),
                "dd_ratio": _safe_ratio(
                    float(a["max_drawdown"]), float(r["max_drawdown"])
                ),
            }
        )
    return pd.DataFrame(rows)


def _empty_artifact(
    strategy_id: str, model_name: str, features: Sequence[str], mapping: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "schema_version": "v1",
        "strategy_id": str(strategy_id),
        "model_name": str(model_name),
        "features": list(features),
        "feature_mapping": mapping,
        "percentile_refs": {},
        "feature_splines": {},
        "distribution_feature_spec": {"enabled": False, "columns": []},
        "reliability_feature_spec": {"global_brier": 0.25},
        "symbol_feature_spec": {
            "global_utility": 0.0,
            "global_bad_rate": 0.5,
            "symbols": {},
        },
        "trust_model": {"enabled": False, "reason": "empty_artifact"},
        "trust_gate_threshold": -0.35,
        "elastic_net": {
            "coef": [],
            "intercept": 0.0,
            "train_prediction_mean": 0.0,
            "params": {},
            "scaler": {"center": [], "scale": []},
        },
        "clips": {
            "total_log_weight_clip": [-0.35, 0.22],
            "regime_weight_clip": [0.70, 1.25],
        },
        "bucket_gates": [],
        "asset_gates": [],
        "rank_normalization": {
            "method": "pandas_rank_pct_average",
            "score": "deployment_score",
        },
        "enable_regime_adaptor": False,
        "enable_regime_adaptor_inference": False,
        "inference_integration_mode": "disabled",
    }


def fixed_bucket_diagnostics(
    regime_df: pd.DataFrame,
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    symbols: np.ndarray,
    strategy_id: str,
    model_name: str,
    percentile_refs: Dict[str, np.ndarray],
) -> pd.DataFrame:
    strategy_tops = score_metrics(scores, returns, timestamps, top_fracs=(0.01, 0.05))
    rows: List[Dict[str, Any]] = []
    buckets = [(0.0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 1.0)]
    for feat in regime_df.columns:
        pct = _apply_percentile(
            regime_df[feat].values,
            percentile_refs.get(feat, _fit_percentile(regime_df[feat].values)),
        )
        for lo, hi in buckets:
            mask = (pct >= lo) & (pct < hi if hi < 1.0 else pct <= hi)
            local_scores = np.where(mask, scores, np.nan)
            bucket_metrics = score_metrics(
                local_scores, returns, timestamps, top_fracs=(0.01, 0.05)
            )
            for _, bm_row in bucket_metrics.iterrows():
                bm = bm_row.to_dict()
                frac = float(bm.get("top_frac", 0.0))
                st_rows = strategy_tops[
                    np.isclose(strategy_tops["top_frac"].astype(float), frac)
                ]
                if st_rows.empty:
                    continue
                strategy_top = st_rows.iloc[0].to_dict()
                gross_ratio = _safe_ratio(
                    bm["mean_gross_return"], strategy_top["mean_gross_return"]
                )
                std_ratio = _safe_ratio(bm["return_std"], strategy_top["return_std"])
                dd_ratio = _safe_ratio(bm["max_drawdown"], strategy_top["max_drawdown"])
                rows.append(
                    {
                        "strategy_id": strategy_id,
                        "model": model_name,
                        "feature": feat,
                        "bucket_type": "fixed",
                        "lo": float(lo),
                        "hi": float(hi),
                        "n": int(np.sum(mask)),
                        "lift_ratio_vs_strategy": _safe_ratio(
                            bm["lift"], strategy_top["lift"]
                        ),
                        "gross_ret_ratio": gross_ratio,
                        "hit_rate_ratio": _safe_ratio(
                            bm["hit_rate"], strategy_top["hit_rate"]
                        ),
                        "return_std_ratio": std_ratio,
                        "drawdown_ratio": dd_ratio,
                        "regime_gated": bool(
                            gross_ratio < 0.7 and std_ratio > 1.3 and dd_ratio > 1.3
                        ),
                        **bm,
                    }
                )
    return pd.DataFrame(rows)


def asset_diagnostics(
    scores: np.ndarray,
    returns: np.ndarray,
    timestamps: Optional[np.ndarray],
    symbols: np.ndarray,
    strategy_id: str,
    model_name: str,
) -> pd.DataFrame:
    strategy_tops = score_metrics(scores, returns, timestamps, top_fracs=(0.01, 0.05))
    rows: List[Dict[str, Any]] = []
    for sym in sorted(set(str(s) for s in symbols)):
        mask = np.asarray([str(s) == sym for s in symbols], dtype=bool)
        if int(np.sum(mask)) < 10:
            continue
        local_scores = np.where(mask, scores, np.nan)
        asset_metrics = score_metrics(
            local_scores, returns, timestamps, top_fracs=(0.01, 0.05)
        )
        for _, bm_row in asset_metrics.iterrows():
            bm = bm_row.to_dict()
            frac = float(bm.get("top_frac", 0.0))
            st_rows = strategy_tops[
                np.isclose(strategy_tops["top_frac"].astype(float), frac)
            ]
            if st_rows.empty:
                continue
            strategy_top = st_rows.iloc[0].to_dict()
            gross_ratio = _safe_ratio(
                bm["mean_gross_return"], strategy_top["mean_gross_return"]
            )
            std_ratio = _safe_ratio(bm["return_std"], strategy_top["return_std"])
            dd_ratio = _safe_ratio(bm["max_drawdown"], strategy_top["max_drawdown"])
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "model": model_name,
                    "symbol": sym,
                    "n": int(np.sum(mask)),
                    "gross_ret_ratio": gross_ratio,
                    "return_std_ratio": std_ratio,
                    "drawdown_ratio": dd_ratio,
                    "asset_gated": bool(
                        gross_ratio < 0.6 and std_ratio > 1.4 and dd_ratio > 1.4
                    ),
                    **bm,
                }
            )
    return pd.DataFrame(rows)


def _bucket_gates(fixed: pd.DataFrame) -> List[Dict[str, Any]]:
    if fixed.empty or "regime_gated" not in fixed.columns:
        return []
    rows = fixed[fixed["regime_gated"]]
    return [
        {"feature": str(r["feature"]), "lo": float(r["lo"]), "hi": float(r["hi"])}
        for _, r in rows.iterrows()
    ]


def _asset_gates(asset_diag: pd.DataFrame) -> List[str]:
    if asset_diag.empty or "asset_gated" not in asset_diag.columns:
        return []
    return sorted(
        str(s) for s in asset_diag.loc[asset_diag["asset_gated"], "symbol"].tolist()
    )


def audit_rolling_regime_readiness(
    artifact: Dict[str, Any],
    *,
    live_feature_frame: Optional[pd.DataFrame] = None,
    downstream_candidate_frame: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Summarize whether a rolling bad-regime artifact is wired for use.

    This is intentionally a lightweight structural audit: it does not judge the
    trading objective itself, but it highlights the common integration states
    that make the rolling adaptor train-only or live-disabled.
    """
    is_rolling = artifact.get("schema_version") in {
        "rolling_bad_regime_v2",
        "rolling_regime_v1",
    } or "selected_combination_params" in artifact
    trade_eval_available = bool(
        artifact.get("trade_candidate_eval_available", False)
    )
    evaluation_universe = str(artifact.get("evaluation_universe", ""))
    enablement_uses_trade_candidates = (
        trade_eval_available and evaluation_universe == "trade_candidates"
    )

    missing_live_cols: List[str]
    if live_feature_frame is None:
        missing_live_cols = list(
            artifact.get("missing_live_p_bad_regime_columns", [])
        )
        if isinstance(missing_live_cols, str):
            missing_live_cols = [c for c in missing_live_cols.split(",") if c]
    else:
        missing_live_cols = [
            c
            for c in REQUIRED_LIVE_BAD_REGIME_COLUMNS
            if c not in live_feature_frame.columns
        ]
    live_columns_available = not missing_live_cols

    downstream_rank_columns = ("final_global_rank", "final_side_rank")
    downstream_has_pre_rank = False
    downstream_rank_available = False
    if downstream_candidate_frame is not None:
        downstream_has_pre_rank = (
            "deployment_score_pre_rank" in downstream_candidate_frame.columns
        )
        downstream_rank_available = any(
            c in downstream_candidate_frame.columns for c in downstream_rank_columns
        )

    live_feature_missingness_by_scope: Dict[str, Dict[str, float]] = {}
    high_missing_live_features: Dict[str, List[str]] = {}
    feature_key_lists = artifact.get("feature_key_lists", {})
    if live_feature_frame is not None and isinstance(feature_key_lists, dict):
        for scope, cols in feature_key_lists.items():
            scope_missingness: Dict[str, float] = {}
            for col in cols or []:
                col = str(col)
                if col not in live_feature_frame.columns:
                    missing_rate = 1.0
                else:
                    vals = pd.to_numeric(
                        live_feature_frame[col], errors="coerce"
                    ).to_numpy(dtype=np.float64)
                    missing_rate = (
                        float(np.mean(~np.isfinite(vals))) if len(vals) else 1.0
                    )
                scope_missingness[col] = missing_rate
            live_feature_missingness_by_scope[str(scope)] = scope_missingness
            high_missing_live_features[str(scope)] = [
                col for col, rate in scope_missingness.items() if rate > 0.50
            ]

    checks = {
        "rolling_artifact": bool(is_rolling),
        "trade_candidate_eval_available": trade_eval_available,
        "evaluation_universe_is_trade_candidates": evaluation_universe
        == "trade_candidates",
        "live_p_bad_regime_columns_available": live_columns_available,
        "downstream_rank_available": downstream_rank_available,
    }
    if downstream_candidate_frame is None:
        checks["downstream_rank_available"] = None

    return {
        "schema_version": artifact.get("schema_version", ""),
        "enable_regime_adaptor": bool(artifact.get("enable_regime_adaptor", False)),
        "reason": artifact.get("reason", ""),
        "trade_candidate_eval_available": trade_eval_available,
        "evaluation_universe": evaluation_universe,
        "enablement_uses_trade_candidates": enablement_uses_trade_candidates,
        "live_required_columns": list(REQUIRED_LIVE_BAD_REGIME_COLUMNS),
        "live_required_columns_available": live_columns_available,
        "missing_live_p_bad_regime_columns": missing_live_cols,
        "downstream_has_deployment_score_pre_rank": downstream_has_pre_rank,
        "downstream_rank_columns": list(downstream_rank_columns),
        "downstream_rank_available": downstream_rank_available
        if downstream_candidate_frame is not None
        else None,
        "live_feature_missingness_by_scope": live_feature_missingness_by_scope,
        "high_missing_live_features": high_missing_live_features,
        "rank_scope": artifact.get("rank_scope", ""),
        "rank_requirement": artifact.get("rank_requirement", ""),
        "checks": checks,
    }


def save_regime_adaptor_outputs(
    data_root: str,
    run_id: str,
    strategy_id: str,
    fit: RegimeAdaptorFit,
    market_mode: str = "spot",
) -> Path:
    out_dir = (
        Path(data_root)
        / "artifacts"
        / run_id
        / "ridge_sizer"
        / "regime_adaptors"
        / safe_strategy_slug(strategy_id)
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    market_mode = normalize_market_mode(market_mode)
    fit.artifact["market_mode"] = market_mode
    artifact_path = market_file_path(out_dir / "regime_adaptor.json", market_mode)
    artifact_path.write_text(
        json.dumps(_jsonify(fit.artifact), indent=2, sort_keys=True)
    )
    for name, frame in (
        ("regime_diagnostics_fixed", fit.fixed_diagnostics),
        ("regime_diagnostics_adaptive", fit.adaptive_diagnostics),
        ("regime_asset_diagnostics", fit.asset_diagnostics),
        ("regime_before_after_metrics", fit.metrics),
    ):
        if frame is None or frame.empty:
            continue
        frame.to_parquet(
            market_file_path(out_dir / f"{name}.parquet", market_mode), index=False
        )
        market_file_path(out_dir / f"{name}.json", market_mode).write_text(
            frame.to_json(orient="records", indent=2)
        )
    return artifact_path


def load_regime_adaptor(
    path: str | Path, market_mode: str | None = None
) -> Dict[str, Any]:
    return json.loads(resolve_market_file_path(Path(path), market_mode).read_text())


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
