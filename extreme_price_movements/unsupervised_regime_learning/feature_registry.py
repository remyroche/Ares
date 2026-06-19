"""Primitive feature registry for unsupervised regime learning.

The lists in this module are deliberately import-light so they can be wired into
``config.py`` without pulling in scipy/sklearn or training code.
"""

from __future__ import annotations

from typing import Iterable


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(str(v) for v in values if str(v)))


BINARY_PRIMITIVE_FEATURES = _dedupe(
    [
        "ema20_gt_ema50",
        "ema50_gt_ema200",
        "price_lt_ema200",
    ]
)


def _exclude_binary(values: Iterable[str]) -> list[str]:
    binary = set(BINARY_PRIMITIVE_FEATURES)
    return [value for value in _dedupe(values) if value not in binary]


CORE_CONTINUOUS_REGIME_FEATURES = _exclude_binary(
    [
        # trend / path / compression
        "ema20_gt_ema50",
        "ema50_gt_ema200",
        "ema50_ema200_spread_atr",
        "price_lt_ema200",
        "ema50_slope",
        "trend_strength_percentile",
        "ema20_slope_5h",
        "ema_slope_norm",
        "trend_acceleration",
        # Path asymmetry / tail ratios
        "vol_shock_asym_8_24",
        "up_down_semivol_ratio_tanh",
        "up_down_return_mass_ratio_tanh",
        "tail_asymmetry_q90_q10_atr_norm",
        # volatility / compression
        "realized_volatility_24h",
        "rv_24h",
        "atr_change_rate",
        "true_range_percentile",
        "bollinger_band_width",
        "rolling_range_20",
        "atr_percentile",
        "prior_range",
        "prior_volatility",
        "compression_ratio",
        "range_expansion_ratio",
        "volatility_ratio_short_long",
        "atr_compression_ratio",
        "compression_score",
        "variance_ratio_10_48",
        "volatility_of_volatility_48",
        "volatility_autocorr_48",
        # path structure / latent dynamics
        "efficiency_ratio_20",
        "choppiness_index_20",
        "direction_entropy_20",
        "return_autocorr_48",
        "bars_since_ema20_ema50_cross_log_norm",
        "bars_in_high_vol_state_log_norm",
        "bars_outside_ema20_atr_band_log_norm",
        "up_down_semivol_ratio_tanh",
        "up_down_return_mass_ratio_tanh",
        "tail_asymmetry_q90_q10_atr_norm",
        "spectral_entropy_ret_24",
        "path_efficiency_24",
        "coherence_24",
        # liquidity / volume
        "volume_percentile",
        "volume_zscore_48h",
        "volume_trend_48",
        "volume_autocorr_48",
        "volume_entropy_24",
        "amihud_illiq",
        "amihud_z",
        # cross-sectional / cross-asset
        "mkt_ret_eq_24h",
        "cs_rank_ret_24h",
        "market_breadth_4h",
        "market_breadth_24h",
        "market_dispersion_4h",
        "market_dispersion_24h",
        "symbol_minus_mkt_ret_24h",
    ]
)


ENTROPY_FEATURES = _dedupe(
    [
        "direction_entropy_20",
        "spectral_entropy_ret_24",
        "volume_entropy_24",
        "choppiness_index_20",
        "efficiency_ratio_20",
        "path_efficiency_24",
        "coherence_24",
    ]
)


PERP_STRUCTURE_FEATURES = _dedupe(
    [
        # funding
        "funding_rate",
        "fund_rate_z_14d",
        "funding_z",
        "funding_abs_z",
        "funding_per_hour",
        "funding_per_hour_z",
        "funding_rank_30d",
        "funding_persistence",
        "funding_mom_2h",
        "funding_mom_4h",
        "funding_mom_8h",
        "funding_mom_w",
    ]
)


OI_FEATURES = _dedupe(
    [
        "oi_value_log",
        "oi_value_1d_log_chg",
        "oi_value_3d_log_chg",
        "oi_value_7d_log_chg",
        "price_1d_ret_z",
        "price_3d_ret_z",
        "price_7d_ret_z",
        "funding_z_90d",
        "oi_1d_chg_z_90d",
        "oi_3d_chg_z_90d",
        "oi_7d_chg_z_90d",
        "funding_1d_chg",
        "funding_1d_chg_z_90d",
        "oi_value_log_z_30d",
        "oi_value_log_z_90d",
        "oi_value_1d_log_chg_z_90d",
        "oi_value_3d_log_chg_z_90d",
        "oi_value_7d_log_chg_z_90d",
        "oi_value_7d_log_chg_z_180d",
        "oi_value_1d_chg_z_90d",
        "oi_value_3d_chg_z_90d",
        "oi_value_7d_chg_z_90d",
        "oi_value_7d_chg_z_180d",
        "oi_value_z_30d",
        "oi_value_z_90d",
        "oi_value_pct_90d",
        "oi_value_log_cp_z_8_32_96",
        "oi_value_log_cp_logstd_8_32",
        "oi_value_log_cp_absratio_8_32",
        "oi_value_1d_log_chg_cp_z_8_32_96",
        "oi_value_1d_log_chg_cp_logstd_8_32",
        "oi_value_1d_log_chg_cp_absratio_8_32",
        "log_oi_to_volume_1d_cp_z_8_32_96",
        "log_oi_to_volume_1d_cp_logstd_8_32",
        "log_oi_to_volume_1d_cp_absratio_8_32",
        "asset_minus_mkt_oi_1d_cp_z_8_32_96",
        "asset_minus_mkt_oi_1d_cp_logstd_8_32",
        "asset_minus_mkt_oi_1d_cp_absratio_8_32",
        "log_oi_to_volume_1d",
        "log_oi_to_volume_7d",
        "oi_to_volume_1d_z_90d",
        "oi_to_volume_7d_z_180d",
        "price_x_oi_1d",
        "price_x_oi_3d",
        "price_x_oi_7d",
        "oi_1d_x_funding",
        "oi_3d_x_funding",
        "oi_7d_x_funding",
        "asset_minus_mkt_oi_1d",
        "asset_minus_mkt_oi_7d",
        "mkt_oi_z_30d",
        "mkt_oi_chg_z_24h",
        "mkt_oi_breadth_rising_24h",
        "mkt_oi_dispersion_24h",
        "cs_rank_oi_value_z_30d",
        "cs_rank_oi_chg_1d_z_90d",
    ]
)


RESIDUAL_FEATURES = _dedupe(
    [
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
        "trend_pct_mkt_resid",
        "atr_expansion_ts_resid",
        "coherence_24_ts_resid",
        "fund_abs_z_mkt_resid",
        "basis_fund_div_mkt_resid",
        "xasset_funding_ts_resid",
        "xasset_funding_peer_resid",
        "funding_1d_chg_ts_resid",
        "funding_1d_chg_peer_resid",
        "asset_minus_mkt_oi_1d_ts_resid",
        "asset_minus_mkt_oi_7d_ts_resid",
        "asset_minus_mkt_oi_1d_peer_resid",
        "asset_minus_mkt_oi_7d_peer_resid",
        "volume_price_corr_ts_resid",
        "path_efficiency_24_ts_resid",
    ]
)


CROSS_ASSET_FEATURES = _dedupe(
    [
        "cs_rank_ret_24h",
        "mkt_ret_eq_1h",
        "mkt_ret_eq_4h",
        "mkt_ret_eq_24h",
        "symbol_minus_mkt_ret_1h",
        "symbol_minus_mkt_ret_4h",
        "symbol_minus_mkt_ret_24h",
        "market_breadth_1h",
        "market_breadth_4h",
        "market_breadth_24h",
        "market_dispersion_1h",
        "market_dispersion_4h",
        "market_dispersion_24h",
    ]
)


PRICE_MEMORY_FEATURES = _dedupe(
    [
        "log_bars_since_above_1atr",
        "log_bars_since_below_1atr",
        "memory_asymmetry_1ATR",
        "log_bars_since_above_2atr",
        "log_bars_since_below_2atr",
        "memory_asymmetry_2ATR",
        "log_bars_since_above_3atr",
        "log_bars_since_below_3atr",
        "memory_asymmetry_3ATR",
    ]
)


UNSUPERVISED_REGIME_FEATURE_GROUPS = {
    "core_continuous_regime_features": CORE_CONTINUOUS_REGIME_FEATURES,
    "entropy_features": ENTROPY_FEATURES,
    "perp_structure_features": PERP_STRUCTURE_FEATURES,
    "oi_features": OI_FEATURES,
    "residual_features": RESIDUAL_FEATURES,
    "cross_asset_features": CROSS_ASSET_FEATURES,
    "price_memory_features": PRICE_MEMORY_FEATURES,
}

UNSUPERVISED_REGIME_PRIMITIVE_FEATURES = _dedupe(
    feature
    for group in UNSUPERVISED_REGIME_FEATURE_GROUPS.values()
    for feature in group
    if feature not in set(BINARY_PRIMITIVE_FEATURES)
)

UNSUPERVISED_REGIME_FEATURE_MECHANISMS = {
    feature: group_name.removesuffix("_features")
    for group_name, group_features in UNSUPERVISED_REGIME_FEATURE_GROUPS.items()
    for feature in group_features
}

UNSUPERVISED_REGIME_LEARNING_DEFAULTS = {
    "primitive_feature_keys": UNSUPERVISED_REGIME_PRIMITIVE_FEATURES,
    "primitive_feature_groups": UNSUPERVISED_REGIME_FEATURE_GROUPS,
    "excluded_primitive_feature_keys": BINARY_PRIMITIVE_FEATURES,
    "feature_mechanisms": UNSUPERVISED_REGIME_FEATURE_MECHANISMS,
    "quality": {
        "symbol_col": "symbol",
        "timestamp_col": "timestamp",
        "warmup_rows": 240,
        "min_good_row_fraction": 0.90,
        "treat_zero_as_low_quality": True,
    },
    "primitive_selection": {
        "target_features": 100,
        "initial_spearman_threshold": 0.96,
        "threshold_step": 0.005,
        "max_spearman_threshold": 0.999,
        "spearman_max_corr_rows": 25000,
        "spearman_corr_time_bins": 24,
        "spearman_max_candidates": 0,
        "block_hours": 24 * 7,
        "min_block_rows": 48,
        "autocorr_lag": 1,
    },
    "operator_selection": {
        "target_features": 400,
        "initial_spearman_threshold": 0.95,
        "pair_initial_spearman_threshold": 0.96,
        "threshold_step": 0.005,
        "max_spearman_threshold": 0.999,
        "quality_dynamics_weighted_score": True,
        "spearman_max_corr_rows": 25000,
        "spearman_corr_time_bins": 24,
        "max_regular_candidates_for_spearman": 0,
        "max_pair_features_for_spearman": 1200,
    },
    "operators": {
        "quantile_window": 168,
        "autocorr_window": 168,
        "autocorr_lag": 1,
        "pair_window": 168,
        "sparse_graph_enabled": True,
        "sparse_graph_block_hours": 168,
        "sparse_graph_min_block_rows": 48,
        "sparse_graph_alpha": 0.05,
        "sparse_graph_partial_corr_threshold": 1e-4,
        "sparse_graph_max_iter": 100,
        "sparse_graph_weight": 0.50,
        "max_pair_candidates_for_generation": 600,
        "pair_candidate_oversample_multiplier": 2.0,
        "eigen_window": 168,
        "eigen_top_k": 3,
        "svd_mode": "walk_forward_prior_only",
        "svd_walk_forward_block_hours": 24 * 7,
        "svd_min_prior_rows": 500,
        "svd_max_reference_rows": 50000,
        "knn_max_reference_rows": 20000,
        "svd_sample_time_bins": 24,
        "svd_components": [8, 16, 32],
        "knn_svd_components": 16,
        "knn_neighbors": 25,
    },
    "regime_models": {
        "enabled": False,
        "max_rows": 50000,
        "sample_time_bins": 32,
        "scaling_mode": "causal_expanding",
        "scaling_min_periods": 64,
        "selector_backend": "lgbm",
        "null_ratio": 1.0,
        "null_block_size": 24,
        "stability_bootstraps": 12,
        "stability_top_m": 80,
        "bootstrap_block_hours": 24 * 7,
        "max_depth": 3,
        "large_sample_depth_threshold": 100000,
        "max_depth_large_sample": 4,
        "min_leaf_fraction": 0.025,
        "n_estimators": 80,
        "learning_rate": 0.05,
        "max_classifier_rows": 60000,
        "lgbm_feature_fraction": 0.85,
        "lgbm_bagging_fraction": 0.85,
        "lgbm_bagging_freq": 1,
        "lgbm_min_gain_to_split": 0.0,
        "lgbm_lambda_l1": 0.0,
        "lgbm_lambda_l2": 0.0,
        "conservative_threshold": 0.80,
        "strong_threshold": 0.70,
        "exploratory_threshold": 0.50,
        "leaf_trees": 80,
        "leaf_embedding_dim": 8,
        "raw_embedding_dim": 8,
        "n_regimes": 5,
        "min_regime_duration": 4,
        "bayesian_gmm_covariance_type": "diag",
        "bayesian_gmm_weight_concentration_prior": 0.0,
        "bayesian_gmm_reg_covar": 1e-6,
        "bayesian_gmm_max_iter": 200,
        "hdbscan_min_cluster_size": 0,
        "hdbscan_min_cluster_size_fraction": 0.0,
        "hdbscan_min_samples": 0,
        "hdbscan_cluster_selection_epsilon": 0.0,
        "hdbscan_cluster_selection_method": "eom",
        "hmm_covariance_type": "diag",
        "hmm_n_iter": 100,
        "hmm_tol": 1e-2,
        "hmm_min_covar": 1e-3,
        "hmm_transmat_self_bias": 0.0,
        "hmm_startprob_prior": 1.0,
        "spectral_n_neighbors": 10,
        "spectral_affinity": "nearest_neighbors",
        "spectral_assign_labels": "kmeans",
        "spectral_gamma": 1.0,
        "kmeans_n_init": 10,
        "kmeans_max_iter": 300,
        "kmeans_tol": 1e-4,
        "kmeans_algorithm": "lloyd",
        "mfa_regimes": 5,
        "mfa_factors": 3,
        "mfa_max_iter": 25,
        "mfa_l1_lambda": 0.001,
        "mfa_tol": 1e-4,
        "mfa_relevance_min": 0.0,
        "mfa_min_keep_features": 8,
        "ae_latent_dim": 8,
        "ae_hidden_dim": 32,
        "ae_epochs": 40,
        "ae_batch_size": 256,
        "ae_backend": "numpy",
        "ae_torch_enabled": False,
        "ae_max_train_rows": 20000,
        "ae_learning_rate": 1e-3,
        "ae_weight_decay": 1e-4,
        "ae_dropout": 0.05,
        "ae_noise": 0.03,
        "ae_family_mask_rate": 0.15,
        "ae_lambda_sparse": 1e-3,
        "ae_lambda_contrastive": 0.20,
        "ae_lambda_smooth": 0.01,
        "ae_temperature": 0.20,
        "keep_candidate_margin": 0.0,
    },
}
