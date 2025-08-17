# src/training/steps/step3_feature_engineering.py

import asyncio
from typing import Any
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.utils.logger import system_logger
import pickle
from src.utils.decorators import with_tracing_span, guard_dataframe_nulls


# Import training pipeline decorators for comprehensive security and troubleshooting
from src.utils.training_pipeline_decorators import (
    validate_step_prerequisites,
    secure_data_processing,
    prevent_data_leakage,
    resource_monitor,
    memory_efficient,
    debug_training_step,
    circuit_breaker_protection,
    validate_step_output,
    quality_gate,
)


@validate_step_prerequisites(
    required_directories=["data/training"],
    min_memory_gb=8.0,
    min_disk_gb=5.0,
    required_packages=["pandas", "numpy", "sklearn", "talib"],
    data_quality_checks={
        "min_rows": 1000,
        "required_columns": ["timestamp", "open", "high", "low", "close", "volume"],
    },
    context="Feature Engineering",
)
@secure_data_processing(
    backup_before=True, integrity_checks=True, memory_cleanup=True, data_validation=True
)
@prevent_data_leakage(
    temporal_validation=True,
    feature_leakage_detection=True,
    lookahead_bias_prevention=True,
)
@resource_monitor(
    memory_threshold_gb=16.0,
    cpu_threshold_percent=90.0,
    disk_threshold_gb=10.0,
    monitor_interval=30.0,
    auto_cleanup=True,
)
@memory_efficient(
    chunk_size=10000, streaming_processing=True, memory_pool=True, cleanup_frequency=25
)
@debug_training_step(
    log_intermediate_results=True,
    save_debug_artifacts=True,
    performance_profiling=True,
    error_context_preservation=True,
)
@circuit_breaker_protection(
    failure_threshold=3,
    recovery_timeout=180.0,
    expected_exception=Exception,
    monitor_interval=30.0,
)
@validate_step_output(
    required_files=["data/training/{exchange}_{symbol}_features_train.parquet"],
    data_quality_checks={"min_rows": 100, "required_columns": ["features", "targets"]},
    performance_thresholds={"engineering_time_minutes": 60.0},
    format_validation=True,
)
@quality_gate(
    data_quality_metrics={"completeness": 0.9, "consistency": 0.8},
    validation_score_requirements={"feature_quality": 0.7},
)
async def run_step(
    symbol: str,
    exchange: str = "BINANCE",
    data_dir: str = "data/training",
    timeframe: str = "1m",
    **kwargs: Any,
) -> bool:
    """
    Step 3: Engineering the features (post-labeling).
    Loads labeled parquet from Step 2 and produces robust feature parquet artifacts for train/val/test.
    Also writes pickle copies with timestamps and a feature hash for Step 5 compatibility.
    """
    logger = system_logger.getChild("Step3.FeatureEngineering")
    try:
        from src.training.steps.vectorized_advanced_feature_engineering import (
            VectorizedAdvancedFeatureEngineering,
        )
        from src.training.enhanced_training_manager_optimized import (
            MemoryEfficientDataManager,
        )

        # 1) Load labeled splits produced by Step 2
        paths = {
            "train": f"{data_dir}/{exchange}_{symbol}_labeled_train.parquet",
            "validation": f"{data_dir}/{exchange}_{symbol}_labeled_validation.parquet",
            "test": f"{data_dir}/{exchange}_{symbol}_labeled_test.parquet",
        }
        labeled = {name: pd.read_parquet(path) for name, path in paths.items()}
        for split, df in labeled.items():
            logger.info(f"📦 Loaded labeled {split}: {len(df)} rows")

        # Ensure timestamp present and set as index for alignment
        for k in labeled.keys():
            if "timestamp" not in labeled[k].columns and isinstance(
                labeled[k].index, pd.DatetimeIndex
            ):
                labeled[k] = (
                    labeled[k].reset_index().rename(columns={"index": "timestamp"})
                )
            if "timestamp" in labeled[k].columns:
                labeled[k]["timestamp"] = pd.to_datetime(
                    labeled[k]["timestamp"], errors="coerce"
                )
                labeled[k] = (
                    labeled[k].dropna(subset=["timestamp"]).sort_values("timestamp")
                )
                labeled[k] = labeled[k].set_index("timestamp")

        # 2) Extract OHLCV inputs
        @with_tracing_span("Step3._extract_inputs", log_args=False)
        @guard_dataframe_nulls(mode="warn", arg_index=0)
        def _extract_inputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            price_cols = [
                c for c in ["open", "high", "low", "close", "volume"] if c in df.columns
            ]
            if len(price_cols) < 4:  # expect at least open/high/low/close
                raise ValueError("🚨 Missing OHLC columns in labeled data")
            price = df[price_cols].copy()
            vol = (
                price[["volume"]].copy()
                if "volume" in price.columns
                else pd.DataFrame({"volume": 1.0}, index=price.index)
            )
            return price, vol

        # SR levels loader with append-and-reuse semantics (prefers Step 2 persisted levels)
        @with_tracing_span("Step3._load_or_build_sr_levels", log_args=False)
        async def _load_or_build_sr_levels(
            price_df: pd.DataFrame, split_name: str
        ) -> dict[str, Any]:
            try:
                data_dir = data_dir_ref
                exchange_local = exchange
                symbol_local = symbol
                import os

                sr_path = (
                    f"{data_dir}/{exchange_local}_{symbol_local}_sr_levels.parquet"
                )
                if os.path.exists(sr_path):
                    sr_df = pd.read_parquet(sr_path)
                    # Retain only recent enough levels and compute age decay; keep full history for training alignment
                    sr_df["timestamp"] = pd.to_datetime(
                        sr_df["timestamp"], errors="coerce"
                    )
                    sr_df = sr_df.dropna(subset=["timestamp"]).sort_values("timestamp")
                    # Align to the split end timestamp; strengths can be decayed by age (optional)
                    end_ts = pd.to_datetime(price_df.index.max())
                    # Build levels list with decayed strength based on age in minutes
                    supports = []
                    resistances = []
                    for _, row in sr_df.iterrows():
                        price = float(row.get("price", np.nan))
                        if not np.isfinite(price):
                            continue
                        base_strength = float(row.get("strength", 0.2))
                        age_min = float(row.get("age", 0.0))
                        # Simple exponential decay by age in hours, configurable via kwargs
                        decay_hl_min = float(
                            kwargs.get("sr_strength_half_life_min", 24 * 60)
                        )
                        if decay_hl_min > 0:
                            lam = np.log(2) / max(decay_hl_min, 1e-6)
                            decayed = base_strength * float(
                                np.exp(-lam * max(0.0, age_min))
                            )
                        else:
                            decayed = base_strength
                        lvl = {
                            "price": price,
                            "strength": float(np.clip(decayed, 0.0, 1.0)),
                        }
                        if (
                            str(row.get("level_type", "support"))
                            .lower()
                            .startswith("support")
                        ):
                            supports.append(lvl)
                        else:
                            resistances.append(lvl)
                    return {
                        "support_levels": supports,
                        "resistance_levels": resistances,
                    }
            except Exception:
                pass
            # Fallback to lightweight builder (percentiles) if persisted SR is unavailable
            try:
                lows = price_df["low"].astype(float)
                highs = price_df["high"].astype(float)
                window = min(len(lows), 2000)
                if window <= 0:
                    return {"support_levels": [], "resistance_levels": []}
                lt = lows.tail(window).dropna()
                ht = highs.tail(window).dropna()
                if lt.empty or ht.empty:
                    return {"support_levels": [], "resistance_levels": []}
                support_prices = np.percentile(lt.values, [5, 15, 30]).tolist()
                resistance_prices = np.percentile(ht.values, [70, 85, 95]).tolist()

                def _mk_levels(vals, strength=0.2):
                    out = []
                    seen: set[float] = set()
                    for v in vals:
                        r = round(float(v), 8)
                        if r in seen:
                            continue
                        seen.add(r)
                        out.append({"price": r, "strength": float(strength)})
                    return out

                return {
                    "support_levels": _mk_levels(support_prices, 0.2),
                    "resistance_levels": _mk_levels(resistance_prices, 0.2),
                }
            except Exception:
                return {"support_levels": [], "resistance_levels": []}

        price_tr, vol_tr = _extract_inputs(labeled["train"])
        price_vl, vol_vl = _extract_inputs(labeled["validation"])
        price_te, vol_te = _extract_inputs(labeled["test"])

        # 3) Initialize FE engine with configuration
        # Get configuration from kwargs or use defaults
        feature_config = kwargs.get("feature_config", {})
        if not feature_config:
            # Default configuration with difference and acceleration features enabled
            feature_config = {
                "vectorized_advanced_features": {
                    "enable_difference_acceleration_features": True,
                    "enable_volatility_modeling": True,
                    "enable_correlation_analysis": True,
                    "enable_momentum_analysis": True,
                    "enable_liquidity_analysis": True,
                    "enable_candlestick_patterns": True,
                    "enable_sr_distance": True,
                    "enable_wavelet_transforms": True,
                    "enable_multi_timeframe": True,
                    "enable_meta_labeling": False,
                    "enable_explicit_meta_labels": False,
                }
            }
        
        # Add symbol and exchange to the feature config for data quality decorator
        feature_config["symbol"] = symbol
        feature_config["exchange"] = exchange
        
        fe = VectorizedAdvancedFeatureEngineering(feature_config)
        await fe.initialize()

        # 4) Engineer features per split
        # Bind data_dir for loader
        data_dir_ref = data_dir
        sr_tr = await _load_or_build_sr_levels(price_tr, "train")
        sr_vl = await _load_or_build_sr_levels(price_vl, "validation")
        sr_te = await _load_or_build_sr_levels(price_te, "test")

        feats_tr = await fe.engineer_features(price_tr, vol_tr, sr_levels=sr_tr)
        feats_vl = await fe.engineer_features(price_vl, vol_vl, sr_levels=sr_vl)
        feats_te = await fe.engineer_features(price_te, vol_te, sr_levels=sr_te)

        X_tr = pd.DataFrame(feats_tr).reindex(price_tr.index)
        X_vl = pd.DataFrame(feats_vl).reindex(price_vl.index)
        X_te = pd.DataFrame(feats_te).reindex(price_te.index)

        # 4a) Join HMM composite clusters (Step 1_7) if available
        try:
            import os

            int_path = os.path.join(
                data_dir,
                f"{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet",
            )

            # Use centralized HMM composite manager
            from src.utils.hmm_composite_manager import get_hmm_composite_manager

            hmm_manager = get_hmm_composite_manager()
            comp_df = hmm_manager.load_composite_clusters(
                exchange, symbol, timeframe, data_dir
            )

            if comp_df is not None:
                if "timestamp" in comp_df.columns:
                    comp_df["timestamp"] = pd.to_datetime(
                        comp_df["timestamp"], errors="coerce", utc=True
                    )
                    comp_df = comp_df.dropna(subset=["timestamp"]).sort_values(
                        "timestamp"
                    )
                    comp_df = comp_df.set_index("timestamp")
                comp_df = comp_df.rename(
                    columns={
                        "combination_id": "hmm_combination_id",
                        "composite_cluster_id": "hmm_composite_cluster_id",
                    }
                )
                # Load intensities if available
                int_df = None
                if os.path.exists(int_path):
                    int_df = pd.read_parquet(int_path)
                    if "timestamp" in int_df.columns:
                        int_df["timestamp"] = pd.to_datetime(
                            int_df["timestamp"], errors="coerce", utc=True
                        )
                        int_df = int_df.dropna(subset=["timestamp"]).sort_values(
                            "timestamp"
                        )
                        int_df = int_df.set_index("timestamp")

                # Align to each split index
                def _merge_clusters(base: pd.DataFrame) -> pd.DataFrame:
                    aligned = comp_df.reindex(base.index)
                    aligned_int = (
                        int_df.reindex(base.index) if int_df is not None else None
                    )
                    merged = base.copy()
                    for c in ["hmm_combination_id", "hmm_composite_cluster_id"]:
                        if c in aligned.columns:
                            merged[c] = aligned[c].astype("float").fillna(-1.0)
                    if aligned_int is not None:
                        # Add all intensity columns
                        for c in aligned_int.columns:
                            if c.startswith("intensity_cluster_"):
                                merged[c] = aligned_int[c].astype("float").fillna(0.0)
                    return merged

                X_tr = _merge_clusters(X_tr)
                X_vl = _merge_clusters(X_vl)
                X_te = _merge_clusters(X_te)
                logger.info(
                    "✅ Joined HMM composite cluster features into Step 3 features"
                )
            else:
                logger.info("ℹ️ HMM composite clusters not available; skipping join")
        except Exception as e:
            logger.warning(f"⚠️ Failed to join HMM composite clusters: {e}")

        # 4b) Optionally augment with Autoencoder features
        @with_tracing_span("Step3._augment_with_autoencoder", log_args=False)
        def _augment_with_autoencoder(
            features_df: pd.DataFrame, split: str
        ) -> pd.DataFrame:
            try:
                from src.analyst.autoencoder_feature_generator import (
                    AutoencoderFeatureGenerator,
                )
            except Exception as e:
                logger.warning(
                    f"⚠️ Autoencoder unavailable for Step 3 augmentation: {e}"
                )
                return features_df
            try:
                ae = AutoencoderFeatureGenerator({})
                y = None
                try:
                    y = (
                        labeled[split]["label"].astype(int).values
                        if "label" in labeled[split].columns
                        else np.zeros(len(features_df))
                    )
                except Exception:
                    y = np.zeros(len(features_df))
                ae_input = features_df.copy()
                ae_df = ae.generate_features(ae_input, f"step3_{split}", y)
                if isinstance(ae_df, pd.DataFrame) and not ae_df.empty:
                    ae_df = ae_df.reindex(features_df.index)
                    merged = pd.concat([features_df, ae_df], axis=1)
                    logger.info(
                        f"✅ Augmented {split} with Autoencoder features: +{ae_df.shape[1]} cols"
                    )
                    return merged
                return features_df
            except Exception as e:
                logger.warning(f"⚠️ Autoencoder augmentation skipped for {split}: {e}")
                return features_df

        # Temporarily disable autoencoder features to avoid validation issues
        if bool(kwargs.get("enable_autoencoder_features", False)):  # Changed from True to False
            X_tr = _augment_with_autoencoder(X_tr, "train")
            X_vl = _augment_with_autoencoder(X_vl, "validation")
            X_te = _augment_with_autoencoder(X_te, "test")

        # 4c) Handle lookahead bias for specific features that need lagging
        @with_tracing_span("Step3._handle_lookahead_bias", log_args=False)
        def _handle_lookahead_bias(features_df: pd.DataFrame) -> pd.DataFrame:
            """Apply lagging to features that may have lookahead bias."""
            try:
                # List of features that commonly have lookahead bias
                features_needing_lagging = [
                    "market_depth_change", "market_depth_returns", "market_depth_imbalance",
                    "ema20_slope", "sma50_slope", "price_impact", "volume_price_impact",
                    "order_flow_imbalance", "bid_ask_spread_returns", "bid_ask_spread_level",
                    "market_depth_change", "market_depth_returns", "market_depth_imbalance"
                ]
                
                # Find features that exist in the DataFrame and need lagging
                existing_features = [col for col in features_needing_lagging if col in features_df.columns]
                
                if existing_features:
                    logger.info(f"🔧 Applying lagging to {len(existing_features)} features to prevent lookahead bias")
                    
                    # Apply 1-period lag to these features
                    for feature in existing_features:
                        lagged_feature_name = f"{feature}_lag1"
                        features_df[lagged_feature_name] = features_df[feature].shift(1)
                        
                        # Replace original feature with lagged version
                        features_df[feature] = features_df[lagged_feature_name]
                        features_df.drop(columns=[lagged_feature_name], inplace=True)
                    
                    logger.info(f"✅ Applied lagging to features: {existing_features}")
                
                return features_df
            except Exception as e:
                logger.warning(f"⚠️ Lookahead bias handling failed: {e}")
                return features_df

        # Apply lookahead bias handling to all splits
        X_tr = _handle_lookahead_bias(X_tr)
        X_vl = _handle_lookahead_bias(X_vl)
        X_te = _handle_lookahead_bias(X_te)

        # 5) Basic sanitization: drop constant columns, handle inf/nan
        @with_tracing_span("Step3._sanitize", log_args=False)
        @guard_dataframe_nulls(mode="warn", arg_index=0)
        def _sanitize(df: pd.DataFrame) -> pd.DataFrame:
            df = df.replace([np.inf, -np.inf], np.nan)
            nunique = df.nunique(dropna=True)
            low_var_cols = nunique[nunique <= 1].index.tolist()
            if low_var_cols:
                logger.info(f"🗑️ Dropping {len(low_var_cols)} constant features")
                df = df.drop(columns=low_var_cols, errors="ignore")
            return df.fillna(0)

        X_tr = _sanitize(X_tr)
        X_vl = _sanitize(X_vl)
        X_te = _sanitize(X_te)

        # 6) Cluster-based correlation pruning with cap (|rho| >= threshold)
        @with_tracing_span("Step3._cluster_corr_prune", log_args=False)
        def _cluster_corr_prune(
            train_df: pd.DataFrame,
            thr: float = 0.95,
            max_to_drop: int | None = None,
        ) -> list[str]:
            if train_df.empty:
                return []
            numeric_df = train_df.select_dtypes(include=[np.number]).copy()
            if numeric_df.shape[1] < 2:
                return []
            numeric_df = numeric_df.fillna(0.0)
            cols = list(numeric_df.columns)
            corr = numeric_df.corr().abs()

            # Build adjacency based on threshold
            neighbors = {c: set() for c in cols}
            for i in range(len(cols)):
                for j in range(i + 1, len(cols)):
                    if corr.iloc[i, j] >= thr:
                        ci, cj = cols[i], cols[j]
                        neighbors[ci].add(cj)
                        neighbors[cj].add(ci)

            # Find connected components (clusters)
            visited = set()
            clusters: list[list[str]] = []
            for c in cols:
                if c in visited:
                    continue
                stack = [c]
                cluster = []
                while stack:
                    node = stack.pop()
                    if node in visited:
                        continue
                    visited.add(node)
                    cluster.append(node)
                    for nb in neighbors.get(node, set()):
                        if nb not in visited:
                            stack.append(nb)
                if len(cluster) > 1:
                    clusters.append(cluster)

            # Pick representative per cluster (keep max-variance feature)
            to_drop: list[str] = []
            for cluster in clusters:
                var_series = numeric_df[cluster].var(ddof=0)
                keep_col = str(var_series.idxmax())
                for col in cluster:
                    if col != keep_col:
                        to_drop.append(col)

            if max_to_drop is not None and len(to_drop) > max_to_drop:
                logger.warning(
                    f"⚠️ Cluster correlation pruning proposed {len(to_drop)} removals; capping at {max_to_drop}."
                )
                to_drop = to_drop[:max_to_drop]

            return to_drop

        # Execute cluster correlation pruning with 50% cap (read from config if available)
        initial_feature_count = X_tr.shape[1]
        try:
            from src.utils.config_loader import ConfigLoader
            loader = ConfigLoader()
            fs_conf = loader.load_yaml_config("src/config/feature_selection_config.yaml").get("feature_selection", {})
            cluster_thr = float(fs_conf.get("cluster_corr_threshold", kwargs.get("cluster_corr_threshold", 0.95)))
            cluster_cap_fraction = float(fs_conf.get("cluster_corr_max_removal_fraction", kwargs.get("cluster_corr_max_removal_fraction", 0.5)))
        except Exception:
            cluster_thr = float(kwargs.get("cluster_corr_threshold", 0.95))
            cluster_cap_fraction = float(kwargs.get("cluster_corr_max_removal_fraction", 0.5))
        cluster_cap_count = int(initial_feature_count * cluster_cap_fraction)
        drop_tr = _cluster_corr_prune(X_tr, thr=cluster_thr, max_to_drop=cluster_cap_count)
        removed_corr_count = len(drop_tr)
        if drop_tr:
            logger.info(
                f"🔗 Cluster correlation prune: dropping {len(drop_tr)} features (|rho|>={cluster_thr:.2f}, cap={cluster_cap_count})"
            )
            X_tr = X_tr.drop(columns=drop_tr, errors="ignore")
            X_vl = X_vl.drop(columns=drop_tr, errors="ignore")
            X_te = X_te.drop(columns=drop_tr, errors="ignore")

        # 7) Mutual information screen (classification target 'label')
        try:
            from sklearn.feature_selection import mutual_info_classif

            y = None
            if "label" in labeled["train"].columns:
                # Use classification labels from Step 2
                y = labeled["train"]["label"].astype(int).values
            if y is not None and len(np.unique(y)) > 1 and not X_tr.empty:
                numX = X_tr.select_dtypes(include=[np.number])
                if not numX.empty:
                    mi = mutual_info_classif(
                        numX.values, y, discrete_features=False, random_state=42
                    )
                    mi_s = pd.Series(mi, index=numX.columns).sort_values(
                        ascending=False
                    )
                    # Persist MI scores
                    os.makedirs("log/mi", exist_ok=True)
                    with open(f"log/mi/{exchange}_{symbol}_step3_mi.json", "w") as f:
                        json.dump({"mi": mi_s.to_dict()}, f, indent=2)
                    # Selection policy: keep top-k if provided; otherwise drop bottom quantile
                    mi_top_k = int(kwargs.get("mi_top_k", 0) or 0)
                    if mi_top_k > 0:
                        keep_cols = list(mi_s.head(mi_top_k).index)
                    else:
                        mi_quantile = float(kwargs.get("mi_quantile", 0.66))
                        thr = mi_s.quantile(mi_quantile)
                        keep_cols = list(mi_s[mi_s >= thr].index)
                    
                    # Safety check: ensure we keep at least some features
                    if len(keep_cols) == 0:
                        # If quantile approach resulted in 0 features, keep top 10% or at least 5 features
                        min_features = max(5, int(len(mi_s) * 0.10))
                        keep_cols = list(mi_s.head(min_features).index)
                        logger.warning(f"⚠️ MI quantile resulted in 0 features, keeping top {min_features} features instead")
                    
                    # Apply keep set safely across splits: skip features missing in any split
                    set_tr = set(X_tr.columns)
                    set_vl = set(X_vl.columns)
                    set_te = set(X_te.columns)

                    missing_tr = [c for c in keep_cols if c not in set_tr]
                    missing_vl = [c for c in keep_cols if c not in set_vl]
                    missing_te = [c for c in keep_cols if c not in set_te]

                    present_all = set_tr & set_vl & set_te
                    final_keep_cols = [c for c in keep_cols if c in present_all]

                    def _shorten(cols: list[str], limit: int = 15) -> str:
                        return (
                            ", ".join(cols[:limit]) + ("..." if len(cols) > limit else "")
                        ) if cols else ""

                    if missing_tr:
                        logger.warning(
                            f"⚠️ MI: skipping {len(missing_tr)} features absent in TRAIN split: [{_shorten(missing_tr)}]"
                        )
                    if missing_vl:
                        logger.warning(
                            f"⚠️ MI: skipping {len(missing_vl)} features absent in VALIDATION split: [{_shorten(missing_vl)}]"
                        )
                    if missing_te:
                        logger.warning(
                            f"⚠️ MI: skipping {len(missing_te)} features absent in TEST split: [{_shorten(missing_te)}]"
                        )

                    if not final_keep_cols:
                        logger.warning(
                            "⚠️ MI: no selected features were common across all splits; skipping MI application"
                        )
                    else:
                        X_tr = X_tr[final_keep_cols]
                        X_vl = X_vl[final_keep_cols]
                        X_te = X_te[final_keep_cols]
                        logger.info(
                            f"📊 MI kept {len(final_keep_cols)} features (top_k={mi_top_k} quantile={kwargs.get('mi_quantile', 0.80)})"
                        )
        except Exception as e:
            logger.warning(f"⚠️ MI screening skipped: {e}")

        # 8) VIF reduction (iterative) with combined 50% cap (corr + VIF)
        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            # Read thresholds from config if available
            try:
                from src.utils.config_loader import ConfigLoader
                loader = ConfigLoader()
                fs_conf = loader.load_yaml_config("src/config/feature_selection_config.yaml").get("feature_selection", {})
                vif_thr = float(fs_conf.get("vif_threshold", kwargs.get("vif_threshold", 10.0)))
                overall_cap_fraction = float(fs_conf.get("max_total_removal_fraction", kwargs.get("max_total_removal_fraction", 0.5)))
            except Exception:
                vif_thr = float(kwargs.get("vif_threshold", 10.0))
                overall_cap_fraction = float(kwargs.get("max_total_removal_fraction", 0.5))
            max_iter = int(kwargs.get("max_vif_iterations", 10))
            num_cols = list(X_tr.select_dtypes(include=[np.number]).columns)
            it = 0
            removed_vif: list[str] = []
            overall_cap_count = int(initial_feature_count * overall_cap_fraction)
            vif_allowed = max(0, overall_cap_count - removed_corr_count)
            if vif_allowed == 0:
                logger.info("ℹ️ Skipping VIF pruning: overall 50% cap already reached by correlation pruning")
            
            while it < max_iter and len(num_cols) > 1:
                it += 1
                Xn = X_tr[num_cols].astype(float).fillna(0.0)
                # Standardize to stabilize VIF
                std = Xn.std(ddof=0).replace(0, 1.0)
                Xn = (Xn - Xn.mean()) / std
                vif_vals = pd.Series(
                    [
                        variance_inflation_factor(Xn.values, i)
                        for i in range(Xn.shape[1])
                    ],
                    index=num_cols,
                )
                max_vif = float(vif_vals.max()) if not vif_vals.empty else 0.0
                if max_vif <= vif_thr:
                    break
                drop_col = str(vif_vals.idxmax())
                logger.info(f"📊 VIF prune: dropping {drop_col} (VIF={max_vif:.2f})")
                num_cols.remove(drop_col)
                removed_vif.append(drop_col)
                if len(removed_vif) >= vif_allowed:
                    logger.warning(
                        f"⚠️ VIF cap reached: removed {len(removed_vif)} via VIF; overall cap={overall_cap_count} incl. {removed_corr_count} correlation removals. Stopping VIF."
                    )
                    break
            # Apply final VIF-selected set
            if num_cols:
                X_tr = X_tr[num_cols]
                X_vl = X_vl[num_cols]
                X_te = X_te[num_cols]
                logger.info(
                    f"📊 VIF kept {len(num_cols)} features (threshold={vif_thr}, removed_vif={len(removed_vif)}, removed_corr={removed_corr_count})"
                )
            else:
                # Safety check: if VIF removed all features, keep original features
                logger.warning(f"⚠️ VIF removed all features, keeping original feature set")
                num_cols = list(X_tr.select_dtypes(include=[np.number]).columns)
                if num_cols:
                    X_tr = X_tr[num_cols]
                    X_vl = X_vl[num_cols]
                    X_te = X_te[num_cols]
                    logger.info(f"📊 VIF fallback: kept {len(num_cols)} original features")
        except Exception as e:
            logger.warning(f"⚠️ VIF reduction skipped: {e}")

        # 9) Save features and selected feature lists
        os.makedirs(data_dir, exist_ok=True)
        mem_mgr = MemoryEfficientDataManager()

        @with_tracing_span("Step3._attach_timestamp", log_args=False)
        def _attach_timestamp(
            df_features: pd.DataFrame, labeled_df: pd.DataFrame
        ) -> pd.DataFrame:
            try:
                if (
                    "timestamp" in labeled_df.columns
                    and "timestamp" not in df_features.columns
                ):
                    df_features = df_features.copy()
                    df_features["timestamp"] = labeled_df["timestamp"].values
            except Exception:
                pass
            return df_features

        @with_tracing_span("Step3._save", log_args=False)
        @guard_dataframe_nulls(mode="warn", arg_index=1)
        def _save(name: str, df: pd.DataFrame, labeled_df: pd.DataFrame):
            df_out = _attach_timestamp(df, labeled_df)
            path_parquet = f"{data_dir}/{exchange}_{symbol}_features_{name}.parquet"
            mem_mgr.save_to_parquet(
                mem_mgr.optimize_dataframe(df_out.copy()), path_parquet
            )
            logger.info(
                f"✅ Saved features {name}: {len(df_out)} rows, {df_out.shape[1]} cols -> {path_parquet}"
            )
            # Also save PKL for downstream steps expecting PKL
            try:
                path_pkl = f"{data_dir}/{exchange}_{symbol}_features_{name}.pkl"
                with open(path_pkl, "wb") as f:
                    pickle.dump(df_out, f)
                logger.info(f"✅ Saved features {name} (PKL): {path_pkl}")
            except Exception as e:
                logger.warning(f"⚠️ Unable to save PKL features for {name}: {e}")

        # NEW: Integrate HMM features into main feature datasets
        @with_tracing_span("Step3._integrate_hmm_features", log_args=False)
        def _integrate_hmm_features():
            """Integrate HMM features from Step 2 into the main feature datasets."""
            try:
                logger.info("🔄 Integrating HMM features into main feature datasets...")
                
                # Load HMM features for the current timeframe
                hmm_features = {}
                
                # Load composite clusters
                composite_clusters_path = f"{data_dir}/{exchange}_{symbol}_hmm_composite_clusters_{timeframe}.parquet"
                if os.path.exists(composite_clusters_path):
                    composite_clusters = pd.read_parquet(composite_clusters_path)
                    composite_clusters["timestamp"] = pd.to_datetime(composite_clusters["timestamp"])
                    composite_clusters = composite_clusters.set_index("timestamp")
                    hmm_features["composite_clusters"] = composite_clusters
                    logger.info(f"✅ Loaded composite clusters: {len(composite_clusters)} rows")
                
                # Load composite intensity
                composite_intensity_path = f"{data_dir}/{exchange}_{symbol}_hmm_composite_intensity_{timeframe}.parquet"
                if os.path.exists(composite_intensity_path):
                    composite_intensity = pd.read_parquet(composite_intensity_path)
                    composite_intensity["timestamp"] = pd.to_datetime(composite_intensity["timestamp"])
                    composite_intensity = composite_intensity.set_index("timestamp")
                    hmm_features["composite_intensity"] = composite_intensity
                    logger.info(f"✅ Loaded composite intensity: {len(composite_intensity)} rows")
                
                # Load block states (regime probabilities)
                block_states_path = f"{data_dir}/{exchange}_{symbol}_hmm_block_states_{timeframe}.parquet"
                if os.path.exists(block_states_path):
                    block_states = pd.read_parquet(block_states_path)
                    block_states["timestamp"] = pd.to_datetime(block_states["timestamp"])
                    block_states = block_states.set_index("timestamp")
                    hmm_features["block_states"] = block_states
                    logger.info(f"✅ Loaded block states: {len(block_states)} rows")
                
                if not hmm_features:
                    logger.warning("⚠️ No HMM features found to integrate")
                    return
                
                # Integrate HMM features into each split
                for split_name, features_df in [("train", X_tr), ("validation", X_vl), ("test", X_te)]:
                    logger.info(f"🔄 Integrating HMM features into {split_name} split...")
                    
                    # Start with original features
                    integrated_df = features_df.copy()
                    
                    # Merge composite clusters
                    if "composite_clusters" in hmm_features:
                        composite_clusters = hmm_features["composite_clusters"]
                        # Align timestamps and merge
                        aligned_clusters = composite_clusters.reindex(features_df.index, method='ffill')
                        integrated_df = pd.concat([integrated_df, aligned_clusters.drop(columns=['combination_id'])], axis=1)
                        logger.info(f"✅ Added composite_cluster_id to {split_name}")
                    
                    # Merge composite intensity
                    if "composite_intensity" in hmm_features:
                        composite_intensity = hmm_features["composite_intensity"]
                        # Align timestamps and merge
                        aligned_intensity = composite_intensity.reindex(features_df.index, method='ffill')
                        integrated_df = pd.concat([integrated_df, aligned_intensity.drop(columns=['combination_id'])], axis=1)
                        logger.info(f"✅ Added {len(composite_intensity.columns)-1} intensity features to {split_name}")
                    
                    # Merge block states (regime probabilities)
                    if "block_states" in hmm_features:
                        block_states = hmm_features["block_states"]
                        # Align timestamps and merge
                        aligned_states = block_states.reindex(features_df.index, method='ffill')
                        # Drop state_id columns, keep only probability columns
                        prob_columns = [col for col in aligned_states.columns if col.endswith('_p_state_')]
                        if prob_columns:
                            integrated_df = pd.concat([integrated_df, aligned_states[prob_columns]], axis=1)
                            logger.info(f"✅ Added {len(prob_columns)} regime probability features to {split_name}")
                    
                    # Update the original dataframe
                    if split_name == "train":
                        X_tr = integrated_df
                    elif split_name == "validation":
                        X_vl = integrated_df
                    elif split_name == "test":
                        X_te = integrated_df
                    
                    logger.info(f"✅ {split_name} split now has {integrated_df.shape[1]} features")
                
                logger.info("✅ HMM feature integration completed successfully")
                
            except Exception as e:
                logger.error(f"🚨 HMM feature integration failed: {e}")
                logger.exception("HMM integration error details:")
        
        # Execute HMM feature integration
        _integrate_hmm_features()
        
        # NEW: Enhance HMM features with additional derived features
        @with_tracing_span("Step3._enhance_hmm_features", log_args=False)
        def _enhance_hmm_features():
            """Enhance HMM features with additional derived features for Step 5 compatibility."""
            try:
                from src.training.steps.hmm_feature_enhancer import HMMFeatureEnhancer
                
                logger.info("🔄 Enhancing HMM features with derived features...")
                
                enhancer = HMMFeatureEnhancer(feature_config)
                
                # Enhance each split
                global X_tr, X_vl, X_te
                X_tr = enhancer.enhance_hmm_features(X_tr)
                X_vl = enhancer.enhance_hmm_features(X_vl)
                X_te = enhancer.enhance_hmm_features(X_te)
                
                logger.info("✅ HMM feature enhancement completed")
                
            except Exception as e:
                logger.error(f"🚨 HMM feature enhancement failed: {e}")
                logger.exception("HMM enhancement error details:")
        
        # Execute HMM feature enhancement
        _enhance_hmm_features()

        _save("train", X_tr, labeled["train"])
        _save("validation", X_vl, labeled["validation"])
        _save("test", X_te, labeled["test"])

        # Save feature lists per split and a feature hash
        feature_lists = {
            "train": list(X_tr.columns),
            "validation": list(X_vl.columns),
            "test": list(X_te.columns),
            "timestamp": datetime.now().isoformat(),
        }
        with open(f"{data_dir}/{exchange}_{symbol}_selected_features.json", "w") as f:
            json.dump(feature_lists, f, indent=2)

        # Update feature metadata and hash to keep artifact loader compatibility
        try:
            metadata = {
                "symbol": symbol,
                "exchange": exchange,
                "created_at": datetime.now().isoformat(),
                "feature_config": feature_config if 'feature_config' in locals() else {},
                "feature_counts": {
                    "train": X_tr.shape[1],
                    "validation": X_vl.shape[1],
                    "test": X_te.shape[1],
                },
                "row_counts": {
                    "train": len(X_tr),
                    "validation": len(X_vl),
                    "test": len(X_te),
                },
                "feature_columns": list(X_tr.columns),
            }
            meta_path = f"{data_dir}/{exchange}_{symbol}_features_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

            # Maintain a simple hash.txt file presence expected by the loader
            def _hash_cols(cols: list[str]) -> str:
                s = ",".join(cols)
                import hashlib
                return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

            hash_value = _hash_cols(list(X_tr.columns))
            hash_path = f"{data_dir}/{exchange}_{symbol}_features_hash.txt"
            with open(hash_path, "w") as f:
                f.write(hash_value)

            logger.info(
                f"💾 Updated feature metadata and hash: cols(train)={X_tr.shape[1]} -> {meta_path}"
            )
        except Exception as e:
            logger.warning(f"⚠️ Failed to update feature metadata/hash: {e}")

        # NEW: HMM Composite Regime Data Splitting
        @with_tracing_span("Step3._hmm_composite_regime_splitting", log_args=False)
        async def _hmm_composite_regime_splitting():
            """Split data by HMM composite regimes for regime-specific training."""
            try:
                logger.info("🔄 Starting HMM composite regime data splitting...")

                # Load HMM composite regime data
                hmm_file = f"{data_dir}/{exchange}_{symbol}_hmm_composite_meta_{timeframe}.json"
                if not os.path.exists(hmm_file):
                    logger.warning(f"⚠️ HMM composite file not found: {hmm_file}")
                return

                with open(hmm_file, "r") as f:
                    hmm_data = json.load(f)

                archetype_descriptions = hmm_data.get("archetype_descriptions", {})
                logger.info(
                    f"📊 Loaded {len(archetype_descriptions)} HMM composite archetypes"
                )

                # Create regime data directory
                regime_data_dir = os.path.join(data_dir, "regime_data")
                os.makedirs(regime_data_dir, exist_ok=True)

                # Split each dataset by composite_cluster_id
                regime_splits = {}
                for split_name, features_df in [
                    ("train", X_tr),
                    ("validation", X_vl),
                    ("test", X_te),
                ]:
                    if "composite_cluster_id" not in features_df.columns:
                        logger.warning(
                            f"⚠️ No composite_cluster_id in {split_name} data"
                        )
                        continue

                    # Get unique regime IDs
                    regime_ids = features_df["composite_cluster_id"].dropna().unique()
                    logger.info(
                        f"📊 {split_name}: Found {len(regime_ids)} unique HMM composite regimes"
                    )

                    for regime_id in regime_ids:
                        regime_key = f"hmm_composite_{regime_id}"
                        regime_mask = features_df["composite_cluster_id"] == regime_id
                        regime_data = features_df[regime_mask].copy()

                        if not regime_data.empty:
                            # Add regime description
                            regime_desc = archetype_descriptions.get(
                                str(regime_id), f"Archetype {regime_id}"
                            )
                            regime_data["regime_description"] = regime_desc

                            # Save regime-specific data
                            regime_file = os.path.join(
                                regime_data_dir, f"{split_name}_{regime_key}.parquet"
                            )
                            regime_data.to_parquet(regime_file, index=True)

                            if regime_key not in regime_splits:
                                regime_splits[regime_key] = {
                                    "description": regime_desc,
                                    "splits": {},
                                }
                            regime_splits[regime_key]["splits"][split_name] = {
                                "rows": len(regime_data),
                                "file": regime_file,
                            }

                            logger.info(
                                f"✅ {split_name} regime {regime_key}: {len(regime_data)} rows -> {regime_file}"
                            )

                # Save regime splitting summary
                regime_summary = {
                    "total_regimes": len(regime_splits),
                    "regime_details": regime_splits,
                    "generated_at": datetime.now().isoformat(),
                    "timeframe": timeframe,
                    "exchange": exchange,
                    "symbol": symbol,
                }

                summary_file = os.path.join(
                    data_dir, f"{exchange}_{symbol}_hmm_composite_regime_splits.json"
                )
                with open(summary_file, "w") as f:
                    json.dump(regime_summary, f, indent=2)

                logger.info(
                    f"✅ HMM composite regime splitting completed: {len(regime_splits)} regimes"
                )
                logger.info(f"📄 Regime summary saved to: {summary_file}")

                # Create gating matrix for ensemble training
                @with_tracing_span("Step3._create_gating_matrix", log_args=False)
                def _create_gating_matrix():
                    """Create gating matrix for regime ensemble training."""
                    try:
                        gating_dir = os.path.join(data_dir, "gating")
                        os.makedirs(gating_dir, exist_ok=True)

                        # Create gating matrix from composite_cluster_id probabilities
                        gating_data = []
                        for split_name, features_df in [
                            ("train", X_tr),
                            ("validation", X_vl),
                            ("test", X_te),
                        ]:
                            if "composite_cluster_id" in features_df.columns:
                                # Get regime probabilities (one-hot encoding)
                                regime_probs = pd.get_dummies(
                                    features_df["composite_cluster_id"], prefix="regime"
                                )

                                # Add timestamp and split info
                                gating_df = regime_probs.copy()
                                gating_df["timestamp"] = features_df.index
                                gating_df["split"] = split_name

                                gating_data.append(gating_df)

                        if gating_data:
                            combined_gating = pd.concat(gating_data, ignore_index=True)
                            gating_file = os.path.join(
                                gating_dir,
                                f"{exchange}_{symbol}_hmm_composite_gating.parquet",
                            )
                            combined_gating.to_parquet(gating_file, index=False)
                            logger.info(
                                f"✅ Gating matrix saved: {gating_file} ({len(combined_gating)} rows)"
                            )

                    except Exception as e:
                        logger.warning(f"⚠️ Gating matrix creation failed: {e}")

                _create_gating_matrix()

            except Exception as e:
                logger.error(f"🚨 HMM composite regime splitting failed: {e}")

        # Execute regime splitting
        await _hmm_composite_regime_splitting()

        # NEW: also persist pickle copies with timestamps for Step 5 compatibility
        try:
            import pickle

            for split_name, X in ("train", X_tr), ("validation", X_vl), ("test", X_te):
                X_pick = X.copy()
                X_pick["timestamp"] = X_pick.index
                X_pick = X_pick.reset_index(drop=True)
                pkl_path = f"{data_dir}/{exchange}_{symbol}_features_{split_name}.pkl"
                with open(pkl_path, "wb") as f:
                    pickle.dump(X_pick, f)
                logger.info(
                    f"✅ Wrote pickle features {split_name}: {pkl_path} rows={len(X_pick)} cols={X_pick.shape[1]}"
                )

            # Write a simple feature hash to ensure downstream consistency
            import hashlib

            def _hash_cols(cols: list[str]) -> str:
                s = ",".join(cols)
                return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

            hash_info = {
                "train_hash": _hash_cols(feature_lists["train"]),
                "validation_hash": _hash_cols(feature_lists["validation"]),
                "test_hash": _hash_cols(feature_lists["test"]),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            with open(f"{data_dir}/{exchange}_{symbol}_feature_hash.json", "w") as f:
                json.dump(hash_info, f, indent=2)
        except Exception as e:
            logger.warning(f"⚠️ Pickle compatibility write skipped: {e}")

        logger.info("✅ Step 3: Feature engineering completed successfully")
        return True
    except Exception as e:
        logger.exception(f"🚨 Step 3 feature engineering failed: {e}")
        return False


if __name__ == "__main__":

    async def _test():
        ok = await run_step("ETHUSDT")
        print(f"Step 3 test result: {ok}")

    asyncio.run(_test())
