import logging
import time
import pandas as pd
import numpy as np
import gc
from typing import Dict, Any, List, Optional, Tuple, Callable
from sklearn.metrics import roc_auc_score
from src.utils.numba_funcs import _numba_generate_dollar_bars, _numba_generate_range_bars

from src.utils.ml_common.afml_utils import (
    get_daily_vol, get_t_events, get_vertical_barrier, 
    apply_triple_barrier, get_bins, get_weights_by_uniqueness,
    frac_diff_fixed, get_sample_weights, compute_master_weight, compute_hunter_weight,
    get_sample_uniqueness, get_num_concurrent_events,
    calculate_rolling_volatility, calculate_dynamic_range_threshold
)
from src.utils.ml_common.wavelet_utils import get_wavelet_features
from src.utils.ml_common.physics_router import AdaptiveHunterRouter
from src.utils.ml_common.anarchy_detector import AnarchyDetector
from src.training.steps.market_analysis.offline_causal_discovery import OfflineCausalDiscovery
from src.utils.data.klines_parquet import get_klines_manager
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success, tprint
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_model_with_oof

logger = logging.getLogger(__name__)

class AFMLSpecialistMixin:
    """
    Mixin providing AFML-specific logic for enhanced specialists:
    - Phase 1: Physics Router Integration (Regime Weights)
    - Phase 2: Feature Conditioning (Partial Orthogonalization)
    - Phase 6: Anarchy & Anomaly Detection
    - Phase 0: Hunter Weighting (Uniqueness * Cleanliness * Wavelet * Router)
    - Phase 3/E: Monotonic Constraints via Ridge
    """
    
    # Class-level cache for expensive feature generation
    _PHYSICS_CACHE = {}
    _ROUTER_CACHE = {}  # Cache for fitted routers
    _ANARCHY_CACHE = {}
    _BAR_CACHE = {}

    def _apply_standard_feature_selection(self, features: pd.DataFrame, max_features: int = 30) -> pd.DataFrame:
        tprint_info(f"[{self.__class__.__name__}] Starting _apply_standard_feature_selection with {features.shape[1] if not features.empty else 0} features")
        if features.empty: return features

        # Causal Pruning Integration (Phase 3 Offline Loading)
        # Try to load causal artifacts
        causal_engine = OfflineCausalDiscovery()
        universal_drivers = causal_engine.load_artifacts()

        # If we have drivers, prioritize them
        if universal_drivers:
            # Keep universal drivers present in features
            keep_cols = [c for c in features.columns if c in universal_drivers]
            # Fill the rest with variance-based selection
            remaining_quota = max_features - len(keep_cols)
            if remaining_quota > 0:
                candidates = features.drop(columns=keep_cols)
                # Remove constant/corr logic on candidates...
                constant_features = candidates.columns[candidates.nunique() <= 1]
                if len(constant_features) > 0:
                    candidates = candidates.drop(columns=constant_features)

                # Simple variance
                top_candidates = candidates.var().nlargest(remaining_quota).index.tolist()
                keep_cols.extend(top_candidates)

            return features[keep_cols]

        # Fallback to standard logic if no artifacts
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)

        if features.empty: return features

        # Optimization: If dataset is large, use a subset for correlation check
        # Correlation is usually stable enough on 10k samples
        if len(features) > 10000:
            corr_data = features.sample(n=10000, random_state=42)
        else:
            corr_data = features

        correlation_matrix = corr_data.corr().abs()
        
        # Optimized vectorized detection of high correlation
        upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
        
        if to_drop:
            tprint_info(f"   [Selection] Dropping {len(to_drop)} highly correlated features")
            features = features.drop(columns=to_drop)

        if len(features.columns) > max_features:
            # Use variance but avoid recomputing if possible
            feature_variances = corr_data.var()
            top_features = feature_variances.nlargest(max_features).index
            # Filter to only include features that actually exist in the current DataFrame
            available_top_features = [f for f in top_features if f in features.columns]
            if available_top_features:
                features = features[available_top_features]
                tprint_info(f"   [Selection] Selected {len(available_top_features)}/{len(top_features)} available top features")
            else:
                # Fallback: just take the first max_features columns
                features = features.iloc[:, :max_features]
                tprint_warning(f"   [Selection] No top features available, using first {max_features} columns")

        return features

    def generate_standard_enhanced_features(
        self,
        df: pd.DataFrame,
        specialist_type: SpecialistType,
        manual_feature_func: Optional[Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = None
    ) -> pd.DataFrame:
        tprint_info(f"[{self.__class__.__name__}] generate_standard_enhanced_features start specialist={specialist_type.name}")
        pipeline_features = pd.DataFrame(index=df.index)
        if hasattr(self, 'feature_pipeline'):
            pipeline_features = self.feature_pipeline.generate_enhanced_features(
                df, specialist_type.value, {'enhanced_features': True}
            )

        manual_features = pd.DataFrame(index=df.index)
        if manual_feature_func:
            manual_features = manual_feature_func(df, pipeline_features)

        # Combine
        combined_features = pd.concat([pipeline_features, manual_features], axis=1)
        combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]

        # Apply Phase 2: Feature Conditioning (Partial Orthogonalization)
        # We need Nuisance factors (W).
        # Approximated by [Vol, Trend] of the market data.
        # W = [Rolling Vol, Efficiency]
        if not combined_features.empty:
            vol = df['close'].pct_change().rolling(20).std().fillna(0)
            eff = (df['close'] - df['close'].shift(20)).abs() / (df['close'].diff().abs().rolling(20).sum() + 1e-9)
            W = pd.concat([vol, eff], axis=1).fillna(0)
            W.columns = ['vol_nuisance', 'eff_nuisance']

            # Simple in-memory orthogonalization (could be moved to CausalEngine)
            # X_resid = X - 0.7 * E[X|W]
            # Since we are doing this online per specialist, we do a simple global LinearRegression fit on the batch
            from sklearn.linear_model import LinearRegression

            # Align
            common_idx = combined_features.index.intersection(W.index)
            X_curr = combined_features.loc[common_idx]
            W_curr = W.loc[common_idx]

            # Helper to avoid heavy sklearn import if not needed? No, we need it.
            lr = LinearRegression()
            X_curr_clean = X_curr.fillna(0)
            W_curr_clean = W_curr.fillna(0)
            
            # Remove non-numeric columns before fitting
            numeric_cols = X_curr_clean.select_dtypes(include=[np.number]).columns
            X_curr_clean = X_curr_clean[numeric_cols]

            lr.fit(W_curr_clean, X_curr_clean)
            X_hat = lr.predict(W_curr_clean)
            X_resid = X_curr_clean - (0.7 * X_hat)

            # Keep both Raw and Resid
            X_resid.columns = [f"{c}_resid" for c in X_curr_clean.columns]
            combined_features = pd.concat([X_curr, X_resid], axis=1)

        # Standard Selection
        selected_features = self._apply_standard_feature_selection(combined_features)
        return selected_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    async def execute_standard_specialist_logic(
        self,
        config: Dict[str, Any],
        specialist_type: SpecialistType,
        manual_feature_func: Optional[Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = None,
        filter_type: str = 'price',
        pt_sl_config_key: Optional[str] = None,
        default_pt_sl: List[float] = [2.0, 1.0],
        suffix: str = "enhanced_features"
    ) -> Dict[str, Any]:
        start_time = time.time()
        tprint_info(f"[{self.__class__.__name__}] execute_standard_specialist_logic start specialist={specialist_type.name}")
        try:
            symbol = config.get('symbol', 'BTCUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            
            # [Fix] Explicitly set specialist context to avoid dumping into 'analyst' directory
            specialist_name = self.__class__.__name__
            # Clean up name: EnhancedMLMomentumPersistenceStep -> momentum_persistence
            clean_name = specialist_name.lower().replace('step', '').replace('enhancedml', '').replace('enhanced', '')
            if clean_name.startswith('_'): clean_name = clean_name[1:]
            
            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                model=clean_name
            )
            
            cache_key = (symbol, exchange, timeframe)

            # 1. Load Market Data
            tprint_info(f"   [Phase 1/6] Loading Market Data...")
            if hasattr(self, '_load_market_data_with_cache'):
                df, market_source = self._load_market_data_with_cache(config, timeframe)
            elif hasattr(self, '_load_market_data'):
                df = self._load_market_data(symbol, exchange, timeframe)
                market_source = "loaded"
            else:
                raise NotImplementedError("Specialist must implement _load_market_data")

            if df is None or len(df) < 1000:
                return {"success": False, "error": "Insufficient data"}

            # 2. Feature Generation (Conditioned)
            tprint_info(f"   [Phase 2/6] Generating Conditioned Features...")
            feature_df = self.generate_standard_enhanced_features(df, specialist_type, manual_feature_func)

            # Phase 6: Add Anarchy Features (with Caching)
            tprint_info(f"   [Phase 3/6] Adding Anarchy Features (IF + Path Sig)...")
            if cache_key in self._ANARCHY_CACHE:
                tprint_info("   [Cache] Using cached Anarchy features")
                anarchy_features = self._ANARCHY_CACHE[cache_key]
            else:
                anarchy_detector = AnarchyDetector()
                anarchy_features = anarchy_detector.generate_anarchy_features(df)
                self._ANARCHY_CACHE[cache_key] = anarchy_features
            
            feature_df = pd.concat([feature_df, anarchy_features], axis=1)

            # 3. AFML & Hunter Prep
            tprint_info(f"   [Phase 4/6] AFML & Hunter Preparation (Bars, Physics, Sampling)...")
            X, y, weights = self.prepare_specialist_data(
                market_data=df,
                feature_df=feature_df,
                config=config,
                filter_type=filter_type,
                pt_sl_config_key=pt_sl_config_key,
                default_pt_sl=default_pt_sl
            )

            # 4. Centralized Training
            # Fast fail if insufficient events for training
            if len(X) < 100:
                raise ValueError(f"Insufficient events for training specialist {clean_name}: {len(X)} < 100")

            tprint_info(f"   [Phase 5/6] Training ExtraTrees Model...")

            # Calculate dynamic purge length based on labeling horizon
            # Default to 24h if unclear to be safe
            horizon_bars = config.get('lookforward_bars', 35)
            # Estimate purge duration: horizon_bars * timeframe duration
            # Default assumption 15m if not parseable
            tf_str = config.get('timeframe', '15m')
            try:
                if tf_str.endswith('m'):
                    min_per_bar = int(tf_str[:-1])
                elif tf_str.endswith('h'):
                    min_per_bar = int(tf_str[:-1]) * 60
                elif tf_str.endswith('d'):
                    min_per_bar = int(tf_str[:-1]) * 1440
                else:
                    min_per_bar = 15
            except:
                min_per_bar = 15
                
            horizon_minutes = horizon_bars * min_per_bar
            # Add small buffer + embargo is separate
            purge_minutes = max(1440, horizon_minutes + 60) # Default minimum 24h as requested by user
            purge_length = pd.Timedelta(minutes=purge_minutes)
            
            tprint_info(f"   [Phase 5/6] Calculated Purge: {purge_length} ({horizon_bars} bars * {min_per_bar}m + buffer)")
            
            training_result = train_specialist_model_with_oof(
                X.fillna(0.0),
                y.fillna(0.0),
                sample_weight=weights,
                n_splits=5,
                params_override=config.get('params_override', {}),
                purge_length=purge_length
            )

            # 5. Output Construction
            tprint_info(f"   [Phase 6/6] Finalizing Output & Aligning Probs...")
            oof_probs = training_result.oof_predictions
            final_probs, final_preds = self._align_probabilities_to_index(
                oof_probs=oof_probs,
                target_index=df.index,
                neutral_value=0.5,
                threshold=config.get("binary_threshold", 0.5),
            )

            # Labels also need to be aligned to df.index
            full_labels = pd.Series(0, index=df.index)
            # y is indexed by t_events, which are part of anchor_df index
            # Align y to df.index
            y_aligned = y.reindex(df.index, method='ffill').fillna(0).astype(int)
            full_labels.update(y_aligned)

            # 6. Save (Clear heavy caches first to free memory for diagnostics)
            tprint_info(f"   [Phase 6/6] Clearing caches & saving results...")
            self._PHYSICS_CACHE.clear()
            self._ROUTER_CACHE.clear()
            self._ANARCHY_CACHE.clear()
            self._BAR_CACHE.clear()
            gc.collect()

            if hasattr(self, 'save_specialist_results'):
                result = self.save_specialist_results(
                    config=config,
                    feature_df=feature_df,
                    labels=full_labels,
                    predictions=final_preds.values,
                    probabilities=final_probs.values,
                    model=training_result.model,
                    metrics=training_result.metrics,
                    specialist_name=self.__class__.__name__
                )
            else:
                result = {"success": True, "metrics": training_result.metrics}

            return result

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.exception(f"❌ {self.step_name} failed: {e}")
            tprint_error(f"[{self.__class__.__name__}] execute_standard_specialist_logic failed: {e}")
            return {"success": False, "error": str(e)}

    def _align_probabilities_to_index(
        self,
        oof_probs: Optional[pd.Series],
        target_index: pd.Index,
        neutral_value: float = 0.5,
        threshold: float = 0.5,
    ) -> Tuple[pd.Series, pd.Series]:
        """Align OOF probabilities to the full market-data index without biasing direction."""
        if not isinstance(target_index, pd.Index):
            raise ValueError("target_index must be a pandas Index")

        if oof_probs is None:
            aligned_probs = pd.Series(neutral_value, index=target_index)
            binary = pd.Series(0, index=target_index, dtype=int)
            return aligned_probs, binary

        if not isinstance(oof_probs, pd.Series):
            oof_probs = pd.Series(oof_probs)

        aligned_probs = oof_probs.reindex(target_index)
        missing_mask = aligned_probs.isna()

        if missing_mask.all():
            aligned_probs = pd.Series(neutral_value, index=target_index)
            binary = pd.Series(0, index=target_index, dtype=int)
            return aligned_probs, binary

        aligned_probs = aligned_probs.fillna(neutral_value)
        binary = pd.Series(0, index=target_index, dtype=int)
        confident_mask = ~missing_mask
        symbol = config.get('symbol', 'BTCUSDT')
        exchange = config.get('exchange', 'binance')
        timeframe = config.get('timeframe', '15m')
        
        # [Fix] Explicitly set specialist context to avoid dumping into 'analyst' directory
        specialist_name = self.__class__.__name__
        # Clean up name: EnhancedMLMomentumPersistenceStep -> momentum_persistence
        clean_name = specialist_name.lower().replace('step', '').replace('enhancedml', '').replace('enhanced', '')
        if clean_name.startswith('_'): clean_name = clean_name[1:]
        
        self.set_context(
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            model=clean_name
        )
        
        cache_key = (symbol, exchange, timeframe)

        # 1. Load Market Data
        tprint_info(f"   [Phase 1/6] Loading Market Data...")
        if hasattr(self, '_load_market_data_with_cache'):
            df, market_source = self._load_market_data_with_cache(config, timeframe)
        elif hasattr(self, '_load_market_data'):
            df = self._load_market_data(symbol, exchange, timeframe)
            market_source = "loaded"
        self._anchor_type = config.get('anchor_type', self._get_anchor_type(specialist_type))
        # Use dollar as context for range, and range as context for dollar
        context_type = 'range' if self._anchor_type == 'dollar' else 'dollar'

        # 2. Generate Bars (with Caching)
        tprint_info(f"      [Sub-Phase 4.1] Generating {self._anchor_type} & {context_type} bars...")
        cache_key_anchor = (*cache_key_base, self._anchor_type)
        cache_key_anchor = (*cache_key_base, anchor_type)
        cache_key_context = (*cache_key_base, context_type)

        if cache_key_anchor in self._BAR_CACHE:
            tprint_info(f"      [Cache] Using cached anchor_df ({anchor_type})")
            anchor_df = self._BAR_CACHE[cache_key_anchor]
        else:
            anchor_df = self._generate_bars(market_data, config, anchor_type)
            self._BAR_CACHE[cache_key_anchor] = anchor_df

        if cache_key_context in self._BAR_CACHE:
            tprint_info(f"      [Cache] Using cached context_df ({context_type})")
            context_df = self._BAR_CACHE[cache_key_context]
        else:
            context_df = self._generate_bars(market_data, config, context_type)
            self._BAR_CACHE[cache_key_context] = context_df
        
        if anchor_df is None or anchor_df.empty:
            tprint_warning(f"      [Prepare] anchor_df ({anchor_type}) is empty. Falling back to time bars.")
            anchor_df = self._fallback_time_bars(market_data)
            if anchor_df is not None and not anchor_df.empty:
                self._BAR_CACHE[cache_key_anchor] = anchor_df
            else:
                tprint_error(f"      [Prepare] anchor_df ({anchor_type}) is empty after fallback!")
                return pd.DataFrame(), pd.Series(), pd.Series()

        min_anchor_rows = int(config.get('min_anchor_rows', 200))
        if len(anchor_df) < min_anchor_rows:
            tprint_error(f"      [Prepare] anchor_df has {len(anchor_df)} rows (< {min_anchor_rows}). Aborting.")
            return pd.DataFrame(), pd.Series(), pd.Series()
        
        tprint_info(f"      [Prepare] anchor_df: {len(anchor_df)} rows, context_df: {len(context_df) if context_df is not None else 0} rows")

        # 3. Join & Cross-Bar Features
        anchor_feats = anchor_df.add_suffix(f"_{anchor_type}_bar")
        if context_df is not None and not context_df.empty:
            context_feats = context_df.add_suffix(f"_{context_type}_bar")
            context_feats['context_ts_join'] = context_feats.index
            joined_df = pd.merge_asof(anchor_feats.sort_index(), context_feats.sort_index(), left_index=True, right_index=True, direction='backward')
            joined_df = joined_df.drop(columns=['context_ts_join'])
        else:
            joined_df = anchor_feats

        pipeline_feats = feature_df.reindex(anchor_df.index, method='ffill').fillna(method='bfill')
        X_combined = pd.concat([joined_df, pipeline_feats], axis=1)
        X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]
        market_data_anchor = anchor_df

        # 4. AFML Sampling & Labeling
        tprint_info(f"      [Sub-Phase 4.2] AFML CUSUM Sampling ({filter_type})...")
        sampled_df, t_events = self.apply_afml_sampling(market_data_anchor, config, filter_type=filter_type)
        if len(t_events) == 0:
            tprint_error(f"      [Prepare] t_events is empty after sampling!")
            return pd.DataFrame(), pd.Series(), pd.Series()

        # Debug logging for investigation
        tprint_info(f"   [Debug] t_events count: {len(t_events)}")
        tprint_info(f"   [Debug] anchor_df shape: {anchor_df.shape}, context_df shape: {context_df.shape if context_df is not None else None}")
        tprint_info(f"   [Debug] X_combined shape: {X_combined.shape}")

        min_event_samples = int(config.get('min_event_samples', 100))
        if len(t_events) < min_event_samples:
            tprint_error(f"      [Prepare] t_events has {len(t_events)} samples (< {min_event_samples}). Aborting.")
            return pd.DataFrame(), pd.Series(), pd.Series()
            
        tprint_info(f"      [Prepare] sampled events: {len(t_events)}")
        
        tprint_info(f"      [Sub-Phase 4.3] Generating TBM Labels...")
        pt_sl = config.get(pt_sl_config_key, default_pt_sl) if pt_sl_config_key else default_pt_sl
        tbm_labels_df = self.generate_tbm_labels(market_data_anchor, t_events, config, pt_sl)
        
        if tbm_labels_df.empty or 'bin' not in tbm_labels_df.columns:
            tprint_error(f"      [Prepare] tbm_labels_df is empty or missing 'bin'!")
            return pd.DataFrame(), pd.Series(), pd.Series()

        train_regime_label = config.get('train_regime_label', None)
        regime_at_event: Optional[pd.Series] = None

        # 5. Physics Router (Phase 1) (with Caching)
        tprint_info(f"      [Sub-Phase 4.4] Physics Router (Matrix Profile + Wavelet)...")
        # Physics features are tied to anchor_df (which is already specific to anchor_type)
        if cache_key_anchor in self._ROUTER_CACHE:
            tprint_info(f"      [Cache] Using cached fitted Router for {anchor_type}")
            router = self._ROUTER_CACHE[cache_key_anchor]
            physics_feats = self._PHYSICS_CACHE[cache_key_anchor]
        else:
            router = AdaptiveHunterRouter()
            physics_feats = router.compute_physics_features(market_data_anchor)
            if not physics_feats.empty:
                router.fit(physics_feats.values)
                self._ROUTER_CACHE[cache_key_anchor] = router
                self._PHYSICS_CACHE[cache_key_anchor] = physics_feats
            else:
                tprint_warning("      [Prepare] physics_feats is empty, router will use uniform weights")

        target_regime = config.get('target_regime', None) 
        regime_weights_series = pd.Series(1.0, index=t_events)
        
        if target_regime and not physics_feats.empty and getattr(router, 'is_fit', False):
            try:
                events_present = t_events.intersection(physics_feats.index)
                if not events_present.empty:
                    X_phys = physics_feats.loc[events_present].values
                    X_phys_scaled = router.scaler.transform(X_phys)
                    probs = router.gmm.predict_proba(X_phys_scaled)
                    rev_map = {v: k for k, v in router.regime_map.items()}
                    regime_idx = rev_map.get(target_regime, 0)
                    regime_weights_series.loc[events_present] = probs[:, regime_idx]
            except Exception as e:
                tprint_warning(f"   [Prepare] Regime weighting failed: {e}")

        if train_regime_label and not physics_feats.empty and getattr(router, 'is_fit', False):
            try:
                X_phys_all = physics_feats.values
                X_phys_scaled = router.scaler.transform(X_phys_all)
                cluster_ids = router.gmm.predict(X_phys_scaled)
                labels = [router.regime_map.get(int(i), str(i)) for i in cluster_ids]
                label_series = pd.Series(labels, index=physics_feats.index)
                regime_at_event = label_series.reindex(t_events, method='ffill')
            except Exception as e:
                tprint_warning(f"   [Prepare] Regime filter assignment failed: {e}")
                regime_at_event = None

        # 6. Wavelet Features & Hunter Weighting (Phase 0)
        anchor_wavelet_fam = 'db4' if anchor_type == 'dollar' else 'sym4'
        wavelet_vals = []
        close_vals = market_data_anchor['close'].values
        for evt in t_events:
            try:
                idx = market_data_anchor.index.get_loc(evt)
                if idx > 32:
                    s = close_vals[idx-32:idx]
                    feat = get_wavelet_features(s, wavelet=anchor_wavelet_fam)
                    wavelet_vals.append(feat['hf_lf_ratio'])
                else:
                    wavelet_vals.append(0.5)
            except Exception:
                wavelet_vals.append(0.5)

        hf_lf_series = pd.Series(wavelet_vals, index=t_events)
        noise_ratio = hf_lf_series / (1.0 + hf_lf_series) 

        volatility = get_daily_vol(market_data_anchor['close'], use_volume_time=True).loc[t_events]
        barrier_size = volatility * pt_sl[0]
        
        num_concurrent = self.get_concurrent_weights(tbm_labels_df['t1'], market_data_anchor.index)
        uniqueness = get_sample_uniqueness(tbm_labels_df['t1'], num_concurrent)

        base_weights = compute_hunter_weight(
            uniqueness=uniqueness.values,
            mfe=tbm_labels_df['mfe'].values,
            mae=tbm_labels_df['mae'].values,
            barrier=barrier_size.values,
            wavelet_noise=noise_ratio.values
        )

        final_weights = base_weights * regime_weights_series.loc[t_events].values
        weights_series = pd.Series(final_weights, index=t_events)

        # 7. Final Clean
        wavelet_df = pd.DataFrame({'wavelet_noise': noise_ratio}, index=t_events)
        X_sampled = pd.concat([X_combined.loc[t_events], wavelet_df], axis=1)

        X = X_sampled.select_dtypes(include=[np.number])
        valid_mask = X.notna().all(axis=1) & tbm_labels_df['bin'].notna()
        X, y, w = X.loc[valid_mask], tbm_labels_df['bin'].loc[valid_mask], weights_series.loc[valid_mask]

        if train_regime_label and regime_at_event is not None:
            try:
                r = regime_at_event.reindex(X.index)
                keep = r == str(train_regime_label)
                X, y, w = X.loc[keep], y.loc[keep], w.loc[keep]
            except Exception:
                pass
        
        tprint_info(f"   [Prepare] Final data: X={X.shape}, y={len(y)}, w={len(w)}")
        return X, y, w

    def _get_anchor_type(self, specialist_type: SpecialistType) -> str:
        tprint_info(f"[{self.__class__.__name__}] _get_anchor_type for {specialist_type.name}")
        # Phase 1: All volume-based specialists now use dollar bars
        volume_anchors = {
            SpecialistType.VOLUME_FORCE, SpecialistType.LIQUIDITY_REGIME, 
            SpecialistType.MICROSTRUCTURE, SpecialistType.RISK_REGIME, 
            SpecialistType.SMC_REGIME, SpecialistType.SPECTRAL
        }
        return 'dollar' if specialist_type in volume_anchors else 'range'

    def _generate_bars(self, df: pd.DataFrame, config: Dict[str, Any], bar_type: str) -> pd.DataFrame:
        tprint_info(f"[{self.__class__.__name__}] _generate_bars type={bar_type}")
        if bar_type == 'dollar': return self._generate_dynamic_dollar_bars(df, config)
        elif bar_type == 'range': return self._generate_range_bars(df, config)
        elif bar_type == 'pit': return self.generate_pit_bars(df, config)
        else: raise ValueError(f"Unknown bar type: {bar_type}")

    def _fallback_time_bars(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Fallback to time-based bars when specialized bar generation fails."""
        if df is None or df.empty:
            return None
        required_cols = {'open', 'high', 'low', 'close'}
        if not required_cols.issubset(df.columns):
            return None
        base_cols = ['open', 'high', 'low', 'close']
        if 'volume' in df.columns:
            base_cols.append('volume')
        fallback = df[base_cols].copy()
        if isinstance(fallback.index, pd.DatetimeIndex):
            fallback['bar_duration'] = fallback.index.to_series().diff().dt.total_seconds().fillna(60.0)
        return fallback

    def _generate_dynamic_dollar_bars(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Phase 1: Dynamic Dollar (USDT) bars (Vectorized).
        Threshold is adaptive based on 7-day rolling dollar volume.
        """
        tprint_info(f"[{self.__class__.__name__}] _generate_dynamic_dollar_bars (Vectorized) start")
        symbol = config.get('symbol', 'BTCUSDT')
        try:
            manager = get_klines_manager(config.get('data_dir', 'historical_data'))
            start_date = df.index.min() - pd.Timedelta(days=8)
            end_date = df.index.max()
            
            df_1m = manager.read_data(symbol, "1m", start_date, end_date, data_type="raw")
            if df_1m is None or df_1m.empty: return None
            
            # Ensure 'quote_volume' is present and valid
            # Ensure 'quote_volume' is present and valid (handle mostly empty columns)
            if 'quote_volume' not in df_1m.columns or df_1m['quote_volume'].count() < 0.5 * len(df_1m):
                df_1m['quote_volume'] = df_1m['volume'] * df_1m['close']
            
            df_1m['quote_volume'] = df_1m['quote_volume'].fillna(0.0)
            
            # Adaptive threshold using 30-day (monthly) rolling mean volume * 15 (for ~96 bars/day = 15 min avg)
            # Using 30-day rolling mean per user request for stability
            # 43200 = 30 days * 24 hours * 60 minutes
            rolling_30d_mean = df_1m['quote_volume'].rolling(window=43200, min_periods=1440).mean()
            
            # Threshold = rolling mean * 15 (minutes per bar at 96 bars/day)
            dynamic_threshold = (rolling_30d_mean * 15.0).ffill().bfill().values
            
            # Vectorized bar generation using Numba
            vols = df_1m['quote_volume'].values
            closes = df_1m['close'].values
            highs = df_1m['high'].values
            lows = df_1m['low'].values
            opens = df_1m['open'].values
            times = df_1m.index.values # datetime64[ns]

            # Use Numba-optimized generation
            out_times, out_opens, out_highs, out_lows, out_closes, out_vols = _numba_generate_dollar_bars(
                 times, opens, highs, lows, closes, vols, dynamic_threshold
            )
            
            if len(out_times) == 0:
                 return None

            res_df = pd.DataFrame({
                'open': out_opens, 'high': out_highs, 'low': out_lows, 
                'close': out_closes, 'volume': out_vols
            }, index=pd.DatetimeIndex(out_times))
            
            res_df['bar_duration'] = res_df.index.to_series().diff().dt.total_seconds().fillna(60.0)
            return res_df
        except Exception as e:
            tprint_error(f"   [DollarBars] Failed: {e}")
            return None

    def _generate_range_bars(self, df: pd.DataFrame, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Phase 1: Dynamic Range bars using volatility-adaptive thresholds (Optimized).
        """
        tprint_info(f"[{self.__class__.__name__}] _generate_range_bars start")
        symbol = config.get('symbol', 'BTCUSDT')
        try:
            vol_15m = calculate_rolling_volatility(df['close'], window_days=28)
            delta_p_series = calculate_dynamic_range_threshold(vol_15m, df['close'])
            delta_p_daily = delta_p_series.resample('1D').last().shift(1)

            manager = get_klines_manager(config.get('data_dir', 'historical_data'))
            start_date = df.index.min() - pd.Timedelta(days=8)
            end_date = df.index.max()
            df_1m = manager.read_data(symbol, "1m", start_date, end_date, data_type="raw")
            if df_1m is None: return None

            thresholds_1m = delta_p_daily.reindex(df_1m.index, method='ffill').bfill().values

            times = df_1m.index.values
            opens = df_1m['open'].values
            highs = df_1m['high'].values
            lows = df_1m['low'].values
            closes = df_1m['close'].values
            vols = df_1m['volume'].values
            
            # Use Numba-optimized generation
            out_times, out_opens, out_highs, out_lows, out_closes, out_vols, out_durations = _numba_generate_range_bars(
                times, opens, highs, lows, closes, vols, thresholds_1m
            )

            if len(out_times) == 0:
                 return None

            return pd.DataFrame({'open': out_opens, 'high': out_highs, 'low': out_lows, 'close': out_closes, 'volume': out_vols, 'bar_duration': out_durations}, index=pd.DatetimeIndex(out_times))
        except Exception as e:
            tprint_error(f"   [RangeBars] Failed: {e}")
            return None

    def apply_afml_sampling(self, df, config, filter_type='price'):
        tprint_info(f"[{self.__class__.__name__}] apply_afml_sampling start filter={filter_type}")
        
        use_volume_bars = config.get('use_volume_bars', True)

        # Select threshold base based on filter type
        if filter_type == 'volume' and 'volume' in df.columns:
            # For volume sampling, use rolling average volume as threshold base
            # Note: For Dollar Bars, volume is roughly constant, so this may degrade to time sampling
            threshold_base = df['volume'].rolling(window=100).mean()
            # If using dollar bars, volume is constant (e.g. 1M USDT). CUSUM on constant volume
            # accumulates linearly. This is equivalent to periodic sampling.
            # If this was "broken" (7%), switching to price/volatility fixes it by using Price CUSUM.
            tprint_info("      [Sampling] Using Volume CUSUM logic")
        else:
            # Default: Price/Volatility CUSUM (Standard AFML)
            threshold_base = get_daily_vol(df['close'], use_volume_time=use_volume_bars)
            tprint_info("      [Sampling] Using Volatility (Price) CUSUM logic")
        
        # Derive specialist identifiers for flexible config lookups
        # 1. Full step name (e.g. "enhanced_ml_smc_regime_step")
        step_name = getattr(self, 'step_name', self.__class__.__name__)

        # 2. Short name (e.g. "smc_regime" from "enhanced_ml_smc_regime_step")
        short_name = step_name.replace('enhanced_ml_', '').replace('enhanced_', '').replace('_step', '')

        # 3. Mashed name (legacy/fallback, e.g. "smcregime")
        class_name = self.__class__.__name__
        mashed_name = class_name.lower().replace('step', '').replace('enhancedml', '').replace('enhanced', '')
        if mashed_name.startswith('_'): mashed_name = mashed_name[1:]

        # Config lookup with priority: step_name > short_name > mashed_name > global default

        # Target Sampling Rate
        default_rate = config.get('afml_target_sampling_rate', 0.20)
        target_rate = config.get(f"{step_name}_target_sampling_rate",
                        config.get(f"{short_name}_target_sampling_rate",
                            config.get(f"{mashed_name}_target_sampling_rate", default_rate)))

        # Minimum Event Samples
        default_min_samples = config.get('min_event_samples', 100)
        min_samples = int(config.get(f"{step_name}_min_event_samples",
                            config.get(f"{short_name}_min_event_samples",
                                config.get(f"{mashed_name}_min_event_samples", default_min_samples))))

        tprint_info(f"      [Sampling Config] {short_name}: target_rate={target_rate}, min_samples={min_samples}")

        # Ensure target count respects minimum samples
        raw_target = int(len(df) * target_rate)
        target_count = min(len(df), max(raw_target, min_samples))

        low, high = 1e-8, 1.0
        threshold_base = threshold_base.fillna(method='bfill').fillna(method='ffill')
        
        best_events = df.index[::5] # Default fallback 20%
        best_diff = float('inf')
        
        # Optimization: Reduce search iterations and exit early if within 5% of target count
        if len(df) > 10:
            for i in range(8):
                mid = (low + high) / 2

                # Apply filter logic
                if filter_type == 'volume' and 'volume' in df.columns:
                    # Volume CUSUM: accum = max(0, accum + vol - threshold)
                    # This is slightly different from standard Symmetric CUSUM on price
                    # Simplified: Trigger when cumulative volume exceeds threshold
                    # Since we don't have a volume-specific CUSUM utility in afml_utils yet,
                    # we can simulate it or use get_t_events if we treat volume as the series.
                    # But get_t_events does pct_change().
                    # Volume CUSUM usually means: sum(volume) >= threshold.
                    # Which is just re-sampling bars.
                    # Here we treat 'volume' filter as: "Sample when cumulative volume deviation is high"?
                    # Or just use the original get_t_events on volume?
                    # get_t_events(volume) -> volume.pct_change() CUSUM.
                    # This detects "Volume Surges".
                    t_events = get_t_events(df['volume'], threshold_base * mid)
                else:
                    # Standard Price CUSUM
                    t_events = get_t_events(df['close'], threshold_base * mid)

                count = len(t_events)
                diff = abs(count - target_count)
                
                # Update best_events using closest match strategy
                if diff < best_diff:
                    best_diff = diff
                    best_events = t_events
                    # tprint_info(f"      [Sampling] New best: {count} events (diff={diff})")

                if count > target_count:
                    low = mid
                else:
                    high = mid
                
                # Early exit if within 5% of target
                if abs(count - target_count) / max(1, target_count) < 0.05:
                    tprint_info(f"      [Sampling] Early exit: within 5% of target ({count}/{target_count})")
                    best_events = t_events
                    break
        
        # Fast fail for regime sparsity if optimized result is still too sparse
        if len(best_events) < min_samples:
            msg = f"Regime sparsity detected: Optimized events {len(best_events)} < {min_samples}. Fast failing."
            tprint_error(f"      [Sampling] {msg}")
            raise ValueError(msg)

        tprint_info(f"[{self.__class__.__name__}] apply_afml_sampling finished with {len(best_events)} events")
        return df.loc[best_events], best_events

    def generate_tbm_labels(self, df, t_events, config, pt_sl):
        tprint_info(f"[{self.__class__.__name__}] generate_tbm_labels start events={len(t_events)}")
        # Delegate to utility
        vol = get_daily_vol(df['close'], use_volume_time=config.get('use_volume_bars', True))
        vb = get_vertical_barrier(df['close'], t_events, config.get('lookforward_bars', 35))
        tbm = apply_triple_barrier(df['close'], t_events, pt_sl, vol, config.get('min_ret', 0.001), vb)
        return get_bins(tbm, df['close'])

    def get_concurrent_weights(self, t1, idx):
        tprint_info(f"[{self.__class__.__name__}] get_concurrent_weights start samples={len(t1)}")
        return get_weights_by_uniqueness(t1, idx)
