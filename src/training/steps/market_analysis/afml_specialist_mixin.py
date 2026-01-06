import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from sklearn.metrics import roc_auc_score

from src.utils.ml_common.afml_utils import (
    get_daily_vol, get_t_events, get_vertical_barrier, 
    apply_triple_barrier, get_bins, get_weights_by_uniqueness,
    frac_diff_fixed, get_sample_weights
)
from src.utils.tprint import tprint_info, tprint_warning, tprint_success, tprint
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_xgb_with_oof

logger = logging.getLogger(__name__)

class AFMLSpecialistMixin:
    """
    Mixin providing AFML-specific logic for enhanced specialists:
    - CUSUM Filtering for event-based sampling
    - Triple Barrier Method for path-aware labeling
    - Sample Uniqueness weighting to handle overlap (Concurrence)
    - Fractional Differentiation for memory preservation
    - Standardized Feature Generation and Execution Flow
    """

    def _apply_standard_feature_selection(self, features: pd.DataFrame, max_features: int = 30) -> pd.DataFrame:
        """
        Standard manual feature selection to reduce redundancy and keep high-quality features.
        """
        if features.empty:
            return features

        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)

        # Manual redundancy reduction - remove highly correlated features
        correlation_matrix = features.corr().abs()
        upper_triangle = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )

        # Find highly correlated pairs (>0.9)
        to_drop = []
        for column in upper_triangle.columns:
            correlated_features = upper_triangle[column][upper_triangle[column] > 0.9]
            if not correlated_features.empty:
                for correlated_feature in correlated_features.index:
                    if correlated_feature > column:
                        to_drop.append(correlated_feature)

        # Remove redundant features
        if to_drop:
            features = features.drop(columns=list(set(to_drop)))

        # Keep only the most informative features by variance
        if len(features.columns) > max_features:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(max_features).index
            features = features[top_features]

        return features

    def generate_standard_enhanced_features(
        self,
        df: pd.DataFrame,
        specialist_type: SpecialistType,
        manual_feature_func: Optional[Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = None
    ) -> pd.DataFrame:
        """
        Standardized interface for feature generation:
        1. Pipeline Features (MIOptimizedFeaturePipeline)
        2. Manual Features (Override)
        3. Redundancy Reduction
        """
        # 1. Pipeline Features
        pipeline_features = pd.DataFrame(index=df.index)
        if hasattr(self, 'feature_pipeline'):
            pipeline_features = self.feature_pipeline.generate_enhanced_features(
                df, specialist_type.value, {'enhanced_features': True}
            )
        else:
            logger.warning("No feature_pipeline found in self. Skipping pipeline features.")

        # 2. Manual Features
        manual_features = pd.DataFrame(index=df.index)
        if manual_feature_func:
            manual_features = manual_feature_func(df, pipeline_features)

        # 3. Combine
        combined_features = pd.concat([pipeline_features, manual_features], axis=1)

        # Remove duplicates
        combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]

        # 4. Standard Selection/Reduction
        selected_features = self._apply_standard_feature_selection(combined_features)

        # Clean infinities and NaNs
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
        """
        Execute the standard specialist workflow:
        1. Load Market Data
        2. Generate Features (Pipeline + Manual)
        3. Prepare AFML Data (CUSUM, TBM, Weights)
        4. Train Model (XGB + OOF)
        5. Save Results & Diagnostics
        """
        start_time = time.time()
        try:
            symbol = config.get('symbol', 'BTCUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'long')

            if hasattr(self, 'set_context'):
                self.set_context(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    direction=direction,
                    model=self.step_name
                )

            # Initialize versioned store if available
            if hasattr(self, 'versioned_store'):
                _ = self.versioned_store

            # 1. Load Market Data
            tprint_info(f"🚀 Starting Enhanced {self.step_name} for {symbol} on {exchange}")
            # Use _load_market_data_with_cache if available (some specialists define it), else generic
            if hasattr(self, '_load_market_data_with_cache'):
                df, market_source = self._load_market_data_with_cache(config, timeframe)
            elif hasattr(self, '_load_market_data'):
                df = self._load_market_data(symbol, exchange, timeframe)
                market_source = "loaded"
            else:
                 # Fallback to BaseStep logic if exposed, or raise error
                raise NotImplementedError("Specialist must implement _load_market_data or _load_market_data_with_cache")

            if df is None or len(df) < 1000:
                return {"success": False, "error": "Insufficient data"}

            # 2. Feature Generation
            tprint_info(f"🛠️ Generating standard enhanced features for {specialist_type.value}...")
            feature_df = self.generate_standard_enhanced_features(df, specialist_type, manual_feature_func)
            tprint_info(f"✅ Features generated: {len(feature_df.columns)} columns")

            # Save feature artifact if not batch run
            if not config.get("is_batch_run", False) and hasattr(self, '_save_artifact'):
                feature_df_reset = feature_df.reset_index().rename(columns={feature_df.index.name or "index": "timestamp"})
                self._save_artifact(
                    data=feature_df_reset,
                    artifact_name=f"{self.step_name}_{suffix}",
                    artifact_type="data",
                    data_category="features",
                    metadata={"source": market_source, "enhanced": True}
                )

            # 3. AFML: Sampling, Labeling, Weighting, Alignment
            X, y, weights = self.prepare_specialist_data(
                market_data=df,
                feature_df=feature_df,
                config=config,
                filter_type=filter_type,
                pt_sl_config_key=pt_sl_config_key,
                default_pt_sl=default_pt_sl
            )

            # 4. Centralized purged-CV training
            tprint_info("🤖 Training model with centralized XGB helper (purged CV & AFML weights)...")
            training_result = train_specialist_xgb_with_oof(
                X.fillna(0.0),
                y.fillna(0.0),
                sample_weight=weights,
                n_splits=5,
            )

            oof_probs = training_result.oof_predictions
            last_model = training_result.model
            metrics = training_result.metrics

            # 5. Align results back to full market index
            # Initialize with NaN
            final_probs = pd.Series(np.nan, index=df.index)

            valid_oof = oof_probs.dropna()
            if len(valid_oof) > 0:
                final_probs.loc[valid_oof.index] = valid_oof.values

            # Ffill probabilities
            final_probs = final_probs.ffill().fillna(0.5)
            final_preds = (final_probs >= 0.5).astype(int)

            full_labels = pd.Series(0, index=df.index)
            full_labels.loc[y.index] = y

            # 6. Save Results & Diagnostics
            # This handles standardized output, artifacts, and diagnostics via SpecialistDiagnosticsMixinEnhancedV2
            if hasattr(self, 'save_specialist_results'):
                result = self.save_specialist_results(
                    config=config,
                    feature_df=feature_df,
                    labels=full_labels,
                    predictions=final_preds.values,
                    probabilities=final_probs.values,
                    model=last_model,
                    metrics=metrics,
                    specialist_name=self.__class__.__name__
                )
            else:
                # Fallback if mixin not present
                result = {"success": True, "metrics": metrics}

            execution_time = time.time() - start_time
            result["execution_time"] = execution_time
            tprint_success(f"✅ {self.step_name} completed in {execution_time:.2f}s")

            return result

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.exception(f"❌ {self.step_name} failed: {e}")
            return {"success": False, "error": str(e)}

    def prepare_specialist_data(
        self,
        market_data: pd.DataFrame,
        feature_df: pd.DataFrame,
        config: Dict[str, Any],
        filter_type: str = 'price',
        pt_sl_config_key: Optional[str] = None,
        default_pt_sl: List[float] = [2.0, 1.0]
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Execute the full AFML pipeline from raw features to weighted training sets.

        Sequence:
        1. CUSUM Sampling (apply_afml_sampling)
        2. Triple Barrier Labeling (generate_tbm_labels)
        3. Concurrence Weighting (get_concurrent_weights)
        4. Sample Weighting (get_sample_weights)
        5. Data Alignment & Cleaning

        Args:
            market_data: Raw market data (OHLCV)
            feature_df: Generated features dataframe
            config: Configuration dictionary
            filter_type: Type of CUSUM filter ('price', 'volatility', 'volume', 'spread')
            pt_sl_config_key: Config key to look up PT/SL thresholds (e.g., 'spectral_pt_sl')
            default_pt_sl: Default PT/SL thresholds if key not found

        Returns:
            Tuple[X, y, weights] aligned and cleaned
        """
        # 1. AFML: CUSUM Sampling
        tprint_info(f"🎯 Applying AFML CUSUM sampling (10% target) using {filter_type} filter...")
        sampled_df, t_events = self.apply_afml_sampling(market_data, config, filter_type=filter_type)

        # 2. AFML: Triple Barrier Labels
        pt_sl = config.get(pt_sl_config_key, default_pt_sl) if pt_sl_config_key else default_pt_sl
        tprint_info(f"🏷️ Generating TBM labels with PT/SL: {pt_sl}")
        tbm_labels_df = self.generate_tbm_labels(market_data, t_events, config, pt_sl)

        # 3. AFML: Alignment and Uniqueness Weighting
        X_sampled = feature_df.loc[t_events]
        y_sampled = tbm_labels_df['bin']
        t1_sampled = tbm_labels_df['t1']
        ret_sampled = tbm_labels_df['ret']

        # 4. AFML Hardening: Sample Weighting (u_bar * |return|)
        num_concurrent = self.get_concurrent_weights(t1_sampled, market_data.index)
        weights_sampled = get_sample_weights(t1_sampled, num_concurrent, ret_sampled)

        # 5. Filter numeric and drop NaNs
        X = X_sampled.select_dtypes(include=[np.number])
        valid_mask = X.notna().all(axis=1) & y_sampled.notna()
        X, y, weights = X.loc[valid_mask], y_sampled.loc[valid_mask], weights_sampled.loc[valid_mask]

        if len(X) < 100:
            tprint_warning(f"⚠️ Low sample count after AFML filtering: {len(X)}")

        tprint_info(f"📊 Training Data (AFML Sampled): {len(X)} samples, {len(X.columns)} features")

        return X, y, weights
    
    def apply_afml_sampling(self, df: pd.DataFrame, config: Dict[str, Any], filter_type: str = 'price') -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        """
        Apply CUSUM filtering to sample meaningful events, targeting ~10% of total bars.
        Uses an enhanced binary search to find the optimal threshold factor.
        Supports: 'price', 'volatility', 'volume', 'spread'
        """
        if filter_type == 'price':
            series = df['close']
            threshold_base = get_daily_vol(series)
        elif filter_type == 'volatility':
            # Log-volatility returns
            vol = df['close'].pct_change().rolling(20).std().fillna(0)
            # Use small epsilon to avoid log(0)
            series = np.log((vol + 1e-9) / (vol.shift(1) + 1e-9)).fillna(0)
            threshold_base = series.rolling(100).std()
        elif filter_type == 'volume':
            volume = df.get('volume', pd.Series(1, index=df.index))
            log_vol = np.log1p(volume)
            series = log_vol.diff().fillna(0)
            threshold_base = series.rolling(100).std()
        elif filter_type == 'spread':
            # Use high-low range as spread proxy
            spread = (df['high'] - df['low']) / (df['close'] + 1e-8)
            series = spread.diff().fillna(0)
            threshold_base = series.rolling(100).std()
        else:
            series = df['close']
            threshold_base = get_daily_vol(series)
            
        threshold_base = threshold_base.fillna(method='bfill').fillna(method='ffill')
        
        # Binary search for threshold_factor to target ~10% sampling rate
        target_rate = config.get('afml_target_sampling_rate', 0.10)
        target_count = int(len(df) * target_rate)
        
        # Wide search range [1e-6, 1e6] for extreme series like spread or volatility
        low, high = 1e-6, 1000000.0
        best_factor = 1.0
        best_events = df.index
        min_diff = float('inf')
        
        # 25 steps of binary search for high precision across wide range
        for _ in range(25):
            mid = (low + high) / 2
            t_events = get_t_events(series, threshold_base * mid)
            count = len(t_events)
            
            if count == 0:
                # Too high, lower it
                high = mid
                continue
                
            if abs(count - target_count) < min_diff:
                min_diff = abs(count - target_count)
                best_factor = mid
                best_events = t_events
                
            if count > target_count:
                # Too many events, increase threshold
                low = mid
            else:
                # Too few events, decrease threshold
                high = mid
                
        t_events = best_events
        if len(t_events) == 0:
            # Emergency fallback if search failed
            t_events = df.index[::int(1/target_rate)]
            logger.warning(f"AFML {filter_type} Sampling FAILED to find threshold. Using periodic fallback.")
            
        logger.info(f"AFML {filter_type.capitalize()} Sampling: {len(t_events)} events (Rate: {len(t_events)/len(df):.1%}, Target: {target_rate:.1%}, Factor: {best_factor:.6f})")
        
        return df.loc[t_events], t_events

    def apply_sequential_bootstrap(self, t1: pd.Series, close_index: pd.Index, num_samples: Optional[int] = None) -> List[pd.Timestamp]:
        """Apply sequential bootstrapping to get robust non-overlapping samples."""
        from src.utils.ml_common.afml_utils import seq_bootstrap
        return seq_bootstrap(t1, close_index, num_samples)

    def generate_tbm_labels(self, df: pd.DataFrame, t_events: pd.DatetimeIndex, 
                           config: Dict[str, Any], pt_sl: List[float]) -> pd.DataFrame:
        """Generate Triple Barrier Method labels."""
        close = df['close']
        vol = get_daily_vol(close)
        vol = vol.fillna(method='bfill').fillna(method='ffill')
        
        lookforward = config.get('lookforward_bars', 35)
        vertical_barrier = get_vertical_barrier(close, t_events, lookforward)
        
        # Apply Triple Barrier
        tbm_events = apply_triple_barrier(
            close=close,
            t_events=t_events,
            pt_sl=pt_sl,
            target=vol,
            min_ret=config.get('min_ret', 0.001),
            vertical_barrier=vertical_barrier
        )
        
        labels_df = get_bins(tbm_events, close)
        return labels_df

    def get_concurrent_weights(self, t1: pd.Series, close_index: pd.Index) -> pd.Series:
        """Calculate sample weights based on average uniqueness (Concurrence fix)."""
        weights = get_weights_by_uniqueness(t1, close_index)
        return weights

    def apply_fractional_diff(self, series: pd.Series, d: float = 0.5) -> pd.Series:
        """Apply fractional differentiation to preserve memory in non-stationary series."""
        return frac_diff_fixed(series, d)

    def compute_binned_mi(self, x: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
        """Compute a fast binned MI score for reporting."""
        try:
            from sklearn.metrics import mutual_info_score
            if len(x) < 2 or len(np.unique(y)) < 2:
                return 0.0
            # Clean data
            mask = ~(np.isnan(x) | np.isnan(y) | np.isinf(x) | np.isinf(y))
            x_c, y_c = x[mask], y[mask]
            if len(x_c) < 2:
                return 0.0
            
            # Bin continuous x
            x_edges = np.histogram_bin_edges(x_c, bins=bins)
            x_binned = np.digitize(x_c, x_edges)
            
            return float(mutual_info_score(x_binned, y_c))
        except Exception:
            return 0.0
