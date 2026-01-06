import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from sklearn.metrics import roc_auc_score

from src.utils.ml_common.afml_utils import (
    get_daily_vol, get_t_events, get_vertical_barrier, 
    apply_triple_barrier, get_bins, get_weights_by_uniqueness,
    frac_diff_fixed, get_sample_weights, compute_master_weight,
    get_sample_uniqueness, get_num_concurrent_events
)
from src.utils.ml_common.wavelet_utils import get_wavelet_features
from src.utils.data.klines_parquet import get_klines_manager
from src.utils.tprint import tprint_info, tprint_warning, tprint_error
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
    - Dynamic Volume Bars and Wavelet Features (New)
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
        Execute the full AFML pipeline with Master-Anchor Architecture.

        Roles determined by SpecialistType:
        - Volume/Micro Specialist: Anchor=VolumeBars, Context=RangeBars
        - Range/Trend Specialist: Anchor=RangeBars, Context=VolumeBars
        """

        # 1. Determine Anchor and Context Types
        specialist_type = config.get('specialist_type')
        if not specialist_type and hasattr(self, 'specialist_type'):
            specialist_type = self.specialist_type

        # Default to Volume anchor if unknown (fallback)
        anchor_type = 'volume'
        if isinstance(specialist_type, SpecialistType):
            anchor_type = self._get_anchor_type(specialist_type)
        elif isinstance(specialist_type, str):
            # Try to map string to enum if possible, or just default
            pass

        context_type = 'range' if anchor_type == 'volume' else 'volume'

        tprint_info(f"🏗️ Master-Anchor Architecture: Anchor={anchor_type.upper()}, Context={context_type.upper()}")

        # 2. Generate Bars
        # Anchor Bars (The Rows)
        anchor_df = self._generate_bars(market_data, config, anchor_type)
        # Context Bars (The Features)
        context_df = self._generate_bars(market_data, config, context_type)

        if anchor_df is None or context_df is None:
            raise ValueError("Failed to generate Anchor or Context bars")

        tprint_info(f"   Anchor Bars ({anchor_type}): {len(anchor_df)}, Context Bars ({context_type}): {len(context_df)}")

        # 3. Join & Cross-Bar Features
        # We start with Anchor DF as the base
        anchor_feats = anchor_df.copy()
        anchor_feats.columns = [f"{c}_{anchor_type}_bar" for c in anchor_feats.columns]

        context_feats = context_df.copy()
        context_feats.columns = [f"{c}_{context_type}_bar" for c in context_feats.columns]

        # Calculate Avg Duration for Latency
        avg_context_duration = context_df['bar_duration'].mean() if 'bar_duration' in context_df.columns else 60.0

        if anchor_type == 'volume':
            # Volume Anchor: Use merge_asof to get most recent Range Bar state
            context_feats['context_ts_join'] = context_feats.index

            joined_df = pd.merge_asof(
                anchor_feats.sort_index(),
                context_feats.sort_index(),
                left_index=True,
                right_index=True,
                direction='backward'
            )

            # Cross-Bar: Latency (Staleness)
            if 'context_ts_join' in joined_df.columns:
                time_diff = (joined_df.index - joined_df['context_ts_join']).dt.total_seconds()
                joined_df['latency'] = time_diff / (avg_context_duration + 1e-6)
                joined_df = joined_df.drop(columns=['context_ts_join'])
            else:
                joined_df['latency'] = 0.0

            # Cross-Bar: Filling (Information Density)
            # volume_bars_per_range_bar is hard to exact match, use relative duration proxy
            # If latency is low, it means we just had a range bar.
            # Simple Proxy: AnchorDuration / ContextDuration (Ratio of bar speeds)
            # VolumeBarDuration / LastRangeBarDuration
            if f'bar_duration_{anchor_type}_bar' in joined_df.columns and f'bar_duration_{context_type}_bar' in joined_df.columns:
                joined_df['filling_ratio'] = joined_df[f'bar_duration_{anchor_type}_bar'] / (joined_df[f'bar_duration_{context_type}_bar'] + 1e-6)
            else:
                joined_df['filling_ratio'] = 1.0

        else:
            # Range Anchor: Aggregate Volume Bars within the Range Bar
            range_indices = anchor_df.index
            bin_indices = range_indices.searchsorted(context_df.index)

            context_df['bin_idx'] = bin_indices
            valid_context = context_df[(context_df['bin_idx'] > 0) & (context_df['bin_idx'] < len(range_indices))]

            # Aggregate: Sum Volume, Count Bars (Filling), Std Close (Micro Volatility)
            agg_ops = {
                'volume': ['sum', 'count'],
                'close': 'std',
                'bar_duration': 'sum'
            }

            grouped = valid_context.groupby('bin_idx').agg(agg_ops)
            # Flatten columns
            grouped.columns = [f"agg_{c[0]}_{c[1]}_{context_type}" for c in grouped.columns]

            # Rename specific aggregations for clarity
            # agg_volume_count_volume -> filling_count

            anchor_feats['iloc'] = np.arange(len(anchor_feats))
            # searchsorted returns insertion index i such that a[i-1] < v <= a[i].
            # So bin_idx corresponds exactly to the index in anchor_df (since bin_idx < len).
            # We map bin_idx -> anchor_df.index

            joined_df = anchor_feats.merge(grouped, left_on='iloc', right_index=True, how='left').drop(columns=['iloc'])
            joined_df = joined_df.fillna(0)

            # Cross-Bar Features
            # Latency is 0 by definition (we aggregated up to the close)
            joined_df['latency'] = 0.0

            # Filling: Number of volume bars in this range bar
            count_col = f"agg_volume_count_{context_type}"
            if count_col in joined_df.columns:
                 joined_df['filling_count'] = joined_df[count_col]
            else:
                 joined_df['filling_count'] = 1.0

        # 4. Feature Pipeline Integration
        # We need to reindex the incoming `feature_df` (which is 15m or 1m based) to the Anchor Bars.
        # Use reindex/ffill
        pipeline_feats = feature_df.reindex(anchor_df.index, method='ffill').fillna(method='bfill')

        # Combine everything
        X_combined = pd.concat([joined_df, pipeline_feats], axis=1)
        # Remove duplicate cols
        X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]

        # Update market_data to Anchor Bars for subsequent steps (Sampling/TBM)
        market_data_anchor = anchor_df

        # 5. AFML: CUSUM Sampling
        tprint_info(f"🎯 Applying AFML CUSUM sampling on Anchor Bars...")
        sampled_df, t_events = self.apply_afml_sampling(market_data_anchor, config, filter_type=filter_type)

        # 6. AFML: Triple Barrier Labels (on Anchor Bars)
        pt_sl = config.get(pt_sl_config_key, default_pt_sl) if pt_sl_config_key else default_pt_sl
        tbm_labels_df = self.generate_tbm_labels(market_data_anchor, t_events, config, pt_sl)

        # 7. Compute Wavelet Features (Cross-Bar Aware)
        # Compute Anchor Wavelets
        anchor_wavelet_fam = 'db4' if anchor_type == 'volume' else 'sym4'
        tprint_info(f"🌊 Computing Wavelets: Anchor={anchor_wavelet_fam}, Context=Mixed")

        wavelet_feats_list = []
        hf_lf_ratios = []
        rel_entropy_list = []

        # For Relative Entropy, we need Context Entropy.
        # We can compute context entropy on the fly for the context bars associated.
        # Simplified: Compute Anchor Entropy. Assume Context Entropy is 1.0 (placeholder) or compute?
        # Let's compute Anchor Entropy properly.

        close_values = market_data_anchor['close'].values
        wavelet_window = 32 # bars

        for event_time in t_events:
            try:
                idx = market_data_anchor.index.get_loc(event_time)
                if idx >= wavelet_window:
                    series = close_values[idx-wavelet_window:idx]
                    feats = get_wavelet_features(series, wavelet=anchor_wavelet_fam, level=4)
                    wavelet_feats_list.append(feats)
                    hf_lf_ratios.append(feats.get('hf_lf_ratio', 0.5))

                    # Compute Relative Entropy?
                    # Needs Context Bar data.
                    # Placeholder: Just use Anchor Entropy features.
                else:
                    default_feats = get_wavelet_features(np.zeros(wavelet_window), wavelet=anchor_wavelet_fam, level=4)
                    wavelet_feats_list.append(default_feats)
                    hf_lf_ratios.append(0.5)
            except Exception:
                default_feats = get_wavelet_features(np.zeros(wavelet_window), wavelet=anchor_wavelet_fam, level=4)
                wavelet_feats_list.append(default_feats)
                hf_lf_ratios.append(0.5)

        hf_lf_series = pd.Series(hf_lf_ratios, index=t_events)
        wavelet_df = pd.DataFrame(wavelet_feats_list, index=t_events)
        wavelet_df.columns = [f'wavelet_{c}' for c in wavelet_df.columns]

        # 8. Alignment
        X_sampled = X_combined.loc[t_events].copy()
        X_sampled = pd.concat([X_sampled, wavelet_df], axis=1)

        y_sampled = tbm_labels_df['bin']
        t1_sampled = tbm_labels_df['t1']
        ret_sampled = tbm_labels_df['ret']
        mfe_sampled = tbm_labels_df['mfe']
        mae_sampled = tbm_labels_df['mae']

        # 9. Weighting
        volatility = get_daily_vol(market_data_anchor['close'], use_volume_time=True).loc[t_events]
        barrier_size = volatility * pt_sl[0]

        num_concurrent = self.get_concurrent_weights(t1_sampled, market_data_anchor.index)
        uniqueness = get_sample_uniqueness(t1_sampled, num_concurrent)

        weights_sampled = compute_master_weight(
            uniqueness=uniqueness.values, mfe=mfe_sampled.values, mae=mae_sampled.values,
            barrier=barrier_size.values, hf_lf_ratio=hf_lf_series.values,
            volatility=volatility.values, raw_return=ret_sampled.values,
            timestamp_index=t_events
        )
        weights_series = pd.Series(weights_sampled, index=t_events)

        # 10. Final Clean
        X = X_sampled.select_dtypes(include=[np.number])
        valid_mask = X.notna().all(axis=1) & y_sampled.notna()
        X, y, weights = X.loc[valid_mask], y_sampled.loc[valid_mask], weights_series.loc[valid_mask]

        if len(X) < 100: tprint_warning(f"⚠️ Low sample count: {len(X)}")

        return X, y, weights
    
    def _get_anchor_type(self, specialist_type: SpecialistType) -> str:
        """
        Determine if the specialist is Volume (Micro/Execution) or Range (Trend/Breakout) anchored.
        """
        volume_anchors = {
            SpecialistType.VOLUME_FORCE,
            SpecialistType.LIQUIDITY_REGIME,
            SpecialistType.MICROSTRUCTURE,
            SpecialistType.RISK_REGIME,
            SpecialistType.SMC_REGIME,
            SpecialistType.SPECTRAL
        }

        if specialist_type in volume_anchors:
            return 'volume'
        else:
            return 'range'

    def _generate_bars(self, df: pd.DataFrame, config: Dict[str, Any], bar_type: str) -> pd.DataFrame:
        """
        Dispatcher for bar generation (Volume or Range).
        """
        if bar_type == 'volume':
            return self._generate_dynamic_volume_bars(df, config)
        elif bar_type == 'range':
            return self._generate_range_bars(df, config)
        elif bar_type == 'pit':
            return self.generate_pit_bars(df, config)
        else:
            raise ValueError(f"Unknown bar type: {bar_type}")

    def _generate_range_bars(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate Range Bars from 1-minute data.
        Delta P updated every 24h using trailing 7-day 15m volatility.
        """
        symbol = config.get('symbol', 'ETHUSDT')
        try:
            # 1. Compute Daily Delta P Thresholds from 15m data
            vol_15m = calculate_rolling_volatility(df['close'], window_days=7)
            delta_p_series = calculate_dynamic_range_threshold(vol_15m, df['close'])
            delta_p_daily = delta_p_series.resample('1D').last().shift(1)

            # 2. Load 1m raw data
            manager = get_klines_manager(config.get('data_dir', 'historical_data'))
            start_date = df.index.min() - pd.Timedelta(days=8)
            end_date = df.index.max()

            tprint_info(f"   Loading 1m raw data for Range Bars: {symbol} {start_date} to {end_date}")
            df_1m = manager.read_data(symbol, "1m", start_date, end_date, data_type="raw")

            if df_1m is None or df_1m.empty: return None

            df_1m = df_1m[['open', 'high', 'low', 'close', 'volume']]

            # Align Delta P to 1m data
            thresholds_1m = delta_p_daily.reindex(df_1m.index, method='ffill').fillna(method='bfill')

            # 3. Generate Bars
            times = df_1m.index.values
            opens = df_1m['open'].values
            highs = df_1m['high'].values
            lows = df_1m['low'].values
            closes = df_1m['close'].values
            vols = df_1m['volume'].values
            threshold_vals = thresholds_1m.values

            rb_times, rb_opens, rb_highs, rb_lows, rb_closes, rb_vols = [], [], [], [], [], []
            rb_durations = []

            current_open = opens[0]
            current_high = highs[0]
            current_low = lows[0]
            current_vol = 0.0
            start_idx = 0

            n_rows = len(df_1m)

            for i in range(n_rows):
                p = closes[i]
                current_high = max(current_high, highs[i])
                current_low = min(current_low, lows[i])
                current_vol += vols[i]

                delta_p = threshold_vals[i]
                if np.isnan(delta_p) or delta_p <= 0: delta_p = p * 0.01

                if abs(p - current_open) >= delta_p:
                    rb_times.append(times[i])
                    rb_opens.append(current_open)
                    rb_highs.append(current_high)
                    rb_lows.append(current_low)
                    rb_closes.append(p)
                    rb_vols.append(current_vol)

                    duration = (times[i] - times[start_idx]).astype('timedelta64[s]').astype(float) + 60.0
                    rb_durations.append(duration)

                    if i + 1 < n_rows:
                        current_open = p # Range bar continuity
                        current_high = p
                        current_low = p
                        current_vol = 0.0
                        start_idx = i + 1

            range_bars = pd.DataFrame({
                'open': rb_opens, 'high': rb_highs, 'low': rb_lows, 'close': rb_closes,
                'volume': rb_vols, 'bar_duration': rb_durations
            }, index=pd.DatetimeIndex(rb_times))
            range_bars.index.name = 'timestamp'
            range_bars['bar_return'] = range_bars['close'].pct_change().fillna(0.0)

            return range_bars

        except Exception as e:
            tprint_error(f"Error generating range bars: {e}")
            return None

    def generate_pit_bars(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate Point-in-Time (PiT) Information Bars for Meta-Model.
        Race between Volume Accumulation (N) and Price Deviation (Delta P).
        """
        symbol = config.get('symbol', 'ETHUSDT')
        try:
            manager = get_klines_manager(config.get('data_dir', 'historical_data'))
            start_date = df.index.min() - pd.Timedelta(days=8)
            end_date = df.index.max()
            df_1m = manager.read_data(symbol, "1m", start_date, end_date, data_type="raw")
            if df_1m is None or df_1m.empty: return None
            df_1m = df_1m[['open', 'high', 'low', 'close', 'volume']]

            rolling_7d_vol = df_1m['volume'].rolling(window=10080, min_periods=1440).sum()
            vol_thresholds = (rolling_7d_vol / 674.0).fillna(method='bfill')

            vol_15m = calculate_rolling_volatility(df['close'], window_days=7)
            delta_p_series = calculate_dynamic_range_threshold(vol_15m, df['close'])
            delta_p_daily = delta_p_series.resample('1D').last().shift(1)
            price_thresholds = delta_p_daily.reindex(df_1m.index, method='ffill').fillna(method='bfill')

            times = df_1m.index.values
            closes = df_1m['close'].values
            vols = df_1m['volume'].values
            vol_thresh_vals = vol_thresholds.values
            price_thresh_vals = price_thresholds.values

            pit_times, pit_reasons = [], []
            current_vol = 0.0
            current_open_price = closes[0]

            n_rows = len(df_1m)

            for i in range(n_rows):
                current_vol += vols[i]
                price_dev = abs(closes[i] - current_open_price)

                v_thresh = vol_thresh_vals[i]
                p_thresh = price_thresh_vals[i]

                if np.isnan(v_thresh) or v_thresh <= 0: v_thresh = 1e9
                if np.isnan(p_thresh) or p_thresh <= 0: p_thresh = 1e9

                vol_trigger = current_vol >= v_thresh
                price_trigger = price_dev >= p_thresh

                if vol_trigger or price_trigger:
                    pit_times.append(times[i])
                    if price_trigger:
                        pit_reasons.append('price')
                    else:
                        pit_reasons.append('volume')

                    current_vol = 0.0
                    current_open_price = closes[i]

            pit_bars = pd.DataFrame({'reason': pit_reasons}, index=pd.DatetimeIndex(pit_times))
            pit_bars.index.name = 'timestamp'
            return pit_bars

        except Exception as e:
            tprint_error(f"Error generating PiT bars: {e}")
            return None

    def _generate_dynamic_volume_bars(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Generate Dynamic Volume Bars from 1-minute data.
        N = 1/674th of trailing 7-day volume sum.
        """
        symbol = config.get('symbol', 'ETHUSDT')
        # Load 1m raw data
        # We need to fetch 1m data. We can use KlinesParquetManager.
        try:
            manager = get_klines_manager(config.get('data_dir', 'historical_data'))
            # Calculate range: we need enough history for rolling 7d.
            # Assuming 'df' passed is 15m data, we find its start/end and load 1m for that range + buffer.
            start_date = df.index.min() - pd.Timedelta(days=8) # 7 days buffer for initial rolling
            end_date = df.index.max()

            tprint_info(f"   Loading 1m raw data for Volume Bars: {symbol} {start_date} to {end_date}")
            df_1m = manager.read_data(symbol, "1m", start_date, end_date, data_type="raw")

            if df_1m is None or df_1m.empty:
                tprint_error("   ❌ Failed to load 1m data for volume bars")
                return None

            # Filter columns
            df_1m = df_1m[['open', 'high', 'low', 'close', 'volume']]

            # Calculate Rolling 7-day Volume Sum
            # 1m bars per 7 days = 7 * 1440 = 10080
            rolling_7d_vol = df_1m['volume'].rolling(window=10080, min_periods=1440).sum()

            # Dynamic Threshold N
            # N = Volume7d / 674
            dynamic_N = rolling_7d_vol / 674.0
            dynamic_N = dynamic_N.fillna(method='bfill') # Fill initial

            # Generate Bars
            # Iterating through 1m bars to construct volume bars
            # This is slow in pure Python. We use a vectorized approach or optimized loop.
            # Optimized Loop:

            # Pre-compute cumulative volume
            df_1m['cum_vol'] = df_1m['volume'].cumsum()

            # We can't easily vectorize dynamic threshold because threshold changes with time.
            # However, N changes slowly.
            # Let's iterate.

            # Extract numpy arrays for speed
            times = df_1m.index.values
            opens = df_1m['open'].values
            highs = df_1m['high'].values
            lows = df_1m['low'].values
            closes = df_1m['close'].values
            vols = df_1m['volume'].values
            thresholds = dynamic_N.values

            vb_times = []
            vb_opens = []
            vb_highs = []
            vb_lows = []
            vb_closes = []
            vb_vols = []

            current_vol = 0.0
            current_open = opens[0]
            current_high = highs[0]
            current_low = lows[0]

            # Iterate
            n_rows = len(df_1m)

            for i in range(n_rows):
                vol = vols[i]
                current_vol += vol
                current_high = max(current_high, highs[i])
                current_low = min(current_low, lows[i])

                target_n = thresholds[i]
                if np.isnan(target_n) or target_n <= 0:
                    target_n = 10000.0 # Fallback

                if current_vol >= target_n:
                    # Bar complete
                    vb_times.append(times[i])
                    vb_opens.append(current_open)
                    vb_highs.append(current_high)
                    vb_lows.append(current_low)
                    vb_closes.append(closes[i])
                    vb_vols.append(current_vol)

                    # Reset
                    current_vol = 0.0
                    if i + 1 < n_rows:
                        current_open = opens[i+1]
                        current_high = highs[i+1]
                        current_low = lows[i+1]

            # Construct DataFrame
            volume_bars = pd.DataFrame({
                'open': vb_opens,
                'high': vb_highs,
                'low': vb_lows,
                'close': vb_closes,
                'volume': vb_vols
            }, index=pd.DatetimeIndex(vb_times))

            volume_bars.index.name = 'timestamp'
            return volume_bars

        except Exception as e:
            tprint_error(f"Error generating volume bars: {e}")
            return None

    def apply_afml_sampling(self, df: pd.DataFrame, config: Dict[str, Any], filter_type: str = 'price') -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        """
        Apply CUSUM filtering to sample meaningful events, targeting ~10% of total bars.
        Uses an enhanced binary search to find the optimal threshold factor.
        Supports: 'price', 'volatility', 'volume', 'spread'
        """
        # Determine if we are using volume time for volatility calc
        # Usually assume yes if this mixin is active with volume bars
        # But here we default to standard get_daily_vol (time based) unless we override
        # We should use the same logic as prepare_specialist_data
        use_volume_bars = config.get('use_volume_bars', True)

        if filter_type == 'price':
            series = df['close']
            threshold_base = get_daily_vol(series, use_volume_time=use_volume_bars)
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
            threshold_base = get_daily_vol(series, use_volume_time=use_volume_bars)
            
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

        # Volatility: use volume time awareness if configured
        use_volume_bars = config.get('use_volume_bars', True)
        vol = get_daily_vol(close, use_volume_time=use_volume_bars)
        vol = vol.fillna(method='bfill').fillna(method='ffill')
        
        # Lookforward: If volume bars, this is number of bars. If time bars, also bars (but roughly time)
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
        
        # We return the full TBM output which now includes mfe/mae/ret
        # But we also need the 'bin' column for compatibility
        tbm_events = get_bins(tbm_events, close)

        return tbm_events

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
