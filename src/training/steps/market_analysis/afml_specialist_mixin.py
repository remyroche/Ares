import logging
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
from sklearn.metrics import roc_auc_score

from src.utils.ml_common.afml_utils import (
    get_daily_vol, get_t_events, get_vertical_barrier, 
    apply_triple_barrier, get_bins, get_weights_by_uniqueness,
    frac_diff_fixed, get_sample_weights, compute_master_weight, compute_hunter_weight,
    get_sample_uniqueness, get_num_concurrent_events, calculate_rolling_volatility,
    calculate_dynamic_range_threshold
)
from src.utils.ml_common.wavelet_utils import get_wavelet_features
from src.utils.ml_common.physics_router import AdaptiveHunterRouter
from src.utils.ml_common.anarchy_detector import AnarchyDetector
from src.training.steps.market_analysis.offline_causal_discovery import OfflineCausalDiscovery
from src.utils.data.klines_parquet import get_klines_manager
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success, tprint
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType
from src.utils.ml_common.specialist_xgb import train_specialist_xgb_with_oof

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

    def _apply_standard_feature_selection(self, features: pd.DataFrame, max_features: int = 30) -> pd.DataFrame:
        tprint_info(f"🔍 Applying feature selection (Max: {max_features})...")
        if features.empty: return features

        # Causal Pruning Integration (Phase 3 Offline Loading)
        causal_engine = OfflineCausalDiscovery()
        universal_drivers = causal_engine.load_artifacts()

        if universal_drivers:
            tprint_info(f"   Found {len(universal_drivers)} Universal Drivers in artifacts.")
            keep_cols = [c for c in features.columns if c in universal_drivers]
            remaining_quota = max_features - len(keep_cols)

            if remaining_quota > 0:
                candidates = features.drop(columns=keep_cols)
                constant_features = candidates.columns[candidates.nunique() <= 1]
                if len(constant_features) > 0:
                    candidates = candidates.drop(columns=constant_features)

                if not candidates.empty:
                    top_candidates = candidates.var().nlargest(remaining_quota).index.tolist()
                    keep_cols.extend(top_candidates)

            tprint_info(f"   Selected {len(keep_cols)} features (Drivers + Variance).")
            return features[keep_cols]

        tprint_info("   No Causal Drivers found. Fallback to correlation/variance.")
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)

        correlation_matrix = features.corr().abs()
        upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
        to_drop = []
        for column in upper_triangle.columns:
            if any(upper_triangle[column] > 0.9):
                to_drop.append(column)
        if to_drop:
            features = features.drop(columns=to_drop)

        if len(features.columns) > max_features:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(max_features).index
            features = features[top_features]

        tprint_info(f"   Final feature count: {len(features.columns)}")
        return features

    def generate_standard_enhanced_features(
        self,
        df: pd.DataFrame,
        specialist_type: SpecialistType,
        manual_feature_func: Optional[Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]] = None
    ) -> pd.DataFrame:
        tprint_info(f"🛠️ Generating Features for {specialist_type}...")

        pipeline_features = pd.DataFrame(index=df.index)
        if hasattr(self, 'feature_pipeline'):
            pipeline_features = self.feature_pipeline.generate_enhanced_features(
                df, specialist_type.value, {'enhanced_features': True}
            )

        manual_features = pd.DataFrame(index=df.index)
        if manual_feature_func:
            manual_features = manual_feature_func(df, pipeline_features)

        combined_features = pd.concat([pipeline_features, manual_features], axis=1)
        combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]

        # Apply Phase 2: Feature Conditioning (Partial Orthogonalization)
        if not combined_features.empty:
            tprint_info("   Applying Phase 2: Partial Orthogonalization (Conditioning)...")
            vol = df['close'].pct_change().rolling(20).std().fillna(0)
            eff = (df['close'] - df['close'].shift(20)).abs() / (df['close'].diff().abs().rolling(20).sum() + 1e-9)
            W = pd.concat([vol, eff], axis=1).fillna(0)
            W.columns = ['vol_nuisance', 'eff_nuisance']

            from sklearn.linear_model import LinearRegression

            common_idx = combined_features.index.intersection(W.index)
            X_curr = combined_features.loc[common_idx]
            W_curr = W.loc[common_idx]

            lr = LinearRegression()
            X_curr_clean = X_curr.fillna(0)
            W_curr_clean = W_curr.fillna(0)

            lr.fit(W_curr_clean, X_curr_clean)
            X_hat = lr.predict(W_curr_clean)
            X_resid = X_curr_clean - (0.7 * X_hat)

            X_resid.columns = [f"{c}_resid" for c in X_curr.columns]
            combined_features = pd.concat([X_curr, X_resid], axis=1)

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
        try:
            symbol = config.get('symbol', 'BTCUSDT')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')

            tprint_info(f"🚀 Starting Hunter Specialist execution for {symbol}...")

            # 1. Load Market Data
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
            feature_df = self.generate_standard_enhanced_features(df, specialist_type, manual_feature_func)

            # Phase 6: Add Anarchy Features
            tprint_info("   Phase 6: Generating Anarchy Features...")
            anarchy_detector = AnarchyDetector()
            anarchy_features = anarchy_detector.generate_anarchy_features(df)
            feature_df = pd.concat([feature_df, anarchy_features], axis=1)

            # 3. AFML & Hunter Prep
            tprint_info("   Preparing AFML/Hunter Data (Sampling, Router, Weighting)...")
            X, y, weights = self.prepare_specialist_data(
                market_data=df,
                feature_df=feature_df,
                config=config,
                filter_type=filter_type,
                pt_sl_config_key=pt_sl_config_key,
                default_pt_sl=default_pt_sl
            )

            # 4. Centralized Training (with Monotonic Constraints)
            tprint_info("🤖 Training Hunter Model (XGB + Monotonic Ridge)...")
            training_result = train_specialist_xgb_with_oof(
                X.fillna(0.0),
                y.fillna(0.0),
                sample_weight=weights,
                n_splits=5,
                apply_monotonic_constraints=True
            )

            # 5. Output Construction
            oof_probs = training_result.oof_predictions
            final_probs = pd.Series(np.nan, index=df.index)
            valid_oof = oof_probs.dropna()
            if len(valid_oof) > 0:
                final_probs.loc[valid_oof.index] = valid_oof.values
            final_probs = final_probs.ffill().fillna(0.5)
            final_preds = (final_probs >= 0.5).astype(int)
            full_labels = pd.Series(0, index=df.index)
            full_labels.loc[y.index] = y

            # 6. Save
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

            tprint_success(f"✅ {self.step_name} completed in {time.time() - start_time:.2f}s")
            return result

        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.exception(f"❌ {self.step_name} failed: {e}")
            tprint_error(f"❌ {self.step_name} failed: {e}")
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

        tprint_info("   [AFML] Starting Data Prep...")

        # 1. Determine Anchor
        specialist_type = config.get('specialist_type', self.specialist_type if hasattr(self, 'specialist_type') else SpecialistType.VOLUME_FORCE)
        anchor_type = self._get_anchor_type(specialist_type)
        context_type = 'range' if anchor_type == 'dollar' else 'dollar'
        tprint_info(f"   [AFML] Anchor: {anchor_type}, Context: {context_type}")

        # 2. Generate Bars
        anchor_df = self._generate_bars(market_data, config, anchor_type)
        context_df = self._generate_bars(market_data, config, context_type)

        # 3. Join & Cross-Bar Features
        anchor_feats = anchor_df.add_suffix(f"_{anchor_type}_bar")
        context_feats = context_df.add_suffix(f"_{context_type}_bar")

        if anchor_type == 'dollar':
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
        sampled_df, t_events = self.apply_afml_sampling(market_data_anchor, config, filter_type=filter_type)
        pt_sl = config.get(pt_sl_config_key, default_pt_sl) if pt_sl_config_key else default_pt_sl
        tbm_labels_df = self.generate_tbm_labels(market_data_anchor, t_events, config, pt_sl)

        # 5. Physics Router (Phase 1)
        tprint_info("   [Phase 1] Physics Router - Calculating Regimes...")
        router = AdaptiveHunterRouter()
        physics_feats = router.compute_physics_features(market_data_anchor)
        router.fit(physics_feats.values)

        target_regime = config.get('target_regime', None)
        regime_weights_series = pd.Series(1.0, index=t_events)

        if target_regime:
            tprint_info(f"   [Phase 1] Filtering for Target Regime: {target_regime}")
            X_phys = physics_feats.loc[t_events].values
            X_phys_scaled = router.scaler.transform(X_phys)
            probs = router.gmm.predict_proba(X_phys_scaled)
            rev_map = {v: k for k, v in router.regime_map.items()}
            regime_idx = rev_map.get(target_regime, 0)
            regime_weights_series = pd.Series(probs[:, regime_idx], index=t_events)

        # 6. Wavelet Features & Hunter Weighting (Phase 0)
        tprint_info("   [Phase 0] Calculating Hunter Weights...")
        anchor_wavelet_fam = 'db4' if anchor_type == 'dollar' else 'sym4'
        wavelet_vals = []
        close_vals = market_data_anchor['close'].values
        for evt in t_events:
            idx = market_data_anchor.index.get_loc(evt)
            if idx > 32:
                s = close_vals[idx-32:idx]
                feat = get_wavelet_features(s, wavelet=anchor_wavelet_fam)
                wavelet_vals.append(feat['hf_lf_ratio'])
            else:
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

        final_weights = base_weights * regime_weights_series.values
        weights_series = pd.Series(final_weights, index=t_events)

        # 7. Final Clean
        wavelet_df = pd.DataFrame({'wavelet_noise': noise_ratio}, index=t_events)
        X_sampled = pd.concat([X_combined.loc[t_events], wavelet_df], axis=1)

        X = X_sampled.select_dtypes(include=[np.number])
        valid_mask = X.notna().all(axis=1) & tbm_labels_df['bin'].notna()
        X, y, w = X.loc[valid_mask], tbm_labels_df['bin'].loc[valid_mask], weights_series.loc[valid_mask]

        tprint_info(f"   [AFML] Data Ready: {len(X)} samples.")
        return X, y, w

    def _get_anchor_type(self, specialist_type: SpecialistType) -> str:
        volume_anchors = {SpecialistType.VOLUME_FORCE, SpecialistType.LIQUIDITY_REGIME, SpecialistType.MICROSTRUCTURE, SpecialistType.RISK_REGIME, SpecialistType.SMC_REGIME, SpecialistType.SPECTRAL}
        return 'dollar' if specialist_type in volume_anchors else 'range'

    def _generate_bars(self, df: pd.DataFrame, config: Dict[str, Any], bar_type: str) -> pd.DataFrame:
        if bar_type == 'dollar': return self._generate_dynamic_dollar_bars(df, config)
        elif bar_type == 'range': return self._generate_range_bars(df, config)
        elif bar_type == 'pit': return self.generate_pit_bars(df, config)
        else: raise ValueError(f"Unknown bar type: {bar_type}")

    def _generate_dynamic_dollar_bars(self, df, config):
        tprint_info("   Generating Dynamic Dollar Bars...")
        symbol = config.get('symbol', 'ETHUSDT')
        try:
            manager = get_klines_manager(config.get('data_dir', 'historical_data'))
            start_date = df.index.min() - pd.Timedelta(days=8)
            end_date = df.index.max()
            df_1m = manager.read_data(symbol, "1m", start_date, end_date, data_type="raw")
            if df_1m is None or df_1m.empty: return None
            df_1m = df_1m[['open', 'high', 'low', 'close', 'volume']]

            # Dollar Volume calculation
            df_1m['dollar_volume'] = df_1m['close'] * df_1m['volume']

            # Rolling 7-day Dollar Volume Sum
            rolling_7d_dollar_vol = df_1m['dollar_volume'].rolling(window=10080, min_periods=1440).sum()
            dynamic_N = rolling_7d_dollar_vol / 674.0
            dynamic_N = dynamic_N.fillna(method='bfill')

            times = df_1m.index.values
            opens = df_1m['open'].values
            highs = df_1m['high'].values
            lows = df_1m['low'].values
            closes = df_1m['close'].values
            vols = df_1m['volume'].values # We still track volume for the bar
            dollar_vols = df_1m['dollar_volume'].values
            thresholds = dynamic_N.values

            db_times, db_opens, db_highs, db_lows, db_closes, db_vols = [],[],[],[],[],[]
            current_dollar_vol, current_vol, current_open, current_high, current_low = 0.0, 0.0, opens[0], highs[0], lows[0]
            n_rows = len(df_1m)

            for i in range(n_rows):
                dollar_vol = dollar_vols[i]
                vol = vols[i]

                current_dollar_vol += dollar_vol
                current_vol += vol

                current_high = max(current_high, highs[i])
                current_low = min(current_low, lows[i])
                target_n = thresholds[i]
                if np.isnan(target_n) or target_n <= 0: target_n = 10000000.0 # High fallback for dollar

                if current_dollar_vol >= target_n:
                    db_times.append(times[i])
                    db_opens.append(current_open)
                    db_highs.append(current_high)
                    db_lows.append(current_low)
                    db_closes.append(closes[i])
                    db_vols.append(current_vol)

                    current_dollar_vol = 0.0
                    current_vol = 0.0

                    if i + 1 < n_rows:
                        current_open = opens[i+1]
                        current_high = highs[i+1]
                        current_low = lows[i+1]

            # We return 'volume' column as the volume accumulated in the dollar bar
            # Or should we return dollar volume? Standard OHLCV implies V is volume.
            # We stick to Volume.
            return pd.DataFrame({'open': db_opens, 'high': db_highs, 'low': db_lows, 'close': db_closes, 'volume': db_vols}, index=pd.DatetimeIndex(db_times))
        except Exception as e:
            tprint_error(f"Error generating dollar bars: {e}")
            return None

    def _generate_range_bars(self, df, config):
        tprint_info("   Generating Range Bars...")
        symbol = config.get('symbol', 'ETHUSDT')
        try:
            vol_15m = calculate_rolling_volatility(df['close'], window_days=7)
            delta_p_series = calculate_dynamic_range_threshold(vol_15m, df['close'])
            delta_p_daily = delta_p_series.resample('1D').last().shift(1)

            manager = get_klines_manager(config.get('data_dir', 'historical_data'))
            start_date = df.index.min() - pd.Timedelta(days=8)
            end_date = df.index.max()
            df_1m = manager.read_data(symbol, "1m", start_date, end_date, data_type="raw")
            if df_1m is None: return None

            thresholds_1m = delta_p_daily.reindex(df_1m.index, method='ffill').fillna(method='bfill')

            times = df_1m.index.values
            opens = df_1m['open'].values
            highs = df_1m['high'].values
            lows = df_1m['low'].values
            closes = df_1m['close'].values
            vols = df_1m['volume'].values
            threshold_vals = thresholds_1m.values

            rb_times, rb_opens, rb_highs, rb_lows, rb_closes, rb_vols, rb_durations = [],[],[],[],[],[],[]
            current_open, current_high, current_low, current_vol = opens[0], highs[0], lows[0], 0.0
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
                    rb_durations.append((times[i] - times[start_idx]).astype('timedelta64[s]').astype(float) + 60.0)

                    if i + 1 < n_rows:
                        current_open = p; current_high = p; current_low = p; current_vol = 0.0; start_idx = i + 1

            return pd.DataFrame({'open': rb_opens, 'high': rb_highs, 'low': rb_lows, 'close': rb_closes, 'volume': rb_vols, 'bar_duration': rb_durations}, index=pd.DatetimeIndex(rb_times))
        except Exception: return None

    def apply_afml_sampling(self, df, config, filter_type='price'):
        tprint_info(f"   Applying AFML CUSUM Sampling ({filter_type})...")
        use_volume_bars = config.get('use_volume_bars', True)
        if filter_type == 'price':
            threshold_base = get_daily_vol(df['close'], use_volume_time=use_volume_bars)
        else:
            threshold_base = get_daily_vol(df['close'], use_volume_time=use_volume_bars)
        
        target_rate = config.get('afml_target_sampling_rate', 0.10)
        low, high = 1e-6, 1000000.0
        target_count = int(len(df) * target_rate)
        threshold_base = threshold_base.fillna(method='bfill')
        
        best_events = df.index[::int(1/target_rate)] if len(df) > 0 else df.index
        
        for _ in range(15):
            mid = (low + high) / 2
            t_events = get_t_events(df['close'], threshold_base * mid)
            if len(t_events) > target_count: low = mid
            else: high = mid
            best_events = t_events
            
        tprint_info(f"   Sampled {len(best_events)} events.")
        return df.loc[best_events], best_events

    def generate_tbm_labels(self, df, t_events, config, pt_sl):
        tprint_info("   Generating Triple Barrier Labels...")
        vol = get_daily_vol(df['close'], use_volume_time=config.get('use_volume_bars', True))
        vb = get_vertical_barrier(df['close'], t_events, config.get('lookforward_bars', 35))
        tbm = apply_triple_barrier(df['close'], t_events, pt_sl, vol, config.get('min_ret', 0.001), vb)
        return get_bins(tbm, df['close'])

    def get_concurrent_weights(self, t1, idx):
        return get_weights_by_uniqueness(t1, idx)

    def generate_pit_bars(self, df, config): return pd.DataFrame()
