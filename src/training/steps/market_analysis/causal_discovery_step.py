from __future__ import annotations

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
try:
    from econml.ortho_forest import DMLOrthoForest
except ImportError:
    from econml.orf import DMLOrthoForest
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import clone

from src.training.steps.base_step import BaseStep
from src.training.steps.labeling.label_based_layer_0 import run_layer0_kalman_vwap
from src.training.steps.market_analysis.base_models.agnostic_cusum import (
    AgnosticCusumFilter,
    CausalFeatureGenerator
)
from src.training.steps.labeling.orthogonal_label_generation import apply_triple_barrier_multi

# Setup Logger
logger = logging.getLogger(__name__)

class CausalDiscoveryStep(BaseStep):
    """
    Step for Causal Discovery using Orthogonal Random Forest.

    1. Loads and denoises market data.
    2. Generates causal features.
    3. Generates CUSUM signals on features.
    4. Applies Triple Barrier Method labeling.
    5. Samples events using sequential bootstrap.
    6. Trains DMLOrthoForest.
    7. Reports causal metrics.
    """

    def __init__(self, **kwargs):
        super().__init__(step_name="causal_discovery_step")

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        logger.info("Starting Causal Discovery Step...")

        # 1. Data Loading & Preprocessing (Layer 0)
        logger.info("Running Layer 0 Denoising...")

        market_data, source_desc = self.load_market_data_or_fail(config)
        outcomes_dir = Path("outcomes")
        outcomes_dir.mkdir(exist_ok=True, parents=True)
        symbol = config.get("symbol", "ETHUSDT")
        timeframe = config.get("timeframe", "15m")

        market_data, layer0_payload = run_layer0_kalman_vwap(
            symbol=symbol,
            timeframe=timeframe,
            market_data=market_data,
            config=config,
            outcomes_dir=outcomes_dir,
            run_optimization=True
        )

        # 2. Feature Generation
        logger.info("Generating Causal Features...")
        features = pd.DataFrame(index=market_data.index)

        # Main Features for CUSUM
        # 1. Volatility (Garman-Klass)
        features['volatility_gk'] = CausalFeatureGenerator.volatility_main_feature(market_data)

        # 2. Path Smoothness (Efficiency Ratio) - Lagged to avoid collider bias
        er = CausalFeatureGenerator.path_smoothness_feature(market_data)
        features['efficiency_ratio'] = er.shift(1) # Lagged t-1

        # 3. Volume
        features['volume_intensity'] = CausalFeatureGenerator.volume_main_feature(market_data)
        # 4. Market Reversion / Trend
        features['kalman_trend'] = CausalFeatureGenerator.kalman_trend_feature(market_data) # Innovation
        # 5. Momentum Persistence
        features['momentum_persistence'] = CausalFeatureGenerator.momentum_persistence_feature(market_data)
        # 6. Liquidity (Spread)
        features['liquidity_spread'] = CausalFeatureGenerator.liquidity_feature(market_data)
        # 7. Shannon Entropy
        features['shannon_entropy'] = CausalFeatureGenerator.shannon_entropy_feature(market_data)
        # 8. Order Imbalance
        features['order_imbalance'] = CausalFeatureGenerator.order_imbalance_feature(market_data)

        # 9. Shadow Variables (Sessionality)
        sessionality = CausalFeatureGenerator.time_of_day_features(market_data.index)
        features['sin_time'] = sessionality['sin_time']
        features['cos_time'] = sessionality['cos_time']

        # 10. Volatility-of-Volatility
        features['vol_of_vol'] = CausalFeatureGenerator.volatility_of_volatility_feature(features['volatility_gk'], window=10)

        # Normalize features for CUSUM (exclude time features from normalization as they are bounded)
        cols_to_norm = [c for c in features.columns if c not in ['sin_time', 'cos_time']]
        features_norm = features.copy()
        features_norm[cols_to_norm] = (features[cols_to_norm] - features[cols_to_norm].rolling(100).mean()) / (features[cols_to_norm].rolling(100).std() + 1e-9)

        # 3. Signal Generation (CUSUM)
        logger.info("Generating CUSUM Signals...")
        filter_gen = AgnosticCusumFilter(target_events_per_day=7.5) # Target 5-10

        events = pd.DatetimeIndex([])

        for col in features_norm.columns:
            # Generate signals for each feature
            sigs = filter_gen.generate_signals(features_norm[col])
            events = events.union(sigs)

        events = events.sort_values()

        # Limit to available data
        events = events[events.isin(market_data.index)]

        logger.info(f"Generated {len(events)} total unique events.")

        if len(events) < 50:
            logger.warning("Too few events for causal analysis.")
            return {"status": "skipped", "reason": "insufficient_events"}

        # 4. Define Geometries (Multiple Horizons)
        # 1) Standard TBM: TP 2x, SL 1x, Horizon 32
        # 2) Trend Model: TP 4x, SL 2x, Horizon 250
        geometries = [
            {"name": "standard_tbm", "pt": 2.0, "sl": 1.0, "horizon": 32},
            {"name": "trend_model",  "pt": 4.0, "sl": 2.0, "horizon": 250}
        ]

        all_metrics = []

        # 5. Loop through Geometries
        for geom in geometries:
            geom_name = geom["name"]
            logger.info(f"Processing Geometry: {geom_name}")

            # Calculating TBM Labels
            labels = self._apply_tbm(market_data['close'], events, features['volatility_gk'],
                                     pt=geom["pt"], sl=geom["sl"], horizon=geom["horizon"])

            # Sequential Bootstrap & Bagging (Uniqueness)
            uniqueness = self._compute_uniqueness(events, market_data.index, horizon=geom["horizon"])

            # 6. Prepare Data for ORF
            df_events = features.loc[events].copy()
            y_events = labels.loc[events]
            w_events = uniqueness.loc[events] # Uniqueness weights

            # Drop NaNs
            valid_mask = ~df_events.isna().any(axis=1) & ~y_events.isna()
            df_events = df_events[valid_mask]
            y_events = y_events[valid_mask]
            w_events = w_events[valid_mask]

            if len(df_events) < 50:
                 logger.warning(f"Too few events after alignment for geometry {geom_name}.")
                 continue

            T_cols = ['order_imbalance', 'kalman_trend']
            X_cols = ['shannon_entropy', 'momentum_persistence', 'sin_time', 'cos_time']
            W_cols = ['volatility_gk', 'volume_intensity', 'liquidity_spread', 'efficiency_ratio', 'vol_of_vol']

            Y = y_events.values
            T = df_events[T_cols].values
            X = df_events[X_cols].values
            W = df_events[W_cols].values

            # 7. Train DMLOrthoForest
            logger.info(f"Training DMLOrthoForest for {geom_name}...")

            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=30,
                bootstrap=False,
                n_jobs=-1
            )

            orf = DMLOrthoForest(
                n_trees=300,
                min_leaf_size=30,
                max_depth=5,
                subsample_ratio=0.5,
                model_T=clone(rf_model),
                model_Y=clone(rf_model),
                discrete_treatment=False,
                random_state=42
            )

            # Fit
            orf.fit(Y, T, X=X, W=W, sample_weight=w_events.values)

            # 8. Causal Metrics
            logger.info(f"Calculating Metrics for {geom_name}...")

            # Calculate CATE (Conditional Average Treatment Effect)
            cate_pred = orf.const_marginal_effect(X)

            # ATE (Average Treatment Effect)
            ate = np.mean(cate_pred, axis=0)
            # CATE Dispersion
            cate_std = np.std(cate_pred, axis=0)

            metrics_row = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "geometry": geom_name,
                "horizon": geom["horizon"],
                "ate_order_imbalance": ate[0],
                "ate_kalman_trend": ate[1] if len(ate) > 1 else 0.0,
                "cate_std_order_imbalance": cate_std[0],
                "cate_std_kalman_trend": cate_std[1] if len(cate_std) > 1 else 0.0,
                "n_events": len(events)
            }
            all_metrics.append(metrics_row)

        # Save metrics
        if not all_metrics:
            return {"status": "skipped", "reason": "no_valid_geometries"}

        results_file = outcomes_dir / f"causal_metrics_{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        pd.DataFrame(all_metrics).to_csv(results_file, index=False)

        return {"status": "success", "metrics_count": len(all_metrics), "file": str(results_file)}

    def _apply_tbm(self, close: pd.Series, events: pd.DatetimeIndex, volatility: pd.Series, pt: float, sl: float, horizon: int) -> pd.Series:
        """
        Apply Triple Barrier Method to get returns.
        Returns the realized return at the first barrier touch or horizon.
        """
        out_labels = pd.Series(np.nan, index=events)

        vol_aligned = volatility.reindex(events).fillna(method='ffill')

        for t in events:
            if t not in close.index: continue

            t_idx = close.index.get_loc(t)
            end_idx = min(t_idx + horizon, len(close) - 1)

            path = close.iloc[t_idx : end_idx+1]
            if len(path) < 2: continue

            ret_path = (path / path.iloc[0]) - 1.0

            vol = vol_aligned.loc[t]
            if np.isnan(vol) or vol <= 0: vol = 0.01 # fallback

            barrier_up = pt * vol
            barrier_down = -sl * vol

            # Check touches
            touch_up = ret_path[ret_path >= barrier_up]
            touch_down = ret_path[ret_path <= barrier_down]

            first_touch_time = end_idx # Default to horizon
            final_ret = ret_path.iloc[-1]

            if not touch_up.empty:
                t_up = close.index.get_loc(touch_up.index[0])
                first_touch_time = min(first_touch_time, t_up)

            if not touch_down.empty:
                t_down = close.index.get_loc(touch_down.index[0])
                if t_down < first_touch_time:
                    first_touch_time = t_down

            # If touched barrier, use barrier return (approx)
            # Or use actual close price at touch.
            # Let's use actual return at touch index.
            # Note: t_idx is 0 in relative path coordinates

            rel_idx = first_touch_time - t_idx
            final_ret = ret_path.iloc[rel_idx]

            out_labels.loc[t] = final_ret

        return out_labels

    def _compute_uniqueness(self, events: pd.DatetimeIndex, index: pd.DatetimeIndex, horizon: int) -> pd.Series:
        """
        Compute average uniqueness of events.
        """
        # Indicator matrix
        concurrency = pd.Series(0, index=index)

        for t in events:
            if t not in index: continue
            t_idx = index.get_loc(t)
            end_idx = min(t_idx + horizon, len(index))
            concurrency.iloc[t_idx:end_idx] += 1

        uniqueness = pd.Series(np.nan, index=events)
        for t in events:
            if t not in index: continue
            t_idx = index.get_loc(t)
            end_idx = min(t_idx + horizon, len(index))
            c = concurrency.iloc[t_idx:end_idx]
            if len(c) > 0:
                uniqueness.loc[t] = (1.0 / c).mean()

        return uniqueness.fillna(1.0)
