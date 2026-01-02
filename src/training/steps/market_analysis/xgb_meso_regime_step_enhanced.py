"""
Enhanced XGB Meso Regime Step with MI Improvements

This enhanced version implements:
- Enhanced feature generation for MI improvement
- Real-time MI monitoring during training
- Hyperparameter optimization for MI > 0.02 target
- Data structure standardization
- Binary output enforcement
"""

import logging
import time
from typing import Any, Dict, Optional, Tuple, List
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
)
from src.utils.versioned_artifacts.temporal_splits import (
    create_temporal_split_config_for_pipeline,
    TemporalSplitConfig,
)
from src.utils.ml_common.standardized_xgb_trainer import (
    StandardizedXGBTrainer,
    XGBTrainingConfig,
)
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2
)
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType

logger = logging.getLogger(__name__)


class EnhancedXGBMesoRegimeStep(SpecialistDiagnosticsMixinEnhancedV2, BaseStep):

    @property
    def artifact_router(self):
        """Override artifact_router property for enhanced specialists."""
        if self._artifact_router is None:
            from src.utils.artifact_router import ArtifactRouter
            self._artifact_router = ArtifactRouter(
                base_dir="artifacts",
                versioned_store_dir="versioned_artifacts",
                historical_data_dir="historical_data",
                enable_versioned_artifacts=self.use_versioned_artifacts
            )
        return self._artifact_router

    """
    Enhanced XGB Meso Regime Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Meso-specific feature engineering
    - Hyperparameter optimization for MI > 0.02
    - Data structure standardization
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_xgb_meso_regime_step"):
        """Initialize the enhanced XGB meso regime step."""
        super().__init__()
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("EnhancedXGBMesoRegimeStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        
    def _generate_enhanced_meso_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate enhanced meso features with MI improvements."""
        # Import original meso features
        try:
            from src.feature_generation.categories.meso_regime_features import generate_meso_regime_features
            meso_features = generate_meso_regime_features(df, config)
        except ImportError:
            # Fallback if original features not available
            meso_features = pd.DataFrame(index=df.index)
        
        # Generate enhanced features
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'meso_regime', config
        )
        
        # Meso-specific enhancements
        meso_enhanced = self._add_meso_specific_features(df, meso_features)
        
        # Combine all features
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        # MI-based feature selection
        if len(combined_features.columns) > 50:
            # Select top features by MI contribution
            from sklearn.feature_selection import mutual_info_regression
            mi_scores = mutual_info_regression(combined_features.fillna(0), df["target_long"] if "target_long" in df.columns else combined_features.iloc[:, 0])
            mi_ranking = np.argsort(mi_scores)[::-1][:50]  # Top 50 features
            combined_features = combined_features.iloc[:, mi_ranking]
        all_features = pd.concat([meso_features, enhanced_features, meso_enhanced], axis=1)
        
        # Remove duplicates and clean
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        all_features = all_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return all_features
    
    def _add_meso_specific_features(self, df: pd.DataFrame, meso_features: pd.DataFrame) -> pd.DataFrame:
        """Add meso-specific enhanced features."""
        features = pd.DataFrame(index=df.index)
        
        # Enhanced meso analysis
        if 'close' in df.columns:
            close = df['close']
            returns = close.pct_change()
            
            # Multi-timeframe meso analysis
            for window in [40,60,80]:
                # Meso trend
                meso_trend = returns.rolling(window).mean()
                features[f'meso_trend_{window}'] = meso_trend
                
                # Meso momentum
                meso_momentum = returns.rolling(window).sum()
                features[f'meso_momentum_{window}'] = meso_momentum
                
                # Meso acceleration
                meso_acceleration = meso_momentum.diff()
                features[f'meso_acceleration_{window}'] = meso_acceleration
                
                # Meso volatility
                meso_volatility = returns.rolling(window).std()
                features[f'meso_volatility_{window}'] = meso_volatility
                
                # Meso risk-adjusted returns
                risk_adjusted = meso_trend / meso_volatility
                features[f'meso_risk_adjusted_{window}'] = risk_adjusted
                
                # Meso regime strength
                regime_strength = abs(meso_trend) / meso_volatility
                features[f'meso_regime_strength_{window}'] = regime_strength
                
                # Meso persistence
                meso_persistence = (meso_trend > 0).rolling(window).mean()
                features[f'meso_persistence_{window}'] = meso_persistence
                
                # Meso regime transitions
                regime_transition = meso_persistence.diff()
                features[f'meso_regime_transition_{window}'] = regime_transition
            
            # Cross-timeframe meso analysis
            for short_window in [5, 10]:
                for long_window in [20, 50]:
                    short_trend = returns.rolling(short_window).mean()
                    long_trend = returns.rolling(long_window).mean()
                    
                    # Trend alignment
                    trend_alignment = (short_trend * long_trend)
                    features[f'meso_trend_alignment_{short_window}_{long_window}'] = trend_alignment
                    
                    # Trend divergence
                    trend_divergence = abs(short_trend - long_trend)
                    features[f'meso_trend_divergence_{short_window}_{long_window}'] = trend_divergence
                    
                    # Momentum convergence
                    momentum_convergence = (short_trend > 0) == (long_trend > 0)
                    features[f'meso_momentum_convergence_{short_window}_{long_window}'] = momentum_convergence.astype(int)
            
            # Meso cycle analysis
            for window in [10, 20, 50]:
                # Cycle detection using autocorrelation
                cycle_strength = returns.rolling(window).apply(lambda x: x.autocorr())
                features[f'meso_cycle_strength_{window}'] = cycle_strength
                
                # Cycle phase
                cycle_phase = np.arctan2(returns.rolling(window).mean(), returns.rolling(window).std())
                features[f'meso_cycle_phase_{window}'] = cycle_phase
                
                # Cycle amplitude
                cycle_amplitude = returns.rolling(window).std()
                features[f'meso_cycle_amplitude_{window}'] = cycle_amplitude
            
            # Meso extreme analysis
            for window in [40,60,80]:
                # Extreme returns
                extreme_returns = returns.rolling(window).apply(lambda x: (x.abs() > x.std() * 1.5).sum())
                features[f'meso_extreme_returns_{window}'] = extreme_returns
                
                # Tail risk
                tail_risk = returns.rolling(window).apply(lambda x: (x < x.quantile(0.1)).mean())
                features[f'meso_tail_risk_{window}'] = tail_risk
                
                # Volatility clustering
                volatility_clustering = returns.rolling(window).std().rolling(window).corr(returns.rolling(window).std())
                features[f'meso_volatility_clustering_{window}'] = volatility_clustering
            
            # Meso regime classification
            for window in [20, 50]:
                # Bull regime
                bull_regime = (returns.rolling(window).mean() > 0) & (returns.rolling(window).std() < returns.rolling(window*2).std() * 0.8)
                features[f'meso_bull_regime_{window}'] = bull_regime.astype(int)
                
                # Bear regime
                bear_regime = (returns.rolling(window).mean() < 0) & (returns.rolling(window).std() < returns.rolling(window*2).std() * 0.8)
                features[f'meso_bear_regime_{window}'] = bear_regime.astype(int)
                
                # Volatile regime
                volatile_regime = returns.rolling(window).std() > returns.rolling(window*2).std() * 1.2
                features[f'meso_volatile_regime_{window}'] = volatile_regime.astype(int)
                
                # Range-bound regime
                range_bound = (abs(returns.rolling(window).mean()) < returns.rolling(window).std() * 0.3) & (returns.rolling(window).std() < returns.rolling(window*2).std() * 0.8)
                features[f'meso_range_bound_{window}'] = range_bound.astype(int)
        
        # Volume-meso relationship
        if 'volume' in df.columns and 'close' in df.columns:
            volume = df['volume']
            returns = df['close'].pct_change()
            
            # Volume-adjusted meso analysis
            volume_ma = volume.rolling(25).mean()
            volume_anomaly = volume / volume_ma
            
            for window in [5, 10, 20]:
                # Volume-weighted meso trend
                volume_weighted_trend = (returns * volume).rolling(window).sum()
                features[f'meso_volume_weighted_trend_{window}'] = volume_weighted_trend
                
                # Volume-meso correlation
                volume_meso_corr = returns.rolling(window).corr(volume)
                features[f'meso_volume_meso_corr_{window}'] = volume_meso_corr
                
                # Volume confirmation of meso moves
                volume_confirmation = (volume_anomaly > 1.5) & (abs(returns.rolling(window).mean()) > returns.rolling(window*2).std() * 0.3)
                features[f'meso_volume_confirmation_{window}'] = volume_confirmation.astype(int)
                
                # Volume-meso divergence
                volume_divergence = abs(volume_meso_corr) < 0.3
                features[f'meso_volume_divergence_{window}'] = volume_divergence.astype(int)
                
                # Volume-meso efficiency
                volume_efficiency = returns.abs() / (volume + 1e-8)
                features[f'meso_volume_efficiency_{window}'] = volume_efficiency.rolling(window).mean()
        
        # Support/resistance meso analysis
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            close = df['close']
            high = df['high']
            low = df['low']
            
            for window in [10, 20, 50]:
                # Meso support/resistance
                rolling_max = close.rolling(window).max()
                rolling_min = close.rolling(window).min()
                
                # Distance to meso levels
                features[f'meso_distance_to_resistance_{window}'] = (rolling_max - close) / rolling_max
                features[f'meso_distance_to_support_{window}'] = (close - rolling_min) / rolling_max
                
                # Meso SR strength
                features[f'meso_sr_strength_{window}'] = (rolling_max - rolling_min) / close
                
                # Meso level breaches
                features[f'meso_resistance_breach_{window}'] = (close > rolling_max.shift(1)).astype(int)
                features[f'meso_support_breach_{window}'] = (close < rolling_min.shift(1)).astype(int)
                
                # Meso range expansion
                range_expansion = (rolling_max - rolling_min) / (rolling_max - rolling_min).rolling(window*2).mean()
                features[f'meso_range_expansion_{window}'] = range_expansion
                
                # Meso range contraction
                range_contraction = range_expansion < 0.8
                features[f'meso_range_contraction_{window}'] = range_contraction.astype(int)
                
                # Meso position
                meso_position = (close - rolling_min) / (rolling_max - rolling_min)
                features[f'meso_position_{window}'] = meso_position
                
                # Meso position momentum
                meso_position_momentum = meso_position.diff()
                features[f'meso_position_momentum_{window}'] = meso_position_momentum
        
        # Time-based meso patterns
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour_of_day'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_london_session'] = ((df.index.hour >= 8) & (df.index.hour <= 16)).astype(int)
            features['is_ny_session'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(int)
            features['is_asia_session'] = ((df.index.hour >= 0) & (df.index.hour <= 8)).astype(int)
            
            # Session overlaps
            features['is_london_ny_overlap'] = ((df.index.hour >= 13) & (df.index.hour <= 16)).astype(int)
            
            # Weekend effects on meso
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Time-based meso transitions
            features['is_end_of_day'] = (df.index.hour >= 20).astype(int)
            features['is_start_of_day'] = (df.index.hour <= 8).astype(int)
            
            # Weekly patterns
            features['is_monday'] = (df.index.dayofweek == 0).astype(int)
            features['is_friday'] = (df.index.dayofweek == 4).astype(int)
        
        return features
    

    def _create_meso_labels(self, df: pd.DataFrame, lookforward: int = 35) -> pd.Series:
        """Create meso labels based on meso regime patterns."""
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            
            # Multi-timeframe meso analysis
            meso_trend_10 = returns.rolling(15).mean()
            meso_trend_20 = returns.rolling(25).mean()
            
            # Meso regime strength
            regime_strength = abs(meso_trend_10) / returns.rolling(15).std()
            
            # Future meso trend
            future_meso_trend = returns.shift(-lookforward).rolling(15).mean()
            
            # Meso regime change detection
            regime_change = abs(future_meso_trend - meso_trend_10)
            regime_change_threshold = returns.rolling(15).std() * 0.3
            
            # Label: 1 for significant meso regime change
            labels = (regime_change > regime_change_threshold).astype(int)
            
            return labels
        else:
            # Fallback to simple trend-based labels
            returns = df['close'].pct_change()
            future_returns = returns.shift(-lookforward)
            labels = (future_returns.abs() > returns.rolling(15).std()).astype(int)
            return labels
    
    def _optimize_xgb_hyperparameters_for_mi(self, X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, Any], float]:
        """Optimize XGBoost hyperparameters specifically for MI improvement."""
        best_params = {}
        best_mi = 0.0
        
        # Parameter grid for XGBoost MI optimization
        # Parameter grid for MI-focused optimization
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [4, 6],
            "learning_rate": [0.03, 0.07, 0.1],
            "subsample": [0.8, 0.9],
            "colsample_bytree": [0.8, 0.9],
            "gamma": [0, 0.1, 0.2],
            "reg_alpha": [0.1, 0.5, 1.0],
            "reg_lambda": [2, 5, 10],
            "min_child_weight": [20, 40]
        }
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        for params in self._generate_param_combinations(param_grid, max_combinations=15):
            mi_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train XGBoost model
                import xgboost as xgb
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    random_state=42,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    **params
                )
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                         early_stopping_rounds=20, verbose=False)
                
                # Compute MI
                val_pred = model.predict_proba(X_val)[:, 1]
                mi_score = mutual_info_regression(
                    val_pred.reshape(-1, 1), y_val.values
                )[0]
                mi_scores.append(mi_score)
            
            avg_mi = np.mean(mi_scores)
            
            if avg_mi > best_mi:
                best_mi = avg_mi
                best_params = params.copy()
                
                tprint_info(f"🔥 New best XGB MI: {avg_mi:.4f} with params: {params}")
        
        tprint_success(f"✅ Best XGBoost hyperparameters found: MI = {best_mi:.4f}")
        return best_params, best_mi
    
    def _generate_param_combinations(self, param_grid: Dict[str, List], max_combinations: int = 20):
        """Generate parameter combinations for optimization."""
        import itertools
        import random
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        # Generate all combinations
        all_combinations = list(itertools.product(*values))
        
        # Randomly sample if too many
        if len(all_combinations) > max_combinations:
            all_combinations = random.sample(all_combinations, max_combinations)
        
        for combination in all_combinations:
            yield dict(zip(keys, combination))
    
    def _train_enhanced_meso_model(self, features: pd.DataFrame, labels: pd.Series, 
                                   config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train enhanced meso model with MI optimization."""
        
        # Optimize hyperparameters for MI
        tprint_info("🔧 Optimizing XGBoost hyperparameters for MI improvement...")
        best_params, best_mi = self._optimize_xgb_hyperparameters_for_mi(features, labels)
        
        # Create temporal split config
        temporal_config = create_temporal_split_config_for_pipeline(
            symbol=config.get("symbol", "ETHUSDT"),
            exchange=config.get("exchange", "binance"),
            timeframe=config.get("timeframe", "15m"),
            direction=config.get("direction", "long"),
            n_splits=config.get("meso_n_splits", 5),
            walk_forward_type="rolling",
            test_size_ratio=config.get("meso_test_size_ratio", 0.2),
            min_train_samples=config.get("meso_min_train_samples", 500),
        )
        
        # Create training config
        training_config = XGBTrainingConfig(
            objective="binary:logistic",
            random_state=42,
            **best_params
        )
        
        # Train with standardized trainer
        trainer = StandardizedXGBTrainer(training_config)
        train_result = trainer.train_time_series_cv(features, labels, temporal_config)
        
        # Extract best model
        best_model = train_result.models[-1] if train_result.models else None
        
        # Compute MI metrics
        oof_preds = train_result.oof_predictions
        if 'probability' in oof_preds.columns:
            mi_score = mutual_info_regression(
                oof_preds['probability'].values.reshape(-1, 1), 
                labels.loc[oof_preds.index].values
            )[0]
        else:
            mi_score = 0.0
        
        # Store training metrics
        self.training_metrics.append({
            'mi_score': mi_score,
            'n_features': len(features.columns),
            'best_params': best_params
        })
        
        metrics = {
            'mi_score': mi_score,
            'auc': train_result.metrics.get('oof_auc', 0.0),
            'log_loss': train_result.metrics.get('oof_log_loss', 0.0),
            'n_features': len(features.columns),
            'optimization_params': best_params,
            'training_time': train_result.metrics.get('training_time', 0.0)
        }
        
        return best_model, metrics
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced XGB meso regime step."""
        start_time = time.time()
        metrics: Dict[str, Any] = {}
        artifacts: List[str] = []

        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            timeframe = str(config.get("timeframe", "15m"))
            direction = str(config.get("direction", "long"))

            if not symbol or not exchange:
                raise ValueError("Config must include 'symbol' and 'exchange'")

            self.set_context(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                direction=direction,
                model="enhanced_xgb_meso_regime",
            )

            tprint_info(f"🚀 Starting Enhanced XGB Meso Regime for {symbol} on {exchange}")

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Generate Enhanced Features
            tprint_info("🛠️ Generating Enhanced XGB Meso Regime features...")
            feature_df = self._generate_enhanced_meso_features(market_data, config)
            
            tprint_info(f"✅ Enhanced XGB Meso Regime features: {len(feature_df.columns)} columns")

            if not config.get("is_batch_run", False):
                feature_df_reset = feature_df.reset_index().rename(columns={feature_df.index.name or "index": "timestamp"})
                features_path = self._save_artifact(
                    data=feature_df_reset,
                    artifact_name="enhanced_xgb_meso_features",
                    artifact_type="data",
                    data_category="features",
                    metadata={"source": market_source, "enhanced": True}
                )
                artifacts.append(features_path)

            # 3. Generate Labels
            tprint_info("🎯 Generating Enhanced XGB Meso Regime labels...")
            labels = self._create_meso_labels(market_data)

            # Align features and labels
            common_index = feature_df.index.intersection(labels.index)
            X = feature_df.loc[common_index]
            y = labels.loc[common_index]

            # Clean data
            valid_mask = X.notna().all(axis=1) & y.notna()
            X = X.loc[valid_mask]
            y = y.loc[valid_mask]

            if len(X) < 500:
                raise RuntimeError(f"Insufficient valid samples: {len(X)} < 500")

            tprint_info(f"📊 Training Data: {len(X)} samples, {len(X.columns)} features")

            # 4. Train Enhanced Model with MI Optimization
            tprint_info("🤖 Training Enhanced XGB Meso Regime model with MI optimization...")
            model, model_metrics = self._train_enhanced_meso_model(X, y, config)
            
            metrics.update(model_metrics)

            # 5. Generate Predictions
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            # 6. Create Standardized Output
            standardized_output = self._create_standardized_output(
                X, y, predictions, probabilities, symbol, exchange, timeframe, direction
            )

            # 7. Save Artifacts
            artifact_name = f"enhanced_xgb_meso_predictions_{timeframe}"
            metadata = SpecialistDataInterface.create_standard_metadata(
                specialist_name="EnhancedXGBMesoRegimeStep",
                config=config,
                metrics=metrics,
                mi_score=metrics['mi_score'],
                hsic_score=0.0
            )
            
            
            artifact_path = self._save_artifact(
                data=standardized_output,
                artifact_name=artifact_name,
                artifact_type="data",
                data_category="predictions",
                metadata=metadata
            )
            artifacts.append(artifact_path)

            # 8. Run Enhanced Diagnostics
            tprint_info("🔍 Running Enhanced Diagnostics...")
            diagnostics_result = self.run_enhanced_diagnostics(symbol, exchange, timeframe, direction)
            
            if diagnostics_result.get('success', False):
                compliance_report = diagnostics_result['compliance_report']
                ensemble_compatibility = diagnostics_result['ensemble_compatibility']
                
                tprint_success(f"✅ Enhanced Diagnostics Complete:")
                tprint_info(f"   MI Score: {compliance_report['metrics']['mi_score']:.4f}")
                tprint_info(f"   Requirements Met: {compliance_report['requirements_met']}/3")
                tprint_info(f"   Ensemble Ready: {ensemble_compatibility['ensemble_ready']}")
                
                metrics.update({
                    'enhanced_mi_score': compliance_report['metrics']['mi_score'],
                    'enhanced_requirements_met': compliance_report['requirements_met'],
                    'enhanced_ensemble_ready': ensemble_compatibility['ensemble_ready']
                })

            # 9. Final Summary
            execution_time = time.time() - start_time
            metrics["execution_time"] = execution_time
            metrics["n_samples"] = len(standardized_output)

            tprint_success(f"✅ Enhanced XGB Meso Regime completed in {execution_time:.2f}s")
            tprint_info(f"📊 Final Metrics: MI={metrics.get('mi_score', 0):.4f}, AUC={metrics.get('auc', 0):.3f}")

            return {
                "success": True,
                "metrics": metrics,
                "n_samples": len(standardized_output),
                "features": list(X.columns),
                "artifacts": artifacts,
                "diagnostics": diagnostics_result,
                "mi_history": self.mi_history,
                "training_metrics": self.training_metrics,
                "execution_time": execution_time
            }

        except Exception as e:
            self.logger.exception(f"❌ Enhanced XGB Meso Regime step failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_standardized_output(self, features: pd.DataFrame, labels: pd.Series,
                                  predictions: np.ndarray, probabilities: np.ndarray,
                                  symbol: str, exchange: str, timeframe: str, direction: str) -> pd.DataFrame:
        """Create standardized output structure."""
        standardized = pd.DataFrame(index=features.index)
        standardized['timestamp'] = features.index
        standardized['specialist_prediction'] = predictions
        standardized['specialist_probability'] = probabilities
        standardized['target_label'] = labels
        
        # Add original features for reference
        for col in features.columns[:20]:  # Limit to first 20 features
            standardized[f'feature_{col}'] = features[col]
        
        return standardized
    
    def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> Tuple[pd.DataFrame, str]:
        """Load market data with caching."""
        # This would be implemented based on the actual data loading mechanism
        # Using alternative data loading approach
        return load_market_data(config['symbol'], config['exchange'], timeframe)
