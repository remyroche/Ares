"""
Enhanced XGB Macro Regime Step with MI Improvements

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


class EnhancedXGBMacroRegimeStep(SpecialistDiagnosticsMixinEnhancedV2, BaseStep):

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
    Enhanced XGB Macro Regime Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Macro-specific feature engineering
    - Hyperparameter optimization for MI > 0.02
    - Data structure standardization
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_xgb_macro_regime_step"):
        """Initialize the enhanced XGB macro regime step."""
        super().__init__()
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("EnhancedXGBMacroRegimeStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        
    def _generate_enhanced_macro_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate enhanced macro features with MI improvements."""
        # Import original macro features
        try:
            from src.feature_generation.categories.macro_regime_features import generate_macro_regime_features
            macro_features = generate_macro_regime_features(df, config)
        except ImportError:
            # Fallback if original features not available
            macro_features = pd.DataFrame(index=df.index)
        
        # Generate enhanced features
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'macro_regime', config
        )
        
        # Macro-specific enhancements
        macro_enhanced = self._add_macro_specific_features(df, macro_features)
        
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
        all_features = pd.concat([macro_features, enhanced_features, macro_enhanced], axis=1)
        
        # Remove duplicates and clean
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        all_features = all_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        
        return all_features
    
    def _add_macro_specific_features(self, df: pd.DataFrame, macro_features: pd.DataFrame) -> pd.DataFrame:
        """Add macro-specific enhanced features."""
        features = pd.DataFrame(index=df.index)
        
        # Enhanced macro analysis
        if 'close' in df.columns:
            close = df['close']
            returns = close.pct_change()
            
            # Multi-timeframe macro analysis
            for window in [10, 20, 50, 100]:
                # Macro trend
                macro_trend = returns.rolling(window).mean()
                features[f'macro_trend_{window}'] = macro_trend
                
                # Macro momentum
                macro_momentum = returns.rolling(window).sum()
                features[f'macro_momentum_{window}'] = macro_momentum
                
                # Macro acceleration
                macro_acceleration = macro_momentum.diff()
                features[f'macro_acceleration_{window}'] = macro_acceleration
                
                # Macro volatility
                macro_volatility = returns.rolling(window).std()
                features[f'macro_volatility_{window}'] = macro_volatility
                
                # Macro risk-adjusted returns
                risk_adjusted = macro_trend / macro_volatility
                features[f'macro_risk_adjusted_{window}'] = risk_adjusted
                
                # Macro regime strength
                regime_strength = abs(macro_trend) / macro_volatility
                features[f'macro_regime_strength_{window}'] = regime_strength
                
                # Macro persistence
                macro_persistence = (macro_trend > 0).rolling(window).mean()
                features[f'macro_persistence_{window}'] = macro_persistence
                
                # Macro regime transitions
                regime_transition = macro_persistence.diff()
                features[f'macro_regime_transition_{window}'] = regime_transition
            
            # Cross-timeframe macro analysis
            for short_window in [10, 20]:
                for long_window in [50, 100]:
                    short_trend = returns.rolling(short_window).mean()
                    long_trend = returns.rolling(long_window).mean()
                    
                    # Trend alignment
                    trend_alignment = (short_trend * long_trend)
                    features[f'trend_alignment_{short_window}_{long_window}'] = trend_alignment
                    
                    # Trend divergence
                    trend_divergence = abs(short_trend - long_trend)
                    features[f'trend_divergence_{short_window}_{long_window}'] = trend_divergence
                    
                    # Momentum convergence
                    momentum_convergence = (short_trend > 0) == (long_trend > 0)
                    features[f'momentum_convergence_{short_window}_{long_window}'] = momentum_convergence.astype(int)
            
            # Macro cycle analysis
            for window in [20, 50, 100]:
                # Cycle detection using autocorrelation
                cycle_strength = returns.rolling(window).apply(lambda x: x.autocorr())
                features[f'cycle_strength_{window}'] = cycle_strength
                
                # Cycle phase
                cycle_phase = np.arctan2(returns.rolling(window).mean(), returns.rolling(window).std())
                features[f'cycle_phase_{window}'] = cycle_phase
                
                # Cycle amplitude
                cycle_amplitude = returns.rolling(window).std()
                features[f'cycle_amplitude_{window}'] = cycle_amplitude
            
            # Macro extreme analysis
            for window in [20, 50]:
                # Extreme returns
                extreme_returns = returns.rolling(window).apply(lambda x: (x.abs() > x.std() * 2).sum())
                features[f'extreme_returns_{window}'] = extreme_returns
                
                # Tail risk
                tail_risk = returns.rolling(window).apply(lambda x: (x < x.quantile(0.05)).mean())
                features[f'tail_risk_{window}'] = tail_risk
                
                # Volatility clustering
                volatility_clustering = returns.rolling(window).std().rolling(window).corr(returns.rolling(window).std())
                features[f'volatility_clustering_{window}'] = volatility_clustering
        
        # Volume-macro relationship
        if 'volume' in df.columns and 'close' in df.columns:
            volume = df['volume']
            returns = df['close'].pct_change()
            
            # Volume-adjusted macro analysis
            volume_ma = volume.rolling(35).mean()
            volume_anomaly = volume / volume_ma
            
            for window in [10, 20, 50]:
                # Volume-weighted macro trend
                volume_weighted_trend = (returns * volume).rolling(window).sum()
                features[f'volume_weighted_trend_{window}'] = volume_weighted_trend
                
                # Volume-macro correlation
                volume_macro_corr = returns.rolling(window).corr(volume)
                features[f'volume_macro_corr_{window}'] = volume_macro_corr
                
                # Volume confirmation of macro moves
                volume_confirmation = (volume_anomaly > 1.5) & (abs(returns.rolling(window).mean()) > returns.rolling(window*2).std() * 0.5)
                features[f'volume_confirmation_{window}'] = volume_confirmation.astype(int)
                
                # Volume-macro divergence
                volume_divergence = abs(volume_macro_corr) < 0.3
                features[f'volume_divergence_{window}'] = volume_divergence.astype(int)
        
        # Support/resistance macro analysis
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            close = df['close']
            high = df['high']
            low = df['low']
            
            for window in [20, 50, 100]:
                # Macro support/resistance
                rolling_max = close.rolling(window).max()
                rolling_min = close.rolling(window).min()
                
                # Distance to macro levels
                features[f'macro_distance_to_resistance_{window}'] = (rolling_max - close) / rolling_max
                features[f'macro_distance_to_support_{window}'] = (close - rolling_min) / rolling_max
                
                # Macro SR strength
                features[f'macro_sr_strength_{window}'] = (rolling_max - rolling_min) / close
                
                # Macro level breaches
                features[f'macro_resistance_breach_{window}'] = (close > rolling_max.shift(1)).astype(int)
                features[f'macro_support_breach_{window}'] = (close < rolling_min.shift(1)).astype(int)
                
                # Macro range expansion
                range_expansion = (rolling_max - rolling_min) / (rolling_max - rolling_min).rolling(window*2).mean()
                features[f'macro_range_expansion_{window}'] = range_expansion
                
                # Macro range contraction
                range_contraction = range_expansion < 0.8
                features[f'macro_range_contraction_{window}'] = range_contraction.astype(int)
        
        # Time-based macro patterns
        if isinstance(df.index, pd.DatetimeIndex):
            features['hour_of_day'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_london_session'] = ((df.index.hour >= 8) & (df.index.hour <= 16)).astype(int)
            features['is_ny_session'] = ((df.index.hour >= 13) & (df.index.hour <= 21)).astype(int)
            features['is_asia_session'] = ((df.index.hour >= 0) & (df.index.hour <= 8)).astype(int)
            
            # Session overlaps
            features['is_london_ny_overlap'] = ((df.index.hour >= 13) & (df.index.hour <= 16)).astype(int)
            
            # Weekend effects on macro
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            # Month-end macro effects
            features['is_month_end'] = (df.index.day >= 28).astype(int)
            features['is_month_start'] = (df.index.day <= 5).astype(int)
            
            # Quarterly effects
            features['is_quarter_end'] = (df.index.month % 3 == 0).astype(int)
            
            # Seasonal patterns
            features['month'] = df.index.month
            features['quarter'] = df.index.month // 4 + 1
        
        return features
    
    def _create_macro_labels(self, df: pd.DataFrame, lookforward: int = 50) -> pd.Series:
        """Create macro labels based on macro regime patterns."""
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            
            # Multi-timeframe macro analysis
            macro_trend_20 = returns.rolling(35).mean()
            macro_trend_50 = returns.rolling(70).mean()
            
            # Macro regime strength
            regime_strength = abs(macro_trend_20) / returns.rolling(35).std()
            
            # Future macro trend
            future_macro_trend = returns.shift(-lookforward).rolling(35).mean()
            
            # Macro regime change detection
            regime_change = abs(future_macro_trend - macro_trend_20)
            regime_change_threshold = returns.rolling(35).std() * 0.5
            
            # Label: 1 for significant macro regime change
            labels = (regime_change > regime_change_threshold).astype(int)
            
            return labels
        else:
            # Fallback to simple trend-based labels
            returns = df['close'].pct_change()
            future_returns = returns.shift(-lookforward)
            labels = (future_returns.abs() > returns.rolling(35).std() * 1.5).astype(int)
            return labels
    

    def save(self, artifact_name: str, data, artifact_type: str = "data", data_category: str = "predictions"):
        """Custom save method for enhanced specialists."""
        try:
            # Use versioned store directly if available
            if hasattr(self, '_versioned_store') and self._versioned_store is not None:
                context = {
                    'symbol': self._current_context.get('symbol', 'UNKNOWN'),
                    'exchange': self._current_context.get('exchange', 'binance'),
                    'timeframe': self._current_context.get('timeframe', '15m'),
                    'direction': self._current_context.get('direction', 'long'),
                    'model': self._current_context.get('model', 'analyst'),
                    'step_name': self.step_name,
                }
                self._versioned_store.save(
                    artifact_name=artifact_name,
                    data=data,
                    artifact_type=artifact_type,
                    data_category=data_category,
                    context=context
                )
                self.logger.info(f"✅ Saved {artifact_name} to versioned store")
            else:
                self.logger.warning(f"⚠️ Cannot save {artifact_name}: no versioned store available")
        except Exception as e:
            self.logger.error(f"❌ Failed to save {artifact_name}: {e}")

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
    
    def _train_enhanced_macro_model(self, features: pd.DataFrame, labels: pd.Series, 
                                    config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train enhanced macro model with MI optimization."""
        
        # Optimize hyperparameters for MI
        tprint_info("🔧 Optimizing XGBoost hyperparameters for MI improvement...")
        best_params, best_mi = self._optimize_xgb_hyperparameters_for_mi(features, labels)
        
        # Create temporal split config
        temporal_config = create_temporal_split_config_for_pipeline(
            symbol=config.get("symbol", "ETHUSDT"),
            exchange=config.get("exchange", "binance"),
            timeframe=config.get("timeframe", "15m"),
            direction=config.get("direction", "long"),
            n_splits=config.get("macro_n_splits", 5),
            walk_forward_type="rolling",
            test_size_ratio=config.get("macro_test_size_ratio", 0.2),
            min_train_samples=config.get("macro_min_train_samples", 500),
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
        """Execute enhanced XGB macro regime step."""
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
                model="enhanced_xgb_macro_regime",
            )

            tprint_info(f"🚀 Starting Enhanced XGB Macro Regime for {symbol} on {exchange}")

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Generate Enhanced Features
            tprint_info("🛠️ Generating Enhanced XGB Macro Regime features...")
            feature_df = self._generate_enhanced_macro_features(market_data, config)
            
            tprint_info(f"✅ Enhanced XGB Macro Regime features: {len(feature_df.columns)} columns")

            if not config.get("is_batch_run", False):
                feature_df_reset = feature_df.reset_index().rename(columns={feature_df.index.name or "index": "timestamp"})
                features_path = self._save_artifact(
                    data=feature_df_reset,
                    artifact_name="enhanced_xgb_macro_features",
                    artifact_type="data",
                    data_category="features",
                    metadata={"source": market_source, "enhanced": True}
                )
                artifacts.append(features_path)

            # 3. Generate Labels
            tprint_info("🎯 Generating Enhanced XGB Macro Regime labels...")
            labels = self._create_macro_labels(market_data)

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
            tprint_info("🤖 Training Enhanced XGB Macro Regime model with MI optimization...")
            model, model_metrics = self._train_enhanced_macro_model(X, y, config)
            
            metrics.update(model_metrics)

            # 5. Generate Predictions
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            # 6. Create Standardized Output
            standardized_output = self._create_standardized_output(
                X, y, predictions, probabilities, symbol, exchange, timeframe, direction
            )

            # 7. Save Artifacts
            artifact_name = f"enhanced_xgb_macro_predictions_{timeframe}"
            metadata = SpecialistDataInterface.create_standard_metadata(
                specialist_name="EnhancedXGBMacroRegimeStep",
                config=config,
                metrics=metrics,
                mi_score=metrics['mi_score'],
                hsic_score=0.0
            )
            
            
            # DEBUG: Check artifact saving setup
            print(f"🐛 DEBUG: About to save artifact: {artifact_name}")
            print(f"🐛 DEBUG: Output df shape: {output_df.shape}")
            print(f"🐛 DEBUG: Artifact router type: {type(self.artifact_router)}")
            print(f"🐛 DEBUG: Versioned store available: {hasattr(self, '_versioned_store') and self._versioned_store is not None}")
            if hasattr(self, '_versioned_store') and self._versioned_store is not None:
                print(f"🐛 DEBUG: Versioned store type: {type(self._versioned_store)}")
            
            self.artifact_router.save(
                artifact_name=artifact_name,
                data=standardized_output,
                metadata=metadata
            )
            artifacts.append(artifact_name)

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

            tprint_success(f"✅ Enhanced XGB Macro Regime completed in {execution_time:.2f}s")
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
            self.logger.exception(f"❌ Enhanced XGB Macro Regime step failed: {e}")
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
