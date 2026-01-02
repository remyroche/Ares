"""
Enhanced ML Liquidity Regime Step with MI Improvements

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
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from src.training.steps.base_step import BaseStep
from src.utils.tprint import (
    tprint,
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
from src.utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


class EnhancedMLLiquidityRegimeStep(SpecialistDiagnosticsMixinEnhancedV2, BaseStep):

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
    Enhanced Liquidity Regime Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Hyperparameter optimization for MI > 0.02
    - Standardized data structure output
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_ml_liquidity_regime_step"):
        """Initialize the enhanced liquidity regime step."""
        super().__init__()
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("EnhancedMLLiquidityRegimeStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        
    def _generate_enhanced_liquidity_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate enhanced liquidity features with manual feature engineering."""
        # Import original liquidity features
        from src.feature_generation.categories.liquidity_regime_features import generate_liquidity_regime_features
        base_liquidity_features = generate_liquidity_regime_features(df, config)
        
        # Enhanced features from pipeline
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'liquidity_regime', {'enhanced_features': True}
        )
        
        # Manual feature engineering for liquidity regime
        manual_features = self._create_manual_liquidity_enhanced_features(df, enhanced_features)
        
        # Combine all features
        all_features = [base_liquidity_features, enhanced_features, manual_features]
        
        # Combine all features with manual redundancy reduction
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            
            # Manual redundancy reduction and feature selection
            combined_features = self._apply_manual_liquidity_feature_selection(combined_features)
            
            # Remove duplicates and clean
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            combined_features = combined_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            return combined_features
        
        return pd.DataFrame(index=df.index)
    
    def _create_manual_liquidity_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features for liquidity regime detection."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # 1. Enhanced liquidity regime features
            # Multi-timeframe liquidity signals
            volume_ma_short = volume.rolling(10).mean()
            volume_ma_medium = volume.rolling(20).mean()
            volume_ma_long = volume.rolling(50).mean()
            
            volume_ratio_short = volume / (volume_ma_short + 1e-8)
            volume_ratio_medium = volume / (volume_ma_medium + 1e-8)
            volume_ratio_long = volume / (volume_ma_long + 1e-8)
            
            manual_features['liquidity_regime_short'] = (volume_ratio_short > 1).astype(int)
            manual_features['liquidity_regime_medium'] = (volume_ratio_medium > 1).astype(int)
            manual_features['liquidity_regime_long'] = (volume_ratio_long > 1).astype(int)
            
            # Liquidity regime consistency
            liquidity_consistency = (volume_ratio_medium > 1).rolling(20).mean()
            manual_features['liquidity_regime_consistency'] = liquidity_consistency
            
            # Liquidity regime transitions
            liquidity_transitions = (volume_ratio_medium > 1).astype(int).diff().abs()
            manual_features['liquidity_regime_transitions'] = liquidity_transitions
            
            # 2. Price-liquidity interaction features
            # Volume-adjusted price changes
            volume_adjusted_returns = returns * volume_ratio_medium
            manual_features['volume_adjusted_returns'] = volume_adjusted_returns
            
            # Liquidity-adjusted volatility
            volatility = returns.rolling(20).std()
            liquidity_adjusted_vol = volatility / (volume_ratio_medium + 1e-8)
            manual_features['liquidity_adjusted_volatility'] = liquidity_adjusted_vol
            
            # Price-liquidity divergence
            price_regime = (returns.rolling(20).mean() > 0).astype(int)
            liquidity_regime = (volume_ratio_medium > 1).astype(int)
            price_liquidity_divergence = np.abs(price_regime - liquidity_regime)
            manual_features['price_liquidity_divergence'] = price_liquidity_divergence
            
            # 3. Range-based liquidity features
            range_ratio = (high - low) / close
            range_volume = range_ratio * volume_ratio_medium
            manual_features['range_volume_liquidity'] = range_volume
            
            # Range-liquidity regime
            range_regime = (range_ratio > range_ratio.rolling(50).mean()).astype(int)
            manual_features['range_liquidity_regime'] = range_regime
            
            # 4. Liquidity persistence features
            liquidity_persistence_short = (volume_ratio_short > 1).rolling(5).sum()
            liquidity_persistence_medium = (volume_ratio_medium > 1).rolling(10).sum()
            manual_features['liquidity_persistence_short'] = liquidity_persistence_short
            manual_features['liquidity_persistence_medium'] = liquidity_persistence_medium
            
            # Liquidity momentum
            liquidity_momentum = volume_ratio_medium.diff().rolling(5).mean()
            manual_features['liquidity_momentum'] = liquidity_momentum
            
            # 5. Enhanced liquidity volatility interaction
            vol_liquidity_regime = (liquidity_adjusted_vol > liquidity_adjusted_vol.rolling(100).mean()).astype(int)
            manual_features['vol_liquidity_regime'] = vol_liquidity_regime
            
            # Liquidity volatility strength
            liq_vol_strength = abs(liquidity_adjusted_vol)
            manual_features['liquidity_vol_strength'] = liq_vol_strength
            
            # 6. Microstructure liquidity features
            # Price impact estimation
            price_impact = abs(returns) / (volume_ratio_medium + 1e-8)
            manual_features['price_impact'] = price_impact
            
            # Liquidity depth proxy
            depth_proxy = volume / (range_ratio + 1e-8)
            manual_features['liquidity_depth'] = depth_proxy
            
            # Market efficiency indicator
            efficiency = abs(returns.rolling(10).mean()) / (volume_ratio_medium + 1e-8)
            manual_features['market_efficiency'] = efficiency
            
            # 7. Liquidity regime classification
            # High liquidity regime
            high_liquidity = (volume_ratio_medium > 1.5).astype(int)
            manual_features['high_liquidity_regime'] = high_liquidity
            
            # Low liquidity regime
            low_liquidity = (volume_ratio_medium < 0.5).astype(int)
            manual_features['low_liquidity_regime'] = low_liquidity
            
            # Liquidity stress indicator
            liquidity_stress = np.where(volume_ratio_medium < 0.3, 2, np.where(volume_ratio_medium > 2, 0, 1))
            manual_features['liquidity_stress'] = liquidity_stress
            
        return manual_features
    
    def _apply_manual_liquidity_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply manual feature selection for liquidity regime features."""
        if features.empty:
            return features
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant liquidity features")
        
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
                # Keep the feature that comes first alphabetically (deterministic)
                for correlated_feature in correlated_features.index:
                    if correlated_feature > column:  # Drop the later feature alphabetically
                        to_drop.append(correlated_feature)
        
        # Remove redundant features
        if to_drop:
            features = features.drop(columns=list(set(to_drop)))
            self.logger.info(f"Removed {len(set(to_drop))} redundant liquidity features: {list(set(to_drop))}")
        
        # Keep only the most informative features (limit to top 30 by variance)
        if len(features.columns) > 30:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(30).index
            features = features[top_features]
            self.logger.info(f"Limited liquidity features to top 30 by variance")
        
        return features
    
    def _add_liquidity_specific_features(self, df: pd.DataFrame, liquidity_features: pd.DataFrame) -> pd.DataFrame:
        """Add liquidity-specific enhanced features."""
        features = pd.DataFrame(index=df.index)
        
        # Enhanced volume analysis
        if 'volume' in df.columns and 'close' in df.columns:
            volume = df['volume']
            price_change = df['close'].pct_change()
            
            # Volume-price relationship enhancements
            features['volume_price_correlation_10'] = price_change.rolling(15).corr(volume)
            features['volume_price_correlation_20'] = price_change.rolling(25).corr(volume)
            features['volume_price_correlation_50'] = price_change.rolling(60).corr(volume)
            
            # Volume pattern recognition
            volume_ma = volume.rolling(25).mean()
            features['volume_pattern_accumulation'] = (volume > volume_ma * 1.5).astype(int)
            features['volume_pattern_distribution'] = (volume < volume_ma * 0.5).astype(int)
            features['volume_pattern_churning'] = ((volume >= volume_ma * 0.5) & (volume <= volume_ma * 1.5)).astype(int)
            
            # Volume efficiency metrics
            features['volume_efficiency_ratio'] = price_change.abs() / (volume + 1e-8)
            features['volume_efficiency_ma'] = features['volume_efficiency_ratio'].rolling(25).mean()
            
            # Volume momentum
            volume_change = volume.pct_change()
            features['volume_momentum_10'] = volume_change.rolling(15).sum()
            features['volume_momentum_20'] = volume_change.rolling(25).sum()
            features['volume_acceleration'] = volume_change.rolling(15).sum() - volume_change.rolling(25).sum()
        
        # Enhanced price analysis
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            high_low_range = df['high'] - df['low']
            close_price = df['close']
            
            # Range analysis
            range_ma = high_low_range.rolling(25).mean()
            features['range_expansion'] = high_low_range / range_ma
            features['range_contraction'] = (high_low_range < range_ma * 0.7).astype(int)
            features['range_breakout_up'] = (high_low_range > high_low_range.rolling(25).max().shift(1)).astype(int)
            features['range_breakout_down'] = (high_low_range < high_low_range.rolling(25).min().shift(1)).astype(int)
            
            # Price efficiency
            mid_price = (df['high'] + df['low']) / 2
            features['price_efficiency'] = (close_price - mid_price) / mid_price
            features['price_efficiency_ma'] = features['price_efficiency'].rolling(25).mean()
            
            # Support/resistance levels
            for window in [20, 50]:
                rolling_max = close_price.rolling(window).max()
                rolling_min = close_price.rolling(window).min()
                
                features[f'distance_to_resistance_{window}'] = (rolling_max - close_price) / rolling_max
                features[f'distance_to_support_{window}'] = (close_price - rolling_min) / rolling_max
                features[f'sr_strength_{window}'] = (rolling_max - rolling_min) / close_price
        
        # Market microstructure features
        if 'volume' in df.columns and 'close' in df.columns:
            # Volume profile analysis
            volume_ma = df['volume'].rolling(25).mean()
            price_change = df['close'].pct_change()
            
            features['volume_anomaly'] = df['volume'] / volume_ma
            features['volume_price_trend'] = (price_change * df['volume']).rolling(15).sum()
            
            # Order flow imbalance proxy
            features['order_flow_proxy'] = (price_change * df['volume']).rolling(15).sum()
            features['order_flow_persistence'] = (features['order_flow_proxy'] > 0).rolling(25).sum()
        
        return features
    
    def _create_liquidity_labels(self, df: pd.DataFrame, lookforward: int = 35) -> pd.Series:
        """Create liquidity regime labels based on volume and price patterns."""
        if 'volume' not in df.columns or 'close' not in df.columns:
            # Fallback to simple return-based labels
            returns = df['close'].pct_change()
            volume_change = df['volume'].pct_change()
            
            # Liquidity stress indicator
            liquidity_stress = returns.rolling(25).std() * volume_change.rolling(25).std()
            future_stress = liquidity_stress.shift(-lookforward)
            
            labels = (future_stress > liquidity_stress.quantile(0.8)).astype(int)
            return labels
        
        # Liquidity-specific labeling
        volume = df['volume']
        close_price = df['close']
        
        # Volume patterns
        volume_ma = volume.rolling(25).mean()
        volume_anomaly = volume / volume_ma
        
        # Price patterns
        price_change = close_price.pct_change()
        price_volatility = price_change.rolling(25).std()
        
        # Liquidity stress indicator
        liquidity_stress = price_volatility * volume_anomaly
        
        # Future liquidity stress
        future_stress = liquidity_stress.shift(-lookforward)
        
        # Label: positive if liquidity stress increases (potential regime change)
        labels = (future_stress > liquidity_stress.quantile(0.75)).astype(int)
        
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
    
    def _train_enhanced_liquidity_model(self, features: pd.DataFrame, labels: pd.Series, 
                                           config: Dict[str, Any]) -> Tuple[Any, Dict[str, float]]:
        """Train enhanced liquidity model with MI optimization."""
        
        # Optimize hyperparameters for MI
        tprint_info("🔧 Optimizing XGBoost hyperparameters for MI improvement...")
        best_params, best_mi = self._optimize_xgb_hyperparameters_for_mi(features, labels)
        
        # Create temporal split config
        temporal_config = create_temporal_split_config_for_pipeline(
            symbol=config.get("symbol", "ETHUSDT"),
            exchange=config.get("exchange", "binance"),
            timeframe=config.get("timeframe", "15m"),
            direction=config.get("direction", "long"),
            n_splits=config.get("liquidity_n_splits", 5),
            walk_forward_type="rolling",
            test_size_ratio=config.get("liquidity_test_size_ratio", 0.2),
            min_train_samples=config.get("liquidity_min_train_samples", 500),
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
        """Execute enhanced liquidity regime step."""
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
                model="enhanced_liquidity_regime",
            )

            tprint_info(f"🚀 Starting Enhanced Liquidity Regime for {symbol} on {exchange}")

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Generate Enhanced Features
            tprint_info("🛠️ Generating Enhanced Liquidity features...")
            feature_df = self._generate_enhanced_liquidity_features(market_data, config)
            
            tprint_info(f"✅ Enhanced Liquidity features: {len(feature_df.columns)} columns")

            if not config.get("is_batch_run", False):
                feature_df_reset = feature_df.reset_index().rename(columns={feature_df.index.name or "index": "timestamp"})
                features_path = self._save_artifact(
                    data=feature_df_reset,
                    artifact_name="enhanced_ml_liquidity_features",
                    artifact_type="data",
                    data_category="features",
                    metadata={"source": market_source, "enhanced": True}
                )
                artifacts.append(features_path)

            # 3. Generate Labels
            tprint_info("🎯 Generating Enhanced Liquidity labels...")
            labels = self._create_liquidity_labels(market_data)

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
            tprint_info("🤖 Training Enhanced Liquidity model with MI optimization...")
            model, model_metrics = self._train_enhanced_liquidity_model(X, y, config)
            
            metrics.update(model_metrics)

            # 5. Generate Predictions
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            # 6. Create Standardized Output
            standardized_output = self._create_standardized_output(
                X, y, predictions, probabilities, symbol, exchange, timeframe, direction
            )

            # 7. Save Artifacts
            artifact_name = f"enhanced_ml_liquidity_predictions_{timeframe}"
            metadata = SpecialistDataInterface.create_standard_metadata(
                specialist_name="EnhancedMLLiquidityRegimeStep",
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

            tprint_success(f"✅ Enhanced Liquidity Regime completed in {execution_time:.2f}s")
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
            self.logger.exception(f"❌ Enhanced Liquidity Regime step failed: {e}")
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
    
def _load_market_data_with_cache(self, config: Dict[str, Any], timeframe: str) -> pd.DataFrame:
    """Load market data with caching."""
    # Use DataLoader to get market data
    data_loader = DataLoader()
    
    # Try to load appropriate data based on timeframe
    if timeframe == "1h":
        market_data = data_loader.load_ethusdt_1h_data()
    elif timeframe == "1m":
        market_data = data_loader.load_ethusdt_1m_data()
    else:
        # For other timeframes, try the general loading function
        market_data = data_loader.load_ethusdt_data_for_analysis(timeframe=timeframe)
        if isinstance(market_data, dict):
            market_data = market_data.get('data')
    
    return market_data
