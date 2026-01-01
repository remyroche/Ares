"""
Enhanced ML Microstructure Step with MI Improvements

This enhanced version implements:
- Enhanced feature generation for MI improvement
- Real-time MI monitoring during training
- Hyperparameter optimization for MI > 0.02 target
- Data structure standardization
- Binary output enforcement
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score

from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.versioned_artifacts import VersionedArtifactStore
from src.training.steps.market_analysis.ml_risk_regime_step import MLRiskRegimeStepHMM
from src.training.steps.market_analysis.specialist_diagnostics_mixin_enhanced_v2 import (
    SpecialistDiagnosticsMixinEnhancedV2
)
from src.training.steps.market_analysis.specialist_diagnostics_mixin import SpecialistDiagnosticsMixin
from src.training.steps.market_analysis.enhanced_feature_generators import MIOptimizedFeaturePipeline
from src.training.steps.market_analysis.specialist_interface import SpecialistDataInterface
from src.training.steps.market_analysis.specialist_data_standard import SpecialistType

logger = logging.getLogger(__name__)


class EnhancedMLMicrostructureStep(MLRiskRegimeStepHMM, SpecialistDiagnosticsMixinEnhancedV2, SpecialistDiagnosticsMixin):

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
    Enhanced Microstructure Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Hyperparameter optimization for MI > 0.02
    - Standardized data structure output
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_ml_microstructure_step"):
        """Initialize the enhanced microstructure step."""
        super().__init__(step_name=step_name)
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        self.logger = logger.getChild("EnhancedMLMicrostructureStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        
    def _compute_enhanced_structural_optimized_horizon_optimized_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute enhanced momentum features with MI improvements."""
        features = pd.DataFrame(index=df.index)
        
        # Basic momentum features
        returns = df['close'].pct_change()
        
        # Multi-timeframe momentum
        for window in [60,80,100]:
            features[f'momentum_{window}'] = returns.rolling(window).sum()
            features[f'momentum_accel_{window}'] = returns.rolling(window).sum() - returns.rolling(window*2).sum()
            features[f'momentum_vol_{window}'] = returns.rolling(window).std()
        
        # RSI-like momentum
        for window in [60,80,100]:
            gains = returns.clip(lower=0)
            losses = -returns.clip(upper=0)
            avg_gains = gains.rolling(window).mean()
            avg_losses = losses.rolling(window).mean()
            rs = avg_gains / (avg_losses + 1e-8)
            features[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        # Price momentum indicators
        for window in [10, 20, 50]:
            sma = df['close'].rolling(window).mean()
            features[f'price_microstructure_{window}'] = (df['close'] - sma) / sma
            features[f'price_microstructure_cross_{window}'] = (df['close'] > sma).astype(int)
        
        # Volume momentum
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change()
            for window in [5, 10, 20]:
                features[f'volume_microstructure_{window}'] = volume_change.rolling(window).sum()
                features[f'volume_price_microstructure_{window}'] = (returns * volume_change).rolling(window).sum()
        
        return features
    
    def _generate_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate enhanced microstructure features with manual feature engineering."""
        # Basic momentum features
        momentum_features = self._compute_enhanced_microstructure_features(df)
        
        # Enhanced features from pipeline
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'microstructure', {'enhanced_features': True}
        )
        
        # Manual feature engineering for microstructure analysis
        manual_features = self._create_manual_microstructure_enhanced_features(df, enhanced_features)
        
        # Combine all features
        all_features = [momentum_features, enhanced_features, manual_features]
        
        # Combine all features with manual redundancy reduction
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            
            # Manual redundancy reduction and feature selection
            combined_features = self._apply_manual_microstructure_feature_selection(combined_features)
            
            # Remove duplicates and clean
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            combined_features = combined_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            return combined_features
        
        return pd.DataFrame(index=df.index)
    
    def _create_manual_microstructure_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features for microstructure analysis."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # 1. Enhanced bid-ask spread proxies
            # Range-based spread
            range_spread = (high - low) / close
            manual_features['range_spread'] = range_spread
            
            # Spread volatility
            spread_volatility = range_spread.rolling(20).std()
            manual_features['spread_volatility'] = spread_volatility
            
            # Spread persistence
            spread_persistence = (range_spread > range_spread.rolling(50).mean()).rolling(10).sum()
            manual_features['spread_persistence'] = spread_persistence
            
            # 2. Order flow imbalance features
            # Volume-weighted price impact
            price_impact = returns * volume
            order_flow_imbalance = price_impact.rolling(10).sum()
            manual_features['order_flow_imbalance'] = order_flow_imbalance
            
            # Order flow direction
            order_flow_direction = np.sign(order_flow_imbalance)
            manual_features['order_flow_direction'] = order_flow_direction
            
            # Order flow intensity
            order_flow_intensity = abs(order_flow_imbalance) / (volume.rolling(10).mean() + 1e-8)
            manual_features['order_flow_intensity'] = order_flow_intensity
            
            # 3. Market depth proxies
            # Volume-depth ratio
            volume_depth = volume / (range_spread + 1e-8)
            manual_features['volume_depth'] = volume_depth
            
            # Depth volatility
            depth_volatility = volume_depth.rolling(20).std()
            manual_features['depth_volatility'] = depth_volatility
            
            # Depth persistence
            depth_persistence = (volume_depth > volume_depth.rolling(50).mean()).rolling(10).sum()
            manual_features['depth_persistence'] = depth_persistence
            
            # 4. Price efficiency measures
            # Price efficiency (how efficiently price incorporates information)
            price_efficiency = abs(returns.rolling(10).mean()) / (returns.rolling(10).std() + 1e-8)
            manual_features['price_efficiency'] = price_efficiency
            
            # Market efficiency (inverse of noise)
            market_noise = returns.rolling(20).std() / abs(returns.rolling(20).mean() + 1e-8)
            market_efficiency = 1 / (market_noise + 1e-8)
            manual_features['market_efficiency'] = market_efficiency
            
            # Information ratio
            info_ratio = returns.rolling(20).mean() / (returns.rolling(20).std() + 1e-8)
            manual_features['information_ratio'] = info_ratio
            
            # 5. Microstructure regime features
            # Microstructure stress (high spread + low depth)
            spread_stress = (range_spread > range_spread.rolling(100).mean()).astype(int)
            depth_stress = (volume_depth < volume_depth.rolling(100).mean()).astype(int)
            microstructure_stress = spread_stress + depth_stress
            manual_features['microstructure_stress'] = microstructure_stress
            
            # Microstructure quality (low spread + high depth)
            spread_quality = (range_spread < range_spread.rolling(100).mean()).astype(int)
            depth_quality = (volume_depth > volume_depth.rolling(100).mean()).astype(int)
            microstructure_quality = spread_quality + depth_quality
            manual_features['microstructure_quality'] = microstructure_quality
            
            # 6. Tick-like features (simulated)
            # Simulated tick frequency (based on volume changes)
            volume_change = volume.diff().abs()
            tick_frequency = (volume_change > volume_change.rolling(20).mean()).astype(int)
            manual_features['tick_frequency'] = tick_frequency
            
            # Tick intensity
            tick_intensity = volume_change / (volume_change.rolling(20).mean() + 1e-8)
            manual_features['tick_intensity'] = tick_intensity
            
            # 7. Liquidity measures
            # Liquidity ratio (volume / price movement)
            liquidity_ratio = volume / (abs(returns) + 1e-8)
            manual_features['liquidity_ratio'] = liquidity_ratio
            
            # Liquidity volatility
            liquidity_volatility = liquidity_ratio.rolling(20).std()
            manual_features['liquidity_volatility'] = liquidity_volatility
            
            # Liquidity stress
            liquidity_stress = (liquidity_ratio < liquidity_ratio.rolling(100).mean()).astype(int)
            manual_features['liquidity_stress'] = liquidity_stress
            
            # 8. Market impact features
            # Price impact coefficient
            impact_coefficient = returns.rolling(20).corr(volume)
            manual_features['impact_coefficient'] = impact_coefficient
            
            # Impact asymmetry (up vs down moves)
            up_moves = returns[returns > 0]
            down_moves = returns[returns < 0]
            up_volume = volume[returns > 0]
            down_volume = volume[returns < 0]
            
            up_impact = up_moves.rolling(20).corr(up_volume) if len(up_moves) > 20 else 0
            down_impact = down_moves.rolling(20).corr(down_volume) if len(down_moves) > 20 else 0
            impact_asymmetry = up_impact - down_impact
            manual_features['impact_asymmetry'] = impact_asymmetry
            
            # 9. Microstructure momentum features
            # Spread momentum
            spread_momentum = range_spread.rolling(10).mean()
            manual_features['spread_momentum'] = spread_momentum
            
            # Depth momentum
            depth_momentum = volume_depth.rolling(10).mean()
            manual_features['depth_momentum'] = depth_momentum
            
            # Efficiency momentum
            efficiency_momentum = price_efficiency.rolling(10).mean()
            manual_features['efficiency_momentum'] = efficiency_momentum
            
            # 10. Composite microstructure indicators
            # Microstructure health index
            health_index = (
                0.3 * (range_spread < range_spread.rolling(100).mean()).astype(int) +
                0.3 * (volume_depth > volume_depth.rolling(100).mean()).astype(int) +
                0.2 * (price_efficiency > price_efficiency.rolling(100).mean()).astype(int) +
                0.2 * (liquidity_ratio > liquidity_ratio.rolling(100).mean()).astype(int)
            )
            manual_features['microstructure_health'] = health_index
            
            # Microstructure stability index
            stability_index = (
                0.25 * (spread_volatility < spread_volatility.rolling(100).mean()).astype(int) +
                0.25 * (depth_volatility < depth_volatility.rolling(100).mean()).astype(int) +
                0.25 * (abs(impact_coefficient) < abs(impact_coefficient).rolling(100).mean()).astype(int) +
                0.25 * (liquidity_volatility < liquidity_volatility.rolling(100).mean()).astype(int)
            )
            manual_features['microstructure_stability'] = stability_index
            
        return manual_features
    
    def _apply_manual_microstructure_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply manual feature selection for microstructure features."""
        if features.empty:
            return features
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant microstructure features")
        
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
            self.logger.info(f"Removed {len(set(to_drop))} redundant microstructure features: {list(set(to_drop))}")
        
        # Keep only the most informative features (limit to top 30 by variance)
        if len(features.columns) > 30:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(30).index
            features = features[top_features]
            self.logger.info(f"Limited microstructure features to top 30 by variance")
        
        return features
    
    def _create_microstructure_labels(self, df: pd.DataFrame, lookforward: int = 35) -> pd.Series:
        """Create momentum persistence labels."""
        returns = df['close'].pct_change()
        
        # Future momentum
        future_returns = returns.shift(-lookforward).rolling(lookforward).sum()
        
        # Binary label: positive future momentum
        labels = (future_returns > returns.rolling(25).std() * 0.5).astype(int)
        
        return labels
    
    def _compute_mi_during_training(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                  X_val: pd.DataFrame, y_val: pd.Series,
                                  model_predictions: np.ndarray) -> Dict[str, float]:
        """Compute MI metrics during training for monitoring."""
        mi_metrics = {}
        
        try:
            # Feature MI to target
            feature_mi_scores = []
            for col in X_train.select_dtypes(include=[np.number]).columns:
                mi_score = mutual_info_regression(
                    X_train[col].values.reshape(-1, 1), y_train.values
                )[0]
                feature_mi_scores.append(mi_score)
            
            if feature_mi_scores:
                mi_metrics['avg_feature_mi'] = np.mean(feature_mi_scores)
                mi_metrics['max_feature_mi'] = np.max(feature_mi_scores)
                mi_metrics['high_mi_features'] = sum(1 for mi in feature_mi_scores if mi > 0.02)
            
            # Prediction MI to target
            mi_metrics['prediction_mi'] = mutual_info_regression(
                model_predictions.reshape(-1, 1), y_val.values
            )[0]
            
            # MI improvement tracking
            self.mi_history.append(mi_metrics['prediction_mi'])
            
        except Exception as e:
            self.logger.warning(f"MI computation failed: {e}")
            mi_metrics = {'prediction_mi': 0.0, 'avg_feature_mi': 0.0}
        
        return mi_metrics
    
    def _optimize_hyperparameters_for_mi(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters specifically for MI improvement."""
        best_params = {}
        best_mi = 0.0
        
        # Parameter grid for MI optimization
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
        
        for params in self._generate_param_combinations(param_grid, max_combinations=20):
            mi_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Train model with current parameters
                model = lgb.LGBMClassifier(
                    objective='binary',
                    random_state=42,
                    verbose=-1,
                    **params
                )
                
                model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                         callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
                
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
                
                tprint_info(f"🔥 New best MI: {avg_mi:.4f} with params: {params}")
        
        tprint_success(f"✅ Best hyperparameters found: MI = {best_mi:.4f}")
        return best_params, best_mi
    

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
    
    def _train_enhanced_microstructure_model(self, features: pd.DataFrame, labels: pd.Series) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
        """Train enhanced momentum model with MI optimization."""
        
        # Optimize hyperparameters for MI
        tprint_info("🔧 Optimizing hyperparameters for MI improvement...")
        best_params, best_mi = self._optimize_hyperparameters_for_mi(features, labels)
        
        # Time series split for final training
        n_splits = 5
        split_idx = len(features) // (n_splits + 1)
        
        models = []
        mi_scores = []
        auc_scores = []
        
        for i in range(n_splits):
            train_start = i * split_idx
            train_end = (i + 2) * split_idx
            val_end = (i + 3) * split_idx
            
            if val_end > len(features):
                break
            
            X_train = features.iloc[train_start:train_end]
            y_train = labels.iloc[train_start:train_end]
            X_val = features.iloc[train_end:val_end]
            y_val = labels.iloc[train_end:val_end]
            
            # Train model with optimized parameters
            model = lgb.LGBMClassifier(
                objective='binary',
                random_state=42 + i,
                verbose=-1,
                **best_params
            )
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
            
            # Evaluate
            val_pred = model.predict_proba(X_val)[:, 1]
            
            # Compute MI
            mi_score = mutual_info_regression(
                val_pred.reshape(-1, 1), y_val.values
            )[0]
            mi_scores.append(mi_score)
            
            # Compute AUC
            auc = roc_auc_score(y_val, val_pred)
            auc_scores.append(auc)
            
            models.append(model)
            
            # Store training metrics
            self.training_metrics.append({
                'fold': i,
                'mi_score': mi_score,
                'auc_score': auc,
                'n_features': len(X_train.columns)
            })
        
        # Select best model based on MI
        best_idx = np.argmax(mi_scores)
        best_model = models[best_idx]
        
        metrics = {
            'mi_score': np.mean(mi_scores),
            'mi_std': np.std(mi_scores),
            'auc': np.mean(auc_scores),
            'auc_std': np.std(auc_scores),
            'best_mi': np.max(mi_scores),
            'best_auc': np.max(auc_scores),
            'n_features': len(features.columns),
            'optimization_params': best_params
        }
        
        return best_model, metrics
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute enhanced momentum persistence step."""
        try:
            symbol = str(config.get("symbol", "ETHUSDT"))
            exchange = str(config.get("exchange", "binance"))
            timeframe = str(config.get("timeframe", "15m"))
            direction = str(config.get("direction", "long"))

            tprint_info(f"🚀 Starting Enhanced Microstructure for {symbol}")

            # Load market data
            market_data = self._load_market_data(config, timeframe)
            
            # Generate enhanced features
            tprint_info("🛠️ Generating enhanced momentum features...")
            features = self._generate_enhanced_features(market_data)
            
            # Create labels
            labels = self._create_microstructure_labels(market_data)
            
            # Align features and labels
            common_index = features.index.intersection(labels.index)
            features = features.loc[common_index]
            labels = labels.loc[common_index]
            
            # Clean data
            valid_mask = ~(features.isna().any(axis=1)) & ~(labels.isna())
            features = features[valid_mask]
            labels = labels[valid_mask]
            
            if len(features) < 500:
                raise ValueError(f"Insufficient data: {len(features)} samples")
            
            tprint_info(f"📊 Training data: {len(features)} samples, {len(features.columns)} features")
            
            # Train enhanced model
            tprint_info("🤖 Training enhanced momentum model with MI optimization...")
            model, metrics = self._train_enhanced_microstructure_model(features, labels)
            
            # Generate predictions
            predictions = model.predict(features)
            probabilities = model.predict_proba(features)[:, 1]
            
            # Create standardized output
            output_df = self._create_standardized_output(
                features, labels, predictions, probabilities, symbol, exchange, timeframe, direction
            )
            
            # Save artifacts
            artifact_name = f"enhanced_microstructure_persistence_{timeframe}"
            metadata = SpecialistDataInterface.create_standard_metadata(
                specialist_name="EnhancedMLMicrostructureStep",
                config=config,
                metrics=metrics,
                mi_score=metrics['mi_score'],
                hsic_score=0.0  # Not computed for this implementation
            )
            
            
            # DEBUG: Check artifact saving setup
            print(f"🐛 DEBUG: About to save artifact: {artifact_name}")
            print(f"🐛 DEBUG: Output df shape: {output_df.shape}")
            print(f"🐛 DEBUG: Artifact router type: {type(self.artifact_router)}")
            print(f"🐛 DEBUG: Versioned store available: {hasattr(self, '_versioned_store') and self._versioned_store is not None}")
            if hasattr(self, '_versioned_store') and self._versioned_store is not None:
                print(f"🐛 DEBUG: Versioned store type: {type(self._versioned_store)}")
            
            self.artifact_router.save(
                data=output_df,
                artifact_name=artifact_name,
                artifact_type="data",
                data_category="predictions",
                metadata=metadata
            )
            
            # Run enhanced diagnostics
            diagnostics_result = self.run_enhanced_diagnostics(symbol, exchange, timeframe, direction)
            
            # Final summary
            tprint_success(f"✅ Enhanced Microstructure completed:")
            tprint_info(f"   MI Score: {metrics['mi_score']:.4f} (target: >0.02)")
            tprint_info(f"   AUC: {metrics['auc']:.3f}")
            tprint_info(f"   Features: {metrics['n_features']}")
            
            return {
                "success": True,
                "metrics": metrics,
                "n_samples": len(output_df),
                "features": list(features.columns),
                "artifact_name": artifact_name,
                "diagnostics": diagnostics_result,
                "mi_history": self.mi_history,
                "training_metrics": self.training_metrics
            }
            
        except Exception as e:
            self.logger.exception(f"❌ Enhanced Microstructure step failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_standardized_output(self, features: pd.DataFrame, labels: pd.Series,
                                  predictions: np.ndarray, probabilities: np.ndarray,
                                  symbol: str, exchange: str, timeframe: str, direction: str) -> pd.DataFrame:
        """Create standardized output structure."""
        output_df = pd.DataFrame(index=features.index)
        output_df['timestamp'] = features.index
        output_df['specialist_prediction'] = predictions
        output_df['specialist_probability'] = probabilities
        output_df['target_label'] = labels
        
        # Add original features for reference
        for col in features.columns[:20]:  # Limit to first 20 features
            output_df[f'feature_{col}'] = features[col]
        
        return output_df
    
    def _load_market_data(self, config: Dict[str, Any], timeframe: str) -> pd.DataFrame:
        """Load market data - placeholder implementation."""
        # This would be implemented based on the actual data loading mechanism
        # Using alternative data loading approach
            # market_data = self._load_alternative_market_data(config, timeframe)
        return load_market_data(config['symbol'], config['exchange'], timeframe)
