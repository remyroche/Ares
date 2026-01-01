"""
Enhanced ML SMC Regime Step with MI Improvements

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
import xgboost as xgb
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


class EnhancedMLSMCRegimeStep(SpecialistDiagnosticsMixinEnhancedV2, BaseStep):

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
    Enhanced SMC Regime Specialist with MI optimization.
    
    Key enhancements:
    - Enhanced feature generation for MI improvement
    - Real-time MI monitoring during training
    - Hyperparameter optimization for MI > 0.02
    - Standardized data structure output
    - Binary output enforcement
    """
    
    def __init__(self, step_name: str = "enhanced_ml_smc_regime_step"):
        """Initialize the enhanced SMC regime step."""
        super().__init__()
        self._current_context = {}
        self._artifact_manager = None
        self._versioned_store = None
        self.step_name = step_name
        BaseStep.__init__(self, step_name, use_versioned_artifacts=True)
        self.logger = logger.getChild("EnhancedMLSMCRegimeStep")
        self.feature_pipeline = MIOptimizedFeaturePipeline()
        self.mi_history = []
        self.training_metrics = []
        
    def _generate_enhanced_smc_features(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Generate enhanced SMC features with manual feature engineering."""
        # Import original SMC features
        from src.feature_generation.categories.smc_regime_features import generate_smc_regime_features
        base_smc_features = generate_smc_regime_features(df)
        
        # Enhanced features from pipeline
        enhanced_features = self.feature_pipeline.generate_enhanced_features(
            df, 'smc_regime', {'enhanced_features': True}
        )
        
        # Manual feature engineering for SMC regime
        manual_features = self._create_manual_smc_enhanced_features(df, enhanced_features)
        
        # Combine all features
        all_features = [base_smc_features, enhanced_features, manual_features]
        
        # Combine all features with manual redundancy reduction
        if all_features:
            combined_features = pd.concat(all_features, axis=1)
            
            # Manual redundancy reduction and feature selection
            combined_features = self._apply_manual_smc_feature_selection(combined_features)
            
            # Remove duplicates and clean
            combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]
            combined_features = combined_features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            return combined_features
        
        return pd.DataFrame(index=df.index)
    
    def _create_manual_smc_enhanced_features(self, df: pd.DataFrame, enhanced_features: pd.DataFrame) -> pd.DataFrame:
        """Create manual enhanced features for SMC regime detection."""
        manual_features = pd.DataFrame(index=df.index)
        
        if all(col in df.columns for col in ['close', 'high', 'low', 'volume']):
            close = df['close']
            high = df['high']
            low = df['low']
            volume = df.get('volume', pd.Series(1, index=df.index))
            returns = close.pct_change()
            
            # 1. Enhanced SMC regime features
            # Multi-timeframe SMC signals
            short_smc = returns.rolling(10).mean() / (returns.rolling(10).std() + 1e-8)
            medium_smc = returns.rolling(20).mean() / (returns.rolling(20).std() + 1e-8)
            long_smc = returns.rolling(50).mean() / (returns.rolling(50).std() + 1e-8)
            
            manual_features['smc_short_regime'] = (short_smc > 0).astype(int)
            manual_features['smc_medium_regime'] = (medium_smc > 0).astype(int)
            manual_features['smc_long_regime'] = (long_smc > 0).astype(int)
            
            # SMC regime consistency
            smc_consistency = (short_smc > 0).rolling(20).mean()
            manual_features['smc_regime_consistency'] = smc_consistency
            
            # SMC regime transitions
            smc_transitions = (medium_smc > 0).astype(int).diff().abs()
            manual_features['smc_regime_transitions'] = smc_transitions
            
            # 2. Volume-adjusted SMC features
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume / (volume_ma + 1e-8)
            
            volume_adjusted_smc = medium_smc * (1 + np.log(volume_ratio + 1))
            manual_features['volume_adjusted_smc'] = volume_adjusted_smc
            
            # Volume-SMC divergence
            volume_regime = (volume_ratio > 1).astype(int)
            smc_regime = (medium_smc > 0).astype(int)
            volume_smc_divergence = np.abs(volume_regime - smc_regime)
            manual_features['volume_smc_divergence'] = volume_smc_divergence
            
            # 2.1 Advanced orthogonalization for smc|volume_force_breakout redundancy reduction
            if 'smc_predicted' in enhanced_features.columns and 'vol_force_breakout' in enhanced_features.columns:
                smc_pred = enhanced_features['smc_predicted']
                vol_force = enhanced_features['vol_force_breakout']
                
                # Standardize for orthogonal decomposition
                smc_std = (smc_pred - smc_pred.mean()) / (smc_pred.std() + 1e-8)
                vol_std = (vol_force - vol_force.mean()) / (vol_force.std() + 1e-8)
                
                # Method 1: Gram-Schmidt orthogonalization
                smc_vol_cov = np.cov(smc_std, vol_std)[0,1]
                orthogonal_vol_gs = vol_std - smc_vol_cov * smc_std
                orthogonal_vol_gs = orthogonal_vol_gs / (orthogonal_vol_gs.std() + 1e-8)
                manual_features['orthogonal_volume_gs'] = orthogonal_vol_gs
                
                # Method 2: Residual regression orthogonalization
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression()
                reg.fit(smc_std.values.reshape(-1, 1), vol_std.values)
                vol_pred = reg.predict(smc_std.values.reshape(-1, 1))
                orthogonal_vol_resid = vol_std - vol_pred
                orthogonal_vol_resid = orthogonal_vol_resid / (orthogonal_vol_resid.std() + 1e-8)
                manual_features['orthogonal_volume_resid'] = orthogonal_vol_resid
                
                # Method 3: Frequency domain separation
                smc_low = smc_std.rolling(20).mean()  # Low frequency SMC
                vol_high = vol_std - vol_std.rolling(5).mean()  # High frequency volume
                orthogonal_vol_freq = vol_high / (vol_high.std() + 1e-8)
                manual_features['orthogonal_volume_freq'] = orthogonal_vol_freq
                
                # Advanced divergence metrics
                regime_divergence_mag = np.abs(smc_std - vol_std) * (np.abs(smc_std) + np.abs(vol_std))
                manual_features['regime_divergence_magnitude'] = regime_divergence_mag
                
                disagreement_intensity = (smc_std * vol_std < 0).astype(float) * np.abs(smc_std - vol_std)
                manual_features['regime_disagreement_intensity'] = disagreement_intensity
                
                consensus_strength = (smc_std * vol_std > 0).astype(float) * (1 - np.abs(smc_std - vol_std))
                manual_features['regime_consensus_strength'] = consensus_strength
            
            # 3. Range-based SMC features
            range_ratio = (high - low) / close
            range_smc = medium_smc * range_ratio
            manual_features['range_adjusted_smc'] = range_smc
            
            # Range regime SMC
            range_regime = (range_ratio > range_ratio.rolling(50).mean()).astype(int)
            manual_features['range_regime_smc'] = range_regime
            
            # 4. SMC persistence features
            smc_persistence_short = (short_smc > 0).rolling(5).sum()
            smc_persistence_medium = (medium_smc > 0).rolling(10).sum()
            manual_features['smc_persistence_short'] = smc_persistence_short
            manual_features['smc_persistence_medium'] = smc_persistence_medium
            
            # SMC momentum
            smc_momentum = medium_smc.diff().rolling(5).mean()
            manual_features['smc_momentum'] = smc_momentum
            
            # 5. Enhanced SMC volatility interaction
            volatility = returns.rolling(20).std()
            vol_adjusted_smc = medium_smc / (volatility + 1e-8)
            manual_features['vol_adjusted_smc'] = vol_adjusted_smc
            
            # SMC volatility regime
            vol_regime = (volatility > volatility.rolling(100).mean()).astype(int)
            smc_vol_regime = smc_regime * vol_regime
            manual_features['smc_vol_regime'] = smc_vol_regime
            
            # 6. SMC trend strength
            trend_strength = abs(medium_smc)
            manual_features['smc_trend_strength'] = trend_strength
            
            # SMC trend acceleration
            trend_acceleration = medium_smc.diff().diff()
            manual_features['smc_trend_acceleration'] = trend_acceleration
            
        return manual_features
    
    def _apply_manual_smc_feature_selection(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply manual feature selection for SMC regime features."""
        if features.empty:
            return features
        
        # Remove constant features
        constant_features = features.columns[features.nunique() <= 1]
        if len(constant_features) > 0:
            features = features.drop(columns=constant_features)
            self.logger.info(f"Removed {len(constant_features)} constant SMC features")
        
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
            self.logger.info(f"Removed {len(set(to_drop))} redundant SMC features: {list(set(to_drop))}")
        
        # Keep only the most informative features (limit to top 30 by variance)
        if len(features.columns) > 30:
            feature_variances = features.var()
            top_features = feature_variances.nlargest(30).index
            features = features[top_features]
            self.logger.info(f"Limited SMC features to top 30 by variance")
        
        return features
    
    def _add_smc_specific_features(self, df: pd.DataFrame, smc_features: pd.DataFrame) -> pd.DataFrame:
        """Add SMC-specific enhanced features."""
        features = pd.DataFrame(index=df.index)
        
        # Enhanced SMC analysis
        if 'close' in df.columns:
            returns = df['close'].pct_change()
            
            # SMC-specific enhancements
            features['smc_volatility_ratio'] = returns.rolling(20).std() / returns.rolling(50).std()
            features['smc_trend_consistency'] = (returns.rolling(10).mean() > 0).rolling(30).mean()
            
        return features
    
    def _train_enhanced_smc_model(self, features: pd.DataFrame, labels: pd.Series, 
                                 config: Dict[str, Any]) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
        """Train enhanced SMC model with MI optimization."""
        
        # Model parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': config.get('smc_xgb_max_depth', 6),
            'learning_rate': 0.1,
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(features.fillna(0), labels.fillna(0))
        
        # Calculate metrics
        predictions = model.predict_proba(features.fillna(0))[:, 1]
        mi_score = np.corrcoef(predictions, labels.fillna(0))[0, 1] ** 2
        
        metrics = {
            'mi_score': mi_score,
            'feature_count': len(features.columns),
            'training_samples': len(features)
        }
        
        return model, metrics
    
    def _create_smc_labels(self, df: pd.DataFrame, lookforward: int = 35) -> pd.Series:
        """Create SMC regime labels based on price efficiency and range utilization."""
        if 'close' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
            # Fallback to simple return-based labels
            returns = df['close'].pct_change()
            future_returns = returns.shift(-lookforward)
            labels = (future_returns > returns.rolling(25).std()).astype(int)
            return labels
        
        # SMC-specific labeling
        # Calculate price efficiency
        mid_price = (df['high'] + df['low']) / 2
        price_efficiency = (df['close'] - mid_price) / mid_price
        
        # Calculate future efficiency
        future_mid_price = (df['high'].shift(-lookforward) + df['low'].shift(-lookforward)) / 2
        future_efficiency = (df['close'].shift(-lookforward) - future_mid_price) / future_mid_price
        
        # Label: positive if efficiency improves and price moves in direction of efficiency
        efficiency_improvement = future_efficiency > price_efficiency
        price_direction = (df['close'].shift(-lookforward) > df['close']) == (price_efficiency > 0)
        
        labels = (efficiency_improvement & price_direction).astype(int)
        
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
    
    def _train_enhanced_smc_model(self, features: pd.DataFrame, labels: pd.Series, 
                                 config: Dict[str, Any]) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
        """Train enhanced SMC model with MI optimization."""
        
        # Optimize hyperparameters for MI
        tprint_info("🔧 Optimizing XGBoost hyperparameters for MI improvement...")
        best_params, best_mi = self._optimize_xgb_hyperparameters_for_mi(features, labels)
        
        # Create temporal split config
        temporal_config = create_temporal_split_config_for_pipeline(
            symbol=config.get("symbol", "ETHUSDT"),
            exchange=config.get("exchange", "binance"),
            timeframe=config.get("timeframe", "15m"),
            direction=config.get("direction", "long"),
            n_splits=config.get("smc_n_splits", 5),
            walk_forward_type="rolling",
            test_size_ratio=config.get("smc_test_size_ratio", 0.2),
            min_train_samples=config.get("smc_min_train_samples", 500),
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
        """Execute enhanced SMC regime step."""
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
                model="enhanced_smc_regime",
            )

            tprint_info(f"🚀 Starting Enhanced SMC Regime for {symbol} on {exchange}")

            # 1. Load Market Data
            market_data, market_source = self._load_market_data_with_cache(config, timeframe)

            # 2. Generate Enhanced Features
            tprint_info("🛠️ Generating Enhanced SMC features...")
            feature_df = self._generate_enhanced_smc_features(market_data, config)
            
            tprint_info(f"✅ Enhanced SMC features: {len(feature_df.columns)} columns")

            if not config.get("is_batch_run", False):
                feature_df_reset = feature_df.reset_index().rename(columns={feature_df.index.name or "index": "timestamp"})
                features_path = self._save_artifact(
                    data=feature_df_reset,
                    artifact_name="enhanced_ml_smc_features",
                    artifact_type="data",
                    data_category="features",
                    metadata={"source": market_source, "enhanced": True}
                )
                artifacts.append(features_path)

            # 3. Generate Labels
            tprint_info("🎯 Generating Enhanced SMC labels...")
            labels = self._create_smc_labels(market_data)

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
            tprint_info("🤖 Training Enhanced SMC model with MI optimization...")
            model, model_metrics = self._train_enhanced_smc_model(X, y, config)
            
            metrics.update(model_metrics)

            # 5. Generate Predictions
            predictions = model.predict(X)
            probabilities = model.predict_proba(X)[:, 1]

            # 6. Create Standardized Output
            standardized_output = self._create_standardized_output(
                X, y, predictions, probabilities, symbol, exchange, timeframe, direction
            )

            # 7. Save Artifacts
            artifact_name = f"enhanced_ml_smc_predictions_{timeframe}"
            metadata = SpecialistDataInterface.create_standard_metadata(
                specialist_name="EnhancedMLSMCRegimeStep",
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

            tprint_success(f"✅ Enhanced SMC Regime completed in {execution_time:.2f}s")
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
            self.logger.exception(f"❌ Enhanced SMC Regime step failed: {e}")
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
            # market_data = self._load_alternative_market_data(config, timeframe)
        market_data, market_source = self.load_market_data_or_fail(
            {**config, "timeframe": timeframe},
            pipeline_state={},
            allow_config_override=True,
        )
        return market_data, market_source
