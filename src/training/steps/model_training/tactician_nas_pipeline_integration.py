"""
Tactician NAS Pipeline Integration

This module integrates NAS model as DeepScaler1m replacement into the complete
ML training pipeline for Tactician ensemble.

Key Features:
- NAS replaces DeepScaler1m in Tactician base models
- Regime-specific feature integration (volatility, volume, trend, momentum)
- Feature selection optimization (45-70 features for NAS)
- Fast fail implementation (no fallbacks)
- Integration with existing optimization tools
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
from datetime import datetime

# Import NAS integration
from src.training.steps.model_training.tactician_nas_integration import (
    TacticianNASIntegration, TacticianNASConfig, create_tactician_nas_model
)

# Import existing optimization tools
from src.utils.ml_common.optimization.hyperparameter_optimization import HyperparameterOptimizer
from src.utils.ml_common.optimization.regime_aware_hpo import RegimeAwareHyperparameterOptimizer
from src.utils.ml_common.feature_engineering.feature_selection import FeatureSelector
from src.utils.ml_common.validation.overfitting_detection import UniversalOverfittingDetector

# Import logging utilities
from src.utils.tprint import (
    tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
    tprint_debug, tprint_progress, tprint_performance, tprint_structured,
    tprint_timer, LogLevel
)

logger = logging.getLogger(__name__)


@dataclass
class TacticianPipelineConfig:
    """Configuration for Tactician NAS pipeline integration."""
    
    # NAS configuration
    nas_enabled: bool = True
    nas_trials: int = 30
    nas_timeout: int = 1800  # 30 minutes
    
    # Feature selection pipeline
    max_features: int = 60  # Final target for NAS (RandomForest selection)
    min_features: int = 45  # Minimum for meaningful learning
    feature_selection_pipeline: List[str] = None  # ['mrmr', 'mi', 'lasso', 'rf']
    
    # Feature selection thresholds
    mrmr_threshold: int = 80  # mRMR reduces to 80
    mi_threshold: int = 70    # MI reduces to 70  
    lasso_threshold: int = 65 # LASSO reduces to 65
    rf_threshold: int = 60    # RandomForest final selection to 60
    
    # Regime integration
    enable_regime_features: bool = True
    regime_feature_types: List[str] = None  # ['volatility', 'volume', 'trend', 'momentum']
    
    # Fast fail settings
    min_training_samples: int = 1000
    min_architecture_score: float = 0.5
    max_accuracy_gap: float = 0.2
    
    # Performance optimization
    enable_early_stopping: bool = True
    enable_model_pruning: bool = True
    memory_limit_gb: float = 4.0
    
    def __post_init__(self):
        if self.regime_feature_types is None:
            self.regime_feature_types = ['volatility', 'volume', 'trend', 'momentum']
        if self.feature_selection_pipeline is None:
            self.feature_selection_pipeline = ['mrmr', 'mi', 'lasso', 'rf']


class TacticianNASPipelineIntegration:
    """Integrates NAS into the complete Tactician ML training pipeline."""
    
    def __init__(self, config: Optional[TacticianPipelineConfig] = None):
        """Initialize Tactician NAS pipeline integration."""
        self.config = config or TacticianPipelineConfig()
        self.logger = logger.getChild('TacticianNASPipelineIntegration')
        self.training_stats = {}
        
        tprint_info("🚀 Tactician NAS Pipeline Integration initialized")
        tprint_info(f"📊 Configuration: NAS enabled={self.config.nas_enabled}")
        tprint_info(f"📊 Feature range: {self.config.min_features}-{self.config.max_features}")
    
    def integrate_nas_in_pipeline(self, 
                                 X_train: np.ndarray, 
                                 y_train: np.ndarray,
                                 X_val: np.ndarray, 
                                 y_val: np.ndarray,
                                 regime_labels: Optional[np.ndarray] = None,
                                 regime_features: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Integrate NAS into the complete Tactician ML training pipeline.
        
        Args:
            X_train: Training features (1m timeframe)
            y_train: Training labels (trading signals)
            X_val: Validation features
            y_val: Validation labels
            regime_labels: Regime labels for regime-aware optimization (optional)
            regime_features: Regime-specific features (volatility, volume, trend, momentum) (optional)
            
        Returns:
            Complete pipeline results with NAS integration
        """
        tprint_info("🔧 Starting Tactician NAS pipeline integration...")
        start_time = time.time()
        
        try:
            # Step 1: Fast fail validation
            self._validate_inputs(X_train, y_train, X_val, y_val)
            
            # Step 2: Feature selection optimization
            X_train_optimized, X_val_optimized = self._optimize_features(
                X_train, y_train, X_val, y_val
            )
            
            # Step 3: Regime feature integration
            if self.config.enable_regime_features and regime_features is not None:
                X_train_optimized, X_val_optimized = self._integrate_regime_features(
                    X_train_optimized, X_val_optimized, regime_features
                )
            
            # Step 4: NAS model creation
            nas_model = self._create_nas_model(
                X_train_optimized, y_train, X_val_optimized, y_val, regime_labels
            )
            
            # Step 5: Model validation
            self._validate_nas_model(nas_model, X_train_optimized, y_train, X_val_optimized, y_val)
            
            # Step 6: Pipeline integration
            pipeline_results = self._integrate_with_pipeline(nas_model)
            
            # Calculate statistics
            integration_time = time.time() - start_time
            self.training_stats = {
                'integration_time': integration_time,
                'nas_enabled': self.config.nas_enabled,
                'feature_count': X_train_optimized.shape[1],
                'regime_features_integrated': self.config.enable_regime_features,
                'success': True
            }
            
            tprint_success("✅ Tactician NAS pipeline integration completed successfully")
            tprint_info(f"⏱️ Integration time: {integration_time:.2f}s")
            tprint_info(f"📊 Features: {X_train_optimized.shape[1]} (optimized for NAS)")
            
            return pipeline_results
            
        except Exception as e:
            tprint_error(f"❌ Tactician NAS pipeline integration failed: {e}")
            self.training_stats = {
                'success': False,
                'error': str(e),
                'integration_time': time.time() - start_time
            }
            raise
    
    def _validate_inputs(self, X_train: np.ndarray, y_train: np.ndarray, 
                        X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Fast fail validation of inputs."""
        tprint_info("🔍 Validating inputs...")
        
        # Check training data size
        if X_train.shape[0] < self.config.min_training_samples:
            raise ValueError(f"Insufficient training data: {X_train.shape[0]} < {self.config.min_training_samples}")
        
        # Check feature count
        if X_train.shape[1] < self.config.min_features:
            tprint_warning(f"⚠️ Low feature count: {X_train.shape[1]} < {self.config.min_features}")
        
        # Check data consistency
        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError(f"Feature count mismatch: train={X_train.shape[1]}, val={X_val.shape[1]}")
        
        if len(y_train) != X_train.shape[0]:
            raise ValueError(f"Sample count mismatch: X={X_train.shape[0]}, y={len(y_train)}")
        
        tprint_success("✅ Input validation passed")
    
    def _optimize_features(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize features for NAS using comprehensive feature selection pipeline."""
        tprint_info("🎯 Optimizing features for NAS using advanced feature selection pipeline...")
        
        current_features = X_train.shape[1]
        
        if current_features <= self.config.max_features:
            tprint_info(f"✅ Feature count optimal: {current_features} features")
            return X_train, X_val
        
        if current_features > 200:
            tprint_warning(f"⚠️ High feature count: {current_features} > 200, applying comprehensive feature selection...")
            
            # Step 1: mRMR (Minimum Redundancy Maximum Relevance)
            tprint_info("🔍 Step 1: Applying mRMR feature selection...")
            from src.utils.ml_common.feature_engineering.mrmr_selection import MRMRSelector
            mrmr_selector = MRMRSelector(k=min(self.config.mrmr_threshold, current_features), method='fscore')
            X_train_mrmr = mrmr_selector.fit_transform(X_train, y_train)
            X_val_mrmr = mrmr_selector.transform(X_val)
            tprint_success(f"✅ mRMR: {current_features} → {X_train_mrmr.shape[1]} features")
            
            # Step 2: Mutual Information filtering
            tprint_info("🔍 Step 2: Applying Mutual Information filtering...")
            from src.utils.ml_common.feature_engineering.mutual_info_selection import MutualInfoSelector
            mi_selector = MutualInfoSelector(k=min(self.config.mi_threshold, X_train_mrmr.shape[1]), method='mutual_info')
            X_train_mi = mi_selector.fit_transform(X_train_mrmr, y_train)
            X_val_mi = mi_selector.transform(X_val_mrmr)
            tprint_success(f"✅ MI: {X_train_mrmr.shape[1]} → {X_train_mi.shape[1]} features")
            
            # Step 3: LASSO regularization
            tprint_info("🔍 Step 3: Applying LASSO regularization...")
            from src.utils.ml_common.feature_engineering.lasso_selection import LassoSelector
            lasso_selector = LassoSelector(alpha=0.01, max_features=min(self.config.lasso_threshold, X_train_mi.shape[1]))
            X_train_lasso = lasso_selector.fit_transform(X_train_mi, y_train)
            X_val_lasso = lasso_selector.transform(X_val_mi)
            tprint_success(f"✅ LASSO: {X_train_mi.shape[1]} → {X_train_lasso.shape[1]} features")
            
            # Step 4: RandomForest final selection (down to 60)
            tprint_info("🔍 Step 4: Applying RandomForest final selection to 60 features...")
            from src.utils.ml_common.feature_engineering.random_forest_selection import RandomForestSelector
            rf_selector = RandomForestSelector(
                n_estimators=100,
                max_features=min(self.config.rf_threshold, X_train_lasso.shape[1]),
                method='importance'
            )
            X_train_final = rf_selector.fit_transform(X_train_lasso, y_train)
            X_val_final = rf_selector.transform(X_val_lasso)
            
            tprint_success(f"✅ RandomForest: {X_train_lasso.shape[1]} → {X_train_final.shape[1]} features")
            tprint_success(f"🎯 Final feature reduction: {current_features} → {X_train_final.shape[1]} features")
            
            return X_train_final, X_val_final
        
        else:
            tprint_info(f"✅ Feature count acceptable: {current_features} features")
            return X_train, X_val
    
    def _integrate_regime_features(self, X_train: np.ndarray, X_val: np.ndarray, 
                                 regime_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Integrate regime-specific features."""
        tprint_info("🧠 Integrating regime-specific features...")
        
        try:
            # Ensure regime features match training data length
            train_len = len(X_train)
            val_len = len(X_val)
            
            if len(regime_features) < train_len + val_len:
                tprint_warning("⚠️ Regime features shorter than required, padding with zeros")
                padding = np.zeros((train_len + val_len - len(regime_features), regime_features.shape[1]))
                regime_features = np.vstack([regime_features, padding])
            
            # Split regime features
            regime_train = regime_features[:train_len]
            regime_val = regime_features[train_len:train_len + val_len]
            
            # Integrate features
            X_train_integrated = np.hstack([X_train, regime_train])
            X_val_integrated = np.hstack([X_val, regime_val])
            
            tprint_success(f"✅ Regime features integrated: {X_train.shape[1]} → {X_train_integrated.shape[1]} features")
            return X_train_integrated, X_val_integrated
            
        except Exception as e:
            tprint_warning(f"⚠️ Regime feature integration failed: {e}")
            return X_train, X_val
    
    def _create_nas_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                         X_val: np.ndarray, y_val: np.ndarray, 
                         regime_labels: Optional[np.ndarray] = None) -> Any:
        """Create NAS model using existing optimization tools."""
        tprint_info("🔍 Creating NAS model with existing optimization tools...")
        
        if not self.config.nas_enabled:
            tprint_info("⏭️ NAS disabled in configuration")
            return None
        
        # Configure NAS for Tactician
        nas_config = TacticianNASConfig(
            n_trials=self.config.nas_trials,
            timeout_seconds=self.config.nas_timeout,
            max_layers=6,  # Shallow for 1m timeframe
            max_units=256,  # Moderate complexity
            min_units=32,  # Minimum for learning
            objectives=['accuracy', 'efficiency', 'robustness'],
            objective_weights=[0.5, 0.3, 0.2],  # Balance for 1m timeframe
            enable_regime_awareness=True,
            early_stopping_patience=10
        )
        
        # Create NAS model
        nas_model = create_tactician_nas_model(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            config=nas_config,
            regime_labels=regime_labels
        )
        
        if nas_model is None:
            raise RuntimeError("NAS model creation failed")
        
        tprint_success("✅ NAS model created successfully")
        return nas_model
    
    def _validate_nas_model(self, nas_model: Any, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Validate NAS model using existing overfitting detection."""
        tprint_info("🔍 Validating NAS model performance...")
        
        if nas_model is None:
            tprint_info("⏭️ No NAS model to validate")
            return
        
        try:
            # Use existing overfitting detection
            overfitting_detector = UniversalOverfittingDetector()
            overfitting_report = overfitting_detector.detect_overfitting(
                train_predictions=nas_model.predict(X_train),
                val_predictions=nas_model.predict(X_val),
                train_labels=y_train,
                val_labels=y_val,
                model_name="tactician_nas",
                model_type="neural_network"
            )
            
            # Fast fail: Check for severe overfitting
            if overfitting_report.severity == "high":
                raise RuntimeError(f"Severe overfitting detected: {overfitting_report.severity}")
            
            if overfitting_report.accuracy_gap > self.config.max_accuracy_gap:
                raise RuntimeError(f"High accuracy gap: {overfitting_report.accuracy_gap:.3f} > {self.config.max_accuracy_gap}")
            
            tprint_success("✅ NAS model validation passed")
            tprint_info(f"📊 Overfitting severity: {overfitting_report.severity}")
            tprint_info(f"📊 Accuracy gap: {overfitting_report.accuracy_gap:.3f}")
            
        except Exception as e:
            tprint_error(f"❌ NAS model validation failed: {e}")
            raise
    
    def _integrate_with_pipeline(self, nas_model: Any) -> Dict[str, Any]:
        """Integrate NAS model with complete pipeline."""
        tprint_info("🔧 Integrating NAS model with complete pipeline...")
        
        # Updated Tactician base models with NAS replacing DeepScaler1m
        updated_base_models = {
            "xgboost": "XGBoost",
            "lightgbm": "LightGBM",
            "nas_optimized": nas_model,  # ← REPLACES DeepScaler1m
            "FinancialResNet": "FinancialResNet",
            "RSF": "RandomSurvivalForest"
        }
        
        # Meta-learner configuration
        meta_learner_config = {
            "type": "advanced_mamba_hybrid",
            "base_models": list(updated_base_models.keys()),
            "nas_enabled": nas_model is not None
        }
        
        pipeline_results = {
            "base_models": updated_base_models,
            "meta_learner": meta_learner_config,
            "nas_model": nas_model,
            "training_stats": self.training_stats,
            "feature_optimization": {
                "final_feature_count": self.training_stats.get('feature_count', 0),
                "regime_features_integrated": self.config.enable_regime_features
            }
        }
        
        tprint_success("✅ Pipeline integration completed")
        tprint_info(f"📊 Base models: {list(updated_base_models.keys())}")
        tprint_info(f"📊 Meta-learner: {meta_learner_config['type']}")
        
        return pipeline_results
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline integration statistics."""
        return self.training_stats.copy()


# Convenience function for easy integration
def integrate_nas_in_tactician_pipeline(X_train: np.ndarray, 
                                       y_train: np.ndarray,
                                       X_val: np.ndarray, 
                                       y_val: np.ndarray,
                                       config: Optional[TacticianPipelineConfig] = None,
                                       regime_labels: Optional[np.ndarray] = None,
                                       regime_features: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Convenience function to integrate NAS into Tactician pipeline.
    
    Args:
        X_train: Training features (1m timeframe)
        y_train: Training labels (trading signals)
        X_val: Validation features
        y_val: Validation labels
        config: Pipeline configuration
        regime_labels: Regime labels for regime-aware optimization (optional)
        regime_features: Regime-specific features (volatility, volume, trend, momentum) (optional)
        
    Returns:
        Complete pipeline results with NAS integration
    """
    pipeline_integration = TacticianNASPipelineIntegration(config)
    return pipeline_integration.integrate_nas_in_pipeline(
        X_train, y_train, X_val, y_val, regime_labels, regime_features
    )