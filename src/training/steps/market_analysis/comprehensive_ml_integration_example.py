"""
Comprehensive ML Common Utilities Integration Example

This example demonstrates how to thoroughly use the extensive ML utilities
from src/utils/ml_common/ across the TAS, NAS, and Hybrid regime detection systems.

Key Features Demonstrated:
- Overfitting Prevention
- Hyperparameter Optimization (Grid, Bayesian, TPE)
- Advanced Cross-Validation (Temporal, Purged, Nested)
- Ensemble Methods and Stacking
- Feature Importance Analysis
- Data Drift Detection
- Model Monitoring and Safeguards
- Lookahead Protection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import time

# Import comprehensive ML common utilities
try:
    from src.utils.ml_common import (
        # Overfitting prevention
        OverfittingPrevention, OverfittingPreventionConfig,
        # Hyperparameter optimization
        HyperparameterOptimization, optimize_hyperparameters, create_search_space,
        # Validation
        UnifiedCrossValidator, perform_cross_validation, temporal_cross_validation,
        PurgedKFold, TemporalCrossValidator, StabilityAnalyzer,
        # Models and ensembles
        EnhancedModelFactory, MultiOutputModel, EnsembleManager,
        # Feature engineering
        FeatureImportanceAnalyzer, DataDriftDetector,
        # Monitoring and safeguards
        LookaheadProtection, MLTrainingSafeguards, RobustErrorHandler
    )
    ML_COMMON_AVAILABLE = True
except ImportError as e:
    print(f"ML common utilities not available: {e}")
    ML_COMMON_AVAILABLE = False

# Import target systems
from tas_regime.core.tas_engine import TreeArchitectureSearchEngine, TASEngineConfig
from nas_regime.core.enhanced_nas_engine import EnhancedNASEngine, NASSearchConfig
from hybrid_nas_tas_regime.enhanced_hybrid_orchestrator import EnhancedHybridOrchestrator

logger = logging.getLogger(__name__)


class ComprehensiveMLIntegration:
    """
    Comprehensive ML Common Utilities Integration Example.
    
    Demonstrates thorough usage of ML utilities across all regime detection systems.
    """
    
    def __init__(self):
        """Initialize comprehensive ML integration."""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize ML common utilities
        self._initialize_ml_utilities()
        
        # Initialize target systems
        self._initialize_target_systems()
        
        self.logger.info("✅ Comprehensive ML Integration initialized")
    
    def _initialize_ml_utilities(self):
        """Initialize all ML common utilities."""
        if not ML_COMMON_AVAILABLE:
            self.logger.warning("⚠️ ML common utilities not available")
            return
        
        try:
            # Initialize overfitting prevention
            self.overfitting_prevention = OverfittingPrevention(
                OverfittingPreventionConfig(
                    enable_early_stopping=True,
                    enable_cross_validation=True,
                    enable_regularization=True,
                    enable_ensemble_diversity=True,
                    early_stopping_patience=20,
                    cv_folds=5,
                    enable_performance_monitoring=True,
                    overfitting_threshold=0.1
                )
            )
            
            # Initialize hyperparameter optimization
            self.hpo_optimizer = HyperparameterOptimization({
                'enable_parallel': True,
                'max_workers': 4,
                'enable_monitoring': True,
                'use_nonlinear_optimization': True,
                'convergence': {
                    'improvement_threshold': 0.001,
                    'patience_trials': 20,
                    'variance_threshold': 0.01
                }
            })
            
            # Initialize validation framework
            self.validation_framework = UnifiedCrossValidator()
            
            # Initialize model factory
            self.model_factory = EnhancedModelFactory()
            
            # Initialize ensemble manager
            self.ensemble_manager = EnsembleManager()
            
            # Initialize feature importance analyzer
            self.feature_analyzer = FeatureImportanceAnalyzer()
            
            # Initialize data drift detector
            self.drift_detector = DataDriftDetector()
            
            # Initialize safeguards
            self.lookahead_protection = LookaheadProtection()
            self.training_safeguards = MLTrainingSafeguards()
            self.error_handler = RobustErrorHandler()
            
            self.logger.info("✅ All ML common utilities initialized")
            
        except Exception as e:
            self.logger.error(f"❌ ML utilities initialization failed: {e}")
    
    def _initialize_target_systems(self):
        """Initialize target regime detection systems."""
        try:
            # Initialize TAS engine
            tas_config = TASEngineConfig(
                enable_meta_learning=True,
                enable_hardware_optimization=True,
                enable_uncertainty_estimation=True,
                enable_regime_analysis=True,
                enable_real_time_adaptation=True
            )
            self.tas_engine = TreeArchitectureSearchEngine(tas_config)
            
            # Initialize NAS engine
            nas_config = NASSearchConfig(
                search_strategy=SearchStrategy.ENHANCED_BAYESIAN,
                population_size=50,
                max_generations=100,
                enable_multi_objective=True
            )
            self.nas_engine = EnhancedNASEngine(nas_config)
            
            # Initialize hybrid orchestrator
            from hybrid_nas_tas_regime.config.hybrid_regime_config import HybridRegimeConfig
            hybrid_config = HybridRegimeConfig(
                enable_multi_timeframe=True,
                use_unified_search=True,
                use_signal_generation=True
            )
            self.hybrid_orchestrator = EnhancedHybridOrchestrator(hybrid_config)
            
            self.logger.info("✅ Target systems initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Target systems initialization failed: {e}")
    
    def demonstrate_comprehensive_ml_usage(self, 
                                         market_data: pd.DataFrame,
                                         target_data: pd.Series) -> Dict[str, Any]:
        """
        Demonstrate comprehensive usage of ML common utilities.
        
        Args:
            market_data: Market data for analysis
            target_data: Target variable for prediction
            
        Returns:
            Dictionary with comprehensive ML analysis results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_shape': market_data.shape,
            'target_distribution': target_data.value_counts().to_dict(),
            'ml_utilities_available': ML_COMMON_AVAILABLE
        }
        
        try:
            # Step 1: Data Validation and Drift Detection
            self.logger.info("🔍 Step 1: Data validation and drift detection")
            validation_results = self._perform_data_validation(market_data, target_data)
            results['data_validation'] = validation_results
            
            # Step 2: Feature Importance Analysis
            self.logger.info("📊 Step 2: Feature importance analysis")
            feature_results = self._analyze_feature_importance(market_data, target_data)
            results['feature_analysis'] = feature_results
            
            # Step 3: Hyperparameter Optimization
            self.logger.info("⚙️ Step 3: Hyperparameter optimization")
            hpo_results = self._perform_hyperparameter_optimization(market_data, target_data)
            results['hyperparameter_optimization'] = hpo_results
            
            # Step 4: Advanced Cross-Validation
            self.logger.info("✅ Step 4: Advanced cross-validation")
            cv_results = self._perform_advanced_cross_validation(market_data, target_data)
            results['cross_validation'] = cv_results
            
            # Step 5: Overfitting Prevention
            self.logger.info("🛡️ Step 5: Overfitting prevention")
            overfitting_results = self._prevent_overfitting(market_data, target_data)
            results['overfitting_prevention'] = overfitting_results
            
            # Step 6: Ensemble Methods
            self.logger.info("🎯 Step 6: Ensemble methods")
            ensemble_results = self._create_ensembles(market_data, target_data)
            results['ensemble_methods'] = ensemble_results
            
            # Step 7: Model Monitoring
            self.logger.info("📈 Step 7: Model monitoring")
            monitoring_results = self._monitor_models(market_data, target_data)
            results['model_monitoring'] = monitoring_results
            
            # Step 8: TAS Integration
            self.logger.info("🌳 Step 8: TAS integration")
            tas_results = self._integrate_tas_system(market_data, target_data)
            results['tas_integration'] = tas_results
            
            # Step 9: NAS Integration
            self.logger.info("🧠 Step 9: NAS integration")
            nas_results = self._integrate_nas_system(market_data, target_data)
            results['nas_integration'] = nas_results
            
            # Step 10: Hybrid Orchestration
            self.logger.info("🔄 Step 10: Hybrid orchestration")
            hybrid_results = self._orchestrate_hybrid_system(market_data, target_data)
            results['hybrid_orchestration'] = hybrid_results
            
            self.logger.info("✅ Comprehensive ML analysis completed")
            
        except Exception as e:
            self.logger.error(f"❌ Comprehensive ML analysis failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def _perform_data_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive data validation."""
        try:
            validation_results = {}
            
            # Basic data validation
            validation_results['data_quality'] = {
                'missing_values': X.isnull().sum().sum(),
                'infinite_values': np.isinf(X.select_dtypes(include=[np.number])).sum().sum(),
                'duplicate_rows': X.duplicated().sum(),
                'data_types': X.dtypes.value_counts().to_dict()
            }
            
            # Data drift detection
            if self.drift_detector:
                drift_result = self.drift_detector.detect_drift(X.values)
                validation_results['data_drift'] = {
                    'drift_detected': drift_result.drift_detected,
                    'drift_score': drift_result.drift_score,
                    'drifted_features': drift_result.drifted_features,
                    'severity': drift_result.severity
                }
            
            # Lookahead protection
            if self.lookahead_protection:
                protection_result = self.lookahead_protection.protect_data(X.values, y.values)
                validation_results['lookahead_protection'] = {
                    'protected': protection_result.protected,
                    'leakage_detected': protection_result.leakage_detected,
                    'protection_score': protection_result.protection_score
                }
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return {'error': str(e)}
    
    def _analyze_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Analyze feature importance using ML common utilities."""
        try:
            if not self.feature_analyzer:
                return {'error': 'Feature analyzer not available'}
            
            # Analyze feature importance
            importance_result = self.feature_analyzer.analyze_importance(
                X, y, method='permutation'
            )
            
            return {
                'feature_importance': importance_result.importance_scores,
                'important_features': importance_result.important_features,
                'importance_ranking': importance_result.ranking,
                'analysis_method': importance_result.method
            }
            
        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {e}")
            return {'error': str(e)}
    
    def _perform_hyperparameter_optimization(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive hyperparameter optimization."""
        try:
            if not self.hpo_optimizer:
                return {'error': 'HPO optimizer not available'}
            
            # Create search space
            search_space = create_search_space('random_forest', X, y)
            
            # Perform staged HPO
            hpo_result = self.hpo_optimizer.staged_hpo(
                model_factory=lambda **params: self._create_model(**params),
                X=X.values,
                y=y.values,
                search_space=search_space,
                n_trials=50
            )
            
            return {
                'best_params': hpo_result.get('final_params', {}),
                'best_score': hpo_result.get('final_score', 0.0),
                'optimization_stages': hpo_result.get('best_stage', 'unknown'),
                'total_time': hpo_result.get('total_time', 0.0),
                'coarse_time': hpo_result.get('coarse_time', 0.0),
                'fine_time': hpo_result.get('fine_time', 0.0),
                'optuna_time': hpo_result.get('optuna_time', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Hyperparameter optimization failed: {e}")
            return {'error': str(e)}
    
    def _perform_advanced_cross_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform advanced cross-validation using ML common utilities."""
        try:
            cv_results = {}
            
            # Standard cross-validation
            if self.validation_framework:
                standard_cv = perform_cross_validation(
                    X, y, self._create_model(), n_folds=5
                )
                cv_results['standard_cv'] = standard_cv
            
            # Temporal cross-validation
            temporal_cv = temporal_cross_validation(
                X, y, self._create_model(), n_folds=5
            )
            cv_results['temporal_cv'] = temporal_cv
            
            # Purged K-fold validation
            purged_kfold = PurgedKFold(n_splits=5, purged_pct=0.01, embargo_pct=0.01)
            cv_results['purged_kfold'] = {
                'n_splits': purged_kfold.n_splits,
                'purged_pct': purged_kfold.purged_pct,
                'embargo_pct': purged_kfold.embargo_pct
            }
            
            return cv_results
            
        except Exception as e:
            self.logger.error(f"Advanced cross-validation failed: {e}")
            return {'error': str(e)}
    
    def _prevent_overfitting(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Apply comprehensive overfitting prevention."""
        try:
            if not self.overfitting_prevention:
                return {'error': 'Overfitting prevention not available'}
            
            # Create a model
            model = self._create_model()
            
            # Apply regularization
            regularized_model = self.overfitting_prevention.apply_regularization(
                model, 'RandomForestRegressor'
            )
            
            # Setup early stopping
            early_stopping_config = self.overfitting_prevention.setup_early_stopping(
                model, 'RandomForestRegressor'
            )
            
            # Perform cross-validation
            cv_result = self.overfitting_prevention.perform_cross_validation(
                model, X.values, y.values, 'test_model'
            )
            
            # Monitor performance
            X_train, X_val, y_train, y_val = self._split_data(X, y)
            performance_result = self.overfitting_prevention.monitor_performance(
                model, X_train, y_train, X_val, y_val, 'test_model'
            )
            
            # Get overfitting summary
            overfitting_summary = self.overfitting_prevention.get_overfitting_summary()
            
            return {
                'regularization_applied': True,
                'early_stopping_config': early_stopping_config,
                'cross_validation': cv_result,
                'performance_monitoring': performance_result,
                'overfitting_summary': overfitting_summary
            }
            
        except Exception as e:
            self.logger.error(f"Overfitting prevention failed: {e}")
            return {'error': str(e)}
    
    def _create_ensembles(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Create and evaluate ensemble methods."""
        try:
            if not self.ensemble_manager:
                return {'error': 'Ensemble manager not available'}
            
            # Create base models
            base_models = {
                'rf': self._create_model(),
                'gb': self._create_model(),
                'svm': self._create_model()
            }
            
            # Create ensemble
            ensemble = self.ensemble_manager.create_ensemble(
                base_models, ensemble_type='voting'
            )
            
            # Evaluate ensemble
            ensemble_score = ensemble.score(X, y)
            
            return {
                'ensemble_created': True,
                'ensemble_type': 'voting',
                'base_models': list(base_models.keys()),
                'ensemble_score': ensemble_score
            }
            
        except Exception as e:
            self.logger.error(f"Ensemble creation failed: {e}")
            return {'error': str(e)}
    
    def _monitor_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Monitor model performance and stability."""
        try:
            monitoring_results = {}
            
            # Training safeguards
            if self.training_safeguards:
                safeguard_result = self.training_safeguards.validate_training_setup(
                    X.values, y.values
                )
                monitoring_results['training_safeguards'] = safeguard_result
            
            # Error handling
            if self.error_handler:
                error_result = self.error_handler.handle_errors(
                    X.values, y.values
                )
                monitoring_results['error_handling'] = error_result
            
            return monitoring_results
            
        except Exception as e:
            self.logger.error(f"Model monitoring failed: {e}")
            return {'error': str(e)}
    
    def _integrate_tas_system(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Integrate TAS system with ML common utilities."""
        try:
            # Convert to numpy arrays
            X_array = X.values
            y_array = y.values
            
            # Split data
            train_size = int(0.7 * len(X_array))
            X_train, X_val = X_array[:train_size], X_array[train_size:]
            y_train, y_val = y_array[:train_size], y_array[train_size:]
            
            # Perform TAS search
            tas_result = self.tas_engine.search(
                train_data=(X_train, y_train),
                validation_data=(X_val, y_val)
            )
            
            return {
                'tas_search_completed': tas_result.success,
                'best_architecture': str(tas_result.best_architecture),
                'best_score': tas_result.best_score,
                'execution_time': tas_result.execution_time,
                'search_history_length': len(tas_result.search_history)
            }
            
        except Exception as e:
            self.logger.error(f"TAS integration failed: {e}")
            return {'error': str(e)}
    
    def _integrate_nas_system(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Integrate NAS system with ML common utilities."""
        try:
            # Convert to numpy arrays
            X_array = X.values
            y_array = y.values
            
            # Split data
            train_size = int(0.7 * len(X_array))
            X_train, X_val = X_array[:train_size], X_array[train_size:]
            y_train, y_val = y_array[:train_size], y_array[train_size:]
            
            # Perform NAS search
            nas_result = self.nas_engine.search(
                train_data=(X_train, y_train),
                validation_data=(X_val, y_val)
            )
            
            return {
                'nas_search_completed': nas_result.success,
                'best_architecture': str(nas_result.best_architecture),
                'best_score': nas_result.best_score,
                'execution_time': nas_result.execution_time,
                'n_evaluations': nas_result.n_evaluations,
                'strategy_used': nas_result.strategy_used
            }
            
        except Exception as e:
            self.logger.error(f"NAS integration failed: {e}")
            return {'error': str(e)}
    
    def _orchestrate_hybrid_system(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Orchestrate hybrid system with ML common utilities."""
        try:
            # Perform hybrid regime analysis
            hybrid_result = self.hybrid_orchestrator.analyze_market_regimes(
                market_data=X,
                timestamps=None,
                enable_multi_timeframe=True
            )
            
            return {
                'hybrid_analysis_completed': True,
                'regime_predictions_shape': hybrid_result.regime_predictions.shape,
                'regime_probabilities_shape': hybrid_result.regime_probabilities.shape,
                'execution_time': hybrid_result.execution_time,
                'tas_contributions': hybrid_result.tas_contributions,
                'nas_contributions': hybrid_result.nas_contributions,
                'hybrid_analysis': hybrid_result.hybrid_analysis
            }
            
        except Exception as e:
            self.logger.error(f"Hybrid orchestration failed: {e}")
            return {'error': str(e)}
    
    def _create_model(self, **params):
        """Create a model instance."""
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(**params)
    
    def _split_data(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
        """Split data into train and validation sets."""
        from sklearn.model_selection import train_test_split
        return train_test_split(X, y, test_size=test_size, random_state=42)


def run_comprehensive_ml_integration_example():
    """Run comprehensive ML integration example."""
    print("🚀 Starting Comprehensive ML Integration Example")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    y = pd.Series(np.random.randint(0, 3, n_samples), name='target')
    
    print(f"📊 Data shape: {X.shape}")
    print(f"🎯 Target distribution: {y.value_counts().to_dict()}")
    print()
    
    # Initialize comprehensive ML integration
    ml_integration = ComprehensiveMLIntegration()
    
    # Run comprehensive analysis
    results = ml_integration.demonstrate_comprehensive_ml_usage(X, y)
    
    # Display results
    print("📈 Comprehensive ML Analysis Results:")
    print("=" * 60)
    
    for key, value in results.items():
        if key == 'timestamp':
            print(f"⏰ {key}: {value}")
        elif key == 'data_shape':
            print(f"📊 {key}: {value}")
        elif key == 'target_distribution':
            print(f"🎯 {key}: {value}")
        elif key == 'ml_utilities_available':
            print(f"🛠️ {key}: {'✅ Yes' if value else '❌ No'}")
        elif isinstance(value, dict) and 'error' not in value:
            print(f"✅ {key}: Completed successfully")
        elif isinstance(value, dict) and 'error' in value:
            print(f"❌ {key}: {value['error']}")
        else:
            print(f"📋 {key}: {value}")
    
    print("\n🎉 Comprehensive ML Integration Example completed!")
    return results


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the example
    results = run_comprehensive_ml_integration_example()