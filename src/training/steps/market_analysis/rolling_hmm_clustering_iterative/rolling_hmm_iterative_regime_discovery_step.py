"""
Rolling HMM Iterative Regime Discovery Step

This module implements the main regime discovery step using iterative HPO optimization
instead of traditional grid search. It uses the RollingHMMIterativeOptimizer to find
optimal parameters through 20% increments until convergence.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_debug
from src.training.steps.market_analysis.rolling_hmm_clustering_iterative.feature_engineering import (
    RollingHMMFeatureEngineer,
    FeatureEngineeringConfig,
    EWMAConfig,
    DEFAULT_EWMA_CONFIGS
)
from src.training.steps.market_analysis.rolling_hmm_clustering_iterative.rolling_hmm_iterative_optimizer import (
    RollingHMMIterativeOptimizer,
    IterativeHPOConfig,
    DEFAULT_ITERATIVE_HPO_CONFIG
)
from src.training.steps.market_analysis.rolling_hmm_clustering_iterative.sticky_hmm_model import (
    StickyHMM,
    StickyHMMConfig
)
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    ClusterQualityAssessor,
    ClusterQualityConfig
)
from src.utils.data.data_loader import DataLoader
from src.utils.data.data_validator import DataValidator
from src.utils.data.data_preprocessor import DataPreprocessor
from src.utils.error_handler import handle_errors

logger = logging.getLogger(__name__)


@dataclass
class IterativeRegimeDiscoveryConfig:
    """Configuration for iterative regime discovery."""
    
    # Data configuration
    data_path: str = "data/market/1h/BTCUSDT_1h.parquet"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Feature engineering
    use_log_returns: bool = True
    use_volatility_features: bool = True
    use_trend_features: bool = True
    use_volume_features: bool = True
    pca_components: Optional[int] = None
    normalize_method: str = 'zscore'
    rolling_normalize_window: int = 100
    
    # Iterative HPO configuration
    iterative_hpo_config: IterativeHPOConfig = DEFAULT_ITERATIVE_HPO_CONFIG
    
    # HMM configuration
    hmm_n_iter: int = 100
    hmm_covariance_type: str = 'diag'
    hmm_kmeans_init: bool = True
    hmm_use_sticky_priors: bool = True
    hmm_post_fit_regularization: bool = True
    hmm_early_stopping_enabled: bool = True
    hmm_early_stopping_patience: int = 5
    
    # Quality assessment
    quality_assessment_enabled: bool = True
    quality_config: ClusterQualityConfig = ClusterQualityConfig()
    
    # Output configuration
    output_dir: str = "results/rolling_hmm_iterative"
    save_models: bool = True
    save_features: bool = True
    save_predictions: bool = True
    save_optimization_history: bool = True
    
    # Resource management
    enable_hardware_optimization: bool = True
    timeout_seconds: int = 300


class RollingHMMIterativeRegimeDiscoveryStep:
    """
    Main step for iterative rolling HMM regime discovery.
    
    Uses iterative optimization instead of grid search to find optimal parameters.
    """
    
    def __init__(self, config: IterativeRegimeDiscoveryConfig):
        """
        Initialize regime discovery step.
        
        Args:
            config: Iterative regime discovery configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        tprint_info("🚀 Initializing RollingHMMIterativeRegimeDiscoveryStep")
        
        # Initialize output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.feature_engineer: Optional[RollingHMMFeatureEngineer] = None
        self.iterative_optimizer: Optional[RollingHMMIterativeOptimizer] = None
        self.quality_assessor: Optional[ClusterQualityAssessor] = None
        self.best_hmm_model: Optional[StickyHMM] = None
        
        # Results storage
        self.market_data: Optional[pd.DataFrame] = None
        self.features: Optional[pd.DataFrame] = None
        self.economic_features: Optional[pd.DataFrame] = None
        self.regime_labels: Optional[np.ndarray] = None
        self.optimization_result: Optional[Dict[str, Any]] = None
        self.quality_metrics: Optional[Dict[str, Any]] = None
        
    @handle_errors
    def run(self) -> Dict[str, Any]:
        """
        Run the complete iterative regime discovery pipeline.
        
        Returns:
            Dictionary with results
        """
        tprint("🔍 Starting Iterative Rolling HMM Regime Discovery")
        
        start_time = time.time()
        
        # 1. Load and validate data
        tprint_info("📊 Step 1: Loading and validating market data")
        self.market_data = self._load_and_validate_data()
        
        if self.market_data is None or len(self.market_data) < 100:
            tprint_error("❌ Insufficient market data for regime discovery")
            return {'success': False, 'error': 'Insufficient data'}
        
        tprint_info(f"✅ Loaded {len(self.market_data)} rows of market data")
        
        # 2. Initialize feature engineering
        tprint_info("⚙️  Step 2: Initializing feature engineering")
        self._initialize_feature_engineering()
        
        # 3. Initialize iterative optimizer
        tprint_info("🧠 Step 3: Initializing iterative HPO optimizer")
        self._initialize_iterative_optimizer()
        
        # 4. Run iterative optimization
        tprint_info("🔧 Step 4: Running iterative parameter optimization")
        self.optimization_result = self._run_iterative_optimization()
        
        if not self.optimization_result.get('success', False):
            tprint_error("❌ Iterative optimization failed")
            return self.optimization_result
        
        # 5. Train final HMM model with best parameters
        tprint_info("🎯 Step 5: Training final HMM model with best parameters")
        self._train_final_hmm_model()
        
        # 6. Generate regime predictions
        tprint_info("📈 Step 6: Generating regime predictions")
        self._generate_regime_predictions()
        
        # 7. Assess regime quality
        if self.config.quality_assessment_enabled:
            tprint_info("📊 Step 7: Assessing regime quality")
            self._assess_regime_quality()
        
        # 8. Save results
        tprint_info("💾 Step 8: Saving results")
        self._save_results()
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Prepare final result
        result = {
            'success': True,
            'execution_time': execution_time,
            'data_rows': len(self.market_data),
            'feature_count': len(self.features.columns) if self.features is not None else 0,
            'economic_feature_count': len(self.economic_features.columns) if self.economic_features is not None else 0,
            'n_regimes': int(np.max(self.regime_labels)) + 1 if self.regime_labels is not None else 0,
            'best_parameters': self.optimization_result.get('best_params', {}),
            'best_score': self.optimization_result.get('best_score', 0.0),
            'optimization_trials': self.optimization_result.get('n_trials', 0),
            'optimization_iterations': self.optimization_result.get('iterations', 0),
            'converged': self.optimization_result.get('converged', False),
            'quality_metrics': self.quality_metrics if self.quality_metrics is not None else {},
            'output_dir': str(self.output_dir)
        }
        
        tprint("")
        tprint("🎉 Iterative Rolling HMM Regime Discovery Complete!")
        tprint(f"📊 Final Results:")
        tprint(f"   → Data: {result['data_rows']} rows")
        tprint(f"   → Features: {result['feature_count']} total, {result['economic_feature_count']} economic")
        tprint(f"   → Regimes: {result['n_regimes']} discovered")
        tprint(f"   → Best Score: {result['best_score']:.4f}")
        tprint(f"   → Optimization: {result['optimization_trials']} trials, {result['optimization_iterations']} iterations")
        tprint(f"   → Converged: {result['converged']}")
        tprint(f"   → Execution Time: {result['execution_time']:.1f}s")
        tprint(f"   → Output: {result['output_dir']}")
        
        return result
    
    def _load_and_validate_data(self) -> Optional[pd.DataFrame]:
        """
        Load and validate market data.
        
        Returns:
            Validated market data DataFrame or None
        """
        try:
            # Load data
            loader = DataLoader()
            market_data = loader.load_parquet(
                self.config.data_path,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
            
            if market_data is None:
                tprint_error(f"❌ Failed to load data from {self.config.data_path}")
                return None
            
            # Validate data
            validator = DataValidator()
            validation_result = validator.validate_market_data(market_data)
            
            if not validation_result['valid']:
                tprint_error(f"❌ Data validation failed: {validation_result['errors']}")
                return None
            
            # Preprocess data
            preprocessor = DataPreprocessor()
            market_data = preprocessor.preprocess_market_data(market_data)
            
            # Ensure required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in market_data.columns]
            
            if missing_columns:
                tprint_error(f"❌ Missing required columns: {missing_columns}")
                return None
            
            # Sort by timestamp
            market_data = market_data.sort_index()
            
            return market_data
            
        except Exception as e:
            tprint_error(f"❌ Error loading market data: {e}")
            return None
    
    def _initialize_feature_engineering(self) -> None:
        """Initialize feature engineering component."""
        # Create feature engineering config
        ewma_configs = DEFAULT_EWMA_CONFIGS
        
        feature_config = FeatureEngineeringConfig(
            ewma_configs=ewma_configs,
            use_log_returns=self.config.use_log_returns,
            use_volatility_features=self.config.use_volatility_features,
            use_trend_features=self.config.use_trend_features,
            use_volume_features=self.config.use_volume_features,
            pca_components=self.config.pca_components,
            normalize_method=self.config.normalize_method,
            rolling_normalize_window=self.config.rolling_normalize_window,
            enable_hardware_optimization=self.config.enable_hardware_optimization,
            cache_dir=self.output_dir / "cache"
        )
        
        # Initialize feature engineer
        self.feature_engineer = RollingHMMFeatureEngineer(feature_config)
        
        # Pre-compute features for all EWMA configs (will be cached)
        tprint_info("🔄 Pre-computing features for all EWMA configurations")
        self.feature_engineer.precompute_all_features(self.market_data)
        
    def _initialize_iterative_optimizer(self) -> None:
        """Initialize iterative HPO optimizer."""
        # Create iterative optimizer
        self.iterative_optimizer = RollingHMMIterativeOptimizer(
            self.config.iterative_hpo_config
        )
        
        # Initialize quality assessor if needed
        if self.config.quality_assessment_enabled:
            self.quality_assessor = ClusterQualityAssessor(self.config.quality_config)
    
    def _run_iterative_optimization(self) -> Dict[str, Any]:
        """
        Run iterative parameter optimization.
        
        Returns:
            Optimization result dictionary
        """
        try:
            # Run optimization
            result = self.iterative_optimizer.optimize(
                market_data=self.market_data,
                feature_engineer=self.feature_engineer,
                hmm_model_class=StickyHMM,
                quality_assessor=self.quality_assessor
            )
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Iterative optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'best_params': {},
                'best_score': -np.inf,
                'n_trials': 0,
                'iterations': 0,
                'converged': False
            }
    
    def _train_final_hmm_model(self) -> None:
        """Train final HMM model with best parameters."""
        if self.optimization_result is None:
            tprint_error("❌ No optimization result available")
            return
        
        best_params = self.optimization_result['best_params']
        
        # Create EWMA config from best parameters
        ewma_config = EWMAConfig(
            short_window=int(best_params['ewma_short']),
            long_window=int(best_params['ewma_long']),
            name=f"{best_params['ewma_short']}+{best_params['ewma_long']}"
        )
        
        # Generate features with best EWMA config
        tprint_info(f"🔧 Generating features with best EWMA config: {ewma_config.name}")
        features = self.feature_engineer.generate_features(
            self.market_data,
            ewma_config=ewma_config
        )
        
        # Extract economic features
        self.economic_features = self.feature_engineer.extract_economic_features(
            features,
            self.market_data,
            ewma_config
        )
        
        # Create HMM config
        hmm_config = StickyHMMConfig(
            n_components=int(best_params['n_components']),
            min_covar=float(best_params['min_covar']),
            kappa=float(best_params['kappa']),
            n_iter=self.config.hmm_n_iter,
            covariance_type=self.config.hmm_covariance_type,
            kmeans_init=self.config.hmm_kmeans_init,
            use_sticky_priors=self.config.hmm_use_sticky_priors,
            post_fit_regularization=self.config.hmm_post_fit_regularization,
            early_stopping_enabled=self.config.hmm_early_stopping_enabled,
            early_stopping_patience=self.config.hmm_early_stopping_patience
        )
        
        # Train HMM model
        tprint_info("🎯 Training final HMM model")
        self.best_hmm_model = StickyHMM(hmm_config)
        self.best_hmm_model.fit(
            self.economic_features.values,
            ewma_config_name=ewma_config.name
        )
        
        # Store features
        self.features = features
        
    def _generate_regime_predictions(self) -> None:
        """Generate regime predictions using the trained HMM model."""
        if self.best_hmm_model is None or self.economic_features is None:
            tprint_error("❌ HMM model or features not available")
            return
        
        # Predict regime labels
        self.regime_labels = self.best_hmm_model.predict(self.economic_features.values)
        
        # Add to market data for analysis
        self.market_data = self.market_data.copy()
        self.market_data['regime_label'] = self.regime_labels
        
        tprint_info(f"📈 Generated {len(np.unique(self.regime_labels))} regime labels")
    
    def _assess_regime_quality(self) -> None:
        """Assess regime quality using quality assessor."""
        if (self.quality_assessor is None or 
            self.regime_labels is None or 
            self.economic_features is None or
            self.best_hmm_model is None):
            tprint_warning("⚠️  Cannot assess regime quality - missing components")
            return
        
        try:
            # Get transition matrix
            transition_matrix = self.best_hmm_model.get_transition_matrix()
            
            # Calculate forward returns
            forward_returns = (
                self.market_data['close'].pct_change(2)
                .shift(-2)
            )
            forward_returns = forward_returns.loc[self.economic_features.index]
            
            # Assess quality
            quality_result = self.quality_assessor.assess_hmm_regime_quality(
                regime_labels=self.regime_labels,
                feature_data=self.economic_features,
                transition_matrix=transition_matrix,
                hmm_model=self.best_hmm_model,
                forward_returns=forward_returns,
                timestamps=self.economic_features.index,
                timeframe='1h',
                min_regime_size=10,
                run_validators=True,
                temporal_sensitivity_mode="standard",
                fast_mode=False
            )
            
            if quality_result is not None:
                self.quality_metrics = quality_result.to_dict() if hasattr(quality_result, 'to_dict') else {}
                tprint_info("✅ Regime quality assessment completed")
            else:
                tprint_warning("⚠️  Quality assessment returned None")
                
        except Exception as e:
            tprint_error(f"❌ Error assessing regime quality: {e}")
            self.quality_metrics = {}
    
    def _save_results(self) -> None:
        """Save all results to output directory."""
        try:
            # Save market data with regime labels
            if self.config.save_predictions and self.market_data is not None:
                predictions_path = self.output_dir / "regime_predictions.parquet"
                self.market_data.to_parquet(predictions_path)
                tprint_info(f"💾 Saved regime predictions to {predictions_path}")
            
            # Save features
            if self.config.save_features:
                if self.features is not None:
                    features_path = self.output_dir / "features.parquet"
                    self.features.to_parquet(features_path)
                    tprint_info(f"💾 Saved features to {features_path}")
                
                if self.economic_features is not None:
                    economic_features_path = self.output_dir / "economic_features.parquet"
                    self.economic_features.to_parquet(economic_features_path)
                    tprint_info(f"💾 Saved economic features to {economic_features_path}")
            
            # Save HMM model
            if self.config.save_models and self.best_hmm_model is not None:
                model_path = self.output_dir / "hmm_model.pkl"
                import joblib
                joblib.dump(self.best_hmm_model, model_path)
                tprint_info(f"💾 Saved HMM model to {model_path}")
            
            # Save optimization result
            if self.optimization_result is not None:
                optimization_path = self.output_dir / "optimization_result.json"
                with open(optimization_path, 'w') as f:
                    json.dump(self.optimization_result, f, indent=2, default=str)
                tprint_info(f"💾 Saved optimization result to {optimization_path}")
            
            # Save optimization history
            if (self.config.save_optimization_history and 
                self.iterative_optimizer is not None):
                history = self.iterative_optimizer.get_optimization_summary()
                if 'optimization_history' in self.optimization_result:
                    history['optimization_history'] = self.optimization_result['optimization_history']
                
                history_path = self.output_dir / "optimization_history.json"
                with open(history_path, 'w') as f:
                    json.dump(history, f, indent=2, default=str)
                tprint_info(f"💾 Saved optimization history to {history_path}")
            
            # Save quality metrics
            if self.quality_metrics is not None:
                quality_path = self.output_dir / "quality_metrics.json"
                with open(quality_path, 'w') as f:
                    json.dump(self.quality_metrics, f, indent=2, default=str)
                tprint_info(f"💾 Saved quality metrics to {quality_path}")
            
            # Save configuration
            config_path = self.output_dir / "config.json"
            config_dict = asdict(self.config)
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2, default=str)
            tprint_info(f"💾 Saved configuration to {config_path}")
            
        except Exception as e:
            tprint_error(f"❌ Error saving results: {e}")


def main():
    """Main function for testing the iterative regime discovery step."""
    # Create configuration
    config = IterativeRegimeDiscoveryConfig(
        data_path="data/market/1h/BTCUSDT_1h.parquet",
        iterative_hpo_config=DEFAULT_ITERATIVE_HPO_CONFIG,
        output_dir="results/rolling_hmm_iterative_test"
    )
    
    # Create and run regime discovery
    discovery = RollingHMMIterativeRegimeDiscoveryStep(config)
    result = discovery.run()
    
    print(f"\nFinal Result: {result}")


if __name__ == "__main__":
    main()
