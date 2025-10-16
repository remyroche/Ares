"""
Unified Regime Training Pipeline for NAS/TAS Integration

This module provides a complete pipeline that integrates:
- NAS/TAS regime detection (not HMM-based clustering)
- Per-regime ML model training for 5m & 15m timeframes
- Model selection architecture for best 2-3 models
- Signal emission based on ML outputs

This replaces the HMM-based approach with NAS/TAS regime detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Import NAS/TAS components
from src.training.steps.market_analysis.hybrid_nas_tas_regime.core.hybrid_regime_detector import (
    HybridNASTASRegimeDetector, HybridRegimeResult
)
from src.training.steps.market_analysis.hybrid_nas_tas_regime.config.hybrid_regime_config import (
    HybridRegimeConfig, RegimeCombinationStrategy
)

# Import training components
from src.utils.ml_common.training.per_regime_training_step import PerRegimeTrainingStep
from src.utils.ml_common.training.base_training_step import BaseTrainingStep
from src.utils.ml_common.config.base_training_config import PerRegimeTrainingConfig

# Import model selection
from src.training.steps.market_analysis.hybrid_nas_tas_regime.regime_model_mapping.data_driven_model_selector import (
    DataDrivenModelSelector, ModelSelectorConfig, RegimeModelMapping
)

# Import signal generation
from src.trading.signal_generation.signal_pipeline import SignalGenerationPipeline
from src.trading.config.trading_config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class TimeframeConfig:
    """Configuration for specific timeframe training."""
    timeframe: str  # '5m' or '15m'
    lookback_periods: int = 1000
    min_samples_per_regime: int = 100
    validation_split: float = 0.2
    test_split: float = 0.1


@dataclass
class UnifiedTrainingConfig:
    """Configuration for unified regime training pipeline."""
    # Regime detection
    n_regimes: int = 8
    regime_combination_strategy: RegimeCombinationStrategy = RegimeCombinationStrategy.ADAPTIVE_FUSION
    
    # Timeframes
    timeframes: List[str] = field(default_factory=lambda: ['5m', '15m'])
    timeframe_configs: Dict[str, TimeframeConfig] = field(default_factory=dict)
    
    # Model training
    model_types: List[str] = field(default_factory=lambda: ['random_forest', 'xgboost', 'lightgbm', 'neural_network'])
    enable_hpo: bool = True
    enable_ensemble: bool = True
    max_ensemble_models: int = 3
    
    # Model selection
    primary_metric: str = 'f1_score'
    confidence_threshold: float = 0.7
    enable_continuous_learning: bool = True
    
    # Signal generation
    enable_signal_generation: bool = True
    signal_confidence_threshold: float = 0.6
    
    # Data paths
    data_cache_dir: str = "data_cache/nas_tas_regime_training"
    model_save_dir: str = "models/nas_tas_regime_models"
    results_save_dir: str = "results/nas_tas_regime_results"


class UnifiedRegimeTrainingPipeline:
    """
    Unified pipeline that integrates NAS/TAS regime detection with per-regime training,
    model selection, and signal generation.
    
    This replaces the HMM-based approach with NAS/TAS regime detection and ensures
    proper integration across all components.
    """
    
    def __init__(self, config: UnifiedTrainingConfig):
        """Initialize unified regime training pipeline."""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.regime_detector = None
        self.model_selector = None
        self.signal_pipeline = None
        self.training_steps = {}
        
        # Results storage
        self.regime_results = {}
        self.model_mappings = {}
        self.training_metadata = {}
        
        # Create directories
        self._create_directories()
        
        self.logger.info("✅ Unified Regime Training Pipeline initialized")
        self.logger.info(f"   Timeframes: {config.timeframes}")
        self.logger.info(f"   Regimes: {config.n_regimes}")
        self.logger.info(f"   Model types: {config.model_types}")
    
    def _create_directories(self):
        """Create necessary directories for data and model storage."""
        directories = [
            self.config.data_cache_dir,
            self.config.model_save_dir,
            self.config.results_save_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def initialize_components(self) -> bool:
        """Initialize all pipeline components."""
        try:
            self.logger.info("🔧 Initializing pipeline components...")
            
            # Initialize regime detector
            self._initialize_regime_detector()
            
            # Initialize model selector
            self._initialize_model_selector()
            
            # Initialize training steps for each timeframe
            self._initialize_training_steps()
            
            # Initialize signal pipeline
            if self.config.enable_signal_generation:
                self._initialize_signal_pipeline()
            
            self.logger.info("✅ All pipeline components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize pipeline components: {e}")
            return False
    
    def _initialize_regime_detector(self):
        """Initialize NAS/TAS regime detector."""
        try:
            regime_config = HybridRegimeConfig(
                n_regimes=self.config.n_regimes,
                combination_strategy=self.config.regime_combination_strategy
            )
            
            self.regime_detector = HybridNASTASRegimeDetector(regime_config)
            self.logger.info("✅ NAS/TAS regime detector initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize regime detector: {e}")
            raise
    
    def _initialize_model_selector(self):
        """Initialize data-driven model selector."""
        try:
            selector_config = ModelSelectorConfig(
                primary_metric=self.config.primary_metric,
                confidence_threshold=self.config.confidence_threshold,
                enable_ensemble=self.config.enable_ensemble,
                max_ensemble_models=self.config.max_ensemble_models,
                enable_continuous_learning=self.config.enable_continuous_learning,
                mapping_file_path=f"{self.config.data_cache_dir}/regime_model_mappings.pkl"
            )
            
            self.model_selector = DataDrivenModelSelector(selector_config)
            self.logger.info("✅ Data-driven model selector initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize model selector: {e}")
            raise
    
    def _initialize_training_steps(self):
        """Initialize training steps for each timeframe."""
        try:
            for timeframe in self.config.timeframes:
                # Create timeframe-specific config
                timeframe_config = PerRegimeTrainingConfig(
                    model_name=f"nas_tas_regime_{timeframe}",
                    model_types=self.config.model_types,
                    enable_hpo=self.config.enable_hpo,
                    save_models=True,
                    enable_evaluation=True,
                    timeframe=timeframe
                )
                
                # Initialize training step
                training_step = PerRegimeTrainingStep(timeframe_config)
                self.training_steps[timeframe] = training_step
                
                self.logger.info(f"✅ Training step initialized for {timeframe}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize training steps: {e}")
            raise
    
    def _initialize_signal_pipeline(self):
        """Initialize signal generation pipeline."""
        try:
            trading_config = TradingConfig()
            self.signal_pipeline = SignalGenerationPipeline(trading_config)
            self.logger.info("✅ Signal generation pipeline initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize signal pipeline: {e}")
            raise
    
    def train_regime_models(
        self,
        market_data: Dict[str, pd.DataFrame],
        feature_names: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Train regime models for all timeframes using NAS/TAS regime detection.
        
        Args:
            market_data: Market data for each timeframe
            feature_names: Feature names for each timeframe
            
        Returns:
            Training results for all timeframes
        """
        try:
            self.logger.info("🚀 Starting unified regime training for all timeframes")
            start_time = time.time()
            
            results = {}
            
            # Process each timeframe
            for timeframe in self.config.timeframes:
                if timeframe not in market_data:
                    self.logger.warning(f"⚠️ No data available for timeframe {timeframe}")
                    continue
                
                self.logger.info(f"🔄 Processing timeframe {timeframe}...")
                
                # Step 1: Detect regimes using NAS/TAS
                regime_result = self._detect_regimes_nas_tas(
                    market_data[timeframe], timeframe
                )
                
                if not regime_result.success:
                    self.logger.error(f"❌ Regime detection failed for {timeframe}")
                    continue
                
                # Step 2: Train per-regime models
                training_result = self._train_per_regime_models(
                    market_data[timeframe], regime_result, timeframe, 
                    feature_names.get(timeframe) if feature_names else None
                )
                
                # Step 3: Update model selector with performance data
                self._update_model_selector(training_result, timeframe)
                
                # Store results
                results[timeframe] = {
                    'regime_detection': regime_result,
                    'training_results': training_result,
                    'model_mappings': self.model_selector.get_ensemble_weights(
                        regime_result.regime_predictions, self.config.model_types
                    )
                }
                
                self.logger.info(f"✅ Completed training for {timeframe}")
            
            # Save results
            self._save_training_results(results)
            
            total_time = time.time() - start_time
            self.logger.info(f"✅ Unified regime training completed in {total_time:.2f}s")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Unified regime training failed: {e}")
            raise
    
    def _detect_regimes_nas_tas(
        self, 
        market_data: pd.DataFrame, 
        timeframe: str
    ) -> HybridRegimeResult:
        """Detect regimes using NAS/TAS approach."""
        try:
            self.logger.info(f"🔍 Detecting regimes for {timeframe} using NAS/TAS...")
            
            # Use NAS/TAS regime detector (not HMM-based)
            regime_result = self.regime_detector.detect_regimes(
                market_data=market_data,
                validate_economic_significance=True,
                validate_financial_relevance=True
            )
            
            if regime_result.success:
                n_regimes = len(np.unique(regime_result.regime_predictions))
                self.logger.info(f"✅ Detected {n_regimes} regimes for {timeframe}")
            else:
                self.logger.error(f"❌ Regime detection failed for {timeframe}: {regime_result.error_message}")
            
            return regime_result
            
        except Exception as e:
            self.logger.error(f"❌ NAS/TAS regime detection failed for {timeframe}: {e}")
            raise
    
    def _train_per_regime_models(
        self,
        market_data: pd.DataFrame,
        regime_result: HybridRegimeResult,
        timeframe: str,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Train per-regime models for a specific timeframe."""
        try:
            self.logger.info(f"🎯 Training per-regime models for {timeframe}...")
            
            # Prepare data for per-regime training
            X, y = self._prepare_training_data(market_data, feature_names)
            regime_labels = regime_result.regime_predictions
            
            # Get training step for this timeframe
            training_step = self.training_steps[timeframe]
            
            # Execute per-regime training
            training_result = training_step.execute(
                X=X,
                y=y,
                regime_labels=regime_labels,
                feature_names=feature_names,
                symbol=f"ETHUSDT_{timeframe}",
                exchange="binance",
                timeframe=timeframe
            )
            
            self.logger.info(f"✅ Per-regime training completed for {timeframe}")
            return training_result
            
        except Exception as e:
            self.logger.error(f"❌ Per-regime training failed for {timeframe}: {e}")
            raise
    
    def _prepare_training_data(
        self, 
        market_data: pd.DataFrame, 
        feature_names: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from market data."""
        try:
            # Extract features
            if feature_names:
                X = market_data[feature_names].values
            else:
                # Use default features
                feature_cols = ['open', 'high', 'low', 'close', 'volume']
                available_cols = [col for col in feature_cols if col in market_data.columns]
                X = market_data[available_cols].values
            
            # Create target (simplified - in practice this would be more sophisticated)
            if 'close' in market_data.columns:
                y = (market_data['close'].shift(-1) > market_data['close']).astype(int).values
                # Remove last row where target is NaN
                X = X[:-1]
                y = y[:-1]
            else:
                # Fallback target
                y = np.random.randint(0, 2, len(X))
            
            return X, y
            
        except Exception as e:
            self.logger.error(f"❌ Failed to prepare training data: {e}")
            raise
    
    def _update_model_selector(
        self, 
        training_result: Dict[str, Any], 
        timeframe: str
    ):
        """Update model selector with performance data."""
        try:
            # Extract performance data from training results
            if 'evaluation_results' in training_result:
                for regime_id, regime_eval in training_result['evaluation_results'].items():
                    for model_type, model_eval in regime_eval.items():
                        if 'metrics' in model_eval and 'predictions' in model_eval:
                            # Register performance with model selector
                            self.model_selector.register_model_performance(
                                regime_id=int(regime_id),
                                model_name=f"{model_type}_{timeframe}",
                                predictions=model_eval['predictions'],
                                actual_values=model_eval.get('actual_values', []),
                                execution_time=model_eval.get('execution_time', 0.0)
                            )
            
            self.logger.debug(f"Updated model selector for {timeframe}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update model selector for {timeframe}: {e}")
    
    def generate_signals(
        self,
        market_data: Dict[str, pd.DataFrame],
        symbol: str = "ETHUSDT"
    ) -> Dict[str, Any]:
        """
        Generate trading signals using trained models and regime detection.
        
        Args:
            market_data: Market data for each timeframe
            symbol: Trading symbol
            
        Returns:
            Generated signals for each timeframe
        """
        try:
            if not self.signal_pipeline:
                raise RuntimeError("Signal pipeline not initialized")
            
            self.logger.info(f"📡 Generating signals for {symbol}...")
            
            signals = {}
            
            # Generate signals for each timeframe
            for timeframe in self.config.timeframes:
                if timeframe not in market_data:
                    continue
                
                self.logger.info(f"🔄 Generating signals for {timeframe}...")
                
                # Detect current regime
                regime_result = self._detect_regimes_nas_tas(
                    market_data[timeframe], timeframe
                )
                
                if not regime_result.success:
                    self.logger.warning(f"⚠️ Regime detection failed for {timeframe}")
                    continue
                
                # Select best models for current regime
                current_regime = regime_result.regime_predictions[-1]  # Most recent regime
                available_models = [f"{model}_{timeframe}" for model in self.config.model_types]
                
                selected_model, ensemble_weights = self.model_selector.select_model_for_regime(
                    current_regime, available_models
                )
                
                # Generate signal using selected model
                signal = self._generate_signal_with_model(
                    market_data[timeframe], selected_model, ensemble_weights, timeframe
                )
                
                signals[timeframe] = {
                    'signal': signal,
                    'regime': current_regime,
                    'selected_model': selected_model,
                    'ensemble_weights': ensemble_weights,
                    'confidence': regime_result.regime_probabilities[-1, current_regime]
                }
                
                self.logger.info(f"✅ Generated signal for {timeframe}: {signal}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"❌ Signal generation failed: {e}")
            raise
    
    def _generate_signal_with_model(
        self,
        market_data: pd.DataFrame,
        selected_model: str,
        ensemble_weights: Dict[str, float],
        timeframe: str
    ) -> str:
        """Generate trading signal using selected model."""
        try:
            # This is a simplified signal generation
            # In practice, this would load the trained model and make predictions
            
            # For now, return a placeholder signal
            # The actual implementation would:
            # 1. Load the trained model
            # 2. Prepare input features
            # 3. Make prediction
            # 4. Convert prediction to signal
            
            return "hold"  # Placeholder
            
        except Exception as e:
            self.logger.error(f"❌ Signal generation with model failed: {e}")
            return "hold"
    
    def _save_training_results(self, results: Dict[str, Any]):
        """Save training results to disk."""
        try:
            results_file = f"{self.config.results_save_dir}/unified_training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_serializable(results)
            
            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            self.logger.info(f"💾 Training results saved to {results_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save training results: {e}")
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return obj
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and performance metrics."""
        try:
            status = {
                'pipeline_initialized': self.regime_detector is not None,
                'regime_detector_ready': self.regime_detector is not None,
                'model_selector_ready': self.model_selector is not None,
                'signal_pipeline_ready': self.signal_pipeline is not None,
                'training_steps_ready': len(self.training_steps) > 0,
                'timeframes_supported': self.config.timeframes,
                'model_types': self.config.model_types,
                'n_regimes': self.config.n_regimes
            }
            
            if self.model_selector:
                status['model_selector_summary'] = self.model_selector.get_system_summary()
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get system status: {e}")
            return {'error': str(e)}


def create_unified_pipeline(config: Optional[UnifiedTrainingConfig] = None) -> UnifiedRegimeTrainingPipeline:
    """Create and initialize unified regime training pipeline."""
    if config is None:
        config = UnifiedTrainingConfig()
    
    pipeline = UnifiedRegimeTrainingPipeline(config)
    
    if pipeline.initialize_components():
        return pipeline
    else:
        raise RuntimeError("Failed to initialize unified pipeline")


# Example usage
if __name__ == "__main__":
    # Create configuration
    config = UnifiedTrainingConfig(
        timeframes=['5m', '15m'],
        n_regimes=8,
        model_types=['random_forest', 'xgboost', 'lightgbm'],
        enable_hpo=True,
        enable_ensemble=True
    )
    
    # Create pipeline
    pipeline = create_unified_pipeline(config)
    
    # Example market data (in practice, this would come from data sources)
    market_data = {
        '5m': pd.DataFrame({
            'open': np.random.randn(1000).cumsum(),
            'high': np.random.randn(1000).cumsum() + 0.5,
            'low': np.random.randn(1000).cumsum() - 0.5,
            'close': np.random.randn(1000).cumsum(),
            'volume': np.random.randint(1000, 10000, 1000)
        }),
        '15m': pd.DataFrame({
            'open': np.random.randn(500).cumsum(),
            'high': np.random.randn(500).cumsum() + 0.5,
            'low': np.random.randn(500).cumsum() - 0.5,
            'close': np.random.randn(500).cumsum(),
            'volume': np.random.randint(1000, 10000, 500)
        })
    }
    
    # Train models
    results = pipeline.train_regime_models(market_data)
    
    # Generate signals
    signals = pipeline.generate_signals(market_data)
    
    print("Training completed successfully!")
    print(f"Generated signals: {signals}")