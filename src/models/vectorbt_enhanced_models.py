"""
Unified Interface for VectorBT-Enhanced Models

This module provides a unified interface for all enhanced models with VectorBT capabilities:
- PatchTST with VectorBT integration
- GRU models with VectorBT integration  
- TFT (Temporal Fusion Transformer) with VectorBT integration
- Unified backtesting, metrics, and feature generation
- Performance monitoring and memory management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Type
from dataclasses import dataclass
from enum import Enum
import logging
import warnings

# Import enhanced models
from .enhanced_patchtst import EnhancedPatchTSTModel, EnhancedPatchTSTConfig
from .patch_gru import PatchOrchestrator, PatchConfig, ModelType
from .enhanced_tft import EnhancedTFTModel, EnhancedTFTConfig

# VectorBT utils imports
try:
    from src.utils.ml_common.vectorbt_backtesting_engine import VectorBTBacktestingEngine, VectorBTBacktestConfig, BacktestMode
    from src.utils.ml_common.vectorbt_financial_metrics import VectorBTFinancialMetrics, FinancialMetricsConfig
    from src.feature_generation.core.vectorbt_feature_generator import VectorBTFeatureGenerator, VectorBTVolatilityGenerator, VectorBTMomentumGenerator, VectorBTTrendGenerator
    from src.utils.ml_common.vectorbt_memory_manager import get_memory_manager, memory_managed_operation, optimize_memory_usage
    from src.utils.ml_common.vectorbt_performance_monitor import get_performance_monitor, monitor_operation
    VECTORBT_UTILS_AVAILABLE = True
except ImportError:
    VECTORBT_UTILS_AVAILABLE = False
    VectorBTBacktestingEngine = None
    VectorBTFinancialMetrics = None
    VectorBTFeatureGenerator = None

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Types of enhanced models."""
    PATCHTST = "patchtst"
    GRU = "gru"
    TFT = "tft"


@dataclass
class UnifiedModelConfig:
    """Unified configuration for all enhanced models."""
    # Model selection
    model_type: ModelType
    model_name: str = "enhanced_model"
    
    # Common parameters
    sequence_length: int = 24
    prediction_horizon: int = 1
    hidden_size: int = 64
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    random_state: int = 42
    
    # VectorBT integration parameters
    enable_vectorbt: bool = True
    enable_vectorbt_backtesting: bool = True
    enable_vectorbt_metrics: bool = True
    enable_vectorbt_features: bool = True
    enable_memory_optimization: bool = True
    enable_performance_monitoring: bool = True
    
    # VectorBT backtesting configuration
    vectorbt_backtest_config: Optional[VectorBTBacktestConfig] = None
    vectorbt_metrics_config: Optional[FinancialMetricsConfig] = None
    
    # Performance settings
    memory_limit_gb: float = 8.0
    enable_gpu: bool = False
    enable_parallel: bool = True
    chunk_size: int = 1000
    
    # Model-specific parameters
    model_specific_params: Optional[Dict[str, Any]] = None


class VectorBTEnhancedModelInterface:
    """
    Unified interface for all VectorBT-enhanced models.
    
    This class provides a common interface for PatchTST, GRU, and TFT models
    with VectorBT backtesting, financial metrics, and feature generation capabilities.
    """
    
    def __init__(self, config: UnifiedModelConfig):
        """Initialize the unified model interface."""
        self.config = config
        self.model = None
        self.fitted = False
        
        # VectorBT components
        self.vectorbt_backtesting_engine = None
        self.vectorbt_metrics_calculator = None
        self.vectorbt_feature_generators = []
        self.memory_manager = None
        self.performance_monitor = None
        
        # Initialize VectorBT components if available
        if self.config.enable_vectorbt and VECTORBT_UTILS_AVAILABLE:
            self._initialize_vectorbt_components()
        
        # Performance tracking
        self.vectorbt_stats = {
            'backtests_run': 0,
            'metrics_calculated': 0,
            'features_generated': 0,
            'memory_optimizations': 0,
            'performance_operations': 0
        }
    
    def _initialize_vectorbt_components(self):
        """Initialize VectorBT components for enhanced functionality."""
        try:
            # Initialize memory manager
            if self.config.enable_memory_optimization:
                self.memory_manager = get_memory_manager()
                logger.info("✅ VectorBT memory manager initialized")
            
            # Initialize performance monitor
            if self.config.enable_performance_monitoring:
                self.performance_monitor = get_performance_monitor()
                logger.info("✅ VectorBT performance monitor initialized")
            
            # Initialize backtesting engine
            if self.config.enable_vectorbt_backtesting and VectorBTBacktestingEngine:
                backtest_config = self.config.vectorbt_backtest_config
                if backtest_config is None:
                    backtest_config = VectorBTBacktestConfig(
                        initial_capital=100000.0,
                        commission_rate=0.001,
                        slippage_rate=0.0005,
                        use_gpu=self.config.enable_gpu,
                        enable_parallel=self.config.enable_parallel,
                        memory_limit_gb=self.config.memory_limit_gb
                    )
                
                self.vectorbt_backtesting_engine = VectorBTBacktestingEngine(backtest_config)
                logger.info("✅ VectorBT backtesting engine initialized")
            
            # Initialize metrics calculator
            if self.config.enable_vectorbt_metrics and VectorBTFinancialMetrics:
                metrics_config = self.config.vectorbt_metrics_config
                if metrics_config is None:
                    metrics_config = FinancialMetricsConfig(
                        risk_free_rate=0.02,
                        annualization_factor=252,
                        enable_regime_analysis=True,
                        enable_parallel=self.config.enable_parallel
                    )
                
                self.vectorbt_metrics_calculator = VectorBTFinancialMetrics(metrics_config)
                logger.info("✅ VectorBT financial metrics calculator initialized")
            
            # Initialize feature generators
            if self.config.enable_vectorbt_features and VectorBTFeatureGenerator:
                self.vectorbt_feature_generators = [
                    VectorBTVolatilityGenerator(period=20),
                    VectorBTMomentumGenerator(period=14),
                    VectorBTTrendGenerator(period=20)
                ]
                logger.info(f"✅ VectorBT feature generators initialized: {len(self.vectorbt_feature_generators)} generators")
            
            logger.info("🚀 VectorBT components initialization completed")
            
        except Exception as e:
            logger.warning(f"⚠️ VectorBT components initialization failed: {e}")
            self.config.enable_vectorbt = False
    
    def create_model(self) -> Any:
        """Create the appropriate model based on configuration."""
        try:
            if self.config.model_type == ModelType.PATCHTST:
                # Create PatchTST model
                patchtst_config = EnhancedPatchTSTConfig(
                    lookback_hours=self.config.sequence_length,
                    d_model=self.config.hidden_size,
                    learning_rate=self.config.learning_rate,
                    batch_size=self.config.batch_size,
                    epochs=self.config.epochs,
                    early_stopping_patience=self.config.early_stopping_patience,
                    random_state=self.config.random_state,
                    enable_vectorbt=self.config.enable_vectorbt,
                    enable_vectorbt_backtesting=self.config.enable_vectorbt_backtesting,
                    enable_vectorbt_metrics=self.config.enable_vectorbt_metrics,
                    enable_vectorbt_features=self.config.enable_vectorbt_features,
                    enable_memory_optimization=self.config.enable_memory_optimization,
                    enable_performance_monitoring=self.config.enable_performance_monitoring,
                    vectorbt_backtest_config=self.config.vectorbt_backtest_config,
                    vectorbt_metrics_config=self.config.vectorbt_metrics_config,
                    memory_limit_gb=self.config.memory_limit_gb,
                    enable_gpu=self.config.enable_gpu,
                    enable_parallel=self.config.enable_parallel,
                    chunk_size=self.config.chunk_size
                )
                
                # Add model-specific parameters
                if self.config.model_specific_params:
                    for key, value in self.config.model_specific_params.items():
                        if hasattr(patchtst_config, key):
                            setattr(patchtst_config, key, value)
                
                self.model = EnhancedPatchTSTModel(patchtst_config)
                logger.info("✅ PatchTST model created")
                
            elif self.config.model_type == ModelType.GRU:
                # Create GRU model
                gru_config = PatchConfig(
                    model_type=ModelType.GRU,
                    sequence_length=self.config.sequence_length,
                    horizons=[self.config.prediction_horizon],
                    hidden_dim=self.config.hidden_size,
                    learning_rate=self.config.learning_rate,
                    batch_size=self.config.batch_size,
                    epochs=self.config.epochs,
                    enable_vectorbt=self.config.enable_vectorbt,
                    enable_vectorbt_backtesting=self.config.enable_vectorbt_backtesting,
                    enable_vectorbt_metrics=self.config.enable_vectorbt_metrics,
                    enable_vectorbt_features=self.config.enable_vectorbt_features,
                    enable_memory_optimization=self.config.enable_memory_optimization,
                    enable_performance_monitoring=self.config.enable_performance_monitoring,
                    vectorbt_backtest_config=self.config.vectorbt_backtest_config,
                    vectorbt_metrics_config=self.config.vectorbt_metrics_config,
                    memory_limit_gb=self.config.memory_limit_gb,
                    enable_gpu=self.config.enable_gpu,
                    enable_parallel=self.config.enable_parallel,
                    chunk_size=self.config.chunk_size
                )
                
                # Add model-specific parameters
                if self.config.model_specific_params:
                    for key, value in self.config.model_specific_params.items():
                        if hasattr(gru_config, key):
                            setattr(gru_config, key, value)
                
                self.model = PatchOrchestrator(gru_config)
                logger.info("✅ GRU model created")
                
            elif self.config.model_type == ModelType.TFT:
                # Create TFT model
                tft_config = EnhancedTFTConfig(
                    hidden_size=self.config.hidden_size,
                    sequence_length=self.config.sequence_length,
                    prediction_horizon=self.config.prediction_horizon,
                    learning_rate=self.config.learning_rate,
                    batch_size=self.config.batch_size,
                    epochs=self.config.epochs,
                    early_stopping_patience=self.config.early_stopping_patience,
                    random_state=self.config.random_state,
                    enable_vectorbt=self.config.enable_vectorbt,
                    enable_vectorbt_backtesting=self.config.enable_vectorbt_backtesting,
                    enable_vectorbt_metrics=self.config.enable_vectorbt_metrics,
                    enable_vectorbt_features=self.config.enable_vectorbt_features,
                    enable_memory_optimization=self.config.enable_memory_optimization,
                    enable_performance_monitoring=self.config.enable_performance_monitoring,
                    vectorbt_backtest_config=self.config.vectorbt_backtest_config,
                    vectorbt_metrics_config=self.config.vectorbt_metrics_config,
                    memory_limit_gb=self.config.memory_limit_gb,
                    enable_gpu=self.config.enable_gpu,
                    enable_parallel=self.config.enable_parallel,
                    chunk_size=self.config.chunk_size
                )
                
                # Add model-specific parameters
                if self.config.model_specific_params:
                    for key, value in self.config.model_specific_params.items():
                        if hasattr(tft_config, key):
                            setattr(tft_config, key, value)
                
                self.model = EnhancedTFTModel(tft_config)
                logger.info("✅ TFT model created")
                
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Model creation failed: {e}")
            raise
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'VectorBTEnhancedModelInterface':
        """Fit the model."""
        if self.model is None:
            self.create_model()
        
        try:
            # Generate VectorBT features if enabled
            if self.config.enable_vectorbt_features and hasattr(self.model, 'generate_vectorbt_features'):
                # Convert to DataFrame for feature generation
                if isinstance(X, np.ndarray):
                    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
                else:
                    X_df = X
                
                vectorbt_features = self.model.generate_vectorbt_features(X_df)
                if not vectorbt_features.empty:
                    # Combine original features with VectorBT features
                    X_combined = np.hstack([X, vectorbt_features.values])
                    logger.info(f"✅ Combined {X.shape[1]} original features with {vectorbt_features.shape[1]} VectorBT features")
                else:
                    X_combined = X
            else:
                X_combined = X
            
            # Fit the model
            if hasattr(self.model, 'fit'):
                self.model.fit(X_combined, y, sample_weight)
            else:
                # For PatchOrchestrator, use different fit method
                if hasattr(self.model, 'fit') and self.config.model_type == ModelType.GRU:
                    # Convert to DataFrame for GRU model
                    X_df = pd.DataFrame(X_combined, columns=[f'feature_{i}' for i in range(X_combined.shape[1])])
                    targets = {self.config.prediction_horizon: pd.Series(y)}
                    self.model.fit(X_df, targets)
                else:
                    raise ValueError(f"Model {self.config.model_type} does not support fit method")
            
            self.fitted = True
            logger.info(f"✅ Model {self.config.model_type.value} fitted successfully")
            
            return self
            
        except Exception as e:
            logger.error(f"❌ Model fitting failed: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        try:
            # Generate VectorBT features if enabled
            if self.config.enable_vectorbt_features and hasattr(self.model, 'generate_vectorbt_features'):
                # Convert to DataFrame for feature generation
                if isinstance(X, np.ndarray):
                    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
                else:
                    X_df = X
                
                vectorbt_features = self.model.generate_vectorbt_features(X_df)
                if not vectorbt_features.empty:
                    # Combine original features with VectorBT features
                    X_combined = np.hstack([X, vectorbt_features.values])
                else:
                    X_combined = X
            else:
                X_combined = X
            
            # Make predictions
            if hasattr(self.model, 'predict'):
                predictions = self.model.predict(X_combined)
            else:
                # For PatchOrchestrator, use different predict method
                if self.config.model_type == ModelType.GRU:
                    # Convert to DataFrame for GRU model
                    X_df = pd.DataFrame(X_combined, columns=[f'feature_{i}' for i in range(X_combined.shape[1])])
                    output = self.model.predict(X_df)
                    predictions = output.y_hat_h1.values if hasattr(output, 'y_hat_h1') else np.zeros(len(X))
                else:
                    raise ValueError(f"Model {self.config.model_type} does not support predict method")
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Model prediction failed: {e}")
            raise
    
    def generate_vectorbt_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate features using VectorBT feature generators."""
        if not self.config.enable_vectorbt_features or not self.vectorbt_feature_generators:
            logger.warning("⚠️ VectorBT feature generation not enabled or generators not available")
            return pd.DataFrame(index=data.index)
        
        try:
            with monitor_operation(
                f"vectorbt_feature_generation_{len(self.vectorbt_feature_generators)}",
                metadata={'n_generators': len(self.vectorbt_feature_generators), 'data_shape': data.shape}
            ):
                features = []
                
                for generator in self.vectorbt_feature_generators:
                    try:
                        feature = generator.generate(data)
                        if isinstance(feature, pd.Series):
                            features.append(feature)
                        elif isinstance(feature, pd.DataFrame):
                            features.extend([feature[col] for col in feature.columns])
                    except Exception as e:
                        logger.warning(f"⚠️ Feature generator {generator.__class__.__name__} failed: {e}")
                        continue
                
                if features:
                    result_df = pd.DataFrame(features).T
                    result_df.index = data.index
                    self.vectorbt_stats['features_generated'] += len(features)
                    logger.info(f"✅ Generated {len(features)} VectorBT features")
                    return result_df
                else:
                    logger.warning("⚠️ No VectorBT features generated")
                    return pd.DataFrame(index=data.index)
        
        except Exception as e:
            logger.error(f"❌ VectorBT feature generation failed: {e}")
            return pd.DataFrame(index=data.index)
    
    def run_vectorbt_backtest(self, signals: Union[np.ndarray, pd.DataFrame], 
                            prices: Union[np.ndarray, pd.DataFrame],
                            timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
                            mode: str = 'cpu') -> Optional[Dict[str, Any]]:
        """Run VectorBT backtest on model predictions."""
        if not self.config.enable_vectorbt_backtesting or not self.vectorbt_backtesting_engine:
            logger.warning("⚠️ VectorBT backtesting not enabled or engine not available")
            return None
        
        try:
            # Convert mode string to BacktestMode enum
            if mode == 'gpu':
                backtest_mode = BacktestMode.VECTORBT_GPU
            elif mode == 'parallel':
                backtest_mode = BacktestMode.VECTORBT_PARALLEL
            elif mode == 'hybrid':
                backtest_mode = BacktestMode.HYBRID
            else:
                backtest_mode = BacktestMode.VECTORBT_CPU
            
            with monitor_operation(
                f"vectorbt_backtest_{mode}",
                metadata={'signals_shape': signals.shape if hasattr(signals, 'shape') else len(signals),
                         'prices_shape': prices.shape if hasattr(prices, 'shape') else len(prices)}
            ):
                results = self.vectorbt_backtesting_engine.run_backtest(
                    signals=signals,
                    prices=prices,
                    timestamps=timestamps,
                    mode=backtest_mode
                )
                
                self.vectorbt_stats['backtests_run'] += 1
                logger.info(f"✅ VectorBT backtest completed with mode: {mode}")
                return {
                    'results': results,
                    'performance_metrics': results.performance_metrics,
                    'risk_metrics': results.risk_metrics,
                    'drawdown_analysis': results.drawdown_analysis,
                    'computation_time': results.computation_time,
                    'memory_usage': results.memory_usage
                }
        
        except Exception as e:
            logger.error(f"❌ VectorBT backtest failed: {e}")
            return None
    
    def calculate_vectorbt_metrics(self, portfolio_values: Union[np.ndarray, pd.Series],
                                 returns: Optional[Union[np.ndarray, pd.Series]] = None,
                                 benchmark_values: Optional[Union[np.ndarray, pd.Series]] = None,
                                 timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None) -> Optional[Dict[str, Any]]:
        """Calculate comprehensive financial metrics using VectorBT."""
        if not self.config.enable_vectorbt_metrics or not self.vectorbt_metrics_calculator:
            logger.warning("⚠️ VectorBT metrics calculation not enabled or calculator not available")
            return None
        
        try:
            with monitor_operation(
                "vectorbt_metrics_calculation",
                metadata={'portfolio_shape': portfolio_values.shape if hasattr(portfolio_values, 'shape') else len(portfolio_values)}
            ):
                metrics = self.vectorbt_metrics_calculator.calculate_comprehensive_metrics(
                    portfolio_values=portfolio_values,
                    returns=returns,
                    benchmark_values=benchmark_values,
                    timestamps=timestamps
                )
                
                self.vectorbt_stats['metrics_calculated'] += 1
                logger.info(f"✅ Calculated {len(metrics)} VectorBT financial metrics")
                return metrics
        
        except Exception as e:
            logger.error(f"❌ VectorBT metrics calculation failed: {e}")
            return None
    
    def get_vectorbt_stats(self) -> Dict[str, Any]:
        """Get VectorBT performance statistics."""
        stats = self.vectorbt_stats.copy()
        
        # Add memory manager stats if available
        if self.memory_manager:
            memory_stats = self.memory_manager.get_memory_stats()
            stats.update({
                'memory_usage_gb': memory_stats.get('current_usage_gb', 0),
                'memory_peak_gb': memory_stats.get('peak_usage_gb', 0),
                'memory_available_gb': memory_stats.get('available_memory_gb', 0),
                'memory_utilization': memory_stats.get('usage_percentage', 0)
            })
        
        # Add performance monitor stats if available
        if self.performance_monitor:
            perf_stats = self.performance_monitor.get_performance_summary()
            stats.update({
                'total_operations_monitored': perf_stats.get('total_operations', 0),
                'average_operation_duration': perf_stats.get('average_duration', 0),
                'gpu_utilization_rate': perf_stats.get('gpu_utilization_rate', 0),
                'cache_hit_rate': perf_stats.get('cache_hit_rate', 0),
                'error_rate': perf_stats.get('error_rate', 0)
            })
        
        return stats
    
    def reset_vectorbt_stats(self):
        """Reset VectorBT performance statistics."""
        self.vectorbt_stats = {
            'backtests_run': 0,
            'metrics_calculated': 0,
            'features_generated': 0,
            'memory_optimizations': 0,
            'performance_operations': 0
        }


# Factory functions
def create_patchtst_model(sequence_length: int = 24,
                         hidden_size: int = 64,
                         enable_vectorbt: bool = True,
                         **kwargs) -> VectorBTEnhancedModelInterface:
    """Create PatchTST model with VectorBT integration."""
    config = UnifiedModelConfig(
        model_type=ModelType.PATCHTST,
        model_name="patchtst_model",
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        enable_vectorbt=enable_vectorbt,
        **kwargs
    )
    return VectorBTEnhancedModelInterface(config)


def create_gru_model(sequence_length: int = 24,
                    hidden_size: int = 64,
                    enable_vectorbt: bool = True,
                    **kwargs) -> VectorBTEnhancedModelInterface:
    """Create GRU model with VectorBT integration."""
    config = UnifiedModelConfig(
        model_type=ModelType.GRU,
        model_name="gru_model",
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        enable_vectorbt=enable_vectorbt,
        **kwargs
    )
    return VectorBTEnhancedModelInterface(config)


def create_tft_model(sequence_length: int = 24,
                    hidden_size: int = 64,
                    enable_vectorbt: bool = True,
                    **kwargs) -> VectorBTEnhancedModelInterface:
    """Create TFT model with VectorBT integration."""
    config = UnifiedModelConfig(
        model_type=ModelType.TFT,
        model_name="tft_model",
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        enable_vectorbt=enable_vectorbt,
        **kwargs
    )
    return VectorBTEnhancedModelInterface(config)


def create_model(model_type: str,
                sequence_length: int = 24,
                hidden_size: int = 64,
                enable_vectorbt: bool = True,
                **kwargs) -> VectorBTEnhancedModelInterface:
    """Create any model type with VectorBT integration."""
    model_type_enum = ModelType(model_type.lower())
    
    config = UnifiedModelConfig(
        model_type=model_type_enum,
        model_name=f"{model_type}_model",
        sequence_length=sequence_length,
        hidden_size=hidden_size,
        enable_vectorbt=enable_vectorbt,
        **kwargs
    )
    return VectorBTEnhancedModelInterface(config)


# Convenience function for creating all models
def create_all_models(sequence_length: int = 24,
                     hidden_size: int = 64,
                     enable_vectorbt: bool = True,
                     **kwargs) -> Dict[str, VectorBTEnhancedModelInterface]:
    """Create all model types with VectorBT integration."""
    models = {}
    
    for model_type in ModelType:
        try:
            models[model_type.value] = create_model(
                model_type=model_type.value,
                sequence_length=sequence_length,
                hidden_size=hidden_size,
                enable_vectorbt=enable_vectorbt,
                **kwargs
            )
            logger.info(f"✅ Created {model_type.value} model")
        except Exception as e:
            logger.warning(f"⚠️ Failed to create {model_type.value} model: {e}")
    
    return models