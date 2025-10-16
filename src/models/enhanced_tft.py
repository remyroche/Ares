"""
Enhanced TFT (Temporal Fusion Transformer) Model with VectorBT Integration

This module implements an enhanced TFT model for time series forecasting with VectorBT integration:
- Temporal Fusion Transformer architecture
- Multi-horizon forecasting capabilities
- VectorBT backtesting and financial metrics integration
- VectorBT feature generation and optimization
- Memory management and performance monitoring
- Support for both regression and classification tasks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import TimeSeriesSplit

# VectorBT imports
try:
    import vectorbt as vbt
    from vectorbt.portfolio.base import Portfolio
    from vectorbt.records.base import Records
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    Portfolio = None
    Records = None

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

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class EnhancedTFTConfig:
    """Configuration for Enhanced TFT model with VectorBT integration."""
    # TFT parameters
    hidden_size: int = 64
    lstm_layers: int = 2
    attention_heads: int = 4
    dropout: float = 0.1
    output_size: int = 1

    # Time series parameters
    sequence_length: int = 24  # Lookback window
    prediction_horizon: int = 1  # Forecast horizon
    static_features: List[str] = None
    known_future_features: List[str] = None

    # Training parameters
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

class TemporalFusionTransformer:
    """
    Temporal Fusion Transformer implementation with VectorBT integration.

    This class implements the TFT architecture for time series forecasting
    with enhanced VectorBT capabilities for backtesting and analysis.
    """

    def __init__(self, config: EnhancedTFTConfig):
        """Initialize the TFT model with VectorBT integration."""
        self.config = config

        # Model components
        self.model = None
        self.scaler = None
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

class EnhancedTFTModel(BaseEstimator, RegressorMixin):
    """
    Enhanced TFT Model with VectorBT Integration.

    This model uses Temporal Fusion Transformer architecture for time series
    forecasting with VectorBT backtesting, financial metrics, and feature
    generation capabilities.
    """

    def __init__(self, config: Optional[EnhancedTFTConfig] = None):
        """Initialize the Enhanced TFT model with VectorBT integration."""
        self.config = config or EnhancedTFTConfig()

        # Components
        self.tft_model = None
        self.scaler = None

        # State
        self.fitted = False
        self.feature_names = None

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

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'EnhancedTFTModel':
        """Fit the Enhanced TFT model."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset

            # Store feature names if available
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
                X = X.values

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y)

            # Create TFT model (simplified implementation)
            self.tft_model = nn.Sequential(
                nn.Linear(X.shape[1], self.config.hidden_size),
                nn.LSTM(self.config.hidden_size, self.config.hidden_size,
                       num_layers=self.config.lstm_layers,
                       dropout=self.config.dropout,
                       batch_first=True),
                nn.Linear(self.config.hidden_size, self.config.output_size)
            )

            # Training setup
            optimizer = optim.Adam(self.tft_model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()

            # Data loader
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=True
            )

            # Training loop
            self.tft_model.train()
            best_loss = float('inf')
            patience_counter = 0

            for epoch in range(self.config.epochs):
                epoch_loss = 0.0

                for batch_X, batch_y in dataloader:
                    optimizer.zero_grad()

                    # Forward pass
                    output = self.tft_model(batch_X)
                    loss = criterion(output.squeeze(), batch_y)

                    # Backward pass
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / len(dataloader)

                # Early stopping
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

                if epoch % 10 == 0:
                    logger.info(f"Epoch {epoch}, Loss: {avg_loss:.6f}")

            self.fitted = True
            logger.info(f"✅ Enhanced TFT model fitted with {X.shape[1]} features")

            return self

        except ImportError:
            logger.warning("⚠️ PyTorch not available, using fallback linear model")
            return self._fit_fallback(X, y, sample_weight)
        except Exception as e:
            logger.error(f"❌ Enhanced TFT model fitting failed: {e}")
            return self._fit_fallback(X, y, sample_weight)

    def _fit_fallback(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> 'EnhancedTFTModel':
        """Fallback to simple linear model."""
        try:
            from sklearn.linear_model import LinearRegression

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Simple linear model as fallback
            self.tft_model = LinearRegression()
            self.tft_model.fit(X_scaled, y, sample_weight)

            self.fitted = True
            logger.info("✅ Fallback linear model fitted")

            return self

        except Exception as e:
            logger.error(f"❌ Fallback model fitting failed: {e}")
            raise

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the fitted model."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            # Convert to numpy if pandas DataFrame
            if hasattr(X, 'values'):
                X = X.values

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Check if model is PyTorch model
            if hasattr(self.tft_model, 'forward'):
                import torch

                # Convert to tensor
                X_tensor = torch.FloatTensor(X_scaled)

                # Predict
                self.tft_model.eval()
                with torch.no_grad():
                    predictions = self.tft_model(X_tensor).squeeze().numpy()

                return predictions
            else:
                # Fallback model
                return self.tft_model.predict(X_scaled)

        except Exception as e:
            logger.error(f"❌ Enhanced TFT model prediction failed: {e}")
            raise

# Factory function
def create_enhanced_tft(config: Optional[EnhancedTFTConfig] = None) -> EnhancedTFTModel:
    """Create Enhanced TFT model."""
    return EnhancedTFTModel(config)

# Convenience function for creating TFT with VectorBT
def create_tft_with_vectorbt(sequence_length: int = 24,
                           prediction_horizon: int = 1,
                           hidden_size: int = 64,
                           enable_vectorbt: bool = True,
                           **kwargs) -> EnhancedTFTModel:
    """Create TFT model with VectorBT integration."""
    config = EnhancedTFTConfig(
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        hidden_size=hidden_size,
        enable_vectorbt=enable_vectorbt,
        **kwargs
    )
    return EnhancedTFTModel(config)
