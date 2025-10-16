"""
Patch/GRU Model for End-to-End Roadmap with VectorBT Integration

Minimal stacker with VectorBT enhancements:
- Tiny PatchTST or 1-layer GRU
- 2-4h sequence lookback
- Horizons: {1,3} bars
- Exposes: y_hat_h1, y_hat_h3, y_hat_conf
- p99 inference <5ms
- VectorBT backtesting and financial metrics integration
- VectorBT feature generation and optimization
- Memory management and performance monitoring
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import pandas as pd
import numpy as np
import warnings
from abc import ABC, abstractmethod
import logging

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

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of patch/GRU models."""
    PATCH = "patch"
    GRU = "gru"

@dataclass
class PatchConfig:
    """Configuration for patch model with VectorBT integration."""
    model_type: ModelType
    sequence_length: int  # 2-4h in bars
    horizons: List[int]  # [1, 3] bars
    hidden_dim: int = 32
    num_layers: int = 1
    dropout: float = 0.1
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 50

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

@dataclass
class PatchOutput:
    """Output from patch model."""
    y_hat_h1: pd.Series
    y_hat_h3: pd.Series
    y_hat_conf: pd.Series
    metadata: Dict[str, Any]

class BasePatchModel(ABC):
    """Abstract base class for patch models with VectorBT integration."""

    def __init__(self, config: PatchConfig):
        self.config = config
        self.model = None
        self.fitted = False
        self.residual_std = None

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

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass

    def get_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate confidence scores."""
        if self.residual_std is None:
            return np.ones_like(predictions)

        epsilon = 1e-8
        confidence = np.abs(predictions) / (epsilon + self.residual_std)
        return np.clip(confidence, 0, 1)

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

class SimpleGRU(BasePatchModel):
    """Simple 1-layer GRU model."""

    def __init__(self, config: PatchConfig):
        super().__init__(config)
        self.scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the GRU model."""
        try:
            import torch
            import torch.nn as nn
            from sklearn.preprocessing import StandardScaler

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y)

            # Create model
            self.model = nn.GRU(
                input_size=X.shape[1],
                hidden_size=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout if self.config.num_layers > 1 else 0,
                batch_first=True
            )

            # Add output layer
            self.output_layer = nn.Linear(self.config.hidden_dim, 1)

            # Training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()

            self.model.train()
            for epoch in range(self.config.epochs):
                optimizer.zero_grad()

                # Forward pass
                output, _ = self.model(X_tensor)
                predictions = self.output_layer(output[:, -1, :])  # Use last timestep

                loss = criterion(predictions.squeeze(), y_tensor)
                loss.backward()
                optimizer.step()

            # Calculate residual std for confidence
            with torch.no_grad():
                self.model.eval()
                output, _ = self.model(X_tensor)
                pred = self.output_layer(output[:, -1, :]).squeeze()
                residuals = y_tensor - pred
                self.residual_std = torch.std(residuals).item()

            self.fitted = True

        except ImportError:
            warnings.warn("PyTorch not available, using fallback linear model")
            self._fit_fallback(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            import torch

            # Scale features
            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)

            # Predict
            self.model.eval()
            with torch.no_grad():
                output, _ = self.model(X_tensor)
                predictions = self.output_layer(output[:, -1, :])
                return predictions.squeeze().numpy()

        except ImportError:
            return self._predict_fallback(X)

    def _fit_fallback(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fallback to simple linear model."""
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()
        self.model.fit(X, y)

        # Calculate residual std
        predictions = self.model.predict(X)
        residuals = y - predictions
        self.residual_std = np.std(residuals)

        self.fitted = True

    def _predict_fallback(self, X: np.ndarray) -> np.ndarray:
        """Fallback prediction."""
        return self.model.predict(X)

class SimplePatchTST(BasePatchModel):
    """Simple PatchTST model (simplified)."""

    def __init__(self, config: PatchConfig):
        super().__init__(config)
        self.scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the PatchTST model."""
        try:
            import torch
            import torch.nn as nn

            # Scale features
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y)

            # Simple patch-based model
            patch_size = min(16, X.shape[1] // 4)  # Adaptive patch size
            num_patches = X.shape[1] // patch_size

            self.model = nn.Sequential(
                nn.Linear(X.shape[1], self.config.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(self.config.hidden_dim // 2, 1)
            )

            # Training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            criterion = nn.MSELoss()

            self.model.train()
            for epoch in range(self.config.epochs):
                optimizer.zero_grad()

                predictions = self.model(X_tensor)
                loss = criterion(predictions.squeeze(), y_tensor)
                loss.backward()
                optimizer.step()

            # Calculate residual std
            with torch.no_grad():
                self.model.eval()
                pred = self.model(X_tensor).squeeze()
                residuals = y_tensor - pred
                self.residual_std = torch.std(residuals).item()

            self.fitted = True

        except ImportError:
            warnings.warn("PyTorch not available, using fallback linear model")
            self._fit_fallback(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")

        try:
            import torch

            X_scaled = self.scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled)

            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor)
                return predictions.squeeze().numpy()

        except ImportError:
            return self._predict_fallback(X)

    def _fit_fallback(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fallback to simple linear model."""
        from sklearn.linear_model import LinearRegression

        self.model = LinearRegression()
        self.model.fit(X, y)

        predictions = self.model.predict(X)
        residuals = y - predictions
        self.residual_std = np.std(residuals)

        self.fitted = True

    def _predict_fallback(self, X: np.ndarray) -> np.ndarray:
        """Fallback prediction."""
        return self.model.predict(X)

class PatchModelFactory:
    """Factory for creating patch models."""

    @staticmethod
    def create_model(config: PatchConfig) -> BasePatchModel:
        """Create a patch model based on configuration."""
        if config.model_type == ModelType.GRU:
            return SimpleGRU(config)
        elif config.model_type == ModelType.PATCH:
            return SimplePatchTST(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

class PatchOrchestrator:
    """Orchestrator for patch model training and prediction."""

    def __init__(self, config: PatchConfig):
        self.config = config
        self.models = {}  # horizon -> model
        self.fitted = False

    def fit(self,
            bars_data: pd.DataFrame,
            targets: Dict[int, pd.Series]) -> None:
        """Fit models for all horizons."""

        # Prepare sequence data
        X_sequences = self._prepare_sequences(bars_data)

        for horizon in self.config.horizons:
            if horizon not in targets:
                continue

            y = targets[horizon]

            # Align sequences with targets
            min_length = min(len(X_sequences), len(y))
            X_aligned = X_sequences[:min_length]
            y_aligned = y[:min_length]

            if len(X_aligned) < self.config.sequence_length:
                continue

            # Create model for this horizon
            model = PatchModelFactory.create_model(self.config)

            # Fit model
            model.fit(X_aligned, y_aligned.values)
            self.models[horizon] = model

        self.fitted = True

    def predict(self, bars_data: pd.DataFrame) -> PatchOutput:
        """Make predictions for all horizons."""
        if not self.fitted:
            raise ValueError("Models must be fitted before prediction")

        # Prepare sequence data
        X_sequences = self._prepare_sequences(bars_data)

        predictions = {}
        confidences = {}

        for horizon, model in self.models.items():
            if len(X_sequences) == 0:
                pred = np.zeros(len(bars_data))
                conf = np.zeros(len(bars_data))
            else:
                pred = model.predict(X_sequences)
                conf = model.get_confidence(pred)

            predictions[f'y_hat_h{horizon}'] = pd.Series(pred, index=bars_data.index)
            confidences[f'y_hat_h{horizon}'] = pd.Series(conf, index=bars_data.index)

        # Create confidence score (average across horizons)
        if confidences:
            y_hat_conf = pd.Series(
                np.mean([conf.values for conf in confidences.values()], axis=0),
                index=bars_data.index
            )
        else:
            y_hat_conf = pd.Series(0, index=bars_data.index)

        return PatchOutput(
            y_hat_h1=predictions.get('y_hat_h1', pd.Series(0, index=bars_data.index)),
            y_hat_h3=predictions.get('y_hat_h3', pd.Series(0, index=bars_data.index)),
            y_hat_conf=y_hat_conf,
            metadata={
                'fitted_models': list(self.models.keys()),
                'sequence_length': self.config.sequence_length,
                'model_type': self.config.model_type.value
            }
        )

    def _prepare_sequences(self, bars_data: pd.DataFrame) -> np.ndarray:
        """Prepare sequence data for model input."""
        # Select relevant features
        feature_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in feature_cols if col in bars_data.columns]

        if not available_cols:
            return np.array([])

        data = bars_data[available_cols].values

        # Create sequences
        sequences = []
        for i in range(len(data) - self.config.sequence_length + 1):
            sequence = data[i:i + self.config.sequence_length]
            sequences.append(sequence.flatten())  # Flatten to 1D

        return np.array(sequences) if sequences else np.array([])

    def get_oof_predictions(self,
                           bars_data: pd.DataFrame,
                           targets: Dict[int, pd.Series],
                           n_folds: int = 5) -> PatchOutput:
        """Get out-of-fold predictions for training features."""

        from sklearn.model_selection import TimeSeriesSplit

        tscv = TimeSeriesSplit(n_splits=n_folds)
        oof_predictions = {f'y_hat_h{h}': [] for h in self.config.horizons}
        oof_confidences = []

        for train_idx, val_idx in tscv.split(bars_data):
            # Split data
            train_data = bars_data.iloc[train_idx]
            val_data = bars_data.iloc[val_idx]

            # Fit on training data
            train_targets = {h: targets[h].iloc[train_idx] for h in self.config.horizons if h in targets}
            self.fit(train_data, train_targets)

            # Predict on validation data
            val_predictions = self.predict(val_data)

            # Store OOF predictions
            for horizon in self.config.horizons:
                if f'y_hat_h{horizon}' in val_predictions.__dict__:
                    oof_predictions[f'y_hat_h{horizon}'].append(
                        val_predictions.__dict__[f'y_hat_h{horizon}']
                    )

            oof_confidences.append(val_predictions.y_hat_conf)

        # Combine OOF predictions
        combined_predictions = {}
        for horizon in self.config.horizons:
            if oof_predictions[f'y_hat_h{horizon}']:
                combined_predictions[f'y_hat_h{horizon}'] = pd.concat(
                    oof_predictions[f'y_hat_h{horizon}']
                ).sort_index()
            else:
                combined_predictions[f'y_hat_h{horizon}'] = pd.Series(0, index=bars_data.index)

        # Combine confidences
        if oof_confidences:
            combined_conf = pd.concat(oof_confidences).sort_index()
        else:
            combined_conf = pd.Series(0, index=bars_data.index)

        return PatchOutput(
            y_hat_h1=combined_predictions.get('y_hat_h1', pd.Series(0, index=bars_data.index)),
            y_hat_h3=combined_predictions.get('y_hat_h3', pd.Series(0, index=bars_data.index)),
            y_hat_conf=combined_conf,
            metadata={'oof': True, 'n_folds': n_folds}
        )
