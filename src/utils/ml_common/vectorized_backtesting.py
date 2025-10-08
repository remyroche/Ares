"""
Vectorized Backtesting Engine

This module provides highly optimized backtesting using vectorized operations,
matrix computations, and GPU acceleration for maximum performance.
"""

import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

from src.training.steps.pre_training.profit_labeling.enhanced_label_definitions import TradingCosts

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# Import matrix operations if available
try:
    from ..matrix_operations import get_unified_matrix_operations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

logger = logging.getLogger(__name__)


class BacktestMode(Enum):
    """Backtesting execution modes."""
    VECTORIZED = "vectorized"
    PARALLEL = "parallel"
    GPU_ACCELERATED = "gpu_accelerated"
    HYBRID = "hybrid"


@dataclass
class VectorizedBacktestConfig:
    """Configuration for vectorized backtesting."""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    max_position_size: float = 0.1
    min_position_size: float = 0.01
    rebalance_frequency: str = 'daily'
    risk_free_rate: float = 0.02
    benchmark_symbol: Optional[str] = None
    trading_costs: TradingCosts = field(default_factory=TradingCosts)
    asset_classes: Optional[List[str]] = None
    stress_scenario: Optional[str] = None

    # Vectorization settings
    use_gpu: bool = True
    batch_size: int = 10000
    enable_parallel: bool = True
    chunk_size: int = 50000

    # Performance settings
    enable_memory_optimization: bool = True
    enable_progress_tracking: bool = True
    cache_intermediate_results: bool = True


@dataclass
class VectorizedBacktestResults:
    """Results from vectorized backtesting."""
    portfolio_values: np.ndarray
    returns: np.ndarray
    positions: np.ndarray
    trades: pd.DataFrame
    performance_metrics: Dict[str, float]
    risk_metrics: Dict[str, float]
    drawdown_analysis: Dict[str, Any]
    computation_time: float
    memory_usage: float


class VectorizedBacktestingEngine:
    """
    High-performance vectorized backtesting engine.

    This engine uses matrix operations and GPU acceleration to provide
    significant performance improvements over traditional loop-based backtesting.
    """

    def __init__(self, config: Optional[VectorizedBacktestConfig] = None):
        """
        Initialize vectorized backtesting engine.

        Args:
            config: Backtesting configuration
        """
        self.config = config or VectorizedBacktestConfig()

        # Initialize matrix operations if available
        if MATRIX_OPS_AVAILABLE:
            self.matrix_ops = get_unified_matrix_operations()
        else:
            self.matrix_ops = None

        # GPU availability check
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                self.gpu_available = True
            elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = torch.device('mps')
                self.gpu_available = True
            else:
                self.device = torch.device('cpu')
                self.gpu_available = False
        else:
            self.device = None
            self.gpu_available = False

        # Performance tracking
        self.performance_stats = {
            'total_simulations': 0,
            'computation_time': 0.0,
            'memory_peak': 0.0,
            'gpu_operations': 0,
            'cpu_operations': 0
        }

        logger.info("✅ Vectorized backtesting engine initialized")
        logger.info(f"📊 GPU available: {self.gpu_available}")
        logger.info(f"📊 Matrix ops available: {self.matrix_ops is not None}")

    def run_vectorized_backtest(self, signals: Union[np.ndarray, pd.DataFrame],
                               prices: Union[np.ndarray, pd.DataFrame],
                               timestamps: Optional[Union[np.ndarray, pd.DatetimeIndex]] = None,
                               mode: BacktestMode = BacktestMode.VECTORIZED) -> VectorizedBacktestResults:
        """
        Run vectorized backtest simulation.

        Args:
            signals: Trading signals (-1, 0, 1 for short, neutral, long)
            prices: Asset prices
            timestamps: Time index for the data
            mode: Execution mode (vectorized, parallel, gpu, hybrid)

        Returns:
            Comprehensive backtest results
        """
        start_time = time.time()
        logger.info(f"🚀 Starting vectorized backtest with mode: {mode.value}")

        # Convert inputs to numpy arrays
        signals_array = self._convert_to_array(signals)
        prices_array = self._convert_to_array(prices)

        if timestamps is not None:
            if isinstance(timestamps, pd.DatetimeIndex):
                timestamps_array = timestamps.values.astype('datetime64[ns]')
            else:
                timestamps_array = timestamps
        else:
            timestamps_array = None

        # Validate inputs
        self._validate_inputs(signals_array, prices_array)

        logger.info(f"📊 Data shapes - Signals: {signals_array.shape}, Prices: {prices_array.shape}")

        # Execute backtest based on mode
        if mode == BacktestMode.GPU_ACCELERATED and self.gpu_available:
            results = self._gpu_accelerated_backtest(signals_array, prices_array, timestamps_array)
        elif mode == BacktestMode.PARALLEL:
            results = self._parallel_backtest(signals_array, prices_array, timestamps_array)
        elif mode == BacktestMode.HYBRID:
            results = self._hybrid_backtest(signals_array, prices_array, timestamps_array)
        else:
            results = self._vectorized_backtest(signals_array, prices_array, timestamps_array)

        # Calculate performance metrics
        results = self._calculate_performance_metrics(results, prices_array, timestamps_array)

        # Update performance stats
        computation_time = time.time() - start_time
        self.performance_stats['computation_time'] = computation_time
        self.performance_stats['total_simulations'] += 1

        results.computation_time = computation_time

        logger.info(f"✅ Vectorized backtest completed in {computation_time:.3f}s")
        logger.info(f"📊 Final portfolio value: ${results.portfolio_values[-1]:.2f}")
        return results

    def _vectorized_backtest(self, signals: np.ndarray, prices: np.ndarray,
                           timestamps: Optional[np.ndarray] = None) -> VectorizedBacktestResults:
        """
        Pure vectorized backtest implementation using NumPy operations.
        """
        n_periods = len(prices)
        n_assets = prices.shape[1] if prices.ndim > 1 else 1

        # Initialize portfolio tracking arrays
        portfolio_values = np.zeros(n_periods)
        portfolio_values[0] = self.config.initial_capital

        positions = np.zeros((n_periods, n_assets))
        returns = np.zeros(n_periods)

        # Calculate position sizes (vectorized)
        position_sizes = self._calculate_position_sizes_vectorized(signals, prices)

        # Calculate returns (vectorized)
        price_returns = self._calculate_price_returns_vectorized(prices)

        # Apply commission and slippage (vectorized)
        trading_costs = self._calculate_trading_costs_vectorized(signals, prices, position_sizes)

        # Calculate portfolio returns (vectorized)
        portfolio_returns = self._calculate_portfolio_returns_vectorized(
            position_sizes, price_returns, trading_costs
        )

        # Calculate cumulative portfolio values (vectorized)
        portfolio_values = self._calculate_portfolio_values_vectorized(
            portfolio_returns, self.config.initial_capital
        )

        # Extract trades
        trades_df = self._extract_trades_vectorized(signals, prices, timestamps)

        # Initialize results object
        results = VectorizedBacktestResults(
            portfolio_values=portfolio_values,
            returns=portfolio_returns,
            positions=position_sizes,
            trades=trades_df,
            performance_metrics={},
            risk_metrics={},
            drawdown_analysis={},
            computation_time=0.0,
            memory_usage=0.0
        )

        return results

    def _gpu_accelerated_backtest(self, signals: np.ndarray, prices: np.ndarray,
                                timestamps: Optional[np.ndarray] = None) -> VectorizedBacktestResults:
        """
        GPU-accelerated backtest implementation using PyTorch.
        """
        if not self.gpu_available:
            logger.warning("⚠️ GPU not available, falling back to CPU")
            return self._vectorized_backtest(signals, prices, timestamps)

        # Convert to PyTorch tensors
        signals_tensor = torch.from_numpy(signals.astype(np.float32)).to(self.device)
        prices_tensor = torch.from_numpy(prices.astype(np.float32)).to(self.device)

        # GPU-accelerated calculations
        position_sizes = self._calculate_position_sizes_gpu(signals_tensor, prices_tensor)
        price_returns = self._calculate_price_returns_gpu(prices_tensor)
        trading_costs = self._calculate_trading_costs_gpu(signals_tensor, prices_tensor, position_sizes)
        portfolio_returns = self._calculate_portfolio_returns_gpu(position_sizes, price_returns, trading_costs)
        portfolio_values = self._calculate_portfolio_values_gpu(portfolio_returns, self.config.initial_capital)

        # Convert back to CPU
        portfolio_values_np = portfolio_values.cpu().numpy()
        portfolio_returns_np = portfolio_returns.cpu().numpy()
        position_sizes_np = position_sizes.cpu().numpy()

        # Extract trades
        trades_df = self._extract_trades_vectorized(signals, prices, timestamps)

        # Update GPU operation count
        self.performance_stats['gpu_operations'] += 1

        results = VectorizedBacktestResults(
            portfolio_values=portfolio_values_np,
            returns=portfolio_returns_np,
            positions=position_sizes_np,
            trades=trades_df,
            performance_metrics={},
            risk_metrics={},
            drawdown_analysis={},
            computation_time=0.0,
            memory_usage=0.0
        )

        return results

    def _parallel_backtest(self, signals: np.ndarray, prices: np.ndarray,
                         timestamps: Optional[np.ndarray] = None) -> VectorizedBacktestResults:
        """
        Parallel backtest implementation using multiprocessing.
        """
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp

        n_cpus = max(1, mp.cpu_count() - 1)  # Leave one CPU free
        chunk_size = max(1, len(prices) // n_cpus)

        # Split data into chunks
        chunks = []
        for i in range(0, len(prices), chunk_size):
            end_idx = min(i + chunk_size, len(prices))
            chunk_signals = signals[i:end_idx]
            chunk_prices = prices[i:end_idx]
            chunk_timestamps = timestamps[i:end_idx] if timestamps is not None else None
            chunks.append((chunk_signals, chunk_prices, chunk_timestamps, i))

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=n_cpus) as executor:
            futures = [
                executor.submit(self._process_backtest_chunk, chunk)
                for chunk in chunks
            ]

            # Collect results
            chunk_results = []
            for future in futures:
                chunk_results.append(future.result())

        # Combine results
        return self._combine_chunk_results(chunk_results)

    def _hybrid_backtest(self, signals: np.ndarray, prices: np.ndarray,
                       timestamps: Optional[np.ndarray] = None) -> VectorizedBacktestResults:
        """
        Hybrid backtest combining GPU and parallel processing.
        """
        if self.gpu_available:
            return self._gpu_accelerated_backtest(signals, prices, timestamps)
        else:
            return self._parallel_backtest(signals, prices, timestamps)

    def _calculate_position_sizes_vectorized(self, signals: np.ndarray, prices: np.ndarray) -> np.ndarray:
        """Calculate position sizes using vectorized operations."""
        # Simple position sizing based on signals
        position_sizes = signals.astype(np.float32)

        # Apply position size limits
        max_size = self.config.max_position_size * self.config.initial_capital
        min_size = self.config.min_position_size * self.config.initial_capital

        # Scale positions by available capital (simplified)
        position_sizes = position_sizes * max_size

        # Apply minimum position size filter
        position_sizes = np.where(np.abs(position_sizes) < min_size, 0, position_sizes)

        return position_sizes

    def _calculate_price_returns_vectorized(self, prices: np.ndarray) -> np.ndarray:
        """Calculate price returns using vectorized operations."""
        if prices.ndim == 1:
            # Single asset
            returns = np.diff(prices, prepend=prices[0]) / prices
            returns[0] = 0  # First return is undefined
        else:
            # Multiple assets
            returns = np.diff(prices, axis=0, prepend=prices[0:1]) / prices
            returns[0] = 0  # First return is undefined

        return returns

    def _compute_additional_trading_costs(
        self,
        position_sizes: np.ndarray,
        scenario: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute borrow/funding costs and stress multipliers for positions."""
        trading_costs = self.config.trading_costs
        if trading_costs is None:
            zeros = np.zeros_like(position_sizes, dtype=float)
            ones = np.ones_like(position_sizes, dtype=float)
            return zeros, ones

        if position_sizes.ndim == 1:
            asset_classes = [trading_costs.default_asset_class]
        else:
            if self.config.asset_classes is not None:
                if len(self.config.asset_classes) != position_sizes.shape[1]:
                    raise ValueError(
                        "asset_classes length must match number of assets in position_sizes"
                    )
                asset_classes = list(self.config.asset_classes)
            else:
                asset_classes = [trading_costs.default_asset_class] * position_sizes.shape[1]

        trading_costs.validate_asset_assumptions(asset_classes)

        scenario_key = scenario or self.config.stress_scenario or trading_costs.active_stress_scenario

        if position_sizes.ndim == 1:
            positions = position_sizes.astype(float)
            directions = np.where(positions >= 0, 'long', 'short')
            borrow_rates = np.array([
                trading_costs.get_borrow_rate(asset_classes[0], direction)
                for direction in directions
            ])
            funding_rates = np.array([
                trading_costs.get_funding_rate(asset_classes[0], direction)
                for direction in directions
            ])
            stress_factors = np.array([
                trading_costs.get_stress_multiplier(asset_classes[0], direction, scenario_key)
                for direction in directions
            ])

            additional_costs = np.abs(positions) * (borrow_rates + funding_rates)
            return additional_costs, stress_factors

        additional_costs = np.zeros_like(position_sizes, dtype=float)
        stress_factors = np.ones_like(position_sizes, dtype=float)

        for col, asset_class in enumerate(asset_classes):
            positions = position_sizes[:, col].astype(float)
            directions = np.where(positions >= 0, 'long', 'short')
            borrow_rates = np.array([
                trading_costs.get_borrow_rate(asset_class, direction)
                for direction in directions
            ])
            funding_rates = np.array([
                trading_costs.get_funding_rate(asset_class, direction)
                for direction in directions
            ])
            stress_factors[:, col] = np.array([
                trading_costs.get_stress_multiplier(asset_class, direction, scenario_key)
                for direction in directions
            ])
            additional_costs[:, col] = np.abs(positions) * (borrow_rates + funding_rates)

        return additional_costs, stress_factors

    def _calculate_trading_costs_vectorized(self, signals: np.ndarray, prices: np.ndarray,
                                          position_sizes: np.ndarray) -> np.ndarray:
        """Calculate trading costs using vectorized operations."""
        # Detect position changes (trades)
        if position_sizes.ndim == 1:
            position_changes = np.diff(position_sizes, prepend=0)
        else:
            position_changes = np.diff(position_sizes, axis=0, prepend=np.zeros((1, position_sizes.shape[1])))

        # Calculate commission costs
        commission_costs = self.config.commission_rate * np.abs(position_changes)

        # Calculate slippage costs (simplified)
        slippage_costs = self.config.slippage_rate * np.abs(position_changes)

        base_costs = commission_costs + slippage_costs

        additional_costs, stress_factors = self._compute_additional_trading_costs(
            position_sizes,
            scenario=self.config.stress_scenario
        )

        total_costs = (base_costs + additional_costs) * stress_factors

        return total_costs

    def _calculate_portfolio_returns_vectorized(self, position_sizes: np.ndarray,
                                              price_returns: np.ndarray,
                                              trading_costs: np.ndarray) -> np.ndarray:
        """Calculate portfolio returns using vectorized operations."""
        if position_sizes.ndim == 1:
            # Single asset
            gross_returns = position_sizes * price_returns
        else:
            # Multiple assets
            gross_returns = np.sum(position_sizes * price_returns, axis=1)

        # Subtract trading costs
        if trading_costs.ndim == 1:
            net_returns = gross_returns - trading_costs
        else:
            net_returns = gross_returns - np.sum(trading_costs, axis=1)

        return net_returns

    def _calculate_portfolio_values_vectorized(self, returns: np.ndarray,
                                             initial_capital: float) -> np.ndarray:
        """Calculate cumulative portfolio values using vectorized operations."""
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + returns)

        # Calculate portfolio values
        portfolio_values = initial_capital * cumulative_returns

        return portfolio_values

    def _calculate_position_sizes_gpu(self, signals: torch.Tensor, prices: torch.Tensor) -> torch.Tensor:
        """Calculate position sizes using GPU operations."""
        # Simple GPU-based position sizing
        position_sizes = signals.float()

        # Apply position size limits
        max_size = self.config.max_position_size * self.config.initial_capital
        position_sizes = position_sizes * max_size

        return position_sizes

    def _calculate_price_returns_gpu(self, prices: torch.Tensor) -> torch.Tensor:
        """Calculate price returns using GPU operations."""
        if prices.dim() == 1:
            returns = torch.diff(prices, prepend=prices[0]) / prices
            returns[0] = 0
        else:
            returns = torch.diff(prices, dim=0, prepend=prices[0:1]) / prices
            returns[0] = 0

        return returns

    def _calculate_trading_costs_gpu(self, signals: torch.Tensor, prices: torch.Tensor,
                                   position_sizes: torch.Tensor) -> torch.Tensor:
        """Calculate trading costs using GPU operations."""
        if position_sizes.dim() == 1:
            position_changes = torch.diff(position_sizes, prepend=torch.tensor(0.0, device=self.device))
        else:
            position_changes = torch.diff(position_sizes, dim=0,
                                        prepend=torch.zeros((1, position_sizes.shape[1]), device=self.device))

        commission_costs = self.config.commission_rate * torch.abs(position_changes)
        slippage_costs = self.config.slippage_rate * torch.abs(position_changes)
        base_costs = commission_costs + slippage_costs

        additional_costs_np, stress_np = self._compute_additional_trading_costs(
            position_sizes.detach().cpu().numpy(),
            scenario=self.config.stress_scenario
        )

        additional_costs = torch.tensor(additional_costs_np, device=self.device, dtype=position_sizes.dtype)
        stress_factors = torch.tensor(stress_np, device=self.device, dtype=position_sizes.dtype)

        return (base_costs + additional_costs) * stress_factors

    def _calculate_portfolio_returns_gpu(self, position_sizes: torch.Tensor,
                                       price_returns: torch.Tensor,
                                       trading_costs: torch.Tensor) -> torch.Tensor:
        """Calculate portfolio returns using GPU operations."""
        if position_sizes.dim() == 1:
            gross_returns = position_sizes * price_returns
        else:
            gross_returns = torch.sum(position_sizes * price_returns, dim=1)

        if trading_costs.dim() == 1:
            net_returns = gross_returns - trading_costs
        else:
            net_returns = gross_returns - torch.sum(trading_costs, dim=1)

        return net_returns

    def _calculate_portfolio_values_gpu(self, returns: torch.Tensor,
                                      initial_capital: float) -> torch.Tensor:
        """Calculate portfolio values using GPU operations."""
        cumulative_returns = torch.cumprod(1 + returns, dim=0)
        portfolio_values = initial_capital * cumulative_returns

        return portfolio_values

    def _extract_trades_vectorized(self, signals: np.ndarray, prices: np.ndarray,
                                 timestamps: Optional[np.ndarray] = None) -> pd.DataFrame:
        """Extract trades using vectorized operations."""
        # Detect signal changes (entry/exit points)
        if signals.ndim == 1:
            signal_changes = np.diff(signals, prepend=0)
        else:
            signal_changes = np.diff(signals, axis=0, prepend=np.zeros((1, signals.shape[1])))

        # Find non-zero changes (trades)
        trade_indices = np.where(np.abs(signal_changes) > 0)

        if len(trade_indices[0]) == 0:
            # No trades
            return pd.DataFrame(columns=['timestamp', 'signal', 'price', 'position_size'])

        # Extract trade information
        trade_timestamps = timestamps[trade_indices[0]] if timestamps is not None else trade_indices[0]
        trade_signals = signals[trade_indices]
        trade_prices = prices[trade_indices]

        if signals.ndim == 1:
            trade_data = {
                'timestamp': trade_timestamps,
                'signal': trade_signals,
                'price': trade_prices,
                'position_size': trade_signals * self.config.max_position_size * self.config.initial_capital
            }
        else:
            # Multi-asset case
            trade_data = {
                'timestamp': trade_timestamps,
                'asset': trade_indices[1],
                'signal': trade_signals,
                'price': trade_prices,
                'position_size': trade_signals * self.config.max_position_size * self.config.initial_capital
            }

        return pd.DataFrame(trade_data)

    def _calculate_performance_metrics(self, results: VectorizedBacktestResults,
                                     prices: np.ndarray,
                                     timestamps: Optional[np.ndarray] = None) -> VectorizedBacktestResults:
        """Calculate comprehensive performance metrics."""
        portfolio_values = results.portfolio_values
        returns = results.returns

        # Basic metrics
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        annualized_return = self._calculate_annualized_return(total_return, len(portfolio_values))

        # Risk metrics
        volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = self._calculate_sharpe_ratio(returns, volatility)
        max_drawdown, drawdown_analysis = self._calculate_max_drawdown(portfolio_values)

        # Update results
        results.performance_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'final_portfolio_value': portfolio_values[-1],
            'total_trades': len(results.trades)
        }

        results.risk_metrics = {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': self._calculate_sortino_ratio(returns),
            'calmar_ratio': annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        }

        results.drawdown_analysis = drawdown_analysis

        return results

    def _calculate_annualized_return(self, total_return: float, n_periods: int) -> float:
        """Calculate annualized return."""
        if n_periods <= 0:
            return 0.0
        return (1 + total_return) ** (252 / n_periods) - 1

    def _calculate_sharpe_ratio(self, returns: np.ndarray, volatility: float) -> float:
        """Calculate Sharpe ratio."""
        avg_return = np.mean(returns)
        if volatility == 0:
            return 0.0
        return (avg_return * 252 - self.config.risk_free_rate) / volatility

    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return 0.0

        downside_volatility = np.std(downside_returns) * np.sqrt(252)
        avg_return = np.mean(returns)

        if downside_volatility == 0:
            return 0.0

        return (avg_return * 252 - self.config.risk_free_rate) / downside_volatility

    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> Tuple[float, Dict[str, Any]]:
        """Calculate maximum drawdown and drawdown analysis."""
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak
        max_drawdown = np.min(drawdown)

        # Drawdown analysis
        drawdown_analysis = {
            'max_drawdown': max_drawdown,
            'avg_drawdown': np.mean(drawdown),
            'drawdown_duration': np.sum(drawdown < 0),
            'recovery_time': self._calculate_recovery_time(portfolio_values, peak)
        }

        return abs(max_drawdown), drawdown_analysis

    def _calculate_recovery_time(self, portfolio_values: np.ndarray, peak: np.ndarray) -> int:
        """Calculate recovery time from maximum drawdown."""
        # Find the peak before the maximum drawdown
        max_dd_idx = np.argmin((portfolio_values - peak) / peak)
        peak_before_dd = peak[max_dd_idx]

        # Find when portfolio recovers to this peak
        recovery_indices = np.where(portfolio_values[max_dd_idx:] >= peak_before_dd)[0]

        if len(recovery_indices) > 0:
            return recovery_indices[0]
        else:
            return len(portfolio_values) - max_dd_idx  # Still in drawdown

    def _convert_to_array(self, data: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """Convert input data to numpy array."""
        if isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, pd.Series):
            return data.values
        elif isinstance(data, np.ndarray):
            return data
        else:
            return np.array(data)

    def _validate_inputs(self, signals: np.ndarray, prices: np.ndarray):
        """Validate input data."""
        if len(signals) != len(prices):
            raise ValueError(f"Signals and prices must have same length: {len(signals)} vs {len(prices)}")

        if len(signals) == 0:
            raise ValueError("Empty signals array")

        if len(prices) == 0:
            raise ValueError("Empty prices array")

    def _process_backtest_chunk(self, chunk_data: Tuple) -> Dict[str, Any]:
        """Process a single chunk of backtest data."""
        signals, prices, timestamps, start_idx = chunk_data

        # Run vectorized backtest on chunk
        chunk_results = self._vectorized_backtest(signals, prices, timestamps)

        return {
            'start_idx': start_idx,
            'portfolio_values': chunk_results.portfolio_values,
            'returns': chunk_results.returns,
            'positions': chunk_results.positions,
            'trades': chunk_results.trades
        }

    def _combine_chunk_results(self, chunk_results: List[Dict[str, Any]]) -> VectorizedBacktestResults:
        """Combine results from multiple chunks."""
        # Sort chunks by start index
        chunk_results.sort(key=lambda x: x['start_idx'])

        # Concatenate arrays
        portfolio_values = np.concatenate([chunk['portfolio_values'] for chunk in chunk_results])
        returns = np.concatenate([chunk['returns'] for chunk in chunk_results])
        positions = np.concatenate([chunk['positions'] for chunk in chunk_results])

        # Combine trades
        all_trades = pd.concat([chunk['trades'] for chunk in chunk_results], ignore_index=True)

        return VectorizedBacktestResults(
            portfolio_values=portfolio_values,
            returns=returns,
            positions=positions,
            trades=all_trades,
            performance_metrics={},
            risk_metrics={},
            drawdown_analysis={},
            computation_time=0.0,
            memory_usage=0.0
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.performance_stats.copy()


def run_vectorized_backtest(signals: Union[np.ndarray, pd.DataFrame],
                           prices: Union[np.ndarray, pd.DataFrame],
                           config: Optional[VectorizedBacktestConfig] = None,
                           mode: BacktestMode = BacktestMode.VECTORIZED) -> VectorizedBacktestResults:
    """
    Convenience function to run vectorized backtest.

    Args:
        signals: Trading signals
        prices: Asset prices
        config: Backtesting configuration
        mode: Execution mode

    Returns:
        Backtest results
    """
    engine = VectorizedBacktestingEngine(config)
    return engine.run_vectorized_backtest(signals, prices, mode=mode)


# Example usage and benchmarking
def benchmark_backtesting():
    """Benchmark traditional vs vectorized backtesting."""

    # Generate sample data
    np.random.seed(42)
    n_periods = 10000
    n_assets = 5

    # Generate random signals and prices
    signals = np.random.choice([-1, 0, 1], size=(n_periods, n_assets))
    prices = np.random.randn(n_periods, n_assets).cumsum(axis=0) + 100

    logger.info("🔬 Benchmarking backtesting methods...")
    logger.info(f"📊 Dataset: {n_periods} periods, {n_assets} assets")

    # Traditional backtesting (simplified loop-based)
    logger.info("⏱️ Running traditional backtesting...")
    start_time = time.time()

    # Simulate traditional loop-based backtesting
    portfolio_value = 100000.0
    portfolio_values_traditional = [portfolio_value]

    for i in range(1, n_periods):
        # Simple position-based returns (simplified)
        position_returns = signals[i] * (prices[i] - prices[i-1]) / prices[i-1]
        total_return = np.sum(position_returns) * 0.1  # 10% allocation per asset
        portfolio_value *= (1 + total_return)
        portfolio_values_traditional.append(portfolio_value)

    traditional_time = time.time() - start_time

    # Vectorized backtesting
    logger.info("⏱️ Running vectorized backtesting...")
    start_time = time.time()

    config = VectorizedBacktestConfig()
    vectorized_results = run_vectorized_backtest(signals, prices, config)

    vectorized_time = time.time() - start_time

    # Compare results
    speedup = traditional_time / vectorized_time if vectorized_time > 0 else float('inf')

    logger.info("\n📊 BENCHMARK RESULTS:")
    logger.info(f"Traditional backtesting time: {traditional_time:.3f}s")
    logger.info(f"Vectorized backtesting time: {vectorized_time:.3f}s")
    logger.info(f"Speedup factor: {speedup:.2f}x")
    logger.info(f"Traditional final value: ${portfolio_values_traditional[-1]:.4f}")
    logger.info(f"Vectorized final value: ${vectorized_results.portfolio_values[-1]:.4f}")
    return {
        'traditional_time': traditional_time,
        'vectorized_time': vectorized_time,
        'speedup': speedup,
        'traditional_final_value': portfolio_values_traditional[-1],
        'vectorized_final_value': vectorized_results.portfolio_values[-1]
    }


if __name__ == "__main__":
    # Run benchmark when executed directly
    benchmark_backtesting()
