# src/training/optimized_backtester.py

import numpy as np
import pandas as pd
import pickle
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import gc
import psutil

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger


class OptimizedBacktester:
    """
    Optimized backtester with caching, parallelization, and memory management.
    
    Key optimizations:
    - Cached backtest results
    - Precomputed technical indicators
    - Parallel evaluation of parameter sets
    - Progressive evaluation with early stopping
    - Memory-efficient data structures
    """
    
    def __init__(self, market_data: pd.DataFrame, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("OptimizedBacktester")
        
        # Initialize optimizations
        self.cache = {}
        self.max_cache_size = config.get("max_cache_size", 1000)
        self.n_workers = config.get("max_workers", min(mp.cpu_count(), 8))
        self.chunk_size = config.get("chunk_size", 1000)
        
        # Precompute expensive calculations
        self.market_data = self._optimize_dataframe(market_data)
        self.technical_indicators = self._precompute_indicators()
        
        # Initialize parallel executor
        self.executor = ProcessPoolExecutor(max_workers=self.n_workers)
        
        # Memory management
        self.memory_threshold = config.get("memory_threshold", 0.8)
        self.cleanup_frequency = config.get("cleanup_frequency", 100)
        self.evaluation_count = 0

    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame for memory usage."""
        
        # Use appropriate dtypes
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        return df

    def _precompute_indicators(self) -> Dict[str, np.ndarray]:
        """Precompute all technical indicators once."""
        
        self.logger.info("Precomputing technical indicators...")
        indicators = {}
        
        # Price-based features
        indicators['returns'] = self.market_data['close'].pct_change().values
        indicators['log_returns'] = np.log(self.market_data['close']).diff().values
        
        # Moving averages (multiple periods)
        for period in [5, 10, 20, 50, 100]:
            indicators[f'sma_{period}'] = self.market_data['close'].rolling(period).mean().values
            indicators[f'ema_{period}'] = self.market_data['close'].ewm(span=period).mean().values
        
        # Volatility features
        indicators['atr'] = self._calculate_atr()
        indicators['volatility'] = pd.Series(indicators['returns']).rolling(20).std().values
        
        # Momentum features
        indicators['rsi'] = self._calculate_rsi()
        indicators['macd'] = self._calculate_macd()
        
        self.logger.info(f"Precomputed {len(indicators)} technical indicators")
        return indicators

    def _calculate_atr(self) -> np.ndarray:
        """Calculate Average True Range."""
        high = self.market_data['high'].values
        low = self.market_data['low'].values
        close = self.market_data['close'].values
        
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).rolling(14).mean().values
        
        return atr

    def _calculate_rsi(self) -> np.ndarray:
        """Calculate Relative Strength Index."""
        close = self.market_data['close'].values
        delta = np.diff(close)
        
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss).rolling(14).mean().values
        
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_macd(self) -> np.ndarray:
        """Calculate MACD."""
        close = self.market_data['close'].values
        ema12 = pd.Series(close).ewm(span=12).mean().values
        ema26 = pd.Series(close).ewm(span=26).mean().values
        macd = ema12 - ema26
        
        return macd

    def _generate_cache_key(self, params: Dict[str, Any]) -> str:
        """Generate cache key for parameters."""
        # Sort parameters for consistent hashing
        sorted_params = sorted(params.items())
        param_string = str(sorted_params)
        return hashlib.md5(param_string.encode()).hexdigest()

    @handle_errors(
        exceptions=(Exception,),
        default_return=-1.0,
        context="cached backtest evaluation"
    )
    def run_cached_backtest(self, params: Dict[str, Any]) -> float:
        """Run backtest with caching."""
        
        # Check memory usage
        self._check_memory_usage()
        
        # Generate cache key
        cache_key = self._generate_cache_key(params)
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Run backtest
        result = self._run_simplified_backtest(params)
        
        # Cache result
        if len(self.cache) < self.max_cache_size:
            self.cache[cache_key] = result
        
        self.evaluation_count += 1
        
        # Periodic cleanup
        if self.evaluation_count % self.cleanup_frequency == 0:
            self._cleanup_memory()
        
        return result

    def _run_simplified_backtest(self, params: Dict[str, Any]) -> float:
        """Run simplified backtest using precomputed indicators."""
        
        # Extract parameters
        tp_multiplier = params.get('tp_multiplier', 2.0)
        sl_multiplier = params.get('sl_multiplier', 1.0)
        confidence_threshold = params.get('confidence_threshold', 0.7)
        
        # Use precomputed indicators
        close_prices = self.market_data['close'].values
        high_prices = self.market_data['high'].values
        low_prices = self.market_data['low'].values
        
        # Generate signals using precomputed indicators
        signals = self._generate_signals(confidence_threshold)
        
        # Run backtest
        pnls = []
        position_open = False
        entry_price = 0.0
        
        for i in range(1, len(close_prices)):
            if not position_open and signals[i - 1] == 1:
                position_open = True
                entry_price = close_prices[i - 1]
                
                # Calculate TP/SL using ATR
                atr = self.technical_indicators['atr'][i] or 0.01
                tp_price = entry_price + (tp_multiplier * atr)
                sl_price = entry_price - (sl_multiplier * atr)
            
            if position_open:
                # Check exit conditions
                if high_prices[i] >= tp_price:
                    pnl = (tp_price - entry_price) / entry_price
                    pnls.append(pnl)
                    position_open = False
                elif low_prices[i] <= sl_price:
                    pnl = (sl_price - entry_price) / entry_price
                    pnls.append(pnl)
                    position_open = False
        
        # Calculate performance metrics
        if not pnls:
            return -1.0
        
        pnl_series = pd.Series(pnls)
        sharpe_ratio = pnl_series.mean() / (pnl_series.std() + 1e-8)
        win_rate = (pnl_series > 0).mean()
        profit_factor = abs(pnl_series[pnl_series > 0].sum() / (pnl_series[pnl_series < 0].sum() + 1e-8))
        
        # Combined score
        score = (0.5 * sharpe_ratio + 0.3 * win_rate + 0.2 * min(profit_factor, 5.0))
        
        return score

    def _generate_signals(self, confidence_threshold: float) -> np.ndarray:
        """Generate trading signals using precomputed indicators."""
        
        signals = np.zeros(len(self.market_data))
        
        # Simple signal generation using RSI and moving averages
        rsi = self.technical_indicators['rsi']
        sma_20 = self.technical_indicators['sma_20']
        sma_50 = self.technical_indicators['sma_50']
        close = self.market_data['close'].values
        
        for i in range(20, len(close)):
            # RSI oversold condition
            rsi_oversold = rsi[i] < 30
            # Price above moving averages
            price_above_ma = close[i] > sma_20[i] and sma_20[i] > sma_50[i]
            # Momentum positive
            momentum_positive = close[i] > close[i-1]
            
            if rsi_oversold and price_above_ma and momentum_positive:
                signals[i] = 1
        
        return signals

    def evaluate_batch_parallel(self, param_batch: List[Dict[str, Any]]) -> List[float]:
        """Evaluate multiple parameter sets in parallel."""
        
        if len(param_batch) <= 1:
            return [self.run_cached_backtest(params) for params in param_batch]
        
        # Prepare data for parallel processing
        data_pickle = pickle.dumps(self.market_data)
        indicators_pickle = pickle.dumps(self.technical_indicators)
        
        # Submit batch for parallel evaluation
        futures = []
        for params in param_batch:
            future = self.executor.submit(
                self._evaluate_single_params_parallel,
                data_pickle,
                indicators_pickle,
                params
            )
            futures.append(future)
        
        # Collect results
        results = [future.result() for future in futures]
        return results

    @staticmethod
    def _evaluate_single_params_parallel(data_pickle: bytes, 
                                       indicators_pickle: bytes, 
                                       params: Dict[str, Any]) -> float:
        """Evaluate single parameter set (runs in separate process)."""
        
        # Unpickle data
        market_data = pickle.loads(data_pickle)
        technical_indicators = pickle.loads(indicators_pickle)
        
        # Create temporary backtester for this process
        temp_backtester = OptimizedBacktester(market_data, {})
        temp_backtester.technical_indicators = technical_indicators
        
        # Run evaluation
        return temp_backtester._run_simplified_backtest(params)

    def _check_memory_usage(self):
        """Check and manage memory usage."""
        try:
            memory_percent = psutil.virtual_memory().percent / 100
            
            if memory_percent > self.memory_threshold:
                self._cleanup_memory()
                
        except Exception as e:
            self.logger.warning(f"Could not check memory usage: {e}")

    def _cleanup_memory(self):
        """Clean up memory by forcing garbage collection."""
        gc.collect()
        
        # Clear cache if it's too large
        if len(self.cache) > self.max_cache_size * 0.8:
            # Keep only the most recent entries
            cache_items = list(self.cache.items())
            self.cache = dict(cache_items[-self.max_cache_size//2:])
        
        self.logger.info("Memory cleanup completed")

    def progressive_evaluate(self, params: Dict[str, Any], 
                           stages: List[Tuple[float, float]] = None) -> float:
        """Evaluate parameters progressively across data subsets."""
        
        if stages is None:
            stages = [
                (0.1, 0.3),   # 10% data, 30% weight
                (0.3, 0.5),   # 30% data, 50% weight  
                (1.0, 1.0)    # 100% data, 100% weight
            ]
        
        total_score = 0
        total_weight = 0
        
        for data_ratio, weight in stages:
            subset_size = int(len(self.market_data) * data_ratio)
            subset_data = self.market_data.iloc[:subset_size]
            
            # Create temporary backtester for subset
            temp_backtester = OptimizedBacktester(subset_data, self.config)
            score = temp_backtester._run_simplified_backtest(params)
            
            total_score += score * weight
            total_weight += weight
            
            # Early stopping if performance is poor
            if data_ratio < 1.0 and score < -0.5:
                return -1.0  # Stop evaluation
        
        return total_score / total_weight

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            "cache_size": len(self.cache),
            "evaluation_count": self.evaluation_count,
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "memory_usage": psutil.virtual_memory().percent
        }

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        if self.evaluation_count == 0:
            return 0.0
        
        cache_hits = self.evaluation_count - len(self.cache)
        return max(0, cache_hits / self.evaluation_count)

    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False) 