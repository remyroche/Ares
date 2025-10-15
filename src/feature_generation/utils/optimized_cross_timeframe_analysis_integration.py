"""
import warnings
Optimized Cross Timeframe Analysis Integration

This module integrates all the optimized components and provides the main interface
for optimized cross timeframe analysis.
"""

import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union

from src.utils.logger import system_logger
from .optimized_cross_timeframe_analysis import (
    OptimizedCrossTimeframeAnalysis,
    OptimizedCrossTimeframeConfig,
    OptimizedCrossTimeframeResult
)
from .optimized_cross_timeframe_analysis_methods import OptimizedCrossTimeframeMethods
from .optimized_cross_timeframe_analysis_advanced import OptimizedCrossTimeframeAdvanced

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None

logger = system_logger.getChild('OptimizedCrossTimeframeIntegration')

class OptimizedCrossTimeframeAnalysisPipeline:
    """
    Complete optimized cross timeframe analysis pipeline.
    
    This class integrates all optimization components and provides a unified interface
    for performing highly optimized cross timeframe analysis.
    """
    
    def __init__(self, config: Optional[OptimizedCrossTimeframeConfig] = None):
        """Initialize the optimized cross timeframe analysis pipeline."""
        self.config = config or OptimizedCrossTimeframeConfig()
        self.logger = logger.getChild('OptimizedPipeline')
        
        # Initialize main analyzer
        self.analyzer = OptimizedCrossTimeframeAnalysis(self.config)
        
        # Initialize method classes (methods is now deprecated due to aggtrades removal)
        self.methods = OptimizedCrossTimeframeMethods(self.analyzer)  # Placeholder - aggtrades removed
        self.advanced = OptimizedCrossTimeframeAdvanced(self.analyzer)
        
        # Integrate methods into analyzer
        self._integrate_methods()
        
        self.logger.info("✅ Optimized Cross Timeframe Analysis Pipeline initialized")
    
    def _generate_multi_output_targets(
        self, 
        data: pd.DataFrame, 
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, pd.Series]:
        """
        Generate multi-output targets for stacking ensemble models.
        
        Args:
            data: Input data DataFrame
            config: Multi-output configuration
            
        Returns:
            Dictionary of target series
        """
        if config is None:
            config = {
                'analyst': {
                    'signal_strength': {'method': 'price_momentum', 'lookback': 20},
                    'confidence': {'method': 'volatility_based', 'lookback': 50},
                    'risk_score': {'method': 'drawdown_based', 'lookback': 100},
                    'regime_label': {'method': 'hmm_based', 'n_regimes': 3}
                },
                'tactician': {
                    'entry_timing': {'method': 'momentum_based', 'lookback': 10},
                    'position_size': {'method': 'kelly_criterion', 'lookback': 50},
                    'stop_loss': {'method': 'atr_based', 'lookback': 20},
                    'take_profit': {'method': 'risk_reward_ratio', 'lookback': 20}
                }
            }
        
        targets = {}
        
        # Generate analyst targets
        if 'analyst' in config:
            analyst_targets = self._create_analyst_outputs(data, config['analyst'])
            targets.update(analyst_targets)
        
        # Generate tactician targets
        if 'tactician' in config:
            tactician_targets = self._create_tactician_outputs(data, config['tactician'])
            targets.update(tactician_targets)
        
        return targets
    
    def _create_analyst_outputs(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> Dict[str, pd.Series]:
        """Create Analyst multi-output targets."""
        outputs = {}
        
        # Signal strength
        if 'signal_strength' in config:
            method = config['signal_strength']['method']
            lookback = config['signal_strength']['lookback']
            
            if method == 'price_momentum':
                returns = data['close'].pct_change()
                outputs['signal_strength'] = returns.rolling(lookback).mean()
            elif method == 'rsi_based':
                rsi = self._calculate_rsi(data['close'], lookback)
                outputs['signal_strength'] = (rsi - 50) / 50
            else:
                outputs['signal_strength'] = pd.Series(0, index=data.index)
        
        # Confidence
        if 'confidence' in config:
            method = config['confidence']['method']
            lookback = config['confidence']['lookback']
            
            if method == 'volatility_based':
                returns = data['close'].pct_change()
                volatility = returns.rolling(lookback).std()
                # Higher volatility = lower confidence
                outputs['confidence'] = 1 / (1 + volatility)
            elif method == 'volume_based':
                volume_ma = data['volume'].rolling(lookback).mean()
                current_volume = data['volume']
                outputs['confidence'] = current_volume / volume_ma
            else:
                outputs['confidence'] = pd.Series(0.5, index=data.index)
        
        # Risk score
        if 'risk_score' in config:
            method = config['risk_score']['method']
            lookback = config['risk_score']['lookback']
            
            if method == 'drawdown_based':
                returns = data['close'].pct_change()
                cumulative_returns = (1 + returns).cumprod()
                rolling_max = cumulative_returns.rolling(lookback).max()
                drawdown = (cumulative_returns - rolling_max) / rolling_max
                outputs['risk_score'] = -drawdown.rolling(lookback).min()
            elif method == 'var_based':
                returns = data['close'].pct_change()
                var = returns.rolling(lookback).quantile(0.05)
                outputs['risk_score'] = -var
            else:
                outputs['risk_score'] = pd.Series(0, index=data.index)
        
        # Regime label
        if 'regime_label' in config:
            method = config['regime_label']['method']
            n_regimes = config['regime_label'].get('n_regimes', 3)
            
            if method == 'hmm_based':
                outputs['regime_label'] = self._generate_regime_labels(data, n_regimes)
            elif method == 'volatility_based':
                returns = data['close'].pct_change()
                volatility = returns.rolling(20).std()
                # Simple 3-regime classification
                low_threshold = volatility.quantile(0.33)
                high_threshold = volatility.quantile(0.67)
                outputs['regime_label'] = pd.cut(volatility, 
                    bins=[0, low_threshold, high_threshold, float('inf')], 
                    labels=[0, 1, 2])
            else:
                outputs['regime_label'] = pd.Series(0, index=data.index)
        
        return outputs
    
    def _create_tactician_outputs(
        self, 
        data: pd.DataFrame, 
        config: Dict[str, Any]
    ) -> Dict[str, pd.Series]:
        """Create Tactician multi-output targets."""
        outputs = {}
        
        # Entry timing
        if 'entry_timing' in config:
            method = config['entry_timing']['method']
            lookback = config['entry_timing']['lookback']
            
            if method == 'momentum_based':
                returns = data['close'].pct_change()
                momentum = returns.rolling(lookback).mean()
                # Normalize to [-1, 1]
                outputs['entry_timing'] = np.tanh(momentum * 10)
            elif method == 'rsi_based':
                rsi = self._calculate_rsi(data['close'], lookback)
                # Convert RSI to entry timing signal
                outputs['entry_timing'] = (rsi - 50) / 50
            else:
                outputs['entry_timing'] = pd.Series(0, index=data.index)
        
        # Position size
        if 'position_size' in config:
            method = config['position_size']['method']
            lookback = config['position_size']['lookback']
            
            if method == 'kelly_criterion':
                returns = data['close'].pct_change()
                win_rate = (returns > 0).rolling(lookback).mean()
                avg_win = returns[returns > 0].rolling(lookback).mean()
                avg_loss = returns[returns < 0].rolling(lookback).mean()
                kelly = win_rate - (1 - win_rate) * abs(avg_loss) / abs(avg_win)
                outputs['position_size'] = np.clip(kelly, 0, 0.25)  # Cap at 25%
            elif method == 'volatility_based':
                returns = data['close'].pct_change()
                volatility = returns.rolling(lookback).std()
                # Inverse relationship: higher volatility = smaller position
                outputs['position_size'] = 1 / (1 + volatility * 10)
            else:
                outputs['position_size'] = pd.Series(0.1, index=data.index)
        
        # Stop loss
        if 'stop_loss' in config:
            method = config['stop_loss']['method']
            lookback = config['stop_loss']['lookback']
            
            if method == 'atr_based':
                atr = self._calculate_atr(data, lookback)
                atr_multiplier = config['stop_loss'].get('atr_multiplier', 2.0)
                outputs['stop_loss'] = atr * atr_multiplier
            elif method == 'percentage_based':
                percentage = config['stop_loss'].get('percentage', 0.02)
                outputs['stop_loss'] = data['close'] * percentage
            else:
                outputs['stop_loss'] = data['close'] * 0.02
        
        # Take profit
        if 'take_profit' in config:
            method = config['take_profit']['method']
            lookback = config['take_profit']['lookback']
            
            if method == 'risk_reward_ratio':
                risk_reward_ratio = config['take_profit'].get('risk_reward_ratio', 2.0)
                # Use stop_loss as base for take profit
                if 'stop_loss' in outputs:
                    outputs['take_profit'] = outputs['stop_loss'] * risk_reward_ratio
                else:
                    outputs['take_profit'] = data['close'] * 0.04
            elif method == 'atr_based':
                atr = self._calculate_atr(data, lookback)
                atr_multiplier = config['take_profit'].get('atr_multiplier', 3.0)
                outputs['take_profit'] = atr * atr_multiplier
            else:
                outputs['take_profit'] = data['close'] * 0.04
        
        return outputs
    
    def _generate_regime_labels(
        self, 
        data: pd.DataFrame, 
        n_regimes: int = 3
    ) -> pd.Series:
        """Generate regime labels using HMM or simple clustering."""
        try:
            # Simple 3-regime classification based on volatility and trend
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std()
            trend = returns.rolling(20).mean()
            
            # Combine volatility and trend for regime classification
            combined_score = volatility * 0.7 + abs(trend) * 0.3
            
            # Create regime labels
            regime_labels = pd.cut(combined_score, 
                bins=n_regimes, 
                labels=range(n_regimes))
            
            return regime_labels.fillna(0).astype(int)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to generate regime labels: {e}")
            return pd.Series(0, index=data.index)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def _integrate_methods(self):
        """Integrate method classes into the main analyzer."""
        # Data loading and validation methods
        self.analyzer._load_and_validate_data = self.methods._load_and_validate_data
        self.analyzer._load_single_timeframe_data = self.methods._load_single_timeframe_data
        self.analyzer._resample_data_optimized = self.methods._resample_data_optimized
        self.analyzer._chunked_resample = self.methods._chunked_resample
        self.analyzer._optimize_dataframe_memory = self.methods._optimize_dataframe_memory
        
        # Timeframe alignment methods
        self.analyzer._align_timeframes_optimized = self.methods._align_timeframes_optimized
        self.analyzer._align_single_timeframe = self.methods._align_single_timeframe
        
        # Feature engineering methods
        self.analyzer._engineer_features_optimized = self.methods._engineer_features_optimized
        self.analyzer._create_base_features_gpu_accelerated = self.methods._create_base_features_gpu_accelerated
        self.analyzer._create_base_features_cpu = self.methods._create_base_features_cpu
        self.analyzer._create_interaction_features_parallel = self.methods._create_interaction_features_parallel
        self.analyzer._create_correlation_features = self.methods._create_correlation_features
        self.analyzer._create_momentum_features = self.methods._create_momentum_features
        self.analyzer._create_volatility_features = self.methods._create_volatility_features
        self.analyzer._create_volume_features = self.methods._create_volume_features
        self.analyzer._create_aggregation_features_parallel = self.methods._create_aggregation_features_parallel
        self.analyzer._create_timeframe_aggregation_features = self.methods._create_timeframe_aggregation_features
        self.analyzer._create_specialized_features_parallel = self.methods._create_specialized_features_parallel
        self.analyzer._create_microstructure_features = self.methods._create_microstructure_features
        self.analyzer._create_order_flow_features = self.methods._create_order_flow_features
        self.analyzer._create_momentum_divergence_features = self.methods._create_momentum_divergence_features
        self.analyzer._create_volatility_spillover_features = self.methods._create_volatility_spillover_features
        
        # Advanced methods
        self.analyzer._perform_advanced_feature_selection = self.advanced._perform_advanced_feature_selection
        self.analyzer._calculate_interaction_metrics_optimized = self.advanced._calculate_interaction_metrics_optimized
        self.analyzer._calculate_timeframe_correlations_optimized = self.advanced._calculate_timeframe_correlations_optimized
        self.analyzer._calculate_feature_importance_optimized = self.advanced._calculate_feature_importance_optimized
        self.analyzer._calculate_financial_risk_metrics = self.advanced._calculate_financial_risk_metrics
        self.analyzer._generate_quality_report = self.advanced._generate_quality_report
        
        # Multi-output target generation methods
        self.analyzer._generate_multi_output_targets = self._generate_multi_output_targets
        self.analyzer._create_analyst_outputs = self._create_analyst_outputs
        self.analyzer._create_tactician_outputs = self._create_tactician_outputs
        self.analyzer._generate_regime_labels = self._generate_regime_labels
    
    async def analyze_cross_timeframes(
        self,
        data_dir: str,
        symbol: str,
        exchange: str,
        timeframes: Optional[List[str]] = None,
        enable_multi_output: bool = False,
        multi_output_config: Optional[Dict[str, Any]] = None
    ) -> OptimizedCrossTimeframeResult:
        """
        Perform optimized cross timeframe analysis.
        
        Args:
            data_dir: Data directory path
            symbol: Trading symbol
            exchange: Exchange name
            timeframes: List of timeframes to analyze (optional)
            
        Returns:
            OptimizedCrossTimeframeResult with comprehensive analysis results
        """
        return await self.analyzer.analyze_cross_timeframes(data_dir, symbol, exchange, timeframes)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get status of available optimizations."""
        return {
            'hardware_optimizations': {
                'm1_memory_optimizer': self.analyzer.memory_optimizer is not None,
                'm1_cpu_optimizer': self.analyzer.cpu_optimizer is not None,
                'm1_gpu_manager': self.analyzer.gpu_manager is not None
            },
            'feature_selection': {
                'advanced_feature_selector': self.analyzer.feature_selector is not None
            },
            'utilities': {
                'data_validator': self.analyzer.data_validator is not None,
                'data_cleaner': self.analyzer.data_cleaner is not None,
                'data_transformer': self.analyzer.data_transformer is not None,
                'parquet_utils': self.analyzer.parquet_utils is not None,
                'json_serializer': self.analyzer.json_serializer is not None,
                'parquet_serializer': self.analyzer.parquet_serializer is not None
            },
            'caching': {
                'intelligent_cache': self.analyzer.cache is not None
            },
            'config': {
                'enable_m1_optimizations': self.config.enable_m1_optimizations,
                'enable_gpu_acceleration': self.config.enable_gpu_acceleration,
                'enable_advanced_feature_selection': self.config.enable_advanced_feature_selection,
                'enable_caching': self.config.enable_caching,
                'memory_limit_gb': self.config.memory_limit_gb,
                'max_workers': self.config.max_workers
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics from the last analysis."""
        if hasattr(self.analyzer, '_last_result') and self.analyzer._last_result:
            return self.analyzer._last_result.performance_metrics
        return {}
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        if self.analyzer.memory_optimizer:
            return self.analyzer.memory_optimizer.get_memory_report()
        return {}
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization."""
        if self.analyzer.memory_optimizer:
            return self.analyzer.memory_optimizer.optimize_memory()
        return {}

# Convenience functions for easy integration
async def analyze_cross_timeframes_optimized(
    data_dir: str,
    symbol: str,
    exchange: str,
    timeframes: Optional[List[str]] = None,
    config: Optional[OptimizedCrossTimeframeConfig] = None
) -> OptimizedCrossTimeframeResult:
    """
    Convenience function to perform optimized cross timeframe analysis.
    
    Args:
        data_dir: Data directory path
        symbol: Trading symbol
        exchange: Exchange name
        timeframes: List of timeframes to analyze (optional)
        config: Configuration for analysis (optional)
        
    Returns:
        OptimizedCrossTimeframeResult with comprehensive analysis results
    """
    pipeline = OptimizedCrossTimeframeAnalysisPipeline(config)
    return await pipeline.analyze_cross_timeframes(data_dir, symbol, exchange, timeframes)

def create_optimized_config(
    timeframes: Optional[List[str]] = None,
    enable_m1_optimizations: bool = True,
    enable_gpu_acceleration: bool = True,
    enable_advanced_feature_selection: bool = True,
    memory_limit_gb: float = 8.0,
    max_workers: int = 4,
    **kwargs
) -> OptimizedCrossTimeframeConfig:
    """
    Create an optimized configuration for cross timeframe analysis.
    
    Args:
        timeframes: List of timeframes to analyze
        enable_m1_optimizations: Enable M1 hardware optimizations
        enable_gpu_acceleration: Enable 
        enable_advanced_feature_selection: Enable advanced feature selection
        memory_limit_gb: Memory limit in GB
        max_workers: Maximum number of workers for parallel processing
        **kwargs: Additional configuration parameters
        
    Returns:
        OptimizedCrossTimeframeConfig instance
    """
    config_dict = {
        'timeframes': timeframes or ['1m', '5m', '15m', '30m'],
        'enable_m1_optimizations': enable_m1_optimizations,
        'enable_gpu_acceleration': enable_gpu_acceleration,
        'enable_advanced_feature_selection': enable_advanced_feature_selection,
        'memory_limit_gb': memory_limit_gb,
        'max_workers': max_workers,
        **kwargs
    }
    
    return OptimizedCrossTimeframeConfig(**config_dict)

# Example usage and testing
async def example_usage():
    """Example usage of the optimized cross timeframe analysis."""
    try:
        # Create optimized configuration
        config = create_optimized_config(
            timeframes=['1m', '5m', '15m', '30m'],
            enable_m1_optimizations=True,
            enable_gpu_acceleration=True,
            enable_advanced_feature_selection=True,
            memory_limit_gb=8.0,
            max_workers=4
        )
        
        # Create pipeline
        pipeline = OptimizedCrossTimeframeAnalysisPipeline(config)
        
        # Check optimization status
        status = pipeline.get_optimization_status()
        print("Optimization Status:", status)
        
        # Perform analysis
        result = await pipeline.analyze_cross_timeframes(
            data_dir="historical_data",
            symbol="ETHUSDT",
            exchange="BINANCE",
            timeframes=['1m', '5m', '15m', '30m']
        )
        
        # Print results
        print(f"Analysis completed successfully!")
        print(f"Features generated: {len(result.cross_timeframe_features.columns)}")
        print(f"Selected features: {len(result.selected_features.get('final', []))}")
        print(f"Performance metrics: {result.performance_metrics}")
        
        return result
        
    except Exception as e:
        print(f"Example usage failed: {e}")
        return None

if __name__ == "__main__":
    # Run example
    asyncio.run(example_usage())
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
