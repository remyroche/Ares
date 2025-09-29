"""
Refactored TAS Engine using Shared Utilities

This module demonstrates how the TAS engine can be refactored to use
shared utilities, eliminating redundancy while maintaining specialized functionality.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import numpy as np
import pandas as pd

# Import shared utilities
from .shared_engine_utilities import (
    create_shared_utilities, EngineType, DataLoadingConfig, SearchMethod
)

# Import existing utilities that are still needed
from ...tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_structured,
    tprint_with_level, tprint_timer, tprint_logged, configure_tprint,
    get_tprint_config, tprint_context, LogLevel
)

from ...serialization_utils import UniversalSerializer

# Import TAS-specific data processing utilities
from ...data.processing.data_processing import DataProcessor
from ...data.basic_returns_engineer import BasicReturnsEngineer
from ...data.feature_engineer import FeatureEngineer
from ...data.gap_detector import GapDetector
from ...data.unified_data_utils import UnifiedDataUtils

# Setup logging
logger = logging.getLogger(__name__)


@tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
class RefactoredTASEngine:
    """
    Refactored TAS Engine using shared utilities.
    
    This engine demonstrates how shared utilities eliminate redundancy
    while maintaining TAS-specific functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the refactored TAS Engine with shared utilities.
        
        Args:
            config: Configuration dictionary for TAS engine
        """
        tprint_info("🚀 Initializing Refactored TAS Engine with shared utilities")
        
        # Initialize configuration
        self.config = config or {}
        self.logger = logger.getChild("RefactoredTASEngine")
        
        # Initialize shared utilities
        tprint_debug("🔧 Initializing shared utilities")
        self.shared_utilities = create_shared_utilities(EngineType.TAS)
        
        # Initialize TAS-specific data processing utilities
        tprint_debug("🔧 Initializing TAS-specific data processing utilities")
        self.data_processor = DataProcessor()
        self.returns_engineer = BasicReturnsEngineer()
        self.feature_engineer = FeatureEngineer()
        self.gap_detector = GapDetector()
        self.unified_data_utils = UnifiedDataUtils()
        
        # Initialize TAS-specific components
        tprint_debug("🔧 Initializing TAS-specific components")
        self.serializer = UniversalSerializer()
        
        # Initialize performance tracking
        self.performance_metrics = {}
        self.strategy_history = []
        self.trading_metrics = {}
        
        tprint_success("✅ Refactored TAS Engine initialized successfully")
    
    @tprint_timer("Data Loading and Processing")
    def load_and_process_data(
        self, 
        symbol: str = "ETHUSDT",
        interval: str = "1m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        apply_feature_engineering: bool = True
    ) -> Optional[pd.DataFrame]:
        """Load and process data using shared data loader.
        
        Args:
            symbol: Trading symbol to load
            interval: Data interval
            start_date: Start date for data loading
            end_date: End date for data loading
            apply_feature_engineering: Whether to apply feature engineering
            
        Returns:
            Processed DataFrame or None if loading fails
        """
        tprint_info(f"📊 Loading and processing data for {symbol} {interval}")
        
        # Create data loading configuration
        config = DataLoadingConfig(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_type="processed",
            apply_feature_engineering=apply_feature_engineering,
            validate_data=True,
            optimize_dtypes=True,
            guard_nulls=True
        )
        
        # Use shared data loader
        data = self.shared_utilities['data_loader'].load_data(config)
        
        if data is not None:
            # Apply TAS-specific processing
            processed_data = self._process_trading_data(data, apply_feature_engineering)
            
            if processed_data is not None:
                tprint_success(f"✅ Data loaded and processed: {len(processed_data)} records")
                return processed_data
            else:
                tprint_error("❌ Data processing failed")
                return None
        else:
            tprint_error("❌ Failed to load data")
            return None
    
    def _process_trading_data(
        self, 
        data: pd.DataFrame, 
        apply_feature_engineering: bool = True
    ) -> Optional[pd.DataFrame]:
        """Process trading data using TAS-specific utilities."""
        try:
            tprint_debug("🔧 Processing trading data with feature engineering")
            
            # Make a copy to avoid modifying original data
            from ...common_operations import safe_copy
            processed_data = safe_copy(data)
            
            # Apply basic returns engineering
            from ...common_operations import memory_checkpoint
            with memory_checkpoint("returns_engineering"):
                processed_data = self.returns_engineer.add_basic_returns(processed_data)
            
            # Detect gaps in data
            with memory_checkpoint("gap_detection"):
                gaps = self.gap_detector.detect_gaps(processed_data)
                if gaps:
                    tprint_info(f"🔍 Detected {len(gaps)} gaps in data")
            
            # Apply feature engineering if requested
            if apply_feature_engineering:
                with memory_checkpoint("feature_engineering"):
                    processed_data = self.feature_engineer.add_technical_indicators(processed_data)
                    processed_data = self.feature_engineer.add_price_features(processed_data)
                    processed_data = self.feature_engineer.add_volume_features(processed_data)
                    processed_data = self.feature_engineer.add_time_features(processed_data)
            
            # Apply unified data processing
            with memory_checkpoint("unified_processing"):
                processed_data = self.unified_data_utils.standardize_data(processed_data)
                processed_data = self.unified_data_utils.add_derived_features(processed_data)
            
            # Validate processed data
            from ...common_operations import validate_dataframe_columns
            if not validate_dataframe_columns(processed_data, ['open', 'high', 'low', 'close', 'volume']):
                tprint_error("❌ Processed data missing required columns")
                return None
            
            tprint_debug(f"🔧 Processed data shape: {processed_data.shape}")
            tprint_debug(f"🔧 Processed data columns: {list(processed_data.columns)}")
            
            return processed_data
            
        except Exception as e:
            tprint_error(f"❌ Error processing trading data: {e}")
            return None
    
    @tprint_timer("Strategy Search")
    def search_strategies(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        optimization_method: str = "bayesian_tpe",
        n_trials: int = 100,
        include_regime_specific: bool = True
    ) -> Dict[str, Any]:
        """Search for optimal trading strategies using shared search framework.
        
        Args:
            data: Input data for strategy search
            search_space: Strategy search space
            optimization_method: Optimization method (bayesian_tpe, grid, hierarchical)
            n_trials: Number of optimization trials
            include_regime_specific: Whether to include regime-specific optimization
            
        Returns:
            Dictionary with search results and best strategy
        """
        tprint_info(f"🔍 Starting strategy search with {optimization_method}")
        
        # Prepare additional parameters for TAS-specific evaluation
        additional_params = {
            'include_regime_specific': include_regime_specific,
            'regime_analysis': self._analyze_regimes(data) if include_regime_specific else None
        }
        
        # Use shared search framework
        search_result = self.shared_utilities['search_framework'].execute_search(
            data=data,
            search_space=search_space,
            optimization_method=optimization_method,
            n_trials=n_trials,
            evaluation_function=self._evaluate_strategy,
            additional_params=additional_params
        )
        
        # Convert to expected format
        result = {
            'method': search_result.method,
            'n_trials': search_result.n_trials,
            'trials': search_result.trials,
            'best_strategy': search_result.best_solution,
            'best_score': search_result.best_score,
            'search_time': search_result.search_time,
            'performance_metrics': search_result.performance_metrics,
            'regime_analysis': additional_params.get('regime_analysis')
        }
        
        tprint_success(f"✅ Strategy search completed in {search_result.search_time:.2f}s")
        tprint_info(f"🏆 Best score: {search_result.best_score:.4f}")
        
        return result
    
    def _analyze_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market regimes using TAS-specific logic."""
        try:
            tprint_debug("🔍 Analyzing market regimes")
            
            # Extract price data for regime analysis
            price_data = data[['open', 'high', 'low', 'close', 'volume']].values
            
            # Use matrix operations for regime detection
            from ...common_operations import memory_checkpoint
            with memory_checkpoint("regime_analysis"):
                # Calculate rolling statistics using matrix operations
                matrix_ops = self.shared_utilities['feature_matrix_builder'].matrix_ops
                rolling_returns = matrix_ops.calculate_rolling_returns(price_data)
                volatility = matrix_ops.calculate_rolling_volatility(rolling_returns)
                trend_strength = matrix_ops.calculate_trend_strength(price_data)
                
                # Combine features for regime classification
                regime_features = np.column_stack([volatility, trend_strength])
                
                # Use vectorized operations for regime classification
                vectorized_core = self.shared_utilities['feature_matrix_builder'].vectorized_core
                regimes = vectorized_core.classify_regimes(regime_features)
            
            # Calculate regime statistics
            regime_stats = {}
            unique_regimes = np.unique(regimes)
            
            for regime in unique_regimes:
                regime_mask = regimes == regime
                regime_data = data[regime_mask]
                
                if not regime_data.empty:
                    from ...math_validation import safe_divide, safe_mean
                    regime_stats[f'regime_{regime}'] = {
                        'count': len(regime_data),
                        'percentage': safe_divide(len(regime_data), len(data)) * 100,
                        'avg_volatility': safe_mean(volatility[regime_mask]),
                        'avg_trend': safe_mean(trend_strength[regime_mask]),
                        'avg_return': safe_mean(rolling_returns[regime_mask])
                    }
            
            tprint_info(f"🔍 Detected {len(unique_regimes)} market regimes")
            return {
                'regimes': regimes,
                'regime_stats': regime_stats,
                'features': {
                    'volatility': volatility,
                    'trend_strength': trend_strength,
                    'returns': rolling_returns
                }
            }
            
        except Exception as e:
            tprint_error(f"❌ Error in regime analysis: {e}")
            return {}
    
    def _evaluate_strategy(
        self, 
        data: pd.DataFrame, 
        strategy_params: Dict[str, Any],
        additional_params: Dict[str, Any] = None
    ) -> float:
        """Evaluate strategy performance using shared evaluation framework.
        
        Args:
            data: Input data for evaluation
            strategy_params: Strategy parameters to evaluate
            additional_params: Additional parameters for evaluation
            
        Returns:
            Strategy performance score
        """
        # Use shared evaluation framework
        score = self.shared_utilities['evaluation_framework'].evaluate_solution(
            data=data,
            solution_params=strategy_params,
            additional_params=additional_params
        )
        
        return score
    
    @tprint_timer("Results Serialization")
    def save_results(
        self, 
        results: Dict[str, Any], 
        filepath: str
    ) -> bool:
        """Save strategy search results using serialization utilities.
        
        Args:
            results: Strategy search results to save
            filepath: Path to save results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tprint_info(f"💾 Saving strategy results to {filepath}")
            
            # Add metadata
            results_with_metadata = {
                'results': results,
                'metadata': {
                    'timestamp': time.time(),
                    'tas_engine_version': '2.0.0',
                    'engine_type': 'refactored_tas',
                    'shared_utilities_used': True,
                    'trading_metrics': self.trading_metrics
                }
            }
            
            # Save using universal serializer
            success = self.serializer.save(results_with_metadata, filepath)
            
            if success:
                tprint_success(f"✅ Strategy results saved successfully to {filepath}")
            else:
                tprint_error(f"❌ Failed to save strategy results to {filepath}")
            
            return success
            
        except Exception as e:
            tprint_error(f"❌ Error saving strategy results: {e}")
            return False
    
    def load_results(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load strategy search results using serialization utilities.
        
        Args:
            filepath: Path to load results from
            
        Returns:
            Loaded results or None if loading fails
        """
        try:
            tprint_info(f"📂 Loading strategy results from {filepath}")
            
            # Load using universal serializer
            results = self.serializer.load(filepath)
            
            if results:
                tprint_success(f"✅ Strategy results loaded successfully from {filepath}")
                return results
            else:
                tprint_error(f"❌ Failed to load strategy results from {filepath}")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Error loading strategy results: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            tprint_info("🧹 Cleaning up Refactored TAS Engine resources")
            
            # Clear strategy history
            self.strategy_history.clear()
            self.trading_metrics.clear()
            
            tprint_success("✅ Refactored TAS Engine cleanup completed")
            
        except Exception as e:
            tprint_error(f"❌ Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


# Convenience function for quick TAS usage
def create_refactored_tas_engine(config: Optional[Dict[str, Any]] = None) -> RefactoredTASEngine:
    """Create a refactored TAS engine instance with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured RefactoredTASEngine instance
    """
    return RefactoredTASEngine(config)


# Example usage
if __name__ == "__main__":
    # Configure tprint for better output
    from ...tprint import TPrintConfig, configure_tprint
    
    config = TPrintConfig(
        use_colors=True,
        output_to_console=True,
        enable_structured_logging=True
    )
    configure_tprint(config)
    
    # Create and use refactored TAS engine
    with create_refactored_tas_engine() as tas_engine:
        # Load and process data
        data = tas_engine.load_and_process_data("ETHUSDT", "1m", apply_feature_engineering=True)
        
        if data is not None:
            # Define search space
            search_space = {
                'entry_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'exit_threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'risk_factor': [0.5, 1.0, 1.5, 2.0],
                'position_size': [0.05, 0.1, 0.15, 0.2, 0.25],
                'stop_loss': [0.01, 0.02, 0.03, 0.04, 0.05],
                'take_profit': [0.02, 0.03, 0.04, 0.05, 0.06]
            }
            
            # Perform strategy search
            results = tas_engine.search_strategies(
                data=data,
                search_space=search_space,
                optimization_method="bayesian_tpe",
                n_trials=50,
                include_regime_specific=True
            )
            
            # Save results
            if results:
                tas_engine.save_results(results, "refactored_tas_results.json")
                tprint_structured(results, LogLevel.INFO)