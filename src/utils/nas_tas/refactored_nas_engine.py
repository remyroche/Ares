"""
Refactored NAS Engine using Shared Utilities

This module demonstrates how the NAS engine can be refactored to use
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

# Setup logging
logger = logging.getLogger(__name__)


@tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
class RefactoredNASEngine:
    """
    Refactored NAS Engine using shared utilities.
    
    This engine demonstrates how shared utilities eliminate redundancy
    while maintaining NAS-specific functionality.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the refactored NAS Engine with shared utilities.
        
        Args:
            config: Configuration dictionary for NAS engine
        """
        tprint_info("🚀 Initializing Refactored NAS Engine with shared utilities")
        
        # Initialize configuration
        self.config = config or {}
        self.logger = logger.getChild("RefactoredNASEngine")
        
        # Initialize shared utilities
        tprint_debug("🔧 Initializing shared utilities")
        self.shared_utilities = create_shared_utilities(EngineType.NAS)
        
        # Initialize NAS-specific components
        tprint_debug("🔧 Initializing NAS-specific components")
        self.serializer = UniversalSerializer()
        
        # Initialize performance tracking
        self.performance_metrics = {}
        self.search_history = []
        
        tprint_success("✅ Refactored NAS Engine initialized successfully")
    
    @tprint_timer("Data Loading and Validation")
    def load_and_validate_data(
        self, 
        symbol: str = "ETHUSDT",
        interval: str = "1m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Load and validate data using shared data loader.
        
        Args:
            symbol: Trading symbol to load
            interval: Data interval
            start_date: Start date for data loading
            end_date: End date for data loading
            
        Returns:
            Validated DataFrame or None if loading fails
        """
        tprint_info(f"📊 Loading data for {symbol} {interval}")
        
        # Create data loading configuration
        config = DataLoadingConfig(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            data_type="processed",
            apply_feature_engineering=False,  # NAS doesn't need feature engineering
            validate_data=True,
            optimize_dtypes=True,
            guard_nulls=True
        )
        
        # Use shared data loader
        data = self.shared_utilities['data_loader'].load_data(config)
        
        if data is not None:
            tprint_success(f"✅ Data loaded and validated: {len(data)} records")
        else:
            tprint_error("❌ Failed to load data")
        
        return data
    
    @tprint_timer("Architecture Search")
    def search_architectures(
        self,
        data: pd.DataFrame,
        search_space: Dict[str, Any],
        optimization_method: str = "bayesian_tpe",
        n_trials: int = 100
    ) -> Dict[str, Any]:
        """Search for optimal architectures using shared search framework.
        
        Args:
            data: Input data for architecture search
            search_space: Architecture search space
            optimization_method: Optimization method (bayesian_tpe, grid, hierarchical)
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary with search results and best architecture
        """
        tprint_info(f"🔍 Starting architecture search with {optimization_method}")
        
        # Use shared search framework
        search_result = self.shared_utilities['search_framework'].execute_search(
            data=data,
            search_space=search_space,
            optimization_method=optimization_method,
            n_trials=n_trials,
            evaluation_function=self._evaluate_architecture,
            additional_params={}
        )
        
        # Convert to expected format
        result = {
            'method': search_result.method,
            'n_trials': search_result.n_trials,
            'trials': search_result.trials,
            'best_architecture': search_result.best_solution,
            'best_score': search_result.best_score,
            'search_time': search_result.search_time,
            'performance_metrics': search_result.performance_metrics
        }
        
        tprint_success(f"✅ Architecture search completed in {search_result.search_time:.2f}s")
        tprint_info(f"🏆 Best score: {search_result.best_score:.4f}")
        
        return result
    
    def _evaluate_architecture(
        self, 
        data: pd.DataFrame, 
        architecture_params: Dict[str, Any],
        additional_params: Dict[str, Any] = None
    ) -> float:
        """Evaluate architecture performance using shared evaluation framework.
        
        Args:
            data: Input data for evaluation
            architecture_params: Architecture parameters to evaluate
            additional_params: Additional parameters for evaluation
            
        Returns:
            Architecture performance score
        """
        # Use shared evaluation framework
        score = self.shared_utilities['evaluation_framework'].evaluate_solution(
            data=data,
            solution_params=architecture_params,
            additional_params=additional_params
        )
        
        return score
    
    @tprint_timer("Results Serialization")
    def save_results(
        self, 
        results: Dict[str, Any], 
        filepath: str
    ) -> bool:
        """Save search results using serialization utilities.
        
        Args:
            results: Search results to save
            filepath: Path to save results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tprint_info(f"💾 Saving results to {filepath}")
            
            # Add metadata
            results_with_metadata = {
                'results': results,
                'metadata': {
                    'timestamp': time.time(),
                    'nas_engine_version': '2.0.0',
                    'engine_type': 'refactored_nas',
                    'shared_utilities_used': True
                }
            }
            
            # Save using universal serializer
            success = self.serializer.save(results_with_metadata, filepath)
            
            if success:
                tprint_success(f"✅ Results saved successfully to {filepath}")
            else:
                tprint_error(f"❌ Failed to save results to {filepath}")
            
            return success
            
        except Exception as e:
            tprint_error(f"❌ Error saving results: {e}")
            return False
    
    def load_results(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load search results using serialization utilities.
        
        Args:
            filepath: Path to load results from
            
        Returns:
            Loaded results or None if loading fails
        """
        try:
            tprint_info(f"📂 Loading results from {filepath}")
            
            # Load using universal serializer
            results = self.serializer.load(filepath)
            
            if results:
                tprint_success(f"✅ Results loaded successfully from {filepath}")
                return results
            else:
                tprint_error(f"❌ Failed to load results from {filepath}")
                return None
                
        except Exception as e:
            tprint_error(f"❌ Error loading results: {e}")
            return None
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            tprint_info("🧹 Cleaning up Refactored NAS Engine resources")
            
            # Clear search history
            self.search_history.clear()
            
            tprint_success("✅ Refactored NAS Engine cleanup completed")
            
        except Exception as e:
            tprint_error(f"❌ Error during cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()


# Convenience function for quick NAS usage
def create_refactored_nas_engine(config: Optional[Dict[str, Any]] = None) -> RefactoredNASEngine:
    """Create a refactored NAS engine instance with default configuration.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured RefactoredNASEngine instance
    """
    return RefactoredNASEngine(config)


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
    
    # Create and use refactored NAS engine
    with create_refactored_nas_engine() as nas_engine:
        # Load data
        data = nas_engine.load_and_validate_data("ETHUSDT", "1m")
        
        if data is not None:
            # Define search space
            search_space = {
                'complexity': [1.0, 1.5, 2.0, 2.5, 3.0],
                'depth': [1, 2, 3, 4, 5],
                'width': [8, 16, 32, 64, 128],
                'activation': ['relu', 'tanh', 'sigmoid']
            }
            
            # Perform architecture search
            results = nas_engine.search_architectures(
                data=data,
                search_space=search_space,
                optimization_method="bayesian_tpe",
                n_trials=50
            )
            
            # Save results
            if results:
                nas_engine.save_results(results, "refactored_nas_results.json")
                tprint_structured(results, LogLevel.INFO)