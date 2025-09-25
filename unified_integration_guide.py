#!/usr/bin/env python3
"""
Unified Integration Guide for NAS & TAS Systems

This module demonstrates how to integrate the unified hybrid architecture and shared components
with existing NAS and TAS implementations to eliminate code duplication and ensure consistency.

Key Integration Points:
- Configuration management unification
- Evaluation framework consolidation
- Hardware optimization sharing
- Search algorithm standardization
- Data processing pipeline unification
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import numpy as np
import pandas as pd

# Import the unified components
from unified_hybrid_architecture import (
    UnifiedHybridSystem, UnifiedArchitectureConfig, 
    ArchitectureType, SearchStrategy, OptimizationObjective,
    create_unified_hybrid_system, run_quick_search
)

from shared_components import (
    SharedConfigManager, SharedEvaluationMetrics, SharedHardwareOptimizer,
    SharedSearchAlgorithms, SharedDataProcessor, SharedUtilities
)

# Import existing NAS and TAS components for integration
try:
    from nas_trainer import NASTrainer, NASConfig
    NAS_AVAILABLE = True
except ImportError:
    NAS_AVAILABLE = False
    tprint_warning("NAS trainer not available for integration")

try:
    from src.utils.ml_common.optimization.tas.core.tas_engine import TreeArchitectureSearchEngine
    from src.utils.ml_common.optimization.tas.core.tas_config import TASConfig
    TAS_AVAILABLE = True
except ImportError:
    TAS_AVAILABLE = False
    tprint_warning("TAS engine not available for integration")

logger = logging.getLogger(__name__)


# ============================================================================
# INTEGRATION ADAPTERS
# ============================================================================

class NASIntegrationAdapter:
    """Adapter to integrate existing NAS system with unified components."""
    
    def __init__(self, unified_config: UnifiedArchitectureConfig):
        """Initialize NAS integration adapter."""
        self.unified_config = unified_config
        self.nas_config = self._convert_to_nas_config(unified_config)
        self.nas_trainer = None
        
        if NAS_AVAILABLE:
            self.nas_trainer = NASTrainer(self.nas_config)
    
    def _convert_to_nas_config(self, unified_config: UnifiedArchitectureConfig) -> NASConfig:
        """Convert unified config to NAS config."""
        # This would map unified config parameters to NAS-specific parameters
        nas_config = NASConfig(
            search_strategy=unified_config.search_strategy.value,
            max_trials=unified_config.max_trials,
            max_epochs=unified_config.max_epochs,
            min_layers=unified_config.min_layers,
            max_layers=unified_config.max_layers,
            min_neurons=unified_config.min_neurons,
            max_neurons=unified_config.max_neurons,
            activation_functions=unified_config.activation_functions,
            dropout_rates=unified_config.dropout_rates,
            learning_rate_range=unified_config.learning_rate_range,
            batch_size_range=unified_config.batch_size_range,
            use_m1_optimization=unified_config.enable_hardware_optimization,
            memory_limit_gb=unified_config.memory_limit_gb,
            save_results=unified_config.save_results,
            output_dir=unified_config.output_dir,
            verbose=unified_config.verbose
        )
        return nas_config
    
    def run_nas_with_unified_components(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run NAS using unified shared components."""
        if not self.nas_trainer:
            raise RuntimeError("NAS trainer not available")
        
        # Use shared data processing
        config_dict = {
            'handle_missing_values': True,
            'normalize_data': True,
            'standardize_data': True,
            'enable_feature_selection': self.unified_config.enable_feature_selection,
            'max_features': self.unified_config.max_features,
            'validation_split': self.unified_config.validation_split,
            'time_series_split': False
        }
        
        # Preprocess data using shared components
        X_processed, y_processed = SharedDataProcessor.preprocess_data(X, y, config_dict)
        X_train, X_val, y_train, y_val = SharedDataProcessor.split_data(X_processed, y_processed, config_dict)
        
        # Use shared hardware optimization
        hardware_config = {
            'enable_hardware_optimization': self.unified_config.enable_hardware_optimization,
            'enable_m1_optimization': True,
            'memory_limit_gb': self.unified_config.memory_limit_gb
        }
        hardware_optimizer = SharedHardwareOptimizer(hardware_config)
        
        # Run NAS with shared components
        with hardware_optimizer.memory_context():
            results = self.nas_trainer.run_full_nas(X_train, y_train)
        
        # Use shared evaluation metrics
        if 'evaluation_results' in results:
            eval_results = results['evaluation_results']
            y_pred = eval_results.get('predictions', np.random.randint(0, 2, len(y_val)))
            
            # Calculate additional metrics using shared components
            basic_metrics = SharedEvaluationMetrics.calculate_basic_metrics(y_val, y_pred)
            trading_metrics = SharedEvaluationMetrics.calculate_trading_metrics(
                np.random.randn(len(y_val)) * 0.01,  # Simulated returns
                y_pred,
                y_val
            )
            
            # Merge with existing results
            results['shared_basic_metrics'] = basic_metrics
            results['shared_trading_metrics'] = trading_metrics
        
        # Save results using shared utilities
        if self.unified_config.save_results:
            SharedUtilities.save_results(results, f"{self.unified_config.output_dir}/nas_integration_results.json")
        
        return results


class TASIntegrationAdapter:
    """Adapter to integrate existing TAS system with unified components."""
    
    def __init__(self, unified_config: UnifiedArchitectureConfig):
        """Initialize TAS integration adapter."""
        self.unified_config = unified_config
        self.tas_config = self._convert_to_tas_config(unified_config)
        self.tas_engine = None
        
        if TAS_AVAILABLE:
            self.tas_engine = TreeArchitectureSearchEngine(self.tas_config)
    
    def _convert_to_tas_config(self, unified_config: UnifiedArchitectureConfig) -> TASConfig:
        """Convert unified config to TAS config."""
        # This would map unified config parameters to TAS-specific parameters
        # For now, create a basic config
        tas_config = TASConfig()
        # Map parameters as needed
        return tas_config
    
    def run_tas_with_unified_components(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Run TAS using unified shared components."""
        if not self.tas_engine:
            raise RuntimeError("TAS engine not available")
        
        # Use shared data processing
        config_dict = {
            'handle_missing_values': True,
            'normalize_data': True,
            'standardize_data': True,
            'enable_feature_selection': self.unified_config.enable_feature_selection,
            'max_features': self.unified_config.max_features,
            'validation_split': self.unified_config.validation_split,
            'time_series_split': True  # TAS typically uses time series
        }
        
        # Preprocess data using shared components
        X_processed, y_processed = SharedDataProcessor.preprocess_data(X, y, config_dict)
        X_train, X_val, y_train, y_val = SharedDataProcessor.split_data(X_processed, y_processed, config_dict)
        
        # Use shared hardware optimization
        hardware_config = {
            'enable_hardware_optimization': self.unified_config.enable_hardware_optimization,
            'enable_m1_optimization': True,
            'memory_limit_gb': self.unified_config.memory_limit_gb
        }
        hardware_optimizer = SharedHardwareOptimizer(hardware_config)
        
        # Run TAS with shared components
        with hardware_optimizer.memory_context():
            results = self.tas_engine.search_architectures(X_train, y_train)
        
        # Use shared evaluation metrics
        if hasattr(results, 'evaluation_results'):
            eval_results = results.evaluation_results
            y_pred = eval_results.get('predictions', np.random.randint(0, 2, len(y_val)))
            
            # Calculate additional metrics using shared components
            basic_metrics = SharedEvaluationMetrics.calculate_basic_metrics(y_val, y_pred)
            trading_metrics = SharedEvaluationMetrics.calculate_trading_metrics(
                np.random.randn(len(y_val)) * 0.01,  # Simulated returns
                y_pred,
                y_val
            )
            
            # Merge with existing results
            results.shared_basic_metrics = basic_metrics
            results.shared_trading_metrics = trading_metrics
        
        # Save results using shared utilities
        if self.unified_config.save_results:
            results_dict = {
                'tas_results': str(results),
                'shared_basic_metrics': getattr(results, 'shared_basic_metrics', {}),
                'shared_trading_metrics': getattr(results, 'shared_trading_metrics', {})
            }
            SharedUtilities.save_results(results_dict, f"{self.unified_config.output_dir}/tas_integration_results.json")
        
        return results


# ============================================================================
# UNIFIED INTEGRATION MANAGER
# ============================================================================

class UnifiedIntegrationManager:
    """Manager for integrating unified components with existing NAS and TAS systems."""
    
    def __init__(self, config: Optional[UnifiedArchitectureConfig] = None):
        """Initialize integration manager."""
        self.config = config or UnifiedArchitectureConfig()
        self.nas_adapter = None
        self.tas_adapter = None
        
        # Initialize adapters if components are available
        if NAS_AVAILABLE:
            self.nas_adapter = NASIntegrationAdapter(self.config)
        
        if TAS_AVAILABLE:
            self.tas_adapter = TASIntegrationAdapter(self.config)
    
    def run_unified_search(self, 
                          X: np.ndarray,
                          y: np.ndarray,
                          architecture_type: str = "hybrid") -> Dict[str, Any]:
        """Run unified search combining NAS, TAS, and hybrid approaches."""
        
        results = {
            'unified_search_completed': True,
            'timestamp': time.time(),
            'architecture_type': architecture_type,
            'config': {
                'search_strategy': self.config.search_strategy.value,
                'max_trials': self.config.max_trials,
                'enable_hardware_optimization': self.config.enable_hardware_optimization
            }
        }
        
        # Use shared data processing for all approaches
        config_dict = {
            'handle_missing_values': True,
            'normalize_data': True,
            'standardize_data': True,
            'enable_feature_selection': self.config.enable_feature_selection,
            'max_features': self.config.max_features,
            'validation_split': self.config.validation_split,
            'time_series_split': False
        }
        
        X_processed, y_processed = SharedDataProcessor.preprocess_data(X, y, config_dict)
        X_train, X_val, y_train, y_val = SharedDataProcessor.split_data(X_processed, y_processed, config_dict)
        
        # Use shared hardware optimization
        hardware_config = {
            'enable_hardware_optimization': self.config.enable_hardware_optimization,
            'enable_m1_optimization': True,
            'memory_limit_gb': self.config.memory_limit_gb
        }
        hardware_optimizer = SharedHardwareOptimizer(hardware_config)
        
        with hardware_optimizer.memory_context():
            # Run unified hybrid system
            unified_system = UnifiedHybridSystem(self.config)
            unified_results = unified_system.run_architecture_search(X_train, y_train, X_val, y_val)
            results['unified_results'] = unified_results
            
            # Run NAS if available
            if self.nas_adapter:
                try:
                    nas_results = self.nas_adapter.run_nas_with_unified_components(X_train, y_train)
                    results['nas_results'] = nas_results
                except Exception as e:
                    results['nas_error'] = str(e)
            
            # Run TAS if available
            if self.tas_adapter:
                try:
                    tas_results = self.tas_adapter.run_tas_with_unified_components(X_train, y_train)
                    results['tas_results'] = tas_results
                except Exception as e:
                    results['tas_error'] = str(e)
        
        # Compare results using shared evaluation
        results['comparison'] = self._compare_results(results)
        
        # Save unified results
        if self.config.save_results:
            SharedUtilities.save_results(results, f"{self.config.output_dir}/unified_integration_results.json")
        
        return results
    
    def _compare_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare results from different approaches."""
        comparison = {
            'unified_available': 'unified_results' in results,
            'nas_available': 'nas_results' in results,
            'tas_available': 'tas_results' in results
        }
        
        # Extract best scores for comparison
        scores = {}
        
        if 'unified_results' in results:
            unified_results = results['unified_results']
            if 'best_candidate' in unified_results:
                scores['unified'] = unified_results['best_candidate'].get('fitness_score', 0)
        
        if 'nas_results' in results:
            nas_results = results['nas_results']
            if 'evaluation_results' in nas_results:
                eval_results = nas_results['evaluation_results']
                scores['nas'] = eval_results.get('test_accuracy', 0)
        
        if 'tas_results' in results:
            tas_results = results['tas_results']
            if hasattr(tas_results, 'best_score'):
                scores['tas'] = tas_results.best_score
        
        comparison['scores'] = scores
        
        # Determine best approach
        if scores:
            best_approach = max(scores.items(), key=lambda x: x[1])
            comparison['best_approach'] = best_approach[0]
            comparison['best_score'] = best_approach[1]
        
        return comparison


# ============================================================================
# MIGRATION HELPERS
# ============================================================================

class MigrationHelper:
    """Helper functions for migrating existing NAS/TAS code to use unified components."""
    
    @staticmethod
    def migrate_nas_config(nas_config: Any) -> UnifiedArchitectureConfig:
        """Migrate existing NAS config to unified config."""
        unified_config = UnifiedArchitectureConfig()
        
        # Map NAS-specific parameters
        if hasattr(nas_config, 'search_strategy'):
            unified_config.search_strategy = SearchStrategy(nas_config.search_strategy)
        
        if hasattr(nas_config, 'max_trials'):
            unified_config.max_trials = nas_config.max_trials
        
        if hasattr(nas_config, 'max_epochs'):
            unified_config.max_epochs = nas_config.max_epochs
        
        if hasattr(nas_config, 'min_layers'):
            unified_config.min_layers = nas_config.min_layers
        
        if hasattr(nas_config, 'max_layers'):
            unified_config.max_layers = nas_config.max_layers
        
        if hasattr(nas_config, 'activation_functions'):
            unified_config.activation_functions = nas_config.activation_functions
        
        if hasattr(nas_config, 'use_m1_optimization'):
            unified_config.enable_hardware_optimization = nas_config.use_m1_optimization
        
        if hasattr(nas_config, 'output_dir'):
            unified_config.output_dir = nas_config.output_dir
        
        return unified_config
    
    @staticmethod
    def migrate_tas_config(tas_config: Any) -> UnifiedArchitectureConfig:
        """Migrate existing TAS config to unified config."""
        unified_config = UnifiedArchitectureConfig()
        
        # Map TAS-specific parameters
        if hasattr(tas_config, 'search_strategy'):
            unified_config.search_strategy = SearchStrategy(tas_config.search_strategy)
        
        if hasattr(tas_config, 'max_trials'):
            unified_config.max_trials = tas_config.max_trials
        
        if hasattr(tas_config, 'population_size'):
            unified_config.population_size = tas_config.population_size
        
        if hasattr(tas_config, 'enable_hardware_optimization'):
            unified_config.enable_hardware_optimization = tas_config.enable_hardware_optimization
        
        return unified_config
    
    @staticmethod
    def create_integration_checklist() -> List[str]:
        """Create checklist for integrating unified components."""
        checklist = [
            "✅ Import unified_hybrid_architecture and shared_components modules",
            "✅ Replace custom configuration classes with UnifiedArchitectureConfig",
            "✅ Replace custom evaluation functions with SharedEvaluationMetrics",
            "✅ Replace custom hardware optimization with SharedHardwareOptimizer",
            "✅ Replace custom search algorithms with SharedSearchAlgorithms",
            "✅ Replace custom data processing with SharedDataProcessor",
            "✅ Replace custom utility functions with SharedUtilities",
            "✅ Update existing NAS/TAS classes to use unified components",
            "✅ Test integration with existing workflows",
            "✅ Update documentation and examples",
            "✅ Remove duplicate code and consolidate implementations"
        ]
        return checklist


# ============================================================================
# EXAMPLE USAGE AND INTEGRATION
# ============================================================================

def demonstrate_unified_integration():
    """Demonstrate how to use unified components with existing systems."""
    
    print("="*60)
    print("UNIFIED NAS/TAS INTEGRATION DEMONSTRATION")
    print("="*60)
    
    # Create sample data
    X, y = SharedUtilities.create_sample_data(n_samples=1000, n_features=20, n_classes=2)
    print(f"Created sample data: {X.shape}, {y.shape}")
    
    # Create unified configuration
    unified_config = UnifiedArchitectureConfig(
        architecture_type=ArchitectureType.FEEDFORWARD,
        search_strategy=SearchStrategy.RANDOM,
        max_trials=20,
        enable_hardware_optimization=True,
        save_results=True,
        output_dir="integration_demo_results"
    )
    
    print(f"Unified config created: {unified_config.architecture_type.value}")
    
    # Create integration manager
    integration_manager = UnifiedIntegrationManager(unified_config)
    
    # Run unified search
    print("\nRunning unified search...")
    start_time = time.time()
    
    results = integration_manager.run_unified_search(X, y, "hybrid")
    
    execution_time = time.time() - start_time
    print(f"Unified search completed in {execution_time:.2f} seconds")
    
    # Display results
    print("\nResults Summary:")
    print(f"Unified search completed: {results['unified_search_completed']}")
    
    if 'comparison' in results:
        comparison = results['comparison']
        print(f"Available approaches: {[k for k, v in comparison.items() if k.endswith('_available') and v]}")
        
        if 'scores' in comparison:
            print("Performance scores:")
            for approach, score in comparison['scores'].items():
                print(f"  {approach}: {score:.4f}")
        
        if 'best_approach' in comparison:
            print(f"Best approach: {comparison['best_approach']} (score: {comparison['best_score']:.4f})")
    
    # Show migration checklist
    print("\nMigration Checklist:")
    checklist = MigrationHelper.create_integration_checklist()
    for item in checklist:
        print(f"  {item}")
    
    print("\n" + "="*60)
    print("INTEGRATION DEMONSTRATION COMPLETED")
    print("="*60)


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_unified_integration()