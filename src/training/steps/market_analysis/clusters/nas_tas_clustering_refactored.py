"""
Refactored NAS-TAS Clustering Component.

This is the streamlined main component that orchestrates the refactored clustering modules.
It maintains the same public API as the original component while using the new modular architecture.
"""

import copy
import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import traceback
from pathlib import Path
from collections import defaultdict
import pickle
import re

from src.utils.tprint import (
    tprint,
    tprint_debug,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
    tprint_progress,
    tprint_performance,
    tprint_timer,
    tprint_structured,
)

from ...shared_utils import (
    # Features
    prepare_market_features,
    FeatureConfig,
    FeaturePreparationResult,

    # Configuration
    validate_regime_count,
    normalize_weights,
    validate_algorithm_type,
    create_default_config,
    ConfigValidator,
    BaseConfig,

    # Logging
    get_logger,
    log_execution,
    log_performance,
    LoggingContext,

    # Metrics
    calculate_consensus_metrics,
    calculate_disagreement_metrics,
    calculate_economic_scores,
    calculate_trading_scores,
    calculate_stability_scores,
    MetricsCalculator,

    # Characteristics
    create_regime_characteristics,
    generate_cluster_characteristics,
    CharacteristicsGenerator,
)

from ...shared_utils.calibration_registry import (
    get_current_calibration,
    get_quality_thresholds as get_calibrated_thresholds,
    update_quality_calibration,
)

# Import the refactored clustering modules
from . import (
    ClusteringOrchestrator,
    ClusteringContext
)


@dataclass
class ClusteringContext:
    """Context for clustering operations."""
    original_features: np.ndarray
    market_data: pd.DataFrame
    memory_optimizer: Any = None
    original_feature_names: Optional[List[str]] = None
    feature_scores: Optional[Dict[str, float]] = None
    
    # Outputs
    optimized_features: Optional[np.ndarray] = None
    optimized_feature_names: Optional[List[str]] = None
    dropped_feature_names: Optional[List[str]] = None
    pca_loading_scores: Optional[Dict[str, float]] = None
    pre_pca_feature_names: Optional[List[str]] = None
    pre_pca_feature_count: Optional[int] = None
    
    # Clustering outputs
    tas_assignments: Optional[np.ndarray] = None
    nas_assignments: Optional[np.ndarray] = None
    initial_assignments: Optional[np.ndarray] = None
    optimized_assignments: Optional[np.ndarray] = None
    optimal_k: Optional[int] = None
    final_k: Optional[int] = None
    
    # Results
    validation_results: Optional[Dict[str, Any]] = None
    stability_results: Optional[Dict[str, Any]] = None
    final_results: Optional[Dict[str, Any]] = None


class NASTASClusteringConfig(BaseConfig):
    """Configuration for NAS-TAS Clustering Component."""
    
    def __post_init__(self):
        """Post-initialization validation."""
        super().__post_init__()
        
        # Set default values
        if not hasattr(self, 'n_regimes') or self.n_regimes is None:
            self.n_regimes = 6
        
        if not hasattr(self, 'feature_categories') or self.feature_categories is None:
            self.feature_categories = [
                'regime_volatility', 
                'regime_volume', 
                'regime_structural_trend', 
                'regime_statistical'
            ]
        
        if not hasattr(self, 'use_standardized_features') or self.use_standardized_features is None:
            self.use_standardized_features = True
        
        if not hasattr(self, 'enable_samples_reallocation') or self.enable_samples_reallocation is None:
            self.enable_samples_reallocation = True


class NASTASClusteringComponent:
    """
    Refactored NAS-TAS Clustering Component.
    
    This component uses the new modular architecture with separate steps and
    iterative optimization processes for improved maintainability and performance.
    """
    
    def __init__(self, config: Optional[NASTASClusteringConfig] = None):
        """Initialize the refactored NAS-TAS clustering component."""
        with LoggingContext('NAS-TAS-Clustering-Refactored', 'Initialization', verbose=True):
            # Initialize configuration
            self.config = config or NASTASClusteringConfig()
            
            # Use shared logging utilities
            self.logger = get_logger('NASTASClusteringRefactored')
            
            # Initialize shared utilities
            self.config_validator = ConfigValidator(verbose=True)
            self.metrics_calculator = MetricsCalculator(verbose=True)
            self.characteristics_generator = CharacteristicsGenerator(verbose=True)
            
            # Initialize the clustering orchestrator
            self.clustering_orchestrator = ClusteringOrchestrator(verbose=True)
            
            # Initialize state
            self.clustering_result = None
            self.execution_metadata = {}
            
            # Performance monitoring
            self.performance_metrics = {
                "start_time": None,
                "end_time": None,
                "memory_usage": [],
                "processing_times": {},
                "error_count": 0,
                "success_count": 0,
                "optimization_trials": 0,
                "cv_folds": 0
            }
            
            tprint_success("🔍 Refactored NAS-TAS Clustering Component initialized")
    
    def _log(self, message: str, level: str = "INFO") -> None:
        """Log a message with the specified level."""
        self.logger.log(level, message)
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts for this component."""
        return ["market_data", "features"]
    
    async def _perform_clustering(self, features: np.ndarray, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform clustering using the refactored pipeline."""
        try:
            tprint("Performing clustering using refactored pipeline...", "INFO")
            
            # Validate inputs
            validation_results = self.clustering_orchestrator.validate_pipeline_requirements(
                features, market_data
            )
            
            if not validation_results["valid"]:
                raise ValueError(f"Pipeline validation failed: {validation_results['issues']}")
            
            if validation_results["warnings"]:
                for warning in validation_results["warnings"]:
                    tprint_warning(f"⚠️ {warning}")
            
            # Execute the clustering pipeline
            clustering_result = await self.clustering_orchestrator.execute_clustering_pipeline(
                features, market_data, self.config
            )
            
            # Store results
            self.clustering_result = clustering_result
            
            tprint("Clustering completed successfully", "SUCCESS")
            return clustering_result
            
        except Exception as e:
            tprint(f"Clustering failed: {e}", "ERROR")
            raise ValueError(f"Clustering failed: {e}")
    
    async def run(self, market_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Run the clustering component."""
        try:
            tprint("Starting NAS-TAS Clustering (Refactored)", "INFO")
            
            # Prepare features using shared utilities
            feature_result = await self._prepare_features(market_data)
            features = feature_result.features
            
            # Perform clustering
            clustering_result = await self._perform_clustering(features, market_data)
            
            # Create consolidated artifacts
            artifacts = await self._create_consolidated_artifacts(clustering_result, market_data)
            
            # Return results
            return {
                'clustering_result': clustering_result,
                'artifacts': artifacts,
                'execution_metadata': self.execution_metadata,
                'performance_metrics': self.performance_metrics
            }
            
        except Exception as e:
            tprint(f"Component execution failed: {e}", "ERROR")
            raise ValueError(f"Component execution failed: {e}")
    
    async def _prepare_features(self, market_data: pd.DataFrame) -> FeaturePreparationResult:
        """Prepare features using shared utilities."""
        try:
            # Use shared feature configuration
            feature_config = FeatureConfig(
                feature_categories=self.config.feature_categories,
                use_standardized_features=self.config.use_standardized_features,
                drop_highly_correlated=True
            )
            
            # Prepare features using shared utilities
            feature_result = prepare_market_features(
                market_data=market_data,
                config=feature_config
            )
            
            tprint(f"Prepared {feature_result.features.shape[1]} features", "SUCCESS")
            return feature_result
            
        except Exception as e:
            tprint(f"Feature preparation failed: {e}", "ERROR")
            raise
    
    async def _create_consolidated_artifacts(
        self, 
        clustering_result: Dict[str, Any], 
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create consolidated artifacts."""
        try:
            artifacts = {
                'nas_tas_clustering_result': clustering_result,
                'clustering_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'component_version': '2.0.0',
                    'refactored': True,
                    'pipeline_steps': [
                        'feature_preparation',
                        'initial_clustering', 
                        'iterative_optimization',
                        'validation',
                        'results_consolidation'
                    ]
                },
                'performance_summary': self.clustering_orchestrator.get_performance_summary()
            }
            
            return artifacts
            
        except Exception as e:
            tprint(f"Artifact creation failed: {e}", "ERROR")
            return {'error': str(e)}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of the clustering component."""
        try:
            orchestrator_summary = self.clustering_orchestrator.get_performance_summary()
            
            summary = {
                "component": "NAS-TAS Clustering (Refactored)",
                "version": "2.0.0",
                "orchestrator_performance": orchestrator_summary,
                "component_metrics": self.performance_metrics,
                "refactored_architecture": True,
                "modular_steps": True
            }
            
            return summary
            
        except Exception as e:
            tprint(f"Performance summary generation failed: {e}", "ERROR")
            return {"error": str(e)}
    
    def reset_performance_metrics(self) -> None:
        """Reset performance metrics."""
        try:
            self.performance_metrics = {
                "start_time": None,
                "end_time": None,
                "memory_usage": [],
                "processing_times": {},
                "error_count": 0,
                "success_count": 0,
                "optimization_trials": 0,
                "cv_folds": 0
            }
            self.clustering_orchestrator.reset_performance_metrics()
            tprint("Performance metrics reset", "INFO")
        except Exception as e:
            tprint(f"Performance metrics reset failed: {e}", "ERROR")
    
    def get_step_info(self) -> Dict[str, str]:
        """Get information about available pipeline steps."""
        return self.clustering_orchestrator.get_step_info()
    
    async def execute_step_individually(
        self, 
        step_name: str, 
        features: np.ndarray, 
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Execute a single step individually for testing/debugging."""
        try:
            # Create context
            context = ClusteringContext(
                original_features=features,
                market_data=market_data
            )
            
            # Execute step
            result_context = await self.clustering_orchestrator.execute_step_individually(
                step_name, context, self.config
            )
            
            return {
                'step_name': step_name,
                'context': result_context,
                'success': True
            }
            
        except Exception as e:
            tprint(f"Individual step execution failed for {step_name}: {e}", "ERROR")
            return {
                'step_name': step_name,
                'error': str(e),
                'success': False
            }