"""
Clustering Integration Layer.

This module provides integration between different clustering approaches
and regime detection systems. It enables seamless data exchange, comparison,
and enhancement of clustering-based regime identification.

Key Integration Features:
- Data format conversion between systems
- Clustering result enhancement with feature analysis
- Comparative analysis between different clustering approaches
- Hybrid regime identification combining multiple methods
- Performance benchmarking and validation
- Migration utilities for existing workflows
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import json
import asyncio

from src.utils.logger import system_logger

# Import clustering components
from .dimension_analyzer import MarketDimensionAnalyzer, DimensionAnalysisConfig
from .regime_clusterer import RegimeClusterer, ClusteringConfig
from .feature_importance import RegimeFeatureImportance, ImportanceConfig
from .validation_metrics import RegimeValidationMetrics, ValidationConfig
from .dimension_discovery_pipeline import DimensionDiscoveryPipeline, DiscoveryConfig


class IntegrationMethod(Enum):
    """Enumeration of integration methods."""
    DIMENSION_FIRST = "dimension_first"  # Discover dimensions → then clustering
    CLUSTERING_FIRST = "clustering_first"  # Clustering → then dimension analysis
    CLUSTERING_ONLY = "clustering_only"  # Pure clustering approach
    CLUSTERING_ENHANCED = "clustering_enhanced"  # Clustering + discovered dimensions
    HYBRID = "hybrid"  # Simultaneous dimension discovery and clustering
    COMPARATIVE = "comparative"  # Compare all approaches


@dataclass
class IntegrationConfig:
    """Configuration for clustering integration."""
    # Integration method
    method: IntegrationMethod = IntegrationMethod.HYBRID
    
    # Clustering parameters
    clustering_n_clusters: int = 5
    clustering_methods: List[str] = None
    
    # Feature analysis parameters
    analyze_dimensions: bool = True
    analyze_feature_importance: bool = True
    
    # Validation parameters
    validate_results: bool = True
    compare_methods: bool = True
    
    # Output parameters
    save_results: bool = True
    output_directory: str = "regime_integration_results"
    
    def __post_init__(self):
        """Set default values after initialization."""
        if self.clustering_methods is None:
            self.clustering_methods = ["kmeans", "gmm", "hierarchical"]


@dataclass
class IntegrationResult:
    """Result container for integration analysis."""
    method: IntegrationMethod
    clustering_results: Optional[Dict[str, Any]]
    dimension_analysis: Optional[Dict[str, Any]]
    feature_importance: Optional[Dict[str, Any]]
    validation_metrics: Optional[Dict[str, Any]]
    comparison_analysis: Optional[Dict[str, Any]]
    
    # Performance metrics
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    
    # Metadata
    timestamp: str = ""
    data_shape: Tuple[int, int] = (0, 0)
    config: Optional[IntegrationConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'method': self.method.value,
            'clustering_results': self.clustering_results,
            'dimension_analysis': self.dimension_analysis,
            'feature_importance': self.feature_importance,
            'validation_metrics': self.validation_metrics,
            'comparison_analysis': self.comparison_analysis,
            'execution_time': self.execution_time,
            'success': self.success,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
            'data_shape': self.data_shape
        }


class IntegrationLayer:
    """
    Integration layer for clustering-based regime detection.
    
    This class provides comprehensive integration between different clustering
    approaches and feature analysis methods for regime detection.
    """
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.logger = system_logger.getChild('IntegrationLayer')
        
        # Initialize components
        self.dimension_analyzer = MarketDimensionAnalyzer(DimensionAnalysisConfig())
        self.regime_clusterer = RegimeClusterer(ClusteringConfig())
        self.feature_importance = RegimeFeatureImportance(ImportanceConfig())
        self.validation_metrics = RegimeValidationMetrics(ValidationConfig())
        self.dimension_pipeline = DimensionDiscoveryPipeline(DiscoveryConfig())
        
        self.logger.info(f"Integration layer initialized with method: {config.method.value}")
    
    async def run_integration_analysis(self, 
                                     data: pd.DataFrame,
                                     symbol: str = "ETHUSDT") -> IntegrationResult:
        """
        Run comprehensive integration analysis.
        
        Args:
            data: Market data DataFrame
            symbol: Trading symbol
            
        Returns:
            IntegrationResult with analysis results
        """
        start_time = pd.Timestamp.now()
        self.logger.info(f"Starting integration analysis for {symbol}")
        self.logger.info(f"Data shape: {data.shape}")
        
        try:
            result = IntegrationResult(
                method=self.config.method,
                clustering_results=None,
                dimension_analysis=None,
                feature_importance=None,
                validation_metrics=None,
                comparison_analysis=None,
                timestamp=start_time.isoformat(),
                data_shape=data.shape,
                config=self.config
            )
            
            # Run analysis based on method
            if self.config.method == IntegrationMethod.DIMENSION_FIRST:
                await self._run_dimension_first_analysis(data, result)
            elif self.config.method == IntegrationMethod.CLUSTERING_FIRST:
                await self._run_clustering_first_analysis(data, result)
            elif self.config.method == IntegrationMethod.CLUSTERING_ONLY:
                await self._run_clustering_only_analysis(data, result)
            elif self.config.method == IntegrationMethod.CLUSTERING_ENHANCED:
                await self._run_clustering_enhanced_analysis(data, result)
            elif self.config.method == IntegrationMethod.HYBRID:
                await self._run_hybrid_analysis(data, result)
            elif self.config.method == IntegrationMethod.COMPARATIVE:
                await self._run_comparative_analysis(data, result)
            else:
                raise ValueError(f"Unknown integration method: {self.config.method}")
            
            # Calculate execution time
            result.execution_time = (pd.Timestamp.now() - start_time).total_seconds()
            
            # Save results if configured
            if self.config.save_results:
                await self._save_integration_results(result, symbol)
            
            self.logger.info(f"Integration analysis completed in {result.execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Integration analysis failed: {e}")
            return IntegrationResult(
                method=self.config.method,
                success=False,
                error_message=str(e),
                timestamp=start_time.isoformat(),
                data_shape=data.shape,
                config=self.config
            )
    
    async def _run_dimension_first_analysis(self, data: pd.DataFrame, result: IntegrationResult):
        """Run dimension-first analysis."""
        self.logger.info("Running dimension-first analysis")
        
        # Step 1: Discover dimensions
        dimension_results = await self.dimension_pipeline.run_complete_analysis(data)
        result.dimension_analysis = dimension_results
        
        # Step 2: Use discovered dimensions for clustering
        if dimension_results and 'optimal_features' in dimension_results:
            optimal_features = dimension_results['optimal_features']
            clustering_results = await self.regime_clusterer.run_all_methods(
                data[optimal_features].values,
                analyze_dimensions=False,  # Already analyzed
                feature_names=optimal_features
            )
            result.clustering_results = clustering_results
        
        # Step 3: Feature importance analysis
        if self.config.analyze_feature_importance and result.clustering_results:
            importance_results = await self.feature_importance.analyze_importance(
                data, result.clustering_results
            )
            result.feature_importance = importance_results
        
        # Step 4: Validation
        if self.config.validate_results and result.clustering_results:
            validation_results = await self.validation_metrics.validate_all_metrics(
                data, result.clustering_results
            )
            result.validation_metrics = validation_results
    
    async def _run_clustering_first_analysis(self, data: pd.DataFrame, result: IntegrationResult):
        """Run clustering-first analysis."""
        self.logger.info("Running clustering-first analysis")
        
        # Step 1: Run clustering
        clustering_results = await self.regime_clusterer.run_all_methods(
            data.values,
            analyze_dimensions=True,
            feature_names=data.columns.tolist()
        )
        result.clustering_results = clustering_results
        
        # Step 2: Analyze dimensions based on clustering results
        if clustering_results and self.config.analyze_dimensions:
            dimension_results = await self.dimension_analyzer.analyze_dimensions(
                data, clustering_results
            )
            result.dimension_analysis = dimension_results
        
        # Step 3: Feature importance analysis
        if self.config.analyze_feature_importance and clustering_results:
            importance_results = await self.feature_importance.analyze_importance(
                data, clustering_results
            )
            result.feature_importance = importance_results
        
        # Step 4: Validation
        if self.config.validate_results and clustering_results:
            validation_results = await self.validation_metrics.validate_all_metrics(
                data, clustering_results
            )
            result.validation_metrics = validation_results
    
    async def _run_clustering_only_analysis(self, data: pd.DataFrame, result: IntegrationResult):
        """Run clustering-only analysis."""
        self.logger.info("Running clustering-only analysis")
        
        # Run clustering
        clustering_results = await self.regime_clusterer.run_all_methods(
            data.values,
            analyze_dimensions=False,
            feature_names=data.columns.tolist()
        )
        result.clustering_results = clustering_results
        
        # Validation only
        if self.config.validate_results and clustering_results:
            validation_results = await self.validation_metrics.validate_all_metrics(
                data, clustering_results
            )
            result.validation_metrics = validation_results
    
    async def _run_clustering_enhanced_analysis(self, data: pd.DataFrame, result: IntegrationResult):
        """Run clustering-enhanced analysis."""
        self.logger.info("Running clustering-enhanced analysis")
        
        # Step 1: Quick dimension analysis
        dimension_results = await self.dimension_analyzer.analyze_dimensions(data)
        result.dimension_analysis = dimension_results
        
        # Step 2: Enhanced clustering with dimension insights
        if dimension_results and 'important_features' in dimension_results:
            important_features = dimension_results['important_features']
            enhanced_data = data[important_features]
        else:
            enhanced_data = data
        
        clustering_results = await self.regime_clusterer.run_all_methods(
            enhanced_data.values,
            analyze_dimensions=False,
            feature_names=enhanced_data.columns.tolist()
        )
        result.clustering_results = clustering_results
        
        # Step 3: Feature importance analysis
        if self.config.analyze_feature_importance and clustering_results:
            importance_results = await self.feature_importance.analyze_importance(
                data, clustering_results
            )
            result.feature_importance = importance_results
        
        # Step 4: Validation
        if self.config.validate_results and clustering_results:
            validation_results = await self.validation_metrics.validate_all_metrics(
                data, clustering_results
            )
            result.validation_metrics = validation_results
    
    async def _run_hybrid_analysis(self, data: pd.DataFrame, result: IntegrationResult):
        """Run hybrid analysis combining dimension discovery and clustering."""
        self.logger.info("Running hybrid analysis")
        
        # Run dimension discovery and clustering in parallel
        dimension_task = self.dimension_pipeline.run_complete_analysis(data)
        clustering_task = self.regime_clusterer.run_all_methods(
            data.values,
            analyze_dimensions=True,
            feature_names=data.columns.tolist()
        )
        
        # Wait for both to complete
        dimension_results, clustering_results = await asyncio.gather(
            dimension_task, clustering_task
        )
        
        result.dimension_analysis = dimension_results
        result.clustering_results = clustering_results
        
        # Feature importance analysis
        if self.config.analyze_feature_importance and clustering_results:
            importance_results = await self.feature_importance.analyze_importance(
                data, clustering_results
            )
            result.feature_importance = importance_results
        
        # Validation
        if self.config.validate_results and clustering_results:
            validation_results = await self.validation_metrics.validate_all_metrics(
                data, clustering_results
            )
            result.validation_metrics = validation_results
    
    async def _run_comparative_analysis(self, data: pd.DataFrame, result: IntegrationResult):
        """Run comparative analysis of all methods."""
        self.logger.info("Running comparative analysis")
        
        # Run all methods
        dimension_task = self.dimension_pipeline.run_complete_analysis(data)
        clustering_task = self.regime_clusterer.run_all_methods(
            data.values,
            analyze_dimensions=True,
            feature_names=data.columns.tolist()
        )
        
        # Wait for completion
        dimension_results, clustering_results = await asyncio.gather(
            dimension_task, clustering_task
        )
        
        result.dimension_analysis = dimension_results
        result.clustering_results = clustering_results
        
        # Feature importance analysis
        if self.config.analyze_feature_importance and clustering_results:
            importance_results = await self.feature_importance.analyze_importance(
                data, clustering_results
            )
            result.feature_importance = importance_results
        
        # Validation
        if self.config.validate_results and clustering_results:
            validation_results = await self.validation_metrics.validate_all_metrics(
                data, clustering_results
            )
            result.validation_metrics = validation_results
        
        # Comparative analysis
        if self.config.compare_methods:
            comparison_results = await self._compare_methods(
                data, dimension_results, clustering_results
            )
            result.comparison_analysis = comparison_results
    
    async def _compare_methods(self, 
                             data: pd.DataFrame,
                             dimension_results: Dict[str, Any],
                             clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare different methods and approaches."""
        comparison = {
            'method_comparison': {},
            'performance_metrics': {},
            'recommendations': []
        }
        
        # Compare clustering methods
        if clustering_results:
            best_method, best_result = self.regime_clusterer.get_best_method()
            comparison['method_comparison']['best_clustering_method'] = best_method.value if best_method else 'none'
            comparison['method_comparison']['clustering_methods_tested'] = len(clustering_results)
        
        # Compare with dimension analysis
        if dimension_results and clustering_results:
            # Simple comparison based on number of features used
            if 'optimal_features' in dimension_results:
                n_optimal_features = len(dimension_results['optimal_features'])
                n_original_features = len(data.columns)
                feature_reduction = (n_original_features - n_optimal_features) / n_original_features
                
                comparison['performance_metrics']['feature_reduction'] = feature_reduction
                comparison['performance_metrics']['n_optimal_features'] = n_optimal_features
                comparison['performance_metrics']['n_original_features'] = n_original_features
        
        # Generate recommendations
        if clustering_results and dimension_results:
            comparison['recommendations'].append("Both clustering and dimension analysis completed successfully")
            
            if 'optimal_features' in dimension_results:
                feature_reduction = comparison['performance_metrics'].get('feature_reduction', 0)
                if feature_reduction > 0.5:
                    comparison['recommendations'].append("Significant feature reduction achieved - consider using optimal features")
                elif feature_reduction > 0.2:
                    comparison['recommendations'].append("Moderate feature reduction - optimal features may improve performance")
                else:
                    comparison['recommendations'].append("Minimal feature reduction - all features may be important")
        
        return comparison
    
    async def _save_integration_results(self, result: IntegrationResult, symbol: str):
        """Save integration results to disk."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main results
            results_file = output_dir / f"integration_results_{symbol}_{result.timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            self.logger.info(f"Integration results saved to {results_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save integration results: {e}")


# Convenience functions
async def run_integration_analysis(data: pd.DataFrame,
                                 method: IntegrationMethod = IntegrationMethod.HYBRID,
                                 symbol: str = "ETHUSDT",
                                 **kwargs) -> IntegrationResult:
    """
    Run integration analysis with default configuration.
    
    Args:
        data: Market data DataFrame
        method: Integration method to use
        symbol: Trading symbol
        **kwargs: Additional configuration parameters
        
    Returns:
        IntegrationResult with analysis results
    """
    config = IntegrationConfig(method=method, **kwargs)
    integration_layer = IntegrationLayer(config)
    return await integration_layer.run_integration_analysis(data, symbol)


def create_integration_config(method: IntegrationMethod = IntegrationMethod.HYBRID,
                            n_clusters: int = 5,
                            **kwargs) -> IntegrationConfig:
    """
    Create integration configuration with common parameters.
    
    Args:
        method: Integration method
        n_clusters: Number of clusters
        **kwargs: Additional configuration parameters
        
    Returns:
        IntegrationConfig instance
    """
    return IntegrationConfig(
        method=method,
        clustering_n_clusters=n_clusters,
        **kwargs
    )