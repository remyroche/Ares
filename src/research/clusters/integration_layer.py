"""
HMM Integration Layer.

This module provides integration between the new regime clustering research
framework and the existing HMM regime discovery and clustering systems.
It enables seamless data exchange, comparison, and enhancement of existing
HMM-based regime identification.

Key Integration Features:
- Data format conversion between systems
- HMM result enhancement with clustering research
- Comparative analysis between HMM and clustering approaches
- Hybrid regime identification combining both methods
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

# Import existing HMM components (with error handling)
try:
    from src.training.steps.market_analysis.components.hmm_regime_discovery import HMMRegimeDiscoveryComponent
    from src.training.steps.market_analysis.components.hmm_clustering import HMMClusteringComponent
    HMM_AVAILABLE = True
except ImportError as e:
    system_logger.warning(f"HMM components not available: {e}")
    HMM_AVAILABLE = False

# Import new clustering components
from .dimension_analyzer import MarketDimensionAnalyzer, DimensionAnalysisConfig
from .regime_clusterer import RegimeClusterer, ClusteringConfig
from .feature_importance import RegimeFeatureImportance, ImportanceConfig
from .validation_metrics import RegimeValidationMetrics, ValidationConfig
from .dimension_discovery_pipeline import DimensionDiscoveryPipeline, DiscoveryConfig


class IntegrationMethod(Enum):
    """Enumeration of integration methods."""
    DIMENSION_FIRST = "dimension_first"  # Discover dimensions → then HMM
    HMM_FIRST = "hmm_first"  # HMM → then dimension analysis
    CLUSTERING_ONLY = "clustering_only"  # Pure clustering approach
    HMM_ENHANCED = "hmm_enhanced"  # HMM + discovered dimensions
    CLUSTERING_ENHANCED = "clustering_enhanced"  # Clustering + HMM priors
    HYBRID = "hybrid"  # Simultaneous dimension discovery and HMM
    COMPARATIVE = "comparative"  # Compare all approaches


@dataclass
class IntegrationConfig:
    """Configuration for HMM integration."""
    # Integration method
    method: IntegrationMethod = IntegrationMethod.HYBRID
    
    # HMM parameters
    hmm_n_components: int = 5
    hmm_optimization_mode: str = "blank"
    
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
    hmm_results: Optional[Dict[str, Any]]
    clustering_results: Optional[Dict[str, Any]]
    dimension_analysis: Optional[Dict[str, Any]]
    feature_importance: Optional[Dict[str, Any]]
    validation_metrics: Optional[Dict[str, Any]]
    comparison_analysis: Optional[Dict[str, Any]]
    recommendations: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'method': self.method.value,
            'hmm_results': self.hmm_results,
            'clustering_results': self.clustering_results,
            'dimension_analysis': self.dimension_analysis,
            'feature_importance': self.feature_importance,
            'validation_metrics': self.validation_metrics,
            'comparison_analysis': self.comparison_analysis,
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }


class HMMDataAdapter:
    """Adapter for converting between HMM and clustering data formats."""
    
    def __init__(self):
        self.logger = system_logger.getChild('HMMDataAdapter')
    
    def hmm_to_clustering_format(self, hmm_result: Dict[str, Any]) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Convert HMM results to clustering format.
        
        Args:
            hmm_result: HMM regime discovery result
            
        Returns:
            Tuple of (features_dataframe, regime_labels)
        """
        try:
            # Extract regime assignments
            if 'regime_assignments' in hmm_result:
                regime_labels = np.array(hmm_result['regime_assignments'])
            else:
                raise ValueError("No regime assignments found in HMM result")
            
            # Create features dataframe (placeholder - would need actual market data)
            # In practice, this would come from the original market data used for HMM
            n_samples = len(regime_labels)
            features = pd.DataFrame({
                'regime_id': regime_labels,
                'sample_index': range(n_samples)
            })
            
            self.logger.info(f"Converted HMM result: {len(regime_labels)} samples, {len(np.unique(regime_labels))} regimes")
            
            return features, regime_labels
            
        except Exception as e:
            self.logger.error(f"Failed to convert HMM to clustering format: {e}")
            raise
    
    def clustering_to_hmm_format(self, 
                                features: pd.DataFrame,
                                regime_labels: np.ndarray,
                                clustering_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert clustering results to HMM format.
        
        Args:
            features: Features used for clustering
            regime_labels: Clustering regime assignments
            clustering_result: Clustering analysis result
            
        Returns:
            HMM-compatible result dictionary
        """
        try:
            # Create HMM-compatible result
            hmm_compatible = {
                'regime_models': [f"regime_{i}" for i in range(len(np.unique(regime_labels)))],
                'regime_assignments': regime_labels.tolist(),
                'regime_metrics': {
                    'total_regimes': len(np.unique(regime_labels)),
                    'total_samples': len(regime_labels),
                    'regime_distribution': {
                        str(regime): int(count) 
                        for regime, count in zip(*np.unique(regime_labels, return_counts=True))
                    }
                },
                'regime_discovery_summary': {
                    'total_regimes': len(np.unique(regime_labels)),
                    'total_assignments': len(regime_labels),
                    'discovery_method': 'clustering',
                    'clustering_method': clustering_result.get('method', 'unknown')
                },
                'metadata': {
                    'data_points': len(features),
                    'n_features': len(features.columns),
                    'clustering_score': clustering_result.get('silhouette_score', 0.0)
                }
            }
            
            self.logger.info(f"Converted clustering to HMM format: {len(np.unique(regime_labels))} regimes")
            
            return hmm_compatible
            
        except Exception as e:
            self.logger.error(f"Failed to convert clustering to HMM format: {e}")
            raise


class HMMIntegrationLayer:
    """
    Main integration layer between HMM and clustering systems.
    
    This class provides comprehensive integration capabilities between the
    existing HMM regime discovery/clustering and the new research framework.
    """
    
    def __init__(self, config: Optional[IntegrationConfig] = None):
        """
        Initialize the HMM integration layer.
        
        Args:
            config: Configuration for integration
        """
        self.config = config or IntegrationConfig()
        self.logger = system_logger.getChild('HMMIntegrationLayer')
        self.data_adapter = HMMDataAdapter()
        
        # Initialize components
        self.dimension_analyzer = MarketDimensionAnalyzer(DimensionAnalysisConfig())
        self.regime_clusterer = RegimeClusterer(ClusteringConfig(n_clusters=self.config.clustering_n_clusters))
        self.feature_importance = RegimeFeatureImportance(ImportanceConfig())
        self.validation_metrics = RegimeValidationMetrics(ValidationConfig())
        self.dimension_discovery = DimensionDiscoveryPipeline(discovery_config=DiscoveryConfig())
        
        # Initialize HMM components if available
        if HMM_AVAILABLE:
            try:
                from src.core.config_service import ConfigService
                config_service = ConfigService()
                
                # Create component configs (simplified)
                component_config = type('ComponentConfig', (), {
                    'symbol': 'ETHUSDT',
                    'exchange': 'binance',
                    'timeframe': '1h',
                    'optimization_mode': self.config.hmm_optimization_mode
                })()
                
                self.hmm_regime_discovery = HMMRegimeDiscoveryComponent(component_config)
                self.hmm_clustering = HMMClusteringComponent(component_config)
                
            except Exception as e:
                self.logger.warning(f"Could not initialize HMM components: {e}")
                self.hmm_regime_discovery = None
                self.hmm_clustering = None
        else:
            self.hmm_regime_discovery = None
            self.hmm_clustering = None
    
    async def run_integration_analysis(self, 
                                     market_data: pd.DataFrame,
                                     target: Optional[np.ndarray] = None) -> IntegrationResult:
        """
        Run comprehensive integration analysis.
        
        Args:
            market_data: Market data for analysis
            target: Optional target variable for supervised analysis
            
        Returns:
            Integration analysis result
        """
        self.logger.info(f"🚀 Starting {self.config.method.value} integration analysis")
        
        # Initialize result containers
        hmm_results = None
        clustering_results = None
        dimension_analysis = None
        feature_importance_results = None
        validation_results = None
        comparison_analysis = None
        recommendations = []
        
        try:
            # Step 1: Run HMM analysis (if available and requested)
            if self._should_run_hmm():
                self.logger.info("📊 Running HMM regime discovery")
                hmm_results = await self._run_hmm_analysis(market_data)
            
            # Step 2: Run clustering analysis
            if self._should_run_clustering():
                self.logger.info("🔍 Running clustering analysis")
                clustering_results = await self._run_clustering_analysis(market_data, target)
            
            # Step 3: Run dimension analysis
            if self.config.analyze_dimensions:
                self.logger.info("📈 Analyzing market dimensions")
                # First, try the new discovery pipeline (dynamic targets + mRMR+PID aggregation)
                try:
                    feature_data = self._prepare_clustering_features(market_data)
                    discovery = self.dimension_discovery.run(market_data, feature_data)
                    dimension_analysis = {
                        'discovery_dimensions': discovery.get('dimensions'),
                        'feature_dynamic_matrix_shape': discovery.get('aggregation', {}).get('feature_dynamic_matrix', pd.DataFrame()).shape,
                        'dimension_scores_head': discovery.get('dimension_scores', pd.DataFrame()).head(3).to_dict() if isinstance(discovery.get('dimension_scores'), pd.DataFrame) else {},
                    }
                    # Optional: cluster on dimension scores
                    dim_scores = discovery.get('dimension_scores', None)
                    if isinstance(dim_scores, pd.DataFrame) and not dim_scores.empty:
                        self.logger.info("🔁 Clustering on dimension scores")
                        dim_cluster_results = self.regime_clusterer.run_all_methods(dim_scores.fillna(0).values)
                        best = self.regime_clusterer.get_best_method()
                        if best:
                            best_m, best_r = best
                            dimension_analysis['dimension_scores_clustering'] = {
                                'best_method': best_m.value,
                                'n_clusters': best_r.n_clusters,
                                'metrics': best_r.metrics,
                            }
                except Exception as e:
                    self.logger.warning(f"New discovery pipeline failed ({e}); falling back to legacy analysis")
                    dimension_analysis = await self._run_dimension_analysis(market_data, target)
            
            # Step 4: Run feature importance analysis
            if self.config.analyze_feature_importance and clustering_results:
                self.logger.info("⚖️ Analyzing feature importance")
                feature_importance_results = await self._run_feature_importance_analysis(
                    market_data, clustering_results.get('regime_labels')
                )
            
            # Step 5: Run validation analysis
            if self.config.validate_results:
                self.logger.info("✅ Running validation analysis")
                validation_results = await self._run_validation_analysis(
                    market_data, hmm_results, clustering_results
                )
            
            # Step 6: Run comparative analysis
            if self.config.compare_methods and hmm_results and clustering_results:
                self.logger.info("⚖️ Running comparative analysis")
                comparison_analysis = await self._run_comparative_analysis(
                    market_data, hmm_results, clustering_results
                )
            
            # Step 7: Generate recommendations
            recommendations = self._generate_recommendations(
                hmm_results, clustering_results, dimension_analysis,
                feature_importance_results, validation_results, comparison_analysis
            )
            
            # Create integration result
            result = IntegrationResult(
                method=self.config.method,
                hmm_results=hmm_results,
                clustering_results=clustering_results,
                dimension_analysis=dimension_analysis,
                feature_importance=feature_importance_results,
                validation_metrics=validation_results,
                comparison_analysis=comparison_analysis,
                recommendations=recommendations,
                metadata={
                    'data_shape': market_data.shape,
                    'analysis_timestamp': pd.Timestamp.now().isoformat(),
                    'config': self.config.__dict__
                }
            )
            
            # Save results if requested
            if self.config.save_results:
                await self._save_integration_results(result)
            
            self.logger.info("✅ Integration analysis completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Integration analysis failed: {e}")
            raise
    
    def _should_run_hmm(self) -> bool:
        """Check if HMM analysis should be run."""
        return (
            HMM_AVAILABLE and 
            self.hmm_regime_discovery is not None and
            self.config.method in [
                IntegrationMethod.HMM_ONLY,
                IntegrationMethod.HMM_ENHANCED,
                IntegrationMethod.HYBRID,
                IntegrationMethod.ENSEMBLE,
                IntegrationMethod.COMPARATIVE
            ]
        )
    
    def _should_run_clustering(self) -> bool:
        """Check if clustering analysis should be run."""
        return self.config.method in [
            IntegrationMethod.CLUSTERING_ONLY,
            IntegrationMethod.CLUSTERING_ENHANCED,
            IntegrationMethod.HYBRID,
            IntegrationMethod.ENSEMBLE,
            IntegrationMethod.COMPARATIVE
        ]
    
    async def _run_hmm_analysis(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Run HMM regime discovery analysis."""
        try:
            # Run HMM regime discovery
            hmm_result = await self.hmm_regime_discovery.execute(market_data, {})
            
            if hmm_result.success:
                return {
                    'regime_discovery': hmm_result.artifacts.get('hmm_regime_discovery_result', {}),
                    'performance_metrics': hmm_result.metadata,
                    'method': 'hmm'
                }
            else:
                self.logger.error(f"HMM regime discovery failed: {hmm_result.error_message}")
                return None
                
        except Exception as e:
            self.logger.error(f"HMM analysis failed: {e}")
            return None
    
    async def _run_clustering_analysis(self, 
                                     market_data: pd.DataFrame,
                                     target: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run clustering regime analysis."""
        try:
            # Prepare features from market data
            features = self._prepare_clustering_features(market_data)
            
            # Run clustering analysis
            clustering_results = self.regime_clusterer.run_all_methods(features.values)
            
            # Get best clustering result
            best_method, best_result = self.regime_clusterer.get_best_method()
            
            if best_result:
                return {
                    'best_method': best_method.value,
                    'best_result': best_result.to_dict(),
                    'all_results': {method.value: result.to_dict() for method, result in clustering_results.items()},
                    'regime_labels': best_result.labels,
                    'n_clusters': best_result.n_clusters,
                    'method': 'clustering'
                }
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Clustering analysis failed: {e}")
            return None
    
    async def _run_dimension_analysis(self, 
                                    market_data: pd.DataFrame,
                                    target: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Run market dimension analysis."""
        try:
            # Create target from market data if not provided
            if target is None:
                target = market_data['close'].pct_change().fillna(0).values
            
            # Run dimension analysis
            dimension_results = self.dimension_analyzer.analyze_all_dimensions(market_data, target)
            
            # Get top dimensions
            top_dimensions = self.dimension_analyzer.get_top_dimensions(5)
            
            return {
                'dimension_results': {dim.value: metrics.to_dict() for dim, metrics in dimension_results.items()},
                'top_dimensions': [(dim.value, metrics.to_dict()) for dim, metrics in top_dimensions],
                'analysis_report': self.dimension_analyzer.generate_analysis_report()
            }
            
        except Exception as e:
            self.logger.error(f"Dimension analysis failed: {e}")
            return None
    
    async def _run_feature_importance_analysis(self, 
                                             market_data: pd.DataFrame,
                                             regime_labels: np.ndarray) -> Dict[str, Any]:
        """Run feature importance analysis."""
        try:
            # Prepare features
            features = self._prepare_clustering_features(market_data)
            
            # Run feature importance analysis
            importance_results = self.feature_importance.analyze_all_methods(features, regime_labels)
            
            # Get consensus features
            consensus_features = self.feature_importance.get_consensus_features(10)
            
            return {
                'importance_results': {method.value: result.to_dict() for method, result in importance_results.items()},
                'consensus_features': consensus_features,
                'analysis_report': self.feature_importance.generate_importance_report()
            }
            
        except Exception as e:
            self.logger.error(f"Feature importance analysis failed: {e}")
            return None
    
    async def _run_validation_analysis(self, 
                                     market_data: pd.DataFrame,
                                     hmm_results: Optional[Dict[str, Any]],
                                     clustering_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Run validation analysis."""
        try:
            validation_results = {}
            
            # Validate HMM results
            if hmm_results and 'regime_discovery' in hmm_results:
                regime_assignments = hmm_results['regime_discovery'].get('regime_assignments', [])
                if regime_assignments:
                    hmm_validation = self.validation_metrics.validate_all_metrics(
                        market_data, np.array(regime_assignments)
                    )
                    validation_results['hmm'] = {
                        method.value: result.to_dict() for method, result in hmm_validation.items()
                    }
            
            # Validate clustering results
            if clustering_results and 'regime_labels' in clustering_results:
                regime_labels = clustering_results['regime_labels']
                if regime_labels is not None:
                    clustering_validation = self.validation_metrics.validate_all_metrics(
                        market_data, regime_labels
                    )
                    validation_results['clustering'] = {
                        method.value: result.to_dict() for method, result in clustering_validation.items()
                    }
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Validation analysis failed: {e}")
            return None
    
    async def _run_comparative_analysis(self, 
                                      market_data: pd.DataFrame,
                                      hmm_results: Dict[str, Any],
                                      clustering_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run comparative analysis between HMM and clustering."""
        try:
            comparison = {}
            
            # Extract regime labels
            hmm_labels = np.array(hmm_results['regime_discovery'].get('regime_assignments', []))
            clustering_labels = clustering_results.get('regime_labels')
            
            if len(hmm_labels) > 0 and clustering_labels is not None:
                # Align lengths if necessary
                min_len = min(len(hmm_labels), len(clustering_labels))
                hmm_labels = hmm_labels[:min_len]
                clustering_labels = clustering_labels[:min_len]
                
                # Calculate agreement metrics
                from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
                
                comparison['agreement_metrics'] = {
                    'adjusted_rand_score': float(adjusted_rand_score(hmm_labels, clustering_labels)),
                    'normalized_mutual_info': float(normalized_mutual_info_score(hmm_labels, clustering_labels))
                }
                
                # Compare regime counts
                comparison['regime_counts'] = {
                    'hmm': len(np.unique(hmm_labels)),
                    'clustering': len(np.unique(clustering_labels))
                }
                
                # Compare regime distributions
                hmm_dist = np.bincount(hmm_labels) / len(hmm_labels)
                clustering_dist = np.bincount(clustering_labels) / len(clustering_labels)
                
                comparison['regime_distributions'] = {
                    'hmm': hmm_dist.tolist(),
                    'clustering': clustering_dist.tolist()
                }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Comparative analysis failed: {e}")
            return None
    
    def _prepare_clustering_features(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for clustering analysis."""
        features = pd.DataFrame()
        
        # Basic OHLCV features
        if 'open' in market_data.columns:
            features['open'] = market_data['open']
        if 'high' in market_data.columns:
            features['high'] = market_data['high']
        if 'low' in market_data.columns:
            features['low'] = market_data['low']
        if 'close' in market_data.columns:
            features['close'] = market_data['close']
            # Returns
            features['returns'] = market_data['close'].pct_change()
            # Volatility proxy
            features['volatility'] = features['returns'].rolling(20).std()
        if 'volume' in market_data.columns:
            features['volume'] = market_data['volume']
            features['volume_ma'] = market_data['volume'].rolling(20).mean()
        
        # Technical indicators
        if 'close' in market_data.columns:
            # Moving averages
            features['ma_10'] = market_data['close'].rolling(10).mean()
            features['ma_50'] = market_data['close'].rolling(50).mean()
            
            # RSI-like indicator
            delta = market_data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta).where(delta < 0, 0).rolling(14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs))
        
        return features.fillna(method='ffill').fillna(0)
    
    def _generate_recommendations(self, 
                                hmm_results: Optional[Dict[str, Any]],
                                clustering_results: Optional[Dict[str, Any]],
                                dimension_analysis: Optional[Dict[str, Any]],
                                feature_importance: Optional[Dict[str, Any]],
                                validation_results: Optional[Dict[str, Any]],
                                comparison_analysis: Optional[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # HMM vs Clustering recommendations
        if comparison_analysis and 'agreement_metrics' in comparison_analysis:
            ari = comparison_analysis['agreement_metrics'].get('adjusted_rand_score', 0)
            if ari > 0.5:
                recommendations.append("✅ High agreement between HMM and clustering methods - both approaches are consistent")
            elif ari > 0.3:
                recommendations.append("⚠️ Moderate agreement between methods - consider ensemble approach")
            else:
                recommendations.append("❌ Low agreement between methods - investigate data quality and feature engineering")
        
        # Dimension analysis recommendations
        if dimension_analysis and 'top_dimensions' in dimension_analysis:
            top_dims = dimension_analysis['top_dimensions'][:3]
            dim_names = [dim[0] for dim in top_dims]
            recommendations.append(f"📊 Focus on {', '.join(dim_names)} dimensions for regime identification")
        
        # Feature importance recommendations
        if feature_importance and 'consensus_features' in feature_importance:
            consensus = feature_importance['consensus_features'][:5]
            if consensus:
                feature_names = [f[0] for f in consensus]
                recommendations.append(f"🎯 Key features for regime prediction: {', '.join(feature_names)}")
        
        # Validation recommendations
        if validation_results:
            for method, results in validation_results.items():
                if 'silhouette_score' in results:
                    score = results['silhouette_score'].get('value', 0)
                    if score > 0.5:
                        recommendations.append(f"✅ {method.upper()} shows good regime separation (silhouette: {score:.2f})")
                    else:
                        recommendations.append(f"⚠️ {method.upper()} shows weak regime separation - consider parameter tuning")
        
        # Method-specific recommendations
        if self.config.method == IntegrationMethod.HYBRID:
            recommendations.append("🔄 Consider combining HMM and clustering in ensemble for robust regime identification")
        
        if not recommendations:
            recommendations.append("ℹ️ Run complete analysis to generate specific recommendations")
        
        return recommendations
    
    async def _save_integration_results(self, result: IntegrationResult):
        """Save integration results to files."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save main result
            result_file = output_dir / "integration_result.json"
            with open(result_file, 'w') as f:
                json.dump(result.to_dict(), f, indent=2, default=str)
            
            # Save individual reports if available
            if result.dimension_analysis and 'analysis_report' in result.dimension_analysis:
                report_file = output_dir / "dimension_analysis_report.md"
                with open(report_file, 'w') as f:
                    f.write(result.dimension_analysis['analysis_report'])
            
            if result.feature_importance and 'analysis_report' in result.feature_importance:
                report_file = output_dir / "feature_importance_report.md"
                with open(report_file, 'w') as f:
                    f.write(result.feature_importance['analysis_report'])
            
            # Save validation report
            if result.validation_metrics:
                validation_report = self.validation_metrics.generate_validation_report()
                report_file = output_dir / "validation_report.md"
                with open(report_file, 'w') as f:
                    f.write(validation_report)
            
            self.logger.info(f"💾 Saved integration results to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Failed to save integration results: {e}")
    
    def generate_integration_report(self, result: IntegrationResult) -> str:
        """Generate comprehensive integration report."""
        report = []
        report.append("# HMM-Clustering Integration Analysis Report")
        report.append("=" * 60)
        report.append("")
        
        # Executive Summary
        report.append("## Executive Summary")
        report.append("")
        report.append(f"**Integration Method**: {result.method.value}")
        report.append(f"**Analysis Date**: {result.metadata.get('analysis_timestamp', 'N/A')}")
        report.append(f"**Data Shape**: {result.metadata.get('data_shape', 'N/A')}")
        report.append("")
        
        # Key Recommendations
        if result.recommendations:
            report.append("## Key Recommendations")
            report.append("")
            for i, rec in enumerate(result.recommendations, 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # Method Comparison
        if result.comparison_analysis:
            report.append("## Method Comparison")
            report.append("")
            
            if 'agreement_metrics' in result.comparison_analysis:
                metrics = result.comparison_analysis['agreement_metrics']
                report.append("**Agreement Between Methods:**")
                for metric, value in metrics.items():
                    report.append(f"- {metric.replace('_', ' ').title()}: {value:.3f}")
                report.append("")
            
            if 'regime_counts' in result.comparison_analysis:
                counts = result.comparison_analysis['regime_counts']
                report.append("**Regime Counts:**")
                for method, count in counts.items():
                    report.append(f"- {method.upper()}: {count} regimes")
                report.append("")
        
        # Results Summary
        if result.hmm_results:
            report.append("## HMM Analysis Results")
            report.append("")
            regime_discovery = result.hmm_results.get('regime_discovery', {})
            if regime_discovery:
                report.append(f"- **Regimes Discovered**: {regime_discovery.get('regime_metrics', {}).get('total_regimes', 'N/A')}")
                report.append(f"- **Total Samples**: {regime_discovery.get('regime_metrics', {}).get('total_samples', 'N/A')}")
                report.append("")
        
        if result.clustering_results:
            report.append("## Clustering Analysis Results")
            report.append("")
            report.append(f"- **Best Method**: {result.clustering_results.get('best_method', 'N/A')}")
            report.append(f"- **Clusters Found**: {result.clustering_results.get('n_clusters', 'N/A')}")
            
            best_result = result.clustering_results.get('best_result', {})
            if 'metrics' in best_result:
                metrics = best_result['metrics']
                if 'silhouette_score' in metrics:
                    report.append(f"- **Silhouette Score**: {metrics['silhouette_score']:.3f}")
            report.append("")
        
        # Dimension Analysis
        if result.dimension_analysis and 'top_dimensions' in result.dimension_analysis:
            report.append("## Top Market Dimensions")
            report.append("")
            
            for i, (dim_name, dim_data) in enumerate(result.dimension_analysis['top_dimensions'][:5], 1):
                composite_score = dim_data.get('metrics', {}).get('composite_score', 0)
                report.append(f"{i}. **{dim_name.upper()}** - Composite Score: {composite_score:.3f}")
            report.append("")
        
        # Feature Importance
        if result.feature_importance and 'consensus_features' in result.feature_importance:
            report.append("## Top Important Features")
            report.append("")
            
            for i, (feature, score, n_methods) in enumerate(result.feature_importance['consensus_features'][:10], 1):
                report.append(f"{i:2d}. **{feature}** - Score: {score:.3f} (agreed by {n_methods} methods)")
            report.append("")
        
        # Validation Summary
        if result.validation_metrics:
            report.append("## Validation Summary")
            report.append("")
            
            for method, validation in result.validation_metrics.items():
                report.append(f"### {method.upper()} Validation")
                
                # Show key validation metrics
                key_metrics = ['silhouette_score', 'temporal_consistency', 'return_separability']
                for metric in key_metrics:
                    if metric in validation:
                        metric_data = validation[metric]
                        value = metric_data.get('value', 0)
                        interpretation = metric_data.get('interpretation', 'N/A')
                        report.append(f"- **{metric.replace('_', ' ').title()}**: {value:.3f} - {interpretation}")
                
                report.append("")
        
        # Next Steps
        report.append("## Recommended Next Steps")
        report.append("")
        report.append("1. **Model Training**: Use identified regimes for ML model training")
        report.append("2. **Feature Engineering**: Focus on top-performing dimensions and features")
        report.append("3. **Validation**: Implement regime-specific validation in trading systems")
        report.append("4. **Monitoring**: Set up regime change detection for live trading")
        report.append("5. **Optimization**: Fine-tune parameters based on validation results")
        report.append("")
        
        return "\n".join(report)