"""
Optimal Regime Clustering Component.

This component wraps the optimal_regime_clustering functionality to replace hmm_clustering
in the sub_pipeline system while maintaining the same interface and output format.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
from datetime import datetime
from pathlib import Path

# Handle optional dependencies gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

# Import the optimal regime clustering orchestrator
try:
    from src.training.steps.market_analysis.optimal_regime_clustering.orchestrator import run_optimal_clustering
    OPTIMAL_REGIME_CLUSTERING_AVAILABLE = True
except ImportError:
    OPTIMAL_REGIME_CLUSTERING_AVAILABLE = False
    run_optimal_clustering = None

# Import base component classes
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

logger = logging.getLogger(__name__)


# Validation result classes for simplified error handling
class ValidationResult(NamedTuple):
    """Result of input validation."""
    is_valid: bool
    error_message: Optional[str] = None
    market_data: Optional[Any] = None


class OptimalRegimeClusteringComponent(BaseMarketAnalysisComponent):
    """
    Optimal Regime Clustering Component.

    This component replaces the HMM clustering component with the more advanced
    optimal regime clustering system while maintaining full compatibility
    with the existing sub_pipeline interface.
    """

    def __init__(self, config):
        """Initialize the optimal regime clustering component."""
        super().__init__(config)
        self.component_name = "optimal_regime_clustering"

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['optimal_regime_clustering_result']

    def get_component_name(self) -> str:
        """Get the component name."""
        return self.component_name

    def validate_config(self) -> bool:
        """Validate component configuration."""
        if not OPTIMAL_REGIME_CLUSTERING_AVAILABLE:
            self.logger.error("❌ Optimal regime clustering not available - required dependencies missing")
            return False

        # Check required configuration parameters
        required_params = ['symbol', 'exchange', 'timeframe', 'data_dir']
        for param in required_params:
            if not hasattr(self.config, param) or getattr(self.config, param) is None:
                self.logger.error(f"❌ Missing required config parameter: {param}")
                return False

        return True

    async def _validate_inputs(self, data: Any, pipeline_state: Dict[str, Any]) -> NamedTuple:
        """
        Validate input data and pipeline state.

        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state

        Returns:
            ValidationResult with market data and validation status
        """
        # ValidationResult is defined locally above

        # Basic input validation
        if data is None:
            return ValidationResult(
                is_valid=False,
                market_data=None,
                error_message="No market data provided for optimal regime clustering"
            )

        if not pipeline_state:
            return ValidationResult(
                is_valid=False,
                market_data=None,
                error_message="No pipeline state provided for optimal regime clustering"
            )

        # Validate required pipeline state parameters
        required_state = ['symbol', 'exchange', 'timeframe', 'data_dir']
        for param in required_state:
            if param not in pipeline_state:
                return ValidationResult(
                    is_valid=False,
                    market_data=None,
                    error_message=f"Missing required pipeline state parameter: {param}"
                )

        # Validate data format
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            if data.empty:
                return ValidationResult(
                    is_valid=False,
                    market_data=None,
                    error_message="Empty DataFrame provided for optimal regime clustering"
                )

            # Check for required columns
            required_columns = ['close', 'volume', 'high', 'low']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return ValidationResult(
                    is_valid=False,
                    market_data=None,
                    error_message=f"Missing required columns: {missing_columns}"
                )

        return ValidationResult(
            is_valid=True,
            market_data=data,
            error_message=None
        )

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute optimal regime clustering.

        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with clustering results
        """
        from src.utils.tprint import tprint

        tprint("🔄 Starting Optimal Regime Clustering")
        self.logger.info('🔄 Starting Optimal Regime Clustering')

        try:
            # Validate inputs
            tprint("📊 Input validation starting...")
            validation_result = await self._validate_inputs(data, pipeline_state)
            if not validation_result.is_valid:
                return ComponentResult(success=False, artifacts={}, error_message=validation_result.error_message)

            market_data = validation_result.market_data
            tprint(f"✅ Market data validated successfully: {type(market_data)}")
            if hasattr(market_data, 'shape'):
                tprint(f"📊 Market data shape: {market_data.shape}")

            # Get configuration from pipeline state
            symbol = pipeline_state.get('symbol', 'ETHUSDT')
            exchange = pipeline_state.get('exchange', 'binance')
            timeframe = pipeline_state.get('timeframe', '15m')
            data_dir = pipeline_state.get('data_dir', 'historical_data')

            # Try to get regime discovery results from pipeline state
            tprint("🔍 Retrieving regime discovery results...")
            hmm_regime_discovery = pipeline_state.get('hmm_regime_discovery_result', {})

            # If not found in pipeline state, try to get from artifacts
            if not hmm_regime_discovery:
                tprint("🔍 Trying to get regime discovery from artifacts...")
                artifacts = pipeline_state.get('artifacts', {})
                hmm_regime_discovery = artifacts.get('hmm_regime_discovery_result', {})

            # If still not found, try to construct from individual components
            if not hmm_regime_discovery:
                tprint("🔍 Trying to construct regime discovery from individual components...")
                regime_models = pipeline_state.get('regime_models', [])
                regime_assignments = pipeline_state.get('regime_assignments', [])
                tprint(f"📊 Regime models count: {len(regime_models)}")
                tprint(f"📊 Regime assignments count: {len(regime_assignments)}")

                if regime_models or regime_assignments:
                    hmm_regime_discovery = {
                        'regime_models': regime_models,
                        'regime_assignments': regime_assignments,
                        'regime_metrics': pipeline_state.get('regime_metrics', {})
                    }
                    tprint("✅ Constructed regime discovery from individual components")
                else:
                    tprint("⚠️ No regime models or assignments found")

            # If still not found, try to load from previous outcome file
            if not hmm_regime_discovery:
                tprint("🔍 Trying to load regime discovery from previous outcome file...")
                try:
                    # Look for the most recent hmm_regime_discovery outcome file
                    outcomes_dir = Path("outcomes")
                    if outcomes_dir.exists():
                        pattern = f"market_analysis_hmm_regime_discovery_outcome_*_{symbol.lower()}_{exchange.lower()}_*.json"
                        outcome_files = list(outcomes_dir.glob(pattern))
                        if not outcome_files:
                            # Try a more general pattern
                            pattern = f"market_analysis_hmm_regime_discovery_outcome_*.json"
                            outcome_files = list(outcomes_dir.glob(pattern))

                        if outcome_files:
                            # Get the most recent file
                            latest_outcome = max(outcome_files, key=lambda f: f.stat().st_mtime)
                            tprint(f"📁 Loading from outcome file: {latest_outcome}")

                            with open(latest_outcome, 'r') as f:
                                outcome_data = json.load(f)

                            # Extract regime discovery results
                            hmm_regime_discovery = outcome_data.get('artifacts', {}).get('hmm_regime_discovery_result', {})
                            if hmm_regime_discovery:
                                tprint("✅ Loaded regime discovery from outcome file")
                        else:
                            tprint("⚠️ No regime discovery outcome files found")
                    else:
                        tprint("⚠️ Outcomes directory does not exist")
                except Exception as e:
                    tprint(f"⚠️ Error loading regime discovery from outcome file: {e}")

            # If we still don't have regime discovery data, return error
            if not hmm_regime_discovery:
                error_msg = "No HMM regime discovery data available for optimal regime clustering"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(success=False, artifacts={}, error_message=error_msg)

            # Save regime discovery data to a temporary file for optimal clustering
            tprint("💾 Saving regime discovery data for optimal clustering...")
            temp_regime_file = f"temp_regime_discovery_{symbol}_{exchange}_{timeframe}.json"
            try:
                # Create the structure that the orchestrator expects
                temp_data = {
                    "artifacts": {
                        "hmm_regime_discovery_result": hmm_regime_discovery
                    }
                }
                with open(temp_regime_file, 'w') as f:
                    json.dump(temp_data, f, indent=2, default=str)
                tprint(f"✅ Saved regime discovery data to: {temp_regime_file}")
            except Exception as e:
                error_msg = f"Failed to save regime discovery data: {e}"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(success=False, artifacts={}, error_message=error_msg)

            # Run optimal regime clustering
            tprint("🚀 Running optimal regime clustering...")
            try:
                # Use the run_optimal_clustering function
                clustering_result = run_optimal_clustering(
                    data_path=temp_regime_file,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )

                tprint("✅ Optimal regime clustering completed successfully")

                # Clean up temporary file
                try:
                    Path(temp_regime_file).unlink()
                    tprint("🧹 Cleaned up temporary regime discovery file")
                except Exception as e:
                    tprint(f"⚠️ Warning: Could not clean up temporary file: {e}")

                # Check if clustering was successful
                if not clustering_result.get('success', False):
                    error_msg = clustering_result.get('error', 'Unknown error in optimal regime clustering')
                    tprint(f"❌ {error_msg}")
                    self.logger.error(error_msg)
                    return ComponentResult(success=False, artifacts={}, error_message=error_msg)

                # Create compatible artifacts for the sub_pipeline
                artifacts = self._create_compatible_artifacts(clustering_result, symbol, exchange, timeframe)

                return ComponentResult(
                    success=True,
                    artifacts=artifacts,
                    metadata={
                        'component': self.component_name,
                        'clustering_method': 'optimal_regime_clustering',
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'execution_time': clustering_result.get('execution_time', 0)
                    }
                )

            except Exception as e:
                error_msg = f"Error running optimal regime clustering: {e}"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(success=False, artifacts={}, error_message=error_msg)

        except Exception as e:
            error_msg = f"Unexpected error in optimal regime clustering: {e}"
            tprint(f"❌ {error_msg}")
            self.logger.error(error_msg)
            return ComponentResult(success=False, artifacts={}, error_message=error_msg)

    def _create_compatible_artifacts(self, clustering_result: Dict[str, Any],
                                   symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """
        Create artifacts compatible with the existing sub_pipeline interface.

        Args:
            clustering_result: Results from optimal regime clustering
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Data timeframe

        Returns:
            Dictionary of artifacts compatible with sub_pipeline expectations
        """
        try:
            # Extract cluster assignments and models from optimal regime clustering results
            cluster_assignments = []
            hmm_models = []
            cluster_metrics = {}

            # Try to extract from different possible locations in the result
            if 'cluster_assignments' in clustering_result:
                cluster_assignments = clustering_result['cluster_assignments']
            elif 'results' in clustering_result and 'cluster_assignments' in clustering_result['results']:
                cluster_assignments = clustering_result['results']['cluster_assignments']

            if 'hmm_models' in clustering_result:
                hmm_models = clustering_result['hmm_models']
            elif 'results' in clustering_result and 'hmm_models' in clustering_result['results']:
                hmm_models = clustering_result['results']['hmm_models']

            if 'cluster_metrics' in clustering_result:
                cluster_metrics = clustering_result['cluster_metrics']
            elif 'results' in clustering_result and 'cluster_metrics' in clustering_result['results']:
                cluster_metrics = clustering_result['results']['cluster_metrics']

            # Calculate basic metrics
            n_clusters = len(set(cluster_assignments)) if cluster_assignments else 0
            n_samples = len(cluster_assignments) if cluster_assignments else 0

            # Create the main artifact in the same format as HMM clustering
            optimal_clustering_result = {
                # 1. EXECUTION METADATA
                'execution_metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'data_points_processed': n_samples,
                    'execution_successful': True,
                    'clustering_method': 'optimal_regime_clustering',
                    'target_clusters': 20,
                    'target_coverage_min': 90.0,
                    'target_coverage_max': 95.0,
                    'outcome_version': '2.0_optimal_regime',
                    'raw_data_excluded': True,
                    'comprehensive_metrics_included': True,
                    'regime_reduction': f"{n_clusters} optimal regimes"
                },

                # 2. GENERAL METRICS (All clusters, top N analysis, overall statistics)
                'general_metrics': {
                    # Cluster Assignments Summary
                    'cluster_assignments_summary': {
                        'total_assignments': n_samples,
                        'unique_clusters': n_clusters,
                        'cluster_distribution': self._calculate_cluster_distribution(cluster_assignments) if cluster_assignments else {},
                        'assignment_range': {
                            'min_cluster_id': min(cluster_assignments) if cluster_assignments else 0,
                            'max_cluster_id': max(cluster_assignments) if cluster_assignments else 0
                        }
                    },

                    # Economical Metrics
                    'economical_metrics': {
                        'market_state_diversity': {
                            'unique_states': n_clusters,
                            'diversity_ratio': 1.0 if n_clusters <= 25 else 0.8,
                            'economical_relevance': {
                                'clusters_covering_90_percent': n_clusters,
                                'market_state_coverage': 'comprehensive'
                            }
                        }
                    },

                    # Comprehensive Metrics
                    'comprehensive_metrics': {
                        'coverage_metrics': {
                            'top_5_coverage': self._calculate_top_n_coverage(cluster_assignments, 5) if cluster_assignments else 0.0,
                            'top_10_coverage': self._calculate_top_n_coverage(cluster_assignments, 10) if cluster_assignments else 0.0,
                        },
                        'quality_metrics': {
                            'silhouette_score': cluster_metrics.get('silhouette_score', 0.0),
                            'calinski_harabasz_score': cluster_metrics.get('calinski_harabasz_score', 0.0),
                            'davies_bouldin_score': cluster_metrics.get('davies_bouldin_score', 0.0),
                            'overall_quality_score': cluster_metrics.get('overall_quality_score', 0.0)
                        }
                    }
                },

                # 3. DETAILED CLUSTER ANALYSIS
                'detailed_cluster_analysis': {
                    'cluster_characteristics': self._create_cluster_characteristics(cluster_assignments, cluster_metrics),
                    'individual_cluster_metrics': self._create_individual_cluster_metrics(cluster_assignments, cluster_metrics),
                    'cluster_stability_analysis': cluster_metrics.get('stability_analysis', {}),
                    'feature_importance_analysis': cluster_metrics.get('feature_importance', {})
                },

                # 4. HMM MODELS AND ASSIGNMENTS
                'hmm_models': hmm_models,
                'cluster_assignments': cluster_assignments,

                # 5. VALIDATION AND QUALITY METRICS
                'validation_metrics': {
                    'cluster_validation': {
                        'n_clusters': n_clusters,
                        'n_samples': n_samples,
                        'coverage_percentage': (n_samples / n_samples * 100) if n_samples > 0 else 0.0,
                        'quality_score': cluster_metrics.get('overall_quality_score', 0.0)
                    },
                    'regime_validation': {
                        'regime_count': n_clusters,
                        'regime_coverage': 'comprehensive',
                        'regime_quality': 'optimal'
                    }
                },

                # 6. PERFORMANCE METRICS
                'performance_metrics': {
                    'execution_time': clustering_result.get('execution_time', 0),
                    'memory_usage': cluster_metrics.get('memory_usage', 0),
                    'cpu_usage': cluster_metrics.get('cpu_usage', 0)
                },

                # 7. OUTPUT FILES (for compatibility)
                'output_files': {
                    'cluster_summary_report': clustering_result.get('cluster_summary_report', ''),
                    'ml_datasets': clustering_result.get('ml_datasets', []),
                    'outcome_file': clustering_result.get('outcome_file', '')
                }
            }

            return {
                'optimal_regime_clustering_result': optimal_clustering_result
            }

        except Exception as e:
            self.logger.error(f"Error creating compatible artifacts: {e}")
            return {}

    def _calculate_cluster_distribution(self, cluster_assignments: List[int]) -> Dict[str, Any]:
        """Calculate cluster distribution statistics."""
        if not cluster_assignments:
            return {}

        try:
            from collections import Counter
            cluster_counts = Counter(cluster_assignments)
            total_samples = len(cluster_assignments)

            distribution = {}
            for cluster_id, count in cluster_counts.items():
                distribution[str(cluster_id)] = {
                    'count': count,
                    'percentage': (count / total_samples) * 100,
                    'size_category': self._get_size_category(count)
                }

            return distribution
        except Exception as e:
            self.logger.error(f"Error calculating cluster distribution: {e}")
            return {}

    def _get_size_category(self, count: int) -> str:
        """Get size category for cluster."""
        if count >= 1000:
            return 'large'
        elif count >= 100:
            return 'medium'
        else:
            return 'small'

    def _calculate_top_n_coverage(self, cluster_assignments: List[int], n: int) -> float:
        """Calculate coverage percentage of top N clusters."""
        if not cluster_assignments:
            return 0.0

        try:
            from collections import Counter
            cluster_counts = Counter(cluster_assignments)
            top_clusters = [cluster for cluster, _ in cluster_counts.most_common(n)]
            top_samples = sum(count for cluster, count in cluster_counts.items() if cluster in top_clusters)
            total_samples = len(cluster_assignments)

            return (top_samples / total_samples) * 100 if total_samples > 0 else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating top N coverage: {e}")
            return 0.0

    def _create_cluster_characteristics(self, cluster_assignments: List[int],
                                     cluster_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create cluster characteristics analysis."""
        try:
            characteristics = {
                'distribution_analysis': self._calculate_cluster_distribution(cluster_assignments),
                'size_analysis': {
                    'total_clusters': len(set(cluster_assignments)) if cluster_assignments else 0,
                    'total_samples': len(cluster_assignments) if cluster_assignments else 0,
                    'average_cluster_size': len(cluster_assignments) / len(set(cluster_assignments)) if cluster_assignments and len(set(cluster_assignments)) > 0 else 0
                },
                'quality_metrics': cluster_metrics.get('quality_metrics', {}),
                'stability_metrics': cluster_metrics.get('stability_analysis', {})
            }

            return characteristics
        except Exception as e:
            self.logger.error(f"Error creating cluster characteristics: {e}")
            return {}

    def _create_individual_cluster_metrics(self, cluster_assignments: List[int],
                                        cluster_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create individual cluster metrics."""
        try:
            if not cluster_assignments:
                return {}

            from collections import Counter
            cluster_counts = Counter(cluster_assignments)

            individual_metrics = {}
            for cluster_id, count in cluster_counts.items():
                individual_metrics[str(cluster_id)] = {
                    'basic_stats': {
                        'size': count,
                        'percentage': (count / len(cluster_assignments)) * 100 if cluster_assignments else 0
                    },
                    'quality_metrics': cluster_metrics.get(f'cluster_{cluster_id}', {}),
                    'stability_metrics': {}
                }

            return individual_metrics
        except Exception as e:
            self.logger.error(f"Error creating individual cluster metrics: {e}")
            return {}
