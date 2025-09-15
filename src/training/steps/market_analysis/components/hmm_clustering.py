"""
HMM Clustering Component.

This component performs HMM-based regime clustering.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
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


# Hardware optimization imports
try:
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    from src.utils.matrix_operations import MatrixOperations
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    M1MemoryOptimizer = None
    M1CPUOptimizer = None
    MatrixOperations = None

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger


class HMMClusteringComponent(BaseMarketAnalysisComponent):
    """
    HMM Clustering Component.
    
    Performs HMM-based regime clustering.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the HMM clustering component."""
        super().__init__(config)
        self.logger = system_logger.getChild('HMMClustering')
        
        # Initialize hardware optimization components
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            self.memory_optimizer = M1MemoryOptimizer()
            self.cpu_optimizer = M1CPUOptimizer()
            self.matrix_ops = MatrixOperations()
        else:
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.matrix_ops = None
            self.logger.warning("⚠️ Hardware optimization not available - using fallback methods")
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['hmm_clustering_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute HMM clustering.
        
        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with clustering results
        """
        self.logger.info('🔄 Starting HMM Clustering')
        
        try:
            # Import HMM clustering utilities
            from src.utils.hmm_composite_manager import EnhancedHMMCompositeManager
            
            # Get market data
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for HMM clustering")
            
            # Get regime discovery results from previous stage
            hmm_regime_discovery = pipeline_state.get('hmm_regime_discovery_result', {})
            if not hmm_regime_discovery:
                raise ValueError("No HMM regime discovery results available for clustering")
            
            input_regimes = len(hmm_regime_discovery.get('regime_models', []))
            self.logger.info(f'🔧 HMM Clustering: Processing {input_regimes} regimes → 3 clusters (Bull/Bear/Sideways)')
            
            # Configure HMM clustering
            clustering_config = {
                'n_clusters': 3,  # Bull, Bear, Sideways
                'clustering_method': 'hmm_based',
                'min_cluster_size': 10,
                'convergence_tolerance': 1e-6,
                'max_iterations': 100,
                
                # Regime constraints
                'max_regimes': 25,  # Maximum 25 clusters allowed (regimes are clustered into fewer groups)
                'min_regime_sample_percentage': 0.01,  # 1% minimum sample threshold
                
                # Hardware optimization
                'enable_parallel_processing': True,
                'enable_gpu_acceleration': True,
                'memory_limit_gb': 8.0
            }
            
            # Create HMM composite manager
            hmm_manager = EnhancedHMMCompositeManager()
            
            # Perform HMM clustering
            clustering_result = await self._perform_hmm_clustering(
                hmm_manager, market_data, hmm_regime_discovery, clustering_config
            )
            
            # Extract results
            hmm_models = clustering_result.get('hmm_models', [])
            cluster_assignments = clustering_result.get('cluster_assignments', [])
            cluster_metrics = clustering_result.get('cluster_metrics', {})
            
            # Validate that we have clustering results
            if not hmm_models or not cluster_assignments:
                raise ValueError("HMM clustering completed but no clusters were created")
            
            # Apply regime constraints
            validated_result = self._apply_regime_constraints(
                hmm_models, cluster_assignments, clustering_config
            )
            hmm_models = validated_result['hmm_models']
            cluster_assignments = validated_result['cluster_assignments']
            
            # Perform comprehensive cluster quality validation
            quality_metrics = self._validate_cluster_quality(
                hmm_models, cluster_assignments, market_data, clustering_config
            )
            
            # Generate detailed metrics for each HMM cluster
            cluster_detailed_metrics = self._generate_cluster_detailed_metrics(
                hmm_models, cluster_assignments, market_data, clustering_config
            )
            
            # Create single consolidated artifact
            artifacts = {
                'hmm_clustering_result': {
                    'hmm_models': hmm_models,
                    'cluster_assignments': cluster_assignments,
                    'cluster_metrics': cluster_metrics,
                    'cluster_quality_metrics': quality_metrics,
                    'cluster_detailed_metrics': cluster_detailed_metrics,
                    'clustering_summary': {
                        'total_clusters': len(hmm_models),
                        'total_assignments': len(cluster_assignments),
                        'cluster_distribution': self._calculate_cluster_distribution(cluster_assignments),
                        'clustering_time': clustering_result.get('clustering_time', 0.0),
                        'quality_score': quality_metrics.get('overall_quality_score', 0.0),
                        'validation_passed': quality_metrics.get('validation_passed', False),
                        'regime_reduction': {
                            'input_regimes': len(hmm_regime_discovery.get('regime_models', [])),
                            'output_clusters': len(hmm_models),
                            'reduction_ratio': len(hmm_models) / max(1, len(hmm_regime_discovery.get('regime_models', [])))
                        }
                    },
                    'metadata': {
                        'symbol': self.config.symbol,
                        'exchange': self.config.exchange,
                        'timeframe': self.config.timeframe,
                        'data_points': len(market_data) if market_data is not None else 0,
                        'execution_timestamp': datetime.now().isoformat(),
                        'clustering_info': {
                            'input_regimes': len(hmm_regime_discovery.get('regime_models', [])),
                            'output_clusters': len(hmm_models),
                            'max_regimes_supported': 150,
                            'max_clusters_allowed': 25
                        }
                    }
                }
            }
            
            self.logger.info(f'✅ HMM Clustering completed: {len(hmm_models)} clusters created (from up to 150 regimes)')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'cluster_count': len(hmm_models),
                    'regime_to_cluster_reduction': f"{len(hmm_regime_discovery.get('regime_models', []))} → {len(hmm_models)}"
                }
            )
            
        except Exception as e:
            self.logger.error(f'❌ HMM Clustering failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
            )
    
    async def _load_market_data(self, data: Any) -> Optional[Any]:
        """Load and prepare market data for clustering."""
        if data is None:
            return None
        
        if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
            return data.copy()
        
        # Handle other data types if needed
        return data
    
    async def _perform_hmm_clustering(
        self, 
        hmm_manager: Any, 
        market_data: Any, 
        regime_discovery: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform the actual HMM clustering process with hardware optimization."""
        start_time = time.time()
        
        try:
            # Prepare data for clustering with memory optimization
            prepared_data = await self._prepare_data_for_clustering_optimized(market_data, regime_discovery, config)
            
            # Perform HMM clustering with hardware optimization
            if self.cpu_optimizer and config.get('enable_parallel_processing', True):
                clustering_result = await self._perform_parallel_hmm_clustering(
                    hmm_manager, prepared_data, config
                )
            else:
                clustering_result = await hmm_manager.perform_hmm_clustering(prepared_data, config)
            
            clustering_time = time.time() - start_time
            clustering_result['clustering_time'] = clustering_time
            
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"HMM clustering process failed: {e}")
            # Return fallback clustering result
            return {
                'hmm_models': [],
                'cluster_assignments': [],
                'cluster_metrics': {
                    'clustering_method': 'fallback',
                    'error': str(e)
                },
                'clustering_time': time.time() - start_time
            }
    
    def _prepare_data_for_clustering(self, data: Any, regime_discovery: Dict[str, Any]) -> Any:
        """Prepare market data and regime discovery results for clustering."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            self.logger.warning("Pandas not available or data is not a DataFrame, using fallback")
            return {
                'market_data': data,
                'regime_discovery': regime_discovery
            }
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for clustering: {missing_columns}")
            # Use available columns or create fallback data
            for col in missing_columns:
                if col == 'volume':
                    data[col] = 1000  # Default volume
                else:
                    data[col] = data.get('close', 100.0)  # Use close price as fallback
        
        return {
            'market_data': data,
            'regime_discovery': regime_discovery
        }
    
    def _calculate_cluster_distribution(self, cluster_assignments: List[int]) -> Dict[str, float]:
        """Calculate the distribution of cluster assignments."""
        if not cluster_assignments:
            return {}
        
        total_assignments = len(cluster_assignments)
        cluster_counts = {}
        
        for assignment in cluster_assignments:
            cluster_counts[assignment] = cluster_counts.get(assignment, 0) + 1
        
        # Convert to percentages
        cluster_distribution = {}
        for cluster, count in cluster_counts.items():
            cluster_distribution[f'cluster_{cluster}'] = (count / total_assignments) * 100
        
        return cluster_distribution
    
    def _apply_regime_constraints(
        self, 
        hmm_models: List[Any], 
        cluster_assignments: List[int], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply cluster constraints: max 25 clusters and 1% sample threshold."""
        max_regimes = config.get('max_regimes', 25)
        min_sample_percentage = config.get('min_regime_sample_percentage', 0.01)
        
        if not cluster_assignments:
            return {'hmm_models': hmm_models, 'cluster_assignments': cluster_assignments}
        
        total_samples = len(cluster_assignments)
        min_samples = int(total_samples * min_sample_percentage)
        
        # Count samples per cluster
        cluster_counts = {}
        for assignment in cluster_assignments:
            cluster_counts[assignment] = cluster_counts.get(assignment, 0) + 1
        
        # Filter clusters that meet the minimum sample threshold
        valid_clusters = []
        for cluster, count in cluster_counts.items():
            if count >= min_samples:
                valid_clusters.append(cluster)
            else:
                self.logger.warning(f"⚠️ Cluster {cluster} has {count} samples ({count/total_samples:.2%}), below 1% threshold - removing")
        
        # Limit to max_clusters
        if len(valid_clusters) > max_regimes:
            # Keep the clusters with the most samples
            cluster_counts_sorted = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
            valid_clusters = [cluster for cluster, _ in cluster_counts_sorted[:max_regimes]]
            self.logger.warning(f"⚠️ Limiting to {max_regimes} clusters (had {len(cluster_counts)} clusters)")
        
        # Filter cluster assignments to only include valid clusters
        filtered_assignments = []
        for assignment in cluster_assignments:
            if assignment in valid_clusters:
                filtered_assignments.append(assignment)
            else:
                # Assign to the most common valid cluster as fallback
                if valid_clusters:
                    filtered_assignments.append(valid_clusters[0])
                else:
                    filtered_assignments.append(0)  # Fallback to cluster 0
        
        # Filter HMM models to match valid clusters
        filtered_models = []
        for i, cluster in enumerate(valid_clusters):
            if i < len(hmm_models):
                filtered_models.append(hmm_models[i])
        
        self.logger.info(f"✅ Applied cluster constraints: {len(valid_clusters)} valid clusters (min {min_samples} samples each, max {max_regimes} clusters)")
        
        return {
            'hmm_models': filtered_models,
            'cluster_assignments': filtered_assignments
        }
    
    def _validate_cluster_quality(
        self, 
        hmm_models: List[Any], 
        cluster_assignments: List[int], 
        market_data: Any,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive cluster quality validation."""
        start_time = time.time()
        
        try:
            quality_metrics = {}
            
            # 1. Cluster Persistence Analysis
            persistence_metrics = self._calculate_cluster_persistence(cluster_assignments)
            quality_metrics['persistence_analysis'] = persistence_metrics
            
            # 2. Economic Significance Validation
            if PANDAS_AVAILABLE and market_data is not None:
                economic_metrics = self._validate_cluster_economic_significance(
                    hmm_models, cluster_assignments, market_data
                )
                quality_metrics['economic_significance'] = economic_metrics
            
            # 3. Cross-validation Stability
            stability_metrics = self._cross_validate_clusters(
                hmm_models, cluster_assignments, market_data
            )
            quality_metrics['stability_analysis'] = stability_metrics
            
            # 4. Cluster Transition Analysis
            transition_metrics = self._analyze_cluster_transitions(cluster_assignments)
            quality_metrics['transition_analysis'] = transition_metrics
            
            # 5. Multi-stage Validation Gates
            validation_gates = self._apply_quality_gates(
                persistence_metrics, economic_metrics, stability_metrics, transition_metrics
            )
            quality_metrics['validation_gates'] = validation_gates
            
            # 6. Overall Quality Score
            overall_score = self._calculate_overall_quality_score(quality_metrics)
            quality_metrics['overall_quality_score'] = overall_score
            quality_metrics['validation_passed'] = overall_score >= 0.7  # 70% threshold
            
            # 7. Quality Recommendations
            recommendations = self._generate_quality_recommendations(quality_metrics)
            quality_metrics['recommendations'] = recommendations
            
            validation_time = time.time() - start_time
            quality_metrics['validation_time'] = validation_time
            
            self.logger.info(f"✅ Cluster quality validation completed in {validation_time:.2f}s")
            self.logger.info(f"📊 Overall quality score: {overall_score:.2f} ({'PASSED' if quality_metrics['validation_passed'] else 'FAILED'})")
            self.logger.info(f"📈 Regime range: 2-150 → Clusters: {len(hmm_models)}")
            self.logger.info(f"📋 Detailed cluster metrics generated for {len(hmm_models)} clusters")
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Cluster quality validation failed: {e}")
            return {
                'overall_quality_score': 0.0,
                'validation_passed': False,
                'error': str(e),
                'validation_time': time.time() - start_time
            }
    
    def _calculate_cluster_persistence(self, cluster_assignments: List[int]) -> Dict[str, Any]:
        """Calculate cluster persistence metrics."""
        if not cluster_assignments or len(cluster_assignments) < 2:
            return {'error': 'Insufficient data for persistence analysis'}
        
        # Calculate cluster durations
        cluster_durations = []
        current_cluster = cluster_assignments[0]
        current_duration = 1
        
        for i in range(1, len(cluster_assignments)):
            if cluster_assignments[i] == current_cluster:
                current_duration += 1
            else:
                cluster_durations.append(current_duration)
                current_cluster = cluster_assignments[i]
                current_duration = 1
        
        # Add the last cluster duration
        cluster_durations.append(current_duration)
        
        if not cluster_durations:
            return {'error': 'No cluster durations calculated'}
        
        # Calculate persistence metrics
        avg_duration = np.mean(cluster_durations) if NUMPY_AVAILABLE else sum(cluster_durations) / len(cluster_durations)
        median_duration = np.median(cluster_durations) if NUMPY_AVAILABLE else sorted(cluster_durations)[len(cluster_durations)//2]
        std_duration = np.std(cluster_durations) if NUMPY_AVAILABLE else 0
        
        # Calculate cluster stability (lower std = more stable)
        stability_score = max(0, 1 - (std_duration / avg_duration)) if avg_duration > 0 else 0
        
        return {
            'avg_duration': avg_duration,
            'median_duration': median_duration,
            'std_duration': std_duration,
            'stability_score': stability_score,
            'total_transitions': len(cluster_durations) - 1,
            'cluster_durations': cluster_durations
        }
    
    def _validate_cluster_economic_significance(
        self, 
        hmm_models: List[Any], 
        cluster_assignments: List[int], 
        market_data: Any
    ) -> Dict[str, Any]:
        """Validate economic significance of clusters."""
        if not PANDAS_AVAILABLE or not isinstance(market_data, pd.DataFrame):
            return {'error': 'Pandas not available or invalid market data'}
        
        try:
            # Calculate returns for each cluster
            cluster_returns = {}
            cluster_volatilities = {}
            
            for cluster in set(cluster_assignments):
                cluster_mask = np.array(cluster_assignments) == cluster
                cluster_data = market_data[cluster_mask]
                
                if len(cluster_data) < 2:
                    continue
                
                # Calculate returns (assuming 'close' column exists)
                if 'close' in cluster_data.columns:
                    returns = cluster_data['close'].pct_change().dropna()
                    cluster_returns[cluster] = returns.mean()
                    cluster_volatilities[cluster] = returns.std()
            
            if not cluster_returns:
                return {'error': 'No valid cluster returns calculated'}
            
            # Calculate economic significance metrics
            return_spread = max(cluster_returns.values()) - min(cluster_returns.values())
            volatility_spread = max(cluster_volatilities.values()) - min(cluster_volatilities.values())
            
            # Economic significance score (higher is better)
            economic_score = min(1.0, (return_spread + volatility_spread) / 0.1)  # Normalize to 0-1
            
            return {
                'cluster_returns': cluster_returns,
                'cluster_volatilities': cluster_volatilities,
                'return_spread': return_spread,
                'volatility_spread': volatility_spread,
                'economic_significance_score': economic_score,
                'is_economically_significant': economic_score >= 0.5
            }
            
        except Exception as e:
            return {'error': f'Economic significance validation failed: {e}'}
    
    def _cross_validate_clusters(
        self, 
        hmm_models: List[Any], 
        cluster_assignments: List[int], 
        market_data: Any
    ) -> Dict[str, Any]:
        """Perform cross-validation to ensure cluster stability."""
        if not cluster_assignments or len(cluster_assignments) < 100:
            return {'error': 'Insufficient data for cross-validation'}
        
        try:
            # Split data into train/test for stability check
            split_point = len(cluster_assignments) // 2
            train_assignments = cluster_assignments[:split_point]
            test_assignments = cluster_assignments[split_point:]
            
            # Calculate cluster distributions
            train_dist = self._calculate_cluster_distribution(train_assignments)
            test_dist = self._calculate_cluster_distribution(test_assignments)
            
            # Calculate stability score (how similar are the distributions)
            stability_score = 0.0
            if train_dist and test_dist:
                # Calculate correlation between distributions
                common_clusters = set(train_dist.keys()) & set(test_dist.keys())
                if common_clusters:
                    train_values = [train_dist.get(cluster, 0) for cluster in common_clusters]
                    test_values = [test_dist.get(cluster, 0) for cluster in common_clusters]
                    
                    if NUMPY_AVAILABLE and len(train_values) > 1:
                        correlation = np.corrcoef(train_values, test_values)[0, 1]
                        stability_score = max(0, correlation) if not np.isnan(correlation) else 0
                    else:
                        # Simple similarity measure
                        diff = sum(abs(t - s) for t, s in zip(train_values, test_values))
                        stability_score = max(0, 1 - diff / len(common_clusters))
            
            return {
                'train_distribution': train_dist,
                'test_distribution': test_dist,
                'stability_score': stability_score,
                'is_stable': stability_score >= 0.7
            }
            
        except Exception as e:
            return {'error': f'Cross-validation failed: {e}'}
    
    def _analyze_cluster_transitions(self, cluster_assignments: List[int]) -> Dict[str, Any]:
        """Analyze cluster transition patterns."""
        if not cluster_assignments or len(cluster_assignments) < 2:
            return {'error': 'Insufficient data for transition analysis'}
        
        try:
            # Count transitions
            transitions = {}
            total_transitions = 0
            
            for i in range(1, len(cluster_assignments)):
                from_cluster = cluster_assignments[i-1]
                to_cluster = cluster_assignments[i]
                
                if from_cluster != to_cluster:
                    transition_key = f"{from_cluster}->{to_cluster}"
                    transitions[transition_key] = transitions.get(transition_key, 0) + 1
                    total_transitions += 1
            
            # Calculate transition probabilities
            transition_probs = {}
            for transition, count in transitions.items():
                from_cluster = int(transition.split('->')[0])
                from_count = cluster_assignments.count(from_cluster)
                if from_count > 0:
                    transition_probs[transition] = count / from_count
            
            # Calculate transition entropy (higher = more random transitions)
            entropy = 0.0
            if total_transitions > 0:
                for count in transitions.values():
                    prob = count / total_transitions
                    if prob > 0:
                        entropy -= prob * np.log2(prob) if NUMPY_AVAILABLE else prob * np.log(prob) / np.log(2)
            
            return {
                'transitions': transitions,
                'transition_probabilities': transition_probs,
                'total_transitions': total_transitions,
                'transition_entropy': entropy,
                'transition_frequency': total_transitions / len(cluster_assignments),
                'is_transition_stable': entropy < 2.0  # Lower entropy = more stable
            }
            
        except Exception as e:
            return {'error': f'Transition analysis failed: {e}'}
    
    def _apply_quality_gates(
        self, 
        persistence_metrics: Dict[str, Any],
        economic_metrics: Dict[str, Any], 
        stability_metrics: Dict[str, Any],
        transition_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply multi-stage validation gates."""
        gates = {}
        
        # Gate 1: Persistence Gate
        gates['persistence_gate'] = {
            'passed': persistence_metrics.get('stability_score', 0) >= 0.5,
            'score': persistence_metrics.get('stability_score', 0),
            'threshold': 0.5
        }
        
        # Gate 2: Economic Significance Gate
        gates['economic_gate'] = {
            'passed': economic_metrics.get('is_economically_significant', False),
            'score': economic_metrics.get('economic_significance_score', 0),
            'threshold': 0.5
        }
        
        # Gate 3: Stability Gate
        gates['stability_gate'] = {
            'passed': stability_metrics.get('is_stable', False),
            'score': stability_metrics.get('stability_score', 0),
            'threshold': 0.7
        }
        
        # Gate 4: Transition Gate
        gates['transition_gate'] = {
            'passed': transition_metrics.get('is_transition_stable', False),
            'score': 1 - (transition_metrics.get('transition_entropy', 0) / 3.0),  # Normalize entropy
            'threshold': 0.5
        }
        
        # Overall gate result
        gates['overall_passed'] = all(gate['passed'] for gate in gates.values() if isinstance(gate, dict) and 'passed' in gate)
        
        return gates
    
    def _calculate_overall_quality_score(self, quality_metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score from all metrics."""
        try:
            scores = []
            
            # Persistence score
            if 'persistence_analysis' in quality_metrics:
                scores.append(quality_metrics['persistence_analysis'].get('stability_score', 0))
            
            # Economic significance score
            if 'economic_significance' in quality_metrics:
                scores.append(quality_metrics['economic_significance'].get('economic_significance_score', 0))
            
            # Stability score
            if 'stability_analysis' in quality_metrics:
                scores.append(quality_metrics['stability_analysis'].get('stability_score', 0))
            
            # Transition score
            if 'transition_analysis' in quality_metrics:
                transition_entropy = quality_metrics['transition_analysis'].get('transition_entropy', 0)
                transition_score = max(0, 1 - (transition_entropy / 3.0))  # Normalize entropy
                scores.append(transition_score)
            
            if not scores:
                return 0.0
            
            return sum(scores) / len(scores)
            
        except Exception as e:
            self.logger.error(f"Error calculating overall quality score: {e}")
            return 0.0
    
    def _generate_quality_recommendations(self, quality_metrics: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []
        
        # Check persistence
        persistence_score = quality_metrics.get('persistence_analysis', {}).get('stability_score', 0)
        if persistence_score < 0.5:
            recommendations.append("Consider increasing minimum cluster duration to improve persistence")
        
        # Check economic significance
        economic_score = quality_metrics.get('economic_significance', {}).get('economic_significance_score', 0)
        if economic_score < 0.5:
            recommendations.append("Clusters may not be economically significant - consider feature engineering")
        
        # Check stability
        stability_score = quality_metrics.get('stability_analysis', {}).get('stability_score', 0)
        if stability_score < 0.7:
            recommendations.append("Clusters show low stability - consider cross-validation improvements")
        
        # Check transitions
        transition_entropy = quality_metrics.get('transition_analysis', {}).get('transition_entropy', 0)
        if transition_entropy > 2.0:
            recommendations.append("High transition entropy - consider cluster smoothing or filtering")
        
        if not recommendations:
            recommendations.append("Cluster quality is good - no specific recommendations")
        
        return recommendations
    
    def _generate_cluster_detailed_metrics(
        self, 
        hmm_models: List[Any], 
        cluster_assignments: List[int], 
        market_data: Any,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate detailed metrics for each HMM cluster."""
        start_time = time.time()
        
        try:
            if not PANDAS_AVAILABLE or not isinstance(market_data, pd.DataFrame):
                return {'error': 'Pandas not available or invalid market data for detailed metrics'}
            
            if not cluster_assignments or not hmm_models:
                return {'error': 'No cluster assignments or models available'}
            
            detailed_metrics = {}
            unique_clusters = sorted(set(cluster_assignments))
            
            self.logger.info(f"📊 Generating detailed metrics for {len(unique_clusters)} clusters")
            
            for cluster_id in unique_clusters:
                cluster_metrics = self._analyze_single_cluster(
                    cluster_id, hmm_models, cluster_assignments, market_data
                )
                detailed_metrics[f'cluster_{cluster_id}'] = cluster_metrics
            
            # Add cluster comparison metrics
            comparison_metrics = self._generate_cluster_comparison_metrics(
                detailed_metrics, cluster_assignments, market_data
            )
            detailed_metrics['cluster_comparison'] = comparison_metrics
            
            # Add cluster performance metrics
            performance_metrics = self._generate_cluster_performance_metrics(
                detailed_metrics, cluster_assignments, market_data
            )
            detailed_metrics['cluster_performance'] = performance_metrics
            
            generation_time = time.time() - start_time
            detailed_metrics['generation_time'] = generation_time
            
            self.logger.info(f"✅ Detailed cluster metrics generated in {generation_time:.2f}s")
            
            return detailed_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Detailed cluster metrics generation failed: {e}")
            return {
                'error': str(e),
                'generation_time': time.time() - start_time
            }
    
    def _analyze_single_cluster(
        self, 
        cluster_id: int, 
        hmm_models: List[Any], 
        cluster_assignments: List[int], 
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Analyze a single cluster in detail."""
        try:
            # Get cluster data
            cluster_mask = np.array(cluster_assignments) == cluster_id
            cluster_data = market_data[cluster_mask]
            
            if len(cluster_data) < 2:
                return {'error': f'Insufficient data for cluster {cluster_id}'}
            
            cluster_metrics = {
                'cluster_id': cluster_id,
                'sample_percentage': (len(cluster_data) / len(market_data)) * 100
            }
            
            
            # Volume analysis
            if 'volume' in cluster_data.columns:
                volume_metrics = self._analyze_cluster_volume(cluster_data)
                cluster_metrics['volume_analysis'] = volume_metrics
            
            # Volatility analysis
            if 'high' in cluster_data.columns and 'low' in cluster_data.columns:
                volatility_metrics = self._analyze_cluster_volatility(cluster_data)
                cluster_metrics['volatility_analysis'] = volatility_metrics
            
            # Trend analysis
            trend_metrics = self._analyze_cluster_trend(cluster_data)
            cluster_metrics['trend_analysis'] = trend_metrics
            
            # HMM model analysis
            if cluster_id < len(hmm_models):
                hmm_metrics = self._analyze_cluster_hmm_model(hmm_models[cluster_id])
                cluster_metrics['hmm_model_analysis'] = hmm_metrics
            
            
            return cluster_metrics
            
        except Exception as e:
            return {'error': f'Analysis failed for cluster {cluster_id}: {e}'}
    
    
    def _analyze_cluster_volume(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume characteristics of a cluster."""
        try:
            volume = cluster_data['volume']
            volume_returns = volume.pct_change().dropna()
            
            return {
                'volume_stats': {
                    'mean_volume': float(volume.mean()),
                    'median_volume': float(volume.median()),
                    'std_volume': float(volume.std()),
                    'min_volume': float(volume.min()),
                    'max_volume': float(volume.max())
                },
                'volume_volatility': {
                    'volume_cv': float(volume.std() / volume.mean()) if volume.mean() > 0 else 0.0,
                    'volume_trend': 'increasing' if volume.iloc[-1] > volume.iloc[0] else 'decreasing'
                },
                'volume_anomalies': {
                    'high_volume_threshold': float(volume.quantile(0.9)),
                    'low_volume_threshold': float(volume.quantile(0.1)),
                    'high_volume_samples': int((volume > volume.quantile(0.9)).sum()),
                    'low_volume_samples': int((volume < volume.quantile(0.1)).sum())
                }
            }
        except Exception as e:
            return {'error': f'Volume analysis failed: {e}'}
    
    def _analyze_cluster_volatility(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility characteristics of a cluster."""
        try:
            high = cluster_data['high']
            low = cluster_data['low']
            close = cluster_data['close']
            
            # True Range and ATR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean()
            
            # Price ranges
            daily_ranges = (high - low) / close * 100
            
            return {
                'volatility_metrics': {
                    'mean_daily_range_pct': float(daily_ranges.mean()),
                    'std_daily_range_pct': float(daily_ranges.std()),
                    'max_daily_range_pct': float(daily_ranges.max()),
                    'min_daily_range_pct': float(daily_ranges.min()),
                    'mean_atr': float(atr.mean()),
                    'atr_volatility': float(atr.std())
                },
                'volatility_classification': {
                    'volatility_level': self._classify_volatility(daily_ranges.mean()),
                    'volatility_consistency': 'consistent' if daily_ranges.std() < daily_ranges.mean() * 0.5 else 'inconsistent'
                }
            }
        except Exception as e:
            return {'error': f'Volatility analysis failed: {e}'}
    
    def _analyze_cluster_trend(self, cluster_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trend characteristics of a cluster."""
        try:
            close = cluster_data['close']
            
            # Simple moving averages
            sma_5 = close.rolling(window=5).mean()
            sma_20 = close.rolling(window=20).mean() if len(close) >= 20 else close.rolling(window=len(close)).mean()
            
            # Trend strength
            trend_strength = abs(sma_5.iloc[-1] - sma_20.iloc[-1]) / sma_20.iloc[-1] * 100 if len(sma_20) > 0 else 0
            
            return {
                'trend_metrics': {
                    'trend_direction': 'upward' if sma_5.iloc[-1] > sma_20.iloc[-1] else 'downward' if len(sma_5) > 0 and len(sma_20) > 0 else 'neutral',
                    'trend_strength': float(trend_strength),
                    'trend_consistency': self._assess_trend_consistency(close)
                },
                'moving_averages': {
                    'sma_5': float(sma_5.iloc[-1]) if len(sma_5) > 0 else None,
                    'sma_20': float(sma_20.iloc[-1]) if len(sma_20) > 0 else None,
                    'ma_cross': 'golden' if sma_5.iloc[-1] > sma_20.iloc[-1] else 'death' if len(sma_5) > 0 and len(sma_20) > 0 else 'none'
                }
            }
        except Exception as e:
            return {'error': f'Trend analysis failed: {e}'}
    
    def _analyze_cluster_hmm_model(self, hmm_model: Any) -> Dict[str, Any]:
        """Analyze HMM model characteristics for a cluster."""
        try:
            model_metrics = {
                'model_type': str(type(hmm_model).__name__),
                'model_available': hmm_model is not None
            }
            
            if hasattr(hmm_model, 'n_components'):
                model_metrics['n_components'] = hmm_model.n_components
            
            if hasattr(hmm_model, 'covariance_type'):
                model_metrics['covariance_type'] = hmm_model.covariance_type
            
            if hasattr(hmm_model, 'means_'):
                model_metrics['means'] = hmm_model.means_.tolist() if hasattr(hmm_model.means_, 'tolist') else str(hmm_model.means_)
            
            if hasattr(hmm_model, 'covars_'):
                model_metrics['covariances_available'] = True
                model_metrics['covariance_shape'] = hmm_model.covars_.shape if hasattr(hmm_model.covars_, 'shape') else 'unknown'
            
            return model_metrics
            
        except Exception as e:
            return {'error': f'HMM model analysis failed: {e}'}
    
    
    def _classify_volatility(self, mean_daily_range: float) -> str:
        """Classify volatility level based on mean daily range."""
        if mean_daily_range > 3.0:
            return 'high'
        elif mean_daily_range > 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _assess_trend_consistency(self, prices: pd.Series) -> str:
        """Assess trend consistency."""
        if len(prices) < 3:
            return 'insufficient_data'
        
        # Count direction changes
        direction_changes = 0
        for i in range(1, len(prices)):
            if (prices.iloc[i] > prices.iloc[i-1]) != (prices.iloc[i-1] > prices.iloc[i-2] if i > 1 else True):
                direction_changes += 1
        
        change_ratio = direction_changes / (len(prices) - 2)
        
        if change_ratio < 0.2:
            return 'very_consistent'
        elif change_ratio < 0.4:
            return 'consistent'
        elif change_ratio < 0.6:
            return 'moderate'
        else:
            return 'inconsistent'
    
    def _generate_cluster_comparison_metrics(
        self, 
        detailed_metrics: Dict[str, Any], 
        cluster_assignments: List[int], 
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate comparison metrics between clusters."""
        try:
            comparison = {
                'cluster_count': len([k for k in detailed_metrics.keys() if k.startswith('cluster_')]),
                'cluster_distribution': self._calculate_cluster_distribution(cluster_assignments),
                'cluster_rankings': {}
            }
            
            # Rank clusters by various metrics
            rankings = {
                'by_sample_count': [],
                'by_return': [],
                'by_volatility': [],
                'by_volume': [],
                'by_trend': []
            }
            
            for cluster_key, metrics in detailed_metrics.items():
                if cluster_key.startswith('cluster_') and 'error' not in metrics:
                    cluster_id = metrics.get('cluster_id', 0)
                    
                    # Sample count ranking
                    sample_percentage = metrics.get('sample_percentage', 0)
                    rankings['by_sample_count'].append((cluster_id, sample_percentage))
                    
                    # Volatility ranking
                    if 'volatility_analysis' in metrics and 'volatility_metrics' in metrics['volatility_analysis']:
                        volatility = metrics['volatility_analysis']['volatility_metrics']['mean_daily_range_pct']
                        rankings['by_volatility'].append((cluster_id, volatility))
                    
                    # Volume ranking
                    if 'volume_analysis' in metrics and 'volume_stats' in metrics['volume_analysis']:
                        volume = metrics['volume_analysis']['volume_stats']['mean_volume']
                        rankings['by_volume'].append((cluster_id, volume))
                    
                    # Trend ranking
                    if 'trend_analysis' in metrics and 'trend_metrics' in metrics['trend_analysis']:
                        trend_strength = metrics['trend_analysis']['trend_metrics']['trend_strength']
                        rankings['by_trend'].append((cluster_id, trend_strength))
            
            # Sort rankings
            for ranking_type, ranking_list in rankings.items():
                if ranking_list:
                    sorted_ranking = sorted(ranking_list, key=lambda x: x[1], reverse=True)
                    comparison['cluster_rankings'][ranking_type] = {
                        'best': sorted_ranking[0] if sorted_ranking else None,
                        'worst': sorted_ranking[-1] if sorted_ranking else None,
                        'all_rankings': sorted_ranking
                    }
            
            return comparison
            
        except Exception as e:
            return {'error': f'Cluster comparison failed: {e}'}
    
    def _generate_cluster_performance_metrics(
        self, 
        detailed_metrics: Dict[str, Any], 
        cluster_assignments: List[int], 
        market_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate performance metrics for clusters."""
        try:
            performance = {
                'overall_performance': {},
                'cluster_performance': {},
                'performance_insights': []
            }
            
            # Calculate overall performance metrics
            total_samples = len(cluster_assignments)
            unique_clusters = set(cluster_assignments)
            
            performance['overall_performance'] = {
                'total_clusters': len(unique_clusters),
                'total_samples': total_samples,
                'avg_samples_per_cluster': total_samples / len(unique_clusters) if unique_clusters else 0,
                'cluster_balance': self._calculate_cluster_balance(cluster_assignments)
            }
            
            # Calculate individual cluster performance
            for cluster_key, metrics in detailed_metrics.items():
                if cluster_key.startswith('cluster_') and 'error' not in metrics:
                    cluster_id = metrics.get('cluster_id', 0)
                    
                    cluster_perf = {
                        'sample_efficiency': metrics.get('sample_percentage', 0),
                        'data_quality': self._assess_data_quality(metrics),
                        'market_impact': self._assess_market_impact(metrics)
                    }
                    
                    performance['cluster_performance'][f'cluster_{cluster_id}'] = cluster_perf
            
            
            return performance
            
        except Exception as e:
            return {'error': f'Performance metrics generation failed: {e}'}
    
    def _calculate_cluster_balance(self, cluster_assignments: List[int]) -> Dict[str, Any]:
        """Calculate cluster balance metrics."""
        cluster_counts = {}
        for assignment in cluster_assignments:
            cluster_counts[assignment] = cluster_counts.get(assignment, 0) + 1
        
        if not cluster_counts:
            return {'balance_score': 0.0, 'is_balanced': False}
        
        counts = list(cluster_counts.values())
        mean_count = sum(counts) / len(counts)
        std_count = np.std(counts) if NUMPY_AVAILABLE else 0
        
        balance_score = max(0, 1 - (std_count / mean_count)) if mean_count > 0 else 0
        is_balanced = balance_score > 0.7
        
        return {
            'balance_score': balance_score,
            'is_balanced': is_balanced,
            'cluster_counts': cluster_counts,
            'count_std': std_count,
            'count_mean': mean_count
        }
    
    def _assess_data_quality(self, cluster_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess data quality for a cluster."""
        quality_score = 0.0
        quality_factors = []
        
        # Check if all analyses are available
        analyses = ['volume_analysis', 'volatility_analysis', 'trend_analysis']
        available_analyses = sum(1 for analysis in analyses if analysis in cluster_metrics and 'error' not in cluster_metrics[analysis])
        
        if available_analyses == len(analyses):
            quality_score += 0.4
            quality_factors.append('all_analyses_available')
        elif available_analyses >= 2:
            quality_score += 0.2
            quality_factors.append('partial_analyses_available')
        
        # Check sample percentage
        sample_percentage = cluster_metrics.get('sample_percentage', 0)
        if sample_percentage >= 30:
            quality_score += 0.3
            quality_factors.append('sufficient_samples')
        elif sample_percentage >= 15:
            quality_score += 0.2
            quality_factors.append('moderate_samples')
        
        return {
            'quality_score': quality_score,
            'quality_level': 'high' if quality_score >= 0.8 else 'medium' if quality_score >= 0.5 else 'low',
            'quality_factors': quality_factors
        }
    
    def _assess_market_impact(self, cluster_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess market impact of a cluster."""
        impact_score = 0.0
        impact_factors = []
        
        # Trend impact
        if 'trend_analysis' in cluster_metrics and 'trend_metrics' in cluster_metrics['trend_analysis']:
            trend_strength = cluster_metrics['trend_analysis']['trend_metrics']['trend_strength']
            if trend_strength > 5:
                impact_score += 0.4
                impact_factors.append('high_trend_impact')
            elif trend_strength > 2:
                impact_score += 0.2
                impact_factors.append('moderate_trend_impact')
        
        # Volume impact
        if 'volume_analysis' in cluster_metrics and 'volume_anomalies' in cluster_metrics['volume_analysis']:
            high_volume_samples = cluster_metrics['volume_analysis']['volume_anomalies']['high_volume_samples']
            # Use sample percentage as proxy for total samples
            sample_percentage = cluster_metrics.get('sample_percentage', 1)
            if high_volume_samples > 0 and sample_percentage > 0:
                volume_ratio = high_volume_samples / (sample_percentage * 10)  # Approximate total samples
                if volume_ratio > 0.2:
                    impact_score += 0.3
                    impact_factors.append('high_volume_activity')
        
        # Volatility impact
        if 'volatility_analysis' in cluster_metrics and 'volatility_classification' in cluster_metrics['volatility_analysis']:
            vol_level = cluster_metrics['volatility_analysis']['volatility_classification']['volatility_level']
            if vol_level == 'high':
                impact_score += 0.3
                impact_factors.append('high_volatility')
        
        return {
            'impact_score': impact_score,
            'impact_level': 'high' if impact_score >= 0.7 else 'medium' if impact_score >= 0.4 else 'low',
            'impact_factors': impact_factors
        }
    
    
    async def _prepare_data_for_clustering_optimized(
        self, 
        data: Any, 
        regime_discovery: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Any:
        """Prepare data for clustering with memory optimization."""
        if not PANDAS_AVAILABLE or not isinstance(data, pd.DataFrame):
            self.logger.warning("Pandas not available or data is not a DataFrame, using fallback")
            return {
                'market_data': data,
                'regime_discovery': regime_discovery
            }
        
        # Use memory optimizer to determine optimal chunk size
        if self.memory_optimizer:
            memory_limit_gb = config.get('memory_limit_gb', 8.0)
            optimal_chunk_size = self.memory_optimizer.calculate_optimal_chunk_size(
                data.shape, memory_limit_gb
            )
            self.logger.info(f"🔧 Memory optimization: Using chunk size {optimal_chunk_size} for {data.shape[0]} rows")
        else:
            optimal_chunk_size = min(10000, len(data))  # Fallback chunk size
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for clustering: {missing_columns}")
            # Use available columns or create fallback data
            for col in missing_columns:
                if col == 'volume':
                    data[col] = 1000  # Default volume
                else:
                    data[col] = data.get('close', 100.0)  # Use close price as fallback
        
        return {
            'market_data': data,
            'regime_discovery': regime_discovery,
            'chunk_size': optimal_chunk_size,
            'memory_optimized': self.memory_optimizer is not None
        }
    
    async def _perform_parallel_hmm_clustering(
        self, 
        hmm_manager: Any, 
        prepared_data: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform HMM clustering with parallel processing optimization."""
        try:
            # Get optimal number of workers
            if self.cpu_optimizer:
                max_workers = self.cpu_optimizer.get_optimal_worker_count()
                self.logger.info(f"🔧 CPU optimization: Using {max_workers} workers for parallel processing")
            else:
                max_workers = 4  # Fallback worker count
            
            # Split data into chunks for parallel processing
            market_data = prepared_data.get('market_data')
            chunk_size = prepared_data.get('chunk_size', 10000)
            
            if PANDAS_AVAILABLE and isinstance(market_data, pd.DataFrame):
                # Create data chunks
                chunks = []
                for i in range(0, len(market_data), chunk_size):
                    chunk = market_data.iloc[i:i+chunk_size].copy()
                    chunks.append(chunk)
                
                self.logger.info(f"🔧 Processing {len(chunks)} data chunks in parallel")
                
                # Process chunks in parallel
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit clustering tasks for each chunk
                    future_to_chunk = {
                        executor.submit(
                            self._cluster_single_chunk, 
                            hmm_manager, chunk, config, i
                        ): i for i, chunk in enumerate(chunks)
                    }
                    
                    # Collect results
                    chunk_results = []
                    for future in concurrent.futures.as_completed(future_to_chunk):
                        chunk_idx = future_to_chunk[future]
                        try:
                            result = future.result()
                            chunk_results.append((chunk_idx, result))
                        except Exception as e:
                            self.logger.error(f"❌ Chunk {chunk_idx} clustering failed: {e}")
                            chunk_results.append((chunk_idx, None))
                
                # Merge chunk results
                merged_result = self._merge_chunk_clustering_results(chunk_results)
                return merged_result
            else:
                # Fallback to single-threaded processing
                return await hmm_manager.perform_hmm_clustering(prepared_data, config)
                
        except Exception as e:
            self.logger.error(f"❌ Parallel HMM clustering failed: {e}")
            # Fallback to single-threaded processing
            return await hmm_manager.perform_hmm_clustering(prepared_data, config)
    
    def _cluster_single_chunk(
        self, 
        hmm_manager: Any, 
        chunk_data: Any, 
        config: Dict[str, Any], 
        chunk_idx: int
    ) -> Dict[str, Any]:
        """Cluster a single data chunk."""
        try:
            # Prepare chunk data
            chunk_prepared = {
                'market_data': chunk_data,
                'regime_discovery': {},  # Empty for individual chunks
                'chunk_index': chunk_idx
            }
            
            # Perform clustering on chunk
            result = hmm_manager.perform_hmm_clustering(chunk_prepared, config)
            result['chunk_index'] = chunk_idx
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Chunk {chunk_idx} clustering failed: {e}")
            return {
                'hmm_models': [],
                'cluster_assignments': [],
                'cluster_metrics': {'error': str(e)},
                'chunk_index': chunk_idx
            }
    
    def _merge_chunk_clustering_results(self, chunk_results: List[Tuple[int, Any]]) -> Dict[str, Any]:
        """Merge results from multiple clustering chunks."""
        try:
            # Sort results by chunk index
            chunk_results.sort(key=lambda x: x[0])
            
            # Merge HMM models
            all_models = []
            all_assignments = []
            all_metrics = []
            
            for chunk_idx, result in chunk_results:
                if result is None:
                    continue
                
                models = result.get('hmm_models', [])
                assignments = result.get('cluster_assignments', [])
                metrics = result.get('cluster_metrics', {})
                
                # Adjust assignment indices to be globally unique
                if assignments:
                    max_assignment = max(all_assignments) if all_assignments else -1
                    adjusted_assignments = [a + max_assignment + 1 for a in assignments]
                    all_assignments.extend(adjusted_assignments)
                
                all_models.extend(models)
                all_metrics.append(metrics)
            
            # Calculate merged metrics
            merged_metrics = {
                'clustering_method': 'parallel_chunked',
                'total_chunks': len(chunk_results),
                'successful_chunks': len([r for r in chunk_results if r[1] is not None]),
                'total_models': len(all_models),
                'total_assignments': len(all_assignments)
            }
            
            self.logger.info(f"✅ Merged {len(chunk_results)} chunks: {len(all_models)} models, {len(all_assignments)} assignments")
            
            return {
                'hmm_models': all_models,
                'cluster_assignments': all_assignments,
                'cluster_metrics': merged_metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to merge chunk results: {e}")
            return {
                'hmm_models': [],
                'cluster_assignments': [],
                'cluster_metrics': {'error': f'Merge failed: {e}'}
            }
    