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

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KMeans = None
    silhouette_score = None
    calinski_harabasz_score = None

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
            
            # Configure HMM clustering
            clustering_config = {
                'n_clusters': 3,  # Bull, Bear, Sideways
                'clustering_method': 'hmm_based',  # Options: 'hmm_based', 'kmeans_only', 'multi_algorithm_consensus'
                'min_cluster_size': 10,
                'convergence_tolerance': 1e-6,
                'max_iterations': 100,
                'random_state': 42,
                
                # Regime constraints
                'max_regimes': 25,  # Maximum 25 regimes allowed
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
            
            # Perform comprehensive regime quality validation
            quality_metrics = self._validate_regime_quality(
                hmm_models, cluster_assignments, market_data, clustering_config
            )
            
            # Create single consolidated artifact
            artifacts = {
                'hmm_clustering_result': {
                    'hmm_models': hmm_models,
                    'cluster_assignments': cluster_assignments,
                    'cluster_metrics': cluster_metrics,
                    'regime_quality_metrics': quality_metrics,
                    'clustering_summary': {
                        'total_clusters': len(hmm_models),
                        'total_assignments': len(cluster_assignments),
                        'cluster_distribution': self._calculate_cluster_distribution(cluster_assignments),
                        'clustering_time': clustering_result.get('clustering_time', 0.0),
                        'quality_score': quality_metrics.get('overall_quality_score', 0.0),
                        'validation_passed': quality_metrics.get('validation_passed', False)
                    },
                    'metadata': {
                        'symbol': self.config.symbol,
                        'exchange': self.config.exchange,
                        'timeframe': self.config.timeframe,
                        'data_points': len(market_data) if market_data is not None else 0,
                        'execution_timestamp': datetime.now().isoformat()
                    }
                }
            }
            
            self.logger.info(f'✅ HMM Clustering completed: {len(hmm_models)} clusters created')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'cluster_count': len(hmm_models)
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
            
            # Perform clustering with multiple algorithms
            clustering_method = config.get('clustering_method', 'hmm_based')
            
            if clustering_method == 'kmeans_only':
                clustering_result = await self._perform_kmeans_clustering(
                    prepared_data, config
                )
            elif clustering_method == 'multi_algorithm_consensus':
                clustering_result = await self._perform_multi_algorithm_clustering(
                    hmm_manager, prepared_data, config
                )
            else:  # Default: HMM-based clustering
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
        """Apply regime constraints: max 25 regimes and 1% sample threshold."""
        max_regimes = config.get('max_regimes', 25)
        min_sample_percentage = config.get('min_regime_sample_percentage', 0.01)
        
        if not cluster_assignments:
            return {'hmm_models': hmm_models, 'cluster_assignments': cluster_assignments}
        
        total_samples = len(cluster_assignments)
        min_samples = int(total_samples * min_sample_percentage)
        
        # Count samples per regime
        regime_counts = {}
        for assignment in cluster_assignments:
            regime_counts[assignment] = regime_counts.get(assignment, 0) + 1
        
        # Filter regimes that meet the minimum sample threshold
        valid_regimes = []
        for regime, count in regime_counts.items():
            if count >= min_samples:
                valid_regimes.append(regime)
            else:
                self.logger.warning(f"⚠️ Regime {regime} has {count} samples ({count/total_samples:.2%}), below 1% threshold - removing")
        
        # Limit to max_regimes
        if len(valid_regimes) > max_regimes:
            # Keep the regimes with the most samples
            regime_counts_sorted = sorted(regime_counts.items(), key=lambda x: x[1], reverse=True)
            valid_regimes = [regime for regime, _ in regime_counts_sorted[:max_regimes]]
            self.logger.warning(f"⚠️ Limiting to {max_regimes} regimes (had {len(regime_counts)} regimes)")
        
        # Filter cluster assignments to only include valid regimes
        filtered_assignments = []
        for assignment in cluster_assignments:
            if assignment in valid_regimes:
                filtered_assignments.append(assignment)
            else:
                # Assign to the most common valid regime as fallback
                if valid_regimes:
                    filtered_assignments.append(valid_regimes[0])
                else:
                    filtered_assignments.append(0)  # Fallback to regime 0
        
        # Filter HMM models to match valid regimes
        filtered_models = []
        for i, regime in enumerate(valid_regimes):
            if i < len(hmm_models):
                filtered_models.append(hmm_models[i])
        
        self.logger.info(f"✅ Applied regime constraints: {len(valid_regimes)} valid regimes (min {min_samples} samples each, max {max_regimes} regimes)")
        
        return {
            'hmm_models': filtered_models,
            'cluster_assignments': filtered_assignments
        }
    
    def _validate_regime_quality(
        self, 
        hmm_models: List[Any], 
        cluster_assignments: List[int], 
        market_data: Any,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform comprehensive regime quality validation."""
        start_time = time.time()
        
        try:
            quality_metrics = {}
            
            # 1. Regime Persistence Analysis
            persistence_metrics = self._calculate_regime_persistence(cluster_assignments)
            quality_metrics['persistence_analysis'] = persistence_metrics
            
            # 2. Economic Significance Validation
            if PANDAS_AVAILABLE and market_data is not None:
                economic_metrics = self._validate_economic_significance(
                    hmm_models, cluster_assignments, market_data
                )
                quality_metrics['economic_significance'] = economic_metrics
            
            # 3. Cross-validation Stability
            stability_metrics = self._cross_validate_regimes(
                hmm_models, cluster_assignments, market_data
            )
            quality_metrics['stability_analysis'] = stability_metrics
            
            # 4. Regime Transition Analysis
            transition_metrics = self._analyze_regime_transitions(cluster_assignments)
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
            
            self.logger.info(f"✅ Regime quality validation completed in {validation_time:.2f}s")
            self.logger.info(f"📊 Overall quality score: {overall_score:.2f} ({'PASSED' if quality_metrics['validation_passed'] else 'FAILED'})")
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Regime quality validation failed: {e}")
            return {
                'overall_quality_score': 0.0,
                'validation_passed': False,
                'error': str(e),
                'validation_time': time.time() - start_time
            }
    
    def _calculate_regime_persistence(self, cluster_assignments: List[int]) -> Dict[str, Any]:
        """Calculate regime persistence metrics."""
        if not cluster_assignments or len(cluster_assignments) < 2:
            return {'error': 'Insufficient data for persistence analysis'}
        
        # Calculate regime durations
        regime_durations = []
        current_regime = cluster_assignments[0]
        current_duration = 1
        
        for i in range(1, len(cluster_assignments)):
            if cluster_assignments[i] == current_regime:
                current_duration += 1
            else:
                regime_durations.append(current_duration)
                current_regime = cluster_assignments[i]
                current_duration = 1
        
        # Add the last regime duration
        regime_durations.append(current_duration)
        
        if not regime_durations:
            return {'error': 'No regime durations calculated'}
        
        # Calculate persistence metrics
        avg_duration = np.mean(regime_durations) if NUMPY_AVAILABLE else sum(regime_durations) / len(regime_durations)
        median_duration = np.median(regime_durations) if NUMPY_AVAILABLE else sorted(regime_durations)[len(regime_durations)//2]
        std_duration = np.std(regime_durations) if NUMPY_AVAILABLE else 0
        
        # Calculate regime stability (lower std = more stable)
        stability_score = max(0, 1 - (std_duration / avg_duration)) if avg_duration > 0 else 0
        
        return {
            'avg_duration': avg_duration,
            'median_duration': median_duration,
            'std_duration': std_duration,
            'stability_score': stability_score,
            'total_transitions': len(regime_durations) - 1,
            'regime_durations': regime_durations
        }
    
    def _validate_economic_significance(
        self, 
        hmm_models: List[Any], 
        cluster_assignments: List[int], 
        market_data: Any
    ) -> Dict[str, Any]:
        """Validate economic significance of regimes."""
        if not PANDAS_AVAILABLE or not isinstance(market_data, pd.DataFrame):
            return {'error': 'Pandas not available or invalid market data'}
        
        try:
            # Calculate returns for each regime
            regime_returns = {}
            regime_volatilities = {}
            
            for regime in set(cluster_assignments):
                regime_mask = np.array(cluster_assignments) == regime
                regime_data = market_data[regime_mask]
                
                if len(regime_data) < 2:
                    continue
                
                # Calculate returns (assuming 'close' column exists)
                if 'close' in regime_data.columns:
                    returns = regime_data['close'].pct_change().dropna()
                    regime_returns[regime] = returns.mean()
                    regime_volatilities[regime] = returns.std()
            
            if not regime_returns:
                return {'error': 'No valid regime returns calculated'}
            
            # Calculate economic significance metrics
            return_spread = max(regime_returns.values()) - min(regime_returns.values())
            volatility_spread = max(regime_volatilities.values()) - min(regime_volatilities.values())
            
            # Economic significance score (higher is better)
            economic_score = min(1.0, (return_spread + volatility_spread) / 0.1)  # Normalize to 0-1
            
            return {
                'regime_returns': regime_returns,
                'regime_volatilities': regime_volatilities,
                'return_spread': return_spread,
                'volatility_spread': volatility_spread,
                'economic_significance_score': economic_score,
                'is_economically_significant': economic_score >= 0.5
            }
            
        except Exception as e:
            return {'error': f'Economic significance validation failed: {e}'}
    
    def _cross_validate_regimes(
        self, 
        hmm_models: List[Any], 
        cluster_assignments: List[int], 
        market_data: Any
    ) -> Dict[str, Any]:
        """Perform cross-validation to ensure regime stability."""
        if not cluster_assignments or len(cluster_assignments) < 100:
            return {'error': 'Insufficient data for cross-validation'}
        
        try:
            # Split data into train/test for stability check
            split_point = len(cluster_assignments) // 2
            train_assignments = cluster_assignments[:split_point]
            test_assignments = cluster_assignments[split_point:]
            
            # Calculate regime distributions
            train_dist = self._calculate_cluster_distribution(train_assignments)
            test_dist = self._calculate_cluster_distribution(test_assignments)
            
            # Calculate stability score (how similar are the distributions)
            stability_score = 0.0
            if train_dist and test_dist:
                # Calculate correlation between distributions
                common_regimes = set(train_dist.keys()) & set(test_dist.keys())
                if common_regimes:
                    train_values = [train_dist.get(regime, 0) for regime in common_regimes]
                    test_values = [test_dist.get(regime, 0) for regime in common_regimes]
                    
                    if NUMPY_AVAILABLE and len(train_values) > 1:
                        correlation = np.corrcoef(train_values, test_values)[0, 1]
                        stability_score = max(0, correlation) if not np.isnan(correlation) else 0
                    else:
                        # Simple similarity measure
                        diff = sum(abs(t - s) for t, s in zip(train_values, test_values))
                        stability_score = max(0, 1 - diff / len(common_regimes))
            
            return {
                'train_distribution': train_dist,
                'test_distribution': test_dist,
                'stability_score': stability_score,
                'is_stable': stability_score >= 0.7
            }
            
        except Exception as e:
            return {'error': f'Cross-validation failed: {e}'}
    
    def _analyze_regime_transitions(self, cluster_assignments: List[int]) -> Dict[str, Any]:
        """Analyze regime transition patterns."""
        if not cluster_assignments or len(cluster_assignments) < 2:
            return {'error': 'Insufficient data for transition analysis'}
        
        try:
            # Count transitions
            transitions = {}
            total_transitions = 0
            
            for i in range(1, len(cluster_assignments)):
                from_regime = cluster_assignments[i-1]
                to_regime = cluster_assignments[i]
                
                if from_regime != to_regime:
                    transition_key = f"{from_regime}->{to_regime}"
                    transitions[transition_key] = transitions.get(transition_key, 0) + 1
                    total_transitions += 1
            
            # Calculate transition probabilities
            transition_probs = {}
            for transition, count in transitions.items():
                from_regime = int(transition.split('->')[0])
                from_count = cluster_assignments.count(from_regime)
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
            recommendations.append("Consider increasing minimum regime duration to improve persistence")
        
        # Check economic significance
        economic_score = quality_metrics.get('economic_significance', {}).get('economic_significance_score', 0)
        if economic_score < 0.5:
            recommendations.append("Regimes may not be economically significant - consider feature engineering")
        
        # Check stability
        stability_score = quality_metrics.get('stability_analysis', {}).get('stability_score', 0)
        if stability_score < 0.7:
            recommendations.append("Regimes show low stability - consider cross-validation improvements")
        
        # Check transitions
        transition_entropy = quality_metrics.get('transition_analysis', {}).get('transition_entropy', 0)
        if transition_entropy > 2.0:
            recommendations.append("High transition entropy - consider regime smoothing or filtering")
        
        if not recommendations:
            recommendations.append("Regime quality is good - no specific recommendations")
        
        return recommendations
    
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
    
    async def _perform_kmeans_clustering(
        self, 
        prepared_data: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform K-means clustering on regime features."""
        try:
            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn not available for K-means clustering")
            
            market_data = prepared_data.get('market_data')
            if not PANDAS_AVAILABLE or not isinstance(market_data, pd.DataFrame):
                raise ValueError("Market data must be a pandas DataFrame for K-means clustering")
            
            # Extract regime features for clustering
            regime_features = self._extract_regime_features(market_data)
            
            # Configure K-means
            n_clusters = config.get('n_clusters', 3)
            random_state = config.get('random_state', 42)
            
            # Perform K-means clustering
            kmeans = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
                max_iter=300
            )
            
            cluster_assignments = kmeans.fit_predict(regime_features)
            
            # Calculate clustering metrics
            if len(set(cluster_assignments)) > 1:  # Need at least 2 clusters for metrics
                silhouette_avg = silhouette_score(regime_features, cluster_assignments)
                calinski_score = calinski_harabasz_score(regime_features, cluster_assignments)
            else:
                silhouette_avg = 0.0
                calinski_score = 0.0
            
            # Create mock HMM models for compatibility
            mock_models = [{'cluster_center': center, 'cluster_id': i} 
                          for i, center in enumerate(kmeans.cluster_centers_)]
            
            self.logger.info(f"✅ K-means clustering completed: {n_clusters} clusters")
            self.logger.info(f"📊 Silhouette score: {silhouette_avg:.3f}, Calinski-Harabasz score: {calinski_score:.3f}")
            
            return {
                'hmm_models': mock_models,
                'cluster_assignments': cluster_assignments.tolist(),
                'cluster_metrics': {
                    'clustering_method': 'kmeans',
                    'n_clusters': n_clusters,
                    'silhouette_score': silhouette_avg,
                    'calinski_harabasz_score': calinski_score,
                    'inertia': kmeans.inertia_,
                    'cluster_centers': kmeans.cluster_centers_.tolist()
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ K-means clustering failed: {e}")
            return {
                'hmm_models': [],
                'cluster_assignments': [],
                'cluster_metrics': {
                    'clustering_method': 'kmeans_fallback',
                    'error': str(e)
                }
            }
    
    async def _perform_multi_algorithm_clustering(
        self, 
        hmm_manager: Any, 
        prepared_data: Any, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform clustering using multiple algorithms and combine results."""
        try:
            # Run HMM clustering
            hmm_result = await self._perform_parallel_hmm_clustering(
                hmm_manager, prepared_data, config
            )
            
            # Run K-means clustering
            kmeans_result = await self._perform_kmeans_clustering(
                prepared_data, config
            )
            
            # Combine results using consensus
            consensus_result = self._combine_clustering_consensus(
                hmm_result, kmeans_result, config
            )
            
            self.logger.info("✅ Multi-algorithm consensus clustering completed")
            return consensus_result
            
        except Exception as e:
            self.logger.error(f"❌ Multi-algorithm clustering failed: {e}")
            # Fallback to HMM clustering
            return await self._perform_parallel_hmm_clustering(hmm_manager, prepared_data, config)
    
    def _extract_regime_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Extract features for K-means clustering from market data."""
        try:
            features = []
            
            # Price-based features
            if 'close' in market_data.columns:
                # Returns
                returns = market_data['close'].pct_change().dropna()
                features.append(returns.values)
                
                # Volatility (rolling standard deviation)
                volatility = returns.rolling(window=20).std().dropna()
                features.append(volatility.values)
                
                # Price momentum
                momentum = market_data['close'].pct_change(periods=5).dropna()
                features.append(momentum.values)
            
            # Volume-based features
            if 'volume' in market_data.columns:
                volume_returns = market_data['volume'].pct_change().dropna()
                features.append(volume_returns.values)
            
            # High-Low features
            if 'high' in market_data.columns and 'low' in market_data.columns:
                price_range = (market_data['high'] - market_data['low']) / market_data['close']
                features.append(price_range.dropna().values)
            
            # Combine features and handle different lengths
            if features:
                # Find minimum length to avoid NaN issues
                min_length = min(len(f) for f in features if len(f) > 0)
                if min_length > 0:
                    combined_features = np.column_stack([
                        f[:min_length] for f in features if len(f) >= min_length
                    ])
                    return combined_features
            
            # Fallback: use close prices only
            if 'close' in market_data.columns:
                close_prices = market_data['close'].values.reshape(-1, 1)
                return close_prices
            
            raise ValueError("No suitable features found for clustering")
            
        except Exception as e:
            self.logger.error(f"❌ Feature extraction failed: {e}")
            # Return dummy features
            return np.random.randn(len(market_data), 3)
    
    def _combine_clustering_consensus(
        self, 
        hmm_result: Dict[str, Any], 
        kmeans_result: Dict[str, Any], 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine HMM and K-means clustering results using consensus."""
        try:
            hmm_assignments = hmm_result.get('cluster_assignments', [])
            kmeans_assignments = kmeans_result.get('cluster_assignments', [])
            
            if not hmm_assignments or not kmeans_assignments:
                # Fallback to whichever method worked
                return hmm_result if hmm_assignments else kmeans_result
            
            # Ensure same length
            min_length = min(len(hmm_assignments), len(kmeans_assignments))
            hmm_assignments = hmm_assignments[:min_length]
            kmeans_assignments = kmeans_assignments[:min_length]
            
            # Calculate consensus assignments
            consensus_assignments = []
            consensus_weights = []
            
            for i in range(min_length):
                hmm_cluster = hmm_assignments[i]
                kmeans_cluster = kmeans_assignments[i]
                
                # Simple consensus: if both agree, use that cluster
                if hmm_cluster == kmeans_cluster:
                    consensus_assignments.append(hmm_cluster)
                    consensus_weights.append(1.0)
                else:
                    # If they disagree, use the more common cluster in the dataset
                    hmm_count = hmm_assignments.count(hmm_cluster)
                    kmeans_count = kmeans_assignments.count(kmeans_cluster)
                    
                    if hmm_count >= kmeans_count:
                        consensus_assignments.append(hmm_cluster)
                        consensus_weights.append(0.5)  # Lower confidence
                    else:
                        consensus_assignments.append(kmeans_cluster)
                        consensus_weights.append(0.5)  # Lower confidence
            
            # Calculate consensus metrics
            agreement_rate = sum(1 for w in consensus_weights if w == 1.0) / len(consensus_weights)
            
            # Combine models (prefer HMM models as they're more sophisticated)
            combined_models = hmm_result.get('hmm_models', [])
            if not combined_models:
                combined_models = kmeans_result.get('hmm_models', [])
            
            # Combine metrics
            combined_metrics = {
                'clustering_method': 'multi_algorithm_consensus',
                'hmm_metrics': hmm_result.get('cluster_metrics', {}),
                'kmeans_metrics': kmeans_result.get('cluster_metrics', {}),
                'agreement_rate': agreement_rate,
                'consensus_confidence': np.mean(consensus_weights) if NUMPY_AVAILABLE else sum(consensus_weights) / len(consensus_weights)
            }
            
            self.logger.info(f"📊 Consensus clustering: {agreement_rate:.1%} agreement between HMM and K-means")
            
            return {
                'hmm_models': combined_models,
                'cluster_assignments': consensus_assignments,
                'cluster_metrics': combined_metrics
            }
            
        except Exception as e:
            self.logger.error(f"❌ Consensus combination failed: {e}")
            # Fallback to HMM result
            return hmm_result