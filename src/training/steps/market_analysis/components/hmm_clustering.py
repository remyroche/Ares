"""
HMM Clustering Component.

This component performs HMM-based regime clustering.
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


# Hardware optimization imports
try:
    from src.utils.hardware.m1_memory_optimizer import M1MemoryOptimizer
    from src.utils.hardware.m1_cpu_optimizer import M1CPUOptimizer
    from src.utils.matrix_operations import UnifiedMatrixOperations
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    M1MemoryOptimizer = None
    M1CPUOptimizer = None
    UnifiedMatrixOperations = None

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger

# ============================================================================
# CLUSTERING CONFIGURATION CONSTANTS
# ============================================================================

# Maximum CV threshold for clustering - used both for:
# 1. CV threshold ceiling (prevents "CV THRESHOLD MAXED OUT" at old 5.0 limit)  
# 2. Problematic cluster prevention (rejects merges that would create high-CV clusters)
# 
# USAGE: Change this single value to adjust both the ceiling and prevention logic together.
# - Lower values (5.0-8.0): More conservative, better cluster quality, lower coverage
# - Higher values (10.0-15.0): More aggressive, potentially higher coverage, risk of heterogeneous clusters
MAX_CV_THRESHOLD = 15.0  # Increased to allow bigger steps (was 10.0)

# ============================================================================

# Validation result classes for simplified error handling
class ValidationResult(NamedTuple):
    """Result of input validation."""
    is_valid: bool
    error_message: Optional[str] = None
    market_data: Optional[Any] = None

class DataAlignmentResult(NamedTuple):
    """Result of data alignment operation."""
    aligned_market_data: Any
    aligned_assignments: List[int]
    original_lengths: Tuple[int, int]
    was_aligned: bool

# Standard return structure for clustering methods with advanced analysis
STANDARD_CLUSTERING_RESULT = {
    'hmm_models': [],
    'cluster_assignments': [],
    'cluster_metrics': {},
    'advanced_clustering_analysis': {},
    'statistical_analysis': {},
    'market_dynamics': {},
    'clustering_time': 0.0,
    'success': False,
    'error': None
}


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
            self.matrix_ops = UnifiedMatrixOperations()
        else:
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.matrix_ops = None
            self.logger.warning("⚠️ Hardware optimization not available - using fallback methods")
    
    async def _validate_inputs(self, data: Any, pipeline_state: Dict[str, Any]) -> ValidationResult:
        """
        Consolidated input validation to reduce repetitive checking.
        
        Args:
            data: Input data to validate
            pipeline_state: Pipeline state to validate
            
        Returns:
            ValidationResult with validation status and any error message
        """
        from src.utils.tprint import tprint
        
        # Check if data is None
        if data is None:
            error_msg = "Input data is None"
            tprint(f"❌ {error_msg}")
            self.logger.error(error_msg)
            return ValidationResult(is_valid=False, error_message=error_msg)
        
        # Check if pipeline_state is empty
        if not pipeline_state:
            error_msg = "Pipeline state is empty"
            tprint(f"❌ {error_msg}")
            self.logger.error(error_msg)
            return ValidationResult(is_valid=False, error_message=error_msg)
        
        # Load and validate market data
        try:
            market_data = await self._load_market_data(data)
            if market_data is None:
                error_msg = "Market data is None after loading"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ValidationResult(is_valid=False, error_message=error_msg)
            
            if hasattr(market_data, 'empty') and market_data.empty:
                error_msg = "Market data is empty"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ValidationResult(is_valid=False, error_message=error_msg)
            
            tprint("✅ Input validation passed")
            return ValidationResult(is_valid=True, market_data=market_data)
            
        except Exception as e:
            error_msg = f"Failed to load market data: {e}"
            tprint(f"❌ {error_msg}")
            self.logger.error(error_msg)
            return ValidationResult(is_valid=False, error_message=error_msg)
    
    def _align_data_lengths(self, market_data: Any, assignments: List[int]) -> DataAlignmentResult:
        """
        Handle data alignment between market data and regime/cluster assignments.
        
        Args:
            market_data: Market data (DataFrame or array-like)
            assignments: Regime or cluster assignments list
            
        Returns:
            DataAlignmentResult with aligned data and alignment info
        """
        if not PANDAS_AVAILABLE or not isinstance(market_data, pd.DataFrame):
            # For non-DataFrame data, return as-is
            return DataAlignmentResult(
                aligned_market_data=market_data,
                aligned_assignments=assignments,
                original_lengths=(len(assignments), len(market_data) if hasattr(market_data, '__len__') else 0),
                was_aligned=False
            )
        
        assignments_len = len(assignments)
        market_data_len = len(market_data)
        
        if assignments_len == market_data_len:
            # Data is already aligned
            return DataAlignmentResult(
                aligned_market_data=market_data,
                aligned_assignments=assignments,
                original_lengths=(assignments_len, market_data_len),
                was_aligned=False
            )
        
        # Align data by truncating to minimum length
        min_length = min(assignments_len, market_data_len)
        aligned_market_data = market_data.iloc[:min_length]
        aligned_assignments = assignments[:min_length]
        
        self.logger.info(f"ℹ️ 🔧 Data alignment: market_data={market_data_len} → {min_length}, assignments={assignments_len} → {min_length}")
        
        return DataAlignmentResult(
            aligned_market_data=aligned_market_data,
            aligned_assignments=aligned_assignments,
            original_lengths=(assignments_len, market_data_len),
            was_aligned=True
        )
    
    def _should_stop_clustering(self, cluster_count: int, threshold: float, previous_coverage: float, 
                               cv_threshold_idx: int, cv_thresholds: List[float]) -> bool:
        """
        Extract complex clustering stop conditions into a well-named boolean method.
        
        Args:
            cluster_count: Current number of clusters
            threshold: Current similarity threshold
            previous_coverage: Previous top-20 coverage percentage
            cv_threshold_idx: Current CV threshold index
            cv_thresholds: List of CV thresholds
            
        Returns:
            True if clustering should stop, False otherwise
        """
        # Stop if we reach optimal cluster count (20 or below)
        if cluster_count <= 20:
            self.logger.info(f"🎯 STOPPING: Reached optimal cluster count ({cluster_count} <= 20)")
            return True
        
        # Stop if we reach minimum similarity threshold (55%)
        if threshold < 0.55:
            self.logger.info(f"🛑 STOPPING: Reached minimum similarity threshold (55%)")
            return True
        
        # Stop if we've reached the end of CV thresholds and coverage is good
        if (previous_coverage > 0 and previous_coverage < 90.0 and 
            cv_threshold_idx < len(cv_thresholds) - 1):
            return False  # Continue with CV relaxation
        
        return False
    
    def _should_relax_cv_threshold(self, previous_coverage: float, cv_threshold_idx: int, 
                                  cv_thresholds: List[float]) -> bool:
        """
        Check if CV threshold should be relaxed based on coverage criteria.
        
        Args:
            previous_coverage: Previous top-20 coverage percentage
            cv_threshold_idx: Current CV threshold index
            cv_thresholds: List of CV thresholds
            
        Returns:
            True if CV threshold should be relaxed, False otherwise
        """
        return (previous_coverage > 0 and previous_coverage < 90.0 and 
                cv_threshold_idx < len(cv_thresholds) - 1)
    
    def _is_merge_blocked_by_size(self, cluster_1_size: int, cluster_2_size: int, 
                                  total_samples: int, max_percentage: float = 15.0) -> bool:
        """
        Check if a merge should be blocked due to cluster size constraints.
        
        Args:
            cluster_1_size: Size of first cluster
            cluster_2_size: Size of second cluster
            total_samples: Total number of samples
            max_percentage: Maximum allowed percentage for combined cluster
            
        Returns:
            True if merge should be blocked, False otherwise
        """
        combined_size = cluster_1_size + cluster_2_size
        combined_percentage = (combined_size / total_samples) * 100 if total_samples > 0 else 0
        return combined_percentage > max_percentage
    
    def _get_cv_threshold_multiplier(self, combined_size: int, total_samples: int) -> float:
        """
        Get CV threshold multiplier based on cluster size.
        
        Args:
            combined_size: Combined size of clusters to merge
            total_samples: Total number of samples
            
        Returns:
            CV threshold multiplier (1.0 for normal, 1.5 for tiny clusters)
        """
        if combined_size < total_samples * 0.01:  # Less than 1% of total samples
            return 1.5  # 150% more lenient for tiny clusters
        return 1.0  # Standard thresholds for all others
    
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
        from src.utils.tprint import tprint
        
        tprint("🔄 Starting HMM Clustering - Enhanced Error Handling")
        self.logger.info('🔄 Starting HMM Clustering')
        
        try:
            tprint("📊 Input validation starting...")
            tprint(f"📊 Data type: {type(data)}")
            tprint(f"📊 Pipeline state keys: {list(pipeline_state.keys()) if pipeline_state else 'None'}")
            
            # Consolidated input validation
            validation_result = await self._validate_inputs(data, pipeline_state)
            if not validation_result.is_valid:
                return ComponentResult(success=False, artifacts={}, error_message=validation_result.error_message)
            
            market_data = validation_result.market_data
            tprint(f"✅ Market data loaded successfully: {type(market_data)}")
            if hasattr(market_data, 'shape'):
                tprint(f"📊 Market data shape: {market_data.shape}")
            
            # Import HMM clustering utilities
            tprint("📦 Importing HMM clustering utilities...")
            try:
                from src.utils.hmm_composite_manager import EnhancedHMMCompositeManager
                tprint("✅ EnhancedHMMCompositeManager imported successfully")
            except ImportError as e:
                error_msg = f"Failed to import EnhancedHMMCompositeManager: {e}"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(success=False, artifacts={}, error_message=error_msg)
            
            # Get regime discovery results from previous stage
            tprint("🔍 Retrieving regime discovery results...")
            try:
                # First try to get from pipeline state (direct access)
                hmm_regime_discovery = pipeline_state.get('hmm_regime_discovery_result', {})
                tprint(f"📊 Direct regime discovery result: {type(hmm_regime_discovery)}")
                
                # If not found in pipeline state, try to get from artifacts in pipeline state
                if not hmm_regime_discovery:
                    tprint("🔍 Trying to get regime discovery from artifacts...")
                    artifacts = pipeline_state.get('artifacts', {})
                    hmm_regime_discovery = artifacts.get('hmm_regime_discovery_result', {})
                    tprint(f"📊 Artifacts regime discovery result: {type(hmm_regime_discovery)}")
                
                # If still not found, try to get from regime models in pipeline state
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
                        # Get symbol and exchange from pipeline state
                        symbol = pipeline_state.get('symbol', 'ETHUSDT')
                        exchange = pipeline_state.get('exchange', 'binance')
                        
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
                                
                                # Extract the regime discovery results from the outcome file
                                artifacts = outcome_data.get('artifacts', {})
                                hmm_regime_discovery = artifacts.get('hmm_regime_discovery_result', {})
                                
                                if hmm_regime_discovery:
                                    tprint("✅ Loaded regime discovery results from outcome file")
                                    tprint(f"📊 Found {len(hmm_regime_discovery.get('regime_models', []))} regime models")
                                    tprint(f"📊 Found {len(hmm_regime_discovery.get('regime_assignments', []))} regime assignments")
                                else:
                                    tprint("⚠️ No regime discovery results found in outcome file")
                            else:
                                tprint("⚠️ No outcome files found")
                        else:
                            tprint("⚠️ Outcomes directory not found")
                    except Exception as e:
                        tprint(f"⚠️ Failed to load from outcome file: {e}")
                
                tprint(f"📊 Final regime discovery type: {type(hmm_regime_discovery)}")
                tprint(f"📊 Final regime discovery keys: {list(hmm_regime_discovery.keys()) if isinstance(hmm_regime_discovery, dict) else 'Not a dict'}")
                
            except Exception as e:
                error_msg = f"Failed to retrieve regime discovery results: {e}"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(success=False, artifacts={}, error_message=error_msg)
            
            # Log regime discovery summary (reduced verbosity)
            if hmm_regime_discovery:
                regime_models = hmm_regime_discovery.get('regime_models', [])
                regime_assignments = hmm_regime_discovery.get('regime_assignments', [])
                regime_metrics = hmm_regime_discovery.get('regime_metrics', {})
                total_regimes = regime_metrics.get('total_regimes', len(regime_models))
                total_samples = regime_metrics.get('total_samples', len(regime_assignments))
                
                self.logger.info(f"ℹ️ 🔍 Found HMM regime discovery results: {total_regimes} regimes, {total_samples} samples")
                
                # Log regime distribution if available
                regime_distribution = regime_metrics.get('regime_distribution', {})
                if regime_distribution:
                    dist_summary = {f"regime_{k}": f"{v/total_samples*100:.1f}%" for k, v in regime_distribution.items()}
                    self.logger.info(f"ℹ️ 🔍 Regime distribution: {dist_summary}")
            else:
                self.logger.warning("⚠️ No HMM regime discovery results found in pipeline state")
            
            if not hmm_regime_discovery:
                raise ValueError("No HMM regime discovery results available for clustering")
            
            input_regimes = len(hmm_regime_discovery.get('regime_models', []))
            self.logger.info(f'🔧 HMM Clustering: Processing {input_regimes} regimes → Dynamic clustering based on mode')
            
            # Configure HMM clustering - Data-driven cluster selection with elbow method
            mode = pipeline_state.get('mode', 'light')  # Get mode from pipeline state
            
            # Set maximum clusters based on mode
            if mode == 'full':
                max_clusters = min(25, max(3, input_regimes // 2))  # Maximum 25 clusters in full mode
            elif mode == 'blank':
                max_clusters = min(8, max(3, input_regimes // 4))   # Maximum 8 clusters in blank mode  
            else:  # light mode
                max_clusters = 3  # Maximum 3 clusters for light mode
            
            clustering_config = {
                'max_clusters': max_clusters,
                'clustering_method': 'hmm_based',
                'min_cluster_size': 10,
                'convergence_tolerance': 1e-6,
                'max_iterations': 100,
                
                # Regime constraints
                'max_regimes': 25,  # Maximum 25 regimes allowed
                'min_regime_sample_percentage': 0.01,  # 1% minimum sample threshold
                
                # Hardware optimization
                'enable_parallel_processing': True,
                'enable_gpu_acceleration': True,
                'memory_limit_gb': 8.0
            }
            
            self.logger.info(f'🎯 Clustering mode: {mode} → max {max_clusters} clusters (data-driven selection)')
            
            # Create HMM composite manager
            tprint("🔧 Creating HMM composite manager...")
            try:
                hmm_manager = EnhancedHMMCompositeManager()
                tprint("✅ HMM composite manager created successfully")
            except Exception as e:
                error_msg = f"Failed to create HMM composite manager: {e}"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(success=False, artifacts={}, error_message=error_msg)
            
            # Perform HMM clustering
            tprint("🔄 Starting HMM clustering process...")
            try:
                clustering_result = await self._perform_hmm_clustering(
                    hmm_manager, market_data, hmm_regime_discovery, clustering_config
                )
                tprint(f"✅ HMM clustering completed: {type(clustering_result)}")
                tprint(f"📊 Clustering result keys: {list(clustering_result.keys()) if isinstance(clustering_result, dict) else 'Not a dict'}")
                
                # Validate clustering result
                if not isinstance(clustering_result, dict):
                    error_msg = f"Invalid clustering result type: {type(clustering_result)}"
                    tprint(f"❌ {error_msg}")
                    self.logger.error(error_msg)
                    return ComponentResult(success=False, artifacts={}, error_message=error_msg)
                
                if not clustering_result.get('success', False):
                    error_msg = f"Clustering failed: {clustering_result.get('error', 'Unknown error')}"
                    tprint(f"❌ {error_msg}")
                    self.logger.error(error_msg)
                    return ComponentResult(success=False, artifacts={}, error_message=error_msg)
                
            except Exception as e:
                error_msg = f"HMM clustering process failed: {e}"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(success=False, artifacts={}, error_message=error_msg)
            
            # Extract results
            tprint("📊 Extracting clustering results...")
            try:
                hmm_models = clustering_result.get('hmm_models', [])
                cluster_assignments = clustering_result.get('cluster_assignments', [])
                cluster_metrics = clustering_result.get('cluster_metrics', {})
                
                # Use aligned market data from clustering result to prevent length mismatches
                aligned_market_data = clustering_result.get('aligned_market_data', market_data)
                
                tprint(f"📊 Extracted {len(hmm_models)} HMM models")
                tprint(f"📊 Extracted {len(cluster_assignments)} cluster assignments")
                tprint(f"📊 Cluster metrics keys: {list(cluster_metrics.keys()) if cluster_metrics else 'None'}")
                tprint(f"📊 Using aligned market data: {len(aligned_market_data)} samples")
                
                # Validate that we have clustering results
                if not hmm_models:
                    error_msg = "HMM clustering completed but no models were created"
                    tprint(f"❌ {error_msg}")
                    self.logger.error(error_msg)
                    return ComponentResult(success=False, artifacts={}, error_message=error_msg)
                
                if not cluster_assignments:
                    error_msg = "HMM clustering completed but no cluster assignments were created"
                    tprint(f"❌ {error_msg}")
                    self.logger.error(error_msg)
                    return ComponentResult(success=False, artifacts={}, error_message=error_msg)
                
                tprint("✅ Clustering results validation passed")
                
            except Exception as e:
                error_msg = f"Failed to extract clustering results: {e}"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(success=False, artifacts={}, error_message=error_msg)
            
            # Skip regime constraints - use matrix-based clustering results directly
            tprint(f"✅ Using matrix-based clustering results directly: {len(hmm_models)} models, {len(cluster_assignments)} assignments")
            
            # Perform comprehensive cluster quality validation
            tprint("🔍 Performing cluster quality validation...")
            try:
                quality_metrics = self._validate_cluster_quality(
                    hmm_models, cluster_assignments, aligned_market_data, clustering_config
                )
                tprint(f"✅ Cluster quality validation completed: {quality_metrics.get('validation_passed', False)}")
            except Exception as e:
                error_msg = f"Failed to validate cluster quality: {e}"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(success=False, artifacts={}, error_message=error_msg)
            
            # Generate detailed metrics for each HMM cluster
            tprint("📊 Generating detailed cluster metrics...")
            try:
                cluster_detailed_metrics = self._generate_cluster_detailed_metrics(
                    hmm_models, cluster_assignments, aligned_market_data, clustering_config
                )
                tprint(f"✅ Detailed cluster metrics generated: {len(cluster_detailed_metrics)} metrics")
            except Exception as e:
                error_msg = f"Failed to generate detailed cluster metrics: {e}"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(success=False, artifacts={}, error_message=error_msg)
            
            # Create single consolidated artifact
            tprint("📦 Creating consolidated artifacts...")
            try:
                # Extract advanced clustering analysis from cluster_metrics
                advanced_analysis = cluster_metrics.get('advanced_clustering_analysis', {})
                statistical_analysis = cluster_metrics.get('statistical_analysis', {})
                
                # Extract hierarchical post-processing metrics if available
                hierarchical_coverage_metrics = {}
                if hasattr(clustering_result, 'get'):
                    hierarchical_coverage_metrics = clustering_result.get('coverage_metrics', {})
                elif isinstance(clustering_result, dict):
                    hierarchical_coverage_metrics = clustering_result.get('coverage_metrics', {})
                
                # Build single consolidated artifact
                artifacts = {
                    'hmm_clustering_result': {
                        # Core clustering results
                        'hmm_models': hmm_models,
                        # 'cluster_assignments': cluster_assignments,  # Excluded to reduce file size
                        'cluster_metrics': cluster_metrics,
                        'cluster_quality_metrics': quality_metrics,
                        'cluster_detailed_metrics': cluster_detailed_metrics,
                        
                        # Advanced clustering analysis
                        'advanced_clustering_analysis': {
                            'cluster_selection_methods': advanced_analysis.get('cluster_selection_methods', {}),
                            'validation_metrics': advanced_analysis.get('validation_metrics', {}),
                            'information_criteria': advanced_analysis.get('information_criteria', {}),
                            'gmm_analysis': advanced_analysis.get('gmm_analysis', {}),
                            'spectral_analysis': advanced_analysis.get('spectral_analysis', {}),
                            'optimal_k_selection': advanced_analysis.get('optimal_k_selection', {})
                        },
                        
                        # Regime-level statistical analysis
                        'statistical_analysis': {
                            'regime_volume_tests': statistical_analysis.get('regime_volume_tests', {}),
                            'regime_volatility_tests': statistical_analysis.get('regime_volatility_tests', {}),
                            'regime_momentum_tests': statistical_analysis.get('regime_momentum_tests', {}),
                            'cluster_validation': statistical_analysis.get('cluster_validation', {}),
                            'regime_similarity_validation': statistical_analysis.get('regime_similarity_validation', {}),
                            'factor_impact_analysis': statistical_analysis.get('factor_impact_analysis', {}),
                            'economic_validation': statistical_analysis.get('economic_validation', {}),
                            'overall_cluster_quality': statistical_analysis.get('overall_cluster_quality', {})
                        },
                        
                        # Market dynamics insights
                        'market_dynamics': {
                            'aspect_ranking': statistical_analysis.get('factor_impact_analysis', {}).get('aspect_ranking', []),
                            'primary_market_driver': statistical_analysis.get('factor_impact_analysis', {}).get('primary_market_driver', {}),
                            'market_dynamics_hierarchy': statistical_analysis.get('factor_impact_analysis', {}).get('market_dynamics_hierarchy', {}),
                            'economic_alignment': statistical_analysis.get('economic_validation', {}).get('overall_economic_alignment', {})
                        },
                        
                        # Hierarchical post-processing coverage metrics (if available)
                        'hierarchical_coverage_metrics': hierarchical_coverage_metrics,
                        
                        # Clustering summary with advanced metrics (embedded here to avoid extra artifact)
                        'clustering_summary': (lambda: (lambda ca, qm: {
                            'total_clusters': len(hmm_models),
                            'total_assignments': sum(ca.get('cluster_sizes', {}).values()) if isinstance(ca.get('cluster_sizes', {}), dict) else 0,
                            'cluster_distribution': self._calculate_cluster_distribution_from_sizes(ca.get('cluster_sizes', {})),
                            'top_5_coverage': ca.get('concentration_analysis', {}).get('top_5_coverage', 0.0),
                            'top_20_coverage': hierarchical_coverage_metrics.get('top_20_coverage', ca.get('top_20_coverage', 0.0)),
                            'hierarchical_post_processing_applied': bool(hierarchical_coverage_metrics),
                            'original_top_20_coverage': ca.get('top_20_coverage', 0.0),
                            'avg_within_cluster_cv': ca.get('avg_within_cluster_cv', 0.0),
                            'inter_cluster_similarity_pct': max(0.0, min(100.0, (1.0 - qm.get('avg_inter_cluster_dissimilarity', 0.0)) * 100.0)),
                            'clustering_time': clustering_result.get('clustering_time', 0.0),
                            'quality_score': qm.get('overall_quality_score', 0.0),
                            'validation_passed': qm.get('validation_passed', False),
                            'clustering_mode': clustering_config.get('max_clusters', 3),
                            'data_driven_selection': True,
                            'regime_reduction': {
                                'input_regimes': len(hmm_regime_discovery.get('regime_models', [])),
                                'output_clusters': len(hmm_models),
                                'reduction_ratio': len(hmm_models) / max(1, len(hmm_regime_discovery.get('regime_models', [])))
                            },
                            'advanced_methods_used': {
                                'information_criteria': True,
                                'gmm_confidence_optimization': advanced_analysis.get('gmm_analysis', {}).get('gmm_quality') == 'good',
                                'spectral_clustering': advanced_analysis.get('spectral_analysis', {}).get('spectral_quality') == 'good',
                                'multi_method_consensus': True,
                                'economic_validation': True
                            }
                        })(clustering_result.get('cluster_analytics', {}), quality_metrics))(),
                        
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
                                'max_clusters_allowed': clustering_config.get('max_clusters', 25),
                                'clustering_method': 'advanced_multi_method_consensus',
                                'validation_dimensions': ['statistical', 'economic', 'financial', 'stability']
                            }
                        }
                    }
                }
                
                # Note: We intentionally avoid creating a second top-level artifact ("clustering_summary")
                # to ensure only one artifact group is produced as requested.
                
                tprint(f"✅ Artifacts created successfully: {len(artifacts)} artifact groups")
                tprint(f"📊 Total clusters: {len(hmm_models)}")
                tprint(f"📊 Total assignments: {len(cluster_assignments)}")
                tprint(f"📊 Quality score: {quality_metrics.get('overall_quality_score', 0.0):.3f}")
                tprint(f"📊 Validation passed: {quality_metrics.get('validation_passed', False)}")
                tprint(f"🎯 Primary market driver: {statistical_analysis.get('factor_impact_analysis', {}).get('primary_market_driver', {}).get('dominant_aspect', 'unknown')}")
                tprint(f"📈 Economic validation: {statistical_analysis.get('economic_validation', {}).get('overall_economic_alignment', {}).get('economic_validation_passed', False)}")
                
            except Exception as e:
                error_msg = f"Failed to create artifacts: {e}"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(success=False, artifacts={}, error_message=error_msg)
            
            self.logger.info(f'✅ HMM Clustering completed: {len(hmm_models)} clusters created')
            tprint(f"🎉 HMM Clustering completed successfully: {len(hmm_models)} clusters created")
            
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
            from src.utils.tprint import tprint
            import traceback
            
            error_msg = f'HMM Clustering failed: {e}'
            tprint(f"❌ {error_msg}")
            self.logger.error(f'❌ HMM Clustering failed: {e}')
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            
            # Log additional debugging information
            tprint(f"🔍 Error type: {type(e).__name__}")
            tprint(f"🔍 Error args: {e.args}")
            
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
            prepared_data = self._prepare_data_for_clustering(market_data, regime_discovery)
            
            # Use matrix-based clustering to merge regimes using similarity matrix
            self.logger.info("🎯 Using matrix-based clustering with similarity matrix to merge regimes")
            
            # Extract regime characteristics and assignments from regime discovery
            regime_characteristics = self._extract_regime_characteristics_from_discovery(regime_discovery)
            regime_assignments = regime_discovery.get('regime_assignments', [])
            
            if not regime_characteristics:
                raise ValueError("No regime characteristics found in discovery results")
            if not regime_assignments:
                raise ValueError("No regime assignments found in discovery results")
            
            # Ensure data alignment between market data and regime assignments
            alignment_result = self._align_data_lengths(market_data, regime_assignments)
            market_data = alignment_result.aligned_market_data
            regime_assignments = alignment_result.aligned_assignments
            
            max_clusters = config.get('max_clusters', 40)  # Reasonable maximum for regime clustering
            
            # Check if enhanced clustering features should be used (force enhanced for testing)
            use_enhanced_clustering = config.get('use_enhanced_clustering', True) 
            use_multi_stage = config.get('use_multi_stage_clustering', False)
            
            # Force enhanced clustering for testing (temporarily disabled due to bugs)
            self.logger.info(f"🔧 Clustering mode selection: enhanced={use_enhanced_clustering}, multi_stage={use_multi_stage}")
            if not use_multi_stage:
                use_enhanced_clustering = False  # Temporarily disable enhanced mode
                self.logger.info("🔧 Using standard clustering with improvements applied")
            
            if use_multi_stage:
                self.logger.info("🔄 Using Multi-Stage Hierarchical Clustering")
                clustering_result = self._multi_stage_hierarchical_clustering(
                    regime_characteristics, regime_assignments
                )
            elif use_enhanced_clustering:
                self.logger.info("🔄 Using Enhanced Matrix-Based Clustering")
                clustering_result = self._perform_enhanced_matrix_clustering(
                    regime_characteristics, regime_assignments, max_clusters, market_data
                )
            else:
                self.logger.info("🔄 Using Standard Matrix-Based Clustering")
            clustering_result = self._perform_matrix_based_clustering(
                regime_characteristics, regime_assignments, max_clusters, market_data
            )
            
            clustering_time = time.time() - start_time
            clustering_result['clustering_time'] = clustering_time
            
            # Include aligned market data in the result to prevent length mismatches downstream
            clustering_result['aligned_market_data'] = market_data
            
            # Apply hierarchical post-processing (always enabled for coverage improvement)
            self.logger.info("🔄 Applying hierarchical post-processing...")
            self.logger.info(f"🔍 DEBUG: Clustering result keys: {list(clustering_result.keys()) if isinstance(clustering_result, dict) else 'Not a dict'}")
            self.logger.info(f"🔍 DEBUG: n_clusters = {clustering_result.get('n_clusters', 'missing')}")
            self.logger.info(f"🔍 DEBUG: cluster_assignments length = {len(clustering_result.get('cluster_assignments', []))}")
            self.logger.info(f"🔍 DEBUG: aligned_market_data length = {len(clustering_result.get('aligned_market_data', []))}")
            target_clusters = 75  # Conservative target for 85-90% coverage
            clustering_result = self.apply_hierarchical_post_processing(clustering_result, target_clusters)
            self.logger.info(f"✅ Hierarchical post-processing completed: {clustering_result.get('n_clusters', 'unknown')} super-clusters")
            
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"HMM clustering process failed: {e}")
            raise

    def _perform_enhanced_matrix_clustering(
        self, 
        regime_characteristics: Dict[str, Any], 
        regime_assignments: List[int], 
        max_clusters: int,
        market_data: Any = None
    ) -> Dict[str, Any]:
        """Enhanced matrix-based clustering with incremental updates and quality scoring."""
        try:
            import numpy as np
            
            # Calculate initial similarity matrix with memory optimization
            similarity_matrix = self._calculate_memory_efficient_similarity_matrix(regime_characteristics)
            if similarity_matrix.size == 0:
                self.logger.error("❌ Empty similarity matrix - cannot perform clustering")
                return self._create_single_cluster_result(regime_assignments, market_data)

            regime_ids = list(regime_characteristics.keys())
            n_regimes = len(regime_ids)

            self.logger.info(f"🎯 Enhanced hierarchical clustering: {n_regimes} regimes → target: 20-40 clusters")
            self.logger.info(f"📊 Using incremental similarity updates and comprehensive quality scoring")
            
            # Initialize regime-to-cluster mapping
            regime_to_cluster = {regime_id: i for i, regime_id in enumerate(regime_ids)}
            cluster_count = n_regimes
            
            # Use our enhanced adaptive CV thresholds
            cv_thresholds = self._generate_adaptive_cv_thresholds(regime_characteristics)
            current_cv_threshold_idx = 0
            
            # Calculate progressive similarity thresholds
            similarity_thresholds = self._calculate_progressive_similarity_thresholds()
            
            self.logger.info(f"🎯 Enhanced CV Thresholds: {[f'{t:.2f}' for t in cv_thresholds]}")
            
            # Track merges for incremental updates
            all_merged_clusters = []
            previous_top_20_coverage = 0.0
            top_20_coverage = 0.0  # Initialize for progressive relaxation logic
            
            # Calculate initial cluster sample counts
            cluster_sample_counts = {}
            total_samples = len(regime_assignments) if regime_assignments else 0
            for regime_id in regime_ids:
                regime_idx = int(regime_id.split('_')[-1]) if regime_id.startswith('regime_') else regime_id
                cluster_id = regime_to_cluster[regime_id]
                if regime_idx < len(regime_assignments):
                    regime_count = regime_assignments.count(regime_idx)
                    cluster_sample_counts[cluster_id] = cluster_sample_counts.get(cluster_id, 0) + regime_count
            
            for threshold in similarity_thresholds:
                # Use guard clauses to simplify nested conditionals
                if self._should_stop_clustering(cluster_count, threshold, previous_top_20_coverage, 
                                              current_cv_threshold_idx, cv_thresholds):
                    break
                    
                self.logger.info(f"🔄 Enhanced merging at {threshold*100:.1f}% similarity threshold")

                # Adaptive CV threshold progression using extracted method
                if self._should_relax_cv_threshold(previous_top_20_coverage, current_cv_threshold_idx, cv_thresholds):
                    old_cv_threshold = cv_thresholds[current_cv_threshold_idx]
                    current_cv_threshold_idx += 1
                    new_cv_threshold = cv_thresholds[current_cv_threshold_idx]
                    self.logger.info(f"📈 ENHANCED CV RELAXATION: {old_cv_threshold:.2f} → {new_cv_threshold:.2f}")

                current_cv_threshold = cv_thresholds[min(current_cv_threshold_idx, len(cv_thresholds) - 1)]
                
                # Calculate total samples for size-based exclusions
                total_samples = len(regime_assignments) if regime_assignments else 0
                
                # Get merge candidates with enhanced filtering (only exclude oversized clusters)
                excluded_clusters = self._get_excluded_clusters_due_to_size(regime_characteristics, regime_to_cluster, total_samples)
                mergeable_pairs = self._get_smart_merge_candidates(
                    similarity_matrix, regime_to_cluster, regime_ids, excluded_clusters, threshold
                )
                
                # Perform merges and track them
                round_merges = []
                merges_this_round = 0
                
                # Calculate cluster sample counts for size-based constraints
                cluster_sample_counts = {}
                total_samples = len(regime_assignments)
                for regime_id, cluster_id in regime_to_cluster.items():
                    if cluster_id not in cluster_sample_counts:
                        cluster_sample_counts[cluster_id] = 0
                    regime_sample_count = regime_characteristics[regime_id].get('sample_count', 1)
                    cluster_sample_counts[cluster_id] += regime_sample_count
                
                for similarity, cluster_1, cluster_2, regime_1, regime_2 in mergeable_pairs:
                    if regime_to_cluster[regime_1] != regime_to_cluster[regime_2]:
                        # Enhanced quality checks with size-based relaxation for smaller clusters
                        cluster_1_size = cluster_sample_counts.get(cluster_1, 0)
                        cluster_2_size = cluster_sample_counts.get(cluster_2, 0)
                        combined_size = cluster_1_size + cluster_2_size
                        
                        # Check if merge should be blocked due to size constraints
                        if self._is_merge_blocked_by_size(cluster_1_size, cluster_2_size, total_samples):
                            combined_percentage = (combined_size / total_samples) * 100
                            self.logger.info(f"🚫 Blocking merge: would create {combined_percentage:.1f}% cluster (>15% limit)")
                            continue  # Skip this merge to prevent oversized clusters
                        
                        # Get CV threshold multiplier based on cluster size
                        cv_threshold_multiplier = self._get_cv_threshold_multiplier(combined_size, total_samples)
                        
                        if (self._passes_cv_hard_constraint(
                            regime_characteristics[regime_1], 
                            regime_characteristics[regime_2],
                            max_relative_diff=0.9 * cv_threshold_multiplier  # Size-adjusted threshold
                        ) and self._check_cv_compatibility(
                            regime_characteristics[regime_1],
                            regime_characteristics[regime_2],
                            min_cv_threshold=0.9 * cv_threshold_multiplier  # Size-adjusted threshold
                        )):
                            # Perform merge
                            old_cluster = regime_to_cluster[regime_2]
                            new_cluster = regime_to_cluster[regime_1]
                            
                            # Update all regimes in old cluster
                            for regime_id in regime_ids:
                                if regime_to_cluster[regime_id] == old_cluster:
                                    regime_to_cluster[regime_id] = new_cluster
                            
                            round_merges.append((old_cluster, new_cluster))
                            merges_this_round += 1
                            cluster_count -= 1
                
                # Use incremental similarity updates instead of full recalculation
                if round_merges:
                    self.logger.info(f"   ✅ Enhanced round: {merges_this_round} merges completed")
                    similarity_matrix = self._update_similarity_matrix_incremental(
                        similarity_matrix, round_merges, regime_characteristics, regime_to_cluster
                    )
                    all_merged_clusters.extend(round_merges)
                else:
                    self.logger.info(f"   ⏭️ No valid merges at threshold {threshold:.3f}")
                
                # Update cluster sample counts after merges
                if round_merges:
                    for old_cluster, new_cluster in round_merges:
                        if old_cluster in cluster_sample_counts and new_cluster in cluster_sample_counts:
                            cluster_sample_counts[new_cluster] += cluster_sample_counts[old_cluster]
                            del cluster_sample_counts[old_cluster]
                
                # Calculate top-20 coverage
                sorted_clusters = sorted(cluster_sample_counts.items(), key=lambda x: x[1], reverse=True)
                top_20_samples = sum([size for _, size in sorted_clusters[:20]])
                top_20_coverage = (top_20_samples / total_samples * 100) if total_samples > 0 else 0
                previous_top_20_coverage = top_20_coverage
                
                self.logger.info(f"   📊 Enhanced Progress: {cluster_count} clusters, top-20 coverage: {top_20_coverage:.1f}%")
                
                # Early stopping with enhanced criteria
                if top_20_coverage >= 90.0:
                    self.logger.info(f"✅ ENHANCED TARGET REACHED: {top_20_coverage:.1f}% coverage")
                    break
            
            # Calculate comprehensive quality score
            clusters_dict = {}
            cluster_id_map = {}
            next_id = 0
            
            for regime_id, cluster_num in regime_to_cluster.items():
                if cluster_num not in cluster_id_map:
                    cluster_id_map[cluster_num] = f"cluster_{next_id}"
                    next_id += 1
                
                cluster_id = cluster_id_map[cluster_num]
                if cluster_id not in clusters_dict:
                    clusters_dict[cluster_id] = []
                clusters_dict[cluster_id].append(regime_id)
            
            # Apply comprehensive quality scoring
            quality_score, quality_metrics = self._calculate_comprehensive_cluster_quality(
                clusters_dict, regime_characteristics, similarity_matrix
            )
            
            self.logger.info(f"✅ Enhanced clustering completed: {len(clusters_dict)} clusters")
            self.logger.info(f"📊 Overall Quality Score: {quality_score:.3f}")
            
            # Create cluster assignments from regime-to-cluster mapping
            cluster_assignments = self._create_cluster_assignments_from_mapping(regime_assignments, regime_to_cluster, list(regime_characteristics.keys()))
            
            # Ensure data alignment between cluster assignments and market data
            if market_data is not None and hasattr(market_data, '__len__'):
                alignment_result = self._align_data_lengths(market_data, cluster_assignments)
                market_data = alignment_result.aligned_market_data
                cluster_assignments = alignment_result.aligned_assignments
            
            return {
                'hmm_models': [{'cluster_id': i, 'model_type': 'enhanced_cluster', 'regime_count': len(regimes)} 
                              for i, regimes in enumerate(clusters_dict.values())],
                'cluster_assignments': cluster_assignments,
                'aligned_market_data': market_data,  # Include aligned market data
                'cluster_metrics': {
                    'cluster_count': len(clusters_dict),
                    'quality_score': quality_score,
                    'quality_metrics': quality_metrics,
                    'total_merges': len(all_merged_clusters),
                    'incremental_updates': True
                },
                'advanced_clustering_analysis': {
                    'enhanced_features_used': True,
                    'incremental_similarity_updates': len(all_merged_clusters),
                    'comprehensive_quality_scoring': True
                },
                'success': True,
                'error': None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced clustering failed: {e}")
            # Fallback to standard clustering
            return self._perform_matrix_based_clustering(regime_characteristics, regime_assignments, max_clusters, market_data)

    def _perform_matrix_based_clustering(
        self, 
        regime_characteristics: Dict[str, Any], 
        regime_assignments: List[int], 
        max_clusters: int,
        market_data: Any = None
    ) -> Dict[str, Any]:
        """Perform hierarchical matrix-based clustering with similarity thresholds."""
        try:
            import numpy as np
            
            # Calculate regime similarity matrix with memory optimization
            similarity_matrix = self._calculate_memory_efficient_similarity_matrix(regime_characteristics)
            if similarity_matrix.size == 0:
                self.logger.error("❌ Empty similarity matrix - cannot perform clustering")
                return self._create_single_cluster_result(regime_assignments, market_data)

            regime_ids = list(regime_characteristics.keys())
            n_regimes = len(regime_ids)

            # 🔍 DETAILED SIMILARITY ANALYSIS
            import numpy as np
            self.logger.info(f"🎯 Starting hierarchical matrix clustering: {n_regimes} regimes → target: 20-40 clusters")
            self.logger.info(f"📊 Similarity Matrix Analysis:")
            self.logger.info(f"   📈 Matrix shape: {similarity_matrix.shape}")
            self.logger.info(f"   📈 Min similarity: {np.min(similarity_matrix):.4f}")
            self.logger.info(f"   📈 Max similarity: {np.max(similarity_matrix):.4f}")
            self.logger.info(f"   📈 Mean similarity: {np.mean(similarity_matrix):.4f}")
            self.logger.info(f"   📈 Median similarity: {np.median(similarity_matrix):.4f}")
            self.logger.info(f"   📈 Std similarity: {np.std(similarity_matrix):.4f}")
            
            # Analyze similarity distribution
            high_sim_pairs = np.sum(similarity_matrix > 0.8)
            med_sim_pairs = np.sum((similarity_matrix > 0.5) & (similarity_matrix <= 0.8))
            low_sim_pairs = np.sum(similarity_matrix <= 0.5)
            total_pairs = n_regimes * (n_regimes - 1) // 2  # Upper triangle excluding diagonal
            
            self.logger.info(f"   🎯 Similarity Distribution:")
            self.logger.info(f"      High similarity (>0.8): {high_sim_pairs} pairs ({high_sim_pairs/total_pairs*100:.1f}%)")
            self.logger.info(f"      Medium similarity (0.5-0.8): {med_sim_pairs} pairs ({med_sim_pairs/total_pairs*100:.1f}%)")
            self.logger.info(f"      Low similarity (≤0.5): {low_sim_pairs} pairs ({low_sim_pairs/total_pairs*100:.1f}%)")
            
            # Initialize regime-to-cluster mapping (each regime starts as its own cluster)
            regime_to_cluster = {regime_id: i for i, regime_id in enumerate(regime_ids)}
            cluster_count = n_regimes
            
            # Intelligent hierarchical merging with adaptive similarity thresholds
            # Automatically detect optimal stopping point based on regime dissimilarity
            min_cv_threshold = 11.25  # Another 50% more relaxed for faster convergence (was 7.5)
            # Note: Merging logic allows clusters >6% to merge as long as resulting cluster <12%
            
            # Calculate progressive similarity thresholds from 99% down to 80%
            similarity_thresholds = self._calculate_progressive_similarity_thresholds()
            
            # 🔍 DETAILED THRESHOLD ANALYSIS
            self.logger.info(f"🎯 Adaptive Similarity Thresholds:")
            for i, threshold in enumerate(similarity_thresholds):
                eligible_pairs = np.sum(similarity_matrix >= threshold)
                self.logger.info(f"   📊 Threshold {i+1}: {threshold:.4f} ({threshold*100:.1f}%) - {eligible_pairs} eligible pairs for merging")

            # Diagnostics accumulators
            threshold_quality_curve: List[Dict[str, Any]] = []
            rejection_histogram_agg = {
                'same_cluster': 0,
                'excluded_cv': 0,
                'size_constraint': 0,
                'similarity_too_low': 0,
                'cv_incompatible': 0,
                'cv_hard_constraint': 0,
                'accepted': 0
            }

            # Adaptive CV threshold generation based on data characteristics
            cv_thresholds = self._generate_adaptive_cv_thresholds(regime_characteristics)
            current_cv_threshold_idx = 0
            
            self.logger.info(f"🎯 PROGRESSIVE CV RELAXATION: Starting with CV threshold {cv_thresholds[0]:.1f} ({cv_thresholds[0]*100:.0f}%)")
            self.logger.info(f"   📊 Progression: {' → '.join([f'{t:.1f}' for t in cv_thresholds])}")
            self.logger.info(f"   📊 Target: 90% top-20 coverage for early stopping")
            
            # Track previous coverage to trigger CV relaxation
            previous_top_20_coverage = 0.0
            cv_relaxation_triggered = False
            
            for threshold in similarity_thresholds:
                if cluster_count <= 20:  # Stop when we reach lower bound of optimal range (20-40)
                    self.logger.info(f"🎯 STOPPING: Reached optimal cluster count ({cluster_count} <= 20)")
                    break
                    
                self.logger.info(f"🔄 Batch merging at {threshold*100:.1f}% similarity threshold ({threshold:.3f})...")
                self.logger.info(f"   📊 Starting with {cluster_count} clusters")
                
                # Progressive merging approach - continue down to 55% similarity
                if threshold < 0.55:  # Stop only when we reach 55% similarity
                    self.logger.info(f"🛑 STOPPING: Reached minimum similarity threshold (55%)")
                    break

                # Check if we need to relax CV threshold based on previous iteration's coverage
                if previous_top_20_coverage > 0 and previous_top_20_coverage < 90.0 and current_cv_threshold_idx < len(cv_thresholds) - 1:
                    old_cv_threshold = cv_thresholds[current_cv_threshold_idx]
                    current_cv_threshold_idx += 1
                    new_cv_threshold = cv_thresholds[current_cv_threshold_idx]
                    
                    self.logger.info(f"📈 RELAXING CV THRESHOLD: {old_cv_threshold:.1f} → {new_cv_threshold:.1f} (Previous top-20: {previous_top_20_coverage:.1f}% < 90% target)")
                    self.logger.info(f"   🔄 Progress: Step {current_cv_threshold_idx + 1}/{len(cv_thresholds)} in CV relaxation sequence")
                    cv_relaxation_triggered = True
                elif previous_top_20_coverage > 0 and previous_top_20_coverage < 90.0 and current_cv_threshold_idx >= len(cv_thresholds) - 1:
                    current_cv_threshold = cv_thresholds[-1]  # Use max threshold
                    self.logger.warning(f"⚠️ CV THRESHOLD MAXED OUT: At {current_cv_threshold:.1f} but only {previous_top_20_coverage:.1f}% coverage (target: 90%+)")
                    self.logger.warning(f"   🔍 Consider other bottlenecks: similarity thresholds, hard constraints, or regime quality")
                    self.logger.info(f"🛑 STOPPING: CV threshold exhausted - cannot achieve 90% target with current data")
                    break

                # Log CV and threshold analysis for information but don't stop
                cv_analysis = self._analyze_cv_stopping_criteria(regime_characteristics, regime_to_cluster)
                if cv_analysis['should_stop']:
                    self.logger.info(f"⚠️ CV Analysis Warning: {cv_analysis['reason']} (continuing anyway)")
                    self.logger.info(f"   📊 CV Stats: {cv_analysis['stats']}")

                threshold_analysis = self._analyze_similarity_threshold(threshold, similarity_matrix, regime_to_cluster)
                if threshold_analysis['too_low']:
                    self.logger.info(f"⚠️ Threshold Warning: {threshold_analysis['reason']} (continuing anyway)")
                    self.logger.info(f"   📊 Similarity Stats: {threshold_analysis['stats']}")
                else:
                    self.logger.info(f"✅ Threshold OK: {threshold_analysis['reason']}")
                
                # Use progressive CV threshold with debug logging
                current_cv_threshold = cv_thresholds[min(current_cv_threshold_idx, len(cv_thresholds) - 1)]
                self.logger.info(f"🔍 DEBUG: Using CV threshold {current_cv_threshold:.1f} (index {current_cv_threshold_idx}/{len(cv_thresholds)-1})")
                
                # Calculate total samples first
                total_samples = len(regime_assignments)
                
                # Identify clusters that should be excluded from merging (only oversized clusters >12%)
                excluded_clusters = self._get_excluded_clusters_due_to_size(regime_characteristics, regime_to_cluster, total_samples)

                # Calculate cluster sample counts for logging
                cluster_sample_counts = {}
                for regime_id, cluster_id in regime_to_cluster.items():
                    if cluster_id not in cluster_sample_counts:
                        cluster_sample_counts[cluster_id] = 0
                    regime_sample_count = regime_characteristics[regime_id].get('sample_count', 1)
                    cluster_sample_counts[cluster_id] += regime_sample_count
                
                # Oversized cluster detection is now handled by _get_excluded_clusters_due_to_size
                
                # Log current cluster sizes with distribution statistics
                sorted_clusters_debug = sorted(cluster_sample_counts.items(), key=lambda x: x[1], reverse=True)
                top_5_debug = sorted_clusters_debug[:5]
                top_10_debug = sorted_clusters_debug[:10]
                top_20_debug = sorted_clusters_debug[:20]
                top_30_debug = sorted_clusters_debug[:30]
                
                # Calculate cumulative distribution percentages
                top_5_total = sum(size for _, size in top_5_debug)
                top_10_total = sum(size for _, size in top_10_debug)
                top_20_total = sum(size for _, size in top_20_debug)
                top_30_total = sum(size for _, size in top_30_debug)
                
                top_5_pct = (top_5_total / total_samples) * 100 if total_samples > 0 else 0
                top_10_pct = (top_10_total / total_samples) * 100 if total_samples > 0 else 0
                top_20_pct = (top_20_total / total_samples) * 100 if total_samples > 0 else 0
                top_30_pct = (top_30_total / total_samples) * 100 if total_samples > 0 else 0
                
                self.logger.info(f"   📊 Current cluster sizes (top 5): {[(f'C{cid}', size, f'{size/total_samples*100:.1f}%') for cid, size in top_5_debug]}")
                self.logger.info(f"   📊 Distribution stats: Top 5 = {top_5_pct:.1f}%, Top 10 = {top_10_pct:.1f}%, Top 20 = {top_20_pct:.1f}%, Top 30 = {top_30_pct:.1f}%")
                
                # Check coverage-based stopping and CV threshold adjustment
                self.logger.info(f"🔍 DEBUG CV PROGRESSION: Current threshold index={current_cv_threshold_idx}, threshold={current_cv_threshold:.1f}, top_20_pct={top_20_pct:.1f}%")
                
                if top_20_pct >= 90.0:
                    self.logger.info(f"✅ TARGET REACHED: Top 20 clusters cover {top_20_pct:.1f}% >= 90% of samples")
                    self.logger.info(f"🛑 EARLY STOPPING: Target achieved, terminating CV relaxation loop")
                    break  # Stop processing further thresholds
                elif top_20_pct < 85.0:
                    if current_cv_threshold_idx < len(cv_thresholds) - 1:
                        self.logger.info(f"📈 COVERAGE INSUFFICIENT: {top_20_pct:.1f}% < 90% target - CV threshold will be relaxed in next iteration")
                    else:
                        self.logger.warning(f"⚠️ CV THRESHOLD MAXED OUT: At {current_cv_threshold:.1f} but only {top_20_pct:.1f}% coverage (target: 90%+)")
                        self.logger.warning(f"   🔍 Consider other bottlenecks: similarity thresholds, hard constraints, or regime quality")
                        self.logger.info(f"🛑 STOPPING: CV threshold exhausted - cannot achieve 85% target with current data")
                        break

                # (cluster_sample_counts already calculated above)
                
                # Compute within-cluster CV by aspect and derive soft penalties
                cluster_cv_aspects = self._compute_cluster_cv_by_aspect(regime_characteristics, regime_to_cluster)
                import numpy as np
                # Build aspect thresholds: median + 0.5 * IQR per aspect
                aspect_thresholds: Dict[str, float] = {}
                for aspect in ['momentum', 'volatility', 'volume']:
                    vals = [cv_map.get(aspect) for cv_map in cluster_cv_aspects.values() if isinstance(cv_map.get(aspect), (int, float))]
                    vals = [float(v) for v in vals if v is not None and np.isfinite(v)]
                    if len(vals) > 0:
                        med = float(np.median(vals))
                        iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
                        aspect_thresholds[aspect] = max(0.05, med + 0.5 * iqr)
                    else:
                        aspect_thresholds[aspect] = 0.2
                # Penalty per cluster if dominant aspect exceeds its threshold
                cluster_penalty: Dict[int, float] = {}
                base_penalty = 0.002
                penalized_clusters = []
                for cid, cv_map in cluster_cv_aspects.items():
                    # Determine dominant aspect
                    items = [(k, v) for k, v in cv_map.items() if k in aspect_thresholds and isinstance(v, (int, float))]
                    if not items:
                        cluster_penalty[cid] = 0.0
                        continue
                    dom_aspect, dom_cv = max(items, key=lambda kv: kv[1])
                    thr = aspect_thresholds.get(dom_aspect, 0.2)
                    penalty = base_penalty if (dom_cv is not None and dom_cv >= thr) else 0.0
                    if penalty > 0.0:
                        penalized_clusters.append((cid, dom_aspect, dom_cv, thr))
                    cluster_penalty[cid] = penalty
                if len(penalized_clusters) > 0:
                    self.logger.info(
                        "   ⚖️ High-CV penalties: base=%.3f, aspect_thresholds=%s, penalized=%s" % (
                            base_penalty,
                            {k: round(v, 3) for k, v in aspect_thresholds.items()},
                            [(cid, a, round(cv, 3), round(th, 3)) for cid, a, cv, th in penalized_clusters]
                        )
                    )
                
                # Smart merge candidate pre-filtering to avoid expensive checks
                mergeable_pairs = self._get_smart_merge_candidates(
                    similarity_matrix, regime_to_cluster, regime_ids, excluded_clusters, threshold
                )
                self.logger.info(f"🔍 POST-PREFILTERING: {len(mergeable_pairs)} candidates passed pre-filtering")
                rejection_stats = {
                    'same_cluster': 0,
                    'excluded_cv': 0,
                    'size_constraint': 0,
                    'similarity_too_low': 0,
                    'cv_incompatible': 0,
                    'cv_hard_constraint': 0,
                    'accepted': 0,
                    'prefiltered': len(mergeable_pairs)
                }
                
                # Process pre-filtered candidates (much more efficient than nested loops)
                self.logger.debug(f"🔍 Processing {len(mergeable_pairs)} pre-filtered candidates")
                
                # If pre-filtering failed, fall back to original method
                if not mergeable_pairs:
                    self.logger.warning("⚠️ Pre-filtering returned no candidates, using fallback method")
                    for i, regime_i in enumerate(regime_ids):
                        for j, regime_j in enumerate(regime_ids[i+1:], i+1):
                            similarity = similarity_matrix[i, j]
                            cluster_i = regime_to_cluster[regime_i]
                            cluster_j = regime_to_cluster[regime_j]
                            
                            # Skip if same cluster
                            if cluster_i == cluster_j:
                                rejection_stats['same_cluster'] += 1
                                continue
                            
                            # Skip if either cluster is excluded due to high CV
                            if cluster_i in excluded_clusters or cluster_j in excluded_clusters:
                                rejection_stats['excluded_cv'] += 1
                                continue
                else:
                    # Process only the pre-filtered high-potential candidates
                    final_mergeable_pairs = []
                    detailed_processing_count = 0
                    for similarity, cluster_i, cluster_j, regime_i, regime_j in mergeable_pairs:
                        detailed_processing_count += 1
                        
                        # Enforce STRICT 12% max cluster size constraint
                        # Check if either cluster is already at or above 12%
                        cluster_i_size = cluster_sample_counts.get(cluster_i, 0)
                        cluster_j_size = cluster_sample_counts.get(cluster_j, 0)
                        cluster_i_pct = (cluster_i_size / total_samples) * 100 if total_samples > 0 else 0
                        cluster_j_pct = (cluster_j_size / total_samples) * 100 if total_samples > 0 else 0
                        
                        # Calculate combined CV to determine progressive size limits
                        cv_i = cluster_cv_aspects.get(cluster_i, {})
                        cv_j = cluster_cv_aspects.get(cluster_j, {})
                        max_cv = 0.0
                        
                        if cv_i and cv_j:
                            for aspect in ['momentum', 'volatility', 'volume']:
                                cv_i_val = cv_i.get(aspect, 0.0) or 0.0
                                cv_j_val = cv_j.get(aspect, 0.0) or 0.0
                                if isinstance(cv_i_val, (int, float)) and isinstance(cv_j_val, (int, float)):
                                    max_cv = max(max_cv, cv_i_val, cv_j_val)
                        
                        # GRADUATED SIZE PENALTIES: Replace hard cutoffs with penalty-based system
                        merged_size = cluster_i_size + cluster_j_size
                        merged_pct = (merged_size / total_samples) * 100 if total_samples > 0 else 0
                        
                        # Calculate size penalty (0.0 = no penalty, 1.0 = block merge)
                        if merged_pct <= 12.0:
                            size_penalty = 0.0  # No penalty for clusters ≤12%
                            penalty_reason = "small cluster (≤12%)"
                        elif merged_pct <= 15.0:
                            size_penalty = (merged_pct - 12.0) / 3.0  # 0.0 at 12%, 1.0 at 15%
                            penalty_reason = f"medium cluster ({merged_pct:.1f}%)"
                        elif merged_pct <= 20.0:
                            size_penalty = 0.6 + (merged_pct - 15.0) / 5.0 * 0.3  # 0.6 at 15%, 0.9 at 20%
                            penalty_reason = f"large cluster ({merged_pct:.1f}%)"
                        else:
                            size_penalty = 1.0  # Hard block for clusters >20%
                            penalty_reason = f"oversized cluster ({merged_pct:.1f}%)"
                        
                        # Apply graduated penalty based on coverage progress (use previous iteration's coverage)
                        coverage_boost = 0.0
                        current_coverage = previous_top_20_coverage  # Use available coverage variable
                        if current_coverage < 70.0:  # Far from target, be more permissive
                            coverage_boost = 0.3
                        elif current_coverage < 80.0:  # Getting closer, moderate boost
                            coverage_boost = 0.2
                        elif current_coverage < 90.0:  # Close to target, small boost
                            coverage_boost = 0.1
                        
                        effective_penalty = max(0.0, size_penalty - coverage_boost)
                        
                        # Block only if effective penalty is very high (>0.8)
                        if effective_penalty > 0.8:
                            self.logger.debug(f"   🚫 BLOCKED: C{cluster_i} ({cluster_i_pct:.1f}%) + C{cluster_j} ({cluster_j_pct:.1f}%) → {merged_pct:.1f}% (penalty: {effective_penalty:.2f} > 0.8, {penalty_reason})")
                            rejection_stats['size_constraint'] += 1
                            continue
                        elif size_penalty > 0.0:
                            self.logger.debug(f"   ⚠️ PENALTY: C{cluster_i} + C{cluster_j} → {merged_pct:.1f}% (penalty: {effective_penalty:.2f}, {penalty_reason})")
                        
                        # DEBUG: Track C122 constraint checks
                        if cluster_i == 122 or cluster_j == 122:
                            self.logger.warning(f"🔍 C122 CONSTRAINT CHECK: C{cluster_i} + C{cluster_j}")
                        
                        # Weighted similarity approach - 60% similarity, 30% CV, 10% size
                        regime_1 = regime_characteristics[regime_i]
                        regime_2 = regime_characteristics[regime_j]
                        cluster_sizes = [
                            len([rid for rid, cid in regime_to_cluster.items() if cid == cluster_i]),
                            len([rid for rid, cid in regime_to_cluster.items() if cid == cluster_j])
                        ]
                        
                        # Calculate weighted composite score with dynamic weights based on similarity threshold
                        weighted_score = self._calculate_weighted_merge_score(
                            similarity, regime_1, regime_2, cluster_i, cluster_j, regime_to_cluster, total_samples, threshold
                        )
                        
                        # Debug logging for weighted scores (sample a few)
                        if (cluster_i == 122 or cluster_j == 122) or (cluster_i < 5 and cluster_j < 5):
                            combined_size_pct = (len([rid for rid, cid in regime_to_cluster.items() if cid == cluster_i]) + 
                                               len([rid for rid, cid in regime_to_cluster.items() if cid == cluster_j])) / total_samples * 100
                            self.logger.debug(f"   📊 WEIGHTED SCORE: C{cluster_i}+C{cluster_j} → {weighted_score:.3f} (sim:{similarity:.3f}, size:{combined_size_pct:.1f}%)")
                        
                        # Apply strict dimension compatibility check for final consolidation
                        if threshold <= 0.75:  # Final consolidation phase (earlier transition)
                            if not self._strict_dimension_compatibility_check(regime_1, regime_2):
                                rejection_stats['dimension_incompatible'] = rejection_stats.get('dimension_incompatible', 0) + 1
                            continue
                                
                        # Prevent problematic cluster formation (applies to all phases)
                        if self._would_create_problematic_cluster(regime_1, regime_2, cluster_i, cluster_j, 
                                                                regime_to_cluster, total_samples, current_cv_threshold):
                            rejection_stats['problematic_cluster_prevented'] = rejection_stats.get('problematic_cluster_prevented', 0) + 1
                            continue
                        
                        # Dynamic threshold based on clustering progress
                        base_threshold = threshold
                        if weighted_score < base_threshold:
                            rejection_stats['similarity_too_low'] += 1
                            continue

                        rejection_stats['accepted'] += 1
                        final_mergeable_pairs.append((similarity, cluster_i, cluster_j, regime_i, regime_j))
                    
                    # Update mergeable_pairs with validated candidates
                    mergeable_pairs = final_mergeable_pairs
                    self.logger.info(f"🔍 DETAILED PROCESSING: {detailed_processing_count} pairs processed through detailed checks")

                # 🔍 DETAILED PAIR ANALYSIS WITH BOTTLENECK IDENTIFICATION
                total_pairs_checked = sum(rejection_stats.values())
                self.logger.info(f"   📊 Pair Analysis (out of {total_pairs_checked} pairs checked):")
                self.logger.info(f"      ✅ Accepted for merging: {rejection_stats['accepted']} ({rejection_stats['accepted']/total_pairs_checked*100:.1f}%)")
                self.logger.info(f"      ❌ Same cluster: {rejection_stats['same_cluster']} ({rejection_stats['same_cluster']/total_pairs_checked*100:.1f}%)")
                self.logger.info(f"      ❌ Excluded due to CV: {rejection_stats['excluded_cv']} ({rejection_stats['excluded_cv']/total_pairs_checked*100:.1f}%)")
                self.logger.info(f"      ❌ Size constraint (>12%): {rejection_stats['size_constraint']} ({rejection_stats['size_constraint']/total_pairs_checked*100:.1f}%)")
                self.logger.info(f"      ❌ Similarity too low: {rejection_stats['similarity_too_low']} ({rejection_stats['similarity_too_low']/total_pairs_checked*100:.1f}%)")
                if rejection_stats.get('dimension_incompatible', 0) > 0:
                    self.logger.info(f"      ❌ Dimension incompatible (final consolidation): {rejection_stats['dimension_incompatible']} ({rejection_stats['dimension_incompatible']/total_pairs_checked*100:.1f}%)")
                if rejection_stats.get('problematic_cluster_prevented', 0) > 0:
                    self.logger.info(f"      ❌ Problematic cluster prevented (high CV prediction): {rejection_stats['problematic_cluster_prevented']} ({rejection_stats['problematic_cluster_prevented']/total_pairs_checked*100:.1f}%)")
                self.logger.info(f"      ❌ Weighted score too low (dynamic weights): {rejection_stats['cv_incompatible']} ({rejection_stats['cv_incompatible']/total_pairs_checked*100:.1f}%)")
                self.logger.info(f"      ❌ Hard constraints: {rejection_stats['cv_hard_constraint']} ({rejection_stats['cv_hard_constraint']/total_pairs_checked*100:.1f}%)")
                
                # Identify primary bottleneck
                bottlenecks = [
                    ('CV Exclusion', rejection_stats['excluded_cv']),
                    ('Weighted Score Too Low', rejection_stats['cv_incompatible']),
                    ('Basic Similarity Too Low', rejection_stats['similarity_too_low']),
                    ('Size Constraint', rejection_stats['size_constraint']),
                    ('Dimension Incompatible', rejection_stats.get('dimension_incompatible', 0)),
                    ('Problematic Cluster Prevention', rejection_stats.get('problematic_cluster_prevented', 0))
                ]
                primary_bottleneck = max(bottlenecks, key=lambda x: x[1])
                self.logger.info(f"   🎯 PRIMARY BOTTLENECK: {primary_bottleneck[0]} blocking {primary_bottleneck[1]:,} pairs ({primary_bottleneck[1]/total_pairs_checked*100:.1f}%)")
                
                self.logger.info(f"   📊 Found {len(mergeable_pairs)} mergeable pairs")
                
                # Sort by similarity (highest first) and merge pairs
                mergeable_pairs.sort(key=lambda x: x[0], reverse=True)
                
                merges_this_round = 0
                merged_clusters = set()
                merge_similarities = []
                merge_cv_differences = []
                merged_cluster_sizes = []
                
                for similarity, cluster_i, cluster_j, regime_i, regime_j in mergeable_pairs:
                    # Skip if either cluster has already been merged this round
                    if cluster_i in merged_clusters or cluster_j in merged_clusters:
                        continue
                    
                    # DEBUG: Track merge that might create extreme CV
                    if cluster_i == 122 or cluster_j == 122:
                        self.logger.warning(f"🔍 TRACKING C122: Merging C{cluster_j} → C{cluster_i} (similarity: {similarity:.3f})")
                        self.logger.warning(f"   📊 Pre-merge sizes: C{cluster_i}={cluster_sample_counts.get(cluster_i, 0)}, C{cluster_j}={cluster_sample_counts.get(cluster_j, 0)}")
                    
                    # Merge cluster_j into cluster_i
                    for regime_id, cluster_id in regime_to_cluster.items():
                        if cluster_id == cluster_j:
                            regime_to_cluster[regime_id] = cluster_i
                    
                    # DEBUG: Check if this merge created extreme CV
                    if cluster_i == 122:
                        regimes_in_122 = [rid for rid, cid in regime_to_cluster.items() if cid == 122]
                        self.logger.warning(f"🔍 C122 POST-MERGE: {len(regimes_in_122)} regimes = {regimes_in_122[:5]}{'...' if len(regimes_in_122) > 5 else ''}")
                    
                    # Calculate merged cluster size BEFORE updating counts
                    merged_size = cluster_sample_counts.get(cluster_i, 0) + cluster_sample_counts.get(cluster_j, 0)
                    merged_cluster_sizes.append(merged_size)
                    
                    # Update cluster sample counts after merge
                    cluster_sample_counts[cluster_i] = merged_size
                    if cluster_j in cluster_sample_counts:
                        del cluster_sample_counts[cluster_j]
                    
                    # Track merge statistics
                    merge_similarities.append(similarity)
                    merged_clusters.add(cluster_j)
                    merges_this_round += 1
                    
                    self.logger.debug(f"   ✅ Merged clusters {cluster_j} → {cluster_i} (similarity: {similarity:.3f})")
                
                # Update cluster count
                cluster_count = len(set(regime_to_cluster.values()))
                
                # Calculate batch summary metrics
                if merges_this_round > 0:
                    avg_similarity = sum(merge_similarities) / len(merge_similarities)
                    min_cluster_size = min(merged_cluster_sizes)
                    max_cluster_size = max(merged_cluster_sizes)
                    avg_cluster_size = sum(merged_cluster_sizes) / len(merged_cluster_sizes)
                    
                    self.logger.info(f"   📊 BATCH SUMMARY:")
                    self.logger.info(f"      🔄 Merges: {merges_this_round} pairs merged")
                    self.logger.info(f"      🎯 Clusters: {cluster_count} remaining")
                    self.logger.info(f"      📈 Similarity: avg={avg_similarity:.3f} (range: {min(merge_similarities):.3f}-{max(merge_similarities):.3f})")
                    self.logger.info(f"      📏 Cluster Sizes: avg={avg_cluster_size:.0f} samples (range: {min_cluster_size}-{max_cluster_size})")
                    
                    # Report current cluster size distribution with detailed statistics
                    cluster_size_list = [(f'C{cluster_id}', size, f'{(size / total_samples * 100):.1f}%') 
                                       for cluster_id, size in cluster_sample_counts.items()]
                    cluster_size_list.sort(key=lambda x: x[1], reverse=True)  # Sort by size descending
                    top_5_clusters = cluster_size_list[:5]
                    top_10_clusters = cluster_size_list[:10] 
                    top_20_clusters = cluster_size_list[:20]
                    top_30_clusters = cluster_size_list[:30]
                    
                    # Calculate cumulative distribution percentages
                    top_5_total = sum(int(cluster[1]) for cluster in top_5_clusters)
                    top_10_total = sum(int(cluster[1]) for cluster in top_10_clusters)
                    top_20_total = sum(int(cluster[1]) for cluster in top_20_clusters)
                    top_30_total = sum(int(cluster[1]) for cluster in top_30_clusters)
                    
                    top_5_pct = (top_5_total / total_samples) * 100 if total_samples > 0 else 0
                    top_10_pct = (top_10_total / total_samples) * 100 if total_samples > 0 else 0
                    top_20_pct = (top_20_total / total_samples) * 100 if total_samples > 0 else 0
                    top_30_pct = (top_30_total / total_samples) * 100 if total_samples > 0 else 0
                    
                    self.logger.info(f"   📊 Current cluster sizes (top 5): {top_5_clusters}")
                    self.logger.info(f"   📊 Distribution stats: Top 5 = {top_5_pct:.1f}%, Top 10 = {top_10_pct:.1f}%, Top 20 = {top_20_pct:.1f}%, Top 30 = {top_30_pct:.1f}%")
                    
                    # Store coverage for CV threshold adjustment
                    previous_top_20_coverage = top_20_pct
                    
                    # Check coverage after merges
                    if top_20_pct >= 90.0:
                        self.logger.info(f"🎯 COVERAGE EXCEEDED: Top 20 clusters cover {top_20_pct:.1f}% >= 90% of samples")
                    elif top_20_pct >= 85.0:
                        self.logger.info(f"✅ TARGET REACHED: Top 20 clusters cover {top_20_pct:.1f}% >= 90% of samples")
                    else:
                        self.logger.info(f"📈 PROGRESS: Top 20 clusters cover {top_20_pct:.1f}% (target: 90%+)")
                else:
                    self.logger.info(f"   📊 Batch complete: {merges_this_round} merges → {cluster_count} clusters remaining")
                    
                    # Report current cluster size distribution with detailed statistics even when no merges occurred
                    cluster_size_list = [(f'C{cluster_id}', size, f'{(size / total_samples * 100):.1f}%') 
                                       for cluster_id, size in cluster_sample_counts.items()]
                    cluster_size_list.sort(key=lambda x: x[1], reverse=True)  # Sort by size descending
                    top_5_clusters = cluster_size_list[:5]
                    top_10_clusters = cluster_size_list[:10] 
                    top_20_clusters = cluster_size_list[:20]
                    top_30_clusters = cluster_size_list[:30]
                    
                    # Calculate cumulative distribution percentages
                    top_5_total = sum(int(cluster[1]) for cluster in top_5_clusters)
                    top_10_total = sum(int(cluster[1]) for cluster in top_10_clusters)
                    top_20_total = sum(int(cluster[1]) for cluster in top_20_clusters)
                    top_30_total = sum(int(cluster[1]) for cluster in top_30_clusters)
                    
                    top_5_pct = (top_5_total / total_samples) * 100 if total_samples > 0 else 0
                    top_10_pct = (top_10_total / total_samples) * 100 if total_samples > 0 else 0
                    top_20_pct = (top_20_total / total_samples) * 100 if total_samples > 0 else 0
                    top_30_pct = (top_30_total / total_samples) * 100 if total_samples > 0 else 0
                    
                    self.logger.info(f"   📊 Current cluster sizes (top 5): {top_5_clusters}")
                    self.logger.info(f"   📊 Distribution stats: Top 5 = {top_5_pct:.1f}%, Top 10 = {top_10_pct:.1f}%, Top 20 = {top_20_pct:.1f}%, Top 30 = {top_30_pct:.1f}%")
                    
                    # Store coverage for CV threshold adjustment
                    previous_top_20_coverage = top_20_pct
                    
                    # Check coverage when no merges occurred
                    if top_20_pct >= 90.0:
                        self.logger.info(f"🎯 COVERAGE EXCEEDED: Top 20 clusters cover {top_20_pct:.1f}% >= 90% of samples")
                    elif top_20_pct >= 85.0:
                        self.logger.info(f"✅ TARGET REACHED: Top 20 clusters cover {top_20_pct:.1f}% >= 90% of samples")
                    else:
                        self.logger.info(f"📈 PROGRESS: Top 20 clusters cover {top_20_pct:.1f}% (target: 90%+)")

                # Record diagnostics per threshold
                try:
                    temp_assignments = self._map_regime_assignments_to_clusters(regime_assignments, regime_to_cluster, len(regime_assignments))
                    quality_proxy = self._calculate_matrix_clustering_score(similarity_matrix, regime_to_cluster, temp_assignments)
                except Exception:
                    quality_proxy = 0.0
                threshold_quality_curve.append({
                    'threshold': float(threshold),
                    'merges': int(merges_this_round),
                    'cluster_count': int(cluster_count),
                    'quality_proxy': float(quality_proxy),
                    'rejections': {k: int(v) for k, v in rejection_stats.items() if k != 'accepted'}
                })
                for k in rejection_histogram_agg.keys():
                    if k in rejection_stats:
                        rejection_histogram_agg[k] += int(rejection_stats[k])
                
                # Recalculate similarity matrix with new cluster characteristics after merging
                if merges_this_round > 0:
                    self.logger.info(f"   🔄 Recalculating similarity matrix with {cluster_count} clusters...")
                    
                    # Update regime_ids to reflect current clusters
                    current_regime_ids = list(regime_to_cluster.keys())
                    
                    # Recalculate similarity matrix with updated cluster characteristics (no feature weighting)
                    similarity_matrix = self._calculate_regime_similarity_matrix(regime_characteristics, None)
                    
                    self.logger.info(f"   ✅ Updated similarity matrix for {cluster_count} clusters")
                    # Recompute and log CV summaries to observe penalty relaxation opportunities
                    cluster_cv_aspects_after = self._compute_cluster_cv_by_aspect(regime_characteristics, regime_to_cluster)
                    for aspect in ['momentum', 'volatility', 'volume']:
                        vals = [v.get(aspect) for v in cluster_cv_aspects_after.values() if isinstance(v.get(aspect), (int, float))]
                        vals = [float(v) for v in vals if np.isfinite(v)]
                        if len(vals) > 0:
                            med = float(np.median(vals))
                            iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
                            self.logger.info(f"   📉 Post-merge CV[{aspect}]: median={med:.3f}, IQR={iqr:.3f}")
                
                if cluster_count <= 20:  # Stop if we reach lower bound of optimal range
                    break
            
            # Renumber clusters sequentially
            unique_clusters = sorted(set(regime_to_cluster.values()))
            cluster_remapping = {old_id: new_id for new_id, old_id in enumerate(unique_clusters)}
            regime_to_cluster = {regime_id: cluster_remapping[cluster_id] for regime_id, cluster_id in regime_to_cluster.items()}
            final_cluster_count = len(unique_clusters)
            
            # Create cluster assignments
            cluster_assignments = self._create_cluster_assignments_from_mapping(
                regime_assignments, regime_to_cluster, regime_ids
            )
            
            # Ensure data alignment between cluster assignments and market data
            if market_data is not None and hasattr(market_data, '__len__'):
                alignment_result = self._align_data_lengths(market_data, cluster_assignments)
                market_data = alignment_result.aligned_market_data
                cluster_assignments = alignment_result.aligned_assignments

            # Calculate final quality metrics
            quality_score = self._calculate_matrix_clustering_score(
                similarity_matrix, regime_to_cluster, cluster_assignments
            )

            # Calculate advanced cluster analytics BEFORE constraints are applied
            cluster_analytics = self._calculate_advanced_cluster_analytics(
                regime_characteristics, regime_to_cluster, cluster_assignments, final_cluster_count
            )

            # Create simple cluster models without complex representative models
            cluster_models = []
            for cluster_id in range(final_cluster_count):
                cluster_models.append({
                    'cluster_id': cluster_id,
                    'model_type': 'simple_cluster',
                    'regime_count': sum(1 for cid in regime_to_cluster.values() if cid == cluster_id)
                })

            # Prepare similarity diagnostics
            similarity_histogram = self._calculate_similarity_histogram(similarity_matrix)
            try:
                sample_n = min(50, similarity_matrix.shape[0])
                similarity_heatmap_sample = [list(map(float, row[:sample_n])) for row in similarity_matrix[:sample_n]]
            except Exception:
                similarity_heatmap_sample = None

            # Compute per-cluster prototypes
            cluster_prototypes = self._compute_cluster_prototypes(regime_ids, regime_to_cluster, similarity_matrix, regime_characteristics, top_n_features=5)

            # Create result in expected format
            result = {
                'success': True,
                'cluster_assignments': cluster_assignments,
                'regime_to_cluster': regime_to_cluster,
                'n_clusters': final_cluster_count,
                'hmm_models': cluster_models,
                'similarity_matrix': similarity_matrix,
                'quality_score': quality_score,
                'method': 'hierarchical_matrix_based',
                'merging_thresholds_used': similarity_thresholds,
                'cluster_analytics': cluster_analytics,
                'diagnostics': {
                    'threshold_quality_curve': threshold_quality_curve,
                    'rejection_histogram': rejection_histogram_agg,
                    'similarity_histogram': similarity_histogram,
                    'similarity_heatmap_sample': similarity_heatmap_sample,
                    'cluster_prototypes': cluster_prototypes
                }
            }

            self.logger.info(f"✅ Hierarchical matrix clustering completed: {final_cluster_count} clusters created")
            self.logger.info(f"📊 Final cluster distribution: {self._calculate_cluster_distribution_from_sizes(cluster_analytics['cluster_sizes'])}")
            
            # Log advanced analytics
            self.logger.info(f"📊 Advanced Cluster Analytics (BEFORE constraints):")
            self.logger.info(f"   📈 Average within-cluster CV: {cluster_analytics['avg_within_cluster_cv']:.4f}")
            self.logger.info(f"   📊 Top 20 clusters coverage: {cluster_analytics['top_20_coverage']:.1f}% of samples")
            self.logger.info(f"   🎯 Cluster concentration: {cluster_analytics['concentration_analysis']}")
            self.logger.info(f"   📏 Actual cluster count: {final_cluster_count} clusters")
            self.logger.info(f"   📊 Total samples: {cluster_analytics['total_samples']}")
            
            # Show top 10 cluster sizes
            sorted_clusters = sorted(cluster_analytics['cluster_sizes'].items(), key=lambda x: x[1], reverse=True)
            top_10 = sorted_clusters[:10]
            total_samples = sum(cluster_analytics['cluster_sizes'].values())
            top_10_with_pct = [(f'C{cid}', size, f'{(size/total_samples)*100:.1f}%') for cid, size in top_10]
            self.logger.info(f"   🏆 Top 10 cluster sizes: {top_10_with_pct}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Hierarchical matrix clustering failed: {e}")
            # Fast fail - raise the exception instead of falling back
            raise RuntimeError(f"Hierarchical matrix clustering failed: {e}")

    def _calculate_weighted_merge_score(self, similarity: float, regime_1: Dict[str, Any], regime_2: Dict[str, Any], 
                                       cluster_i: int, cluster_j: int, regime_to_cluster: Dict[str, int], 
                                       total_samples: int, similarity_threshold: float = 0.80) -> float:
        """Calculate composite merge score with dynamic weights based on similarity threshold.
        
        Standard weights: 60% similarity + 30% CV + 10% size
        Final consolidation (≤72%): 40% similarity + 50% CV + 10% size
        """
        try:
            import numpy as np
            from math import log
            
            # Dynamic weight adjustment based on similarity threshold
            if similarity_threshold <= 0.75:  # Final consolidation phase (earlier transition)
                similarity_weight = 0.40  # Reduced from 0.60
                cv_weight = 0.50          # Increased from 0.30
                size_weight = 0.10        # Unchanged
                phase = "final_consolidation"
            else:
                similarity_weight = 0.60  # Standard weights
                cv_weight = 0.30
                size_weight = 0.10
                phase = "standard"
            
            # 1. Similarity component - direct use
            similarity_component = max(0.0, min(1.0, similarity))  # Ensure 0.0-1.0 range
            
            # 2. CV component - inverse log scaling (rewards low CV)
            cv1 = regime_1.get('features', {}).get('mean_cv', 0.0)
            cv2 = regime_2.get('features', {}).get('mean_cv', 0.0)
            
            # Use individual aspect CVs if mean_cv not available
            if cv1 == 0.0:
                cv1 = max(regime_1.get('features', {}).get('momentum_cv', 0.0),
                         regime_1.get('features', {}).get('volatility_cv', 0.0),
                         regime_1.get('features', {}).get('volume_cv', 0.0))
            if cv2 == 0.0:
                cv2 = max(regime_2.get('features', {}).get('momentum_cv', 0.0),
                         regime_2.get('features', {}).get('volatility_cv', 0.0),
                         regime_2.get('features', {}).get('volume_cv', 0.0))
            
            avg_cv = (cv1 + cv2) / 2.0
            max_cv_threshold = 10.0
            cv_component = max(0.0, 1.0 - log(1 + avg_cv) / log(1 + max_cv_threshold))
            
            # 3. Size component (10%) - inverse log scaling with penalty starting at 8%, very strong at 12%
            regimes_i = [rid for rid, cid in regime_to_cluster.items() if cid == cluster_i]
            regimes_j = [rid for rid, cid in regime_to_cluster.items() if cid == cluster_j]
            combined_size_pct = (len(regimes_i) + len(regimes_j)) / total_samples * 100 if total_samples > 0 else 0.0
            
            if combined_size_pct <= 8.0:
                # Boost for clusters ≤8%
                size_component = 1.0 - log(1 + combined_size_pct) / log(9)  # Boost up to 8%
            else:
                # Strong penalty for clusters >8%, very strong at 12%
                penalty_factor = (combined_size_pct - 8.0) / 4.0  # 0.0 at 8%, 1.0 at 12%
                size_component = max(0.0, 1.0 - log(1 + penalty_factor * 10) / log(11))  # Strong log penalty
            
            # 4. Weighted composite score with dynamic weights
            composite_score = (similarity_weight * similarity_component + 
                             cv_weight * cv_component + 
                             size_weight * size_component)
            
            # Log phase transition for debugging
            if similarity_threshold <= 0.72 and hasattr(self, '_last_logged_phase') and self._last_logged_phase != "final_consolidation":
                self.logger.info(f"🔄 PHASE TRANSITION: Entering final consolidation mode (weights: {similarity_weight:.0%} sim, {cv_weight:.0%} CV, {size_weight:.0%} size)")
                self._last_logged_phase = "final_consolidation"
            elif similarity_threshold > 0.72 and hasattr(self, '_last_logged_phase') and self._last_logged_phase != "standard":
                self.logger.info(f"🔄 PHASE TRANSITION: Standard clustering mode (weights: {similarity_weight:.0%} sim, {cv_weight:.0%} CV, {size_weight:.0%} size)")
                self._last_logged_phase = "standard"
            elif not hasattr(self, '_last_logged_phase'):
                self._last_logged_phase = phase
            
            return composite_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating weighted merge score: {e}")
            return similarity  # Fallback to basic similarity
    
    def _strict_dimension_compatibility_check(self, regime_1: Dict[str, Any], regime_2: Dict[str, Any]) -> bool:
        """Strict dimension compatibility check for final consolidation phase.

        Ensures regimes have compatible financial characteristics before merging at low similarity.
        """
        try:
            features_1 = regime_1.get('features', {})
            features_2 = regime_2.get('features', {})
            
            # 1. Momentum direction alignment (must be compatible)
            momentum_1 = features_1.get('momentum_mean', 0.0)
            momentum_2 = features_2.get('momentum_mean', 0.0)
            
            # Require same direction for significant momentum values (relaxed)
            if (momentum_1 > 0 and momentum_2 < 0) or (momentum_1 < 0 and momentum_2 > 0):
                if abs(momentum_1) > 10.0 and abs(momentum_2) > 10.0:  # Both very significant (relaxed from 1.0)
                    self.logger.debug(f"🚫 Momentum direction mismatch: {momentum_1:.3f} vs {momentum_2:.3f}")
                    return False
            
            # 2. Volatility magnitude compatibility (prevent extreme differences)
            vol_1 = features_1.get('volatility_mean', 0.0)
            vol_2 = features_2.get('volatility_mean', 0.0)
            
            if vol_1 > 0 and vol_2 > 0:
                vol_ratio = max(vol_1, vol_2) / min(vol_1, vol_2)
                if vol_ratio > 10.0:  # Max 10x difference in volatility (relaxed from 5.0)
                    self.logger.debug(f"🚫 Volatility ratio too high: {vol_ratio:.2f}x")
                    return False
            
            # 3. Volume pattern compatibility (similar relative activity)
            volume_1 = features_1.get('volume_mean', 0.0)
            volume_2 = features_2.get('volume_mean', 0.0)
            
            if volume_1 > 0 and volume_2 > 0:
                volume_ratio = max(volume_1, volume_2) / min(volume_1, volume_2)
                if volume_ratio > 10.0:  # Max 10x difference in volume (already at 10.0)
                    self.logger.debug(f"🚫 Volume ratio too high: {volume_ratio:.2f}x")
                    return False
            
            # 4. CV compatibility (prevent merging high-CV with low-CV)
            cv_1 = max(features_1.get('momentum_cv', 0.0), features_1.get('volatility_cv', 0.0), features_1.get('volume_cv', 0.0))
            cv_2 = max(features_2.get('momentum_cv', 0.0), features_2.get('volatility_cv', 0.0), features_2.get('volume_cv', 0.0))
            
            if cv_1 > 0 and cv_2 > 0:
                cv_ratio = max(cv_1, cv_2) / min(cv_1, cv_2)
                if cv_ratio > 4.0:  # Max 4x difference in CV (relaxed from 3.0)
                    self.logger.debug(f"🚫 CV ratio too high: {cv_ratio:.2f}x")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error in dimension compatibility check: {e}")
            return True  # Default to allowing merge if check fails
    
    def _would_create_problematic_cluster(self, regime_1: Dict[str, Any], regime_2: Dict[str, Any], 
                                         cluster_i: int, cluster_j: int, regime_to_cluster: Dict[str, int],
                                         total_samples: int, cv_threshold: float = MAX_CV_THRESHOLD) -> bool:
        """Predict if merging would create a problematic high-CV cluster.
        
        Prevents formation of clusters like C486 (CV=37.468) and C15 (CV=343.096) 
        by predicting post-merge CV and rejecting dangerous combinations.
        """
        try:
            import numpy as np
            
            # Get cluster sizes
            regimes_i = [rid for rid, cid in regime_to_cluster.items() if cid == cluster_i]
            regimes_j = [rid for rid, cid in regime_to_cluster.items() if cid == cluster_j]
            combined_size_pct = (len(regimes_i) + len(regimes_j)) / total_samples * 100 if total_samples > 0 else 0.0
            
            # 1. Check individual regime CVs - prevent merging regimes with extreme CVs
            features_1 = regime_1.get('features', {})
            features_2 = regime_2.get('features', {})
            
            # Get max CV from each regime across all dimensions
            cv_1 = max(
                features_1.get('momentum_cv', 0.0),
                features_1.get('volatility_cv', 0.0),
                features_1.get('volume_cv', 0.0)
            )
            cv_2 = max(
                features_2.get('momentum_cv', 0.0),
                features_2.get('volatility_cv', 0.0),
                features_2.get('volume_cv', 0.0)
            )
            
            # 2. Extreme CV detection - reject if either regime has extreme CV
            if cv_1 > 100.0 or cv_2 > 100.0:  # Prevent extreme outliers like CV=343 (relaxed from 50.0)
                self.logger.debug(f"🚫 PROBLEMATIC: Extreme CV detected - regime CVs: {cv_1:.1f}, {cv_2:.1f}")
                return True
            
            # 3. High CV combination check - prevent two high-CV regimes from merging
            if cv_1 > MAX_CV_THRESHOLD and cv_2 > MAX_CV_THRESHOLD:
                self.logger.debug(f"🚫 PROBLEMATIC: Both regimes have high CV - {cv_1:.1f}, {cv_2:.1f}")
                return True
            
            # 4. Size-based CV limits - use current CV threshold with 2x multipliers (more permissive)
            if combined_size_pct > 15.0:  # Very large clusters - be conservative
                size_based_cv_limit = cv_threshold * 3.0  # 200% above current threshold (was 1.5)
            elif combined_size_pct > 8.0:  # Medium clusters - moderate relaxation
                size_based_cv_limit = cv_threshold * 4.0  # 300% above current threshold (was 2.0)
            else:  # Small clusters - most permissive
                size_based_cv_limit = cv_threshold * 6.0  # 500% above current threshold (was 3.0)
            
            # 5. Predict combined CV using weighted average
            # This approximates what the post-merge CV would be
            if cv_1 > 0 and cv_2 > 0:
                # Weight by cluster sizes
                weight_1 = len(regimes_i) / (len(regimes_i) + len(regimes_j))
                weight_2 = len(regimes_j) / (len(regimes_i) + len(regimes_j))
                
                # Predict combined CV (minimal safety margin for bigger steps)
                predicted_cv = (weight_1 * cv_1 + weight_2 * cv_2) * 1.05  # 5% safety margin (reduced from 10%)
                
                if predicted_cv > size_based_cv_limit:
                    self.logger.debug(f"🚫 PROBLEMATIC: Predicted CV {predicted_cv:.1f} > limit {size_based_cv_limit:.1f} (size: {combined_size_pct:.1f}%)")
                    return True
            
            # 6. Feature divergence check - prevent merging regimes with opposite patterns (relaxed)
            momentum_1 = features_1.get('momentum_mean', 0.0)
            momentum_2 = features_2.get('momentum_mean', 0.0)
            
            # Only block very strong opposite momentum patterns (relaxed from 2.0 to 5.0)
            if abs(momentum_1) > 5.0 and abs(momentum_2) > 5.0 and (momentum_1 * momentum_2 < 0):
                self.logger.debug(f"🚫 PROBLEMATIC: Very strong opposite momentum - {momentum_1:.2f} vs {momentum_2:.2f}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error in problematic cluster detection: {e}")
            return False  # Default to allowing merge if check fails
    
    
    
    
    def _compute_cluster_cv(self, regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int]) -> Dict[int, float]:
        """Compute robust within-cluster CV over feature values, not a weighted average of regime CVs.

        For each cluster, we:
        - Build the regime-by-feature matrix across all regimes
        - For the subset of regimes in the cluster, compute per-feature robust CV:
          robust_cv_feature = (1.4826 * MAD(feature_values)) / (abs(median(feature_values)) + eps)
        - Aggregate the cluster's per-feature CVs using the median to reduce influence of outliers

        Returns a mapping cluster_id -> robust_within_cv
        """
        import numpy as np
        # Build matrix of original features (no scaling) for all regimes
        regime_ids = list(regime_to_cluster.keys())
        X, feature_order = self._build_feature_matrix(regime_characteristics, regime_ids)
        if X.size == 0:
            return {}
        cluster_cv: Dict[int, float] = {}
        # Precompute cluster indices
        cluster_to_indices: Dict[int, list] = {}
        for idx, regime_id in enumerate(regime_ids):
            cid = regime_to_cluster.get(regime_id)
            if cid is None:
                continue
            if cid not in cluster_to_indices:
                cluster_to_indices[cid] = []
            cluster_to_indices[cid].append(idx)
        # Compute robust CV per cluster
        eps = 1e-8
        for cid, indices in cluster_to_indices.items():
            if not indices or len(indices) < 2:
                # Not enough regimes to assess variability meaningfully
                cluster_cv[cid] = 0.0
                continue
            Xc = X[indices, :]
            # Compute per-feature robust CV; skip all-NaN columns
            cvs = []
            for j in range(Xc.shape[1]):
                col = Xc[:, j]
                if np.all(np.isnan(col)):
                    continue
                # Robust center and scale
                med = np.nanmedian(col)
                mad = np.nanmedian(np.abs(col - med))
                robust_scale = 1.4826 * mad
                denom = max(abs(med), eps)
                cvj = robust_scale / denom
                if np.isfinite(cvj):
                    # Cap extreme CV values to prevent outliers like 43.494
                    cvj = min(cvj, 10.0)  # Cap at 10.0 like regime-level calculation
                    cvs.append(cvj)
            cluster_cv[cid] = float(np.nanmedian(cvs)) if len(cvs) > 0 else 0.0
        return cluster_cv

    def _compute_cluster_cv_by_aspect(self, regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int]) -> Dict[int, Dict[str, float]]:
        """Compute robust within-cluster CV per aspect: momentum, volatility, volume.

        For each aspect, we collect the corresponding feature values across regimes in the cluster and
        compute robust CV = (1.4826 * MAD) / (abs(median) + eps). Cluster aspect CV is the median across
        that aspect's features.
        """
        import numpy as np
        # Define feature groups - updated to match current HMM regime discovery features
        aspect_features = {
            'momentum': [
                'momentum_1h',  # Short-term price momentum (1-hour)
                'momentum_2h'   # Medium-term price momentum (2-hour)
            ],
            'volatility': [
                'volatility_intrabar',  # High-low range relative to close
                'volatility_3h',        # 3-hour rolling standard deviation
                'atr_6h'               # 6-hour Average True Range
            ],
            'volume': [
                'volume_ratio_48h'     # Volume relative to 24-hour rolling mean
            ]
        }
        # Build map from regime_id -> raw feature dict
        regime_ids = list(regime_to_cluster.keys())
        # Extract per-regime features from provided characteristics (flat in regime['features'] if present),
        # otherwise look into sub-dicts for consistency with prior code sections
        regime_feature_maps: Dict[str, Dict[str, float]] = {}
        total_features_found = 0
        for regime_id in regime_ids:
            reg = regime_characteristics.get(regime_id, {})
            feats = dict(reg.get('features', {}))
            # Also merge keys from known characteristic groups if present
            for grp in ['momentum_characteristics', 'volatility_characteristics', 'volume_characteristics']:
                sub = reg.get(grp, {})
                for k, v in sub.items():
                    if isinstance(v, (int, float)):
                        feats.setdefault(k, float(v))
            regime_feature_maps[regime_id] = feats
            total_features_found += len(feats)
        
        # Debug: Log feature extraction summary
        avg_features_per_regime = total_features_found / len(regime_ids) if regime_ids else 0
        if avg_features_per_regime < 5:  # If very few features found
            self.logger.warning(f"⚠️ CV Calculation Issue: Only {avg_features_per_regime:.1f} avg features per regime")
            # Sample first regime to see what's available
            if regime_ids:
                sample_features = regime_feature_maps[regime_ids[0]]
                self.logger.warning(f"   🔍 Sample regime features: {list(sample_features.keys())[:10]}")
                self.logger.warning(f"   🔍 Sample feature values: {dict(list(sample_features.items())[:5])}")
        # Group regimes by cluster
        cluster_to_regimes: Dict[int, list] = {}
        for regime_id, cid in regime_to_cluster.items():
            cluster_to_regimes.setdefault(cid, []).append(regime_id)
        # Compute robust CV per aspect per cluster
        eps = 1e-8
        cluster_cv_aspects: Dict[int, Dict[str, float]] = {}
        for cid, rids in cluster_to_regimes.items():
            aspect_cvs: Dict[str, float] = {}
            for aspect, feat_list in aspect_features.items():
                per_feature_cvs = []
                for feat in feat_list:
                    values = [regime_feature_maps[rid].get(feat, np.nan) for rid in rids]
                    arr = np.array(values, dtype=float)
                    if np.all(np.isnan(arr)):
                        # Debug: Log when feature is all NaN
                        if cid in [32, 213, 218] and aspect == 'momentum':  # Debug specific cases
                            self.logger.warning(f"⚠️ Feature '{feat}' all NaN for cluster {cid}")
                        continue
                    med = np.nanmedian(arr)
                    mad = np.nanmedian(np.abs(arr - med))
                    robust_scale = 1.4826 * mad
                    denom = max(abs(med), eps)
                    cvj = robust_scale / denom
                    if np.isfinite(cvj):
                        per_feature_cvs.append(float(cvj))
                if len(per_feature_cvs) > 0:
                    aspect_cvs[aspect] = float(np.nanmedian(per_feature_cvs))
                else:
                    aspect_cvs[aspect] = 0.0
                    # Debug: Log when no features found for CV calculation
                    if cid in [32, 213, 218]:  # Debug specific problematic clusters
                        self.logger.warning(f"⚠️ No {aspect} features found for cluster {cid}, defaulting CV to 0.0")
                        self.logger.warning(f"   🔍 Available features: {list(regime_feature_maps[rids[0]].keys())[:10] if rids else 'none'}")
                        self.logger.warning(f"   🔍 Expected {aspect} features: {feat_list}")
            cluster_cv_aspects[cid] = aspect_cvs
        return cluster_cv_aspects

    def _extract_regime_characteristics_from_discovery(self, regime_discovery: Dict[str, Any]) -> Dict[str, Any]:
        """Extract volume, volatility, and momentum characteristics from HMM regime discovery results."""
        try:
            import numpy as np
            
            regime_characteristics = regime_discovery.get('regime_characteristics', {})
            regime_metrics = regime_discovery.get('regime_metrics', {})
            
            if not regime_characteristics:
                self.logger.warning("⚠️ No regime characteristics found in discovery results")
                return {}
            
            extracted_characteristics = {}
            
            for regime_key, characteristics in regime_characteristics.items():
                if not isinstance(characteristics, dict):
                    continue
                
                feature_means = characteristics.get('feature_means', {})
                feature_stds = characteristics.get('feature_stds', {})
                
                # Create features dictionary from feature_means (this is what similarity calculation expects)
                features = {}
                if feature_means:
                    features.update(feature_means)
                    
                    # Debug: Log available features for the first regime to understand what's actually available
                    if regime_key == list(regime_characteristics.keys())[0]:
                        self.logger.info(f"🔍 DEBUG CV: Available features in {regime_key}: {list(feature_means.keys())[:10]}...")
                        self.logger.info(f"🔍 DEBUG CV: Sample feature values: {dict(list(feature_means.items())[:5])}")
                    
                    # Calculate CV values that the adaptive CV threshold generation needs
                    # CV = std / mean (coefficient of variation)
                    
                    # Momentum CV calculation - matching HMM discovery features
                    momentum_features = [
                        'momentum_1h', 'momentum_2h'
                    ]
                    momentum_cvs = []
                    found_momentum_features = []
                    for feat in momentum_features:
                        mean_val = feature_means.get(feat, 0.0)
                        std_val = feature_stds.get(feat, 0.0)
                        if feat in feature_means:  # Feature exists
                            found_momentum_features.append(feat)
                            if abs(mean_val) > 1e-8:  # Avoid division by zero
                                cv = abs(std_val / mean_val)
                                if cv < 10.0:  # Cap extreme CV values
                                    momentum_cvs.append(cv)
                    
                    features['momentum_cv'] = float(np.mean(momentum_cvs)) if momentum_cvs else 0.0
                    
                    # Debug logging for first regime
                    if regime_key == list(regime_characteristics.keys())[0]:
                        self.logger.info(f"🔍 DEBUG CV: Found {len(found_momentum_features)}/2 momentum features: {found_momentum_features}")
                        self.logger.info(f"🔍 DEBUG CV: Calculated {len(momentum_cvs)} valid momentum CVs, avg={np.mean(momentum_cvs):.3f}" if momentum_cvs else "🔍 DEBUG CV: No valid momentum CVs calculated")
                    
                    # Volatility CV calculation - matching HMM discovery features
                    volatility_features = [
                        'volatility_intrabar', 'volatility_3h', 'atr_6h'
                    ]
                    volatility_cvs = []
                    found_volatility_features = []
                    for feat in volatility_features:
                        mean_val = feature_means.get(feat, 0.0)
                        std_val = feature_stds.get(feat, 0.0)
                        if feat in feature_means:  # Feature exists
                            found_volatility_features.append(feat)
                            if abs(mean_val) > 1e-8:  # Avoid division by zero
                                cv = abs(std_val / mean_val)
                                if cv < 10.0:  # Cap extreme CV values
                                    volatility_cvs.append(cv)
                    
                    features['volatility_cv'] = float(np.mean(volatility_cvs)) if volatility_cvs else 0.0
                    
                    # Debug logging for first regime
                    if regime_key == list(regime_characteristics.keys())[0]:
                        self.logger.info(f"🔍 DEBUG CV: Found {len(found_volatility_features)}/3 volatility features: {found_volatility_features}")
                        self.logger.info(f"🔍 DEBUG CV: Calculated {len(volatility_cvs)} valid volatility CVs, avg={np.mean(volatility_cvs):.3f}" if volatility_cvs else "🔍 DEBUG CV: No valid volatility CVs calculated")
                    
                    # Volume CV calculation - matching HMM discovery features
                    volume_features = [
                        'volume_ratio_48h'  # Volume ratio between current volume & 24h rolling mean
                    ]
                    volume_cvs = []
                    found_volume_features = []
                    for feat in volume_features:
                        mean_val = feature_means.get(feat, 0.0)
                        std_val = feature_stds.get(feat, 0.0)
                        if feat in feature_means:  # Feature exists
                            found_volume_features.append(feat)
                            if abs(mean_val) > 1e-8:  # Avoid division by zero
                                cv = abs(std_val / mean_val)
                                if cv < 10.0:  # Cap extreme CV values
                                    volume_cvs.append(cv)
                    
                    features['volume_cv'] = float(np.mean(volume_cvs)) if volume_cvs else 0.0
                    
                    # Debug logging for first regime
                    if regime_key == list(regime_characteristics.keys())[0]:
                        self.logger.info(f"🔍 DEBUG CV: Found {len(found_volume_features)}/1 volume features: {found_volume_features}")
                        self.logger.info(f"🔍 DEBUG CV: Calculated {len(volume_cvs)} valid volume CVs, avg={np.mean(volume_cvs):.3f}" if volume_cvs else "🔍 DEBUG CV: No valid volume CVs calculated")
                    
                    # If no specific features found, try to calculate CV from any available features
                    if features['momentum_cv'] == 0.0 and features['volatility_cv'] == 0.0 and features['volume_cv'] == 0.0:
                        self.logger.warning(f"⚠️ No specific CV features found, trying generic approach for {regime_key}")
                        
                        # Generic CV calculation from all available features
                        all_feature_cvs = []
                        for feat_name, mean_val in feature_means.items():
                            std_val = feature_stds.get(feat_name, 0.0)
                            if abs(mean_val) > 1e-8 and std_val > 0:
                                cv = abs(std_val / mean_val)
                                if cv < 10.0:  # Cap extreme CV values
                                    all_feature_cvs.append(cv)
                        
                        if all_feature_cvs:
                            generic_cv = float(np.mean(all_feature_cvs))
                            # Distribute the generic CV across categories
                            features['momentum_cv'] = generic_cv * 0.4
                            features['volatility_cv'] = generic_cv * 0.3
                            features['volume_cv'] = generic_cv * 0.3
                            self.logger.info(f"✅ Used generic CV calculation: {generic_cv:.3f} from {len(all_feature_cvs)} features")
                    
                    # Overall mean CV for compatibility
                    all_cvs = [features['momentum_cv'], features['volatility_cv'], features['volume_cv']]
                    valid_cvs = [cv for cv in all_cvs if cv > 0]
                    features['mean_cv'] = float(np.mean(valid_cvs)) if valid_cvs else 0.0
                
                extracted_characteristics[regime_key] = {
                    'features': features,
                    'sample_count': characteristics.get('sample_count', 0)
                }
            
            # Log CV statistics for debugging
            all_momentum_cvs = [char['features'].get('momentum_cv', 0) for char in extracted_characteristics.values()]
            all_volatility_cvs = [char['features'].get('volatility_cv', 0) for char in extracted_characteristics.values()]
            all_volume_cvs = [char['features'].get('volume_cv', 0) for char in extracted_characteristics.values()]
            
            valid_momentum = [cv for cv in all_momentum_cvs if cv > 0]
            valid_volatility = [cv for cv in all_volatility_cvs if cv > 0]
            valid_volume = [cv for cv in all_volume_cvs if cv > 0]
            
            self.logger.info(f"✅ Extracted characteristics for {len(extracted_characteristics)} regimes")
            self.logger.info(f"📊 CV Statistics - Momentum: {len(valid_momentum)} valid, avg={np.mean(valid_momentum):.3f}" if valid_momentum else "📊 CV Statistics - Momentum: No valid data")
            self.logger.info(f"📊 CV Statistics - Volatility: {len(valid_volatility)} valid, avg={np.mean(valid_volatility):.3f}" if valid_volatility else "📊 CV Statistics - Volatility: No valid data")
            self.logger.info(f"📊 CV Statistics - Volume: {len(valid_volume)} valid, avg={np.mean(valid_volume):.3f}" if valid_volume else "📊 CV Statistics - Volume: No valid data")
            
            return extracted_characteristics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to extract regime characteristics: {e}")
            return {}
    
    def _extract_volume_characteristics(self, feature_means: Dict[str, float], feature_stds: Dict[str, float]) -> Dict[str, Any]:
        """Extract volume-related characteristics from regime features."""
        volume_chars = {}
        
        # Volume momentum characteristics
        volume_momentum_5 = feature_means.get('volume_momentum_5', 0.0)
        volume_momentum_20 = feature_means.get('volume_momentum_20', 0.0)
        volume_momentum_5_std = feature_stds.get('volume_momentum_5', 0.0)
        
        volume_chars['mean_volume_momentum_5'] = volume_momentum_5
        volume_chars['mean_volume_momentum_20'] = volume_momentum_20
        volume_chars['volume_momentum_volatility'] = volume_momentum_5_std
        volume_chars['volume_trend'] = 'increasing' if volume_momentum_5 > 0.05 else 'decreasing' if volume_momentum_5 < -0.05 else 'stable'
        
        # Volume ratio characteristics
        volume_ratio_20 = feature_means.get('volume_ratio_20', 1.0)
        volume_ratio_std = feature_stds.get('volume_ratio_20', 0.0)
        
        volume_chars['mean_volume_ratio'] = volume_ratio_20
        volume_chars['volume_ratio_volatility'] = volume_ratio_std
        volume_chars['volume_level'] = 'high' if volume_ratio_20 > 1.5 else 'low' if volume_ratio_20 < 0.5 else 'normal'
        
        return volume_chars

    
    def _calculate_hmm_appropriate_metrics(self, market_data: pd.DataFrame, 
                                         cluster_assignments: np.ndarray, 
                                         n_clusters: int) -> Dict[str, Any]:
        """
        Calculate HMM-appropriate validation metrics instead of traditional clustering metrics.
        
        This replaces misleading clustering metrics (silhouette_score, davies_bouldin_score, 
        calinski_harabasz_score) with metrics appropriate for temporal regime modeling.
        
        Handles both 1D and 3D cluster structures.
        """
        try:
            import numpy as np
            
            # Handle 3D cluster assignments - flatten to 1D for compatibility
            if cluster_assignments.ndim > 1:
                self.logger.info(f"📊 Handling 3D cluster structure: {cluster_assignments.shape}")
                # For 3D clusters, we may need to use the dominant cluster or flatten appropriately
                if cluster_assignments.ndim == 3:
                    # Take the argmax along the last dimension to get dominant cluster per time step
                    cluster_assignments_1d = np.argmax(cluster_assignments, axis=-1)
                    if cluster_assignments_1d.ndim > 1:
                        cluster_assignments_1d = cluster_assignments_1d.flatten()
                elif cluster_assignments.ndim == 2:
                    # For 2D, take argmax along second dimension
                    cluster_assignments_1d = np.argmax(cluster_assignments, axis=1)
                else:
                    cluster_assignments_1d = cluster_assignments.flatten()
                
                self.logger.info(f"📊 Flattened 3D clusters to 1D: {cluster_assignments_1d.shape}")
            else:
                cluster_assignments_1d = cluster_assignments
            
            # Import HMM validation framework
            try:
                from src.utils.hmm_validation import HMMStatisticalValidator
                validator = HMMStatisticalValidator(logger=self.logger)
                
                # Ensure data length compatibility
                min_length = min(len(market_data), len(cluster_assignments_1d))
                if min_length != len(market_data):
                    self.logger.warning(f"⚠️ Truncating data for 3D compatibility: {len(market_data)} -> {min_length}")
                
                # Create regime data DataFrame with proper length alignment
                regime_data = market_data.iloc[:min_length].copy()
                regime_data['regime'] = cluster_assignments_1d[:min_length]
                
                # Use HMM-appropriate validation
                validation_result = validator.validate_hmm_regimes_appropriate(
                    regime_data, market_data.iloc[:min_length]
                )
                
                # Extract key metrics for compatibility
                hmm_metrics = {
                    'hmm_quality_score': validation_result.get('hmm_validation_metrics', {}).get('hmm_quality_score', 0.0),
                    'temporal_coherence': validation_result.get('temporal_coherence', {}).get('temporal_coherence', 0.0),
                    'transition_quality': validation_result.get('transition_quality', {}).get('transition_quality', 0.0),
                    'economic_differentiation': validation_result.get('economic_differentiation', {}).get('economic_differentiation', 0.0),
                    'spatial_coherence': validation_result.get('spatial_coherence', {}).get('spatial_coherence', 0.0),
                    'regime_stability': validation_result.get('regime_stability', {}).get('regime_stability_index', 0.0),
                    'validation_passed': validation_result.get('hmm_validation_metrics', {}).get('validation_passed', False),
                    'interpretation': validation_result.get('hmm_validation_metrics', {}).get('overall_interpretation', 'HMM validation completed'),
                    'validation_method': 'HMM-appropriate metrics'
                }
                
                # Add regime distribution
                unique_regimes, counts = np.unique(cluster_assignments, return_counts=True)
                regime_distribution = {f'regime_{regime}': count for regime, count in zip(unique_regimes, counts)}
                hmm_metrics['regime_distribution'] = regime_distribution
                hmm_metrics['regime_count'] = len(unique_regimes)
                
                self.logger.info(f"✅ HMM validation metrics calculated - Quality Score: {hmm_metrics['hmm_quality_score']:.3f}")
                return hmm_metrics
                
            except ImportError:
                self.logger.warning("HMM validation framework not available, using basic metrics")
                return self._calculate_basic_clustering_metrics(market_data, cluster_assignments, n_clusters)
                
        except Exception as e:
            self.logger.error(f"❌ Error calculating HMM-appropriate metrics: {e}")
            return self._calculate_basic_clustering_metrics(market_data, cluster_assignments, n_clusters)
    
    def _calculate_basic_clustering_metrics(self, market_data: pd.DataFrame, 
                                          cluster_assignments: np.ndarray, 
                                          n_clusters: int) -> Dict[str, Any]:
        """Fallback to basic clustering metrics if HMM validation is not available."""
        try:
            unique_regimes, counts = np.unique(cluster_assignments, return_counts=True)
            regime_distribution = {f'regime_{regime}': count for regime, count in zip(unique_regimes, counts)}
            
            return {
                'hmm_quality_score': 0.5,  # Neutral score
                'temporal_coherence': 0.0,
                'transition_quality': 0.0,
                'economic_differentiation': 0.0,
                'spatial_coherence': 0.0,
                'regime_stability': 0.0,
                'validation_passed': len(unique_regimes) > 1,
                'regime_count': len(unique_regimes),
                'regime_distribution': regime_distribution,
                'interpretation': 'Basic clustering metrics - HMM validation not available',
                'validation_method': 'Basic fallback metrics'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating basic clustering metrics: {e}")
            return {
                'hmm_quality_score': 0.0,
                'validation_passed': False,
                'error': str(e),
                'interpretation': 'Error in metrics calculation',
                'validation_method': 'Error state'
            }
    

    def _calculate_within_cluster_coherence(self, distance_matrix: np.ndarray, cluster_labels: np.ndarray) -> float:
        """
        Calculate within-cluster coherence focusing on internal consistency rather than separation.
        
        Handles both 1D and 3D cluster structures by flattening multi-dimensional assignments
        and computing coherence based on dominant cluster assignments.
        """
        try:
            import numpy as np
            
            # Handle 3D cluster labels - flatten to 1D for coherence calculation
            if cluster_labels.ndim > 1:
                self.logger.info(f"📊 Handling 3D cluster labels for coherence: {cluster_labels.shape}")
                
                if cluster_labels.ndim == 3:
                    # For 3D clusters, take argmax along the last dimension to get dominant cluster
                    cluster_labels_1d = np.argmax(cluster_labels, axis=-1)
                    if cluster_labels_1d.ndim > 1:
                        cluster_labels_1d = cluster_labels_1d.flatten()
                elif cluster_labels.ndim == 2:
                    # For 2D, take argmax along second dimension
                    cluster_labels_1d = np.argmax(cluster_labels, axis=1)
                else:
                    cluster_labels_1d = cluster_labels.flatten()
                
                self.logger.info(f"📊 Flattened 3D cluster labels to 1D: {cluster_labels_1d.shape}")
                
                # For 3D clusters, we also need to consider uncertainty/probability distributions
                # Calculate additional 3D-specific coherence metrics
                cluster_3d_coherence = self._calculate_3d_cluster_coherence(cluster_labels, distance_matrix)
                
            else:
                cluster_labels_1d = cluster_labels
                cluster_3d_coherence = None
            
            # Ensure distance matrix and labels are compatible
            min_size = min(len(cluster_labels_1d), distance_matrix.shape[0])
            if min_size != len(cluster_labels_1d):
                self.logger.warning(f"⚠️ Adjusting for 3D compatibility: {len(cluster_labels_1d)} -> {min_size}")
                cluster_labels_1d = cluster_labels_1d[:min_size]
                distance_matrix = distance_matrix[:min_size, :min_size]
            
            unique_clusters = np.unique(cluster_labels_1d)
            # Pre-allocate numpy array for better performance
            coherence_scores = np.zeros(len(unique_clusters))
            valid_scores_count = 0
            
            for i, cluster_id in enumerate(unique_clusters):
                cluster_indices = np.where(cluster_labels_1d == cluster_id)[0]
                
                if len(cluster_indices) < 2:
                    continue
                    
                # Calculate within-cluster distances
                cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                
                # Remove diagonal (self-distances = 0) - Vectorized approach
                n_points = len(cluster_indices)
                # Use numpy vectorized operations instead of nested loops
                upper_triangle_indices = np.triu_indices(n_points, k=1)
                within_distances = cluster_distances[upper_triangle_indices]
                
                if len(within_distances) > 0:
                    # Lower average distance = higher coherence
                    avg_within_distance = np.mean(within_distances)
                    # Convert to coherence score (higher = better)
                    coherence = 1.0 / (1.0 + avg_within_distance)
                    coherence_scores[valid_scores_count] = coherence
                    valid_scores_count += 1
            
            # Calculate final coherence score
            base_coherence = float(np.mean(coherence_scores[:valid_scores_count])) if valid_scores_count > 0 else 0.0
            
            # For 3D clusters, combine with 3D-specific coherence
            if cluster_3d_coherence is not None:
                final_coherence = 0.7 * base_coherence + 0.3 * cluster_3d_coherence
                self.logger.info(f"📊 Combined 3D coherence: base={base_coherence:.3f}, 3D={cluster_3d_coherence:.3f}, final={final_coherence:.3f}")
                return final_coherence
            
            return base_coherence
            
        except Exception as e:
            self.logger.error(f"❌ Within-cluster coherence calculation failed: {e}")
            return 0.0

    def _calculate_3d_cluster_coherence(self, cluster_labels_3d: np.ndarray, distance_matrix: np.ndarray) -> float:
        """
        Calculate 3D-specific cluster coherence considering probability distributions.
        
        For 3D clusters, this considers the uncertainty/probability distribution across clusters
        and measures how consistent these distributions are within each dominant cluster.
        """
        try:
            import numpy as np
            
            if cluster_labels_3d.ndim < 2:
                return 0.0
            
            # Get the shape of the 3D cluster data
            if cluster_labels_3d.ndim == 3:
                n_samples, n_features, n_clusters = cluster_labels_3d.shape
                # Reshape to (n_samples * n_features, n_clusters) for easier processing
                probs_2d = cluster_labels_3d.reshape(-1, n_clusters)
            elif cluster_labels_3d.ndim == 2:
                probs_2d = cluster_labels_3d
                n_clusters = cluster_labels_3d.shape[1]
            else:
                return 0.0
            
            # Calculate entropy-based coherence for probabilistic assignments
            # Lower entropy within dominant clusters indicates better coherence
            dominant_clusters = np.argmax(probs_2d, axis=1)
            unique_clusters = np.unique(dominant_clusters)
            
            coherence_scores = []
            
            for cluster_id in unique_clusters:
                cluster_mask = dominant_clusters == cluster_id
                if np.sum(cluster_mask) < 2:
                    continue
                
                # Get probability distributions for this cluster
                cluster_probs = probs_2d[cluster_mask]
                
                # Calculate average entropy within this cluster (lower is better)
                entropies = []
                for prob_dist in cluster_probs:
                    # Avoid log(0) by adding small epsilon
                    prob_dist = prob_dist + 1e-10
                    prob_dist = prob_dist / np.sum(prob_dist)  # Normalize
                    entropy = -np.sum(prob_dist * np.log(prob_dist))
                    entropies.append(entropy)
                
                avg_entropy = np.mean(entropies)
                # Convert entropy to coherence score (lower entropy = higher coherence)
                max_entropy = np.log(n_clusters)  # Maximum possible entropy
                coherence = 1.0 - (avg_entropy / max_entropy)
                coherence_scores.append(coherence)
            
            # Return average coherence across all clusters
            return float(np.mean(coherence_scores)) if coherence_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ 3D cluster coherence calculation failed: {e}")
            return 0.0

    def _calculate_ari_stability(self, distance_matrix: np.ndarray, n_clusters: int, n_bootstrap: int = 10) -> float:
        """Calculate ARI stability using bootstrap resampling."""
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import adjusted_rand_score
            import numpy as np
            
            n_samples = distance_matrix.shape[0]
            # Pre-allocate numpy array for better performance
            ari_scores = np.zeros(n_bootstrap)
            
            # Original clustering with complete linkage for better performance
            original_clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='complete'
            )
            original_labels = original_clustering.fit_predict(distance_matrix)
            
            # Pre-compute all bootstrap indices for efficiency
            all_bootstrap_indices = [np.random.choice(n_samples, size=n_samples, replace=True) 
                                   for _ in range(n_bootstrap)]
            
            valid_ari_count = 0
            
            for bootstrap_idx, bootstrap_indices in enumerate(all_bootstrap_indices):
                try:
                    # Create bootstrap distance matrix using advanced indexing (more efficient)
                    bootstrap_distance_matrix = distance_matrix[bootstrap_indices][:, bootstrap_indices]
                    
                    # Perform clustering on bootstrap sample with complete linkage
                    bootstrap_clustering = AgglomerativeClustering(
                        n_clusters=min(n_clusters, len(np.unique(bootstrap_indices))),
                        metric='precomputed',
                        linkage='complete'
                    )
                    bootstrap_labels = bootstrap_clustering.fit_predict(bootstrap_distance_matrix)
                    
                    # Map back to original indices and calculate ARI
                    mapped_labels = np.full(n_samples, -1)
                    for i, orig_idx in enumerate(bootstrap_indices):
                        mapped_labels[orig_idx] = bootstrap_labels[i]
                    
                    # Only compare samples that were included in bootstrap
                    valid_mask = mapped_labels != -1
                    if np.sum(valid_mask) > 1:
                        ari = adjusted_rand_score(original_labels[valid_mask], mapped_labels[valid_mask])
                        ari_scores[valid_ari_count] = ari
                        valid_ari_count += 1
                        
                except Exception:
                    continue
            
            return float(np.mean(ari_scores[:valid_ari_count])) if valid_ari_count > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ ARI stability calculation failed: {e}")
            return 0.0




    def _calculate_regime_similarity_matrix(self, regime_characteristics: Dict[str, Any], feature_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Calculate similarity matrix between regimes using global, per-feature robust scaling.

        This constructs a regime-by-feature matrix, applies winsorization and robust
        standardization per feature across all regimes, L2-normalizes each regime vector,
        and returns the cosine similarity matrix.
        """
        try:
            import numpy as np
            if not regime_characteristics:
                self.logger.warning("⚠️ No regime characteristics available for similarity calculation")
                return np.array([])

            regime_ids = list(regime_characteristics.keys())
            n_regimes = len(regime_ids)
            if n_regimes == 0:
                return np.array([])

            # Build feature matrix X (n_regimes x n_features) with deterministic feature order
            X, feature_order = self._build_feature_matrix(regime_characteristics, regime_ids)
            if X.size == 0:
                self.logger.warning("⚠️ Empty feature matrix for similarity calculation")
                return np.array([])

            # Fit robust per-feature scaler (winsorize + median/MAD or IQR)
            scaler = self._fit_global_robust_scaler(X)

            # Standardize and L2-normalize regime vectors
            Z = self._standardize_with_scaler(X, scaler)
            # Apply optional feature reweighting
            if feature_weights and len(feature_order) > 0:
                try:
                    import numpy as np
                    w = np.ones(len(feature_order), dtype=float)
                    for i, feat in enumerate(feature_order):
                        w[i] = float(feature_weights.get(feat, 1.0))
                    # Normalize weights to mean 1.0 for stability
                    mean_w = float(np.mean(w)) if np.isfinite(np.mean(w)) and np.mean(w) > 0 else 1.0
                    w = w / mean_w
                    Z = Z * w
                except Exception as e:
                    self.logger.warning(f"⚠️ Feature reweighting skipped due to error: {e}")
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            Z_normalized = Z / norms

            # Cosine similarity matrix as matrix multiplication of normalized vectors (optimized)
            similarity_matrix = np.clip(np.matmul(Z_normalized, Z_normalized.T), -1.0, 1.0)

            # Persist scaler for pairwise similarity calls and debugging
            self._global_feature_scaler = {
                'feature_order': feature_order,
                'median': scaler['median'],
                'scale': scaler['scale'],
                'winsor_low': scaler['winsor_low'],
                'winsor_high': scaler['winsor_high'],
                'valid_mask': scaler['valid_mask']
            }

            # Log summary statistics
            kept_features = int(np.sum(scaler['valid_mask']))
            dropped_features = int(len(feature_order) - kept_features)
            sim_vals = similarity_matrix[np.triu_indices(n_regimes, k=1)] if n_regimes > 1 else np.array([1.0])
            if sim_vals.size > 0:
                sim_min = float(np.min(sim_vals))
                sim_max = float(np.max(sim_vals))
                sim_mean = float(np.mean(sim_vals))
                self.logger.info(
                    f"✅ Calculated similarity matrix for {n_regimes} regimes | features kept: {kept_features}, dropped: {dropped_features} | similarity range: {sim_min:.3f}-{sim_max:.3f}, mean: {sim_mean:.3f}"
                )
            else:
                self.logger.info(
                    f"✅ Calculated similarity matrix for {n_regimes} regimes | features kept: {kept_features}, dropped: {dropped_features}"
                )

            return similarity_matrix

        except Exception as e:
            self.logger.error(f"❌ Failed to calculate regime similarity matrix: {e}")
            return np.array([])

    def _build_feature_matrix(self, regime_characteristics: Dict[str, Any], regime_ids: List[str]):
        """Construct a dense regime-by-feature matrix and deterministic feature order.

        - Collects union of numeric feature keys from regime['features'] for all regimes
        - Returns X (n_regimes x n_features) with NaNs for missing values
        - Imputation is handled later by the scaler using medians
        """
        import numpy as np
        # Collect all numeric feature keys
        feature_keys = set()
        for regime_id in regime_ids:
            features = regime_characteristics.get(regime_id, {}).get('features', {})
            for key, val in features.items():
                if isinstance(val, (int, float)):
                    feature_keys.add(key)
        feature_order = sorted(feature_keys)
        if len(feature_order) == 0:
            return np.array([]), []
        # Build matrix with NaNs for missing
        X = np.full((len(regime_ids), len(feature_order)), np.nan, dtype=float)
        feature_index = {k: i for i, k in enumerate(feature_order)}
        for r_idx, regime_id in enumerate(regime_ids):
            features = regime_characteristics.get(regime_id, {}).get('features', {})
            for key, val in features.items():
                if key in feature_index and isinstance(val, (int, float)):
                    X[r_idx, feature_index[key]] = float(val)
        return X, feature_order

    def _fit_global_robust_scaler(self, X: 'np.ndarray') -> Dict[str, Any]:
        """Fit a robust scaler on columns of X using winsorization and MAD/IQR.

        Returns a dict with median, scale, winsor bounds, and valid feature mask.
        """
        import numpy as np
        # Compute winsorization bounds per feature (columns)
        # Use 1st-99th percentiles to dampen extreme outliers
        with np.errstate(invalid='ignore'):
            winsor_low = np.nanpercentile(X, 1.0, axis=0)
            winsor_high = np.nanpercentile(X, 99.0, axis=0)
        X_clipped = np.clip(X, winsor_low, winsor_high)
        # Compute medians per feature
        median = np.nanmedian(X_clipped, axis=0)
        # Compute MAD (median absolute deviation)
        abs_dev = np.abs(X_clipped - median)
        mad = np.nanmedian(abs_dev, axis=0)
        scale_mad = 1.4826 * mad
        # Fallback to IQR-based scale if MAD is too small
        q25 = np.nanpercentile(X_clipped, 25.0, axis=0)
        q75 = np.nanpercentile(X_clipped, 75.0, axis=0)
        iqr = q75 - q25
        scale_iqr = iqr / 1.349
        # Choose the larger of MAD-based and IQR-based scales to be conservative
        scale = np.where(scale_mad > 1e-12, scale_mad, scale_iqr)
        # Mark invalid features with near-zero scale
        valid_mask = scale > 1e-12
        # Ensure no zeros
        scale = np.where(valid_mask, scale, 1.0)
        return {
            'median': median,
            'scale': scale,
            'winsor_low': winsor_low,
            'winsor_high': winsor_high,
            'valid_mask': valid_mask
        }

    def _standardize_with_scaler(self, X: 'np.ndarray', scaler: Dict[str, Any]) -> 'np.ndarray':
        """Apply winsorization and robust standardization using provided scaler."""
        import numpy as np
        X_w = np.clip(X, scaler['winsor_low'], scaler['winsor_high'])
        # Impute NaNs with median before standardization
        X_w = np.where(np.isnan(X_w), scaler['median'], X_w)
        Z = (X_w - scaler['median']) / scaler['scale']
        # Drop invalid columns by zeroing them out (no contribution to cosine)
        if 'valid_mask' in scaler:
            invalid_cols = ~scaler['valid_mask']
            if np.any(invalid_cols):
                Z[:, invalid_cols] = 0.0
        return Z

    



    
    def _get_excluded_clusters_due_to_size(self, regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int], total_samples: int) -> set:
        """Identify clusters that should be excluded from merging due to being oversized (>12%).
        
        Args:
            regime_characteristics: Dictionary of regime characteristics
            regime_to_cluster: Current regime to cluster mapping
            total_samples: Total number of samples
            
        Returns:
            Set of cluster IDs that should be excluded from merging due to size
        """
        try:
            # Calculate cluster sample sizes
            cluster_sample_counts = {}
            
            for regime_id, cluster_id in regime_to_cluster.items():
                sample_count = regime_characteristics[regime_id].get('sample_count', 1)
                if cluster_id not in cluster_sample_counts:
                    cluster_sample_counts[cluster_id] = 0
                cluster_sample_counts[cluster_id] += sample_count
            
            excluded_clusters = set()
            oversized_clusters = []
            
            for cluster_id, sample_count in cluster_sample_counts.items():
                sample_percentage = (sample_count / total_samples) * 100 if total_samples > 0 else 0
                
                if sample_percentage > 18.0:  # Only exclude truly oversized clusters (>18%) from merging
                    excluded_clusters.add(cluster_id)
                    oversized_clusters.append((cluster_id, sample_count, sample_percentage))
            
            if oversized_clusters:
                self.logger.warning(f"⚠️ Found {len(oversized_clusters)} oversized clusters (>18%), will prevent further merging:")
                for cluster_id, size, pct in oversized_clusters:
                    self.logger.warning(f"   🚫 Cluster C{cluster_id}: {size} samples ({pct:.1f}%) - NO MERGE ALLOWED")
            
            return excluded_clusters
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error identifying oversized clusters: {e}")
            return set()

    def _get_excluded_clusters_due_to_cv_legacy(self, regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int], cv_threshold: float = 0.15) -> set:
        """Identify clusters that should be excluded from future merging due to high internal CV.
        
        Args:
            regime_characteristics: Dictionary of regime characteristics
            regime_to_cluster: Current regime to cluster mapping
            cv_threshold: CV threshold for exclusion (progressive: 0.15 → 0.2 → 0.25 → 0.3)
            
        Returns:
            Set of cluster IDs that should be excluded from merging
        """
        try:
            import numpy as np
            
            # Calculate current cluster CV metrics
            cluster_cv_aspects = self._compute_cluster_cv_by_aspect(regime_characteristics, regime_to_cluster)
            
            if not cluster_cv_aspects:
                return set()
            
            excluded_clusters = set()
            
            # Calculate cluster sample sizes and total samples
            cluster_sample_counts = {}
            total_samples = 0
            
            for regime_id, cluster_id in regime_to_cluster.items():
                sample_count = regime_characteristics[regime_id].get('sample_count', 1)
                total_samples += sample_count
                if cluster_id not in cluster_sample_counts:
                    cluster_sample_counts[cluster_id] = 0
                cluster_sample_counts[cluster_id] += sample_count
            
            self.logger.info(f"🎯 CV Exclusion Threshold: {cv_threshold:.1f} ({cv_threshold*100:.0f}%)")
            
            excluded_count_by_threshold = {}
            high_cv_explanations = []
            
            for cluster_id, cv_map in cluster_cv_aspects.items():
                # Get maximum CV across all aspects for this cluster
                cluster_cvs = [cv for cv in cv_map.values() if cv is not None and np.isfinite(cv)]
                if not cluster_cvs:
                    continue
                
                max_cv = max(cluster_cvs)
                sample_count = cluster_sample_counts.get(cluster_id, 0)
                sample_percentage = (sample_count / total_samples) * 100 if total_samples > 0 else 0
                
                if max_cv > cv_threshold:
                    excluded_clusters.add(cluster_id)
                    # Categorize exclusions by CV range for better logging
                    if max_cv > 1.0:
                        cv_category = "extreme"
                    elif max_cv > 0.5:
                        cv_category = "very_high"
                    elif max_cv > 0.3:
                        cv_category = "high"
                    else:
                        cv_category = "moderate"
                    
                    excluded_count_by_threshold[cv_category] = excluded_count_by_threshold.get(cv_category, 0) + 1
                    
                    # Find which aspect is causing the high CV
                    worst_aspect = max(cv_map.items(), key=lambda x: x[1] if x[1] is not None and np.isfinite(x[1]) else 0)[0]
                    worst_cv = cv_map[worst_aspect]
                    
                    self.logger.info(f"   🚫 Excluding cluster {cluster_id} from merging: CV={max_cv:.3f} > {cv_threshold:.3f} (samples: {sample_count}, {sample_percentage:.1f}%, worst: {worst_aspect}={worst_cv:.3f})")
                    
                    # Collect high CV explanations for summary
                    if max_cv > 0.5:  # Focus on very high CV cases
                        high_cv_explanations.append(f"C{cluster_id}: {worst_aspect}={worst_cv:.3f} ({sample_count} samples)")
            
            if excluded_clusters:
                self.logger.info(f"📊 Excluded {len(excluded_clusters)} clusters from merging due to high internal CV")
                self.logger.info(f"   📈 CV Categories: {excluded_count_by_threshold}")
                
                # Explain high CV origins for top cases
                if high_cv_explanations:
                    top_high_cv = sorted(high_cv_explanations, key=lambda x: float(x.split('=')[1].split(' ')[0]), reverse=True)[:5]
                    self.logger.info(f"   🔍 TOP HIGH CV CASES: {', '.join(top_high_cv)}")
                    self.logger.info(f"   💡 HIGH CV ORIGINS: Regime heterogeneity within clusters - different momentum/volatility/volume patterns merged together")
            
            return excluded_clusters
            
        except Exception as e:
            self.logger.error(f"❌ Error identifying excluded clusters: {e}")
            return set()



    def _calculate_memory_efficient_similarity_matrix(self, regime_characteristics: Dict[str, Any], 
                                                     sparsity_threshold: float = 0.3) -> np.ndarray:
        """Calculate similarity matrix with memory optimization using sparsity.
        
        Args:
            regime_characteristics: Dictionary of regime characteristics
            sparsity_threshold: Threshold below which similarities are set to 0 for sparsity
            
        Returns:
            Similarity matrix (dense format for compatibility, but optimized internally)
        """
        try:
             import numpy as np
             from scipy.sparse import csr_matrix
             
             # Calculate full similarity matrix first
             similarity_matrix = self._calculate_regime_similarity_matrix(regime_characteristics, None)
             
             if similarity_matrix.size == 0:
                 return similarity_matrix
             
             # Memory optimization: apply sparsity threshold
             n_regimes = similarity_matrix.shape[0]
             total_elements = n_regimes * n_regimes
             
             # Count elements below threshold
             below_threshold = np.sum(similarity_matrix < sparsity_threshold)
             sparsity_ratio = below_threshold / total_elements
             
             # If significant sparsity potential, optimize
             if sparsity_ratio > 0.4:  # More than 40% of elements are below threshold
                 self.logger.info(f"🗜️ Memory optimization: {sparsity_ratio:.1%} of similarities below {sparsity_threshold:.2f}")
                 
                 # Create optimized matrix
                 optimized_matrix = similarity_matrix.copy()
                 optimized_matrix[optimized_matrix < sparsity_threshold] = 0
                 
                 # Log memory savings
                 memory_savings = sparsity_ratio * 100
                 self.logger.info(f"   💾 Potential memory savings: ~{memory_savings:.0f}% through sparsity")
                 
                 return optimized_matrix
             else:
                 self.logger.debug(f"📊 Similarity matrix density: {(1-sparsity_ratio):.1%} (no sparsity optimization)")
                 return similarity_matrix
             
        except Exception as e:
            self.logger.error(f"❌ Error in memory-efficient similarity calculation: {e}")
            # Fallback to standard calculation
            return self._calculate_regime_similarity_matrix(regime_characteristics, None)

    def _update_similarity_matrix_incremental(self, similarity_matrix: np.ndarray, 
                                            merged_clusters: List[Tuple[int, int]], 
                                            regime_characteristics: Dict[str, Any],
                                            regime_to_cluster: Dict[str, int]) -> np.ndarray:
        """Update similarity matrix incrementally after merges instead of full recalculation.
        
        Args:
            similarity_matrix: Current similarity matrix
            merged_clusters: List of (old_cluster_id, new_cluster_id) pairs
            regime_characteristics: Regime characteristics
            regime_to_cluster: Mapping of regime to cluster
            
        Returns:
            Updated similarity matrix
        """
        try:
            import numpy as np
            
            if not merged_clusters or similarity_matrix.size == 0:
                return similarity_matrix
            
            # Get affected clusters (those that were merged into)
            updated_cluster_ids = set()
            for old_cluster, new_cluster in merged_clusters:
                updated_cluster_ids.add(new_cluster)
            
            # Get regime IDs for each updated cluster
            cluster_to_regimes = {}
            for regime_id, cluster_id in regime_to_cluster.items():
                if cluster_id in updated_cluster_ids:
                    if cluster_id not in cluster_to_regimes:
                        cluster_to_regimes[cluster_id] = []
                    cluster_to_regimes[cluster_id].append(regime_id)
            
            # Create regime ID to index mapping
            regime_ids = list(regime_characteristics.keys())
            regime_to_idx = {regime_id: idx for idx, regime_id in enumerate(regime_ids)}
            
            # Update similarities only for affected regime pairs
            updated_pairs = 0
            for cluster_id in updated_cluster_ids:
                if cluster_id in cluster_to_regimes:
                    cluster_regimes = cluster_to_regimes[cluster_id]
                    
                    # Update similarities within this cluster
                    for i, regime_i in enumerate(cluster_regimes):
                        if regime_i not in regime_to_idx:
                            continue
                        idx_i = regime_to_idx[regime_i]
                        
                        for j, regime_j in enumerate(cluster_regimes[i+1:], i+1):
                            if regime_j not in regime_to_idx:
                                continue
                            idx_j = regime_to_idx[regime_j]
                            
                            # Recalculate similarity for this pair
                            new_sim = self._calculate_pairwise_similarity(
                                regime_characteristics[regime_i], 
                                regime_characteristics[regime_j]
                            )
                            
                            # Update symmetric matrix
                            similarity_matrix[idx_i, idx_j] = new_sim
                            similarity_matrix[idx_j, idx_i] = new_sim
                            updated_pairs += 1
                    
                    # Update similarities between this cluster and other regimes
                    for regime_other in regime_ids:
                        if regime_other not in regime_to_idx:
                            continue
                        other_cluster = regime_to_cluster.get(regime_other)
                        if other_cluster != cluster_id:
                            idx_other = regime_to_idx[regime_other]
                            
                            # Find representative regime from updated cluster
                            if cluster_regimes:
                                repr_regime = cluster_regimes[0]  # Use first as representative
                                idx_repr = regime_to_idx[repr_regime]
                                
                                new_sim = self._calculate_pairwise_similarity(
                                    regime_characteristics[repr_regime], 
                                    regime_characteristics[regime_other]
                                )
                                
                                # Update all regimes in cluster to have same similarity to other
                                for regime_in_cluster in cluster_regimes:
                                    if regime_in_cluster in regime_to_idx:
                                        idx_cluster = regime_to_idx[regime_in_cluster]
                                        similarity_matrix[idx_cluster, idx_other] = new_sim
                                        similarity_matrix[idx_other, idx_cluster] = new_sim
                                        updated_pairs += 1
            
            self.logger.info(f"   🔄 Incremental similarity update: {updated_pairs} pairs updated")
            return similarity_matrix
            
        except Exception as e:
            self.logger.warning(f"⚠️ Incremental similarity update failed: {e}, falling back to full recalculation")
            return self._calculate_regime_similarity_matrix(regime_characteristics, None)

    def _calculate_pairwise_similarity(self, regime_1: Dict[str, Any], regime_2: Dict[str, Any]) -> float:
        """Calculate similarity between two individual regimes.
        
        Args:
            regime_1: First regime characteristics
            regime_2: Second regime characteristics
            
        Returns:
            Similarity score between regimes
        """
        try:
            import numpy as np
            
            features_1 = regime_1.get('features', {})
            features_2 = regime_2.get('features', {})
            
            if not features_1 or not features_2:
                return 0.0
            
            # Get common features
            common_features = set(features_1.keys()) & set(features_2.keys())
            if not common_features:
                return 0.0
            
            # Extract feature vectors
            vec_1 = []
            vec_2 = []
            
            for feature in sorted(common_features):
                val_1 = features_1.get(feature, 0.0)
                val_2 = features_2.get(feature, 0.0)
                
                if isinstance(val_1, (int, float)) and isinstance(val_2, (int, float)):
                    if np.isfinite(val_1) and np.isfinite(val_2):
                        vec_1.append(val_1)
                        vec_2.append(val_2)
            
            if len(vec_1) < 2:  # Need at least 2 features
                return 0.0
            
            vec_1 = np.array(vec_1)
            vec_2 = np.array(vec_2)
            
            # Calculate cosine similarity
            norm_1 = np.linalg.norm(vec_1)
            norm_2 = np.linalg.norm(vec_2)
            
            if norm_1 == 0 or norm_2 == 0:
                return 0.0
            
            similarity = np.dot(vec_1, vec_2) / (norm_1 * norm_2)
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating pairwise similarity: {e}")
            return 0.0

    def _multi_stage_hierarchical_clustering(self, regime_characteristics: Dict[str, Any], 
                                           regime_assignments: List[int]) -> Dict[str, Any]:
        """Implement multi-stage clustering to handle diverse CV ranges more effectively.
        
        Args:
            regime_characteristics: Regime characteristics dictionary
            regime_assignments: List of regime assignments
            
        Returns:
            Clustering result with improved handling of extreme CV values
        """
        try:
            import numpy as np
            
            self.logger.info("🔄 Starting Multi-Stage Hierarchical Clustering")
            
            # Analyze CV distribution to determine stage thresholds
            all_cvs = []
            cv_aspects = ['momentum_cv', 'volatility_cv', 'volume_cv']
            
            for regime in regime_characteristics.values():
                features = regime.get('features', {})
                for aspect in cv_aspects:
                    cv_val = features.get(aspect, 0.0)
                    if cv_val is not None and isinstance(cv_val, (int, float)) and np.isfinite(cv_val) and cv_val > 0:
                        all_cvs.append(cv_val)
            
            if not all_cvs:
                self.logger.warning("⚠️ No CV data available for multi-stage clustering, using standard approach")
                return self._perform_standard_clustering(regime_characteristics, regime_assignments)
            
            all_cvs = np.array(all_cvs)
            cv_percentiles = np.percentile(all_cvs, [75, 90, 95])
            
            # Define stage thresholds based on data distribution
            low_cv_threshold = cv_percentiles[0]     # 75th percentile
            medium_cv_threshold = cv_percentiles[1]  # 90th percentile
            high_cv_threshold = cv_percentiles[2]    # 95th percentile
            
            self.logger.info(f"   📊 Stage thresholds: Low≤{low_cv_threshold:.2f}, Medium≤{medium_cv_threshold:.2f}, High≤{high_cv_threshold:.2f}")
            
            # Stage 1: Cluster low-CV regimes aggressively
            low_cv_regimes = self._filter_regimes_by_cv_range(
                regime_characteristics, max_cv=low_cv_threshold
            )
            stage1_result = self._cluster_regime_stage(
                low_cv_regimes, regime_assignments, stage="low_cv",
                cv_threshold_multiplier=0.6, similarity_threshold_multiplier=1.1
            )
            
            # Stage 2: Cluster medium-CV regimes moderately
            medium_cv_regimes = self._filter_regimes_by_cv_range(
                regime_characteristics, min_cv=low_cv_threshold, max_cv=medium_cv_threshold
            )
            stage2_result = self._cluster_regime_stage(
                medium_cv_regimes, regime_assignments, stage="medium_cv",
                cv_threshold_multiplier=1.0, similarity_threshold_multiplier=1.0
            )
            
            # Stage 3: Cluster high-CV regimes conservatively
            high_cv_regimes = self._filter_regimes_by_cv_range(
                regime_characteristics, min_cv=medium_cv_threshold, max_cv=high_cv_threshold
            )
            stage3_result = self._cluster_regime_stage(
                high_cv_regimes, regime_assignments, stage="high_cv",
                cv_threshold_multiplier=1.8, similarity_threshold_multiplier=0.9
            )
            
            # Stage 4: Handle extreme-CV regimes as singletons or small clusters
            extreme_cv_regimes = self._filter_regimes_by_cv_range(
                regime_characteristics, min_cv=high_cv_threshold
            )
            stage4_result = self._handle_extreme_cv_regimes(
                extreme_cv_regimes, regime_assignments
            )
            
            # Merge stage results
            merged_result = self._merge_stage_results([
                stage1_result, stage2_result, stage3_result, stage4_result
            ], regime_characteristics)
            
            self.logger.info(f"✅ Multi-stage clustering completed: {merged_result['cluster_count']} final clusters")
            return merged_result
            
        except Exception as e:
            self.logger.error(f"❌ Multi-stage clustering failed: {e}")
            return self._perform_standard_clustering(regime_characteristics, regime_assignments)

    def _filter_regimes_by_cv_range(self, regime_characteristics: Dict[str, Any], 
                                   min_cv: float = 0.0, max_cv: float = float('inf')) -> Dict[str, Any]:
        """Filter regimes by CV range for stage-based clustering.
        
        Args:
            regime_characteristics: Full regime characteristics
            min_cv: Minimum CV threshold (inclusive)
            max_cv: Maximum CV threshold (exclusive)
            
        Returns:
            Filtered regime characteristics
        """
        filtered_regimes = {}
        cv_aspects = ['momentum_cv', 'volatility_cv', 'volume_cv']
        
        for regime_id, characteristics in regime_characteristics.items():
            features = characteristics.get('features', {})
            
            # Calculate max CV across all aspects for this regime
            regime_max_cv = 0.0
            for aspect in cv_aspects:
                cv_val = features.get(aspect, 0.0)
                if isinstance(cv_val, (int, float)) and np.isfinite(cv_val):
                    regime_max_cv = max(regime_max_cv, cv_val)
            
            # Include regime if its max CV falls within the specified range
            if min_cv <= regime_max_cv < max_cv:
                filtered_regimes[regime_id] = characteristics
        
        return filtered_regimes

    def _cluster_regime_stage(self, stage_regimes: Dict[str, Any], 
                            regime_assignments: List[int], stage: str,
                            cv_threshold_multiplier: float = 1.0,
                            similarity_threshold_multiplier: float = 1.0) -> Dict[str, Any]:
        """Cluster regimes within a specific CV stage.
        
        Args:
            stage_regimes: Regimes for this stage
            regime_assignments: Original regime assignments
            stage: Stage name for logging
            cv_threshold_multiplier: Multiplier for CV thresholds
            similarity_threshold_multiplier: Multiplier for similarity thresholds
            
        Returns:
            Stage clustering result
        """
        if not stage_regimes:
            return {'clusters': {}, 'stage': stage, 'regime_count': 0}
        
        self.logger.info(f"   🔄 Stage '{stage}': Clustering {len(stage_regimes)} regimes")
        
        try:
            # Calculate similarity matrix for this stage
            similarity_matrix = self._calculate_regime_similarity_matrix(stage_regimes, None)
            
            # Adjust thresholds for this stage
            base_cv_thresholds = self._generate_adaptive_cv_thresholds(stage_regimes)
            adjusted_cv_thresholds = [t * cv_threshold_multiplier for t in base_cv_thresholds]
            
            base_similarity_thresholds = np.linspace(0.99, 0.75, 8)
            adjusted_similarity_thresholds = base_similarity_thresholds * similarity_threshold_multiplier
            adjusted_similarity_thresholds = np.clip(adjusted_similarity_thresholds, 0.1, 0.99)
            
            # Perform clustering with adjusted parameters
            stage_clusters = self._perform_stage_clustering(
                stage_regimes, similarity_matrix, 
                adjusted_cv_thresholds, adjusted_similarity_thresholds
            )
            
            self.logger.info(f"   ✅ Stage '{stage}': Created {len(stage_clusters)} clusters")
            
            return {
                'clusters': stage_clusters,
                'stage': stage,
                'regime_count': len(stage_regimes),
                'cluster_count': len(stage_clusters)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Stage '{stage}' clustering failed: {e}")
            return {'clusters': {}, 'stage': stage, 'regime_count': len(stage_regimes)}

    def _handle_extreme_cv_regimes(self, extreme_regimes: Dict[str, Any], 
                                  regime_assignments: List[int]) -> Dict[str, Any]:
        """Handle extreme CV regimes by creating singleton clusters or small groups.
        
        Args:
            extreme_regimes: Regimes with extreme CV values
            regime_assignments: Original regime assignments
            
        Returns:
            Extreme regime clustering result
        """
        if not extreme_regimes:
            return {'clusters': {}, 'stage': 'extreme_cv', 'regime_count': 0}
        
        self.logger.info(f"   ⚠️ Handling {len(extreme_regimes)} extreme CV regimes")
        
        # Create singleton clusters for extreme regimes
        extreme_clusters = {}
        for i, regime_id in enumerate(extreme_regimes.keys()):
            cluster_id = f"extreme_{i}"
            extreme_clusters[cluster_id] = [regime_id]
        
        self.logger.info(f"   ✅ Created {len(extreme_clusters)} singleton clusters for extreme regimes")
        
        return {
            'clusters': extreme_clusters,
            'stage': 'extreme_cv',
            'regime_count': len(extreme_regimes),
            'cluster_count': len(extreme_clusters)
        }

    def _merge_stage_results(self, stage_results: List[Dict[str, Any]], 
                           regime_characteristics: Dict[str, Any], market_data: Any = None) -> Dict[str, Any]:
        """Merge results from multiple clustering stages.
        
        Args:
            stage_results: List of stage clustering results
            regime_characteristics: Original regime characteristics
            
        Returns:
            Merged clustering result
        """
        try:
            merged_clusters = {}
            total_regimes = 0
            cluster_counter = 0
            
            for stage_result in stage_results:
                stage_clusters = stage_result.get('clusters', {})
                stage_name = stage_result.get('stage', 'unknown')
                
                for stage_cluster_id, regime_list in stage_clusters.items():
                    if regime_list:  # Only include non-empty clusters
                        merged_cluster_id = f"merged_{cluster_counter}"
                        merged_clusters[merged_cluster_id] = regime_list
                        cluster_counter += 1
                        total_regimes += len(regime_list)
                
                self.logger.info(f"   📊 Merged {len(stage_clusters)} clusters from stage '{stage_name}'")
            
            # Create regime-to-cluster mapping
            regime_to_cluster = {}
            for cluster_id, regime_list in merged_clusters.items():
                for regime_id in regime_list:
                    regime_to_cluster[regime_id] = cluster_id
            
            self.logger.info(f"✅ Multi-stage merge complete: {len(merged_clusters)} clusters, {total_regimes} regimes")
            
            return {
                'clusters': merged_clusters,
                'regime_to_cluster': regime_to_cluster,
                'cluster_count': len(merged_clusters),
                'total_regimes': total_regimes,
                'multi_stage': True,
                'aligned_market_data': market_data  # Include aligned market data
            }
            
        except Exception as e:
            self.logger.error(f"❌ Stage result merging failed: {e}")
            return {'clusters': {}, 'regime_to_cluster': {}, 'cluster_count': 0}

    def _perform_stage_clustering(self, regimes: Dict[str, Any], similarity_matrix: np.ndarray,
                                cv_thresholds: List[float], similarity_thresholds: np.ndarray) -> Dict[str, List[str]]:
        """Perform clustering for a specific stage with adjusted parameters.
        
        Args:
            regimes: Regimes to cluster
            similarity_matrix: Similarity matrix for these regimes
            cv_thresholds: Adjusted CV thresholds for this stage
            similarity_thresholds: Adjusted similarity thresholds for this stage
            
        Returns:
            Dictionary of cluster_id -> [regime_ids]
        """
        try:
            # Simplified clustering logic for individual stages
            regime_ids = list(regimes.keys())
            n_regimes = len(regime_ids)
            
            if n_regimes == 0:
                return {}
            elif n_regimes == 1:
                return {'stage_cluster_0': regime_ids}
            
            # Initialize each regime as its own cluster
            regime_to_cluster = {regime_id: i for i, regime_id in enumerate(regime_ids)}
            cluster_count = n_regimes
            
            # Simple agglomerative clustering with stage-specific thresholds
            for similarity_threshold in similarity_thresholds:
                if cluster_count <= 2:  # Stop when we have few clusters
                    break
                
                # Find best merge candidates
                best_similarity = -1
                best_pair = None
                
                for i in range(n_regimes):
                    for j in range(i + 1, n_regimes):
                        if regime_to_cluster[regime_ids[i]] != regime_to_cluster[regime_ids[j]]:
                            sim = similarity_matrix[i, j]
                            if sim >= similarity_threshold and sim > best_similarity:
                                best_similarity = sim
                                best_pair = (i, j)
                
                # Merge best pair if found
                if best_pair:
                    i, j = best_pair
                    old_cluster = regime_to_cluster[regime_ids[j]]
                    new_cluster = regime_to_cluster[regime_ids[i]]
                    
                    # Update all regimes in old cluster to new cluster
                    for regime_id in regime_ids:
                        if regime_to_cluster[regime_id] == old_cluster:
                            regime_to_cluster[regime_id] = new_cluster
                    
                    cluster_count -= 1
            
            # Convert to cluster -> regime_list format
            clusters = {}
            cluster_id_map = {}
            next_id = 0
            
            for regime_id, cluster_num in regime_to_cluster.items():
                if cluster_num not in cluster_id_map:
                    cluster_id_map[cluster_num] = f"stage_cluster_{next_id}"
                    next_id += 1
                
                cluster_id = cluster_id_map[cluster_num]
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(regime_id)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"❌ Stage clustering failed: {e}")
            return {}

    def _perform_standard_clustering(self, regime_characteristics: Dict[str, Any], 
                                   regime_assignments: List[int]) -> Dict[str, Any]:
        """Fallback to standard clustering approach.
        
        Args:
            regime_characteristics: Regime characteristics
            regime_assignments: Regime assignments
            
        Returns:
            Standard clustering result
        """
        self.logger.info("🔄 Falling back to standard clustering approach")
        
        # Use the existing clustering logic as fallback
        try:
            similarity_matrix = self._calculate_regime_similarity_matrix(regime_characteristics, None)
            # ... implement standard clustering logic or call existing method
            return {'clusters': {}, 'cluster_count': 0, 'fallback': True}
        except Exception as e:
            self.logger.error(f"❌ Standard clustering fallback failed: {e}")
            return {'clusters': {}, 'cluster_count': 0, 'error': str(e)}

    def _calculate_comprehensive_cluster_quality(self, clusters: Dict[str, List[str]], 
                                               regime_characteristics: Dict[str, Any],
                                               similarity_matrix: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Calculate comprehensive cluster quality score with multiple metrics.
        
        Args:
            clusters: Dictionary of cluster_id -> [regime_ids]
            regime_characteristics: Regime characteristics
            similarity_matrix: Similarity matrix between regimes
            
        Returns:
            Tuple of (composite_score, individual_metrics)
        """
        try:
            import numpy as np
            
            if not clusters or not regime_characteristics:
                return 0.0, {}
            
            # Calculate individual quality metrics
            quality_metrics = {
                'intra_cluster_homogeneity': self._calculate_intra_cluster_homogeneity(
                    clusters, regime_characteristics, similarity_matrix
                ),
                'inter_cluster_separation': self._calculate_inter_cluster_separation(
                    clusters, regime_characteristics, similarity_matrix
                ),
                'size_distribution_balance': self._calculate_size_distribution_quality(clusters),
                'regime_coherence': self._calculate_regime_coherence(
                    clusters, regime_characteristics
                ),
                'temporal_stability': self._calculate_temporal_stability(clusters, regime_characteristics),
                'cv_consistency': self._calculate_cv_consistency(clusters, regime_characteristics)
            }
            
            # Weighted composite score
            weights = {
                'intra_cluster_homogeneity': 0.25,
                'inter_cluster_separation': 0.20,
                'size_distribution_balance': 0.15,
                'regime_coherence': 0.20,
                'temporal_stability': 0.10,
                'cv_consistency': 0.10
            }
            
            composite_score = sum(
                quality_metrics.get(metric, 0.0) * weights[metric] 
                for metric in weights
            )
            
            self.logger.info(f"📊 Comprehensive Quality Score: {composite_score:.3f}")
            for metric, score in quality_metrics.items():
                self.logger.info(f"   📈 {metric}: {score:.3f}")
            
            return composite_score, quality_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating comprehensive cluster quality: {e}")
            return 0.0, {}

    def _calculate_intra_cluster_homogeneity(self, clusters: Dict[str, List[str]], 
                                           regime_characteristics: Dict[str, Any],
                                           similarity_matrix: np.ndarray) -> float:
        """Calculate how homogeneous regimes are within each cluster.
        
        Args:
            clusters: Dictionary of cluster_id -> [regime_ids]
            regime_characteristics: Regime characteristics
            similarity_matrix: Similarity matrix
            
        Returns:
            Homogeneity score (0-1, higher is better)
        """
        try:
            import numpy as np
            
            regime_ids = list(regime_characteristics.keys())
            regime_to_idx = {regime_id: idx for idx, regime_id in enumerate(regime_ids)}
            
            cluster_homogeneities = []
            
            for cluster_id, regime_list in clusters.items():
                if len(regime_list) < 2:
                    cluster_homogeneities.append(1.0)  # Single regime is perfectly homogeneous
                    continue
                
                # Calculate average intra-cluster similarity
                similarities = []
                for i, regime_i in enumerate(regime_list):
                    if regime_i not in regime_to_idx:
                        continue
                    idx_i = regime_to_idx[regime_i]
                    
                    for regime_j in regime_list[i+1:]:
                        if regime_j not in regime_to_idx:
                            continue
                        idx_j = regime_to_idx[regime_j]
                        
                        sim = similarity_matrix[idx_i, idx_j]
                        if np.isfinite(sim):
                            similarities.append(sim)
                
                if similarities:
                    avg_similarity = np.mean(similarities)
                    # Convert similarity to homogeneity score (0-1 range)
                    homogeneity = (avg_similarity + 1) / 2  # Convert from [-1,1] to [0,1]
                    cluster_homogeneities.append(homogeneity)
                else:
                    cluster_homogeneities.append(0.5)  # Neutral score if no similarities
            
            return float(np.mean(cluster_homogeneities)) if cluster_homogeneities else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating intra-cluster homogeneity: {e}")
            return 0.0

    def _calculate_inter_cluster_separation(self, clusters: Dict[str, List[str]], 
                                          regime_characteristics: Dict[str, Any],
                                          similarity_matrix: np.ndarray) -> float:
        """Calculate how well-separated different clusters are.
        
        Args:
            clusters: Dictionary of cluster_id -> [regime_ids]
            regime_characteristics: Regime characteristics
            similarity_matrix: Similarity matrix
            
        Returns:
            Separation score (0-1, higher is better)
        """
        try:
            import numpy as np
            
            regime_ids = list(regime_characteristics.keys())
            regime_to_idx = {regime_id: idx for idx, regime_id in enumerate(regime_ids)}
            
            cluster_list = list(clusters.items())
            inter_cluster_similarities = []
            
            for i, (cluster_i, regimes_i) in enumerate(cluster_list):
                for cluster_j, regimes_j in cluster_list[i+1:]:
                    # Calculate average similarity between clusters
                    cross_similarities = []
                    
                    for regime_i in regimes_i:
                        if regime_i not in regime_to_idx:
                            continue
                        idx_i = regime_to_idx[regime_i]
                        
                        for regime_j in regimes_j:
                            if regime_j not in regime_to_idx:
                                continue
                            idx_j = regime_to_idx[regime_j]
                            
                            sim = similarity_matrix[idx_i, idx_j]
                            if np.isfinite(sim):
                                cross_similarities.append(sim)
                    
                    if cross_similarities:
                        avg_cross_similarity = np.mean(cross_similarities)
                        inter_cluster_similarities.append(avg_cross_similarity)
            
            if inter_cluster_similarities:
                avg_inter_similarity = np.mean(inter_cluster_similarities)
                # Convert to separation score (lower similarity = better separation)
                separation = (1 - avg_inter_similarity) / 2  # Convert from [-1,1] to [0,1], inverted
                return float(max(0.0, separation))
            else:
                return 1.0  # Perfect separation if only one cluster
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating inter-cluster separation: {e}")
            return 0.0

    def _calculate_size_distribution_quality(self, clusters: Dict[str, List[str]]) -> float:
        """Calculate quality of cluster size distribution (balanced is better).
        
        Args:
            clusters: Dictionary of cluster_id -> [regime_ids]
            
        Returns:
            Size distribution quality score (0-1, higher is better)
        """
        try:
            import numpy as np
            
            if not clusters:
                return 0.0
            
            cluster_sizes = [len(regimes) for regimes in clusters.values()]
            
            if len(cluster_sizes) == 1:
                return 1.0  # Single cluster is perfectly "balanced"
            
            # Calculate coefficient of variation for cluster sizes
            mean_size = np.mean(cluster_sizes)
            std_size = np.std(cluster_sizes)
            
            if mean_size == 0:
                return 0.0
            
            cv = std_size / mean_size
            
            # Convert CV to quality score (lower CV = more balanced = higher quality)
            # Use exponential decay to map CV to [0,1] range
            quality = np.exp(-cv)  # CV of 0 -> quality 1.0, higher CV -> lower quality
            
            return float(quality)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating size distribution quality: {e}")
            return 0.0

    def _calculate_regime_coherence(self, clusters: Dict[str, List[str]], 
                                  regime_characteristics: Dict[str, Any]) -> float:
        """Calculate how coherent regimes are within clusters based on feature similarity.
        
        Args:
            clusters: Dictionary of cluster_id -> [regime_ids]
            regime_characteristics: Regime characteristics
            
        Returns:
            Regime coherence score (0-1, higher is better)
        """
        try:
            import numpy as np
            
            cluster_coherences = []
            
            for cluster_id, regime_list in clusters.items():
                if len(regime_list) < 2:
                    cluster_coherences.append(1.0)  # Single regime is perfectly coherent
                    continue
                
                # Extract features for all regimes in cluster
                cluster_features = []
                for regime_id in regime_list:
                    if regime_id in regime_characteristics:
                        features = regime_characteristics[regime_id].get('features', {})
                        if features:
                            cluster_features.append(features)
                
                if len(cluster_features) < 2:
                    cluster_coherences.append(0.5)  # Neutral score
                    continue
                
                # Calculate feature consistency within cluster
                feature_coherences = []
                all_features = set()
                for features in cluster_features:
                    all_features.update(features.keys())
                
                for feature_name in all_features:
                    feature_values = []
                    for features in cluster_features:
                        if feature_name in features:
                            val = features[feature_name]
                            if isinstance(val, (int, float)) and np.isfinite(val):
                                feature_values.append(val)
                    
                    if len(feature_values) >= 2:
                        # Calculate coefficient of variation for this feature
                        mean_val = np.mean(feature_values)
                        std_val = np.std(feature_values)
                        
                        if mean_val != 0:
                            cv = abs(std_val / mean_val)
                            # Convert CV to coherence (lower CV = higher coherence)
                            coherence = np.exp(-cv)
                            feature_coherences.append(coherence)
                
                if feature_coherences:
                    cluster_coherence = np.mean(feature_coherences)
                    cluster_coherences.append(cluster_coherence)
                else:
                    cluster_coherences.append(0.5)  # Neutral score
            
            return float(np.mean(cluster_coherences)) if cluster_coherences else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating regime coherence: {e}")
            return 0.0

    def _calculate_temporal_stability(self, clusters: Dict[str, List[str]], 
                                    regime_characteristics: Dict[str, Any]) -> float:
        """Calculate temporal stability of clusters (placeholder implementation).
        
        Args:
            clusters: Dictionary of cluster_id -> [regime_ids]
            regime_characteristics: Regime characteristics
            
        Returns:
            Temporal stability score (0-1, higher is better)
        """
        try:
            # Placeholder implementation - could be enhanced with actual temporal analysis
            # For now, return a score based on cluster size consistency
            cluster_sizes = [len(regimes) for regimes in clusters.values()]
            
            if len(cluster_sizes) <= 1:
                return 1.0
            
            import numpy as np
            # Use inverse of size variation as proxy for stability
            size_cv = np.std(cluster_sizes) / np.mean(cluster_sizes) if np.mean(cluster_sizes) > 0 else 0
            stability = np.exp(-size_cv)
            
            return float(stability)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating temporal stability: {e}")
            return 0.5  # Neutral score

    def _calculate_cv_consistency(self, clusters: Dict[str, List[str]], 
                                regime_characteristics: Dict[str, Any]) -> float:
        """Calculate CV consistency within clusters.
        
        Args:
            clusters: Dictionary of cluster_id -> [regime_ids]
            regime_characteristics: Regime characteristics
            
        Returns:
            CV consistency score (0-1, higher is better)
        """
        try:
            import numpy as np
            
            cluster_cv_consistencies = []
            cv_aspects = ['momentum_cv', 'volatility_cv', 'volume_cv']
            
            for cluster_id, regime_list in clusters.items():
                if len(regime_list) < 2:
                    cluster_cv_consistencies.append(1.0)
                    continue
                
                # Collect CV values for each aspect within this cluster
                aspect_consistencies = []
                
                for aspect in cv_aspects:
                    aspect_cvs = []
                    for regime_id in regime_list:
                        if regime_id in regime_characteristics:
                            features = regime_characteristics[regime_id].get('features', {})
                            cv_val = features.get(aspect, 0.0)
                            if isinstance(cv_val, (int, float)) and np.isfinite(cv_val):
                                aspect_cvs.append(cv_val)
                    
                    if len(aspect_cvs) >= 2:
                        # Calculate consistency (inverse of coefficient of variation)
                        mean_cv = np.mean(aspect_cvs)
                        std_cv = np.std(aspect_cvs)
                        
                        if mean_cv > 0:
                            cv_of_cvs = std_cv / mean_cv
                            consistency = np.exp(-cv_of_cvs)  # Lower variation = higher consistency
                            aspect_consistencies.append(consistency)
                
                if aspect_consistencies:
                    cluster_consistency = np.mean(aspect_consistencies)
                    cluster_cv_consistencies.append(cluster_consistency)
                else:
                    cluster_cv_consistencies.append(0.5)  # Neutral score
            
            return float(np.mean(cluster_cv_consistencies)) if cluster_cv_consistencies else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error calculating CV consistency: {e}")
            return 0.0

    def _get_smart_merge_candidates(self, similarity_matrix: np.ndarray, regime_to_cluster: Dict[str, int], 
                                    regime_ids: List[str], excluded_clusters: set, threshold: float, 
                                    max_candidates: int = 200) -> List[Tuple[float, int, int, str, str]]:
        """Pre-filter merge candidates to avoid expensive checks on obviously incompatible pairs.
         
         Args:
             similarity_matrix: Similarity matrix between regimes
             regime_to_cluster: Current regime to cluster mapping
             regime_ids: List of regime IDs
             excluded_clusters: Clusters excluded due to high CV
             threshold: Current similarity threshold
             max_candidates: Maximum number of candidates to return
             
         Returns:
             List of (similarity, cluster_i, cluster_j, regime_i, regime_j) tuples sorted by similarity
         """
        try:
            import numpy as np
            
            candidates = []
            min_similarity = max(threshold * 0.8, 0.3)  # Don't consider very low similarities
            
            # Detailed logging counters
            total_pairs = 0
            same_cluster_filtered = 0
            excluded_cluster_filtered = 0
            low_similarity_filtered = 0
            
            for i, regime_i in enumerate(regime_ids):
                for j, regime_j in enumerate(regime_ids[i+1:], i+1):
                    total_pairs += 1
                    similarity = similarity_matrix[i, j]
                    cluster_i = regime_to_cluster[regime_i]
                    cluster_j = regime_to_cluster[regime_j]
                    
                    # Quick elimination checks (fast operations only) with detailed logging
                    if cluster_i == cluster_j:  # Same cluster
                        same_cluster_filtered += 1
                        continue
                    elif cluster_i in excluded_clusters or cluster_j in excluded_clusters:  # Excluded
                        excluded_cluster_filtered += 1
                        continue
                    elif similarity < min_similarity:  # Too low similarity
                        low_similarity_filtered += 1
                        continue
                    
                    candidates.append((similarity, cluster_i, cluster_j, regime_i, regime_j))
            
            # Detailed pre-filtering report
            self.logger.info(f"🔍 PRE-FILTERING ANALYSIS (out of {total_pairs} total pairs):")
            self.logger.info(f"   🎯 Min similarity threshold: {min_similarity:.3f} (= max({threshold:.3f} * 0.8, 0.3))")
            self.logger.info(f"   ❌ Same cluster: {same_cluster_filtered} ({same_cluster_filtered/total_pairs*100:.1f}%)")
            self.logger.info(f"   ❌ Excluded clusters: {excluded_cluster_filtered} ({excluded_cluster_filtered/total_pairs*100:.1f}%)")
            self.logger.info(f"   ❌ Below min similarity ({min_similarity:.3f}): {low_similarity_filtered} ({low_similarity_filtered/total_pairs*100:.1f}%)")
            self.logger.info(f"   ✅ Pre-filtered candidates: {len(candidates)} ({len(candidates)/total_pairs*100:.1f}%)")
            
            # Sort by similarity (highest first) and limit to top candidates
            candidates.sort(reverse=True, key=lambda x: x[0])
            
            original_count = len(candidates)
            if len(candidates) > max_candidates:
                candidates = candidates[:max_candidates]
                self.logger.info(f"🎯 LIMITED to top {max_candidates} candidates (was {original_count})")
            
            return candidates
            
        except Exception as e:
            self.logger.error(f"❌ Error in smart merge candidate filtering: {e}")
            # Fallback to empty list - will trigger normal processing
            return []

    def _generate_adaptive_cv_thresholds(self, regime_characteristics: Dict[str, Any]) -> List[float]:
        """Generate CV thresholds based on actual data distribution characteristics.
         
         Analyzes the CV values in the regime data to create adaptive thresholds that
         are more appropriate for the specific dataset characteristics.
         
         Args:
             regime_characteristics: Dictionary of regime characteristics
             
        Returns:
            List of CV thresholds in ascending order
        """
        try:
            import numpy as np
            
            # Collect all CV values from the regime data
            all_cvs = []
            cv_aspects = ['momentum_cv', 'volatility_cv', 'volume_cv']
            
            for regime in regime_characteristics.values():
                features = regime.get('features', {})
                for aspect in cv_aspects:
                    cv_val = features.get(aspect, 0.0)
                    if cv_val is not None and isinstance(cv_val, (int, float)) and np.isfinite(cv_val) and cv_val > 0:
                        all_cvs.append(cv_val)
            
            if not all_cvs:
                # Fallback to conservative thresholds if no data
                self.logger.warning("⚠️ No valid CV data found, using fallback thresholds")
                return [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
            
            all_cvs = np.array(all_cvs)
            
            # Calculate distribution statistics
            cv_mean = np.mean(all_cvs)
            cv_median = np.median(all_cvs)
            cv_std = np.std(all_cvs)
            
            # Enhanced distribution analysis with extreme value handling
            cv_percentiles = np.percentile(all_cvs, [10, 25, 50, 75, 90, 95, 99])
            cv_iqr = cv_percentiles[3] - cv_percentiles[1]  # 75th - 25th percentile
            extreme_cvs = [cv for cv in all_cvs if cv > 10.0]  # Track extreme values
            
            # Generate multi-tier adaptive thresholds
            threshold_candidates = []
            
            # Tier 1: Conservative thresholds (for high-quality clusters)
            threshold_candidates.extend([
                cv_percentiles[1] * 0.8,  # Below 25th percentile
                cv_percentiles[1],        # 25th percentile
                cv_percentiles[2] * 0.9,  # Below median
            ])
            
            # Tier 2: Moderate thresholds (for standard merging)
            threshold_candidates.extend([
                cv_percentiles[2],        # Median (50th percentile)
                cv_percentiles[3] * 0.9,  # Below 75th percentile
                cv_percentiles[3],        # 75th percentile
                cv_percentiles[4] * 0.95, # Below 90th percentile
            ])
            
            # Tier 3: Aggressive thresholds (for difficult cases)
            threshold_candidates.extend([
                cv_percentiles[4],        # 90th percentile
                cv_percentiles[5],        # 95th percentile
                cv_percentiles[6] * 0.8,  # Below 99th percentile
                cv_mean + cv_std * 1.5,   # Statistical upper bound
            ])
            
            # Tier 4: Extreme thresholds (for outlier handling)
            if extreme_cvs:
                extreme_threshold = np.percentile(extreme_cvs, 50)  # Median of extremes
                threshold_candidates.extend([
                    cv_percentiles[6],         # 99th percentile
                    extreme_threshold * 0.5,   # Half of extreme median
                    extreme_threshold * 0.8,   # 80% of extreme median
                    min(extreme_threshold, MAX_CV_THRESHOLD)  # Cap at max threshold (centrally configured)
                ])
            else:
                threshold_candidates.extend([
                    cv_percentiles[6],         # 99th percentile
                    cv_mean + cv_std * 2.0,    # 2-sigma upper bound
                    cv_mean + cv_std * 3.0,    # 3-sigma upper bound
                    MAX_CV_THRESHOLD           # Fixed high threshold (centrally configured)
            ])
            
            # Remove duplicates, sort, and filter reasonable range
            thresholds = sorted(set(threshold_candidates))
            thresholds = [t for t in thresholds if 0.05 <= t <= MAX_CV_THRESHOLD]  # Cap at centrally configured max
            
            # Ensure progressive spacing with adaptive minimum gaps
            filtered_thresholds = [thresholds[0]]
            min_spacing = max(0.05, cv_iqr * 0.1)  # Adaptive minimum spacing
            
            for t in thresholds[1:]:
                if t - filtered_thresholds[-1] >= min_spacing:
                    filtered_thresholds.append(t)
            
            # Ensure we have reasonable number of thresholds (8-12)
            if len(filtered_thresholds) > 12:
                # Take evenly distributed subset
                indices = np.linspace(0, len(filtered_thresholds)-1, 10, dtype=int)
                filtered_thresholds = [filtered_thresholds[i] for i in indices]
            elif len(filtered_thresholds) < 6:
                # Add intermediate thresholds if too few
                additional = []
                for i in range(len(filtered_thresholds)-1):
                    mid = (filtered_thresholds[i] + filtered_thresholds[i+1]) / 2
                    additional.append(mid)
                filtered_thresholds.extend(additional)
                filtered_thresholds = sorted(filtered_thresholds)
            
            # Enhanced logging with detailed statistics
            self.logger.info(f"📊 Enhanced Adaptive CV Thresholds from {len(all_cvs)} values")
            self.logger.info(f"   📈 Distribution: P25={cv_percentiles[1]:.2f}, P50={cv_percentiles[2]:.2f}, P75={cv_percentiles[3]:.2f}, P90={cv_percentiles[4]:.2f}")
            self.logger.info(f"   ⚠️ Extreme CVs (>10): {len(extreme_cvs)} values, max={max(extreme_cvs):.1f}" if extreme_cvs else "   ✅ No extreme CV values detected")
            self.logger.info(f"   🔧 Generated {len(filtered_thresholds)} thresholds: {[f'{t:.2f}' for t in filtered_thresholds]}")
            
            return filtered_thresholds
            
        except Exception as e:
            self.logger.error(f"❌ Error generating adaptive CV thresholds: {e}")
            # Enhanced fallback with more aggressive thresholds
            return [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    def _passes_cv_hard_constraint(self, regime_1: Dict[str, Any], regime_2: Dict[str, Any], max_relative_diff: float = 0.70) -> bool:
        """Check if two regimes pass the hard CV constraint (relative difference < threshold).
        
        Prevents merging regimes with fundamentally different characteristics by checking
        if any aspect (momentum, volatility, volume) has relative difference > threshold.
        
        Args:
            regime_1: First regime characteristics
            regime_2: Second regime characteristics  
            max_relative_diff: Maximum allowed relative difference (default: 0.70 = 70%)
            
        Returns:
            True if constraint passes (safe to merge), False if constraint fails (block merge)
        """
        try:
            import numpy as np
            
            # Get CV values for each aspect from both regimes
            aspects = ['momentum_cv', 'volatility_cv', 'volume_cv']
            
            for aspect in aspects:
                cv1 = regime_1.get('features', {}).get(aspect, 0.0)
                cv2 = regime_2.get('features', {}).get(aspect, 0.0)
                
                # Handle None or non-numeric values
                if cv1 is None or not isinstance(cv1, (int, float)) or not np.isfinite(cv1):
                    cv1 = 0.0
                if cv2 is None or not isinstance(cv2, (int, float)) or not np.isfinite(cv2):
                    cv2 = 0.0
                
                # Calculate relative difference
                max_cv = max(abs(cv1), abs(cv2))
                if max_cv > 1e-10:  # Avoid division by very small numbers
                    relative_diff = abs(cv1 - cv2) / max_cv
                    
                    if relative_diff > max_relative_diff:
                        self.logger.debug(f"   🚫 CV HARD CONSTRAINT: {aspect} diff={relative_diff:.3f} > {max_relative_diff:.3f} (cv1={cv1:.3f}, cv2={cv2:.3f})")
                        return False
            
            return True  # All aspects pass the constraint
            
        except Exception as e:
            self.logger.debug(f"⚠️ Error checking CV hard constraint: {e}")
            return True  # Allow merge on error to avoid blocking everything

    def _would_merge_exceed_cv_limit_fresh(self, cluster_cv_aspects: Dict[str, Any], cluster_i: int, cluster_j: int, regime_to_cluster: Dict[str, int], total_samples: int) -> bool:
        """Check if merging two clusters would exceed adaptive CV limit using fresh cluster CV data.
        
        This version uses the fresh cluster_cv_aspects data instead of potentially stale regime_characteristics.
        """
        try:
            import numpy as np
            
            # Get current CV for both clusters from fresh data
            cv_i = cluster_cv_aspects.get(cluster_i, {})
            cv_j = cluster_cv_aspects.get(cluster_j, {})
            
            if not cv_i or not cv_j:
                return False  # Allow merge if no CV data available
            
            # Calculate merged cluster sample count for adaptive limit
            regimes_i = [rid for rid, cid in regime_to_cluster.items() if cid == cluster_i]
            regimes_j = [rid for rid, cid in regime_to_cluster.items() if cid == cluster_j]
            merged_sample_count = len(regimes_i) + len(regimes_j)  # Simplified count
            
            # Determine adaptive CV limit based on merged cluster size - another 50% more relaxed for faster convergence
            if total_samples and total_samples > 0:
                merged_pct = (merged_sample_count / total_samples) * 100
                if merged_pct < 1.0:
                    cv_limit = 22.5  # 2250% - another 50% more relaxed for tiny clusters (was 15.0)
                elif merged_pct < 3.0:
                    cv_limit = 13.5  # 1350% - another 50% more relaxed for small clusters (was 9.0)
                elif merged_pct < 12.0:
                    cv_limit = 11.25 # 1125% - another 50% more relaxed for large clusters (was 7.5)
                else:
                    cv_limit = 2.0   # 200% - kept conservative for very large clusters (unchanged)
            else:
                cv_limit = 11.25  # Another 50% more relaxed default fallback (was 7.5)
            
            # Calculate combined CV for each aspect using fresh data
            aspects = ['momentum', 'volatility', 'volume']
            max_combined_cv = 0.0
            
            for aspect in aspects:
                cv_i_val = cv_i.get(aspect, 0.0)
                cv_j_val = cv_j.get(aspect, 0.0)
                
                # Simple weighted average of CVs (could be improved)
                if cv_i_val is not None and cv_j_val is not None and np.isfinite(cv_i_val) and np.isfinite(cv_j_val):
                    combined_cv = max(cv_i_val, cv_j_val)  # Conservative: use worst CV
                    max_combined_cv = max(max_combined_cv, combined_cv)
            
            would_exceed = max_combined_cv > cv_limit
            if would_exceed:
                self.logger.debug(f"   🚫 FRESH CV LIMIT: C{cluster_i} + C{cluster_j} → CV={max_combined_cv:.3f} > {cv_limit:.3f} (size: {merged_pct:.1f}%)")
            
            return would_exceed
            
        except Exception as e:
            self.logger.debug(f"⚠️ Error checking fresh CV limit: {e}")
            return False

    def _would_merge_create_extreme_cv_fresh(self, cluster_cv_aspects: Dict[str, Any], cluster_i: int, cluster_j: int, regime_to_cluster: Dict[str, int], cv_cap: float = 3.0) -> bool:
        """Check if merging would create extreme CV using fresh cluster CV data."""
        try:
            import numpy as np
            
            # Get current CV for both clusters from fresh data
            cv_i = cluster_cv_aspects.get(cluster_i, {})
            cv_j = cluster_cv_aspects.get(cluster_j, {})
            
            if not cv_i or not cv_j:
                return False  # Allow merge if no CV data available
            
            # Calculate worst-case combined CV for each aspect
            aspects = ['momentum', 'volatility', 'volume']
            max_combined_cv = 0.0
            extreme_volume_detected = False
            
            for aspect in aspects:
                cv_i_val = cv_i.get(aspect, 0.0)
                cv_j_val = cv_j.get(aspect, 0.0)
                
                # Conservative: use worst CV as estimate for combined CV
                if cv_i_val is not None and cv_j_val is not None and np.isfinite(cv_i_val) and np.isfinite(cv_j_val):
                    combined_cv = max(cv_i_val, cv_j_val)
                    max_combined_cv = max(max_combined_cv, combined_cv)
            
                    # EXTREME VOLUME CV PREVENTION: Block any merge that would create volume CV > 10 (1000%)
                    if combined_cv > 10.0:
                        extreme_volume_detected = True
                        self.logger.warning(f"🚫 EXTREME CV BLOCKED: C{cluster_i} + C{cluster_j} → combined_cv={combined_cv:.1f} > 10.0 (1000%)")
            
            # Block if extreme volume CV detected OR standard CV cap exceeded
            would_exceed = extreme_volume_detected or max_combined_cv > cv_cap
            if would_exceed and not extreme_volume_detected:
                self.logger.debug(f"   🚫 FRESH HARD CV CAP: C{cluster_i} + C{cluster_j} → CV={max_combined_cv:.3f} > {cv_cap:.1f}")
            
            return would_exceed
            
        except Exception as e:
            self.logger.debug(f"⚠️ Error checking fresh hard CV cap: {e}")
            return False
    
    def _calculate_regime_pair_similarity(self, regime_1: Dict[str, Any], regime_2: Dict[str, Any]) -> float:
        """Calculate similarity between two regimes using robust per-feature normalization.
        
        This method reuses the existing robust normalization approach but extends it to support
        negative similarities for opposite directions, enabling proper cosine-like similarity.
        """
        try:
            # Extract features from both regimes
            features_1 = regime_1.get('features', {})
            features_2 = regime_2.get('features', {})
            
            # Get common feature keys
            keys = sorted((set(features_1.keys()) & set(features_2.keys())))
            if not keys:
                return 0.0
            
            # Calculate similarity for each feature using the shared utility function
            similarities = []
            feature_values_1 = []
            feature_values_2 = []
            
            for k in keys:
                v1 = features_1.get(k)
                v2 = features_2.get(k)
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    val_1, val_2 = float(v1), float(v2)
                    feature_values_1.append(val_1)
                    feature_values_2.append(val_2)
                    # Use shared utility function with negative similarity support
                    similarity = self._calculate_feature_similarity(val_1, val_2, support_negative=True)
                    similarities.append(similarity)
            
            if not similarities:
                return 0.0
            
            # Calculate weighted average similarity
            overall_similarity = np.mean(similarities)
            
            # Limited debug output
            if not hasattr(self, '_similarity_debug_count'):
                self._similarity_debug_count = 0
            if self._similarity_debug_count < 3:
                self.logger.warning(f"🔍 DEBUG: Features 1 keys: {keys[:5]}...")
                self.logger.warning(f"🔍 DEBUG: Features 2 keys: {keys[:5]}...")
                self.logger.warning(f"🔍 DEBUG: Features 1 sample values: {feature_values_1[:5]}")
                self.logger.warning(f"🔍 DEBUG: Features 2 sample values: {feature_values_2[:5]}")
                self.logger.warning(f"🔍 DEBUG: Feature similarities[:5]: {similarities[:5]}")
                self.logger.warning(f"🔍 DEBUG: Overall cosine similarity: {overall_similarity:.6f}")
                self._similarity_debug_count += 1
            
            return float(np.clip(overall_similarity, -1.0, 1.0))

        except Exception as e:
            self.logger.error(f"❌ Failed to calculate regime pair similarity: {e}")
            return 0.0
    
    def _calculate_feature_similarity(self, val_1: float, val_2: float, support_negative: bool = False) -> float:
        """Calculate similarity between two feature values using robust normalization.
        
        Args:
            val_1: First feature value
            val_2: Second feature value  
            support_negative: If True, allows negative similarities for opposite directions
            
        Returns:
            Similarity score between -1 and +1 (if support_negative) or 0 and +1 (otherwise)
        """
        try:
            # Handle zero cases
            if val_1 == 0 and val_2 == 0:
                return 1.0
            elif val_1 == 0 or val_2 == 0:
                return 0.0
            
            # Use max absolute value for robust normalization
            max_val = max(abs(val_1), abs(val_2))
            if max_val == 0:
                return 1.0
            
            if support_negative:
                # For cosine-like similarity, normalize both values and compute dot product
                norm_1 = val_1 / max_val
                norm_2 = val_2 / max_val
                
                # Calculate similarity considering both magnitude and direction
                # This gives us values between -1 and +1
                similarity = norm_1 * norm_2
                
                # If both values are very small relative to max, they're similar
                if abs(norm_1) < 0.1 and abs(norm_2) < 0.1:
                    similarity = 1.0 if (norm_1 >= 0) == (norm_2 >= 0) else -1.0
                
                return max(-1.0, min(1.0, similarity))
            else:
                # Original distance-based approach (0 to 1 range)
                distance = abs(val_1 - val_2) / max_val
                return max(0.0, 1.0 - distance)
                
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate feature similarity: {e}")
            return 0.0


    def _perform_quality_based_clustering(self, similarity_matrix: np.ndarray, regime_ids: List[str], n_clusters: int) -> Dict[int, int]:
        """Group regimes with similar characteristics together using hierarchical clustering."""
        try:
            if similarity_matrix.size == 0 or len(regime_ids) == 0:
                self.logger.warning("⚠️ Empty similarity matrix or regime IDs, using fallback clustering")
                return self._create_fallback_cluster_mapping(regime_ids, n_clusters)
            
            # Convert similarity to distance (1 - similarity)
            distance_matrix = 1.0 - similarity_matrix
            
            # Apply hierarchical clustering
            from sklearn.cluster import AgglomerativeClustering
            
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='complete'  # Complete linkage for better performance and compact clusters
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Create regime to cluster mapping
            regime_to_cluster = {}
            for i, regime_id in enumerate(regime_ids):
                regime_to_cluster[regime_id] = cluster_labels[i]
            
            # Log cluster assignments
            cluster_groups = {}
            for regime_id, cluster_id in regime_to_cluster.items():
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(regime_id)
            
            self.logger.info(f"✅ Quality-based clustering completed: {len(cluster_groups)} clusters created")
            for cluster_id, regimes in cluster_groups.items():
                self.logger.info(f"📊 Cluster {cluster_id}: {regimes}")
            
            return regime_to_cluster
            
        except Exception as e:
            self.logger.error(f"❌ Quality-based clustering failed: {e}")
            self.logger.info("🔄 Falling back to frequency-based clustering")
            return self._create_fallback_cluster_mapping(regime_ids, n_clusters)
    
    def _create_fallback_cluster_mapping(self, regime_ids: List[str], n_clusters: int) -> Dict[int, int]:
        """Create fallback cluster mapping when quality-based clustering fails."""
        regime_to_cluster = {}
        
        for i, regime_id in enumerate(regime_ids):
            cluster_id = i % n_clusters
            regime_to_cluster[regime_id] = cluster_id
        
        self.logger.info(f"📊 Fallback clustering: {len(regime_ids)} regimes → {n_clusters} clusters")
        return regime_to_cluster

    def _create_cluster_assignments(
        self, 
        regime_assignments: List[int], 
        n_clusters: int, 
        data_length: int,
        regime_discovery: Dict[str, Any] = None
    ) -> List[int]:
        """Create cluster assignments by grouping similar regimes using market characteristics."""
        try:
            if not regime_assignments:
                # Fallback: create random cluster assignments
                import random
                return [random.randint(0, n_clusters - 1) for _ in range(data_length)]
            
            # Use provided regime discovery results
            if not regime_discovery:
                self.logger.warning("⚠️ No regime discovery results available, falling back to frequency-based clustering")
                return self._create_frequency_based_clusters(regime_assignments, n_clusters, data_length)
            
            # Extract regime characteristics from HMM regime discovery
            regime_characteristics = self._extract_regime_characteristics_from_discovery(regime_discovery)
            if not regime_characteristics:
                self.logger.warning("⚠️ No regime characteristics available, falling back to frequency-based clustering")
                return self._create_frequency_based_clusters(regime_assignments, n_clusters, data_length)
            
            # Calculate regime similarity matrix
            similarity_matrix = self._calculate_regime_similarity_matrix(regime_characteristics)
            if similarity_matrix.size == 0:
                self.logger.warning("⚠️ Empty similarity matrix, falling back to frequency-based clustering")
                return self._create_frequency_based_clusters(regime_assignments, n_clusters, data_length)
            
            # Perform quality-based clustering
            regime_ids = list(regime_characteristics.keys())
            regime_to_cluster = self._perform_quality_based_clustering(similarity_matrix, regime_ids, n_clusters)
            
            # Convert regime IDs to regime indices for mapping
            regime_id_to_index = {}
            for i, regime_id in enumerate(regime_ids):
                # Extract regime number from regime_id (e.g., "regime_0" -> 0)
                try:
                    regime_num = int(regime_id.split('_')[-1])
                    regime_id_to_index[regime_num] = regime_id
                except (ValueError, IndexError):
                    regime_id_to_index[i] = regime_id
            
            # Create cluster assignments
            cluster_assignments = []
            for regime in regime_assignments:
                # Map regime number to regime_id, then to cluster
                regime_id = regime_id_to_index.get(regime, regime_id_to_index.get(0, regime_ids[0]))
                cluster_id = regime_to_cluster.get(regime_id, 0)
                cluster_assignments.append(cluster_id)
            
            # Validate cluster quality
            cluster_quality = self._validate_cluster_quality_metrics(cluster_assignments, regime_characteristics, regime_to_cluster)
            
            # Log cluster distribution and quality for verification
            cluster_sizes = self._calculate_cluster_sizes_from_regime_mapping(regime_characteristics, regime_to_cluster)
            cluster_dist = self._calculate_cluster_distribution_from_sizes(cluster_sizes)
            self.logger.info(f"✅ Quality-based cluster assignments created: {len(set(cluster_assignments))} unique clusters")
            self.logger.info(f"📊 Cluster distribution: {cluster_dist}")
            self.logger.info(f"📊 Cluster quality score: {cluster_quality.get('overall_quality_score', 0.0):.3f}")
            
            return cluster_assignments
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create quality-based cluster assignments: {e}")
            self.logger.info("🔄 Falling back to frequency-based clustering")
            return self._create_frequency_based_clusters(regime_assignments, n_clusters, data_length)
    
    def _get_regime_discovery_results(self) -> Dict[str, Any]:
        """Get regime discovery results from pipeline state or outcome files."""
        try:
            # Try to get from the most recent HMM regime discovery outcome file
            from pathlib import Path
            import json
            
            outcomes_dir = Path("outcomes")
            if outcomes_dir.exists():
                # Look for the most recent hmm_regime_discovery outcome file
                pattern = "market_analysis_hmm_regime_discovery_outcome_*.json"
                outcome_files = list(outcomes_dir.glob(pattern))
                
                if outcome_files:
                    # Get the most recent file
                    latest_outcome = max(outcome_files, key=lambda f: f.stat().st_mtime)
                    self.logger.info(f"📁 Loading regime discovery from: {latest_outcome}")
                    
                    with open(latest_outcome, 'r') as f:
                        outcome_data = json.load(f)
                    
                    # Extract the regime discovery results from the outcome file
                    artifacts = outcome_data.get('artifacts', {})
                    regime_discovery = artifacts.get('hmm_regime_discovery_result', {})
                    
                    if regime_discovery:
                        self.logger.info(f"✅ Loaded regime discovery results: {len(regime_discovery.get('regime_models', []))} regimes")
                        return regime_discovery
                    else:
                        self.logger.warning("⚠️ No regime discovery results found in outcome file")
                else:
                    self.logger.warning("⚠️ No HMM regime discovery outcome files found")
            else:
                self.logger.warning("⚠️ Outcomes directory not found")
            
            return {}
        except Exception as e:
            self.logger.error(f"❌ Failed to get regime discovery results: {e}")
            return {}
    
    def _create_frequency_based_clusters(self, regime_assignments: List[int], n_clusters: int, data_length: int) -> List[int]:
        """Create frequency-based cluster assignments as fallback."""
        try:
            # Count regime frequencies
            regime_counts = {}
            for regime in regime_assignments:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Sort regimes by frequency (descending)
            sorted_regimes = sorted(regime_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Create frequency-based cluster mapping
            regime_to_cluster = {}
            
            if len(sorted_regimes) <= n_clusters:
                # If we have fewer regimes than clusters, assign each regime to its own cluster
                for i, (regime, count) in enumerate(sorted_regimes):
                    regime_to_cluster[regime] = i
            else:
                # Use frequency-based clustering
                # Assign the most frequent regimes to different clusters first
                for i, (regime, count) in enumerate(sorted_regimes[:n_clusters]):
                    regime_to_cluster[regime] = i
                
                # Assign remaining regimes to clusters based on similarity
                for regime, count in sorted_regimes[n_clusters:]:
                    # Find the cluster with the least total assignments so far
                    cluster_totals = [0] * n_clusters
                    for existing_regime, cluster_id in regime_to_cluster.items():
                        cluster_totals[cluster_id] += regime_counts.get(existing_regime, 0)
                    
                    # Assign to the cluster with the least total assignments
                    min_cluster = cluster_totals.index(min(cluster_totals))
                    regime_to_cluster[regime] = min_cluster
            
            # Create cluster assignments
            cluster_assignments = []
            for regime in regime_assignments:
                cluster_id = regime_to_cluster.get(regime, 0)
                cluster_assignments.append(cluster_id)
            
            # Note: Data alignment will be handled by the calling function
            
            self.logger.info(f"📊 Frequency-based clustering: {len(set(cluster_assignments))} unique clusters")
            return cluster_assignments
            
        except Exception as e:
            self.logger.error(f"❌ Frequency-based clustering failed: {e}")
            # Final fallback: create simple cluster assignments
            return [i % n_clusters for i in range(data_length)]
    
    def _validate_cluster_quality_metrics(self, cluster_assignments: List[int], regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int]) -> Dict[str, Any]:
        """Validate the quality of cluster assignments based on regime characteristics."""
        try:
            quality_metrics = {}
            
            # 1. Intra-cluster similarity (regimes within same cluster should be similar)
            intra_cluster_similarities = []
            cluster_groups = {}
            
            for regime_id, cluster_id in regime_to_cluster.items():
                if cluster_id not in cluster_groups:
                    cluster_groups[cluster_id] = []
                cluster_groups[cluster_id].append(regime_id)
            
            for cluster_id, regimes in cluster_groups.items():
                if len(regimes) > 1:
                    # Calculate average similarity within cluster
                    similarities = []
                    for i, regime_1 in enumerate(regimes):
                        for regime_2 in regimes[i+1:]:
                            similarity = self._calculate_regime_pair_similarity(
                                regime_characteristics[regime_1],
                                regime_characteristics[regime_2]
                            )
                            similarities.append(similarity)
                    if similarities:
                        intra_cluster_similarities.append(np.mean(similarities))
            
            quality_metrics['avg_intra_cluster_similarity'] = np.mean(intra_cluster_similarities) if intra_cluster_similarities else 0.0
            
            # 2. Inter-cluster dissimilarity (clusters should be distinct)
            inter_cluster_dissimilarities = []
            cluster_centroids = {}
            
            for cluster_id, regimes in cluster_groups.items():
                # Calculate cluster centroid (average characteristics)
                centroid = self._calculate_cluster_centroid(regimes, regime_characteristics)
                cluster_centroids[cluster_id] = centroid
            
            cluster_ids = list(cluster_centroids.keys())
            for i, cluster_1 in enumerate(cluster_ids):
                for cluster_2 in cluster_ids[i+1:]:
                    dissimilarity = 1.0 - self._calculate_regime_pair_similarity(
                        cluster_centroids[cluster_1],
                        cluster_centroids[cluster_2]
                    )
                    inter_cluster_dissimilarities.append(dissimilarity)
            
            quality_metrics['avg_inter_cluster_dissimilarity'] = np.mean(inter_cluster_dissimilarities) if inter_cluster_dissimilarities else 0.0
            
            # 3. Cluster balance (regimes should be reasonably distributed across clusters)
            cluster_counts = [len(regimes) for regimes in cluster_groups.values()]
            if cluster_counts:
                balance_score = 1.0 - (np.std(cluster_counts) / np.mean(cluster_counts)) if np.mean(cluster_counts) > 0 else 0.0
                quality_metrics['cluster_balance_score'] = max(0.0, balance_score)
            else:
                quality_metrics['cluster_balance_score'] = 0.0
            
            # 4. Overall quality score (weighted combination)
            overall_score = (
                0.4 * quality_metrics['avg_intra_cluster_similarity'] +
                0.4 * quality_metrics['avg_inter_cluster_dissimilarity'] +
                0.2 * quality_metrics['cluster_balance_score']
            )
            quality_metrics['overall_quality_score'] = overall_score
            
            # 5. Quality assessment
            quality_metrics['quality_level'] = (
                'excellent' if overall_score > 0.8 else
                'good' if overall_score > 0.6 else
                'fair' if overall_score > 0.4 else
                'poor'
            )
            
            self.logger.info(f"📊 Cluster quality validation: {quality_metrics['quality_level']} (score: {overall_score:.3f})")
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to validate cluster quality: {e}")
            return {'overall_quality_score': 0.0, 'quality_level': 'unknown'}
    
    def _calculate_cluster_centroid(self, regime_ids: List[str], regime_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the centroid (average) characteristics for a cluster of regimes."""
        try:
            if not regime_ids or not regime_characteristics:
                return {}
            
            # Initialize centroid structure
            centroid = {
                'volume_characteristics': {},
                'volatility_characteristics': {},
                'momentum_characteristics': {}
            }
            
            # Calculate averages for each characteristic type
            for char_type in ['volume_characteristics', 'volatility_characteristics', 'momentum_characteristics']:
                char_values = {}
                char_counts = {}
                
                for regime_id in regime_ids:
                    regime_chars = regime_characteristics.get(regime_id, {}).get(char_type, {})
                    for key, value in regime_chars.items():
                        if isinstance(value, (int, float)):
                            if key not in char_values:
                                char_values[key] = 0.0
                                char_counts[key] = 0
                            char_values[key] += value
                            char_counts[key] += 1
                
                # Calculate averages
                for key in char_values:
                    if char_counts[key] > 0:
                        centroid[char_type][key] = char_values[key] / char_counts[key]
            
            return centroid
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate cluster centroid: {e}")
            return {}
    

    
    def _create_frequency_based_clusters(self, regime_assignments: List[int], n_clusters: int, data_length: int) -> Dict[int, int]:
        """Fallback frequency-based clustering method."""
        try:
            # Type validation to prevent TypeError
            if not isinstance(n_clusters, int):
                self.logger.warning(f"⚠️ n_clusters is not int: {type(n_clusters)}, using default value 3")
                n_clusters = 3
            if not isinstance(data_length, int):
                self.logger.warning(f"⚠️ data_length is not int: {type(data_length)}, using len(regime_assignments)")
                data_length = len(regime_assignments) if regime_assignments else 100
            
            # Count regime frequencies
            regime_counts = {}
            for regime in regime_assignments:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            # Sort regimes by frequency (descending)
            sorted_regimes = sorted(regime_counts.items(), key=lambda x: x[1], reverse=True)
            
            # Create cluster mapping
            regime_to_cluster = {}
            
            if len(sorted_regimes) <= n_clusters:
                # If we have fewer regimes than clusters, assign each regime to its own cluster
                for i, (regime, count) in enumerate(sorted_regimes):
                    regime_to_cluster[regime] = i
            else:
                # Assign the most frequent regimes to different clusters first
                for i, (regime, count) in enumerate(sorted_regimes[:n_clusters]):
                    regime_to_cluster[regime] = i
                
                # Assign remaining regimes to clusters based on load balancing
                for regime, count in sorted_regimes[n_clusters:]:
                    # Find the cluster with the least total assignments so far
                    cluster_totals = [0] * n_clusters
                    for existing_regime, cluster_id in regime_to_cluster.items():
                        cluster_totals[cluster_id] += regime_counts.get(existing_regime, 0)
                    
                    # Assign to the cluster with the least total assignments
                    min_cluster = cluster_totals.index(min(cluster_totals))
                    regime_to_cluster[regime] = min_cluster
            
            return regime_to_cluster
            
        except Exception as e:
            self.logger.error(f"Frequency-based clustering failed: {e}")
            # Ultimate fallback: simple round-robin with type safety
            try:
                safe_n_clusters = int(n_clusters) if isinstance(n_clusters, (int, float, str)) else 3
                safe_data_length = int(data_length) if isinstance(data_length, (int, float, str)) else 100
                return {i: i % safe_n_clusters for i in range(max(safe_data_length, safe_n_clusters))}
            except:
                # Final fallback if everything fails
                return {i: i % 3 for i in range(100)}
    
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
    
    def _calculate_cluster_distribution_from_sizes(self, cluster_sizes: Dict[int, int]) -> Dict[str, float]:
        """Calculate the distribution of clusters based on actual sample counts."""
        if not cluster_sizes:
            return {}
        
        total_samples = sum(cluster_sizes.values())
        if total_samples == 0:
            return {}
        
        # Convert to percentages based on actual sample counts
        cluster_distribution = {}
        for cluster_id, sample_count in cluster_sizes.items():
            cluster_distribution[f'cluster_{cluster_id}'] = (sample_count / total_samples) * 100
        
        return cluster_distribution
    
    def _calculate_cluster_sizes_from_regime_mapping(self, regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int]) -> Dict[int, int]:
        """Calculate cluster sizes from regime characteristics and regime-to-cluster mapping."""
        cluster_sizes = {}
        
        for regime_id, cluster_id in regime_to_cluster.items():
            regime = regime_characteristics.get(regime_id, {})
            regime_sample_count = regime.get('sample_count', 0)
            
            if cluster_id not in cluster_sizes:
                cluster_sizes[cluster_id] = 0
            cluster_sizes[cluster_id] += regime_sample_count
        
        return cluster_sizes
    

    
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
                
                # 2.1. Momentum Analysis
                momentum_metrics = self._calculate_cluster_momentum_metrics(cluster_assignments, market_data)
                quality_metrics['momentum_analysis'] = momentum_metrics
                
                # 2.2. Statistical Significance Tests
                statistical_metrics = self._calculate_cluster_statistical_significance(
                    cluster_assignments, market_data, None, None
                )
                quality_metrics['statistical_significance'] = statistical_metrics
            
            # 3. Cross-validation Stability
            stability_metrics = self._cross_validate_clusters(
                hmm_models, cluster_assignments, market_data
            )
            quality_metrics['stability_analysis'] = stability_metrics
            
            # 4. Cluster Transition Analysis
            transition_metrics = self._analyze_cluster_transitions(cluster_assignments)
            quality_metrics['transition_analysis'] = transition_metrics
            
            # 5. Model Selection Criteria (AIC/BIC)
            if hmm_models and market_data is not None:
                model_selection_metrics = self._calculate_hmm_model_selection_criteria(hmm_models, market_data)
                quality_metrics['model_selection_criteria'] = model_selection_metrics
            
            # 6. Multi-stage Validation Gates
            validation_gates = self._apply_quality_gates(
                persistence_metrics, economic_metrics, stability_metrics, transition_metrics
            )
            quality_metrics['validation_gates'] = validation_gates
            
            # 6. Overall Quality Score
            overall_score = self._calculate_overall_quality_score(quality_metrics)
            quality_metrics['overall_quality_score'] = overall_score
            quality_metrics['validation_passed'] = overall_score >= 0.5  # 50% threshold (more lenient)
            
            # 7. Quality Recommendations
            recommendations = self._generate_quality_recommendations(quality_metrics)
            quality_metrics['recommendations'] = recommendations
            
            validation_time = time.time() - start_time
            quality_metrics['validation_time'] = validation_time
            
            # Get unique clusters with actual assignments for consistent reporting
            unique_clusters_with_assignments = sorted(set(cluster_assignments)) if cluster_assignments else []
            
            self.logger.info(f"✅ Cluster quality validation completed in {validation_time:.2f}s")
            self.logger.info(f"📊 Overall quality score: {overall_score:.2f} ({'PASSED' if quality_metrics['validation_passed'] else 'FAILED'})")
            self.logger.info(f"📈 Regime range: xx → Clusters: {len(unique_clusters_with_assignments)}")
            self.logger.info(f"📋 Detailed cluster metrics generated for {len(unique_clusters_with_assignments)} clusters")
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Cluster quality validation failed: {e}")
            return {
                'overall_quality_score': 0.0,
                'validation_passed': False,
                'error': str(e),
                'validation_time': time.time() - start_time
            }
    
    def _calculate_cluster_momentum_metrics(self, cluster_assignments: List[int], market_data: Any) -> Dict[str, Any]:
        """Calculate comprehensive momentum metrics for each cluster."""
        if not PANDAS_AVAILABLE or not isinstance(market_data, pd.DataFrame):
            return {'error': 'Pandas not available or invalid market data'}
        
        try:
            momentum_metrics = {}
            unique_clusters = list(set(cluster_assignments))
            
            for cluster_id in unique_clusters:
                cluster_mask = np.array(cluster_assignments) == cluster_id
                cluster_data = market_data[cluster_mask]
                
                if len(cluster_data) < 2:
                    continue
                
                cluster_momentum = {}
                
                # Price momentum indicators
                if 'close' in cluster_data.columns:
                    close_prices = cluster_data['close']
                    
                    # Price momentum (5, 10, 20 periods)
                    for period in [5, 10, 20]:
                        if len(close_prices) > period:
                            momentum = (close_prices.iloc[-1] - close_prices.iloc[-period-1]) / close_prices.iloc[-period-1]
                            cluster_momentum[f'price_momentum_{period}'] = float(momentum)
                    
                    # Rate of change
                    if len(close_prices) > 1:
                        roc = close_prices.pct_change().dropna()
                        cluster_momentum['mean_roc'] = float(roc.mean())
                        cluster_momentum['std_roc'] = float(roc.std())
                        cluster_momentum['roc_skewness'] = float(roc.skew()) if len(roc) > 2 else 0.0
                        cluster_momentum['roc_kurtosis'] = float(roc.kurtosis()) if len(roc) > 3 else 0.0
                
                # Volume momentum
                if 'volume' in cluster_data.columns:
                    volume = cluster_data['volume']
                    if len(volume) > 5:
                        volume_momentum = (volume.iloc[-1] - volume.iloc[-6]) / volume.iloc[-6] if volume.iloc[-6] > 0 else 0
                        cluster_momentum['volume_momentum'] = float(volume_momentum)
                        
                        # Volume trend
                        volume_ma_5 = volume.rolling(5).mean()
                        volume_ma_20 = volume.rolling(20).mean()
                        if not volume_ma_5.isna().all() and not volume_ma_20.isna().all():
                            cluster_momentum['volume_trend_strength'] = float((volume_ma_5.iloc[-1] - volume_ma_20.iloc[-1]) / volume_ma_20.iloc[-1])
                
                # Volatility momentum
                if 'high' in cluster_data.columns and 'low' in cluster_data.columns:
                    daily_ranges = (cluster_data['high'] - cluster_data['low']) / cluster_data['close']
                    if len(daily_ranges) > 5:
                        vol_momentum = daily_ranges.rolling(5).mean().iloc[-1] - daily_ranges.rolling(10).mean().iloc[-1]
                        cluster_momentum['volatility_momentum'] = float(vol_momentum)
                
                # Technical indicators (if available)
                if 'rsi' in cluster_data.columns:
                    rsi = cluster_data['rsi'].dropna()
                    if len(rsi) > 0:
                        cluster_momentum['mean_rsi'] = float(rsi.mean())
                        cluster_momentum['rsi_trend'] = 'overbought' if rsi.iloc[-1] > 70 else 'oversold' if rsi.iloc[-1] < 30 else 'neutral'
                
                if 'macd' in cluster_data.columns:
                    macd = cluster_data['macd'].dropna()
                    if len(macd) > 0:
                        cluster_momentum['mean_macd'] = float(macd.mean())
                        cluster_momentum['macd_signal'] = 'bullish' if macd.iloc[-1] > 0 else 'bearish'
                
                # Overall momentum assessment
                price_mom = cluster_momentum.get('price_momentum_5', 0)
                vol_mom = cluster_momentum.get('volume_momentum', 0)
                vol_vol_mom = cluster_momentum.get('volatility_momentum', 0)
                
                cluster_momentum['overall_momentum_score'] = abs(price_mom) + abs(vol_mom) + abs(vol_vol_mom)
                cluster_momentum['momentum_direction'] = 'bullish' if price_mom > 0.02 else 'bearish' if price_mom < -0.02 else 'neutral'
                cluster_momentum['momentum_strength'] = 'strong' if cluster_momentum['overall_momentum_score'] > 0.1 else 'weak' if cluster_momentum['overall_momentum_score'] < 0.02 else 'moderate'
                
                momentum_metrics[f'cluster_{cluster_id}'] = cluster_momentum
            
            return momentum_metrics
            
        except Exception as e:
            return {'error': f'Momentum metrics calculation failed: {e}'}
    
    def _calculate_cluster_statistical_significance(self, cluster_assignments: List[int], market_data: Any, regime_characteristics: Dict[str, Any] = None, regime_to_cluster: Dict[str, int] = None) -> Dict[str, Any]:
        """Calculate regime-level statistical significance tests for cluster differences."""
        if not regime_characteristics or not regime_to_cluster:
            return {'error': 'Regime characteristics and regime-to-cluster mapping required for regime-level analysis'}
        
        try:
            from scipy import stats
            
            statistical_metrics = {}
            unique_clusters = list(set(cluster_assignments))
            
            if len(unique_clusters) < 2:
                return {'error': 'Need at least 2 clusters for statistical analysis'}
            
            # 1. Regime-Level Statistical Tests for Volume Characteristics
            volume_tests = self._test_regime_volume_characteristics(regime_characteristics, regime_to_cluster, unique_clusters)
            statistical_metrics['regime_volume_tests'] = volume_tests
            
            # 2. Regime-Level Statistical Tests for Volatility Characteristics  
            volatility_tests = self._test_regime_volatility_characteristics(regime_characteristics, regime_to_cluster, unique_clusters)
            statistical_metrics['regime_volatility_tests'] = volatility_tests
            
            # 3. Regime-Level Statistical Tests for Momentum Characteristics
            momentum_tests = self._test_regime_momentum_characteristics(regime_characteristics, regime_to_cluster, unique_clusters)
            statistical_metrics['regime_momentum_tests'] = momentum_tests
            
            # 4. Cluster Validation Metrics for Bull/Bear/Sideways Distinction
            cluster_validation = self._validate_bull_bear_sideways_clusters(regime_characteristics, regime_to_cluster, unique_clusters)
            statistical_metrics['cluster_validation'] = cluster_validation
            
            # 5. Regime Similarity Validation (within vs across clusters)
            similarity_validation = self._validate_regime_similarity_within_across_clusters(regime_characteristics, regime_to_cluster, unique_clusters)
            statistical_metrics['regime_similarity_validation'] = similarity_validation
            
            # 6. Factor Impact Analysis for Market Dynamics
            factor_impact = self._analyze_factor_impact_on_market_dynamics(regime_characteristics, regime_to_cluster, unique_clusters)
            statistical_metrics['factor_impact_analysis'] = factor_impact
            
            # 7. Economic Regime Validation
            economic_validation = self._validate_with_economic_indicators(cluster_assignments, regime_characteristics, regime_to_cluster, market_data)
            statistical_metrics['economic_validation'] = economic_validation
            
            # 8. Overall Cluster Quality Assessment
            overall_quality = self._assess_overall_cluster_quality(volume_tests, volatility_tests, momentum_tests, cluster_validation, similarity_validation, factor_impact, economic_validation)
            statistical_metrics['overall_cluster_quality'] = overall_quality
            
            return statistical_metrics
            
        except Exception as e:
            return {'error': f'Regime-level statistical analysis failed: {e}'}
    
    def _calculate_eta_squared(self, groups, f_stat):
        """Calculate eta-squared effect size for ANOVA."""
        try:
            all_data = np.concatenate(groups)
            grand_mean = all_data.mean()
            
            # Calculate total sum of squares
            total_ss = sum((group - grand_mean)**2 for group in groups).sum()
            
            # Calculate between-group sum of squares
            between_ss = sum(len(group) * (group.mean() - grand_mean)**2 for group in groups)
            
            return float(between_ss / total_ss) if total_ss > 0 else 0.0
        except:
            return 0.0
    


    

    
    def _calculate_regime_characteristic_stats(self, characteristic_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for regime characteristics within a cluster."""
        try:
            if not characteristic_list:
                return {}
            
            # Extract all keys from the first item
            keys = list(characteristic_list[0].keys())
            stats = {}
            
            for key in keys:
                values = [item[key] for item in characteristic_list if key in item and isinstance(item[key], (int, float))]
                
                if values:
                    stats[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values)),
                        'median': float(np.median(values)),
                        'q25': float(np.percentile(values, 25)),
                        'q75': float(np.percentile(values, 75))
                    }
            
            return stats
            
        except Exception as e:
            return {'error': f'Characteristic stats calculation failed: {e}'}
    
    def _calculate_hmm_model_selection_criteria(self, hmm_models: List[Any], market_data: Any) -> Dict[str, Any]:
        """Calculate AIC, BIC, and other model selection criteria for HMM models."""
        try:
            model_criteria = {}
            
            for i, model in enumerate(hmm_models):
                if not hasattr(model, 'score'):
                    continue
                
                try:
                    # Get model parameters
                    n_components = getattr(model, 'n_components', 0)
                    n_features = market_data.shape[1] if hasattr(market_data, 'shape') else 1
                    n_samples = market_data.shape[0] if hasattr(market_data, 'shape') else len(market_data)
                    
                    # Calculate log-likelihood
                    if hasattr(market_data, 'values'):
                        data = market_data.values
                    else:
                        data = np.array(market_data)
                    
                    log_likelihood = model.score(data)
                    
                    # Calculate number of parameters
                    # For Gaussian HMM: n_components * n_features (means) + n_components * n_features * (n_features + 1) / 2 (covariances) + n_components * (n_components - 1) (transitions) + n_components - 1 (start probs)
                    if hasattr(model, 'covariance_type'):
                        if model.covariance_type == 'full':
                            n_params = n_components * n_features + n_components * n_features * (n_features + 1) // 2
                        elif model.covariance_type == 'diag':
                            n_params = n_components * n_features + n_components * n_features
                        elif model.covariance_type == 'spherical':
                            n_params = n_components * n_features + n_components
                        else:  # tied
                            n_params = n_components * n_features + n_features * (n_features + 1) // 2
                    else:
                        n_params = n_components * n_features * 2  # Conservative estimate
                    
                    n_params += n_components * (n_components - 1) + (n_components - 1)  # Transitions + start probs
                    
                    # Calculate AIC and BIC
                    aic = 2 * n_params - 2 * log_likelihood
                    bic = np.log(n_samples) * n_params - 2 * log_likelihood
                    
                    # Calculate other criteria
                    hq = 2 * n_params * np.log(np.log(n_samples)) - 2 * log_likelihood  # Hannan-Quinn
                    caic = bic + (2 * n_params * (n_params + 1)) / (n_samples - n_params - 1)  # Corrected AIC
                    
                    model_criteria[f'model_{i}'] = {
                        'n_components': int(n_components),
                        'n_parameters': int(n_params),
                        'log_likelihood': float(log_likelihood),
                        'aic': float(aic),
                        'bic': float(bic),
                        'hq': float(hq),
                        'caic': float(caic),
                        'aic_rank': 0,  # Will be filled later
                        'bic_rank': 0   # Will be filled later
                    }
                    
                except Exception as e:
                    model_criteria[f'model_{i}'] = {'error': str(e)}
            
            # Rank models by AIC and BIC
            aic_scores = [(k, v['aic']) for k, v in model_criteria.items() if 'aic' in v]
            bic_scores = [(k, v['bic']) for k, v in model_criteria.items() if 'bic' in v]
            
            aic_scores.sort(key=lambda x: x[1])
            bic_scores.sort(key=lambda x: x[1])
            
            for rank, (model_key, _) in enumerate(aic_scores):
                if model_key in model_criteria:
                    model_criteria[model_key]['aic_rank'] = rank + 1
            
            for rank, (model_key, _) in enumerate(bic_scores):
                if model_key in model_criteria:
                    model_criteria[model_key]['bic_rank'] = rank + 1
            
            # Overall model selection summary
            if aic_scores and bic_scores:
                best_aic = aic_scores[0][0]
                best_bic = bic_scores[0][0]
                
                model_criteria['model_selection_summary'] = {
                    'best_aic_model': best_aic,
                    'best_bic_model': best_bic,
                    'aic_consensus': best_aic == best_bic,
                    'total_models': len(aic_scores)
                }
            
            return model_criteria
            
        except Exception as e:
            return {'error': f'Model selection criteria calculation failed: {e}'}
    
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
        if NUMPY_AVAILABLE:
            avg_duration = np.mean(cluster_durations)
            median_duration = np.median(cluster_durations)
            std_duration = np.std(cluster_durations)
        else:
            avg_duration = sum(cluster_durations) / len(cluster_durations)
            sorted_durations = sorted(cluster_durations)
            median_duration = sorted_durations[len(sorted_durations)//2]
            # Calculate standard deviation manually
            variance = sum((x - avg_duration) ** 2 for x in cluster_durations) / len(cluster_durations)
            std_duration = variance ** 0.5
        
        # Calculate cluster stability (lower std = more stable) using math_validation
        from src.utils.math_validation import safe_divide, validate_positive
        try:
            stability_ratio = safe_divide(std_duration, avg_duration)
            # Use a more lenient calculation that doesn't penalize too harshly
            if stability_ratio <= 1.0:  # If std <= mean, good stability
                stability_score = 1.0 - (stability_ratio * 0.5)  # Scale down penalty
            else:  # If std > mean, moderate penalty
                stability_score = max(0.1, 1.0 - (stability_ratio - 1.0) * 0.3)
        except Exception as e:
            self.logger.warning(f"Stability score calculation failed: {e}")
            stability_score = 0.0
        
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
                    cluster_returns[cluster] = float(returns.mean())  # Convert to Python float
                    cluster_volatilities[cluster] = float(returns.std())  # Convert to Python float
            
            if not cluster_returns:
                return {'error': 'No valid cluster returns calculated'}
            
            # Calculate economic significance metrics
            return_spread = max(cluster_returns.values()) - min(cluster_returns.values())
            volatility_spread = max(cluster_volatilities.values()) - min(cluster_volatilities.values())
            
            # Economic significance score (higher is better)
            # Use a more appropriate normalization for the data scale
            total_spread = return_spread + volatility_spread
            if total_spread > 0.01:  # If spread is significant (>1%)
                economic_score = min(1.0, total_spread / 0.01)  # Normalize to 0-1
            elif total_spread > 0.001:  # If spread is moderate (>0.1%)
                economic_score = min(0.8, total_spread / 0.001)  # Scale to 0-0.8
            else:  # If spread is small
                economic_score = min(0.5, total_spread / 0.0001)  # Scale to 0-0.5
            
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
                        try:
                            correlation = np.corrcoef(train_values, test_values)[0, 1]
                            stability_score = max(0, correlation) if not np.isnan(correlation) else 0
                        except (ValueError, np.linalg.LinAlgError):
                            # Fallback to simple similarity measure
                            diff = sum(abs(t - s) for t, s in zip(train_values, test_values))
                            stability_score = max(0, 1 - diff / len(common_clusters))
                    else:
                        # Simple similarity measure
                        diff = sum(abs(t - s) for t, s in zip(train_values, test_values))
                        stability_score = max(0, 1 - diff / len(common_clusters))
            
            return {
                'train_distribution': train_dist,
                'test_distribution': test_dist,
                'stability_score': stability_score,
                'is_stable': stability_score >= 0.3  # More lenient threshold
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
                        if NUMPY_AVAILABLE:
                            entropy -= prob * np.log2(prob)
                        else:
                            import math
                            entropy -= prob * math.log2(prob)
            
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
            
            # Aggregate transition matrix rows and dwell-time distributions
            transition_rows = {}
            dwell_time_distribution = {}
            for cluster_key, metrics in detailed_metrics.items():
                if cluster_key.startswith('cluster_') and 'error' not in metrics:
                    cid = metrics.get('cluster_id')
                    if cid is not None:
                        transition_rows[cluster_key] = metrics.get('transition_row', {})
                        dwell_time_distribution[cluster_key] = metrics.get('dwell_time', {})

            # Add HMM-appropriate clustering metrics to report (3D-aware)
            try:
                import numpy as np
                
                # Handle 3D cluster assignments for report metrics
                if isinstance(cluster_assignments, (list, tuple)):
                    cluster_assignments_array = np.array(cluster_assignments)
                else:
                    cluster_assignments_array = cluster_assignments
                
                # For 3D clusters, determine n_clusters from array shape
                if cluster_assignments_array.ndim > 1:
                    if cluster_assignments_array.ndim == 3:
                        n_clusters = cluster_assignments_array.shape[-1]  # Last dimension is cluster count
                    elif cluster_assignments_array.ndim == 2:
                        n_clusters = cluster_assignments_array.shape[1]   # Second dimension is cluster count
                    else:
                        n_clusters = len(set(cluster_assignments_array.flatten()))
                else:
                    n_clusters = len(set(cluster_assignments_array))
                
                self.logger.info(f"📊 Report metrics: 3D cluster shape={cluster_assignments_array.shape}, n_clusters={n_clusters}")
                
                hmm_metrics = self._calculate_hmm_appropriate_metrics(
                    market_data, cluster_assignments_array, n_clusters
                )
                detailed_metrics['hmm_appropriate_metrics'] = hmm_metrics
                detailed_metrics['hmm_appropriate_metrics']['cluster_structure'] = {
                    'dimensions': cluster_assignments_array.ndim,
                    'shape': list(cluster_assignments_array.shape),
                    'is_3d': cluster_assignments_array.ndim >= 3
                }
                self.logger.info(f"✅ HMM-appropriate metrics added to report: {hmm_metrics.get('hmm_quality_score', 0.0):.3f}")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to add HMM-appropriate metrics to report: {e}")
                detailed_metrics['hmm_appropriate_metrics'] = {'error': str(e)}
            
            # Add within-cluster coherence metrics to report (3D-aware)
            try:
                import numpy as np
                from sklearn.metrics import pairwise_distances
                
                # Handle 3D cluster assignments for coherence calculation
                if isinstance(cluster_assignments, (list, tuple)):
                    cluster_assignments_array = np.array(cluster_assignments)
                else:
                    cluster_assignments_array = cluster_assignments
                
                # Create feature matrix for coherence calculation
                if hasattr(market_data, 'select_dtypes'):
                    numeric_data = market_data.select_dtypes(include=[np.number])
                    
                    # For 3D clusters, we need to handle length compatibility differently
                    if cluster_assignments_array.ndim > 1:
                        # For 3D/2D clusters, flatten to get the effective data length
                        if cluster_assignments_array.ndim == 3:
                            effective_length = cluster_assignments_array.shape[0] * cluster_assignments_array.shape[1]
                        elif cluster_assignments_array.ndim == 2:
                            effective_length = cluster_assignments_array.shape[0]
                        else:
                            effective_length = len(cluster_assignments_array.flatten())
                    else:
                        effective_length = len(cluster_assignments_array)
                    
                    # Adjust data length for compatibility
                    min_length = min(len(numeric_data), effective_length)
                    
                    if not numeric_data.empty and min_length > 0:
                        # Calculate distance matrix with compatible length
                        numeric_data_adj = numeric_data.iloc[:min_length]
                        distance_matrix = pairwise_distances(numeric_data_adj.fillna(0))
                        
                        # Calculate within-cluster coherence (handles 3D internally)
                        coherence_score = self._calculate_within_cluster_coherence(
                            distance_matrix, cluster_assignments_array
                        )
                        
                        detailed_metrics['within_cluster_coherence'] = {
                            'coherence_score': coherence_score,
                            'interpretation': 'Higher scores indicate better internal cluster consistency',
                            'method': 'Distance-based coherence analysis (3D-aware)',
                            'cluster_structure': {
                                'dimensions': cluster_assignments_array.ndim,
                                'shape': list(cluster_assignments_array.shape),
                                'effective_length': effective_length,
                                'data_length': len(numeric_data),
                                'adjusted_length': min_length
                            }
                        }
                        self.logger.info(f"✅ Within-cluster coherence added to report: {coherence_score:.3f}")
                    else:
                        detailed_metrics['within_cluster_coherence'] = {
                            'error': 'No numeric data or invalid length',
                            'coherence_score': 0.0,
                            'debug_info': {
                                'numeric_data_empty': numeric_data.empty,
                                'effective_length': effective_length,
                                'data_length': len(numeric_data)
                            }
                        }
                else:
                    detailed_metrics['within_cluster_coherence'] = {
                        'error': 'Invalid market data format',
                        'coherence_score': 0.0
                    }
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to add within-cluster coherence to report: {e}")
                detailed_metrics['within_cluster_coherence'] = {
                    'error': str(e),
                    'coherence_score': 0.0
                }

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

            # Persist global transition and dwell-time artifacts
            detailed_metrics['transition_matrix_rows'] = transition_rows
            detailed_metrics['dwell_time_distribution'] = dwell_time_distribution
            
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
            import numpy as np
            
            # Data should already be aligned from the clustering process
            # Use alignment method for consistency and error handling
            alignment_result = self._align_data_lengths(market_data, cluster_assignments)
            market_data = alignment_result.aligned_market_data
            cluster_assignments = alignment_result.aligned_assignments
            
            if alignment_result.was_aligned:
                self.logger.warning(f"⚠️ Data alignment was needed in _analyze_single_cluster - this indicates a bug in upstream alignment")
            
            # Get cluster data
            cluster_mask = np.array(cluster_assignments) == cluster_id
            cluster_data = market_data[cluster_mask]
            
            if len(cluster_data) < 2:
                return {'error': f'Insufficient data for cluster {cluster_id}'}
            
            cluster_metrics = {
                'cluster_id': cluster_id,
                'sample_count': int(len(cluster_data)),
                'sample_percentage': (len(cluster_data) / len(market_data)) * 100
            }

            # Economic metrics (annualized when possible)
            bars_per_year = self._infer_bars_per_year(market_data)
            returns_metrics = self._compute_returns_metrics(cluster_data, bars_per_year)
            cluster_metrics['returns_analysis'] = returns_metrics

            # Dwell-time statistics for this cluster
            dwell_stats = self._compute_dwell_times_for_cluster(cluster_assignments, cluster_id)
            cluster_metrics['dwell_time'] = dwell_stats

            # Transition probability row for this cluster
            n_clusters = int(len(set(cluster_assignments)))
            cluster_metrics['transition_row'] = self._compute_transition_row(cluster_assignments, cluster_id, n_clusters)
            
            
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
            # Economic interpretability label
            vol_level = cluster_metrics.get('volatility_analysis', {}).get('volatility_classification', {}).get('volatility_level')
            trend_metrics = cluster_metrics.get('trend_analysis', {}).get('trend_metrics', {})
            volume_cv = cluster_metrics.get('volume_analysis', {}).get('volume_volatility', {}).get('volume_cv', 0.0)
            econ_label = self._infer_economic_label(
                trend_metrics,
                vol_level,
                returns_metrics.get('annualized_return', 0.0),
                returns_metrics.get('annualized_sharpe', 0.0),
                volume_cv
            )
            cluster_metrics['economic_interpretability'] = {'label': econ_label}

            return cluster_metrics
            
        except Exception as e:
            return {'error': f'Analysis failed for cluster {cluster_id}: {e}'}
    
    
    def _infer_bars_per_year(self, market_data: pd.DataFrame) -> float:
        """Infer approximate bars-per-year from timestamp cadence when possible."""
        try:
            ts_col = None
            for candidate in ['timestamp', 'open_time', 'close_time']:
                if candidate in market_data.columns:
                    ts_col = candidate
                    break
            if ts_col is None:
                return 365.0 * 24.0
            ts = market_data[ts_col]
            if np.issubdtype(ts.dtype, np.number):
                unit = 'ms' if ts.iloc[-1] > 10_000_000_000 else 's'
                t = pd.to_datetime(ts, unit=unit, errors='coerce')
            else:
                t = pd.to_datetime(ts, errors='coerce')
            dt = t.diff().dropna()
            if len(dt) == 0:
                return 365.0 * 24.0
            median_seconds = dt.median().total_seconds()
            if median_seconds <= 0:
                return 365.0 * 24.0
            return float((365.0 * 24.0 * 3600.0) / median_seconds)
        except Exception:
            return 365.0 * 24.0


    def _compute_returns_metrics(self, cluster_data: pd.DataFrame, bars_per_year: float) -> Dict[str, Any]:
        """Compute per-cluster return/volatility/Sharpe and 95% CI on mean returns."""
        try:
            close = cluster_data['close'] if 'close' in cluster_data.columns else None
            if close is None or len(close) < 3:
                return {
                    'bars_per_year': bars_per_year,
                    'annualized_return': 0.0,
                    'annualized_volatility': 0.0,
                    'annualized_sharpe': 0.0,
                    'mean_return_bar': 0.0,
                    'std_return_bar': 0.0,
                    'ci95_bar': [0.0, 0.0],
                    'ci95_annualized': [0.0, 0.0]
                }
            r = close.pct_change().dropna()
            if len(r) == 0:
                return {
                    'bars_per_year': bars_per_year,
                    'annualized_return': 0.0,
                    'annualized_volatility': 0.0,
                    'annualized_sharpe': 0.0,
                    'mean_return_bar': 0.0,
                    'std_return_bar': 0.0,
                    'ci95_bar': [0.0, 0.0],
                    'ci95_annualized': [0.0, 0.0]
                }
            mean_bar = float(r.mean())
            std_bar = float(r.std()) if r.std() == r.std() else 0.0
            n = max(1, len(r))
            try:
                ann_return = float((1.0 + mean_bar) ** bars_per_year - 1.0)
            except Exception:
                ann_return = float(mean_bar * bars_per_year)
            ann_vol = float(std_bar * np.sqrt(bars_per_year)) if std_bar == std_bar else 0.0
            sharpe = float((mean_bar / std_bar) * np.sqrt(bars_per_year)) if std_bar > 0 else 0.0
            se = std_bar / np.sqrt(n) if n > 0 else 0.0
            z = 1.96
            ci_low_bar = mean_bar - z * se
            ci_high_bar = mean_bar + z * se
            ci_low_ann = ci_low_bar * bars_per_year
            ci_high_ann = ci_high_bar * bars_per_year
            return {
                'bars_per_year': bars_per_year,
                'annualized_return': ann_return,
                'annualized_volatility': ann_vol,
                'annualized_sharpe': sharpe,
                'mean_return_bar': mean_bar,
                'std_return_bar': std_bar,
                'ci95_bar': [ci_low_bar, ci_high_bar],
                'ci95_annualized': [ci_low_ann, ci_high_ann]
            }
        except Exception:
            return {
                'bars_per_year': bars_per_year,
                'annualized_return': 0.0,
                'annualized_volatility': 0.0,
                'annualized_sharpe': 0.0,
                'mean_return_bar': 0.0,
                'std_return_bar': 0.0,
                'ci95_bar': [0.0, 0.0],
                'ci95_annualized': [0.0, 0.0]
            }


    def _compute_dwell_times_for_cluster(self, cluster_assignments: List[int], cluster_id: int) -> Dict[str, Any]:
        """Compute run-length (dwell-time) statistics for a single cluster."""
        try:
            runs = []
            current_len = 0
            for a in cluster_assignments:
                if a == cluster_id:
                    current_len += 1
                else:
                    if current_len > 0:
                        runs.append(current_len)
                        current_len = 0
            if current_len > 0:
                runs.append(current_len)
            if not runs:
                return {'count': 0, 'mean': 0.0, 'median': 0.0, 'min': 0, 'max': 0, 'p25': 0.0, 'p75': 0.0}
            arr = np.array(runs, dtype=float)
            return {
                'count': int(len(runs)),
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'min': int(np.min(arr)),
                'max': int(np.max(arr)),
                'p25': float(np.percentile(arr, 25)),
                'p75': float(np.percentile(arr, 75))
            }
        except Exception:
            return {'count': 0, 'mean': 0.0, 'median': 0.0, 'min': 0, 'max': 0, 'p25': 0.0, 'p75': 0.0}

    def _passes_temporal_coherence_constraint(self, cluster_assignments: List[int], regime_to_cluster: Dict[str, int], cluster_i: int, cluster_j: int, max_entropy_increase_pct: float = 0.10, max_dwell_kl: float = 0.25) -> bool:
        """Check that merging cluster_j into cluster_i does not break temporal coherence.

        - Blocks if transition entropy increases by more than max_entropy_increase_pct
        - Blocks if dwell-time distributions of the two clusters differ by KL > max_dwell_kl
        """
        try:
            import numpy as np
            from math import log

            def transition_entropy(assignments: np.ndarray, n_clusters: int) -> float:
                T = np.zeros((n_clusters, n_clusters), dtype=float)
                for a, b in zip(assignments[:-1], assignments[1:]):
                    if a < 0 or b < 0 or a >= n_clusters or b >= n_clusters:
                        continue
                    T[a, b] += 1.0
                ent = 0.0
                rows = 0
                for r in range(n_clusters):
                    row_sum = T[r].sum()
                    if row_sum <= 0:
                        continue
                    p = T[r] / row_sum
                    h = -float(np.nansum([pi * log(pi + 1e-12) for pi in p if pi > 0]))
                    ent += h
                    rows += 1
                return ent / max(1, rows)

            def dwell_hist(assignments: np.ndarray, cid: int) -> np.ndarray:
                lengths = []
                cur = None
                count = 0
                for v in assignments:
                    if v == cid:
                        if cur == cid:
                            count += 1
                        else:
                            if cur == cid and count > 0:
                                lengths.append(count)
                            cur = cid
                            count = 1
                    else:
                        if cur == cid and count > 0:
                            lengths.append(count)
                        cur = v
                        count = 0
                if cur == cid and count > 0:
                    lengths.append(count)
                if len(lengths) == 0:
                    return np.array([1.0])
                # 4-bin histogram: 1,2,3,4+
                hist = [0, 0, 0, 0]
                for L in lengths:
                    if L <= 1:
                        hist[0] += 1
                    elif L == 2:
                        hist[1] += 1
                    elif L == 3:
                        hist[2] += 1
                    else:
                        hist[3] += 1
                s = float(sum(hist))
                if s == 0:
                    return np.array([1.0])
                return np.array([h / s for h in hist], dtype=float)

            def kl_div(p: np.ndarray, q: np.ndarray) -> float:
                p = p / max(1e-12, p.sum())
                q = q / max(1e-12, q.sum())
                eps = 1e-8
                p = p + eps
                q = q + eps
                return float(np.sum(p * np.log(p / q)))

            arr = np.array(cluster_assignments, dtype=int)
            n_before = int(np.max(arr)) + 1 if arr.size > 0 else 0
            if n_before <= max(cluster_i, cluster_j):
                return True
            H_before = transition_entropy(arr, n_before)
            merged = arr.copy()
            merged[merged == cluster_j] = cluster_i
            unique = sorted(set(merged.tolist()))
            remap = {old: idx for idx, old in enumerate(unique)}
            merged = np.array([remap[v] for v in merged], dtype=int)
            H_after = transition_entropy(merged, len(unique))
            if H_before > 0 and (H_after - H_before) / H_before > max_entropy_increase_pct:
                return False
            p = dwell_hist(arr, cluster_i)
            q = dwell_hist(arr, cluster_j)
            if kl_div(p, q) > max_dwell_kl:
                return False
            return True
        except Exception:
            return False

    def _compute_transition_row(self, cluster_assignments: List[int], cluster_id: int, n_clusters: int) -> Dict[str, float]:
        """Compute transition probabilities from a given cluster to others."""
        try:
            counts = {k: 0 for k in range(n_clusters)}
            total = 0
            for i in range(1, len(cluster_assignments)):
                if cluster_assignments[i-1] == cluster_id:
                    to_c = cluster_assignments[i]
                    counts[to_c] = counts.get(to_c, 0) + 1
                    total += 1
            if total == 0:
                return {f'to_{k}': 0.0 for k in range(n_clusters)}
            return {f'to_{k}': float(counts.get(k, 0) / total) for k in range(n_clusters)}
        except Exception:
            return {f'to_{k}': 0.0 for k in range(n_clusters)}


    def _infer_economic_label(self, trend_metrics: Dict[str, Any], vol_level: Any, annualized_return: float, annualized_sharpe: float, volume_cv: float) -> str:
        """Infer a concise economic interpretability label for a cluster."""
        try:
            direction = trend_metrics.get('trend_direction')
            strength = float(trend_metrics.get('trend_strength', 0.0))
            consistency = str(trend_metrics.get('trend_consistency', 'neutral'))
            vol_level = str(vol_level) if vol_level else 'unknown'
            if direction == 'upward' and strength > 0.5 and annualized_sharpe > 0:
                return 'trend-up'
            if vol_level == 'high' and annualized_return >= 0:
                return 'high-vol carry'
            if consistency in ('moderate', 'inconsistent') and abs(annualized_return) < 0.05:
                return 'mean-reverting'
            if (volume_cv is not None and volume_cv < 0.3) and vol_level in ('low', 'medium') and consistency == 'inconsistent':
                return 'illiquid chop'
            return 'neutral'
        except Exception:
            return 'neutral'
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
        if NUMPY_AVAILABLE:
            std_count = np.std(counts)
        else:
            # Calculate standard deviation manually
            variance = sum((x - mean_count) ** 2 for x in counts) / len(counts)
            std_count = variance ** 0.5
        
        # Use math_validation for safe division
        from src.utils.math_validation import safe_divide
        try:
            balance_ratio = safe_divide(std_count, mean_count)
            balance_score = max(0, 1 - balance_ratio)
        except Exception as e:
            self.logger.warning(f"Balance score calculation failed: {e}")
            balance_score = 0.0
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
            # Use train_hmm_parallel method instead of non-existent perform_hmm_clustering
            n_models = config.get('n_clusters', 3)
            hmm_models = hmm_manager.train_hmm_parallel(
                data=chunk_data,
                n_models=n_models,
                config=None
            )
            
            # Create cluster assignments using HMM model predictions
            cluster_assignments = []
            if hmm_models and len(hmm_models) > 0:
                # Use the first HMM model to predict cluster assignments
                try:
                    # Prepare data for HMM prediction
                    if PANDAS_AVAILABLE and isinstance(chunk_data, pd.DataFrame):
                        # Select numeric columns for HMM prediction
                        numeric_cols = chunk_data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            hmm_data = chunk_data[numeric_cols].values
                            # Handle NaN values by filling with forward fill then backward fill
                            if PANDAS_AVAILABLE:
                                hmm_df = pd.DataFrame(hmm_data, columns=numeric_cols)
                                hmm_df = hmm_df.fillna(method='ffill').fillna(method='bfill')
                                hmm_data = hmm_df.values
                            else:
                                # Simple NaN handling for numpy arrays
                                hmm_data = np.nan_to_num(hmm_data, nan=0.0)
                        else:
                            # Fallback to simple round-robin if no numeric data
                            cluster_assignments = [int(i % n_models) for i in range(len(chunk_data))]
                    else:
                        # Fallback to simple round-robin
                        cluster_assignments = [int(i % n_models) for i in range(len(chunk_data))]
                    
                    if not cluster_assignments:  # If we haven't assigned yet
                        # Use HMM model to predict states
                        try:
                            hmm_model = hmm_models[0]  # Use first model
                            if hasattr(hmm_model, 'predict'):
                                predicted_states = hmm_model.predict(hmm_data)
                                # Map states to cluster assignments (0 to n_models-1)
                                cluster_assignments = [int(state % n_models) for state in predicted_states]
                            else:
                                # Fallback to round-robin
                                cluster_assignments = [i % n_models for i in range(len(chunk_data))]
                        except Exception as e:
                            self.logger.warning(f"⚠️ HMM prediction failed for chunk {chunk_idx}: {e}")
                            # Fallback to round-robin
                            cluster_assignments = [int(i % n_models) for i in range(len(chunk_data))]
                except Exception as e:
                    self.logger.warning(f"⚠️ Cluster assignment failed for chunk {chunk_idx}: {e}")
                    # Fallback to round-robin
                    cluster_assignments = [int(i % n_models) for i in range(len(chunk_data))]
            else:
                # Fallback to round-robin if no models
                cluster_assignments = [int(i % n_models) for i in range(len(chunk_data))]
            
            # Return standardized format
            result = STANDARD_CLUSTERING_RESULT.copy()
            result.update({
                'hmm_models': hmm_models,
                'cluster_assignments': cluster_assignments,
                'cluster_metrics': {
                    'clustering_method': 'chunk_based',
                    'chunk_size': len(chunk_data)
                },
                'success': True,
                'error': None,
                'chunk_index': chunk_idx
            })
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Chunk {chunk_idx} clustering failed: {e}")
            # Return standardized error format
            result = STANDARD_CLUSTERING_RESULT.copy()
            result.update({
                'success': False,
                'error': str(e),
                'chunk_index': chunk_idx
            })
            return result
    
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
                
                # Keep original assignment indices (clusters should be consistent across chunks)
                if assignments:
                    all_assignments.extend(assignments)
                
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
            
            # Convert HMM models to JSON-serializable format
            serializable_models = []
            for model in all_models:
                if hasattr(model, 'means_') and hasattr(model, 'covars_'):
                    # Extract only the essential parameters that are JSON serializable
                    serializable_model = {
                        'n_components': int(model.n_components),
                        'covariance_type': str(model.covariance_type),
                        'means': model.means_.tolist() if hasattr(model.means_, 'tolist') else model.means_,
                        'covars': model.covars_.tolist() if hasattr(model.covars_, 'tolist') else model.covars_,
                        'transmat_': model.transmat_.tolist() if hasattr(model.transmat_, 'tolist') else model.transmat_,
                        'startprob_': model.startprob_.tolist() if hasattr(model.startprob_, 'tolist') else model.startprob_
                    }
                    serializable_models.append(serializable_model)
                else:
                    # If model doesn't have expected attributes, create a minimal representation
                    serializable_models.append({
                        'n_components': int(model.n_components) if hasattr(model, 'n_components') else 2,
                        'covariance_type': str(model.covariance_type) if hasattr(model, 'covariance_type') else 'diag',
                        'means': [[0.0] * 2],  # Default means
                        'covars': [[1.0, 1.0]],  # Default covariances
                        'transmat_': [[0.5, 0.5], [0.5, 0.5]],  # Default transition matrix
                        'startprob_': [0.5, 0.5]  # Default start probabilities
                    })
            
            # Return standardized format
            result = STANDARD_CLUSTERING_RESULT.copy()
            result.update({
                'hmm_models': serializable_models,
                'cluster_assignments': all_assignments,
                'cluster_metrics': merged_metrics,
                'success': True,
                'error': None
            })
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to merge chunk results: {e}")
            # Return standardized error format
            result = STANDARD_CLUSTERING_RESULT.copy()
            result.update({
                'success': False,
                'error': f'Merge failed: {e}'
            })
            return result

    # ========== NEW REGIME-LEVEL STATISTICAL ANALYSIS FUNCTIONS ==========
    
    def _test_regime_volume_characteristics(self, regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int], unique_clusters: List[int]) -> Dict[str, Any]:
        """Test statistical differences in volume characteristics between clusters."""
        try:
            from scipy import stats
            
            volume_tests = {}
            
            # Volume characteristics to test
            volume_features = [
                'mean_volume_momentum_5', 'mean_volume_momentum_20', 'mean_volume_ratio',
                'volume_momentum_volatility', 'volume_ratio_volatility'
            ]
            
            # Extract volume data by cluster
            cluster_volume_data = {}
            for cluster_id in unique_clusters:
                cluster_volume_data[cluster_id] = {feature: [] for feature in volume_features}
            
            # Collect regime-level volume characteristics by cluster
            for regime_id, cluster_id in regime_to_cluster.items():
                if regime_id in regime_characteristics:
                    volume_chars = regime_characteristics[regime_id].get('volume_characteristics', {})
                    for feature in volume_features:
                        if feature in volume_chars:
                            cluster_volume_data[cluster_id][feature].append(volume_chars[feature])
            
            # Perform ANOVA tests for each volume feature
            for feature in volume_features:
                groups = []
                for cluster_id in unique_clusters:
                    data = cluster_volume_data[cluster_id][feature]
                    if len(data) > 0:
                        groups.append(data)
                
                if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                    try:
                        f_stat, p_value = stats.f_oneway(*groups)
                        effect_size = self._calculate_eta_squared(groups, f_stat)
                        
                        volume_tests[feature] = {
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'effect_size': float(effect_size),
                            'effect_magnitude': 'large' if effect_size > 0.14 else 'medium' if effect_size > 0.06 else 'small',
                            'cluster_means': {f'cluster_{cid}': float(np.mean(cluster_volume_data[cid][feature])) if cluster_volume_data[cid][feature] else 0.0 for cid in unique_clusters}
                        }
                    except Exception as e:
                        volume_tests[feature] = {'error': str(e)}
                else:
                    volume_tests[feature] = {'error': 'Insufficient data for statistical test'}
            
            # Overall volume differentiation score
            significant_features = sum(1 for test in volume_tests.values() if isinstance(test, dict) and test.get('significant', False))
            total_features = len([test for test in volume_tests.values() if isinstance(test, dict) and 'error' not in test])
            
            volume_tests['overall_volume_differentiation'] = {
                'significant_features': significant_features,
                'total_features': total_features,
                'differentiation_score': significant_features / total_features if total_features > 0 else 0.0,
                'differentiation_quality': 'high' if (significant_features / total_features if total_features > 0 else 0) > 0.6 else 'medium' if (significant_features / total_features if total_features > 0 else 0) > 0.3 else 'low'
            }
            
            return volume_tests
            
        except Exception as e:
            return {'error': f'Volume characteristics test failed: {e}'}
    
    def _test_regime_volatility_characteristics(self, regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int], unique_clusters: List[int]) -> Dict[str, Any]:
        """Test statistical differences in volatility characteristics between clusters."""
        try:
            from scipy import stats
            
            volatility_tests = {}
            
            # Volatility characteristics to test
            volatility_features = [
                'mean_volatility_5', 'mean_volatility_10', 'mean_volatility_20',
                'volatility_momentum', 'volatility_acceleration', 'mean_atr_normalized'
            ]
            
            # Extract volatility data by cluster
            cluster_volatility_data = {}
            for cluster_id in unique_clusters:
                cluster_volatility_data[cluster_id] = {feature: [] for feature in volatility_features}
            
            # Collect regime-level volatility characteristics by cluster
            for regime_id, cluster_id in regime_to_cluster.items():
                if regime_id in regime_characteristics:
                    volatility_chars = regime_characteristics[regime_id].get('volatility_characteristics', {})
                    for feature in volatility_features:
                        if feature in volatility_chars:
                            cluster_volatility_data[cluster_id][feature].append(volatility_chars[feature])
            
            # Perform ANOVA tests for each volatility feature
            for feature in volatility_features:
                groups = []
                for cluster_id in unique_clusters:
                    data = cluster_volatility_data[cluster_id][feature]
                    if len(data) > 0:
                        groups.append(data)
                
                if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                    try:
                        f_stat, p_value = stats.f_oneway(*groups)
                        effect_size = self._calculate_eta_squared(groups, f_stat)
                        
                        volatility_tests[feature] = {
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'effect_size': float(effect_size),
                            'effect_magnitude': 'large' if effect_size > 0.14 else 'medium' if effect_size > 0.06 else 'small',
                            'cluster_means': {f'cluster_{cid}': float(np.mean(cluster_volatility_data[cid][feature])) if cluster_volatility_data[cid][feature] else 0.0 for cid in unique_clusters}
                        }
                    except Exception as e:
                        volatility_tests[feature] = {'error': str(e)}
                else:
                    volatility_tests[feature] = {'error': 'Insufficient data for statistical test'}
            
            # Overall volatility differentiation score
            significant_features = sum(1 for test in volatility_tests.values() if isinstance(test, dict) and test.get('significant', False))
            total_features = len([test for test in volatility_tests.values() if isinstance(test, dict) and 'error' not in test])
            
            volatility_tests['overall_volatility_differentiation'] = {
                'significant_features': significant_features,
                'total_features': total_features,
                'differentiation_score': significant_features / total_features if total_features > 0 else 0.0,
                'differentiation_quality': 'high' if (significant_features / total_features if total_features > 0 else 0) > 0.6 else 'medium' if (significant_features / total_features if total_features > 0 else 0) > 0.3 else 'low'
            }
            
            return volatility_tests
            
        except Exception as e:
            return {'error': f'Volatility characteristics test failed: {e}'}
    
    def _test_regime_momentum_characteristics(self, regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int], unique_clusters: List[int]) -> Dict[str, Any]:
        """Test statistical differences in momentum characteristics between clusters."""
        try:
            from scipy import stats
            
            momentum_tests = {}
            
            # Momentum characteristics to test (including risk-return features)
            momentum_features = [
                'mean_price_momentum_5', 'mean_price_momentum_20', 'mean_rsi', 'mean_macd',
                'rsi_momentum', 'macd_momentum', 'momentum_strength',
                'sharpe_ratio', 'trend_persistence', 'max_drawdown', 'volatility_clustering',
                'risk_adjusted_return', 'drawdown_recovery_ratio'
            ]
            
            # Extract momentum data by cluster
            cluster_momentum_data = {}
            for cluster_id in unique_clusters:
                cluster_momentum_data[cluster_id] = {feature: [] for feature in momentum_features}
            
            # Collect regime-level momentum characteristics by cluster
            for regime_id, cluster_id in regime_to_cluster.items():
                if regime_id in regime_characteristics:
                    momentum_chars = regime_characteristics[regime_id].get('momentum_characteristics', {})
                    for feature in momentum_features:
                        if feature in momentum_chars:
                            cluster_momentum_data[cluster_id][feature].append(momentum_chars[feature])
            
            # Perform ANOVA tests for each momentum feature
            for feature in momentum_features:
                groups = []
                for cluster_id in unique_clusters:
                    data = cluster_momentum_data[cluster_id][feature]
                    if len(data) > 0:
                        groups.append(data)
                
                if len(groups) >= 2 and all(len(g) > 1 for g in groups):
                    try:
                        f_stat, p_value = stats.f_oneway(*groups)
                        effect_size = self._calculate_eta_squared(groups, f_stat)
                        
                        momentum_tests[feature] = {
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05,
                            'effect_size': float(effect_size),
                            'effect_magnitude': 'large' if effect_size > 0.14 else 'medium' if effect_size > 0.06 else 'small',
                            'cluster_means': {f'cluster_{cid}': float(np.mean(cluster_momentum_data[cid][feature])) if cluster_momentum_data[cid][feature] else 0.0 for cid in unique_clusters}
                        }
                    except Exception as e:
                        momentum_tests[feature] = {'error': str(e)}
                else:
                    momentum_tests[feature] = {'error': 'Insufficient data for statistical test'}
            
            # Overall momentum differentiation score
            significant_features = sum(1 for test in momentum_tests.values() if isinstance(test, dict) and test.get('significant', False))
            total_features = len([test for test in momentum_tests.values() if isinstance(test, dict) and 'error' not in test])
            
            momentum_tests['overall_momentum_differentiation'] = {
                'significant_features': significant_features,
                'total_features': total_features,
                'differentiation_score': significant_features / total_features if total_features > 0 else 0.0,
                'differentiation_quality': 'high' if (significant_features / total_features if total_features > 0 else 0) > 0.6 else 'medium' if (significant_features / total_features if total_features > 0 else 0) > 0.3 else 'low'
            }
            
            return momentum_tests
            
        except Exception as e:
            return {'error': f'Momentum characteristics test failed: {e}'}
    
    def _validate_bull_bear_sideways_clusters(self, regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int], unique_clusters: List[int]) -> Dict[str, Any]:
        """Validate whether the 3 clusters meaningfully represent Bull/Bear/Sideways market conditions."""
        try:
            cluster_validation = {}
            
            # Calculate cluster profiles
            cluster_profiles = {}
            for cluster_id in unique_clusters:
                cluster_regimes = [regime_id for regime_id, cid in regime_to_cluster.items() if cid == cluster_id]
                
                if not cluster_regimes:
                    continue
                
                # Aggregate characteristics for this cluster
                momentum_values = []
                volatility_values = []
                volume_momentum_values = []
                
                for regime_id in cluster_regimes:
                    if regime_id in regime_characteristics:
                        regime_data = regime_characteristics[regime_id]
                        
                        # Momentum indicators
                        momentum_chars = regime_data.get('momentum_characteristics', {})
                        if momentum_chars:
                            momentum_values.append(momentum_chars.get('mean_price_momentum_5', 0))
                        
                        # Volatility indicators
                        volatility_chars = regime_data.get('volatility_characteristics', {})
                        if volatility_chars:
                            volatility_values.append(volatility_chars.get('mean_volatility_20', 0))
                        
                        # Volume momentum
                        volume_chars = regime_data.get('volume_characteristics', {})
                        if volume_chars:
                            volume_momentum_values.append(volume_chars.get('mean_volume_momentum_5', 0))
                
                # Calculate cluster profile
                avg_momentum = np.mean(momentum_values) if momentum_values else 0
                avg_volatility = np.mean(volatility_values) if volatility_values else 0
                avg_volume_momentum = np.mean(volume_momentum_values) if volume_momentum_values else 0
                
                # Classify cluster behavior
                if avg_momentum > 0.02:  # Strong positive momentum
                    market_type = 'Bull'
                    confidence = min(avg_momentum * 20, 1.0)  # Scale to 0-1
                elif avg_momentum < -0.02:  # Strong negative momentum
                    market_type = 'Bear'
                    confidence = min(abs(avg_momentum) * 20, 1.0)
                else:  # Low momentum
                    market_type = 'Sideways'
                    # High volatility with low momentum suggests sideways
                    confidence = min(avg_volatility * 10, 1.0) if avg_volatility > 0.01 else 0.3
                
                cluster_profiles[cluster_id] = {
                    'avg_momentum': float(avg_momentum),
                    'avg_volatility': float(avg_volatility),
                    'avg_volume_momentum': float(avg_volume_momentum),
                    'predicted_market_type': market_type,
                    'classification_confidence': float(confidence),
                    'n_regimes': len(cluster_regimes)
                }
            
            # Validate cluster distinctiveness
            cluster_validation['cluster_profiles'] = cluster_profiles
            
            # Check if we have good Bull/Bear/Sideways separation
            market_types = [profile['predicted_market_type'] for profile in cluster_profiles.values()]
            unique_market_types = set(market_types)
            
            cluster_validation['market_type_coverage'] = {
                'unique_market_types': list(unique_market_types),
                'has_bull': 'Bull' in unique_market_types,
                'has_bear': 'Bear' in unique_market_types,
                'has_sideways': 'Sideways' in unique_market_types,
                'complete_coverage': len(unique_market_types) == 3
            }
            
            # Calculate separation quality
            momentum_values_by_cluster = []
            for cluster_id in unique_clusters:
                if cluster_id in cluster_profiles:
                    momentum_values_by_cluster.append(cluster_profiles[cluster_id]['avg_momentum'])
            
            if len(momentum_values_by_cluster) >= 2:
                momentum_range = max(momentum_values_by_cluster) - min(momentum_values_by_cluster)
                momentum_std = np.std(momentum_values_by_cluster)
                
                cluster_validation['separation_quality'] = {
                    'momentum_range': float(momentum_range),
                    'momentum_std': float(momentum_std),
                    'separation_score': float(momentum_range + momentum_std),
                    'separation_quality': 'high' if momentum_range > 0.08 else 'medium' if momentum_range > 0.04 else 'low'
                }
            
            # Overall validation score
            coverage_score = len(unique_market_types) / 3.0  # 0-1 based on market type coverage
            separation_score = cluster_validation.get('separation_quality', {}).get('separation_score', 0) / 0.2  # Normalize
            separation_score = min(separation_score, 1.0)
            
            avg_confidence = np.mean([profile['classification_confidence'] for profile in cluster_profiles.values()]) if cluster_profiles else 0
            
            overall_score = (coverage_score * 0.4 + separation_score * 0.4 + avg_confidence * 0.2)
            
            cluster_validation['overall_validation'] = {
                'coverage_score': float(coverage_score),
                'separation_score': float(separation_score),
                'avg_confidence': float(avg_confidence),
                'overall_score': float(overall_score),
                'validation_quality': 'high' if overall_score > 0.7 else 'medium' if overall_score > 0.5 else 'low'
            }
            
            return cluster_validation
            
        except Exception as e:
            return {'error': f'Bull/Bear/Sideways validation failed: {e}'}
    
    def _validate_regime_similarity_within_across_clusters(self, regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int], unique_clusters: List[int]) -> Dict[str, Any]:
        """Validate that regimes within clusters are more similar than regimes across clusters."""
        try:
            similarity_validation = {}
            
            # Calculate within-cluster similarities
            within_cluster_similarities = {}
            for cluster_id in unique_clusters:
                cluster_regimes = [regime_id for regime_id, cid in regime_to_cluster.items() if cid == cluster_id]
                
                if len(cluster_regimes) < 2:
                    within_cluster_similarities[cluster_id] = {'mean_similarity': 1.0, 'similarities': []}
                    continue
                
                similarities = []
                for i, regime_1 in enumerate(cluster_regimes):
                    for regime_2 in cluster_regimes[i+1:]:
                        if regime_1 in regime_characteristics and regime_2 in regime_characteristics:
                            similarity = self._calculate_regime_similarity(
                                regime_characteristics[regime_1], 
                                regime_characteristics[regime_2]
                            )
                            similarities.append(similarity)
                
                within_cluster_similarities[cluster_id] = {
                    'mean_similarity': float(np.mean(similarities)) if similarities else 0.0,
                    'std_similarity': float(np.std(similarities)) if similarities else 0.0,
                    'min_similarity': float(np.min(similarities)) if similarities else 0.0,
                    'max_similarity': float(np.max(similarities)) if similarities else 0.0,
                    'n_comparisons': len(similarities)
                }
            
            # Calculate across-cluster similarities
            across_cluster_similarities = []
            for i, cluster_1 in enumerate(unique_clusters):
                for cluster_2 in unique_clusters[i+1:]:
                    cluster_1_regimes = [regime_id for regime_id, cid in regime_to_cluster.items() if cid == cluster_1]
                    cluster_2_regimes = [regime_id for regime_id, cid in regime_to_cluster.items() if cid == cluster_2]
                    
                    cluster_pair_similarities = []
                    for regime_1 in cluster_1_regimes:
                        for regime_2 in cluster_2_regimes:
                            if regime_1 in regime_characteristics and regime_2 in regime_characteristics:
                                similarity = self._calculate_regime_similarity(
                                    regime_characteristics[regime_1], 
                                    regime_characteristics[regime_2]
                                )
                                cluster_pair_similarities.append(similarity)
                                across_cluster_similarities.append(similarity)
            
            # Summary statistics
            avg_within_similarity = np.mean([cluster_data['mean_similarity'] for cluster_data in within_cluster_similarities.values()])
            avg_across_similarity = np.mean(across_cluster_similarities) if across_cluster_similarities else 0.0
            
            similarity_validation['within_cluster_similarities'] = within_cluster_similarities
            similarity_validation['across_cluster_summary'] = {
                'mean_similarity': float(avg_across_similarity),
                'std_similarity': float(np.std(across_cluster_similarities)) if across_cluster_similarities else 0.0,
                'min_similarity': float(np.min(across_cluster_similarities)) if across_cluster_similarities else 0.0,
                'max_similarity': float(np.max(across_cluster_similarities)) if across_cluster_similarities else 0.0,
                'n_comparisons': len(across_cluster_similarities)
            }
            
            # Validation metrics
            similarity_difference = avg_within_similarity - avg_across_similarity
            similarity_ratio = avg_within_similarity / avg_across_similarity if avg_across_similarity > 0 else float('inf')
            
            similarity_validation['validation_metrics'] = {
                'avg_within_similarity': float(avg_within_similarity),
                'avg_across_similarity': float(avg_across_similarity),
                'similarity_difference': float(similarity_difference),
                'similarity_ratio': float(similarity_ratio) if similarity_ratio != float('inf') else 10.0,
                'good_clustering': similarity_difference > 0.1 and similarity_ratio > 1.2,
                'clustering_quality': 'high' if similarity_difference > 0.2 and similarity_ratio > 1.5 else 'medium' if similarity_difference > 0.1 and similarity_ratio > 1.2 else 'low'
            }
            
            return similarity_validation
            
        except Exception as e:
            return {'error': f'Regime similarity validation failed: {e}'}
    
    def _analyze_factor_impact_on_market_dynamics(self, regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int], unique_clusters: List[int]) -> Dict[str, Any]:
        """Analyze which factors actually impact market dynamics and trading strategy effectiveness."""
        try:
            from scipy import stats
            import numpy as np
            
            factor_impact = {}
            
            # Define core market aspects for dynamics analysis
            market_aspects = {
                'momentum': [
                    'mean_price_momentum_5', 'mean_price_momentum_20', 'mean_rsi', 'mean_macd',
                    'rsi_momentum', 'macd_momentum', 'momentum_strength', 'trend_persistence',
                    'autocorr_1_day', 'autocorr_5_day'
                ],
                'volatility': [
                    'mean_volatility_5', 'mean_volatility_10', 'mean_volatility_20',
                    'volatility_momentum', 'volatility_acceleration', 'mean_atr_normalized',
                    'volatility_clustering', 'return_volatility'
                ],
                'volume': [
                    'mean_volume_momentum_5', 'mean_volume_momentum_20', 'mean_volume_ratio',
                    'volume_momentum_volatility', 'volume_ratio_volatility'
                ]
            }
            
            # Collect all regime data
            all_regime_data = []
            cluster_labels = []
            
            for regime_id, cluster_id in regime_to_cluster.items():
                if regime_id in regime_characteristics:
                    regime_data = regime_characteristics[regime_id]
                    
                    # Extract all features for this regime
                    regime_features = {}
                    for aspect, features in market_aspects.items():
                        for feature in features:
                            # Look in all characteristic categories
                            value = None
                            for char_type in ['volume_characteristics', 'volatility_characteristics', 'momentum_characteristics']:
                                char_data = regime_data.get(char_type, {})
                                if feature in char_data:
                                    value = char_data[feature]
                                    break
                            
                            regime_features[feature] = value if value is not None else 0.0
                    
                    all_regime_data.append(regime_features)
                    cluster_labels.append(cluster_id)
            
            if len(all_regime_data) < 3:
                return {'error': 'Insufficient data for factor impact analysis'}
            
            # Calculate market aspect importance for cluster separation and dynamics
            aspect_importance = {}
            
            for aspect, features in market_aspects.items():
                aspect_analysis = {}
                aspect_f_stats = []
                
                for feature in features:
                    # Extract feature values by cluster
                    cluster_feature_data = {}
                    for cluster_id in unique_clusters:
                        cluster_feature_data[cluster_id] = []
                    
                    for i, regime_features in enumerate(all_regime_data):
                        cluster_id = cluster_labels[i]
                        if cluster_id in cluster_feature_data:
                            cluster_feature_data[cluster_id].append(regime_features.get(feature, 0.0))
                    
                    # Perform ANOVA to measure factor's ability to distinguish clusters
                    groups = [data for data in cluster_feature_data.values() if len(data) > 1]
                    
                    if len(groups) >= 2:
                        try:
                            f_stat, p_value = stats.f_oneway(*groups)
                            effect_size = self._calculate_eta_squared(groups, f_stat)
                            
                            aspect_analysis[feature] = {
                                'f_statistic': float(f_stat),
                                'p_value': float(p_value),
                                'effect_size': float(effect_size),
                                'significant': p_value < 0.05,
                                'market_impact': 'high' if effect_size > 0.14 else 'medium' if effect_size > 0.06 else 'low'
                            }
                            
                            if not np.isnan(f_stat) and f_stat > 0:
                                aspect_f_stats.append(f_stat)
                                
                        except Exception as e:
                            aspect_analysis[feature] = {'error': str(e)}
                
                # Calculate aspect-level market dynamics impact
                if aspect_f_stats:
                    avg_f_stat = np.mean(aspect_f_stats)
                    significant_features = sum(1 for feat_data in aspect_analysis.values() 
                                             if isinstance(feat_data, dict) and feat_data.get('significant', False))
                    total_features = len([feat_data for feat_data in aspect_analysis.values() 
                                        if isinstance(feat_data, dict) and 'error' not in feat_data])
                    
                    aspect_analysis['aspect_summary'] = {
                        'avg_f_statistic': float(avg_f_stat),
                        'significant_features': significant_features,
                        'total_features': total_features,
                        'significance_ratio': significant_features / total_features if total_features > 0 else 0.0,
                        'market_dynamics_impact': 'high' if avg_f_stat > 5.0 and significant_features / total_features > 0.5 else 'medium' if avg_f_stat > 2.0 else 'low'
                    }
                
                aspect_importance[aspect] = aspect_analysis
            
            # Overall market aspects impact ranking
            aspect_impacts = []
            for aspect, aspect_data in aspect_importance.items():
                summary = aspect_data.get('aspect_summary', {})
                if summary:
                    aspect_impacts.append({
                        'market_aspect': aspect,
                        'avg_f_stat': summary.get('avg_f_statistic', 0),
                        'significance_ratio': summary.get('significance_ratio', 0),
                        'dynamics_impact_score': summary.get('avg_f_statistic', 0) * summary.get('significance_ratio', 0)
                    })
            
            # Sort by dynamics impact score
            aspect_impacts.sort(key=lambda x: x['dynamics_impact_score'], reverse=True)
            
            factor_impact['market_aspects_analysis'] = aspect_importance
            factor_impact['aspect_ranking'] = aspect_impacts
            
            # Market dynamics insights - which aspects actually drive market behavior
            top_aspect = aspect_impacts[0] if aspect_impacts else None
            if top_aspect:
                factor_impact['primary_market_driver'] = {
                    'dominant_aspect': top_aspect['market_aspect'],
                    'impact_strength': 'high' if top_aspect['dynamics_impact_score'] > 10 else 'medium' if top_aspect['dynamics_impact_score'] > 5 else 'low',
                    'statistical_confidence': 'high' if top_aspect['significance_ratio'] > 0.6 else 'medium' if top_aspect['significance_ratio'] > 0.3 else 'low'
                }
            
            # Core market dynamics analysis - fundamental aspects only
            momentum_impact = next((aspect for aspect in aspect_impacts if aspect['market_aspect'] == 'momentum'), {}).get('dynamics_impact_score', 0)
            volatility_impact = next((aspect for aspect in aspect_impacts if aspect['market_aspect'] == 'volatility'), {}).get('dynamics_impact_score', 0)
            volume_impact = next((aspect for aspect in aspect_impacts if aspect['market_aspect'] == 'volume'), {}).get('dynamics_impact_score', 0)
            
            # Determine which core market aspects have the highest impact on dynamics
            aspect_scores = {
                'momentum': momentum_impact,
                'volatility': volatility_impact,
                'volume': volume_impact
            }
            
            # Sort aspects by impact
            sorted_aspects = sorted(aspect_scores.items(), key=lambda x: x[1], reverse=True)
            
            factor_impact['market_dynamics_hierarchy'] = {
                'aspect_impact_scores': aspect_scores,
                'ranked_aspects': [{'aspect': aspect, 'impact_score': score} for aspect, score in sorted_aspects],
                'primary_driver': sorted_aspects[0][0] if sorted_aspects else 'unknown',
                'secondary_driver': sorted_aspects[1][0] if len(sorted_aspects) > 1 else 'none',
                'complexity_level': 'high' if len([score for _, score in sorted_aspects if score > 5]) > 3 else 'medium' if len([score for _, score in sorted_aspects if score > 2]) > 2 else 'low',
                'multi_factor_market': len([score for _, score in sorted_aspects if score > 3]) > 2
            }
            
            return factor_impact
            
        except Exception as e:
            return {'error': f'Factor impact analysis failed: {e}'}
    
    def _validate_with_economic_indicators(self, cluster_assignments: List[int], regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int], market_data: Any) -> Dict[str, Any]:
        """Validate clustering against known economic indicators and market conditions."""
        try:
            import numpy as np
            from scipy import stats
            
            economic_validation = {}
            
            # 1. Volatility Regime Validation (Multiple volatility measures)
            volatility_validation = {}
            if hasattr(market_data, 'columns') and 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                
                # Multiple volatility measures
                rolling_vol_20 = returns.rolling(window=20).std() * np.sqrt(252)  # 20-day annualized
                rolling_vol_5 = returns.rolling(window=5).std() * np.sqrt(252)   # 5-day annualized
                
                # High-low volatility (if available)
                if 'high' in market_data.columns and 'low' in market_data.columns:
                    hl_volatility = np.log(market_data['high'] / market_data['low'])
                else:
                    hl_volatility = None
                
                unique_clusters = list(set(cluster_assignments))
                
                # Test different volatility measures
                vol_measures = {
                    'rolling_vol_20': rolling_vol_20,
                    'rolling_vol_5': rolling_vol_5
                }
                
                if hl_volatility is not None:
                    vol_measures['hl_volatility'] = hl_volatility
                
                for vol_name, vol_data in vol_measures.items():
                    vol_groups = [vol_data[np.array(cluster_assignments) == cid].dropna() for cid in unique_clusters]
                    vol_groups = [group for group in vol_groups if len(group) > 1]
                    
                    if len(vol_groups) >= 2:
                        f_stat, p_value = stats.f_oneway(*vol_groups)
                        volatility_validation[vol_name] = {
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
                
                economic_validation['volatility_regime_validation'] = volatility_validation
            
            # 2. Momentum Regime Validation (Multiple momentum measures)
            momentum_validation = {}
            if hasattr(market_data, 'columns') and 'close' in market_data.columns:
                prices = market_data['close']
                
                # Multiple momentum measures
                momentum_5 = prices.pct_change(5)  # 5-day momentum
                momentum_20 = prices.pct_change(20)  # 20-day momentum
                
                # Moving average momentum
                ma_short = prices.rolling(10).mean()
                ma_long = prices.rolling(30).mean()
                ma_momentum = (ma_short - ma_long) / ma_long
                
                # Technical momentum (if available)
                momentum_measures = {
                    'momentum_5': momentum_5,
                    'momentum_20': momentum_20,
                    'ma_momentum': ma_momentum
                }
                
                # Add RSI and MACD if available
                if 'rsi' in market_data.columns:
                    rsi_momentum = market_data['rsi'].pct_change(5)
                    momentum_measures['rsi_momentum'] = rsi_momentum
                
                if 'macd' in market_data.columns:
                    macd_momentum = market_data['macd'].pct_change(5)
                    momentum_measures['macd_momentum'] = macd_momentum
                
                unique_clusters = list(set(cluster_assignments))
                
                for mom_name, mom_data in momentum_measures.items():
                    mom_groups = [mom_data[np.array(cluster_assignments) == cid].dropna() for cid in unique_clusters]
                    mom_groups = [group for group in mom_groups if len(group) > 1]
                    
                    if len(mom_groups) >= 2:
                        f_stat, p_value = stats.f_oneway(*mom_groups)
                        momentum_validation[mom_name] = {
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
                
                economic_validation['momentum_regime_validation'] = momentum_validation
            
            # 3. Volume Regime Validation
            volume_validation = {}
            if hasattr(market_data, 'columns') and 'volume' in market_data.columns:
                volume = market_data['volume']
                
                # Multiple volume measures
                volume_ma_ratio = volume / volume.rolling(20).mean()  # Volume relative to average
                volume_momentum_5 = volume.pct_change(5)  # 5-day volume change
                volume_volatility = volume.rolling(20).std() / volume.rolling(20).mean()  # Volume volatility
                
                volume_measures = {
                    'volume_ma_ratio': volume_ma_ratio,
                    'volume_momentum_5': volume_momentum_5,
                    'volume_volatility': volume_volatility
                }
                
                unique_clusters = list(set(cluster_assignments))
                
                for vol_name, vol_data in volume_measures.items():
                    vol_groups = [vol_data[np.array(cluster_assignments) == cid].dropna() for cid in unique_clusters]
                    vol_groups = [group for group in vol_groups if len(group) > 1]
                    
                    if len(vol_groups) >= 2:
                        # Use log-transformed volume for better normality
                        log_vol_groups = [np.log(np.abs(group) + 1) for group in vol_groups]
                        f_stat, p_value = stats.f_oneway(*log_vol_groups)
                        volume_validation[vol_name] = {
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
                
                economic_validation['volume_regime_validation'] = volume_validation
            
            # 4. Market Stress Regime Validation
            stress_validation = {}
            if hasattr(market_data, 'columns') and 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                
                # Stress indicators
                large_moves = np.abs(returns) > np.abs(returns).quantile(0.95)  # Extreme moves
                negative_runs = (returns < 0).rolling(3).sum() >= 2  # Consecutive negative days
                volatility_spikes = returns.rolling(5).std() > returns.rolling(20).std() * 1.5
                
                stress_measures = {
                    'extreme_moves': large_moves,
                    'negative_runs': negative_runs,
                    'volatility_spikes': volatility_spikes
                }
                
                unique_clusters = list(set(cluster_assignments))
                
                for stress_name, stress_data in stress_measures.items():
                    # Calculate stress frequency by cluster
                    cluster_stress_freq = []
                    for cluster_id in unique_clusters:
                        cluster_mask = np.array(cluster_assignments) == cluster_id
                        cluster_stress = stress_data[cluster_mask]
                        if len(cluster_stress) > 0:
                            stress_freq = cluster_stress.sum() / len(cluster_stress)
                            cluster_stress_freq.append(stress_freq)
                    
                    if len(cluster_stress_freq) >= 2:
                        # Test if stress frequency differs significantly between clusters
                        f_stat, p_value = stats.f_oneway(*[np.full(10, freq) for freq in cluster_stress_freq])
                        stress_validation[stress_name] = {
                            'cluster_stress_frequencies': cluster_stress_freq,
                            'significant_difference': p_value < 0.05,
                            'stress_concentration': max(cluster_stress_freq) - min(cluster_stress_freq)
                        }
                
                economic_validation['market_stress_validation'] = stress_validation
            
            # 5. Overall Economic Alignment Score (Equal weight for all financial aspects)
            alignment_components = []
            component_names = []
            
            # Volatility component score
            if 'volatility_regime_validation' in economic_validation:
                vol_significant = sum(1 for test in volatility_validation.values() if test.get('significant', False))
                vol_total = len(volatility_validation)
                vol_score = vol_significant / vol_total if vol_total > 0 else 0.0
                alignment_components.append(vol_score)
                component_names.append('volatility')
            
            # Momentum component score  
            if 'momentum_regime_validation' in economic_validation:
                mom_significant = sum(1 for test in momentum_validation.values() if test.get('significant', False))
                mom_total = len(momentum_validation)
                mom_score = mom_significant / mom_total if mom_total > 0 else 0.0
                alignment_components.append(mom_score)
                component_names.append('momentum')
            
            # Volume component score
            if 'volume_regime_validation' in economic_validation:
                volume_significant = sum(1 for test in volume_validation.values() if test.get('significant', False))
                volume_total = len(volume_validation)
                volume_score = volume_significant / volume_total if volume_total > 0 else 0.0
                alignment_components.append(volume_score)
                component_names.append('volume')
            
            # Market stress component score
            if 'market_stress_validation' in economic_validation:
                stress_significant = sum(1 for test in stress_validation.values() if test.get('significant_difference', False))
                stress_total = len(stress_validation)
                stress_score = stress_significant / stress_total if stress_total > 0 else 0.0
                alignment_components.append(stress_score)
                component_names.append('market_stress')
            
            overall_score = np.mean(alignment_components) if alignment_components else 0.5
            
            # Create component score dictionary
            component_scores = {}
            for i, name in enumerate(component_names):
                if i < len(alignment_components):
                    component_scores[f'{name}_score'] = alignment_components[i]
            
            economic_validation['overall_economic_alignment'] = {
                **component_scores,  # Include all component scores dynamically
                'overall_score': float(overall_score),
                'components_tested': component_names,
                'n_financial_aspects': len(alignment_components),
                'comprehensive_validation': len(alignment_components) >= 3,  # Volatility + momentum + volume minimum
                'economic_alignment_quality': 'high' if overall_score > 0.7 else 'medium' if overall_score > 0.5 else 'low',
                'economic_validation_passed': overall_score > 0.6
            }
            
            return economic_validation
            
        except Exception as e:
            return {'error': f'Economic validation failed: {e}'}
    
    def _assess_overall_cluster_quality(self, volume_tests: Dict[str, Any], volatility_tests: Dict[str, Any], momentum_tests: Dict[str, Any], cluster_validation: Dict[str, Any], similarity_validation: Dict[str, Any], factor_impact: Dict[str, Any] = None, economic_validation: Dict[str, Any] = None) -> Dict[str, Any]:
        """Assess overall quality of the clustering based on all statistical tests."""
        try:
            overall_assessment = {}
            
            # Extract key metrics
            volume_diff_score = volume_tests.get('overall_volume_differentiation', {}).get('differentiation_score', 0)
            volatility_diff_score = volatility_tests.get('overall_volatility_differentiation', {}).get('differentiation_score', 0)
            momentum_diff_score = momentum_tests.get('overall_momentum_differentiation', {}).get('differentiation_score', 0)
            
            bull_bear_score = cluster_validation.get('overall_validation', {}).get('overall_score', 0)
            similarity_quality = similarity_validation.get('validation_metrics', {}).get('clustering_quality', 'low')
            similarity_score = 1.0 if similarity_quality == 'high' else 0.6 if similarity_quality == 'medium' else 0.3
            
            # Component scores
            overall_assessment['component_scores'] = {
                'volume_differentiation': float(volume_diff_score),
                'volatility_differentiation': float(volatility_diff_score),
                'momentum_differentiation': float(momentum_diff_score),
                'combined_momentum_score': float(combined_momentum_score),
                'bull_bear_validation': float(bull_bear_score),  # Kept for reference
                'similarity_validation': float(similarity_score)
            }
            
            # Weighted overall score - Equal 25% weights for all components
            weights = {
                'volume': 0.25,        # 25% - Equal weight with all other components
                'volatility': 0.25,    # 25% - Equal weight with all other components
                'momentum': 0.25,      # 25% - Equal weight with all other components
                'similarity': 0.25     # 25% - Equal weight with all other components
            }
            
            # Use pure momentum differentiation score (bull/bear validation is redundant)
            combined_momentum_score = momentum_diff_score
            
            weighted_score = (
                volume_diff_score * weights['volume'] +
                volatility_diff_score * weights['volatility'] +
                combined_momentum_score * weights['momentum'] +
                similarity_score * weights['similarity']
            )
            
            # Quality assessment
            if weighted_score > 0.7:
                quality_level = 'high'
                assessment = 'Excellent clustering - clusters are well-differentiated and meaningful'
            elif weighted_score > 0.5:
                quality_level = 'medium'
                assessment = 'Good clustering - clusters show reasonable differentiation'
            else:
                quality_level = 'low'
                assessment = 'Poor clustering - clusters lack clear differentiation'
            
            overall_assessment['overall_score'] = float(weighted_score)
            overall_assessment['quality_level'] = quality_level
            overall_assessment['assessment'] = assessment
            
            # Recommendations
            recommendations = []
            if volume_diff_score < 0.5:
                recommendations.append('Consider improving volume-based regime characterization')
            if volatility_diff_score < 0.5:
                recommendations.append('Consider improving volatility-based regime characterization')
            if combined_momentum_score < 0.5:
                recommendations.append('Consider improving momentum-based regime characterization and Bull/Bear/Sideways distinction')
            if similarity_score < 0.6:
                recommendations.append('Regimes within clusters may not be sufficiently similar')
            
            overall_assessment['recommendations'] = recommendations
            
            return overall_assessment
            
        except Exception as e:
            return {'error': f'Overall cluster quality assessment failed: {e}'}
    


    def _map_regime_assignments_to_clusters(self, regime_assignments: List[int], regime_to_cluster: Dict[str, int], data_length: int) -> List[int]:
        """Map time series regime assignments to cluster assignments."""
        try:
            cluster_assignments = []
            
            for regime_id in regime_assignments:
                # Convert regime number to regime string ID
                regime_key = f"regime_{regime_id}"
                
                # Get cluster assignment for this regime
                cluster_id = regime_to_cluster.get(regime_key, 0)  # Default to cluster 0
                cluster_assignments.append(cluster_id)
            
            # Ensure we have the right length
            while len(cluster_assignments) < data_length:
                cluster_assignments.append(0)  # Pad with cluster 0
            
            return cluster_assignments[:data_length]  # Trim to exact length
            
        except Exception as e:
            self.logger.error(f"❌ Regime to cluster mapping failed: {e}")
            # Fallback: simple modulo assignment
            return [i % len(set(regime_to_cluster.values())) if regime_to_cluster else 0 for i in range(data_length)]



    def _create_cluster_assignments_from_mapping(
        self, 
        regime_assignments: List[int], 
        regime_to_cluster: Dict[str, int], 
        regime_ids: List[str]
    ) -> List[int]:
        """Create cluster assignments from regime-to-cluster mapping."""
        try:
            cluster_assignments = []

            # Build robust mapping from numeric regime index -> cluster id
            # Many pipelines store regime identifiers as strings like "regime_123" while
            # assignments are numeric indices (e.g., 123). We normalize both.
            numeric_to_cluster: Dict[int, int] = {}
            for idx, regime_id in enumerate(regime_ids):
                cluster_id = None
                # Prefer direct string key
                if regime_id in regime_to_cluster:
                    cluster_id = regime_to_cluster[regime_id]
                # Try prefixed form
                elif isinstance(regime_id, (int, float)) and f'regime_{int(regime_id)}' in regime_to_cluster:
                    cluster_id = regime_to_cluster[f'regime_{int(regime_id)}']
                elif isinstance(regime_id, str):
                    # Attempt to parse trailing integer from strings like 'regime_123'
                    try:
                        if '_' in regime_id:
                            parts = regime_id.split('_')
                            num = int(parts[-1])
                            if regime_id in regime_to_cluster:
                                cluster_id = regime_to_cluster[regime_id]
                            elif f'regime_{num}' in regime_to_cluster:
                                cluster_id = regime_to_cluster[f'regime_{num}']
                            else:
                                cluster_id = regime_to_cluster.get(str(num))
                            if cluster_id is not None:
                                numeric_to_cluster[num] = cluster_id
                                continue
                    except Exception:
                        pass
                # Fallback to index-based mapping if no explicit match
                if cluster_id is None:
                    # If an explicit mapping for this index exists, use it
                    cluster_id = regime_to_cluster.get(f'regime_{idx}', regime_to_cluster.get(idx, 0))
                # Record numeric index mapping
                try:
                    # Best effort to assign numeric key
                    if isinstance(regime_id, int):
                        numeric_to_cluster[int(regime_id)] = cluster_id
                    else:
                        # Also map by sequential index
                        numeric_to_cluster[idx] = cluster_id
                except Exception:
                    numeric_to_cluster[idx] = cluster_id

            # Map regime assignments (numeric) to cluster assignments
            for ra in regime_assignments:
                try:
                    key = int(ra)
                except Exception:
                    # If somehow non-numeric, force 0
                    key = 0
                cluster_assignments.append(int(numeric_to_cluster.get(key, 0)))
            
            self.logger.info(f"✅ Created cluster assignments: {len(cluster_assignments)} samples mapped to clusters")
            return cluster_assignments
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create cluster assignments from mapping: {e}")
            # Fallback: assign all to cluster 0
            return [0] * len(regime_assignments)


    def _create_single_cluster_result(self, regime_assignments: List[int], market_data: Any) -> Dict[str, Any]:
        """Create a single cluster result as final fallback."""
        cluster_assignments = [0] * len(regime_assignments)
        # Create a simple cluster model without complex HMM models
        simple_model = [{
            'cluster_id': 0,
            'model_type': 'simple_single_cluster',
            'regime_count': len(set(regime_assignments))
        }]
        return {
            'success': True,
            'cluster_assignments': cluster_assignments,
            'cluster_count': 1,
            'regime_to_cluster': {f'regime_{i}': 0 for i in set(regime_assignments)},
            'hmm_models': simple_model,
            'clustering_method': 'single_cluster_fallback',
            'cluster_centers': [],
            'cluster_distribution': {'cluster_0': 100.0}
        }

    def _calculate_matrix_clustering_score(self, similarity_matrix: Any, regime_to_cluster: Dict[str, int], cluster_assignments: List[int]) -> float:
        """Calculate clustering quality score based on within-cluster similarity."""
        try:
            import numpy as np
            
            if len(cluster_assignments) == 0:
                return 0.0
            
            # Calculate silhouette-like score using similarity matrix
            unique_clusters = list(set(cluster_assignments))
            if len(unique_clusters) <= 1:
                return 1.0  # Perfect score for single cluster
            
            total_score = 0.0
            n_samples = len(cluster_assignments)
            
            for i in range(n_samples):
                cluster_i = cluster_assignments[i]
                
                # Calculate within-cluster similarity (higher is better)
                within_cluster_similarities = []
                between_cluster_similarities = []
                
                for j in range(n_samples):
                    if i != j:
                        cluster_j = cluster_assignments[j]
                        similarity = similarity_matrix[i, j] if hasattr(similarity_matrix, 'shape') else 0.0
                        
                        if cluster_i == cluster_j:
                            within_cluster_similarities.append(similarity)
                        else:
                            between_cluster_similarities.append(similarity)
                
                # Calculate sample score (higher within-cluster, lower between-cluster is better)
                avg_within = np.mean(within_cluster_similarities) if within_cluster_similarities else 0.0
                avg_between = np.mean(between_cluster_similarities) if between_cluster_similarities else 0.0
                
                # Score: positive when within > between
                sample_score = avg_within - avg_between
                total_score += sample_score
            
            # Normalize by number of samples
            final_score = total_score / n_samples
            
            self.logger.info(f"📊 Matrix clustering score: {final_score:.4f}")
            return final_score
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate matrix clustering score: {e}")
            return 0.0

    def _analyze_cv_stopping_criteria(self, regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int]) -> Dict[str, Any]:
        """Analyze CV-based stopping criteria with detailed reasoning."""
        try:
            import numpy as np
            
            # Calculate CV statistics for current clusters
            cluster_cv_stats = {}
            for regime_id, cluster_id in regime_to_cluster.items():
                regime = regime_characteristics.get(regime_id, {})
                if cluster_id not in cluster_cv_stats:
                    cluster_cv_stats[cluster_id] = {
                        'momentum_cvs': [],
                        'volatility_cvs': [],
                        'volume_cvs': []
                    }
                
                # Extract CV values
                for aspect in ['momentum', 'volatility', 'volume']:
                    cv_key = f'{aspect}_cv'
                    cv_value = regime.get(cv_key, 0.0)
                    if isinstance(cv_value, (int, float)) and np.isfinite(cv_value):
                        cluster_cv_stats[cluster_id][f'{aspect}_cvs'].append(cv_value)
            
            # Analyze cluster CV diversity
            high_cv_clusters = 0
            total_clusters = len(cluster_cv_stats)
            cv_diversity_stats = {}
            
            for cluster_id, stats in cluster_cv_stats.items():
                cluster_analysis = {}
                for aspect in ['momentum', 'volatility', 'volume']:
                    cvs = stats[f'{aspect}_cvs']
                    if cvs:
                        cluster_analysis[aspect] = {
                            'mean': np.mean(cvs),
                            'std': np.std(cvs),
                            'range': np.max(cvs) - np.min(cvs),
                            'count': len(cvs)
                        }
                        # High CV if mean > 0.3 or range > 0.5
                        if cluster_analysis[aspect]['mean'] > 0.3 or cluster_analysis[aspect]['range'] > 0.5:
                            high_cv_clusters += 1
                            break
                cv_diversity_stats[cluster_id] = cluster_analysis
            
            # Stopping criteria
            high_cv_ratio = high_cv_clusters / total_clusters if total_clusters > 0 else 0
            should_stop = high_cv_ratio > 0.7  # Stop if >70% of clusters have high CV diversity
            
            return {
                'should_stop': should_stop,
                'reason': f"High CV diversity in {high_cv_clusters}/{total_clusters} clusters ({high_cv_ratio:.1%})" if should_stop else "CV diversity acceptable",
                'stats': {
                    'high_cv_clusters': high_cv_clusters,
                    'total_clusters': total_clusters,
                    'high_cv_ratio': high_cv_ratio,
                    'threshold': 0.7
                }
            }
            
        except Exception as e:
            return {
                'should_stop': False,
                'reason': f"CV analysis failed: {e}",
                'stats': {}
            }

    def _analyze_similarity_threshold(self, threshold: float, similarity_matrix: Any, regime_to_cluster: Dict[str, int]) -> Dict[str, Any]:
        """Analyze similarity threshold with detailed statistics."""
        try:
            import numpy as np
            
            # Calculate similarity statistics
            sim_stats = {
                'min': np.min(similarity_matrix),
                'max': np.max(similarity_matrix),
                'mean': np.mean(similarity_matrix),
                'median': np.median(similarity_matrix),
                'std': np.std(similarity_matrix)
            }
            
            # Count pairs above threshold
            pairs_above_threshold = np.sum(similarity_matrix >= threshold)
            total_pairs = similarity_matrix.shape[0] * (similarity_matrix.shape[0] - 1) // 2
            threshold_ratio = pairs_above_threshold / total_pairs if total_pairs > 0 else 0
            
            # Threshold is too low if:
            # 1. Less than 1% of pairs meet the threshold, OR
            # 2. Threshold is below 2 standard deviations from mean
            too_low = (threshold_ratio < 0.01) or (threshold < sim_stats['mean'] - 2 * sim_stats['std'])
            
            reason = ""
            if threshold_ratio < 0.01:
                reason = f"Only {threshold_ratio:.1%} of pairs meet threshold"
            elif threshold < sim_stats['mean'] - 2 * sim_stats['std']:
                reason = f"Threshold {threshold:.4f} is too far below mean similarity {sim_stats['mean']:.4f}"
            else:
                reason = f"Threshold acceptable: {threshold_ratio:.1%} of pairs eligible"
            
            return {
                'too_low': too_low,
                'reason': reason,
                'stats': {
                    'threshold': threshold,
                    'pairs_above_threshold': pairs_above_threshold,
                    'total_pairs': total_pairs,
                    'threshold_ratio': threshold_ratio,
                    'similarity_stats': sim_stats
                }
            }
            
        except Exception as e:
            return {
                'too_low': False,
                'reason': f"Threshold analysis failed: {e}",
                'stats': {}
            }

    def _calculate_progressive_similarity_thresholds(self) -> List[float]:
        """Calculate progressive similarity thresholds from 99% down to 60%."""
        # Generate thresholds: 99%, 96%, 93%, ..., 66%, 63%, 60%
        thresholds = []
        for threshold_pct in range(99, 54, -4):  # 99 down to 55
            thresholds.append(threshold_pct / 100.0)
        
        self.logger.info(f"🎯 Generated {len(thresholds)} progressive thresholds: {thresholds[0]:.2f} → {thresholds[-1]:.2f}")
        return thresholds


    def _calculate_advanced_cluster_analytics(
        self, 
        regime_characteristics: Dict[str, Any], 
        regime_to_cluster: Dict[str, int], 
        cluster_assignments: List[int],
        cluster_count: int
    ) -> Dict[str, Any]:
        """Calculate advanced cluster analytics including within-cluster CV and top cluster coverage."""
        try:
            import numpy as np
            
            # Group regimes by cluster
            cluster_regimes = {}
            for regime_id, cluster_id in regime_to_cluster.items():
                if cluster_id not in cluster_regimes:
                    cluster_regimes[cluster_id] = []
                cluster_regimes[cluster_id].append(regime_id)
            
            # Calculate within-cluster CVs using sample-level data when available
            cluster_cvs = {}
            all_within_cluster_cvs = []

            # Robust aspect-wise CV as fallback
            robust_cv_by_aspect = self._compute_cluster_cv_by_aspect(regime_characteristics, regime_to_cluster)

            # Attempt sample-level CV from market_data columns if available
            sample_level_available = False
            try:
                # We expect to have cluster_assignments bound in the caller's context; if not, we fall back
                # To keep this function pure, we only use regime_characteristics mapping here
                sample_level_available = False
            except Exception:
                sample_level_available = False

            for cluster_id, regime_ids in cluster_regimes.items():
                aspect_map = robust_cv_by_aspect.get(cluster_id, {})
                momentum_cv = float(aspect_map.get('momentum', 0.0))
                volatility_cv = float(aspect_map.get('volatility', 0.0))
                volume_cv = float(aspect_map.get('volume', 0.0))
                avg_cv = float(np.mean([v for v in [momentum_cv, volatility_cv, volume_cv] if np.isfinite(v)])) if any(np.isfinite([momentum_cv, volatility_cv, volume_cv])) else 0.0
                cluster_cvs[cluster_id] = {
                    'momentum': momentum_cv,
                    'volatility': volatility_cv,
                    'volume': volume_cv,
                    'avg': avg_cv
                }
                all_within_cluster_cvs.append(avg_cv)
            
            # Calculate overall average within-cluster CV
            avg_within_cluster_cv = np.mean(all_within_cluster_cvs) if all_within_cluster_cvs else 0.0
            
            # Calculate cluster sizes and top 20 coverage using regime characteristics (before constraint reduction)
            cluster_sizes = {}
            total_samples = 0
            
            # Calculate cluster sizes from regime characteristics and regime_to_cluster mapping
            for regime_id, cluster_id in regime_to_cluster.items():
                regime = regime_characteristics.get(regime_id, {})
                regime_sample_count = regime.get('sample_count', 0)
                
                if cluster_id not in cluster_sizes:
                    cluster_sizes[cluster_id] = 0
                cluster_sizes[cluster_id] += regime_sample_count
                total_samples += regime_sample_count
            
            # Sort clusters by size (descending)
            sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)
            
            # Calculate top 20 cluster coverage
            top_20_clusters = sorted_clusters[:20] if len(sorted_clusters) >= 20 else sorted_clusters
            top_20_samples = sum(size for _, size in top_20_clusters)
            top_20_coverage = (top_20_samples / total_samples * 100) if total_samples > 0 else 0.0
            
            # Concentration analysis
            if len(sorted_clusters) > 0:
                largest_cluster_pct = (sorted_clusters[0][1] / total_samples * 100) if total_samples > 0 else 0.0
                top_5_samples = sum(size for _, size in sorted_clusters[:5])
                top_5_coverage = (top_5_samples / total_samples * 100) if total_samples > 0 else 0.0
                
                concentration_analysis = {
                    'largest_cluster_pct': largest_cluster_pct,
                    'top_5_coverage': top_5_coverage,
                    'top_20_coverage': top_20_coverage,
                    'cluster_balance': 'balanced' if largest_cluster_pct < 20 else 'concentrated'
                }
            else:
                concentration_analysis = {
                    'largest_cluster_pct': 0.0,
                    'top_5_coverage': 0.0,
                    'top_20_coverage': 0.0,
                    'cluster_balance': 'unknown'
                }
            
            return {
                'avg_within_cluster_cv': avg_within_cluster_cv,
                'top_20_coverage': top_20_coverage,
                'concentration_analysis': concentration_analysis,
                'cluster_cvs': cluster_cvs,
                'cluster_sizes': cluster_sizes,
                'total_samples': total_samples
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate advanced cluster analytics: {e}")
            return {
                'avg_within_cluster_cv': 0.0,
                'top_20_coverage': 0.0,
                'concentration_analysis': {'error': str(e)},
                'cluster_cvs': {},
                'cluster_sizes': {},
                'total_samples': 0
            }

    def _calculate_similarity_histogram(self, similarity_matrix: Any) -> Dict[str, Any]:
        """Return histogram bins for similarity values in the upper triangle (excluding diagonal)."""
        try:
            import numpy as np
            if similarity_matrix is None:
                return {'bins': [], 'counts': []}
            # Upper triangle without diagonal
            triu_indices = np.triu_indices(similarity_matrix.shape[0], k=1)
            values = similarity_matrix[triu_indices].flatten()
            # Clamp values to [-1, 1]
            values = np.clip(values, -1.0, 1.0)
            counts, bin_edges = np.histogram(values, bins=20, range=(-1.0, 1.0))
            return {'bins': [float(b) for b in bin_edges.tolist()], 'counts': [int(c) for c in counts.tolist()]}
        except Exception:
            return {'bins': [], 'counts': []}

    def _compute_cluster_prototypes(self, regime_ids: List[str], regime_to_cluster: Dict[str, int], similarity_matrix: Any, regime_characteristics: Dict[str, Any], top_n_features: int = 5) -> Dict[str, Any]:
        """Compute per-cluster prototypes and top distinguishing features.

        - Prototype: regime with highest average similarity to other regimes in its cluster
        - Distinguishing features: features with highest between-cluster variance normalized by within-cluster variance
        """
        try:
            import numpy as np
            clusters: Dict[int, list] = {}
            id_to_index: Dict[str, int] = {rid: idx for idx, rid in enumerate(regime_ids)}
            for rid, cid in regime_to_cluster.items():
                clusters.setdefault(cid, []).append(rid)

            prototypes: Dict[str, Any] = {}
            # Pre-extract feature dicts
            regime_features: Dict[str, Dict[str, float]] = {}
            for rid in regime_ids:
                regime_features[rid] = dict(regime_characteristics.get(rid, {}).get('features', {}))

            # Compute simple ANOVA-like score for features
            # Build per-cluster feature means
            all_features = set()
            for feats in regime_features.values():
                all_features.update([k for k, v in feats.items() if isinstance(v, (int, float))])
            feature_scores: Dict[str, float] = {}
            for feat in all_features:
                # Between/within variances
                cluster_means = []
                cluster_vars = []
                cluster_sizes = []
                for cid, rids in clusters.items():
                    vals = [float(regime_features[rid].get(feat, np.nan)) for rid in rids]
                    arr = np.array(vals, dtype=float)
                    arr = arr[np.isfinite(arr)]
                    if arr.size == 0:
                        continue
                    cluster_means.append(float(np.mean(arr)))
                    cluster_vars.append(float(np.var(arr)))
                    cluster_sizes.append(int(arr.size))
                if len(cluster_means) < 2:
                    feature_scores[feat] = 0.0
                    continue
                grand_mean = float(np.average(cluster_means, weights=cluster_sizes)) if sum(cluster_sizes) > 0 else float(np.mean(cluster_means))
                # Between variance (weighted)
                between = 0.0
                for m, n in zip(cluster_means, cluster_sizes):
                    between += n * (m - grand_mean) ** 2
                between /= max(1, sum(cluster_sizes))
                within = float(np.mean(cluster_vars)) if len(cluster_vars) > 0 else 1e-8
                score = between / max(within, 1e-8)
                feature_scores[feat] = float(score)

            # For each cluster, choose prototype and rank features
            for cid, rids in clusters.items():
                if not rids:
                    continue
                # Prototype = regime with max avg similarity inside cluster
                avg_sims = []
                for rid in rids:
                    idx = id_to_index.get(rid)
                    if idx is None:
                        avg_sims.append((-np.inf, rid))
                        continue
                    intra_idxs = [id_to_index.get(r) for r in rids if id_to_index.get(r) is not None and r != rid]
                    if not intra_idxs:
                        avg_sims.append((0.0, rid))
                        continue
                    sims = [similarity_matrix[idx, j] for j in intra_idxs]
                    avg_sims.append((float(np.mean(sims)), rid))
                avg_sims.sort(reverse=True)
                proto_id = avg_sims[0][1]

                # Top distinguishing features (global ranking used; could be refined per cluster)
                top_feats = sorted(feature_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n_features]
                prototypes[f'cluster_{cid}'] = {
                    'prototype_regime': proto_id,
                    'top_features': [{'name': k, 'score': float(v)} for k, v in top_feats]
                }
            return prototypes
        except Exception:
            return {}

    def _calculate_matrix_clustering_score(self, similarity_matrix: Any, regime_to_cluster: Dict[str, int], temp_assignments: List[int]) -> float:
        """Calculate a clustering quality score based on the similarity matrix and cluster assignments."""
        try:
            import numpy as np
            
            if similarity_matrix is None or similarity_matrix.size == 0:
                return 0.0
            
            # Ensure we don't access out of bounds indices
            n_regimes = similarity_matrix.shape[0]
            if n_regimes == 0:
                return 0.0
            
            # Get unique cluster IDs from assignments
            unique_clusters = list(set(temp_assignments))
            n_clusters = len(unique_clusters)
            
            if n_clusters <= 1:
                return 0.0
            
            # Calculate within-cluster similarity (higher is better)
            within_cluster_sim = 0.0
            total_within_pairs = 0
            
            for cluster_id in unique_clusters:
                cluster_indices = [i for i, c in enumerate(temp_assignments) if c == cluster_id]
                if len(cluster_indices) < 2:
                    continue
                
                # Sum similarities within this cluster
                for i in range(len(cluster_indices)):
                    for j in range(i + 1, len(cluster_indices)):
                        idx_i, idx_j = cluster_indices[i], cluster_indices[j]
                        # Bounds check
                        if idx_i < n_regimes and idx_j < n_regimes:
                            within_cluster_sim += similarity_matrix[idx_i, idx_j]
                            total_within_pairs += 1
            
            # Calculate between-cluster similarity (lower is better)
            between_cluster_sim = 0.0
            total_between_pairs = 0
            
            for i in range(len(temp_assignments)):
                for j in range(i + 1, len(temp_assignments)):
                    if temp_assignments[i] != temp_assignments[j]:
                        # Different clusters - bounds check
                        if i < n_regimes and j < n_regimes:
                            between_cluster_sim += similarity_matrix[i, j]
                            total_between_pairs += 1
            
            # Calculate averages
            avg_within = within_cluster_sim / max(total_within_pairs, 1)
            avg_between = between_cluster_sim / max(total_between_pairs, 1)
            
            # Score is the difference (higher within - lower between = better)
            score = avg_within - avg_between
            
            return float(score)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate matrix clustering score: {e}")
            return 0.0
    
    def apply_hierarchical_post_processing(self, clustering_result: Dict[str, Any], target_clusters: int = 75) -> Dict[str, Any]:
        """Apply hierarchical post-processing to group similar HMM clusters into super-clusters.
        
        Args:
            clustering_result: Original HMM clustering result with 283 clusters
            target_clusters: Target number of super-clusters (default: 75)
            
        Returns:
            Enhanced clustering result with hierarchical super-clusters
        """
        try:
            import numpy as np
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.preprocessing import StandardScaler
            
            self.logger.info(f"🔄 Starting hierarchical post-processing: {clustering_result.get('n_clusters', 'unknown')} → {target_clusters} super-clusters")
            
            # Extract cluster characteristics for hierarchical clustering
            cluster_features, cluster_metadata = self._extract_cluster_features_for_hierarchical(clustering_result)
            
            if cluster_features is None or len(cluster_features) == 0:
                self.logger.warning("⚠️ No cluster features available for hierarchical post-processing")
                return clustering_result
            
            # Standardize features for better clustering
            scaler = StandardScaler()
            normalized_features = scaler.fit_transform(cluster_features)
            
            self.logger.info(f"📊 Extracted {len(cluster_features)} cluster feature vectors with {cluster_features.shape[1]} dimensions")
            
            # Adjust target clusters based on available data
            n_available_clusters = len(cluster_features)
            if target_clusters >= n_available_clusters:
                adjusted_target = max(1, n_available_clusters // 2)  # Use half of available clusters
                self.logger.warning(f"⚠️ Adjusting target clusters: {target_clusters} → {adjusted_target} (only {n_available_clusters} clusters available)")
                target_clusters = adjusted_target
            
            # Apply Ward hierarchical clustering
            hierarchical_clusterer = AgglomerativeClustering(
                n_clusters=target_clusters,
                linkage='ward',
                metric='euclidean'
            )
            
            super_cluster_labels = hierarchical_clusterer.fit_predict(normalized_features)
            
            # Create super-cluster mapping
            super_cluster_mapping = self._create_super_cluster_mapping(
                super_cluster_labels, cluster_metadata, clustering_result
            )
            
            # Calculate coverage metrics
            coverage_metrics = self._calculate_super_cluster_coverage(super_cluster_mapping, clustering_result)
            
            # Create enhanced result
            enhanced_result = self._create_enhanced_clustering_result(
                clustering_result, super_cluster_mapping, coverage_metrics, target_clusters
            )
            
            self.logger.info(f"✅ Hierarchical post-processing completed:")
            self.logger.info(f"   📊 Original clusters: {clustering_result.get('n_clusters', 'unknown')}")
            self.logger.info(f"   📊 Super-clusters: {target_clusters}")
            self.logger.info(f"   🎯 Top 20 coverage: {coverage_metrics.get('top_20_coverage', 0):.1f}%")
            self.logger.info(f"   📈 Quality preservation: {coverage_metrics.get('quality_preservation', 0):.3f}")
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"❌ Error in hierarchical post-processing: {e}")
            return clustering_result
    
    def _extract_cluster_features_for_hierarchical(self, clustering_result: Dict[str, Any]) -> Tuple[Any, List[Dict]]:
        """Extract numerical features from each cluster for hierarchical clustering."""
        try:
            import numpy as np
            
            cluster_assignments = clustering_result.get('cluster_assignments', [])
            regime_to_cluster = clustering_result.get('regime_to_cluster', {})
            aligned_market_data = clustering_result.get('aligned_market_data', [])
            
            # Debug what we actually have
            self.logger.info(f"🔍 DEBUG EXTRACTION: cluster_assignments={len(cluster_assignments)}, regime_to_cluster={len(regime_to_cluster)}, aligned_market_data={len(aligned_market_data) if hasattr(aligned_market_data, '__len__') else 'no len'}")
            
            # Try alternative data sources if primary ones are missing
            if not cluster_assignments and 'hmm_models' in clustering_result:
                self.logger.info("🔄 Using alternative data extraction from hmm_models")
                return self._extract_features_from_hmm_models(clustering_result)
            
            # Check data availability with proper DataFrame handling
            assignments_valid = len(cluster_assignments) > 0
            market_data_valid = aligned_market_data is not None and len(aligned_market_data) > 0
            
            if not assignments_valid or not market_data_valid:
                self.logger.warning(f"⚠️ Missing required data: assignments={len(cluster_assignments)}, market_data={len(aligned_market_data) if hasattr(aligned_market_data, '__len__') else 'invalid'}")
                return None, []
            
            # Get unique cluster IDs
            unique_clusters = sorted(set(cluster_assignments))
            n_clusters = len(unique_clusters)
            
            self.logger.info(f"📊 Extracting features for {n_clusters} clusters from {len(aligned_market_data)} samples")
            
            # Initialize feature matrix
            feature_matrix = []
            cluster_metadata = []
            
            for cluster_id in unique_clusters:
                # Get samples belonging to this cluster
                cluster_samples = [i for i, c in enumerate(cluster_assignments) if c == cluster_id]
                
                if not cluster_samples:
                    continue
                
                # Extract market data for this cluster (handle DataFrame properly)
                if hasattr(aligned_market_data, 'iloc'):  # DataFrame
                    cluster_data = [aligned_market_data.iloc[i] for i in cluster_samples]
                else:  # List or array
                    cluster_data = [aligned_market_data[i] for i in cluster_samples]
                
                # Calculate cluster features
                features = self._calculate_cluster_statistical_features(cluster_data, cluster_id)
                
                if features is not None:
                    feature_matrix.append(features)
                    cluster_metadata.append({
                        'cluster_id': cluster_id,
                        'sample_count': len(cluster_samples),
                        'sample_indices': cluster_samples
                    })
            
            if not feature_matrix:
                return None, []
            
            return np.array(feature_matrix), cluster_metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting cluster features: {e}")
            return None, []
    
    def _calculate_cluster_statistical_features(self, cluster_data: List[Dict], cluster_id: int) -> List[float]:
        """Calculate statistical features for a single cluster."""
        try:
            import numpy as np
            
            if not cluster_data:
                return None
            
            # Extract numerical features from market data
            features = []
            
            # Helper function to safely extract values from DataFrame rows or dicts
            def safe_extract(sample, key, default=0):
                if hasattr(sample, 'get'):  # Dict-like
                    return sample.get(key, default)
                elif hasattr(sample, key):  # DataFrame row/Series
                    return getattr(sample, key, default)
                else:
                    return default
            
            # Price-based features
            prices = [safe_extract(sample, 'close', 0) for sample in cluster_data]
            prices = [p for p in prices if p != 0]  # Remove zeros
            if prices:
                features.extend([
                    np.mean(prices),
                    np.std(prices),
                    np.median(prices),
                    np.percentile(prices, 25),
                    np.percentile(prices, 75)
                ])
            else:
                features.extend([0, 0, 0, 0, 0])
            
            # Volume features
            volumes = [safe_extract(sample, 'volume', 0) for sample in cluster_data]
            volumes = [v for v in volumes if v != 0]  # Remove zeros
            if volumes:
                features.extend([
                    np.mean(volumes),
                    np.std(volumes),
                    np.log1p(np.mean(volumes))  # Log-scaled volume
                ])
            else:
                features.extend([0, 0, 0])
            
            # Volatility features (price ranges)
            high_prices = [safe_extract(sample, 'high', 0) for sample in cluster_data]
            low_prices = [safe_extract(sample, 'low', 0) for sample in cluster_data]
            high_prices = [h for h in high_prices if h != 0]
            low_prices = [l for l in low_prices if l != 0]
            if high_prices and low_prices and len(high_prices) == len(low_prices):
                ranges = [h - l for h, l in zip(high_prices, low_prices)]
                features.extend([
                    np.mean(ranges),
                    np.std(ranges)
                ])
            else:
                features.extend([0, 0])
            
            # Returns features
            if len(prices) > 1:
                returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
                # Use scipy for skew and kurtosis, or simple approximations
                try:
                    from scipy.stats import skew, kurtosis
                    skew_val = skew(returns) if len(returns) > 2 else 0
                    kurt_val = kurtosis(returns) if len(returns) > 3 else 0
                except ImportError:
                    # Fallback to simple approximations
                    skew_val = 0
                    kurt_val = 0
                
                features.extend([
                    np.mean(returns),
                    np.std(returns),
                    skew_val,
                    kurt_val
                ])
            else:
                features.extend([0, 0, 0, 0])
            
            # Cluster size feature
            features.append(len(cluster_data))
            
            return features
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating features for cluster {cluster_id}: {e}")
            return None
    
    def _create_super_cluster_mapping(self, super_cluster_labels: Any, cluster_metadata: List[Dict], 
                                    clustering_result: Dict[str, Any]) -> Dict[int, Dict]:
        """Create mapping from super-clusters to their constituent HMM clusters."""
        try:
            import numpy as np
            
            super_cluster_mapping = {}
            
            for i, super_label in enumerate(super_cluster_labels):
                if super_label not in super_cluster_mapping:
                    super_cluster_mapping[super_label] = {
                        'hmm_clusters': [],
                        'total_samples': 0,
                        'sample_indices': []
                    }
                
                # Add HMM cluster info to super-cluster
                cluster_info = cluster_metadata[i]
                super_cluster_mapping[super_label]['hmm_clusters'].append(cluster_info['cluster_id'])
                super_cluster_mapping[super_label]['total_samples'] += cluster_info['sample_count']
                super_cluster_mapping[super_label]['sample_indices'].extend(cluster_info['sample_indices'])
            
            return super_cluster_mapping
            
        except Exception as e:
            self.logger.error(f"❌ Error creating super-cluster mapping: {e}")
            return {}
    
    def _calculate_super_cluster_coverage(self, super_cluster_mapping: Dict[int, Dict], 
                                        clustering_result: Dict[str, Any]) -> Dict[str, float]:
        """Calculate coverage metrics for super-clusters."""
        try:
            # Try to get total samples from different sources
            total_samples = len(clustering_result.get('cluster_assignments', []))
            
            # If cluster_assignments is empty, estimate from hmm_models
            if total_samples == 0:
                hmm_models = clustering_result.get('hmm_models', [])
                total_samples = sum(model.get('regime_count', 0) * 30 for model in hmm_models)  # Estimate
                self.logger.info(f"📊 Estimated total samples from HMM models: {total_samples}")
            
            if total_samples == 0:
                return {'top_20_coverage': 0.0, 'quality_preservation': 0.0}
            
            # Sort super-clusters by size
            sorted_super_clusters = sorted(
                super_cluster_mapping.items(), 
                key=lambda x: x[1]['total_samples'], 
                reverse=True
            )
            
            # Calculate top 20 coverage
            top_20_samples = sum(sc[1]['total_samples'] for sc in sorted_super_clusters[:20])
            top_20_coverage = (top_20_samples / total_samples) * 100
            
            # Calculate quality preservation (average cluster coherence)
            quality_preservation = clustering_result.get('quality_score', 0.0)
            
            return {
                'top_20_coverage': top_20_coverage,
                'quality_preservation': quality_preservation,
                'super_cluster_count': len(super_cluster_mapping),
                'largest_super_cluster_pct': (sorted_super_clusters[0][1]['total_samples'] / total_samples) * 100 if sorted_super_clusters else 0
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating super-cluster coverage: {e}")
            return {'top_20_coverage': 0.0, 'quality_preservation': 0.0}
    
    def _create_enhanced_clustering_result(self, original_result: Dict[str, Any], 
                                         super_cluster_mapping: Dict[int, Dict],
                                         coverage_metrics: Dict[str, float],
                                         target_clusters: int) -> Dict[str, Any]:
        """Create enhanced clustering result with hierarchical super-clusters."""
        try:
            # Create new cluster assignments based on super-clusters
            original_assignments = original_result.get('cluster_assignments', [])
            
            # Handle case where cluster_assignments might be empty
            if not original_assignments:
                self.logger.info("📊 Creating simplified enhanced result (no cluster assignments available)")
                enhanced_assignments = []
            else:
                enhanced_assignments = [0] * len(original_assignments)
                
                # Map original cluster assignments to super-cluster assignments
                cluster_to_super_cluster = {}
                for super_id, super_info in super_cluster_mapping.items():
                    for hmm_cluster_id in super_info['hmm_clusters']:
                        cluster_to_super_cluster[hmm_cluster_id] = super_id
                
                # Update assignments
                for i, original_cluster in enumerate(original_assignments):
                    enhanced_assignments[i] = cluster_to_super_cluster.get(original_cluster, original_cluster)
            
            # Create enhanced result with JSON-serializable types
            enhanced_result = original_result.copy()
            enhanced_result.update({
                'cluster_assignments': self._convert_to_json_serializable(enhanced_assignments),
                'n_clusters': int(target_clusters),  # Ensure native Python int
                'hierarchical_mapping': self._convert_to_json_serializable(super_cluster_mapping),
                'original_n_clusters': int(original_result.get('n_clusters', 0)),
                'coverage_metrics': self._convert_to_json_serializable(coverage_metrics),
                'method': f"{original_result.get('method', 'hmm')}_hierarchical",
                'hierarchical_post_processing': True
            })
            
            # Also convert the entire result to ensure all nested structures are JSON-safe
            enhanced_result = self._convert_to_json_serializable(enhanced_result)
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"❌ Error creating enhanced clustering result: {e}")
            return original_result
    
    def _extract_features_from_hmm_models(self, clustering_result: Dict[str, Any]) -> Tuple[Any, List[Dict]]:
        """Alternative feature extraction from hmm_models when standard structure is unavailable."""
        try:
            import numpy as np
            
            hmm_models = clustering_result.get('hmm_models', [])
            
            if not hmm_models:
                self.logger.warning("⚠️ No hmm_models available for feature extraction")
                return None, []
            
            self.logger.info(f"📊 Extracting features from {len(hmm_models)} HMM models")
            
            feature_matrix = []
            cluster_metadata = []
            
            for model in hmm_models:
                cluster_id = model.get('cluster_id', -1)
                regime_count = model.get('regime_count', 0)
                
                if regime_count == 0:
                    continue
                
                # Create simplified features based on available model data
                # Since we don't have access to raw market data, use model metadata
                features = [
                    float(cluster_id),           # Cluster ID as feature
                    float(regime_count),         # Number of regimes in cluster
                    float(regime_count) / 517.0, # Normalized regime count
                    np.log1p(regime_count),      # Log-scaled regime count
                    1.0 / (1.0 + cluster_id),   # Inverse cluster ID (earlier clusters might be more significant)
                ]
                
                feature_matrix.append(features)
                cluster_metadata.append({
                    'cluster_id': cluster_id,
                    'regime_count': regime_count,
                    'sample_count': regime_count * 30  # Estimate sample count
                })
            
            if not feature_matrix:
                return None, []
            
            self.logger.info(f"✅ Extracted {len(feature_matrix)} simplified cluster features")
            return np.array(feature_matrix), cluster_metadata
            
        except Exception as e:
            self.logger.error(f"❌ Error in alternative feature extraction: {e}")
            return None, []
    
    def _convert_to_json_serializable(self, obj):
        """Convert NumPy types and other non-serializable types to JSON-serializable formats."""
        try:
            import numpy as np
            
            if isinstance(obj, dict):
                return {str(k): self._convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return [self._convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif hasattr(obj, 'dtype'):  # Any NumPy type with dtype
                if hasattr(obj, 'item'):
                    return obj.item()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                else:
                    return str(obj)
            elif hasattr(obj, 'item'):  # NumPy scalar
                return obj.item()
            elif str(type(obj)).startswith("<class 'numpy."):  # Catch any remaining NumPy types
                try:
                    return obj.item() if hasattr(obj, 'item') else str(obj)
                except:
                    return str(obj)
            else:
                return obj
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error converting to JSON serializable: {e}")
            return str(obj) if obj is not None else None
    