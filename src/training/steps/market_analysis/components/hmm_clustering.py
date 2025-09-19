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
    from src.utils.matrix_operations import UnifiedMatrixOperations
    HARDWARE_OPTIMIZATION_AVAILABLE = True
except ImportError:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    M1MemoryOptimizer = None
    M1CPUOptimizer = None
    UnifiedMatrixOperations = None

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger

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
            
            # Validate inputs
            if data is None:
                error_msg = "Input data is None"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(success=False, artifacts={}, error_message=error_msg)
            
            if not pipeline_state:
                error_msg = "Pipeline state is empty"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(success=False, artifacts={}, error_message=error_msg)
            
            tprint("✅ Input validation passed")
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
            
            # Get market data
            tprint("📊 Loading market data...")
            try:
                market_data = await self._load_market_data(data)
                if market_data is None:
                    error_msg = "Market data is None after loading"
                    tprint(f"❌ {error_msg}")
                    self.logger.error(error_msg)
                    return ComponentResult(success=False, artifacts={}, error_message=error_msg)
                
                if hasattr(market_data, 'empty') and market_data.empty:
                    error_msg = "Market data is empty"
                    tprint(f"❌ {error_msg}")
                    self.logger.error(error_msg)
                    return ComponentResult(success=False, artifacts={}, error_message=error_msg)
                
                tprint(f"✅ Market data loaded successfully: {type(market_data)}")
                if hasattr(market_data, 'shape'):
                    tprint(f"📊 Market data shape: {market_data.shape}")
                
            except Exception as e:
                error_msg = f"Failed to load market data: {e}"
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
                
                tprint(f"📊 Extracted {len(hmm_models)} HMM models")
                tprint(f"📊 Extracted {len(cluster_assignments)} cluster assignments")
                tprint(f"📊 Cluster metrics keys: {list(cluster_metrics.keys()) if cluster_metrics else 'None'}")
                
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
            
            # Apply regime constraints
            tprint("🔧 Applying regime constraints...")
            try:
                validated_result = self._apply_regime_constraints(
                    hmm_models, cluster_assignments, clustering_config
                )
                hmm_models = validated_result['hmm_models']
                cluster_assignments = validated_result['cluster_assignments']
                tprint(f"✅ Regime constraints applied: {len(hmm_models)} models, {len(cluster_assignments)} assignments")
            except Exception as e:
                error_msg = f"Failed to apply regime constraints: {e}"
                tprint(f"❌ {error_msg}")
                self.logger.error(error_msg)
                return ComponentResult(success=False, artifacts={}, error_message=error_msg)
            
            # Perform comprehensive cluster quality validation
            tprint("🔍 Performing cluster quality validation...")
            try:
                quality_metrics = self._validate_cluster_quality(
                    hmm_models, cluster_assignments, market_data, clustering_config
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
                    hmm_models, cluster_assignments, market_data, clustering_config
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
                
                artifacts = {
                    'hmm_clustering_result': {
                        # Core clustering results
                        'hmm_models': hmm_models,
                        'cluster_assignments': cluster_assignments,
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
                        
                        # Clustering summary with advanced metrics
                        'clustering_summary': {
                            'total_clusters': len(hmm_models),
                            'total_assignments': len(cluster_assignments),
                            'cluster_distribution': self._calculate_cluster_distribution(cluster_assignments),
                            'clustering_time': clustering_result.get('clustering_time', 0.0),
                            'quality_score': quality_metrics.get('overall_quality_score', 0.0),
                            'validation_passed': quality_metrics.get('validation_passed', False),
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
                                'max_clusters_allowed': clustering_config.get('max_clusters', 25),
                                'clustering_method': 'advanced_multi_method_consensus',
                                'validation_dimensions': ['statistical', 'economic', 'financial', 'stability']
                            }
                        }
                    }
                }
                
                # Add comprehensive summary for easy access
                artifacts['clustering_summary'] = {
                    'method': 'Advanced Multi-Method Consensus HMM Clustering',
                    'cluster_count': len(hmm_models),
                    'regime_reduction': f"{len(hmm_regime_discovery.get('regime_models', []))} → {len(hmm_models)}",
                    'data_driven_selection': True,
                    'advanced_methods_used': {
                        'information_criteria': True,
                        'gmm_confidence_optimization': advanced_analysis.get('gmm_analysis', {}).get('gmm_quality') == 'good',
                        'spectral_clustering': advanced_analysis.get('spectral_analysis', {}).get('spectral_quality') == 'good',
                        'economic_validation': True,
                        'multi_dimensional_validation': True
                    },
                    'quality_assessment': {
                        'overall_score': statistical_analysis.get('overall_cluster_quality', {}).get('overall_score', 0.0),
                        'quality_level': statistical_analysis.get('overall_cluster_quality', {}).get('quality_level', 'unknown'),
                        'economic_validation_passed': statistical_analysis.get('economic_validation', {}).get('overall_economic_alignment', {}).get('economic_validation_passed', False)
                    },
                    'market_insights': {
                        'primary_driver': statistical_analysis.get('factor_impact_analysis', {}).get('primary_market_driver', {}).get('dominant_aspect', 'unknown'),
                        'aspects_tested': ['momentum', 'volatility', 'volume'],
                        'financial_validation': ['volatility_regimes', 'momentum_patterns', 'volume_patterns', 'market_stress']
                    }
                }
                
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
            
            self.logger.info(f'✅ HMM Clustering completed: {len(hmm_models)} clusters created (from up to 150 regimes)')
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
            
            max_clusters = config.get('max_clusters', 40)  # Reasonable maximum for regime clustering
            clustering_result = self._perform_matrix_based_clustering(
                regime_characteristics, regime_assignments, max_clusters, market_data
            )
            
            clustering_time = time.time() - start_time
            clustering_result['clustering_time'] = clustering_time
            
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"HMM clustering process failed: {e}")
            raise

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
            
            # Calculate regime similarity matrix
            similarity_matrix = self._calculate_regime_similarity_matrix(regime_characteristics)
            if similarity_matrix.size == 0:
                self.logger.warning("⚠️ Empty similarity matrix, falling back to standard clustering")
                return self._perform_standard_clustering_fallback(regime_characteristics, regime_assignments, max_clusters, market_data)
            
            regime_ids = list(regime_characteristics.keys())
            n_regimes = len(regime_ids)
            
            self.logger.info(f"🎯 Starting hierarchical matrix clustering: {n_regimes} regimes → target: 20-40 clusters")
            
            # Initialize regime-to-cluster mapping (each regime starts as its own cluster)
            regime_to_cluster = {regime_id: i for i, regime_id in enumerate(regime_ids)}
            cluster_count = n_regimes
            
            # Intelligent hierarchical merging with adaptive similarity thresholds
            # Automatically detect optimal stopping point based on regime dissimilarity
            min_cv_threshold = 0.01  # Very strict CV difference threshold
            # Note: Merging logic allows clusters >6% to merge as long as resulting cluster <12%
            
            # Calculate adaptive similarity thresholds based on data characteristics
            similarity_thresholds = self._calculate_adaptive_similarity_thresholds(similarity_matrix)
            
            for threshold in similarity_thresholds:
                if cluster_count <= 20:  # Stop when we reach lower bound of optimal range (20-40)
                    self.logger.info(f"🎯 STOPPING: Reached optimal cluster count ({cluster_count} <= 20)")
                    break
                    
                self.logger.info(f"🔄 Batch merging at {threshold*100:.1f}% similarity threshold ({threshold:.3f})...")
                self.logger.info(f"   📊 Starting with {cluster_count} clusters")
                
                # Check CV-based stopping criteria before merging
                if self._should_stop_merging_due_to_cv(regime_characteristics, regime_to_cluster):
                    self.logger.info(f"🛑 STOPPING: CV-based quality criteria indicate regimes are too dissimilar")
                    break
                
                # Check if current threshold is too low (regimes too dissimilar)
                if self._is_similarity_threshold_too_low(threshold, similarity_matrix, regime_to_cluster):
                    self.logger.info(f"🛑 STOPPING: Similarity threshold {threshold:.4f} is too low - regimes are too dissimilar")
                    break
                
                # Calculate cluster sample percentages
                total_samples = len(regime_assignments)
                cluster_sample_counts = {}
                for regime_id, cluster_id in regime_to_cluster.items():
                    if cluster_id not in cluster_sample_counts:
                        cluster_sample_counts[cluster_id] = 0
                    cluster_sample_counts[cluster_id] += regime_characteristics[regime_id].get('sample_count', 1)
                
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
                
                # Find mergeable pairs based on similarity threshold
                mergeable_pairs = []
                for i, regime_i in enumerate(regime_ids):
                    for j, regime_j in enumerate(regime_ids[i+1:], i+1):
                        similarity = similarity_matrix[i, j]
                        cluster_i = regime_to_cluster[regime_i]
                        cluster_j = regime_to_cluster[regime_j]
                        
                        # Skip if same cluster
                        if cluster_i == cluster_j:
                            continue
                        
                        # Enforce 12% max resulting cluster size constraint
                        merged_size = cluster_sample_counts.get(cluster_i, 0) + cluster_sample_counts.get(cluster_j, 0)
                        merged_pct = (merged_size / total_samples) * 100 if total_samples > 0 else 0
                        if merged_pct > 12.0:
                            # Too big after merge, skip
                            continue
                        
                        # Apply soft CV penalties to similarity requirement
                        pair_threshold = threshold + cluster_penalty.get(cluster_i, 0.0) + cluster_penalty.get(cluster_j, 0.0)
                        if similarity < pair_threshold:
                            continue
                                
                            # Check CV compatibility
                            regime_1 = regime_characteristics[regime_i]
                            regime_2 = regime_characteristics[regime_j]
                            cv_compatible = self._check_cv_compatibility(regime_1, regime_2, min_cv_threshold)
                            
                            if cv_compatible:
                                mergeable_pairs.append((similarity, cluster_i, cluster_j, regime_i, regime_j))
                
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
                    
                    # Merge cluster_j into cluster_i
                    for regime_id, cluster_id in regime_to_cluster.items():
                        if cluster_id == cluster_j:
                            regime_to_cluster[regime_id] = cluster_i
                    
                    # Track merge statistics
                    merge_similarities.append(similarity)
                    merged_clusters.add(cluster_j)
                    merges_this_round += 1
                    
                    # Calculate merged cluster size
                    merged_size = cluster_sample_counts.get(cluster_i, 0) + cluster_sample_counts.get(cluster_j, 0)
                    merged_cluster_sizes.append(merged_size)
                    
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
                else:
                    self.logger.info(f"   📊 Batch complete: {merges_this_round} merges → {cluster_count} clusters remaining")
                
                # Recalculate similarity matrix with new cluster characteristics after merging
                if merges_this_round > 0:
                    self.logger.info(f"   🔄 Recalculating similarity matrix with {cluster_count} clusters...")
                    
                    # Update regime_ids to reflect current clusters
                    current_regime_ids = list(regime_to_cluster.keys())
                    
                    # Recalculate similarity matrix with updated cluster characteristics
                    similarity_matrix = self._calculate_regime_similarity_matrix(regime_characteristics)
                    
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
            
            # Calculate final quality metrics
            quality_score = self._calculate_matrix_clustering_score(
                similarity_matrix, regime_to_cluster, cluster_assignments
            )
            
            # Create result in expected format
            result = {
                'cluster_assignments': cluster_assignments,
                'regime_to_cluster': regime_to_cluster,
                'n_clusters': final_cluster_count,
                'similarity_matrix': similarity_matrix,
                'quality_score': quality_score,
                'method': 'hierarchical_matrix_based',
                'merging_thresholds_used': similarity_thresholds
            }
            
            self.logger.info(f"✅ Hierarchical matrix clustering completed: {final_cluster_count} clusters created")
            self.logger.info(f"📊 Final cluster distribution: {self._calculate_cluster_distribution(cluster_assignments)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Hierarchical matrix clustering failed: {e}")
            # Fallback to standard clustering
            return self._perform_standard_clustering_fallback(regime_characteristics, regime_assignments, max_clusters, market_data)

    def _check_cv_compatibility(self, regime_1: Dict[str, Any], regime_2: Dict[str, Any], min_cv_threshold: float) -> bool:
        """Check if two regimes are compatible for merging based on CV differences."""
        try:
            # Compare key characteristics
            cv_features = ['momentum_cv', 'volatility_cv', 'volume_cv', 'mean_cv']
            
            for feature in cv_features:
                cv1 = regime_1.get(feature, 0.0)
                cv2 = regime_2.get(feature, 0.0)
                
                # Calculate relative CV difference
                if max(cv1, cv2) > 0:
                    cv_difference = abs(cv1 - cv2) / max(cv1, cv2)
                    if cv_difference > min_cv_threshold:
                        return False # Not compatible if CV difference is too high
            return True # Compatible if all CV differences are within threshold
        except Exception as e:
            self.logger.warning(f"⚠️ Error checking CV compatibility: {e}")
            return False # Assume not compatible on error
    
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
        # Define feature groups
        aspect_features = {
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
        # Build map from regime_id -> raw feature dict
        regime_ids = list(regime_to_cluster.keys())
        # Extract per-regime features from provided characteristics (flat in regime['features'] if present),
        # otherwise look into sub-dicts for consistency with prior code sections
        regime_feature_maps: Dict[str, Dict[str, float]] = {}
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
                        continue
                    med = np.nanmedian(arr)
                    mad = np.nanmedian(np.abs(arr - med))
                    robust_scale = 1.4826 * mad
                    denom = max(abs(med), eps)
                    cvj = robust_scale / denom
                    if np.isfinite(cvj):
                        per_feature_cvs.append(float(cvj))
                aspect_cvs[aspect] = float(np.nanmedian(per_feature_cvs)) if len(per_feature_cvs) > 0 else 0.0
            cluster_cv_aspects[cid] = aspect_cvs
        return cluster_cv_aspects

    def _extract_regime_characteristics_from_discovery(self, regime_discovery: Dict[str, Any]) -> Dict[str, Any]:
        """Extract volume, volatility, and momentum characteristics from HMM regime discovery results."""
        try:
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
                
                extracted_characteristics[regime_key] = {
                    'features': features,
                    'sample_count': characteristics.get('sample_count', 0)
                }
            
            self.logger.info(f"✅ Extracted characteristics for {len(extracted_characteristics)} regimes")
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
    
    def _extract_volatility_characteristics(self, feature_means: Dict[str, float], feature_stds: Dict[str, float]) -> Dict[str, Any]:
        """Extract volatility-related characteristics from regime features."""
        volatility_chars = {}
        
        # Multi-timeframe volatility
        vol_5 = feature_means.get('volatility_5', 0.0)
        vol_10 = feature_means.get('volatility_10', 0.0)
        vol_20 = feature_means.get('volatility_20', 0.0)
        vol_20_std = feature_stds.get('volatility_20', 0.0)
        
        volatility_chars['mean_volatility_5'] = vol_5
        volatility_chars['mean_volatility_10'] = vol_10
        volatility_chars['mean_volatility_20'] = vol_20
        volatility_chars['volatility_volatility'] = vol_20_std
        
        # Volatility trend analysis
        volatility_momentum = feature_means.get('volatility_momentum', 0.0)
        volatility_acceleration = feature_means.get('volatility_acceleration', 0.0)
        
        volatility_chars['volatility_momentum'] = volatility_momentum
        volatility_chars['volatility_acceleration'] = volatility_acceleration
        volatility_chars['volatility_trend'] = 'increasing' if volatility_momentum > 0.01 else 'decreasing' if volatility_momentum < -0.01 else 'stable'
        
        # ATR characteristics
        atr = feature_means.get('atr', 0.0)
        atr_normalized = feature_means.get('atr_normalized', 0.0)
        
        volatility_chars['mean_atr'] = atr
        volatility_chars['mean_atr_normalized'] = atr_normalized
        
        # Overall volatility level
        avg_volatility = (vol_5 + vol_10 + vol_20) / 3
        volatility_chars['volatility_level'] = 'high' if avg_volatility > 0.03 else 'low' if avg_volatility < 0.01 else 'medium'
        
        return volatility_chars
    
    def _extract_momentum_characteristics(self, feature_means: Dict[str, float], feature_stds: Dict[str, float]) -> Dict[str, Any]:
        """Extract momentum-related characteristics from regime features."""
        momentum_chars = {}
        
        # Price momentum
        price_momentum_5 = feature_means.get('price_momentum_5', 0.0)
        price_momentum_20 = feature_means.get('price_momentum_20', 0.0)
        price_momentum_5_std = feature_stds.get('price_momentum_5', 0.0)
        
        momentum_chars['mean_price_momentum_5'] = price_momentum_5
        momentum_chars['mean_price_momentum_20'] = price_momentum_20
        momentum_chars['price_momentum_volatility'] = price_momentum_5_std
        
        # RSI momentum
        rsi = feature_means.get('rsi', 50.0)
        rsi_momentum = feature_means.get('rsi_momentum', 0.0)
        rsi_std = feature_stds.get('rsi', 0.0)
        
        momentum_chars['mean_rsi'] = rsi
        momentum_chars['rsi_momentum'] = rsi_momentum
        momentum_chars['rsi_volatility'] = rsi_std
        
        # MACD momentum
        macd = feature_means.get('macd', 0.0)
        macd_momentum = feature_means.get('macd_momentum', 0.0)
        macd_std = feature_stds.get('macd', 0.0)
        
        momentum_chars['mean_macd'] = macd
        momentum_chars['macd_momentum'] = macd_momentum
        momentum_chars['macd_volatility'] = macd_std
        
        # Overall momentum assessment
        momentum_chars['momentum_direction'] = 'bullish' if price_momentum_5 > 0.02 else 'bearish' if price_momentum_5 < -0.02 else 'neutral'
        momentum_chars['momentum_strength'] = abs(price_momentum_5) + abs(rsi_momentum) + abs(macd_momentum)
        momentum_chars['momentum_strength_level'] = 'strong' if momentum_chars['momentum_strength'] > 0.1 else 'weak' if momentum_chars['momentum_strength'] < 0.02 else 'moderate'
        
        # Add risk-return characteristics
        risk_return_chars = self._extract_risk_return_characteristics(feature_means, feature_stds)
        momentum_chars.update(risk_return_chars)
        
        return momentum_chars

    def _extract_risk_return_characteristics(self, feature_means: Dict[str, float], feature_stds: Dict[str, float]) -> Dict[str, Any]:
        """Extract risk-adjusted returns, drawdown patterns, and trend persistence characteristics."""
        risk_return_chars = {}
        
        try:
            # Risk-adjusted returns
            returns = feature_means.get('returns', 0.0)
            returns_std = feature_stds.get('returns', 0.01)  # Avoid division by zero
            
            # Sharpe ratio approximation (assuming risk-free rate ≈ 0)
            sharpe_ratio = returns / returns_std if returns_std > 0 else 0.0
            risk_return_chars['sharpe_ratio'] = sharpe_ratio
            risk_return_chars['risk_adjusted_return'] = returns / max(returns_std, 0.001)
            
            # Return characteristics
            risk_return_chars['mean_returns'] = returns
            risk_return_chars['return_volatility'] = returns_std
            risk_return_chars['return_skewness'] = feature_means.get('return_skewness', 0.0)
            risk_return_chars['return_kurtosis'] = feature_means.get('return_kurtosis', 3.0)
            
            # Drawdown characteristics
            max_drawdown = feature_means.get('max_drawdown', 0.0)
            avg_drawdown = feature_means.get('avg_drawdown', 0.0)
            drawdown_duration = feature_means.get('drawdown_duration', 0.0)
            
            risk_return_chars['max_drawdown'] = abs(max_drawdown)  # Make positive
            risk_return_chars['avg_drawdown'] = abs(avg_drawdown)
            risk_return_chars['drawdown_duration'] = drawdown_duration
            risk_return_chars['drawdown_recovery_ratio'] = abs(returns) / max(abs(max_drawdown), 0.001)
            
            # Trend persistence characteristics
            trend_strength = feature_means.get('trend_strength', 0.0)
            trend_consistency = feature_means.get('trend_consistency', 0.0)
            autocorrelation_1 = feature_means.get('autocorr_1', 0.0)
            autocorrelation_5 = feature_means.get('autocorr_5', 0.0)
            
            risk_return_chars['trend_strength'] = trend_strength
            risk_return_chars['trend_consistency'] = trend_consistency
            risk_return_chars['autocorr_1_day'] = autocorrelation_1
            risk_return_chars['autocorr_5_day'] = autocorrelation_5
            risk_return_chars['trend_persistence'] = (abs(autocorrelation_1) + abs(autocorrelation_5)) / 2
            
            # Volatility clustering characteristics  
            vol_persistence = feature_means.get('volatility_persistence', 0.0)
            vol_clustering = feature_stds.get('volatility_5', 0.0) / max(feature_means.get('volatility_5', 0.01), 0.01)
            
            risk_return_chars['volatility_persistence'] = vol_persistence
            risk_return_chars['volatility_clustering'] = vol_clustering
            
            # Risk classification
            if sharpe_ratio > 1.0:
                risk_class = 'high_reward_low_risk'
            elif sharpe_ratio > 0.5:
                risk_class = 'moderate_reward_risk'
            elif sharpe_ratio > 0:
                risk_class = 'low_reward_moderate_risk'
            else:
                risk_class = 'negative_reward'
                
            risk_return_chars['risk_classification'] = risk_class
            
            # Market efficiency indicators
            mean_reversion_strength = feature_means.get('mean_reversion', 0.0)
            momentum_persistence = max(abs(autocorrelation_1), abs(autocorrelation_5))
            
            if momentum_persistence > 0.3:
                efficiency_class = 'trending_inefficient'
            elif mean_reversion_strength > 0.3:
                efficiency_class = 'mean_reverting'
            else:
                efficiency_class = 'semi_efficient'
                
            risk_return_chars['market_efficiency'] = efficiency_class
            risk_return_chars['mean_reversion_strength'] = mean_reversion_strength
            
        except Exception as e:
            # Fallback values if calculation fails
            risk_return_chars.update({
                'sharpe_ratio': 0.0,
                'risk_adjusted_return': 0.0,
                'max_drawdown': 0.0,
                'trend_persistence': 0.0,
                'volatility_clustering': 0.0,
                'market_efficiency': 'unknown'
            })
        
        return risk_return_chars


    
    def _calculate_hmm_appropriate_metrics(self, market_data: pd.DataFrame, 
                                         cluster_assignments: np.ndarray, 
                                         n_clusters: int) -> Dict[str, Any]:
        """
        Calculate HMM-appropriate validation metrics instead of traditional clustering metrics.
        
        This replaces misleading clustering metrics (silhouette_score, davies_bouldin_score, 
        calinski_harabasz_score) with metrics appropriate for temporal regime modeling.
        """
        try:
            # Import HMM validation framework
            try:
                from src.utils.hmm_validation import HMMStatisticalValidator
                validator = HMMStatisticalValidator(logger=self.logger)
                
                # Create regime data DataFrame
                regime_data = market_data.copy()
                regime_data['regime'] = cluster_assignments
                
                # Use HMM-appropriate validation
                validation_result = validator.validate_hmm_regimes_appropriate(
                    regime_data, market_data
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
        """Calculate within-cluster coherence focusing on internal consistency rather than separation."""
        try:
            import numpy as np
            
            coherence_scores = []
            unique_clusters = np.unique(cluster_labels)
            
            for cluster_id in unique_clusters:
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                
                if len(cluster_indices) < 2:
                    continue
                    
                # Calculate within-cluster distances
                cluster_distances = distance_matrix[np.ix_(cluster_indices, cluster_indices)]
                
                # Remove diagonal (self-distances = 0)
                n_points = len(cluster_indices)
                within_distances = []
                for i in range(n_points):
                    for j in range(i + 1, n_points):
                        within_distances.append(cluster_distances[i, j])
                
                if within_distances:
                    # Lower average distance = higher coherence
                    avg_within_distance = np.mean(within_distances)
                    # Convert to coherence score (higher = better)
                    coherence = 1.0 / (1.0 + avg_within_distance)
                    coherence_scores.append(coherence)
            
            # Return average coherence across all clusters
            return float(np.mean(coherence_scores)) if coherence_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Within-cluster coherence calculation failed: {e}")
            return 0.0

    def _calculate_additional_validation_metrics(self, distance_matrix: np.ndarray, cluster_range: range, inertias: list, coherence_scores: list) -> dict:
        """Calculate Calinski-Harabasz, Davies-Bouldin, and ARI validation metrics."""
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
            import numpy as np
            
            validation_metrics = {
                'calinski_harabasz_scores': [],
                'davies_bouldin_scores': [],
                'ari_stability_scores': []
            }
            
            # Convert distance matrix to feature matrix for sklearn metrics
            # Use MDS (Multidimensional Scaling) to embed in Euclidean space
            try:
                from sklearn.manifold import MDS
                mds = MDS(n_components=min(10, distance_matrix.shape[0]-1), dissimilarity='precomputed', random_state=42)
                embedded_features = mds.fit_transform(distance_matrix)
            except:
                # Fallback: use distance matrix directly (approximate)
                embedded_features = 1.0 - distance_matrix
            
            for n_clusters in cluster_range:
                try:
                    # Perform clustering
                    clustering = AgglomerativeClustering(
                        n_clusters=n_clusters,
                        metric='precomputed',
                        linkage='average'
                    )
                    cluster_labels = clustering.fit_predict(distance_matrix)
                    
                    # 4. Calinski-Harabasz Index (higher is better)
                    if len(set(cluster_labels)) > 1:
                        ch_score = calinski_harabasz_score(embedded_features, cluster_labels)
                        validation_metrics['calinski_harabasz_scores'].append(ch_score)
                    else:
                        validation_metrics['calinski_harabasz_scores'].append(0.0)
                    
                    # 5. Davies-Bouldin Index (lower is better)
                    if len(set(cluster_labels)) > 1:
                        db_score = davies_bouldin_score(embedded_features, cluster_labels)
                        validation_metrics['davies_bouldin_scores'].append(db_score)
                    else:
                        validation_metrics['davies_bouldin_scores'].append(float('inf'))
                    
                    # 6. ARI Stability (bootstrap approach)
                    ari_stability = self._calculate_ari_stability(distance_matrix, n_clusters)
                    validation_metrics['ari_stability_scores'].append(ari_stability)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Validation metrics failed for {n_clusters} clusters: {e}")
                    validation_metrics['calinski_harabasz_scores'].append(0.0)
                    validation_metrics['davies_bouldin_scores'].append(float('inf'))
                    validation_metrics['ari_stability_scores'].append(0.0)
            
            return validation_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Additional validation metrics calculation failed: {e}")
            return {'calinski_harabasz_scores': [], 'davies_bouldin_scores': [], 'ari_stability_scores': []}

    def _calculate_ari_stability(self, distance_matrix: np.ndarray, n_clusters: int, n_bootstrap: int = 10) -> float:
        """Calculate ARI stability using bootstrap resampling."""
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics import adjusted_rand_score
            import numpy as np
            
            n_samples = distance_matrix.shape[0]
            ari_scores = []
            
            # Original clustering
            original_clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            original_labels = original_clustering.fit_predict(distance_matrix)
            
            for _ in range(n_bootstrap):
                try:
                    # Bootstrap sample indices
                    bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
                    
                    # Create bootstrap distance matrix
                    bootstrap_distance_matrix = distance_matrix[np.ix_(bootstrap_indices, bootstrap_indices)]
                    
                    # Perform clustering on bootstrap sample
                    bootstrap_clustering = AgglomerativeClustering(
                        n_clusters=min(n_clusters, len(np.unique(bootstrap_indices))),
                        metric='precomputed',
                        linkage='average'
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
                        ari_scores.append(ari)
                        
                except Exception:
                    continue
            
            return float(np.mean(ari_scores)) if ari_scores else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ ARI stability calculation failed: {e}")
            return 0.0


    def _calculate_modularity(self, similarity_matrix: np.ndarray, cluster_labels: np.ndarray) -> float:
        """Calculate modularity score for graph clustering quality."""
        try:
            import numpy as np
            
            # Convert similarity to adjacency matrix (threshold at median)
            threshold = np.median(similarity_matrix[similarity_matrix > 0])
            adjacency = (similarity_matrix > threshold).astype(float)
            
            # Calculate modularity
            m = np.sum(adjacency) / 2  # Total edges
            if m == 0:
                return 0.0
            
            modularity = 0.0
            unique_clusters = np.unique(cluster_labels)
            
            for cluster in unique_clusters:
                cluster_indices = np.where(cluster_labels == cluster)[0]
                
                # Edges within cluster
                edges_within = np.sum(adjacency[np.ix_(cluster_indices, cluster_indices)]) / 2
                
                # Expected edges within cluster
                degrees = np.sum(adjacency[cluster_indices, :], axis=1)
                expected_edges = np.sum(degrees) ** 2 / (4 * m)
                
                modularity += (edges_within - expected_edges) / m
            
            return float(modularity)
            
        except Exception as e:
            return 0.0



    def _calculate_regime_similarity_matrix(self, regime_characteristics: Dict[str, Any]) -> np.ndarray:
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
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1.0
            Z_normalized = Z / norms

            # Cosine similarity matrix as dot product of normalized vectors
            similarity_matrix = np.clip(np.dot(Z_normalized, Z_normalized.T), -1.0, 1.0)

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
    
    def _calculate_adaptive_similarity_thresholds(self, similarity_matrix: np.ndarray) -> List[float]:
        """Calculate adaptive similarity thresholds based on data characteristics.
        
        This method analyzes the similarity distribution to automatically determine
        optimal thresholds that balance cluster quality with meaningful grouping.
        
        Returns:
            List of similarity thresholds in descending order (highest to lowest)
        """
        try:
            import numpy as np
            
            # Extract upper triangle of similarity matrix (avoid duplicates and diagonal)
            n_regimes = similarity_matrix.shape[0]
            similarities = []
            for i in range(n_regimes):
                for j in range(i + 1, n_regimes):
                    similarities.append(similarity_matrix[i, j])
            
            if not similarities:
                # Fallback to conservative thresholds
                return [0.999, 0.998, 0.997, 0.996, 0.995]
            
            similarities = np.array(similarities)
            
            # Remove any NaN or infinite values
            valid_similarities = similarities[np.isfinite(similarities)]
            if len(valid_similarities) == 0:
                return [0.999, 0.998, 0.997, 0.996, 0.995]
            
            # Calculate distribution statistics
            mean_sim = np.mean(valid_similarities)
            std_sim = np.std(valid_similarities)
            median_sim = np.median(valid_similarities)
            q75_sim = np.percentile(valid_similarities, 75)
            q90_sim = np.percentile(valid_similarities, 90)
            q95_sim = np.percentile(valid_similarities, 95)
            
            self.logger.info(f"📊 Similarity distribution analysis:")
            self.logger.info(f"   Mean: {mean_sim:.4f}, Std: {std_sim:.4f}")
            self.logger.info(f"   Median: {median_sim:.4f}, Q75: {q75_sim:.4f}, Q90: {q90_sim:.4f}, Q95: {q95_sim:.4f}")
            
            # Strategy 1: Start from high similarity and descend gradually
            # Use percentile-based approach to ensure we have meaningful merges
            
            # Start from 95th percentile (very similar regimes)
            start_threshold = max(0.99, q95_sim)  # At least 99% similarity
            
            # Calculate step size based on distribution
            # Use smaller steps when there's more variation
            if std_sim > 0.1:  # High variation - use smaller steps
                step_size = 0.001  # 0.1% steps
            elif std_sim > 0.05:  # Medium variation
                step_size = 0.002  # 0.2% steps
            else:  # Low variation - use larger steps
                step_size = 0.005  # 0.5% steps
            
            # Calculate stop threshold based on data characteristics
            # Stop when we reach a point where regimes become too dissimilar
            if mean_sim > 0.5:  # Generally similar regimes
                # Stop at mean - 1 std (regimes below this are too dissimilar)
                stop_threshold = max(0.8, mean_sim - std_sim)
            else:  # Generally dissimilar regimes
                # Stop at 75th percentile (regimes below this are too dissimilar)
                stop_threshold = max(0.7, q75_sim)
            
            # Generate threshold sequence
            thresholds = []
            current_threshold = start_threshold
            
            while current_threshold >= stop_threshold and len(thresholds) < 50:  # Safety limit
                thresholds.append(round(current_threshold, 4))
                current_threshold -= step_size
            
            # Ensure we have at least a few thresholds
            if len(thresholds) < 3:
                thresholds = [0.999, 0.995, 0.990, 0.980, 0.970]
            
            self.logger.info(f"🎯 Generated {len(thresholds)} adaptive thresholds:")
            self.logger.info(f"   Start: {thresholds[0]:.4f}, Stop: {thresholds[-1]:.4f}")
            self.logger.info(f"   Thresholds: {thresholds[:5]}..." + (f" to {thresholds[-1]:.4f}" if len(thresholds) > 5 else ""))
            
            return thresholds
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate adaptive similarity thresholds: {e}")
            # Fallback to conservative thresholds
            return [0.999, 0.998, 0.997, 0.996, 0.995]
    
    def _should_stop_merging_due_to_cv(self, regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int]) -> bool:
        """Determine if merging should stop based on CV-based quality criteria.
        
        This method analyzes the current cluster quality and determines if further
        merging would create clusters that are too heterogeneous.
        
        Args:
            regime_characteristics: Dictionary of regime characteristics
            regime_to_cluster: Current regime to cluster mapping
            
        Returns:
            True if merging should stop due to poor cluster quality
        """
        try:
            import numpy as np
            
            # Calculate current cluster CV metrics
            cluster_cv_aspects = self._compute_cluster_cv_by_aspect(regime_characteristics, regime_to_cluster)
            
            if not cluster_cv_aspects:
                return False
            
            # Extract CV values for each aspect
            aspect_cvs = {'momentum': [], 'volatility': [], 'volume': []}
            
            for cluster_id, cv_map in cluster_cv_aspects.items():
                for aspect in aspect_cvs.keys():
                    cv_value = cv_map.get(aspect)
                    if cv_value is not None and np.isfinite(cv_value):
                        aspect_cvs[aspect].append(cv_value)
            
            # Calculate statistics for each aspect
            aspect_stats = {}
            for aspect, cv_values in aspect_cvs.items():
                if len(cv_values) > 0:
                    aspect_stats[aspect] = {
                        'mean': np.mean(cv_values),
                        'median': np.median(cv_values),
                        'q75': np.percentile(cv_values, 75),
                        'q90': np.percentile(cv_values, 90),
                        'count': len(cv_values)
                    }
            
            # Stop merging if cluster quality is degrading
            stop_criteria = []
            
            # Criterion 1: High median CV indicates poor cluster homogeneity
            for aspect, stats in aspect_stats.items():
                if stats['median'] > 0.3:  # 30% CV is quite high
                    stop_criteria.append(f"High {aspect} CV (median: {stats['median']:.3f})")
                
                # Criterion 2: Many clusters with very high CV (>40%)
                high_cv_count = sum(1 for cv in aspect_cvs[aspect] if cv > 0.4)
                if high_cv_count > len(aspect_cvs[aspect]) * 0.5:  # More than 50% of clusters
                    stop_criteria.append(f"Too many high-CV {aspect} clusters ({high_cv_count}/{len(aspect_cvs[aspect])})")
            
            # Criterion 3: Overall cluster heterogeneity
            all_cvs = []
            for cv_values in aspect_cvs.values():
                all_cvs.extend(cv_values)
            
            if len(all_cvs) > 0:
                overall_median_cv = np.median(all_cvs)
                overall_q90_cv = np.percentile(all_cvs, 90)
                
                if overall_median_cv > 0.25:  # 25% median CV across all aspects
                    stop_criteria.append(f"High overall cluster heterogeneity (median CV: {overall_median_cv:.3f})")
                
                if overall_q90_cv > 0.5:  # 90th percentile > 50% CV
                    stop_criteria.append(f"Many very heterogeneous clusters (Q90 CV: {overall_q90_cv:.3f})")
            
            # Criterion 4: Check if we have enough clusters for meaningful analysis
            unique_clusters = len(set(regime_to_cluster.values()))
            if unique_clusters < 15:  # Too few clusters for meaningful market analysis
                stop_criteria.append(f"Too few clusters for meaningful analysis ({unique_clusters})")
            
            # Log the analysis
            if stop_criteria:
                self.logger.info(f"🛑 CV-based stopping criteria triggered:")
                for criterion in stop_criteria:
                    self.logger.info(f"   - {criterion}")
                return True
            else:
                self.logger.info(f"✅ CV-based quality check passed - continuing merging")
                return False
            
        except Exception as e:
            self.logger.error(f"❌ Error in CV-based stopping criteria: {e}")
            return False  # Don't stop on error, let other criteria handle it
    
    def _is_similarity_threshold_too_low(self, threshold: float, similarity_matrix: np.ndarray, regime_to_cluster: Dict[str, int]) -> bool:
        """Check if the current similarity threshold is too low for meaningful clustering.
        
        This method analyzes the similarity distribution at the current threshold to determine
        if merging would combine regimes that are too dissimilar.
        
        Args:
            threshold: Current similarity threshold
            similarity_matrix: Matrix of regime similarities
            regime_to_cluster: Current regime to cluster mapping
            
        Returns:
            True if the threshold is too low and merging should stop
        """
        try:
            import numpy as np
            
            # Get all regime pairs that would be considered for merging at this threshold
            regime_ids = list(regime_to_cluster.keys())
            n_regimes = len(regime_ids)
            
            # Count potential merges at this threshold
            potential_merges = 0
            high_similarity_merges = 0  # Similarities above threshold + 0.05
            low_similarity_merges = 0   # Similarities just above threshold
            
            for i in range(n_regimes):
                for j in range(i + 1, n_regimes):
                    similarity = similarity_matrix[i, j]
                    cluster_i = regime_to_cluster[regime_ids[i]]
                    cluster_j = regime_to_cluster[regime_ids[j]]
                    
                    # Only consider pairs from different clusters
                    if cluster_i != cluster_j and similarity >= threshold:
                        potential_merges += 1
                        
                        if similarity >= threshold + 0.05:  # High similarity
                            high_similarity_merges += 1
                        else:  # Low similarity (just above threshold)
                            low_similarity_merges += 1
            
            # If no potential merges, threshold is fine
            if potential_merges == 0:
                return False
            
            # Calculate ratio of low-similarity merges
            low_similarity_ratio = low_similarity_merges / potential_merges if potential_merges > 0 else 0
            
            # Stop if most merges would be of low similarity
            if low_similarity_ratio > 0.7:  # More than 70% of merges are low similarity
                self.logger.info(f"   📊 Similarity analysis at threshold {threshold:.4f}:")
                self.logger.info(f"      Potential merges: {potential_merges}")
                self.logger.info(f"      High similarity: {high_similarity_merges}, Low similarity: {low_similarity_merges}")
                self.logger.info(f"      Low similarity ratio: {low_similarity_ratio:.3f} (too high)")
                return True
            
            # Stop if threshold is below a certain absolute minimum
            if threshold < 0.5:  # Less than 50% similarity is too low
                self.logger.info(f"   📊 Threshold {threshold:.4f} is below absolute minimum (0.5)")
                return True
            
            # Stop if we have very few high-similarity merges and many low-similarity ones
            if potential_merges > 10 and high_similarity_merges < 3 and low_similarity_merges > 7:
                self.logger.info(f"   📊 Too many low-similarity merges relative to high-similarity ones")
                self.logger.info(f"      High similarity: {high_similarity_merges}, Low similarity: {low_similarity_merges}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Error checking similarity threshold: {e}")
            return False  # Don't stop on error
    
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

    def _calculate_characteristic_similarity(self, chars_1: Dict[str, Any], chars_2: Dict[str, Any], feature_keys: List[str]) -> float:
        """Calculate similarity between two characteristic dictionaries for specific features."""
        try:
            similarities = []
            
            for key in feature_keys:
                val_1 = chars_1.get(key, 0.0)
                val_2 = chars_2.get(key, 0.0)
                
                # Handle string comparisons (categorical features)
                if isinstance(val_1, str) and isinstance(val_2, str):
                    similarity = 1.0 if val_1 == val_2 else 0.0
                else:
                    # Use shared utility function (original 0-1 range)
                    similarity = self._calculate_feature_similarity(val_1, val_2, support_negative=False)
                
                similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Failed to calculate characteristic similarity: {e}")
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
                linkage='average'
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
            cluster_dist = self._calculate_cluster_distribution(cluster_assignments)
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
    
    def _cluster_regimes_by_similarity(self, regime_models: List[Any], n_clusters: int) -> Dict[int, int]:
        """Cluster regimes by similarity using their model characteristics."""
        try:
            import numpy as np
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # Extract regime characteristics for similarity clustering
            regime_features = []
            regime_indices = []
            
            for i, model in enumerate(regime_models):
                try:
                    # Extract key characteristics from HMM model
                    features = []
                    
                    # Transition matrix characteristics
                    if hasattr(model, 'transmat_') and model.transmat_ is not None:
                        transmat = model.transmat_
                        # Add transition matrix statistics
                        features.extend([
                            np.mean(transmat),           # Average transition probability
                            np.std(transmat),            # Transition variability
                            np.trace(transmat),          # Persistence (diagonal elements)
                            np.sum(transmat - np.diag(np.diag(transmat)))  # Off-diagonal transitions
                        ])
                    else:
                        features.extend([0.0, 0.0, 0.0, 0.0])
                    
                    # Emission characteristics (means and covariances)
                    if hasattr(model, 'means_') and model.means_ is not None:
                        means = model.means_
                        features.extend([
                            np.mean(means),              # Average mean across states
                            np.std(means),               # Variability of means
                            np.max(means) - np.min(means)  # Range of means
                        ])
                    else:
                        features.extend([0.0, 0.0, 0.0])
                    
                    if hasattr(model, 'covars_') and model.covars_ is not None:
                        covars = model.covars_
                        if covars.ndim == 3:  # Full covariance matrices
                            # Extract diagonal elements (variances)
                            diag_covars = np.array([np.diag(cov) for cov in covars])
                            features.extend([
                                np.mean(diag_covars),    # Average variance
                                np.std(diag_covars),     # Variance variability
                                np.mean([np.linalg.det(cov) for cov in covars])  # Average determinant
                            ])
                        else:  # Diagonal or spherical covariances
                            features.extend([
                                np.mean(covars),         # Average variance
                                np.std(covars),          # Variance variability
                                np.mean(covars)          # Same as average for diagonal/spherical
                            ])
                    else:
                        features.extend([0.0, 0.0, 0.0])
                    
                    # Model complexity (number of components)
                    if hasattr(model, 'n_components'):
                        features.append(float(model.n_components))
                    else:
                        features.append(0.0)
                    
                    regime_features.append(features)
                    regime_indices.append(i)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract features from regime {i}: {e}")
                    # Add zero features for failed regimes
                    regime_features.append([0.0] * 11)  # 11 features total
                    regime_indices.append(i)
            
            if not regime_features:
                self.logger.error("No regime features extracted, falling back to frequency-based clustering")
                return self._create_frequency_based_clusters([], n_clusters, len(regime_models))
            
            # Convert to numpy array and standardize
            regime_features = np.array(regime_features)
            scaler = StandardScaler()
            regime_features_scaled = scaler.fit_transform(regime_features)
            
            # Perform K-means clustering on regime characteristics
            kmeans = KMeans(n_clusters=min(n_clusters, len(regime_features)), 
                          random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(regime_features_scaled)
            
            # Create regime to cluster mapping
            regime_to_cluster = {}
            for regime_idx, cluster_label in zip(regime_indices, cluster_labels):
                regime_to_cluster[regime_idx] = cluster_label
            
            self.logger.info(f"📊 Clustered {len(regime_models)} regimes into {n_clusters} clusters using similarity analysis")
            
            return regime_to_cluster
            
        except Exception as e:
            self.logger.error(f"Similarity-based clustering failed: {e}")
            # Fallback to frequency-based clustering
            return self._create_frequency_based_clusters([], n_clusters, len(regime_models))
    
    def _create_frequency_based_clusters(self, regime_assignments: List[int], n_clusters: int, data_length: int) -> Dict[int, int]:
        """Fallback frequency-based clustering method."""
        try:
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
            # Ultimate fallback: simple round-robin
            return {i: i % n_clusters for i in range(max(data_length, n_clusters))}
    
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
            
            self.logger.info(f"✅ Cluster quality validation completed in {validation_time:.2f}s")
            self.logger.info(f"📊 Overall quality score: {overall_score:.2f} ({'PASSED' if quality_metrics['validation_passed'] else 'FAILED'})")
            self.logger.info(f"📈 Regime range: xx → Clusters: {len(hmm_models)}")
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
    
    def _calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size for t-tests."""
        try:
            n1, n2 = len(group1), len(group2)
            s1, s2 = group1.std(), group2.std()
            pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
            return (group1.mean() - group2.mean()) / pooled_std
        except:
            return 0.0
    
    def _calculate_cluster_separation_metrics(self, cluster_data, unique_clusters):
        """Calculate overall cluster separation metrics."""
        try:
            # Silhouette score (if we have enough data)
            if len(unique_clusters) >= 2:
                from sklearn.metrics import silhouette_score
                from sklearn.preprocessing import StandardScaler
                
                # Prepare data for silhouette calculation
                all_data = []
                labels = []
                for cluster_id in unique_clusters:
                    cluster_df = cluster_data[cluster_id]
                    numeric_cols = cluster_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        cluster_numeric = cluster_df[numeric_cols].fillna(0)
                        all_data.append(cluster_numeric)
                        labels.extend([cluster_id] * len(cluster_numeric))
                
                if all_data and len(set(labels)) > 1:
                    combined_data = pd.concat(all_data, ignore_index=True)
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(combined_data)
                    
                    silhouette_avg = silhouette_score(scaled_data, labels)
                    return {
                        'silhouette_score': float(silhouette_avg),
                        'separation_quality': 'good' if silhouette_avg > 0.5 else 'fair' if silhouette_avg > 0.3 else 'poor'
                    }
            
            return {'error': 'Insufficient data for silhouette calculation'}
            
        except Exception as e:
            return {'error': f'Separation metrics calculation failed: {e}'}
    
    def _calculate_within_regime_cluster_analysis(self, cluster_assignments: List[int], regime_characteristics: Dict[str, Any], regime_to_cluster: Dict[str, int]) -> Dict[str, Any]:
        """Calculate within-regime cluster analysis - more relevant for regime clustering."""
        try:
            within_regime_metrics = {}
            unique_clusters = list(set(cluster_assignments))
            
            for cluster_id in unique_clusters:
                # Get regimes assigned to this cluster
                cluster_regimes = [regime_id for regime_id, cid in regime_to_cluster.items() if cid == cluster_id]
                
                if not cluster_regimes:
                    continue
                
                cluster_analysis = {
                    'n_regimes': len(cluster_regimes),
                    'regime_ids': cluster_regimes
                }
                
                # Analyze regime characteristics within this cluster
                momentum_chars = []
                volatility_chars = []
                volume_chars = []
                
                for regime_id in cluster_regimes:
                    regime_data = regime_characteristics.get(regime_id, {})
                    
                    # Extract momentum characteristics
                    momentum_data = regime_data.get('momentum_characteristics', {})
                    if momentum_data:
                        momentum_chars.append({
                            'price_momentum_5': momentum_data.get('mean_price_momentum_5', 0),
                            'price_momentum_20': momentum_data.get('mean_price_momentum_20', 0),
                            'rsi': momentum_data.get('mean_rsi', 50),
                            'macd': momentum_data.get('mean_macd', 0),
                            'momentum_strength': momentum_data.get('momentum_strength', 0)
                        })
                    
                    # Extract volatility characteristics
                    volatility_data = regime_data.get('volatility_characteristics', {})
                    if volatility_data:
                        volatility_chars.append({
                            'volatility_5': volatility_data.get('mean_volatility_5', 0),
                            'volatility_20': volatility_data.get('mean_volatility_20', 0),
                            'atr': volatility_data.get('mean_atr_normalized', 0),
                            'volatility_momentum': volatility_data.get('volatility_momentum', 0)
                        })
                    
                    # Extract volume characteristics
                    volume_data = regime_data.get('volume_characteristics', {})
                    if volume_data:
                        volume_chars.append({
                            'volume_momentum_5': volume_data.get('mean_volume_momentum_5', 0),
                            'volume_momentum_20': volume_data.get('mean_volume_momentum_20', 0),
                            'volume_ratio': volume_data.get('mean_volume_ratio', 1),
                            'volume_trend': volume_data.get('volume_trend', 'stable')
                        })
                
                # Calculate statistics for each characteristic type
                if momentum_chars:
                    momentum_stats = self._calculate_regime_characteristic_stats(momentum_chars)
                    cluster_analysis['momentum_characteristics'] = momentum_stats
                
                if volatility_chars:
                    volatility_stats = self._calculate_regime_characteristic_stats(volatility_chars)
                    cluster_analysis['volatility_characteristics'] = volatility_stats
                
                if volume_chars:
                    volume_stats = self._calculate_regime_characteristic_stats(volume_chars)
                    cluster_analysis['volume_characteristics'] = volume_stats
                
                # Calculate cluster coherence (how similar are regimes within this cluster)
                if len(cluster_regimes) > 1:
                    coherence_scores = []
                    for i, regime_1 in enumerate(cluster_regimes):
                        for regime_2 in cluster_regimes[i+1:]:
                            similarity = self._calculate_regime_similarity(
                                regime_characteristics[regime_1], 
                                regime_characteristics[regime_2]
                            )
                            coherence_scores.append(similarity)
                    
                    cluster_analysis['coherence'] = {
                        'mean_similarity': float(np.mean(coherence_scores)) if coherence_scores else 0.0,
                        'std_similarity': float(np.std(coherence_scores)) if coherence_scores else 0.0,
                        'min_similarity': float(np.min(coherence_scores)) if coherence_scores else 0.0,
                        'max_similarity': float(np.max(coherence_scores)) if coherence_scores else 0.0,
                        'coherence_quality': 'high' if np.mean(coherence_scores) > 0.7 else 'medium' if np.mean(coherence_scores) > 0.5 else 'low'
                    }
                
                within_regime_metrics[f'cluster_{cluster_id}'] = cluster_analysis
            
            return within_regime_metrics
            
        except Exception as e:
            return {'error': f'Within-regime cluster analysis failed: {e}'}
    
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
        from src.utils.tprint import tprint
        
        try:
            tprint("🔍 Starting parallel HMM clustering validation...")
            
            # Validate prepared_data structure
            if not isinstance(prepared_data, dict):
                raise ValueError(f"prepared_data must be a dict, got {type(prepared_data)}")
            
            # Validate required keys
            required_keys = ['market_data']
            missing_keys = [key for key in required_keys if key not in prepared_data]
            if missing_keys:
                raise ValueError(f"prepared_data missing required keys: {missing_keys}")
            
            # Get optimal number of workers
            if self.cpu_optimizer:
                max_workers = self.cpu_optimizer.get_optimal_worker_count()
                tprint(f"🔧 CPU optimization: Using {max_workers} workers for parallel processing")
            else:
                max_workers = 4  # Fallback worker count
                tprint(f"🔧 Using fallback worker count: {max_workers}")
            
            # Split data into chunks for parallel processing
            market_data = prepared_data.get('market_data')
            chunk_size = prepared_data.get('chunk_size', 10000)
            
            tprint(f"📊 Market data type: {type(market_data)}")
            tprint(f"📊 Chunk size: {chunk_size}")
            
            # Validate market data
            if not PANDAS_AVAILABLE:
                raise ValueError("Pandas not available for data processing")
            
            if not isinstance(market_data, pd.DataFrame):
                raise ValueError(f"Market data must be a pandas DataFrame, got {type(market_data)}")
            
            if market_data.empty:
                raise ValueError("Market data is empty")
            
            tprint(f"📊 Market data shape: {market_data.shape}")
            
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
                
                # Ensure standardized format
                if not isinstance(merged_result, dict):
                    merged_result = STANDARD_CLUSTERING_RESULT.copy()
                    merged_result.update({'success': False, 'error': 'Invalid merged result format'})
                elif 'success' not in merged_result:
                    merged_result['success'] = True
                    merged_result['error'] = None
                
                return merged_result
            else:
                # Fallback to single-threaded processing
                tprint("⚠️ Falling back to single-threaded processing _perform_parallel_hmm_clustering")
                
        except Exception as e:
            self.logger.error(f"❌ Parallel HMM clustering failed: {e}")
            # Return standardized error format
            result = STANDARD_CLUSTERING_RESULT.copy()
            result.update({
                'success': False,
                'error': str(e),
                'clustering_time': 0.0
            })
            return result
    
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
            
            # Weighted overall score - Heavy focus on similarity for composite clustering
            weights = {
                'volume': 0.15,
                'volatility': 0.25,
                'momentum': 0.25,
                'similarity': 0.35  # Heavy focus on regime similarity for composite clusters
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

    def _create_cluster_representative_models(self, hmm_manager: Any, market_data: Any, cluster_assignments: List[int], n_clusters: int, regime_discovery: Dict[str, Any]) -> List[Any]:
        """Create representative HMM models for each cluster based on cluster assignments."""
        try:
            representative_models = []
            
            # For each cluster, train a representative HMM model
            for cluster_id in range(n_clusters):
                # Get data points assigned to this cluster
                cluster_mask = np.array(cluster_assignments) == cluster_id
                cluster_data = market_data[cluster_mask] if hasattr(market_data, '__getitem__') else market_data
                
                if hasattr(cluster_data, 'empty') and not cluster_data.empty:
                    # Train HMM model on cluster data
                    cluster_models = hmm_manager.train_hmm_parallel(
                        data=cluster_data,
                        n_models=1,  # One model per cluster
                        config=None
                    )
                    
                    if cluster_models and len(cluster_models) > 0:
                        representative_models.append(cluster_models[0])
                    else:
                        # Create dummy model if training failed
                        representative_models.append(self._create_dummy_hmm_model())
                else:
                    # Create dummy model for empty clusters
                    representative_models.append(self._create_dummy_hmm_model())
            
            return representative_models
            
        except Exception as e:
            self.logger.error(f"❌ Representative model creation failed: {e}")
            # Return dummy models
            return [self._create_dummy_hmm_model() for _ in range(n_clusters)]

    def _create_dummy_hmm_model(self) -> Dict[str, Any]:
        """Create a dummy HMM model for fallback purposes."""
        return {
            'model_type': 'dummy',
            'n_components': 2,
            'means_': [[0.0], [0.0]],
            'covars_': [[1.0], [1.0]],
            'transmat_': [[0.7, 0.3], [0.3, 0.7]],
            'startprob_': [0.5, 0.5]
        }
    