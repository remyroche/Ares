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

# Import the optimal regime clustering orchestrator from the new structure
try:
    from ..integration.orchestrator import run_optimal_clustering
    OPTIMAL_REGIME_CLUSTERING_AVAILABLE = True
except ImportError:
    OPTIMAL_REGIME_CLUSTERING_AVAILABLE = False
    run_optimal_clustering = None

# Import base component classes
try:
    from ...components.base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
except ImportError:
    # Fallback if base component not available
    class BaseMarketAnalysisComponent:
        def __init__(self, config):
            self.config = config
            self.logger = logging.getLogger(self.__class__.__name__)
    
    class ComponentConfig:
        pass
    
    class ComponentResult:
        def __init__(self, success=True, artifacts=None, metadata=None, error_message=None):
            self.success = success
            self.artifacts = artifacts or {}
            self.metadata = metadata or {}
            self.error_message = error_message

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
        try:
            self.logger.info("🔍 Validating inputs for optimal regime clustering")

            # Check if we have market data
            if data is None:
                return ValidationResult(
                    is_valid=False,
                    error_message="No market data provided for clustering"
                )

            # Validate data format
            if PANDAS_AVAILABLE and isinstance(data, pd.DataFrame):
                if data.empty:
                    return ValidationResult(
                        is_valid=False,
                        error_message="Market data DataFrame is empty"
                    )
                market_data = data
            elif NUMPY_AVAILABLE and isinstance(data, np.ndarray):
                if data.size == 0:
                    return ValidationResult(
                        is_valid=False,
                        error_message="Market data array is empty"
                    )
                market_data = data
            else:
                return ValidationResult(
                    is_valid=False,
                    error_message=f"Unsupported data type: {type(data)}"
                )

            # Check for HMM regime discovery results in pipeline state
            if 'hmm_regime_discovery_result' not in pipeline_state:
                self.logger.warning("⚠️ No HMM regime discovery results found in pipeline state")
                # Continue anyway as we might have regime data in the market data

            self.logger.info("✅ Input validation successful")
            return ValidationResult(
                is_valid=True,
                market_data=market_data
            )

        except Exception as e:
            self.logger.error(f"❌ Input validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                error_message=f"Input validation error: {str(e)}"
            )

    async def _execute_clustering(self, market_data: Any, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the optimal regime clustering.

        Args:
            market_data: Market data for clustering
            pipeline_state: Current pipeline state

        Returns:
            Dictionary with clustering results
        """
        try:
            self.logger.info("🚀 Executing optimal regime clustering")

            # Prepare clustering configuration
            clustering_config = self._prepare_clustering_config(pipeline_state)

            # Execute clustering using the orchestrator
            clustering_result = await asyncio.get_event_loop().run_in_executor(
                None, run_optimal_clustering, market_data, clustering_config
            )

            if not clustering_result.get('success', False):
                raise RuntimeError(f"Clustering failed: {clustering_result.get('error_message', 'Unknown error')}")

            self.logger.info("✅ Optimal regime clustering completed successfully")
            return clustering_result

        except Exception as e:
            self.logger.error(f"❌ Clustering execution failed: {e}")
            raise

    def _prepare_clustering_config(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare clustering configuration from component config and pipeline state.

        Args:
            pipeline_state: Current pipeline state

        Returns:
            Clustering configuration dictionary
        """
        try:
            # Base configuration from component config
            config = {
                'symbol': getattr(self.config, 'symbol', 'BTCUSDT'),
                'exchange': getattr(self.config, 'exchange', 'binance'),
                'timeframe': getattr(self.config, 'timeframe', '15m'),
                'data_dir': getattr(self.config, 'data_dir', 'historical_data'),
                'use_matrix_optimization': True,
                'use_enhanced_clustering': True,
                'enable_fast_fail': True,
                'timeout_seconds': 300,
                'memory_limit_gb': 8.0,
                'quality_threshold': 0.3
            }

            # Add any custom parameters from component config
            if hasattr(self.config, 'custom_params') and self.config.custom_params:
                config.update(self.config.custom_params)

            # Add any relevant information from pipeline state
            if 'hmm_regime_discovery_result' in pipeline_state:
                config['hmm_regime_data'] = pipeline_state['hmm_regime_discovery_result']

            return config

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to prepare clustering config: {e}")
            return {}

    def _create_component_result(self, clustering_result: Dict[str, Any]) -> ComponentResult:
        """
        Create component result from clustering results.

        Args:
            clustering_result: Results from clustering execution

        Returns:
            ComponentResult object
        """
        try:
            # Extract key results
            success = clustering_result.get('success', False)
            
            if success:
                # Create comprehensive artifacts
                artifacts = {
                    'optimal_regime_clustering_result': {
                        'success': success,
                        'execution_time': clustering_result.get('execution_time', 0.0),
                        'timestamp': clustering_result.get('timestamp', datetime.now().isoformat()),
                        
                        # Standard clustering results
                        'standard_clustering': clustering_result.get('standard_clustering', {}),
                        
                        # Enhanced clustering results
                        'enhanced_clustering': clustering_result.get('enhanced_clustering', {}),
                        
                        # Comprehensive metrics
                        'comprehensive_metrics': clustering_result.get('comprehensive_metrics', {}),
                        
                        # Metrics evolution report
                        'metrics_evolution_report': clustering_result.get('metrics_evolution_report', {}),
                        
                        # Configuration used
                        'configuration': clustering_result.get('configuration', {})
                    }
                }

                # Create metadata
                metadata = {
                    'component_name': self.component_name,
                    'execution_time': clustering_result.get('execution_time', 0.0),
                    'timestamp': clustering_result.get('timestamp', datetime.now().isoformat()),
                    'clustering_success': success,
                    'standard_clustering_success': clustering_result.get('standard_clustering', {}).get('success', False),
                    'enhanced_clustering_success': clustering_result.get('enhanced_clustering', {}).get('success', False)
                }

                self.logger.info("✅ Component result created successfully")
                return ComponentResult(
                    success=True,
                    artifacts=artifacts,
                    metadata=metadata
                )
            else:
                error_message = clustering_result.get('error_message', 'Clustering failed')
                self.logger.error(f"❌ Clustering failed: {error_message}")
                
                return ComponentResult(
                    success=False,
                    artifacts={},
                    metadata={
                        'component_name': self.component_name,
                        'error': error_message,
                        'timestamp': datetime.now().isoformat()
                    },
                    error_message=error_message
                )

        except Exception as e:
            self.logger.error(f"❌ Failed to create component result: {e}")
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={
                    'component_name': self.component_name,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                },
                error_message=str(e)
            )

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute the optimal regime clustering component.

        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state

        Returns:
            ComponentResult with clustering results
        """
        start_time = time.time()
        
        try:
            self.logger.info("🎯 Starting optimal regime clustering component execution")

            # Validate configuration
            if not self.validate_config():
                raise ValueError("Component configuration validation failed")

            # Validate inputs
            validation_result = await self._validate_inputs(data, pipeline_state)
            if not validation_result.is_valid:
                raise ValueError(f"Input validation failed: {validation_result.error_message}")

            # Execute clustering
            clustering_result = await self._execute_clustering(validation_result.market_data, pipeline_state)

            # Create component result
            result = self._create_component_result(clustering_result)

            execution_time = time.time() - start_time
            self.logger.info(f"✅ Optimal regime clustering component completed in {execution_time:.2f}s")

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Optimal regime clustering component failed after {execution_time:.2f}s: {e}")
            
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={
                    'component_name': self.component_name,
                    'execution_time': execution_time,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                },
                error_message=str(e)
            )

    def get_component_info(self) -> Dict[str, Any]:
        """Get component information."""
        return {
            'component_name': self.component_name,
            'component_type': 'optimal_regime_clustering',
            'version': '1.0.0',
            'description': 'Advanced optimal regime clustering with matrix optimization and enhanced 4D frontier optimization',
            'required_artifacts': self.get_required_artifacts(),
            'dependencies': {
                'numpy': NUMPY_AVAILABLE,
                'pandas': PANDAS_AVAILABLE,
                'optimal_regime_clustering': OPTIMAL_REGIME_CLUSTERING_AVAILABLE
            },
            'features': [
                'Matrix-optimized clustering with GPU acceleration',
                'Enhanced clustering with 4D frontier optimization',
                'Comprehensive metrics tracking and evolution reporting',
                'Fast fail mechanisms for quality assurance',
                'Hardware acceleration and memory optimization'
            ]
        }


def create_optimal_regime_clustering_component(config) -> OptimalRegimeClusteringComponent:
    """Create an optimal regime clustering component instance.

    Args:
        config: Component configuration

    Returns:
        OptimalRegimeClusteringComponent instance
    """
    return OptimalRegimeClusteringComponent(config)