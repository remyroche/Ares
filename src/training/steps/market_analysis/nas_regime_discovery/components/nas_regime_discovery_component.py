"""
NAS Regime Discovery Component for HMM pipeline replacement.

This component provides NAS-driven regime discovery that replaces the existing
HMM regime discovery with enhanced capabilities.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, NamedTuple
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Import NAS clustering components
from ...nas_clustering import NASOrchestrator, NASClusteringConfig

logger = logging.getLogger(__name__)


class NASRegimeDiscoveryComponent:
    """
    NAS Regime Discovery Component for HMM pipeline replacement.
    
    This component provides NAS-driven regime discovery that replaces the existing
    HMM regime discovery with enhanced capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NAS regime discovery component.
        
        Args:
            config: Component configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize NAS configuration
        self.nas_config = NASClusteringConfig.create_short_term_trading_config()
        
        # Update configuration with provided values
        if 'nas_config' in config:
            self.nas_config.update_config(config['nas_config'])
        
        # Initialize NAS orchestrator
        self.nas_orchestrator = NASOrchestrator({
            'symbol': config.get('symbol', 'BTCUSDT'),
            'exchange': config.get('exchange', 'binance'),
            'timeframe': config.get('timeframe', '15m'),
            'data_dir': config.get('data_dir', 'historical_data'),
            'nas_config': self.nas_config.__dict__
        })
        
        # Component metadata
        self.component_name = "nas_regime_discovery"
        self.component_version = "1.0.0"
        
        self.logger.info(f"✅ NAS Regime Discovery Component initialized for {self.nas_config.timeframe} timeframe")
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return [
            'nas_regime_discovery_result',
            'nas_regime_models',
            'nas_regime_assignments',
            'nas_regime_metrics',
            'nas_economic_significance',
            'nas_trading_viability'
        ]
    
    def get_component_name(self) -> str:
        """Get the component name."""
        return self.component_name
    
    def validate_config(self) -> bool:
        """Validate component configuration."""
        try:
            # Validate NAS configuration
            if not self.nas_config.validate_config():
                self.logger.error("❌ NAS configuration validation failed")
                return False
            
            # Check required parameters
            required_params = ['symbol', 'exchange', 'timeframe', 'data_dir']
            for param in required_params:
                if param not in self.config:
                    self.logger.error(f"❌ Missing required config parameter: {param}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            return False
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute NAS regime discovery component.
        
        Args:
            data: Market data for regime discovery
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary with NAS regime discovery results
        """
        start_time = time.time()
        
        try:
            self.logger.info("🎯 Starting NAS regime discovery component execution")
            
            # Validate configuration
            if not self.validate_config():
                raise ValueError("Component configuration validation failed")
            
            # Validate inputs
            validation_result = await self._validate_inputs(data, pipeline_state)
            if not validation_result.is_valid:
                raise ValueError(f"Input validation failed: {validation_result.error_message}")
            
            # Execute NAS regime discovery
            regime_discovery_result = await self._execute_nas_regime_discovery(
                validation_result.market_data, pipeline_state
            )
            
            # Format output for pipeline compatibility
            formatted_result = self._format_output_for_pipeline(
                regime_discovery_result, pipeline_state
            )
            
            execution_time = time.time() - start_time
            self.logger.info(f"✅ NAS regime discovery component completed in {execution_time:.2f}s")
            
            return formatted_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ NAS regime discovery component failed after {execution_time:.2f}s: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _validate_inputs(self, data: Any, pipeline_state: Dict[str, Any]) -> NamedTuple:
        """Validate input data and pipeline state.
        
        Args:
            data: Market data for regime discovery
            pipeline_state: Current pipeline state
            
        Returns:
            ValidationResult with market data and validation status
        """
        try:
            self.logger.info("🔍 Validating inputs for NAS regime discovery")
            
            # Check if we have market data
            if data is None:
                return ValidationResult(
                    is_valid=False,
                    error_message="No market data provided for NAS regime discovery"
                )
            
            # Validate data format
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    return ValidationResult(
                        is_valid=False,
                        error_message="Market data DataFrame is empty"
                    )
                market_data = data
            elif isinstance(data, np.ndarray):
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
            
            # Check for required columns in DataFrame
            if isinstance(market_data, pd.DataFrame):
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                missing_columns = [col for col in required_columns if col not in market_data.columns]
                if missing_columns:
                    self.logger.warning(f"⚠️ Missing columns: {missing_columns}")
                    # Continue with available data
            
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
    
    async def _execute_nas_regime_discovery(self, market_data: Any, 
                                         pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute NAS regime discovery on market data.
        
        Args:
            market_data: Market data for regime discovery
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary with NAS regime discovery results
        """
        try:
            self.logger.info("🚀 Executing NAS regime discovery")
            
            # Prepare data for NAS clustering
            if isinstance(market_data, pd.DataFrame):
                data_array = market_data[['open', 'high', 'low', 'close', 'volume']].values
                timestamps = market_data.index.values
            else:
                data_array = market_data
                timestamps = np.arange(len(market_data))
            
            # Run NAS clustering for regime discovery
            nas_results = await self.nas_orchestrator.run_nas_clustering(
                data=data_array,
                timestamps=timestamps,
                symbol=self.config.get('symbol', 'BTCUSDT'),
                exchange=self.config.get('exchange', 'binance'),
                timeframe=self.config.get('timeframe', '15m')
            )
            
            if not nas_results['success']:
                raise RuntimeError(f"NAS regime discovery failed: {nas_results.get('error', 'Unknown error')}")
            
            self.logger.info("✅ NAS regime discovery completed successfully")
            return nas_results
            
        except Exception as e:
            self.logger.error(f"❌ NAS regime discovery execution failed: {e}")
            raise
    
    def _format_output_for_pipeline(self, regime_discovery_result: Dict[str, Any],
                                  pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Format NAS regime discovery result for pipeline compatibility.
        
        Args:
            regime_discovery_result: NAS regime discovery result
            pipeline_state: Current pipeline state
            
        Returns:
            Formatted result dictionary
        """
        try:
            # Extract clustering result
            clustering_result = regime_discovery_result.get('clustering_result')
            if not clustering_result:
                raise ValueError("No clustering result found in regime discovery")
            
            # Create regime models
            regime_models = self._create_regime_models(clustering_result)
            
            # Create regime assignments
            regime_assignments = self._create_regime_assignments(clustering_result)
            
            # Create regime metrics
            regime_metrics = self._create_regime_metrics(clustering_result)
            
            # Create formatted result
            formatted_result = {
                'success': regime_discovery_result['success'],
                'execution_time': regime_discovery_result['execution_time'],
                'timestamp': regime_discovery_result['timestamp'],
                'method': 'nas_regime_discovery',
                'timeframe': self.nas_config.timeframe,
                'n_regimes': self.nas_config.n_regimes,
                
                # Regime discovery results
                'regime_models': regime_models,
                'regime_assignments': regime_assignments,
                'regime_metrics': regime_metrics,
                
                # NAS-specific results
                'nas_architectures': clustering_result.nas_architectures,
                'micro_regimes': regime_discovery_result.get('micro_regime_result'),
                'economic_significance_scores': clustering_result.economic_significance_scores.tolist(),
                'trading_viability_scores': clustering_result.trading_viability_scores.tolist(),
                
                # Pipeline integration fields
                'pipeline_replacement': True,
                'hmm_replacement': True,
                'regime_data_available': True,
                'ml_training_ready': True
            }
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"❌ Output formatting failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': regime_discovery_result.get('execution_time', 0.0),
                'timestamp': regime_discovery_result.get('timestamp', datetime.now().isoformat())
            }
    
    def _create_regime_models(self, clustering_result: Any) -> Dict[str, Any]:
        """Create regime models from clustering result."""
        try:
            return {
                'nas_architectures': clustering_result.nas_architectures,
                'cluster_centers': clustering_result.cluster_centers.tolist(),
                'regime_statistics': clustering_result.statistics,
                'regime_quality_metrics': clustering_result.quality_metrics,
                'regime_validation': clustering_result.validation,
                'regime_metadata': clustering_result.metadata
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime models creation failed: {e}")
            return {}
    
    def _create_regime_assignments(self, clustering_result: Any) -> Dict[str, Any]:
        """Create regime assignments from clustering result."""
        try:
            return {
                'regime_labels': clustering_result.labels.tolist(),
                'regime_transitions': clustering_result.regime_transitions.tolist() if clustering_result.regime_transitions is not None else [],
                'economic_significance_scores': clustering_result.economic_significance_scores.tolist(),
                'trading_viability_scores': clustering_result.trading_viability_scores.tolist(),
                'n_regimes': len(np.unique(clustering_result.labels)),
                'regime_distribution': self._calculate_regime_distribution(clustering_result.labels)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime assignments creation failed: {e}")
            return {}
    
    def _create_regime_metrics(self, clustering_result: Any) -> Dict[str, Any]:
        """Create regime metrics from clustering result."""
        try:
            return {
                'silhouette_score': clustering_result.quality_metrics.get('silhouette_score', 0.0),
                'nas_score': clustering_result.quality_metrics.get('nas_score', 0.0),
                'calinski_harabasz_score': clustering_result.quality_metrics.get('calinski_harabasz_score', 0.0),
                'regime_stability': self._calculate_regime_stability(clustering_result.labels),
                'economic_significance_mean': float(np.mean(clustering_result.economic_significance_scores)),
                'trading_viability_mean': float(np.mean(clustering_result.trading_viability_scores)),
                'execution_time': clustering_result.execution_time
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime metrics creation failed: {e}")
            return {}
    
    def _calculate_regime_distribution(self, labels: np.ndarray) -> Dict[str, int]:
        """Calculate regime distribution."""
        try:
            unique_labels = np.unique(labels)
            distribution = {}
            
            for label in unique_labels:
                count = np.sum(labels == label)
                distribution[f'regime_{label}'] = int(count)
            
            return distribution
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime distribution calculation failed: {e}")
            return {}
    
    def _calculate_regime_stability(self, labels: np.ndarray) -> float:
        """Calculate regime stability."""
        try:
            if len(labels) < 2:
                return 0.0
            
            # Calculate regime changes
            regime_changes = np.sum(np.diff(labels) != 0)
            total_periods = len(labels) - 1
            
            # Stability is inverse of change frequency
            stability = 1.0 - (regime_changes / total_periods) if total_periods > 0 else 0.0
            
            return float(stability)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime stability calculation failed: {e}")
            return 0.0
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information."""
        return {
            'component_name': self.component_name,
            'component_version': self.component_version,
            'component_type': 'nas_regime_discovery',
            'description': 'NAS-driven regime discovery for short-term trading regime detection',
            'timeframe': self.nas_config.timeframe,
            'n_regimes': self.nas_config.n_regimes,
            'nas_architecture_type': self.nas_config.nas_architecture_type.value,
            'micro_regime_detection': self.nas_config.enable_micro_regime_detection,
            'required_artifacts': self.get_required_artifacts(),
            'features': [
                'NAS-driven regime discovery',
                'Short-term trading optimization (5-30m)',
                'Micro-regime detection',
                'Economic significance scoring',
                'Trading viability assessment',
                'HMM pipeline replacement',
                'ML model training support'
            ]
        }


# Validation result class
class ValidationResult(NamedTuple):
    """Result of input validation."""
    is_valid: bool
    error_message: Optional[str] = None
    market_data: Optional[Any] = None