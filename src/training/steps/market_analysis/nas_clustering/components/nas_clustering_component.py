"""
NAS Clustering Component for HMM pipeline replacement.

This component provides NAS-driven clustering that replaces the existing
HMM clustering pipeline with enhanced capabilities.
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

from ..core.nas_clusterer import NASClusterer, NASClusteringResult
from ..core.nas_config import NASClusteringConfig, NASConfig
from ..core.nas_feature_extractor import NASFeatureExtractor
from ..core.micro_regime_detector import MicroRegimeDetector

logger = logging.getLogger(__name__)


class NASClusteringComponent:
    """
    NAS Clustering Component for HMM pipeline replacement.
    
    This component provides NAS-driven clustering that replaces the existing
    HMM clustering pipeline with enhanced capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NAS clustering component.
        
        Args:
            config: Component configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize NAS clustering configuration
        self.nas_config = NASClusteringConfig.create_short_term_trading_config()
        
        # Update configuration with provided values
        if 'nas_config' in config:
            self.nas_config.update_config(config['nas_config'])
        
        # Initialize NAS clusterer
        self.nas_clusterer = NASClusterer(self.nas_config)
        
        # Component metadata
        self.component_name = "nas_clustering"
        self.component_version = "1.0.0"
        
        self.logger.info(f"✅ NAS Clustering Component initialized for {self.nas_config.timeframe} timeframe")
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return [
            'nas_clustering_result',
            'nas_regime_data',
            'nas_micro_regimes',
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
        """Execute NAS clustering component.
        
        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary with NAS clustering results
        """
        start_time = time.time()
        
        try:
            self.logger.info("🎯 Starting NAS clustering component execution")
            
            # Validate configuration
            if not self.validate_config():
                raise ValueError("Component configuration validation failed")
            
            # Validate inputs
            validation_result = await self._validate_inputs(data, pipeline_state)
            if not validation_result.is_valid:
                raise ValueError(f"Input validation failed: {validation_result.error_message}")
            
            # Execute NAS clustering
            clustering_result = await self._execute_nas_clustering(
                validation_result.market_data, pipeline_state
            )
            
            # Format output for pipeline compatibility
            formatted_result = self._format_output_for_pipeline(
                clustering_result, pipeline_state
            )
            
            execution_time = time.time() - start_time
            self.logger.info(f"✅ NAS clustering component completed in {execution_time:.2f}s")
            
            return formatted_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ NAS clustering component failed after {execution_time:.2f}s: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            }
    
    async def _validate_inputs(self, data: Any, pipeline_state: Dict[str, Any]) -> NamedTuple:
        """Validate input data and pipeline state.
        
        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state
            
        Returns:
            ValidationResult with market data and validation status
        """
        try:
            self.logger.info("🔍 Validating inputs for NAS clustering")
            
            # Check if we have market data
            if data is None:
                return ValidationResult(
                    is_valid=False,
                    error_message="No market data provided for NAS clustering"
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
    
    async def _execute_nas_clustering(self, market_data: Any, 
                                    pipeline_state: Dict[str, Any]) -> NASClusteringResult:
        """Execute NAS clustering on market data.
        
        Args:
            market_data: Market data for clustering
            pipeline_state: Current pipeline state
            
        Returns:
            NASClusteringResult with clustering results
        """
        try:
            self.logger.info("🚀 Executing NAS clustering")
            
            # Prepare clustering configuration
            clustering_config = self._prepare_clustering_config(pipeline_state)
            
            # Execute NAS clustering
            clustering_result = self.nas_clusterer.cluster(
                market_data,
                timestamps=market_data.index if hasattr(market_data, 'index') else None,
                optimize_parameters=True,
                generate_report=True
            )
            
            if not clustering_result.success:
                raise RuntimeError(f"NAS clustering failed: {clustering_result.error_message}")
            
            self.logger.info("✅ NAS clustering completed successfully")
            return clustering_result
            
        except Exception as e:
            self.logger.error(f"❌ NAS clustering execution failed: {e}")
            raise
    
    def _prepare_clustering_config(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare clustering configuration from component config and pipeline state.
        
        Args:
            pipeline_state: Current pipeline state
            
        Returns:
            Clustering configuration dictionary
        """
        try:
            # Base configuration from component config
            config = {
                'symbol': self.config.get('symbol', 'BTCUSDT'),
                'exchange': self.config.get('exchange', 'binance'),
                'timeframe': self.config.get('timeframe', '15m'),
                'data_dir': self.config.get('data_dir', 'historical_data'),
                'nas_architecture_type': self.nas_config.nas_architecture_type.value,
                'n_regimes': self.nas_config.n_regimes,
                'min_regime_duration': self.nas_config.min_regime_duration,
                'max_regime_duration': self.nas_config.max_regime_duration,
                'enable_micro_regime_detection': self.nas_config.enable_micro_regime_detection,
                'economic_significance_threshold': self.nas_config.economic_significance_threshold,
                'trading_viability_threshold': self.nas_config.trading_viability_threshold
            }
            
            # Add any custom parameters from component config
            if 'custom_params' in self.config:
                config.update(self.config['custom_params'])
            
            # Add any relevant information from pipeline state
            if 'previous_clustering_result' in pipeline_state:
                config['previous_clustering_result'] = pipeline_state['previous_clustering_result']
            
            return config
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to prepare clustering config: {e}")
            return {}
    
    def _format_output_for_pipeline(self, clustering_result: NASClusteringResult,
                                   pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Format NAS clustering result for pipeline compatibility.
        
        Args:
            clustering_result: NAS clustering result
            pipeline_state: Current pipeline state
            
        Returns:
            Formatted result dictionary
        """
        try:
            # Create base result structure compatible with HMM clustering
            formatted_result = {
                'success': clustering_result.success,
                'execution_time': clustering_result.execution_time,
                'timestamp': clustering_result.timestamp,
                'method': 'nas_clustering',
                'timeframe': self.nas_config.timeframe,
                'n_regimes': self.nas_config.n_regimes,
                
                # Standard clustering results (HMM compatible)
                'labels': clustering_result.labels.tolist(),
                'cluster_centers': clustering_result.cluster_centers.tolist(),
                'statistics': clustering_result.statistics,
                'quality_metrics': clustering_result.quality_metrics,
                'validation': clustering_result.validation,
                'metadata': clustering_result.metadata,
                
                # NAS-specific results
                'nas_architectures': clustering_result.nas_architectures,
                'micro_regimes': {
                    'regimes': clustering_result.micro_regimes.micro_regimes.tolist() if clustering_result.micro_regimes else [],
                    'types': [t.value for t in clustering_result.micro_regimes.micro_regime_types] if clustering_result.micro_regimes else [],
                    'scores': clustering_result.micro_regimes.micro_regime_scores.tolist() if clustering_result.micro_regimes else [],
                    'detection_accuracy': clustering_result.micro_regimes.detection_accuracy if clustering_result.micro_regimes else 0.0
                },
                'regime_transitions': clustering_result.regime_transitions.tolist() if clustering_result.regime_transitions is not None else [],
                'economic_significance_scores': clustering_result.economic_significance_scores.tolist(),
                'trading_viability_scores': clustering_result.trading_viability_scores.tolist(),
                
                # Pipeline integration fields
                'pipeline_replacement': True,
                'hmm_replacement': True,
                'regime_data_available': True,
                'timestamped_regime_data': self._create_timestamped_regime_data(clustering_result)
            }
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"❌ Output formatting failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time': clustering_result.execution_time,
                'timestamp': clustering_result.timestamp
            }
    
    def _create_timestamped_regime_data(self, clustering_result: NASClusteringResult) -> Dict[str, Any]:
        """Create timestamped regime data for LM model training.
        
        Args:
            clustering_result: NAS clustering result
            
        Returns:
            Dictionary with timestamped regime data
        """
        try:
            # Create timestamped regime data structure
            timestamped_data = {
                'regime_labels': clustering_result.labels.tolist(),
                'regime_centers': clustering_result.cluster_centers.tolist(),
                'regime_statistics': clustering_result.statistics,
                'regime_quality_metrics': clustering_result.quality_metrics,
                'regime_validation': clustering_result.validation,
                'regime_metadata': clustering_result.metadata,
                
                # NAS-specific regime data
                'nas_architectures': clustering_result.nas_architectures,
                'micro_regimes': {
                    'regimes': clustering_result.micro_regimes.micro_regimes.tolist() if clustering_result.micro_regimes else [],
                    'types': [t.value for t in clustering_result.micro_regimes.micro_regime_types] if clustering_result.micro_regimes else [],
                    'scores': clustering_result.micro_regimes.micro_regime_scores.tolist() if clustering_result.micro_regimes else [],
                    'detection_accuracy': clustering_result.micro_regimes.detection_accuracy if clustering_result.micro_regimes else 0.0
                },
                'regime_transitions': clustering_result.regime_transitions.tolist() if clustering_result.regime_transitions is not None else [],
                'economic_significance_scores': clustering_result.economic_significance_scores.tolist(),
                'trading_viability_scores': clustering_result.trading_viability_scores.tolist(),
                
                # LM model training fields
                'lm_training_data': {
                    'regime_sequences': clustering_result.labels.tolist(),
                    'regime_transitions': clustering_result.regime_transitions.tolist() if clustering_result.regime_transitions is not None else [],
                    'economic_significance': clustering_result.economic_significance_scores.tolist(),
                    'trading_viability': clustering_result.trading_viability_scores.tolist(),
                    'micro_regime_sequences': clustering_result.micro_regimes.micro_regimes.tolist() if clustering_result.micro_regimes else [],
                    'micro_regime_types': [t.value for t in clustering_result.micro_regimes.micro_regime_types] if clustering_result.micro_regimes else []
                }
            }
            
            return timestamped_data
            
        except Exception as e:
            self.logger.error(f"❌ Timestamped regime data creation failed: {e}")
            return {}
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get component information."""
        return {
            'component_name': self.component_name,
            'component_version': self.component_version,
            'component_type': 'nas_clustering',
            'description': 'NAS-driven clustering for short-term trading regime detection',
            'timeframe': self.nas_config.timeframe,
            'n_regimes': self.nas_config.n_regimes,
            'nas_architecture_type': self.nas_config.nas_architecture_type.value,
            'micro_regime_detection': self.nas_config.enable_micro_regime_detection,
            'required_artifacts': self.get_required_artifacts(),
            'features': [
                'NAS-driven regime detection',
                'Short-term trading optimization (5-30m)',
                'Micro-regime detection',
                'Economic significance scoring',
                'Trading viability assessment',
                'Full pipeline compatibility',
                'LM model training support'
            ]
        }


# Validation result class
class ValidationResult(NamedTuple):
    """Result of input validation."""
    is_valid: bool
    error_message: Optional[str] = None
    market_data: Optional[Any] = None