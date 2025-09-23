"""
NAS Clustering Pipeline Component for HMM pipeline replacement.

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

# Import NAS clustering components
from ..core.nas_clusterer import NASClusterer, NASClusteringResult
from ..core.nas_config import NASClusteringConfig
from ..core.nas_feature_extractor import NASFeatureExtractor
from ..core.micro_regime_detector import MicroRegimeDetector
from ..core.nas_regime_optimizer import NASRegimeOptimizer
from ..utils.nas_detailed_reporter import NASDetailedReporter

logger = logging.getLogger(__name__)


class NASClusteringPipelineComponent:
    """
    NAS Clustering Pipeline Component for HMM pipeline replacement.
    
    This component provides NAS-driven clustering that replaces the existing
    HMM clustering pipeline with enhanced capabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NAS clustering pipeline component.
        
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
        
        # Initialize NAS components
        self.nas_clusterer = NASClusterer(self.nas_config)
        self.feature_extractor = NASFeatureExtractor(self.nas_config.get_feature_config())
        self.micro_regime_detector = MicroRegimeDetector(self.nas_config.get_micro_regime_config())
        self.regime_optimizer = NASRegimeOptimizer({
            'min_regimes': 5,
            'max_regimes': 20,
            'optimization_methods': ['silhouette', 'calinski_harabasz', 'davies_bouldin'],
            'quality_threshold': 0.6,
            'stability_threshold': 0.7,
            'enable_data_analysis': True,
            'enable_volatility_analysis': True,
            'enable_trend_analysis': True,
            'enable_volume_analysis': True
        })
        self.detailed_reporter = NASDetailedReporter({
            'enable_detailed_analysis': True,
            'enable_economic_reporting': True,
            'enable_trading_reporting': True,
            'enable_ml_training_reporting': True,
            'enable_micro_regime_reporting': True,
            'output_format': 'json',
            'include_recommendations': True
        })
        
        # Component metadata
        self.component_name = "nas_clustering"
        self.component_version = "1.0.0"
        
        self.logger.info(f"✅ NAS Clustering Pipeline Component initialized for {self.nas_config.timeframe} timeframe")
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return [
            'optimal_regime_clustering_result',
            'nas_clustering_result',
            'nas_regime_data',
            'nas_micro_regimes',
            'nas_economic_significance',
            'nas_trading_viability',
            'nas_ml_training_features',
            'nas_detailed_report'
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
        """Execute NAS clustering pipeline component.
        
        Args:
            data: Market data for clustering
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary with NAS clustering results
        """
        start_time = time.time()
        
        try:
            self.logger.info("🎯 Starting NAS clustering pipeline component execution")
            
            # Validate configuration
            if not self.validate_config():
                raise ValueError("Component configuration validation failed")
            
            # Validate inputs
            validation_result = await self._validate_inputs(data, pipeline_state)
            if not validation_result.is_valid:
                raise ValueError(f"Input validation failed: {validation_result.error_message}")
            
            # Execute NAS clustering pipeline
            clustering_result = await self._execute_nas_clustering_pipeline(
                validation_result.market_data, pipeline_state
            )
            
            # Format output for pipeline compatibility
            formatted_result = self._format_output_for_pipeline(
                clustering_result, pipeline_state
            )
            
            execution_time = time.time() - start_time
            self.logger.info(f"✅ NAS clustering pipeline component completed in {execution_time:.2f}s")
            
            return formatted_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ NAS clustering pipeline component failed after {execution_time:.2f}s: {e}")
            
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
    
    async def _execute_nas_clustering_pipeline(self, market_data: Any, 
                                            pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute NAS clustering pipeline on market data.
        
        Args:
            market_data: Market data for clustering
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary with NAS clustering results
        """
        try:
            self.logger.info("🚀 Executing NAS clustering pipeline")
            
            # Prepare data for NAS clustering
            if isinstance(market_data, pd.DataFrame):
                data_array = market_data[['open', 'high', 'low', 'close', 'volume']].values
                timestamps = market_data.index.values
            else:
                data_array = market_data
                timestamps = np.arange(len(market_data))
            
            # Step 1: Extract NAS features
            self.logger.info("📊 Step 1: Extracting NAS features")
            feature_result = self.feature_extractor.extract_features(data_array, timestamps)
            
            if feature_result.features.size == 0:
                raise ValueError("No features extracted for NAS clustering")
            
            # Step 2: Optimize regime count if data-driven
            regime_optimization = None
            if self.nas_config.data_driven_regimes:
                self.logger.info("🔍 Step 2: Optimizing regime count based on data characteristics")
                regime_optimization = self.regime_optimizer.optimize_regime_count(
                    feature_result.features, data_array, timestamps, self.nas_config.n_regimes
                )
                self.nas_config.n_regimes = regime_optimization.optimal_n_regimes
                self.logger.info(f"📊 Optimal regime count determined: {self.nas_config.n_regimes}")
            
            # Step 3: Detect micro-regimes
            self.logger.info("🔍 Step 3: Detecting micro-regimes")
            micro_regime_result = self.micro_regime_detector.detect_micro_regimes(
                data_array, timestamps, feature_result.features
            )
            
            # Step 4: Perform NAS clustering
            self.logger.info("🧠 Step 4: Performing NAS clustering")
            clustering_result = self.nas_clusterer.cluster(
                data_array, timestamps, optimize_parameters=True, generate_report=True
            )
            
            if not clustering_result.success:
                raise RuntimeError(f"NAS clustering failed: {clustering_result.error_message}")
            
            # Step 5: Generate detailed report
            self.logger.info("📊 Step 5: Generating detailed report")
            detailed_report = self.detailed_reporter.generate_comprehensive_report(
                clustering_result, feature_result, micro_regime_result, regime_optimization,
                data_array, timestamps
            )
            
            # Create comprehensive result
            pipeline_result = {
                'success': True,
                'execution_time': clustering_result.execution_time,
                'timestamp': clustering_result.timestamp,
                'method': 'nas_clustering_pipeline',
                
                # Core clustering results
                'clustering_result': clustering_result,
                'feature_result': feature_result,
                'micro_regime_result': micro_regime_result,
                'regime_optimization': regime_optimization,
                'detailed_report': detailed_report,
                
                # Pipeline integration
                'pipeline_replacement': True,
                'hmm_replacement': True,
                'regime_data_available': True,
                'ml_training_ready': True
            }
            
            self.logger.info("✅ NAS clustering pipeline completed successfully")
            return pipeline_result
            
        except Exception as e:
            self.logger.error(f"❌ NAS clustering pipeline execution failed: {e}")
            raise
    
    def _format_output_for_pipeline(self, pipeline_result: Dict[str, Any],
                                   pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Format NAS clustering pipeline result for pipeline compatibility.
        
        Args:
            pipeline_result: NAS clustering pipeline result
            pipeline_state: Current pipeline state
            
        Returns:
            Formatted result dictionary
        """
        try:
            clustering_result = pipeline_result['clustering_result']
            feature_result = pipeline_result['feature_result']
            micro_regime_result = pipeline_result['micro_regime_result']
            regime_optimization = pipeline_result['regime_optimization']
            detailed_report = pipeline_result['detailed_report']
            
            # Create optimal regime clustering result (HMM-compatible format)
            optimal_regime_clustering_result = {
                'success': clustering_result.success,
                'execution_time': clustering_result.execution_time,
                'timestamp': clustering_result.timestamp,
                'method': 'nas_clustering',
                'timeframe': self.nas_config.timeframe,
                'n_regimes': self.nas_config.n_regimes,
                
                # Standard clustering results (HMM-compatible)
                'labels': clustering_result.labels.tolist(),
                'cluster_centers': clustering_result.cluster_centers.tolist(),
                'statistics': clustering_result.statistics,
                'quality_metrics': clustering_result.quality_metrics,
                'validation': clustering_result.validation,
                'metadata': clustering_result.metadata,
                
                # HMM-replacement fields (enhanced NAS capabilities)
                'transition_matrix': clustering_result.regime_transitions.tolist() if clustering_result.regime_transitions is not None else [],
                'eigenvalues': self._calculate_eigenvalues(clustering_result.regime_transitions),
                'eigenvectors': self._calculate_eigenvectors(clustering_result.regime_transitions),
                'stationary_distribution': self._calculate_stationary_distribution(clustering_result.regime_transitions),
                'implied_timescales': self._calculate_implied_timescales(clustering_result.regime_transitions),
                'msm_score': clustering_result.quality_metrics.get('nas_score', 0.0),
                'lag_time': 1,
                
                # NAS-specific fields (additional)
                'nas_architectures': clustering_result.nas_architectures,
                'micro_regimes': self._format_micro_regimes(micro_regime_result),
                'economic_significance_scores': clustering_result.economic_significance_scores.tolist(),
                'trading_viability_scores': clustering_result.trading_viability_scores.tolist(),
                
                # ML training features
                'ml_training_features': self._create_ml_training_features(
                    clustering_result, feature_result, micro_regime_result
                ),
                
                # Pipeline replacement fields
                'pipeline_replacement': True,
                'hmm_replacement': True,
                'enhanced_nas_capabilities': True,
                'regime_data_available': True
            }
            
            # Create comprehensive result
            formatted_result = {
                'success': pipeline_result['success'],
                'execution_time': pipeline_result['execution_time'],
                'timestamp': pipeline_result['timestamp'],
                'method': 'nas_clustering_pipeline',
                
                # Main result (HMM-compatible)
                'optimal_regime_clustering_result': optimal_regime_clustering_result,
                
                # Additional NAS results
                'nas_clustering_result': clustering_result,
                'nas_regime_data': self._create_nas_regime_data(clustering_result, feature_result, micro_regime_result),
                'nas_micro_regimes': self._format_micro_regimes(micro_regime_result),
                'nas_economic_significance': clustering_result.economic_significance_scores.tolist(),
                'nas_trading_viability': clustering_result.trading_viability_scores.tolist(),
                'nas_ml_training_features': self._create_ml_training_features(
                    clustering_result, feature_result, micro_regime_result
                ),
                'nas_detailed_report': detailed_report,
                
                # Pipeline integration
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
                'execution_time': pipeline_result.get('execution_time', 0.0),
                'timestamp': pipeline_result.get('timestamp', datetime.now().isoformat())
            }
    
    def _calculate_eigenvalues(self, transition_matrix: np.ndarray) -> List[float]:
        """Calculate eigenvalues for HMM compatibility."""
        try:
            if transition_matrix is None or transition_matrix.size == 0:
                return []
            
            eigenvalues = np.linalg.eig(transition_matrix)[0]
            return eigenvalues.tolist()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Eigenvalue calculation failed: {e}")
            return []
    
    def _calculate_eigenvectors(self, transition_matrix: np.ndarray) -> List[List[float]]:
        """Calculate eigenvectors for HMM compatibility."""
        try:
            if transition_matrix is None or transition_matrix.size == 0:
                return []
            
            eigenvalues, eigenvectors = np.linalg.eig(transition_matrix)
            return eigenvectors.tolist()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Eigenvector calculation failed: {e}")
            return []
    
    def _calculate_stationary_distribution(self, transition_matrix: np.ndarray) -> List[float]:
        """Calculate stationary distribution for HMM compatibility."""
        try:
            if transition_matrix is None or transition_matrix.size == 0:
                return []
            
            eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
            stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
            stationary_dist = np.real(eigenvectors[:, stationary_idx])
            stationary_dist = stationary_dist / np.sum(stationary_dist)
            
            return stationary_dist.tolist()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Stationary distribution calculation failed: {e}")
            return []
    
    def _calculate_implied_timescales(self, transition_matrix: np.ndarray) -> List[float]:
        """Calculate implied timescales for HMM compatibility."""
        try:
            if transition_matrix is None or transition_matrix.size == 0:
                return []
            
            eigenvalues = np.linalg.eig(transition_matrix)[0]
            valid_eigenvals = eigenvalues[(np.abs(eigenvalues) < 1) & (np.abs(eigenvalues) > 1e-10)]
            timescales = -1 / np.log(np.abs(valid_eigenvals))
            
            return timescales.tolist()
            
        except Exception as e:
            self.logger.warning(f"⚠️ Implied timescales calculation failed: {e}")
            return []
    
    def _format_micro_regimes(self, micro_regime_result: Any) -> Dict[str, Any]:
        """Format micro-regimes for output."""
        try:
            if micro_regime_result is None:
                return {}
            
            return {
                'regimes': micro_regime_result.micro_regimes.tolist(),
                'types': [t.value for t in micro_regime_result.micro_regime_types],
                'scores': micro_regime_result.micro_regime_scores.tolist(),
                'detection_accuracy': micro_regime_result.detection_accuracy,
                'metadata': micro_regime_result.micro_regime_metadata
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Micro-regime formatting failed: {e}")
            return {}
    
    def _create_ml_training_features(self, clustering_result: Any, feature_result: Any,
                                   micro_regime_result: Any) -> Dict[str, Any]:
        """Create ML training features for downstream models."""
        try:
            return {
                'regime_features': {
                    'regime_labels': clustering_result.labels.tolist(),
                    'regime_centers': clustering_result.cluster_centers.tolist(),
                    'regime_statistics': clustering_result.statistics,
                    'regime_quality_metrics': clustering_result.quality_metrics
                },
                'transition_features': {
                    'regime_transitions': clustering_result.regime_transitions.tolist() if clustering_result.regime_transitions is not None else [],
                    'regime_persistence': self._calculate_regime_persistence(clustering_result.labels),
                    'regime_change_frequency': float(np.sum(np.diff(clustering_result.labels) != 0) / (len(clustering_result.labels) - 1))
                },
                'economic_features': {
                    'economic_significance_scores': clustering_result.economic_significance_scores.tolist(),
                    'mean_economic_significance': float(np.mean(clustering_result.economic_significance_scores)),
                    'std_economic_significance': float(np.std(clustering_result.economic_significance_scores))
                },
                'trading_features': {
                    'trading_viability_scores': clustering_result.trading_viability_scores.tolist(),
                    'mean_trading_viability': float(np.mean(clustering_result.trading_viability_scores)),
                    'std_trading_viability': float(np.std(clustering_result.trading_viability_scores))
                },
                'micro_regime_features': {
                    'micro_regime_labels': micro_regime_result.micro_regimes.tolist() if micro_regime_result else [],
                    'micro_regime_types': [t.value for t in micro_regime_result.micro_regime_types] if micro_regime_result else [],
                    'micro_regime_scores': micro_regime_result.micro_regime_scores.tolist() if micro_regime_result else [],
                    'detection_accuracy': micro_regime_result.detection_accuracy if micro_regime_result else 0.0
                },
                'market_features': {
                    'feature_names': feature_result.feature_names,
                    'feature_count': len(feature_result.feature_names),
                    'feature_metadata': feature_result.feature_metadata
                },
                'supported_ml_models': ['DeepScale', 'LGBM', 'XGBoost', 'RandomForest', 'SVM', 'NeuralNetwork'],
                'feature_types': ['regime_features', 'transition_features', 'economic_features', 'trading_features', 'micro_regime_features'],
                'data_ready': True
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ ML training features creation failed: {e}")
            return {}
    
    def _create_nas_regime_data(self, clustering_result: Any, feature_result: Any,
                              micro_regime_result: Any) -> Dict[str, Any]:
        """Create NAS regime data for pipeline integration."""
        try:
            return {
                'regime_data': {
                    'regime_labels': clustering_result.labels.tolist(),
                    'regime_centers': clustering_result.cluster_centers.tolist(),
                    'regime_statistics': clustering_result.statistics,
                    'regime_quality_metrics': clustering_result.quality_metrics,
                    'regime_validation': clustering_result.validation,
                    'regime_metadata': clustering_result.metadata
                },
                'nas_data': {
                    'nas_architectures': clustering_result.nas_architectures,
                    'nas_score': clustering_result.quality_metrics.get('nas_score', 0.0),
                    'nas_architecture_type': clustering_result.metadata.get('nas_architecture_type', 'hybrid')
                },
                'micro_regime_data': {
                    'micro_regimes': micro_regime_result.micro_regimes.tolist() if micro_regime_result else [],
                    'micro_regime_types': [t.value for t in micro_regime_result.micro_regime_types] if micro_regime_result else [],
                    'micro_regime_scores': micro_regime_result.micro_regime_scores.tolist() if micro_regime_result else [],
                    'micro_regime_detection_accuracy': micro_regime_result.detection_accuracy if micro_regime_result else 0.0
                },
                'economic_data': {
                    'economic_significance_scores': clustering_result.economic_significance_scores.tolist(),
                    'trading_viability_scores': clustering_result.trading_viability_scores.tolist(),
                    'regime_transitions': clustering_result.regime_transitions.tolist() if clustering_result.regime_transitions is not None else []
                },
                'feature_data': {
                    'feature_names': feature_result.feature_names,
                    'feature_count': len(feature_result.feature_names),
                    'feature_metadata': feature_result.feature_metadata
                }
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ NAS regime data creation failed: {e}")
            return {}
    
    def _calculate_regime_persistence(self, labels: np.ndarray) -> Dict[str, float]:
        """Calculate regime persistence."""
        try:
            unique_labels = np.unique(labels)
            persistence = {}
            
            for label in unique_labels:
                regime_mask = labels == label
                if np.any(regime_mask):
                    regime_indices = np.where(regime_mask)[0]
                    persistence[f'regime_{label}'] = float(len(regime_indices) / len(labels))
            
            return persistence
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime persistence calculation failed: {e}")
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
            'data_driven_regimes': self.nas_config.data_driven_regimes,
            'required_artifacts': self.get_required_artifacts(),
            'features': [
                'NAS-driven regime detection',
                'Short-term trading optimization (5-30m)',
                'Micro-regime detection',
                'Economic significance scoring',
                'Trading viability assessment',
                'Data-driven regime count determination',
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