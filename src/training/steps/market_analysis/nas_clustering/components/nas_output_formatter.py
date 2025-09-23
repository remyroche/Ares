"""
NAS Output Formatter for pipeline compatibility.

This module provides output formatting to ensure full compatibility
with the existing HMM clustering pipeline while adding NAS-specific features.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class NASOutputFormatter:
    """Formatter for NAS clustering output to ensure pipeline compatibility."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize NAS output formatter.
        
        Args:
            config: Formatter configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Output format settings
        self.output_format = config.get('output_format', 'hmm_clustering_compatible')
        self.include_nas_features = config.get('include_nas_features', True)
        self.include_micro_regimes = config.get('include_micro_regimes', True)
        self.include_economic_significance = config.get('include_economic_significance', True)
        self.include_trading_viability = config.get('include_trading_viability', True)
        
        self.logger.info(f"✅ NAS Output Formatter initialized with format: {self.output_format}")
    
    def format_clustering_result(self, nas_result: Any, 
                               feature_result: Any = None) -> Dict[str, Any]:
        """Format NAS clustering result for pipeline compatibility.
        
        Args:
            nas_result: NAS clustering result
            feature_result: Optional feature extraction result
            
        Returns:
            Formatted result dictionary
        """
        try:
            if self.output_format == 'hmm_clustering_compatible':
                return self._format_hmm_compatible(nas_result, feature_result)
            elif self.output_format == 'nas_enhanced':
                return self._format_nas_enhanced(nas_result, feature_result)
            else:
                return self._format_default(nas_result, feature_result)
                
        except Exception as e:
            self.logger.error(f"❌ Output formatting failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _format_hmm_compatible(self, nas_result: Any, 
                             feature_result: Any = None) -> Dict[str, Any]:
        """Format result to be compatible with HMM clustering output."""
        try:
            # Create HMM-compatible structure
            formatted_result = {
                'success': nas_result.success,
                'execution_time': nas_result.execution_time,
                'timestamp': nas_result.timestamp,
                'method': 'nas_clustering',
                
                # Standard HMM clustering fields
                'labels': nas_result.labels.tolist(),
                'cluster_centers': nas_result.cluster_centers.tolist(),
                'statistics': nas_result.statistics,
                'quality_metrics': nas_result.quality_metrics,
                'validation': nas_result.validation,
                'metadata': nas_result.metadata,
                
                # HMM-specific fields (compatible)
                'transition_matrix': nas_result.regime_transitions.tolist() if nas_result.regime_transitions is not None else [],
                'eigenvalues': self._calculate_eigenvalues(nas_result.regime_transitions),
                'eigenvectors': self._calculate_eigenvectors(nas_result.regime_transitions),
                'stationary_distribution': self._calculate_stationary_distribution(nas_result.regime_transitions),
                'implied_timescales': self._calculate_implied_timescales(nas_result.regime_transitions),
                'msm_score': nas_result.quality_metrics.get('nas_score', 0.0),
                'lag_time': 1,
                
                # NAS-specific fields (additional)
                'nas_architectures': nas_result.nas_architectures if self.include_nas_features else {},
                'micro_regimes': self._format_micro_regimes(nas_result.micro_regimes) if self.include_micro_regimes else {},
                'economic_significance_scores': nas_result.economic_significance_scores.tolist() if self.include_economic_significance else [],
                'trading_viability_scores': nas_result.trading_viability_scores.tolist() if self.include_trading_viability else [],
                
                # Pipeline compatibility
                'pipeline_compatible': True,
                'hmm_compatible': True,
                'regime_data_available': True
            }
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"❌ HMM-compatible formatting failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _format_nas_enhanced(self, nas_result: Any, 
                           feature_result: Any = None) -> Dict[str, Any]:
        """Format result with enhanced NAS features."""
        try:
            # Create NAS-enhanced structure
            formatted_result = {
                'success': nas_result.success,
                'execution_time': nas_result.execution_time,
                'timestamp': nas_result.timestamp,
                'method': 'nas_clustering_enhanced',
                
                # Standard clustering fields
                'labels': nas_result.labels.tolist(),
                'cluster_centers': nas_result.cluster_centers.tolist(),
                'statistics': nas_result.statistics,
                'quality_metrics': nas_result.quality_metrics,
                'validation': nas_result.validation,
                'metadata': nas_result.metadata,
                
                # NAS-specific fields
                'nas_architectures': nas_result.nas_architectures,
                'nas_architecture_type': nas_result.metadata.get('nas_architecture_type', 'hybrid'),
                'nas_score': nas_result.quality_metrics.get('nas_score', 0.0),
                'nas_optimization': {
                    'economic_significance_threshold': nas_result.metadata.get('economic_significance_threshold', 0.7),
                    'trading_viability_threshold': nas_result.metadata.get('trading_viability_threshold', 0.6),
                    'regime_transition_cost': nas_result.metadata.get('regime_transition_cost', 0.05)
                },
                
                # Micro-regime fields
                'micro_regimes': self._format_micro_regimes(nas_result.micro_regimes),
                'micro_regime_detection': {
                    'enabled': nas_result.metadata.get('micro_regime_detection', False),
                    'sensitivity': nas_result.metadata.get('micro_regime_sensitivity', 0.7),
                    'detection_accuracy': nas_result.micro_regimes.detection_accuracy if nas_result.micro_regimes else 0.0
                },
                
                # Economic and trading fields
                'economic_significance': {
                    'scores': nas_result.economic_significance_scores.tolist(),
                    'mean_score': float(np.mean(nas_result.economic_significance_scores)),
                    'threshold': nas_result.metadata.get('economic_significance_threshold', 0.7)
                },
                'trading_viability': {
                    'scores': nas_result.trading_viability_scores.tolist(),
                    'mean_score': float(np.mean(nas_result.trading_viability_scores)),
                    'threshold': nas_result.metadata.get('trading_viability_threshold', 0.6)
                },
                
                # Regime transition fields
                'regime_transitions': {
                    'matrix': nas_result.regime_transitions.tolist() if nas_result.regime_transitions is not None else [],
                    'transition_probabilities': self._calculate_transition_probabilities(nas_result.regime_transitions),
                    'regime_persistence': self._calculate_regime_persistence(nas_result.labels)
                },
                
                # LM model training fields
                'lm_training_data': self._create_lm_training_data(nas_result),
                
                # Pipeline compatibility
                'pipeline_compatible': True,
                'hmm_compatible': True,
                'nas_enhanced': True
            }
            
            return formatted_result
            
        except Exception as e:
            self.logger.error(f"❌ NAS-enhanced formatting failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _format_default(self, nas_result: Any, 
                       feature_result: Any = None) -> Dict[str, Any]:
        """Format result with default structure."""
        try:
            return {
                'success': nas_result.success,
                'execution_time': nas_result.execution_time,
                'timestamp': nas_result.timestamp,
                'method': 'nas_clustering',
                'labels': nas_result.labels.tolist(),
                'cluster_centers': nas_result.cluster_centers.tolist(),
                'statistics': nas_result.statistics,
                'quality_metrics': nas_result.quality_metrics,
                'validation': nas_result.validation,
                'metadata': nas_result.metadata
            }
            
        except Exception as e:
            self.logger.error(f"❌ Default formatting failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _format_micro_regimes(self, micro_regimes: Any) -> Dict[str, Any]:
        """Format micro-regimes for output."""
        try:
            if micro_regimes is None:
                return {}
            
            return {
                'regimes': micro_regimes.micro_regimes.tolist(),
                'types': [t.value for t in micro_regimes.micro_regime_types],
                'scores': micro_regimes.micro_regime_scores.tolist(),
                'detection_accuracy': micro_regimes.detection_accuracy,
                'metadata': micro_regimes.micro_regime_metadata
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Micro-regime formatting failed: {e}")
            return {}
    
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
    
    def _calculate_transition_probabilities(self, transition_matrix: np.ndarray) -> Dict[str, Any]:
        """Calculate transition probabilities."""
        try:
            if transition_matrix is None or transition_matrix.size == 0:
                return {}
            
            # Calculate transition probabilities
            row_sums = transition_matrix.sum(axis=1)
            transition_probs = transition_matrix / (row_sums[:, np.newaxis] + 1e-8)
            
            return {
                'matrix': transition_probs.tolist(),
                'row_sums': row_sums.tolist(),
                'is_stochastic': np.allclose(row_sums, 1.0, atol=1e-6)
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Transition probability calculation failed: {e}")
            return {}
    
    def _calculate_regime_persistence(self, labels: np.ndarray) -> Dict[str, Any]:
        """Calculate regime persistence statistics."""
        try:
            if len(labels) == 0:
                return {}
            
            # Calculate regime durations
            regime_changes = np.diff(labels) != 0
            regime_starts = np.concatenate([[True], regime_changes])
            regime_ends = np.concatenate([regime_changes, [True]])
            
            regime_durations = []
            current_duration = 0
            
            for i, (start, end) in enumerate(zip(regime_starts, regime_ends)):
                if start:
                    current_duration = 1
                else:
                    current_duration += 1
                
                if end:
                    regime_durations.append(current_duration)
            
            return {
                'regime_durations': regime_durations,
                'mean_duration': float(np.mean(regime_durations)),
                'median_duration': float(np.median(regime_durations)),
                'max_duration': int(np.max(regime_durations)),
                'min_duration': int(np.min(regime_durations)),
                'total_regime_changes': int(np.sum(regime_changes))
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ Regime persistence calculation failed: {e}")
            return {}
    
    def _create_lm_training_data(self, nas_result: Any) -> Dict[str, Any]:
        """Create LM model training data."""
        try:
            return {
                'regime_sequences': nas_result.labels.tolist(),
                'regime_transitions': nas_result.regime_transitions.tolist() if nas_result.regime_transitions is not None else [],
                'economic_significance': nas_result.economic_significance_scores.tolist(),
                'trading_viability': nas_result.trading_viability_scores.tolist(),
                'micro_regime_sequences': nas_result.micro_regimes.micro_regimes.tolist() if nas_result.micro_regimes else [],
                'micro_regime_types': [t.value for t in nas_result.micro_regimes.micro_regime_types] if nas_result.micro_regimes else [],
                'regime_statistics': nas_result.statistics,
                'regime_quality_metrics': nas_result.quality_metrics,
                'regime_metadata': nas_result.metadata,
                'nas_architectures': nas_result.nas_architectures,
                'timestamp': nas_result.timestamp,
                'execution_time': nas_result.execution_time
            }
            
        except Exception as e:
            self.logger.warning(f"⚠️ LM training data creation failed: {e}")
            return {}
    
    def save_formatted_result(self, formatted_result: Dict[str, Any], 
                            output_path: str) -> bool:
        """Save formatted result to file.
        
        Args:
            formatted_result: Formatted result dictionary
            output_path: Output file path
            
        Returns:
            Success status
        """
        try:
            # Create output directory if it doesn't exist
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(formatted_result, f, indent=2, default=str)
            
            self.logger.info(f"✅ Formatted result saved to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save formatted result: {e}")
            return False
    
    def load_formatted_result(self, input_path: str) -> Dict[str, Any]:
        """Load formatted result from file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Loaded result dictionary
        """
        try:
            with open(input_path, 'r') as f:
                result = json.load(f)
            
            self.logger.info(f"✅ Formatted result loaded from {input_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load formatted result: {e}")
            return {}