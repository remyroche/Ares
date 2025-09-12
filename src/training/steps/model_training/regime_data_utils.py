"""
Regime Data Utilities

Utility functions for accessing regime data in model training steps.
Ensures that all subsequent ML models use HMM-retagged regime data
when available, falling back to original MARKET_ANALYSIS data.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime

from src.utils.logger import system_logger

logger = logging.getLogger(__name__)

class RegimeDataAccessor:
    """Utility class for accessing regime data in model training steps."""
    
    def __init__(self):
        """Initialize regime data accessor."""
        self.logger = system_logger.getChild('RegimeDataAccessor')
    
    def get_regime_data(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get regime data, preferring HMM-retagged data over original MARKET_ANALYSIS data.
        
        Args:
            pipeline_state: Current pipeline state
            
        Returns:
            Regime data dictionary with labels, probabilities, and confidence scores
        """
        try:
            # Check if HMM retagging was completed and use HMM data
            if pipeline_state.get('hmm_retagging_completed', False):
                self.logger.info("🔄 Using HMM-retagged regime data")
                return self._extract_hmm_regime_data(pipeline_state)
            
            # Fall back to original MARKET_ANALYSIS data
            self.logger.info("🔄 Using original MARKET_ANALYSIS regime data")
            return self._extract_original_regime_data(pipeline_state)
            
        except Exception as e:
            self.logger.error(f"❌ Error accessing regime data: {e}")
            return self._get_empty_regime_data()
    
    def _extract_hmm_regime_data(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract HMM-retagged regime data from pipeline state."""
        try:
            return {
                'regime_labels': np.array(pipeline_state.get('regime_states', [])),
                'regime_probabilities': np.array(pipeline_state.get('regime_probabilities', [])),
                'regime_confidence': np.array(pipeline_state.get('regime_confidence', [])),
                'hmm_state_sequence': np.array(pipeline_state.get('hmm_state_sequence', [])),
                'hmm_state_probs': np.array(pipeline_state.get('hmm_state_probs', [])),
                'n_regimes': len(np.unique(pipeline_state.get('regime_states', []))),
                'regime_characteristics': pipeline_state.get('regime_characteristics', {}),
                'transition_matrix': pipeline_state.get('transition_matrix', None),
                'data_source': 'hmm_retagged',
                'retagging_timestamp': pipeline_state.get('retagging_timestamp', ''),
                'hmm_model_used': pipeline_state.get('hmm_model_used_for_retagging', False)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting HMM regime data: {e}")
            return self._get_empty_regime_data()
    
    def _extract_original_regime_data(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract original MARKET_ANALYSIS regime data from pipeline state."""
        try:
            return {
                'regime_labels': np.array(pipeline_state.get('regime_states', [])),
                'regime_probabilities': np.array(pipeline_state.get('regime_probabilities', [])),
                'regime_confidence': np.array(pipeline_state.get('regime_confidence', [])),
                'hmm_state_sequence': np.array(pipeline_state.get('hmm_state_sequence', [])),
                'hmm_state_probs': np.array(pipeline_state.get('hmm_state_probs', [])),
                'n_regimes': len(np.unique(pipeline_state.get('regime_states', []))),
                'regime_characteristics': pipeline_state.get('regime_characteristics', {}),
                'transition_matrix': pipeline_state.get('transition_matrix', None),
                'data_source': 'market_analysis',
                'retagging_timestamp': '',
                'hmm_model_used': False
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting original regime data: {e}")
            return self._get_empty_regime_data()
    
    def _get_empty_regime_data(self) -> Dict[str, Any]:
        """Get empty regime data structure."""
        return {
            'regime_labels': np.array([]),
            'regime_probabilities': np.array([]),
            'regime_confidence': np.array([]),
            'hmm_state_sequence': np.array([]),
            'hmm_state_probs': np.array([]),
            'n_regimes': 0,
            'regime_characteristics': {},
            'transition_matrix': None,
            'data_source': 'none',
            'retagging_timestamp': '',
            'hmm_model_used': False
        }
    
    def validate_regime_data(self, regime_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate regime data for consistency.
        
        Args:
            regime_data: Regime data to validate
            
        Returns:
            Validation result dictionary
        """
        try:
            validation_result = {
                'is_valid': True,
                'issues': [],
                'warnings': [],
                'stats': {}
            }
            
            # Check if data is empty
            if len(regime_data['regime_labels']) == 0:
                validation_result['is_valid'] = False
                validation_result['issues'].append("No regime labels found")
                return validation_result
            
            # Check data consistency
            regime_labels = regime_data['regime_labels']
            regime_probabilities = regime_data['regime_probabilities']
            regime_confidence = regime_data['regime_confidence']
            
            if len(regime_labels) != len(regime_probabilities):
                validation_result['is_valid'] = False
                validation_result['issues'].append("Regime labels and probabilities length mismatch")
            
            if len(regime_labels) != len(regime_confidence):
                validation_result['is_valid'] = False
                validation_result['issues'].append("Regime labels and confidence length mismatch")
            
            # Check regime probability consistency
            if len(regime_probabilities.shape) == 2:
                prob_sums = np.sum(regime_probabilities, axis=1)
                if not np.allclose(prob_sums, 1.0, atol=1e-6):
                    validation_result['warnings'].append("Regime probabilities don't sum to 1.0")
            
            # Calculate statistics
            unique_regimes = np.unique(regime_labels)
            regime_counts = [np.sum(regime_labels == regime) for regime in unique_regimes]
            
            validation_result['stats'] = {
                'n_regimes': len(unique_regimes),
                'total_samples': len(regime_labels),
                'regime_distribution': dict(zip(unique_regimes.tolist(), regime_counts)),
                'mean_confidence': float(np.mean(regime_confidence)) if len(regime_confidence) > 0 else 0.0,
                'min_confidence': float(np.min(regime_confidence)) if len(regime_confidence) > 0 else 0.0,
                'max_confidence': float(np.max(regime_confidence)) if len(regime_confidence) > 0 else 0.0,
                'data_source': regime_data.get('data_source', 'unknown')
            }
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Error validating regime data: {e}")
            return {
                'is_valid': False,
                'issues': [f"Validation error: {str(e)}"],
                'warnings': [],
                'stats': {}
            }
    
    def log_regime_data_summary(self, regime_data: Dict[str, Any], validation_result: Dict[str, Any]) -> None:
        """Log a summary of the regime data being used."""
        try:
            data_source = regime_data.get('data_source', 'unknown')
            n_regimes = regime_data.get('n_regimes', 0)
            total_samples = len(regime_data.get('regime_labels', []))
            
            self.logger.info(f"📊 Regime Data Summary:")
            self.logger.info(f"   - Data Source: {data_source}")
            self.logger.info(f"   - Regimes: {n_regimes}")
            self.logger.info(f"   - Samples: {total_samples}")
            
            if validation_result.get('is_valid', False):
                self.logger.info("   - Validation: ✅ PASSED")
            else:
                self.logger.info("   - Validation: ❌ FAILED")
                for issue in validation_result.get('issues', []):
                    self.logger.info(f"     - {issue}")
            
            # Log warnings
            warnings = validation_result.get('warnings', [])
            if warnings:
                self.logger.info("   - Warnings:")
                for warning in warnings:
                    self.logger.info(f"     - {warning}")
            
            # Log regime distribution
            stats = validation_result.get('stats', {})
            distribution = stats.get('regime_distribution', {})
            if distribution:
                self.logger.info("   - Regime Distribution:")
                for regime, count in distribution.items():
                    percentage = count / total_samples * 100 if total_samples > 0 else 0
                    self.logger.info(f"     Regime {regime}: {count} ({percentage:.1f}%)")
            
        except Exception as e:
            self.logger.error(f"❌ Error logging regime data summary: {e}")

# Global instance for convenience
regime_data_accessor = RegimeDataAccessor()

def get_regime_data(pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to get regime data."""
    return regime_data_accessor.get_regime_data(pipeline_state)

def validate_regime_data(regime_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function to validate regime data."""
    return regime_data_accessor.validate_regime_data(regime_data)