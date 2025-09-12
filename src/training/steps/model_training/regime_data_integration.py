"""
Regime Data Integration Step

This step ensures that all subsequent ML models use the HMM-retagged regime data
instead of the original MARKET_ANALYSIS regime tags, ensuring consistency with
real trading situations.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

from src.utils.logger import system_logger
from src.utils.comprehensive_function_logger import log_important_calls, log_all_calls

logger = logging.getLogger(__name__)

class RegimeDataIntegrator:
    """Integrates HMM-retagged regime data into the pipeline for subsequent steps."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize regime data integrator.
        
        Args:
            config: Integration configuration
        """
        self.config = config or {}
        self.logger = system_logger.getChild('RegimeDataIntegrator')
        
    @log_important_calls
    async def integrate_hmm_regime_data(
        self,
        pipeline_state: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Integrate HMM-retagged regime data into pipeline state.
        
        Args:
            pipeline_state: Current pipeline state
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            
        Returns:
            Updated pipeline state with HMM regime data
        """
        try:
            self.logger.info(f"🔄 Integrating HMM regime data for {symbol}/{exchange}/{timeframe}")
            
            # Check if HMM retagging was completed
            if not pipeline_state.get('hmm_retagging_completed', False):
                self.logger.warning("⚠️ HMM retagging not completed, using original regime data")
                return pipeline_state
            
            # Extract HMM regime data
            hmm_regime_data = self._extract_hmm_regime_data(pipeline_state)
            
            if not hmm_regime_data:
                self.logger.warning("⚠️ No HMM regime data found, using original regime data")
                return pipeline_state
            
            # Update pipeline state with HMM regime data
            updated_pipeline_state = self._update_pipeline_state_with_hmm_data(
                pipeline_state, hmm_regime_data
            )
            
            # Validate regime data consistency
            validation_result = await self._validate_regime_data_consistency(
                updated_pipeline_state, symbol, exchange, timeframe
            )
            
            # Log integration summary
            self._log_integration_summary(hmm_regime_data, validation_result)
            
            self.logger.info("✅ HMM regime data integration completed successfully")
            
            return updated_pipeline_state
            
        except Exception as e:
            self.logger.error(f"❌ HMM regime data integration failed: {e}")
            return pipeline_state
    
    def _extract_hmm_regime_data(self, pipeline_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract HMM regime data from pipeline state."""
        try:
            # Check for HMM retagged data
            if pipeline_state.get('hmm_retagged_regimes', False):
                return {
                    'regime_states': pipeline_state.get('regime_states', []),
                    'regime_probabilities': pipeline_state.get('regime_probabilities', []),
                    'regime_confidence': pipeline_state.get('regime_confidence', []),
                    'hmm_state_sequence': pipeline_state.get('hmm_state_sequence', []),
                    'hmm_state_probs': pipeline_state.get('hmm_state_probs', []),
                    'regime_transitions': pipeline_state.get('regime_transitions', {}),
                    'retagging_timestamp': pipeline_state.get('retagging_timestamp', ''),
                    'hmm_model_used_for_retagging': pipeline_state.get('hmm_model_used_for_retagging', False)
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting HMM regime data: {e}")
            return None
    
    def _update_pipeline_state_with_hmm_data(
        self, 
        pipeline_state: Dict[str, Any], 
        hmm_regime_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update pipeline state with HMM regime data."""
        try:
            updated_state = pipeline_state.copy()
            
            # Update regime-related keys with HMM data
            regime_keys = [
                'regime_states', 'regime_probabilities', 'regime_confidence',
                'hmm_state_sequence', 'hmm_state_probs'
            ]
            
            for key in regime_keys:
                if key in hmm_regime_data:
                    updated_state[key] = hmm_regime_data[key]
            
            # Add HMM-specific metadata
            updated_state.update({
                'regime_data_source': 'hmm_retagged',
                'hmm_integration_completed': True,
                'hmm_integration_timestamp': datetime.now().isoformat(),
                'regime_transitions': hmm_regime_data.get('regime_transitions', {}),
                'hmm_model_used_for_retagging': hmm_regime_data.get('hmm_model_used_for_retagging', False)
            })
            
            # Mark original MARKET_ANALYSIS regime data as superseded
            updated_state['market_analysis_regimes_superseded'] = True
            updated_state['superseded_by_hmm'] = True
            
            return updated_state
            
        except Exception as e:
            self.logger.error(f"❌ Error updating pipeline state: {e}")
            return pipeline_state
    
    @log_all_calls
    async def _validate_regime_data_consistency(
        self,
        pipeline_state: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> Dict[str, Any]:
        """Validate consistency of HMM regime data."""
        try:
            validation_result = {
                'is_valid': True,
                'issues': [],
                'warnings': [],
                'regime_stats': {}
            }
            
            # Check required regime data
            required_keys = ['regime_states', 'regime_probabilities', 'regime_confidence']
            for key in required_keys:
                if key not in pipeline_state or not pipeline_state[key]:
                    validation_result['is_valid'] = False
                    validation_result['issues'].append(f"Missing {key}")
            
            if not validation_result['is_valid']:
                return validation_result
            
            # Validate regime states
            regime_states = np.array(pipeline_state['regime_states'])
            regime_probabilities = np.array(pipeline_state['regime_probabilities'])
            regime_confidence = np.array(pipeline_state['regime_confidence'])
            
            # Check data consistency
            if len(regime_states) != len(regime_probabilities):
                validation_result['is_valid'] = False
                validation_result['issues'].append("Regime states and probabilities length mismatch")
            
            if len(regime_states) != len(regime_confidence):
                validation_result['is_valid'] = False
                validation_result['issues'].append("Regime states and confidence length mismatch")
            
            # Check regime probability consistency
            if len(regime_probabilities.shape) == 2:
                prob_sums = np.sum(regime_probabilities, axis=1)
                if not np.allclose(prob_sums, 1.0, atol=1e-6):
                    validation_result['warnings'].append("Regime probabilities don't sum to 1.0")
            
            # Calculate regime statistics
            unique_regimes = np.unique(regime_states)
            regime_counts = [np.sum(regime_states == regime) for regime in unique_regimes]
            
            validation_result['regime_stats'] = {
                'n_regimes': len(unique_regimes),
                'total_samples': len(regime_states),
                'regime_distribution': dict(zip(unique_regimes.tolist(), regime_counts)),
                'mean_confidence': float(np.mean(regime_confidence)),
                'min_confidence': float(np.min(regime_confidence)),
                'max_confidence': float(np.max(regime_confidence)),
                'confidence_std': float(np.std(regime_confidence))
            }
            
            # Check for balanced regime distribution
            if len(regime_counts) > 1:
                regime_balance = np.std(regime_counts) / np.mean(regime_counts)
                if regime_balance > 0.5:  # More than 50% imbalance
                    validation_result['warnings'].append(f"Unbalanced regime distribution (balance: {regime_balance:.2f})")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"❌ Error validating regime data consistency: {e}")
            return {
                'is_valid': False,
                'issues': [f"Validation error: {str(e)}"],
                'warnings': [],
                'regime_stats': {}
            }
    
    def _log_integration_summary(
        self, 
        hmm_regime_data: Dict[str, Any], 
        validation_result: Dict[str, Any]
    ) -> None:
        """Log integration summary."""
        try:
            self.logger.info("📊 HMM Regime Data Integration Summary:")
            
            # Log validation status
            if validation_result.get('is_valid', False):
                self.logger.info("   ✅ Validation: PASSED")
            else:
                self.logger.info("   ❌ Validation: FAILED")
                for issue in validation_result.get('issues', []):
                    self.logger.info(f"      - {issue}")
            
            # Log warnings
            warnings = validation_result.get('warnings', [])
            if warnings:
                self.logger.info("   ⚠️ Warnings:")
                for warning in warnings:
                    self.logger.info(f"      - {warning}")
            
            # Log regime statistics
            regime_stats = validation_result.get('regime_stats', {})
            if regime_stats:
                self.logger.info(f"   📈 Regime Statistics:")
                self.logger.info(f"      - Regimes: {regime_stats.get('n_regimes', 0)}")
                self.logger.info(f"      - Samples: {regime_stats.get('total_samples', 0)}")
                self.logger.info(f"      - Mean Confidence: {regime_stats.get('mean_confidence', 0.0):.3f}")
                
                # Log regime distribution
                distribution = regime_stats.get('regime_distribution', {})
                if distribution:
                    self.logger.info(f"      - Regime Distribution:")
                    for regime, count in distribution.items():
                        percentage = count / regime_stats.get('total_samples', 1) * 100
                        self.logger.info(f"        Regime {regime}: {count} ({percentage:.1f}%)")
            
            # Log HMM metadata
            if hmm_regime_data.get('hmm_model_used_for_retagging', False):
                self.logger.info("   🤖 HMM Model: Used for retagging")
            
            retagging_timestamp = hmm_regime_data.get('retagging_timestamp', '')
            if retagging_timestamp:
                self.logger.info(f"   ⏰ Retagging Time: {retagging_timestamp}")
            
        except Exception as e:
            self.logger.error(f"❌ Error logging integration summary: {e}")

# Global instance for convenience
regime_data_integrator = RegimeDataIntegrator()