#!/usr/bin/env python3
"""Validator for enhanced HMM clustering step."""

import asyncio
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.core.decorators import validates, handles_errors

logger = system_logger.getChild("HMMClusteringValidator")


class HMMClusteringValidator:
    """Validator for enhanced HMM clustering results."""
    
    def __init__(self):
        self.logger = system_logger.getChild('HMMClusteringValidator')
        self.validation_results = {}
        
    @validates(step_name='hmm_clustering_validation')
    @handles_errors(default_return={'validation_passed': False, 'errors': ['Validation failed']})
    async def validate(self, results: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate HMM clustering results.
        
        Args:
            results: Results from HMM clustering
            config: Configuration used
            
        Returns:
            Validation results dictionary
        """
        self.logger.info("🔍 Starting HMM clustering validation...")
        
        errors = []
        warnings = []
        
        # 1. Check required outputs
        required_outputs = [
            'regime_states',
            'regime_labels', 
            'n_regimes',
            'regime_distribution',
            'regime_transitions',
            'hmm_regime_discovery_completed'
        ]
        
        for output in required_outputs:
            if output not in results:
                errors.append(f"Missing required output: {output}")
        
        # 2. Validate regime states
        if 'regime_states' in results:
            regime_states = results['regime_states']
            
            if not isinstance(regime_states, list):
                errors.append("regime_states must be a list")
            elif len(regime_states) == 0:
                errors.append("regime_states is empty")
            else:
                # Check regime values
                unique_regimes = set(regime_states)
                n_regimes = results.get('n_regimes', 0)
                
                if len(unique_regimes) != n_regimes:
                    warnings.append(
                        f"Mismatch: {len(unique_regimes)} unique regimes found, "
                        f"but n_regimes={n_regimes}"
                    )
                
                # Check regime numbering
                expected_regimes = set(range(n_regimes))
                if unique_regimes != expected_regimes:
                    warnings.append(
                        f"Regime numbering issue: expected {expected_regimes}, "
                        f"got {unique_regimes}"
                    )
        
        # 3. Validate composite DataFrame
        if 'composite_df' in results:
            composite_df = results['composite_df']
            
            if not isinstance(composite_df, pd.DataFrame):
                errors.append("composite_df must be a pandas DataFrame")
            else:
                # Check required columns
                required_cols = ['composite_cluster_id']
                missing_cols = [col for col in required_cols if col not in composite_df.columns]
                
                if missing_cols:
                    errors.append(f"composite_df missing columns: {missing_cols}")
                
                # Check data consistency
                if 'composite_cluster_id' in composite_df.columns:
                    cluster_ids = composite_df['composite_cluster_id'].tolist()
                    if cluster_ids != results.get('regime_states', [])[:len(cluster_ids)]:
                        warnings.append("Mismatch between composite_df and regime_states")
        
        # 4. Validate regime distribution
        if 'regime_distribution' in results:
            regime_dist = results['regime_distribution']
            
            if not isinstance(regime_dist, dict):
                errors.append("regime_distribution must be a dictionary")
            else:
                # Check total count
                total_count = sum(regime_dist.values())
                expected_count = len(results.get('regime_states', []))
                
                if total_count != expected_count:
                    errors.append(
                        f"Regime distribution count mismatch: "
                        f"sum={total_count}, expected={expected_count}"
                    )
        
        # 5. Validate transitions
        if 'regime_transitions' in results:
            transitions = results['regime_transitions']
            
            if not isinstance(transitions, dict):
                errors.append("regime_transitions must be a dictionary")
            else:
                # Check required fields
                required_fields = ['total_transitions', 'transition_rate']
                missing_fields = [f for f in required_fields if f not in transitions]
                
                if missing_fields:
                    warnings.append(f"regime_transitions missing fields: {missing_fields}")
                
                # Validate transition rate
                if 'transition_rate' in transitions:
                    rate = transitions['transition_rate']
                    if not 0 <= rate <= 1:
                        warnings.append(f"Invalid transition_rate: {rate} (should be 0-1)")
        
        # 6. Validate quality metrics
        if 'overall_quality_score' in results:
            score = results['overall_quality_score']
            
            if not isinstance(score, (int, float)):
                errors.append("overall_quality_score must be numeric")
            elif not 0 <= score <= 1:
                warnings.append(f"overall_quality_score {score} outside expected range [0,1]")
        
        # 7. Validate enhanced features
        if results.get('enhanced_ml_transition_detection', False):
            if 'transition_models' not in results:
                warnings.append("Enhanced ML detection enabled but no transition_models found")
        
        # 8. Validate saved files
        if 'saved_files' in results:
            saved_files = results['saved_files']
            
            # Check if key files were saved
            expected_files = ['composite_clusters', 'metadata']
            missing_files = [f for f in expected_files if f not in saved_files]
            
            if missing_files:
                warnings.append(f"Expected files not saved: {missing_files}")
            
            # Verify files exist
            for file_type, file_path in saved_files.items():
                if not Path(file_path).exists():
                    errors.append(f"Saved file does not exist: {file_path}")
        
        # 9. Performance checks
        if 'execution_time' in results:
            exec_time = results['execution_time']
            
            if exec_time > 1800:  # 30 minutes
                warnings.append(f"Execution time too long: {exec_time:.1f} seconds")
        
        # 10. Data consistency checks
        if 'n_regimes' in results:
            n_regimes = results['n_regimes']
            
            if n_regimes < 2:
                errors.append(f"Too few regimes discovered: {n_regimes}")
            elif n_regimes > 10:
                warnings.append(f"Many regimes discovered: {n_regimes} (may be overfitting)")
        
        # Compile validation results
        validation_passed = len(errors) == 0
        
        validation_results = {
            'validation_passed': validation_passed,
            'errors': errors,
            'warnings': warnings,
            'checks_performed': {
                'required_outputs': len(errors) == 0 or not any('Missing required' in e for e in errors),
                'regime_states_valid': 'regime_states' in results and isinstance(results['regime_states'], list),
                'composite_df_valid': 'composite_df' not in results or isinstance(results.get('composite_df'), pd.DataFrame),
                'distribution_valid': 'regime_distribution' in results and isinstance(results['regime_distribution'], dict),
                'transitions_valid': 'regime_transitions' in results and isinstance(results['regime_transitions'], dict),
                'quality_metrics_valid': 'overall_quality_score' in results,
                'files_saved': 'saved_files' in results
            },
            'summary': {
                'n_errors': len(errors),
                'n_warnings': len(warnings),
                'n_regimes': results.get('n_regimes', 0),
                'n_periods': len(results.get('regime_states', [])),
                'quality_score': results.get('overall_quality_score', 0)
            }
        }
        
        # Log results
        if validation_passed:
            self.logger.info("✅ HMM clustering validation PASSED")
            if warnings:
                self.logger.warning(f"⚠️ {len(warnings)} warnings found:")
                for warning in warnings[:3]:
                    self.logger.warning(f"   - {warning}")
                if len(warnings) > 3:
                    self.logger.warning(f"   ... and {len(warnings) - 3} more")
        else:
            self.logger.error("❌ HMM clustering validation FAILED")
            self.logger.error(f"   Found {len(errors)} errors:")
            for error in errors[:5]:
                self.logger.error(f"   - {error}")
            if len(errors) > 5:
                self.logger.error(f"   ... and {len(errors) - 5} more")
        
        return validation_results


@handles_errors(fallback=False)
async def run_validator(results: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Run the HMM clustering validator.
    
    Args:
        results: Results to validate
        config: Configuration used
        
    Returns:
        True if validation passed
    """
    validator = HMMClusteringValidator()
    validation_results = await validator.validate(results, config)
    
    return validation_results['validation_passed']


if __name__ == "__main__":
    # Test the validator
    async def test_validator():
        # Create test results
        test_results = {
            'regime_states': [0, 0, 1, 1, 2, 2, 0, 1, 2],
            'regime_labels': [0, 0, 1, 1, 2, 2, 0, 1, 2],
            'n_regimes': 3,
            'regime_distribution': {
                'regime_0': 3,
                'regime_1': 3,
                'regime_2': 3
            },
            'regime_transitions': {
                'total_transitions': 5,
                'transition_rate': 0.625
            },
            'hmm_regime_discovery_completed': True,
            'overall_quality_score': 0.85,
            'execution_time': 120.5,
            'composite_df': pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=9, freq='1min'),
                'composite_cluster_id': [0, 0, 1, 1, 2, 2, 0, 1, 2]
            })
        }
        
        test_config = {
            'symbol': 'ETHUSDT',
            'exchange': 'BINANCE',
            'timeframe': '1m'
        }
        
        # Run validation
        passed = await run_validator(test_results, test_config)
        
        print(f"\n{'✅' if passed else '❌'} Test validation {'PASSED' if passed else 'FAILED'}")
        
    asyncio.run(test_validator())