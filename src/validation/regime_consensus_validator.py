"""
Regime Consensus Validator for Pipeline-Level Semantic Consensus Validation.

This module provides pipeline-level validation of regime consensus using semantic
consensus approach to detect and resolve regime disagreements between different systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
from dataclasses import dataclass

from src.utils.logger import system_logger
from src.core.decorators import handles_errors, validates
from src.core.error_classes import (
    initialization_error,
    validation_error,
    execution_error,
)

# Import semantic consensus utilities
try:
    from src.training.steps.market_analysis.shared_utils.metrics import (
        calculate_consensus_metrics,
        calculate_disagreement_metrics,
        MetricsCalculator
    )
    SEMANTIC_CONSENSUS_AVAILABLE = True
except ImportError:
    SEMANTIC_CONSENSUS_AVAILABLE = False
    calculate_consensus_metrics = None
    calculate_disagreement_metrics = None
    MetricsCalculator = None

@dataclass
class RegimeConsensusConfig:
    """Configuration for regime consensus validation."""
    enable_semantic_consensus: bool = True
    consensus_threshold: float = 0.6
    disagreement_tolerance: float = 0.3
    enable_regime_mapping: bool = True
    enable_feature_based_mapping: bool = True
    min_samples_for_validation: int = 100
    enable_alerting: bool = True
    alert_consensus_threshold: float = 0.3  # Alert if consensus below this
    enable_regime_quality_assessment: bool = True

class RegimeConsensusValidator:
    """
    Pipeline-level regime consensus validator using semantic consensus approach.

    This validator provides end-to-end validation of regime consensus between
    different systems, detecting disagreements and providing semantic mapping
    to resolve apparent conflicts.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the regime consensus validator.

        Args:
            config: Configuration dictionary
        """
        self.config = RegimeConsensusConfig(**config)
        self.logger = system_logger.getChild("RegimeConsensusValidator")

        # State tracking
        self.is_initialized: bool = False
        self.validation_history: List[Dict[str, Any]] = []
        self.max_history: int = 1000

        # Semantic consensus components
        if SEMANTIC_CONSENSUS_AVAILABLE:
            self.metrics_calculator = MetricsCalculator(verbose=True)
        else:
            self.metrics_calculator = None
            self.logger.warning("⚠️ Semantic consensus utilities not available")

        # Validation statistics
        self.total_validations: int = 0
        self.successful_validations: int = 0
        self.failed_validations: int = 0
        self.consensus_alerts: int = 0

        self.logger.info("🧠 Regime consensus validator initialized")

    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="regime consensus validator initialization",
    )
    async def initialize(self) -> bool:
        """
        Initialize the regime consensus validator.

        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing regime consensus validator...")

            # Validate configuration
            if not self._validate_configuration():
                self.logger.error(validation_error("Invalid regime consensus validator configuration"))
                return False

            # Clear history
            self.validation_history.clear()

            # Reset statistics
            self.total_validations = 0
            self.successful_validations = 0
            self.failed_validations = 0
            self.consensus_alerts = 0

            self.is_initialized = True
            self.logger.info("✅ Regime consensus validator initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(execution_error(f"❌ Regime consensus validator initialization failed: {e}"))
            return False

    def _validate_configuration(self) -> bool:
        """Validate regime consensus validator configuration."""
        try:
            if self.config.consensus_threshold <= 0 or self.config.consensus_threshold > 1:
                self.logger.error(validation_error("Consensus threshold must be between 0 and 1"))
                return False

            if self.config.disagreement_tolerance <= 0 or self.config.disagreement_tolerance > 1:
                self.logger.error(validation_error("Disagreement tolerance must be between 0 and 1"))
                return False

            if self.config.min_samples_for_validation <= 0:
                self.logger.error(validation_error("Minimum samples for validation must be positive"))
                return False

            return True

        except Exception as e:
            self.logger.exception(execution_error(f"Configuration validation failed: {e}"))
            return False

    @validates()
    @handles_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="regime consensus validation",
    )
    async def validate_regime_consensus(
        self,
        system_a_assignments: List[int],
        system_b_assignments: List[int],
        market_data: Optional[pd.DataFrame] = None,
        context: str = "pipeline_validation"
    ) -> Dict[str, Any]:
        """
        Validate regime consensus between two systems using semantic approach.

        Args:
            system_a_assignments: System A regime assignments
            system_b_assignments: System B regime assignments
            market_data: Optional market data for feature-based mapping
            context: Context for validation (e.g., "pipeline_validation", "training_step")

        Returns:
            Dictionary containing validation results
        """
        if not self.is_initialized:
            self.logger.error(initialization_error("Regime consensus validator not initialized"))
            return None

        try:
            self.total_validations += 1
            self.logger.info(f"🧠 Validating regime consensus (context: {context})")

            # Validate input data
            if not self._validate_input_data(system_a_assignments, system_b_assignments):
                self.failed_validations += 1
                return self._create_validation_result(
                    success=False,
                    error="Invalid input data",
                    context=context
                )

            # Check minimum samples requirement
            min_samples = min(len(system_a_assignments), len(system_b_assignments))
            if min_samples < self.config.min_samples_for_validation:
                self.logger.warning(f"⚠️ Insufficient samples for validation: {min_samples} < {self.config.min_samples_for_validation}")
                self.failed_validations += 1
                return self._create_validation_result(
                    success=False,
                    error=f"Insufficient samples: {min_samples} < {self.config.min_samples_for_validation}",
                    context=context
                )

            # Perform semantic consensus validation
            if self.config.enable_semantic_consensus and SEMANTIC_CONSENSUS_AVAILABLE:
                validation_result = await self._perform_semantic_consensus_validation(
                    system_a_assignments, system_b_assignments, market_data, context
                )
            else:
                validation_result = await self._perform_basic_consensus_validation(
                    system_a_assignments, system_b_assignments, context
                )

            # Check for consensus alerts
            if self.config.enable_alerting:
                self._check_consensus_alerts(validation_result, context)

            # Record validation history
            self._record_validation_history(validation_result, context)

            # Update statistics
            if validation_result.get('success', False):
                self.successful_validations += 1
            else:
                self.failed_validations += 1

            self.logger.info(f"✅ Regime consensus validation completed: {validation_result.get('consensus_score', 0.0):.3f}")
            return validation_result

        except Exception as e:
            self.logger.exception(execution_error(f"❌ Regime consensus validation failed: {e}"))
            self.failed_validations += 1
            return self._create_validation_result(
                success=False,
                error=str(e),
                context=context
            )

    def _validate_input_data(self, system_a_assignments: List[int], system_b_assignments: List[int]) -> bool:
        """Validate input data for regime consensus validation."""
        try:
            if not system_a_assignments or not system_b_assignments:
                self.logger.error("❌ Empty assignments provided")
                return False

            if len(system_a_assignments) == 0 or len(system_b_assignments) == 0:
                self.logger.error("❌ Zero-length assignments provided")
                return False

            # Check for valid regime IDs (non-negative integers)
            if any(not isinstance(x, int) or x < 0 for x in system_a_assignments):
                self.logger.error("❌ Invalid system A regime assignments (must be non-negative integers)")
                return False

            if any(not isinstance(x, int) or x < 0 for x in system_b_assignments):
                self.logger.error("❌ Invalid system B regime assignments (must be non-negative integers)")
                return False

            return True

        except Exception as e:
            self.logger.error(f"❌ Input data validation failed: {e}")
            return False

    async def _perform_semantic_consensus_validation(
        self,
        system_a_assignments: List[int],
        system_b_assignments: List[int],
        market_data: Optional[pd.DataFrame],
        context: str
    ) -> Dict[str, Any]:
        """Perform semantic consensus validation using regime mapping."""
        try:
            self.logger.info("🧠 Performing semantic consensus validation")

            # Perform semantic divergence assessment
            semantic_assessment = await self._perform_semantic_divergence_assessment(
                system_a_assignments, system_b_assignments, market_data
            )

            # Calculate semantic consensus metrics
            regime_mapping = semantic_assessment.get('regime_mapping', {})
            consensus_metrics = calculate_consensus_metrics(
                system_a_assignments, system_b_assignments,
                regime_mapping=regime_mapping,
                verbose=True
            )

            # Calculate disagreement metrics
            disagreement_metrics = calculate_disagreement_metrics(
                system_a_assignments, system_b_assignments,
                verbose=True
            )

            # Determine validation success
            consensus_score = consensus_metrics.get('consensus_score', 0.0)
            success = consensus_score >= self.config.consensus_threshold

            # Create comprehensive validation result
            validation_result = {
                'success': success,
                'consensus_score': consensus_score,
                'consensus_metrics': consensus_metrics,
                'disagreement_metrics': disagreement_metrics,
                'semantic_assessment': semantic_assessment,
                'regime_mapping': regime_mapping,
                'used_semantic_approach': True,
                'validation_method': 'semantic_consensus',
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'statistics': {
                    'system_a_regime_count': len(set(system_a_assignments)),
                    'system_b_regime_count': len(set(system_b_assignments)),
                    'total_samples': min(len(system_a_assignments), len(system_b_assignments)),
                    'mapping_quality': semantic_assessment.get('mapping_quality', 0.0),
                    'consensus_improvement': semantic_assessment.get('consensus_improvement', 0.0)
                }
            }

            return validation_result

        except Exception as e:
            self.logger.error(f"❌ Semantic consensus validation failed: {e}")
            return self._create_validation_result(
                success=False,
                error=f"Semantic consensus validation failed: {e}",
                context=context
            )

    async def _perform_basic_consensus_validation(
        self,
        system_a_assignments: List[int],
        system_b_assignments: List[int],
        context: str
    ) -> Dict[str, Any]:
        """Perform basic consensus validation without semantic mapping."""
        try:
            self.logger.info("📊 Performing basic consensus validation")

            # Simple consensus calculation
            min_length = min(len(system_a_assignments), len(system_b_assignments))
            agreements = sum(1 for i in range(min_length) if system_a_assignments[i] == system_b_assignments[i])
            consensus_score = agreements / min_length if min_length > 0 else 0.0

            # Determine validation success
            success = consensus_score >= self.config.consensus_threshold

            return self._create_validation_result(
                success=success,
                consensus_score=consensus_score,
                context=context,
                validation_method='basic_consensus',
                used_semantic_approach=False
            )

        except Exception as e:
            self.logger.error(f"❌ Basic consensus validation failed: {e}")
            return self._create_validation_result(
                success=False,
                error=f"Basic consensus validation failed: {e}",
                context=context
            )

    async def _perform_semantic_divergence_assessment(
        self,
        system_a_assignments: List[int],
        system_b_assignments: List[int],
        market_data: Optional[pd.DataFrame]
    ) -> Dict[str, Any]:
        """Perform semantic divergence assessment for regime mapping."""
        try:
            min_length = min(len(system_a_assignments), len(system_b_assignments))
            system_a_assignments = np.array(system_a_assignments[:min_length])
            system_b_assignments = np.array(system_b_assignments[:min_length])

            # For pipeline-level validation, we'll use a distribution-based approach
            # since we may not have direct access to market data features

            # Calculate regime distributions
            system_a_distribution = self._calculate_regime_distribution(system_a_assignments)
            system_b_distribution = self._calculate_regime_distribution(system_b_assignments)

            # Find optimal regime mapping using distribution similarity
            regime_mapping = self._find_optimal_regime_mapping_by_distribution(system_a_distribution, system_b_distribution)

            if not regime_mapping:
                # Fallback to numerical comparison
                disagreement_mask = system_a_assignments != system_b_assignments
                semantic_divergence_rate = np.mean(disagreement_mask)

                return {
                    'semantic_divergence_rate': semantic_divergence_rate,
                    'regime_mapping': {},
                    'mapping_quality': 0.5,
                    'raw_consensus': 1.0 - semantic_divergence_rate,
                    'semantic_consensus': 1.0 - semantic_divergence_rate,
                    'consensus_improvement': 0.0,
                    'assessment_method': 'numerical_fallback'
                }

            # Calculate semantic divergence using mapped regimes
            semantic_assignments = self._apply_regime_mapping(system_b_assignments, regime_mapping)
            semantic_disagreement_mask = system_a_assignments != semantic_assignments
            semantic_divergence_rate = np.mean(semantic_disagreement_mask)

            # Calculate mapping quality metrics
            mapping_quality = self._calculate_mapping_quality_by_distribution(system_a_distribution, system_b_distribution, regime_mapping)

            # Calculate semantic consensus improvement
            raw_agreements = np.sum(system_a_assignments == system_b_assignments)
            raw_consensus = raw_agreements / min_length if min_length > 0 else 0.0
            semantic_agreements = np.sum(system_a_assignments == semantic_assignments)
            semantic_consensus = semantic_agreements / min_length if min_length > 0 else 0.0
            consensus_improvement = semantic_consensus - raw_consensus

            return {
                'semantic_divergence_rate': semantic_divergence_rate,
                'regime_mapping': regime_mapping,
                'mapping_quality': mapping_quality,
                'raw_consensus': raw_consensus,
                'semantic_consensus': semantic_consensus,
                'consensus_improvement': consensus_improvement,
                'assessment_method': 'distribution_based',
                'system_a_distribution': system_a_distribution,
                'system_b_distribution': system_b_distribution
            }

        except Exception as e:
            self.logger.error(f"❌ Semantic divergence assessment failed: {e}")
            return {
                'semantic_divergence_rate': 1.0,
                'regime_mapping': {},
                'mapping_quality': 0.0,
                'raw_consensus': 0.0,
                'semantic_consensus': 0.0,
                'consensus_improvement': 0.0,
                'assessment_method': 'failed'
            }

    def _calculate_regime_distribution(self, assignments: np.ndarray) -> Dict[str, float]:
        """Calculate the distribution of regime assignments."""
        try:
            if len(assignments) == 0:
                return {}

            total_assignments = len(assignments)
            regime_counts = {}

            for assignment in assignments:
                regime_counts[assignment] = regime_counts.get(assignment, 0) + 1

            # Convert to percentages
            regime_distribution = {}
            for regime, count in regime_counts.items():
                key = f'regime_{regime}'
                percentage = (count / total_assignments) * 100
                regime_distribution[key] = percentage

            return regime_distribution

        except Exception as e:
            self.logger.warning(f"⚠️ Distribution calculation failed: {e}")
            return {}

    def _find_optimal_regime_mapping_by_distribution(self, system_a_distribution: Dict[str, float], system_b_distribution: Dict[str, float]) -> Dict[int, int]:
        """Find optimal mapping between two systems' regimes using distribution similarity."""
        try:
            if not system_a_distribution or not system_b_distribution:
                return {}

            # Extract regime IDs and their percentages
            system_a_regimes = {}
            system_b_regimes = {}

            for key, percentage in system_a_distribution.items():
                regime_id = int(key.replace('regime_', ''))
                system_a_regimes[regime_id] = percentage

            for key, percentage in system_b_distribution.items():
                regime_id = int(key.replace('regime_', ''))
                system_b_regimes[regime_id] = percentage

            # Create mapping based on distribution similarity
            regime_mapping = {}
            used_system_a_regimes = set()

            # Sort regimes by size (largest first) for better mapping
            system_a_sorted = sorted(system_a_regimes.items(), key=lambda x: x[1], reverse=True)
            system_b_sorted = sorted(system_b_regimes.items(), key=lambda x: x[1], reverse=True)

            # Map largest system B regime to largest system A regime, etc.
            for i, (system_b_regime, system_b_percentage) in enumerate(system_b_sorted):
                if i < len(system_a_sorted) and system_a_sorted[i][0] not in used_system_a_regimes:
                    system_a_regime = system_a_sorted[i][0]
                    regime_mapping[system_b_regime] = system_a_regime
                    used_system_a_regimes.add(system_a_regime)

            return regime_mapping

        except Exception as e:
            self.logger.warning(f"⚠️ Distribution-based mapping failed: {e}")
            return {}

    def _apply_regime_mapping(self, system_b_assignments: np.ndarray, regime_mapping: Dict[int, int]) -> np.ndarray:
        """Apply regime mapping to system B assignments."""
        try:
            mapped_assignments = system_b_assignments.copy()

            for system_b_regime, system_a_regime in regime_mapping.items():
                mask = system_b_assignments == system_b_regime
                mapped_assignments[mask] = system_a_regime

            return mapped_assignments

        except Exception as e:
            self.logger.warning(f"⚠️ Regime mapping application failed: {e}")
            return system_b_assignments

    def _calculate_mapping_quality_by_distribution(self, system_a_distribution: Dict[str, float], system_b_distribution: Dict[str, float], regime_mapping: Dict[int, int]) -> float:
        """Calculate quality metrics for the regime mapping based on distribution similarity."""
        try:
            if not regime_mapping:
                return 0.0

            total_similarity = 0.0
            mapping_count = 0

            for system_b_regime, system_a_regime in regime_mapping.items():
                system_b_key = f'regime_{system_b_regime}'
                system_a_key = f'regime_{system_a_regime}'

                if system_b_key in system_b_distribution and system_a_key in system_a_distribution:
                    system_b_percentage = system_b_distribution[system_b_key]
                    system_a_percentage = system_a_distribution[system_a_key]

                    # Calculate similarity (higher is better, max difference is 100%)
                    similarity = 1.0 - abs(system_b_percentage - system_a_percentage) / 100.0
                    total_similarity += similarity
                    mapping_count += 1

            if mapping_count == 0:
                return 0.0

            # Average similarity as quality metric
            quality = total_similarity / mapping_count
            return max(0.0, quality)

        except Exception as e:
            self.logger.warning(f"⚠️ Mapping quality calculation failed: {e}")
            return 0.0

    def _check_consensus_alerts(self, validation_result: Dict[str, Any], context: str) -> None:
        """Check for consensus alerts and log them."""
        try:
            consensus_score = validation_result.get('consensus_score', 0.0)

            if consensus_score < self.config.alert_consensus_threshold:
                self.consensus_alerts += 1
                self.logger.warning(
                    f"🚨 CONSENSUS ALERT: Low consensus detected in {context} "
                    f"(score: {consensus_score:.3f} < {self.config.alert_consensus_threshold})"
                )

                # Log additional details
                if validation_result.get('used_semantic_approach', False):
                    semantic_assessment = validation_result.get('semantic_assessment', {})
                    mapping_quality = semantic_assessment.get('mapping_quality', 0.0)
                    consensus_improvement = semantic_assessment.get('consensus_improvement', 0.0)

                    self.logger.warning(
                        f"   📊 Mapping quality: {mapping_quality:.3f}, "
                        f"Consensus improvement: {consensus_improvement:.3f}"
                    )

        except Exception as e:
            self.logger.error(f"❌ Consensus alert checking failed: {e}")

    def _record_validation_history(self, validation_result: Dict[str, Any], context: str) -> None:
        """Record validation result in history."""
        try:
            # Add timestamp and context
            history_entry = validation_result.copy()
            history_entry['timestamp'] = datetime.now().isoformat()
            history_entry['context'] = context

            # Add to history
            self.validation_history.append(history_entry)

            # Maintain history size limit
            if len(self.validation_history) > self.max_history:
                self.validation_history = self.validation_history[-self.max_history:]

        except Exception as e:
            self.logger.error(f"❌ Validation history recording failed: {e}")

    def _create_validation_result(
        self,
        success: bool,
        consensus_score: float = 0.0,
        error: str = None,
        context: str = "unknown",
        validation_method: str = "unknown",
        used_semantic_approach: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a standardized validation result."""
        result = {
            'success': success,
            'consensus_score': consensus_score,
            'validation_method': validation_method,
            'used_semantic_approach': used_semantic_approach,
            'context': context,
            'timestamp': datetime.now().isoformat(),
            **kwargs
        }

        if error:
            result['error'] = error

        return result

    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return {
            'total_validations': self.total_validations,
            'successful_validations': self.successful_validations,
            'failed_validations': self.failed_validations,
            'consensus_alerts': self.consensus_alerts,
            'success_rate': self.successful_validations / max(1, self.total_validations),
            'is_initialized': self.is_initialized,
            'semantic_consensus_available': SEMANTIC_CONSENSUS_AVAILABLE,
            'history_size': len(self.validation_history)
        }

    def get_recent_validations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent validation results."""
        return self.validation_history[-limit:] if self.validation_history else []
