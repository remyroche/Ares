"""
Bayesian Probability Updates for Continuous Learning
Continuously updates probability estimates with new market evidence
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from collections import deque
from dataclasses import dataclass
from scipy.stats import beta, norm

from src.utils.logger import system_logger
from src.core.decorators import handles_errors


@dataclass
class BayesianUpdate:
    """Result of a Bayesian probability update"""
    target: str
    regime: str
    prior_alpha: float
    prior_beta: float
    posterior_alpha: float
    posterior_beta: float
    updated_probability: float
    confidence_interval: Tuple[float, float]
    confidence: float
    timestamp: datetime
    evidence_count: int


class BayesianProbabilityUpdater:
    """
    Bayesian Probability Updates for continuous learning.
    Continuously updates probability estimates with new market evidence.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Bayesian Probability Updater.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild('BayesianUpdater')
        
        # Configuration
        self.bayesian_config = config.get('bayesian_updates', {})
        self.regime_names = [f"regime_{i:02d}" for i in range(20)]  # regime_00 to regime_19
        self.price_targets = [f"{i*0.1:.1f}%" for i in range(1, 21)]  # 0.1% to 2.0%
        
        # Bayesian parameters
        self.prior_alpha = self.bayesian_config.get('prior_alpha', 1.0)  # Prior success count
        self.prior_beta = self.bayesian_config.get('prior_beta', 1.0)    # Prior failure count
        self.confidence_level = self.bayesian_config.get('confidence_level', 0.95)
        self.min_evidence = self.bayesian_config.get('min_evidence', 10)
        
        # Storage
        self.bayesian_models: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.update_history: Dict[str, List[BayesianUpdate]] = {}
        self.evidence_cache: Dict[str, deque] = {}
        
        # Initialize models
        self._initialize_bayesian_models()
        
    def _initialize_bayesian_models(self) -> None:
        """Initialize Bayesian models for all regime-target combinations"""
        
        for regime in self.regime_names:
            self.bayesian_models[regime] = {}
            self.update_history[regime] = []
            self.evidence_cache[regime] = deque(maxlen=1000)
            
            for target in self.price_targets:
                self.bayesian_models[regime][target] = {
                    'alpha': self.prior_alpha,
                    'beta': self.prior_beta,
                    'evidence_count': 0,
                    'last_updated': datetime.now()
                }
    
    @handles_errors(exceptions=(ValueError, AttributeError), default_return=False, context='Bayesian updater initialization')
    async def initialize(self) -> bool:
        """
        Initialize the Bayesian Probability Updater.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.logger.info("Initializing Bayesian Probability Updater...")
            
            # Load existing models if available
            await self._load_existing_models()
            
            self.logger.info("✅ Bayesian Probability Updater initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Bayesian Probability Updater initialization failed: {e}")
            return False
    
    async def _load_existing_models(self) -> None:
        """Load existing Bayesian models from storage"""
        try:
            # This would load from your existing model storage
            # For now, models are initialized with default priors
            self.logger.info("Loaded existing Bayesian models (or initialized with defaults)")
        except Exception as e:
            self.logger.warning(f"Could not load existing models: {e}")
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='Bayesian probability update')
    async def update_probability(
        self,
        target: str,
        regime: str,
        success: bool,
        confidence: float = 1.0,
        metadata: Dict[str, Any] = None
    ) -> Optional[BayesianUpdate]:
        """
        Update probability for a specific target-regime combination.
        
        Args:
            target: Price target (e.g., "0.2%")
            regime: HMM regime name
            success: Whether the target was hit (True) or not (False)
            confidence: Confidence in the observation (0-1)
            metadata: Additional metadata about the observation
            
        Returns:
            BayesianUpdate: Result of the Bayesian update
        """
        try:
            # Validate inputs
            if regime not in self.regime_names:
                self.logger.error(f"Invalid regime: {regime}")
                return None
            
            if target not in self.price_targets:
                self.logger.error(f"Invalid target: {target}")
                return None
            
            # Get current model parameters
            model = self.bayesian_models[regime][target]
            current_alpha = model['alpha']
            current_beta = model['beta']
            
            # Update parameters based on new evidence
            if success:
                # Success: increment alpha
                new_alpha = current_alpha + confidence
                new_beta = current_beta
            else:
                # Failure: increment beta
                new_alpha = current_alpha
                new_beta = current_beta + confidence
            
            # Calculate updated probability
            updated_probability = new_alpha / (new_alpha + new_beta)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_credible_interval(
                new_alpha, new_beta, self.confidence_level
            )
            
            # Calculate confidence in the estimate
            confidence_in_estimate = self._calculate_bayesian_confidence(new_alpha, new_beta)
            
            # Update model
            model['alpha'] = new_alpha
            model['beta'] = new_beta
            model['evidence_count'] += 1
            model['last_updated'] = datetime.now()
            
            # Store evidence
            evidence = {
                'target': target,
                'success': success,
                'confidence': confidence,
                'timestamp': datetime.now(),
                'metadata': metadata or {}
            }
            self.evidence_cache[regime].append(evidence)
            
            # Create update result
            update_result = BayesianUpdate(
                target=target,
                regime=regime,
                prior_alpha=current_alpha,
                prior_beta=current_beta,
                posterior_alpha=new_alpha,
                posterior_beta=new_beta,
                updated_probability=updated_probability,
                confidence_interval=confidence_interval,
                confidence=confidence_in_estimate,
                timestamp=datetime.now(),
                evidence_count=model['evidence_count']
            )
            
            # Store in history
            self.update_history[regime].append(update_result)
            
            self.logger.debug(f"Updated probability for {regime}-{target}: {updated_probability:.3f}")
            
            return update_result
            
        except Exception as e:
            self.logger.error(f"Error updating probability for {regime}-{target}: {e}")
            return None
    
    def _calculate_credible_interval(
        self,
        alpha: float,
        beta: float,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate credible interval for beta distribution"""
        
        try:
            # Calculate percentiles
            lower_percentile = (1 - confidence_level) / 2
            upper_percentile = 1 - lower_percentile
            
            lower_bound = beta.ppf(lower_percentile, alpha, beta)
            upper_bound = beta.ppf(upper_percentile, alpha, beta)
            
            return (lower_bound, upper_bound)
            
        except Exception as e:
            self.logger.error(f"Error calculating credible interval: {e}")
            return (0.0, 1.0)
    
    def _calculate_bayesian_confidence(self, alpha: float, beta: float) -> float:
        """Calculate confidence in the Bayesian estimate"""
        
        # Higher confidence when we have more evidence
        total_evidence = alpha + beta - 2  # Subtract initial prior
        max_confidence_evidence = 100  # Max confidence at 100 observations
        
        confidence = min(1.0, total_evidence / max_confidence_evidence)
        
        # Also consider the variance (lower variance = higher confidence)
        if alpha + beta > 2:
            variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
            # Convert variance to confidence (lower variance = higher confidence)
            variance_confidence = max(0.0, 1.0 - variance * 10)  # Scale factor
            confidence = (confidence + variance_confidence) / 2
        
        return confidence
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='get updated probabilities')
    async def get_updated_probabilities(
        self,
        regime: str,
        min_evidence: int = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get updated probabilities for a specific regime.
        
        Args:
            regime: HMM regime name
            min_evidence: Minimum evidence required for inclusion
            
        Returns:
            Dict: Updated probabilities with confidence intervals
        """
        try:
            if regime not in self.regime_names:
                self.logger.error(f"Invalid regime: {regime}")
                return None
            
            min_evidence = min_evidence or self.min_evidence
            probabilities = {}
            
            for target in self.price_targets:
                model = self.bayesian_models[regime][target]
                
                if model['evidence_count'] >= min_evidence:
                    alpha = model['alpha']
                    beta = model['beta']
                    
                    probability = alpha / (alpha + beta)
                    confidence_interval = self._calculate_credible_interval(
                        alpha, beta, self.confidence_level
                    )
                    confidence = self._calculate_bayesian_confidence(alpha, beta)
                    
                    probabilities[target] = {
                        'probability': probability,
                        'confidence_interval': confidence_interval,
                        'confidence': confidence,
                        'evidence_count': model['evidence_count'],
                        'last_updated': model['last_updated']
                    }
            
            return {
                'regime': regime,
                'probabilities': probabilities,
                'total_targets': len(probabilities),
                'min_evidence': min_evidence,
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting updated probabilities for regime {regime}: {e}")
            return None
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='batch probability update')
    async def batch_update_probabilities(
        self,
        regime: str,
        observations: List[Dict[str, Any]]
    ) -> Optional[List[BayesianUpdate]]:
        """
        Update probabilities for multiple observations at once.
        
        Args:
            regime: HMM regime name
            observations: List of observations with target, success, confidence
            
        Returns:
            List[BayesianUpdate]: Results of all updates
        """
        try:
            if regime not in self.regime_names:
                self.logger.error(f"Invalid regime: {regime}")
                return None
            
            update_results = []
            
            for observation in observations:
                target = observation.get('target')
                success = observation.get('success')
                confidence = observation.get('confidence', 1.0)
                metadata = observation.get('metadata', {})
                
                if target and success is not None:
                    update_result = await self.update_probability(
                        target, regime, success, confidence, metadata
                    )
                    if update_result:
                        update_results.append(update_result)
            
            self.logger.info(f"Batch updated {len(update_results)} probabilities for regime {regime}")
            
            return update_results
            
        except Exception as e:
            self.logger.error(f"Error in batch update for regime {regime}: {e}")
            return None
    
    @handles_errors(exceptions=(ValueError, KeyError), default_return=None, context='regime comparison')
    async def compare_regime_probabilities(
        self,
        regime1: str,
        regime2: str,
        target: str = None
    ) -> Optional[Dict[str, Any]]:
        """
        Compare probabilities between two regimes.
        
        Args:
            regime1: First regime to compare
            regime2: Second regime to compare
            target: Specific target to compare (None for all targets)
            
        Returns:
            Dict: Comparison results
        """
        try:
            if regime1 not in self.regime_names or regime2 not in self.regime_names:
                self.logger.error("Invalid regime names for comparison")
                return None
            
            comparison_results = {
                'regime1': regime1,
                'regime2': regime2,
                'target_comparisons': {},
                'overall_comparison': {},
                'timestamp': datetime.now()
            }
            
            targets_to_compare = [target] if target else self.price_targets
            
            for target_name in targets_to_compare:
                model1 = self.bayesian_models[regime1][target_name]
                model2 = self.bayesian_models[regime2][target_name]
                
                if model1['evidence_count'] >= self.min_evidence and model2['evidence_count'] >= self.min_evidence:
                    prob1 = model1['alpha'] / (model1['alpha'] + model1['beta'])
                    prob2 = model2['alpha'] / (model2['alpha'] + model2['beta'])
                    
                    # Statistical significance test
                    significance = self._test_probability_difference(
                        model1['alpha'], model1['beta'],
                        model2['alpha'], model2['beta']
                    )
                    
                    comparison_results['target_comparisons'][target_name] = {
                        'regime1_probability': prob1,
                        'regime2_probability': prob2,
                        'difference': prob1 - prob2,
                        'relative_difference': (prob1 - prob2) / prob2 if prob2 > 0 else 0,
                        'statistical_significance': significance,
                        'regime1_evidence': model1['evidence_count'],
                        'regime2_evidence': model2['evidence_count']
                    }
            
            # Overall comparison
            if comparison_results['target_comparisons']:
                differences = [comp['difference'] for comp in comparison_results['target_comparisons'].values()]
                significant_differences = [comp['statistical_significance'] for comp in comparison_results['target_comparisons'].values()]
                
                comparison_results['overall_comparison'] = {
                    'avg_difference': np.mean(differences),
                    'significant_differences': sum(significant_differences),
                    'total_comparisons': len(differences),
                    'significance_rate': sum(significant_differences) / len(differences) if differences else 0
                }
            
            return comparison_results
            
        except Exception as e:
            self.logger.error(f"Error comparing regimes {regime1} and {regime2}: {e}")
            return None
    
    def _test_probability_difference(
        self,
        alpha1: float, beta1: float,
        alpha2: float, beta2: float
    ) -> bool:
        """Test if two probabilities are significantly different"""
        
        try:
            # Calculate probabilities
            prob1 = alpha1 / (alpha1 + beta1)
            prob2 = alpha2 / (alpha2 + beta2)
            
            # Calculate standard errors
            se1 = np.sqrt((alpha1 * beta1) / ((alpha1 + beta1) ** 2 * (alpha1 + beta1 + 1)))
            se2 = np.sqrt((alpha2 * beta2) / ((alpha2 + beta2) ** 2 * (alpha2 + beta2 + 1)))
            
            # Calculate difference and combined standard error
            diff = prob1 - prob2
            se_diff = np.sqrt(se1**2 + se2**2)
            
            # Z-test for difference
            if se_diff > 0:
                z_score = diff / se_diff
                # Two-tailed test at 95% confidence
                significant = abs(z_score) > 1.96
            else:
                significant = False
            
            return significant
            
        except Exception as e:
            self.logger.error(f"Error testing probability difference: {e}")
            return False
    
    def get_bayesian_summary(self) -> Dict[str, Any]:
        """Get summary of Bayesian models"""
        
        summary = {
            'system_status': 'active',
            'regime_count': len(self.regime_names),
            'target_count': len(self.price_targets),
            'total_models': len(self.regime_names) * len(self.price_targets),
            'prior_parameters': {
                'alpha': self.prior_alpha,
                'beta': self.prior_beta
            },
            'confidence_level': self.confidence_level,
            'min_evidence': self.min_evidence,
            'regime_summaries': {}
        }
        
        for regime in self.regime_names:
            regime_models = self.bayesian_models[regime]
            evidence_counts = [model['evidence_count'] for model in regime_models.values()]
            
            summary['regime_summaries'][regime] = {
                'total_evidence': sum(evidence_counts),
                'avg_evidence': np.mean(evidence_counts),
                'max_evidence': max(evidence_counts),
                'min_evidence': min(evidence_counts),
                'models_with_sufficient_evidence': sum(1 for count in evidence_counts if count >= self.min_evidence)
            }
        
        return summary