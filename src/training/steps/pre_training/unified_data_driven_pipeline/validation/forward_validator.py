"""
Forward Validation Module

This module implements forward validation with walk-forward holdout testing
to validate pipeline performance on unseen future data and regime changes.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std,
    validate_finite, validate_positive, memory_checkpoint
)


@dataclass
class ForwardValidationConfig:
    """Configuration for forward validation."""
    
    # Holdout parameters
    holdout_ratio: float = 0.2                # Ratio of data for holdout
    min_holdout_days: int = 30                # Minimum holdout days
    max_holdout_days: int = 90                # Maximum holdout days
    
    # Walk-forward parameters
    walk_forward_steps: int = 5               # Number of walk-forward steps
    step_size_days: int = 7                   # Days per step
    min_step_size_days: int = 1               # Minimum step size
    
    # Regime detection
    enable_regime_detection: bool = True     # Enable regime change detection
    regime_window_days: int = 30             # Window for regime detection
    regime_change_threshold: float = 0.15    # Threshold for regime change
    
    # Performance metrics
    min_forward_ic: float = 0.01             # Minimum forward IC
    min_forward_sharpe: float = 0.1          # Minimum forward Sharpe
    max_ic_decay: float = 0.5                # Maximum IC decay
    max_sharpe_decay: float = 0.3            # Maximum Sharpe decay
    
    # Validation criteria
    require_positive_forward_performance: bool = True
    require_regime_stability: bool = True
    require_consistent_performance: bool = True
    
    # Logging
    verbose: bool = True


@dataclass
class ForwardValidationStep:
    """Represents a single forward validation step."""
    step_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    regime_stable: bool = True
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    predictions: Optional[pd.Series] = None
    actual: Optional[pd.Series] = None


@dataclass
class ForwardValidationResult:
    """Result of forward validation."""
    
    # Validation steps
    steps: List[ForwardValidationStep] = field(default_factory=list)
    total_steps: int = 0
    successful_steps: int = 0
    
    # Performance metrics
    forward_ic: float = 0.0
    forward_sharpe: float = 0.0
    ic_decay: float = 0.0
    sharpe_decay: float = 0.0
    regime_sensitivity: float = 0.0
    
    # Validation status
    passed_forward_validation: bool = False
    regime_stable: bool = True
    performance_consistent: bool = True
    
    # Issues detected
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ForwardValidator:
    """
    Forward validation with walk-forward holdout testing.
    
    Validates:
    1. Performance on unseen future data
    2. IC and Sharpe decay over time
    3. Regime sensitivity and stability
    4. Consistent performance across steps
    """
    
    def __init__(self, config: Optional[ForwardValidationConfig] = None):
        """Initialize the forward validator."""
        self.config = config or ForwardValidationConfig()
        self.logger = logging.getLogger(__name__)
        
        if self.config.verbose:
            tprint("🔮 Initializing ForwardValidator")
    
    def perform_forward_validation(self, 
                                 data: pd.DataFrame,
                                 targets: pd.Series,
                                 pipeline: callable,
                                 start_date: Optional[datetime] = None,
                                 end_date: Optional[datetime] = None) -> ForwardValidationResult:
        """
        Perform forward validation with walk-forward holdout.
        
        Args:
            data: Input features
            targets: Target labels
            pipeline: Trained pipeline to validate
            start_date: Start date for validation
            end_date: End date for validation
            
        Returns:
            ForwardValidationResult
        """
        if self.config.verbose:
            tprint("🔮 Starting forward validation")
        
        result = ForwardValidationResult()
        
        # Determine validation period
        if start_date is None:
            start_date = data.index[0]
        if end_date is None:
            end_date = data.index[-1]
        
        # Calculate holdout period
        total_days = (end_date - start_date).days
        holdout_days = min(
            max(int(total_days * self.config.holdout_ratio), self.config.min_holdout_days),
            self.config.max_holdout_days
        )
        
        # Split data into train and holdout
        split_date = end_date - timedelta(days=holdout_days)
        train_data = data[data.index < split_date]
        train_targets = targets[targets.index < split_date]
        holdout_data = data[data.index >= split_date]
        holdout_targets = targets[targets.index >= split_date]
        
        if self.config.verbose:
            tprint(f"📊 Train period: {train_data.index[0]} to {train_data.index[-1]}")
            tprint(f"📊 Holdout period: {holdout_data.index[0]} to {holdout_data.index[-1]}")
        
        # Perform walk-forward validation
        steps = self._create_walk_forward_steps(holdout_data, holdout_targets)
        result.steps = steps
        result.total_steps = len(steps)
        
        # Execute each step
        for step in steps:
            if self.config.verbose:
                tprint(f"🔄 Executing step {step.step_id}/{len(steps)}")
            
            # Train on historical data + step training data
            step_train_data = pd.concat([train_data, data[data.index < step.test_start]])
            step_train_targets = pd.concat([train_targets, targets[targets.index < step.test_start]])
            
            # Test on step test data
            step_test_data = data[
                (data.index >= step.test_start) & 
                (data.index < step.test_end)
            ]
            step_test_targets = targets[
                (targets.index >= step.test_start) & 
                (targets.index < step.test_end)
            ]
            
            # Execute pipeline
            try:
                predictions = pipeline.predict(step_test_data)
                step.predictions = predictions
                step.actual = step_test_targets
                
                # Calculate performance metrics
                step.performance_metrics = self._calculate_step_metrics(
                    predictions, step_test_targets
                )
                
                # Check regime stability
                step.regime_stable = self._check_regime_stability(
                    step_train_data, step_test_data
                )
                
                result.successful_steps += 1
                
            except Exception as e:
                self.logger.warning(f"Step {step.step_id} failed: {e}")
                step.performance_metrics = {'ic': 0.0, 'sharpe': 0.0}
                step.regime_stable = False
        
        # Calculate overall metrics
        self._calculate_overall_metrics(result)
        
        # Validate results
        self._validate_forward_results(result)
        
        # Generate recommendations
        result.recommendations = self._generate_recommendations(result)
        
        if self.config.verbose:
            tprint_success(f"✅ Forward validation completed")
            tprint(f"📊 Forward IC: {result.forward_ic:.4f}")
            tprint(f"📊 Forward Sharpe: {result.forward_sharpe:.4f}")
            tprint(f"📊 IC Decay: {result.ic_decay:.4f}")
            tprint(f"📊 Passed: {result.passed_forward_validation}")
        
        return result
    
    def _create_walk_forward_steps(self, 
                                 holdout_data: pd.DataFrame,
                                 holdout_targets: pd.Series) -> List[ForwardValidationStep]:
        """Create walk-forward validation steps."""
        steps = []
        
        # Calculate step size
        total_days = (holdout_data.index[-1] - holdout_data.index[0]).days
        step_size_days = max(
            min(self.config.step_size_days, total_days // self.config.walk_forward_steps),
            self.config.min_step_size_days
        )
        
        # Create steps
        current_date = holdout_data.index[0]
        step_id = 1
        
        while current_date < holdout_data.index[-1]:
            # Calculate step boundaries
            step_end = current_date + timedelta(days=step_size_days)
            if step_end > holdout_data.index[-1]:
                step_end = holdout_data.index[-1]
            
            # Create step
            step = ForwardValidationStep(
                step_id=step_id,
                train_start=holdout_data.index[0],
                train_end=current_date,
                test_start=current_date,
                test_end=step_end
            )
            
            steps.append(step)
            
            # Move to next step
            current_date = step_end
            step_id += 1
            
            # Limit number of steps
            if len(steps) >= self.config.walk_forward_steps:
                break
        
        return steps
    
    def _calculate_step_metrics(self, 
                              predictions: pd.Series,
                              actual: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics for a step."""
        try:
            # Align predictions and actual
            common_index = predictions.index.intersection(actual.index)
            if len(common_index) == 0:
                return {'ic': 0.0, 'sharpe': 0.0, 'mse': 0.0}
            
            pred_aligned = predictions.loc[common_index]
            actual_aligned = actual.loc[common_index]
            
            # Calculate IC
            ic = pred_aligned.corr(actual_aligned)
            ic = ic if not np.isnan(ic) else 0.0
            
            # Calculate Sharpe ratio
            returns = actual_aligned.pct_change().dropna()
            if len(returns) == 0:
                sharpe = 0.0
            else:
                sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0.0
            
            # Calculate MSE
            mse = np.mean((pred_aligned - actual_aligned) ** 2)
            
            return {
                'ic': ic,
                'sharpe': sharpe,
                'mse': mse
            }
        except Exception as e:
            self.logger.warning(f"Step metrics calculation failed: {e}")
            return {'ic': 0.0, 'sharpe': 0.0, 'mse': 0.0}
    
    def _check_regime_stability(self, 
                              train_data: pd.DataFrame,
                              test_data: pd.DataFrame) -> bool:
        """Check regime stability between train and test periods."""
        try:
            if not self.config.enable_regime_detection:
                return True
            
            # Calculate regime indicators
            train_vol = train_data['close'].pct_change().rolling(20).std().mean()
            test_vol = test_data['close'].pct_change().rolling(20).std().mean()
            
            train_trend = train_data['close'].rolling(20).mean().iloc[-1]
            test_trend = test_data['close'].rolling(20).mean().iloc[-1]
            
            # Check for regime change
            vol_change = abs(test_vol - train_vol) / train_vol if train_vol > 0 else 0
            trend_change = abs(test_trend - train_trend) / train_trend if train_trend > 0 else 0
            
            # Regime is stable if changes are below threshold
            return (vol_change < self.config.regime_change_threshold and 
                   trend_change < self.config.regime_change_threshold)
        except:
            return False
    
    def _calculate_overall_metrics(self, result: ForwardValidationResult) -> None:
        """Calculate overall forward validation metrics."""
        try:
            # Calculate forward IC and Sharpe
            ic_scores = [step.performance_metrics.get('ic', 0.0) for step in result.steps]
            sharpe_scores = [step.performance_metrics.get('sharpe', 0.0) for step in result.steps]
            
            result.forward_ic = np.mean(ic_scores)
            result.forward_sharpe = np.mean(sharpe_scores)
            
            # Calculate decay metrics
            if len(ic_scores) >= 2:
                result.ic_decay = ic_scores[0] - ic_scores[-1] if ic_scores[0] > 0 else 0
                result.sharpe_decay = sharpe_scores[0] - sharpe_scores[-1] if sharpe_scores[0] > 0 else 0
            
            # Calculate regime sensitivity
            stable_steps = sum(1 for step in result.steps if step.regime_stable)
            result.regime_sensitivity = 1.0 - (stable_steps / len(result.steps)) if result.steps else 0.0
            
        except Exception as e:
            self.logger.warning(f"Overall metrics calculation failed: {e}")
    
    def _validate_forward_results(self, result: ForwardValidationResult) -> None:
        """Validate forward validation results."""
        try:
            # Check forward performance
            if result.forward_ic < self.config.min_forward_ic:
                result.issues.append(f"Forward IC too low: {result.forward_ic:.4f}")
                result.passed_forward_validation = False
            
            if result.forward_sharpe < self.config.min_forward_sharpe:
                result.issues.append(f"Forward Sharpe too low: {result.forward_sharpe:.4f}")
                result.passed_forward_validation = False
            
            # Check decay
            if result.ic_decay > self.config.max_ic_decay:
                result.issues.append(f"IC decay too high: {result.ic_decay:.4f}")
                result.passed_forward_validation = False
            
            if result.sharpe_decay > self.config.max_sharpe_decay:
                result.issues.append(f"Sharpe decay too high: {result.sharpe_decay:.4f}")
                result.passed_forward_validation = False
            
            # Check regime stability
            stable_steps = sum(1 for step in result.steps if step.regime_stable)
            regime_stability_ratio = stable_steps / len(result.steps) if result.steps else 0
            
            if regime_stability_ratio < 0.8:  # 80% of steps should be regime stable
                result.issues.append(f"Regime instability: {regime_stability_ratio:.2f}")
                result.regime_stable = False
            
            # Check performance consistency
            ic_scores = [step.performance_metrics.get('ic', 0.0) for step in result.steps]
            ic_std = np.std(ic_scores)
            if ic_std > 0.1:  # High variance in performance
                result.issues.append(f"Performance inconsistency: IC std {ic_std:.4f}")
                result.performance_consistent = False
            
            # Overall validation
            if not result.issues:
                result.passed_forward_validation = True
            
        except Exception as e:
            self.logger.warning(f"Forward validation failed: {e}")
            result.passed_forward_validation = False
    
    def _generate_recommendations(self, result: ForwardValidationResult) -> List[str]:
        """Generate recommendations based on forward validation results."""
        recommendations = []
        
        if not result.passed_forward_validation:
            recommendations.append("Improve forward validation performance")
        
        if result.forward_ic < 0.05:
            recommendations.append("Enhance Information Coefficient through better feature engineering")
        
        if result.forward_sharpe < 0.2:
            recommendations.append("Improve Sharpe ratio through risk management")
        
        if result.ic_decay > 0.3:
            recommendations.append("Reduce IC decay through regularization")
        
        if result.sharpe_decay > 0.2:
            recommendations.append("Reduce Sharpe decay through stability improvements")
        
        if not result.regime_stable:
            recommendations.append("Improve regime stability through regime-aware modeling")
        
        if not result.performance_consistent:
            recommendations.append("Improve performance consistency across time periods")
        
        if result.successful_steps < result.total_steps * 0.8:
            recommendations.append("Increase successful validation steps")
        
        return recommendations
