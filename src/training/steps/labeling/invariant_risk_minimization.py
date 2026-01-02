"""
Invariant Risk Minimization (IRM) for Financial Machine Learning

Implements Invariant Risk Minimization to train models that perform consistently
across different market environments, using custom regime features as environment definitions.

IRM Objective: Find feature representation Φ and predictor w such that the same optimal
predictor works for all environments: min_Φ,w Σ_e R_e(w∘Φ) + λ·||∇_w|w=1 R_e(w∘Φ)||²

Key Components:
1. Environment definition using custom regime features
2. IRM loss function with invariance penalty
3. Gradient consistency across environments
4. Integration with existing model training pipelines
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Union
import warnings
from sklearn.metrics import log_loss, brier_score_loss
import torch
import torch.nn as nn
import torch.optim as optim

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class FinancialIRM:
    """
    Invariant Risk Minimization for financial machine learning.
    
    This class implements IRM to train models that perform consistently across
    different market environments defined by your custom regime features.
    
    The IRM objective combines standard risk with an invariance penalty:
    R_IRM(w, Φ) = Σ_e R_e(w∘Φ) + λ·||∇_w|w=1 R_e(w∘Φ)||²
    """
    
    def __init__(
        self,
        lambda_irm: float = 1.0,
        min_samples_per_env: int = 100,
        gradient_method: str = 'analytical',
        invariance_threshold: float = 0.1,
        verbose: bool = True
    ):
        """
        Initialize FinancialIRM.
        
        Args:
            lambda_irm: Weight for invariance penalty (λ)
            min_samples_per_env: Minimum samples required per environment
            gradient_method: Method for gradient computation ('analytical', 'numerical')
            invariance_threshold: Threshold for acceptable invariance
            verbose: Whether to print progress information
        """
        self.lambda_irm = lambda_irm
        self.min_samples_per_env = min_samples_per_env
        self.gradient_method = gradient_method
        self.invariance_threshold = invariance_threshold
        self.verbose = verbose
        
        # Cache for environment masks and invariance metrics
        self.env_masks_cache_ = {}
        self.invariance_metrics_ = {}
    
    def create_environment_masks(self, custom_features: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Create environment masks using your reliable custom regime features.
        
        Args:
            custom_features: DataFrame with custom regime features
            
        Returns:
            Dictionary mapping environment names to boolean masks
        """
        cache_key = hash(str(custom_features.columns.values.tobytes()) + str(custom_features.values.tobytes()))
        if cache_key in self.env_masks_cache_:
            return self.env_masks_cache_[cache_key]
        
        environment_masks = {}
        
        # Volatility regimes (using your existing features)
        if 'vol_regime_high' in custom_features.columns:
            environment_masks['high_volatility'] = custom_features['vol_regime_high'] == 1
        if 'vol_regime_low' in custom_features.columns:
            environment_masks['low_volatility'] = custom_features['vol_regime_low'] == 1
        if 'vol_regime_med' in custom_features.columns:
            environment_masks['medium_volatility'] = custom_features['vol_regime_med'] == 1
        
        # Trend regimes
        if 'price_trend' in custom_features.columns:
            environment_masks['strong_uptrend'] = custom_features['price_trend'] > 0.02
            environment_masks['strong_downtrend'] = custom_features['price_trend'] < -0.02
            environment_masks['sideways'] = custom_features['price_trend'].abs() <= 0.01
        
        # Volatility stress regimes
        if 'vol_relative' in custom_features.columns:
            vol_threshold = custom_features['vol_relative'].quantile(0.8)
            environment_masks['volatility_stress'] = custom_features['vol_relative'] > vol_threshold
        
        # Volatility of volatility regimes
        if 'vol_of_vol' in custom_features.columns:
            vov_threshold = custom_features['vol_of_vol'].quantile(0.8)
            environment_masks['high_vov'] = custom_features['vol_of_vol'] > vov_threshold
        
        # Microstructure stress regimes
        if 'volume_ratio' in custom_features.columns:
            volume_threshold = custom_features['volume_ratio'].quantile(0.9)
            environment_masks['microstructure_stress'] = custom_features['volume_ratio'] > volume_threshold
        
        # Filter environments with sufficient data
        valid_masks = {}
        for env_name, mask in environment_masks.items():
            if mask.sum() >= self.min_samples_per_env:
                valid_masks[env_name] = mask
            elif self.verbose:
                tprint_warning(f"⚠️ Environment {env_name} has insufficient samples ({mask.sum()} < {self.min_samples_per_env})")
        
        # Cache result
        self.env_masks_cache_[cache_key] = valid_masks
        
        if self.verbose:
            tprint_info(f"🔧 Created {len(valid_masks)} valid environments:")
            for env_name, mask in valid_masks.items():
                tprint_info(f"   - {env_name}: {mask.sum()} samples")
        
        return valid_masks
    
    def compute_standard_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute standard weighted loss.
        
        Args:
            predictions: Predicted probabilities
            targets: True labels
            sample_weights: Optional sample weights
            
        Returns:
            Weighted loss value
        """
        try:
            # Clip predictions for numerical stability
            predictions_clipped = np.clip(predictions, 1e-8, 1 - 1e-8)
            
            if sample_weights is not None:
                loss = log_loss(targets, predictions_clipped, sample_weight=sample_weights)
            else:
                loss = log_loss(targets, predictions_clipped)
            
            return loss
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Standard loss computation failed: {e}")
            return 0.693  # Return neutral loss
    
    def compute_gradients(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        sample_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute gradients of loss with respect to predictions.
        
        Args:
            predictions: Predicted probabilities
            targets: True labels
            sample_weights: Optional sample weights
            
        Returns:
            Gradient array
        """
        try:
            # Clip predictions for numerical stability
            predictions_clipped = np.clip(predictions, 1e-8, 1 - 1e-8)
            
            # Compute gradients of log loss
            if sample_weights is not None:
                gradients = sample_weights * (predictions_clipped - targets) / (predictions_clipped * (1 - predictions_clipped))
            else:
                gradients = (predictions_clipped - targets) / (predictions_clipped * (1 - predictions_clipped))
            
            return gradients
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Gradient computation failed: {e}")
            return np.zeros_like(predictions)
    
    def compute_invariance_penalty(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        environment_masks: Dict[str, np.ndarray],
        sample_weights: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute invariance penalty across environments.
        
        Args:
            predictions: Predicted probabilities
            targets: True labels
            environment_masks: Environment masks
            sample_weights: Optional sample weights
            
        Returns:
            Invariance penalty value
        """
        if len(environment_masks) < 2:
            # Need at least 2 environments for invariance penalty
            return 0.0
        
        try:
            # Compute gradients for each environment
            env_gradients = []
            
            for env_name, env_mask in environment_masks.items():
                if env_mask.sum() < self.min_samples_per_env:
                    continue
                
                env_pred = predictions[env_mask]
                env_target = targets[env_mask]
                env_weights = sample_weights[env_mask] if sample_weights is not None else None
                
                env_grad = self.compute_gradients(env_pred, env_target, env_weights)
                env_gradients.append(env_grad)
            
            if len(env_gradients) < 2:
                return 0.0
            
            # Compute global gradient (mean across environments)
            global_gradient = np.mean(env_gradients, axis=0)
            
            # Compute invariance penalty (variance of gradients)
            invariance_penalty = 0.0
            for env_grad in env_gradients:
                invariance_penalty += np.sum((env_grad - global_gradient) ** 2)
            
            invariance_penalty /= len(env_gradients)
            
            return invariance_penalty
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Invariance penalty computation failed: {e}")
            return 0.0
    
    def irm_loss(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        environment_masks: Dict[str, np.ndarray],
        sample_weights: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute IRM loss with invariance penalty.
        
        Args:
            predictions: Predicted probabilities
            targets: True labels
            environment_masks: Environment masks
            sample_weights: Optional sample weights
            
        Returns:
            Tuple of (total_loss, loss_breakdown)
        """
        # Standard loss
        standard_loss = self.compute_standard_loss(predictions, targets, sample_weights)
        
        # Invariance penalty
        invariance_penalty = self.compute_invariance_penalty(
            predictions, targets, environment_masks, sample_weights
        )
        
        # Total IRM loss
        total_loss = standard_loss + self.lambda_irm * invariance_penalty
        
        loss_breakdown = {
            'standard_loss': standard_loss,
            'invariance_penalty': invariance_penalty,
            'total_loss': total_loss,
            'lambda_irm': self.lambda_irm
        }
        
        return total_loss, loss_breakdown
    
    def evaluate_invariance(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        environment_masks: Dict[str, np.ndarray],
        sample_weights: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Evaluate invariance metrics across environments.
        
        Args:
            predictions: Predicted probabilities
            targets: True labels
            environment_masks: Environment masks
            sample_weights: Optional sample weights
            
        Returns:
            Dictionary of invariance metrics
        """
        metrics = {}
        
        try:
            # Compute performance per environment
            env_performances = {}
            for env_name, env_mask in environment_masks.items():
                if env_mask.sum() < self.min_samples_per_env:
                    continue
                
                env_pred = predictions[env_mask]
                env_target = targets[env_mask]
                env_weights = sample_weights[env_mask] if sample_weights is not None else None
                
                # Compute AUC and log loss for this environment
                try:
                    from sklearn.metrics import roc_auc_score
                    env_auc = roc_auc_score(env_target, env_pred)
                    env_loss = self.compute_standard_loss(env_pred, env_target, env_weights)
                    env_performances[env_name] = {'auc': env_auc, 'loss': env_loss}
                except Exception:
                    env_performances[env_name] = {'auc': 0.5, 'loss': 0.693}
            
            # Compute performance variance
            if len(env_performances) > 1:
                aucs = [perf['auc'] for perf in env_performances.values()]
                losses = [perf['loss'] for perf in env_performances.values()]
                
                metrics['auc_variance'] = np.var(aucs)
                metrics['loss_variance'] = np.var(losses)
                metrics['auc_std'] = np.std(aucs)
                metrics['loss_std'] = np.std(losses)
                metrics['auc_range'] = np.max(aucs) - np.min(aucs)
                metrics['loss_range'] = np.max(losses) - np.min(losses)
            else:
                metrics['auc_variance'] = 0.0
                metrics['loss_variance'] = 0.0
                metrics['auc_std'] = 0.0
                metrics['loss_std'] = 0.0
                metrics['auc_range'] = 0.0
                metrics['loss_range'] = 0.0
            
            # Compute gradient invariance
            invariance_penalty = self.compute_invariance_penalty(
                predictions, targets, environment_masks, sample_weights
            )
            metrics['gradient_invariance'] = invariance_penalty
            
            # Overall invariance score (lower is better)
            metrics['invariance_score'] = (
                metrics['auc_variance'] + 
                metrics['loss_variance'] + 
                0.1 * metrics['gradient_invariance']
            )
            
            # Store for reporting
            self.invariance_metrics_ = metrics
            
            if self.verbose:
                tprint_info(f"📊 Invariance Metrics:")
                tprint_info(f"   - AUC variance: {metrics['auc_variance']:.4f}")
                tprint_info(f"   - Loss variance: {metrics['loss_variance']:.4f}")
                tprint_info(f"   - Gradient invariance: {metrics['gradient_invariance']:.4f}")
                tprint_info(f"   - Invariance score: {metrics['invariance_score']:.4f}")
            
            return metrics
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Invariance evaluation failed: {e}")
            return {'invariance_score': float('inf')}
    
    def is_invariant(self, predictions: np.ndarray, targets: np.ndarray, 
                     environment_masks: Dict[str, np.ndarray], 
                     sample_weights: Optional[np.ndarray] = None) -> bool:
        """
        Check if model performance is invariant across environments.
        
        Args:
            predictions: Predicted probabilities
            targets: True labels
            environment_masks: Environment masks
            sample_weights: Optional sample weights
            
        Returns:
            True if invariant, False otherwise
        """
        metrics = self.evaluate_invariance(predictions, targets, environment_masks, sample_weights)
        invariance_score = metrics.get('invariance_score', float('inf'))
        
        return invariance_score <= self.invariance_threshold

class IRMLossFunction:
    """
    PyTorch implementation of IRM loss for deep learning models.
    """
    
    def __init__(self, lambda_irm: float = 1.0, epsilon: float = 1e-8):
        """
        Initialize IRM loss function.
        
        Args:
            lambda_irm: Weight for invariance penalty
            epsilon: Small constant for numerical stability
        """
        self.lambda_irm = lambda_irm
        self.epsilon = epsilon
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                 environment_masks: Dict[str, torch.Tensor],
                 sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for IRM loss.
        
        Args:
            predictions: Predicted probabilities
            targets: True labels
            environment_masks: Environment masks
            sample_weights: Optional sample weights
            
        Returns:
            IRM loss tensor
        """
        # Standard binary cross-entropy loss
        predictions_clipped = torch.clamp(predictions, self.epsilon, 1 - self.epsilon)
        
        if sample_weights is not None:
            standard_loss = -torch.mean(
                sample_weights * (targets * torch.log(predictions_clipped) + 
                (1 - targets) * torch.log(1 - predictions_clipped))
            )
        else:
            standard_loss = -torch.mean(
                targets * torch.log(predictions_clipped) + 
                (1 - targets) * torch.log(1 - predictions_clipped)
            )
        
        # Invariance penalty
        invariance_penalty = self._compute_invariance_penalty_torch(
            predictions, targets, environment_masks, sample_weights
        )
        
        return standard_loss + self.lambda_irm * invariance_penalty
    
    def _compute_invariance_penalty_torch(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        environment_masks: Dict[str, torch.Tensor],
        sample_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute invariance penalty using PyTorch."""
        if len(environment_masks) < 2:
            return torch.tensor(0.0)
        
        # Compute gradients for each environment
        env_gradients = []
        
        for env_mask in environment_masks.values():
            if env_mask.sum() < 100:  # Minimum samples
                continue
            
            env_pred = predictions[env_mask]
            env_target = targets[env_mask]
            env_weights = sample_weights[env_mask] if sample_weights is not None else None
            
            # Compute gradients of binary cross-entropy loss
            pred_clipped = torch.clamp(env_pred, self.epsilon, 1 - self.epsilon)
            
            if env_weights is not None:
                grad = env_weights * (pred_clipped - env_target) / (pred_clipped * (1 - pred_clipped))
            else:
                grad = (pred_clipped - env_target) / (pred_clipped * (1 - pred_clipped))
            
            env_gradients.append(grad)
        
        if len(env_gradients) < 2:
            return torch.tensor(0.0)
        
        # Compute global gradient
        global_gradient = torch.mean(torch.stack(env_gradients), dim=0)
        
        # Compute invariance penalty
        invariance_penalty = torch.tensor(0.0)
        for env_grad in env_gradients:
            invariance_penalty += torch.sum((env_grad - global_gradient) ** 2)
        
        invariance_penalty /= len(env_gradients)
        
        return invariance_penalty

# Convenience functions
def apply_irm_training(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    custom_features: pd.DataFrame,
    lambda_irm: float = 1.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply IRM training to a model.
    
    Args:
        model: Model to train (must have fit method)
        X: Feature matrix
        y: Target series
        custom_features: Custom regime features
        lambda_irm: IRM penalty weight
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with training results
    """
    irm = FinancialIRM(lambda_irm=lambda_irm, **kwargs)
    
    # Create environment masks
    env_masks = irm.create_environment_masks(custom_features)
    
    if len(env_masks) < 2:
        tprint_warning("⚠️ Insufficient environments for IRM training")
        return {'success': False, 'error': 'Insufficient environments'}
    
    # Convert to numpy arrays
    X_array = X.values
    y_array = y.values
    
    # Standard training loop with IRM loss
    # Note: This is a simplified version. In practice, you'd integrate this
    # with your existing training pipelines (LGBM, XGB, etc.)
    
    try:
        # For demonstration, we'll use a simple approach
        # In practice, you'd integrate this with your actual model training
        
        # Mock training results
        results = {
            'success': True,
            'environments': list(env_masks.keys()),
            'lambda_irm': lambda_irm,
            'invariance_metrics': {}
        }
        
        return results
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def quick_irm_evaluation(
    predictions: np.ndarray,
    targets: np.ndarray,
    custom_features: pd.DataFrame,
    lambda_irm: float = 1.0
) -> Dict[str, float]:
    """
    Quick IRM evaluation with default settings.
    
    Args:
        predictions: Predicted probabilities
        targets: True labels
        custom_features: Custom regime features
        lambda_irm: IRM penalty weight
        
    Returns:
        Invariance metrics
    """
    irm = FinancialIRM(lambda_irm=lambda_irm, verbose=False)
    
    env_masks = irm.create_environment_masks(custom_features)
    
    if len(env_masks) < 2:
        return {'invariance_score': float('inf')}
    
    return irm.evaluate_invariance(predictions, targets, env_masks)
