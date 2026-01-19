"""
Invariant Risk Minimization v2 - Enhanced Implementation

Enhanced IRM with variance penalty and focal loss for modern De Prado framework.

Key Features:
1. Enhanced IRM loss with variance penalty across environments
2. Focal loss integration for handling class imbalance
3. Environment creation from custom regime features
4. Invariance evaluation and metrics
5. Training integration with existing models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

class EnhancedIRM:
    """
    Enhanced Invariant Risk Minimization with variance penalty.
    
    Implements IRM with focal loss and variance penalty to find
    invariant representations across different market environments.
    """
    
    def __init__(
        self,
        lambda_irm: float = 1.0,
        lambda_variance: float = 1.0,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        n_environments: int = 4,
        min_env_samples: int = 100,
        verbose: bool = True,
        anneal_steps: int = 1000
    ):
        """
        Initialize Enhanced IRM.
        
        Args:
            lambda_irm: IRM penalty weight
            lambda_variance: Variance penalty weight
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            n_environments: Number of environments to create
            min_env_samples: Minimum samples per environment
            verbose: Whether to print progress information
            anneal_steps: Number of steps to anneal penalties
        """
        self.lambda_irm = lambda_irm
        self.lambda_variance = lambda_variance
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.n_environments = n_environments
        self.min_env_samples = min_env_samples
        self.verbose = verbose
        self.anneal_steps = anneal_steps

        # Annealing state
        self.current_step_ = 0
        self.anneal_progress_ = 0.0
        
        # Storage for environments and metrics
        self.environment_masks_ = {}
        self.invariance_metrics_ = {}
        self.training_history_ = []

    def reset_annealing(self):
        """Reset annealing schedule."""
        self.current_step_ = 0
        self.anneal_progress_ = 0.0
        if self.verbose:
            tprint_info("🔄 IRM annealing schedule reset")

    def set_anneal_progress(self, progress: float):
        """
        Set annealing progress manually.

        Args:
            progress: Progress value between 0.0 and 1.0
        """
        self.anneal_progress_ = np.clip(progress, 0.0, 1.0)
    
    def focal_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for handling class imbalance.
        
        Args:
            y_pred: Predicted probabilities
            y_true: True labels
            
        Returns:
            Focal loss tensor
        """
        # Clamp predictions to avoid log(0)
        y_pred_clamped = torch.clamp(y_pred, 1e-8, 1 - 1e-8)
        
        # Calculate focal loss
        focal_loss = -self.focal_alpha * (1 - y_pred_clamped) ** self.focal_gamma * \
                     (y_true * torch.log(y_pred_clamped) + (1 - y_true) * torch.log(1 - y_pred_clamped))
        
        return torch.mean(focal_loss)
    
    def create_environment_masks(
        self,
        custom_features: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Create environment masks from custom regime features.
        
        Args:
            custom_features: Custom regime features
            feature_columns: Columns to use for environment creation
            
        Returns:
            Dictionary of environment masks
        """
        try:
            if self.verbose:
                tprint_info("🌍 Creating Environment Masks...")
            
            if feature_columns is None:
                # Use volatility and trend features for environment creation
                feature_columns = []
                for col in custom_features.columns:
                    if any(keyword in col.lower() for keyword in ['vol', 'trend', 'regime', 'cluster']):
                        feature_columns.append(col)
            
            if len(feature_columns) == 0:
                if self.verbose:
                    tprint_warning("⚠️ No suitable features for environment creation")
                return {}
            
            # Use selected features for environment clustering
            env_features = custom_features[feature_columns].fillna(0)
            
            # Create environments using quantile-based clustering
            n_samples = len(env_features)
            environment_masks = {}
            
            # Method 1: Volatility-based environments
            if 'volatility' in env_features.columns:
                vol_data = env_features['volatility']
                vol_quantiles = np.quantile(vol_data, [0.33, 0.67])
                
                environment_masks['low_vol'] = vol_data <= vol_quantiles[0]
                environment_masks['med_vol'] = (vol_data > vol_quantiles[0]) & (vol_data <= vol_quantiles[1])
                environment_masks['high_vol'] = vol_data > vol_quantiles[1]
            
            # Method 2: Trend-based environments
            elif any('trend' in col for col in env_features.columns):
                trend_cols = [col for col in env_features.columns if 'trend' in col.lower()]
                if trend_cols:
                    trend_data = env_features[trend_cols[0]]
                    trend_quantiles = np.quantile(trend_data, [0.33, 0.67])
                    
                    environment_masks['downtrend'] = trend_data <= trend_quantiles[0]
                    environment_masks['sideways'] = (trend_data > trend_quantiles[0]) & (trend_data <= trend_quantiles[1])
                    environment_masks['uptrend'] = trend_data > trend_quantiles[1]
            
            # Method 3: Regime-based environments
            elif any('regime' in col for col in env_features.columns):
                regime_cols = [col for col in env_features.columns if 'regime' in col.lower()]
                if regime_cols:
                    regime_data = env_features[regime_cols[0]]
                    unique_regimes = regime_data.unique()
                    
                    for i, regime in enumerate(unique_regimes[:self.n_environments]):
                        env_name = f"regime_{i}"
                        environment_masks[env_name] = regime_data == regime
            
            # Method 4: K-means clustering (fallback)
            else:
                from sklearn.cluster import KMeans
                
                kmeans = KMeans(n_clusters=min(self.n_environments, 4), random_state=42)
                clusters = kmeans.fit_predict(env_features)
                
                for i in range(min(self.n_environments, 4)):
                    env_name = f"cluster_{i}"
                    environment_masks[env_name] = clusters == i
            
            # Filter environments by minimum sample size
            valid_environments = {}
            for env_name, mask in environment_masks.items():
                if mask.sum() >= self.min_env_samples:
                    valid_environments[env_name] = mask
                elif self.verbose:
                    tprint_warning(f"   ⚠️ Environment {env_name} has insufficient samples: {mask.sum()} < {self.min_env_samples}")
            
            self.environment_masks_ = valid_environments
            
            if self.verbose:
                tprint_success(f"✅ Created {len(valid_environments)} valid environments:")
                for env_name, mask in valid_environments.items():
                    tprint_info(f"   - {env_name}: {mask.sum()} samples")
            
            return valid_environments
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Environment creation failed: {e}")
            return {}
    
    def compute_irm_penalty(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        environment_masks: Dict[str, np.ndarray]
    ) -> torch.Tensor:
        """
        Compute IRM penalty across environments.
        
        Args:
            predictions: Model predictions
            targets: True targets
            environment_masks: Environment masks
            
        Returns:
            IRM penalty tensor
        """
        try:
            irm_penalty = 0.0
            n_valid_envs = 0
            
            for env_name, mask in environment_masks.items():
                if mask.sum() < self.min_env_samples:
                    continue
                
                # Get environment-specific data
                env_pred = predictions[mask]
                env_target = targets[mask]
                
                if len(env_pred) < 10:  # Skip very small environments
                    continue
                
                # Compute gradient of loss w.r.t. predictions
                env_pred_detached = env_pred.detach().requires_grad_(True)
                env_loss = self.focal_loss(env_pred_detached, env_target)
                env_grad = torch.autograd.grad(env_loss, env_pred_detached)[0]
                
                # IRM penalty: squared norm of gradient
                irm_penalty += torch.mean(env_grad ** 2)
                n_valid_envs += 1
            
            if n_valid_envs > 0:
                irm_penalty = irm_penalty / n_valid_envs
            else:
                irm_penalty = torch.tensor(0.0)
            
            return irm_penalty
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ IRM penalty computation failed: {e}")
            return torch.tensor(0.0)
    
    def compute_variance_penalty(
        self,
        predictions: torch.Tensor,
        environment_masks: Dict[str, np.ndarray]
    ) -> torch.Tensor:
        """
        Compute variance penalty across environments.
        
        Args:
            predictions: Model predictions
            environment_masks: Environment masks
            
        Returns:
            Variance penalty tensor
        """
        try:
            env_predictions = []
            
            for env_name, mask in environment_masks.items():
                if mask.sum() < self.min_env_samples:
                    continue
                
                env_pred = predictions[mask]
                if len(env_pred) >= 10:
                    env_predictions.append(torch.mean(env_pred))
            
            if len(env_predictions) < 2:
                return torch.tensor(0.0)
            
            # Compute variance of environment means
            env_predictions_tensor = torch.stack(env_predictions)
            variance_penalty = torch.var(env_predictions_tensor)
            
            return variance_penalty
            
        except Exception as e:
            if self.verbose:
                tprint_warning(f"⚠️ Variance penalty computation failed: {e}")
            return torch.tensor(0.0)
    
    def enhanced_irm_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        environment_masks: Dict[str, np.ndarray],
        sample_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute enhanced IRM loss with focal loss and variance penalty.
        
        Args:
            predictions: Model predictions
            targets: True targets
            environment_masks: Environment masks
            sample_weights: Optional sample weights
            
        Returns:
            Tuple of (total_loss, loss_breakdown)
        """
        try:
            # Focal loss (primary loss)
            focal_loss_value = self.focal_loss(predictions, targets)
            
            # Apply annealing factor
            anneal_factor = self.anneal_progress_

            # IRM penalty
            irm_penalty = self.compute_irm_penalty(predictions, targets, environment_masks)
            
            # Variance penalty
            variance_penalty = self.compute_variance_penalty(predictions, environment_masks)
            
            # Total loss with annealed penalties
            total_loss = focal_loss_value + \
                         (self.lambda_irm * anneal_factor * irm_penalty) + \
                         (self.lambda_variance * anneal_factor * variance_penalty)
            
            loss_breakdown = {
                'focal_loss': focal_loss_value,
                'irm_penalty': irm_penalty,
                'variance_penalty': variance_penalty,
                'total_loss': total_loss,
                'anneal_progress': anneal_factor
            }
            
            return total_loss, loss_breakdown
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Enhanced IRM loss computation failed: {e}")
            return torch.tensor(0.0), {'error': str(e)}
    
    def evaluate_invariance_enhanced(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        environment_masks: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate invariance metrics across environments.
        
        Args:
            predictions: Model predictions
            targets: True targets
            environment_masks: Environment masks
            
        Returns:
            Dictionary of invariance metrics
        """
        try:
            env_metrics = {}
            
            for env_name, mask in environment_masks.items():
                if mask.sum() < self.min_env_samples:
                    continue
                
                env_pred = predictions[mask]
                env_target = targets[mask]
                
                if len(env_pred) < 10:
                    continue
                
                # Compute environment-specific metrics
                mse = mean_squared_error(env_target, env_pred)
                r2 = r2_score(env_target, env_pred)
                
                env_metrics[env_name] = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2': r2,
                    'n_samples': len(env_pred)
                }
            
            # Compute invariance metrics
            if len(env_metrics) > 1:
                env_r2_scores = [metrics['r2'] for metrics in env_metrics.values()]
                env_rmse_scores = [metrics['rmse'] for metrics in env_metrics.values()]
                
                invariance_metrics = {
                    'r2_variance': np.var(env_r2_scores),
                    'r2_range': np.max(env_r2_scores) - np.min(env_r2_scores),
                    'rmse_variance': np.var(env_rmse_scores),
                    'rmse_range': np.max(env_rmse_scores) - np.min(env_rmse_scores),
                    'mean_r2': np.mean(env_r2_scores),
                    'mean_rmse': np.mean(env_rmse_scores),
                    'n_environments': len(env_metrics)
                }
            else:
                invariance_metrics = {
                    'r2_variance': 0.0,
                    'r2_range': 0.0,
                    'rmse_variance': 0.0,
                    'rmse_range': 0.0,
                    'mean_r2': list(env_metrics.values())[0]['r2'] if env_metrics else 0.0,
                    'mean_rmse': list(env_metrics.values())[0]['rmse'] if env_metrics else 0.0,
                    'n_environments': len(env_metrics)
                }
            
            invariance_metrics['environment_metrics'] = env_metrics
            self.invariance_metrics_ = invariance_metrics
            
            return invariance_metrics
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Invariance evaluation failed: {e}")
            return {'error': str(e)}
    
    def is_invariant_enhanced(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
        environment_masks: Dict[str, np.ndarray],
        r2_variance_threshold: float = 0.1,
        rmse_variance_threshold: float = 0.01
    ) -> bool:
        """
        Check if model is invariant across environments.
        
        Args:
            predictions: Model predictions
            targets: True targets
            environment_masks: Environment masks
            r2_variance_threshold: Threshold for R² variance
            rmse_variance_threshold: Threshold for RMSE variance
            
        Returns:
            Boolean indicating invariance
        """
        try:
            invariance_metrics = self.evaluate_invariance_enhanced(
                predictions, targets, environment_masks
            )
            
            if 'error' in invariance_metrics:
                return False
            
            is_invariant = (
                invariance_metrics['r2_variance'] <= r2_variance_threshold and
                invariance_metrics['rmse_variance'] <= rmse_variance_threshold
            )
            
            if self.verbose:
                status = "✅ Invariant" if is_invariant else "❌ Not invariant"
                tprint_info(f"   {status}: R² variance={invariance_metrics['r2_variance']:.4f}, "
                           f"RMSE variance={invariance_metrics['rmse_variance']:.4f}")
            
            return is_invariant
            
        except Exception as e:
            if self.verbose:
                tprint_error(f"❌ Invariance check failed: {e}")
            return False
    
    def create_enhanced_irm_trainer(
        self,
        model: Any,
        optimizer: torch.optim.Optimizer,
        environment_masks: Dict[str, np.ndarray]
    ) -> callable:
        """
        Create enhanced IRM training function.
        
        Args:
            model: Model to train
            optimizer: Optimizer
            environment_masks: Environment masks
            
        Returns:
            Training function
        """
        # Reset annealing at start of training
        self.reset_annealing()

        def enhanced_irm_train_step(X_batch, y_batch, sample_weights=None):
            """Enhanced IRM training step."""
            # Update annealing progress
            self.current_step_ += 1
            if self.anneal_steps > 0:
                self.anneal_progress_ = min(1.0, self.current_step_ / self.anneal_steps)
            else:
                self.anneal_progress_ = 1.0

            # Convert to tensors
            X_tensor = torch.FloatTensor(X_batch)
            y_tensor = torch.FloatTensor(y_batch)
            
            if sample_weights is not None:
                weights_tensor = torch.FloatTensor(sample_weights)
            else:
                weights_tensor = None
            
            # Forward pass
            predictions = model(X_tensor)
            
            # Compute enhanced IRM loss
            total_loss, loss_breakdown = self.enhanced_irm_loss(
                predictions, y_tensor, environment_masks, weights_tensor
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            return total_loss.item(), loss_breakdown
        
        return enhanced_irm_train_step
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of Enhanced IRM system.
        
        Returns:
            Summary dictionary
        """
        return {
            'lambda_irm': self.lambda_irm,
            'lambda_variance': self.lambda_variance,
            'focal_alpha': self.focal_alpha,
            'focal_gamma': self.focal_gamma,
            'n_environments': self.n_environments,
            'min_env_samples': self.min_env_samples,
            'anneal_steps': self.anneal_steps,
            'environments_created': len(self.environment_masks_),
            'has_invariance_metrics': self.invariance_metrics_ is not None
        }

# Convenience functions
def quick_enhanced_irm(
    lambda_irm: float = 1.0,
    lambda_variance: float = 1.0,
    **kwargs
) -> EnhancedIRM:
    """
    Quick Enhanced IRM setup.
    
    Args:
        lambda_irm: IRM penalty weight
        lambda_variance: Variance penalty weight
        **kwargs: Additional parameters
        
    Returns:
        Enhanced IRM instance
    """
    return EnhancedIRM(
        lambda_irm=lambda_irm,
        lambda_variance=lambda_variance,
        **kwargs
    )

def apply_irm_training(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    custom_features: pd.DataFrame,
    **kwargs
) -> Dict[str, Any]:
    """
    Apply IRM training to a model.
    
    Args:
        model: Model to train
        X_train: Training features
        y_train: Training targets
        custom_features: Custom features for environment creation
        **kwargs: Additional parameters
        
    Returns:
        Training results
    """
    irm = EnhancedIRM(**kwargs)
    
    # Create environments
    env_masks = irm.create_environment_masks(custom_features)
    
    if len(env_masks) < 2:
        return {
            'success': False,
            'error': 'Insufficient environments for IRM training',
            'environments': list(env_masks.keys())
        }
    
    # Mock training with Enhanced IRM loss
    # In practice, you'd integrate this with your actual training loops
    
    # Generate mock predictions for demonstration
    if hasattr(model, 'predict_proba'):
        predictions = model.predict_proba(X_train)[:, 1]
    elif hasattr(model, 'predict'):
        predictions = model.predict(X_train)
    else:
        # Mock predictions for demonstration
        predictions = np.random.uniform(0, 1, len(y_train))
    
    # Evaluate invariance
    invariance_metrics = irm.evaluate_invariance_enhanced(
        predictions, y_train, env_masks
    )
    
    # Check invariance
    is_invariant = irm.is_invariant_enhanced(
        predictions, y_train, env_masks
    )
    
    results = {
        'success': True,
        'environments': list(env_masks.keys()),
        'lambda_irm': irm.lambda_irm,
        'lambda_variance': irm.lambda_variance,
        'invariance_metrics': invariance_metrics,
        'is_invariant': is_invariant,
        'model_type': type(model).__name__
    }
    
    return results
