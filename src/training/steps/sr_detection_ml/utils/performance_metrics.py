"""
Performance Metrics and Analysis

Comprehensive evaluation metrics for SR ML models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    explained_variance_score
)
from scipy.stats import spearmanr, pearsonr

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyze model performance with comprehensive metrics.
    
    Provides:
    - Regression metrics (R², RMSE, MAE, etc.)
    - Correlation analysis
    - Residual analysis
    - Prediction distribution analysis
    """
    
    def __init__(self, output_dir: str = "outputs/sr_ml/performance"):
        """
        Initialize performance analyzer.
        
        Args:
            output_dir: Directory to save plots and reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        sns.set_style("whitegrid")
    
    def full_evaluation(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str = "evaluation"
    ) -> Dict[str, Any]:
        """
        Perform full evaluation with all metrics and plots.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            prefix: Filename prefix
        
        Returns:
            Dictionary with all metrics
        """
        self.logger.info("📊 Performing comprehensive performance analysis...")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Log metrics
        self.logger.info("\n   Regression Metrics:")
        self.logger.info(f"      R²: {metrics['r2']:.4f}")
        self.logger.info(f"      RMSE: {metrics['rmse']:.6f}")
        self.logger.info(f"      MAE: {metrics['mae']:.6f}")
        self.logger.info(f"      Explained Variance: {metrics['explained_variance']:.4f}")
        
        self.logger.info("\n   Correlation:")
        self.logger.info(f"      Pearson: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e})")
        self.logger.info(f"      Spearman: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4e})")
        
        # Generate plots
        self.logger.info("\n   Generating diagnostic plots...")
        self.prediction_scatter(y_true, y_pred, prefix)
        self.residual_plot(y_true, y_pred, prefix)
        self.distribution_comparison(y_true, y_pred, prefix)
        
        # Save metrics
        self._save_metrics(metrics, prefix)
        
        self.logger.info(f"✅ Performance analysis complete. Results saved to {self.output_dir}")
        
        return metrics
    
    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate all regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
        
        Returns:
            Dictionary with metrics
        """
        metrics = {}
        
        # Basic regression metrics
        metrics['r2'] = r2_score(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
        
        # Correlation metrics
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
        
        metrics['pearson_r'] = pearson_r
        metrics['pearson_p'] = pearson_p
        metrics['spearman_r'] = spearman_r
        metrics['spearman_p'] = spearman_p
        
        # Residual statistics
        residuals = y_true - y_pred
        metrics['mean_residual'] = np.mean(residuals)
        metrics['std_residual'] = np.std(residuals)
        
        return metrics
    
    def prediction_scatter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str
    ):
        """Generate prediction vs actual scatter plot."""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            ax.scatter(y_true, y_pred, alpha=0.5, s=20)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
            
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predicted Values')
            ax.set_title('Predictions vs Actual')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            filepath = self.output_dir / f"{prefix}_scatter.png"
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create scatter plot: {e}")
    
    def residual_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str
    ):
        """Generate residual plot."""
        try:
            residuals = y_true - y_pred
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Residuals vs predicted
            ax1.scatter(y_pred, residuals, alpha=0.5, s=20)
            ax1.axhline(y=0, color='r', linestyle='--', lw=2)
            ax1.set_xlabel('Predicted Values')
            ax1.set_ylabel('Residuals')
            ax1.set_title('Residual Plot')
            ax1.grid(True, alpha=0.3)
            
            # Residual distribution
            ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
            ax2.axvline(x=0, color='r', linestyle='--', lw=2)
            ax2.set_xlabel('Residuals')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Residual Distribution')
            ax2.grid(True, alpha=0.3)
            
            filepath = self.output_dir / f"{prefix}_residuals.png"
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create residual plot: {e}")
    
    def distribution_comparison(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        prefix: str
    ):
        """Compare distributions of true vs predicted values."""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.hist(y_true, bins=50, alpha=0.5, label='True', edgecolor='black')
            ax.hist(y_pred, bins=50, alpha=0.5, label='Predicted', edgecolor='black')
            
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Distribution Comparison: True vs Predicted')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            filepath = self.output_dir / f"{prefix}_distributions.png"
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to create distribution plot: {e}")
    
    def _save_metrics(self, metrics: Dict[str, float], prefix: str):
        """Save metrics to JSON file."""
        try:
            import json
            
            filepath = self.output_dir / f"{prefix}_metrics.json"
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2)
            
            self.logger.debug(f"   Saved metrics: {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save metrics: {e}")

