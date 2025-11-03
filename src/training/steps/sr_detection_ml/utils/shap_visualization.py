"""
SHAP Visualization Utilities

Generate comprehensive SHAP analysis visualizations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


class ShapVisualizer:
    """
    Generate SHAP interpretability visualizations.
    
    Provides:
    - Summary plots (global feature importance)
    - Dependence plots (feature interactions)
    - Force plots (individual predictions)
    - Waterfall plots (detailed breakdowns)
    """
    
    def __init__(self, output_dir: str = "outputs/sr_ml/shap"):
        """
        Initialize SHAP visualizer.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 100
    
    def generate_all_plots(
        self,
        explainer: shap.TreeExplainer,
        X: pd.DataFrame,
        shap_values: np.ndarray,
        feature_names: List[str],
        prefix: str = "shap"
    ):
        """
        Generate all SHAP visualizations.
        
        Args:
            explainer: SHAP explainer object
            X: Feature matrix
            shap_values: SHAP values
            feature_names: List of feature names
            prefix: Filename prefix
        """
        self.logger.info("🎨 Generating SHAP visualizations...")
        
        # 1. Summary plot (global importance)
        self.logger.info("   Creating summary plot...")
        self.summary_plot(shap_values, X, feature_names, prefix)
        
        # 2. Bar plot (mean absolute SHAP)
        self.logger.info("   Creating bar plot...")
        self.bar_plot(shap_values, feature_names, prefix)
        
        # 3. Dependence plots for top features
        self.logger.info("   Creating dependence plots...")
        self.dependence_plots(shap_values, X, feature_names, prefix, top_n=10)
        
        # 4. Force plots (sample of predictions)
        self.logger.info("   Creating force plots...")
        self.force_plots(explainer, X, prefix, n_samples=5)
        
        self.logger.info(f"✅ SHAP visualizations saved to {self.output_dir}")
    
    def summary_plot(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        feature_names: List[str],
        prefix: str
    ):
        """Generate SHAP summary plot."""
        try:
            plt.figure(figsize=(12, 10))
            shap.summary_plot(
                shap_values,
                X,
                feature_names=feature_names,
                show=False,
                max_display=20
            )
            
            filepath = self.output_dir / f"{prefix}_summary.png"
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.debug(f"   Saved summary plot: {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create summary plot: {e}")
    
    def bar_plot(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        prefix: str
    ):
        """Generate SHAP bar plot (mean absolute importance)."""
        try:
            # Calculate mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            # Sort
            sorted_idx = np.argsort(mean_shap)[-20:]  # Top 20
            
            # Plot
            _, ax = plt.subplots(figsize=(10, 8))
            
            y_pos = np.arange(len(sorted_idx))
            ax.barh(y_pos, mean_shap[sorted_idx])
            ax.set_yticks(y_pos)
            ax.set_yticklabels([feature_names[i] for i in sorted_idx])
            ax.set_xlabel('Mean |SHAP value|')
            ax.set_title('Feature Importance (Top 20)')
            
            filepath = self.output_dir / f"{prefix}_bar.png"
            plt.tight_layout()
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.debug(f"   Saved bar plot: {filepath}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create bar plot: {e}")
    
    def dependence_plots(
        self,
        shap_values: np.ndarray,
        X: pd.DataFrame,
        feature_names: List[str],
        prefix: str,
        top_n: int = 10
    ):
        """Generate SHAP dependence plots for top features."""
        try:
            # Get top features by importance
            mean_shap = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_shap)[-top_n:]
            
            for idx in top_indices:
                feature = feature_names[idx]
                
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(
                    idx,
                    shap_values,
                    X,
                    feature_names=feature_names,
                    show=False
                )
                
                filepath = self.output_dir / f"{prefix}_dependence_{feature.replace('/', '_')}.png"
                plt.tight_layout()
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
            
            self.logger.debug(f"   Saved {top_n} dependence plots")
            
        except Exception as e:
            self.logger.warning(f"Failed to create dependence plots: {e}")
    
    def force_plots(
        self,
        explainer: shap.TreeExplainer,
        X: pd.DataFrame,
        prefix: str,
        n_samples: int = 5
    ):
        """Generate SHAP force plots for sample predictions."""
        try:
            # Sample random indices
            sample_indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
            
            for i, idx in enumerate(sample_indices):
                shap_values_single = explainer.shap_values(X.iloc[idx:idx+1])
                
                # Force plot
                shap.force_plot(
                    explainer.expected_value,
                    shap_values_single[0],
                    X.iloc[idx],
                    matplotlib=True,
                    show=False
                )
                
                filepath = self.output_dir / f"{prefix}_force_{i}.png"
                plt.tight_layout()
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
            
            self.logger.debug(f"   Saved {n_samples} force plots")
            
        except Exception as e:
            self.logger.warning(f"Failed to create force plots: {e}")
    
    def feature_importance_table(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
        top_n: int = 50
    ) -> pd.DataFrame:
        """
        Generate feature importance table.
        
        Args:
            shap_values: SHAP values
            feature_names: List of feature names
            top_n: Number of top features
        
        Returns:
            DataFrame with feature importance ranking
        """
        mean_shap = np.abs(shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_shap
        })
        
        df = df.sort_values('mean_abs_shap', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df.head(top_n)

