"""
LGBM-SHAP with Recursive Feature Elimination (RFE)

This module implements a sophisticated feature selection method that combines
LGBM importance scores with SHAP values and uses RFE to iteratively remove
the least important features until reaching the target count.

Key Features:
- Removes 25% of features per iteration when above target
- Uses LGBM importance + SHAP values for feature ranking
- Comprehensive logging with tprint
- Detailed reporting with global and per-feature metrics
- Handles NaN values and data validation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import os
import json
import warnings
from dataclasses import dataclass, field

# Import LGBM and SHAP
try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    lgb = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

# Import project utilities
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_performance, tprint_debug
from src.utils.ml_common.utils.feature_selection import FeatureImportanceRanker

logger = logging.getLogger(__name__)

@dataclass
class LGBMSHAPRFEConfig:
    """Configuration for LGBM-SHAP RFE selector."""
    
    # Target settings
    target_features: int = 60
    removal_percentage: float = 0.25  # Remove 25% of features per iteration
    
    # LGBM settings
    lgb_params: Dict[str, Any] = field(default_factory=lambda: {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    })
    
    # SHAP settings
    shap_explainer: str = 'tree'  # 'tree', 'linear', 'kernel'
    shap_sample_size: Optional[int] = None  # None for all samples
    
    # RFE settings
    min_features_to_keep: int = 10
    max_iterations: int = 20
    early_stopping_patience: int = 3
    
    # Validation settings
    cv_folds: int = 5
    validation_size: float = 0.2
    
    # Reporting settings
    enable_detailed_logging: bool = True
    save_intermediate_results: bool = True
    report_format: str = 'both'  # 'json', 'markdown', 'both'

class LGBMSHAPRFESelector:
    """
    LGBM-SHAP with Recursive Feature Elimination selector.
    
    This selector combines LGBM importance scores with SHAP values to rank features
    and uses RFE to iteratively remove the least important features.
    """
    
    def __init__(self, config: Optional[LGBMSHAPRFEConfig] = None):
        """Initialize the LGBM-SHAP RFE selector."""
        self.config = config or LGBMSHAPRFEConfig()
        self.logger = logger.getChild('LGBMSHAPRFESelector')
        
        # Validate dependencies
        if not LGBM_AVAILABLE:
            raise ImportError("LightGBM is required but not available")
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is required but not available")
        
        # Initialize components
        self.feature_ranker = FeatureImportanceRanker()
        
        # Tracking variables
        self.selection_history = []
        self.feature_importance_history = []
        self.performance_history = []
        self.removed_features_history = []
        
        tprint_success("🔧 LGBM-SHAP RFE Selector initialized")
    
    def select_features(self, 
                       X: Union[np.ndarray, pd.DataFrame], 
                       y: Union[np.ndarray, pd.Series],
                       feature_names: Optional[List[str]] = None,
                       target_features: Optional[int] = None) -> Dict[str, Any]:
        """
        Select features using LGBM-SHAP with RFE.
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: List of feature names
            target_features: Target number of features (overrides config)
            
        Returns:
            Dictionary containing selection results and metrics
        """
        tprint("🚀 Starting LGBM-SHAP RFE feature selection")
        
        # Set target features
        if target_features is not None:
            self.config.target_features = target_features
        
        # Prepare data
        X_processed, y_processed, feature_names_processed = self._prepare_data(X, y, feature_names)
        
        # Initialize tracking
        current_features = list(range(X_processed.shape[1]))
        current_feature_names = feature_names_processed.copy()
        iteration = 0
        best_performance = -np.inf
        patience_counter = 0
        
        tprint(f"📊 Starting with {len(current_features)} features, target: {self.config.target_features}")
        
        # Main RFE loop - continue until we reach exactly the target features
        while len(current_features) > self.config.target_features and iteration < self.config.max_iterations:
            iteration += 1
            tprint(f"\n🔄 RFE Iteration {iteration}")
            tprint(f"📈 Current features: {len(current_features)}")
            
            # Check if we can remove features
            if len(current_features) <= self.config.min_features_to_keep:
                tprint_warning(f"⚠️ Reached minimum features limit: {self.config.min_features_to_keep}")
                break
            
            # Calculate number of features to remove
            features_to_remove = max(1, int(len(current_features) * self.config.removal_percentage))
            features_to_remove = min(features_to_remove, len(current_features) - self.config.target_features)
            
            tprint(f"🎯 Will remove {features_to_remove} features (25% of {len(current_features)})")
            
            # Get current feature subset
            X_current = X_processed[:, current_features]
            
            # Train LGBM model
            model, performance = self._train_lgbm_model(X_current, y_processed)
            
            # Calculate feature importance and SHAP values
            importance_scores, shap_values = self._calculate_importance_and_shap(
                model, X_current, y_processed, current_feature_names
            )
            
            # Combine importance and SHAP scores
            combined_scores = self._combine_scores(importance_scores, shap_values)
            
            # Select features to remove (lowest scores)
            features_to_remove_indices = self._select_features_to_remove(
                combined_scores, features_to_remove
            )
            
            # Get names of features to be removed
            removed_feature_names = [current_feature_names[i] for i in features_to_remove_indices]
            
            # Log removed features
            tprint(f"🗑️ Removing {len(removed_feature_names)} features:")
            for i, feature_name in enumerate(removed_feature_names):
                score = combined_scores[features_to_remove_indices[i]]
                tprint(f"   {i+1:2d}. {feature_name} (score: {score:.6f})")
            
            # Update feature lists
            remaining_indices = [i for i in range(len(current_features)) if i not in features_to_remove_indices]
            current_features = [current_features[i] for i in remaining_indices]
            current_feature_names = [current_feature_names[i] for i in remaining_indices]
            
            # Store iteration results
            iteration_result = {
                'iteration': iteration,
                'features_removed': removed_feature_names,
                'features_remaining': len(current_features),
                'performance': performance,
                'importance_scores': importance_scores.tolist(),
                'shap_values': shap_values.tolist() if shap_values is not None else None,
                'combined_scores': combined_scores.tolist()
            }
            self.selection_history.append(iteration_result)
            self.removed_features_history.extend(removed_feature_names)
            
            tprint(f"✅ Iteration {iteration} complete. Features remaining: {len(current_features)}")
            
            # Check for early stopping
            if performance > best_performance:
                best_performance = performance
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.config.early_stopping_patience:
                tprint_warning(f"⚠️ Early stopping triggered after {patience_counter} iterations without improvement")
                break
        
        # Final results
        final_features = current_features
        final_feature_names = current_feature_names
        
        # Ensure we have exactly the target number of features
        if len(final_features) > self.config.target_features:
            # If we still have too many features, remove the least important ones
            tprint(f"⚠️ Still have {len(final_features)} features, need exactly {self.config.target_features}")
            
            # Get final feature subset for final selection
            X_final = X_processed[:, final_features]
            model_final, _ = self._train_lgbm_model(X_final, y_processed)
            importance_scores, shap_values = self._calculate_importance_and_shap(
                model_final, X_final, y_processed, final_feature_names
            )
            combined_scores = self._combine_scores(importance_scores, shap_values)
            
            # Select exactly target_features
            top_indices = np.argsort(combined_scores)[-self.config.target_features:]
            final_features = [final_features[i] for i in top_indices]
            final_feature_names = [final_feature_names[i] for i in top_indices]
            
            tprint(f"✅ Final selection: {len(final_features)} features")
        
        tprint(f"\n🎉 RFE Complete!")
        tprint(f"📊 Final features: {len(final_features)} (target: {self.config.target_features})")
        tprint(f"📈 Total iterations: {iteration}")
        tprint(f"🗑️ Total features removed: {len(self.removed_features_history)}")
        
        # Generate detailed report
        report = self._generate_detailed_report(
            X_processed, y_processed, final_features, final_feature_names
        )
        
        # Save report
        if self.config.save_intermediate_results:
            self._save_report(report)
        
        return {
            'selected_features': final_features,
            'selected_feature_names': final_feature_names,
            'selection_history': self.selection_history,
            'removed_features': self.removed_features_history,
            'report': report,
            'success': True
        }
    
    def _prepare_data(self, X: Union[np.ndarray, pd.DataFrame], 
                     y: Union[np.ndarray, pd.Series],
                     feature_names: Optional[List[str]]) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare and validate input data."""
        tprint_debug("🔧 Preparing input data")
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X_array = X.values
            if feature_names is None:
                feature_names = X.columns.tolist()
        else:
            X_array = X
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_array.shape[1])]
        
        if isinstance(y, pd.Series):
            y_array = y.values
        else:
            y_array = y
        
        # Handle NaN values
        nan_mask = np.isnan(X_array).any(axis=1) | np.isnan(y_array)
        if np.any(nan_mask):
            tprint_warning(f"⚠️ Removing {np.sum(nan_mask)} rows with NaN values")
            X_array = X_array[~nan_mask]
            y_array = y_array[~nan_mask]
        
        # Validate data
        if X_array.shape[0] == 0:
            raise ValueError("No valid data remaining after NaN removal")
        
        tprint(f"📊 Data prepared: {X_array.shape[0]} samples, {X_array.shape[1]} features")
        
        return X_array, y_array, feature_names
    
    def _train_lgbm_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, float]:
        """Train LGBM model and return performance."""
        tprint_debug("🌲 Training LGBM model")
        
        # Split data for validation
        n_samples = X.shape[0]
        val_size = int(n_samples * self.config.validation_size)
        indices = np.random.permutation(n_samples)
        
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            self.config.lgb_params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=100,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
        )
        
        # Calculate performance
        y_pred = model.predict(X_val)
        performance = -np.mean((y_val - y_pred) ** 2)  # Negative MSE for maximization
        
        return model, performance
    
    def _calculate_importance_and_shap(self, model: Any, X: np.ndarray, y: np.ndarray,
                                     feature_names: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Calculate LGBM importance and SHAP values."""
        tprint_debug("📊 Calculating importance and SHAP values")
        
        # Get LGBM importance
        importance_scores = model.feature_importance(importance_type='gain')
        
        # Calculate SHAP values
        shap_values = None
        try:
            if self.config.shap_explainer == 'tree':
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
            elif self.config.shap_explainer == 'linear':
                explainer = shap.LinearExplainer(model, X)
                shap_values = explainer.shap_values(X)
            else:
                # Use kernel explainer as fallback
                explainer = shap.KernelExplainer(model.predict, X[:100])  # Sample for efficiency
                shap_values = explainer.shap_values(X[:100])
        except Exception as e:
            tprint_warning(f"⚠️ SHAP calculation failed: {e}")
            shap_values = None
        
        return importance_scores, shap_values
    
    def _combine_scores(self, importance_scores: np.ndarray, 
                       shap_values: Optional[np.ndarray]) -> np.ndarray:
        """Combine LGBM importance and SHAP values."""
        tprint_debug("🔗 Combining importance and SHAP scores")
        
        # Normalize importance scores
        importance_normalized = importance_scores / (np.sum(importance_scores) + 1e-10)
        
        if shap_values is not None:
            # Calculate mean absolute SHAP values
            shap_mean_abs = np.mean(np.abs(shap_values), axis=0)
            shap_normalized = shap_mean_abs / (np.sum(shap_mean_abs) + 1e-10)
            
            # Combine with equal weights
            combined_scores = 0.5 * importance_normalized + 0.5 * shap_normalized
        else:
            # Use only importance scores
            combined_scores = importance_normalized
        
        return combined_scores
    
    def _select_features_to_remove(self, scores: np.ndarray, 
                                  n_to_remove: int) -> List[int]:
        """Select features to remove based on scores."""
        tprint_debug(f"🎯 Selecting {n_to_remove} features to remove")
        
        # Get indices of features with lowest scores
        sorted_indices = np.argsort(scores)
        features_to_remove = sorted_indices[:n_to_remove].tolist()
        
        return features_to_remove
    
    def _generate_detailed_report(self, X: np.ndarray, y: np.ndarray,
                                selected_features: List[int], 
                                selected_feature_names: List[str]) -> Dict[str, Any]:
        """Generate detailed report with global and per-feature metrics."""
        tprint("📋 Generating detailed report")
        
        # Global metrics
        global_metrics = self._calculate_global_metrics(X, y, selected_features)
        
        # Per-feature metrics
        per_feature_metrics = self._calculate_per_feature_metrics(
            X, y, selected_features, selected_feature_names
        )
        
        # Selection summary
        selection_summary = {
            'total_iterations': len(self.selection_history),
            'total_features_removed': len(self.removed_features_history),
            'final_feature_count': len(selected_features),
            'target_feature_count': self.config.target_features,
            'removal_percentage': self.config.removal_percentage
        }
        
        # Performance over iterations
        performance_over_time = {
            'iterations': [h['iteration'] for h in self.selection_history],
            'features_remaining': [h['features_remaining'] for h in self.selection_history],
            'performance': [h['performance'] for h in self.selection_history]
        }
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'target_features': self.config.target_features,
                'removal_percentage': self.config.removal_percentage,
                'lgb_params': self.config.lgb_params,
                'shap_explainer': self.config.shap_explainer
            },
            'selection_summary': selection_summary,
            'global_metrics': global_metrics,
            'per_feature_metrics': per_feature_metrics,
            'performance_over_time': performance_over_time,
            'removed_features': self.removed_features_history,
            'selected_features': selected_feature_names
        }
        
        return report
    
    def _calculate_global_metrics(self, X: np.ndarray, y: np.ndarray,
                                selected_features: List[int]) -> Dict[str, Any]:
        """Calculate global performance metrics."""
        tprint_debug("📊 Calculating global metrics")
        
        # Get selected feature subset
        X_selected = X[:, selected_features]
        
        # Calculate basic statistics
        n_samples, n_features = X_selected.shape
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X_selected.T)
        avg_correlation = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
        
        # Calculate variance
        feature_variances = np.var(X_selected, axis=0)
        avg_variance = np.mean(feature_variances)
        
        # Calculate mutual information (simplified)
        target_variance = np.var(y)
        
        return {
            'n_samples': int(n_samples),
            'n_features': int(n_features),
            'avg_correlation': float(avg_correlation),
            'avg_variance': float(avg_variance),
            'target_variance': float(target_variance),
            'feature_variance_std': float(np.std(feature_variances))
        }
    
    def _calculate_per_feature_metrics(self, X: np.ndarray, y: np.ndarray,
                                     selected_features: List[int],
                                     selected_feature_names: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        """Calculate per-feature metrics."""
        tprint_debug("📊 Calculating per-feature metrics")
        
        X_selected = X[:, selected_features]
        per_feature_metrics = []
        
        for i, feature_name in enumerate(selected_feature_names):
            feature_values = X_selected[:, i]
            
            # Basic statistics
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            
            # Correlation with target
            correlation = np.corrcoef(feature_values, y)[0, 1] if len(feature_values) > 1 else 0
            
            # Variance
            variance = np.var(feature_values)
            
            # Skewness and kurtosis
            from scipy import stats
            skewness = stats.skew(feature_values)
            kurtosis = stats.kurtosis(feature_values)
            
            per_feature_metrics.append({
                'feature_name': feature_name,
                'feature_index': selected_features[i],
                'mean': float(mean_val),
                'std': float(std_val),
                'min': float(min_val),
                'max': float(max_val),
                'variance': float(variance),
                'correlation_with_target': float(correlation),
                'skewness': float(skewness),
                'kurtosis': float(kurtosis)
            })
        
        return per_feature_metrics
    
    def _save_report(self, report: Dict[str, Any]) -> str:
        """Save detailed report to outcomes directory."""
        tprint("💾 Saving detailed report")
        
        # Create outcomes directory if it doesn't exist
        outcomes_dir = "/workspace/outcomes"
        os.makedirs(outcomes_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"lgbm_shap_rfe_report_{timestamp}"
        
        # Save JSON report
        if self.config.report_format in ['json', 'both']:
            json_filename = f"{base_filename}.json"
            json_path = os.path.join(outcomes_dir, json_filename)
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2)
            tprint_success(f"✅ JSON report saved: {json_path}")
        
        # Save Markdown report
        if self.config.report_format in ['markdown', 'both']:
            md_filename = f"{base_filename}.md"
            md_path = os.path.join(outcomes_dir, md_filename)
            self._save_markdown_report(report, md_path)
            tprint_success(f"✅ Markdown report saved: {md_path}")
        
        return base_filename
    
    def _save_markdown_report(self, report: Dict[str, Any], filepath: str) -> None:
        """Save report in Markdown format."""
        with open(filepath, 'w') as f:
            f.write(f"# LGBM-SHAP RFE Feature Selection Report\n\n")
            f.write(f"**Generated:** {report['timestamp']}\n\n")
            
            # Selection Summary
            f.write("## 📊 Selection Summary\n\n")
            summary = report['selection_summary']
            f.write(f"- **Total Iterations:** {summary['total_iterations']}\n")
            f.write(f"- **Features Removed:** {summary['total_features_removed']}\n")
            f.write(f"- **Final Features:** {summary['final_feature_count']}\n")
            f.write(f"- **Target Features:** {summary['target_feature_count']}\n")
            f.write(f"- **Removal Percentage:** {summary['removal_percentage']:.1%}\n\n")
            
            # Global Metrics
            f.write("## 🌍 Global Metrics\n\n")
            global_metrics = report['global_metrics']
            f.write(f"- **Samples:** {global_metrics['n_samples']:,}\n")
            f.write(f"- **Features:** {global_metrics['n_features']}\n")
            f.write(f"- **Avg Correlation:** {global_metrics['avg_correlation']:.4f}\n")
            f.write(f"- **Avg Variance:** {global_metrics['avg_variance']:.4f}\n")
            f.write(f"- **Target Variance:** {global_metrics['target_variance']:.4f}\n\n")
            
            # Selected Features
            f.write("## ✅ Selected Features\n\n")
            for i, feature_name in enumerate(report['selected_features'], 1):
                f.write(f"{i:2d}. {feature_name}\n")
            f.write("\n")
            
            # Removed Features
            f.write("## 🗑️ Removed Features\n\n")
            for i, feature_name in enumerate(report['removed_features'], 1):
                f.write(f"{i:2d}. {feature_name}\n")
            f.write("\n")
            
            # Per-Feature Metrics
            f.write("## 📈 Per-Feature Metrics\n\n")
            f.write("| Feature | Mean | Std | Min | Max | Variance | Correlation | Skewness | Kurtosis |\n")
            f.write("|---------|------|-----|-----|-----|----------|-------------|----------|----------|\n")
            
            for feature_metrics in report['per_feature_metrics']:
                f.write(f"| {feature_metrics['feature_name']} | "
                       f"{feature_metrics['mean']:.4f} | "
                       f"{feature_metrics['std']:.4f} | "
                       f"{feature_metrics['min']:.4f} | "
                       f"{feature_metrics['max']:.4f} | "
                       f"{feature_metrics['variance']:.4f} | "
                       f"{feature_metrics['correlation_with_target']:.4f} | "
                       f"{feature_metrics['skewness']:.4f} | "
                       f"{feature_metrics['kurtosis']:.4f} |\n")
            
            f.write("\n")
            
            # Performance Over Time
            f.write("## 📊 Performance Over Time\n\n")
            f.write("| Iteration | Features Remaining | Performance |\n")
            f.write("|-----------|-------------------|-------------|\n")
            
            perf_data = report['performance_over_time']
            for i in range(len(perf_data['iterations'])):
                f.write(f"| {perf_data['iterations'][i]} | "
                       f"{perf_data['features_remaining'][i]} | "
                       f"{perf_data['performance'][i]:.6f} |\n")
            
            f.write("\n")


def create_lgbm_shap_rfe_selector(config: Optional[LGBMSHAPRFEConfig] = None) -> LGBMSHAPRFESelector:
    """Create and return a LGBM-SHAP RFE selector instance."""
    return LGBMSHAPRFESelector(config)


# Example usage
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    n_samples, n_features = 1000, 200
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create selector
    config = LGBMSHAPRFEConfig(target_features=60, removal_percentage=0.25)
    selector = create_lgbm_shap_rfe_selector(config)
    
    # Run selection
    result = selector.select_features(X, y, feature_names)
    
    print(f"Selected {len(result['selected_features'])} features")
    print(f"Removed {len(result['removed_features'])} features")