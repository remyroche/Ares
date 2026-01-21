"""
Layer 3 Model Race Reporting System

Comprehensive reporting for Layer 3 multi-horizon model comparison.
Generates detailed .md and .csv reports for model race results,
including IC, IC_IR, stability metrics, calibration, and ensemble analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from scipy import stats
import logging

# Import tprint functions
try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    # Fallback print functions
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")

logger = logging.getLogger(__name__)

class Layer3ModelRaceReporter:
    """Comprehensive model race reporting for Layer 3 multi-horizon pipeline."""
    
    def __init__(self, outcomes_dir: Optional[Path] = None):
        tprint_info("Starting Layer3ModelRaceReporter.__init__")
        self.outcomes_dir = outcomes_dir or Path('outcomes')
        self.outcomes_dir.mkdir(exist_ok=True, parents=True)
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tprint_info("Finished Layer3ModelRaceReporter.__init__")
    
    def calculate_extended_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  task_type: str) -> Dict[str, float]:
        """
        Calculate extended metrics for model evaluation.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary of extended metrics
        """
        tprint_info(f"Starting calculate_extended_metrics (task_type={task_type})")
        metrics = {}
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        
        if np.sum(valid_mask) < 10:
            tprint_warning("Insufficient valid samples in calculate_extended_metrics")
            tprint_info("Finished calculate_extended_metrics (early exit)")
            return {'error': 'Insufficient valid samples'}
        
        y_true_clean = y_true[valid_mask]
        y_pred_clean = y_pred[valid_mask]
        
        if task_type == 'regression':
            # Information Coefficient (IC)
            try:
                ic = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
                metrics['IC'] = ic if not np.isnan(ic) else 0.0
                
                # IC Information Ratio (IC_IR)
                if len(y_true_clean) > 20:
                    # Rolling IC to calculate stability
                    window = min(20, len(y_true_clean) // 4)
                    rolling_ics = []
                    for i in range(window, len(y_true_clean)):
                        window_ic = np.corrcoef(y_true_clean[i-window:i], y_pred_clean[i-window:i])[0, 1]
                        if not np.isnan(window_ic):
                            rolling_ics.append(window_ic)
                    
                    if rolling_ics:
                        ic_mean = np.mean(rolling_ics)
                        ic_std = np.std(rolling_ics)
                        metrics['IC_IR'] = ic_mean / (ic_std + 1e-6)
                        metrics['IC_Stability'] = 1.0 - (ic_std / (abs(ic_mean) + 1e-6))
                    else:
                        metrics['IC_IR'] = 0.0
                        metrics['IC_Stability'] = 0.0
                else:
                    metrics['IC_IR'] = 0.0
                    metrics['IC_Stability'] = 0.0
                    
            except:
                metrics['IC'] = 0.0
                metrics['IC_IR'] = 0.0
                metrics['IC_Stability'] = 0.0
            
            # Additional regression metrics
            try:
                mse = np.mean((y_true_clean - y_pred_clean) ** 2)
                metrics['MSE'] = mse
                metrics['RMSE'] = np.sqrt(mse)
            except:
                metrics['MSE'] = float('inf')
                metrics['RMSE'] = float('inf')
                
        else:
            # Classification metrics
            try:
                from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision
                
                y_binary = (y_true_clean > 0).astype(int)
                metrics['AUC'] = roc_auc_score(y_binary, y_pred_clean)
                metrics['PR_AUC'] = average_precision(y_binary, y_pred_clean)
                
                # Calibration metrics
                # Brier score
                metrics['Brier_Score'] = np.mean((y_pred_clean - y_binary) ** 2)
                
                # Expected Calibration Error (simplified)
                n_bins = 10
                bin_edges = np.linspace(0, 1, n_bins + 1)
                bin_lowers = bin_edges[:-1]
                bin_uppers = bin_edges[1:]
                
                ece = 0.0
                for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                    in_bin = (y_pred_clean > bin_lower) & (y_pred_clean <= bin_upper)
                    prop_in_bin = in_bin.mean()
                    if prop_in_bin > 0:
                        accuracy_in_bin = y_binary[in_bin].mean()
                        avg_confidence_in_bin = y_pred_clean[in_bin].mean()
                        ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
                metrics['ECE'] = ece
                
            except Exception as e:
                metrics['AUC'] = 0.5
                metrics['PR_AUC'] = 0.5
                metrics['Brier_Score'] = 1.0
                metrics['ECE'] = 1.0
        
        tprint_info("Finished calculate_extended_metrics")
        return metrics
    
    def calculate_model_correlations(self, models_dict: Dict[str, Any], 
                                   horizon: str, task_type: str) -> Dict[str, float]:
        """
        Calculate model correlation matrix for ensemble analysis.
        
        Args:
            models_dict: Dictionary of model predictions
            horizon: '12' or '48'
            task_type: 'regression' or 'classification'
            
        Returns:
            Dictionary with correlation metrics
        """
        tprint_info(f"Starting calculate_model_correlations (horizon={horizon}, task_type={task_type})")
        suffix = f"{horizon}_{'reg' if task_type == 'regression' else 'cls'}"
        model_keys = [k for k in models_dict.keys() if k.endswith(suffix)]
        
        if len(model_keys) < 2:
            tprint_info("Finished calculate_model_correlations (insufficient models)")
            return {'avg_correlation': 0.0, 'max_correlation': 0.0, 'min_correlation': 0.0}
        
        # Collect predictions
        predictions = []
        for key in model_keys:
            if key in models_dict and models_dict[key] is not None:
                pred = models_dict[key]['cate']
                if not np.all(np.isnan(pred)):
                    predictions.append(pred)
        
        if len(predictions) < 2:
            return {'avg_correlation': 0.0, 'max_correlation': 0.0, 'min_correlation': 0.0}
        
        # Calculate correlation matrix
        pred_matrix = np.column_stack(predictions)
        corr_matrix = np.corrcoef(pred_matrix.T)
        
        # Extract upper triangle (excluding diagonal)
        upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
        
        # Handle NaN values
        valid_corrs = upper_triangle[~np.isnan(upper_triangle)]
        
        if len(valid_corrs) == 0:
            tprint_info("Finished calculate_model_correlations (no valid correlations)")
            return {'avg_correlation': 0.0, 'max_correlation': 0.0, 'min_correlation': 0.0}
        
        tprint_info("Finished calculate_model_correlations")
        return {
            'avg_correlation': np.mean(valid_corrs),
            'max_correlation': np.max(valid_corrs),
            'min_correlation': np.min(valid_corrs),
            'diversity_score': 1.0 - np.mean(valid_corrs)  # Higher diversity = lower correlation
        }
    
    def generate_model_race_report(self, models_dict: Dict[str, Any], 
                                 y_alpha_12: np.ndarray, y_prob_12: np.ndarray,
                                 y_alpha_48: np.ndarray, y_prob_48: np.ndarray) -> None:
        """
        Generate comprehensive model race report for Layer 3.
        
        Args:
            models_dict: Dictionary of all trained models
            y_alpha_12: 12-bar alpha targets
            y_prob_12: 12-bar probability targets
            y_alpha_48: 48-bar alpha targets
            y_prob_48: 48-bar probability targets
        """
        try:
            tprint_info("Starting generate_model_race_report")
            tprint_info("🏁 Generating Layer 3 Model Race Report...")
            
            # Define tasks
            tasks = [
                ('12', 'alpha', y_alpha_12, 'regression'),
                ('12', 'prob', y_prob_12, 'classification'),
                ('48', 'alpha', y_alpha_48, 'regression'),
                ('48', 'prob', y_prob_48, 'classification')
            ]
            
            # Collect results for all tasks
            all_results = []
            
            for horizon, target_name, y_true, task_type in tasks:
                suffix = f"{horizon}_{'reg' if task_type == 'regression' else 'cls'}"
                
                # Calculate correlations for this task
                correlation_metrics = self.calculate_model_correlations(models_dict, horizon, task_type)
                
                # Evaluate each model
                for model_family in ['et', 'lgbm', 'xgb', 'catboost', 'huber', 'ridge']:
                    model_key = f"{model_family}_{suffix}"
                    
                    if model_key not in models_dict or models_dict[model_key] is None:
                        continue
                    
                    model_result = models_dict[model_key]
                    predictions = model_result['cate']
                    
                    # Calculate extended metrics
                    metrics = self.calculate_extended_metrics(y_true, predictions, task_type)
                    
                    if 'error' in metrics:
                        continue
                    
                    # Build result row
                    result = {
                        'horizon': horizon,
                        'task': target_name,
                        'model_family': model_family,
                        'model_key': model_key,
                        **metrics,
                        **correlation_metrics
                    }
                    
                    # Add model-specific info
                    if model_family == 'ridge' and 'best_alpha' in model_result:
                        result['best_alpha'] = model_result['best_alpha']
                    
                    all_results.append(result)
            
            if not all_results:
                tprint_warning("⚠️ No valid model results for race report")
                return
            
            # Create DataFrame
            results_df = pd.DataFrame(all_results)
            
            # Save CSV
            csv_path = self.outcomes_dir / f"layer3_model_race_{self.ts}.csv"
            results_df.to_csv(csv_path, index=False)
            
            # Generate markdown report
            self._generate_markdown_report(results_df)
            
            tprint_success(f"✅ Layer 3 model race report saved: {csv_path}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate model race report: {e}")
            import traceback
            traceback.print_exc()
        finally:
            tprint_info("Finished generate_model_race_report")
    
    def _generate_markdown_report(self, results_df: pd.DataFrame) -> None:
        """Generate markdown report from results DataFrame."""
        tprint_info("Starting _generate_markdown_report")
        
        lines = ["# Layer 3 Model Race Report\n\n"]
        lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Overall summary
        lines.append("## Overall Summary\n")
        lines.append(f"- **Models Evaluated**: {len(results_df)}\n")
        lines.append(f"- **Model Families**: {', '.join(sorted(results_df['model_family'].unique()))}\n")
        lines.append(f"- **Horizons**: {', '.join(sorted(results_df['horizon'].unique()))}\n")
        lines.append(f"- **Tasks**: {', '.join(sorted(results_df['task'].unique()))}\n")
        lines.append("\n")
        
        # Per-horizon summaries
        for horizon in sorted(results_df['horizon'].unique()):
            horizon_data = results_df[results_df['horizon'] == horizon]
            lines.append(f"## {horizon}-Bar Results\n")
            
            for task in sorted(horizon_data['task'].unique()):
                task_data = horizon_data[horizon_data['task'] == task]
                task_type = 'regression' if task == 'alpha' else 'classification'
                
                lines.append(f"### {task.title()} ({task_type})\n")
                
                # Sort by primary metric
                primary_metric = 'IC' if task_type == 'regression' else 'AUC'
                task_data_sorted = task_data.sort_values(primary_metric, ascending=False)
                
                # Best model
                best_model = task_data_sorted.iloc[0]
                lines.append(f"- **Best Model**: {best_model['model_family']} ({primary_metric}: {best_model[primary_metric]:.4f})\n")
                
                if task_type == 'regression':
                    lines.append(f"- **Best IC_IR**: {task_data_sorted['IC_IR'].max():.4f}\n")
                    lines.append(f"- **Best IC_Stability**: {task_data_sorted['IC_Stability'].max():.4f}\n")
                else:
                    lines.append(f"- **Best PR_AUC**: {task_data_sorted['PR_AUC'].max():.4f}\n")
                    lines.append(f"- **Best ECE**: {task_data_sorted['ECE'].min():.4f}\n")
                
                lines.append(f"- **Avg Diversity**: {task_data_sorted['diversity_score'].mean():.4f}\n")
                lines.append("\n")
        
        # Detailed comparison tables
        lines.append("## Detailed Model Comparison\n")
        
        for horizon in sorted(results_df['horizon'].unique()):
            for task in sorted(results_df[results_df['horizon'] == horizon]['task'].unique()):
                task_data = results_df[(results_df['horizon'] == horizon) & (results_df['task'] == task)]
                task_type = 'regression' if task == 'alpha' else 'classification'
                primary_metric = 'IC' if task_type == 'regression' else 'AUC'
                
                task_data_sorted = task_data.sort_values(primary_metric, ascending=False)
                
                lines.append(f"### {horizon}-Bar {task.title()}\n")
                
                # Table header
                if task_type == 'regression':
                    lines.append("| Model | IC | IC_IR | IC_Stability | MSE | Diversity |\n")
                    lines.append("|-------|----|-------|--------------|-----|-----------|\n")
                else:
                    lines.append("| Model | AUC | PR_AUC | Brier_Score | ECE | Diversity |\n")
                    lines.append("|-------|-----|--------|-------------|-----|-----------|\n")
                
                # Table rows
                for _, row in task_data_sorted.iterrows():
                    if task_type == 'regression':
                        lines.append(f"| {row['model_family']} | {row['IC']:.4f} | "
                                   f"{row['IC_IR']:.4f} | {row['IC_Stability']:.4f} | "
                                   f"{row['MSE']:.4f} | {row['diversity_score']:.4f} |\n")
                    else:
                        lines.append(f"| {row['model_family']} | {row['AUC']:.4f} | "
                                   f"{row['PR_AUC']:.4f} | {row['Brier_Score']:.4f} | "
                                   f"{row['ECE']:.4f} | {row['diversity_score']:.4f} |\n")
                
                lines.append("\n")
        
        # Ensemble analysis
        lines.append("## Ensemble Analysis\n")
        
        for horizon in sorted(results_df['horizon'].unique()):
            horizon_data = results_df[results_df['horizon'] == horizon]
            lines.append(f"### {horizon}-Bar Ensemble Metrics\n")
            
            for task in sorted(horizon_data['task'].unique()):
                task_data = horizon_data[horizon_data['task'] == task]
                
                lines.append(f"**{task.title()}**:\n")
                lines.append(f"- Average Correlation: {task_data['avg_correlation'].mean():.4f}\n")
                lines.append(f"- Max Correlation: {task_data['max_correlation'].max():.4f}\n")
                lines.append(f"- Diversity Score: {task_data['diversity_score'].mean():.4f}\n")
                lines.append("\n")
        
        # Model family performance summary
        lines.append("## Model Family Performance Summary\n")
        
        family_summary = results_df.groupby('model_family').agg({
            'IC': ['mean', 'std'] if 'IC' in results_df.columns else None,
            'AUC': ['mean', 'std'] if 'AUC' in results_df.columns else None,
            'diversity_score': 'mean'
        }).round(4)
        
        lines.append("### Overall Performance by Model Family\n")
        lines.append("| Model Family | Avg IC | IC Std | Avg AUC | AUC Std | Avg Diversity |\n")
        lines.append("|--------------|--------|--------|---------|---------|---------------|\n")
        
        for family in sorted(results_df['model_family'].unique()):
            family_data = results_df[results_df['model_family'] == family]
            
            avg_ic = family_data['IC'].mean() if 'IC' in family_data.columns else 'N/A'
            std_ic = family_data['IC'].std() if 'IC' in family_data.columns else 'N/A'
            avg_auc = family_data['AUC'].mean() if 'AUC' in family_data.columns else 'N/A'
            std_auc = family_data['AUC'].std() if 'AUC' in family_data.columns else 'N/A'
            avg_diversity = family_data['diversity_score'].mean()
            
            ic_str = f"{avg_ic:.4f}" if avg_ic != 'N/A' else 'N/A'
            ic_std_str = f"{std_ic:.4f}" if std_ic != 'N/A' else 'N/A'
            auc_str = f"{avg_auc:.4f}" if avg_auc != 'N/A' else 'N/A'
            auc_std_str = f"{std_auc:.4f}" if std_auc != 'N/A' else 'N/A'
            
            lines.append(f"| {family} | {ic_str} | {ic_std_str} | {auc_str} | {auc_std_str} | {avg_diversity:.4f} |\n")
        
        lines.append("\n")
        
        # Save markdown report
        report_path = self.outcomes_dir / f"layer3_model_race_{self.ts}.md"
        report_path.write_text("".join(lines))
        
        tprint_success(f"✅ Markdown report saved: {report_path}")
        tprint_info("Finished _generate_markdown_report")
