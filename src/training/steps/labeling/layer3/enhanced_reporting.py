"""
Enhanced Layer 3 Reporting System

Handles comprehensive reporting and diagnostics for Layer 3 models with additional
reports for dual-head analysis, feature engineering impact, and cross-layer analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
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

class EnhancedLayer3Reporter:
    """Enhanced reporting for Layer 3 meta-modeling pipeline."""
    
    def __init__(self, outcomes_dir: Optional[Path] = None):
        self.outcomes_dir = outcomes_dir or Path('outcomes')
        self.outcomes_dir.mkdir(exist_ok=True, parents=True)
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_all_reports(self, df: pd.DataFrame, models: Dict[str, Any], 
                           geometry_metrics: List[Dict[str, Any]], 
                           meta_features: List[str], target_col: str,
                           config: Dict[str, Any]) -> None:
        """
        Generate all enhanced Layer 3 reports.
        
        Args:
            df: DataFrame with meta-model predictions
            models: Dictionary containing alpha and probability models
            geometry_metrics: List of geometry performance metrics
            meta_features: List of meta-feature names
            target_col: Target column name
            config: Configuration dictionary
        """
        tprint_info("📊 Generating Enhanced Layer 3 Reports...")
        
        # Generate existing reports
        self._generate_meta_report(df, geometry_metrics, models, target_col, config)
        self._generate_feature_importance_report(models, meta_features)
        self._generate_calibration_diagnostics(df, target_col)
        self._generate_performance_summary(df, models, target_col)
        
        # Generate new enhanced reports
        self._generate_dual_head_analysis_report(df, models, target_col)
        self._generate_feature_engineering_impact_report(df, meta_features, target_col)
        self._generate_cross_layer_analysis_report(df, target_col, config)
        self._generate_model_ensemble_report(models, meta_features)
        self._generate_prediction_confidence_report(df, target_col)
        
        # New sections per de Prado framework
        self._generate_regime_performance_report(df, target_col)
        self._generate_structural_feature_report(df, meta_features, target_col)
        
        tprint_success(f"✅ Enhanced Layer 3 reports saved to {self.outcomes_dir}")
    
    def _generate_meta_report(self, df: pd.DataFrame, geometry_metrics: List[Dict[str, Any]], 
                            models: Dict[str, Any], target_col: str, config: Dict[str, Any]) -> None:
        """Generate enhanced meta-report."""
        try:
            lines = ["# Enhanced Layer 3 Meta-Labeling Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            lines.append(f"Instrument: {config.get('symbol', 'UNKNOWN')} | Timeframe: {config.get('timeframe', '15m')}\n\n")

            # Geometry Performance Summary
            lines.append("## Geometry Performance Summary\n")
            if geometry_metrics:
                metric_df = pd.DataFrame(geometry_metrics)
                metric_df = metric_df.sort_values('score', ascending=False)
                lines.append(self._safe_to_markdown(metric_df) + "\n\n")
                
                # Save CSV
                metric_df.to_csv(self.outcomes_dir / f"layer3_geometry_metrics_{self.ts}.csv", index=False)
            else:
                lines.append("No geometry-specific metrics available.\n\n")

            # Alpha Head Performance
            lines.append("## Alpha Head Performance\n")
            if 'alpha_metrics' in models:
                alpha_metrics = models['alpha_metrics']
                lines.append(f"- **Final IC**: {alpha_metrics.get('final_ic', 'N/A'):.4f}\n")
                lines.append(f"- **Selected Models**: {', '.join(alpha_metrics.get('selected_models', []))}\n")
                
                if 'model_scores' in alpha_metrics:
                    lines.append("\n### Model Scores\n")
                    scores_df = pd.DataFrame(list(alpha_metrics['model_scores'].items()), 
                                           columns=['Model', 'IC'])
                    lines.append(self._safe_to_markdown(scores_df) + "\n\n")
                    scores_df.to_csv(self.outcomes_dir / f"layer3_alpha_model_scores_{self.ts}.csv", index=False)
            else:
                lines.append("Alpha metrics not available.\n\n")

            # Probability Head Performance
            lines.append("## Probability Head Performance\n")
            if 'prob_metrics' in models:
                prob_metrics = models['prob_metrics']
                lines.append(f"- **Final AUC**: {prob_metrics.get('final_auc', 'N/A'):.4f}\n")
                lines.append(f"- **Final LogLoss**: {prob_metrics.get('final_logloss', 'N/A'):.4f}\n")
                lines.append(f"- **Selected Models**: {', '.join(prob_metrics.get('selected_models', []))}\n")
                
                if 'model_scores' in prob_metrics:
                    lines.append("\n### Model Scores\n")
                    scores_df = pd.DataFrame(list(prob_metrics['model_scores'].items()), 
                                           columns=['Model', 'AUC'])
                    lines.append(self._safe_to_markdown(scores_df) + "\n\n")
                    scores_df.to_csv(self.outcomes_dir / f"layer3_prob_model_scores_{self.ts}.csv", index=False)
            else:
                lines.append("Probability metrics not available.\n\n")

            # Feature Summary
            lines.append("## Feature Summary\n")
            feature_counts = {
                'total': len(df.columns),
                'meta': len([c for c in df.columns if 'meta_' in c]),
                'layer0': len([c for c in df.columns if any(x in c for x in ['unified', 'adaptive', 'noise', 'filter'])]),
                'layer1': len([c for c in df.columns if 'layer1_weight' in c])
            }
            
            for category, count in feature_counts.items():
                lines.append(f"- **{category.capitalize()} Features**: {count}\n")
            
            # Save feature summary
            feature_df = pd.DataFrame([feature_counts])
            feature_df.to_csv(self.outcomes_dir / f"layer3_feature_summary_{self.ts}.csv", index=False)

            report_path = self.outcomes_dir / f"layer3_enhanced_meta_report_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Enhanced meta-report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate enhanced meta-report: {e}")
    
    def _generate_dual_head_analysis_report(self, df: pd.DataFrame, models: Dict[str, Any], 
                                          target_col: str) -> None:
        """Generate dual-head analysis report."""
        try:
            lines = ["# Dual-Head Analysis Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if 'meta_alpha' in df.columns and 'meta_prob' in df.columns and target_col in df.columns:
                # Correlation analysis
                from scipy.stats import spearmanr, pearsonr
                alpha_prob_corr, _ = pearsonr(df['meta_alpha'], df['meta_prob'])
                lines.append("## Head Correlation Analysis\n")
                lines.append(f"- **Alpha-Probability Correlation**: {alpha_prob_corr:.4f}\n")
                
                # Performance by head
                lines.append("\n## Individual Head Performance\n")
                
                # Alpha head performance
                ic, _ = spearmanr(df[target_col], df['meta_alpha'])
                lines.append(f"### Alpha Head\n")
                lines.append(f"- **Information Coefficient (IC)**: {ic:.4f}\n")
                
                # Probability head performance
                from sklearn.metrics import roc_auc_score, log_loss
                y_true = (df[target_col] > 0.5).astype(int)
                y_prob = df['meta_prob']
                auc = roc_auc_score(y_true, y_prob)
                logloss = log_loss(y_true, y_prob)
                
                lines.append(f"### Probability Head\n")
                lines.append(f"- **AUC**: {auc:.4f}\n")
                lines.append(f"- **LogLoss**: {logloss:.4f}\n")
                
                # Combined performance analysis
                lines.append("\n## Combined Performance Analysis\n")
                
                # Performance by probability bins
                prob_bins = pd.qcut(df['meta_prob'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                bin_performance = df.groupby(prob_bins).agg({
                    target_col: ['mean', 'count'],
                    'meta_alpha': ['mean', 'std']
                }).round(4)
                
                lines.append("### Performance by Probability Bins\n")
                lines.append(self._safe_to_markdown(bin_performance) + "\n\n")
                
                # Save analysis data
                dual_head_data = {
                    'alpha_prob_correlation': alpha_prob_corr,
                    'alpha_ic': ic,
                    'probability_auc': auc,
                    'probability_logloss': logloss
                }
                
                dual_df = pd.DataFrame([dual_head_data])
                dual_df.to_csv(self.outcomes_dir / f"layer3_dual_head_analysis_{self.ts}.csv", index=False)
                bin_performance.to_csv(self.outcomes_dir / f"layer3_performance_by_prob_bins_{self.ts}.csv")

            else:
                lines.append("Required columns not available for dual-head analysis.\n")

            report_path = self.outcomes_dir / f"layer3_dual_head_report_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Dual-head analysis report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate dual-head analysis report: {e}")
    
    def _generate_feature_engineering_impact_report(self, df: pd.DataFrame, meta_features: List[str], 
                                                   target_col: str) -> None:
        """Generate feature engineering impact report."""
        try:
            lines = ["# Feature Engineering Impact Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Feature category analysis
            feature_categories = {
                'meta_features': [f for f in meta_features if 'meta_' in f],
                'layer0_features': [f for f in df.columns if any(x in f for x in ['unified', 'adaptive', 'noise', 'filter'])],
                'layer1_features': [f for f in df.columns if 'layer1_weight' in f],
                'other_features': [f for f in df.columns if f not in meta_features and 
                                 not any(x in f for x in ['meta_', 'unified', 'adaptive', 'noise', 'filter', 'layer1_weight'])]
            }

            lines.append("## Feature Category Breakdown\n")
            for category, features in feature_categories.items():
                lines.append(f"- **{category.replace('_', ' ').title()}**: {len(features)}\n")
            
            # Feature importance by category (if available)
            if 'meta_alpha' in df.columns and target_col in df.columns:
                lines.append("\n## Feature Impact Analysis\n")
                
                from scipy.stats import spearmanr
                feature_impacts = []
                
                for feature in meta_features:
                    if feature in df.columns:
                        corr, p_val = spearmanr(df[feature], df[target_col])
                        feature_impacts.append({
                            'feature': feature,
                            'correlation': corr,
                            'p_value': p_val,
                            'category': self._categorize_feature(feature, feature_categories)
                        })
                
                if feature_impacts:
                    impact_df = pd.DataFrame(feature_impacts)
                    impact_df = impact_df.sort_values('correlation', key=abs, ascending=False)
                    
                    lines.append("### Top Features by Impact\n")
                    lines.append(self._safe_to_markdown(impact_df.head(20)) + "\n\n")
                    
                    # Category summary
                    category_summary = impact_df.groupby('category')['correlation'].agg(['mean', 'std', 'count']).round(4)
                    lines.append("### Impact by Category\n")
                    lines.append(self._safe_to_markdown(category_summary) + "\n\n")
                    
                    # Save data
                    impact_df.to_csv(self.outcomes_dir / f"layer3_feature_impacts_{self.ts}.csv", index=False)
                    category_summary.to_csv(self.outcomes_dir / f"layer3_category_impacts_{self.ts}.csv")

            report_path = self.outcomes_dir / f"layer3_feature_engineering_impact_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Feature engineering impact report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate feature engineering impact report: {e}")
    
    def _generate_cross_layer_analysis_report(self, df: pd.DataFrame, target_col: str, 
                                            config: Dict[str, Any]) -> None:
        """Generate cross-layer analysis report."""
        try:
            lines = ["# Cross-Layer Analysis Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Layer integration analysis
            lines.append("## Layer Integration Analysis\n")
            
            # Check for layer-specific features
            layer_features = {
                'Layer 0': [f for f in df.columns if any(x in f for x in ['unified', 'adaptive', 'noise', 'filter'])],
                'Layer 1': [f for f in df.columns if 'layer1_weight' in f],
                'Layer 2': [f for f in df.columns if 'meta_' in f],
                'Layer 3': [f for f in df.columns if f in ['meta_alpha', 'meta_prob']]
            }
            
            for layer, features in layer_features.items():
                lines.append(f"- **{layer} Features**: {len(features)}\n")
                if features and layer != 'Layer 3':
                    sample_features = features[:3]
                    lines.append(f"  - Sample: {', '.join(sample_features)}\n")
            
            # Performance progression (if we have historical data)
            lines.append("\n## Performance Progression\n")
            lines.append("Note: This would show performance metrics across layers if historical data is available.\n")
            
            # Feature flow analysis
            lines.append("\n## Feature Flow Analysis\n")
            total_features = len(df.columns)
            lines.append(f"- **Total Feature Count**: {total_features}\n")
            lines.append(f"- **Feature Reduction Ratio**: {(1 - len(layer_features['Layer 3'])/total_features)*100:.1f}%\n")
            
            # Save cross-layer data
            cross_layer_df = pd.DataFrame([
                {'layer': layer, 'feature_count': len(features)}
                for layer, features in layer_features.items()
            ])
            cross_layer_df.to_csv(self.outcomes_dir / f"layer3_cross_layer_analysis_{self.ts}.csv", index=False)

            report_path = self.outcomes_dir / f"layer3_cross_layer_report_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Cross-layer analysis report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate cross-layer analysis report: {e}")
    
    def _generate_model_ensemble_report(self, models: Dict[str, Any], meta_features: List[str]) -> None:
        """Generate model ensemble analysis report."""
        try:
            lines = ["# Model Ensemble Analysis Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Alpha ensemble analysis
            lines.append("## Alpha Head Ensemble\n")
            if 'alpha_models' in models and len(models['alpha_models']) > 0:
                alpha_models = models['alpha_models']
                lines.append(f"- **Number of Models**: {len(alpha_models)}\n")
                
                # Feature importance consensus
                if hasattr(alpha_models[0], 'feature_importances_'):
                    consensus_importance = np.zeros(len(meta_features))
                    for model in alpha_models:
                        if hasattr(model, 'feature_importances_'):
                            consensus_importance += model.feature_importances_
                    consensus_importance /= len(alpha_models)
                    
                    consensus_df = pd.DataFrame({
                        'feature': meta_features,
                        'consensus_importance': consensus_importance
                    }).sort_values('consensus_importance', ascending=False)
                    
                    lines.append("### Top Features by Consensus Importance\n")
                    lines.append(self._safe_to_markdown(consensus_df.head(15)) + "\n\n")
                    
                    consensus_df.to_csv(self.outcomes_dir / f"layer3_alpha_consensus_importance_{self.ts}.csv", index=False)

            # Probability ensemble analysis
            lines.append("## Probability Head Ensemble\n")
            if 'prob_models' in models and len(models['prob_models']) > 0:
                prob_models = models['prob_models']
                lines.append(f"- **Number of Models**: {len(prob_models)}\n")
                
                # Feature importance consensus
                if hasattr(prob_models[0], 'feature_importances_'):
                    consensus_importance = np.zeros(len(meta_features))
                    for model in prob_models:
                        if hasattr(model, 'feature_importances_'):
                            consensus_importance += model.feature_importances_
                    consensus_importance /= len(prob_models)
                    
                    consensus_df = pd.DataFrame({
                        'feature': meta_features,
                        'consensus_importance': consensus_importance
                    }).sort_values('consensus_importance', ascending=False)
                    
                    lines.append("### Top Features by Consensus Importance\n")
                    lines.append(self._safe_to_markdown(consensus_df.head(15)) + "\n\n")
                    
                    consensus_df.to_csv(self.outcomes_dir / f"layer3_prob_consensus_importance_{self.ts}.csv", index=False)

            report_path = self.outcomes_dir / f"layer3_model_ensemble_report_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Model ensemble report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate model ensemble report: {e}")
    
    def _generate_prediction_confidence_report(self, df: pd.DataFrame, target_col: str) -> None:
        """Generate prediction confidence analysis report."""
        try:
            lines = ["# Prediction Confidence Analysis Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if 'meta_prob' in df.columns and target_col in df.columns:
                # Confidence distribution analysis
                lines.append("## Confidence Distribution Analysis\n")
                
                y_true = (df[target_col] > 0.5).astype(int)
                y_prob = df['meta_prob']
                
                # Overall statistics
                lines.append(f"- **Mean Confidence**: {y_prob.mean():.3f}\n")
                lines.append(f"- **Confidence Std**: {y_prob.std():.3f}\n")
                lines.append(f"- **Min Confidence**: {y_prob.min():.3f}\n")
                lines.append(f"- **Max Confidence**: {y_prob.max():.3f}\n")
                
                # Confidence bins analysis
                confidence_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                bin_analysis = []
                
                for i in range(len(confidence_bins) - 1):
                    lower, upper = confidence_bins[i], confidence_bins[i + 1]
                    mask = (y_prob >= lower) & (y_prob < upper)
                    
                    if mask.sum() > 0:
                        accuracy = y_true[mask].mean()
                        count = mask.sum()
                        bin_analysis.append({
                            'confidence_range': f"{lower:.1f}-{upper:.1f}",
                            'sample_count': count,
                            'accuracy': accuracy,
                            'confidence': (lower + upper) / 2
                        })
                
                if bin_analysis:
                    bin_df = pd.DataFrame(bin_analysis)
                    lines.append("\n### Performance by Confidence Bins\n")
                    lines.append(self._safe_to_markdown(bin_df) + "\n\n")
                    
                    # Confidence-accuracy gap analysis
                    bin_df['confidence_accuracy_gap'] = abs(bin_df['confidence'] - bin_df['accuracy'])
                    lines.append("### Confidence-Accuracy Gap Analysis\n")
                    lines.append(self._safe_to_markdown(bin_df[['confidence_range', 'confidence_accuracy_gap']]) + "\n\n")
                    
                    # Save data
                    bin_df.to_csv(self.outcomes_dir / f"layer3_confidence_analysis_{self.ts}.csv", index=False)

            report_path = self.outcomes_dir / f"layer3_prediction_confidence_report_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Prediction confidence report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate prediction confidence report: {e}")
    
    def _generate_feature_importance_report(self, models: Dict[str, Any], meta_features: List[str]) -> None:
        """Generate feature importance report (from original)."""
        try:
            importance_data = []
            
            # Extract feature importances from models
            if 'alpha_models' in models:
                for i, model in enumerate(models['alpha_models']):
                    if hasattr(model, 'feature_importances_'):
                        for j, importance in enumerate(model.feature_importances_):
                            if j < len(meta_features):
                                importance_data.append({
                                    'feature': meta_features[j],
                                    'importance': importance,
                                    'model': f'alpha_model_{i}',
                                    'head': 'alpha'
                                })
            
            if 'prob_models' in models:
                for i, model in enumerate(models['prob_models']):
                    if hasattr(model, 'feature_importances_'):
                        for j, importance in enumerate(model.feature_importances_):
                            if j < len(meta_features):
                                importance_data.append({
                                    'feature': meta_features[j],
                                    'importance': importance,
                                    'model': f'prob_model_{i}',
                                    'head': 'probability'
                                })
            
            if importance_data:
                importance_df = pd.DataFrame(importance_data)
                
                # Aggregate by feature
                feature_summary = importance_df.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
                feature_summary.columns = ['feature', 'mean_importance', 'std_importance', 'model_count']
                feature_summary = feature_summary.sort_values('mean_importance', ascending=False)
                
                # Save detailed importance
                importance_df.to_csv(self.outcomes_dir / f"layer3_feature_importance_detailed_{self.ts}.csv", index=False)
                feature_summary.to_csv(self.outcomes_dir / f"layer3_feature_importance_summary_{self.ts}.csv", index=False)

        except Exception as e:
            tprint_error(f"❌ Failed to generate feature importance report: {e}")
    
    def _generate_calibration_diagnostics(self, df: pd.DataFrame, target_col: str) -> None:
        """Generate calibration diagnostics (from original)."""
        try:
            if 'meta_prob' not in df.columns or target_col not in df.columns:
                return
            
            y_true = (df[target_col] > 0.5).astype(int).values
            y_prob = df['meta_prob'].values
            
            # Calculate calibration metrics
            ece = self._fast_expected_calibration_error(y_true, y_prob)
            
            # Create calibration plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Reliability diagram
            from sklearn.calibration import calibration_curve
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            
            axes[0, 0].plot(prob_pred, prob_true, marker='o', linewidth=2, label='Meta-Model')
            axes[0, 0].plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.5, label='Perfect')
            axes[0, 0].set_xlabel('Predicted Probability')
            axes[0, 0].set_ylabel('Actual Win Rate')
            axes[0, 0].set_title('Calibration (Reliability)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Probability distribution
            axes[0, 1].hist(y_prob, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 1].set_xlabel('Predicted Probability')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Probability Distribution')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Confidence vs Accuracy
            confidence_bins = np.linspace(0.5, 1.0, 10)
            accuracy_by_confidence = []
            
            for i in range(len(confidence_bins) - 1):
                mask = (y_prob >= confidence_bins[i]) & (y_prob < confidence_bins[i + 1])
                if mask.sum() > 0:
                    accuracy = y_true[mask].mean()
                    accuracy_by_confidence.append(accuracy)
                else:
                    accuracy_by_confidence.append(np.nan)
            
            axes[1, 0].plot(confidence_bins[:-1], accuracy_by_confidence, marker='o', linewidth=2)
            axes[1, 0].plot([0.5, 1.0], [0.5, 1.0], linestyle='--', color='gray', alpha=0.5)
            axes[1, 0].set_xlabel('Confidence Threshold')
            axes[1, 0].set_ylabel('Actual Accuracy')
            axes[1, 0].set_title('Confidence vs Accuracy')
            axes[1, 0].grid(True, alpha=0.3)
            
            # ECE by threshold
            ece_by_threshold = []
            for threshold in np.linspace(0.5, 0.95, 10):
                mask = y_prob >= threshold
                if mask.sum() > 0:
                    ece = self._fast_expected_calibration_error(y_true[mask], y_prob[mask])
                    ece_by_threshold.append(ece)
                else:
                    ece_by_threshold.append(np.nan)
            
            axes[1, 1].plot(np.linspace(0.5, 0.95, 10), ece_by_threshold, marker='o', linewidth=2)
            axes[1, 1].set_xlabel('Probability Threshold')
            axes[1, 1].set_ylabel('ECE')
            axes[1, 1].set_title('ECE by Probability Threshold')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.outcomes_dir / f"layer3_calibration_diagnostics_{self.ts}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            tprint_error(f"❌ Failed to generate calibration diagnostics: {e}")
    
    def _generate_performance_summary(self, df: pd.DataFrame, models: Dict[str, Any], target_col: str) -> None:
        """Generate performance summary (from original)."""
        try:
            summary_stats = {}
            
            # Alpha performance
            if 'meta_alpha' in df.columns and target_col in df.columns:
                from scipy.stats import spearmanr
                ic, _ = spearmanr(df[target_col], df['meta_alpha'])
                summary_stats['alpha_ic'] = ic
            
            # Probability performance
            if 'meta_prob' in df.columns and target_col in df.columns:
                from sklearn.metrics import roc_auc_score, log_loss
                y_true = (df[target_col] > 0.5).astype(int)
                y_prob = df['meta_prob']
                
                summary_stats['prob_auc'] = roc_auc_score(y_true, y_prob)
                summary_stats['prob_logloss'] = log_loss(y_true, y_prob)
                summary_stats['prob_ece'] = self._fast_expected_calibration_error(y_true, y_prob)
            
            # Data statistics
            summary_stats['total_samples'] = len(df)
            summary_stats['positive_rate'] = (df[target_col] > 0.5).mean() if target_col in df.columns else np.nan
            summary_stats['meta_prob_mean'] = df['meta_prob'].mean() if 'meta_prob' in df.columns else np.nan
            summary_stats['meta_prob_std'] = df['meta_prob'].std() if 'meta_prob' in df.columns else np.nan
            
            # Save summary
            summary_df = pd.DataFrame([summary_stats])
            summary_df.to_csv(self.outcomes_dir / f"layer3_performance_summary_{self.ts}.csv", index=False)

        except Exception as e:
            tprint_error(f"❌ Failed to generate performance summary: {e}")
    
    def _categorize_feature(self, feature: str, categories: Dict[str, List[str]]) -> str:
        """Categorize a feature name."""
        for category, features in categories.items():
            if feature in features:
                return category
        return 'other'
    
    def _generate_regime_performance_report(self, df: pd.DataFrame, target_col: str) -> None:
        """Generate per-regime performance report."""
        try:
            if 'regime_label' not in df.columns or target_col not in df.columns:
                return

            lines = ["# Per-Regime Performance Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            regimes = df['regime_label'].unique()
            regime_metrics = []

            from sklearn.metrics import roc_auc_score
            from scipy.stats import spearmanr

            for regime in regimes:
                mask = df['regime_label'] == regime
                regime_df = df[mask]
                
                if len(regime_df) < 5: # Minimal threshold
                    continue
                
                metrics = {
                    'regime': regime, 
                    'samples': len(regime_df),
                    'pos_rate': (regime_df[target_col] > 0.5).mean()
                }
                
                if 'meta_alpha' in df.columns:
                    ic, _ = spearmanr(regime_df[target_col], regime_df['meta_alpha'])
                    metrics['alpha_ic'] = ic
                
                if 'meta_prob' in df.columns:
                    y_true = (regime_df[target_col] > 0.5).astype(int)
                    if len(y_true.unique()) > 1:
                        metrics['prob_auc'] = roc_auc_score(y_true, regime_df['meta_prob'])
                    else:
                        metrics['prob_auc'] = np.nan
                
                regime_metrics.append(metrics)

            if regime_metrics:
                reg_df = pd.DataFrame(regime_metrics)
                lines.append(self._safe_to_markdown(reg_df) + "\n\n")
                reg_df.to_csv(self.outcomes_dir / f"layer3_regime_performance_{self.ts}.csv", index=False)

            report_path = self.outcomes_dir / f"layer3_regime_report_{self.ts}.md"
            report_path.write_text("".join(lines))
        except Exception as e:
            tprint_error(f"❌ Failed to generate regime report: {e}")

    def _generate_structural_feature_report(self, df: pd.DataFrame, meta_features: List[str], target_col: str) -> None:
        """Generate structural feature impact report (Anchor and Drift)."""
        try:
            anchor_features = [f for f in meta_features if 'anchor_' in f or 'stability' in f]
            if not anchor_features or target_col not in df.columns:
                return

            lines = ["# Structural Feature Analysis (Anchor and Drift)\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            from scipy.stats import spearmanr
            impacts = []
            for f in anchor_features:
                if f in df.columns:
                    corr, p = spearmanr(df[f], df[target_col])
                    impacts.append({
                        'feature': f, 
                        'correlation': corr, 
                        'p_value': p,
                        'type': 'drift' if 'stability' in f else 'anchor'
                    })

            if impacts:
                impact_df = pd.DataFrame(impacts).sort_values('correlation', key=abs, ascending=False)
                lines.append("## Anchor/Drift Feature Impacts\n")
                lines.append(self._safe_to_markdown(impact_df) + "\n\n")
                impact_df.to_csv(self.outcomes_dir / f"layer3_structural_impacts_{self.ts}.csv", index=False)

            report_path = self.outcomes_dir / f"layer3_structural_report_{self.ts}.md"
            report_path.write_text("".join(lines))
        except Exception as e:
            tprint_error(f"❌ Failed to generate structural report: {e}")

    def _fast_expected_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Fast Expected Calibration Error calculation."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                avg_confidence_in_bin = y_prob[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def _safe_to_markdown(self, df: pd.DataFrame) -> str:
        """Fallback for to_markdown() if tabulate is missing."""
        try:
            return df.to_markdown(index=False)
        except Exception:
            cols = df.columns
            res = [" | " + " | ".join(map(str, cols)) + " | "]
            res.append(" | " + " | ".join(["---"] * len(cols)) + " | ")
            for _, row in df.iterrows():
                formatted_row = [f"{x:.4f}" if isinstance(x, (float, np.float64, np.float32)) else str(x) for x in row]
                res.append(" | " + " | ".join(formatted_row) + " | ")
            return "\n".join(res)
