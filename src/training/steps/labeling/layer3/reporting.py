"""
Layer 3 Reporting System

Handles comprehensive reporting and diagnostics for Layer 3 models.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
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

def generate_layer3_reports(
    df: pd.DataFrame,
    models: Dict[str, Any],
    geometry_metrics: List[Dict[str, Any]],
    meta_features: List[str],
    target_col: str,
    outcomes_dir: Path,
    ts: str,
    config: Dict[str, Any]
) -> None:
    """
    Generate comprehensive Layer 3 reports.
    """
    tprint_info("📊 Generating Comprehensive Layer 3 Reports...")
    tprint_info(f"📂 Output directory: {outcomes_dir}")
    tprint_info(f"🕐 Timestamp: {ts}")
    
    report_start_time = datetime.now()
    
    # Generate meta-report
    tprint_info("📋 Generating Meta-Report...")
    _generate_layer3_meta_report(df, geometry_metrics, models, target_col, outcomes_dir, ts, config)
    
    # Generate feature importance report
    tprint_info("📈 Generating Feature Importance Report...")
    _generate_feature_importance_report(models, meta_features, outcomes_dir, ts)
    
    # Generate calibration diagnostics
    tprint_info("🎯 Generating Calibration Diagnostics...")
    _generate_calibration_diagnostics(df, target_col, outcomes_dir, ts)
    
    # Generate performance summary
    tprint_info("📊 Generating Performance Summary...")
    _generate_performance_summary(df, models, target_col, outcomes_dir, ts)
    
    report_time = (datetime.now() - report_start_time).total_seconds()
    tprint_success(f"✅ All reports generated in {report_time:.2f}s")

def _generate_layer3_meta_report(
    df: pd.DataFrame,
    geometry_metrics: List[Dict[str, Any]],
    models: Dict[str, Any],
    target_col: str,
    outcomes_dir: Path,
    ts: str,
    config: Dict[str, Any]
) -> None:
    """
    Generate detailed Layer 3 meta-report.
    """
    try:
        tprint_info("📝 Writing Layer 3 Meta-Report...")
        
        lines = ["# Layer 3 Meta-Labeling Consolidated Report\n\n"]
        lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        lines.append(f"Instrument: {config.get('symbol', 'UNKNOWN')} | Timeframe: {config.get('timeframe', '15m')}\n\n")

        # 1. Geometry Performance Summary
        lines.append("## Geometry Performance Summary\n")
        if geometry_metrics:
            metric_df = pd.DataFrame(geometry_metrics)
            metric_df = metric_df.sort_values('score', ascending=False)
            lines.append(_safe_to_markdown(metric_df) + "\n\n")
            tprint_info(f"📊 Geometry metrics: {len(metric_df)} geometries")
        else:
            lines.append("No geometry-specific metrics available.\n\n")
            tprint_warning("⚠️ No geometry metrics available")

        # 2. Alpha Head Performance
        lines.append("## Alpha Head Performance\n")
        if 'alpha_metrics' in models:
            alpha_metrics = models['alpha_metrics']
            lines.append(f"- **Final IC**: {alpha_metrics.get('final_ic', 'N/A'):.4f}\n")
            lines.append(f"- **Selected Models**: {', '.join(alpha_metrics.get('selected_models', []))}\n")
            
            if 'model_scores' in alpha_metrics:
                lines.append("\n### Model Scores\n")
                for model, score in alpha_metrics['model_scores'].items():
                    lines.append(f"- {model}: {score:.4f}\n")
            lines.append("\n")
            
            tprint_info(f"📈 Alpha IC: {alpha_metrics.get('final_ic', 'N/A'):.4f}")
        else:
            lines.append("Alpha metrics not available.\n\n")
            tprint_warning("⚠️ Alpha metrics not available")

        # 3. Probability Head Performance
        lines.append("## Probability Head Performance\n")
        if 'prob_metrics' in models:
            prob_metrics = models['prob_metrics']
            lines.append(f"- **Final AUC**: {prob_metrics.get('final_auc', 'N/A'):.4f}\n")
            lines.append(f"- **Final LogLoss**: {prob_metrics.get('final_logloss', 'N/A'):.4f}\n")
            lines.append(f"- **Selected Models**: {', '.join(prob_metrics.get('selected_models', []))}\n")
            
            if 'model_scores' in prob_metrics:
                lines.append("\n### Model Scores\n")
                for model, score in prob_metrics['model_scores'].items():
                    lines.append(f"- {model}: {score:.4f}\n")
            lines.append("\n")
            
            tprint_info(f"🎯 Probability AUC: {prob_metrics.get('final_auc', 'N/A'):.4f}")
            tprint_info(f"🎯 Probability LogLoss: {prob_metrics.get('final_logloss', 'N/A'):.4f}")
        else:
            lines.append("Probability metrics not available.\n\n")
            tprint_warning("⚠️ Probability metrics not available")

        # 4. Calibration Diagnostics
        lines.append("## Calibration Diagnostics\n")
        if 'meta_prob' in df.columns and target_col in df.columns:
            y_true = (df[target_col] > 0.5).astype(int).values
            y_prob = df['meta_prob'].values
            
            ece = _fast_expected_calibration_error(y_true, y_prob)
            lines.append(f"- **Expected Calibration Error (ECE)**: {ece:.4f}\n")
            lines.append("- *Note: ECE targets < 0.05 for high-confidence trading.*\n\n")
            
            tprint_info(f"🎯 Calibration ECE: {ece:.4f}")
        else:
            lines.append("Calibration metrics not available.\n\n")
            tprint_warning("⚠️ Calibration metrics not available")

        # 5. Feature Summary
        lines.append("## Feature Summary\n")
        lines.append(f"- **Total Features**: {len(df.columns)}\n")
        lines.append(f"- **Meta Features**: {len([c for c in df.columns if 'meta_' in c])}\n")
        lines.append(f"- **Layer 0 Features**: {len([c for c in df.columns if any(x in c for x in ['unified', 'adaptive', 'noise', 'filter'])])}\n")
        lines.append(f"- **Layer 1 Weight Features**: {len([c for c in df.columns if 'layer1_weight' in c])}\n\n")
        
        feature_counts = {
            'total': len(df.columns),
            'meta': len([c for c in df.columns if 'meta_' in c]),
            'layer0': len([c for c in df.columns if any(x in c for x in ['unified', 'adaptive', 'noise', 'filter'])]),
            'layer1': len([c for c in df.columns if 'layer1_weight' in c])
        }
        
        for category, count in feature_counts.items():
            tprint_info(f"📊 {category.capitalize()} features: {count}")

        # 6. Framework Details
        lines.append("## Causal Framework Summary\n")
        causal_summary = config.get('causal_summary', {})
        if causal_summary:
            lines.append(f"- **Framework Type**: {causal_summary.get('framework_type', 'N/A')}\n")
            lines.append(f"- **Specialist Count**: {causal_summary.get('specialist_count', 'N/A')}\n")
            lines.append(f"- **Causal Event Count**: {causal_summary.get('causal_event_count', 'N/A')}\n")
            lines.append(f"- **Surprise Density**: {causal_summary.get('surprise_density', 'N/A')}\n")
            lines.append(f"- **Causal Target Columns**: {causal_summary.get('causal_target_columns', 'N/A')}\n")
            lines.append(f"- **Dataset Fingerprint**: {causal_summary.get('dataset_fingerprint', 'N/A')}\n\n")
        else:
            lines.append("Causal summary not available.\n\n")

        # 7. Hyperparameter Parity
        lines.append("## Causal Hyperparameters\n")
        causal_hps = config.get('causal_hyperparams', {})
        if causal_hps:
            for key, value in causal_hps.items():
                lines.append(f"- **{key}**: {value}\n")
            lines.append("\n")
        else:
            lines.append("No causal hyperparameters were propagated from Layer 2.\n\n")

        # 8. Selected Weighting Schemes
        lines.append("## Weighting Schemes\n")
        selected_scheme = models.get('selected_weighting_scheme', 'N/A')
        available_schemes = models.get('available_weighting_schemes', [])
        lines.append(f"- **Selected Scheme**: {selected_scheme}\n")
        if available_schemes:
            lines.append("- **Available Schemes**:\n")
            for scheme in available_schemes:
                lines.append(f"  - {scheme}\n")
        lines.append("\n")

        report_path = outcomes_dir / f"layer3_meta_report_{ts}.md"
        report_path.write_text("".join(lines))
        tprint_success(f"✅ Meta-report saved to {report_path}")

    except Exception as e:
        tprint_error(f"❌ Failed to generate Layer 3 meta-report: {e}")

def _generate_feature_importance_report(
    models: Dict[str, Any],
    meta_features: List[str],
    outcomes_dir: Path,
    ts: str
) -> None:
    """
    Generate feature importance report.
    """
    try:
        tprint_info("📈 Analyzing Feature Importances...")
        
        importance_data = []
        
        # Extract feature importances from models
        if 'alpha_models' in models:
            tprint_info(f"📊 Analyzing {len(models['alpha_models'])} alpha models...")
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
            tprint_info(f"🎯 Analyzing {len(models['prob_models'])} probability models...")
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
            tprint_info(f"📊 Total importance entries: {len(importance_df)}")
            
            # Aggregate by feature
            feature_summary = importance_df.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
            feature_summary.columns = ['feature', 'mean_importance', 'std_importance', 'model_count']
            feature_summary = feature_summary.sort_values('mean_importance', ascending=False)
            
            tprint_info(f"📈 Features with importance data: {len(feature_summary)}")
            
            # Save detailed importance
            importance_df.to_csv(outcomes_dir / f"layer3_feature_importance_detailed_{ts}.csv", index=False)
            feature_summary.to_csv(outcomes_dir / f"layer3_feature_importance_summary_{ts}.csv", index=False)
            
            # Show top features
            top_features = feature_summary.head(10)
            tprint_success("✅ Top 10 features by mean importance:")
            for _, row in top_features.iterrows():
                tprint_info(f"   - {row['feature']}: {row['mean_importance']:.4f} (±{row['std_importance']:.4f})")
            
            tprint_success(f"✅ Feature importance reports saved")
        else:
            tprint_warning("⚠️ No feature importance data available")

    except Exception as e:
        tprint_error(f"❌ Failed to generate feature importance report: {e}")

def _generate_calibration_diagnostics(
    df: pd.DataFrame,
    target_col: str,
    outcomes_dir: Path,
    ts: str
) -> None:
    """
    Generate calibration diagnostics plots and metrics.
    """
    try:
        if 'meta_prob' not in df.columns or target_col not in df.columns:
            tprint_warning("⚠️ Missing required columns for calibration diagnostics")
            return
        
        tprint_info("🎯 Creating Calibration Diagnostics...")
        
        y_true = (df[target_col] > 0.5).astype(int).values
        y_prob = df['meta_prob'].values
        
        # Calculate calibration metrics
        ece = _fast_expected_calibration_error(y_true, y_prob)
        tprint_info(f"🎯 Expected Calibration Error: {ece:.4f}")
        
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
        
        prob_stats = f"mean={y_prob.mean():.3f}, std={y_prob.std():.3f}"
        tprint_info(f"📊 Probability distribution: {prob_stats}")
        
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
                ece = _fast_expected_calibration_error(y_true[mask], y_prob[mask])
                ece_by_threshold.append(ece)
            else:
                ece_by_threshold.append(np.nan)
        
        axes[1, 1].plot(np.linspace(0.5, 0.95, 10), ece_by_threshold, marker='o', linewidth=2)
        axes[1, 1].set_xlabel('Probability Threshold')
        axes[1, 1].set_ylabel('ECE')
        axes[1, 1].set_title('ECE by Probability Threshold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = outcomes_dir / f"layer3_calibration_diagnostics_{ts}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        tprint_success(f"✅ Calibration diagnostics plot saved to {plot_path}")

    except Exception as e:
        tprint_error(f"❌ Failed to generate calibration diagnostics: {e}")

def _generate_performance_summary(
    df: pd.DataFrame,
    models: Dict[str, Any],
    target_col: str,
    outcomes_dir: Path,
    ts: str
) -> None:
    """
    Generate performance summary statistics.
    """
    try:
        tprint_info("📊 Generating Performance Summary...")
        
        summary_stats = {}
        
        # Alpha performance
        if 'meta_alpha' in df.columns and target_col in df.columns:
            from scipy.stats import spearmanr
            ic, _ = spearmanr(df[target_col], df['meta_alpha'])
            summary_stats['alpha_ic'] = ic
            tprint_info(f"📈 Alpha IC: {ic:.4f}")
        
        # Probability performance
        if 'meta_prob' in df.columns and target_col in df.columns:
            from sklearn.metrics import roc_auc_score, log_loss
            y_true = (df[target_col] > 0.5).astype(int)
            y_prob = df['meta_prob']
            
            summary_stats['prob_auc'] = roc_auc_score(y_true, y_prob)
            summary_stats['prob_logloss'] = log_loss(y_true, y_prob)
            summary_stats['prob_ece'] = _fast_expected_calibration_error(y_true, y_prob)
            
            tprint_info(f"🎯 Probability AUC: {summary_stats['prob_auc']:.4f}")
            tprint_info(f"🎯 Probability LogLoss: {summary_stats['prob_logloss']:.4f}")
            tprint_info(f"🎯 Probability ECE: {summary_stats['prob_ece']:.4f}")
        
        # Data statistics
        summary_stats['total_samples'] = len(df)
        summary_stats['positive_rate'] = (df[target_col] > 0.5).mean() if target_col in df.columns else np.nan
        summary_stats['meta_prob_mean'] = df['meta_prob'].mean() if 'meta_prob' in df.columns else np.nan
        summary_stats['meta_prob_std'] = df['meta_prob'].std() if 'meta_prob' in df.columns else np.nan
        
        tprint_info(f"📊 Total samples: {summary_stats['total_samples']}")
        tprint_info(f"📊 Positive rate: {summary_stats['positive_rate']:.1%}")
        tprint_info(f"📊 Meta prob mean: {summary_stats['meta_prob_mean']:.3f}")
        
        # Save summary
        summary_df = pd.DataFrame([summary_stats])
        summary_path = outcomes_dir / f"layer3_performance_summary_{ts}.csv"
        summary_df.to_csv(summary_path, index=False)
        
        tprint_success(f"✅ Performance summary saved to {summary_path}")

    except Exception as e:
        tprint_error(f"❌ Failed to generate performance summary: {e}")

def _fast_expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Fast Expected Calibration Error calculation.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Determine samples in bin
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Calculate accuracy in bin
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            # Add to ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece

def _safe_to_markdown(df: pd.DataFrame) -> str:
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
