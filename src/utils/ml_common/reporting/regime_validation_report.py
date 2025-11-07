"""
Markdown Report Generator for Regime Walk-Forward Validation Results

Generates comprehensive markdown reports with:
- Model rankings with confidence intervals
- Temporal metrics (MEL, TR, SFPR, etc.)
- Fold-by-fold performance
- Summary statistics
"""

from typing import Dict, Any, Optional
from datetime import datetime
from src.utils.ml_common.validation.regime_walk_forward_validator import RegimeValidationResult


def generate_validation_markdown_report(
    wf_result: RegimeValidationResult,
    component_name: str = "Regime Training",
    output_path: Optional[str] = None
) -> str:
    """
    Generate a comprehensive markdown report from walk-forward validation results.

    Args:
        wf_result: Walk-forward validation result
        component_name: Name of the component (e.g., "Base Models", "Ensemble")
        output_path: Optional path to save the report

    Returns:
        Markdown report as a string
    """
    report_lines = []

    # Header
    report_lines.append(f"# {component_name} - Walk-Forward Validation Report")
    report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"\n**Validation Folds:** {wf_result.metadata.get('n_folds_completed', 0)}/{wf_result.metadata.get('n_folds_attempted', 0)}")
    report_lines.append(f"\n**Models Evaluated:** {wf_result.metadata.get('n_models_evaluated', 0)}")
    report_lines.append("\n---\n")

    # Model Rankings
    report_lines.append("## Model Rankings (by OOS Performance)\n")
    report_lines.append("| Rank | Model | Composite Score | Accuracy | F1-Score | MEL | SFPR |")
    report_lines.append("|------|-------|----------------|----------|----------|-----|------|")

    for i, ranking in enumerate(wf_result.model_rankings, 1):
        acc_ci = ranking.get('accuracy_ci', (0, 0))
        f1_ci = ranking.get('f1_ci', (0, 0))

        report_lines.append(
            f"| {i} | `{ranking['model_name']}` | "
            f"{ranking['composite_score']:.4f} | "
            f"{ranking['accuracy']:.4f} [{acc_ci[0]:.4f}, {acc_ci[1]:.4f}] | "
            f"{ranking['f1_score']:.4f} [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}] | "
            f"{ranking.get('mel', 0):.2f} | "
            f"{ranking.get('sfpr', 0):.4f} |"
        )

    report_lines.append("\n")

    # Detailed Metrics
    report_lines.append("## Detailed Performance Metrics\n")

    # Accuracy
    report_lines.append("### Accuracy\n")
    acc = wf_result.accuracy
    report_lines.append(f"- **Mean:** {acc.get('mean', 0):.4f}")
    report_lines.append(f"- **Std:** {acc.get('std', 0):.4f}")
    report_lines.append(f"- **95% CI:** [{acc.get('ci_lower', 0):.4f}, {acc.get('ci_upper', 0):.4f}]")
    report_lines.append(f"- **Range:** [{acc.get('min', 0):.4f}, {acc.get('max', 0):.4f}]")
    report_lines.append(f"- **Folds:** {acc.get('n_folds', 0)}\n")

    # F1-Score
    report_lines.append("### F1-Score (Weighted)\n")
    f1 = wf_result.f1_score
    report_lines.append(f"- **Mean:** {f1.get('mean', 0):.4f}")
    report_lines.append(f"- **Std:** {f1.get('std', 0):.4f}")
    report_lines.append(f"- **95% CI:** [{f1.get('ci_lower', 0):.4f}, {f1.get('ci_upper', 0):.4f}]")
    report_lines.append(f"- **Range:** [{f1.get('min', 0):.4f}, {f1.get('max', 0):.4f}]")
    report_lines.append(f"- **Folds:** {f1.get('n_folds', 0)}\n")

    # Precision
    report_lines.append("### Precision (Weighted)\n")
    precision = wf_result.precision
    report_lines.append(f"- **Mean:** {precision.get('mean', 0):.4f}")
    report_lines.append(f"- **Std:** {precision.get('std', 0):.4f}")
    report_lines.append(f"- **95% CI:** [{precision.get('ci_lower', 0):.4f}, {precision.get('ci_upper', 0):.4f}]")
    report_lines.append(f"- **Range:** [{precision.get('min', 0):.4f}, {precision.get('max', 0):.4f}]\n")

    # Recall
    report_lines.append("### Recall (Weighted)\n")
    recall = wf_result.recall
    report_lines.append(f"- **Mean:** {recall.get('mean', 0):.4f}")
    report_lines.append(f"- **Std:** {recall.get('std', 0):.4f}")
    report_lines.append(f"- **95% CI:** [{recall.get('ci_lower', 0):.4f}, {recall.get('ci_upper', 0):.4f}]")
    report_lines.append(f"- **Range:** [{recall.get('min', 0):.4f}, {recall.get('max', 0):.4f}]\n")

    # Temporal Metrics
    report_lines.append("## Temporal Stability Metrics\n")

    temporal = wf_result.temporal_metrics

    # MEL (Mean Episode Length)
    if 'mel' in temporal:
        mel = temporal['mel']
        report_lines.append("### Mean Episode Length (MEL)\n")
        report_lines.append(f"- **Mean:** {mel.get('mean', 0):.2f} bars")
        report_lines.append(f"- **Std:** {mel.get('std', 0):.2f}")
        report_lines.append(f"- **95% CI:** [{mel.get('ci_lower', 0):.2f}, {mel.get('ci_upper', 0):.2f}]")
        report_lines.append(f"- **Range:** [{mel.get('min', 0):.2f}, {mel.get('max', 0):.2f}]\n")
        report_lines.append("*Higher MEL indicates more stable regime predictions*\n")

    # Transition Rate
    if 'transition_rate' in temporal:
        tr = temporal['transition_rate']
        report_lines.append("### Transition Rate\n")
        report_lines.append(f"- **Mean:** {tr.get('mean', 0):.4f} transitions/bar")
        report_lines.append(f"- **Std:** {tr.get('std', 0):.4f}")
        report_lines.append(f"- **95% CI:** [{tr.get('ci_lower', 0):.4f}, {tr.get('ci_upper', 0):.4f}]\n")
        report_lines.append("*Lower transition rate indicates more stable predictions*\n")

    # SFPR (Switch False Positive Rate)
    if 'sfpr' in temporal:
        sfpr = temporal['sfpr']
        report_lines.append("### Switch False Positive Rate (SFPR)\n")
        report_lines.append(f"- **Mean:** {sfpr.get('mean', 0):.4f}")
        report_lines.append(f"- **Std:** {sfpr.get('std', 0):.4f}")
        report_lines.append(f"- **95% CI:** [{sfpr.get('ci_lower', 0):.4f}, {sfpr.get('ci_upper', 0):.4f}]\n")
        report_lines.append("*Measures fraction of regime switches that immediately revert (A→B→A)*\n")
        report_lines.append("*Lower SFPR indicates more consistent predictions*\n")

    # Confidence Metrics
    if 'mean_confidence' in temporal:
        conf = temporal['mean_confidence']
        report_lines.append("### Prediction Confidence\n")
        report_lines.append(f"- **Mean:** {conf.get('mean', 0):.4f}")
        report_lines.append(f"- **Std:** {conf.get('std', 0):.4f}")
        report_lines.append(f"- **95% CI:** [{conf.get('ci_lower', 0):.4f}, {conf.get('ci_upper', 0):.4f}]\n")

    # Fold-by-Fold Results
    if wf_result.fold_metrics:
        report_lines.append("## Fold-by-Fold Performance\n")
        report_lines.append("| Fold | Accuracy | F1-Score | MEL | SFPR | Confidence |")
        report_lines.append("|------|----------|----------|-----|------|------------|")

        for fold_result in wf_result.fold_metrics:
            fold_num = fold_result.get('fold', 'N/A')
            accuracy = fold_result.get('accuracy', 0)
            f1 = fold_result.get('f1_score', 0)
            mel = fold_result.get('mel', 0)
            sfpr = fold_result.get('sfpr', 0)
            conf = fold_result.get('mean_confidence', 0)

            report_lines.append(
                f"| {fold_num} | {accuracy:.4f} | {f1:.4f} | "
                f"{mel:.2f} | {sfpr:.4f} | {conf:.4f} |"
            )

        report_lines.append("\n")

    # Summary and Recommendations
    report_lines.append("## Summary\n")

    best_model = wf_result.model_rankings[0] if wf_result.model_rankings else None
    if best_model:
        report_lines.append(f"### Best Model: `{best_model['model_name']}`\n")
        report_lines.append(f"- **Composite Score:** {best_model['composite_score']:.4f}")
        report_lines.append(f"- **OOS Accuracy:** {best_model['accuracy']:.4f} (95% CI: [{best_model['accuracy_ci'][0]:.4f}, {best_model['accuracy_ci'][1]:.4f}])")
        report_lines.append(f"- **F1-Score:** {best_model['f1_score']:.4f}")
        report_lines.append(f"- **Stability (MEL):** {best_model.get('mel', 0):.2f} bars")
        report_lines.append(f"- **SFPR:** {best_model.get('sfpr', 0):.4f}\n")

    # Recommendations
    report_lines.append("### Recommendations\n")

    if wf_result.accuracy.get('std', 0) > 0.1:
        report_lines.append("⚠️ **High variance in accuracy across folds** - Consider:")
        report_lines.append("  - Increasing training data")
        report_lines.append("  - Simplifying model complexity")
        report_lines.append("  - Investigating regime distribution imbalance\n")

    mel_mean = wf_result.temporal_metrics.get('mel', {}).get('mean', 0)
    if mel_mean < 3:
        report_lines.append("⚠️ **Low Mean Episode Length (< 3 bars)** - Predictions are unstable:")
        report_lines.append("  - Consider temporal smoothing")
        report_lines.append("  - Increase minimum episode length penalty")
        report_lines.append("  - Review feature engineering for temporal consistency\n")

    sfpr_mean = wf_result.temporal_metrics.get('sfpr', {}).get('mean', 0)
    if sfpr_mean > 0.3:
        report_lines.append("⚠️ **High Switch False Positive Rate (> 0.3)** - Too many immediate reversals:")
        report_lines.append("  - Apply transition-aware loss function")
        report_lines.append("  - Implement regime change confirmation logic")
        report_lines.append("  - Increase switching cost penalty\n")

    if wf_result.metadata.get('n_folds_completed', 0) < wf_result.metadata.get('n_folds_attempted', 0):
        completed = wf_result.metadata.get('n_folds_completed', 0)
        attempted = wf_result.metadata.get('n_folds_attempted', 0)
        report_lines.append(f"⚠️ **Only {completed}/{attempted} folds completed** - Some folds failed:")
        report_lines.append("  - Check data sufficiency per fold")
        report_lines.append("  - Verify regime distribution in failed folds\n")

    report_lines.append("\n---\n")
    report_lines.append(f"\n*Report generated by Regime Walk-Forward Validator*")

    # Join into final report
    report = "\n".join(report_lines)

    # Save to file if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)

    return report


def generate_simple_summary(walk_forward_metrics: Dict[str, Any]) -> str:
    """
    Generate a simple summary for inclusion in larger reports.

    Args:
        walk_forward_metrics: Walk-forward validation metrics dictionary

    Returns:
        Simple markdown summary
    """
    if not walk_forward_metrics.get('validation_completed', False):
        return "**Walk-Forward Validation:** Not completed\n"

    lines = []
    lines.append("### Walk-Forward Validation Summary\n")

    accuracy = walk_forward_metrics.get('accuracy', {})
    f1_score = walk_forward_metrics.get('f1_score', {})

    lines.append(f"- **Folds Completed:** {walk_forward_metrics.get('n_folds', 'N/A')}")
    lines.append(f"- **Accuracy:** {accuracy.get('mean', 0):.4f} ± {accuracy.get('std', 0):.4f} (95% CI: [{accuracy.get('ci_lower', 0):.4f}, {accuracy.get('ci_upper', 0):.4f}])")
    lines.append(f"- **F1-Score:** {f1_score.get('mean', 0):.4f} ± {f1_score.get('std', 0):.4f}")

    temporal = walk_forward_metrics.get('temporal_metrics', {})
    if 'mel' in temporal:
        mel = temporal['mel']
        lines.append(f"- **MEL:** {mel.get('mean', 0):.2f} bars")

    if 'sfpr' in temporal:
        sfpr = temporal['sfpr']
        lines.append(f"- **SFPR:** {sfpr.get('mean', 0):.4f}")

    lines.append("\n")
    return "\n".join(lines)
