"""
Comprehensive Reporter for SR Quality Model

Generates detailed .md and .csv reports with all metrics:
- Financial metrics (global and per-level)
- Model metrics (HPO outcomes, CV scores)
- Model quality metrics (overfitting, calibration, etc.)
- Feature importance (LGBM, permutation, SHAP)
- Individual level quality scores
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import json
import logging


class ComprehensiveReporter:
    """Generate comprehensive reports for SR quality model training."""
    
    def __init__(self, output_dir: str = 'outcomes'):
        """Initialize reporter.
        
        Args:
            output_dir: Directory to save reports (default: 'outcomes')
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Generate timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def generate_complete_report(self, 
                                training_data: pd.DataFrame,
                                model,
                                training_metrics: Dict,
                                quality_assessment: Dict,
                                importance_analysis: Dict,
                                timeframe: str,
                                symbol: str) -> Dict[str, str]:
        """Generate complete training report with all metrics.
        
        Args:
            training_data: Full training dataset
            model: Trained model
            training_metrics: Training/CV metrics
            quality_assessment: Model quality assessment
            importance_analysis: Feature importance analysis
            timeframe: Timeframe used
            symbol: Symbol used
            
        Returns:
            Dict with paths to generated files
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("📊 GENERATING COMPREHENSIVE REPORT")
        self.logger.info("="*80)
        
        # Generate filename base
        filename_base = f"sr_quality_report_{symbol}_{timeframe}_{self.timestamp}"
        
        # 1. Generate markdown report
        md_path = self.output_dir / f"{filename_base}.md"
        self._generate_markdown_report(
            md_path, training_data, model, training_metrics,
            quality_assessment, importance_analysis, timeframe, symbol
        )
        
        # 2. Generate CSV with level-by-level metrics
        csv_path = self.output_dir / f"{filename_base}.csv"
        self._generate_level_csv(csv_path, training_data, model)
        
        # 3. Generate JSON with all metrics (for programmatic access)
        json_path = self.output_dir / f"{filename_base}.json"
        self._generate_json_report(
            json_path, training_metrics, quality_assessment, importance_analysis
        )
        
        self.logger.info(f"\n✅ Reports generated:")
        self.logger.info(f"   📄 Markdown: {md_path}")
        self.logger.info(f"   📊 CSV: {csv_path}")
        self.logger.info(f"   📋 JSON: {json_path}")
        
        return {
            'markdown': str(md_path),
            'csv': str(csv_path),
            'json': str(json_path)
        }
    
    def _generate_markdown_report(self, output_path: Path, training_data: pd.DataFrame,
                                  model, training_metrics: Dict, quality_assessment: Dict,
                                  importance_analysis: Dict, timeframe: str, symbol: str):
        """Generate comprehensive markdown report."""
        
        lines = []
        
        # Header
        lines.append(f"# SR Quality Model Training Report")
        lines.append(f"")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Symbol:** {symbol}")
        lines.append(f"**Timeframe:** {timeframe}")
        lines.append(f"**Total Samples:** {len(training_data):,}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        
        # Executive Summary
        lines.extend(self._section_executive_summary(training_metrics, quality_assessment))
        
        # Model Performance Metrics
        lines.extend(self._section_model_performance(training_metrics))
        
        # Model Quality Metrics
        lines.extend(self._section_model_quality(quality_assessment))
        
        # Financial Metrics
        lines.extend(self._section_financial_metrics(training_data))
        
        # Feature Importance
        lines.extend(self._section_feature_importance(importance_analysis))
        
        # Per-Level Analysis
        lines.extend(self._section_per_level_analysis(training_data, model))
        
        # Production Readiness
        lines.extend(self._section_production_readiness(quality_assessment))
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
        
        self.logger.info(f"   ✅ Markdown report: {len(lines)} lines")
    
    def _section_executive_summary(self, training_metrics: Dict, 
                                   quality_assessment: Dict) -> List[str]:
        """Generate executive summary section."""
        lines = []
        lines.append("## 📊 Executive Summary")
        lines.append("")
        
        # Overall health
        health_score = quality_assessment.get('health_score', 0)
        production_ready = quality_assessment.get('production_ready', False)
        
        if production_ready:
            lines.append(f"**Status:** ✅ **PRODUCTION READY** (Health Score: {health_score:.2f}/1.00)")
        else:
            lines.append(f"**Status:** ⚠️  **NEEDS IMPROVEMENT** (Health Score: {health_score:.2f}/1.00)")
        
        lines.append("")
        
        # Key metrics
        avg_metrics = training_metrics.get('avg_metrics', {})
        lines.append(f"### Key Metrics:")
        lines.append(f"")
        lines.append(f"| Metric | Value | Status |")
        lines.append(f"|--------|-------|--------|")
        lines.append(f"| **Validation R²** | {avg_metrics.get('avg_val_r2', 0):.3f} | {'✅' if avg_metrics.get('avg_val_r2', 0) > 0.5 else '⚠️'} |")
        lines.append(f"| **Validation RMSE** | {avg_metrics.get('avg_val_rmse', 0):.4f} | {'✅' if avg_metrics.get('avg_val_rmse', 0) < 0.20 else '⚠️'} |")
        
        # Overfitting
        overfitting = quality_assessment.get('overfitting', {})
        of_status = overfitting.get('status', '❌')
        of_severity = overfitting.get('severity', 'unknown')
        lines.append(f"| **Overfitting** | {of_severity} | {of_status} |")
        
        # Calibration
        calibration = quality_assessment.get('calibration', {})
        ece = calibration.get('expected_calibration_error', 1.0)
        lines.append(f"| **Calibration (ECE)** | {ece:.4f} | {'✅' if ece < 0.05 else '⚠️'} |")
        
        # Health score
        lines.append(f"| **Overall Health** | {health_score:.2f}/1.00 | {'✅' if health_score >= 0.70 else '❌'} |")
        
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        
        return lines
    
    def _section_model_performance(self, training_metrics: Dict) -> List[str]:
        """Generate model performance section."""
        lines = []
        lines.append("## 📈 Model Performance Metrics")
        lines.append("")
        
        # Cross-validation scores
        cv_scores = training_metrics.get('cv_scores', [])
        if cv_scores:
            lines.append("### Cross-Validation Results:")
            lines.append("")
            lines.append("| Fold | Train RMSE | Val RMSE | Train R² | Val R² | Val MAE | Boost Rounds |")
            lines.append("|------|------------|----------|----------|--------|---------|--------------|")
            
            for score in cv_scores:
                lines.append(
                    f"| {score['fold']+1} | "
                    f"{score['train_rmse']:.4f} | {score['val_rmse']:.4f} | "
                    f"{score['train_r2']:.3f} | {score['val_r2']:.3f} | "
                    f"{score['val_mae']:.4f} | {score['num_boost_rounds']} |"
                )
            
            lines.append("")
            
            # Average scores
            avg_metrics = training_metrics.get('avg_metrics', {})
            lines.append("### Average Performance:")
            lines.append("")
            lines.append(f"- **Validation RMSE:** {avg_metrics.get('avg_val_rmse', 0):.4f} ± {avg_metrics.get('std_val_rmse', 0):.4f}")
            lines.append(f"- **Validation R²:** {avg_metrics.get('avg_val_r2', 0):.3f} ± {avg_metrics.get('std_val_r2', 0):.3f}")
            lines.append(f"- **Validation MAE:** {avg_metrics.get('avg_val_mae', 0):.4f}")
            lines.append(f"")
        
        # HPO results
        if 'hpo_best_params' in training_metrics:
            lines.append("### Hyperparameter Optimization:")
            lines.append("")
            lines.append("**Best Parameters Found:**")
            lines.append("```python")
            for param, value in training_metrics['hpo_best_params'].items():
                lines.append(f"{param}: {value}")
            lines.append("```")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _section_model_quality(self, quality_assessment: Dict) -> List[str]:
        """Generate model quality metrics section."""
        lines = []
        lines.append("## 🔬 Model Quality Metrics")
        lines.append("")
        
        # Overfitting analysis
        lines.append("### 1. Overfitting Detection")
        lines.append("")
        overfitting = quality_assessment.get('overfitting', {})
        
        lines.append("| Metric | Train | Validation | Gap | Status |")
        lines.append("|--------|-------|------------|-----|--------|")
        
        # Would need train metrics - adding placeholder logic
        rmse_gap = overfitting.get('rmse_gap', 0)
        r2_gap = overfitting.get('r2_gap', 0)
        severity = overfitting.get('severity', 'unknown')
        status = overfitting.get('status', '❌')
        
        lines.append(f"| RMSE | - | - | {rmse_gap:+.4f} | {status} |")
        lines.append(f"| R² | - | - | {r2_gap:+.4f} | {status} |")
        lines.append("")
        lines.append(f"**Severity:** {severity}")
        lines.append(f"**Recommendation:** {overfitting.get('recommendation', 'N/A')}")
        lines.append("")
        
        # Calibration
        lines.append("### 2. Calibration Analysis")
        lines.append("")
        calibration = quality_assessment.get('calibration', {})
        
        lines.append(f"- **Expected Calibration Error (ECE):** {calibration.get('expected_calibration_error', 0):.4f}")
        lines.append(f"- **Mean Calibration Error:** {calibration.get('mean_calibration_error', 0):.4f}")
        lines.append(f"- **Status:** {'✅ Well calibrated' if calibration.get('well_calibrated', False) else '⚠️  Needs calibration'}")
        lines.append("")
        
        # Calibration by bin
        cal_bins = calibration.get('calibration_by_bin', [])
        if cal_bins:
            lines.append("**Calibration by Prediction Range:**")
            lines.append("")
            lines.append("| Predicted Range | Count | Predicted Mean | Actual Mean | Error | Status |")
            lines.append("|-----------------|-------|----------------|-------------|-------|--------|")
            
            for b in cal_bins:
                status = '✅' if b['well_calibrated'] else '❌'
                lines.append(
                    f"| {b['bin_range'][0]:.1f}-{b['bin_range'][1]:.1f} | "
                    f"{b['count']} | {b['predicted_mean']:.3f} | {b['actual_mean']:.3f} | "
                    f"{b['calibration_error']:.3f} | {status} |"
                )
            lines.append("")
        
        # Prediction distribution
        lines.append("### 3. Prediction Distribution")
        lines.append("")
        pred_dist = quality_assessment.get('prediction_distribution', {})
        pred_stats = pred_dist.get('prediction_stats', {})
        true_stats = pred_dist.get('true_stats', {})
        
        lines.append("| Metric | Predictions | True Values | Ratio |")
        lines.append("|--------|-------------|-------------|-------|")
        lines.append(f"| Mean | {pred_stats.get('mean', 0):.4f} | {true_stats.get('mean', 0):.4f} | - |")
        lines.append(f"| Std | {pred_stats.get('std', 0):.4f} | {true_stats.get('std', 0):.4f} | {pred_dist.get('variance_ratio', 0):.2f} |")
        lines.append(f"| Range | {pred_stats.get('range', 0):.4f} | {true_stats.get('range', 0):.4f} | {pred_dist.get('range_coverage', 0):.2f} |")
        lines.append("")
        
        # Health issues
        health_issues = pred_dist.get('health_issues', [])
        if health_issues:
            lines.append("**⚠️  Distribution Issues:**")
            for issue in health_issues:
                lines.append(f"- {issue}")
            lines.append("")
        else:
            lines.append("**✅ Distribution is healthy**")
            lines.append("")
        
        # Feature stability
        if 'feature_stability' in quality_assessment:
            lines.append("### 4. Feature Importance Stability")
            lines.append("")
            feat_stab = quality_assessment['feature_stability']
            
            lines.append(f"- **Top 10 Stable:** {feat_stab['top_10_stable_count']}/10")
            lines.append(f"- **Mean CV:** {feat_stab['top_10_cv_mean']:.3f}")
            lines.append(f"- **Status:** {'✅ Stable' if feat_stab['top_10_stable'] else '⚠️  Unstable'}")
            lines.append("")
        
        # Error analysis by quality bin
        if 'error_by_bin' in quality_assessment:
            lines.append("### 5. Error Analysis by Quality Bin")
            lines.append("")
            lines.append("| Quality Bin | Samples | MAE | RMSE | Bias | R² |")
            lines.append("|-------------|---------|-----|------|------|-----|")
            
            for bin_name, stats in quality_assessment['error_by_bin'].items():
                lines.append(
                    f"| {bin_name} | {stats['count']} | {stats['mae']:.4f} | "
                    f"{stats['rmse']:.4f} | {stats['bias']:+.4f} | {stats['r2']:.3f} |"
                )
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _section_financial_metrics(self, training_data: pd.DataFrame) -> List[str]:
        """Generate financial metrics section."""
        lines = []
        lines.append("## 💰 Financial Metrics")
        lines.append("")
        
        # Global financial metrics
        lines.append("### Global Statistics:")
        lines.append("")
        
        # Trade profit analysis
        if 'trade_profit' in training_data.columns:
            trade_profits = training_data['trade_profit']
            win_rate = (trade_profits > 0).sum() / len(trade_profits) * 100
            
            lines.append(f"**Trade Performance:**")
            lines.append(f"- Mean Profit: {trade_profits.mean():.4f}")
            lines.append(f"- Win Rate: {win_rate:.1f}%")
            lines.append(f"- Best Trade: {trade_profits.max():.4f}")
            lines.append(f"- Worst Trade: {trade_profits.min():.4f}")
            lines.append(f"- Profit Std: {trade_profits.std():.4f}")
            lines.append("")
        
        # Quality score distribution
        if 'quality_score' in training_data.columns:
            quality = training_data['quality_score']
            
            lines.append(f"**Quality Distribution:**")
            lines.append(f"- Mean: {quality.mean():.4f}")
            lines.append(f"- Median: {quality.median():.4f}")
            lines.append(f"- Std: {quality.std():.4f}")
            lines.append(f"- Range: [{quality.min():.4f}, {quality.max():.4f}]")
            lines.append(f"- IQR: {quality.quantile(0.75) - quality.quantile(0.25):.4f}")
            lines.append("")
        
        # Component performance
        lines.append(f"**Component Metrics:**")
        lines.append("")
        lines.append("| Component | Mean | Median | Std | Min | Max |")
        lines.append("|-----------|------|--------|-----|-----|-----|")
        
        components = ['bounce_strength', 'hold_strength', 'trade_profit', 
                     'rejection_speed', 'volume_quality']
        for comp in components:
            if comp in training_data.columns:
                c = training_data[comp]
                lines.append(f"| {comp} | {c.mean():.4f} | {c.median():.4f} | "
                           f"{c.std():.4f} | {c.min():.4f} | {c.max():.4f} |")
        
        lines.append("")
        
        # Per-level analysis (Top 5, Mid 5, Bottom 5)
        lines.extend(self._subsection_per_level_financial(training_data))
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _subsection_per_level_financial(self, training_data: pd.DataFrame) -> List[str]:
        """Per-level financial analysis for top/mid/bottom levels."""
        lines = []
        lines.append("### Per-Level Analysis:")
        lines.append("")
        
        if 'quality_score' not in training_data.columns:
            lines.append("*Quality score not available*")
            lines.append("")
            return lines
        
        # Sort by quality
        sorted_data = training_data.sort_values('quality_score', ascending=False)
        n = len(sorted_data)
        
        # Top 5
        lines.append("#### Top 5 Levels (Highest Quality):")
        lines.append("")
        lines.extend(self._format_level_table(sorted_data.head(5)))
        
        # Middle 5
        mid_start = (n // 2) - 2
        mid_end = mid_start + 5
        lines.append("#### Middle 5 Levels (Average Quality):")
        lines.append("")
        lines.extend(self._format_level_table(sorted_data.iloc[mid_start:mid_end]))
        
        # Bottom 5
        lines.append("#### Bottom 5 Levels (Lowest Quality):")
        lines.append("")
        lines.extend(self._format_level_table(sorted_data.tail(5)))
        
        return lines
    
    def _format_level_table(self, levels_df: pd.DataFrame) -> List[str]:
        """Format levels into markdown table."""
        lines = []
        
        lines.append("| # | Quality | Bounce | Hold | Trade | Rejection | Volume |")
        lines.append("|---|---------|--------|------|-------|-----------|--------|")
        
        for idx, row in levels_df.iterrows():
            lines.append(
                f"| {idx+1} | "
                f"{row.get('quality_score', 0):.3f} | "
                f"{row.get('bounce_strength', 0):.3f} | "
                f"{row.get('hold_strength', 0):.3f} | "
                f"{row.get('trade_profit', 0):.3f} | "
                f"{row.get('rejection_speed', 0):.3f} | "
                f"{row.get('volume_quality', 0):.3f} |"
            )
        
        lines.append("")
        
        return lines
    
    def _section_feature_importance(self, importance_analysis: Dict) -> List[str]:
        """Generate feature importance section."""
        lines = []
        lines.append("## 🎯 Feature Importance Analysis")
        lines.append("")
        
        combined = importance_analysis.get('combined_ranking')
        shap_available = importance_analysis.get('shap_available', False)
        
        if combined is not None:
            lines.append("### Top 20 Features (Combined Ranking):")
            lines.append("")
            
            if shap_available:
                lines.append("| Rank | Feature | LGBM Gain | Permutation | SHAP | Avg Rank |")
                lines.append("|------|---------|-----------|-------------|------|----------|")
                
                for idx, row in combined.head(20).iterrows():
                    lines.append(
                        f"| {idx+1} | {row['feature'].replace('feature_', '')} | "
                        f"{row.get('lgbm_gain_rank', 0):.0f} | "
                        f"{row.get('permutation_rank', 0):.0f} | "
                        f"{row.get('shap_rank', 0):.0f} | "
                        f"{row.get('avg_rank', 0):.1f} |"
                    )
            else:
                lines.append("| Rank | Feature | LGBM Gain | Permutation | Avg Rank |")
                lines.append("|------|---------|-----------|-------------|----------|")
                
                for idx, row in combined.head(20).iterrows():
                    lines.append(
                        f"| {idx+1} | {row['feature'].replace('feature_', '')} | "
                        f"{row.get('lgbm_gain_rank', 0):.0f} | "
                        f"{row.get('permutation_rank', 0):.0f} | "
                        f"{row.get('avg_rank', 0):.1f} |"
                    )
            
            lines.append("")
            
            # Interpretation
            lines.append("### Key Insights:")
            lines.append("")
            top_features = combined.head(5)['feature'].tolist()
            lines.append(f"**Most Important Features:**")
            for i, feat in enumerate(top_features, 1):
                lines.append(f"{i}. `{feat.replace('feature_', '')}`")
            lines.append("")
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _section_per_level_analysis(self, training_data: pd.DataFrame, model) -> List[str]:
        """Generate detailed per-level analysis."""
        lines = []
        lines.append("## 📋 Detailed Level Analysis")
        lines.append("")
        
        # Sort by predicted quality
        feature_cols = [c for c in training_data.columns if c.startswith('feature_')]
        if feature_cols and 'quality_score' in training_data.columns:
            X = training_data[feature_cols].fillna(0.0)
            y_pred = model.predict(X)
            
            analysis_df = training_data.copy()
            analysis_df['predicted_quality'] = y_pred
            analysis_df['prediction_error'] = y_pred - analysis_df['quality_score']
            analysis_df = analysis_df.sort_values('predicted_quality', ascending=False)
            
            # Summary statistics
            lines.append(f"### Prediction Accuracy:")
            lines.append(f"")
            lines.append(f"- Mean Absolute Error: {np.abs(analysis_df['prediction_error']).mean():.4f}")
            lines.append(f"- RMSE: {np.sqrt((analysis_df['prediction_error']**2).mean()):.4f}")
            lines.append(f"- Correlation: {analysis_df['predicted_quality'].corr(analysis_df['quality_score']):.3f}")
            lines.append("")
            
            # Largest errors
            largest_errors = analysis_df.nlargest(5, 'prediction_error')
            lines.append(f"### Top 5 Over-Predictions (Model too optimistic):")
            lines.append("")
            lines.extend(self._format_error_table(largest_errors))
            
            smallest_errors = analysis_df.nsmallest(5, 'prediction_error')
            lines.append(f"### Top 5 Under-Predictions (Model too pessimistic):")
            lines.append("")
            lines.extend(self._format_error_table(smallest_errors))
        
        lines.append("---")
        lines.append("")
        
        return lines
    
    def _format_error_table(self, levels_df: pd.DataFrame) -> List[str]:
        """Format error analysis table."""
        lines = []
        lines.append("| True | Predicted | Error | Bounce | Hold | Trade |")
        lines.append("|------|-----------|-------|--------|------|-------|")
        
        for idx, row in levels_df.iterrows():
            lines.append(
                f"| {row.get('quality_score', 0):.3f} | "
                f"{row.get('predicted_quality', 0):.3f} | "
                f"{row.get('prediction_error', 0):+.3f} | "
                f"{row.get('bounce_strength', 0):.3f} | "
                f"{row.get('hold_strength', 0):.3f} | "
                f"{row.get('trade_profit', 0):.3f} |"
            )
        
        lines.append("")
        return lines
    
    def _section_production_readiness(self, quality_assessment: Dict) -> List[str]:
        """Generate production readiness section."""
        lines = []
        lines.append("## 🚀 Production Readiness")
        lines.append("")
        
        health_score = quality_assessment.get('health_score', 0)
        production_ready = quality_assessment.get('production_ready', False)
        
        lines.append(f"### Overall Assessment:")
        lines.append("")
        lines.append(f"**Health Score:** {health_score:.2f}/1.00")
        lines.append("")
        
        # Criteria checklist
        lines.append("### Criteria Checklist:")
        lines.append("")
        lines.append("| Criterion | Status | Details |")
        lines.append("|-----------|--------|---------|")
        
        # Overfitting
        overfitting = quality_assessment.get('overfitting', {})
        of_ok = overfitting.get('severity') in ['none', 'mild']
        lines.append(f"| No significant overfitting | {'✅' if of_ok else '❌'} | {overfitting.get('severity', 'unknown')} |")
        
        # Calibration
        calibration = quality_assessment.get('calibration', {})
        cal_ok = calibration.get('well_calibrated', False)
        ece = calibration.get('expected_calibration_error', 1.0)
        lines.append(f"| Well calibrated | {'✅' if cal_ok else '❌'} | ECE={ece:.4f} |")
        
        # Distribution
        pred_dist = quality_assessment.get('prediction_distribution', {})
        dist_ok = pred_dist.get('healthy', False)
        lines.append(f"| Healthy predictions | {'✅' if dist_ok else '❌'} | {len(pred_dist.get('health_issues', []))} issues |")
        
        # Feature stability
        if 'feature_stability' in quality_assessment:
            feat_ok = quality_assessment['feature_stability']['top_10_stable']
            lines.append(f"| Stable features | {'✅' if feat_ok else '❌'} | {quality_assessment['feature_stability']['top_10_stable_count']}/10 stable |")
        
        # CV stability
        cv_ok = quality_assessment.get('overfitting', {}).get('cv_stable', True)
        lines.append(f"| Stable CV | {'✅' if cv_ok else '❌'} | - |")
        
        lines.append("")
        
        # Final verdict
        if production_ready:
            lines.append(f"### ✅ **PRODUCTION READY**")
            lines.append("")
            lines.append(f"Model meets all criteria for production deployment.")
        else:
            lines.append(f"### ⚠️  **NEEDS IMPROVEMENT**")
            lines.append("")
            lines.append(f"Address the issues above before production deployment.")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Recommendations
        lines.append("### Recommendations:")
        lines.append("")
        
        recommendations = []
        
        if not of_ok:
            recommendations.append(overfitting.get('recommendation', 'Reduce overfitting'))
        
        if not cal_ok:
            recommendations.append("Improve calibration with isotonic regression or recalibration")
        
        if not dist_ok:
            for issue in pred_dist.get('health_issues', []):
                recommendations.append(f"Fix prediction distribution: {issue}")
        
        if recommendations:
            for rec in recommendations:
                lines.append(f"- {rec}")
        else:
            lines.append(f"- No major issues detected - ready for deployment!")
        
        lines.append("")
        
        return lines
    
    def _generate_level_csv(self, output_path: Path, training_data: pd.DataFrame, model):
        """Generate CSV with each level and its 11 quality metrics."""
        
        # Get predictions
        feature_cols = [c for c in training_data.columns if c.startswith('feature_')]
        
        if not feature_cols:
            self.logger.warning("No features found, skipping CSV generation")
            return
        
        X = training_data[feature_cols].fillna(0.0)
        y_pred = model.predict(X)
        
        # Create output dataframe
        output_df = pd.DataFrame()
        
        # Metadata
        for col in ['date', 'symbol', 'timeframe']:
            if col in training_data.columns:
                output_df[col] = training_data[col]
        
        # 11 Quality metrics
        quality_metrics = [
            'bounce_strength',
            'max_bounce_strength',
            'hold_strength',
            'trade_profit',
            'rejection_speed',
            'volume_quality',
            'quality_score',
            'bounce_quality',
            'hold_quality',
            'trade_quality',
            'speed_quality',
            'volume_confirmation_quality'
        ]
        
        for metric in quality_metrics:
            if metric in training_data.columns:
                output_df[metric] = training_data[metric]
            else:
                output_df[metric] = 0.0  # Placeholder for missing metrics
        
        # Add prediction
        output_df['predicted_quality'] = y_pred
        output_df['prediction_error'] = y_pred - training_data['quality_score']
        
        # Add key features for context
        key_features = ['feature_strength', 'feature_prominence', 'feature_touch_count', 
                       'feature_distance_to_current_pct', 'feature_weighted_touch_count']
        for feat in key_features:
            if feat in training_data.columns:
                output_df[feat] = training_data[feat]
        
        # Save
        output_df.to_csv(output_path, index=False)
        
        self.logger.info(f"   ✅ Level CSV: {len(output_df)} levels, {len(output_df.columns)} columns")
    
    def _generate_json_report(self, output_path: Path, training_metrics: Dict,
                             quality_assessment: Dict, importance_analysis: Dict):
        """Generate JSON report for programmatic access."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_metrics': self._make_serializable(training_metrics),
            'quality_assessment': self._make_serializable(quality_assessment),
            'importance_summary': {
                'shap_available': importance_analysis.get('shap_available', False),
                'top_10_features': importance_analysis.get('combined_ranking').head(10)['feature'].tolist()
                                  if 'combined_ranking' in importance_analysis else []
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"   ✅ JSON report: {output_path.name}")
    
    def _make_serializable(self, obj):
        """Convert to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif pd.isna(obj):
            return None
        else:
            return obj

