"""
Layer 4 Reporting System

Handles comprehensive reporting and diagnostics for Layer 4 position sizing and risk management.
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

class Layer4Reporter:
    """Comprehensive reporting for Layer 4 position sizing pipeline."""
    
    def __init__(self, outcomes_dir: Optional[Path] = None):
        self.outcomes_dir = outcomes_dir or Path('outcomes')
        self.outcomes_dir.mkdir(exist_ok=True, parents=True)
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_all_reports(self, oof_df: pd.DataFrame, models_dict: Dict[str, Any], 
                           fold_results: List[Dict[str, Any]], overall_metrics: Dict[str, Any],
                           config: Dict[str, Any]) -> None:
        """
        Generate all Layer 4 reports.
        
        Args:
            oof_df: Out-of-fold predictions DataFrame
            models_dict: Dictionary containing trained models and metadata
            fold_results: List of fold-specific results
            overall_metrics: Overall performance metrics
            config: Configuration dictionary
        """
        tprint_info("📊 Generating Comprehensive Layer 4 Reports...")
        
        # Generate reports
        self._generate_meta_report(oof_df, models_dict, fold_results, overall_metrics, config)
        self._generate_position_sizing_report(oof_df, overall_metrics)
        self._generate_risk_metrics_report(fold_results, overall_metrics)
        self._generate_model_performance_report(models_dict, fold_results)
        self._generate_betting_analysis_report(oof_df)
        self._generate_fold_analysis_report(fold_results)
        self._generate_feature_importance_report(models_dict)
        self._generate_performance_attribution_report(oof_df, overall_metrics)
        
        tprint_success(f"✅ Layer 4 reports saved to {self.outcomes_dir}")
    
    def _generate_meta_report(self, oof_df: pd.DataFrame, models_dict: Dict[str, Any], 
                            fold_results: List[Dict[str, Any]], overall_metrics: Dict[str, Any],
                            config: Dict[str, Any]) -> None:
        """Generate comprehensive meta-report."""
        try:
            lines = ["# Layer 4 Position Sizing & Risk Management Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            lines.append(f"Instrument: {config.get('symbol', 'UNKNOWN')} | Timeframe: {config.get('timeframe', '15m')}\n\n")

            # Executive Summary
            lines.append("## Executive Summary\n")
            lines.append(f"- **Total PnL**: {overall_metrics.get('total_pnl', 0):.4f}\n")
            lines.append(f"- **Sharpe Ratio**: {overall_metrics.get('sharpe_ratio', 0):.4f}\n")
            lines.append(f"- **Sortino Ratio**: {overall_metrics.get('sortino_ratio', 0):.4f}\n")
            lines.append(f"- **Max Drawdown**: {overall_metrics.get('max_drawdown', 0):.4f}\n")
            lines.append(f"- **Win Rate**: {overall_metrics.get('win_rate', 0):.1%}\n")
            lines.append(f"- **Profit Factor**: {overall_metrics.get('profit_factor', 0):.2f}\n")
            lines.append(f"- **Annualized Return**: {overall_metrics.get('annualized_return', 0):.2%}\n")

            # Model Configuration
            lines.append("\n## Model Configuration\n")
            if 'model_type' in models_dict:
                lines.append(f"- **Model Type**: {models_dict['model_type']}\n")
            if 'n_estimators' in models_dict:
                lines.append(f"- **Number of Estimators**: {models_dict['n_estimators']}\n")
            if 'max_features' in models_dict:
                lines.append(f"- **Max Features**: {models_dict['max_features']}\n")
            
            lines.append(f"- **Number of Folds**: {len(fold_results)}\n")
            lines.append(f"- **Total OOF Samples**: {len(oof_df)}\n")

            # Risk Engine Summary
            lines.append("\n## Risk Engine Summary\n")
            if 'risk_engine' in models_dict:
                risk_engine = models_dict['risk_engine']
                lines.append(f"- **Risk Engine Type**: {type(risk_engine).__name__}\n")
                if hasattr(risk_engine, 'max_bet_size'):
                    lines.append(f"- **Max Bet Size**: {risk_engine.max_bet_size}\n")
                if hasattr(risk_engine, 'min_bet_size'):
                    lines.append(f"- **Min Bet Size**: {risk_engine.min_bet_size}\n")

            # Training Summary
            lines.append("\n## Training Summary\n")
            valid_samples = overall_metrics.get('valid_oof_samples', 0)
            total_samples = overall_metrics.get('total_oof_samples', 1)
            coverage = overall_metrics.get('oof_coverage', 0)
            
            lines.append(f"- **Training Coverage**: {coverage:.1%}\n")
            lines.append(f"- **Valid Predictions**: {valid_samples:,}\n")
            lines.append(f"- **Mean Bet Size**: {overall_metrics.get('mean_bet_size', 0):.4f}\n")
            lines.append(f"- **Bet Size Std**: {overall_metrics.get('bet_size_std', 0):.4f}\n")

            # Save meta data
            meta_data = {
                'total_pnl': overall_metrics.get('total_pnl', 0),
                'sharpe_ratio': overall_metrics.get('sharpe_ratio', 0),
                'sortino_ratio': overall_metrics.get('sortino_ratio', 0),
                'max_drawdown': overall_metrics.get('max_drawdown', 0),
                'win_rate': overall_metrics.get('win_rate', 0),
                'profit_factor': overall_metrics.get('profit_factor', 0),
                'annualized_return': overall_metrics.get('annualized_return', 0),
                'total_folds': len(fold_results),
                'total_oof_samples': len(oof_df),
                'valid_oof_samples': valid_samples,
                'oof_coverage': coverage,
                'mean_bet_size': overall_metrics.get('mean_bet_size', 0),
                'bet_size_std': overall_metrics.get('bet_size_std', 0)
            }
            
            meta_df = pd.DataFrame([meta_data])
            meta_df.to_csv(self.outcomes_dir / f"layer4_meta_summary_{self.ts}.csv", index=False)

            report_path = self.outcomes_dir / f"layer4_meta_report_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Meta-report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate meta-report: {e}")
    
    def _generate_position_sizing_report(self, oof_df: pd.DataFrame, overall_metrics: Dict[str, Any]) -> None:
        """Generate position sizing analysis report."""
        try:
            lines = ["# Position Sizing Analysis Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if 'bet_size' in oof_df.columns:
                bet_sizes = oof_df['bet_size'].dropna()
                
                # Bet size statistics
                lines.append("## Bet Size Statistics\n")
                lines.append(f"- **Mean Bet Size**: {bet_sizes.mean():.4f}\n")
                lines.append(f"- **Median Bet Size**: {bet_sizes.median():.4f}\n")
                lines.append(f"- **Std Dev**: {bet_sizes.std():.4f}\n")
                lines.append(f"- **Min Bet Size**: {bet_sizes.min():.4f}\n")
                lines.append(f"- **Max Bet Size**: {bet_sizes.max():.4f}\n")
                lines.append(f"- **Range**: {bet_sizes.max() - bet_sizes.min():.4f}\n")

                # Bet size distribution
                lines.append("\n## Bet Size Distribution\n")
                size_bins = pd.qcut(bet_sizes, q=5, labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
                size_distribution = size_bins.value_counts().sort_index()
                
                for size, count in size_distribution.items():
                    lines.append(f"- **{size}**: {count} ({count/len(bet_sizes):.1%})\n")

                # Bet size effectiveness
                if 'realized_return' in oof_df.columns:
                    lines.append("\n## Bet Size Effectiveness\n")
                    
                    # Performance by bet size bins
                    oof_df_valid = oof_df.dropna(subset=['bet_size', 'realized_return'])
                    oof_df_valid['bet_size_bin'] = pd.qcut(oof_df_valid['bet_size'], q=5, 
                                                          labels=['Very Small', 'Small', 'Medium', 'Large', 'Very Large'])
                    
                    size_performance = oof_df_valid.groupby('bet_size_bin').agg({
                        'realized_return': ['mean', 'std', 'count'],
                        'bet_size': 'mean'
                    }).round(4)
                    
                    lines.append("### Performance by Bet Size Bins\n")
                    lines.append(self._safe_to_markdown(size_performance) + "\n\n")
                    
                    # Save data
                    size_performance.to_csv(self.outcomes_dir / f"layer4_bet_size_performance_{self.ts}.csv")
                    size_distribution.to_csv(self.outcomes_dir / f"layer4_bet_size_distribution_{self.ts}.csv")

            report_path = self.outcomes_dir / f"layer4_position_sizing_report_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Position sizing report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate position sizing report: {e}")
    
    def _generate_risk_metrics_report(self, fold_results: List[Dict[str, Any]], overall_metrics: Dict[str, Any]) -> None:
        """Generate risk metrics analysis report."""
        try:
            lines = ["# Risk Metrics Analysis Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Overall risk metrics
            lines.append("## Overall Risk Metrics\n")
            risk_metrics = ['total_pnl', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 
                          'win_rate', 'profit_factor', 'calmar_ratio', 'annualized_return']
            
            for metric in risk_metrics:
                value = overall_metrics.get(metric, 0)
                if metric in ['win_rate']:
                    lines.append(f"- **{metric.replace('_', ' ').title()}**: {value:.1%}\n")
                elif metric in ['total_pnl', 'annualized_return']:
                    lines.append(f"- **{metric.replace('_', ' ').title()}**: {value:.2%}\n")
                else:
                    lines.append(f"- **{metric.replace('_', ' ').title()}**: {value:.4f}\n")

            # Fold-wise risk metrics
            if fold_results:
                lines.append("\n## Fold-wise Risk Metrics\n")
                
                fold_metrics = []
                for i, fold in enumerate(fold_results):
                    fold_data = {'fold': i + 1}
                    for metric in risk_metrics:
                        if metric in fold:
                            fold_data[metric] = fold[metric]
                    fold_metrics.append(fold_data)
                
                if fold_metrics:
                    fold_df = pd.DataFrame(fold_metrics)
                    fold_df = fold_df.set_index('fold')
                    
                    lines.append("### Risk Metrics by Fold\n")
                    lines.append(self._safe_to_markdown(fold_df) + "\n\n")
                    
                    # Statistics across folds
                    lines.append("### Risk Metrics Statistics\n")
                    numeric_cols = [col for col in fold_df.columns if pd.api.types.is_numeric_dtype(fold_df[col])]
                    if numeric_cols:
                        stats_df = fold_df[numeric_cols].agg(['mean', 'std', 'min', 'max']).round(4)
                        lines.append(self._safe_to_markdown(stats_df) + "\n\n")
                        
                        # Save data
                        fold_df.to_csv(self.outcomes_dir / f"layer4_fold_risk_metrics_{self.ts}.csv")
                        stats_df.to_csv(self.outcomes_dir / f"layer4_risk_metrics_stats_{self.ts}.csv)

            # Risk assessment
            lines.append("\n## Risk Assessment\n")
            
            # Risk categories
            sharpe = overall_metrics.get('sharpe_ratio', 0)
            max_dd = overall_metrics.get('max_drawdown', 0)
            win_rate = overall_metrics.get('win_rate', 0)
            
            if sharpe > 1.5:
                lines.append("- **Sharpe Ratio**: Excellent (> 1.5)\n")
            elif sharpe > 1.0:
                lines.append("- **Sharpe Ratio**: Good (> 1.0)\n")
            elif sharpe > 0.5:
                lines.append("- **Sharpe Ratio**: Moderate (> 0.5)\n")
            else:
                lines.append("- **Sharpe Ratio**: Poor (< 0.5)\n")
            
            if max_dd > -0.05:
                lines.append("- **Max Drawdown**: Excellent (> -5%)\n")
            elif max_dd > -0.10:
                lines.append("- **Max Drawdown**: Good (> -10%)\n")
            elif max_dd > -0.20:
                lines.append("- **Max Drawdown**: Moderate (> -20%)\n")
            else:
                lines.append("- **Max Drawdown**: Poor (< -20%)\n")
            
            if win_rate > 0.55:
                lines.append("- **Win Rate**: Excellent (> 55%)\n")
            elif win_rate > 0.50:
                lines.append("- **Win Rate**: Good (> 50%)\n")
            elif win_rate > 0.45:
                lines.append("- **Win Rate**: Moderate (> 45%)\n")
            else:
                lines.append("- **Win Rate**: Poor (< 45%)\n")

            report_path = self.outcomes_dir / f"layer4_risk_metrics_report_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Risk metrics report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate risk metrics report: {e}")
    
    def _generate_model_performance_report(self, models_dict: Dict[str, Any], fold_results: List[Dict[str, Any]]) -> None:
        """Generate model performance analysis report."""
        try:
            lines = ["# Model Performance Analysis Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Model information
            lines.append("## Model Information\n")
            lines.append(f"- **Model Type**: {models_dict.get('model_type', 'Unknown')}\n")
            lines.append(f"- **Number of Models**: {len(models_dict.get('models', []))}\n")
            
            if 'n_estimators' in models_dict:
                lines.append(f"- **Estimators per Model**: {models_dict['n_estimators']}\n")
            if 'max_features' in models_dict:
                lines.append(f"- **Max Features**: {models_dict['max_features']}\n")

            # Cross-validation performance
            if fold_results:
                lines.append("\n## Cross-Validation Performance\n")
                
                # Extract common metrics across folds
                cv_metrics = {}
                for metric in ['total_pnl', 'sharpe_ratio', 'sortino_ratio', 'max_drawdown', 'win_rate']:
                    values = [fold.get(metric) for fold in fold_results if metric in fold]
                    if values:
                        cv_metrics[metric] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
                
                if cv_metrics:
                    cv_df = pd.DataFrame(cv_metrics).round(4)
                    lines.append("### CV Performance Statistics\n")
                    lines.append(self._safe_to_markdown(cv_df.T) + "\n\n")
                    
                    # Save CV data
                    cv_df.to_csv(self.outcomes_dir / f"layer4_cv_performance_{self.ts}.csv")

            # Model stability analysis
            lines.append("\n## Model Stability Analysis\n")
            
            if len(fold_results) > 1:
                # Calculate coefficient of variation for key metrics
                stability_metrics = {}
                for metric in ['total_pnl', 'sharpe_ratio', 'win_rate']:
                    values = [fold.get(metric) for fold in fold_results if metric in fold]
                    if values and np.mean(values) != 0:
                        cv = np.std(values) / abs(np.mean(values))
                        stability_metrics[metric] = cv
                
                if stability_metrics:
                    lines.append("### Metric Stability (Coefficient of Variation)\n")
                    for metric, cv in stability_metrics.items():
                        if cv < 0.1:
                            stability = "Excellent"
                        elif cv < 0.2:
                            stability = "Good"
                        elif cv < 0.3:
                            stability = "Moderate"
                        else:
                            stability = "Poor"
                        lines.append(f"- **{metric.replace('_', ' ').title()}**: {cv:.3f} ({stability})\n")

            report_path = self.outcomes_dir / f"layer4_model_performance_report_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Model performance report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate model performance report: {e}")
    
    def _generate_betting_analysis_report(self, oof_df: pd.DataFrame) -> None:
        """Generate betting analysis report."""
        try:
            lines = ["# Betting Analysis Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if 'bet_size' in oof_df.columns and 'realized_return' in oof_df.columns:
                df_valid = oof_df.dropna(subset=['bet_size', 'realized_return'])
                
                # Betting statistics
                lines.append("## Betting Statistics\n")
                total_bets = len(df_valid)
                winning_bets = (df_valid['realized_return'] > 0).sum()
                losing_bets = total_bets - winning_bets
                
                lines.append(f"- **Total Bets**: {total_bets:,}\n")
                lines.append(f"- **Winning Bets**: {winning_bets:,} ({winning_bets/total_bets:.1%})\n")
                lines.append(f"- **Losing Bets**: {losing_bets:,} ({losing_bets/total_bets:.1%})\n")

                # Bet size analysis
                lines.append("\n## Bet Size Analysis\n")
                lines.append(f"- **Average Bet Size**: {df_valid['bet_size'].mean():.4f}\n")
                lines.append(f"- **Total Bet Volume**: {df_valid['bet_size'].sum():.2f}\n")
                
                # Performance by bet size deciles
                df_valid['bet_size_decile'] = pd.qcut(df_valid['bet_size'], q=10, labels=False)
                decile_performance = df_valid.groupby('bet_size_decile').agg({
                    'realized_return': ['mean', 'std', 'count'],
                    'bet_size': 'mean'
                }).round(4)
                
                lines.append("\n### Performance by Bet Size Decile\n")
                lines.append(self._safe_to_markdown(decile_performance) + "\n\n")
                
                # Save betting data
                betting_stats = {
                    'total_bets': total_bets,
                    'winning_bets': winning_bets,
                    'losing_bets': losing_bets,
                    'win_rate': winning_bets / total_bets,
                    'avg_bet_size': df_valid['bet_size'].mean(),
                    'total_bet_volume': df_valid['bet_size'].sum()
                }
                
                betting_df = pd.DataFrame([betting_stats])
                betting_df.to_csv(self.outcomes_dir / f"layer4_betting_statistics_{self.ts}.csv", index=False)
                decile_performance.to_csv(self.outcomes_dir / f"layer4_bet_size_decile_performance_{self.ts}.csv")

            report_path = self.outcomes_dir / f"layer4_betting_analysis_report_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Betting analysis report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate betting analysis report: {e}")
    
    def _generate_fold_analysis_report(self, fold_results: List[Dict[str, Any]]) -> None:
        """Generate fold-wise analysis report."""
        try:
            lines = ["# Fold-wise Analysis Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if fold_results:
                # Create fold summary table
                fold_summary = []
                for i, fold in enumerate(fold_results):
                    summary = {
                        'fold': i + 1,
                        'train_samples': fold.get('train_samples', 0),
                        'val_samples': fold.get('val_samples', 0),
                        'total_pnl': fold.get('total_pnl', 0),
                        'sharpe_ratio': fold.get('sharpe_ratio', 0),
                        'max_drawdown': fold.get('max_drawdown', 0),
                        'win_rate': fold.get('win_rate', 0)
                    }
                    fold_summary.append(summary)
                
                fold_df = pd.DataFrame(fold_summary)
                fold_df = fold_df.set_index('fold')
                
                lines.append("## Fold Summary\n")
                lines.append(self._safe_to_markdown(fold_df) + "\n\n")
                
                # Best and worst performing folds
                lines.append("## Best and Worst Performing Folds\n")
                
                best_pnl_fold = fold_df['total_pnl'].idxmax()
                worst_pnl_fold = fold_df['total_pnl'].idxmin()
                best_sharpe_fold = fold_df['sharpe_ratio'].idxmax()
                worst_sharpe_fold = fold_df['sharpe_ratio'].idxmin()
                
                lines.append(f"### Total PnL\n")
                lines.append(f"- **Best Fold**: {best_pnl_fold} ({fold_df.loc[best_pnl_fold, 'total_pnl']:.4f})\n")
                lines.append(f"- **Worst Fold**: {worst_pnl_fold} ({fold_df.loc[worst_pnl_fold, 'total_pnl']:.4f})\n")
                
                lines.append(f"### Sharpe Ratio\n")
                lines.append(f"- **Best Fold**: {best_sharpe_fold} ({fold_df.loc[best_sharpe_fold, 'sharpe_ratio']:.4f})\n")
                lines.append(f"- **Worst Fold**: {worst_sharpe_fold} ({fold_df.loc[worst_sharpe_fold, 'sharpe_ratio']:.4f})\n")
                
                # Save fold data
                fold_df.to_csv(self.outcomes_dir / f"layer4_fold_summary_{self.ts}.csv")

            report_path = self.outcomes_dir / f"layer4_fold_analysis_report_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Fold analysis report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate fold analysis report: {e}")
    
    def _generate_feature_importance_report(self, models_dict: Dict[str, Any]) -> None:
        """Generate feature importance report."""
        try:
            lines = ["# Feature Importance Analysis Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            models = models_dict.get('models', [])
            if models and len(models) > 0:
                # Collect feature importances from all models
                all_importances = []
                
                for i, model in enumerate(models):
                    if hasattr(model, 'feature_importances_'):
                        importances = model.feature_importances_
                        feature_names = [f'feature_{j}' for j in range(len(importances))]
                        
                        for j, importance in enumerate(importances):
                            all_importances.append({
                                'feature': feature_names[j],
                                'importance': importance,
                                'model': f'model_{i}'
                            })
                
                if all_importances:
                    importance_df = pd.DataFrame(all_importances)
                    
                    # Aggregate by feature
                    feature_summary = importance_df.groupby('feature')['importance'].agg(['mean', 'std', 'count']).reset_index()
                    feature_summary.columns = ['feature', 'mean_importance', 'std_importance', 'model_count']
                    feature_summary = feature_summary.sort_values('mean_importance', ascending=False)
                    
                    lines.append("## Top Features by Importance\n")
                    lines.append(self._safe_to_markdown(feature_summary.head(20)) + "\n\n")
                    
                    # Feature importance statistics
                    lines.append("## Feature Importance Statistics\n")
                    lines.append(f"- **Total Features**: {len(feature_summary)}\n")
                    lines.append(f"- **Mean Importance**: {feature_summary['mean_importance'].mean():.4f}\n")
                    lines.append(f"- **Importance Std**: {feature_summary['mean_importance'].std():.4f}\n")
                    lines.append(f"- **Max Importance**: {feature_summary['mean_importance'].max():.4f}\n")
                    lines.append(f"- **Min Importance**: {feature_summary['mean_importance'].min():.4f}\n")
                    
                    # Save feature importance data
                    importance_df.to_csv(self.outcomes_dir / f"layer4_feature_importance_detailed_{self.ts}.csv", index=False)
                    feature_summary.to_csv(self.outcomes_dir / f"layer4_feature_importance_summary_{self.ts}.csv", index=False)

            else:
                lines.append("No feature importance data available.\n")

            report_path = self.outcomes_dir / f"layer4_feature_importance_report_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Feature importance report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate feature importance report: {e}")
    
    def _generate_performance_attribution_report(self, oof_df: pd.DataFrame, overall_metrics: Dict[str, Any]) -> None:
        """Generate performance attribution report."""
        try:
            lines = ["# Performance Attribution Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            if 'bet_size' in oof_df.columns and 'realized_return' in oof_df.columns:
                df_valid = oof_df.dropna(subset=['bet_size', 'realized_return'])
                
                # PnL attribution
                lines.append("## PnL Attribution\n")
                
                total_pnl = overall_metrics.get('total_pnl', 0)
                winning_pnl = df_valid[df_valid['realized_return'] > 0]['realized_return'].sum()
                losing_pnl = df_valid[df_valid['realized_return'] < 0]['realized_return'].sum()
                
                lines.append(f"- **Total PnL**: {total_pnl:.4f}\n")
                lines.append(f"- **Winning PnL**: {winning_pnl:.4f} ({winning_pnl/abs(total_pnl)*100:.1f}% of total)\n")
                lines.append(f"- **Losing PnL**: {losing_pnl:.4f} ({losing_pnl/abs(total_pnl)*100:.1f}% of total)\n")
                
                # Bet size contribution
                lines.append("\n## Bet Size Contribution Analysis\n")
                
                # Large bets vs small bets
                median_bet = df_valid['bet_size'].median()
                large_bets = df_valid[df_valid['bet_size'] > median_bet]
                small_bets = df_valid[df_valid['bet_size'] <= median_bet]
                
                large_pnl = (large_bets['bet_size'] * large_bets['realized_return']).sum()
                small_pnl = (small_bets['bet_size'] * small_bets['realized_return']).sum()
                
                lines.append(f"### Large Bets (> median) vs Small Bets (<= median)\n")
                lines.append(f"- **Large Bets PnL**: {large_pnl:.4f} ({len(large_bets)} bets)\n")
                lines.append(f"- **Small Bets PnL**: {small_pnl:.4f} ({len(small_bets)} bets)\n")
                lines.append(f"- **Large Bet Efficiency**: {large_pnl/len(large_bets):.6f}\n")
                lines.append(f"- **Small Bet Efficiency**: {small_pnl/len(small_bets):.6f}\n")
                
                # Performance attribution data
                attribution_data = {
                    'total_pnl': total_pnl,
                    'winning_pnl': winning_pnl,
                    'losing_pnl': losing_pnl,
                    'large_bets_pnl': large_pnl,
                    'small_bets_pnl': small_pnl,
                    'large_bets_count': len(large_bets),
                    'small_bets_count': len(small_bets),
                    'large_bet_efficiency': large_pnl/len(large_bets) if len(large_bets) > 0 else 0,
                    'small_bet_efficiency': small_pnl/len(small_bets) if len(small_bets) > 0 else 0
                }
                
                attribution_df = pd.DataFrame([attribution_data])
                attribution_df.to_csv(self.outcomes_dir / f"layer4_performance_attribution_{self.ts}.csv", index=False)

            report_path = self.outcomes_dir / f"layer4_performance_attribution_report_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Performance attribution report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate performance attribution report: {e}")
    
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
