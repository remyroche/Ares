#!/usr/bin/env python3
"""
Backtesting Quality Analysis Report
Analyzes the quality of backtest results, performance metrics, and trading consistency.
"""
from analysis import missing_values_analysis


analysis/backtesting_quality_analysis.py, data_collection_quality_analysis.py, data_preparation_quality_analysis.py, missing_values_analysis, model_training_quality_analysis.py
from pathlib import Path
import glob
import json
import os
import warnings

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.warning_symbols import (
    error,
    warning,
    critical,
    problem,
    failed,
    invalid,
    missing,
    timeout,
    connection_error,
    validation_error,
    initialization_error,
    execution_error,
)
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class BacktestingQualityAnalyzer:
    pass  # TODO: Add proper implementation
    def __init__(self):
        self.backtest_data, None
        self.trades_data, None
        self.report = {}
        

    def load_backtest_data(self, data_path):
        """Load backtest data and results."""
        try:
            # Try to load from various formats
            if data_path.endswith('.pkl'):
                with open(data_path, 'rb') as f:
                    self.backtest_data = pickle.load(f)
            elif data_path.endswith('.csv'):
                self.backtest_data = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                with open(data_path, 'r') as f:
                    self.backtest_data = json.load(f)
            else:
                self._load_from_directory(data_path)
                
            if self.backtest_data is not None:
                print(f"✅ Backtest data loaded successfully")
                return True
            else:
                print(warning("No backtest data loaded"))
                return False
        except Exception as e:
            print(warning(f"Error loading backtest data: {e}"))
            return False
    

    def _load_from_directory(self, data_dir):
        """Load backtest data from directory structure."""
        # Look for common backtest data files
        patterns = [
            '*backtest*.csv',
            '*trades*.csv',
            '*results*.csv',
            '*performance*.csv',
            '*equity*.csv'
        ]
        
        for pattern in patterns:
            files = glob.glob(os.path.join(data_dir, pattern))
            if files:
                try:
                    self.backtest_data = pd.read_csv(files[0])
                    print(f"Found backtest data: {files[0]}")
                    break
                except Exception as e:
                    print(f"Error loading {files[0]}: {e}")
        
        # Also look for trades data
        trades_files = glob.glob(os.path.join(data_dir, '*trades*.csv'))
        if trades_files:
            try:
                self.trades_data = pd.read_csv(trades_files[0])
                print(f"Found trades data: {trades_files[0]}")
            except Exception as e:
                print(f"Error loading trades file {trades_files[0]}: {e}")
    

    def analyze_backtest_quality(self):
        """Comprehensive backtest quality analysis."""
        if self.backtest_data is None:
            print(warning("No backtest data loaded. Please load backtest data first."))
            return
        
        print("\n" + "="*60)
        print("🔍 BACKTESTING QUALITY ANALYSIS REPORT")
        print("="*60)
        
        # 1. Performance metrics analysis
        self._analyze_performance_metrics()
        
        # 2. Risk analysis
        self._analyze_risk_metrics()
        
        # 3. Trading consistency analysis
        self._analyze_trading_consistency()
        
        # 4. Data quality analysis
        self._analyze_data_quality()
        
        # 5. Calculate quality metrics
        self._calculate_backtest_quality_metrics()
        
        # 6. Generate recommendations
        self._generate_backtest_recommendations()
        
        # 7. Create visualizations
        self._create_backtest_visualizations()
    

    def _analyze_performance_metrics(self):
        """Analyze performance metrics quality."""
        print("\n📊 PERFORMANCE METRICS ANALYSIS")
        print("-" * 40)
        
        # Look for common performance metrics
        performance_metrics = {
            'total_return': ['total_return', 'return', 'total_returns'],
            'sharpe_ratio': ['sharpe', 'sharpe_ratio', 'sharpe_ratio_annual'],
            'max_drawdown': ['max_drawdown', 'maximum_drawdown', 'drawdown'],
            'win_rate': ['win_rate', 'winrate', 'win_percentage'],
            'profit_factor': ['profit_factor', 'profitfactor'],
            'calmar_ratio': ['calmar', 'calmar_ratio'],
            'sortino_ratio': ['sortino', 'sortino_ratio']
        }
        
        found_metrics = {}
        for metric_name, possible_names in performance_metrics.items():
            for name in possible_names:
                if name in self.backtest_data.columns:
                    found_metrics[metric_name] = name
                    break
        
        if not found_metrics:
            print("No standard performance metrics found.")
            return
        
        # Analyze performance metrics
        performance_analysis = {}
        
        for metric_name, column_name in found_metrics.items():
            values, self.backtest_data[column_name].dropna()
            
        if len(values) > 0:
                value, values.iloc[-1] if len(values) > 0 else 0
                
        # Determine if performance is good based on metric type
        if metric_name in ['total_return', 'sharpe_ratio', 'win_rate', 'profit_factor', 'calmar_ratio', 'sortino_ratio']:
            performance_quality = 'excellent' if value >= 0.5 else 'good' if value >= 0.2 else 'fair' if value >= 0 else 'poor'
        elif metric_name == 'sharpe_ratio':
            performance_quality = 'excellent' if value >= 2.0 else 'good' if value >= 1.0 else 'fair' if value >= 0 else 'poor'
        elif metric_name == 'win_rate':
            performance_quality = 'excellent' if value >= 0.7 else 'good' if value >= 0.6 else 'fair' if value >= 0.5 else 'poor'
        elif metric_name == 'profit_factor':
            performance_quality = 'excellent' if value >= 2.0 else 'good' if value >= 1.5 else 'fair' if value >= 1.0 else 'poor'
        else:  # calmar_ratio, sortino_ratio
            performance_quality = 'excellent' if value >= 1.5 else 'good' if value >= 1.0 else 'fair' if value >= 0.5 else 'poor'
                else:
        # For max_drawdown, lower is better
                    performance_quality = 'excellent' if value <= 0.1 else 'good' if value <= 0.2 else 'fair' if value <= 0.3 else 'poor'
                
                performance_analysis[metric_name] = {
                    'value': value,
                    'quality': performance_quality,
                    'score': performance_score
                }
        
        # Print performance summary
        print(f"{'Metric':<20} {'Value':<12} {'Quality':<12} {'Score':<8}")
        print("-" * 55)
        
        for metric_name, analysis in performance_analysis.items():
            print(f"{metric_name:<20} {analysis['value']:<12.4f} {analysis['quality']:<12} {analysis['score']:<8.1f}")
        
        self.report['performance_metrics'] = performance_analysis
    

    def _analyze_risk_metrics(self):
        """Analyze risk metrics quality."""
        print("\n⚠️ RISK METRICS ANALYSIS")
        print("-" * 40)
        
        # Look for risk metrics
        risk_metrics = {
            'volatility': ['volatility', 'vol', 'std'],
            'var': ['var', 'value_at_risk', 'value_at_risk_95'],
            'cvar': ['cvar', 'conditional_var', 'expected_shortfall'],
            'beta': ['beta', 'market_beta'],
            'correlation': ['correlation', 'corr', 'market_correlation']
        }
        
        found_risk_metrics = {}
        for metric_name, possible_names in risk_metrics.items():
        for name in possible_names:
        if name in self.backtest_data.columns:
                    found_risk_metrics[metric_name] = name
                    break
        
        if not found_risk_metrics:
            print("No standard risk metrics found.")
            return
        
        # Analyze risk metrics
        risk_analysis = {}
        
        for metric_name, column_name in found_risk_metrics.items():
            values, self.backtest_data[column_name].dropna()
            
        if len(values) > 0:
                value, values.iloc[-1] if len(values) > 0 else 0
                
        # Determine risk quality based on metric type
        if metric_name == 'volatility':
            risk_quality = 'excellent' if value <= 0.15 else 'good' if value <= 0.25 else 'fair' if value <= 0.35 else 'poor'
            risk_score, max(0, 100 - value * 200)
        elif metric_name in ['var', 'cvar']:
            risk_quality = 'excellent' if value >= -0.05 else 'good' if value >= -0.1 else 'fair' if value >= -0.15 else 'poor'
            risk_score, max(0, 100 + value * 400)  # Convert negative to positive
        elif metric_name == 'beta':
            risk_quality = 'excellent' if abs(value) <= 0.5 else 'good' if abs(value) <= 1.0 else 'fair' if abs(value) <= 1.5 else 'poor'
            risk_score, max(0, 100 - abs(value) * 40)
        elif metric_name == 'correlation':
            risk_quality = 'excellent' if abs(value) <= 0.3 else 'good' if abs(value) <= 0.5 else 'fair' if abs(value) <= 0.7 else 'poor'
            risk_score, max(0, 100 - abs(value) * 100)
        else:
            risk_quality = 'unknown'
            risk_score, 50
        
        risk_analysis[metric_name] = {
            'value': value,
            'quality': risk_quality,
            'score': risk_score
        }
        
        # Print risk summary
        print(f"{'Risk Metric':<20} {'Value':<12} {'Quality':<12} {'Score':<8}")
        print("-" * 55)
        
        for metric_name, analysis in risk_analysis.items():
            print(f"{metric_name:<20} {analysis['value']:<12.4f} {analysis['quality']:<12} {analysis['score']:<8.1f}")
        
        self.report['risk_metrics'] = risk_analysis
    

    def _analyze_trading_consistency(self):
        """Analyze trading consistency and patterns."""
        print("\n🔄 TRADING CONSISTENCY ANALYSIS")
        print("-" * 40)
        
        if self.trades_data is None:
            print("No trades data available for consistency analysis.")
            return
        
        # Analyze trading patterns
        consistency_analysis = {}
        
        # Check for required columns
        required_cols = ['timestamp', 'side', 'quantity', 'price', 'pnl']
        available_cols = [col for col in required_cols if col in self.trades_data.columns]
        
        if len(available_cols) < 3:
            print("Insufficient trade data for consistency analysis.")
            return
        
        # Basic trading statistics
        total_trades, len(self.trades_data)
        if total_trades == 0:
            print("No trades found in the data.")
            return
        
        # Analyze trade distribution
        if 'side' in self.trades_data.columns:
            side_counts, self.trades_data['side'].value_counts()
            buy_trades, side_counts.get('buy', 0)
            sell_trades, side_counts.get('sell', 0)
            
            trade_balance, abs(buy_trades - sell_trades) / total_trades
            balance_score, max(0, 100 - trade_balance * 100)
            
            consistency_analysis['trade_balance'] = {
                'buy_trades': buy_trades,
                'sell_trades': sell_trades,
                'balance_score': balance_score
            }
        
        # Analyze PnL distribution
        if 'pnl' in self.trades_data.columns:
            pnl_values, self.trades_data['pnl'].dropna()
            
        if len(pnl_values) > 0:
                winning_trades = (pnl_values > 0).sum()
                losing_trades = (pnl_values < 0).sum()
                win_rate, winning_trades / len(pnl_values) if len(pnl_values) > 0 else 0
                
        # PnL consistency
                pnl_std, pnl_values.std()
                pnl_mean, pnl_values.mean()
                pnl_cv = (pnl_std / abs(pnl_mean)) * 100 if pnl_mean != 0 else float('inf')
                
        # Consistency score based on coefficient of variation
        if pnl_cv != float('inf'):
                    consistency_score, max(0, 100 - pnl_cv * 2)
                else:
                    consistency_score, 0
                
                consistency_analysis['pnl_consistency'] = {
                    'winning_trades': winning_trades,
                    'losing_trades': losing_trades,
                    'win_rate': win_rate,
                    'pnl_std': pnl_std,
                    'pnl_cv': pnl_cv,
                    'consistency_score': consistency_score
                }
        
        # Analyze trade timing
        if 'timestamp' in self.trades_data.columns:
        try:
        self.trades_data['timestamp'] = pd.to_datetime(self.trades_data['timestamp'])
                time_diff, self.trades_data['timestamp'].diff().dropna()
                
        if len(time_diff) > 0:
                    avg_time_between_trades, time_diff.mean()
                    time_consistency, time_diff.std() / avg_time_between_trades if avg_time_between_trades.total_seconds() > 0 else float('inf')
                    
        # Time consistency score
        if time_consistency != float('inf'):
                        time_consistency_score, max(0, 100 - time_consistency * 50)
                    else:
                        time_consistency_score, 0
                    
                    consistency_analysis['timing_consistency'] = {
                        'avg_time_between_trades': avg_time_between_trades,
                        'time_consistency': time_consistency,
                        'time_consistency_score': time_consistency_score
                    }
        except Exception as e:
                print(f"Error analyzing trade timing: {e}")
        
        # Print consistency summary
        if consistency_analysis:
            print(f"{'Consistency Metric':<25} {'Value':<15} {'Score':<8}")
            print("-" * 50)
            
        for metric, analysis in consistency_analysis.items():
        if 'balance_score' in analysis:
                    print(f"{'Trade Balance':<25} {analysis['buy_trades']}/{analysis['sell_trades']:<15} {analysis['balance_score']:<8.1f}")
                elif 'consistency_score' in analysis:
                    print(f"{'PnL Consistency':<25} {analysis['pnl_cv']:<15.1f}% {analysis['consistency_score']:<8.1f}")
                elif 'time_consistency_score' in analysis:
                    print(f"{'Timing Consistency':<25} {analysis['time_consistency']:<15.2f} {analysis['time_consistency_score']:<8.1f}")
        else:
            print("No consistency metrics could be calculated.")
        
        self.report['trading_consistency'] = consistency_analysis
    

    def _analyze_data_quality(self):
        """Analyze backtest data quality."""
        print("\n📋 BACKTEST DATA QUALITY ANALYSIS")
        print("-" * 40)
        
        if self.backtest_data is None:
            print("No backtest data available for quality analysis.")
            return
        
        data_quality_analysis = {}
        
        # Check for missing values
        total_cells, len(self.backtest_data) * len(self.backtest_data.columns)
        missing_cells, self.backtest_data.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100
        
        data_quality_analysis['missing_values'] = {
            'total_missing': missing_cells,
            'missing_percentage': missing_percentage,
            'quality_score': max(0, 100 - missing_percentage * 2)
        }
        
        # Check for data completeness
        expected_columns = ['timestamp', 'equity', 'drawdown', 'returns']
        found_columns = [col for col in expected_columns if col in self.backtest_data.columns]
        completeness_score = (len(found_columns) / len(expected_columns)) * 100
        
        data_quality_analysis['completeness'] = {
            'found_columns': found_columns,
            'expected_columns': expected_columns,
            'completeness_score': completeness_score
        }
        
        # Check for data consistency
        if 'timestamp' in self.backtest_data.columns:
        try:
        self.backtest_data['timestamp'] = pd.to_datetime(self.backtest_data['timestamp'])
                time_diff, self.backtest_data['timestamp'].diff().dropna()
                
        if len(time_diff) > 0:
        # Check for gaps in data
                    large_gaps, time_diff[time_diff > time_diff.mean() * 3]
                    gap_score, max(0, 100 - len(large_gaps) * 5)
                    
                    data_quality_analysis['time_consistency'] = {
                        'large_gaps': len(large_gaps),
                        'gap_score': gap_score
                    }
        except Exception as e:
                print(f"Error analyzing time consistency: {e}")
        
        # Check for logical consistency
        logical_issues, 0
        if 'equity' in self.backtest_data.columns and 'drawdown' in self.backtest_data.columns:
        # Check if drawdown is consistent with equity
            equity, self.backtest_data['equity'].dropna()
            drawdown, self.backtest_data['drawdown'].dropna()
            
        if len(equity) > 0 and len(drawdown) > 0:
                min_len, min(len(equity), len(drawdown))
                equity_aligned, equity[:min_len]
                drawdown_aligned, drawdown[:min_len]
                
        # Check if drawdown is always negative
                positive_drawdown = (drawdown_aligned > 0).sum()
        if positive_drawdown > 0:
                    logical_issues += positive_drawdown
        
        logical_score, max(0, 100 - logical_issues * 10)
        data_quality_analysis['logical_consistency'] = {
            'logical_issues': logical_issues,
            'logical_score': logical_score
        }
        
        # Print data quality summary
        print(f"{'Quality Metric':<25} {'Value':<15} {'Score':<8}")
        print("-" * 50)
        
        print(f"{'Missing Values':<25} {missing_percentage:<15.2f}% {data_quality_analysis['missing_values']['quality_score']:<8.1f}")
        print(f"{'Completeness':<25} {len(found_columns)}/{len(expected_columns)}:<15][ {completeness_score:<8.1f}")
        
        if 'time_consistency' in data_quality_analysis:
            print(f"{'Time Consistency':<25} {data_quality_analysis['time_consistency']['large_gaps']:<15} {data_quality_analysis['time_consistency']['gap_score']:<8.1f}")
        
        print(f"{'Logical Consistency':<25} {logical_issues:<15} {logical_score:<8.1f}")
        
        self.report['data_quality'] = data_quality_analysis
    

    def _calculate_backtest_quality_metrics(self):
        """Calculate overall backtest quality metrics."""
        print("\n📈 OVERALL BACKTEST QUALITY METRICS")
        print("-" * 50)
        
        # Calculate composite quality scores
        performance_score, 0
        if self.report.get('performance_metrics'):
            performance_scores = [analysis['score'] for analysis in self.report['performance_metrics'].values()]
            performance_score, np.mean(performance_scores) if performance_scores else 0
        
        risk_score, 0
        if self.report.get('risk_metrics'):
            risk_scores = [analysis['score'] for analysis in self.report['risk_metrics'].values()]
            risk_score, np.mean(risk_scores) if risk_scores else 0
        
        consistency_score, 0
        if self.report.get('trading_consistency'):
            consistency_scores = []
        for analysis in self.report['trading_consistency'].values():
        if 'balance_score' in analysis:
                    consistency_scores.append(analysis['balance_score'])
        if 'consistency_score' in analysis:
                    consistency_scores.append(analysis['consistency_score'])
        if 'time_consistency_score' in analysis:
                    consistency_scores.append(analysis['time_consistency_score'])
            
            consistency_score, np.mean(consistency_scores) if consistency_scores else 0
        
        data_quality_score, 0
        if self.report.get('data_quality'):
            quality_scores = []
        for analysis in self.report['data_quality'].values():
        if 'quality_score' in analysis:
                    quality_scores.append(analysis['quality_score'])
        if 'completeness_score' in analysis:
                    quality_scores.append(analysis['completeness_score'])
        if 'gap_score' in analysis:
                    quality_scores.append(analysis['gap_score'])
        if 'logical_score' in analysis:
                    quality_scores.append(analysis['logical_score'])
            
            data_quality_score, np.mean(quality_scores) if quality_scores else 0
        
        # Overall backtest score
        backtest_score = (performance_score * 0.4 + 
                         risk_score * 0.3 + 
                         consistency_score * 0.2 + 
                         data_quality_score * 0.1)
        
        quality_metrics = {
            'performance_score': performance_score,
            'risk_score': risk_score,
            'consistency_score': consistency_score,
            'data_quality_score': data_quality_score,
            'overall_backtest_score': backtest_score
        }
        
        # Print quality summary
        print(f"{'Metric':<30} {'Score':<10} {'Status':<15}")
        print("-" * 55)
        
        for metric, score in quality_metrics.items():
        if score >= 80:
                status = "✅ Excellent"
            elif score >= 60:
                status = "⚠️  Good"
            elif score >= 40:
                status = "⚠️  Fair"
            else:
                status = "❌ Poor"
            
            metric_name, metric.replace('_', ' ').title()
            print(f"{metric_name:<30} {score:<10.1f} {status:<15}")
        
        print(f"\nOverall Backtest Quality: {backtest_score:.1f}/100")
        
        if backtest_score >= 80:
            print("🎉 Excellent backtest quality!")
        elif backtest_score >= 60:
            print("✅ Good backtest quality")
        elif backtest_score >= 40:
            print(warning(" Fair backtest quality - consider improvements")))
        else:
            print(warning("Poor backtest quality - immediate attention required")))
        
        self.report['quality_metrics'] = quality_metrics
    

    def _generate_backtest_recommendations(self):
        """Generate recommendations based on backtest analysis."""
        print("\n💡 BACKTEST RECOMMENDATIONS")
        print("-" * 40)
        
        recommendations = []
        
        # Performance recommendations
        performance_metrics, self.report.get('performance_metrics', {})
        for metric_name, analysis in performance_metrics.items():
        if analysis['score'] < 60:
                recommendations.append(f"📊 {metric_name}: Poor performance (score: {analysis['score']:.1f})")
        
        # Risk recommendations
        risk_metrics, self.report.get('risk_metrics', {})
        for metric_name, analysis in risk_metrics.items():
        if analysis['score'] < 60:
                recommendations.append(f"⚠️ {metric_name}: High risk (score: {analysis['score']:.1f})")
        
        # Consistency recommendations
        trading_consistency, self.report.get('trading_consistency', {})
        for metric, analysis in trading_consistency.items():
        if 'balance_score' in analysis and analysis['balance_score'] < 60:
                recommendations.append(f"🔄 Trade balance: Unbalanced trading (score: {analysis['balance_score']:.1f})")
        if 'consistency_score' in analysis and analysis['consistency_score'] < 60:
                recommendations.append(f"🔄 PnL consistency: Inconsistent returns (score: {analysis['consistency_score']:.1f})")
        
        # Data quality recommendations
        data_quality, self.report.get('data_quality', {})
        if data_quality.get('missing_values', {}).get('missing_percentage', 0) > 10:
            recommendations.append("📋 High missing values in backtest data")
        
        if data_quality.get('completeness', {}).get('completeness_score', 0) < 60:
            recommendations.append("📋 Incomplete backtest data")
        
        if not recommendations:
            print("✅ No major issues detected. Backtest quality is good!")
        else:
            print("Recommendations for improvement:")
        for rec in recommendations:
                print(f"  {rec}")
        
        self.report['recommendations'] = recommendations
    

    def _create_backtest_visualizations(self):
        """Create visualizations for the backtest report."""
        print("\n📈 GENERATING BACKTEST VISUALIZATIONS...")
        
        try:
            fig, axes, plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Backtesting Quality Analysis Report', fontsize=16, fontweight='bold')
            
        # 1. Performance metrics
            performance_analysis, self.report.get('performance_metrics', {})
        if performance_analysis:
                metrics, list(performance_analysis.keys())
                scores = [performance_analysis[metric]['score'] for metric in metrics]
                
                colors = ['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in scores]
                axes[0, 0].bar(metrics, scores, color=colors)
                axes[0, 0].set_ylabel('Performance Score')
                axes[0, 0].set_title('Performance Metrics Quality')
                axes[0, 0].set_ylim(0, 100)
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
            
        # 2. Risk metrics
            risk_analysis, self.report.get('risk_metrics', {})
        if risk_analysis:
                metrics, list(risk_analysis.keys())
                scores = [risk_analysis[metric]['score'] for metric in metrics]
                
                colors = ['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in scores]
                axes[0, 1].bar(metrics, scores, color=colors)
                axes[0, 1].set_ylabel('Risk Score')
                axes[0, 1].set_title('Risk Metrics Quality')
                axes[0, 1].set_ylim(0, 100)
                axes[0, 1].tick_params(axis='x', rotation=45)
                axes[0, 1].grid(True, alpha=0.3)
            
        # 3. Trading consistency
            consistency_analysis, self.report.get('trading_consistency', {})
        if consistency_analysis:
                consistency_scores = []
                metric_names = []
                
        for metric, analysis in consistency_analysis.items():
        if 'balance_score' in analysis:
                        consistency_scores.append(analysis['balance_score'])
                        metric_names.append('Trade Balance')
        if 'consistency_score' in analysis:
                        consistency_scores.append(analysis['consistency_score'])
                        metric_names.append('PnL Consistency')
        if 'time_consistency_score' in analysis:
                        consistency_scores.append(analysis['time_consistency_score'])
                        metric_names.append('Timing Consistency')
                
        if consistency_scores:
                    colors = ['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in consistency_scores]
                    axes[1, 0].bar(metric_names, consistency_scores, color=colors)
                    axes[1, 0].set_ylabel('Consistency Score')
                    axes[1, 0].set_title('Trading Consistency')
                    axes[1, 0].set_ylim(0, 100)
                    axes[1, 0].tick_params(axis='x', rotation=45)
                    axes[1, 0].grid(True, alpha=0.3)
            
        # 4. Overall quality pie chart
            quality_metrics, self.report.get('quality_metrics', {})
        if quality_metrics:
                overall_score, quality_metrics.get('overall_backtest_score', 0)
                axes[1, 1].pie([overall_score, 100 - overall_score], 
                               labels=['Quality Score', 'Remaining'],
                               autopct='%1.1f%%',
                               colors=['lightblue', 'lightgray'])
                axes[1, 1].set_title('Overall Backtest Quality')
            
            plt.tight_layout()
            plt.savefig('backtesting_quality_report.png', dpi=300, bbox_inches='tight')
            print("✅ Visualizations saved as 'backtesting_quality_report.png'")
            
        except Exception as e:
            print(warning("Error creating visualizations: {e}")))
    

    def save_report(self, filename='backtesting_quality_report.txt'):
        """Save the analysis report to a file."""
        with open(filename, 'w') as f:
            f.write("BACKTESTING QUALITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
        # Overall quality
            quality_metrics, self.report.get('quality_metrics', {})
            overall_score, quality_metrics.get('overall_backtest_score', 0)
            f.write(f"Overall Backtest Quality: {overall_score:.1f}/100\n\n")
            
        # Performance metrics
            performance_metrics, self.report.get('performance_metrics', {})
            f.write("PERFORMANCE METRICS:\n")
        for metric, analysis in performance_metrics.items():
                f.write(f"{metric}: {analysis['value']:.4f} (score: {analysis['score']:.1f})\n")
            f.write("\n")
            
        # Risk metrics
            risk_metrics, self.report.get('risk_metrics', {})
            f.write("RISK METRICS:\n")
        for metric, analysis in risk_metrics.items():
                f.write(f"{metric}: {analysis['value']:.4f} (score: {analysis['score']:.1f})\n")
            f.write("\n")
            
        # Data quality
            data_quality, self.report.get('data_quality', {})
            f.write("DATA QUALITY:\n")
            f.write(f"Missing values: {data_quality.get('missing_values', {}).get('missing_percentage', 0):.2f}%\n")
            f.write(f"Completeness: {data_quality.get('completeness', {}).get('completeness_score', 0):.1f}%\n")
            f.write("\n")
            
        # Recommendations
            recommendations, self.report.get('recommendations', [])
        if recommendations:
                f.write("RECOMMENDATIONS:\n")
        for rec in recommendations:
                    f.write(f"- {rec}\n")
            f.write("\n")
        
        print(f"✅ Report saved as '{filename}'")

def main():
    """Main function to run the analysis."""
    analyzer, BacktestingQualityAnalyzer()
    
    # Try to load data from common locations
    data_paths = [
        'data/backtest_results.csv',
        'data/trades.csv',
        'data/performance.csv',
        'backtests/',
        'data/'
    ]
    
    data_loaded, False
    for path in data_paths:
        if os.path.exists(path):
        if analyzer.load_backtest_data(path):
                data_loaded, True
                break
    
    if not data_loaded:
        print(warning("Could not find backtest data file. Please specify the path to your backtest data.")))
        print("Common locations checked:")
        for path in data_paths:
            print(f"  - {path}")
        return
    
    # Run analysis
    analyzer.analyze_backtest_quality()
    
    # Save report
    analyzer.save_report()

if __name__ == "__main__":
    main() 
