#!/usr/bin/env python3
"""
Data Preparation Quality Analysis Report
Analyzes the quality of feature engineering, data cleaning, and data transformation processes.
"""

import glob
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

class DataPreparationQualityAnalyzer:
    pass  # TODO: Add proper implementation
    def __init__(self):
        self.data = None
        self.report = {}
        

    def load_data(self, data_path):
        """Load the prepared data for analysis."""
        try:
            if data_path.endswith('.pkl'):
                with open(data_path, 'rb') as f:
                    self.data = pickle.load(f)
            elif data_path.endswith('.csv'):
                self.data = pd.read_csv(data_path)
            else:
                self._load_from_directory(data_path)
                
            if self.data is not None and not self.data.empty:
                print(f"✅ Data loaded: {len(self.data)} rows, {len(self.data.columns)} columns")
                return True
            else:
                print(warning("No data loaded or data is empty"))
                return False
        except Exception as e:
            print(warning(f"Error loading data: {e}"))
            return False
    

    def _load_from_directory(self, data_dir):
        """Load data from directory structure."""
        patterns = ['*features*.csv', '*prepared*.csv', '*processed*.csv', '*training*.csv']
        for pattern in patterns:
            files = glob.glob(os.path.join(data_dir, pattern))
            if files:
                try:
                    self.data = pd.read_csv(files[0])
                    print(f"Found data file: {files[0]}")
                    break
                except Exception as e:
                    print(f"Error loading {files[0]}: {e}")
    

    def analyze_preparation_quality(self):
        """Comprehensive data preparation quality analysis."""
        if self.data is None or self.data.empty:
            print(warning("No data loaded. Please load data first."))
            return
        
        print("\n" + "="*60)
        print("🔍 DATA PREPARATION QUALITY ANALYSIS REPORT")
        print("="*60)
        
        # 1. Feature engineering quality analysis
        self._analyze_feature_engineering_quality()
        
        # 2. Data cleaning effectiveness
        self._analyze_data_cleaning_effectiveness()
        
        # 3. Feature correlation analysis
        self._analyze_feature_correlations()
        
        # 4. Calculate quality metrics
        self._calculate_preparation_quality_metrics()
        
        # 5. Generate recommendations
        self._generate_preparation_recommendations()
        
        # 6. Create visualizations
        self._create_preparation_visualizations()
    

    def _analyze_feature_engineering_quality(self):
        """Analyze the quality of feature engineering."""
        print("\n🔧 FEATURE ENGINEERING QUALITY ANALYSIS")
        print("-" * 50)
        
        # Check for common feature categories
        feature_categories = {
            'price_features': ['open', 'high', 'low', 'close', 'volume'],
            'technical_indicators': ['RSI', 'MACD', 'BB', 'ATR', 'ADX'],
            'volatility_features': ['volatility', 'vol'],
            'momentum_features': ['momentum', 'mom'],
            'funding_features': ['funding', 'fund'],
            'target_variables': ['target', 'reward', 'risk']
        }
        
        feature_stats = {}
        
        for category, keywords in feature_categories.items():
            matching_features = []
            for keyword in keywords:
                matching_cols = [col for col in self.data.columns if keyword.lower() in col.lower()]
                matching_features.extend(matching_cols)

            if matching_features:
                # Analyze feature quality
                missing_pct = np.mean([(self.data[col].isnull().sum() / len(self.data)) * 100 
                                       for col in matching_features if col in self.data.columns])
                inf_count = np.sum([np.isinf(self.data[col]).sum() 
                                    for col in matching_features if col in self.data.columns])

                quality_score = 100
                if missing_pct > 10:
                    quality_score -= (missing_pct - 10) * 2
                if inf_count > 0:
                    quality_score -= inf_count * 0.1

                feature_stats[category] = {
                    'found_features': len(matching_features),
                    'missing_percentage': missing_pct,
                    'infinite_count': inf_count,
                    'quality_score': max(0, quality_score)
                }
            else:
                feature_stats[category] = {
                    'found_features': 0,
                    'missing_percentage': 0,
                    'infinite_count': 0,
                    'quality_score': 0
                }
        
        # Print feature engineering summary
        print(f"{'Category':<20} {'Features':<10} {'Missing %':<12} {'Quality':<10}")
        print("-" * 55)
        
        for category, stats in feature_stats.items():
            print(f"{category:<20} {stats['found_features']:<10} {stats['missing_percentage']:<12.1f} {stats['quality_score']:<10.1f}")
        
        self.report['feature_engineering'] = feature_stats
    

    def _analyze_data_cleaning_effectiveness(self):
        """Analyze the effectiveness of data cleaning processes."""
        print("\n🧹 DATA CLEANING EFFECTIVENESS ANALYSIS")
        print("-" * 50)
        
        # Check for remaining missing values
        total_missing = self.data.isnull().sum().sum()
        total_cells = len(self.data) * len(self.data.columns)
        missing_percentage = (total_missing / total_cells) * 100
        
        # Check for infinite values
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        inf_count = self.data[numeric_cols].isin([np.inf, -np.inf]).sum().sum()
        
        # Check for duplicate rows
        duplicates = self.data.duplicated().sum()
        duplicate_percentage = (duplicates / len(self.data)) * 100
        
        # Calculate cleaning scores
        missing_score = max(0, 100 - missing_percentage * 2)
        inf_score = max(0, 100 - (inf_count / len(self.data)) * 10)
        duplicate_score = max(0, 100 - duplicate_percentage * 5)
        
        overall_cleaning_score = np.mean([missing_score, inf_score, duplicate_score])
        
        cleaning_stats = {
            'missing_values': {'count': total_missing, 'percentage': missing_percentage, 'score': missing_score},
            'infinite_values': {'count': inf_count, 'score': inf_score},
            'duplicates': {'count': duplicates, 'percentage': duplicate_percentage, 'score': duplicate_score},
            'overall_score': overall_cleaning_score
        }
        
        # Print cleaning summary
        print(f"{'Issue':<20} {'Count':<10} {'Percentage':<12} {'Score':<8}")
        print("-" * 50)
        print(f"{'Missing Values':<20} {total_missing:<10,} {missing_percentage:<12.2f} {missing_score:<8.1f}")
        print(f"{'Infinite Values':<20} {inf_count:<10,} {'N/A':<12} {inf_score:<8.1f}")
        print(f"{'Duplicates':<20} {duplicates:<10,} {duplicate_percentage:<12.2f} {duplicate_score:<8.1f}")
        print(f"{'Overall':<20} {'N/A':<10} {'N/A':<12} {overall_cleaning_score:<8.1f}")
        
        self.report['data_cleaning'] = cleaning_stats
    

    def _analyze_feature_correlations(self):
        """Analyze feature correlations and multicollinearity."""
        print("\n🔗 FEATURE CORRELATION ANALYSIS")
        print("-" * 40)
        
        # Select numeric features for correlation analysis
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            print("Not enough numeric features for correlation analysis.")
            return
        
        # Calculate correlation matrix
        correlation_matrix = self.data[numeric_cols].corr()
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        # Find features with very low correlation
        low_corr_features = []
        for col in numeric_cols:
            correlations = correlation_matrix[col].abs()
            avg_corr = correlations.mean()
            if avg_corr < 0.1:  # Very low average correlation
                low_corr_features.append({
                    'feature': col,
                    'avg_correlation': avg_corr
                })
        
        correlation_stats = {
            'high_correlation_pairs': high_corr_pairs,
            'low_correlation_features': low_corr_features
        }
        
        # Print correlation summary
        print(f"High correlation pairs (|r| > 0.8): {len(high_corr_pairs)}")
        if high_corr_pairs:
            print("Top high correlation pairs:")
            for pair in sorted(high_corr_pairs, key=lambda x: abs(x['correlation']), reverse=True)[:5]:
                print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
        
        print(f"\nLow correlation features (avg |r| < 0.1): {len(low_corr_features)}")
        if low_corr_features:
            print("Features with low correlations:")
            for feature in sorted(low_corr_features, key=lambda x: x['avg_correlation'])[:5]:
                print(f"  {feature['feature']}: {feature['avg_correlation']:.3f}")
        
        self.report['correlation_analysis'] = correlation_stats
    

    def _calculate_preparation_quality_metrics(self):
        """Calculate overall preparation quality metrics."""
        print("\n📈 OVERALL PREPARATION QUALITY METRICS")
        print("-" * 50)
        
        # Calculate composite quality score
        feature_engineering_score = 0
        if self.report.get('feature_engineering'):
            scores = [stats['quality_score'] for stats in self.report['feature_engineering'].values()]
            feature_engineering_score = np.mean(scores) if scores else 0
        
        data_cleaning_score = self.report.get('data_cleaning', {}).get('overall_score', 0)
        
        # Overall preparation score
        preparation_score = (feature_engineering_score * 0.6 + data_cleaning_score * 0.4)
        
        quality_metrics = {
            'feature_engineering_score': feature_engineering_score,
            'data_cleaning_score': data_cleaning_score,
            'overall_preparation_score': preparation_score
        }
        
        # Print quality summary
        print(f"{'Metric':<30} {'Score':<10} {'Status':<15}")
        print("-" * 55)
        
        for metric, score in quality_metrics.items():
            if score >= 80:
                status="✅ Excellent"
            elif score >= 60:
                status = "⚠️  Good"
            elif score >= 40:
                status = "⚠️  Fair"
            else:
                status = "❌ Poor"
            
            metric_name = metric.replace('_', ' ').title()
            print(f"{metric_name:<30} {score:<10.1f} {status:<15}")
        
        print(f"\nOverall Preparation Quality: {preparation_score:.1f}/100")
        
        if preparation_score >= 80:
            print("🎉 Excellent data preparation quality!")
        elif preparation_score >= 60:
            print("✅ Good data preparation quality")
        elif preparation_score >= 40:
            print(warning("Fair data preparation quality - consider improvements"))
        else:
            print(warning("Poor data preparation quality - immediate attention required"))
        
        self.report['quality_metrics'] = quality_metrics
    

    def _generate_preparation_recommendations(self):
        """Generate recommendations based on preparation analysis."""
        print("\n💡 DATA PREPARATION RECOMMENDATIONS")
        print("-" * 40)
        
        recommendations = []
        
        # Feature engineering recommendations
        feature_engineering = self.report.get('feature_engineering', {})
        for category, stats in feature_engineering.items():
            if stats['found_features'] == 0:
                recommendations.append(f"🔧 {category}: No features found in this category")
            elif stats['quality_score'] < 60:
                recommendations.append(f"🔧 {category}: Poor feature quality (score: {stats['quality_score']:.1f})")
        
        # Data cleaning recommendations
        data_cleaning = self.report.get('data_cleaning', {})
        if data_cleaning.get('overall_score', 0) < 70:
            recommendations.append("🧹 Data cleaning effectiveness is below target")
        
        missing_pct = data_cleaning.get('missing_values', {}).get('percentage', 0)
        if missing_pct > 10:
            recommendations.append(f"🔍 Missing values are high: {missing_pct:.1f}%")
        
        # Correlation recommendations
        correlation_analysis = self.report.get('correlation_analysis', {})
        high_corr_pairs = correlation_analysis.get('high_correlation_pairs', [])
        if len(high_corr_pairs) > 10:
            recommendations.append("🔗 Too many highly correlated feature pairs")
        
        if not recommendations:
            print("✅ No major issues detected. Data preparation quality is good!")
        else:
            print("Recommendations for improvement:")
            for rec in recommendations:
                print(f"  {rec}")
        
        self.report['recommendations'] = recommendations
    

    def _create_preparation_visualizations(self):
        """Create visualizations for the preparation report."""
        print("\n📈 GENERATING PREPARATION VISUALIZATIONS...")
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Data Preparation Quality Analysis Report', fontsize=16, fontweight='bold')
            
            # 1. Feature engineering quality
            feature_engineering = self.report.get('feature_engineering', {})
            if feature_engineering:
                categories = list(feature_engineering.keys())
                quality_scores = [feature_engineering[cat]['quality_score'] for cat in categories]
                
                colors = ['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in quality_scores]
                axes[0, 0].bar(categories, quality_scores, color=colors)
                axes[0, 0].set_ylabel('Quality Score')
                axes[0, 0].set_title('Feature Engineering Quality')
                axes[0, 0].set_ylim(0, 100)
                axes[0, 0].tick_params(axis='x', rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
            
            # 2. Data cleaning scores
            data_cleaning = self.report.get('data_cleaning', {})
            if data_cleaning:
                metric_names=['missing_values', 'infinite_values', 'duplicates']
                cleaning_scores=[]
                for metric in metric_names:
                    if metric in data_cleaning:
                        score = data_cleaning[metric].get('score', 0)
                        cleaning_scores.append(score)
                
                if cleaning_scores:
                    colors = ['green' if score >= 80 else 'orange' if score >= 60 else 'red' for score in cleaning_scores]
                    axes[0, 1].bar(metric_names, cleaning_scores, color=colors)
                    axes[0, 1].set_ylabel('Cleaning Score')
                    axes[0, 1].set_title('Data Cleaning Effectiveness')
                    axes[0, 1].set_ylim(0, 100)
                    axes[0, 1].tick_params(axis='x', rotation=45)
                    axes[0, 1].grid(True, alpha=0.3)
            
            # 3. Correlation analysis
            correlation_analysis = self.report.get('correlation_analysis', {})
            if correlation_analysis.get('high_correlation_pairs'):
                high_corr_pairs = correlation_analysis['high_correlation_pairs'][:10]
                pair_names = [f"{pair['feature1'][:10]}...{pair['feature2'][:10]}" for pair in high_corr_pairs]
                pair_values = [abs(pair['correlation']) for pair in high_corr_pairs]
                colors = ['green' if v >= 0.8 else 'orange' if v >= 0.6 else 'red' for v in pair_values]
                axes[1, 0].bar(pair_names, pair_values, color=colors)
                axes[1, 0].set_ylabel('Correlation (|r|)')
                axes[1, 0].set_title('Top High Correlation Pairs')
                axes[1, 0].set_ylim(0, 1)
                axes[1, 0].tick_params(axis='x', rotation=45)
                axes[1, 0].grid(True, alpha=0.3)
            
            # 4. Overall quality pie chart
            quality_metrics = self.report.get('quality_metrics', {})
            if quality_metrics:
                overall_score = quality_metrics.get('overall_preparation_score', 0)
                axes[1, 1].pie([overall_score, 100 - overall_score], 
                               labels=['Quality Score', 'Remaining'],
                               autopct='%1.1f%%',
                               colors=['lightblue', 'lightgray'])
                axes[1, 1].set_title('Overall Preparation Quality')
            
            plt.tight_layout()
            plt.savefig('data_preparation_quality_report.png', dpi=300, bbox_inches='tight')
            print("✅ Visualizations saved as 'data_preparation_quality_report.png'")
            
        except Exception as e:
            print(warning(f"Error creating visualizations: {e}"))
    

    def save_report(self, filename='data_preparation_quality_report.txt'):
        """Save the analysis report to a file."""
        with open(filename, 'w') as f:
            f.write("DATA PREPARATION QUALITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall quality
            quality_metrics = self.report.get('quality_metrics', {})
            overall_score = quality_metrics.get('overall_preparation_score', 0)
            f.write(f"Overall Preparation Quality: {overall_score:.1f}/100\n\n")
            
            # Feature engineering
            feature_engineering = self.report.get('feature_engineering', {})
            f.write("FEATURE ENGINEERING QUALITY:\n")
            for category, stats in feature_engineering.items():
                f.write(f"{category}: {stats['quality_score']:.1f}/100\n")
            f.write("\n")
            
            # Data cleaning
            data_cleaning = self.report.get('data_cleaning', {})
            f.write("DATA CLEANING EFFECTIVENESS:\n")
            f.write(f"Missing values: {data_cleaning.get('missing_values', {}).get('percentage', 0):.2f}%\n")
            f.write(f"Overall cleaning score: {data_cleaning.get('overall_score', 0):.1f}/100\n\n")
            
            # Recommendations
            recommendations = self.report.get('recommendations', [])
            if recommendations:
                f.write("RECOMMENDATIONS:\n")
                for rec in recommendations:
                    f.write(f"- {rec}\n")
            f.write("\n")
        
        print(f"✅ Report saved as '{filename}'")

def main():
    """Main function to run the analysis."""
    analyzer = DataPreparationQualityAnalyzer()
    
    # Try to load data from common locations
    data_paths = [
        'data/prepared_data.pkl',
        'data/features.csv',
        'data/training_data.csv',
        'data/processed_data.pkl',
        'data/'
    ]
    
    data_loaded = False
    for path in data_paths:
        if os.path.exists(path):
            if analyzer.load_data(path):
                data_loaded = True
                break
    
    if not data_loaded:
        print(warning("Could not find data file. Please specify the path to your prepared data."))
        print("Common locations checked:")
        for path in data_paths:
            print(f"  - {path}")
        return
    
    # Run analysis
    analyzer.analyze_preparation_quality()
    
    # Save report
    analyzer.save_report()

if __name__ == "__main__":
    main() 
