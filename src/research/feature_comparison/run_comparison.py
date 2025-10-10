"""
Main Feature Comparison Script

This script runs the complete feature comparison analysis comparing:
1. Initial features (basic OHLCV)
2. VWAP-based features  
3. Volatility normalized features
4. VWAP + volatility normalized features

Using relevance metrics: LGBM/SHAP, LASSO, Mutual Information, Correlation
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from .feature_comparison_utils import FeatureComparisonUtils
from .feature_versions import FeatureVersions
from .relevance_analyzer import RelevanceAnalyzer
from .comparison_report import ComparisonReport

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureComparisonRunner:
    """
    Main class to run the complete feature comparison analysis.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None, 
                 target_col: str = 'returns',
                 task_type: str = 'regression'):
        """
        Initialize the feature comparison runner.
        
        Args:
            data: Input DataFrame with OHLCV data
            target_col: Name of target column
            task_type: 'regression' or 'classification'
        """
        self.data = data
        self.target_col = target_col
        self.task_type = task_type
        self.utils = FeatureComparisonUtils()
        self.analyzer = RelevanceAnalyzer()
        self.report_generator = ComparisonReport()
        
    def load_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Load or generate sample data for testing.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Sample DataFrame with OHLCV data
        """
        logger.info(f"Generating sample data with {n_samples} samples...")
        
        # Generate synthetic OHLCV data
        np.random.seed(42)
        n = n_samples
        
        # Generate price data with trend and volatility
        returns = np.random.normal(0.001, 0.02, n)  # 0.1% mean return, 2% volatility
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=n, freq='1H'),
            'open': prices * (1 + np.random.normal(0, 0.001, n)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n)
        })
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        data['high'] = np.maximum(data['high'], np.maximum(data['open'], data['close']))
        data['low'] = np.minimum(data['low'], np.minimum(data['open'], data['close']))
        
        logger.info(f"Generated sample data with shape: {data.shape}")
        return data
    
    def run_complete_analysis(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run the complete feature comparison analysis.
        
        Args:
            data: Input DataFrame (if None, will use self.data or generate sample)
            
        Returns:
            Dictionary with complete analysis results
        """
        logger.info("Starting complete feature comparison analysis...")
        
        # Use provided data or load sample data
        if data is not None:
            self.data = data
        elif self.data is None:
            self.data = self.load_sample_data()
        
        logger.info(f"Using data with shape: {self.data.shape}")
        
        # Step 1: Create feature versions
        logger.info("Step 1: Creating feature versions...")
        feature_versions = FeatureVersions(self.data, self.target_col)
        
        # Create target variable
        target = feature_versions.create_target(method='future_returns', periods=1)
        logger.info(f"Created target variable with {target.notna().sum()} valid values")
        
        # Generate all feature versions
        versions = feature_versions.generate_all_versions()
        logger.info(f"Generated {len(versions)} feature versions")
        
        # Step 2: Analyze each version
        logger.info("Step 2: Analyzing feature relevance for each version...")
        analysis_results = {}
        
        for version_name, version_data in versions.items():
            logger.info(f"Analyzing version: {version_name}")
            
            # Get feature matrix
            X = feature_versions.get_feature_matrix(version_name)
            
            # Align target with features (remove NaN values)
            valid_idx = ~(X.isna().any(axis=1) | target.isna())
            X_clean = X[valid_idx]
            y_clean = target[valid_idx]
            
            if len(X_clean) == 0:
                logger.warning(f"No valid data for version {version_name}")
                continue
            
            logger.info(f"Analyzing {X_clean.shape[1]} features with {X_clean.shape[0]} samples")
            
            # Run comprehensive analysis
            try:
                results = self.analyzer.comprehensive_analysis(
                    X_clean, y_clean, self.task_type
                )
                analysis_results[version_name] = results
                logger.info(f"Completed analysis for {version_name}")
            except Exception as e:
                logger.error(f"Error analyzing {version_name}: {e}")
                analysis_results[version_name] = {}
        
        # Step 3: Generate comparison report
        logger.info("Step 3: Generating comparison report...")
        report = self.report_generator.generate_comprehensive_report(
            analysis_results, feature_versions, save_plots=True
        )
        
        # Step 4: Generate markdown report
        logger.info("Step 4: Generating markdown report...")
        markdown_report = self.report_generator.generate_markdown_report(
            analysis_results, feature_versions
        )
        
        # Save markdown report
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        markdown_path = Path(self.report_generator.output_dir) / f'feature_comparison_report_{timestamp}.md'
        with open(markdown_path, 'w') as f:
            f.write(markdown_report)
        
        logger.info(f"Markdown report saved to: {markdown_path}")
        
        # Return complete results
        complete_results = {
            'feature_versions': feature_versions,
            'analysis_results': analysis_results,
            'report': report,
            'markdown_report': markdown_report,
            'markdown_path': str(markdown_path)
        }
        
        logger.info("Complete feature comparison analysis finished!")
        return complete_results
    
    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a summary of the analysis results.
        
        Args:
            results: Results from run_complete_analysis
        """
        print("\n" + "="*80)
        print("FEATURE COMPARISON ANALYSIS SUMMARY")
        print("="*80)
        
        # Feature counts
        feature_versions = results['feature_versions']
        version_info = feature_versions.get_version_info()
        
        print("\nFeature Counts by Version:")
        print("-" * 40)
        for version_name, info in version_info.items():
            print(f"{version_name:25}: {info['n_features']:4d} features")
        
        # Performance summary
        analysis_results = results['analysis_results']
        print("\nPerformance Summary (LGBM R² Score):")
        print("-" * 40)
        for version_name, analysis in analysis_results.items():
            if 'lgbm_shap' in analysis and 'performance' in analysis['lgbm_shap']:
                r2 = analysis['lgbm_shap']['performance'].get('r2', 'N/A')
                print(f"{version_name:25}: {r2:.4f}")
            else:
                print(f"{version_name:25}: N/A")
        
        # Top features
        print("\nTop 5 Features by Version (Combined Ranking):")
        print("-" * 60)
        for version_name, analysis in analysis_results.items():
            if 'combined_ranking' in analysis and not analysis['combined_ranking'].empty:
                top_5 = analysis['combined_ranking'].head(5)
                print(f"\n{version_name}:")
                for idx, row in top_5.iterrows():
                    print(f"  {idx+1}. {row['feature']} (rank: {row['avg_rank']:.2f})")
        
        print("\n" + "="*80)
        print(f"Detailed report saved to: {results['markdown_path']}")
        print("="*80)

def main():
    """Main function to run the feature comparison analysis."""
    print("Starting Feature Engineering Comparison Analysis...")
    print("Comparing: Initial vs VWAP-based vs Vol-normalized vs VWAP+Vol-normalized")
    print("Using methods: LGBM/SHAP, LASSO, Mutual Information, Correlation")
    print("-" * 80)
    
    # Initialize runner
    runner = FeatureComparisonRunner(task_type='regression')
    
    # Run analysis
    try:
        results = runner.run_complete_analysis()
        
        # Print summary
        runner.print_summary(results)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main analysis: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())