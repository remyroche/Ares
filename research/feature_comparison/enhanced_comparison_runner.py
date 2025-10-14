"""
Enhanced Feature Comparison Runner

This module provides an enhanced comparison runner that uses standardized
feature definitions and handles consolidation and validation.
"""

import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from .standardized_features import StandardizedFeatureGenerator
from .feature_consolidation import FeatureConsolidator, FeatureValidator
from .relevance_analyzer import RelevanceAnalyzer
from .comparison_report import ComparisonReport

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class EnhancedFeatureComparisonRunner:
    """
    Enhanced feature comparison runner with standardized definitions and consolidation.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None, 
                 target_col: str = 'ret_t1',
                 task_type: str = 'regression',
                 scaling_method: str = 'robust',
                 enable_consolidation: bool = True,
                 enable_validation: bool = True):
        """
        Initialize enhanced feature comparison runner.
        
        Args:
            data: Input DataFrame with OHLCV data
            target_col: Name of target column
            task_type: 'regression' or 'classification'
            scaling_method: Scaling method for robust analysis
            enable_consolidation: Whether to enable feature consolidation
            enable_validation: Whether to enable feature validation
        """
        self.data = data
        self.target_col = target_col
        self.task_type = task_type
        self.scaling_method = scaling_method
        self.enable_consolidation = enable_consolidation
        self.enable_validation = enable_validation
        
        # Initialize components
        self.feature_generator = None
        self.consolidator = FeatureConsolidator() if enable_consolidation else None
        self.validator = FeatureValidator() if enable_validation else None
        self.analyzer = RelevanceAnalyzer(scaling_method=scaling_method)
        self.report_generator = ComparisonReport()
        
        self.versions = {}
        self.analysis_results = {}
        self.consolidation_summary = {}
        self.validation_summary = {}
    
    def load_sample_data(self, n_samples: int = 2000) -> pd.DataFrame:
        """
        Load or generate sample data for testing.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Sample DataFrame with OHLCV data
        """
        logger.info(f"Generating sample data with {n_samples} samples...")
        
        # Generate synthetic OHLCV data with realistic patterns
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
    
    def run_enhanced_analysis(self, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Run enhanced feature comparison analysis.
        
        Args:
            data: Input DataFrame (if None, will use self.data or generate sample)
            
        Returns:
            Dictionary with complete analysis results
        """
        logger.info("Starting enhanced feature comparison analysis...")
        
        # Use provided data or load sample data
        if data is not None:
            self.data = data
        elif self.data is None:
            self.data = self.load_sample_data()
        
        logger.info(f"Using data with shape: {self.data.shape}")
        
        # Step 1: Generate standardized features
        logger.info("Step 1: Generating standardized features...")
        self.feature_generator = StandardizedFeatureGenerator(
            self.data, enable_matrix_ops=True
        )
        
        # Generate all feature versions
        self.versions = self.feature_generator.generate_standardized_features()
        logger.info(f"Generated {len(self.versions)} standardized feature versions")
        
        # Step 2: Consolidate and validate features
        if self.enable_consolidation or self.enable_validation:
            logger.info("Step 2: Consolidating and validating features...")
            self._consolidate_and_validate_features()
        
        # Step 3: Create target variable
        logger.info("Step 3: Creating target variable...")
        target = self._create_target_variable()
        logger.info(f"Created target variable with {target.notna().sum()} valid values")
        
        # Step 4: Analyze each version
        logger.info("Step 4: Analyzing feature relevance for each version...")
        self.analysis_results = {}
        
        for version_name, version_data in self.versions.items():
            logger.info(f"Analyzing version: {version_name}")
            
            # Get feature matrix
            X = self._get_feature_matrix(version_name)
            
            # Align target with features (remove NaN values)
            valid_idx = ~(X.isna().any(axis=1) | target.isna())
            X_clean = X[valid_idx]
            y_clean = target[valid_idx]
            
            if len(X_clean) == 0:
                logger.warning(f"No valid data for version {version_name}")
                continue
            
            logger.info(f"Analyzing {X_clean.shape[1]} features with {X_clean.shape[0]} samples")
            
            # Run robust comprehensive analysis
            try:
                results = self.analyzer.robust_comprehensive_analysis(
                    X_clean, y_clean, self.task_type,
                    include_bootstrap=True,
                    include_temporal=True,
                    n_bootstrap=10,
                    n_temporal_windows=5
                )
                self.analysis_results[version_name] = results
                logger.info(f"Completed analysis for {version_name}")
            except Exception as e:
                logger.error(f"Error analyzing {version_name}: {e}")
                self.analysis_results[version_name] = {}
        
        # Step 5: Generate comparison report
        logger.info("Step 5: Generating comparison report...")
        report = self._generate_enhanced_report()
        
        # Step 6: Generate markdown report
        logger.info("Step 6: Generating markdown report...")
        markdown_report = self._generate_markdown_report()
        
        # Save markdown report
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        markdown_path = Path(self.report_generator.output_dir) / f'enhanced_feature_comparison_report_{timestamp}.md'
        with open(markdown_path, 'w') as f:
            f.write(markdown_report)
        
        logger.info(f"Enhanced markdown report saved to: {markdown_path}")
        
        # Return complete results
        complete_results = {
            'feature_versions': self.versions,
            'analysis_results': self.analysis_results,
            'consolidation_summary': self.consolidation_summary,
            'validation_summary': self.validation_summary,
            'report': report,
            'markdown_report': markdown_report,
            'markdown_path': str(markdown_path),
            'feature_definitions': self.feature_generator.get_feature_definitions()
        }
        
        logger.info("Enhanced feature comparison analysis completed!")
        return complete_results
    
    def _consolidate_and_validate_features(self):
        """Consolidate and validate features."""
        if self.enable_consolidation:
            logger.info("Consolidating features...")
            for version_name, version_data in self.versions.items():
                # Consolidate features
                consolidated_data = self.consolidator.consolidate_features(version_data, version_name)
                
                # Remove multicollinearity
                cleaned_data = self.consolidator.remove_multicollinearity(consolidated_data, version_name)
                
                # Winsorize features
                winsorized_data = self.consolidator.winsorize_features(cleaned_data)
                
                # Update version
                self.versions[version_name] = winsorized_data
            
            # Get consolidation summary
            self.consolidation_summary = self.consolidator.get_consolidation_summary()
            logger.info(f"Consolidation completed. Removed {self.consolidation_summary['total_removed']} features")
        
        if self.enable_validation:
            logger.info("Validating features...")
            for version_name, version_data in self.versions.items():
                self.validator.validate_features(version_data, version_name)
            
            # Get validation summary
            self.validation_summary = self.validator.get_validation_summary()
            logger.info("Feature validation completed")
    
    def _create_target_variable(self) -> pd.Series:
        """Create target variable using standardized returns."""
        # Use the standardized returns as target
        if 'ret_t1' in self.versions['initial'].columns:
            return self.versions['initial']['ret_t1']
        else:
            # Fallback to calculating returns
            return self.data['close'].pct_change()
    
    def _get_feature_matrix(self, version: str) -> pd.DataFrame:
        """Get feature matrix for a specific version."""
        if version not in self.versions:
            raise ValueError(f"Unknown version: {version}")
        
        df = self.versions[version].copy()
        
        # Exclude non-feature columns
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        return df[feature_cols]
    
    def _generate_enhanced_report(self) -> Dict[str, Any]:
        """Generate enhanced comparison report."""
        # Create a mock FeatureVersions object for compatibility
        class MockFeatureVersions:
            def __init__(self, versions):
                self.versions = versions
            
            def get_version_info(self):
                info = {}
                for version_name, version_df in self.versions.items():
                    feature_cols = [col for col in version_df.columns 
                                  if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
                    info[version_name] = {
                        'n_features': len(feature_cols),
                        'feature_names': list(feature_cols),
                        'n_samples': len(version_df),
                        'has_nan': version_df.isna().any().any(),
                        'nan_count': version_df.isna().sum().sum()
                    }
                return info
        
        mock_feature_versions = MockFeatureVersions(self.versions)
        
        # Generate report
        report = self.report_generator.generate_comprehensive_report(
            self.analysis_results, mock_feature_versions, save_plots=True
        )
        
        # Add enhanced information
        report['enhanced_info'] = {
            'consolidation_summary': self.consolidation_summary,
            'validation_summary': self.validation_summary,
            'feature_definitions': self.feature_generator.get_feature_definitions(),
            'standardization_applied': True,
            'consolidation_applied': self.enable_consolidation,
            'validation_applied': self.enable_validation
        }
        
        return report
    
    def _generate_markdown_report(self) -> str:
        """Generate enhanced markdown report."""
        report_lines = []
        
        # Header
        report_lines.append("# Enhanced Feature Comparison Report")
        report_lines.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Enhanced features section
        report_lines.append("## Enhanced Features")
        report_lines.append("")
        report_lines.append("### Standardized Naming Conventions")
        report_lines.append("- `ret_t(h)` = log(P_t / P_{t-h})")
        report_lines.append("- `vwap_t` = volume-weighted average over window W")
        report_lines.append("- `vol_t(W)` = realized vol proxy (std of returns over W)")
        report_lines.append("- `_normvolW` → divided by vol_t(W)")
        report_lines.append("- `_zcs` → cross-sectional z-score at time t")
        report_lines.append("- `_ewmA` → EWMA with span A")
        report_lines.append("- `_wW` → rolling window W")
        report_lines.append("- `_leadH` / `_lagH` → H-step lead/lag")
        report_lines.append("")
        
        # Feature consolidation section
        if self.consolidation_summary:
            report_lines.append("### Feature Consolidation")
            report_lines.append(f"- Total features removed: {self.consolidation_summary['total_removed']}")
            report_lines.append("- Redundancy removal applied")
            report_lines.append("- Multicollinearity screening applied")
            report_lines.append("")
        
        # Validation section
        if self.validation_summary:
            report_lines.append("### Feature Validation")
            report_lines.append(f"- Total versions validated: {self.validation_summary['total_versions']}")
            report_lines.append(f"- Total warnings: {self.validation_summary['overall_quality']['total_warnings']}")
            report_lines.append(f"- Total errors: {self.validation_summary['overall_quality']['total_errors']}")
            report_lines.append("")
        
        # Version comparison
        report_lines.append("## Version Comparison")
        report_lines.append("")
        
        for version_name, analysis in self.analysis_results.items():
            report_lines.append(f"### {version_name.replace('_', ' ').title()}")
            
            # Get version info
            if version_name in self.versions:
                version_df = self.versions[version_name]
                feature_cols = [col for col in version_df.columns 
                              if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
                report_lines.append(f"- Features: {len(feature_cols)}")
                report_lines.append(f"- Samples: {len(version_df)}")
            
            # Performance metrics
            if 'lgbm_shap' in analysis and 'performance' in analysis['lgbm_shap']:
                perf = analysis['lgbm_shap']['performance']
                report_lines.append("**LGBM Performance:**")
                for metric, value in perf.items():
                    report_lines.append(f"- {metric}: {value:.4f}")
                report_lines.append("")
            
            # Top features
            if 'combined_ranking' in analysis and not analysis['combined_ranking'].empty:
                top_10 = analysis['combined_ranking'].head(10)
                report_lines.append("**Top 10 Features:**")
                report_lines.append("| Rank | Feature | Average Rank |")
                report_lines.append("|------|---------|--------------|")
                for idx, row in top_10.iterrows():
                    report_lines.append(f"| {idx + 1} | {row['feature']} | {row['avg_rank']:.2f} |")
                report_lines.append("")
        
        # Feature definitions
        if hasattr(self.feature_generator, 'get_feature_definitions'):
            definitions = self.feature_generator.get_feature_definitions()
            report_lines.append("## Feature Definitions")
            report_lines.append("")
            for feature, definition in list(definitions.items())[:20]:  # Show first 20
                report_lines.append(f"- **{feature}**: {definition}")
            report_lines.append("")
            report_lines.append(f"... and {len(definitions) - 20} more features")
            report_lines.append("")
        
        return "\n".join(report_lines)
    
    def print_enhanced_summary(self, results: Dict[str, Any]) -> None:
        """Print enhanced summary of the analysis results."""
        print("\n" + "="*80)
        print("ENHANCED FEATURE COMPARISON ANALYSIS SUMMARY")
        print("="*80)
        
        # Standardization info
        print("\nStandardization Applied:")
        print("-" * 40)
        print("✅ Standardized naming conventions")
        print("✅ Returns-based calculations (no raw prices)")
        print("✅ Explicit window specifications")
        print("✅ Consolidated redundant features")
        print("✅ Multicollinearity screening")
        print("✅ Feature validation")
        
        # Version comparison
        print("\nFeature Counts by Version:")
        print("-" * 40)
        for version_name, version_df in self.versions.items():
            feature_cols = [col for col in version_df.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            print(f"{version_name:25}: {len(feature_cols):4d} features")
        
        # Performance summary
        print("\nPerformance Summary (LGBM R² Score):")
        print("-" * 40)
        for version_name, analysis in self.analysis_results.items():
            if 'lgbm_shap' in analysis and 'performance' in analysis['lgbm_shap']:
                r2 = analysis['lgbm_shap']['performance'].get('r2', 'N/A')
                print(f"{version_name:25}: {r2:.4f}")
            else:
                print(f"{version_name:25}: N/A")
        
        # Robust evaluation metrics
        print("\nRobust Evaluation Metrics:")
        print("-" * 60)
        for version_name, analysis in self.analysis_results.items():
            print(f"\n{version_name}:")
            
            # Rank correlations
            if 'rank_correlations' in analysis:
                rank_corr = analysis['rank_correlations']
                mean_corr = rank_corr.get('mean_correlation', 0)
                print(f"  Mean Rank Correlation: {mean_corr:.3f}")
            
            # Bootstrap stability
            if 'bootstrap_analysis' in analysis and 'method_results' in analysis['bootstrap_analysis']:
                bootstrap = analysis['bootstrap_analysis']['method_results']
                for method, method_results in bootstrap.items():
                    mean_cv = method_results.get('cv_importance', pd.Series()).mean()
                    print(f"  {method.upper()} Mean CV: {mean_cv:.3f}")
            
            # Temporal stability
            if 'temporal_stability' in analysis and 'stability_metrics' in analysis['temporal_stability']:
                temporal = analysis['temporal_stability']['stability_metrics']
                mean_stability = temporal.get('mean_stability', 0)
                stable_count = len(temporal.get('stable_features', []))
                print(f"  Mean Temporal Stability: {mean_stability:.3f}")
                print(f"  Stable Features Count: {stable_count}")
        
        # Consolidation summary
        if self.consolidation_summary:
            print(f"\nFeature Consolidation:")
            print("-" * 40)
            print(f"Total features removed: {self.consolidation_summary['total_removed']}")
            for version, removed in self.consolidation_summary['removed_features'].items():
                if removed:
                    print(f"  {version}: {len(removed)} features removed")
        
        print("\n" + "="*80)
        print(f"Enhanced report saved to: {results['markdown_path']}")
        print("="*80)

def main():
    """Main function to run the enhanced feature comparison analysis."""
    print("Enhanced Feature Comparison Framework")
    print("=" * 60)
    print("Features:")
    print("- Standardized naming conventions")
    print("- Returns-based calculations")
    print("- Feature consolidation and validation")
    print("- Multicollinearity screening")
    print("- Robust evaluation with 10 bootstrap samples")
    print("=" * 60)
    
    # Initialize runner
    runner = EnhancedFeatureComparisonRunner(
        task_type='regression',
        scaling_method='robust',
        enable_consolidation=True,
        enable_validation=True
    )
    
    # Run analysis
    try:
        results = runner.run_enhanced_analysis()
        
        # Print summary
        runner.print_enhanced_summary(results)
        
        print("\nEnhanced analysis completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in enhanced analysis: {e}")
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())