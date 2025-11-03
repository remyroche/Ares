#!/usr/bin/env python3
"""
SR Feature Engineering Investigation Script

This script analyzes the current feature set used in SR quality prediction
and provides insights into feature importance, missing features, and opportunities
for feature engineering improvements.

Usage:
    python scripts/investigate_sr_features.py --training-data data_cache/sr_training_data.parquet
    python scripts/investigate_sr_features.py --model models/sr_quality_model.lgb
    python scripts/investigate_sr_features.py --training-data data_cache/sr_training_data.parquet --analyze-missing
"""

import sys
import argparse
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger

logger = system_logger.getChild('FeatureInvestigation')


class SRFeatureInvestigator:
    """Investigates SR quality prediction features."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.training_data = None
        self.model = None
        self.feature_importance = None
        
    def load_training_data(self, path: str):
        """Load training data from parquet file."""
        self.logger.info(f"📥 Loading training data from: {path}")
        
        if not Path(path).exists():
            raise FileNotFoundError(f"Training data not found: {path}")
        
        self.training_data = pd.read_parquet(path)
        self.logger.info(f"✅ Loaded {len(self.training_data)} training samples")
        self.logger.info(f"   Columns: {len(self.training_data.columns)}")
        
    def load_model(self, path: str):
        """Load trained model to extract feature importance."""
        self.logger.info(f"📥 Loading model from: {path}")
        
        try:
            import lightgbm as lgb
            from src.tactician.sr_levels.ml_quality import SRQualityModel
            
            model = SRQualityModel()
            model.load(path)
            self.model = model
            
            self.logger.info(f"✅ Model loaded successfully")
            self.logger.info(f"   Features: {len(model.feature_names)}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load model: {e}")
            raise
            
    def analyze_current_features(self) -> Dict:
        """Analyze current feature set in training data."""
        if self.training_data is None:
            raise ValueError("No training data loaded. Call load_training_data() first.")
        
        self.logger.info("\n" + "="*80)
        self.logger.info("📊 CURRENT FEATURE ANALYSIS")
        self.logger.info("="*80)
        
        # Identify feature columns
        feature_cols = [col for col in self.training_data.columns if col.startswith('feature_')]
        non_feature_cols = [col for col in self.training_data.columns if not col.startswith('feature_')]
        
        self.logger.info(f"\n📋 Feature Columns: {len(feature_cols)}")
        self.logger.info(f"   Non-feature Columns: {len(non_feature_cols)}")
        
        # Categorize features
        feature_categories = self._categorize_features(feature_cols)
        
        self.logger.info(f"\n📂 Feature Categories:")
        for category, features in feature_categories.items():
            self.logger.info(f"   {category}: {len(features)} features")
        
        # Print all features by category
        self.logger.info(f"\n📝 All Features by Category:\n")
        for category, features in feature_categories.items():
            self.logger.info(f"\n{category.upper()} ({len(features)} features):")
            for feature in sorted(features):
                self.logger.info(f"  - {feature}")
        
        # Print non-feature columns
        self.logger.info(f"\n📌 Non-Feature Columns ({len(non_feature_cols)}):")
        for col in sorted(non_feature_cols):
            self.logger.info(f"  - {col}")
        
        # Feature statistics
        self.logger.info(f"\n📊 Feature Statistics:")
        for col in feature_cols[:10]:  # Show first 10
            self.logger.info(f"  {col:<40} mean={self.training_data[col].mean():.4f}, std={self.training_data[col].std():.4f}")
        
        return {
            'total_features': len(feature_cols),
            'feature_cols': feature_cols,
            'feature_categories': feature_categories,
            'non_feature_cols': non_feature_cols
        }
    
    def _categorize_features(self, feature_cols: List[str]) -> Dict[str, List[str]]:
        """Categorize features by type."""
        categories = {
            'Basic SR': [],
            'Bounce Metrics': [],
            'Volume': [],
            'Temporal': [],
            'Market Context': [],
            'Position': [],
            'Interaction': [],
            'Multi-TF': [],
            'Regime': [],
            'Statistical': [],
            'Method Confluence': [],
            'Other': []
        }
        
        for feature in feature_cols:
            feature_lower = feature.lower()
            
            # Categorize by keywords
            if any(kw in feature_lower for kw in ['strength', 'prominence', 'width', 'consistency', 'touch_count', 'failure']):
                categories['Basic SR'].append(feature)
            elif any(kw in feature_lower for kw in ['bounce', 'rejection']):
                categories['Bounce Metrics'].append(feature)
            elif 'volume' in feature_lower:
                categories['Volume'].append(feature)
            elif any(kw in feature_lower for kw in ['age', 'recency', 'time', 'decay', 'dwell', 'hour', 'day']):
                categories['Temporal'].append(feature)
            elif any(kw in feature_lower for kw in ['market', 'volatility', 'trend', 'momentum']):
                categories['Market Context'].append(feature)
            elif any(kw in feature_lower for kw in ['price_position', 'distance', 'percentile', 'zscore']):
                categories['Position'].append(feature)
            elif '_x_' in feature_lower or 'interaction' in feature_lower:
                categories['Interaction'].append(feature)
            elif 'multi_tf' in feature_lower or 'confirmation' in feature_lower:
                categories['Multi-TF'].append(feature)
            elif any(kw in feature_lower for kw in ['regime', 'vol_adjusted', 'trend_alignment']):
                categories['Regime'].append(feature)
            elif any(kw in feature_lower for kw in ['zscore', 'percentile', 'statistical']):
                categories['Statistical'].append(feature)
            elif any(kw in feature_lower for kw in ['method', 'confluence', 'agreement', 'diversity']):
                categories['Method Confluence'].append(feature)
            else:
                categories['Other'].append(feature)
        
        return categories
    
    def analyze_feature_importance(self, top_n: int = 30) -> pd.DataFrame:
        """Analyze and display feature importance from trained model."""
        if self.model is None:
            raise ValueError("No model loaded. Call load_model() first.")
        
        self.logger.info("\n" + "="*80)
        self.logger.info(f"🏆 TOP {top_n} FEATURE IMPORTANCE")
        self.logger.info("="*80)
        
        # Extract feature importance
        importance = self.model.model.feature_importance(importance_type='gain')
        
        importance_df = pd.DataFrame({
            'feature': self.model.feature_names,
            'importance': importance,
            'importance_pct': importance / importance.sum() * 100
        }).sort_values('importance', ascending=False)
        
        # Display top N
        self.logger.info(f"\n{'Rank':<6}{'Feature':<45}{'Importance':<12}{'%':<8}")
        self.logger.info("-" * 80)
        
        for idx, (_, row) in enumerate(importance_df.head(top_n).iterrows(), 1):
            self.logger.info(f"{idx:<6}{row['feature']:<45}{row['importance']:>10.0f}  {row['importance_pct']:>6.1f}%")
        
        # Category-wise importance
        self.logger.info(f"\n📊 Feature Importance by Category:")
        feature_categories = self._categorize_features(self.model.feature_names)
        
        category_importance = {}
        for category, features in feature_categories.items():
            cat_importance = importance_df[importance_df['feature'].isin(features)]['importance_pct'].sum()
            category_importance[category] = cat_importance
        
        for category, importance_pct in sorted(category_importance.items(), key=lambda x: x[1], reverse=True):
            if importance_pct > 0:
                self.logger.info(f"  {category:<25} {importance_pct:>6.1f}%")
        
        self.feature_importance = importance_df
        return importance_df
    
    def identify_missing_features(self) -> Dict[str, List[str]]:
        """Identify missing high-impact features."""
        if self.training_data is None:
            raise ValueError("No training data loaded.")
        
        self.logger.info("\n" + "="*80)
        self.logger.info("🔍 MISSING FEATURE ANALYSIS")
        self.logger.info("="*80)
        
        current_features = set([col for col in self.training_data.columns if col.startswith('feature_')])
        
        # Define desired high-impact features
        desired_features = {
            'Temporal Features': [
                'feature_time_since_formation',
                'feature_level_age_days',
                'feature_bars_since_last_touch',
                'feature_touch_frequency',  # touches / age
                'feature_avg_time_between_touches',
                'feature_recent_touch_rate',  # touches in last N bars
            ],
            'Market Regime Features': [
                'feature_regime_volatility',  # current_vol / avg_vol
                'feature_regime_trend_strength',
                'feature_distance_to_price_pct',
                'feature_distance_to_price_atr',  # Distance in ATR units
                'feature_volume_regime',  # current_volume / avg_volume
            ],
            'Relative Ranking Features': [
                'feature_strength_percentile',  # vs all levels
                'feature_touches_percentile',
                'feature_distance_to_nearest_support',
                'feature_distance_to_nearest_resistance',
                'feature_level_density_nearby',  # how crowded
            ],
            'Temporal Decay Features': [
                'feature_recency_weighted_touches',
                'feature_exponential_decay_30',
                'feature_exponential_decay_100',
                'feature_time_weighted_strength',
            ],
            'Statistical Significance Features': [
                'feature_volume_spike_ratio',  # level_volume / avg_volume
                'feature_price_reaction_strength',  # bounce / avg_bounce
                'feature_volume_profile_score',
                'feature_price_action_quality',
            ],
            'Interaction Features (Advanced)': [
                'feature_touches_x_recency',
                'feature_volume_x_proximity',
                'feature_strength_x_volatility_regime',
                'feature_quality_composite',  # touches * strength * recency
            ],
            'Level Quality Clusters': [
                'feature_is_top_10_pct',
                'feature_is_top_20_pct',
                'feature_quality_tier',  # 0-3 (weak to critical)
                'feature_relative_strength_rank',
            ]
        }
        
        missing_by_category = {}
        
        for category, features in desired_features.items():
            missing = [f for f in features if f not in current_features]
            if missing:
                missing_by_category[category] = missing
        
        # Report missing features
        self.logger.info(f"\n📋 Missing High-Impact Features:\n")
        
        total_missing = 0
        for category, missing in missing_by_category.items():
            self.logger.info(f"\n{category}:")
            for feature in missing:
                self.logger.info(f"  ❌ {feature}")
                total_missing += 1
        
        if total_missing == 0:
            self.logger.info("✅ All desired features are present!")
        else:
            self.logger.info(f"\n⚠️  Total missing features: {total_missing}")
        
        # Check which desired features exist
        self.logger.info(f"\n✅ Existing Desired Features:\n")
        for category, features in desired_features.items():
            existing = [f for f in features if f in current_features]
            if existing:
                self.logger.info(f"\n{category}:")
                for feature in existing:
                    self.logger.info(f"  ✓ {feature}")
        
        return missing_by_category
    
    def plot_feature_importance(self, output_path: str = 'outcomes/feature_importance.png', top_n: int = 20):
        """Plot feature importance chart."""
        if self.feature_importance is None:
            self.logger.warning("No feature importance available. Run analyze_feature_importance() first.")
            return
        
        self.logger.info(f"\n📊 Generating feature importance plot...")
        
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(top_n)
        
        plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance (Gain)', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - SR Quality Model', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Plot saved to: {output_path}")
    
    def generate_feature_correlation_heatmap(self, output_path: str = 'outcomes/feature_correlation.png', top_n: int = 20):
        """Generate correlation heatmap for top features."""
        if self.training_data is None or self.feature_importance is None:
            self.logger.warning("Missing data. Load training data and analyze importance first.")
            return
        
        self.logger.info(f"\n📊 Generating feature correlation heatmap...")
        
        # Get top N features
        top_features = self.feature_importance.head(top_n)['feature'].tolist()
        
        # Calculate correlation matrix
        corr_matrix = self.training_data[top_features].corr()
        
        # Plot heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(f'Feature Correlation Heatmap - Top {top_n} Features', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"✅ Correlation heatmap saved to: {output_path}")
    
    def generate_comprehensive_report(self, output_path: str = 'outcomes/feature_engineering_report.md'):
        """Generate comprehensive feature engineering report."""
        if self.training_data is None:
            raise ValueError("No training data loaded.")
        
        self.logger.info(f"\n📄 Generating comprehensive feature engineering report...")
        
        from datetime import datetime
        
        report_lines = [
            "# SR Feature Engineering Investigation Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
        ]
        
        # Current features summary
        feature_cols = [col for col in self.training_data.columns if col.startswith('feature_')]
        report_lines.extend([
            f"- **Total Features:** {len(feature_cols)}",
            f"- **Training Samples:** {len(self.training_data):,}",
            "",
        ])
        
        # Feature categories
        feature_categories = self._categorize_features(feature_cols)
        report_lines.extend([
            "## Feature Categories",
            "",
        ])
        
        for category, features in sorted(feature_categories.items(), key=lambda x: len(x[1]), reverse=True):
            if features:
                report_lines.extend([
                    f"### {category} ({len(features)} features)",
                    "",
                ])
                for feature in sorted(features):
                    report_lines.append(f"- `{feature}`")
                report_lines.append("")
        
        # Feature importance (if available)
        if self.feature_importance is not None:
            report_lines.extend([
                "## Top 30 Most Important Features",
                "",
                "| Rank | Feature | Importance | % |",
                "|------|---------|------------|---|",
            ])
            
            for idx, (_, row) in enumerate(self.feature_importance.head(30).iterrows(), 1):
                report_lines.append(f"| {idx} | `{row['feature']}` | {row['importance']:.0f} | {row['importance_pct']:.1f}% |")
            
            report_lines.append("")
        
        # Missing features
        missing = self.identify_missing_features()
        if missing:
            report_lines.extend([
                "## Missing High-Impact Features",
                "",
                "The following features could potentially improve model performance:",
                "",
            ])
            
            for category, features in missing.items():
                report_lines.extend([
                    f"### {category}",
                    "",
                ])
                for feature in features:
                    report_lines.append(f"- [ ] `{feature}`")
                report_lines.append("")
        
        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            "### High Priority",
            "",
            "1. **Add Temporal Decay Features** - Recency matters for SR quality",
            "   - `feature_recency_weighted_touches`",
            "   - `feature_exponential_decay_30`",
            "   - `feature_time_weighted_strength`",
            "",
            "2. **Add Market Regime Features** - Context is critical",
            "   - `feature_regime_volatility` (current_vol / avg_vol)",
            "   - `feature_regime_trend_strength`",
            "   - `feature_distance_to_price_atr`",
            "",
            "3. **Add Relative Ranking Features** - Compare levels to each other",
            "   - `feature_strength_percentile`",
            "   - `feature_touches_percentile`",
            "   - `feature_level_density_nearby`",
            "",
            "### Medium Priority",
            "",
            "4. **Add Statistical Significance Features**",
            "   - `feature_volume_spike_ratio`",
            "   - `feature_price_reaction_strength`",
            "",
            "5. **Add Advanced Interaction Features**",
            "   - `feature_touches_x_recency`",
            "   - `feature_volume_x_proximity`",
            "   - `feature_quality_composite`",
            "",
            "### Implementation Notes",
            "",
            "- Update `src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`",
            "- Add feature extraction in `_extract_all_features()` method",
            "- Retrain model after adding new features",
            "- Compare model performance before/after",
            "",
        ])
        
        # Write report
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"✅ Report saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Investigate SR quality prediction features",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--training-data', type=str, 
                       help='Path to training data parquet file')
    parser.add_argument('--model', type=str,
                       help='Path to trained model file')
    parser.add_argument('--analyze-missing', action='store_true',
                       help='Analyze missing high-impact features')
    parser.add_argument('--generate-plots', action='store_true',
                       help='Generate feature importance plots')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate comprehensive feature engineering report')
    parser.add_argument('--top-n', type=int, default=30,
                       help='Number of top features to display (default: 30)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.training_data and not args.model:
        parser.error("At least one of --training-data or --model must be provided")
    
    # Initialize investigator
    investigator = SRFeatureInvestigator()
    
    # Load data
    if args.training_data:
        investigator.load_training_data(args.training_data)
        investigator.analyze_current_features()
    
    # Load model
    if args.model:
        investigator.load_model(args.model)
        investigator.analyze_feature_importance(top_n=args.top_n)
    
    # Analyze missing features
    if args.analyze_missing:
        if args.training_data:
            investigator.identify_missing_features()
        else:
            logger.error("--analyze-missing requires --training-data")
    
    # Generate plots
    if args.generate_plots:
        if args.model:
            investigator.plot_feature_importance(top_n=20)
        if args.training_data and args.model:
            investigator.generate_feature_correlation_heatmap(top_n=20)
    
    # Generate report
    if args.generate_report:
        if args.training_data:
            investigator.generate_comprehensive_report()
        else:
            logger.error("--generate-report requires --training-data")
    
    logger.info("\n" + "="*80)
    logger.info("✅ FEATURE INVESTIGATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()

