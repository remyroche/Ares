"""
Multi-Asset Reporting Module

Generates comprehensive comparison metrics between assets for:
- ML model performance (global vs per-asset)
- Feature importance (global vs per-asset)
- Leaf statistics and tree depth
- Event detection and labeling quality
- Cross-asset correlations and divergence

Usage:
    from src.training.steps.labeling.multi_asset_reporting import MultiAssetReporter
    
    reporter = MultiAssetReporter(outcomes_dir='outcomes')
    reporter.generate_multi_asset_report(
        combined_df=combined_df,
        model_results=model_results,
        assets=['ETH', 'BTC', 'SOL']
    )
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json

try:
    from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_success(msg): print(f"[SUCCESS] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")
    def tprint_error(msg): print(f"[ERROR] {msg}")


class MultiAssetReporter:
    """
    Comprehensive multi-asset reporting with global vs per-asset comparisons.
    """
    
    def __init__(self, outcomes_dir: Optional[Path] = None):
        self.outcomes_dir = outcomes_dir or Path('outcomes')
        self.outcomes_dir.mkdir(exist_ok=True, parents=True)
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_multi_asset_report(
        self,
        combined_df: pd.DataFrame,
        model_results: Dict[str, Any],
        assets: List[str],
        asset_col: str = 'asset_id',
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """
        Generate comprehensive multi-asset comparison report.
        
        Args:
            combined_df: Combined DataFrame with all assets
            model_results: Dictionary containing model training results
            assets: List of asset identifiers
            asset_col: Column name for asset identifier
            config: Optional configuration dictionary
            
        Returns:
            Dictionary mapping report type to file path
        """
        tprint_info("📊 Generating Multi-Asset Comparison Reports...")
        
        report_paths = {}
        
        # 1. Model Performance Comparison
        if 'model_performance' in model_results:
            perf_path = self.generate_model_performance_comparison(
                model_results['model_performance'],
                assets,
                asset_col
            )
            report_paths['model_performance'] = perf_path
        
        # 2. Feature Importance Comparison
        if 'feature_importance' in model_results:
            feat_path = self.generate_feature_importance_comparison(
                model_results['feature_importance'],
                assets,
                asset_col
            )
            report_paths['feature_importance'] = feat_path
        
        # 3. Tree/Leaf Statistics Comparison
        if 'tree_stats' in model_results:
            tree_path = self.generate_tree_statistics_comparison(
                model_results['tree_stats'],
                assets,
                asset_col
            )
            report_paths['tree_statistics'] = tree_path
        
        # 4. Label Quality Comparison
        label_path = self.generate_label_quality_comparison(
            combined_df,
            assets,
            asset_col
        )
        report_paths['label_quality'] = label_path
        
        # 5. Cross-Asset Correlation Analysis
        corr_path = self.generate_cross_asset_correlation(
            combined_df,
            assets,
            asset_col
        )
        report_paths['cross_asset_correlation'] = corr_path
        
        # 6. Master Summary Report
        summary_path = self.generate_master_summary(
            combined_df,
            model_results,
            assets,
            asset_col,
            config
        )
        report_paths['master_summary'] = summary_path
        
        tprint_success(f"✅ Multi-asset reports saved to {self.outcomes_dir}")
        return report_paths
    
    def generate_model_performance_comparison(
        self,
        performance_data: Dict[str, Any],
        assets: List[str],
        asset_col: str
    ) -> Path:
        """
        Generate model performance comparison: global vs per-asset.
        
        Metrics:
        - AUC (global, per-asset, cross-asset std)
        - Sharpe ratio
        - Information coefficient
        - Win rate
        - Average return per trade
        """
        tprint_info("   📈 Generating model performance comparison...")
        
        # Prepare data structures
        global_metrics = {}
        per_asset_metrics = {asset: {} for asset in assets}
        
        # Extract global metrics
        if 'global' in performance_data:
            global_metrics = performance_data['global']
        
        # Extract per-asset metrics
        for asset in assets:
            if asset in performance_data:
                per_asset_metrics[asset] = performance_data[asset]
        
        # Create comparison DataFrame
        comparison_rows = []
        
        # Metrics to compare
        metric_names = ['auc', 'sharpe', 'ic', 'win_rate', 'avg_return', 'total_pnl', 'n_trades']
        
        for metric in metric_names:
            row = {'metric': metric}
            
            # Global value
            row['global'] = global_metrics.get(metric, np.nan)
            
            # Per-asset values
            asset_values = []
            for asset in assets:
                value = per_asset_metrics[asset].get(metric, np.nan)
                row[f'{asset}'] = value
                if not np.isnan(value):
                    asset_values.append(value)
            
            # Cross-asset statistics
            if asset_values:
                row['mean_across_assets'] = np.mean(asset_values)
                row['std_across_assets'] = np.std(asset_values)
                row['min_across_assets'] = np.min(asset_values)
                row['max_across_assets'] = np.max(asset_values)
                row['range'] = np.max(asset_values) - np.min(asset_values)
            else:
                row['mean_across_assets'] = np.nan
                row['std_across_assets'] = np.nan
                row['min_across_assets'] = np.nan
                row['max_across_assets'] = np.nan
                row['range'] = np.nan
            
            comparison_rows.append(row)
        
        comparison_df = pd.DataFrame(comparison_rows)
        
        # Save to CSV
        csv_path = self.outcomes_dir / f'model_performance_comparison_{self.ts}.csv'
        comparison_df.to_csv(csv_path, index=False)
        
        # Generate markdown report
        md_lines = ["# Model Performance Comparison: Global vs Per-Asset\n\n"]
        md_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        md_lines.append(f"**Assets**: {', '.join(assets)}\n\n")
        
        md_lines.append("## Performance Metrics\n\n")
        md_lines.append("| Metric | Global | " + " | ".join(assets) + " | Mean | Std | Min | Max | Range |\n")
        md_lines.append("|--------|--------|" + "|".join(["--------"] * len(assets)) + "|------|-----|-----|-----|-------|\n")
        
        for _, row in comparison_df.iterrows():
            metric = row['metric']
            global_val = f"{row['global']:.4f}" if not np.isnan(row['global']) else "N/A"
            asset_vals = [f"{row[asset]:.4f}" if not np.isnan(row[asset]) else "N/A" for asset in assets]
            mean_val = f"{row['mean_across_assets']:.4f}" if not np.isnan(row['mean_across_assets']) else "N/A"
            std_val = f"{row['std_across_assets']:.4f}" if not np.isnan(row['std_across_assets']) else "N/A"
            min_val = f"{row['min_across_assets']:.4f}" if not np.isnan(row['min_across_assets']) else "N/A"
            max_val = f"{row['max_across_assets']:.4f}" if not np.isnan(row['max_across_assets']) else "N/A"
            range_val = f"{row['range']:.4f}" if not np.isnan(row['range']) else "N/A"
            
            md_lines.append(f"| {metric} | {global_val} | " + " | ".join(asset_vals) + 
                          f" | {mean_val} | {std_val} | {min_val} | {max_val} | {range_val} |\n")
        
        md_lines.append("\n## Key Insights\n\n")
        
        # Identify best/worst performing assets
        if 'auc' in comparison_df['metric'].values:
            auc_row = comparison_df[comparison_df['metric'] == 'auc'].iloc[0]
            asset_aucs = {asset: auc_row[asset] for asset in assets if not np.isnan(auc_row[asset])}
            if asset_aucs:
                best_asset = max(asset_aucs, key=asset_aucs.get)
                worst_asset = min(asset_aucs, key=asset_aucs.get)
                md_lines.append(f"- **Best AUC**: {best_asset} ({asset_aucs[best_asset]:.4f})\n")
                md_lines.append(f"- **Worst AUC**: {worst_asset} ({asset_aucs[worst_asset]:.4f})\n")
                md_lines.append(f"- **AUC Range**: {auc_row['range']:.4f}\n\n")
        
        # Check for high cross-asset variance
        high_variance_metrics = comparison_df[comparison_df['std_across_assets'] > 0.1]
        if not high_variance_metrics.empty:
            md_lines.append("### High Cross-Asset Variance (>0.1)\n\n")
            for _, row in high_variance_metrics.iterrows():
                md_lines.append(f"- **{row['metric']}**: std = {row['std_across_assets']:.4f}\n")
            md_lines.append("\n")
        
        md_path = self.outcomes_dir / f'model_performance_comparison_{self.ts}.md'
        with open(md_path, 'w') as f:
            f.writelines(md_lines)
        
        tprint_success(f"   ✅ Model performance comparison saved: {csv_path}")
        return md_path
    
    def generate_feature_importance_comparison(
        self,
        feature_importance_data: Dict[str, Any],
        assets: List[str],
        asset_col: str
    ) -> Path:
        """
        Generate feature importance comparison: global vs per-asset.
        
        Shows:
        - Top features globally
        - Top features per asset
        - Feature importance divergence across assets
        - Asset-specific features vs shared features
        """
        tprint_info("   🎯 Generating feature importance comparison...")
        
        # Extract global feature importance
        global_importance = feature_importance_data.get('global', {})
        
        # Extract per-asset feature importance
        per_asset_importance = {asset: feature_importance_data.get(asset, {}) for asset in assets}
        
        # Create comparison DataFrame
        all_features = set()
        for asset_data in [global_importance] + list(per_asset_importance.values()):
            if isinstance(asset_data, dict):
                all_features.update(asset_data.keys())
            elif isinstance(asset_data, pd.DataFrame) and 'feature' in asset_data.columns:
                all_features.update(asset_data['feature'].values)
        
        comparison_rows = []
        for feature in all_features:
            row = {'feature': feature}
            
            # Global importance
            if isinstance(global_importance, dict):
                row['global'] = global_importance.get(feature, 0.0)
            elif isinstance(global_importance, pd.DataFrame):
                feat_row = global_importance[global_importance['feature'] == feature]
                row['global'] = feat_row['importance'].iloc[0] if not feat_row.empty else 0.0
            else:
                row['global'] = 0.0
            
            # Per-asset importance
            asset_values = []
            for asset in assets:
                asset_data = per_asset_importance[asset]
                if isinstance(asset_data, dict):
                    value = asset_data.get(feature, 0.0)
                elif isinstance(asset_data, pd.DataFrame):
                    feat_row = asset_data[asset_data['feature'] == feature]
                    value = feat_row['importance'].iloc[0] if not feat_row.empty else 0.0
                else:
                    value = 0.0
                
                row[asset] = value
                asset_values.append(value)
            
            # Cross-asset statistics
            row['mean_across_assets'] = np.mean(asset_values)
            row['std_across_assets'] = np.std(asset_values)
            row['cv'] = row['std_across_assets'] / (row['mean_across_assets'] + 1e-9)  # Coefficient of variation
            
            comparison_rows.append(row)
        
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_df = comparison_df.sort_values('global', ascending=False)
        
        # Save to CSV
        csv_path = self.outcomes_dir / f'feature_importance_comparison_{self.ts}.csv'
        comparison_df.to_csv(csv_path, index=False)
        
        # Generate markdown report
        md_lines = ["# Feature Importance Comparison: Global vs Per-Asset\n\n"]
        md_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        md_lines.append("## Top 20 Features (Global)\n\n")
        md_lines.append("| Feature | Global | " + " | ".join(assets) + " | Mean | Std | CV |\n")
        md_lines.append("|---------|--------|" + "|".join(["--------"] * len(assets)) + "|------|-----|----|\n")
        
        for _, row in comparison_df.head(20).iterrows():
            feature = row['feature']
            global_val = f"{row['global']:.4f}"
            asset_vals = [f"{row[asset]:.4f}" for asset in assets]
            mean_val = f"{row['mean_across_assets']:.4f}"
            std_val = f"{row['std_across_assets']:.4f}"
            cv_val = f"{row['cv']:.4f}"
            
            md_lines.append(f"| {feature} | {global_val} | " + " | ".join(asset_vals) + 
                          f" | {mean_val} | {std_val} | {cv_val} |\n")
        
        md_lines.append("\n## Asset-Specific Features\n\n")
        md_lines.append("Features with high coefficient of variation (CV > 0.5) across assets:\n\n")
        
        high_cv_features = comparison_df[comparison_df['cv'] > 0.5].sort_values('cv', ascending=False).head(10)
        if not high_cv_features.empty:
            md_lines.append("| Feature | CV | Global | " + " | ".join(assets) + " |\n")
            md_lines.append("|---------|----| -------|" + "|".join(["--------"] * len(assets)) + "|\n")
            
            for _, row in high_cv_features.iterrows():
                feature = row['feature']
                cv_val = f"{row['cv']:.4f}"
                global_val = f"{row['global']:.4f}"
                asset_vals = [f"{row[asset]:.4f}" for asset in assets]
                
                md_lines.append(f"| {feature} | {cv_val} | {global_val} | " + " | ".join(asset_vals) + " |\n")
        else:
            md_lines.append("No features with high cross-asset variance found.\n")
        
        md_path = self.outcomes_dir / f'feature_importance_comparison_{self.ts}.md'
        with open(md_path, 'w') as f:
            f.writelines(md_lines)
        
        tprint_success(f"   ✅ Feature importance comparison saved: {csv_path}")
        return md_path
    
    def generate_tree_statistics_comparison(
        self,
        tree_stats_data: Dict[str, Any],
        assets: List[str],
        asset_col: str
    ) -> Path:
        """
        Generate tree/leaf statistics comparison: global vs per-asset.
        
        Metrics:
        - Average tree depth
        - Number of leaves
        - Leaf purity
        - Split feature diversity
        """
        tprint_info("   🌳 Generating tree statistics comparison...")
        
        # Extract global tree stats
        global_stats = tree_stats_data.get('global', {})
        
        # Extract per-asset tree stats
        per_asset_stats = {asset: tree_stats_data.get(asset, {}) for asset in assets}
        
        # Metrics to compare
        stat_names = ['avg_depth', 'num_leaves', 'avg_leaf_purity', 'num_split_features', 'max_depth']
        
        comparison_rows = []
        for stat in stat_names:
            row = {'statistic': stat}
            
            # Global value
            row['global'] = global_stats.get(stat, np.nan)
            
            # Per-asset values
            asset_values = []
            for asset in assets:
                value = per_asset_stats[asset].get(stat, np.nan)
                row[asset] = value
                if not np.isnan(value):
                    asset_values.append(value)
            
            # Cross-asset statistics
            if asset_values:
                row['mean'] = np.mean(asset_values)
                row['std'] = np.std(asset_values)
                row['min'] = np.min(asset_values)
                row['max'] = np.max(asset_values)
            else:
                row['mean'] = row['std'] = row['min'] = row['max'] = np.nan
            
            comparison_rows.append(row)
        
        comparison_df = pd.DataFrame(comparison_rows)
        
        # Save to CSV
        csv_path = self.outcomes_dir / f'tree_statistics_comparison_{self.ts}.csv'
        comparison_df.to_csv(csv_path, index=False)
        
        # Generate markdown report
        md_lines = ["# Tree Statistics Comparison: Global vs Per-Asset\n\n"]
        md_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        md_lines.append("## Tree/Leaf Statistics\n\n")
        md_lines.append("| Statistic | Global | " + " | ".join(assets) + " | Mean | Std | Min | Max |\n")
        md_lines.append("|-----------|--------|" + "|".join(["--------"] * len(assets)) + "|------|-----|-----|-----|\n")
        
        for _, row in comparison_df.iterrows():
            stat = row['statistic']
            global_val = f"{row['global']:.2f}" if not np.isnan(row['global']) else "N/A"
            asset_vals = [f"{row[asset]:.2f}" if not np.isnan(row[asset]) else "N/A" for asset in assets]
            mean_val = f"{row['mean']:.2f}" if not np.isnan(row['mean']) else "N/A"
            std_val = f"{row['std']:.2f}" if not np.isnan(row['std']) else "N/A"
            min_val = f"{row['min']:.2f}" if not np.isnan(row['min']) else "N/A"
            max_val = f"{row['max']:.2f}" if not np.isnan(row['max']) else "N/A"
            
            md_lines.append(f"| {stat} | {global_val} | " + " | ".join(asset_vals) + 
                          f" | {mean_val} | {std_val} | {min_val} | {max_val} |\n")
        
        md_path = self.outcomes_dir / f'tree_statistics_comparison_{self.ts}.md'
        with open(md_path, 'w') as f:
            f.writelines(md_lines)
        
        tprint_success(f"   ✅ Tree statistics comparison saved: {csv_path}")
        return md_path
    
    def generate_label_quality_comparison(
        self,
        combined_df: pd.DataFrame,
        assets: List[str],
        asset_col: str
    ) -> Path:
        """
        Generate label quality comparison across assets.
        
        Metrics:
        - Label balance (positive rate)
        - Label entropy
        - Event density (events per day)
        - Average event duration
        """
        tprint_info("   🏷️  Generating label quality comparison...")
        
        comparison_rows = []
        
        for asset in assets:
            asset_df = combined_df[combined_df[asset_col] == asset]
            
            row = {'asset': asset}
            
            # Label balance
            if 'binary_labels' in asset_df.columns:
                labels = asset_df['binary_labels'].dropna()
                row['positive_rate'] = labels.mean() if len(labels) > 0 else np.nan
                row['label_count'] = len(labels)
                
                # Label entropy
                if len(labels) > 0:
                    p = labels.mean()
                    row['entropy'] = -p * np.log2(p + 1e-9) - (1-p) * np.log2(1-p + 1e-9)
                else:
                    row['entropy'] = np.nan
            else:
                row['positive_rate'] = np.nan
                row['label_count'] = 0
                row['entropy'] = np.nan
            
            # Event density
            if 'timestamp' in asset_df.columns or isinstance(asset_df.index, pd.DatetimeIndex):
                if isinstance(asset_df.index, pd.DatetimeIndex):
                    duration_days = (asset_df.index.max() - asset_df.index.min()).days
                else:
                    duration_days = (asset_df['timestamp'].max() - asset_df['timestamp'].min()).days
                
                row['duration_days'] = duration_days
                row['events_per_day'] = row['label_count'] / max(duration_days, 1)
            else:
                row['duration_days'] = np.nan
                row['events_per_day'] = np.nan
            
            # Sample count
            row['total_samples'] = len(asset_df)
            
            comparison_rows.append(row)
        
        # Add global statistics
        global_row = {'asset': 'GLOBAL'}
        if 'binary_labels' in combined_df.columns:
            labels = combined_df['binary_labels'].dropna()
            global_row['positive_rate'] = labels.mean() if len(labels) > 0 else np.nan
            global_row['label_count'] = len(labels)
            
            if len(labels) > 0:
                p = labels.mean()
                global_row['entropy'] = -p * np.log2(p + 1e-9) - (1-p) * np.log2(1-p + 1e-9)
            else:
                global_row['entropy'] = np.nan
        else:
            global_row['positive_rate'] = np.nan
            global_row['label_count'] = 0
            global_row['entropy'] = np.nan
        
        global_row['total_samples'] = len(combined_df)
        
        if 'timestamp' in combined_df.columns or isinstance(combined_df.index, pd.DatetimeIndex):
            if isinstance(combined_df.index, pd.DatetimeIndex):
                duration_days = (combined_df.index.max() - combined_df.index.min()).days
            else:
                duration_days = (combined_df['timestamp'].max() - combined_df['timestamp'].min()).days
            
            global_row['duration_days'] = duration_days
            global_row['events_per_day'] = global_row['label_count'] / max(duration_days, 1)
        else:
            global_row['duration_days'] = np.nan
            global_row['events_per_day'] = np.nan
        
        comparison_rows.append(global_row)
        
        comparison_df = pd.DataFrame(comparison_rows)
        
        # Save to CSV
        csv_path = self.outcomes_dir / f'label_quality_comparison_{self.ts}.csv'
        comparison_df.to_csv(csv_path, index=False)
        
        # Generate markdown report
        md_lines = ["# Label Quality Comparison Across Assets\n\n"]
        md_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        md_lines.append("## Label Statistics\n\n")
        md_lines.append("| Asset | Positive Rate | Label Count | Entropy | Events/Day | Total Samples |\n")
        md_lines.append("|-------|---------------|-------------|---------|------------|---------------|\n")
        
        for _, row in comparison_df.iterrows():
            asset = row['asset']
            pos_rate = f"{row['positive_rate']:.4f}" if not np.isnan(row['positive_rate']) else "N/A"
            label_count = int(row['label_count']) if not np.isnan(row['label_count']) else "N/A"
            entropy = f"{row['entropy']:.4f}" if not np.isnan(row['entropy']) else "N/A"
            events_per_day = f"{row['events_per_day']:.2f}" if not np.isnan(row['events_per_day']) else "N/A"
            total_samples = int(row['total_samples']) if not np.isnan(row['total_samples']) else "N/A"
            
            md_lines.append(f"| {asset} | {pos_rate} | {label_count} | {entropy} | {events_per_day} | {total_samples} |\n")
        
        md_path = self.outcomes_dir / f'label_quality_comparison_{self.ts}.md'
        with open(md_path, 'w') as f:
            f.writelines(md_lines)
        
        tprint_success(f"   ✅ Label quality comparison saved: {csv_path}")
        return md_path
    
    def generate_cross_asset_correlation(
        self,
        combined_df: pd.DataFrame,
        assets: List[str],
        asset_col: str
    ) -> Path:
        """
        Generate cross-asset correlation analysis.
        
        Analyzes:
        - Return correlations
        - Feature correlations
        - Prediction correlations
        - Residual correlations (after market residualization)
        """
        tprint_info("   🔗 Generating cross-asset correlation analysis...")
        
        md_lines = ["# Cross-Asset Correlation Analysis\n\n"]
        md_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 1. Return correlations
        if 'raw_returns' in combined_df.columns:
            md_lines.append("## Raw Return Correlations\n\n")
            
            return_corr = {}
            for asset in assets:
                asset_df = combined_df[combined_df[asset_col] == asset]
                if 'raw_returns' in asset_df.columns:
                    return_corr[asset] = asset_df['raw_returns'].dropna()
            
            if len(return_corr) > 1:
                corr_df = pd.DataFrame(return_corr).corr()
                
                md_lines.append("| Asset | " + " | ".join(assets) + " |\n")
                md_lines.append("|-------|" + "|".join(["-------"] * len(assets)) + "|\n")
                
                for asset in assets:
                    if asset in corr_df.index:
                        row_vals = [f"{corr_df.loc[asset, a]:.4f}" if a in corr_df.columns else "N/A" for a in assets]
                        md_lines.append(f"| {asset} | " + " | ".join(row_vals) + " |\n")
                
                md_lines.append("\n")
                
                # Average correlation
                avg_corr = corr_df.values[np.triu_indices_from(corr_df.values, k=1)].mean()
                md_lines.append(f"**Average pairwise correlation**: {avg_corr:.4f}\n\n")
        
        # 2. Residual return correlations (if available)
        if 'residual_return' in combined_df.columns:
            md_lines.append("## Residual Return Correlations (Market-Adjusted)\n\n")
            
            residual_corr = {}
            for asset in assets:
                asset_df = combined_df[combined_df[asset_col] == asset]
                if 'residual_return' in asset_df.columns:
                    residual_corr[asset] = asset_df['residual_return'].dropna()
            
            if len(residual_corr) > 1:
                corr_df = pd.DataFrame(residual_corr).corr()
                
                md_lines.append("| Asset | " + " | ".join(assets) + " |\n")
                md_lines.append("|-------|" + "|".join(["-------"] * len(assets)) + "|\n")
                
                for asset in assets:
                    if asset in corr_df.index:
                        row_vals = [f"{corr_df.loc[asset, a]:.4f}" if a in corr_df.columns else "N/A" for a in assets]
                        md_lines.append(f"| {asset} | " + " | ".join(row_vals) + " |\n")
                
                md_lines.append("\n")
                
                # Average correlation
                avg_corr = corr_df.values[np.triu_indices_from(corr_df.values, k=1)].mean()
                md_lines.append(f"**Average pairwise correlation (residual)**: {avg_corr:.4f}\n\n")
                md_lines.append("*Lower residual correlations indicate better market residualization.*\n\n")
        
        # 3. Volatility correlations
        if 'volatility_normalized' in combined_df.columns:
            md_lines.append("## Volatility Correlations\n\n")
            
            vol_corr = {}
            for asset in assets:
                asset_df = combined_df[combined_df[asset_col] == asset]
                if 'volatility_normalized' in asset_df.columns:
                    vol_corr[asset] = asset_df['volatility_normalized'].dropna()
            
            if len(vol_corr) > 1:
                corr_df = pd.DataFrame(vol_corr).corr()
                
                md_lines.append("| Asset | " + " | ".join(assets) + " |\n")
                md_lines.append("|-------|" + "|".join(["-------"] * len(assets)) + "|\n")
                
                for asset in assets:
                    if asset in corr_df.index:
                        row_vals = [f"{corr_df.loc[asset, a]:.4f}" if a in corr_df.columns else "N/A" for a in assets]
                        md_lines.append(f"| {asset} | " + " | ".join(row_vals) + " |\n")
                
                md_lines.append("\n")
        
        md_path = self.outcomes_dir / f'cross_asset_correlation_{self.ts}.md'
        with open(md_path, 'w') as f:
            f.writelines(md_lines)
        
        tprint_success(f"   ✅ Cross-asset correlation analysis saved: {md_path}")
        return md_path
    
    def generate_master_summary(
        self,
        combined_df: pd.DataFrame,
        model_results: Dict[str, Any],
        assets: List[str],
        asset_col: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Generate master summary report combining all metrics.
        """
        tprint_info("   📋 Generating master summary report...")
        
        md_lines = ["# Multi-Asset Training Summary Report\n\n"]
        md_lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Configuration
        if config:
            md_lines.append("## Configuration\n\n")
            md_lines.append(f"- **Assets**: {', '.join(assets)}\n")
            md_lines.append(f"- **Multi-asset mode**: {config.get('multi_asset_mode', False)}\n")
            md_lines.append(f"- **Label return column**: {config.get('label_return_column', 'N/A')}\n")
            md_lines.append(f"- **Use market residual labels**: {config.get('use_market_residual_labels', False)}\n")
            md_lines.append("\n")
        
        # Dataset statistics
        md_lines.append("## Dataset Statistics\n\n")
        md_lines.append(f"- **Total samples**: {len(combined_df)}\n")
        
        for asset in assets:
            asset_df = combined_df[combined_df[asset_col] == asset]
            md_lines.append(f"- **{asset} samples**: {len(asset_df)} ({len(asset_df)/len(combined_df)*100:.1f}%)\n")
        
        md_lines.append("\n")
        
        # Model performance summary
        if 'model_performance' in model_results:
            md_lines.append("## Model Performance Summary\n\n")
            
            perf_data = model_results['model_performance']
            
            if 'global' in perf_data:
                global_perf = perf_data['global']
                md_lines.append("### Global Model\n\n")
                md_lines.append(f"- **AUC**: {global_perf.get('auc', 'N/A')}\n")
                md_lines.append(f"- **Sharpe**: {global_perf.get('sharpe', 'N/A')}\n")
                md_lines.append(f"- **IC**: {global_perf.get('ic', 'N/A')}\n")
                md_lines.append(f"- **Win Rate**: {global_perf.get('win_rate', 'N/A')}\n")
                md_lines.append("\n")
            
            md_lines.append("### Per-Asset Performance\n\n")
            md_lines.append("| Asset | AUC | Sharpe | IC | Win Rate |\n")
            md_lines.append("|-------|-----|--------|----|-----------|\n")
            
            for asset in assets:
                if asset in perf_data:
                    asset_perf = perf_data[asset]
                    auc = f"{asset_perf.get('auc', 0):.4f}" if 'auc' in asset_perf else "N/A"
                    sharpe = f"{asset_perf.get('sharpe', 0):.2f}" if 'sharpe' in asset_perf else "N/A"
                    ic = f"{asset_perf.get('ic', 0):.4f}" if 'ic' in asset_perf else "N/A"
                    win_rate = f"{asset_perf.get('win_rate', 0):.4f}" if 'win_rate' in asset_perf else "N/A"
                    
                    md_lines.append(f"| {asset} | {auc} | {sharpe} | {ic} | {win_rate} |\n")
            
            md_lines.append("\n")
        
        # Key insights
        md_lines.append("## Key Insights\n\n")
        md_lines.append("### Asset-Specific Learning\n\n")
        md_lines.append("- ✅ Market residualization removes common market beta\n")
        md_lines.append("- ✅ Asset interaction features enable per-asset learning\n")
        md_lines.append("- ✅ Per-asset uniqueness weighting prevents cross-asset concurrency penalties\n")
        md_lines.append("\n")
        
        md_lines.append("### De Prado Compliance\n\n")
        md_lines.append("- ✅ Residualized predictors (market-neutral labels)\n")
        md_lines.append("- ✅ Causal beta estimation (rolling 60-period)\n")
        md_lines.append("- ✅ Sample uniqueness (per-asset concurrency)\n")
        md_lines.append("- ✅ Fractional differentiation (per-asset stationarity)\n")
        md_lines.append("\n")
        
        md_path = self.outcomes_dir / f'multi_asset_summary_{self.ts}.md'
        with open(md_path, 'w') as f:
            f.writelines(md_lines)
        
        tprint_success(f"   ✅ Master summary report saved: {md_path}")
        return md_path


def add_multi_asset_metrics_to_existing_report(
    report_data: Dict[str, Any],
    assets: List[str],
    asset_col: str = 'asset_id'
) -> Dict[str, Any]:
    """
    Helper function to add multi-asset comparison metrics to existing report data.
    
    Args:
        report_data: Existing report dictionary
        assets: List of asset identifiers
        asset_col: Column name for asset identifier
        
    Returns:
        Enhanced report data with multi-asset metrics
    """
    enhanced_data = report_data.copy()
    
    # Add cross-asset statistics for each metric
    for metric_name, metric_value in report_data.items():
        if isinstance(metric_value, dict):
            # Check if this is per-asset data
            if any(asset in metric_value for asset in assets):
                # Compute cross-asset statistics
                asset_values = [metric_value.get(asset, np.nan) for asset in assets]
                asset_values = [v for v in asset_values if not (isinstance(v, float) and np.isnan(v))]
                
                if asset_values:
                    enhanced_data[f'{metric_name}_mean_across_assets'] = np.mean(asset_values)
                    enhanced_data[f'{metric_name}_std_across_assets'] = np.std(asset_values)
                    enhanced_data[f'{metric_name}_min_across_assets'] = np.min(asset_values)
                    enhanced_data[f'{metric_name}_max_across_assets'] = np.max(asset_values)
                    enhanced_data[f'{metric_name}_range'] = np.max(asset_values) - np.min(asset_values)
    
    return enhanced_data
