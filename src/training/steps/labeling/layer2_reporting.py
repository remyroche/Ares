"""
Layer 2 Reporting Module

Comprehensive reporting system for Layer 2 meta-labeling pipeline.
Generates detailed .md and .csv reports for geometry optimization,
model race results, feature selection, and event statistics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
from datetime import datetime
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

class Layer2Reporter:
    """Comprehensive reporting for Layer 2 meta-labeling pipeline."""
    
    def __init__(self, outcomes_dir: Optional[Path] = None):
        self.outcomes_dir = outcomes_dir or Path('outcomes')
        self.outcomes_dir.mkdir(exist_ok=True, parents=True)
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def generate_all_reports(self, df: pd.DataFrame, events_df: pd.DataFrame, 
                           production_geometries: List[Any], 
                           oof_results: Dict[str, Any]) -> None:
        """
        Generate all Layer 2 reports.
        
        Args:
            df: Original dataframe with market data
            events_df: Events dataframe
            production_geometries: List of selected geometry trials
            oof_results: Out-of-fold results dictionary
        """
        tprint_info("📊 Generating Comprehensive Layer 2 Reports...")
        
        # Generate reports
        self.generate_meta_report(df, production_geometries, oof_results)
        self.generate_geometry_optimization_report(production_geometries)
        self.generate_model_race_report(oof_results)
        self.generate_feature_selection_report(oof_results)
        self.generate_event_statistics_report(df, events_df)
        
        tprint_success(f"✅ Layer 2 reports saved to {self.outcomes_dir}")
    
    def generate_meta_report(self, df: pd.DataFrame, 
                           production_geometries: List[Any],
                           oof_results: Dict[str, Any]) -> None:
        """Generate Layer 2 meta-report with summary statistics."""
        try:
            lines = ["# Layer 2 Meta-Labeling Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            lines.append(f"Dataset: {len(df)} rows | Timeframe: 15m\n\n")

            # Geometry Summary
            lines.append("## Geometry Optimization Summary\n")
            if production_geometries:
                lines.append(f"- **Total Geometries Selected**: {len(production_geometries)}\n")
                family_counts = {}
                for geom in production_geometries:
                    family = getattr(geom, 'family', 'Unknown')
                    family_counts[family] = family_counts.get(family, 0) + 1
                lines.append("- **Family Distribution**:\n")
                for family, count in family_counts.items():
                    lines.append(f"  - {family}: {count}\n")
                
                # Performance summary
                scores = [getattr(geom, 'final_score', 0) for geom in production_geometries]
                learnability = [getattr(geom, 'learnability', 0) for geom in production_geometries]
                lines.append(f"- **Average Geometry Score**: {np.mean(scores):.4f}\n")
                lines.append(f"- **Average Learnability**: {np.mean(learnability):.4f}\n")
            else:
                lines.append("No geometries selected.\n")
            lines.append("\n")

            # Baseline Comparison
            if 'baseline_metrics' in oof_results:
                lines.append("## Baseline Comparison\n")
                lines.append("| Model | AUC | Sharpe | Total PnL | Trades | N Samples |\n")
                lines.append("| --- | --- | --- | --- | --- | --- |\n")
                for name, metrics in oof_results['baseline_metrics'].items():
                    lines.append(f"| {name} | {metrics.get('auc', 0.0):.4f} | {metrics.get('sharpe', 0.0):.2f} | "
                               f"{metrics.get('total_pnl', 0.0):.4f} | {metrics.get('n_trades', 0)} | {metrics.get('n_samples', 0)} |\n")
                lines.append("\n")

            # Model Performance Summary
            lines.append("## Model Performance Summary\n")
            lines.append("- **Validation Strategy**: Purged K-Fold (De Prado) [Prevents Leakage]\n")
            if 'model_metrics' in oof_results:
                metrics = oof_results['model_metrics']
                lines.append(f"- **Overall AUC**: {metrics.get('mean_auc', 'N/A'):.4f}\n")
                lines.append(f"- **Overall PR-AUC**: {metrics.get('mean_ap', 'N/A'):.4f}\n")
                lines.append(f"- **Cross-Validation Folds**: {metrics.get('n_folds', 'N/A')}\n")
                
                if 'best_model' in metrics:
                    lines.append(f"- **Best Model**: {metrics['best_model']}\n")
            lines.append("\n")

            # Event Statistics
            lines.append("## Event Statistics\n")
            if events_df is not None:
                lines.append(f"- **Total Events**: {len(events_df)}\n")
                lines.append(f"- **Event Coverage**: {len(events_df)/len(df)*100:.2f}%\n")
                lines.append(f"- **Events per Day**: {len(events_df)/(len(df)/96):.1f}\n")
            lines.append("\n")

            # Feature Selection
            lines.append("## Feature Engineering\n")
            if 'production_selected_features' in oof_results:
                features = oof_results['production_selected_features']
                lines.append(f"- **Selected Features**: {len(features)}\n")
                lines.append(f"- **Feature Categories**: Layer 0, Layer 1, Market Microstructure\n")
            lines.append("\n")

            report_path = self.outcomes_dir / f"layer2_meta_report_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Meta-report saved to {report_path}")

        except Exception as e:
            tprint_error(f"❌ Failed to generate Layer 2 meta-report: {e}")

    def generate_geometry_optimization_report(self, production_geometries: List[Any]) -> None:
        """Generate detailed geometry optimization report."""
        try:
            if not production_geometries:
                tprint_warning("⚠️ No geometries available for optimization report")
                return

            # Create geometry data
            geometry_data = []
            for geom in production_geometries:
                params = getattr(geom, 'params', {})
                geometry_data.append({
                    'family': getattr(geom, 'family', 'Unknown'),
                    'uuid': getattr(geom, 'uuid', 'Unknown'),
                    'final_score': getattr(geom, 'final_score', 0),
                    'learnability': getattr(geom, 'learnability', 0),
                    'robust_magnitude': getattr(geom, 'robust_magnitude', 0),
                    'stability': getattr(geom, 'stability', 0),
                    'balance': getattr(geom, 'balance', 0),
                    'kappa': params.get('kappa', 'N/A'),
                    'horizon': params.get('horizon', 'N/A'),
                    'sl_mult': params.get('sl_mult', 'N/A'),
                    'pt_mult': params.get('pt_mult', 'N/A'),
                    'alpha': params.get('alpha', 'N/A'),
                    'beta': params.get('beta', 'N/A')
                })

            geometry_df = pd.DataFrame(geometry_data)
            geometry_df = geometry_df.sort_values('final_score', ascending=False)

            # Save CSV
            csv_path = self.outcomes_dir / f"layer2_geometry_optimization_{self.ts}.csv"
            geometry_df.to_csv(csv_path, index=False)

            # Generate markdown report
            lines = ["# Layer 2 Geometry Optimization Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary statistics
            lines.append("## Optimization Summary\n")
            lines.append(f"- **Total Geometries**: {len(geometry_df)}\n")
            lines.append(f"- **Families**: {geometry_df['family'].nunique()}\n")
            lines.append(f"- **Average Score**: {geometry_df['final_score'].mean():.4f}\n")
            lines.append(f"- **Score Std**: {geometry_df['final_score'].std():.4f}\n")
            lines.append(f"- **Best Score**: {geometry_df['final_score'].max():.4f}\n")
            lines.append(f"- **Worst Score**: {geometry_df['final_score'].min():.4f}\n\n")

            # Family performance
            lines.append("## Family Performance\n")
            family_stats = geometry_df.groupby('family')['final_score'].agg(['count', 'mean', 'std', 'max'])
            for family, stats in family_stats.iterrows():
                lines.append(f"### {family}\n")
                lines.append(f"- Count: {stats['count']}\n")
                lines.append(f"- Mean Score: {stats['mean']:.4f}\n")
                lines.append(f"- Std Score: {stats['std']:.4f}\n")
                lines.append(f"- Best Score: {stats['max']:.4f}\n\n")

            # Top geometries table
            lines.append("## Top 10 Geometries\n")
            lines.append("| Rank | Family | UUID | Score | Learnability | Kappa | Horizon |\n")
            lines.append("|------|--------|------|-------|--------------|-------|----------|\n")
            for i, (_, row) in enumerate(geometry_df.head(10).iterrows()):
                uuid_short = str(row['uuid'])[:8] if row['uuid'] != 'N/A' else 'N/A'
                lines.append(f"| {i+1} | {row['family']} | {uuid_short} | {row['final_score']:.4f} | "
                           f"{row['learnability']:.4f} | {row['kappa']} | {row['horizon']} |\n")

            report_path = self.outcomes_dir / f"layer2_geometry_optimization_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Geometry optimization report saved")

        except Exception as e:
            tprint_error(f"❌ Failed to generate geometry optimization report: {e}")

    def generate_model_race_report(self, oof_results: Dict[str, Any]) -> None:
        """Generate model race comparison report."""
        try:
            if 'model_race_results' not in oof_results:
                tprint_warning("⚠️ No model race results available")
                return

            race_results = oof_results['model_race_results']
            
            # Create model comparison data
            model_data = []
            for model_name, metrics in race_results.items():
                model_data.append({
                    'model': model_name,
                    'auc': metrics.get('auc', 'N/A'),
                    'pr_auc': metrics.get('pr_auc', 'N/A'),
                    'bss': metrics.get('bss', 'N/A'),
                    'dir_consistency': metrics.get('dir_consistency', 'N/A'),
                    'stability': metrics.get('stability', 'N/A'),
                    'recall_at_40': metrics.get('recall_at_40', 'N/A'),
                    'precision_at_40': metrics.get('precision_at_40', 'N/A'),
                    'log_loss': metrics.get('log_loss', 'N/A'),
                    'training_time': metrics.get('training_time', 'N/A')
                })

            model_df = pd.DataFrame(model_data)
            model_df = model_df.sort_values('auc', ascending=False)

            # Save CSV
            csv_path = self.outcomes_dir / f"layer2_model_race_{self.ts}.csv"
            model_df.to_csv(csv_path, index=False)

            # Generate markdown report
            lines = ["# Layer 2 Model Race Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Summary
            lines.append("## Model Race Summary\n")
            lines.append(f"- **Models Tested**: {len(model_df)}\n")
            lines.append(f"- **Best AUC**: {model_df['auc'].max():.4f}\n")
            lines.append(f"- **Best PR-AUC**: {model_df['pr_auc'].max():.4f}\n")
            if 'LGBM_Focal' in model_df['model'].values:
                lgbm_row = model_df[model_df['model'] == 'LGBM_Focal'].iloc[0]
                lines.append(f"- **LGBM_Focal AUC**: {lgbm_row['auc']:.4f}\n")
            lines.append("\n")

            # Model comparison table
            lines.append("## Model Comparison\n")
            lines.append("| Model | AUC | PR-AUC | BSS | DirCons | Stability | Recall@40 | Precision@40 | LogLoss | Time(s) |\n")
            lines.append("|-------|-----|--------|-----|---------|-----------|-----------|--------------|---------|----------|\n")
            for _, row in model_df.iterrows():
                time_str = f"{row['training_time']:.2f}" if row['training_time'] != 'N/A' else 'N/A'
                lines.append(
                    f"| {row['model']} | {row['auc']:.4f} | {row['pr_auc']:.4f} | {row['bss']:.4f} | "
                    f"{row['dir_consistency']:.4f} | {row['stability']:.4f} | {row['recall_at_40']:.3f} | "
                    f"{row['precision_at_40']:.3f} | {row['log_loss']:.4f} | {time_str} |\n"
                )

            report_path = self.outcomes_dir / f"layer2_model_race_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Model race report saved")

        except Exception as e:
            tprint_error(f"❌ Failed to generate model race report: {e}")

    def generate_feature_selection_report(self, oof_results: Dict[str, Any]) -> None:
        """Generate feature selection report."""
        try:
            if 'feature_selection_results' not in oof_results:
                tprint_warning("⚠️ No feature selection results available")
                return

            fs_results = oof_results['feature_selection_results']
            
            # Feature importance data
            if 'feature_importance' in fs_results:
                importance_data = []
                for feature, importance in fs_results['feature_importance'].items():
                    importance_data.append({
                        'feature': feature,
                        'importance': importance,
                        'category': self._categorize_feature(feature)
                    })
                
                importance_df = pd.DataFrame(importance_data)
                importance_df = importance_df.sort_values('importance', ascending=False)

                # Save CSV
                csv_path = self.outcomes_dir / f"layer2_feature_importance_{self.ts}.csv"
                importance_df.to_csv(csv_path, index=False)

                # Generate markdown report
                lines = ["# Layer 2 Feature Selection Report\n\n"]
                lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # Summary
                lines.append("## Feature Selection Summary\n")
                lines.append(f"- **Total Features**: {len(importance_df)}\n")
                lines.append(f"- **Selected Features**: {fs_results.get('n_selected', 'N/A')}\n")
                lines.append(f"- **Selection Method**: {fs_results.get('method', 'Titan RFE')}\n")
                lines.append("\n")

                # Category breakdown
                lines.append("## Feature Categories\n")
                category_stats = importance_df.groupby('category')['importance'].agg(['count', 'mean', 'sum'])
                for category, stats in category_stats.iterrows():
                    lines.append(f"### {category}\n")
                    lines.append(f"- Count: {stats['count']}\n")
                    lines.append(f"- Mean Importance: {stats['mean']:.4f}\n")
                    lines.append(f"- Total Importance: {stats['sum']:.4f}\n\n")

                # Top features
                lines.append("## Top 20 Features\n")
                lines.append("| Rank | Feature | Importance | Category |\n")
                lines.append("|------|---------|------------|----------|\n")
                for i, (_, row) in enumerate(importance_df.head(20).iterrows()):
                    feature_truncated = str(row['feature'])[:50] + "..." if len(str(row['feature'])) > 50 else str(row['feature'])
                    lines.append(f"| {i+1} | {feature_truncated} | {row['importance']:.4f} | {row['category']} |\n")

                report_path = self.outcomes_dir / f"layer2_feature_selection_{self.ts}.md"
                report_path.write_text("".join(lines))
                tprint_success(f"✅ Feature selection report saved")

        except Exception as e:
            tprint_error(f"❌ Failed to generate feature selection report: {e}")

    def generate_event_statistics_report(self, df: pd.DataFrame, events_df: pd.DataFrame) -> None:
        """Generate event statistics and temporal analysis report."""
        try:
            if events_df is None or len(events_df) == 0:
                tprint_warning("⚠️ No events available for statistics report")
                return

            # Event statistics
            total_events = len(events_df)
            event_coverage = total_events / len(df)
            events_per_day = total_events / (len(df) / 96)  # Assuming 15m timeframe

            # Temporal analysis
            events_temporal = events_df.copy()
            events_temporal['hour'] = events_temporal.index.hour
            events_temporal['day_of_week'] = events_temporal.index.dayofweek
            events_temporal['month'] = events_temporal.index.month

            hourly_dist = events_temporal['hour'].value_counts().sort_index()
            dow_dist = events_temporal['day_of_week'].value_counts().sort_index()
            monthly_dist = events_temporal['month'].value_counts().sort_index()

            # Generate markdown report
            lines = ["# Layer 2 Event Statistics Report\n\n"]
            lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Basic statistics
            lines.append("## Event Statistics\n")
            lines.append(f"- **Total Events**: {total_events:,}\n")
            lines.append(f"- **Event Coverage**: {event_coverage:.2%}\n")
            lines.append(f"- **Events per Day**: {events_per_day:.1f}\n")
            lines.append(f"- **Data Period**: {df.index[0]} to {df.index[-1]}\n")
            lines.append(f"- **Trading Days**: {(df.index[-1] - df.index[0]).days}\n")
            lines.append("\n")

            # Temporal distribution
            lines.append("## Temporal Distribution\n")
            
            lines.append("### Hourly Distribution\n")
            lines.append("| Hour | Events | Percentage |\n")
            lines.append("|------|--------|------------|\n")
            for hour, count in hourly_dist.items():
                pct = count / total_events * 100
                lines.append(f"| {hour:02d} | {count:,} | {pct:.1f}% |\n")
            lines.append("\n")

            lines.append("### Day of Week Distribution\n")
            lines.append("| Day | Events | Percentage |\n")
            lines.append("|-----|--------|------------|\n")
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            for day, count in dow_dist.items():
                pct = count / total_events * 100
                lines.append(f"| {day_names[day]} | {count:,} | {pct:.1f}% |\n")
            lines.append("\n")

            # Event clustering analysis
            lines.append("## Event Clustering Analysis\n")
            event_gaps = events_df.index.to_series().diff().dropna()
            avg_gap = event_gaps.mean()
            median_gap = event_gaps.median()
            max_gap = event_gaps.max()

            lines.append(f"- **Average Gap**: {avg_gap.total_seconds()/3600:.1f} hours\n")
            lines.append(f"- **Median Gap**: {median_gap.total_seconds()/3600:.1f} hours\n")
            lines.append(f"- **Maximum Gap**: {max_gap.total_seconds()/3600:.1f} hours\n")
            lines.append("\n")

            # Save event data
            event_stats_df = pd.DataFrame({
                'timestamp': events_df.index,
                'hour': events_temporal['hour'],
                'day_of_week': events_temporal['day_of_week'],
                'month': events_temporal['month']
            })
            csv_path = self.outcomes_dir / f"layer2_event_statistics_{self.ts}.csv"
            event_stats_df.to_csv(csv_path, index=False)

            report_path = self.outcomes_dir / f"layer2_event_statistics_{self.ts}.md"
            report_path.write_text("".join(lines))
            tprint_success(f"✅ Event statistics report saved")

        except Exception as e:
            tprint_error(f"❌ Failed to generate event statistics report: {e}")

    def _categorize_feature(self, feature_name: str) -> str:
        """Categorize feature based on naming patterns."""
        feature_name_lower = str(feature_name).lower()
        
        if any(x in feature_name_lower for x in ['unified', 'adaptive', 'noise', 'filter', 'layer0']):
            return 'Layer 0'
        elif any(x in feature_name_lower for x in ['layer1', 'weight']):
            return 'Layer 1'
        elif any(x in feature_name_lower for x in ['volume', 'vwap', 'spread', 'bid', 'ask']):
            return 'Market Microstructure'
        elif any(x in feature_name_lower for x in ['rsi', 'macd', 'bb', 'sma', 'ema']):
            return 'Technical'
        elif any(x in feature_name_lower for x in ['cusum', 'trend', 'regime', 'volatility']):
            return 'Regime'
        else:
            return 'Other'
