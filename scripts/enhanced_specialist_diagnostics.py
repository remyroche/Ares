#!/usr/bin/env python3
"""Enhanced Specialist Feature Diagnostics CLI.

This script provides comprehensive specialist model training and comparison:
- Trains all specialist models independently
- Compares artifacts across all specialists  
- Generates consolidated performance reports
- Supports both legacy and independent specialist modes

Usage:
    # Train all specialists + compare artifacts
    python scripts/enhanced_specialist_diagnostics.py --auto-train --symbol ETHUSDT --exchange binance --timeframe 15m --direction long
    
    # Compare existing artifacts only
    python scripts/enhanced_specialist_diagnostics.py --compare-only --symbol ETHUSDT
    
    # Train specific specialists
    python scripts/enhanced_specialist_diagnostics.py --auto-train --selected-specialists ml_momentum_persistence_step ml_volatility_burst_step --symbol ETHUSDT
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import system_logger
from src.utils.ml_common.get_specialist_models_outputs import get_specialist_models_outputs
from src.utils.ml_common.feature_selection import get_feature_selection_utils
from src.training.steps.labeling.feature_generation_meta_labeling_step import FeatureGenerationMetaLabelingStep
from src.training.steps.labeling.snr_diagnostics import _load_labeled_data
from src.training.steps.pre_training.components.final_feature_selection import (
    FinalFeatureSelectionConfig,
    FinalFeatureSelectionComponent,
)
from src.training.steps.market_analysis import step_registry

logger = system_logger.getChild("enhanced_specialist_diagnostics")

OUTCOMES_DIR = Path("outcomes")

# Independent specialist steps
INDEPENDENT_SPECIALISTS = [
    'ml_momentum_persistence_step',
    'ml_volatility_burst_step',
    'ml_risk_regime_step', 
    'ml_liquidity_regime_step',
    'ml_breakout_bounce_regime_step',
    'ml_path_regime_step',
    'ml_reversion_regime_step',
    'ml_smc_regime_step',
    'ml_volume_force_step',
]

async def train_all_specialists(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    selected_specialists: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Train all specialist models sequentially with progress tracking."""
    
    specialist_steps = selected_specialists or INDEPENDENT_SPECIALISTS
    training_results = {}
    
    logger.info(f"🚀 Starting training for {len(specialist_steps)} specialists...")
    
    for i, step_name in enumerate(specialist_steps, 1):
        logger.info(f"[{i}/{len(specialist_steps)}] Training {step_name}...")
        
        try:
            # Check if specialist is registered
            if not step_registry.is_registered(step_name):
                logger.warning(f"⚠️ Step '{step_name}' not found in registry. Skipping.")
                continue
            
            # Get specialist class and instantiate
            StepClass = step_registry.get_step(step_name)
            specialist = StepClass()
            
            # Execute training
            config = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'direction': direction
            }
            
            result = await specialist.execute(config)
            training_results[step_name] = result
            
            if result.get('success'):
                logger.info(f"✅ {step_name} trained successfully")
                if 'metrics' in result:
                    metrics = result['metrics']
                    logger.info(f"   Metrics: AUC={metrics.get('auc', 'N/A'):.3f}, R²={metrics.get('r2', 'N/A'):.3f}")
            else:
                logger.error(f"❌ {step_name} training failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"❌ {step_name} training error: {e}")
            training_results[step_name] = {'success': False, 'error': str(e)}
    
    return training_results


def compare_all_specialist_artifacts(
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
    selected_specialists: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Load and compare artifacts from all trained specialists."""
    
    specialist_steps = selected_specialists or INDEPENDENT_SPECIALISTS
    comparison_data = {}
    
    logger.info(f"🔍 Comparing artifacts for {len(specialist_steps)} specialists...")
    
    for i, specialist_name in enumerate(specialist_steps, 1):
        logger.info(f"[{i}/{len(specialist_steps)}] Analyzing {specialist_name}...")
        
        try:
            # Check if specialist is registered
            if not step_registry.is_registered(specialist_name):
                logger.warning(f"⚠️ Specialist '{specialist_name}' not found in registry. Skipping.")
                continue
            
            # Get specialist class and run diagnostics
            StepClass = step_registry.get_step(specialist_name)
            specialist = StepClass()
            
            # Check if specialist has independent diagnostics
            if hasattr(specialist, 'run_diagnostics'):
                result = specialist.run_diagnostics(symbol, exchange, timeframe, direction)
                
                if result.get('success'):
                    comparison_data[specialist_name] = {
                        'metrics': result.get('metrics', {}),
                        'feature_importance': result.get('feature_importance', {}),
                        'stability': result.get('stability', {}),
                        'report_path': result.get('report_path'),
                        'csv_path': result.get('csv_path')
                    }
                    logger.info(f"✅ {specialist_name} diagnostics completed")
                else:
                    logger.warning(f"⚠️ {specialist_name} diagnostics failed: {result.get('error')}")
            else:
                logger.warning(f"⚠️ {specialist_name} does not have independent diagnostics")
                
        except Exception as e:
            logger.error(f"❌ Error analyzing {specialist_name}: {e}")
    
    return comparison_data


def create_performance_table(comparison_data: Dict[str, Any]) -> str:
    """Create performance comparison table."""
    
    table_lines = [
        "| Specialist | AUC | R² | Stability | Top Feature | Feature Count |",
        "|------------|-----|----|-----------|-------------|---------------|"
    ]
    
    for specialist_name, data in comparison_data.items():
        metrics = data.get('metrics', {})
        feature_importance = data.get('feature_importance', {})
        
        auc = metrics.get('auc', 'N/A')
        r2 = metrics.get('r2', 'N/A')
        stability = data.get('stability', {}).get('stability_mean', 'N/A')
        
        # Get top feature
        top_feature = 'N/A'
        if feature_importance:
            top_feature = max(feature_importance.items(), key=lambda x: x[1])[0]
            # Truncate long feature names
            if len(top_feature) > 20:
                top_feature = top_feature[:17] + "..."
        
        feature_count = len(feature_importance)
        
        # Format values
        auc_str = f"{auc:.3f}" if isinstance(auc, (int, float)) else str(auc)
        r2_str = f"{r2:.3f}" if isinstance(r2, (int, float)) else str(r2)
        stability_str = f"{stability:.3f}" if isinstance(stability, (int, float)) else str(stability)
        
        table_lines.append(
            f"| {specialist_name.replace('_', ' ').title()} | {auc_str} | {r2_str} | "
            f"{stability_str} | {top_feature} | {feature_count} |"
        )
    
    return "\n".join(table_lines)


def compute_cross_specialist_correlations(comparison_data: Dict[str, Any]) -> str:
    """Compute cross-specialist prediction correlations."""
    
    # For now, return a placeholder since we'd need actual prediction data
    # In a full implementation, we'd load predictions and compute correlations
    
    correlations_lines = [
        "### Cross-Specialist Correlation Matrix",
        "",
        "Note: Correlation analysis requires loading actual prediction data from artifacts.",
        "This would show how specialist predictions correlate with each other.",
        "",
        "Example format:",
        "| Specialist A | Specialist B | Correlation |",
        "|-------------|-------------|-------------|",
        "| Momentum | Volatility | 0.23 |",
        "| Momentum | Risk | 0.45 |",
        "| Volatility | Risk | 0.67 |",
    ]
    
    return "\n".join(correlations_lines)


def aggregate_feature_importance(comparison_data: Dict[str, Any]) -> str:
    """Aggregate and rank features across all specialists."""
    
    all_features = {}
    
    for specialist_name, data in comparison_data.items():
        feature_importance = data.get('feature_importance', {})
        for feature, importance in feature_importance.items():
            if feature not in all_features:
                all_features[feature] = []
            all_features[feature].append(importance)
    
    # Compute average importance across specialists
    avg_importance = {}
    for feature, values in all_features.items():
        avg_importance[feature] = np.mean(values)
    
    # Sort by average importance
    top_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:15]
    
    feature_lines = [
        "### Top 15 Features Across All Specialists",
        "",
        "| Rank | Feature | Avg Importance | Specialists |",
        "|------|---------|---------------|-------------|"
    ]
    
    for rank, (feature, avg_importance) in enumerate(top_features, 1):
        # Count how many specialists use this feature
        specialist_count = len(all_features[feature])
        
        feature_lines.append(
            f"| {rank} | {feature} | {avg_importance:.4f} | {specialist_count} |"
        )
    
    return "\n".join(feature_lines)


def analyze_ensemble_potential(comparison_data: Dict[str, Any]) -> str:
    """Analyze potential for ensemble combination."""
    
    ensemble_lines = [
        "### Ensemble Potential Analysis",
        "",
        "**Performance Distribution:**"
    ]
    
    # Collect AUC values
    auc_values = []
    for data in comparison_data.values():
        auc = data.get('metrics', {}).get('auc')
        if isinstance(auc, (int, float)):
            auc_values.append(auc)
    
    if auc_values:
        mean_auc = np.mean(auc_values)
        std_auc = np.std(auc_values)
        max_auc = np.max(auc_values)
        min_auc = np.min(auc_values)
        
        ensemble_lines.extend([
            f"- Mean AUC: {mean_auc:.3f}",
            f"- Std AUC: {std_auc:.3f}",
            f"- Range: [{min_auc:.3f}, {max_auc:.3f}]",
            f"- Specialist Count: {len(auc_values)}",
            "",
            "**Diversification Potential:**"
        ])
        
        if std_auc > 0.05:
            ensemble_lines.append("- ✅ Good diversity (std > 0.05) - ensemble likely beneficial")
        else:
            ensemble_lines.append("- ⚠️ Low diversity - ensemble may have limited benefit")
        
        if mean_auc > 0.55:
            ensemble_lines.append("- ✅ Good base performance - ensemble should be strong")
        else:
            ensemble_lines.append("- ⚠️ Low base performance - ensemble may need improvement")
    
    ensemble_lines.extend([
        "",
        "**Recommended Ensemble Approach:**",
        "- Simple averaging of specialist predictions",
        "- Consider weighted averaging based on individual AUC",
        "- Meta-model (LogisticRegression) on specialist predictions",
    ])
    
    return "\n".join(ensemble_lines)


def generate_specialist_links(comparison_data: Dict[str, Any]) -> str:
    """Generate links to individual specialist reports."""
    
    links_lines = [
        "### Individual Specialist Reports",
        ""
    ]
    
    for specialist_name, data in comparison_data.items():
        report_path = data.get('report_path')
        csv_path = data.get('csv_path')
        
        links_lines.append(f"#### {specialist_name.replace('_', ' ').title()}")
        
        if report_path:
            links_lines.append(f"- **Report:** `{report_path}`")
        if csv_path:
            links_lines.append(f"- **Metrics:** `{csv_path}`")
        
        # Add key metrics
        metrics = data.get('metrics', {})
        if metrics:
            key_metrics = []
            for metric in ['auc', 'r2', 'accuracy']:
                if metric in metrics:
                    value = metrics[metric]
                    if isinstance(value, (int, float)):
                        key_metrics.append(f"{metric.upper()}: {value:.3f}")
                    else:
                        key_metrics.append(f"{metric.upper()}: {value}")
            
            if key_metrics:
                links_lines.append(f"- **Key Metrics:** {', '.join(key_metrics)}")
        
        links_lines.append("")
    
    return "\n".join(links_lines)


def generate_consolidated_comparison_report(
    comparison_data: Dict[str, Any],
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
) -> str:
    """Generate comprehensive comparison report."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# Enhanced Specialist Model Comparison Report

**Symbol:** {symbol} | **Exchange:** {exchange} | **Timeframe:** {timeframe} | **Direction:** {direction}  
**Generated:** {timestamp}  
**Specialists Analyzed:** {len(comparison_data)}

## Executive Summary

This report provides a comprehensive comparison of {len(comparison_data)} specialist models trained independently.
Each specialist focuses on different market patterns and uses the 1.5-3% range optimization.

## Performance Summary

{create_performance_table(comparison_data)}

## Cross-Specialist Analysis

{compute_cross_specialist_correlations(comparison_data)}

## Feature Analysis

{aggregate_feature_importance(comparison_data)}

## Ensemble Potential

{analyze_ensemble_potential(comparison_data)}

## Detailed Specialist Reports

{generate_specialist_links(comparison_data)}

## Recommendations

### Next Steps
1. **Top Performing Specialists:** Focus on specialists with AUC > 0.55
2. **Ensemble Development:** Implement weighted averaging based on individual performance
3. **Feature Engineering:** Leverage top features across multiple specialists
4. **Monitoring:** Track temporal stability and performance degradation

### Trading Strategy Integration
- Consider combining top 3-5 specialists in a weighted ensemble
- Use specialist confidence scores for position sizing
- Implement regime-based specialist selection

---
*Report generated by Enhanced Specialist Diagnostics Pipeline*
"""
    
    return report


def save_consolidated_report(
    report: str,
    comparison_data: Dict[str, Any],
    symbol: str,
    exchange: str,
    timeframe: str,
    direction: str,
) -> Tuple[str, str]:
    """Save consolidated report and metrics CSV."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"enhanced_specialist_comparison_{symbol}_{timeframe}_{direction}_{timestamp}"
    
    # Save markdown report
    md_path = OUTCOMES_DIR / f"{base_name}.md"
    with open(md_path, 'w') as f:
        f.write(report)
    
    # Create consolidated metrics CSV
    metrics_data = []
    for specialist_name, data in comparison_data.items():
        metrics = data.get('metrics', {})
        stability = data.get('stability', {})
        
        row = {
            'specialist': specialist_name,
            'auc': metrics.get('auc'),
            'r2': metrics.get('r2'),
            'accuracy': metrics.get('accuracy'),
            'stability_mean': stability.get('stability_mean'),
            'stability_cv': stability.get('stability_cv'),
            'feature_count': len(data.get('feature_importance', {})),
            'report_path': data.get('report_path'),
            'csv_path': data.get('csv_path'),
        }
        metrics_data.append(row)
    
    metrics_df = pd.DataFrame(metrics_data)
    csv_path = OUTCOMES_DIR / f"{base_name}_metrics.csv"
    metrics_df.to_csv(csv_path, index=False)
    
    logger.info(f"✅ Consolidated report saved: {md_path}")
    logger.info(f"✅ Metrics CSV saved: {csv_path}")
    
    return str(md_path), str(csv_path)


async def main_async() -> None:
    """Main async entry point."""
    ap = argparse.ArgumentParser(
        description="Enhanced Specialist Model Training and Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all specialists + compare artifacts
  python scripts/enhanced_specialist_diagnostics.py --auto-train --symbol ETHUSDT --exchange binance --timeframe 15m --direction long
  
  # Compare existing artifacts only
  python scripts/enhanced_specialist_diagnostics.py --compare-only --symbol ETHUSDT
  
  # Train specific specialists
  python scripts/enhanced_specialist_diagnostics.py --auto-train --selected-specialists ml_momentum_persistence_step ml_volatility_burst_step --symbol ETHUSDT
        """
    )
    
    # Basic arguments
    ap.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    ap.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    ap.add_argument("--timeframe", type=str, default="15m", help="Timeframe")
    ap.add_argument("--direction", type=str, default="long", choices=["long", "short"], help="Trading direction")
    
    # Mode arguments
    ap.add_argument("--auto-train", action="store_true", help="Train all specialists before comparison")
    ap.add_argument("--train-only", action="store_true", help="Only train specialists, no comparison")
    ap.add_argument("--compare-only", action="store_true", help="Only compare existing artifacts, no training")
    
    # Specialist selection
    ap.add_argument("--selected-specialists", nargs="+", 
                   help="Specific specialists to train/compare",
                   choices=INDEPENDENT_SPECIALISTS)
    
    args = ap.parse_args()
    
    # Validate arguments
    if args.auto_train and args.compare_only:
        logger.error("❌ Cannot use --auto-train and --compare-only together")
        return
    
    if args.train_only and args.compare_only:
        logger.error("❌ Cannot use --train-only and --compare-only together")
        return
    
    logger.info("🚀 Enhanced Specialist Diagnostics Pipeline")
    logger.info(f"📈 Symbol: {args.symbol}/{args.exchange}")
    logger.info(f"⏱️ Timeframe: {args.timeframe}")
    logger.info(f"📊 Direction: {args.direction}")
    
    selected_specialists = args.selected_specialists
    
    # Mode 1: Train only
    if args.train_only:
        logger.info("🔧 Training mode only...")
        training_results = await train_all_specialists(
            args.symbol, args.exchange, args.timeframe, args.direction, selected_specialists
        )
        
        # Print training summary
        successful = sum(1 for r in training_results.values() if r.get('success'))
        total = len(training_results)
        logger.info(f"✅ Training completed: {successful}/{total} specialists successful")
        
        return
    
    # Mode 2: Train + Compare
    if args.auto_train:
        logger.info("🚀 Training + comparison mode...")
        training_results = await train_all_specialists(
            args.symbol, args.exchange, args.timeframe, args.direction, selected_specialists
        )
        
        successful = sum(1 for r in training_results.values() if r.get('success'))
        total = len(training_results)
        logger.info(f"✅ Training completed: {successful}/{total} specialists successful")
        
        if successful == 0:
            logger.error("❌ No specialists trained successfully. Exiting.")
            return
    
    # Mode 3: Compare only (or after training)
    logger.info("📊 Comparison mode...")
    comparison_data = compare_all_specialist_artifacts(
        args.symbol, args.exchange, args.timeframe, args.direction, selected_specialists
    )
    
    if not comparison_data:
        logger.error("❌ No specialist data available for comparison")
        return
    
    # Generate and save report
    logger.info("📝 Generating consolidated report...")
    report = generate_consolidated_comparison_report(
        comparison_data, args.symbol, args.exchange, args.timeframe, args.direction
    )
    
    md_path, csv_path = save_consolidated_report(
        report, comparison_data, args.symbol, args.exchange, args.timeframe, args.direction
    )
    
    # Print summary
    logger.info("🎉 Enhanced Specialist Diagnostics completed!")
    logger.info(f"📊 Report: {md_path}")
    logger.info(f"📈 Metrics: {csv_path}")
    logger.info(f"🔍 Specialists analyzed: {len(comparison_data)}")
    
    # Print top performers
    if comparison_data:
        logger.info("🏆 Top performers by AUC:")
        specialist_scores = []
        for name, data in comparison_data.items():
            auc = data.get('metrics', {}).get('auc', 0)
            if isinstance(auc, (int, float)):
                specialist_scores.append((name, auc))
        
        specialist_scores.sort(key=lambda x: x[1], reverse=True)
        for name, auc in specialist_scores[:3]:
            logger.info(f"   {name}: {auc:.3f}")


def main() -> None:
    """Main entry point."""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
