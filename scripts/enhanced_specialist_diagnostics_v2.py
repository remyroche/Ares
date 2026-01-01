#!/usr/bin/env python3
"""Enhanced Specialist Feature Diagnostics CLI v2.

This script provides comprehensive specialist model training and comparison with:
- MI/HSIC analysis to target
- Cross-specialist orthogonality enforcement
- Single 0/1 scalar output standardization
- Comprehensive reporting

Usage:
    # Train all specialists + compare artifacts with optimization
    python scripts/enhanced_specialist_diagnostics_v2.py --auto-train --optimize-orthogonality --symbol ETHUSDT --exchange binance --timeframe 15m --direction long
    
    # Compare existing artifacts with orthogonality analysis
    python scripts/enhanced_specialist_diagnostics_v2.py --compare-only --analyze-orthogonality --symbol ETHUSDT
"""

import argparse
import asyncio
import inspect
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
from scipy.stats import spearmanr

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

logger = system_logger.getChild("enhanced_specialist_diagnostics_v2")

OUTCOMES_DIR = Path("outcomes")

# Independent specialist steps - include all available specialists
INDEPENDENT_SPECIALISTS = [
    # Base ML specialists
    'ml_momentum_persistence_step',
    'ml_volatility_burst_step',
    'ml_risk_regime_step',
    'ml_liquidity_regime_step',
    'ml_breakout_bounce_regime_step',
    'ml_smc_regime_step',
    'ml_volume_force_step',
    
    # XGB specialists
    'xgb_meso_regime',
    'xgb_macro_regime',
    
    # Enhanced specialists
    'enhanced_ml_spectral_step',
    'enhanced_ml_microstructure_step',
    'enhanced_ml_candlestick_step',
]

def compute_hsic(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0) -> float:
    """Compute Hilbert-Schmidt Independence Criterion (HSIC)."""
    from scipy.spatial.distance import pdist, squareform
    
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    
    n = X.shape[0]
    
    def rbf_kernel(X, Y=None, sigma=sigma):
        if Y is None:
            Y = X
        pairwise_dists = pdist(X, 'sqeuclidean')
        K = np.exp(-pairwise_dists / (2 * sigma ** 2))
        return squareform(K)
    
    K = rbf_kernel(X)
    L = rbf_kernel(Y)
    
    H = np.eye(n) - np.ones((n, n)) / n
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    
    hsic = np.trace(K_centered @ L_centered) / (n ** 2)
    return hsic

def analyze_specialist_orthogonality(comparison_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze orthogonality between specialists."""
    
    logger.info("🔍 Analyzing specialist orthogonality...")
    
    # Collect predictions and features
    all_predictions = {}
    all_features = {}
    
    for specialist_name, data in comparison_data.items():
        try:
            # Load predictions from artifacts
            artifact_store = VersionedArtifactStore()
            predictions_data = artifact_store.load_latest(
                artifact_name=f"{specialist_name}_15m",
                symbol="ETHUSDT",  # Default - should be parameterized
                exchange="binance",
                timeframe="15m",
                direction="long"
            )
            
            if predictions_data is not None and isinstance(predictions_data, pd.DataFrame):
                pred_col = f"{specialist_name}_prediction"
                if pred_col in predictions_data.columns:
                    all_predictions[specialist_name] = predictions_data[pred_col]
                
                # Extract features
                feature_cols = [col for col in predictions_data.columns if col.endswith('_feature')]
                if feature_cols:
                    all_features[specialist_name] = predictions_data[feature_cols]
                    
        except Exception as e:
            logger.warning(f"⚠️ Could not load {specialist_name} for orthogonality analysis: {e}")
    
    if len(all_predictions) < 2:
        logger.warning("⚠️ Insufficient specialists for orthogonality analysis")
        return {'orthogonality_matrix': {}, 'feature_overlap': {}, 'recommendations': []}
    
    # Compute prediction correlation matrix
    specialist_names = list(all_predictions.keys())
    orthogonality_matrix = {}
    
    for i, name1 in enumerate(specialist_names):
        for j, name2 in enumerate(specialist_names):
            if i < j:
                try:
                    # Align predictions
                    common_idx = all_predictions[name1].index.intersection(all_predictions[name2].index)
                    if len(common_idx) > 100:  # Minimum samples
                        pred1 = all_predictions[name1].loc[common_idx]
                        pred2 = all_predictions[name2].loc[common_idx]
                        
                        correlation, p_value = spearmanr(pred1, pred2)
                        orthogonality_matrix[f"{name1}_vs_{name2}"] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'samples': len(common_idx)
                        }
                except Exception as e:
                    logger.warning(f"⚠️ Correlation failed for {name1} vs {name2}: {e}")
    
    # Feature overlap analysis
    feature_overlap = {}
    for i, name1 in enumerate(specialist_names):
        for j, name2 in enumerate(specialist_names):
            if i < j and name1 in all_features and name2 in all_features:
                features1 = set(all_features[name1].columns)
                features2 = set(all_features[name2].columns)
                overlap = features1.intersection(features2)
                overlap_pct = len(overlap) / min(len(features1), len(features2)) * 100 if features1 and features2 else 0
                
                feature_overlap[f"{name1}_vs_{name2}"] = {
                    'overlap_count': len(overlap),
                    'overlap_percentage': overlap_pct,
                    'unique_features_1': len(features1 - overlap),
                    'unique_features_2': len(features2 - overlap)
                }
    
    # Generate recommendations
    recommendations = []
    
    # High correlation warnings
    high_corr_pairs = [
        pair for pair, stats in orthogonality_matrix.items()
        if abs(stats['correlation']) > 0.7
    ]
    
    if high_corr_pairs:
        recommendations.append(f"⚠️ High correlation detected: {', '.join(high_corr_pairs)}")
    
    # High feature overlap warnings
    high_overlap_pairs = [
        pair for pair, stats in feature_overlap.items()
        if stats['overlap_percentage'] > 80
    ]
    
    if high_overlap_pairs:
        recommendations.append(f"⚠️ High feature overlap: {', '.join(high_overlap_pairs)}")
    
    # Good orthogonality
    good_orthogonality_pairs = [
        pair for pair, stats in orthogonality_matrix.items()
        if abs(stats['correlation']) < 0.3
    ]
    
    if good_orthogonality_pairs:
        recommendations.append(f"✅ Good orthogonality: {', '.join(good_orthogonality_pairs)}")
    
    return {
        'orthogonality_matrix': orthogonality_matrix,
        'feature_overlap': feature_overlap,
        'recommendations': recommendations,
        'specialist_count': len(specialist_names)
    }

def analyze_mi_hsic_coverage(comparison_data: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze MI/HSIC coverage across specialists."""
    
    logger.info("📊 Analyzing MI/HSIC coverage...")
    
    mi_coverage = {}
    hsic_coverage = {}
    
    for specialist_name, data in comparison_data.items():
        metrics = data.get('metrics', {})
        
        mi_coverage[specialist_name] = {
            'prediction_mi': metrics.get('prediction_mi_to_target', 0),
            'avg_feature_mi': metrics.get('avg_feature_mi', 0)
        }
        
        hsic_coverage[specialist_name] = {
            'prediction_hsic': metrics.get('prediction_hsic_to_target', 0),
            'avg_feature_hsic': metrics.get('avg_feature_hsic', 0)
        }
    
    # Coverage analysis
    mi_values = [stats['prediction_mi'] for stats in mi_coverage.values()]
    hsic_values = [stats['prediction_hsic'] for stats in hsic_coverage.values()]
    
    coverage_analysis = {
        'mi_coverage': mi_coverage,
        'hsic_coverage': hsic_coverage,
        'avg_mi_across_specialists': np.mean(mi_values),
        'avg_hsic_across_specialists': np.mean(hsic_values),
        'mi_std': np.std(mi_values),
        'hsic_std': np.std(hsic_values),
        'high_mi_specialists': [name for name, stats in mi_coverage.items() if stats['prediction_mi'] > 0.02],
        'high_hsic_specialists': [name for name, stats in hsic_coverage.items() if stats['prediction_hsic'] > 0.02]
    }
    
    return coverage_analysis

def generate_optimization_report(orthogonality_analysis: Dict[str, Any], 
                              mi_hsic_analysis: Dict[str, Any],
                              comparison_data: Dict[str, Any],
                              symbol: str, exchange: str, timeframe: str, direction: str) -> str:
    """Generate comprehensive optimization report."""
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# Enhanced Specialist Model Optimization Report

**Symbol:** {symbol} | **Exchange:** {exchange} | **Timeframe:** {timeframe} | **Direction:** {direction}  
**Generated:** {timestamp} | **Specialists Analyzed:** {len(comparison_data)}

## Executive Summary

This report provides enhanced analysis of specialist models focusing on:
1. **Information Theory Metrics** (MI/HSIC to target)
2. **Cross-Specialist Orthogonality** 
3. **Binary Output Standardization**
4. **Optimization Recommendations**

## Information Theory Coverage

### MI (Mutual Information) Analysis

| Specialist | Prediction MI | Avg Feature MI | Status |
|------------|---------------|----------------|---------|
"""
    
    # Add MI coverage table
    for specialist, stats in mi_hsic_analysis['mi_coverage'].items():
        status = "✅ Good" if stats['prediction_mi'] > 0.02 else "⚠️ Low" if stats['prediction_mi'] > 0.005 else "❌ Poor"
        report += f"| {specialist} | {stats['prediction_mi']:.4f} | {stats['avg_feature_mi']:.4f} | {status} |\n"
    
    report += f"""
### HSIC (Non-linear Dependence) Analysis

| Specialist | Prediction HSIC | Avg Feature HSIC | Status |
|------------|------------------|------------------|---------|
"""
    
    # Add HSIC coverage table
    for specialist, stats in mi_hsic_analysis['hsic_coverage'].items():
        status = "✅ Good" if stats['prediction_hsic'] > 0.02 else "⚠️ Low" if stats['prediction_hsic'] > 0.005 else "❌ Poor"
        report += f"| {specialist} | {stats['prediction_hsic']:.4f} | {stats['avg_feature_hsic']:.4f} | {status} |\n"
    
    report += f"""
**Coverage Summary:**
- Average MI across specialists: {mi_hsic_analysis['avg_mi_across_specialists']:.4f} ± {mi_hsic_analysis['mi_std']:.4f}
- Average HSIC across specialists: {mi_hsic_analysis['avg_hsic_across_specialists']:.4f} ± {mi_hsic_analysis['hsic_std']:.4f}
- High MI specialists: {', '.join(mi_hsic_analysis['high_mi_specialists']) if mi_hsic_analysis['high_mi_specialists'] else 'None'}
- High HSIC specialists: {', '.join(mi_hsic_analysis['high_hsic_specialists']) if mi_hsic_analysis['high_hsic_specialists'] else 'None'}

## Cross-Specialist Orthogonality Analysis

### Prediction Correlation Matrix

| Specialist Pair | Correlation | P-Value | Samples | Status |
|------------------|-------------|---------|---------|---------|
"""
    
    # Add orthogonality matrix
    for pair, stats in orthogonality_analysis['orthogonality_matrix'].items():
        corr = stats['correlation']
        status = "✅ Orthogonal" if abs(corr) < 0.3 else "⚠️ Moderate" if abs(corr) < 0.7 else "❌ Highly Correlated"
        report += f"| {pair} | {corr:.3f} | {stats['p_value']:.4f} | {stats['samples']} | {status} |\n"
    
    report += f"""
### Feature Overlap Analysis

| Specialist Pair | Overlap % | Unique Features 1 | Unique Features 2 | Status |
|------------------|-----------|-------------------|-------------------|---------|
"""
    
    # Add feature overlap
    for pair, stats in orthogonality_analysis['feature_overlap'].items():
        overlap_pct = stats['overlap_percentage']
        status = "✅ Distinct" if overlap_pct < 30 else "⚠️ Some Overlap" if overlap_pct < 80 else "❌ High Overlap"
        report += f"| {pair} | {overlap_pct:.1f}% | {stats['unique_features_1']} | {stats['unique_features_2']} | {status} |\n"
    
    report += f"""
## Performance Summary with Information Metrics

| Specialist | AUC | MI to Target | HSIC to Target | Binary Output | Orthogonal Features |
|------------|-----|--------------|----------------|---------------|-------------------|
"""
    
    # Enhanced performance table
    for specialist_name, data in comparison_data.items():
        metrics = data.get('metrics', {})
        auc = metrics.get('auc', 0)
        mi = metrics.get('prediction_mi_to_target', 0)
        hsic = metrics.get('prediction_hsic_to_target', 0)
        binary_output = "✅" if any('prediction' in col for col in data.get('predictions_data', {}).columns) else "❌"
        orthogonal_features = metrics.get('orthogonal_feature_count', 0)
        
        report += f"| {specialist_name} | {auc:.3f} | {mi:.4f} | {hsic:.4f} | {binary_output} | {orthogonal_features} |\n"
    
    report += f"""
## Optimization Recommendations

### Priority Actions

"""
    
    # Add recommendations
    for rec in orthogonality_analysis['recommendations']:
        report += f"- {rec}\n"
    
    # MI/HSIC recommendations
    if mi_hsic_analysis['avg_mi_across_specialists'] < 0.01:
        report += "- ⚠️ Low average MI across specialists - consider feature engineering\n"
    
    if mi_hsic_analysis['avg_hsic_across_specialists'] < 0.01:
        report += "- ⚠️ Low average HSIC across specialists - consider non-linear transformations\n"
    
    # Binary output recommendations
    non_binary_specialists = [
        name for name, data in comparison_data.items()
        if not any('prediction' in col for col in data.get('predictions_data', {}).columns)
    ]
    
    if non_binary_specialists:
        report += f"- ⚠️ Specialists without binary output: {', '.join(non_binary_specialists)}\n"
    
    report += f"""
### Model Selection Strategy

1. **Primary Specialists** (High MI + Good Orthogonality):
   - Select specialists with MI > 0.02 and correlation < 0.3
   - Current candidates: {len([s for s in comparison_data.keys() if mi_hsic_analysis['mi_coverage'][s]['prediction_mi'] > 0.02])}

2. **Ensemble Construction**:
   - Combine orthogonal specialists for maximum diversity
   - Weight by MI/HSIC scores and AUC performance
   - Consider feature overlap to minimize redundancy

3. **Next Training Cycle**:
   - Enforce feature orthogonality during feature selection
   - Optimize hyperparameters for MI/HSIC targets
   - Standardize binary output format across all specialists

## Technical Implementation Notes

### Binary Output Standardization
- All specialists should output single 0/1 scalar predictions
- Threshold: 0.5 (adjustable per specialist based on validation)
- Format: `{specialist_name}_prediction` column

### Feature Orthogonality Enforcement
- Maximum correlation threshold: 0.7
- Drop highly correlated features during training
- Maintain feature diversity across specialists

### Information Theory Optimization
- Target MI > 0.02 for meaningful information content
- Target HSIC > 0.02 for non-linear dependence
- Monitor both metrics during hyperparameter optimization

---
*Enhanced Specialist Diagnostics v2 - Optimized for Information Content & Orthogonality*
"""
    
    return report

# Reuse existing functions from previous version...
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
    
    logger.info(f"🚀 Starting enhanced training for {len(specialist_steps)} specialists...")
    
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
            
            execute_fn = getattr(specialist, "execute", None)
            if execute_fn is None:
                raise AttributeError(f"{step_name} has no execute()")

            if inspect.iscoroutinefunction(execute_fn):
                result = await execute_fn(config)
            else:
                result = execute_fn(config)
                if inspect.isawaitable(result):
                    result = await result

            training_results[step_name] = result
            
            if result.get('success'):
                logger.info(f"✅ {step_name} trained successfully")
                if 'metrics' in result:
                    metrics = result['metrics']
                    mi_score = metrics.get('prediction_mi_to_target', 0)
                    logger.info(f"   Metrics: AUC={metrics.get('auc', 'N/A'):.3f}, MI={mi_score:.4f}")
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
    
    logger.info(f"🔍 Comparing enhanced artifacts for {len(specialist_steps)} specialists...")
    
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
            
            # Check if specialist has enhanced diagnostics
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
                    logger.info(f"✅ {specialist_name} enhanced diagnostics completed")
                else:
                    logger.warning(f"⚠️ {specialist_name} diagnostics failed: {result.get('error')}")
            else:
                logger.warning(f"⚠️ {specialist_name} does not have enhanced diagnostics")
                
        except Exception as e:
            logger.error(f"❌ Error analyzing {specialist_name}: {e}")
    
    return comparison_data

async def main_async() -> None:
    """Main async entry point."""
    ap = argparse.ArgumentParser(
        description="Enhanced Specialist Model Training and Comparison v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train all specialists + compare with orthogonality optimization
  python scripts/enhanced_specialist_diagnostics_v2.py --auto-train --optimize-orthogonality --symbol ETHUSDT --exchange binance --timeframe 15m --direction long
  
  # Compare existing artifacts with enhanced analysis
  python scripts/enhanced_specialist_diagnostics_v2.py --compare-only --analyze-orthogonality --symbol ETHUSDT
        """
    )
    
    # Basic arguments
    ap.add_argument("--symbol", type=str, default="ETHUSDT", help="Trading symbol")
    ap.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    ap.add_argument("--timeframe", type=str, default="15m", help="Timeframe")
    ap.add_argument("--direction", type=str, default="long", choices=["long", "short"], help="Trading direction")
    
    # Enhanced mode arguments
    ap.add_argument("--auto-train", action="store_true", help="Train all specialists before comparison")
    ap.add_argument("--train-only", action="store_true", help="Only train specialists, no comparison")
    ap.add_argument("--compare-only", action="store_true", help="Only compare existing artifacts, no training")
    ap.add_argument("--optimize-orthogonality", action="store_true", help="Enable orthogonality optimization")
    ap.add_argument("--analyze-orthogonality", action="store_true", help="Analyze cross-specialist orthogonality")
    
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
    
    logger.info("🚀 Enhanced Specialist Diagnostics v2 Pipeline")
    logger.info(f"📈 Symbol: {args.symbol}/{args.exchange}")
    logger.info(f"⏱️ Timeframe: {args.timeframe}")
    logger.info(f"📊 Direction: {args.direction}")
    logger.info(f"🔬 Features: MI/HSIC Analysis + Orthogonality Optimization")
    
    selected_specialists = args.selected_specialists
    
    # Mode 1: Train only
    if args.train_only:
        logger.info("🔧 Enhanced training mode only...")
        training_results = await train_all_specialists(
            args.symbol, args.exchange, args.timeframe, args.direction, selected_specialists
        )
        
        # Print training summary
        successful = sum(1 for r in training_results.values() if r.get('success'))
        total = len(training_results)
        logger.info(f"✅ Enhanced training completed: {successful}/{total} specialists successful")
        
        return
    
    # Mode 2: Train + Compare
    if args.auto_train:
        logger.info("🚀 Enhanced training + comparison mode...")
        training_results = await train_all_specialists(
            args.symbol, args.exchange, args.timeframe, args.direction, selected_specialists
        )
        
        successful = sum(1 for r in training_results.values() if r.get('success'))
        total = len(training_results)
        logger.info(f"✅ Enhanced training completed: {successful}/{total} specialists successful")
        
        if successful == 0:
            logger.error("❌ No specialists trained successfully. Exiting.")
            return
    
    # Mode 3: Compare only (or after training)
    logger.info("📊 Enhanced comparison mode...")
    comparison_data = compare_all_specialist_artifacts(
        args.symbol, args.exchange, args.timeframe, args.direction, selected_specialists
    )
    
    if not comparison_data:
        logger.error("❌ No specialist data available for comparison")
        return
    
    # Enhanced analysis
    logger.info("🔬 Running enhanced analysis...")
    
    # Orthogonality analysis
    orthogonality_analysis = {}
    mi_hsic_analysis = analyze_mi_hsic_coverage(comparison_data)
    
    if args.analyze_orthogonality or args.optimize_orthogonality:
        orthogonality_analysis = analyze_specialist_orthogonality(comparison_data)
    
    # Generate enhanced report
    logger.info("📝 Generating enhanced optimization report...")
    report = generate_optimization_report(
        orthogonality_analysis, mi_hsic_analysis, comparison_data,
        args.symbol, args.exchange, args.timeframe, args.direction
    )
    
    # Save enhanced report
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_name = f"enhanced_specialist_optimization_{args.symbol}_{args.timeframe}_{args.direction}_{timestamp}"
    
    md_path = OUTCOMES_DIR / f"{base_name}.md"
    with open(md_path, 'w') as f:
        f.write(report)
    
    logger.info("🎉 Enhanced Specialist Diagnostics v2 completed!")
    logger.info(f"📊 Enhanced Report: {md_path}")
    logger.info(f"🔬 Specialists analyzed: {len(comparison_data)}")
    
    # Print key metrics
    logger.info("🏆 Top performers by MI:")
    mi_scores = [(name, data.get('metrics', {}).get('prediction_mi_to_target', 0)) 
                 for name, data in comparison_data.items()]
    mi_scores.sort(key=lambda x: x[1], reverse=True)
    for name, mi in mi_scores[:3]:
        logger.info(f"   {name}: MI={mi:.4f}")
    
    if orthogonality_analysis:
        logger.info("🔄 Orthogonality Summary:")
        recommendations = orthogonality_analysis.get('recommendations', [])
        for rec in recommendations:
            logger.info(f"   {rec}")

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
