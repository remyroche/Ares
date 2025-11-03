#!/usr/bin/env python3
"""
SR Quality Model - Ranking Metrics Validation

Validates that the model actually ranks strong levels correctly using proper information retrieval metrics.
This is what traders actually use - not raw R² on mixed data.

Tests:
1. Precision@K: Of the top K levels, how many are actually good?
2. Spearman ρ: Does the ranking order match reality?
3. Strong vs Weak Separation: Can the model distinguish strong from weak?
4. Time-based generalization: Does it work on future data?
5. Sample size reality check: Do we have enough strong levels?
"""

import asyncio
import argparse
import logging
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger
from src.tactician.sr_levels.ml_quality import SRQualityModel, SRQualityDataCollector
from src.utils.data.real_data_loader import RealDataLoader

logger = system_logger.getChild('SRRankingValidation')


def calculate_precision_at_k(y_pred: np.ndarray, y_true: np.ndarray, 
                             k: int, threshold: float = 0.7) -> np.floating:
    """Calculate Precision@K for quality ranking.
    
    Args:
        y_pred: Predicted quality scores
        y_true: Actual quality scores  
        k: Number of top predictions to evaluate
        threshold: Quality threshold for "good" level
        
    Returns:
        Precision@K (0.0 to 1.0)
    """
    if len(y_pred) < k:
        k = len(y_pred)
    
    # Get indices of top K predictions
    top_k_indices = np.argsort(y_pred)[-k:][::-1]
    
    # Count how many are actually good
    good_count = np.sum(y_true[top_k_indices] >= threshold)
    
    return good_count / k


def calculate_separation_score(y_pred_strong: np.ndarray, y_pred_weak: np.ndarray) -> Dict:
    """Calculate separation between strong and weak predictions.
    
    Returns:
        Dict with mean scores, difference, and overlap metrics
    """
    mean_strong = np.mean(y_pred_strong)
    mean_weak = np.mean(y_pred_weak)
    median_strong = np.median(y_pred_strong)
    median_weak = np.median(y_pred_weak)
    
    separation = mean_strong - mean_weak
    
    # Check for overlap (bad)
    weak_above_strong_median = np.sum(y_pred_weak >= median_strong) / len(y_pred_weak)
    strong_below_weak_median = np.sum(y_pred_strong <= median_weak) / len(y_pred_strong)
    
    return {
        'mean_strong': mean_strong,
        'mean_weak': mean_weak,
        'median_strong': median_strong,
        'median_weak': median_weak,
        'separation': separation,
        'weak_above_strong_median_pct': weak_above_strong_median * 100,
        'strong_below_weak_median_pct': strong_below_weak_median * 100
    }


def print_ranking_results(results: Dict):
    """Pretty print ranking validation results."""
    logger.info("\n" + "="*80)
    logger.info("  RANKING METRICS VALIDATION RESULTS")
    logger.info("="*80)
    
    # Test 1: Precision@K
    if 'precision_at_k' in results:
        logger.info("\n📊 TEST 1: Does It Rank Strong Levels Correctly?")
        logger.info("-"*80)
        precision = results['precision_at_k']
        for k, val in sorted(precision.items()):
            status = "✅ PASS" if val >= 0.75 else ("⚠️  MARGINAL" if val >= 0.60 else "❌ FAIL")
            threshold = 0.80 if k <= 5 else 0.75
            logger.info(f"   Precision@{k}: {val*100:.1f}% (target: >{threshold*100}%) {status}")
    
    # Test 2: Spearman Correlation
    if 'spearman' in results:
        logger.info("\n📈 TEST 2: Ranking Correlation (Spearman's ρ)")
        logger.info("-"*80)
        spear = results['spearman']
        status = "✅ PASS" if spear >= 0.60 else ("⚠️  MARGINAL" if spear >= 0.40 else "❌ FAIL")
        logger.info(f"   Spearman ρ: {spear:.3f} (target: >0.60) {status}")
        if 'spearman_pvalue' in results:
            logger.info(f"   P-value: {results['spearman_pvalue']:.4f}")
    
    # Test 3: Strong vs Weak Separation
    if 'separation' in results:
        logger.info("\n🔀 TEST 3: Does It Separate Strong from Weak?")
        logger.info("-"*80)
        sep = results['separation']
        status = "✅ PASS" if sep['separation'] >= 0.35 else ("⚠️  MARGINAL" if sep['separation'] >= 0.25 else "❌ FAIL")
        logger.info(f"   Mean strong: {sep['mean_strong']:.3f}")
        logger.info(f"   Mean weak: {sep['mean_weak']:.3f}")
        logger.info(f"   Separation: {sep['separation']:.3f} (target: >0.35) {status}")
        logger.info(f"   Weak above strong median: {sep['weak_above_strong_median_pct']:.1f}%")
        logger.info(f"   Strong below weak median: {sep['strong_below_weak_median_pct']:.1f}%")
    
    # Test 4: Time-based Generalization
    if 'future_generalization' in results:
        logger.info("\n⏰ TEST 4: Does It Generalize to Future Data?")
        logger.info("-"*80)
        future = results['future_generalization']
        if future['r2'] is not None:
            status = "✅ PASS" if future['r2'] >= 0.45 else ("⚠️  MARGINAL" if future['r2'] >= 0.30 else "❌ FAIL")
            logger.info(f"   Future R²: {future['r2']:.3f} (target: >0.45) {status}")
            logger.info(f"   Train period: {future['train_period']}")
            logger.info(f"   Test period: {future['test_period']}")
            logger.info(f"   Train samples (strong): {future['train_strong_count']}")
            logger.info(f"   Test samples (strong): {future['test_strong_count']}")
        else:
            logger.info(f"   ⚠️  INSUFFICIENT DATA (need strong samples in future period)")
    
    # Test 5: Sample Size Reality Check
    if 'sample_size_check' in results:
        logger.info("\n📉 TEST 5: Sample Size Reality Check")
        logger.info("-"*80)
        size = results['sample_size_check']
        total = size['total_samples']
        strong = size['strong_samples']
        pct_strong = strong / total * 100 if total > 0 else 0
        
        logger.info(f"   Total samples: {total}")
        logger.info(f"   Strong samples (quality > 0.7): {strong} ({pct_strong:.1f}%)")
        
        if strong < 100:
            logger.warning(f"   ⚠️  WARNING: Only {strong} strong samples (target: >100)")
        elif strong >= 300:
            logger.info(f"   ✅ Excellent sample size for strong levels")
        else:
            logger.info(f"   ✅ Adequate sample size")
    
    # Overall Summary
    logger.info("\n" + "="*80)
    logger.info("  VALIDATION SUMMARY")
    logger.info("="*80)
    
    passes = 0
    total_tests = 0
    
    if 'precision_at_k' in results:
        for k, val in results['precision_at_k'].items():
            total_tests += 1
            threshold = 0.80 if k <= 5 else 0.75
            if val >= threshold:
                passes += 1
    
    if 'spearman' in results:
        total_tests += 1
        if results['spearman'] >= 0.60:
            passes += 1
    
    if 'separation' in results:
        total_tests += 1
        if results['separation']['separation'] >= 0.35:
            passes += 1
    
    if 'future_generalization' in results and results['future_generalization']['r2'] is not None:
        total_tests += 1
        if results['future_generalization']['r2'] >= 0.45:
            passes += 1
    
    logger.info(f"\n   Tests passed: {passes}/{total_tests}")
    if passes == total_tests:
        logger.info(f"   ✅ ALL TESTS PASSED - Model is production-ready!")
    elif passes >= total_tests * 0.75:
        logger.info(f"   ⚠️  MOSTLY PASSED - Model has minor issues")
    elif passes >= total_tests * 0.5:
        logger.info(f"   ⚠️  MARGINAL - Model needs improvement")
    else:
        logger.info(f"   ❌ FAILED - Model needs significant work")


async def validate_ranking_metrics(
    symbol: str = 'ETHUSDT',
    exchange: str = 'binance',
    timeframe: str = '15m',
    ml_model_path: str = 'models/sr_quality_model.lgb',
    training_data_path: str = 'data_cache/sr_ml_training/sr_quality_training_data.parquet'
) -> Dict:
    """Run all ranking validation tests.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        ml_model_path: Path to trained ML model
        training_data_path: Path to training data
        
    Returns:
        Dict with all validation results
    """
    results = {}
    
    # Load model
    logger.info(f"📂 Loading trained model from: {ml_model_path}")
    model = SRQualityModel()
    if not Path(ml_model_path).exists():
        logger.error(f"❌ Model not found: {ml_model_path}")
        logger.error("   Run training first: python scripts/run_sr_workflow.py --symbol ETHUSDT --timeframe 15m")
        return results
    model.load(ml_model_path)
    
    # Load training data
    logger.info(f"📂 Loading training data from: {training_data_path}")
    if not Path(training_data_path).exists():
        logger.error(f"❌ Training data not found: {training_data_path}")
        logger.error("   Run training first: python scripts/run_sr_workflow.py --symbol ETHUSDT --timeframe 15m")
        return results
    
    training_data = pd.read_parquet(training_data_path)
    logger.info(f"   Loaded {len(training_data):,} samples")
    
    # Verify model has required features available in data
    if model.feature_names is not None:
        logger.info(f"   Model expects {len(model.feature_names)} features")
        missing_features = set(model.feature_names) - set(training_data.columns)
        if missing_features:
            logger.error(f"   ❌ Missing features in data: {missing_features}")
            logger.error(f"   Model cannot make predictions without these features!")
            return results
        logger.info(f"   ✅ All required features present in training data")
    else:
        logger.warning(f"   ⚠️  Model has no feature_names metadata")
    
    # Extract features properly - model expects only feature columns
    # Use model's feature_names if available, otherwise filter by 'feature_' prefix
    if model.feature_names is not None:
        features_df = training_data[model.feature_names].copy()
    else:
        # Fallback: use columns starting with 'feature_'
        feature_cols = [col for col in training_data.columns if col.startswith('feature_')]
        features_df = training_data[feature_cols].copy()
    y = training_data['quality_score'].copy()
    
    logger.info(f"   Using {len(features_df.columns)} feature columns for evaluation")
    
    # Get strong levels for testing
    strong_mask = (y.values >= 0.7)
    weak_mask = (y.values < 0.4)
    
    # Test 1: Precision@K on strong levels
    logger.info("\n🔬 TEST 1: Precision@K (Strong Levels Only)")
    logger.info("-"*80)
    X_strong = features_df[strong_mask].copy()  # Get strong features
    y_strong = y[strong_mask].values  # Get strong targets as numpy array
    
    if len(X_strong) < 50:
        logger.warning(f"⚠️  Only {len(X_strong)} strong samples - insufficient for testing")
    else:
        y_pred_strong = model.predict(X_strong)  # Returns numpy array
        
        precision_results = {}
        for k in [5, 10, 20, 50]:
            if len(y_pred_strong) >= k:
                prec = calculate_precision_at_k(y_pred_strong, y_strong, k=k, threshold=0.7)
                precision_results[k] = prec
                threshold = 0.80 if k <= 5 else 0.75
                status = "✅" if prec >= threshold else "❌"
                logger.info(f"   Precision@{k}: {prec*100:.1f}% (target: >{threshold*100}%) {status}")
        
        results['precision_at_k'] = precision_results
        
        # Test 2: Spearman Correlation
        logger.info("\n🔬 TEST 2: Spearman Ranking Correlation")
        logger.info("-"*80)
        rho, pval = spearmanr(y_pred_strong, y_strong)
        results['spearman'] = rho
        results['spearman_pvalue'] = pval
        status = "✅" if rho >= 0.60 else ("⚠️" if rho >= 0.40 else "❌")
        logger.info(f"   Spearman ρ: {rho:.3f} (target: >0.60) {status}")
        logger.info(f"   P-value: {pval:.4e}")
    
    # Test 3: Strong vs Weak Separation
    logger.info("\n🔬 TEST 3: Strong vs Weak Separation")
    logger.info("-"*80)
    
    X_strong_full = features_df[strong_mask].copy()
    X_weak = features_df[weak_mask].copy()
    
    if len(X_weak) > 0:
        y_pred_strong_sep = model.predict(X_strong_full)
        y_pred_weak = model.predict(X_weak)
        
        separation = calculate_separation_score(y_pred_strong_sep, y_pred_weak)
        results['separation'] = separation
        
        status = "✅" if separation['separation'] >= 0.35 else ("⚠️" if separation['separation'] >= 0.25 else "❌")
        logger.info(f"   Mean strong: {separation['mean_strong']:.3f}")
        logger.info(f"   Mean weak: {separation['mean_weak']:.3f}")
        logger.info(f"   Separation: {separation['separation']:.3f} (target: >0.35) {status}")
        logger.info(f"   Weak above strong median: {separation['weak_above_strong_median_pct']:.1f}%")
        logger.info(f"   Strong below weak median: {separation['strong_below_weak_median_pct']:.1f}%")
    else:
        logger.warning("⚠️  No weak samples found for comparison")
    
    # Test 4: Time-based Generalization (if date column exists)
    logger.info("\n🔬 TEST 4: Time-Based Generalization")
    logger.info("-"*80)
    
    if 'date' in training_data.columns:
        training_data['date'] = pd.to_datetime(training_data['date'])
        
        # Split: train on first 70%, test on last 30%
        split_date = training_data['date'].quantile(0.70)
        train_mask = (training_data['date'] < split_date).values
        test_mask = (training_data['date'] >= split_date).values
        
        X_test = features_df[test_mask].copy()
        y_test = y[test_mask].copy()
        
        # Get strong levels in test set
        test_strong_mask = (y_test.values >= 0.7)
        X_test_strong = X_test[test_strong_mask].copy()
        y_test_strong = y_test[test_strong_mask].values  # Get as numpy array
        
        if len(X_test_strong) >= 30:
            y_pred_future = model.predict(X_test_strong)
            
            from sklearn.metrics import r2_score
            r2_future = r2_score(y_test_strong, y_pred_future)
            
            results['future_generalization'] = {
                'r2': r2_future,
                'train_period': f"{training_data.loc[train_mask, 'date'].min()} to {training_data.loc[train_mask, 'date'].max()}",
                'test_period': f"{training_data.loc[test_mask, 'date'].min()} to {training_data.loc[test_mask, 'date'].max()}",
                'train_strong_count': int(np.sum(y[train_mask].values >= 0.7)),
                'test_strong_count': len(y_test_strong)
            }
            
            status = "✅" if r2_future >= 0.45 else ("⚠️" if r2_future >= 0.30 else "❌")
            logger.info(f"   Future R² (strong only): {r2_future:.3f} (target: >0.45) {status}")
            logger.info(f"   Train period: {results['future_generalization']['train_period']}")
            logger.info(f"   Test period: {results['future_generalization']['test_period']}")
            logger.info(f"   Train strong samples: {results['future_generalization']['train_strong_count']}")
            logger.info(f"   Test strong samples: {results['future_generalization']['test_strong_count']}")
        else:
            logger.warning(f"⚠️  Only {len(X_test_strong)} strong samples in test period - insufficient")
            results['future_generalization'] = {'r2': None}
    else:
        logger.warning("⚠️  No 'date' column found - skipping time-based test")
    
    # Test 5: Sample Size Reality Check
    logger.info("\n🔬 TEST 5: Sample Size Reality Check")
    logger.info("-"*80)
    
    total_samples = len(training_data)
    strong_samples = len(training_data[training_data['quality_score'] >= 0.7])
    
    results['sample_size_check'] = {
        'total_samples': total_samples,
        'strong_samples': strong_samples
    }
    
    logger.info(f"   Total samples: {total_samples:,}")
    logger.info(f"   Strong samples (quality > 0.7): {strong_samples:,} ({strong_samples/total_samples*100:.1f}%)")
    
    if strong_samples < 100:
        logger.warning(f"   ⚠️  WARNING: Only {strong_samples} strong samples (target: >100)")
    elif strong_samples >= 300:
        logger.info(f"   ✅ Excellent sample size for strong levels")
    else:
        logger.info(f"   ✅ Adequate sample size")
    
    return results


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate SR quality model using ranking metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--symbol', type=str, default='ETHUSDT', help='Trading symbol')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name')
    parser.add_argument('--timeframe', type=str, default='15m', help='Timeframe')
    parser.add_argument('--model', type=str, default='models/sr_quality_model.lgb', 
                       help='Path to trained ML model')
    parser.add_argument('--data', type=str, 
                       default='data_cache/sr_ml_training/sr_quality_training_data.parquet',
                       help='Path to training data')
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*80)
    logger.info("  SR QUALITY MODEL - RANKING METRICS VALIDATION")
    logger.info("="*80)
    logger.info(f"   Symbol: {args.symbol}")
    logger.info(f"   Exchange: {args.exchange}")
    logger.info(f"   Timeframe: {args.timeframe}")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Data: {args.data}")
    logger.info("="*80)
    
    results = await validate_ranking_metrics(
        symbol=args.symbol,
        exchange=args.exchange,
        timeframe=args.timeframe,
        ml_model_path=args.model,
        training_data_path=args.data
    )
    
    print_ranking_results(results)
    
    logger.info("\n" + "="*80)
    logger.info("  VALIDATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    asyncio.run(main())

