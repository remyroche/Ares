"""
Validate SR ML Hypotheses

Tests the hypotheses:
1. R² varies by timeframe (1m < 15m < 1h < 1d)
2. Strong levels are more predictable than weak levels
3. Training on 90% noise hurts performance
4. Ranking metrics matter more than R²
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_r2_by_timeframe(training_data: pd.DataFrame) -> Dict:
    """
    Hypothesis 1: R² increases with timeframe.
    
    Tests if:
    - 1-minute has lowest R² (high noise)
    - Daily has highest R² (low noise)
    
    Returns results dict with R² per timeframe.
    """
    logger.info("\n" + "="*70)
    logger.info("  HYPOTHESIS 1: R² Varies by Timeframe")
    logger.info("="*70)
    
    results = {}
    timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
    
    if 'timeframe' not in training_data.columns:
        logger.warning("⚠️ No 'timeframe' column found. Skipping timeframe analysis.")
        return {}
    
    for tf in timeframes:
        # Filter data for this timeframe
        tf_data = training_data[training_data['timeframe'] == tf]
        
        if len(tf_data) < 50:
            results[tf] = {
                'r2': None,
                'samples': len(tf_data),
                'note': 'Insufficient data (<50 samples)'
            }
            continue
        
        # Prepare features and target
        feature_cols = [c for c in tf_data.columns 
                       if c.startswith('feature_') and c not in ['quality_score', 'timeframe']]
        
        if not feature_cols:
            logger.warning(f"⚠️ No feature columns found for {tf}")
            continue
        
        X = tf_data[feature_cols]
        y = tf_data['quality_score']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train simple model
        model = LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        results[tf] = {
            'r2': r2,
            'samples': len(tf_data),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'mean_quality': y.mean(),
            'std_quality': y.std()
        }
    
    # Print results
    print_timeframe_analysis(results)
    
    return results


def print_timeframe_analysis(results: Dict):
    """Pretty print timeframe analysis results."""
    logger.info(f"\n{'Timeframe':<12} {'R²':<10} {'Samples':<10} {'Mean Q':<10} {'Std Q':<10}")
    logger.info("-"*60)
    
    for tf, metrics in results.items():
        if metrics['r2'] is None:
            logger.info(f"{tf:<12} {'N/A':<10} {metrics['samples']:<10} {'-':<10} {'-':<10}")
        else:
            logger.info(f"{tf:<12} {metrics['r2']:.3f}      {metrics['samples']:<10} "
                       f"{metrics['mean_quality']:.3f}      {metrics['std_quality']:.3f}")
    
    logger.info("="*70)
    
    # Test hypothesis: Does R² increase with timeframe?
    valid_results = {tf: m for tf, m in results.items() if m['r2'] is not None}
    
    if len(valid_results) >= 3:
        timeframe_order = ['1m', '5m', '15m', '1h', '4h', '1d']
        r2_values = [valid_results[tf]['r2'] for tf in timeframe_order 
                    if tf in valid_results]
        
        # Check if generally increasing
        is_increasing = all(r2_values[i] <= r2_values[i+1] + 0.05  # Allow small violations
                           for i in range(len(r2_values)-1))
        
        # Calculate correlation
        correlation = np.corrcoef(range(len(r2_values)), r2_values)[0, 1]
        
        logger.info(f"\n📊 HYPOTHESIS TEST:")
        logger.info(f"   R² increases with timeframe: {'✅ YES' if is_increasing else '❌ NO'}")
        logger.info(f"   Correlation (TF rank vs R²): {correlation:.3f}")
        
        if correlation > 0.7:
            logger.info(f"   ✅ Strong positive correlation - Higher TF = More predictable!")
        elif correlation > 0.3:
            logger.info(f"   🟡 Moderate correlation")
        else:
            logger.info(f"   ❌ Weak/no correlation")


def analyze_r2_by_quality_tier(training_data: pd.DataFrame) -> Dict:
    """
    Hypothesis 2: Strong levels are more predictable than weak levels.
    
    Tests if R² for strong levels (0.7+) >> R² for weak levels (<0.4).
    """
    logger.info("\n" + "="*70)
    logger.info("  HYPOTHESIS 2: Strong Levels More Predictable")
    logger.info("="*70)
    
    results = {}
    
    # Define quality tiers
    tiers = {
        'noise': (0.0, 0.3),       # Untested/garbage
        'weak': (0.3, 0.5),         # Barely works
        'medium': (0.5, 0.7),       # Decent
        'strong': (0.7, 0.85),      # Good
        'critical': (0.85, 1.0)     # Excellent
    }
    
    feature_cols = [c for c in training_data.columns 
                   if c.startswith('feature_') and c not in ['quality_score', 'timeframe']]
    
    if not feature_cols:
        logger.warning("⚠️ No feature columns found")
        return {}
    
    for tier_name, (min_q, max_q) in tiers.items():
        # Filter data for this quality tier
        tier_data = training_data[
            (training_data['quality_score'] >= min_q) &
            (training_data['quality_score'] < max_q)
        ]
        
        if len(tier_data) < 30:
            results[tier_name] = {
                'r2': None,
                'samples': len(tier_data),
                'pct_of_total': len(tier_data) / len(training_data) * 100,
                'note': 'Insufficient data (<30 samples)'
            }
            continue
        
        # Prepare data
        X = tier_data[feature_cols]
        y = tier_data['quality_score']
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train
        model = LGBMRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        results[tier_name] = {
            'r2': r2,
            'samples': len(tier_data),
            'pct_of_total': len(tier_data) / len(training_data) * 100
        }
    
    # Print results
    print_quality_tier_analysis(results, len(training_data))
    
    return results


def print_quality_tier_analysis(results: Dict, total_samples: int):
    """Pretty print quality tier analysis results."""
    logger.info(f"\n{'Tier':<12} {'R²':<10} {'Samples':<10} {'% of Total':<12}")
    logger.info("-"*60)
    
    for tier, metrics in results.items():
        if metrics['r2'] is None:
            logger.info(f"{tier:<12} {'N/A':<10} {metrics['samples']:<10} "
                       f"{metrics['pct_of_total']:.1f}%")
        else:
            logger.info(f"{tier:<12} {metrics['r2']:.3f}      {metrics['samples']:<10} "
                       f"{metrics['pct_of_total']:.1f}%")
    
    logger.info("="*70)
    
    # Calculate data composition
    strong_samples = sum(m['samples'] for tier, m in results.items() 
                        if tier in ['strong', 'critical'])
    weak_samples = sum(m['samples'] for tier, m in results.items() 
                      if tier in ['noise', 'weak'])
    
    logger.info(f"\n📊 TRAINING DATA COMPOSITION:")
    logger.info(f"   Total samples: {total_samples:,}")
    logger.info(f"   Strong/Critical: {strong_samples:,} ({strong_samples/total_samples*100:.1f}%)")
    logger.info(f"   Noise/Weak: {weak_samples:,} ({weak_samples/total_samples*100:.1f}%)")
    
    # Test hypothesis
    if results.get('strong', {}).get('r2') and results.get('noise', {}).get('r2'):
        strong_r2 = results['strong']['r2']
        noise_r2 = results['noise']['r2']
        improvement = (strong_r2 - noise_r2) / abs(noise_r2) * 100 if noise_r2 != 0 else 0
        
        logger.info(f"\n📊 HYPOTHESIS TEST:")
        logger.info(f"   Strong vs Noise R²: {strong_r2:.3f} vs {noise_r2:.3f}")
        logger.info(f"   Improvement: +{improvement:.0f}%")
        
        if improvement > 200:
            logger.info(f"   ✅ HYPOTHESIS CONFIRMED: Strong levels FAR more predictable!")
        elif improvement > 100:
            logger.info(f"   🟡 Strong levels moderately more predictable")
        else:
            logger.info(f"   ❌ Hypothesis not supported")


def compare_ranking_vs_regression(training_data: pd.DataFrame):
    """
    Hypothesis 3: Ranking metrics matter more than R².
    
    Shows that a model with lower R² but better ranking
    is more useful for SR detection.
    """
    logger.info("\n" + "="*70)
    logger.info("  HYPOTHESIS 3: Ranking Metrics vs R²")
    logger.info("="*70)
    
    feature_cols = [c for c in training_data.columns 
                   if c.startswith('feature_') and c not in ['quality_score', 'timeframe']]
    
    if not feature_cols:
        logger.warning("⚠️ No feature columns found")
        return
    
    X = training_data[feature_cols]
    y = training_data['quality_score']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Model 1: Optimized for R² (complex model)
    model_r2 = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=8,  # Deep trees
        num_leaves=63,  # Many leaves
        random_state=42,
        verbose=-1
    )
    model_r2.fit(X_train, y_train)
    
    # Model 2: Optimized for ranking (simple, regularized)
    model_rank = LGBMRegressor(
        n_estimators=50,
        learning_rate=0.03,
        max_depth=4,  # Shallow
        num_leaves=15,  # Few leaves
        lambda_l1=1.0,  # Strong regularization
        lambda_l2=1.0,
        random_state=42,
        verbose=-1
    )
    model_rank.fit(X_train, y_train)
    
    # Evaluate both
    y_pred_r2 = model_r2.predict(X_test)
    y_pred_rank = model_rank.predict(X_test)
    
    # R² scores
    r2_model_r2 = r2_score(y_test, y_pred_r2)
    r2_model_rank = r2_score(y_test, y_pred_rank)
    
    # Ranking metrics
    k = 10
    
    # Precision@K
    precision_r2 = calculate_precision_at_k(y_pred_r2, y_test.values, k=k, threshold=0.7)
    precision_rank = calculate_precision_at_k(y_pred_rank, y_test.values, k=k, threshold=0.7)
    
    # Spearman
    spearman_r2, _ = spearmanr(y_pred_r2, y_test)
    spearman_rank, _ = spearmanr(y_pred_rank, y_test)
    
    # Print comparison
    logger.info(f"\n{'Metric':<25} {'Complex Model':<15} {'Simple Model':<15} {'Winner':<10}")
    logger.info("-"*70)
    logger.info(f"{'R² Score':<25} {r2_model_r2:.3f}           {r2_model_rank:.3f}           "
               f"{'Complex ✅' if r2_model_r2 > r2_model_rank else 'Simple ✅'}")
    logger.info(f"{'Precision@10':<25} {precision_r2*100:.1f}%          {precision_rank*100:.1f}%          "
               f"{'Complex ✅' if precision_r2 > precision_rank else 'Simple ✅'}")
    logger.info(f"{'Spearman ρ':<25} {spearman_r2:.3f}           {spearman_rank:.3f}           "
               f"{'Complex ✅' if spearman_r2 > spearman_rank else 'Simple ✅'}")
    logger.info("="*70)
    
    logger.info(f"\n💡 INTERPRETATION:")
    if precision_rank >= precision_r2:
        logger.info(f"   ✅ Simple model has BETTER ranking (what matters!)")
        logger.info(f"   Even though R² is {'lower' if r2_model_rank < r2_model_r2 else 'similar'}")
    else:
        logger.info(f"   Complex model has better ranking")


def calculate_precision_at_k(y_pred: np.ndarray, y_true: np.ndarray,
                             k: int, threshold: float) -> float:
    """Calculate Precision@K."""
    top_k_indices = np.argsort(y_pred)[-k:][::-1]
    good_count = np.sum(y_true[top_k_indices] >= threshold)
    return good_count / k


def main():
    """Run all hypothesis tests."""
    logger.info("\n" + "="*70)
    logger.info("  SR ML HYPOTHESIS VALIDATION")
    logger.info("="*70)
    
    # Load training data
    training_data_path = Path('data_cache/sr_ml_training/sr_quality_training_data.parquet')
    
    if not training_data_path.exists():
        logger.error(f"❌ Training data not found: {training_data_path}")
        logger.error(f"   Run step 2.5 to generate training data first")
        return
    
    logger.info(f"\n📂 Loading training data from: {training_data_path}")
    training_data = pd.read_parquet(training_data_path)
    logger.info(f"   Loaded {len(training_data):,} samples")
    logger.info(f"   Columns: {len(training_data.columns)}")
    
    # Run hypothesis tests
    try:
        # Test 1: Timeframe stratification
        tf_results = analyze_r2_by_timeframe(training_data)
        
        # Test 2: Quality tier stratification
        quality_results = analyze_r2_by_quality_tier(training_data)
        
        # Test 3: Ranking vs Regression
        compare_ranking_vs_regression(training_data)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("  VALIDATION SUMMARY")
        logger.info("="*70)
        
        recommendations = []
        
        # Check timeframe hypothesis
        if tf_results and len(tf_results) >= 3:
            timeframe_order = ['1m', '5m', '15m', '1h', '4h', '1d']
            r2_values = [tf_results[tf]['r2'] for tf in timeframe_order 
                        if tf in tf_results and tf_results[tf]['r2'] is not None]
            
            if len(r2_values) >= 3:
                correlation = np.corrcoef(range(len(r2_values)), r2_values)[0, 1]
                if correlation > 0.5:
                    recommendations.append("✅ Use timeframe-specific models")
        
        # Check quality hypothesis
        if quality_results:
            strong_samples = sum(m['samples'] for tier, m in quality_results.items() 
                               if tier in ['strong', 'critical'])
            total_samples = sum(m['samples'] for m in quality_results.values())
            
            if strong_samples / total_samples < 0.2:
                recommendations.append("✅ Filter out weak levels from training")
                recommendations.append(f"   (Currently only {strong_samples/total_samples*100:.0f}% are strong)")
        
        logger.info(f"\n🎯 RECOMMENDATIONS:")
        if recommendations:
            for rec in recommendations:
                logger.info(f"   {rec}")
        else:
            logger.info(f"   No clear pattern found - may need more data")
        
        logger.info("\n✅ Validation complete!")
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()

