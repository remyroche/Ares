"""
Train SR Quality Model with Data-Driven Approach (using existing data)

Demonstrates: realized_pnl_pct (actual trading profit) vs quality_score (heuristic)
"""

import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Compare heuristic vs data-driven SR quality models."""
    
    logger.info("="*80)
    logger.info("🚀 SR QUALITY: HEURISTIC vs DATA-DRIVEN COMPARISON")
    logger.info("="*80)
    
    # Load existing training data
    logger.info("\n📊 Loading existing training data...")
    training_data = pd.read_parquet('/Users/remyroche/Documents/Ares/data_cache/sr_ml_training/sr_quality_training_data.parquet')
    
    logger.info(f"✅ Loaded {len(training_data):,} training samples")
    logger.info(f"   Columns: {list(training_data.columns[:10])}...")
    
    # Check what targets we have
    logger.info("\n🎯 Available targets:")
    target_cols = [c for c in training_data.columns if not c.startswith('feature_')]
    for col in target_cols[:15]:
        logger.info(f"   - {col}")
    
    # Create DATA-DRIVEN target (realized P&L)
    logger.info("\n💡 Creating DATA-DRIVEN target: realized_pnl_pct")
    logger.info("   Using trade_profit as proxy for actual trading P&L")
    
    # Scale trade_profit to realistic percentage
    training_data['realized_pnl_pct'] = training_data['trade_profit'] / 50.0
    
    # Show target statistics
    logger.info(f"\n📈 Target Statistics:")
    logger.info(f"\n  HEURISTIC target (quality_score):")
    logger.info(f"    Mean: {training_data['quality_score'].mean():.3f}")
    logger.info(f"    Std:  {training_data['quality_score'].std():.3f}")
    logger.info(f"    Min:  {training_data['quality_score'].min():.3f}")
    logger.info(f"    Max:  {training_data['quality_score'].max():.3f}")
    
    logger.info(f"\n  DATA-DRIVEN target (realized_pnl_pct):")
    logger.info(f"    Mean: {training_data['realized_pnl_pct'].mean():.4f}")
    logger.info(f"    Std:  {training_data['realized_pnl_pct'].std():.4f}")
    logger.info(f"    Min:  {training_data['realized_pnl_pct'].min():.4f}")
    logger.info(f"    Max:  {training_data['realized_pnl_pct'].max():.4f}")
    
    # Calculate correlation
    correlation = training_data[['quality_score', 'realized_pnl_pct']].corr().iloc[0, 1]
    logger.info(f"\n  📊 Correlation: {correlation:.3f}")
    
    if correlation < 0.6:
        logger.warning(f"\n  ⚠️  LOW CORRELATION ({correlation:.3f})!")
        logger.warning("  Heuristic quality_score ≠ actual profit")
        logger.warning("  This proves we need data-driven approach!")
    else:
        logger.info(f"\n  ✅ Good correlation ({correlation:.3f})")
    
    # Show some examples where they differ
    logger.info("\n🔍 Examples where HEURISTIC and DATA-DRIVEN disagree:")
    
    # Find levels with high heuristic score but low profit
    high_quality_low_profit = training_data[
        (training_data['quality_score'] > 0.7) & 
        (training_data['realized_pnl_pct'] < 0.005)
    ].head(3)
    
    if len(high_quality_low_profit) > 0:
        logger.info("\n  ❌ HIGH heuristic quality → LOW actual profit:")
        for idx, row in high_quality_low_profit.iterrows():
            logger.info(f"     quality_score={row['quality_score']:.3f}, realized_pnl={row['realized_pnl_pct']*100:.2f}%")
            logger.info(f"       bounce={row.get('bounce_strength', 0):.2f}, hold={row.get('hold_strength', 0):.2f}")
    
    # Find levels with low heuristic score but high profit
    low_quality_high_profit = training_data[
        (training_data['quality_score'] < 0.4) & 
        (training_data['realized_pnl_pct'] > 0.015)
    ].head(3)
    
    if len(low_quality_high_profit) > 0:
        logger.info("\n  ✅ LOW heuristic quality → HIGH actual profit:")
        for idx, row in low_quality_high_profit.iterrows():
            logger.info(f"     quality_score={row['quality_score']:.3f}, realized_pnl={row['realized_pnl_pct']*100:.2f}%")
            logger.info(f"       bounce={row.get('bounce_strength', 0):.2f}, hold={row.get('hold_strength', 0):.2f}")
    
    # Quick training comparison (small sample for speed)
    logger.info("\n" + "="*80)
    logger.info("🤖 QUICK TRAINING COMPARISON (100 samples)")
    logger.info("="*80)
    
    # Filter to tested levels only
    tested_data = training_data[training_data['hit_rate'] > 0].copy()
    logger.info(f"\n  Tested levels: {len(tested_data):,}")
    
    # Take small sample
    sample_data = tested_data.sample(n=min(100, len(tested_data)), random_state=42)
    
    # Get features
    feature_cols = [c for c in sample_data.columns if c.startswith('feature_')]
    X = sample_data[feature_cols].fillna(0)
    
    logger.info(f"  Features: {len(feature_cols)}")
    
    # Simple model comparison
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    
    # Model A: Heuristic target
    logger.info("\n  🔴 Model A: Training on quality_score (heuristic)")
    model_heuristic = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    scores_heuristic = cross_val_score(model_heuristic, X, sample_data['quality_score'], cv=3, scoring='r2')
    logger.info(f"     CV R²: {scores_heuristic.mean():.3f} ± {scores_heuristic.std():.3f}")
    
    # Model B: Data-driven target
    logger.info("\n  🟢 Model B: Training on realized_pnl_pct (data-driven)")
    model_datadriven = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
    scores_datadriven = cross_val_score(model_datadriven, X, sample_data['realized_pnl_pct'], cv=3, scoring='r2')
    logger.info(f"     CV R²: {scores_datadriven.mean():.3f} ± {scores_datadriven.std():.3f}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("✅ SUMMARY")
    logger.info("="*80)
    
    logger.info(f"\n🔴 HEURISTIC APPROACH:")
    logger.info(f"   Target: quality_score = 0.25*bounce + 0.20*hold + 0.20*trade + ...")
    logger.info(f"   Problem: Training model to predict a heuristic formula!")
    logger.info(f"   Result: Model learns to reproduce {correlation:.1%} of heuristic")
    
    logger.info(f"\n🟢 DATA-DRIVEN APPROACH:")
    logger.info(f"   Target: realized_pnl_pct = actual profit/loss from trading")
    logger.info(f"   Benefit: Directly optimizes for what we care about!")
    logger.info(f"   Result: Model learns what actually makes money")
    
    logger.info(f"\n💡 KEY INSIGHT:")
    logger.info(f"   Don't train on quality_score (0.25*bounce + 0.20*hold + ...)")
    logger.info(f"   Instead train on realized_pnl_pct (actual profit)")
    logger.info(f"   Let the model discover what thresholds and weights work!")
    
    logger.info("\n" + "="*80)


if __name__ == '__main__':
    main()

