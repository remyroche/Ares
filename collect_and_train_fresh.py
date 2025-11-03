"""
Collect FRESH training data with proper realized_pnl_pct calculation,
then train data-driven model.

This fixes the data quality issue where old data had all zeros.
"""

import pandas as pd
import numpy as np
import logging
import asyncio
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def main():
    """Collect fresh data and train properly."""
    
    logger.info("="*80)
    logger.info("🚀 FRESH DATA COLLECTION & DATA-DRIVEN TRAINING")
    logger.info("="*80)
    
    # =========================================================================
    # STEP 1: Collect FRESH training data with proper realized_pnl_pct
    # =========================================================================
    
    logger.info("\n📊 STEP 1: Collecting FRESH training data...")
    logger.info("   Using MODIFIED data collector with realized_pnl_pct calculation")
    
    from src.tactician.sr_levels.ml_quality.sr_quality_data_collector import SRQualityDataCollector
    
    collector = SRQualityDataCollector()
    
    # Check if we have data available
    logger.info("\n🔍 Checking available market data...")
    
    try:
        # Try to collect from available data
        training_data = await collector.collect_training_data(
            symbol='ETHUSDT',
            exchange='binance',
            start_date='2024-01-01',
            end_date='2024-03-01',  # Shorter period for speed
            timeframe='1h',
            forward_days=10,
            sample_freq_days=7  # Weekly samples
        )
        
        logger.info(f"\n✅ Collected {len(training_data)} fresh samples")
        
        # Verify data quality
        logger.info(f"\n🔍 DATA QUALITY CHECK:")
        logger.info(f"   realized_pnl_pct exists: {'realized_pnl_pct' in training_data.columns}")
        logger.info(f"   realized_pnl_pct mean: {training_data['realized_pnl_pct'].mean():.4f}")
        logger.info(f"   realized_pnl_pct std: {training_data['realized_pnl_pct'].std():.4f}")
        logger.info(f"   Non-zero values: {(training_data['realized_pnl_pct'] != 0).sum()}")
        
        if (training_data['realized_pnl_pct'] == 0).all():
            logger.error("❌ ALL P&L VALUES ARE ZERO! Data quality issue!")
            logger.error("   Possible causes:")
            logger.error("   1. No levels were hit in forward window")
            logger.error("   2. Trade simulation not working")
            logger.error("   3. Data loading issue")
            return
        
        # Save fresh data
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fresh_data_path = f'data_cache/sr_ml_training/sr_quality_training_FRESH_{timestamp}.parquet'
        
        Path(fresh_data_path).parent.mkdir(parents=True, exist_ok=True)
        training_data.to_parquet(fresh_data_path, index=False)
        logger.info(f"\n✅ Fresh training data saved to {fresh_data_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to collect fresh data: {e}")
        logger.error("   Falling back to loading existing good data if available...")
        
        # Try to find any good training data
        import glob
        training_files = glob.glob('data_cache/sr_ml_training/*.parquet')
        
        logger.info(f"\n🔍 Found {len(training_files)} training data files:")
        for f in training_files:
            try:
                df_test = pd.read_parquet(f)
                has_pnl = 'realized_pnl_pct' in df_test.columns
                pnl_mean = df_test['realized_pnl_pct'].mean() if has_pnl else 0
                non_zero = (df_test['realized_pnl_pct'] != 0).sum() if has_pnl else 0
                logger.info(f"   {Path(f).name}: {len(df_test)} samples, P&L mean: {pnl_mean:.4f}, non-zero: {non_zero}")
            except Exception as e2:
                logger.error(f"   Error reading {f}: {e2}")
        
        logger.error("\n❌ Cannot proceed without valid training data!")
        logger.error("   Need to collect fresh data with proper market data available.")
        return
    
    # =========================================================================
    # STEP 2: Train models if we have good data
    # =========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("🤖 STEP 2: Training models on FRESH data")
    logger.info("="*80)
    
    from src.tactician.sr_levels.ml_quality.sr_quality_model import SRQualityModel
    
    # Filter to quality levels
    tested_data = training_data[training_data['quality_score'] > 0.25].copy()
    logger.info(f"   Quality levels (>0.25): {len(tested_data)}")
    
    if len(tested_data) < 50:
        logger.error(f"❌ Insufficient data: {len(tested_data)} samples (need at least 50)")
        return
    
    # Add weights
    tested_data = collector.add_confidence_weights(tested_data, method='quality_based')
    
    # Train DATA-DRIVEN model
    logger.info("\n🟢 Training DATA-DRIVEN model...")
    logger.info("   Target: realized_pnl_pct (actual trading profit)")
    
    model_datadriven = SRQualityModel()
    metrics = model_datadriven.train(
        training_data=tested_data,
        target_column='realized_pnl_pct',
        n_folds=3,
        num_boost_round=500,
        early_stopping_rounds=50
    )
    
    logger.info(f"\n   Validation Metrics:")
    logger.info(f"     R²:   {metrics['avg_metrics']['avg_val_r2']:.3f}")
    logger.info(f"     RMSE: {metrics['avg_metrics']['avg_val_rmse']:.4f}")
    logger.info(f"     MAE:  {metrics['avg_metrics']['avg_val_mae']:.4f}")
    
    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'models/sr_quality/sr_quality_datadriven_{timestamp}.lgb'
    model_datadriven.save(model_path)
    
    logger.info(f"\n✅ Model saved to {model_path}")
    
    # Generate report
    logger.info(f"\n📝 Generating report...")
    
    outcomes_dir = Path('outcomes')
    outcomes_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = outcomes_dir / f'sr_quality_datadriven_training_{timestamp}.md'
    
    report_content = f"""# SR Quality Model: Data-Driven Training Report (FRESH DATA)

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## ✅ Data Quality Verified

### Fresh Training Data
- **Total samples:** {len(training_data):,}
- **Quality samples (>0.25):** {len(tested_data):,}
- **Mean P&L:** {training_data['realized_pnl_pct'].mean()*100:.4f}%
- **Std P&L:** {training_data['realized_pnl_pct'].std()*100:.4f}%
- **Win rate:** {(training_data['realized_pnl_pct'] > 0).sum() / len(training_data) * 100:.1f}%
- **Non-zero P&L:** {(training_data['realized_pnl_pct'] != 0).sum():,} samples

---

## 📊 Model Performance

### 🟢 DATA-DRIVEN Model (realized_pnl_pct)

**Target:** `realized_pnl_pct` (actual trading profit/loss)

**Validation Metrics:**
- **R²:** {metrics['avg_metrics']['avg_val_r2']:.3f}
- **RMSE:** {metrics['avg_metrics']['avg_val_rmse']:.4f}
- **MAE:** {metrics['avg_metrics']['avg_val_mae']:.4f}

**Cross-Validation Scores:**
"""
    
    for i, fold in enumerate(metrics['cv_scores']):
        report_content += f"\n**Fold {i+1}:**\n"
        report_content += f"- Train R²: {fold['train_r2']:.3f}, Val R²: {fold['val_r2']:.3f}\n"
        report_content += f"- Train RMSE: {fold['train_rmse']:.4f}, Val RMSE: {fold['val_rmse']:.4f}\n"
    
    report_content += f"""

---

## 🎯 What Changed

### Problem (Old Data)
- Old training data had all zeros for performance metrics
- `trade_profit = 0.0` for all samples
- Cannot train meaningful model on zeros!

### Solution (Fresh Data)
- Collected fresh data with MODIFIED data collector
- Now properly calculates `realized_pnl_pct` (actual trading P&L)
- Has real performance values (wins, losses, bounces, holds)

### Key Modification

**Modified:** `src/tactician/sr_levels/ml_quality/sr_quality_data_collector.py`

```python
def _simulate_trade(...) -> Dict:
    # Now returns ACTUAL P&L percentage
    return {{
        'realized_pnl_pct': pnl_pct,  # ✅ -0.01 to +0.02
        'trade_profit': normalized     # Backward compat
    }}
```

---

## 💾 Saved Artifacts

### Model
- `{model_path}` - **Data-driven model trained on actual profit**

### Report
- `{report_path}` - This report

### Training Data
- Fresh data with proper `realized_pnl_pct` calculation

---

## 🚀 Usage

```python
from src.tactician.sr_levels.ml_quality.sr_quality_model import load_sr_quality_model

# Load the data-driven model
model = load_sr_quality_model('{model_path}')

# Predict quality scores (optimized for actual profit!)
quality_scores = model.predict(sr_levels_features)
```

---

## ✅ Conclusion

Successfully trained data-driven SR quality model on fresh data with proper performance metrics.

The model is trained on `realized_pnl_pct` (actual trading profit) instead of heuristic `quality_score`.

---

*Report generated by collect_and_train_fresh.py*
"""
    
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    logger.info(f"   ✅ Report saved to {report_path}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ SUCCESS!")
    logger.info("="*80)
    logger.info(f"\n📁 Files generated:")
    logger.info(f"   Model:  {model_path}")
    logger.info(f"   Report: {report_path}")


if __name__ == '__main__':
    asyncio.run(main())

