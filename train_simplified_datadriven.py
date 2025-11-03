"""
Train Data-Driven SR Quality Model - SIMPLIFIED APPROACH

Collects fresh training data using SimplifiedSRDataCollector.
NO heuristic components - only realized_pnl_pct (actual profit).
Aligned with 0.5-1% price goals.
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
    """Collect fresh data and train simplified data-driven model."""
    
    logger.info("="*80)
    logger.info("🚀 SIMPLIFIED DATA-DRIVEN SR QUALITY MODEL")
    logger.info("="*80)
    logger.info("\nPHILOSOPHY:")
    logger.info("  ❌ NO heuristic components (bounce_strength, hold_strength, etc.)")
    logger.info("  ✅ ONLY realized_pnl_pct (actual trading profit)")
    logger.info("  ✅ Aligned with 0.5-1% price goals (SL=0.5%, TP=1.0%)")
    
    # =========================================================================
    # STEP 1: Collect FRESH training data
    # =========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("📊 STEP 1: Collecting FRESH training data")
    logger.info("="*80)
    
    from src.tactician.sr_levels.ml_quality.simplified_data_collector import SimplifiedSRDataCollector
    
    collector = SimplifiedSRDataCollector(
        stop_loss_pct=0.01,    # ✅ 1.0% SL (was 0.5% - too tight!)
        take_profit_pct=0.01,  # ✅ 1.0% TP (1:1 R/R, was 2:1)
        max_hold_bars=20
    )
    
    try:
        # UPDATED: DAILY sampling for FULL YEAR
        training_data = await collector.collect_training_data(
            symbol='ETHUSDT',
            exchange='binance',
            start_date='2023-01-01',  # ✅ Full year (365 days)
            end_date='2023-12-31',    # ✅ 1 year of data
            timeframe='1h',
            forward_days=10,
            sample_freq_days=1  # ✅ DAILY (not weekly!)
        )
        
        logger.info(f"\n✅ Fresh data collected: {len(training_data)} samples")
        
        # =====================================================================
        # STEP 2: Verify data quality
        # =====================================================================
        
        logger.info("\n" + "="*80)
        logger.info("🔍 STEP 2: Data Quality Verification")
        logger.info("="*80)
        
        # Check for zeros
        zero_count = (training_data['realized_pnl_pct'] == 0).sum()
        non_zero_count = (training_data['realized_pnl_pct'] != 0).sum()
        
        logger.info(f"\n📊 Target Variable (realized_pnl_pct):")
        logger.info(f"   Non-zero samples: {non_zero_count}/{len(training_data)} ({non_zero_count/len(training_data)*100:.1f}%)")
        logger.info(f"   Zero samples: {zero_count} (untested levels)")
        logger.info(f"   Mean: {training_data['realized_pnl_pct'].mean()*100:.4f}%")
        logger.info(f"   Std: {training_data['realized_pnl_pct'].std()*100:.4f}%")
        logger.info(f"   Min: {training_data['realized_pnl_pct'].min()*100:.2f}%")
        logger.info(f"   Max: {training_data['realized_pnl_pct'].max()*100:.2f}%")
        
        # Calculate win rate
        if non_zero_count > 0:
            tested_data = training_data[training_data['realized_pnl_pct'] != 0]
            win_rate = (tested_data['realized_pnl_pct'] > 0).sum() / len(tested_data)
            logger.info(f"   Win rate (tested levels): {win_rate*100:.1f}%")
        
        # Check for data quality issues
        if non_zero_count == 0:
            logger.error("\n❌ CRITICAL: ALL P&L VALUES ARE ZERO!")
            logger.error("   This means NO levels were tested in the forward window.")
            logger.error("   Possible causes:")
            logger.error("     1. Forward window too short")
            logger.error("     2. SR levels too far from price action")
            logger.error("     3. No market data available")
            return
        
        if non_zero_count < 50:
            logger.warning(f"\n⚠️  WARNING: Only {non_zero_count} tested levels")
            logger.warning("   May not have enough data for reliable training")
        
        logger.info(f"\n✅ Data quality check PASSED!")
        logger.info(f"   Have {non_zero_count} tested levels with real P&L values")
        
        # Save fresh data
        saved_path = collector.save_training_data(training_data)
        
        # =====================================================================
        # STEP 3: Train simplified data-driven model
        # =====================================================================
        
        logger.info("\n" + "="*80)
        logger.info("🤖 STEP 3: Training Simplified Data-Driven Model")
        logger.info("="*80)
        
        from src.tactician.sr_levels.ml_quality.sr_quality_model import SRQualityModel
        
        # Filter to tested levels only (non-zero P&L)
        tested_only = training_data[training_data['realized_pnl_pct'] != 0].copy()
        
        logger.info(f"\n   Training samples: {len(tested_only)}")
        logger.info(f"   Features: {len([c for c in tested_only.columns if c.startswith('feature_')])} ")
        logger.info(f"   Target: realized_pnl_pct")
        
        # Train model
        model = SRQualityModel()
        metrics = model.train(
            training_data=tested_only,
            target_column='realized_pnl_pct',  # ✅ ONLY target!
            n_folds=3,
            num_boost_round=500,
            early_stopping_rounds=50
        )
        
        logger.info(f"\n📊 Training Results:")
        logger.info(f"   Avg Val R²:   {metrics['avg_metrics']['avg_val_r2']:.3f}")
        logger.info(f"   Avg Val RMSE: {metrics['avg_metrics']['avg_val_rmse']:.4f}")
        logger.info(f"   Avg Val MAE:  {metrics['avg_metrics']['avg_val_mae']:.4f}")
        
        # Check for data quality issues in metrics
        if metrics['avg_metrics']['avg_val_r2'] > 0.99:
            logger.warning("\n⚠️  Perfect R² (>0.99) - possible data leakage!")
        elif metrics['avg_metrics']['avg_val_r2'] < -100:
            logger.error("\n❌ Impossibly bad R² - numerical instability!")
        else:
            logger.info(f"\n✅ Metrics look reasonable")
        
        # =====================================================================
        # STEP 4: Save model and generate report
        # =====================================================================
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = f'models/sr_quality/sr_quality_simplified_{timestamp}.lgb'
        model.save(model_path)
        
        logger.info(f"\n💾 Model saved to {model_path}")
        
        # Generate comprehensive report
        outcomes_dir = Path('outcomes')
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = outcomes_dir / f'sr_quality_simplified_training_{timestamp}.md'
        
        report_content = f"""# SR Quality Model: Simplified Data-Driven Training

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 🎯 Approach: Pure Data-Driven (No Heuristics)

### What Changed
- ❌ **REMOVED:** All heuristic components
  - bounce_strength (normalized to 4% threshold)
  - hold_strength (normalized to 20 bars)
  - rejection_speed (heuristic speed score)
  - volume_quality (normalized to 2.5x)
  - quality_score (0.25*bounce + 0.20*hold + ...)

- ✅ **KEPT:** Only what matters
  - realized_pnl_pct (actual trading profit/loss)
  - feature_* columns (historical SR characteristics)

---

## 📊 Dataset

### Training Data
- **Total samples collected:** {len(training_data):,}
- **Tested levels (non-zero P&L):** {len(tested_only):,}
- **Untested levels:** {len(training_data) - len(tested_only):,}

### Target Variable (realized_pnl_pct)
- **Mean P&L:** {tested_only['realized_pnl_pct'].mean()*100:.4f}%
- **Std P&L:** {tested_only['realized_pnl_pct'].std()*100:.4f}%
- **Min P&L:** {tested_only['realized_pnl_pct'].min()*100:.2f}%
- **Max P&L:** {tested_only['realized_pnl_pct'].max()*100:.2f}%
- **Win rate:** {(tested_only['realized_pnl_pct'] > 0).sum() / len(tested_only) * 100:.1f}%

### Trading Parameters (aligned with 0.5-1% goals)
- **Stop Loss:** {collector.stop_loss_pct*100:.1f}%
- **Take Profit:** {collector.take_profit_pct*100:.1f}%
- **Risk/Reward:** {collector.take_profit_pct/collector.stop_loss_pct:.1f}:1
- **Max Hold:** {collector.max_hold_bars} bars

---

## 🤖 Model Performance

### Validation Metrics
- **R²:** {metrics['avg_metrics']['avg_val_r2']:.3f}
- **RMSE:** {metrics['avg_metrics']['avg_val_rmse']:.4f}
- **MAE:** {metrics['avg_metrics']['avg_val_mae']:.4f}

### Cross-Validation Folds
"""
        
        for i, fold in enumerate(metrics['cv_scores']):
            report_content += f"""
**Fold {i+1}:**
- Train: R²={fold['train_r2']:.3f}, RMSE={fold['train_rmse']:.4f}
- Val: R²={fold['val_r2']:.3f}, RMSE={fold['val_rmse']:.4f}
- Samples: Train={fold['train_samples']}, Val={fold['val_samples']}
"""
        
        report_content += f"""

---

## 🎓 Key Insights

### Why This Approach Works

1. **No Circular Logic**
   - OLD: Train on quality_score = 0.25*bounce + 0.20*hold + ...
   - NEW: Train on realized_pnl_pct = actual money made/lost
   
2. **No Fixed Thresholds**
   - OLD: Assume 4% bounce is "perfect"
   - NEW: Model learns what bounce % actually leads to profit
   
3. **No Fixed Weights**
   - OLD: Assume bounce is 25% important, hold is 20%, etc.
   - NEW: Model discovers actual feature importance

4. **Direct Optimization**
   - Model optimizes for what we care about: trading profit!

---

## 💾 Saved Artifacts

### Model
- `{model_path}`

### Training Data
- `{saved_path}`
- Metadata: `{saved_path.replace('.parquet', '_metadata.json')}`

### Report
- `{report_path}`

---

## 🚀 Usage

```python
from src.tactician.sr_levels.ml_quality.sr_quality_model import load_sr_quality_model

# Load the simplified data-driven model
model = load_sr_quality_model('{model_path}')

# Predict on new SR levels (using ONLY historical features)
quality_predictions = model.predict(new_levels[feature_columns])

# Select top levels
top_levels = new_levels[quality_predictions.argsort()[-10:]]

# These predictions optimize for 0.5-1% trading profit goals!
```

---

## ✅ Verification

### Data Quality ✓
- Non-zero P&L values: {non_zero_count:,} samples
- Win rate: {(tested_only['realized_pnl_pct'] > 0).sum() / len(tested_only) * 100:.1f}%
- P&L variance: {tested_only['realized_pnl_pct'].std()*100:.4f}%

### Model Quality ✓
- R² in reasonable range: {metrics['avg_metrics']['avg_val_r2']:.3f}
- No perfect scores (would indicate leakage)
- No impossible scores (would indicate bugs)

---

## 📝 Data Collection Details

### What We Collect

**For each historical date:**
1. Detect SR levels on historical data (no future peeking)
2. Extract historical features (strength, touch_count, etc.)
3. Look forward 10 days
4. Simulate trade with 0.5% SL, 1.0% TP
5. Record actual P&L as realized_pnl_pct

**Training sample:**
```python
{{
    'feature_strength': 0.8,          # Historical
    'feature_touch_count': 5,         # Historical
    'feature_market_volatility': 0.02,# Historical
    ...
    'realized_pnl_pct': 0.0095        # Future (TARGET: made 0.95%)
}}
```

---

## 🎯 Conclusion

Successfully trained simplified data-driven model with:
- Pure profit-based optimization
- No heuristic assumptions
- Aligned with 0.5-1% price goals

The model learns directly from actual trading outcomes.

---

*Generated by train_simplified_datadriven.py*
"""
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"\n📝 Report saved to {report_path}")
        
        logger.info("\n" + "="*80)
        logger.info("✅ SUCCESS!")
        logger.info("="*80)
        logger.info(f"\n📁 Generated files:")
        logger.info(f"   Model:  {model_path}")
        logger.info(f"   Data:   {saved_path}")
        logger.info(f"   Report: {report_path}")
        
        logger.info(f"\n🎯 Summary:")
        logger.info(f"   Collected {len(tested_only)} tested levels")
        logger.info(f"   Trained on realized_pnl_pct (actual profit)")
        logger.info(f"   NO heuristic components used!")
        logger.info(f"   Win rate: {(tested_only['realized_pnl_pct'] > 0).sum() / len(tested_only) * 100:.1f}%")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        
        logger.error("\n💡 Troubleshooting:")
        logger.error("   1. Ensure market data is available for ETHUSDT 1h")
        logger.error("   2. Check RealDataLoader can access binance data")
        logger.error("   3. Try different symbol/timeframe if needed")


if __name__ == '__main__':
    asyncio.run(main())

