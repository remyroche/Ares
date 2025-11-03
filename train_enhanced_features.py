"""
Train SR Quality Model with ENHANCED FEATURES from FeatureBank

Integrates:
1. SR-specific features
2. Market regime features (volatility, trend states)
3. Price action features (momentum, candlestick patterns)
4. Multi-timeframe features (1D SR on 1h timeframe)
5. Recent SR performance features
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
    """Collect data with enhanced features and train."""
    
    logger.info("="*80)
    logger.info("🚀 SR QUALITY MODEL: ENHANCED FEATURES FROM FEATUREBANK")
    logger.info("="*80)
    logger.info("\nNew Features Added:")
    logger.info("  1. ✅ SR-specific features (from FeatureBank)")
    logger.info("  2. ✅ Market regime features (volatility, trend states)")
    logger.info("  3. ✅ Price action features (momentum, candlestick patterns)")
    logger.info("  4. ✅ Multi-timeframe features (1D SR tested on 1h)")
    logger.info("  5. ✅ Recent SR performance (bounced_last_test, etc.)")
    
    # =========================================================================
    # STEP 1: Collect data with enhanced features
    # =========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("📊 STEP 1: Collecting data with ENHANCED FEATURES")
    logger.info("="*80)
    
    from src.tactician.sr_levels.ml_quality.enhanced_feature_data_collector import EnhancedFeatureSRDataCollector
    
    collector = EnhancedFeatureSRDataCollector(
        stop_loss_pct=0.01,    # 1.0% SL
        take_profit_pct=0.01,  # 1.0% TP (1:1 R/R)
        max_hold_bars=20
    )
    
    try:
        training_data = await collector.collect_training_data(
            symbol='ETHUSDT',
            exchange='binance',
            start_date='2023-01-01',  # Full year
            end_date='2023-12-31',
            timeframe='1h',
            forward_days=10,
            sample_freq_days=1  # DAILY
        )
        
        logger.info(f"\n✅ Data collected: {len(training_data)} samples")
        
        # Check feature count
        feature_cols = [c for c in training_data.columns if c.startswith('feature_')]
        logger.info(f"   Total features: {len(feature_cols)} (was 19!)")
        
        # Categorize features
        basic_features = [c for c in feature_cols if not any(x in c for x in ['regime', 'price_action', 'sr_', 'mtf', 'recent'])]
        regime_features = [c for c in feature_cols if 'regime' in c]
        price_action_features = [c for c in feature_cols if 'price_action' in c]
        sr_specific_features = [c for c in feature_cols if 'sr_' in c]
        mtf_features = [c for c in feature_cols if 'mtf' in c]
        recent_features = [c for c in feature_cols if 'recent' in c or 'bounced' in c]
        
        logger.info(f"\n   Feature breakdown:")
        logger.info(f"     Basic SR: {len(basic_features)}")
        logger.info(f"     Regime: {len(regime_features)}")
        logger.info(f"     Price action: {len(price_action_features)}")
        logger.info(f"     SR-specific: {len(sr_specific_features)}")
        logger.info(f"     Multi-timeframe: {len(mtf_features)}")
        logger.info(f"     Recent performance: {len(recent_features)}")
        
        # Data quality check
        logger.info(f"\n🔍 Data Quality:")
        logger.info(f"   Mean P&L: {training_data['realized_pnl_pct'].mean()*100:.4f}%")
        logger.info(f"   Std P&L: {training_data['realized_pnl_pct'].std()*100:.4f}%")
        logger.info(f"   Win rate: {(training_data['realized_pnl_pct'] > 0).sum() / len(training_data) * 100:.1f}%")
        
        if (training_data['realized_pnl_pct'] == 0).all():
            logger.error("❌ All P&L values are zero!")
            return
        
        # Save enhanced data
        saved_path = collector.save_training_data(training_data)
        
        # =====================================================================
        # STEP 2: Train model
        # =====================================================================
        
        logger.info("\n" + "="*80)
        logger.info("🤖 STEP 2: Training with ENHANCED FEATURES")
        logger.info("="*80)
        
        from src.tactician.sr_levels.ml_quality.sr_quality_model import SRQualityModel
        
        # Filter to tested levels
        tested_data = training_data[training_data['realized_pnl_pct'] != 0].copy()
        
        logger.info(f"   Training samples: {len(tested_data)}")
        logger.info(f"   Features: {len(feature_cols)}")
        logger.info(f"   Samples/feature: {len(tested_data) / len(feature_cols):.1f}")
        
        if len(tested_data) < 100:
            logger.error(f"❌ Insufficient data: {len(tested_data)} samples")
            return
        
        # Train model
        model = SRQualityModel()
        metrics = model.train(
            training_data=tested_data,
            target_column='realized_pnl_pct',
            n_folds=3,
            num_boost_round=500,
            early_stopping_rounds=50
        )
        
        logger.info(f"\n📊 Training Results:")
        logger.info(f"   Avg Val R²:   {metrics['avg_metrics']['avg_val_r2']:.3f}")
        logger.info(f"   Avg Val RMSE: {metrics['avg_metrics']['avg_val_rmse']:.4f}")
        logger.info(f"   Avg Val MAE:  {metrics['avg_metrics']['avg_val_mae']:.4f}")
        
        # Check for improvement
        if metrics['avg_metrics']['avg_val_r2'] > 0.10:
            logger.info("\n✅ SUCCESS! R² > 0.10 - Model learned something useful!")
        elif metrics['avg_metrics']['avg_val_r2'] > 0.05:
            logger.info("\n🟡 MODERATE: R² > 0.05 - Some signal detected")
        else:
            logger.warning("\n⚠️  WEAK: R² < 0.05 - Enhanced features didn't help much")
            logger.warning("   May need even more features or different approach")
        
        # =====================================================================
        # STEP 3: Save and report
        # =====================================================================
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save model
        model_path = f'models/sr_quality/sr_quality_enhanced_{timestamp}.lgb'
        model.save(model_path)
        
        logger.info(f"\n💾 Model saved to {model_path}")
        
        # Generate report
        outcomes_dir = Path('outcomes')
        report_path = outcomes_dir / f'sr_quality_enhanced_training_{timestamp}.md'
        
        report_content = f"""# SR Quality Model: Enhanced Features Training Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 🎯 Enhanced Features Approach

### New Features Added

1. **SR-Specific Features** (from FeatureBank)
   - Advanced SR level detection
   - SR strength and quality metrics
   - Volume-weighted SR
   - Count: {len(sr_specific_features)}

2. **Market Regime Features** (from FeatureBank)
   - Volatility regime state
   - Trend regime state
   - Regime transitions and stability
   - Count: {len(regime_features)}

3. **Price Action Features** (from FeatureBank)
   - Momentum indicators (RSI, MACD, etc.)
   - Candlestick patterns
   - Price dynamics
   - Count: {len(price_action_features)}

4. **Multi-Timeframe Features**
   - 1D SR levels tested on 1h timeframe
   - Timeframe alignment
   - Cross-TF confirmation
   - Count: {len(mtf_features)}

5. **Recent Performance Features**
   - How many times tested recently
   - Did it bounce last time
   - Days since last test
   - Count: {len(recent_features)}

**Total features:** {len(feature_cols)} (was 19!)

---

## 📊 Dataset

### Training Data
- **Samples:** {len(tested_data):,}
- **Features:** {len(feature_cols)}
- **Samples/feature:** {len(tested_data) / len(feature_cols):.1f}

### Target (realized_pnl_pct)
- **Mean P&L:** {tested_data['realized_pnl_pct'].mean()*100:.4f}%
- **Std P&L:** {tested_data['realized_pnl_pct'].std()*100:.4f}%
- **Win rate:** {(tested_data['realized_pnl_pct'] > 0).sum() / len(tested_data) * 100:.1f}%

### Trading Parameters
- **SL:** {collector.stop_loss_pct*100:.1f}%
- **TP:** {collector.take_profit_pct*100:.1f}%
- **R/R:** {collector.take_profit_pct/collector.stop_loss_pct:.1f}:1

---

## 🤖 Model Performance

### Validation Metrics
- **R²:** {metrics['avg_metrics']['avg_val_r2']:.3f}
- **RMSE:** {metrics['avg_metrics']['avg_val_rmse']:.4f}
- **MAE:** {metrics['avg_metrics']['avg_val_mae']:.4f}

### Assessment

"""
        
        if metrics['avg_metrics']['avg_val_r2'] > 0.15:
            report_content += "✅ **EXCELLENT!** R² > 0.15 - Enhanced features made a BIG difference!\n\n"
        elif metrics['avg_metrics']['avg_val_r2'] > 0.10:
            report_content += "✅ **GOOD!** R² > 0.10 - Enhanced features helped significantly!\n\n"
        elif metrics['avg_metrics']['avg_val_r2'] > 0.05:
            report_content += "🟡 **MODERATE:** R² > 0.05 - Enhanced features helped somewhat\n\n"
        else:
            report_content += "⚠️  **WEAK:** R² < 0.05 - Enhanced features didn't help much\n\n"
        
        report_content += f"""
### Cross-Validation Folds
"""
        
        for i, fold in enumerate(metrics['cv_scores']):
            report_content += f"""
**Fold {i+1}:**
- Train: R²={fold['train_r2']:.3f}, RMSE={fold['train_rmse']:.4f}
- Val: R²={fold['val_r2']:.3f}, RMSE={fold['val_rmse']:.4f}
"""
        
        report_content += f"""

---

## 🎓 Comparison to Baseline

### Baseline (19 features, simple)
- R²: -0.002
- Win rate: 36.6%
- Features: Basic SR characteristics only

### Enhanced ({len(feature_cols)} features, FeatureBank)
- R²: {metrics['avg_metrics']['avg_val_r2']:.3f}
- Win rate: {(tested_data['realized_pnl_pct'] > 0).sum() / len(tested_data) * 100:.1f}%
- Features: SR + Regime + Price Action + Multi-TF + Recent Performance

### Improvement
- ΔR²: {metrics['avg_metrics']['avg_val_r2'] - (-0.002):+.3f}
- ΔWin rate: {((tested_data['realized_pnl_pct'] > 0).sum() / len(tested_data) - 0.366)*100:+.1f} percentage points

---

## 💾 Saved Artifacts

- Model: `{model_path}`
- Data: `{saved_path}`
- Report: `{report_path}`

---

## ✅ Conclusion

{'Enhanced features from FeatureBank significantly improved model performance!' if metrics['avg_metrics']['avg_val_r2'] > 0.10 else 'Enhanced features added but more work needed for strong predictive power.'}

---

*Generated by train_enhanced_features.py*
"""
        
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"\n📝 Report saved to {report_path}")
        
        logger.info("\n" + "="*80)
        logger.info("✅ ENHANCED TRAINING COMPLETE!")
        logger.info("="*80)
        
        logger.info(f"\n📁 Generated files:")
        logger.info(f"   Model:  {model_path}")
        logger.info(f"   Data:   {saved_path}")
        logger.info(f"   Report: {report_path}")
        
        logger.info(f"\n🎯 Result:")
        logger.info(f"   Features: {len(feature_cols)} (was 19)")
        logger.info(f"   R²: {metrics['avg_metrics']['avg_val_r2']:.3f} (was -0.002)")
        
        if metrics['avg_metrics']['avg_val_r2'] > 0.10:
            improvement = metrics['avg_metrics']['avg_val_r2'] - (-0.002)
            logger.info(f"   ✅ MAJOR IMPROVEMENT: +{improvement:.3f} R² gain!")
        
    except Exception as e:
        logger.error(f"\n❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == '__main__':
    asyncio.run(main())

