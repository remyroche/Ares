#!/usr/bin/env python3
"""
SR Quality Model Training Script

Collects historical SR data and trains LightGBM quality model.
Uses artifact_manager to load existing downloaded data.

Usage:
    python train_sr_quality_model.py --symbol BTCUSDT --exchange binance --timeframe 1h
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import SR ML components
from src.tactician.sr_levels.ml_quality import (
    SRQualityDataCollector,
    SRQualityModel,
    train_sr_quality_model
)


def main():
    parser = argparse.ArgumentParser(description='Train SR Quality Model')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe')
    parser.add_argument('--start-date', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default='2024-01-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', type=str, default='data_cache/sr_ml_training', help='Output directory')
    parser.add_argument('--model-output', type=str, default='models/sr_quality_model.lgb', help='Model output path')
    parser.add_argument('--sample-freq-days', type=int, default=7, help='Sampling frequency in days')
    parser.add_argument('--forward-days', type=int, default=10, help='Forward window for performance measurement')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("SR QUALITY MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Symbol: {args.symbol}")
    logger.info(f"Exchange: {args.exchange}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info("="*80)
    
    # Step 1: Collect training data
    logger.info("\n" + "="*80)
    logger.info("STEP 1: COLLECTING TRAINING DATA")
    logger.info("="*80)
    
    collector = SRQualityDataCollector()
    
    try:
        training_df = collector.collect_training_data(
            symbol=args.symbol,
            exchange=args.exchange,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
            forward_days=args.forward_days,
            sample_freq_days=args.sample_freq_days
        )
        
        # Save training data
        output_path = Path(args.output_dir) / f"sr_training_{args.symbol}_{args.timeframe}.parquet"
        saved_path = collector.save_training_data(training_df, str(output_path))
        
        logger.info(f"\n✅ Training data collected and saved!")
        logger.info(f"   Samples: {len(training_df)}")
        logger.info(f"   Features: {len([c for c in training_df.columns if c.startswith('feature_')])}")
        logger.info(f"   Path: {saved_path}")
        
    except Exception as e:
        logger.error(f"❌ Data collection failed: {e}", exc_info=True)
        return 1
    
    # Step 2: Train ML model
    logger.info("\n" + "="*80)
    logger.info("STEP 2: TRAINING LIGHTGBM MODEL")
    logger.info("="*80)
    logger.info("🔬 Training with confidence weighting (label smoothing) + top 30% filtering")
    
    try:
        model = SRQualityModel()
        
        # Add GENTLE confidence weights (NO HARD FILTERING)
        from src.tactician.sr_levels.ml_quality.sr_quality_data_collector import SRQualityDataCollector
        collector = SRQualityDataCollector()
        logger.info(f"   🔬 Using gentle confidence weighting (preserve variance)")
        logger.info(f"      Noise: 0.3x, Weak: 0.5x, Medium: 0.8x, Strong: 1.2x, Critical: 2.0x")
        training_df_weighted = collector.add_confidence_weights(
            training_df,
            method='tiered'  # Gentle: 0.3x to 2.0x (was 0.1x to 3.0x)
        )
        
        # NO HARD FILTERING - use confidence weights only to preserve variance
        logger.info(f"   📊 Using ALL data with confidence weighting: {len(training_df_weighted):,} samples")
        logger.info(f"      Reason: Hard filtering caused model collapse (predicted ~0.81 for everything)")
        
        metrics = model.train(training_df_weighted, target_column='quality_score', n_folds=5)
        
        # Save trained model
        model_path = Path(args.model_output)
        model.save(str(model_path))
        
        logger.info(f"\n✅ Model trained and saved!")
        logger.info(f"   Model path: {model_path}")
        logger.info(f"   Val R²: {metrics['avg_metrics']['avg_val_r2']:.4f}")
        logger.info(f"   Val RMSE: {metrics['avg_metrics']['avg_val_rmse']:.4f}")
        
    except Exception as e:
        logger.error(f"❌ Model training failed: {e}", exc_info=True)
        return 1
    
    # Step 3: Validate model with RANKING metrics
    logger.info("\n" + "="*80)
    logger.info("STEP 3: MODEL VALIDATION (Ranking-Focused)")
    logger.info("="*80)
    
    try:
        # Test predictions on sample data
        sample_features = training_df_weighted.filter(like='feature_').iloc[:10]
        predictions = model.predict(sample_features)
        
        logger.info(f"   Sample predictions: {predictions[:5]}")
        logger.info(f"   Prediction range: [{predictions.min():.3f}, {predictions.max():.3f}]")
        logger.info(f"   Prediction mean: {predictions.mean():.3f}")
        
        # Evaluate RANKING metrics (what matters!)
        X_test = training_df_weighted.filter(like='feature_')
        y_test = training_df_weighted['quality_score']
        
        ranking_results = model.evaluate_ranking(X_test, y_test, k=10)
        
        logger.info(f"\n📊 RANKING METRICS:")
        logger.info(f"   Precision@10:  {ranking_results['precision_at_k']*100:.1f}%")
        logger.info(f"   Spearman ρ:    {ranking_results['spearman_rho']:.3f}")
        logger.info(f"   NDCG@10:       {ranking_results['ndcg_at_k']:.3f}")
        
        logger.info("\n✅ Model validation complete!")
        
    except Exception as e:
        logger.error(f"⚠️ Model validation failed: {e}")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"✅ Training data: {saved_path}")
    logger.info(f"✅ Model file: {args.model_output}")
    logger.info(f"✅ Ready to use for SR level quality prediction")
    logger.info("\nTo use this model, enable it in config:")
    logger.info("  sr_detection:")
    logger.info("    enable_ml_quality: true")
    logger.info(f"    ml_quality_model_path: '{args.model_output}'")
    logger.info("="*80)
    
    return 0


if __name__ == '__main__':
    exit(main())

