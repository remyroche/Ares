"""
Comprehensive SR Quality Model Training

Trains model and generates complete assessment reports with all metrics.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.tactician.sr_levels.ml_quality.sr_quality_model import SRQualityModel


def train_and_evaluate_comprehensive(data_path: str, 
                                     target_column: str = 'quality_score',
                                     save_model: bool = True):
    """Train model with comprehensive evaluation and reporting.
    
    Args:
        data_path: Path to training data parquet
        target_column: Target column to predict
        save_model: Whether to save the trained model
        
    Returns:
        Training metrics with all assessments
    """
    logger = logging.getLogger(__name__)
    
    print("\n" + "="*80)
    print("🚀 SR QUALITY MODEL - COMPREHENSIVE TRAINING & EVALUATION")
    print("="*80)
    
    # Load training data
    logger.info(f"\n📂 Loading training data from: {data_path}")
    
    if not Path(data_path).exists():
        logger.error(f"❌ Data file not found: {data_path}")
        sys.exit(1)
    
    training_data = pd.read_parquet(data_path)
    
    logger.info(f"✅ Loaded {len(training_data):,} samples")
    logger.info(f"   Columns: {len(training_data.columns)}")
    
    # Check for target column
    if target_column not in training_data.columns:
        logger.error(f"❌ Target column '{target_column}' not found in data")
        logger.error(f"   Available columns: {training_data.columns.tolist()}")
        sys.exit(1)
    
    # Check for new enhanced columns
    enhanced_cols = ['rejection_speed', 'volume_quality', 'bounce_quality', 
                     'hold_quality', 'trade_quality']
    missing_enhanced = [c for c in enhanced_cols if c not in training_data.columns]
    
    if missing_enhanced:
        logger.warning(f"\n⚠️  Missing enhanced columns: {missing_enhanced}")
        logger.warning(f"   Data might be from old collection run")
        logger.warning(f"   Consider recollecting data with latest improvements")
    else:
        logger.info(f"\n✅ All enhanced columns present")
    
    # Display data summary
    logger.info(f"\n📊 Data Summary:")
    logger.info(f"   Target ({target_column}):")
    logger.info(f"      Mean: {training_data[target_column].mean():.4f}")
    logger.info(f"      Std: {training_data[target_column].std():.4f}")
    logger.info(f"      Range: [{training_data[target_column].min():.4f}, {training_data[target_column].max():.4f}]")
    
    # Count features
    feature_cols = [c for c in training_data.columns if c.startswith('feature_')]
    logger.info(f"   Features: {len(feature_cols)}")
    
    # Create model
    logger.info(f"\n🤖 Creating SR Quality Model...")
    model = SRQualityModel()
    
    # Train with comprehensive assessment
    logger.info(f"\n🎓 Training model with {len(training_data)} samples...")
    logger.info(f"   This will include:")
    logger.info(f"      • 5-fold time series cross-validation")
    logger.info(f"      • Overfitting detection")
    logger.info(f"      • Calibration analysis")
    logger.info(f"      • Prediction distribution analysis")
    logger.info(f"      • Feature importance (LGBM + Permutation + SHAP)")
    logger.info(f"      • Comprehensive report generation")
    
    print("\n" + "="*80)
    print("⏳ TRAINING IN PROGRESS...")
    print("="*80)
    
    training_metrics = model.train(
        training_data=training_data,
        target_column=target_column,
        n_folds=5,
        num_boost_round=1000,
        early_stopping_rounds=50
    )
    
    # Save model if requested
    if save_model:
        model_dir = Path('models/sr_quality')
        model_dir.mkdir(parents=True, exist_ok=True)
        
        symbol_val = training_data['symbol'].iloc[0] if 'symbol' in training_data.columns else 'UNKNOWN'
        timeframe_val = training_data['timeframe'].iloc[0] if 'timeframe' in training_data.columns else 'unknown'
        
        model_path = model_dir / f"sr_quality_model_{symbol_val}_{timeframe_val}_{target_column}.lgb"
        model.save(str(model_path))
        
        logger.info(f"\n💾 Model saved to: {model_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    
    # Display key results
    avg_metrics = training_metrics.get('avg_metrics', {})
    logger.info(f"\n📊 Final Results:")
    logger.info(f"   Validation RMSE: {avg_metrics.get('avg_val_rmse', 0):.4f}")
    logger.info(f"   Validation R²: {avg_metrics.get('avg_val_r2', 0):.3f}")
    logger.info(f"   Validation MAE: {avg_metrics.get('avg_val_mae', 0):.4f}")
    
    # Quality assessment summary
    if 'quality_assessment' in training_metrics:
        qa = training_metrics['quality_assessment']
        logger.info(f"\n🔬 Model Quality:")
        logger.info(f"   Health Score: {qa.get('health_score', 0):.2f}/1.00")
        logger.info(f"   Production Ready: {qa.get('production_ready', False)}")
        logger.info(f"   Overfitting: {qa.get('overfitting', {}).get('severity', 'unknown')}")
        logger.info(f"   Calibration ECE: {qa.get('calibration', {}).get('expected_calibration_error', 0):.4f}")
    
    # Report paths
    if 'report_paths' in training_metrics:
        paths = training_metrics['report_paths']
        logger.info(f"\n📁 Reports Generated:")
        logger.info(f"   {paths.get('markdown', '')}")
        logger.info(f"   {paths.get('csv', '')}")
        logger.info(f"   {paths.get('json', '')}")
    
    print("\n" + "="*80 + "\n")
    
    return training_metrics


def main():
    """Main execution."""
    
    # Check for available training data
    possible_paths = [
        'data_cache/sr_ml_training/multi_timeframe/sr_quality_1h_ETHUSDT.parquet',
        'data_cache/sr_ml_training/sr_quality_training_data.parquet',
    ]
    
    data_path = None
    for path in possible_paths:
        if Path(path).exists():
            data_path = path
            break
    
    if data_path is None:
        print("❌ No training data found!")
        print("   Expected locations:")
        for path in possible_paths:
            print(f"      • {path}")
        print("\n   Run data collection first:")
        print("      python3 validate_multi_timeframe_quality.py")
        sys.exit(1)
    
    # Train composite model
    print(f"\n{'='*80}")
    print(f"TRAINING COMPOSITE MODEL (quality_score)")
    print(f"{'='*80}")
    
    train_and_evaluate_comprehensive(
        data_path=data_path,
        target_column='quality_score',
        save_model=True
    )
    
    # Train specialized models if enhanced columns exist
    test_data = pd.read_parquet(data_path)
    
    specialized_targets = ['bounce_quality', 'hold_quality', 'trade_quality']
    
    for target in specialized_targets:
        if target in test_data.columns:
            print(f"\n{'='*80}")
            print(f"TRAINING SPECIALIZED MODEL ({target})")
            print(f"{'='*80}")
            
            train_and_evaluate_comprehensive(
                data_path=data_path,
                target_column=target,
                save_model=True
            )
    
    print("\n" + "="*80)
    print("✅ ALL MODELS TRAINED SUCCESSFULLY!")
    print("="*80)
    print("\n📁 Check outcomes/ directory for comprehensive reports")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()

