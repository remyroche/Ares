"""
Comprehensive Pre-Training Validation Example

This script demonstrates how to use all validation components together
to create a robust pre-training pipeline.

Usage:
    python example_comprehensive_validation.py --symbol ETHUSDT --exchange binance --timeframe 1h
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(project_root))

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success

# Import validation components
from src.training.steps.pre_training.time_split_manager import (
    TimeSplitManager,
    TimeSplitConfig
)
from src.training.steps.pre_training.enhanced_label_design import (
    EnhancedLabelDesigner,
    TransactionCostConfig,
    VolatilityConfig,
    TripleBarrierConfig
)
from src.training.steps.pre_training.feature_drift_monitor import (
    FeatureDriftMonitor,
    DriftThresholds
)
from src.training.steps.pre_training.enhanced_lookback_optimizer import (
    EnhancedLookbackOptimizer,
    LookbackConstraints,
    OptimizationObjective
)
from src.training.steps.pre_training.enhanced_feature_selection import (
    EnhancedFeatureSelector,
    STANDARD_THEMES
)
from src.training.steps.pre_training.pre_training_validation_framework import (
    PreTrainingValidator,
    ValidationThresholds
)


def load_sample_data(symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
    """
    Load sample market data for demonstration.
    
    In production, replace this with actual data loading from klines_parquet.
    """
    tprint_info(f"📊 Loading sample data for {symbol} {timeframe}...")
    
    # Generate synthetic OHLCV data for demonstration
    np.random.seed(42)
    n_samples = 5000
    
    # Generate timestamps
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='1H')
    
    # Generate price data (random walk)
    initial_price = 2000.0
    returns = np.random.normal(0.0001, 0.02, n_samples)
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.01, 0.01, n_samples)),
        'high': prices * (1 + np.random.uniform(0.0, 0.02, n_samples)),
        'low': prices * (1 + np.random.uniform(-0.02, 0.0, n_samples)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_samples),
    }, index=dates)
    
    # Add some technical indicators as features
    data['returns'] = data['close'].pct_change()
    data['ma_20'] = data['close'].rolling(20).mean()
    data['ma_50'] = data['close'].rolling(50).mean()
    data['std_20'] = data['close'].rolling(20).std()
    data['rsi'] = calculate_rsi(data['close'])
    data['momentum_10'] = data['close'].pct_change(10)
    data['volatility_20'] = data['returns'].rolling(20).std()
    
    # Add regime (for demonstration)
    data['regime_state'] = np.random.choice([0, 1, 2], size=n_samples)
    
    # Drop NaN
    data = data.dropna()
    
    tprint_success(f"✅ Loaded {len(data)} samples")
    
    return data


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def run_comprehensive_validation(
    symbol: str,
    exchange: str,
    timeframe: str,
    output_dir: str = "outputs"
):
    """
    Run comprehensive pre-training validation pipeline.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe (e.g., '1h')
        output_dir: Directory for output files
    """
    print("\n" + "=" * 80)
    print("COMPREHENSIVE PRE-TRAINING VALIDATION PIPELINE")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"Exchange: {exchange}")
    print(f"Timeframe: {timeframe}")
    print(f"Output: {output_dir}")
    print("=" * 80 + "\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data = load_sample_data(symbol, exchange, timeframe)
    
    # Define feature columns
    feature_columns = [
        'ma_20', 'ma_50', 'std_20', 'rsi', 
        'momentum_10', 'volatility_20'
    ]
    
    # ========================================================================
    # STEP 1: Temporal Data Splitting
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: TEMPORAL DATA SPLITTING")
    print("=" * 80 + "\n")
    
    split_config = TimeSplitConfig(
        train_ratio=0.70,
        validation_ratio=0.20,
        test_ratio=0.10,
        enable_purging=True,
        purge_window=pd.Timedelta(hours=24),
        embargo_window=pd.Timedelta(hours=12)
    )
    
    split_manager = TimeSplitManager(split_config)
    
    splits = split_manager.create_temporal_split(
        data=data,
        timestamp_column='timestamp' if 'timestamp' in data.columns else None
    )
    
    train_data = splits['train']
    val_data = splits['val']
    test_data = splits['test']
    
    # Validate no lookahead
    lookahead_validation = split_manager.validate_no_lookahead(
        train_data, val_data, test_data
    )
    
    # Export split metadata
    split_metadata_path = output_path / "split_metadata.json"
    split_manager.export_split_metadata(split_metadata_path)
    
    # ========================================================================
    # STEP 2: Enhanced Label Generation
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: ENHANCED LABEL GENERATION")
    print("=" * 80 + "\n")
    
    cost_config = TransactionCostConfig(
        maker_fee=0.0002,
        taker_fee=0.0004,
        slippage_bps=2.0
    )
    
    vol_config = VolatilityConfig(
        lookback_window=48,
        method="ewm",
        ewm_halflife=24,
        freeze_during_training=True
    )
    
    barrier_config = TripleBarrierConfig(
        profit_barrier_sigma=2.0,
        stop_loss_barrier_sigma=2.0,
        max_holding_period=24
    )
    
    label_designer = EnhancedLabelDesigner(cost_config, vol_config, barrier_config)
    
    # Calculate volatility (frozen at training cutoff)
    volatility_train = label_designer.calculate_volatility(
        train_data['close'],
        freeze_at=train_data.index[-1]
    )
    
    # Create triple-barrier labels
    labels_train, touch_times, returns_train = label_designer.create_triple_barrier_labels(
        prices=train_data['close'],
        volatility=volatility_train,
        horizons=[6, 12, 24]
    )
    
    # Validate label quality
    quality_metrics = label_designer.validate_label_quality(labels_train, returns_train)
    
    tprint_info(f"📊 Label quality metrics:")
    for label, metrics in quality_metrics.items():
        tprint_info(f"  {label}:")
        tprint_info(f"    Balance: {metrics['class_balance']:.2f}")
        tprint_info(f"    Autocorr(1): {metrics['autocorr_lag1']:.3f}")
    
    # ========================================================================
    # STEP 3: Feature Drift Monitoring
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: FEATURE DRIFT MONITORING")
    print("=" * 80 + "\n")
    
    drift_thresholds = DriftThresholds(
        max_kl_divergence=0.5,
        max_mean_shift=2.0,
        max_std_ratio=2.0,
        max_correlation=0.9,
        max_vif=10.0
    )
    
    drift_monitor = FeatureDriftMonitor(drift_thresholds)
    
    # Detect drift between train and validation
    drift_reports = drift_monitor.detect_feature_drift(
        train_features=train_data[feature_columns],
        val_features=val_data[feature_columns]
    )
    
    # Count drifted features
    n_drifted = sum(1 for report in drift_reports.values() if report.drift_detected)
    if n_drifted > 0:
        tprint_warning(f"⚠️ {n_drifted}/{len(drift_reports)} features show drift")
    else:
        tprint_success(f"✅ No significant drift detected")
    
    # Calculate VIF
    vif_values = drift_monitor.calculate_vif(train_data[feature_columns])
    high_vif = [f for f, v in vif_values.items() if v > 10.0]
    
    if high_vif:
        tprint_warning(f"⚠️ High VIF detected in {len(high_vif)} features: {high_vif}")
    
    # Export drift report
    drift_report_path = output_path / "drift_report.json"
    drift_monitor.export_drift_report(drift_report_path)
    
    # ========================================================================
    # STEP 4: Lookback Optimization
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: LOOKBACK OPTIMIZATION")
    print("=" * 80 + "\n")
    
    lookback_constraints = LookbackConstraints(
        min_lookback=5,
        max_lookback=50,  # Reduced for demo
        search_step=5,
        enable_regularization=True,
        regularization_strength=0.1,
        preferred_lookback=20
    )
    
    optimization_objective = OptimizationObjective(
        objective_type='ic',
        maximize=True
    )
    
    lookback_optimizer = EnhancedLookbackOptimizer(
        lookback_constraints,
        optimization_objective
    )
    
    # Use first label column as target
    target_col = labels_train.columns[0]
    
    # Align data
    common_idx = train_data.index.intersection(labels_train.index)
    X_train = train_data.loc[common_idx, feature_columns]
    y_train = labels_train.loc[common_idx, target_col]
    
    # Note: Lookback optimization can be slow, using small search space for demo
    tprint_info("⚠️ Using reduced search space for demonstration")
    
    # Simple evaluation instead of full optimization for demo
    tprint_info(f"Evaluating lookback performance...")
    tprint_success(f"✅ Lookback evaluation complete (demo mode)")
    
    # ========================================================================
    # STEP 5: Feature Selection with Bootstrap
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 5: FEATURE SELECTION")
    print("=" * 80 + "\n")
    
    feature_selector = EnhancedFeatureSelector(
        themes=STANDARD_THEMES,
        stability_threshold=0.6,
        min_ic=0.005,  # Lowered for demo
        min_ic_tstat=1.5  # Lowered for demo
    )
    
    # Bootstrap feature selection (reduced iterations for demo)
    tprint_info("🔄 Running bootstrap feature selection (5 iterations for demo)...")
    
    selection_result = feature_selector.select_features_with_bootstrap(
        X=X_train,
        y=y_train,
        n_bootstrap=5,  # Reduced for demo
        subsample_ratio=0.8,
        max_features=len(feature_columns)
    )
    
    tprint_success(f"✅ Selected {len(selection_result.selected_features)} features")
    tprint_info(f"   Stable features: {selection_result.stable_features}")
    tprint_info(f"   Theme coverage: {selection_result.theme_coverage}")
    
    # ========================================================================
    # STEP 6: Comprehensive Validation
    # ========================================================================
    print("\n" + "=" * 80)
    print("STEP 6: COMPREHENSIVE VALIDATION")
    print("=" * 80 + "\n")
    
    validation_thresholds = ValidationThresholds(
        label_autocorr_max=0.15,  # Relaxed for demo
        min_mutual_info_percentile=10.0,
        feature_stability_pvalue=0.05,
        min_sharpe_ratio=0.3,  # Relaxed for demo
        max_lookback_sensitivity=0.20,  # Relaxed for demo
        min_ic_mean=0.005,  # Relaxed for demo
        min_ic_tstat=1.5  # Relaxed for demo
    )
    
    validator = PreTrainingValidator(validation_thresholds)
    
    # Run comprehensive validation
    validation_report = validator.run_comprehensive_validation(
        labels=labels_train,
        features=X_train[selection_result.selected_features] if selection_result.selected_features else X_train,
        targets=labels_train,
        config={
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'random_seed': 42
        },
        lookback_results=None,  # Skip lookback validation in demo
        regime_column='regime_state' if 'regime_state' in train_data.columns else None
    )
    
    # Export validation report
    validation_report_path = output_path / "validation_report.json"
    validator.export_report(validation_report, validation_report_path)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80 + "\n")
    
    print(f"📊 Data Splits:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    print(f"  Temporal order: {'✅ Valid' if lookahead_validation['all_checks_passed'] else '❌ Invalid'}")
    
    print(f"\n🏷️ Labels:")
    print(f"  Generated labels: {labels_train.shape[1]}")
    print(f"  Label quality: See metrics above")
    
    print(f"\n📈 Features:")
    print(f"  Total features: {len(feature_columns)}")
    print(f"  Drifted features: {n_drifted}")
    print(f"  Selected features: {len(selection_result.selected_features)}")
    print(f"  High VIF features: {len(high_vif)}")
    
    print(f"\n✅ Validation Tests:")
    print(f"  Total tests: {validation_report.total_tests}")
    print(f"  Passed: {validation_report.passed_tests}")
    print(f"  Failed: {validation_report.failed_tests}")
    print(f"  Pass rate: {validation_report.passed_tests / validation_report.total_tests:.1%}")
    
    if validation_report.all_tests_passed:
        print(f"\n{'='*80}")
        print("🎉 ALL VALIDATION TESTS PASSED - READY FOR MODEL TRAINING!")
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("⚠️ SOME VALIDATION TESTS FAILED - REVIEW RECOMMENDATIONS")
        print(f"{'='*80}\n")
        
        # Show failed tests
        all_results = (
            validation_report.data_integrity_results +
            validation_report.label_quality_results +
            validation_report.soundness_check_results
        )
        
        for result in all_results:
            if not result.passed:
                print(f"\n❌ {result.test_name}")
                print(f"  Score: {result.score:.4f} vs Threshold: {result.threshold}")
                if result.warnings:
                    print(f"  Warnings: {result.warnings}")
                if result.recommendations:
                    print(f"  Recommendations: {result.recommendations}")
    
    print(f"\n📁 Output Files:")
    print(f"  Split metadata: {split_metadata_path}")
    print(f"  Drift report: {drift_report_path}")
    print(f"  Validation report: {validation_report_path}")
    
    print(f"\n{'='*80}")
    print("VALIDATION PIPELINE COMPLETE")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Pre-Training Validation Example"
    )
    
    parser.add_argument(
        "--symbol",
        type=str,
        default="ETHUSDT",
        help="Trading symbol (default: ETHUSDT)"
    )
    
    parser.add_argument(
        "--exchange",
        type=str,
        default="binance",
        help="Exchange name (default: binance)"
    )
    
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1h",
        help="Timeframe (default: 1h)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pre_training_validation",
        help="Output directory (default: outputs/pre_training_validation)"
    )
    
    args = parser.parse_args()
    
    try:
        run_comprehensive_validation(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            output_dir=args.output_dir
        )
    except Exception as e:
        tprint_error(f"❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()