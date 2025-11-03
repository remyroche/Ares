"""
Demo script for 100% data-driven SR ML system.

Example usage:
    python demo_train.py --symbol ETHUSDT --exchange binance --timeframe 1h
"""

import sys
import logging
import argparse
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from src.training.steps.sr_detection_ml.fully_data_driven_trainer import FullyDataDrivenSRSystem
from src.training.steps.sr_detection_ml.utils.shap_visualization import ShapVisualizer
from src.training.steps.sr_detection_ml.utils.performance_metrics import PerformanceAnalyzer


def main():
    """Run end-to-end training demo."""
    parser = argparse.ArgumentParser(description='Train 100% data-driven SR ML model')
    parser.add_argument('--symbol', type=str, default='ETHUSDT', help='Trading symbol')
    parser.add_argument('--exchange', type=str, default='binance', help='Exchange name')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe')
    parser.add_argument('--start-date', type=str, default='2023-06-01', help='Start date (default: 6 months)')
    parser.add_argument('--end-date', type=str, default='2023-12-01', help='End date')
    parser.add_argument('--n-features', type=int, default=40, help='Number of features to select')
    parser.add_argument('--sample-every', type=int, default=20, help='Sample every N bars (default: 20 for 1000+ samples)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🚀 100% DATA-DRIVEN SR LEVEL ML SYSTEM - DEMO")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Symbol: {args.symbol}")
    print(f"  Exchange: {args.exchange}")
    print(f"  Timeframe: {args.timeframe}")
    print(f"  Period: {args.start_date} to {args.end_date}")
    print(f"  Features to select: {args.n_features}")
    print(f"  Sample frequency: every {args.sample_every} bars")
    print("=" * 80 + "\n")
    
    # Initialize system
    system = FullyDataDrivenSRSystem()
    
    # Train from scratch
    try:
        results = system.train_from_scratch(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            n_features=args.n_features,
            sample_every_n_bars=args.sample_every
        )
        
        print("\n" + "=" * 80)
        print("📊 GENERATING VISUALIZATIONS")
        print("=" * 80)
        
        # Generate SHAP visualizations
        visualizer = ShapVisualizer()
        
        # Get validation data for visualizations
        from src.training.steps.sr_detection_ml.sr_data_collector import SRDataCollector
        collector = SRDataCollector()
        
        # Load data again to get validation set
        raw_data = collector.collect_training_data(
            symbol=args.symbol,
            exchange=args.exchange,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date,
            sample_every_n_bars=args.sample_every
        )
        
        # Extract features
        feature_cols = [c for c in raw_data.columns if any(c.startswith(p) for p in [
            'dist_', 'crosses_', 'vol_', 'ret_', 'range_', 'atr_', 'time_at_', 'close_'
        ])]
        
        X_all = raw_data[feature_cols].fillna(0)
        X_selected = X_all[results['selected_features']]
        
        # Get validation split
        split_idx = int(len(X_selected) * 0.8)
        X_val = X_selected.iloc[split_idx:]
        
        # Generate SHAP plots
        prefix = f"sr_ml_{args.symbol}_{args.exchange}_{args.timeframe}"
        visualizer.generate_all_plots(
            results['explainer'],
            X_val,
            results['shap_values'],
            results['selected_features'],
            prefix=prefix
        )
        
        # Performance analysis
        analyzer = PerformanceAnalyzer()
        
        # Get targets
        target_cols = [c for c in raw_data.columns if any(c.startswith(p) for p in [
            'max_', 'touch_', 'break_', 'reversal_', 'vol_change', 'volume_surge'
        ])]
        y_all = raw_data[target_cols][results['best_target']].fillna(0)
        y_val = y_all.iloc[split_idx:]
        
        # Get predictions
        y_pred = results['model'].predict(X_val)
        
        # Full evaluation
        metrics = analyzer.full_evaluation(
            y_val.values,
            y_pred,
            prefix=prefix
        )
        
        print("\n" + "=" * 80)
        print("✅ TRAINING COMPLETE!")
        print("=" * 80)
        print(f"\nModel Details:")
        print(f"  Best Target: {results['best_target']}")
        print(f"  Val R²: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  Features Selected: {len(results['selected_features'])}")
        print(f"\nOutputs saved to:")
        print(f"  📊 Comprehensive Report: outcomes/SR_ML_TRAINING_REPORT_*_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
        print(f"  🤖 Models: models/sr_ml/")
        print(f"  📈 SHAP plots: outputs/sr_ml/shap/")
        print(f"  📉 Performance: outputs/sr_ml/performance/")
        print(f"  💾 Training data: artifacts/pre_training/artifact_store/")
        print("=" * 80 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())

