"""
SR Prediction Runner

Standalone CLI for training SR performance prediction models.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(project_root))

from src.training.steps.market_analysis.sr_prediction.sr_performance_predictor import SRPerformancePredictor
from src.training.steps.market_analysis.sr_prediction.sr_training_data_builder import SRTrainingDataBuilder
from src.utils.logger import system_logger

logger = logging.getLogger(__name__)


class SRPredictionRunner:
    """Runner for SR performance prediction training."""
    
    def __init__(self, args):
        """Initialize runner with arguments."""
        self.args = args
        self.logger = system_logger.getChild('SRPredictionRunner')
        
        # Create output directory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_builder = SRTrainingDataBuilder()
        self.predictor = SRPerformancePredictor()
    
    async def run(self):
        """Run the complete training pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("SR PERFORMANCE PREDICTION TRAINING")
        self.logger.info("=" * 80)
        
        # Step 1: Collect or load data
        if self.args.load_data:
            self.logger.info(f"\n📂 Loading training data from {self.args.load_data}")
            training_data = self.data_builder.load_data(Path(self.args.load_data))
        else:
            self.logger.info("\n📊 Collecting training data...")
            training_data = await self._collect_data()
            
            # Save collected data
            if self.args.save_data:
                save_path = Path(self.args.save_data)
                self.data_builder.save_data(training_data, save_path)
        
        # Step 2: Check data quality
        self.logger.info("\n🔍 Checking data quality...")
        quality_stats = self.data_builder.check_data_quality(training_data)
        
        # Save quality stats
        stats_path = self.output_dir / 'data_quality_stats.txt'
        with open(stats_path, 'w') as f:
            for key, value in quality_stats.items():
                f.write(f"{key}: {value}\n")
        
        # Step 3: Prepare data
        self.logger.info("\n📊 Preparing training data...")
        
        # Filter untested levels if requested
        if self.args.filter_untested:
            training_data = self.data_builder.filter_untested_levels(training_data)
        
        # Apply confidence weighting if requested
        if self.args.use_weights:
            training_data = self.data_builder.apply_confidence_weighting(
                training_data, 
                method=self.args.weight_method
            )
        
        # Split train/val
        if not self.args.no_validation:
            train_data, val_data = self.data_builder.prepare_train_val_split(
                training_data,
                val_ratio=self.args.val_ratio,
                time_based=True
            )
        else:
            train_data = training_data
            val_data = None
        
        # Step 4: Train model
        self.logger.info("\n🤖 Training SR performance predictor...")
        
        if self.args.use_hpo:
            self.logger.info(f"   Using HPO with {self.args.hpo_trials} trials")
            metrics = self.predictor.train_with_hpo(
                training_data=train_data,
                n_trials=self.args.hpo_trials,
                hpo_method=self.args.hpo_method,
                n_folds=self.args.n_folds,
                num_boost_round=self.args.num_boost_round,
                early_stopping_rounds=self.args.early_stopping_rounds,
                filter_untested=self.args.filter_untested
            )
        else:
            metrics = self.predictor.train(
                training_data=train_data,
                n_folds=self.args.n_folds,
                num_boost_round=self.args.num_boost_round,
                early_stopping_rounds=self.args.early_stopping_rounds,
                filter_untested=self.args.filter_untested
            )
        
        # Step 5: Evaluate on validation set
        if val_data is not None:
            self.logger.info("\n📈 Evaluating on validation set...")
            self._evaluate_validation(val_data)
        
        # Step 6: Generate SHAP analysis
        if self.args.generate_shap:
            self.logger.info("\n🔍 Generating SHAP analysis...")
            self._generate_shap_analysis(train_data)
        
        # Step 7: Save model
        self.logger.info("\n💾 Saving model...")
        model_dir = self.output_dir / 'models'
        self.predictor.save(model_dir)
        
        # Step 8: Save metrics
        self._save_metrics(metrics)
        
        self.logger.info("\n✅ Training complete!")
        self.logger.info(f"   Models saved to: {model_dir}")
        self.logger.info(f"   Outputs saved to: {self.output_dir}")
    
    async def _collect_data(self) -> pd.DataFrame:
        """Collect training data based on arguments."""
        if self.args.multi_symbol:
            # Collect for multiple symbols
            symbols = self.args.symbol if isinstance(self.args.symbol, list) else [self.args.symbol]
            
            data = await self.data_builder.collect_multi_symbol(
                symbols=symbols,
                exchange=self.args.exchange,
                start_date=self.args.start_date,
                end_date=self.args.end_date,
                timeframe=self.args.timeframe,
                forward_days=self.args.forward_days,
                sample_freq_days=self.args.sample_freq_days
            )
        else:
            # Single symbol
            data = await self.data_builder.collect_data(
                symbol=self.args.symbol,
                exchange=self.args.exchange,
                start_date=self.args.start_date,
                end_date=self.args.end_date,
                timeframe=self.args.timeframe,
                forward_days=self.args.forward_days,
                sample_freq_days=self.args.sample_freq_days
            )
        
        return data
    
    def _evaluate_validation(self, val_data: pd.DataFrame):
        """Evaluate model on validation set."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np
        
        # Get predictions
        predictions = self.predictor.predict(val_data)
        
        # Compute metrics for each target
        results = {}
        
        for target in self.predictor.targets:
            if target not in val_data.columns:
                continue
            
            y_true = val_data[target].values
            y_pred = predictions[target]
            
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            results[target] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            self.logger.info(f"   {target}:")
            self.logger.info(f"      RMSE: {rmse:.4f}")
            self.logger.info(f"      MAE:  {mae:.4f}")
            self.logger.info(f"      R²:   {r2:.4f}")
        
        # Save validation results
        val_results_path = self.output_dir / 'validation_results.txt'
        with open(val_results_path, 'w') as f:
            for target, metrics in results.items():
                f.write(f"\n{target}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
    
    def _generate_shap_analysis(self, train_data: pd.DataFrame):
        """Generate SHAP analysis and plots."""
        shap_dir = self.output_dir / 'shap_analysis'
        shap_dir.mkdir(exist_ok=True)
        
        # Subsample for SHAP (can be slow on large datasets)
        if len(train_data) > 1000:
            self.logger.info(f"   Subsampling {len(train_data)} → 1000 for SHAP")
            shap_data = train_data.sample(n=1000, random_state=42)
        else:
            shap_data = train_data
        
        # Generate summary plot for each target
        for target in self.predictor.targets:
            if target in self.predictor.models:
                self.logger.info(f"   Generating SHAP plot for {target}...")
                
                plot_path = shap_dir / f'shap_summary_{target}.png'
                
                try:
                    self.predictor.plot_shap_summary(
                        training_data=shap_data,
                        target=target,
                        save_path=plot_path
                    )
                except Exception as e:
                    self.logger.error(f"Failed to generate SHAP plot for {target}: {e}")
        
        # Save feature importance
        for target in self.predictor.targets:
            if target in self.predictor.models:
                importance_df = self.predictor.get_feature_importance(target, method='gain', top_n=30)
                importance_path = shap_dir / f'feature_importance_{target}.csv'
                importance_df.to_csv(importance_path, index=False)
                
                self.logger.info(f"   Saved feature importance for {target}")
    
    def _save_metrics(self, metrics: dict):
        """Save training metrics to file."""
        metrics_path = self.output_dir / 'training_metrics.txt'
        
        with open(metrics_path, 'w') as f:
            f.write("SR PERFORMANCE PREDICTION - TRAINING METRICS\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Training completed: {datetime.now().isoformat()}\n\n")
            
            for target, target_metrics in metrics.items():
                f.write(f"\n{target.upper()}:\n")
                f.write("-" * 40 + "\n")
                for metric, value in target_metrics.items():
                    f.write(f"  {metric}: {value:.6f}\n")
        
        self.logger.info(f"   Saved metrics to {metrics_path}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SR Performance Prediction Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data collection arguments
    data_group = parser.add_argument_group('Data Collection')
    data_group.add_argument('--symbol', type=str, default='BTCUSDT',
                           help='Trading symbol (or comma-separated list for multi-symbol)')
    data_group.add_argument('--exchange', type=str, default='binance',
                           help='Exchange name')
    data_group.add_argument('--start-date', type=str, default='2023-01-01',
                           help='Start date (YYYY-MM-DD)')
    data_group.add_argument('--end-date', type=str, default='2024-01-01',
                           help='End date (YYYY-MM-DD)')
    data_group.add_argument('--timeframe', type=str, default='1h',
                           help='Timeframe (1h, 4h, 1d, etc.)')
    data_group.add_argument('--forward-days', type=int, default=10,
                           help='Days to look forward for labeling')
    data_group.add_argument('--sample-freq-days', type=int, default=7,
                           help='Sample frequency in days')
    data_group.add_argument('--multi-symbol', action='store_true',
                           help='Collect data for multiple symbols (comma-separated)')
    
    # Data loading/saving
    io_group = parser.add_argument_group('Data I/O')
    io_group.add_argument('--load-data', type=str,
                         help='Load pre-collected data from file instead of collecting')
    io_group.add_argument('--save-data', type=str,
                         help='Save collected data to file')
    
    # Training arguments
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--n-folds', type=int, default=5,
                            help='Number of CV folds')
    train_group.add_argument('--num-boost-round', type=int, default=1000,
                            help='Max boosting rounds')
    train_group.add_argument('--early-stopping-rounds', type=int, default=50,
                            help='Early stopping patience')
    train_group.add_argument('--filter-untested', action='store_true', default=True,
                            help='Filter out untested SR levels')
    train_group.add_argument('--no-filter-untested', dest='filter_untested', 
                            action='store_false',
                            help='Do not filter untested levels')
    
    # Validation
    val_group = parser.add_argument_group('Validation')
    val_group.add_argument('--no-validation', action='store_true',
                          help='Skip validation split (use all data for training)')
    val_group.add_argument('--val-ratio', type=float, default=0.2,
                          help='Validation set ratio')
    
    # Weighting
    weight_group = parser.add_argument_group('Sample Weighting')
    weight_group.add_argument('--use-weights', action='store_true',
                             help='Apply confidence-based sample weighting')
    weight_group.add_argument('--weight-method', type=str, 
                             choices=['quality_based', 'tiered', 'exponential'],
                             default='quality_based',
                             help='Sample weighting method')
    
    # HPO arguments
    hpo_group = parser.add_argument_group('Hyperparameter Optimization')
    hpo_group.add_argument('--use-hpo', action='store_true',
                          help='Use hyperparameter optimization')
    hpo_group.add_argument('--hpo-trials', type=int, default=50,
                          help='Number of HPO trials per target')
    hpo_group.add_argument('--hpo-method', type=str,
                          choices=['bayesian', 'staged', 'multi_objective'],
                          default='bayesian',
                          help='HPO method')
    
    # SHAP analysis
    shap_group = parser.add_argument_group('SHAP Analysis')
    shap_group.add_argument('--generate-shap', action='store_true', default=True,
                           help='Generate SHAP analysis and plots')
    shap_group.add_argument('--no-shap', dest='generate_shap', action='store_false',
                           help='Skip SHAP analysis')
    
    # Output
    output_group = parser.add_argument_group('Output')
    output_group.add_argument('--output-dir', type=str, 
                             default='outputs/sr_prediction',
                             help='Output directory for models and results')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse multi-symbol if needed
    if args.multi_symbol and ',' in args.symbol:
        args.symbol = [s.strip() for s in args.symbol.split(',')]
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run training
    runner = SRPredictionRunner(args)
    asyncio.run(runner.run())


if __name__ == '__main__':
    main()

