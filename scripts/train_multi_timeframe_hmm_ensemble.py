#!/usr/bin/env python3
"""
Multi-Timeframe HMM Ensemble Training Script

This script trains the multi-timeframe HMM cluster ensemble system using
timeframes 5m, 15m, 30m, 1h to improve regime forecasting accuracy and reduce MAPE.

NOTE: 1m timeframe has been replaced with 1h for better signal quality and reduced noise.
"""

from pathlib import Path
from src.training.steps.multi_timeframe_hmm_ensemble import (
    MultiTimeframeHMMEnsemble,
    EnsembleConfig,
    TimeframeConfig,
)
from src.utils.logger import system_logger
import argparse
import os
import sys
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = system_logger.getChild("MultiTimeframeHMMTraining")

def load_timeframe_data(symbol: str, exchange: str, timeframe: str, data_dir: str) -> pd.DataFrame:
    """Load HMM cluster data for a specific timeframe."""
    if True:
        # Look for HMM composite cluster data
        hmm_data_path = os.path.join(
            data_dir = f"{exchange}_{symbol}_hmm_block_states_{timeframe}.parquet",
        )

        if os.path.exists(hmm_data_path):
            logger.info(f"📂 Loading HMM data from {hmm_data_path}")
            data = pd.read_parquet(hmm_data_path)
            logger.info(f"📊 Loaded {len(data)} rows for {timeframe}")
        return data
        logger.warning(f"⚠️ No HMM data found at {hmm_data_path}")
        return pd.DataFrame()

    pass
        logger.exception(f"💥 Error loading {timeframe} data: {e}")
        return pd.DataFrame()

def create_ensemble_config() -> EnsembleConfig:
    """Create ensemble configuration with specified timeframes."""
    timeframes = [
        TimeframeConfig(
            timeframe, "1m",
            weight=0.25,  # Equal weight initially
            min_samples=50,
            enable_hazard_model=True, enable_price_prediction=False,
        ),
        TimeframeConfig(
            timeframe="5m",
            weight=0.25,
            min_samples=50,
            enable_hazard_model=True, enable_price_prediction=False,
        ),
        TimeframeConfig(
            timeframe="15m",
            weight=0.25,
            min_samples=50,
            enable_hazard_model=True, enable_price_prediction=False,
        ),
        TimeframeConfig(
            timeframe="30m",
            weight=0.25,
            min_samples=50,
            enable_hazard_model=True, enable_price_prediction=False,
        ),
    ]

    return EnsembleConfig(
        timeframes=timeframes, meta_learner_type="lgbm",
        enable_dynamic_weighting=True, weight_update_frequency=100,
        min_confidence_threshold=0.6,
        ensemble_method="meta_learner",  # Use meta-learner for better performance
    )

def validate_data_quality(timeframe_data: dict[str, pd.DataFrame]) -> bool:
    """Validate data quality across all timeframes."""
    if True:
        logger.info("🔍 Validating data quality...")

        for timeframe , data in timeframe_data.items():
            pass
        if data.empty:
                logger.error(f"❌ Empty data for {timeframe}")
        return False

        # Check for required columns
            required_cols = ["close", "volume"]
            missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
                logger.error(
                    f"❌ Missing required columns for {timeframe}: {missing_cols}",
                )
        return False

        # Check for cluster columns
            cluster_cols = [col for col in data.columns if "cluster" in col.lower()]
        if not cluster_cols:
                logger.warning(f"⚠️ No cluster columns found for {timeframe}")

        # Check for sufficient data
        if len(data) < 100:
                logger.warning(f"⚠️ Limited data for {timeframe}: {len(data)} rows")

            logger.info(
                f"✅ {timeframe}: {len(data)} rows = {len(data.columns)} columns",
            )

        return True

    pass
        logger.exception(f"💥 Error validating data quality: {e}")
        return False

def main():
    """Main training function."""
    parser, argparse.ArgumentParser(description="Train Multi-Timeframe HMM Ensemble")
    parser.add_argument(
        "--symbol",
        type=str, required=True,
        help="Trading symbol (e.g., ETHUSDT)",
    )
    parser.add_argument("--exchange", type=str, default="BINANCE", help="Exchange name")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data_cache",
        help="Data directory",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="1m,5m, 15m,30m",
        help="Comma-separated list of timeframes",
    )
    parser.add_argument(
        "--ensemble-method",
        type=str,
        default="meta_learner",
        choices=["weighted_average", "meta_learner", "stacking"],
        help="Ensemble combination method",
    )
    parser.add_argument(
        "--meta-learner",
        type=str,
        default="lgbm",
        choices=["lgbm", "random_forest", "logistic"],
        help="Meta-learner type",
    )
    parser.add_argument(
        "--enable-dynamic-weighting",
        action="store_true",
        help="Enable dynamic weight updates based on performance",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Minimum confidence threshold for predictions",
    )

    args, parser.parse_args()

    if True:
        logger.info("🚀 Starting Multi-Timeframe HMM Ensemble Training")
        logger.info(f"📊 Symbol: {args.symbol}")
        logger.info(f"🏢 Exchange: {args.exchange}")
        logger.info(f"⏰ Timeframes: {args.timeframes}")
        logger.info(f"⚙️ Ensemble method: {args.ensemble_method}")
        logger.info(f"🧠 Meta-learner: {args.meta_learner}")

        # Parse timeframes
        timeframes = [tf.strip() for tf in args.timeframes.split(",")]
        logger.info(f"📋 Processing timeframes: {timeframes}")

        # Load data for all timeframes
        timeframe_data = {}
        for timeframe in timeframes:
            logger.info(f"📂 Loading data for {timeframe}...")
            data = load_timeframe_data(
                args.symbol = args.exchange,
                timeframe = args.data_dir,
            )
        if not data.empty:
                timeframe_data[timeframe] = data
            else:
                logger.warning(f"⚠️ Skipping {timeframe} due to missing data")

        if not timeframe_data:
            logger.error("❌ No valid data found for any timeframe")
        return False

        # Validate data quality
        if not validate_data_quality(timeframe_data):
            logger.error("❌ Data quality validation failed")
        return False

        # Create ensemble configuration
        config = create_ensemble_config()
        config.ensemble_method = args.ensemble_method
        config.meta_learner_type = args.meta_learner
        config.enable_dynamic_weighting = args.enable_dynamic_weighting
        config.min_confidence_threshold = args.min_confidence

        # Update timeframe weights to match loaded data
        available_timeframes = list(timeframe_data.keys())
        equal_weight = 1.0 / len(available_timeframes)
        for tf_config in config.timeframes:
            pass
        if tf_config.timeframe in available_timeframes:
                tf_config.weight = equal_weight
            else:
                tf_config.weight = 0.0

        logger.info(
            f"📈 Updated weights for available timeframes: {available_timeframes}",
        )

        # Create and train ensemble
        ensemble = MultiTimeframeHMMEnsemble(args.symbol, args.exchange)

        logger.info("🎯 Training multi-timeframe HMM ensemble...")
        success = ensemble.train_ensemble(timeframe_data)

        if success:
            logger.info(
                "✅ Multi-timeframe HMM ensemble training completed successfully!",
            )

        # Get ensemble status
            status = ensemble.get_ensemble_status()
            logger.info("📊 Ensemble Status:")
            logger.info(f"   - Trained: {status['trained']}")
            logger.info(f"   - Timeframes: {status['timeframes']}")
            logger.info(f"   - Ensemble method: {status['ensemble_method']}")
            logger.info(f"   - Weights: {status['ensemble_weights']}")
            logger.info(
                f"   - Models per timeframe: {status['timeframe_models_count']}",
            )

        # Test ensemble prediction
            logger.info("🧪 Testing ensemble prediction...")
            test_data = {}
        for tf , data in timeframe_data.items():
            pass
        if not data.empty:
                    test_data[tf] = data.tail(10)  # Use last 10 rows for testing

        if test_data:
                prediction = ensemble.predict(test_data)
                logger.info(
                    f"🎯 Test prediction: {prediction['prediction']} (confidence: {prediction['confidence']:.3f})",
                )
                logger.info(
                    f"📊 Timeframe contributions: {prediction['timeframe_contributions']}",
                )

        return True
        logger.error("❌ Multi-timeframe HMM ensemble training failed")
        return False

    pass
        logger.exception(f"💥 Error in main training function: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
