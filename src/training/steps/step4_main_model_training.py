import asyncio
import os
import sys
import pickle
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, Any, Optional, Tuple

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.analyst.ml_target_generator import MLTargetGenerator
from src.analyst.feature_engineering import FeatureEngineeringEngine
from src.analyst.predictive_ensembles.ensemble_orchestrator import (
    RegimePredictiveEnsembles,
)
from src.utils.logger import system_logger, setup_logging
from src.config import CONFIG
from src.utils.state_manager import StateManager  # Import StateManager
from src.analyst.sr_analyzer import SRLevelAnalyzer  # Import SRLevelAnalyzer
from src.utils.error_handler import handle_errors


@handle_errors(
    exceptions=(Exception,), default_return=False, context="main_model_training_step"
)
async def run_step(
    symbol: str,
    timeframe: str,
    data_dir: str,
    data_file_path: str,
) -> bool:
    """
    Trains the main analyst models (including global meta-learner) using the optimized target parameters.
    Loads data from the specified pickle file.
    """
    setup_logging()
    logger = system_logger.getChild("Step4MainModelTraining")
    
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 STEP 4: MAIN MODEL TRAINING")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Symbol: {symbol}")
    logger.info(f"⏱️  Timeframe: {timeframe}")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"📦 Data file path: {data_file_path}")

    try:
        # Step 4.1: Load data from pickle file
        logger.info("📊 STEP 4.1: Loading data from pickle file...")
        data_load_start = time.time()
        
        if not os.path.exists(data_file_path):
            logger.error(f"❌ Data file not found: {data_file_path}")
            return False
            
        with open(data_file_path, "rb") as f:
            collected_data = pickle.load(f)
        klines_df = collected_data["klines"]
        agg_trades_df = collected_data["agg_trades"]
        futures_df = collected_data["futures"]
        
        data_load_duration = time.time() - data_load_start
        logger.info(f"⏱️  Data loading completed in {data_load_duration:.2f} seconds")
        logger.info(f"📊 Loaded data summary:")
        logger.info(f"   - Klines: {len(klines_df)} rows, {len(klines_df.columns)} columns")
        logger.info(f"   - Aggregated trades: {len(agg_trades_df)} rows, {len(agg_trades_df.columns)} columns")
        logger.info(f"   - Futures: {len(futures_df)} rows, {len(futures_df.columns)} columns")

        if klines_df.empty:
            logger.error("❌ Klines data is empty for main model training. Aborting.")
            return False

        # Step 4.2: Load configuration files
        logger.info("📋 STEP 4.2: Loading configuration files...")
        config_load_start = time.time()

        optimal_target_params_path = os.path.join(data_dir, f"{symbol}_optimal_target_params.json")
        pruned_features_path = os.path.join(data_dir, f"{symbol}_pruned_features.json")

        if not os.path.exists(optimal_target_params_path):
            logger.error(f"❌ Optimal target params file not found: {optimal_target_params_path}")
            return False

        if not os.path.exists(pruned_features_path):
            logger.error(f"❌ Pruned features file not found: {pruned_features_path}")
            return False

        with open(optimal_target_params_path, 'r') as f:
            optimal_target_params = json.load(f)
        with open(pruned_features_path, 'r') as f:
            pruned_features = json.load(f)

        config_load_duration = time.time() - config_load_start
        logger.info(f"⏱️  Configuration loading completed in {config_load_duration:.2f} seconds")
        logger.info(f"✅ Loaded configuration:")
        logger.info(f"   - Optimal target params count: {len(optimal_target_params)}")
        logger.info(f"   - Pruned features count: {len(pruned_features)}")

        # Step 4.3: Initialize components
        logger.info("🔧 STEP 4.3: Initializing components...")
        init_start = time.time()
        
        # Initialize StateManager and SRLevelAnalyzer for feature engineering
        state_manager = StateManager()  # For SR levels
        sr_analyzer = SRLevelAnalyzer(CONFIG["analyst"]["sr_analyzer"])
        
        init_duration = time.time() - init_start
        logger.info(f"⏱️  Component initialization completed in {init_duration:.2f} seconds")

        # Step 4.4: Calculate SR levels
        logger.info("📈 STEP 4.4: Calculating SR levels...")
        sr_start = time.time()
        
        # Calculate SR levels from daily klines (needed for feature engineering)
        daily_df_for_sr = (
            klines_df.resample("D")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna()
        )
        daily_df_for_sr.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )
        sr_levels = sr_analyzer.analyze(daily_df_for_sr)
        state_manager.set_state("sr_levels", sr_levels)  # Store in state for consistency
        
        sr_duration = time.time() - sr_start
        logger.info(f"⏱️  SR levels calculation completed in {sr_duration:.2f} seconds")
        logger.info(f"📊 SR levels summary: {len(sr_levels)} levels calculated")

        # Step 4.5: Feature engineering
        logger.info("🔧 STEP 4.5: Running feature engineering...")
        feature_start = time.time()
        
        # Initialize FeatureEngineeringEngine
        feature_engineering = FeatureEngineeringEngine(CONFIG)

        # Ensure ATR is calculated on klines before passing to feature_engineering
        klines_df_copy = klines_df.copy()
        klines_df_copy.ta.atr(
            length=CONFIG["best_params"]["atr_period"], append=True, col_names=("ATR")
        )

        data_with_features = feature_engineering.generate_all_features(
            klines_df_copy, agg_trades_df.copy(), futures_df.copy(), sr_levels
        )
        
        feature_duration = time.time() - feature_start
        logger.info(f"⏱️  Feature engineering completed in {feature_duration:.2f} seconds")
        logger.info(f"📊 Feature engineering results:")
        logger.info(f"   - Input shape: {klines_df.shape}")
        logger.info(f"   - Output shape: {data_with_features.shape}")
        logger.info(f"   - Features added: {len(data_with_features.columns) - len(klines_df.columns)}")

        if data_with_features.empty:
            logger.error("❌ Feature engineering resulted in empty DataFrame. Aborting training.")
            return False

        # Step 4.6: Target generation
        logger.info("🎯 STEP 4.6: Generating targets...")
        target_start = time.time()
        
        logger.info(f"🔢 Target generation parameters:")
        logger.info(f"   - Optimal params: {optimal_target_params}")
        logger.info(f"   - Leverage: {CONFIG.get('tactician', {}).get('initial_leverage', 50)}")
        
        target_generator = MLTargetGenerator(config=CONFIG)
        data_with_targets = target_generator.generate_targets(
            features_df=data_with_features,
            leverage=CONFIG.get("tactician", {}).get("initial_leverage", 50),
        )
        data_with_targets.dropna(inplace=True)
        
        target_duration = time.time() - target_start
        logger.info(f"⏱️  Target generation completed in {target_duration:.2f} seconds")
        logger.info(f"📊 Target generation results:")
        logger.info(f"   - Data with targets shape: {data_with_targets.shape}")
        logger.info(f"   - Target column present: {'target' in data_with_targets.columns}")

        # Step 4.7: Feature filtering
        logger.info("✂️  STEP 4.7: Filtering features to pruned set...")
        filter_start = time.time()
        
        # Filter features to only include pruned ones
        final_features = [f for f in pruned_features if f in data_with_targets.columns]
        if not final_features:
            logger.error("❌ No pruned features found in the data with targets. Aborting.")
            return False

        X_train_full = data_with_targets[final_features]
        y_train_full = data_with_targets["target"]
        
        filter_duration = time.time() - filter_start
        logger.info(f"⏱️  Feature filtering completed in {filter_duration:.2f} seconds")
        logger.info(f"📊 Feature filtering results:")
        logger.info(f"   - Original features: {len(pruned_features)}")
        logger.info(f"   - Available features: {len(final_features)}")
        logger.info(f"   - X shape: {X_train_full.shape}")
        logger.info(f"   - y shape: {y_train_full.shape}")

        if X_train_full.empty or y_train_full.empty:
            logger.error("❌ Feature or target set is empty after processing. Aborting.")
            return False

        # Step 4.8: Model training
        logger.info("🤖 STEP 4.8: Training ensemble models...")
        training_start = time.time()
        
        # Initialize RegimePredictiveEnsembles (EnsembleOrchestrator)
        ensemble_orchestrator = RegimePredictiveEnsembles(CONFIG)
        
        model_path_prefix = os.path.join(
            CONFIG["CHECKPOINT_DIR"], f"{CONFIG['ENSEMBLE_MODEL_PREFIX']}{symbol}_"
        )
        logger.info(f"📁 Model save path: {model_path_prefix}")

        logger.info("🎯 Training all ensemble models and the global meta-learner...")
        ensemble_orchestrator.train_all_models(
            asset=symbol,
            prepared_data=data_with_targets,  # Pass the data with targets and regime labels
            model_path_prefix=model_path_prefix,
        )
        
        training_duration = time.time() - training_start
        logger.info(f"⏱️  Model training completed in {training_duration:.2f} seconds")

        # Step 4.9: Save the fully prepared data for subsequent steps
        logger.info("💾 STEP 4.9: Saving prepared data for validation steps...")
        save_start = time.time()
        prepared_data_path = os.path.join(data_dir, f"{symbol}_prepared_data.pkl")
        try:
            with open(prepared_data_path, "wb") as f:
                pickle.dump(data_with_targets, f)
            logger.info(f"✅ Prepared data saved to: {prepared_data_path}")
        except Exception as e:
            logger.error(f"❌ Failed to save prepared data: {e}", exc_info=True)
            return False
        save_duration = time.time() - save_start
        logger.info(f"⏱️  Prepared data saved in {save_duration:.2f} seconds")

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 4: MAIN MODEL TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info(f"📊 Performance breakdown:")
        logger.info(f"   - Data loading: {data_load_duration:.2f}s ({(data_load_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Config loading: {config_load_duration:.2f}s ({(config_load_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Component init: {init_duration:.2f}s ({(init_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - SR calculation: {sr_duration:.2f}s ({(sr_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Feature engineering: {feature_duration:.2f}s ({(feature_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Target generation: {target_duration:.2f}s ({(target_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Feature filtering: {filter_duration:.2f}s ({(filter_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Model training: {training_duration:.2f}s ({(training_duration/total_duration)*100:.1f}%)")
        logger.info(f"📁 Model save location: {model_path_prefix}")
        logger.info(f"✅ Success: True")

        return True
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ STEP 4: MAIN MODEL TRAINING FAILED")
        logger.error("=" * 80)
        logger.error(f"⏱️  Duration before failure: {total_duration:.2f} seconds")
        logger.error(f"💥 Error: {e}")
        logger.error(f"📋 Full traceback:", exc_info=True)
        return False


if __name__ == "__main__":
    # Command-line arguments: symbol, timeframe, data_dir, data_file_path
    symbol = sys.argv[1]
    timeframe = sys.argv[2]
    data_dir = sys.argv[3]
    data_file_path = sys.argv[4]

    success = asyncio.run(
        run_step(
            symbol,
            timeframe,
            data_dir,
            data_file_path,
        )
    )

    if not success:
        sys.exit(1)  # Indicate failure
    sys.exit(0)  # Indicate success
