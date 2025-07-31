import asyncio
import os
import json
import pickle
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.enhanced_coarse_optimizer import EnhancedCoarseOptimizer
from src.utils.logger import system_logger, setup_logging
from src.database.sqlite_manager import SQLiteManager  # Import SQLiteManager
from src.utils.error_handler import handle_errors
from src.config import CONFIG


# Check if this is a blank training run and override configuration accordingly
if os.environ.get('BLANK_TRAINING_MODE') == '1':
    print("🔧 BLANK TRAINING MODE DETECTED - Overriding configuration for faster execution")
    CONFIG["MODEL_TRAINING"]["hyperparameter_tuning"]["max_trials"] = 3
    CONFIG["MODEL_TRAINING"]["coarse_hpo"] = {
        "n_trials": 5
    }
    print(f"✅ Configuration overridden: coarse_hpo.n_trials = {CONFIG['MODEL_TRAINING']['coarse_hpo']['n_trials']}")


@handle_errors(
    exceptions=(Exception,), default_return=False, context="coarse_optimization_step"
)
async def run_step(
    symbol: str,
    timeframe: str,
    data_dir: str,
    data_file_path: str,
    optimal_target_params_json: str,
) -> bool:
    """
    Runs the coarse optimization for feature pruning and hyperparameter range narrowing.
    Loads data from the specified pickle file and optimal target parameters from JSON.
    """
    setup_logging()
    logger = system_logger.getChild("Step3CoarseOpt")
    
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 STEP 3: COARSE OPTIMIZATION AND FEATURE PRUNING")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Symbol: {symbol}")
    logger.info(f"⏱️  Timeframe: {timeframe}")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"📦 Data file path: {data_file_path}")
    logger.info(f"🎯 Optimal target params JSON: {optimal_target_params_json}")

    try:
        # Step 3.1: Load data from pickle file
        logger.info("📊 STEP 3.1: Loading data from pickle file...")
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
            logger.error("❌ Klines data is empty for coarse optimization. Aborting.")
            return False

        # Step 3.2: Load optimal target parameters
        logger.info("🎯 STEP 3.2: Loading optimal target parameters...")
        params_load_start = time.time()
        
        # Construct the file path based on the symbol
        optimal_target_params_file = os.path.join(data_dir, f"{symbol}_optimal_target_params.json")
        
        if not os.path.exists(optimal_target_params_file):
            logger.error(f"❌ Optimal target params file not found: {optimal_target_params_file}")
            return False
            
        with open(optimal_target_params_file, "r") as f:
            optimal_target_params = json.load(f)
        
        params_load_duration = time.time() - params_load_start
        logger.info(f"⏱️  Parameters loading completed in {params_load_duration:.2f} seconds")
        logger.info(f"✅ Loaded optimal target parameters:")
        for param, value in optimal_target_params.items():
            logger.info(f"   - {param}: {value}")

        # Step 3.3: Initialize database manager
        logger.info("🗄️  STEP 3.3: Initializing database manager...")
        db_init_start = time.time()
        
        db_manager = SQLiteManager()
        await db_manager.initialize()
        
        db_init_duration = time.time() - db_init_start
        logger.info(f"⏱️  Database initialization completed in {db_init_duration:.2f} seconds")

        # Step 3.4: Initialize enhanced coarse optimizer
        logger.info("🔧 STEP 3.4: Initializing enhanced coarse optimizer...")
        optimizer_init_start = time.time()
        
        coarse_optimizer = EnhancedCoarseOptimizer(
            db_manager=db_manager,
            symbol=symbol,
            timeframe=timeframe,
            optimal_target_params=optimal_target_params,
            klines_data=klines_df,
            agg_trades_data=agg_trades_df,
            futures_data=futures_df,
        )
        
        optimizer_init_duration = time.time() - optimizer_init_start
        logger.info(f"⏱️  Optimizer initialization completed in {optimizer_init_duration:.2f} seconds")

        # Step 3.5: Run enhanced coarse optimization
        logger.info("🎯 STEP 3.5: Running enhanced coarse optimization and feature pruning...")
        optimization_start = time.time()
        
        logger.info(f"🔢 Enhanced optimization parameters:")
        logger.info(f"   - Symbol: {symbol}")
        logger.info(f"   - Timeframe: {timeframe}")
        logger.info(f"   - Klines shape: {klines_df.shape}")
        logger.info(f"   - Agg trades shape: {agg_trades_df.shape}")
        logger.info(f"   - Futures shape: {futures_df.shape}")
        logger.info(f"   - Multi-model approach: LightGBM, XGBoost, Random Forest, CatBoost")
        logger.info(f"   - Enhanced pruning: Variance + Correlation + MI + SHAP")
        logger.info(f"   - Wider hyperparameter search: Extended parameter ranges")
        
        pruned_features, hpo_ranges = coarse_optimizer.run()
        
        optimization_duration = time.time() - optimization_start
        logger.info(f"⏱️  Optimization completed in {optimization_duration:.2f} seconds")
        logger.info(f"✅ Optimization results:")
        logger.info(f"   - Pruned features count: {len(pruned_features)}")
        logger.info(f"   - HPO ranges count: {len(hpo_ranges)}")

        # Step 3.6: Save results
        logger.info("💾 STEP 3.6: Saving optimization results...")
        save_start = time.time()
        
        output_pruned_features_file = os.path.join(data_dir, f"{symbol}_pruned_features.json")
        output_hpo_ranges_file = os.path.join(data_dir, f"{symbol}_hpo_ranges.json")

        os.makedirs(data_dir, exist_ok=True)
        
        with open(output_pruned_features_file, "w") as f:
            json.dump(pruned_features, f, indent=4)
        with open(output_hpo_ranges_file, "w") as f:
            json.dump(hpo_ranges, f, indent=4)
        
        save_duration = time.time() - save_start
        logger.info(f"⏱️  Results saved in {save_duration:.2f} seconds")
        logger.info(f"📄 Results saved to:")
        logger.info(f"   - Pruned features: {output_pruned_features_file}")
        logger.info(f"   - HPO ranges: {output_hpo_ranges_file}")

        # Step 3.7: Cleanup
        logger.info("🧹 STEP 3.7: Cleaning up resources...")
        cleanup_start = time.time()
        
        await db_manager.close()  # Close DB connection
        
        cleanup_duration = time.time() - cleanup_start
        logger.info(f"⏱️  Cleanup completed in {cleanup_duration:.2f} seconds")

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 3: ENHANCED COARSE OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info(f"📊 Performance breakdown:")
        logger.info(f"   - Data loading: {data_load_duration:.2f}s ({(data_load_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Params loading: {params_load_duration:.2f}s ({(params_load_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - DB initialization: {db_init_duration:.2f}s ({(db_init_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Optimizer setup: {optimizer_init_duration:.2f}s ({(optimizer_init_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Optimization: {optimization_duration:.2f}s ({(optimization_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Results saving: {save_duration:.2f}s ({(save_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Cleanup: {cleanup_duration:.2f}s ({(cleanup_duration/total_duration)*100:.1f}%)")
        logger.info(f"📄 Output files:")
        logger.info(f"   - Pruned features: {output_pruned_features_file}")
        logger.info(f"   - HPO ranges: {output_hpo_ranges_file}")
        logger.info(f"✅ Success: True")

        return True
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ STEP 3: COARSE OPTIMIZATION FAILED")
        logger.error("=" * 80)
        logger.error(f"⏱️  Duration before failure: {total_duration:.2f} seconds")
        logger.error(f"💥 Error: {e}")
        logger.error(f"📋 Full traceback:", exc_info=True)
        return False


if __name__ == "__main__":
    # Command-line arguments: symbol, timeframe, data_dir, data_file_path, optimal_target_params_json
    symbol = sys.argv[1]
    timeframe = sys.argv[2]
    data_dir = sys.argv[3]
    data_file_path = sys.argv[4]
    optimal_target_params_json = sys.argv[5]

    success = asyncio.run(
        run_step(
            symbol, timeframe, data_dir, data_file_path, optimal_target_params_json
        )
    )

    if not success:
        sys.exit(1)  # Indicate failure
    sys.exit(0)  # Indicate success
