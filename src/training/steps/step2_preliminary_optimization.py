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

from src.training.target_parameter_optimizer import TargetParameterOptimizer
from src.utils.logger import system_logger, setup_logging
from src.config import CONFIG  # Import CONFIG
from src.database.sqlite_manager import SQLiteManager  # Import SQLiteManager
from src.utils.error_handler import handle_errors


@handle_errors(
    exceptions=(Exception,),
    default_return=False,
    context="preliminary_optimization_step",
)
async def run_step(
    symbol: str, timeframe: str, data_dir: str, data_file_path: str
) -> bool:
    """
    Runs the initial optimization for TP, SL, and Holding Period.
    Loads data from the specified pickle file.
    """
    setup_logging()
    logger = system_logger.getChild("Step2PrelimOpt")
    
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 STEP 2: PRELIMINARY TARGET PARAMETER OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Symbol: {symbol}")
    logger.info(f"⏱️  Timeframe: {timeframe}")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"📦 Data file path: {data_file_path}")

    try:
        # Step 2.1: Load data from pickle file
        logger.info("📊 STEP 2.1: Loading data from pickle file...")
        data_load_start = time.time()
        
        if not os.path.exists(data_file_path):
            logger.error(f"❌ Data file not found: {data_file_path}")
            return False
            
        with open(data_file_path, "rb") as f:
            collected_data = pickle.load(f)
        klines_df = collected_data["klines"]
        
        data_load_duration = time.time() - data_load_start
        logger.info(f"⏱️  Data loading completed in {data_load_duration:.2f} seconds")
        logger.info(f"📊 Loaded klines data: {len(klines_df)} rows, {len(klines_df.columns)} columns")

        if klines_df.empty:
            logger.error("❌ Klines data is empty for preliminary optimization. Aborting.")
            return False

        # Step 2.2: Initialize database manager
        logger.info("🗄️  STEP 2.2: Initializing database manager...")
        db_init_start = time.time()
        
        db_manager = SQLiteManager()
        await db_manager.initialize()
        
        db_init_duration = time.time() - db_init_start
        logger.info(f"⏱️  Database initialization completed in {db_init_duration:.2f} seconds")

        # Step 2.3: Initialize target parameter optimizer
        logger.info("🔧 STEP 2.3: Initializing target parameter optimizer...")
        optimizer_init_start = time.time()
        
        # Check if we're in blank training mode from environment variable
        blank_training_mode = os.environ.get("BLANK_TRAINING_MODE", "0") == "1"
        logger.info(f"🔧 Blank training mode: {blank_training_mode}")
        logger.info(f"🔧 Environment BLANK_TRAINING_MODE: {os.environ.get('BLANK_TRAINING_MODE', '0')}")
        
        param_optimizer = TargetParameterOptimizer(
            db_manager=db_manager,
            symbol=symbol,
            timeframe=timeframe,
            klines_data=klines_df,  # Pass klines_df directly
            blank_training_mode=blank_training_mode,
        )
        
        optimizer_init_duration = time.time() - optimizer_init_start
        logger.info(f"⏱️  Optimizer initialization completed in {optimizer_init_duration:.2f} seconds")

        # Step 2.4: Run optimization
        logger.info("🎯 STEP 2.4: Running target parameter optimization...")
        optimization_start = time.time()
        
        max_trials = CONFIG["MODEL_TRAINING"]["hyperparameter_tuning"].get("max_trials", 200)
        logger.info(f"🔢 Optimization parameters:")
        logger.info(f"   - Max trials: {max_trials}")
        logger.info(f"   - Symbol: {symbol}")
        logger.info(f"   - Timeframe: {timeframe}")
        logger.info(f"   - Data shape: {klines_df.shape}")
        
        best_params = param_optimizer.run_optimization(n_trials=max_trials)
        
        optimization_duration = time.time() - optimization_start
        logger.info(f"⏱️  Optimization completed in {optimization_duration:.2f} seconds")
        logger.info(f"✅ Best parameters found:")
        for param, value in best_params.items():
            logger.info(f"   - {param}: {value}")

        # Step 2.5: Save results
        logger.info("💾 STEP 2.5: Saving optimization results...")
        save_start = time.time()
        
        output_file = os.path.join(data_dir, f"{symbol}_optimal_target_params.json")
        os.makedirs(data_dir, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(best_params, f, indent=4)
        
        save_duration = time.time() - save_start
        logger.info(f"⏱️  Results saved in {save_duration:.2f} seconds")
        logger.info(f"📄 Results saved to: {output_file}")

        # Step 2.6: Cleanup
        logger.info("🧹 STEP 2.6: Cleaning up resources...")
        cleanup_start = time.time()
        
        await db_manager.close()  # Close DB connection
        
        cleanup_duration = time.time() - cleanup_start
        logger.info(f"⏱️  Cleanup completed in {cleanup_duration:.2f} seconds")

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 2: PRELIMINARY OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info(f"📊 Performance breakdown:")
        logger.info(f"   - Data loading: {data_load_duration:.2f}s ({(data_load_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - DB initialization: {db_init_duration:.2f}s ({(db_init_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Optimizer setup: {optimizer_init_duration:.2f}s ({(optimizer_init_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Optimization: {optimization_duration:.2f}s ({(optimization_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Results saving: {save_duration:.2f}s ({(save_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Cleanup: {cleanup_duration:.2f}s ({(cleanup_duration/total_duration)*100:.1f}%)")
        logger.info(f"📄 Output file: {output_file}")
        logger.info(f"✅ Success: True")

        return True
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ STEP 2: PRELIMINARY OPTIMIZATION FAILED")
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

    success = asyncio.run(run_step(symbol, timeframe, data_dir, data_file_path))

    if not success:
        sys.exit(1)  # Indicate failure
    sys.exit(0)  # Indicate success
