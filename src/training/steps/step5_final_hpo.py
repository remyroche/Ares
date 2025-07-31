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

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.supervisor.optimizer import Optimizer  # Import Optimizer
from src.utils.logger import system_logger, setup_logging
from src.config import CONFIG  # Import CONFIG
from src.database.sqlite_manager import SQLiteManager  # Import SQLiteManager
from src.utils.error_handler import handle_errors


@handle_errors(exceptions=(Exception,), default_return=False, context="final_hpo_step")
async def run_step(
    symbol: str, data_dir: str, data_file_path: str
) -> bool:
    """
    Runs the final hyperparameter optimization using the Supervisor's Optimizer.
    Loads data from the specified pickle file and HPO ranges from JSON.
    """
    setup_logging()
    logger = system_logger.getChild("Step5FinalHPO")
    
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 STEP 5: FINAL HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Symbol: {symbol}")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"📦 Data file path: {data_file_path}")

    try:
        # Step 5.1: Load data from pickle file
        logger.info("📊 STEP 5.1: Loading data from pickle file...")
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
            logger.error("❌ Klines data is empty for final HPO. Aborting.")
            return False

        # Step 5.2: Load HPO ranges
        logger.info("🎯 STEP 5.2: Loading HPO ranges...")
        hpo_load_start = time.time()

        hpo_ranges_path = os.path.join(data_dir, f"{symbol}_hpo_ranges.json")
        if not os.path.exists(hpo_ranges_path):
            logger.error(f"❌ HPO ranges file not found: {hpo_ranges_path}")
            return False

        with open(hpo_ranges_path, 'r') as f:
            hpo_ranges = json.load(f)

        hpo_load_duration = time.time() - hpo_load_start
        logger.info(f"⏱️  HPO ranges loading completed in {hpo_load_duration:.2f} seconds")
        logger.info(f"✅ Loaded HPO ranges:")
        logger.info(f"   - Number of parameter ranges: {len(hpo_ranges)}")
        for param, range_info in hpo_ranges.items():
            logger.info(f"   - {param}: {range_info}")

        # Step 5.3: Initialize database manager
        logger.info("🗄️  STEP 5.3: Initializing database manager...")
        db_init_start = time.time()
        
        db_manager = SQLiteManager()
        await db_manager.initialize()
        
        db_init_duration = time.time() - db_init_start
        logger.info(f"⏱️  Database initialization completed in {db_init_duration:.2f} seconds")

        # Step 5.4: Initialize optimizer
        logger.info("🔧 STEP 5.4: Initializing optimizer...")
        optimizer_init_start = time.time()
        
        optimizer = Optimizer(config=CONFIG, db_manager=db_manager)
        
        optimizer_init_duration = time.time() - optimizer_init_start
        logger.info(f"⏱️  Optimizer initialization completed in {optimizer_init_duration:.2f} seconds")

        # Step 5.5: Run final hyperparameter optimization
        logger.info("🎯 STEP 5.5: Running final hyperparameter optimization...")
        optimization_start = time.time()
        
        checkpoint_file_path = os.path.join(
            CONFIG["CHECKPOINT_DIR"], f"{symbol}_optimization_checkpoint.pkl"
        )
        logger.info(f"📁 Checkpoint file path: {checkpoint_file_path}")
        logger.info(f"🔢 Optimization parameters:")
        logger.info(f"   - Symbol: {symbol}")
        logger.info(f"   - HPO ranges count: {len(hpo_ranges)}")
        logger.info(f"   - Klines shape: {klines_df.shape}")
        logger.info(f"   - Agg trades shape: {agg_trades_df.shape}")
        logger.info(f"   - Futures shape: {futures_df.shape}")
        
        optimization_result = await optimizer.implement_global_system_optimization(
            historical_pnl_data=pd.DataFrame(),  # Not directly used by Optimizer's current logic
            strategy_breakdown_data={},  # Not directly used by Optimizer's current logic
            checkpoint_file_path=checkpoint_file_path,
            hpo_ranges=hpo_ranges,  # Pass the narrowed HPO ranges
            klines_df=klines_df,  # Pass klines_df
            agg_trades_df=agg_trades_df,  # Pass agg_trades_df
            futures_df=futures_df,  # Pass futures_df
        )
        
        optimization_duration = time.time() - optimization_start
        logger.info(f"⏱️  Optimization completed in {optimization_duration:.2f} seconds")
        logger.info(f"✅ Optimization result: {optimization_result}")

        if optimization_result:
            logger.info(f"✅ Final hyperparameter optimization completed for {symbol}")
        else:
            logger.error(f"❌ Final hyperparameter optimization failed for {symbol}")
            return False

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 5: FINAL HYPERPARAMETER OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info(f"📊 Performance breakdown:")
        logger.info(f"   - Data loading: {data_load_duration:.2f}s ({(data_load_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - HPO loading: {hpo_load_duration:.2f}s ({(hpo_load_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - DB initialization: {db_init_duration:.2f}s ({(db_init_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Optimizer setup: {optimizer_init_duration:.2f}s ({(optimizer_init_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Optimization: {optimization_duration:.2f}s ({(optimization_duration/total_duration)*100:.1f}%)")
        logger.info(f"📁 Checkpoint file: {checkpoint_file_path}")
        logger.info(f"✅ Success: True")

        return True
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ STEP 5: FINAL HYPERPARAMETER OPTIMIZATION FAILED")
        logger.error("=" * 80)
        logger.error(f"⏱️  Duration before failure: {total_duration:.2f} seconds")
        logger.error(f"💥 Error: {e}")
        logger.error(f"📋 Full traceback:", exc_info=True)
        return False


if __name__ == "__main__":
    # Command-line arguments: symbol, data_dir, data_file_path
    symbol = sys.argv[1]
    data_dir = sys.argv[2]
    data_file_path = sys.argv[3]

    success = asyncio.run(run_step(symbol, data_dir, data_file_path))

    if not success:
        sys.exit(1)  # Indicate failure
    sys.exit(0)  # Indicate success
