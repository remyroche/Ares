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

from src.supervisor.optimizer import Optimizer
from src.utils.logger import system_logger, setup_logging
from src.config import CONFIG
from src.database.sqlite_manager import SQLiteManager
from src.utils.error_handler import handle_errors


# Check if this is a blank training run and override configuration accordingly
if os.environ.get('BLANK_TRAINING_MODE') == '1':
    print("🔧 BLANK TRAINING MODE DETECTED - Overriding configuration for faster execution")
    # For blank runs, use minimal trials per stage
    STAGE_TRIALS = [1, 1, 1, 1]  # Total: 4 trials for ultra-fast testing
    print(f"✅ Configuration overridden: 4-stage HPO with trials {STAGE_TRIALS}")
else:
    # Full 4-stage optimization
    STAGE_TRIALS = [5, 20, 30, 50]  # Total: 105 trials


@handle_errors(exceptions=(Exception,), default_return=False, context="multi_stage_hpo_step")
async def run_step(
    symbol: str, data_dir: str, data_file_path: str
) -> bool:
    """
    Runs a 4-stage hyperparameter optimization system:
    - Stage 1: Ultra-Coarse (5 trials) - Very wide ranges
    - Stage 2: Coarse (20 trials) - Narrowed ranges from Stage 1
    - Stage 3: Medium (30 trials) - Further narrowed ranges from Stage 2
    - Stage 4: Fine (50 trials) - Final optimization with precise ranges
    """
    setup_logging()
    logger = system_logger.getChild("Step5MultiStageHPO")
    
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 STEP 5: 4-STAGE HYPERPARAMETER OPTIMIZATION")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Symbol: {symbol}")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"📦 Data file path: {data_file_path}")
    logger.info(f"🔢 Stage trials: {STAGE_TRIALS} (Total: {sum(STAGE_TRIALS)} trials)")

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

        if klines_df.empty:
            logger.error("❌ Klines data is empty for multi-stage HPO. Aborting.")
            return False

        # Step 5.2: Load initial HPO ranges
        logger.info("🎯 STEP 5.2: Loading initial HPO ranges...")
        hpo_load_start = time.time()

        hpo_ranges_path = os.path.join(data_dir, f"{symbol}_hpo_ranges.json")
        if not os.path.exists(hpo_ranges_path):
            logger.error(f"❌ HPO ranges file not found: {hpo_ranges_path}")
            return False

        with open(hpo_ranges_path, 'r') as f:
            initial_hpo_ranges = json.load(f)

        hpo_load_duration = time.time() - hpo_load_start
        logger.info(f"⏱️  HPO ranges loading completed in {hpo_load_duration:.2f} seconds")
        logger.info(f"✅ Loaded initial HPO ranges: {len(initial_hpo_ranges)} parameters")

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

        # Step 5.5: Run 4-stage hyperparameter optimization
        logger.info("🎯 STEP 5.5: Running 4-stage hyperparameter optimization...")
        optimization_start = time.time()
        
        checkpoint_file_path = os.path.join(
            CONFIG["CHECKPOINT_DIR"], f"{symbol}_multi_stage_optimization_checkpoint.pkl"
        )
        
        current_ranges = initial_hpo_ranges.copy()
        stage_results = []
        
        for stage_num, trials in enumerate(STAGE_TRIALS, 1):
            stage_start_time = time.time()
            logger.info(f"🔄 STAGE {stage_num}/4: Running {trials} trials...")
            logger.info(f"   - Trials: {trials}")
            logger.info(f"   - Parameters: {len(current_ranges)}")
            
            # Run optimization for this stage
            stage_result = await optimizer.implement_global_system_optimization(
                historical_pnl_data=pd.DataFrame(),
                strategy_breakdown_data={},
                checkpoint_file_path=checkpoint_file_path,
                hpo_ranges=current_ranges,
                klines_df=klines_df,
                agg_trades_df=agg_trades_df,
                futures_df=futures_df,
                max_trials=trials,  # Limit trials for this stage
                stage_name=f"stage_{stage_num}"
            )
            
            if not stage_result:
                logger.error(f"❌ Stage {stage_num} failed")
                return False
            
            stage_duration = time.time() - stage_start_time
            stage_results.append({
                'stage': stage_num,
                'trials': trials,
                'duration': stage_duration,
                'result': stage_result
            })
            
            logger.info(f"✅ Stage {stage_num} completed in {stage_duration:.2f} seconds")
            
            # For stages 1-3, narrow the parameter ranges based on results
            if stage_num < 4:
                # This would require implementing range narrowing logic
                # For now, we'll use the same ranges for the next stage
                logger.info(f"📊 Stage {stage_num} results: {stage_result}")
                # TODO: Implement range narrowing based on stage results
        
        optimization_duration = time.time() - optimization_start
        logger.info(f"⏱️  4-stage optimization completed in {optimization_duration:.2f} seconds")
        
        # Step 5.6: Save final results
        logger.info("💾 STEP 5.6: Saving final results...")
        save_start = time.time()
        
        results_file = os.path.join(data_dir, f"{symbol}_multi_stage_hpo_results.json")
        with open(results_file, "w") as f:
            json.dump({
                'stage_results': stage_results,
                'total_trials': sum(STAGE_TRIALS),
                'total_duration': optimization_duration,
                'final_ranges': current_ranges
            }, f, indent=4)
        
        save_duration = time.time() - save_start
        logger.info(f"⏱️  Results saved in {save_duration:.2f} seconds")

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 5: 4-STAGE HYPERPARAMETER OPTIMIZATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info(f"📊 Performance breakdown:")
        logger.info(f"   - Data loading: {data_load_duration:.2f}s ({(data_load_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - HPO loading: {hpo_load_duration:.2f}s ({(hpo_load_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - DB initialization: {db_init_duration:.2f}s ({(db_init_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Optimizer setup: {optimizer_init_duration:.2f}s ({(optimizer_init_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - 4-stage optimization: {optimization_duration:.2f}s ({(optimization_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Saving: {save_duration:.2f}s ({(save_duration/total_duration)*100:.1f}%)")
        
        logger.info(f"📈 Stage breakdown:")
        for stage_result in stage_results:
            logger.info(f"   - Stage {stage_result['stage']}: {stage_result['trials']} trials in {stage_result['duration']:.2f}s")
        
        logger.info(f"📁 Results saved to: {results_file}")
        logger.info(f"✅ Success: True")

        return True
        
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ STEP 5: 4-STAGE HYPERPARAMETER OPTIMIZATION FAILED")
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