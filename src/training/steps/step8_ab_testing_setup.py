import asyncio
import time
import traceback
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import system_logger, setup_logging
from src.config import CONFIG  # Import CONFIG
from src.database.sqlite_manager import SQLiteManager  # Import SQLiteManager
from src.utils.error_handler import handle_errors


@handle_errors(
    exceptions=(Exception,), default_return=False, context="ab_testing_setup_step"
)
async def run_step(symbol: str) -> bool:
    """
    Sets up A/B testing by saving configuration to the database.
    """
    setup_logging()
    logger = system_logger.getChild("Step8ABTestingSetup")
    
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 STEP 8: A/B TESTING SETUP")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Symbol: {symbol}")

    try:
        # Step 8.1: Initialize database manager
        logger.info("🗄️  STEP 8.1: Initializing database manager...")
        db_init_start = time.time()
        
        db_manager = SQLiteManager()
        await db_manager.initialize()
        
        db_init_duration = time.time() - db_init_start
        logger.info(f"⏱️  Database initialization completed in {db_init_duration:.2f} seconds")

        # Step 8.2: Prepare A/B testing configuration
        logger.info("⚙️  STEP 8.2: Preparing A/B testing configuration...")
        config_start = time.time()
        
        ab_duration = CONFIG["MODEL_TRAINING"]["ab_test_duration_days"]
        start_date = datetime.now()
        end_date = start_date + timedelta(days=ab_duration)
        
        ab_config = {
            "symbol": symbol,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "duration_days": ab_duration,
            "status": "active",
            "models": {
                "model_a": "current_model",  # This refers to the 'champion' model
                "model_b": "new_model",  # This refers to the newly trained 'candidate' model
            },
            "metrics": ["accuracy", "sharpe_ratio", "max_drawdown"],
        }
        
        config_duration = time.time() - config_start
        logger.info(f"⏱️  Configuration preparation completed in {config_duration:.2f} seconds")
        logger.info(f"✅ A/B testing configuration:")
        logger.info(f"   - Duration: {ab_duration} days")
        logger.info(f"   - Start date: {start_date.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   - End date: {end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"   - Models: {list(ab_config['models'].keys())}")
        logger.info(f"   - Metrics: {ab_config['metrics']}")

        # Step 8.3: Save configuration to database
        logger.info("💾 STEP 8.3: Saving A/B testing configuration to database...")
        save_start = time.time()
        
        document_key = f"{symbol}_ab_test"
        await db_manager.set_document("ab_tests", document_key, ab_config)
        
        save_duration = time.time() - save_start
        logger.info(f"⏱️  Configuration saved in {save_duration:.2f} seconds")
        logger.info(f"📄 Database document key: {document_key}")

        # Step 8.4: Cleanup
        logger.info("🧹 STEP 8.4: Cleaning up resources...")
        cleanup_start = time.time()
        
        await db_manager.close()  # Close DB connection
        
        cleanup_duration = time.time() - cleanup_start
        logger.info(f"⏱️  Cleanup completed in {cleanup_duration:.2f} seconds")

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 8: A/B TESTING SETUP COMPLETE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info(f"📊 Performance breakdown:")
        logger.info(f"   - DB initialization: {db_init_duration:.2f}s ({(db_init_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Config preparation: {config_duration:.2f}s ({(config_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Database save: {save_duration:.2f}s ({(save_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Cleanup: {cleanup_duration:.2f}s ({(cleanup_duration/total_duration)*100:.1f}%)")
        logger.info(f"📄 Database document: {document_key}")
        logger.info(f"✅ Success: True")

        return True
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ STEP 8: A/B TESTING SETUP FAILED")
        logger.error("=" * 80)
        logger.error(f"⏱️  Duration before failure: {total_duration:.2f} seconds")
        logger.error(f"💥 Error: {e}")
        logger.error(f"📋 Full traceback:", exc_info=True)
        return False


if __name__ == "__main__":
    # Command-line arguments: symbol
    symbol = sys.argv[1]

    success = asyncio.run(run_step(symbol))

    if not success:
        sys.exit(1)  # Indicate failure
    sys.exit(0)  # Indicate success
