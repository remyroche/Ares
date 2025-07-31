import asyncio
import os
import time
import traceback
from datetime import datetime
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
    exceptions=(Exception,), default_return=False, context="save_training_results_step"
)
async def run_step(
    symbol: str,
    session_id: str,
    mlflow_run_id: str,
    data_dir: str,
    reports_dir: str,
    models_dir: str,
) -> bool:
    """
    Saves training results and metadata to the database and local files.
    """
    setup_logging()
    logger = system_logger.getChild("Step9SaveResults")
    
    start_time = time.time()
    logger.info("=" * 80)
    logger.info("🚀 STEP 9: SAVE TRAINING RESULTS")
    logger.info("=" * 80)
    logger.info(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"🎯 Symbol: {symbol}")
    logger.info(f"🆔 Session ID: {session_id}")
    logger.info(f"🔬 MLflow Run ID: {mlflow_run_id}")
    logger.info(f"📁 Data directory: {data_dir}")
    logger.info(f"📄 Reports directory: {reports_dir}")
    logger.info(f"🤖 Models directory: {models_dir}")

    try:
        # Step 9.1: Initialize database manager
        logger.info("🗄️  STEP 9.1: Initializing database manager...")
        db_init_start = time.time()
        
        db_manager = SQLiteManager()
        await db_manager.initialize()
        
        db_init_duration = time.time() - db_init_start
        logger.info(f"⏱️  Database initialization completed in {db_init_duration:.2f} seconds")

        # Step 9.2: Prepare training session metadata
        logger.info("📋 STEP 9.2: Preparing training session metadata...")
        session_start = time.time()
        
        current_training_session = {
            "session_id": session_id,
            "symbol": symbol,
            "mlflow_run_id": mlflow_run_id,
            "end_time": datetime.now().isoformat(),
            "status": "completed",  # Assume completed if this script runs
        }
        
        session_duration = time.time() - session_start
        logger.info(f"⏱️  Session metadata preparation completed in {session_duration:.2f} seconds")
        logger.info(f"✅ Training session metadata:")
        logger.info(f"   - Session ID: {session_id}")
        logger.info(f"   - Symbol: {symbol}")
        logger.info(f"   - MLflow Run ID: {mlflow_run_id}")
        logger.info(f"   - Status: completed")

        # Step 9.3: Save training session metadata
        logger.info("💾 STEP 9.3: Saving training session metadata...")
        session_save_start = time.time()
        
        await db_manager.set_document(
            "training_sessions", session_id, current_training_session
        )
        
        session_save_duration = time.time() - session_save_start
        logger.info(f"⏱️  Session metadata saved in {session_save_duration:.2f} seconds")

        # Step 9.4: Prepare model checkpoint metadata
        logger.info("📋 STEP 9.4: Preparing model checkpoint metadata...")
        checkpoint_start = time.time()
        
        model_checkpoint = {
            "symbol": symbol,
            "session_id": session_id,
            "trained_at": datetime.now().isoformat(),
            "model_paths": {
                "analyst_models": os.path.join(
                    CONFIG["CHECKPOINT_DIR"], "analyst_models"
                ),
                "supervisor_models": os.path.join(
                    CONFIG["CHECKPOINT_DIR"], "supervisor_models"
                ),
                "optimization_results": os.path.join(
                    models_dir, f"{symbol}_optimization_checkpoint.pkl"
                ),
            },
            "validation_reports": {
                "walk_forward": os.path.join(
                    reports_dir, f"{symbol}_walk_forward_report.txt"
                ),
                "monte_carlo": os.path.join(
                    reports_dir, f"{symbol}_monte_carlo_report.txt"
                ),
            },
        }
        
        checkpoint_duration = time.time() - checkpoint_start
        logger.info(f"⏱️  Model checkpoint metadata preparation completed in {checkpoint_duration:.2f} seconds")
        logger.info(f"✅ Model checkpoint metadata:")
        logger.info(f"   - Symbol: {symbol}")
        logger.info(f"   - Session ID: {session_id}")
        logger.info(f"   - Model paths count: {len(model_checkpoint['model_paths'])}")
        logger.info(f"   - Validation reports count: {len(model_checkpoint['validation_reports'])}")

        # Step 9.5: Save model checkpoint metadata
        logger.info("💾 STEP 9.5: Saving model checkpoint metadata...")
        checkpoint_save_start = time.time()
        
        checkpoint_key = f"{symbol}_{session_id}"
        await db_manager.set_document(
            "model_checkpoints", checkpoint_key, model_checkpoint
        )
        
        checkpoint_save_duration = time.time() - checkpoint_save_start
        logger.info(f"⏱️  Model checkpoint metadata saved in {checkpoint_save_duration:.2f} seconds")
        logger.info(f"📄 Database checkpoint key: {checkpoint_key}")

        # Step 9.6: Verify file existence
        logger.info("🔍 STEP 9.6: Verifying file existence...")
        verify_start = time.time()
        
        files_to_check = [
            model_checkpoint["model_paths"]["analyst_models"],
            model_checkpoint["model_paths"]["supervisor_models"],
            model_checkpoint["model_paths"]["optimization_results"],
            model_checkpoint["validation_reports"]["walk_forward"],
            model_checkpoint["validation_reports"]["monte_carlo"],
        ]
        
        existing_files = []
        missing_files = []
        
        for file_path in files_to_check:
            if os.path.exists(file_path):
                existing_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        verify_duration = time.time() - verify_start
        logger.info(f"⏱️  File verification completed in {verify_duration:.2f} seconds")
        logger.info(f"📊 File verification results:")
        logger.info(f"   - Existing files: {len(existing_files)}")
        logger.info(f"   - Missing files: {len(missing_files)}")
        if missing_files:
            logger.warning(f"⚠️  Missing files:")
            for file_path in missing_files:
                logger.warning(f"   - {file_path}")

        # Step 9.7: Cleanup
        logger.info("🧹 STEP 9.7: Cleaning up resources...")
        cleanup_start = time.time()
        
        await db_manager.close()  # Close DB connection
        
        cleanup_duration = time.time() - cleanup_start
        logger.info(f"⏱️  Cleanup completed in {cleanup_duration:.2f} seconds")

        # Final summary
        total_duration = time.time() - start_time
        logger.info("=" * 80)
        logger.info("🎉 STEP 9: SAVE TRAINING RESULTS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"⏱️  Total duration: {total_duration:.2f} seconds")
        logger.info(f"📊 Performance breakdown:")
        logger.info(f"   - DB initialization: {db_init_duration:.2f}s ({(db_init_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Session prep: {session_duration:.2f}s ({(session_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Session save: {session_save_duration:.2f}s ({(session_save_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Checkpoint prep: {checkpoint_duration:.2f}s ({(checkpoint_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Checkpoint save: {checkpoint_save_duration:.2f}s ({(checkpoint_save_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - File verification: {verify_duration:.2f}s ({(verify_duration/total_duration)*100:.1f}%)")
        logger.info(f"   - Cleanup: {cleanup_duration:.2f}s ({(cleanup_duration/total_duration)*100:.1f}%)")
        logger.info(f"📄 Database documents:")
        logger.info(f"   - Training session: {session_id}")
        logger.info(f"   - Model checkpoint: {checkpoint_key}")
        logger.info(f"📁 Files verified: {len(existing_files)}/{len(files_to_check)} exist")
        logger.info(f"✅ Success: True")

        return True
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error("=" * 80)
        logger.error("❌ STEP 9: SAVE TRAINING RESULTS FAILED")
        logger.error("=" * 80)
        logger.error(f"⏱️  Duration before failure: {total_duration:.2f} seconds")
        logger.error(f"💥 Error: {e}")
        logger.error(f"📋 Full traceback:", exc_info=True)
        return False


if __name__ == "__main__":
    # Command-line arguments: symbol, session_id, mlflow_run_id, data_dir, reports_dir, models_dir
    symbol = sys.argv[1]
    session_id = sys.argv[2]
    mlflow_run_id = sys.argv[3]
    data_dir = sys.argv[4]
    reports_dir = sys.argv[5]
    models_dir = sys.argv[6]

    success = asyncio.run(
        run_step(symbol, session_id, mlflow_run_id, data_dir, reports_dir, models_dir)
    )

    if not success:
        sys.exit(1)  # Indicate failure
    sys.exit(0)  # Indicate success
