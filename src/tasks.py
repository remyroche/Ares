# src/tasks.py
import os

from celery import Celery
from celery.schedules import crontab

# Configure Celery
app = Celery("ares_tasks", broker="redis://localhost:6379/0")


@app.task
def run_trading_bot_instance(symbol: str, exchange: str) -> None:
    """
    Celery task to run a single trading bot instance.
    This is now called by the main pipeline, not directly by the user.
    """
    from src.ares_pipeline import (
        AresPipeline,
    )  # Import locally to avoid circular dependencies

    # Set environment variables for this specific instance
    os.environ["ARES_SYMBOL"] = symbol
    os.environ["ARES_EXCHANGE"] = exchange

    pipeline = AresPipeline()
    # The pipeline's run_async method will be called by the worker
    # We assume the pipeline is designed to run indefinitely.
    import asyncio

    asyncio.run(pipeline.run_async())


@app.task
def run_monthly_training_pipeline() -> None:
    """
    Celery task to run the monthly retraining and validation pipeline using TrainingManager.
    """
    print("Celery Task: Kicking off monthly training pipeline...")
    try:
        import asyncio

        from src.config import get_environment_settings
        from src.database.sqlite_manager import SQLiteManager
        from src.training.enhanced_training_manager import EnhancedTrainingManager

        async def run_training():
            # Initialize database manager
            db_manager = SQLiteManager({})
            await db_manager.initialize()

            # Initialize enhanced training manager
            training_manager = EnhancedTrainingManager(db_manager)

            # Get current trading symbol and exchange
            env_settings = get_environment_settings()
            symbol = env_settings.trade_symbol
            exchange_name = env_settings.exchange_name

            # Run full training pipeline
            success = await training_manager.run_full_training(symbol, exchange_name)

            if success:
                print(f"Monthly training pipeline completed successfully for {symbol}")
            else:
                print(f"Monthly training pipeline failed for {symbol}")

            # Close database connection
            await db_manager.close()

        # Run the async training function
        asyncio.run(run_training())

    except Exception as e:
        print(
            f"An unexpected error occurred while running the training pipeline task: {e}",
        )


# --- Celery Beat Schedule ---
# This schedule automatically triggers tasks at specified times.
# To run the beat scheduler: celery -A src.tasks beat --loglevel=info
app.conf.beat_schedule = {
    "run-monthly-training": {
        "task": "src.tasks.run_monthly_training_pipeline",
        # Executes at midnight on the first day of every month.
        "schedule": crontab(day_of_month="1", hour=0, minute=0),
    },
}

app.conf.timezone = "UTC"
