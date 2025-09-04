from __future__ import annotations
import asyncio
import os
from celery import Celery
from celery.schedules import crontab
from src.ares_pipeline import AresPipeline
from src.database.sqlite_manager import SQLiteManager
from src.training.core.training_manager import create_training_manager
app = Celery('ares_tasks', broker='redis://localhost:6379/0')

@app.task
def run_trading_bot_instance(symbol: str, exchange: str) -> None:
    """
    Celery task to run a single trading bot instance.
    This is now called by the main pipeline, not directly by the user.
    """
    os.environ['ARES_SYMBOL'] = symbol
    os.environ['ARES_EXCHANGE'] = exchange
    pipeline = AresPipeline()
    asyncio.run(pipeline.run_async())

@app.task
def run_monthly_training_pipeline() -> None:
    """
    Celery task to run the monthly retraining and validation pipeline using TrainingManager.
    """
    print('Celery Task: Kicking off monthly training pipeline...')
    try:

        async def run_training() -> None:
            db_manager = SQLiteManager({})
            await db_manager.initialize()
            training_manager = TrainingManager(db_manager)
            env_settings = get_environment_settings()
            symbol = env_settings.trade_symbol
            exchange_name = env_settings.exchange_name
            success = await training_manager.run_full_training(symbol, exchange_name)
            if success:
                print(f'Monthly training pipeline completed successfully for {symbol}')
            else:
                print(f'Monthly training pipeline failed for {symbol}')
            await db_manager.close()
        asyncio.run(run_training())
    except Exception as e:
        print(f'An unexpected error occurred while running the training pipeline task: {e}')
app.conf.beat_schedule = {'run-monthly-training': {'task': 'src.tasks.run_monthly_training_pipeline', 'schedule': crontab(day_of_month='1', hour=0, minute=0)}}
app.conf.timezone = 'UTC'