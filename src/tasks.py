# src/tasks.py
from celery import Celery
from celery.schedules import crontab
import subprocess
import sys
import os

# Configure Celery
app = Celery('ares_tasks', broker='redis://localhost:6379/0')

@app.task
def run_trading_bot_instance(symbol, exchange):
    """
    Celery task to run a single trading bot instance.
    This is now called by the main pipeline, not directly by the user.
    """
    from src.ares_pipeline import AresPipeline # Import locally to avoid circular dependencies
    
    # Set environment variables for this specific instance
    os.environ['ARES_SYMBOL'] = symbol
    os.environ['ARES_EXCHANGE'] = exchange

    pipeline = AresPipeline()
    # The pipeline's run_async method will be called by the worker
    # We assume the pipeline is designed to run indefinitely.
    import asyncio
    asyncio.run(pipeline.run_async())


@app.task
def run_monthly_training_pipeline():
    """
    Celery task to run the monthly retraining and validation pipeline.
    """
    print("Celery Task: Kicking off monthly training pipeline...")
    try:
        # We run this as a subprocess to ensure it uses the latest code
        # and has a clean environment.
        script_path = os.path.join(os.path.dirname(__file__), '..', 'backtesting', 'training_pipeline.py')
        result = subprocess.run([sys.executable, script_path], capture_output=True, text=True, check=True)
        print("Monthly training pipeline completed successfully.")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Monthly training pipeline failed.")
        print(e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while running the training pipeline task: {e}")


# --- Celery Beat Schedule ---
# This schedule automatically triggers tasks at specified times.
# To run the beat scheduler: celery -A src.tasks beat --loglevel=info
app.conf.beat_schedule = {
    'run-monthly-training': {
        'task': 'src.tasks.run_monthly_training_pipeline',
        # Executes at midnight on the first day of every month.
        'schedule': crontab(day_of_month='1', hour=0, minute=0),
    },
}

app.conf.timezone = 'UTC'
