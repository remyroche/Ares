# src/tasks.py
from celery import Celery
from .ares_pipeline import AresPipeline
import os

# Configure Celery
app = Celery('ares_tasks', broker='redis://localhost:6379/0')

@app.task
def run_trading_bot(symbol, exchange):
    """
    Celery task to run the trading bot for a specific symbol and exchange.
    """
    # You might need to adjust your config loading here to be dynamic
    # based on the symbol and exchange.
    # For now, let's assume the config is adapted inside the pipeline.
    
    # Set environment variables for the pipeline to use
    os.environ['ARES_SYMBOL'] = symbol
    os.environ['ARES_EXCHANGE'] = exchange

    pipeline = AresPipeline()
    pipeline.run()
