# main_launcher.py
import subprocess
import time
import sys
import os
from src.tasks import run_trading_bot_instance

def start_service(command, name):
    """Starts a background service in a new terminal window."""
    print(f"Starting {name}...")
    try:
        # This command is OS-specific. This example is for macOS.
        # For Linux, you might use 'gnome-terminal --' or 'xterm -e'.
        # For Windows, you might use 'start cmd /c'.
        if sys.platform == "darwin": # macOS
            script = f'tell app "Terminal" to do script "{command}"'
            subprocess.Popen(['osascript', '-e', script])
        elif sys.platform.startswith("linux"):
            # Try gnome-terminal first, then xterm
            try:
                subprocess.Popen(['gnome-terminal', '--', 'bash', '-c', f'{command}; exec bash'])
            except FileNotFoundError:
                subprocess.Popen(['xterm', '-e', f'{command}; exec bash'])
        elif sys.platform == "win32":
            subprocess.Popen(f'start cmd /k "{command}"', shell=True)
        else:
            print(f"Unsupported OS '{sys.platform}'. Please start {name} manually with command: {command}")
            return False
        print(f"{name} started in a new terminal.")
        return True
    except Exception as e:
        print(f"Failed to start {name}. Please start it manually. Error: {e}")
        print(f"Command: {command}")
        return False

def main():
    """
    Main entry point for launching the entire Ares live trading system.
    Starts Redis, Celery workers, and dispatches trading bot tasks.
    """
    print("--- Launching Ares Live Trading System ---")

    # 1. Start Redis (if not already running)
    # This is a simple check; a more robust solution would use `redis-cli ping`
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379)
        r.ping()
        print("Redis is already running.")
    except (redis.exceptions.ConnectionError, ImportError):
        print("Redis not found or not running. Attempting to start...")
        start_service("redis-server", "Redis")
        time.sleep(3) # Give Redis a moment to start

    # 2. Start Celery Worker
    celery_command = "celery -A src.tasks worker --loglevel=info"
    start_service(celery_command, "Celery Worker")
    time.sleep(5) # Give the worker a moment to initialize

    # 3. Start Celery Beat for scheduled tasks (like monthly retraining)
    celery_beat_command = "celery -A src.tasks beat --loglevel=info"
    start_service(celery_beat_command, "Celery Beat Scheduler")
    time.sleep(3)

    # 4. Define the symbols and exchanges you want to trade
    trading_pairs = [
        {'symbol': 'BTCUSDT', 'exchange': 'binance'},
        {'symbol': 'ETHUSDT', 'exchange': 'binance'},
        # Add more pairs and exchanges here
    ]

    # 5. Dispatch a trading task for each pair
    print("\nDispatching trading bot tasks...")
    for pair in trading_pairs:
        run_trading_bot_instance.delay(pair['symbol'], pair['exchange'])
        print(f"  -> Dispatched task for {pair['symbol']} on {pair['exchange']}")
    
    print("\n--- All services started and tasks dispatched. ---")
    print("Monitor the Celery Worker and individual bot logs for real-time activity.")
    print("You can close this launcher window.")

if __name__ == "__main__":
    main()
