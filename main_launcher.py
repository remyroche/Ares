# main_launcher.py (new file)
from src.tasks import run_trading_bot

def main():
    # Define the symbols and exchanges you want to trade
    trading_pairs = [
        {'symbol': 'BTCUSDT', 'exchange': 'binance'},
        # {'symbol': 'ETHUSDT', 'exchange': 'binance'},
        # Add more pairs and exchanges here
    ]

    for pair in trading_pairs:
        run_trading_bot.delay(pair['symbol'], pair['exchange'])
        print(f"Launched trading bot for {pair['symbol']} on {pair['exchange']}")

if __name__ == "__main__":
    main()
