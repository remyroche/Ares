# src/paper_trader.py
from datetime import datetime

class PaperTrader:
    def __init__(self, initial_equity):
        self.equity = initial_equity
        self.trades = []
        self.positions = {}

    def place_order(self, symbol, side, type, quantity, price, etc=None):
        # Record the trade instead of executing it
        trade = {
            'symbol': symbol,
            'side': side,
            'type': type,
            'quantity': quantity,
            'price': price,
            'timestamp': datetime.now(),
            'status': 'FILLED'
        }
        self.trades.append(trade)

        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = 0
        
        if side.upper() == 'BUY':
            self.positions[symbol] += quantity
        elif side.upper() == 'SELL':
            self.positions[symbol] -= quantity
            
        return {'status': 'FILLED', 'orderId': f'paper_trade_{len(self.trades)}'}

    def get_position(self, symbol):
        return self.positions.get(symbol, 0)

    def get_equity(self):
        # In a real paper trader, you'd calculate equity based on PnL
        return self.equity
