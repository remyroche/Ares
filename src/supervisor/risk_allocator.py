# src/supervisor/risk_allocator.py
import pandas as pd
import numpy as np
from src.config import CONFIG
from src.utils.logger import system_logger

class RiskAllocator:
    def __init__(self, config=CONFIG, portfolio_risk_pct=None):
        self.config = config.get("supervisor", {})
        self.global_config = config
        self.initial_equity = self.global_config['INITIAL_EQUITY']
        self.logger = system_logger.getChild('RiskAllocator')
        self.allocated_capital_multiplier = self.config.get("initial_allocated_capital_multiplier", 1.0)
        self.portfolio_risk_pct = portfolio_risk_pct if portfolio_risk_pct is not None else CONFIG['risk_allocator'].get('portfolio_risk_pct', 0.01)


    def calculate_position_size(self, portfolio_balance, atr_value, price):
        """
        Calculates position size based on portfolio risk and volatility (ATR).
        """
        if atr_value == 0:
            return 0
            
        # How much cash are we willing to risk on this trade?
        cash_at_risk = portfolio_balance * self.portfolio_risk_pct
        
        # Position size = (Cash at Risk) / (Volatility in Dollars)
        # We use 2 * ATR as a proxy for a stop-loss distance
        position_size = cash_at_risk / (2 * atr_value)
        
        return position_size
        
    def calculate_dynamic_capital_allocation(self, historical_pnl_data: pd.DataFrame):
        self.logger.info("Calculating Dynamic Capital Allocation...")
        lookback_days = self.config.get("risk_allocation_lookback_days", 30)
        
        if historical_pnl_data.empty or len(historical_pnl_data) < lookback_days:
            self.logger.warning("Insufficient historical P&L data. Keeping current allocation.")
            return

        recent_pnl = historical_pnl_data['NetPnL'].tail(lookback_days)
        total_pnl_over_lookback = recent_pnl.sum()
        
        current_effective_capital = self.initial_equity * self.allocated_capital_multiplier
        if current_effective_capital == 0:
            return

        avg_daily_pnl_pct = total_pnl_over_lookback / (lookback_days * current_effective_capital)
        adjustment_factor = 0.1
        change_in_multiplier = avg_daily_pnl_pct * adjustment_factor * lookback_days
        new_multiplier = self.allocated_capital_multiplier + change_in_multiplier

        min_allowed_multiplier = 1.0 - self.config.get("max_capital_allocation_decrease_pct", 0.75)
        max_allowed_multiplier = 1.0 + self.config.get("max_capital_allocation_increase_pct", 1.0)

        self.allocated_capital_multiplier = np.clip(new_multiplier, min_allowed_multiplier, max_allowed_multiplier)
        
        self.logger.info(f"New Allocated Capital Multiplier: {self.allocated_capital_multiplier:.2f}x")

    def get_current_allocated_capital(self):
        return self.initial_equity * self.allocated_capital_multiplier
