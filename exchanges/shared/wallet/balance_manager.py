"""
Balance Management Utilities

Handles balance tracking, equity calculations, and balance validation.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger


class AccountType(Enum):
    """Account type enumeration"""
    SPOT = "spot"
    FUTURES = "futures"
    MARGIN = "margin"
    UNIFIED = "unified"
    CLASSIC = "classic"


@dataclass
class Balance:
    """Balance data structure"""
    currency: str
    available: float
    frozen: float
    total: float
    account_type: AccountType
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AccountEquity:
    """Account equity data structure"""
    account_type: AccountType
    total_equity: float
    available_equity: float
    frozen_equity: float
    unrealized_pnl: float
    realized_pnl: float
    balances: List[Balance]
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BalanceManager:
    """
    Manages balance tracking and equity calculations.
    """
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"BalanceManager.{exchange_name}")
        
        # Balance storage
        self.balances: Dict[str, Dict[AccountType, Balance]] = {}  # currency -> account_type -> balance
        self.account_equities: Dict[AccountType, AccountEquity] = {}
        
        # Balance fetching functions
        self.fetch_functions: Dict[str, callable] = {}
        
        # Cache settings
        self.cache_ttl = timedelta(seconds=30)
        self.last_fetch: Optional[datetime] = None
        
    def register_fetch_functions(
        self,
        get_balances: callable,
        get_account_info: Optional[callable] = None
    ) -> None:
        """
        Register exchange-specific balance fetching functions.
        
        Args:
            get_balances: Function to get balances
            get_account_info: Optional function to get account information
        """
        self.fetch_functions = {
            "get_balances": get_balances,
            "get_account_info": get_account_info
        }
        
        self.logger.info("Registered balance fetching functions")
    
    async def fetch_balances(self, account_type: AccountType) -> List[Balance]:
        """
        Fetch balances for a specific account type.
        
        Args:
            account_type: Account type to fetch balances for
            
        Returns:
            List of Balance objects
        """
        try:
            if "get_balances" not in self.fetch_functions:
                self.logger.warning("No get_balances function registered")
                return []
            
            raw_data = await self.fetch_functions["get_balances"](account_type.value)
            if not raw_data:
                self.logger.warning(f"No balance data received for {account_type.value}")
                return []
            
            # Parse balance data
            balances = self._parse_balance_data(raw_data, account_type)
            
            # Update cache
            self._update_balance_cache(balances, account_type)
            
            self.last_fetch = datetime.now()
            return balances
            
        except Exception as e:
            self.logger.error(f"Error fetching balances for {account_type.value}: {e}")
            return []
    
    def _parse_balance_data(self, raw_data: List[Dict[str, Any]], account_type: AccountType) -> List[Balance]:
        """Parse raw balance data into Balance objects."""
        balances = []
        
        for item in raw_data:
            try:
                currency = item.get("currency") or item.get("ccy", "")
                if not currency:
                    continue
                
                available = float(item.get("available", 0) or item.get("avail", 0))
                frozen = float(item.get("frozen", 0) or item.get("frozenBal", 0))
                total = float(item.get("total", 0) or item.get("totalEq", 0))
                
                # If total is not provided, calculate it
                if total == 0:
                    total = available + frozen
                
                balance = Balance(
                    currency=currency,
                    available=available,
                    frozen=frozen,
                    total=total,
                    account_type=account_type,
                    metadata=item
                )
                
                balances.append(balance)
                
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Error parsing balance data: {e}")
                continue
        
        return balances
    
    def _update_balance_cache(self, balances: List[Balance], account_type: AccountType) -> None:
        """Update balance cache with new data."""
        for balance in balances:
            if balance.currency not in self.balances:
                self.balances[balance.currency] = {}
            
            self.balances[balance.currency][account_type] = balance
    
    def get_balance(self, currency: str, account_type: AccountType) -> Optional[Balance]:
        """Get balance for a specific currency and account type."""
        return self.balances.get(currency, {}).get(account_type)
    
    def get_all_balances(self, account_type: AccountType) -> List[Balance]:
        """Get all balances for an account type."""
        return [
            balance for currency_balances in self.balances.values()
            for balance in [currency_balances.get(account_type)]
            if balance is not None
        ]
    
    def get_balance_summary(self, account_type: AccountType) -> Dict[str, Any]:
        """Get balance summary for an account type."""
        balances = self.get_all_balances(account_type)
        
        total_equity = sum(balance.total for balance in balances)
        available_equity = sum(balance.available for balance in balances)
        frozen_equity = sum(balance.frozen for balance in balances)
        
        return {
            "account_type": account_type.value,
            "total_equity": total_equity,
            "available_equity": available_equity,
            "frozen_equity": frozen_equity,
            "currency_count": len(balances),
            "non_zero_balances": len([b for b in balances if b.total > 0])
        }
    
    def calculate_total_equity(self, account_type: AccountType) -> float:
        """Calculate total equity for an account type."""
        balances = self.get_all_balances(account_type)
        return sum(balance.total for balance in balances)
    
    def calculate_available_equity(self, account_type: AccountType) -> float:
        """Calculate available equity for an account type."""
        balances = self.get_all_balances(account_type)
        return sum(balance.available for balance in balances)
    
    def get_usdt_balance(self, account_type: AccountType) -> float:
        """Get USDT balance for an account type."""
        usdt_balance = self.get_balance("USDT", account_type)
        return usdt_balance.total if usdt_balance else 0.0
    
    def get_btc_balance(self, account_type: AccountType) -> float:
        """Get BTC balance for an account type."""
        btc_balance = self.get_balance("BTC", account_type)
        return btc_balance.total if btc_balance else 0.0
    
    def has_sufficient_balance(
        self,
        currency: str,
        amount: float,
        account_type: AccountType
    ) -> bool:
        """Check if there's sufficient balance for a transaction."""
        balance = self.get_balance(currency, account_type)
        if not balance:
            return False
        
        return balance.available >= amount
    
    def get_balance_utilization(self, currency: str, account_type: AccountType) -> float:
        """Get balance utilization ratio (frozen / total)."""
        balance = self.get_balance(currency, account_type)
        if not balance or balance.total == 0:
            return 0.0
        
        return balance.frozen / balance.total
    
    def get_top_balances(self, account_type: AccountType, limit: int = 10) -> List[Balance]:
        """Get top balances by total value."""
        balances = self.get_all_balances(account_type)
        return sorted(balances, key=lambda x: x.total, reverse=True)[:limit]
    
    def get_non_zero_balances(self, account_type: AccountType) -> List[Balance]:
        """Get all non-zero balances."""
        balances = self.get_all_balances(account_type)
        return [balance for balance in balances if balance.total > 0]
    
    def calculate_portfolio_value(
        self,
        account_type: AccountType,
        prices: Dict[str, float],
        base_currency: str = "USDT"
    ) -> float:
        """
        Calculate portfolio value in base currency.
        
        Args:
            account_type: Account type
            prices: Price dictionary (currency -> price)
            base_currency: Base currency for calculation
            
        Returns:
            Portfolio value in base currency
        """
        balances = self.get_all_balances(account_type)
        total_value = 0.0
        
        for balance in balances:
            if balance.currency == base_currency:
                total_value += balance.total
            elif balance.currency in prices:
                total_value += balance.total * prices[balance.currency]
        
        return total_value
    
    def should_refresh_balances(self) -> bool:
        """Check if balances should be refreshed."""
        if not self.last_fetch:
            return True
        
        return datetime.now() - self.last_fetch > self.cache_ttl
    
    async def ensure_fresh_balances(self, account_type: AccountType) -> List[Balance]:
        """Ensure balances are fresh, refresh if needed."""
        if self.should_refresh_balances():
            return await self.fetch_balances(account_type)
        
        return self.get_all_balances(account_type)
    
    def get_balance_statistics(self) -> Dict[str, Any]:
        """Get balance statistics."""
        total_currencies = len(self.balances)
        total_account_types = len(self.account_equities)
        
        account_summaries = {}
        for account_type in AccountType:
            summary = self.get_balance_summary(account_type)
            if summary["currency_count"] > 0:
                account_summaries[account_type.value] = summary
        
        return {
            "total_currencies": total_currencies,
            "total_account_types": total_account_types,
            "account_summaries": account_summaries,
            "last_fetch": self.last_fetch.isoformat() if self.last_fetch else None,
            "cache_ttl_seconds": self.cache_ttl.total_seconds()
        }
    
    def cleanup_old_balances(self, max_age_hours: int = 24) -> int:
        """Clean up old balance data."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        for currency in list(self.balances.keys()):
            for account_type in list(self.balances[currency].keys()):
                balance = self.balances[currency][account_type]
                if balance.timestamp < cutoff_time:
                    del self.balances[currency][account_type]
                    cleaned_count += 1
            
            # Remove empty currency entries
            if not self.balances[currency]:
                del self.balances[currency]
        
        if cleaned_count > 0:
            self.logger.info(f"Cleaned up {cleaned_count} old balance entries")
        
        return cleaned_count
    
    def set_cache_ttl(self, ttl_seconds: int) -> None:
        """Set cache TTL in seconds."""
        self.cache_ttl = timedelta(seconds=ttl_seconds)
        self.logger.info(f"Set balance cache TTL to {ttl_seconds} seconds")
    
    def invalidate_cache(self, currency: Optional[str] = None, account_type: Optional[AccountType] = None) -> None:
        """Invalidate balance cache."""
        if currency and account_type:
            if currency in self.balances and account_type in self.balances[currency]:
                del self.balances[currency][account_type]
                self.logger.debug(f"Invalidated cache for {currency} {account_type.value}")
        elif currency:
            self.balances.pop(currency, None)
            self.logger.debug(f"Invalidated cache for {currency}")
        elif account_type:
            for currency_balances in self.balances.values():
                currency_balances.pop(account_type, None)
            self.logger.debug(f"Invalidated cache for {account_type.value}")
        else:
            self.balances.clear()
            self.logger.debug("Invalidated all balance cache")