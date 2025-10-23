"""
Balance Management

Handles account balance tracking and management across exchanges.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum

from src.utils.logger import system_logger


class AccountType(Enum):
    """Account type enumeration."""
    SPOT = "spot"
    MARGIN = "margin"
    FUTURES = "futures"
    OPTIONS = "options"


@dataclass
class Balance:
    """Balance representation."""
    currency: str
    available: float
    frozen: float
    total: float
    account_type: AccountType = AccountType.SPOT
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.total == 0.0:
            self.total = self.available + self.frozen


class BalanceManager:
    """Manages account balances across exchanges."""
    
    def __init__(self, exchange_name: str):
        self.exchange_name = exchange_name
        self.logger = system_logger.getChild(f"BalanceManager.{exchange_name}")
        
        # Balance storage
        self.balances: Dict[str, Dict[str, Balance]] = {}  # account_type -> currency -> Balance
        self.balance_history: List[Dict[str, Any]] = []
        
        # Exchange-specific functions
        self.exchange_functions: Dict[str, Callable] = {}
        
        # Cache settings
        self.cache_duration = timedelta(seconds=30)
        self.last_update: Dict[str, datetime] = {}
        
        # Statistics
        self.total_balance_updates = 0
        self.failed_balance_updates = 0
    
    def register_fetch_functions(
        self,
        get_balances: Optional[Callable] = None,
        get_account_info: Optional[Callable] = None
    ) -> None:
        """Register exchange-specific fetch functions."""
        if get_balances:
            self.exchange_functions["get_balances"] = get_balances
        if get_account_info:
            self.exchange_functions["get_account_info"] = get_account_info
    
    async def get_balance(
        self,
        currency: str,
        account_type: str = "spot"
    ) -> Optional[float]:
        """Get balance for a specific currency."""
        try:
            # Check cache first
            if self._is_cache_valid(account_type):
                balance = self._get_cached_balance(currency, account_type)
                if balance is not None:
                    return balance
            
            # Fetch from exchange
            await self._refresh_balances(account_type)
            
            # Return cached balance
            return self._get_cached_balance(currency, account_type)
            
        except Exception as e:
            self.logger.error(f"Failed to get balance for {currency}: {e}")
            return None
    
    async def get_all_balances(
        self,
        account_type: str = "spot"
    ) -> Dict[str, float]:
        """Get all balances for an account type."""
        try:
            # Check cache first
            if self._is_cache_valid(account_type):
                return self._get_all_cached_balances(account_type)
            
            # Fetch from exchange
            await self._refresh_balances(account_type)
            
            # Return cached balances
            return self._get_all_cached_balances(account_type)
            
        except Exception as e:
            self.logger.error(f"Failed to get all balances for {account_type}: {e}")
            return {}
    
    async def get_balance_details(
        self,
        currency: str,
        account_type: str = "spot"
    ) -> Optional[Balance]:
        """Get detailed balance information."""
        try:
            # Check cache first
            if self._is_cache_valid(account_type):
                return self._get_cached_balance_details(currency, account_type)
            
            # Fetch from exchange
            await self._refresh_balances(account_type)
            
            # Return cached balance details
            return self._get_cached_balance_details(currency, account_type)
            
        except Exception as e:
            self.logger.error(f"Failed to get balance details for {currency}: {e}")
            return None
    
    async def get_all_balance_details(
        self,
        account_type: str = "spot"
    ) -> List[Balance]:
        """Get all detailed balance information."""
        try:
            # Check cache first
            if self._is_cache_valid(account_type):
                return self._get_all_cached_balance_details(account_type)
            
            # Fetch from exchange
            await self._refresh_balances(account_type)
            
            # Return cached balance details
            return self._get_all_cached_balance_details(account_type)
            
        except Exception as e:
            self.logger.error(f"Failed to get all balance details for {account_type}: {e}")
            return []
    
    def has_sufficient_balance(
        self,
        currency: str,
        amount: float,
        account_type: str = "spot"
    ) -> bool:
        """Check if account has sufficient balance."""
        try:
            balance = self._get_cached_balance_details(currency, account_type)
            if balance is None:
                return False
            
            return balance.available >= amount
            
        except Exception as e:
            self.logger.error(f"Failed to check sufficient balance for {currency}: {e}")
            return False
    
    def calculate_portfolio_value(
        self,
        prices: Dict[str, float],
        base_currency: str = "USDT",
        account_type: str = "spot"
    ) -> float:
        """Calculate total portfolio value."""
        try:
            total_value = 0.0
            balances = self._get_all_cached_balance_details(account_type)
            
            for balance in balances:
                if balance.currency == base_currency:
                    total_value += balance.total
                elif balance.currency in prices:
                    total_value += balance.total * prices[balance.currency]
            
            return total_value
            
        except Exception as e:
            self.logger.error(f"Failed to calculate portfolio value: {e}")
            return 0.0
    
    async def _refresh_balances(self, account_type: str) -> None:
        """Refresh balances from exchange."""
        try:
            if "get_balances" not in self.exchange_functions:
                self.logger.warning("No get_balances function registered")
                return
            
            # Fetch balances from exchange
            result = await self.exchange_functions["get_balances"](account_type)
            
            if result:
                # Update local balances
                self._update_balances_from_exchange(result, account_type)
                self.last_update[account_type] = datetime.now()
                self.total_balance_updates += 1
                
                self.logger.debug(f"Refreshed balances for {account_type}")
            else:
                self.failed_balance_updates += 1
                self.logger.warning(f"Failed to refresh balances for {account_type}")
                
        except Exception as e:
            self.failed_balance_updates += 1
            self.logger.error(f"Failed to refresh balances for {account_type}: {e}")
    
    def _update_balances_from_exchange(
        self,
        exchange_data: List[Dict[str, Any]],
        account_type: str
    ) -> None:
        """Update balances from exchange data."""
        try:
            if account_type not in self.balances:
                self.balances[account_type] = {}
            
            for balance_data in exchange_data:
                currency = balance_data.get("asset", balance_data.get("currency", ""))
                if not currency:
                    continue
                
                # Extract balance values
                available = float(balance_data.get("free", balance_data.get("available", 0)))
                frozen = float(balance_data.get("locked", balance_data.get("frozen", 0)))
                total = float(balance_data.get("total", available + frozen))
                
                # Create balance object
                balance = Balance(
                    currency=currency.upper(),
                    available=available,
                    frozen=frozen,
                    total=total,
                    account_type=AccountType(account_type.lower()),
                    updated_at=datetime.now()
                )
                
                # Store balance
                self.balances[account_type][currency.upper()] = balance
                
                # Record in history
                self._record_balance_history(balance)
                
        except Exception as e:
            self.logger.error(f"Failed to update balances from exchange: {e}")
    
    def _record_balance_history(self, balance: Balance) -> None:
        """Record balance in history."""
        try:
            history_entry = {
                "timestamp": balance.updated_at.isoformat(),
                "currency": balance.currency,
                "account_type": balance.account_type.value,
                "available": balance.available,
                "frozen": balance.frozen,
                "total": balance.total
            }
            
            self.balance_history.append(history_entry)
            
            # Keep only last 1000 entries
            if len(self.balance_history) > 1000:
                self.balance_history = self.balance_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Failed to record balance history: {e}")
    
    def _is_cache_valid(self, account_type: str) -> bool:
        """Check if cache is valid for account type."""
        if account_type not in self.last_update:
            return False
        
        return datetime.now() - self.last_update[account_type] < self.cache_duration
    
    def _get_cached_balance(self, currency: str, account_type: str) -> Optional[float]:
        """Get cached balance for currency."""
        if account_type not in self.balances:
            return None
        
        balance = self.balances[account_type].get(currency.upper())
        return balance.available if balance else None
    
    def _get_all_cached_balances(self, account_type: str) -> Dict[str, float]:
        """Get all cached balances."""
        if account_type not in self.balances:
            return {}
        
        return {
            currency: balance.available
            for currency, balance in self.balances[account_type].items()
        }
    
    def _get_cached_balance_details(self, currency: str, account_type: str) -> Optional[Balance]:
        """Get cached balance details for currency."""
        if account_type not in self.balances:
            return None
        
        return self.balances[account_type].get(currency.upper())
    
    def _get_all_cached_balance_details(self, account_type: str) -> List[Balance]:
        """Get all cached balance details."""
        if account_type not in self.balances:
            return []
        
        return list(self.balances[account_type].values())
    
    def get_balance_history(
        self,
        currency: Optional[str] = None,
        account_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get balance history."""
        try:
            history = self.balance_history.copy()
            
            # Filter by currency
            if currency:
                history = [h for h in history if h["currency"] == currency.upper()]
            
            # Filter by account type
            if account_type:
                history = [h for h in history if h["account_type"] == account_type.lower()]
            
            # Limit results
            return history[-limit:] if limit > 0 else history
            
        except Exception as e:
            self.logger.error(f"Failed to get balance history: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get balance manager statistics."""
        return {
            "total_balance_updates": self.total_balance_updates,
            "failed_balance_updates": self.failed_balance_updates,
            "cached_account_types": list(self.balances.keys()),
            "balance_history_entries": len(self.balance_history),
            "cache_duration_seconds": self.cache_duration.total_seconds()
        }
    
    def clear_cache(self, account_type: Optional[str] = None) -> None:
        """Clear balance cache."""
        if account_type:
            self.balances.pop(account_type, None)
            self.last_update.pop(account_type, None)
            self.logger.info(f"Cleared cache for {account_type}")
        else:
            self.balances.clear()
            self.last_update.clear()
            self.logger.info("Cleared all balance cache")
    
    def set_cache_duration(self, duration_seconds: int) -> None:
        """Set cache duration in seconds."""
        self.cache_duration = timedelta(seconds=duration_seconds)
        self.logger.info(f"Set cache duration to {duration_seconds} seconds")