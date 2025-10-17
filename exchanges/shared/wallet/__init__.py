"""
Wallet and Balance Management Utilities

Provides utilities for balance management, wallet operations,
and account type handling.
"""

from .balance_manager import BalanceManager

# Wallet management class
class WalletManager:
    """
    Comprehensive wallet management for trading accounts.
    
    Provides functionality for balance tracking, transaction history,
    multi-currency support, and wallet operations.
    """
    
    def __init__(self, account_id: str = None):
        """
        Initialize the WalletManager.
        
        Args:
            account_id: Unique account identifier
        """
        self.account_id = account_id or self._generate_account_id()
        self.balances = {}  # {currency: {'free': float, 'locked': float, 'total': float}}
        self.transaction_history = []
        self.wallet_operations = []
        self.fee_rates = {}  # {currency: fee_rate}
    
    def _generate_account_id(self) -> str:
        """Generate a unique account ID."""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def add_currency(self, currency: str, initial_balance: float = 0.0, fee_rate: float = 0.001):
        """
        Add a new currency to the wallet.
        
        Args:
            currency: Currency symbol (e.g., 'USDT', 'BTC')
            initial_balance: Initial balance for the currency
            fee_rate: Trading fee rate for the currency
        """
        self.balances[currency] = {
            'free': initial_balance,
            'locked': 0.0,
            'total': initial_balance
        }
        self.fee_rates[currency] = fee_rate
    
    def get_balance(self, currency: str) -> dict:
        """
        Get balance information for a currency.
        
        Args:
            currency: Currency symbol
            
        Returns:
            Dictionary with 'free', 'locked', and 'total' balances
        """
        return self.balances.get(currency, {'free': 0.0, 'locked': 0.0, 'total': 0.0})
    
    def get_all_balances(self) -> dict:
        """Get all currency balances."""
        return self.balances.copy()
    
    def deposit(self, currency: str, amount: float, transaction_id: str = None) -> bool:
        """
        Deposit funds to the wallet.
        
        Args:
            currency: Currency symbol
            amount: Amount to deposit
            transaction_id: Optional transaction ID
            
        Returns:
            True if successful
        """
        if currency not in self.balances:
            self.add_currency(currency)
        
        self.balances[currency]['free'] += amount
        self.balances[currency]['total'] += amount
        
        # Record transaction
        transaction = {
            'id': transaction_id or self._generate_transaction_id(),
            'type': 'deposit',
            'currency': currency,
            'amount': amount,
            'timestamp': self._get_timestamp(),
            'status': 'completed'
        }
        self.transaction_history.append(transaction)
        
        return True
    
    def withdraw(self, currency: str, amount: float, transaction_id: str = None) -> bool:
        """
        Withdraw funds from the wallet.
        
        Args:
            currency: Currency symbol
            amount: Amount to withdraw
            transaction_id: Optional transaction ID
            
        Returns:
            True if successful, False if insufficient funds
        """
        if currency not in self.balances:
            return False
        
        if self.balances[currency]['free'] < amount:
            return False
        
        self.balances[currency]['free'] -= amount
        self.balances[currency]['total'] -= amount
        
        # Record transaction
        transaction = {
            'id': transaction_id or self._generate_transaction_id(),
            'type': 'withdrawal',
            'currency': currency,
            'amount': amount,
            'timestamp': self._get_timestamp(),
            'status': 'completed'
        }
        self.transaction_history.append(transaction)
        
        return True
    
    def lock_funds(self, currency: str, amount: float, purpose: str = None) -> bool:
        """
        Lock funds for trading or other purposes.
        
        Args:
            currency: Currency symbol
            amount: Amount to lock
            purpose: Purpose of locking (e.g., 'order', 'margin')
            
        Returns:
            True if successful, False if insufficient free funds
        """
        if currency not in self.balances:
            return False
        
        if self.balances[currency]['free'] < amount:
            return False
        
        self.balances[currency]['free'] -= amount
        self.balances[currency]['locked'] += amount
        
        # Record wallet operation
        operation = {
            'type': 'lock',
            'currency': currency,
            'amount': amount,
            'purpose': purpose,
            'timestamp': self._get_timestamp()
        }
        self.wallet_operations.append(operation)
        
        return True
    
    def unlock_funds(self, currency: str, amount: float, purpose: str = None) -> bool:
        """
        Unlock previously locked funds.
        
        Args:
            currency: Currency symbol
            amount: Amount to unlock
            purpose: Purpose of unlocking
            
        Returns:
            True if successful, False if insufficient locked funds
        """
        if currency not in self.balances:
            return False
        
        if self.balances[currency]['locked'] < amount:
            return False
        
        self.balances[currency]['locked'] -= amount
        self.balances[currency]['free'] += amount
        
        # Record wallet operation
        operation = {
            'type': 'unlock',
            'currency': currency,
            'amount': amount,
            'purpose': purpose,
            'timestamp': self._get_timestamp()
        }
        self.wallet_operations.append(operation)
        
        return True
    
    def calculate_trading_fee(self, currency: str, amount: float) -> float:
        """
        Calculate trading fee for a transaction.
        
        Args:
            currency: Currency symbol
            amount: Transaction amount
            
        Returns:
            Calculated fee amount
        """
        fee_rate = self.fee_rates.get(currency, 0.001)
        return amount * fee_rate
    
    def get_transaction_history(self, currency: str = None, limit: int = 100) -> list:
        """
        Get transaction history.
        
        Args:
            currency: Filter by currency (optional)
            limit: Maximum number of transactions to return
            
        Returns:
            List of transactions
        """
        history = self.transaction_history.copy()
        
        if currency:
            history = [tx for tx in history if tx['currency'] == currency]
        
        return history[-limit:] if limit else history
    
    def get_wallet_summary(self) -> dict:
        """Get comprehensive wallet summary."""
        total_currencies = len(self.balances)
        total_balance_value = sum(balance['total'] for balance in self.balances.values())
        
        return {
            'account_id': self.account_id,
            'total_currencies': total_currencies,
            'total_balance_value': total_balance_value,
            'currencies': list(self.balances.keys()),
            'recent_transactions': len(self.transaction_history),
            'recent_operations': len(self.wallet_operations)
        }
    
    def transfer_between_currencies(self, from_currency: str, to_currency: str, 
                                  amount: float, exchange_rate: float) -> bool:
        """
        Transfer funds between currencies.
        
        Args:
            from_currency: Source currency
            to_currency: Target currency
            amount: Amount to transfer
            exchange_rate: Exchange rate from source to target currency
            
        Returns:
            True if successful
        """
        if from_currency not in self.balances or to_currency not in self.balances:
            return False
        
        if self.balances[from_currency]['free'] < amount:
            return False
        
        # Calculate converted amount
        converted_amount = amount * exchange_rate
        
        # Update balances
        self.balances[from_currency]['free'] -= amount
        self.balances[from_currency]['total'] -= amount
        
        self.balances[to_currency]['free'] += converted_amount
        self.balances[to_currency]['total'] += converted_amount
        
        # Record transaction
        transaction = {
            'id': self._generate_transaction_id(),
            'type': 'currency_transfer',
            'from_currency': from_currency,
            'to_currency': to_currency,
            'amount': amount,
            'converted_amount': converted_amount,
            'exchange_rate': exchange_rate,
            'timestamp': self._get_timestamp(),
            'status': 'completed'
        }
        self.transaction_history.append(transaction)
        
        return True
    
    def _generate_transaction_id(self) -> str:
        """Generate a unique transaction ID."""
        import uuid
        return f"tx_{str(uuid.uuid4())[:8]}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()

__all__ = [
    "BalanceManager",
    "WalletManager"
]