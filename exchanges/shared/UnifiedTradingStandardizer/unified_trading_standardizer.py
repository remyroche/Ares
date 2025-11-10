"""
Unified Trading Standardizer

Main standardizer class for trading operations (orders, positions, balances, account info, trades).
Ensures complete equivalency between all exchanges for trading data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Import src/utils/data utilities
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.utils.logger import system_logger
from src.utils.tprint import tprint

from exchanges.exchange_types import ExchangeType
from exchanges.base_exchange.exchange_interface import OrderSide, OrderType, OrderStatus

from .standardized_order import StandardizedOrder
from .standardized_position import StandardizedPosition
from .standardized_balance import StandardizedBalance
from .standardized_account_info import StandardizedAccountInfo
from .standardized_trade import StandardizedTrade
from .field_mappings import (
    ORDER_FIELD_MAPPINGS,
    POSITION_FIELD_MAPPINGS,
    BALANCE_FIELD_MAPPINGS,
    TRADE_FIELD_MAPPINGS,
)
from .status_mappings import (
    normalize_order_status,
    normalize_order_type,
    normalize_order_side,
    normalize_position_side,
)

logger = logging.getLogger(__name__)


class DataQualityLevel(Enum):
    """Data quality validation levels"""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CRITICAL = "critical"


class UnifiedTradingStandardizer:
    """
    Unified trading data standardizer that ensures complete equivalency across all exchanges.

    This class provides a single interface for standardizing trading data from any exchange
    to a unified format that's fully compatible with trading utilities.
    """

    def __init__(
        self,
        quality_level: DataQualityLevel = DataQualityLevel.STANDARD,
        strict_mode: bool = True
    ):
        """
        Initialize the unified trading standardizer

        Args:
            quality_level: Quality validation level
            strict_mode: If True (default), raise exceptions on standardization errors.
                        If False, return invalid objects with validation errors.
        """
        tprint(f"UnifiedTradingStandardizer.__init__ called with quality_level={quality_level.value}, strict_mode={strict_mode}", "INFO")

        self.quality_level = quality_level
        self.strict_mode = strict_mode
        self.logger = system_logger.getChild("UnifiedTradingStandardizer")

        # Store mappings
        self.order_mappings = ORDER_FIELD_MAPPINGS
        self.position_mappings = POSITION_FIELD_MAPPINGS
        self.balance_mappings = BALANCE_FIELD_MAPPINGS
        self.trade_mappings = TRADE_FIELD_MAPPINGS
        tprint(f"Loaded field mappings for {len(self.order_mappings)} exchanges", "INFO")

        # Telemetry
        self.failure_counts = {
            'orders': 0,
            'positions': 0,
            'balances': 0,
            'trades': 0,
            'account_info': 0
        }

        # Validate mappings at startup
        self._validate_field_mappings()

        mode_str = "strict (exceptions on errors)" if strict_mode else "lenient (invalid objects on errors)"
        self.logger.info(f"✅ UnifiedTradingStandardizer initialized with {quality_level.value} quality level, {mode_str}")
        tprint(f"UnifiedTradingStandardizer initialized successfully with {quality_level.value} quality level, {mode_str}", "SUCCESS")
    
    # ============================================================================
    # ORDER STANDARDIZATION
    # ============================================================================
    
    def standardize_order(
        self,
        raw_order: Dict[str, Any],
        exchange: ExchangeType,
        symbol: Optional[str] = None
    ) -> StandardizedOrder:
        """
        Standardize order response from exchange to unified format.

        Args:
            raw_order: Raw order data from exchange
            exchange: Exchange source
            symbol: Trading symbol (optional, will try to extract from raw_order)

        Returns:
            StandardizedOrder object
        """
        tprint(f"standardize_order called for exchange={exchange.value}, symbol={symbol}", "INFO")
        try:
            mapping = self.order_mappings.get(exchange)
            if not mapping:
                raise ValueError(f"No field mapping found for exchange: {exchange}")
            
            # Extract fields using mapping
            extracted = self._extract_fields(raw_order, mapping)
            
            # Get symbol
            if not symbol:
                symbol = extracted.get('symbol') or raw_order.get('symbol', '')
            if not symbol:
                raise ValueError("Symbol is required and could not be extracted from raw_order")
            
            # Normalize enum fields
            side = self._normalize_value(
                extracted.get('side'),
                normalize_order_side,
                exchange,
                default=OrderSide.BUY
            )
            order_type = self._normalize_value(
                extracted.get('order_type'),
                normalize_order_type,
                exchange,
                default=OrderType.MARKET
            )
            status = self._normalize_value(
                extracted.get('status'),
                normalize_order_status,
                exchange,
                default=OrderStatus.PENDING
            )
            
            # Extract quantities and prices
            quantity = float(extracted.get('quantity', 0))
            executed_quantity = float(extracted.get('executed_quantity', 0))
            remaining_quantity = extracted.get('remaining_quantity')
            if remaining_quantity is None:
                remaining_quantity = quantity - executed_quantity
            else:
                remaining_quantity = float(remaining_quantity)
            
            price = self._safe_float(extracted.get('price'))
            stop_price = self._safe_float(extracted.get('stop_price'))
            executed_price_avg = self._safe_float(extracted.get('executed_price_avg'))
            
            # Extract timestamps
            timestamp = self._convert_timestamp(
                extracted.get('timestamp'),
                exchange,
                default=datetime.now(timezone.utc)
            )
            update_time = self._convert_timestamp(
                extracted.get('update_time'),
                exchange,
                default=timestamp
            )
            
            # Create standardized order
            order = StandardizedOrder(
                order_id=str(extracted.get('order_id', raw_order.get('orderId', ''))),
                symbol=str(symbol),
                exchange=exchange.value,
                side=side,
                order_type=order_type,
                status=status,
                quantity=quantity,
                price=price,
                executed_quantity=executed_quantity,
                remaining_quantity=remaining_quantity,
                executed_price_avg=executed_price_avg,
                stop_price=stop_price,
                time_in_force=extracted.get('time_in_force'),
                fee=self._safe_float(extracted.get('fee')),
                fee_currency=extracted.get('fee_currency'),
                client_order_id=extracted.get('client_order_id'),
                exchange_order_id=extracted.get('exchange_order_id'),
                timestamp=timestamp,
                update_time=update_time,
                raw_order_data=raw_order,
                source_exchange_type=exchange.value,
            )
            
            # Apply quality processing
            if self.quality_level != DataQualityLevel.BASIC:
                order = self._apply_order_quality_processing(order)

            tprint(f"Order standardization successful: order_id={order.order_id}, symbol={order.symbol}, status={order.status}", "SUCCESS")
            return order

        except Exception as e:
            self.failure_counts['orders'] += 1
            error_msg = f"Failed to standardize order from {exchange.value}: {e}"
            self.logger.error(error_msg)
            tprint(f"Order standardization failed for {exchange.value}: {e}", "ERROR")

            # In strict mode, re-raise the exception
            if self.strict_mode:
                raise ValueError(error_msg) from e

            # In lenient mode, return invalid order for tracking
            return StandardizedOrder(
                order_id=str(raw_order.get('orderId', raw_order.get('order_id', 'unknown'))),
                symbol=symbol or 'UNKNOWN',
                exchange=exchange.value,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                status=OrderStatus.REJECTED,
                quantity=0.0,
                timestamp=datetime.now(timezone.utc),
                update_time=datetime.now(timezone.utc),
                is_valid=False,
                validation_errors=[str(e)],
            )
    
    def standardize_orders(
        self,
        raw_orders: Union[List[Dict], List[List], pd.DataFrame],
        exchange: ExchangeType,
        symbol: Optional[str] = None
    ) -> List[StandardizedOrder]:
        """Standardize multiple orders"""
        tprint(f"standardize_orders called for exchange={exchange.value}, count={len(raw_orders) if hasattr(raw_orders, '__len__') else 'unknown'}", "INFO")

        if isinstance(raw_orders, pd.DataFrame):
            raw_orders = raw_orders.to_dict('records')

        standardized = []
        for raw_order in raw_orders:
            try:
                order = self.standardize_order(raw_order, exchange, symbol)
                standardized.append(order)
            except Exception as e:
                self.logger.warning(f"Failed to standardize order: {e}")
                tprint(f"Skipped order due to standardization error: {e}", "WARNING")
                continue

        tprint(f"Standardized {len(standardized)} orders from {exchange.value}", "SUCCESS")
        return standardized
    
    def standardize_orders_to_dataframe(
        self,
        raw_orders: Union[List[Dict], List[List], pd.DataFrame],
        exchange: ExchangeType,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """Standardize orders to DataFrame format"""
        standardized_orders = self.standardize_orders(raw_orders, exchange, symbol)
        
        if not standardized_orders:
            return pd.DataFrame()
        
        data = [order.to_dataframe_row() for order in standardized_orders]
        df = pd.DataFrame(data)
        
        # Set timestamp as index if available
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        return df
    
    # ============================================================================
    # POSITION STANDARDIZATION
    # ============================================================================
    
    def standardize_position(
        self,
        raw_position: Dict[str, Any],
        exchange: ExchangeType,
        symbol: Optional[str] = None
    ) -> StandardizedPosition:
        """Standardize position response from exchange"""
        tprint(f"standardize_position called for exchange={exchange.value}, symbol={symbol}", "INFO")
        try:
            mapping = self.position_mappings.get(exchange)
            if not mapping:
                raise ValueError(f"No field mapping found for exchange: {exchange}")
            
            # Extract fields
            extracted = self._extract_fields(raw_position, mapping)
            
            # Get symbol
            if not symbol:
                symbol = extracted.get('symbol') or raw_position.get('symbol', '')
            if not symbol:
                raise ValueError("Symbol is required and could not be extracted from raw_position")
            
            # Normalize position side
            side = normalize_position_side(
                extracted.get('side', 'neutral'),
                exchange
            )
            
            # Extract numeric fields
            size = abs(float(extracted.get('size', 0)))  # Always positive
            entry_price = float(extracted.get('entry_price', 0))
            mark_price = self._safe_float(extracted.get('mark_price'))
            liquidation_price = self._safe_float(extracted.get('liquidation_price'))
            unrealized_pnl = float(extracted.get('unrealized_pnl', 0))
            realized_pnl = float(extracted.get('realized_pnl', 0))
            leverage = self._safe_float(extracted.get('leverage'))
            margin = self._safe_float(extracted.get('margin'))
            isolated_margin = self._safe_float(extracted.get('isolated_margin'))
            
            # Extract timestamps
            timestamp = self._convert_timestamp(
                extracted.get('timestamp'),
                exchange,
                default=datetime.now(timezone.utc)
            )
            
            # Create standardized position
            position = StandardizedPosition(
                symbol=str(symbol),
                exchange=exchange.value,
                side=side,
                size=size,
                entry_price=entry_price,
                mark_price=mark_price,
                liquidation_price=liquidation_price,
                unrealized_pnl=unrealized_pnl,
                realized_pnl=realized_pnl,
                leverage=leverage,
                margin=margin,
                isolated_margin=isolated_margin,
                position_value=extracted.get('position_value'),
                margin_mode=extracted.get('margin_mode'),
                position_mode=extracted.get('position_mode'),
                exchange_position_id=extracted.get('exchange_position_id'),
                timestamp=timestamp,
                update_time=timestamp,
                raw_position_data=raw_position,
                source_exchange_type=exchange.value,
            )
            
            # Apply quality processing
            if self.quality_level != DataQualityLevel.BASIC:
                position = self._apply_position_quality_processing(position)

            tprint(f"Position standardization successful: symbol={position.symbol}, side={position.side}, size={position.size}", "SUCCESS")
            return position

        except Exception as e:
            self.failure_counts['positions'] += 1
            error_msg = f"Failed to standardize position from {exchange.value}: {e}"
            self.logger.error(error_msg)
            tprint(f"Position standardization failed for {exchange.value}: {e}", "ERROR")

            # In strict mode, re-raise the exception
            if self.strict_mode:
                raise ValueError(error_msg) from e

            # In lenient mode, return invalid position for tracking
            return StandardizedPosition(
                symbol=symbol or 'UNKNOWN',
                exchange=exchange.value,
                side='neutral',
                size=0.0,
                entry_price=0.0,
                timestamp=datetime.now(timezone.utc),
                update_time=datetime.now(timezone.utc),
                is_valid=False,
                validation_errors=[str(e)],
            )
    
    def standardize_positions(
        self,
        raw_positions: Union[List[Dict], Dict[str, Dict]],
        exchange: ExchangeType,
        symbol: Optional[str] = None
    ) -> List[StandardizedPosition]:
        """Standardize multiple positions"""
        tprint(f"standardize_positions called for exchange={exchange.value}", "INFO")

        # Handle dict of positions (keyed by symbol)
        if isinstance(raw_positions, dict):
            positions_list = []
            for pos_symbol, pos_data in raw_positions.items():
                if isinstance(pos_data, dict):
                    positions_list.append(pos_data)
                else:
                    positions_list.append({'symbol': pos_symbol, **pos_data})
            raw_positions = positions_list

        standardized = []
        for raw_position in raw_positions:
            try:
                position = self.standardize_position(raw_position, exchange, symbol)
                standardized.append(position)
            except Exception as e:
                self.logger.warning(f"Failed to standardize position: {e}")
                tprint(f"Skipped position due to standardization error: {e}", "WARNING")
                continue

        tprint(f"Standardized {len(standardized)} positions from {exchange.value}", "SUCCESS")
        return standardized
    
    def standardize_positions_to_dataframe(
        self,
        raw_positions: Union[List[Dict], Dict[str, Dict]],
        exchange: ExchangeType,
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """Standardize positions to DataFrame format"""
        standardized_positions = self.standardize_positions(raw_positions, exchange, symbol)
        
        if not standardized_positions:
            return pd.DataFrame()
        
        data = [pos.to_dataframe_row() for pos in standardized_positions]
        df = pd.DataFrame(data)
        
        return df
    
    # ============================================================================
    # BALANCE STANDARDIZATION
    # ============================================================================
    
    def standardize_balance(
        self,
        raw_balance: Dict[str, Any],
        exchange: ExchangeType,
        currency: str
    ) -> StandardizedBalance:
        """Standardize balance response from exchange"""
        tprint(f"standardize_balance called for exchange={exchange.value}, currency={currency}", "INFO")
        try:
            mapping = self.balance_mappings.get(exchange)
            if not mapping:
                raise ValueError(f"No field mapping found for exchange: {exchange}")
            
            # Extract fields
            extracted = self._extract_fields(raw_balance, mapping)
            
            # Get currency (prioritize provided, then extracted)
            currency = currency or extracted.get('currency') or raw_balance.get('currency', '')
            if not currency:
                raise ValueError("Currency is required")
            
            # Extract balance values
            free = float(extracted.get('free', 0))
            used = float(extracted.get('used', 0))
            total = extracted.get('total')
            if total is None:
                total = free + used
            else:
                total = float(total)
            
            available_balance = self._safe_float(extracted.get('available_balance'))
            frozen_balance = self._safe_float(extracted.get('frozen_balance'))
            
            # Extract timestamp
            timestamp = self._convert_timestamp(
                extracted.get('timestamp'),
                exchange,
                default=datetime.now(timezone.utc)
            )
            
            # Create standardized balance
            balance = StandardizedBalance(
                currency=str(currency).upper(),
                exchange=exchange.value,
                free=free,
                used=used,
                total=total,
                available_balance=available_balance,
                frozen_balance=frozen_balance,
                account_type=extracted.get('account_type'),
                timestamp=timestamp,
                raw_balance_data=raw_balance,
                source_exchange_type=exchange.value,
            )
            
            # Apply quality processing
            if self.quality_level != DataQualityLevel.BASIC:
                balance = self._apply_balance_quality_processing(balance)

            tprint(f"Balance standardization successful: currency={balance.currency}, total={balance.total}", "SUCCESS")
            return balance

        except Exception as e:
            self.failure_counts['balances'] += 1
            error_msg = f"Failed to standardize balance from {exchange.value}: {e}"
            self.logger.error(error_msg)
            tprint(f"Balance standardization failed for {exchange.value}, currency={currency}: {e}", "ERROR")

            # In strict mode, re-raise the exception
            if self.strict_mode:
                raise ValueError(error_msg) from e

            # In lenient mode, return invalid balance for tracking
            return StandardizedBalance(
                currency=str(currency).upper(),
                exchange=exchange.value,
                free=0.0,
                used=0.0,
                total=0.0,
                timestamp=datetime.now(timezone.utc),
                is_valid=False,
                validation_errors=[str(e)],
            )
    
    def standardize_balances(
        self,
        raw_balances: Union[List[Dict], Dict[str, Dict]],
        exchange: ExchangeType
    ) -> List[StandardizedBalance]:
        """Standardize all balances from account"""
        tprint(f"standardize_balances called for exchange={exchange.value}", "INFO")

        # Handle dict of balances (keyed by currency)
        if isinstance(raw_balances, dict):
            balances_list = []
            for currency, balance_data in raw_balances.items():
                if isinstance(balance_data, dict):
                    balance_data['currency'] = balance_data.get('currency', currency)
                    balances_list.append(balance_data)
                else:
                    balances_list.append({'currency': currency})
            raw_balances = balances_list

        standardized = []
        for raw_balance in raw_balances:
            try:
                currency = raw_balance.get('currency', '')
                balance = self.standardize_balance(raw_balance, exchange, currency)
                standardized.append(balance)
            except Exception as e:
                self.logger.warning(f"Failed to standardize balance: {e}")
                tprint(f"Skipped balance due to standardization error: {e}", "WARNING")
                continue

        tprint(f"Standardized {len(standardized)} balances from {exchange.value}", "SUCCESS")
        return standardized
    
    def standardize_balances_to_dataframe(
        self,
        raw_balances: Union[List[Dict], Dict[str, Dict]],
        exchange: ExchangeType
    ) -> pd.DataFrame:
        """Standardize balances to DataFrame format"""
        standardized_balances = self.standardize_balances(raw_balances, exchange)
        
        if not standardized_balances:
            return pd.DataFrame()
        
        data = [balance.to_dataframe_row() for balance in standardized_balances]
        df = pd.DataFrame(data)
        
        return df
    
    # ============================================================================
    # ACCOUNT INFO STANDARDIZATION
    # ============================================================================
    
    def standardize_account_info(
        self,
        raw_account: Dict[str, Any],
        exchange: ExchangeType
    ) -> StandardizedAccountInfo:
        """Standardize account information response"""
        tprint(f"standardize_account_info called for exchange={exchange.value}", "INFO")
        try:
            # Extract account info fields
            account_type = raw_account.get('accountType') or raw_account.get('account_type', 'SPOT')
            can_trade = raw_account.get('canTrade', raw_account.get('can_trade', True))
            can_withdraw = raw_account.get('canWithdraw', raw_account.get('can_withdraw', True))
            can_deposit = raw_account.get('canDeposit', raw_account.get('can_deposit', True))
            
            # Extract permissions
            permissions = raw_account.get('permissions', raw_account.get('permission', []))
            if isinstance(permissions, str):
                permissions = permissions.split(',')
            
            # Extract balances if present
            balances = []
            raw_balances = raw_account.get('balances', raw_account.get('balance', []))
            if raw_balances:
                balances = self.standardize_balances(raw_balances, exchange)
            
            # Extract margin info
            total_equity = self._safe_float(raw_account.get('totalEquity', raw_account.get('total_equity')))
            available_margin = self._safe_float(raw_account.get('availableMargin', raw_account.get('available_margin')))
            used_margin = self._safe_float(raw_account.get('usedMargin', raw_account.get('used_margin')))
            margin_ratio = self._safe_float(raw_account.get('marginRatio', raw_account.get('margin_ratio')))
            
            # Extract timestamp
            timestamp = self._convert_timestamp(
                raw_account.get('updateTime', raw_account.get('update_time')),
                exchange,
                default=datetime.now(timezone.utc)
            )
            
            # Create standardized account info
            account_info = StandardizedAccountInfo(
                exchange=exchange.value,
                account_type=str(account_type).upper(),
                can_trade=bool(can_trade),
                can_withdraw=bool(can_withdraw),
                can_deposit=bool(can_deposit),
                permissions=list(permissions),
                balances=balances,
                total_equity=total_equity,
                available_margin=available_margin,
                used_margin=used_margin,
                margin_ratio=margin_ratio,
                timestamp=timestamp,
                raw_account_data=raw_account,
                source_exchange_type=exchange.value,
            )
            
            # Apply quality processing
            if self.quality_level != DataQualityLevel.BASIC:
                account_info = self._apply_account_quality_processing(account_info)

            tprint(f"Account info standardization successful: exchange={account_info.exchange}, account_type={account_info.account_type}, balances={len(account_info.balances)}", "SUCCESS")
            return account_info

        except Exception as e:
            self.logger.error(f"Failed to standardize account info from {exchange.value}: {e}")
            tprint(f"Account info standardization failed for {exchange.value}: {e}", "ERROR")
            return StandardizedAccountInfo(
                exchange=exchange.value,
                account_type='SPOT',
                can_trade=False,
                can_withdraw=False,
                can_deposit=False,
                timestamp=datetime.now(timezone.utc),
                is_valid=False,
                validation_errors=[str(e)],
            )
    
    # ============================================================================
    # TRADE STANDARDIZATION
    # ============================================================================
    
    def standardize_trade(
        self,
        raw_trade: Dict[str, Any],
        exchange: ExchangeType,
        symbol: str,
        order_id: Optional[str] = None
    ) -> StandardizedTrade:
        """Standardize trade/execution response"""
        tprint(f"standardize_trade called for exchange={exchange.value}, symbol={symbol}, order_id={order_id}", "INFO")
        try:
            mapping = self.trade_mappings.get(exchange)
            if not mapping:
                raise ValueError(f"No field mapping found for exchange: {exchange}")
            
            # Extract fields
            extracted = self._extract_fields(raw_trade, mapping)
            
            # Get order_id
            if not order_id:
                order_id = extracted.get('order_id') or raw_trade.get('orderId', '')
            
            # Normalize side
            side = self._normalize_value(
                extracted.get('side'),
                normalize_order_side,
                exchange,
                default=OrderSide.BUY
            )
            
            # Extract numeric fields
            price = float(extracted.get('price', 0))
            quantity = float(extracted.get('quantity', 0))
            fee = float(extracted.get('fee', 0))
            fee_currency = extracted.get('fee_currency') or 'USDT'
            
            # Extract flags
            is_maker = extracted.get('is_maker')
            if isinstance(is_maker, str):
                is_maker = is_maker.lower() in ['true', '1', 'yes', 'maker']
            is_buyer = extracted.get('is_buyer')
            if isinstance(is_buyer, str):
                is_buyer = is_buyer.lower() in ['true', '1', 'yes']
            
            # Extract timestamp
            timestamp = self._convert_timestamp(
                extracted.get('timestamp'),
                exchange,
                default=datetime.now(timezone.utc)
            )
            
            # Create standardized trade
            trade = StandardizedTrade(
                trade_id=str(extracted.get('trade_id', raw_trade.get('id', ''))),
                order_id=str(order_id),
                symbol=str(symbol),
                exchange=exchange.value,
                side=side,
                price=price,
                quantity=quantity,
                fee=fee,
                fee_currency=str(fee_currency),
                is_maker=is_maker,
                is_buyer=is_buyer,
                trade_type=extracted.get('trade_type'),
                timestamp=timestamp,
                raw_trade_data=raw_trade,
                source_exchange_type=exchange.value,
            )
            
            # Apply quality processing
            if self.quality_level != DataQualityLevel.BASIC:
                trade = self._apply_trade_quality_processing(trade)

            tprint(f"Trade standardization successful: trade_id={trade.trade_id}, symbol={trade.symbol}, quantity={trade.quantity}@{trade.price}", "SUCCESS")
            return trade

        except Exception as e:
            self.failure_counts['trades'] += 1
            error_msg = f"Failed to standardize trade from {exchange.value}: {e}"
            self.logger.error(error_msg)
            tprint(f"Trade standardization failed for {exchange.value}: {e}", "ERROR")

            # In strict mode, re-raise the exception
            if self.strict_mode:
                raise ValueError(error_msg) from e

            # In lenient mode, return invalid trade for tracking
            invalid_trade = StandardizedTrade(
                trade_id=str(raw_trade.get('id', 'unknown')),
                order_id=str(order_id or 'unknown'),
                symbol=str(symbol),
                exchange=exchange.value,
                side=OrderSide.BUY,
                price=0.0,
                quantity=0.0,
                fee=0.0,
                fee_currency='USDT',
                timestamp=datetime.now(timezone.utc),
                is_valid=False,
                validation_errors=[str(e)],
            )
            return invalid_trade
    
    def standardize_trades(
        self,
        raw_trades: Union[List[Dict], pd.DataFrame],
        exchange: ExchangeType,
        symbol: str
    ) -> List[StandardizedTrade]:
        """Standardize multiple trades"""
        tprint(f"standardize_trades called for exchange={exchange.value}, symbol={symbol}, count={len(raw_trades) if hasattr(raw_trades, '__len__') else 'unknown'}", "INFO")

        if isinstance(raw_trades, pd.DataFrame):
            raw_trades = raw_trades.to_dict('records')

        standardized = []
        for raw_trade in raw_trades:
            try:
                trade = self.standardize_trade(raw_trade, exchange, symbol)
                standardized.append(trade)
            except Exception as e:
                self.logger.warning(f"Failed to standardize trade: {e}")
                tprint(f"Skipped trade due to standardization error: {e}", "WARNING")
                continue

        tprint(f"Standardized {len(standardized)} trades from {exchange.value}", "SUCCESS")
        return standardized
    
    def standardize_trades_to_dataframe(
        self,
        raw_trades: Union[List[Dict], pd.DataFrame],
        exchange: ExchangeType,
        symbol: str
    ) -> pd.DataFrame:
        """Standardize trades to DataFrame format"""
        standardized_trades = self.standardize_trades(raw_trades, exchange, symbol)
        
        if not standardized_trades:
            return pd.DataFrame()
        
        data = [trade.to_dataframe_row() for trade in standardized_trades]
        df = pd.DataFrame(data)
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        return df
    
    # ============================================================================
    # INTEGRATION METHODS
    # ============================================================================
    
    def standardize_dispatcher_response(
        self,
        response_type: str,
        raw_response: Any,
        exchange: ExchangeType,
        **kwargs
    ) -> Union[StandardizedOrder, StandardizedPosition, StandardizedBalance, 
               StandardizedAccountInfo, StandardizedTrade, List]:
        """
        Convenience method that works with ExchangeDispatcher responses.
        
        Args:
            response_type: Type of response ('order', 'position', 'balance', 'account', 'trade')
            raw_response: Raw response from dispatcher
            exchange: Exchange type
            **kwargs: Additional arguments (symbol, currency, order_id, etc.)
            
        Returns:
            Standardized data object(s)
        """
        response_type_lower = response_type.lower()
        
        if response_type_lower == 'order':
            if isinstance(raw_response, list):
                return self.standardize_orders(raw_response, exchange, kwargs.get('symbol'))
            else:
                return self.standardize_order(raw_response, exchange, kwargs.get('symbol'))
        
        elif response_type_lower == 'position':
            if isinstance(raw_response, list) or isinstance(raw_response, dict):
                return self.standardize_positions(raw_response, exchange, kwargs.get('symbol'))
            else:
                return self.standardize_position(raw_response, exchange, kwargs.get('symbol'))
        
        elif response_type_lower == 'balance':
            if isinstance(raw_response, list) or isinstance(raw_response, dict):
                return self.standardize_balances(raw_response, exchange)
            else:
                return self.standardize_balance(raw_response, exchange, kwargs.get('currency', ''))
        
        elif response_type_lower in ['account', 'account_info']:
            return self.standardize_account_info(raw_response, exchange)
        
        elif response_type_lower == 'trade':
            if isinstance(raw_response, list):
                return self.standardize_trades(raw_response, exchange, kwargs.get('symbol', ''))
            else:
                return self.standardize_trade(
                    raw_response, 
                    exchange, 
                    kwargs.get('symbol', ''),
                    kwargs.get('order_id')
                )
        
        else:
            raise ValueError(f"Unknown response type: {response_type}")
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _extract_fields(self, data: Dict[str, Any], mapping: Dict[str, List[str]]) -> Dict[str, Any]:
        """Extract fields from data using mapping"""
        extracted = {}
        
        for target_field, source_fields in mapping.items():
            for source_field in source_fields:
                # Try exact match
                if source_field in data:
                    extracted[target_field] = data[source_field]
                    break
                
                # Try case-insensitive match
                for key in data.keys():
                    if key.lower() == source_field.lower():
                        extracted[target_field] = data[key]
                        break
            
            # Try nested access (e.g., 'order.orderId')
            if target_field not in extracted:
                for source_field in source_fields:
                    parts = source_field.split('.')
                    value = data
                    try:
                        for part in parts:
                            value = value[part]
                        extracted[target_field] = value
                        break
                    except (KeyError, TypeError):
                        continue
        
        return extracted
    
    def _normalize_value(self, value: Any, normalizer_func, exchange: ExchangeType, default: Any = None) -> Any:
        """Normalize a value using normalization function"""
        if value is None:
            return default
        
        try:
            return normalizer_func(str(value), exchange)
        except Exception as e:
            self.logger.warning(f"Failed to normalize value {value}: {e}, using default")
            return default
    
    def _convert_timestamp(self, timestamp: Any, exchange: ExchangeType, default: Optional[datetime] = None) -> datetime:
        """Convert timestamp to datetime"""
        if timestamp is None:
            return default or datetime.now(timezone.utc)
        
        try:
            # Handle different timestamp formats
            if isinstance(timestamp, datetime):
                return timestamp
            
            if isinstance(timestamp, str):
                # Try ISO format
                try:
                    return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except:
                    pass
            
            # Try as number (milliseconds or seconds)
            ts_num = float(timestamp)
            
            # Determine if milliseconds or seconds (typically > 1e10 = seconds, < 1e10 = ms)
            if ts_num > 1e10:
                # Seconds
                return datetime.fromtimestamp(ts_num, tz=timezone.utc)
            else:
                # Milliseconds
                return datetime.fromtimestamp(ts_num / 1000.0, tz=timezone.utc)
        
        except Exception as e:
            self.logger.warning(f"Failed to convert timestamp {timestamp}: {e}")
            return default or datetime.now(timezone.utc)
    
    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float"""
        if value is None:
            return None
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    # ============================================================================
    # QUALITY PROCESSING
    # ============================================================================
    
    def _apply_order_quality_processing(self, order: StandardizedOrder) -> StandardizedOrder:
        """Apply quality processing to order"""
        if self.quality_level in [DataQualityLevel.STRICT, DataQualityLevel.CRITICAL]:
            if order.quality_score < 80.0:
                order.is_valid = False
                order.validation_errors.append(f"Quality score {order.quality_score} below threshold")
        return order
    
    def _apply_position_quality_processing(self, position: StandardizedPosition) -> StandardizedPosition:
        """Apply quality processing to position"""
        if self.quality_level in [DataQualityLevel.STRICT, DataQualityLevel.CRITICAL]:
            if position.quality_score < 80.0:
                position.is_valid = False
                position.validation_errors.append(f"Quality score {position.quality_score} below threshold")
        return position
    
    def _apply_balance_quality_processing(self, balance: StandardizedBalance) -> StandardizedBalance:
        """Apply quality processing to balance"""
        if self.quality_level in [DataQualityLevel.STRICT, DataQualityLevel.CRITICAL]:
            if balance.quality_score < 80.0:
                balance.is_valid = False
                balance.validation_errors.append(f"Quality score {balance.quality_score} below threshold")
        return balance
    
    def _apply_account_quality_processing(self, account: StandardizedAccountInfo) -> StandardizedAccountInfo:
        """Apply quality processing to account info"""
        if self.quality_level in [DataQualityLevel.STRICT, DataQualityLevel.CRITICAL]:
            if account.quality_score < 80.0:
                account.is_valid = False
                account.validation_errors.append(f"Quality score {account.quality_score} below threshold")
        return account
    
    def _apply_trade_quality_processing(self, trade: StandardizedTrade) -> StandardizedTrade:
        """Apply quality processing to trade"""
        if self.quality_level in [DataQualityLevel.STRICT, DataQualityLevel.CRITICAL]:
            if trade.quality_score < 80.0:
                trade.is_valid = False
                trade.validation_errors.append(f"Quality score {trade.quality_score} below threshold")
        return trade
    
    # ============================================================================
    # VALIDATION METHODS
    # ============================================================================
    
    def validate_trading_data_consistency(
        self,
        orders: List[StandardizedOrder],
        positions: List[StandardizedPosition],
        balances: List[StandardizedBalance]
    ) -> Dict[str, Any]:
        """
        Validate consistency across orders, positions, and balances.
        
        Returns:
            Dictionary with validation results
        """
        errors = []
        warnings = []
        
        # Check order-position consistency
        filled_orders = [o for o in orders if o.status == OrderStatus.FILLED]
        if filled_orders and positions:
            # Could implement logic to verify order fills match position changes
            pass
        
        # Check balance consistency
        if balances:
            total_check = all(b.total == (b.free + b.used) for b in balances)
            if not total_check:
                warnings.append("Some balances have inconsistent total (free + used)")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
        }

    def _validate_field_mappings(self) -> None:
        """Validate that all exchange type mappings are configured correctly"""
        tprint("Validating field mappings for all exchanges", "INFO")

        required_exchanges = [
            ExchangeType.BINANCE,
            ExchangeType.OKX,
            ExchangeType.BINGX,
            ExchangeType.MEXC,
            ExchangeType.GATEIO,
            ExchangeType.PHEMEX,
        ]

        missing_mappings = []

        for exchange in required_exchanges:
            if exchange not in self.order_mappings:
                missing_mappings.append(f"Order mapping for {exchange.value}")
            if exchange not in self.position_mappings:
                missing_mappings.append(f"Position mapping for {exchange.value}")
            if exchange not in self.balance_mappings:
                missing_mappings.append(f"Balance mapping for {exchange.value}")
            if exchange not in self.trade_mappings:
                missing_mappings.append(f"Trade mapping for {exchange.value}")

        if missing_mappings:
            error_msg = f"Missing field mappings: {', '.join(missing_mappings)}"
            self.logger.error(f"❌ {error_msg}")
            tprint(f"Field mapping validation failed: {error_msg}", "ERROR")
            if self.strict_mode:
                raise ValueError(error_msg)
            else:
                self.logger.warning(f"⚠️ Continuing in lenient mode despite missing mappings")
                tprint("Continuing in lenient mode despite missing mappings", "WARNING")
        else:
            self.logger.info(f"✅ Field mappings validated for {len(required_exchanges)} exchanges")
            tprint(f"Field mappings validated successfully for {len(required_exchanges)} exchanges", "SUCCESS")

    def get_telemetry(self) -> Dict[str, Any]:
        """Get standardization telemetry and failure statistics"""
        tprint("Getting standardization telemetry", "INFO")
        total_failures = sum(self.failure_counts.values())

        telemetry = {
            'total_failures': total_failures,
            'failure_counts': dict(self.failure_counts),
            'quality_level': self.quality_level.value,
            'strict_mode': self.strict_mode,
            'failure_rate_by_type': {
                data_type: count / max(count, 1) * 100
                for data_type, count in self.failure_counts.items()
            }
        }

        tprint(f"Telemetry retrieved: total_failures={total_failures}, quality_level={self.quality_level.value}", "SUCCESS")
        return telemetry

    def reset_telemetry(self) -> None:
        """Reset telemetry counters"""
        tprint("Resetting telemetry counters", "INFO")

        self.failure_counts = {
            'orders': 0,
            'positions': 0,
            'balances': 0,
            'trades': 0,
            'account_info': 0
        }

        self.logger.info("Telemetry counters reset")
        tprint("Telemetry counters reset successfully", "SUCCESS")


# Global instance for easy access (with strict mode enabled by default)
unified_trading_standardizer = UnifiedTradingStandardizer(strict_mode=True)