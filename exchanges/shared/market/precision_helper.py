"""
Precision and Rounding Utilities

Handles price and quantity precision, rounding, and validation.
"""

import decimal
from decimal import Decimal, ROUND_DOWN, ROUND_UP, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

from src.utils.logger import system_logger


@dataclass
class PrecisionConfig:
    """Precision configuration for a symbol"""
    symbol: str
    price_precision: int
    quantity_precision: int
    tick_size: float
    lot_size: float
    min_notional: float
    max_notional: Optional[float] = None


class PrecisionHelper:
    """
    Handles precision, rounding, and validation for prices and quantities.
    """
    
    def __init__(self):
        self.logger = system_logger.getChild("PrecisionHelper")
        self.precision_configs: Dict[str, PrecisionConfig] = {}
        
        # Set high precision for decimal calculations
        decimal.getcontext().prec = 28
    
    def set_precision_config(self, config: PrecisionConfig) -> None:
        """Set precision configuration for a symbol."""
        self.precision_configs[config.symbol] = config
        self.logger.debug(f"Set precision config for {config.symbol}")
    
    def get_precision_config(self, symbol: str) -> Optional[PrecisionConfig]:
        """Get precision configuration for a symbol."""
        return self.precision_configs.get(symbol)
    
    def round_price(self, price: Union[float, str, Decimal], symbol: str) -> float:
        """
        Round price to appropriate precision for symbol.
        
        Args:
            price: Price to round
            symbol: Trading symbol
            
        Returns:
            Rounded price
        """
        config = self.get_precision_config(symbol)
        if not config:
            # Default to 8 decimal places
            return round(float(price), 8)
        
        # Convert to Decimal for precise calculation
        price_decimal = Decimal(str(price))
        tick_size = Decimal(str(config.tick_size))
        
        # Round to tick size
        rounded = (price_decimal / tick_size).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * tick_size
        
        # Round to price precision
        precision = Decimal('0.1') ** config.price_precision
        rounded = rounded.quantize(precision, rounding=ROUND_HALF_UP)
        
        return float(rounded)
    
    def round_quantity(self, quantity: Union[float, str, Decimal], symbol: str) -> float:
        """
        Round quantity to appropriate precision for symbol.
        
        Args:
            quantity: Quantity to round
            symbol: Trading symbol
            
        Returns:
            Rounded quantity
        """
        config = self.get_precision_config(symbol)
        if not config:
            # Default to 8 decimal places
            return round(float(quantity), 8)
        
        # Convert to Decimal for precise calculation
        quantity_decimal = Decimal(str(quantity))
        lot_size = Decimal(str(config.lot_size))
        
        # Round to lot size
        rounded = (quantity_decimal / lot_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * lot_size
        
        # Round to quantity precision
        precision = Decimal('0.1') ** config.quantity_precision
        rounded = rounded.quantize(precision, rounding=ROUND_DOWN)
        
        return float(rounded)
    
    def validate_price(self, price: Union[float, str, Decimal], symbol: str) -> Tuple[bool, str]:
        """
        Validate price for symbol.
        
        Args:
            price: Price to validate
            symbol: Trading symbol
            
        Returns:
            (is_valid, error_message)
        """
        try:
            price_decimal = Decimal(str(price))
            config = self.get_precision_config(symbol)
            
            if not config:
                return True, ""
            
            # Check if price is positive
            if price_decimal <= 0:
                return False, "Price must be positive"
            
            # Check if price is multiple of tick size
            tick_size = Decimal(str(config.tick_size))
            if price_decimal % tick_size != 0:
                return False, f"Price must be multiple of tick size {config.tick_size}"
            
            # Check precision
            expected_precision = Decimal('0.1') ** config.price_precision
            if price_decimal.quantize(expected_precision) != price_decimal:
                return False, f"Price precision must be {config.price_precision} decimal places"
            
            return True, ""
            
        except (ValueError, TypeError, decimal.InvalidOperation) as e:
            return False, f"Invalid price format: {e}"
    
    def validate_quantity(self, quantity: Union[float, str, Decimal], symbol: str) -> Tuple[bool, str]:
        """
        Validate quantity for symbol.
        
        Args:
            quantity: Quantity to validate
            symbol: Trading symbol
            
        Returns:
            (is_valid, error_message)
        """
        try:
            quantity_decimal = Decimal(str(quantity))
            config = self.get_precision_config(symbol)
            
            if not config:
                return True, ""
            
            # Check if quantity is positive
            if quantity_decimal <= 0:
                return False, "Quantity must be positive"
            
            # Check if quantity is multiple of lot size
            lot_size = Decimal(str(config.lot_size))
            if quantity_decimal % lot_size != 0:
                return False, f"Quantity must be multiple of lot size {config.lot_size}"
            
            # Check precision
            expected_precision = Decimal('0.1') ** config.quantity_precision
            if quantity_decimal.quantize(expected_precision) != quantity_decimal:
                return False, f"Quantity precision must be {config.quantity_precision} decimal places"
            
            return True, ""
            
        except (ValueError, TypeError, decimal.InvalidOperation) as e:
            return False, f"Invalid quantity format: {e}"
    
    def validate_notional(self, price: Union[float, str, Decimal], 
                         quantity: Union[float, str, Decimal], symbol: str) -> Tuple[bool, str]:
        """
        Validate notional value (price * quantity) for symbol.
        
        Args:
            price: Price
            quantity: Quantity
            symbol: Trading symbol
            
        Returns:
            (is_valid, error_message)
        """
        try:
            price_decimal = Decimal(str(price))
            quantity_decimal = Decimal(str(quantity))
            notional = price_decimal * quantity_decimal
            
            config = self.get_precision_config(symbol)
            if not config:
                return True, ""
            
            # Check minimum notional
            min_notional = Decimal(str(config.min_notional))
            if notional < min_notional:
                return False, f"Notional value {notional} is below minimum {min_notional}"
            
            # Check maximum notional
            if config.max_notional:
                max_notional = Decimal(str(config.max_notional))
                if notional > max_notional:
                    return False, f"Notional value {notional} exceeds maximum {max_notional}"
            
            return True, ""
            
        except (ValueError, TypeError, decimal.InvalidOperation) as e:
            return False, f"Invalid notional calculation: {e}"
    
    def calculate_minimum_quantity(self, price: Union[float, str, Decimal], symbol: str) -> float:
        """
        Calculate minimum quantity for a given price.
        
        Args:
            price: Price
            symbol: Trading symbol
            
        Returns:
            Minimum quantity
        """
        config = self.get_precision_config(symbol)
        if not config:
            return 0.001  # Default minimum
        
        price_decimal = Decimal(str(price))
        min_notional = Decimal(str(config.min_notional))
        lot_size = Decimal(str(config.lot_size))
        
        # Calculate minimum quantity
        min_quantity = min_notional / price_decimal
        
        # Round up to lot size
        min_quantity = (min_quantity / lot_size).quantize(Decimal('1'), rounding=ROUND_UP) * lot_size
        
        return float(min_quantity)
    
    def calculate_maximum_quantity(self, price: Union[float, str, Decimal], symbol: str) -> Optional[float]:
        """
        Calculate maximum quantity for a given price.
        
        Args:
            price: Price
            symbol: Trading symbol
            
        Returns:
            Maximum quantity or None if no limit
        """
        config = self.get_precision_config(symbol)
        if not config or not config.max_notional:
            return None
        
        price_decimal = Decimal(str(price))
        max_notional = Decimal(str(config.max_notional))
        lot_size = Decimal(str(config.lot_size))
        
        # Calculate maximum quantity
        max_quantity = max_notional / price_decimal
        
        # Round down to lot size
        max_quantity = (max_quantity / lot_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * lot_size
        
        return float(max_quantity)
    
    def adjust_quantity_to_lot_size(self, quantity: Union[float, str, Decimal], symbol: str) -> float:
        """
        Adjust quantity to be a multiple of lot size.
        
        Args:
            quantity: Quantity to adjust
            symbol: Trading symbol
            
        Returns:
            Adjusted quantity
        """
        config = self.get_precision_config(symbol)
        if not config:
            return float(quantity)
        
        quantity_decimal = Decimal(str(quantity))
        lot_size = Decimal(str(config.lot_size))
        
        # Round down to lot size
        adjusted = (quantity_decimal / lot_size).quantize(Decimal('1'), rounding=ROUND_DOWN) * lot_size
        
        return float(adjusted)
    
    def adjust_price_to_tick_size(self, price: Union[float, str, Decimal], symbol: str) -> float:
        """
        Adjust price to be a multiple of tick size.
        
        Args:
            price: Price to adjust
            symbol: Trading symbol
            
        Returns:
            Adjusted price
        """
        config = self.get_precision_config(symbol)
        if not config:
            return float(price)
        
        price_decimal = Decimal(str(price))
        tick_size = Decimal(str(config.tick_size))
        
        # Round to nearest tick size
        adjusted = (price_decimal / tick_size).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * tick_size
        
        return float(adjusted)
    
    def format_price(self, price: Union[float, str, Decimal], symbol: str) -> str:
        """
        Format price with appropriate precision.
        
        Args:
            price: Price to format
            symbol: Trading symbol
            
        Returns:
            Formatted price string
        """
        config = self.get_precision_config(symbol)
        if not config:
            return f"{float(price):.8f}"
        
        precision = config.price_precision
        return f"{float(price):.{precision}f}"
    
    def format_quantity(self, quantity: Union[float, str, Decimal], symbol: str) -> str:
        """
        Format quantity with appropriate precision.
        
        Args:
            quantity: Quantity to format
            symbol: Trading symbol
            
        Returns:
            Formatted quantity string
        """
        config = self.get_precision_config(symbol)
        if not config:
            return f"{float(quantity):.8f}"
        
        precision = config.quantity_precision
        return f"{float(quantity):.{precision}f}"
    
    def get_tick_size(self, symbol: str) -> Optional[float]:
        """Get tick size for symbol."""
        config = self.get_precision_config(symbol)
        return config.tick_size if config else None
    
    def get_lot_size(self, symbol: str) -> Optional[float]:
        """Get lot size for symbol."""
        config = self.get_precision_config(symbol)
        return config.lot_size if config else None
    
    def get_min_notional(self, symbol: str) -> Optional[float]:
        """Get minimum notional for symbol."""
        config = self.get_precision_config(symbol)
        return config.min_notional if config else None
    
    def get_max_notional(self, symbol: str) -> Optional[float]:
        """Get maximum notional for symbol."""
        config = self.get_precision_config(symbol)
        return config.max_notional if config else None
    
    def validate_order(self, symbol: str, side: str, order_type: str, 
                      price: Optional[Union[float, str, Decimal]], 
                      quantity: Union[float, str, Decimal]) -> Tuple[bool, List[str]]:
        """
        Validate complete order parameters.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            order_type: Order type (market/limit)
            price: Order price (optional for market orders)
            quantity: Order quantity
            
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Validate quantity
        is_valid_qty, qty_error = self.validate_quantity(quantity, symbol)
        if not is_valid_qty:
            errors.append(f"Quantity: {qty_error}")
        
        # Validate price for limit orders
        if order_type.lower() == "limit":
            if price is None:
                errors.append("Price is required for limit orders")
            else:
                is_valid_price, price_error = self.validate_price(price, symbol)
                if not is_valid_price:
                    errors.append(f"Price: {price_error}")
                
                # Validate notional
                if is_valid_qty and is_valid_price:
                    is_valid_notional, notional_error = self.validate_notional(price, quantity, symbol)
                    if not is_valid_notional:
                        errors.append(f"Notional: {notional_error}")
        
        # Validate notional for market orders (using current price estimate)
        elif order_type.lower() == "market":
            # For market orders, we can only validate quantity
            # Notional validation would require current market price
            pass
        
        return len(errors) == 0, errors
    
    def get_precision_summary(self, symbol: str) -> Dict[str, Any]:
        """Get precision summary for symbol."""
        config = self.get_precision_config(symbol)
        if not config:
            return {"error": "No precision config found"}
        
        return {
            "symbol": symbol,
            "price_precision": config.price_precision,
            "quantity_precision": config.quantity_precision,
            "tick_size": config.tick_size,
            "lot_size": config.lot_size,
            "min_notional": config.min_notional,
            "max_notional": config.max_notional
        }
    
    def bulk_validate_orders(self, orders: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate multiple orders in bulk.
        
        Args:
            orders: List of order dictionaries
            
        Returns:
            Validation results
        """
        results = {
            "valid_orders": [],
            "invalid_orders": [],
            "total_orders": len(orders),
            "valid_count": 0,
            "invalid_count": 0
        }
        
        for i, order in enumerate(orders):
            symbol = order.get("symbol", "")
            side = order.get("side", "")
            order_type = order.get("order_type", "")
            price = order.get("price")
            quantity = order.get("quantity")
            
            is_valid, errors = self.validate_order(symbol, side, order_type, price, quantity)
            
            order_result = {
                "index": i,
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "price": price,
                "quantity": quantity,
                "is_valid": is_valid,
                "errors": errors
            }
            
            if is_valid:
                results["valid_orders"].append(order_result)
                results["valid_count"] += 1
            else:
                results["invalid_orders"].append(order_result)
                results["invalid_count"] += 1
        
        return results