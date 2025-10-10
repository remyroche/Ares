"""
Example usage of high-level shared exchange utilities with comprehensive type hints.

This example demonstrates how to use the high-level interfaces
for common exchange operations with full type safety and error handling.
"""

import asyncio
from typing import Dict, List, Optional, Any
from exchanges.shared import (
    HighLevelAuthManager,
    HighLevelMarketManager,
    HighLevelOrderManager,
    HighLevelRiskManager,
    HighLevelBalanceManager,
    HighLevelRateLimitManager,
    DataSource,
    ValidationResult,
    tprint,
    handle_errors,
    handle_async_errors
)


@handle_async_errors(default_return=False)
async def main() -> bool:
    """Example of using high-level exchange utilities with type safety."""
    
    # Initialize all managers with proper type hints
    auth_manager: HighLevelAuthManager = HighLevelAuthManager("okx")
    market_manager: HighLevelMarketManager = HighLevelMarketManager("okx")
    order_manager: HighLevelOrderManager = HighLevelOrderManager("okx")
    risk_manager: HighLevelRiskManager = HighLevelRiskManager("okx")
    balance_manager: HighLevelBalanceManager = HighLevelBalanceManager("okx")
    rate_limit_manager: HighLevelRateLimitManager = HighLevelRateLimitManager("okx")
    
    # Initialize all managers
    auth_manager.initialize()
    market_manager.initialize()
    order_manager.initialize()
    risk_manager.initialize()
    balance_manager.initialize()
    rate_limit_manager.initialize()
    
    try:
        # 1. Authentication with type safety
        tprint("=== Authentication ===", "INFO")
        credentials: Dict[str, Any] = {
            "api_key": "your_api_key",
            "api_secret": "your_api_secret",
            "passphrase": "your_passphrase",
            "permissions": ["read", "trade"],
            "auto_sync_time": True
        }
        
        auth_success: bool = await auth_manager.authenticate(credentials)
        tprint(f"Authentication successful: {auth_success}", "INFO")
        
        if auth_success:
            has_trade_permission: bool = auth_manager.has_permission("trade")
            tprint(f"Has trade permission: {has_trade_permission}", "INFO")
        
        # 2. Market Data with type safety
        tprint("\n=== Market Data ===", "INFO")
        
        # Get instrument information with proper typing
        instrument_info: Optional[Dict[str, Any]] = await market_manager.get_instrument_info("BTCUSDT")
        if instrument_info:
            tprint(f"Instrument info: {instrument_info}", "INFO")
        
        # Get current price with type safety
        price: Optional[float] = await market_manager.get_price("BTCUSDT", DataSource.CACHE)
        if price:
            tprint(f"Current BTC price: ${price}", "INFO")
        
        # Search for trading pairs with type safety
        trading_pairs: List[Dict[str, Any]] = market_manager.search_instruments({
            "base_currency": "BTC",
            "type": "spot"
        })
        tprint(f"Found {len(trading_pairs)} BTC trading pairs", "INFO")
        
        # 3. Balance Management with type safety
        tprint("\n=== Balance Management ===", "INFO")
        
        # Get USDT balance with type safety
        usdt_balance: Optional[float] = await balance_manager.get_balance("USDT", "spot")
        if usdt_balance:
            tprint(f"USDT balance: {usdt_balance}", "INFO")
        
        # Get all balances with type safety
        all_balances: Dict[str, float] = await balance_manager.get_all_balances("spot")
        tprint(f"All balances: {all_balances}", "INFO")
        
        # Check if sufficient balance with type safety
        has_balance: bool = balance_manager.has_sufficient_balance("USDT", 100.0, "spot")
        tprint(f"Has sufficient USDT balance: {has_balance}", "INFO")
        
        # 4. Risk Management with type safety
        tprint("\n=== Risk Management ===", "INFO")
        
        # Calculate position risk with type safety
        position_risk: Dict[str, Any] = risk_manager.calculate_position_risk(
            symbol="BTCUSDT",
            position_size=0.001,
            current_price=50000.0,
            leverage=2.0
        )
        tprint(f"Position risk: {position_risk}", "INFO")
        
        # Validate risk limits with type safety
        validation: ValidationResult = risk_manager.validate_risk_limits(position_risk)
        tprint(f"Risk validation - Valid: {validation.is_valid}", "INFO")
        if validation.warnings:
            tprint(f"Warnings: {validation.warnings}", "WARNING")
        if validation.errors:
            tprint(f"Errors: {validation.errors}", "ERROR")
        
        # Calculate max position size with type safety
        max_size: float = risk_manager.get_max_position_size(
            symbol="BTCUSDT",
            available_margin=1000.0,
            risk_tolerance=0.8
        )
        tprint(f"Max position size: {max_size}", "INFO")
        
        # 5. Order Management with type safety
        tprint("\n=== Order Management ===", "INFO")
        
        # Validate order parameters with type safety
        order_params: Dict[str, Any] = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "order_type": "limit",
            "quantity": 0.001,
            "price": 49000.0
        }
        
        order_validation: ValidationResult = order_manager.validate_order_params(order_params)
        tprint(f"Order validation - Valid: {order_validation.is_valid}", "INFO")
        if order_validation.errors:
            tprint(f"Order errors: {order_validation.errors}", "ERROR")
        
        # Create order with type safety (if validation passes)
        if order_validation.is_valid:
            order_id: Optional[str] = await order_manager.create_order(**order_params)
            if order_id:
                tprint(f"Created order: {order_id}", "INFO")
                
                # Get order status with type safety
                order_status: Optional[Dict[str, Any]] = await order_manager.get_order_status(order_id)
                if order_status:
                    tprint(f"Order status: {order_status}", "INFO")
        
        # Get open orders with type safety
        open_orders: List[Dict[str, Any]] = await order_manager.get_open_orders("BTCUSDT")
        tprint(f"Open BTCUSDT orders: {len(open_orders)}", "INFO")
        
        # 6. Rate Limiting with type safety
        tprint("\n=== Rate Limiting ===", "INFO")
        
        # Set rate limits with type safety
        rate_limits: Dict[str, int] = {
            "per_second": 5,
            "per_minute": 300,
            "per_hour": 18000,
            "burst": 10
        }
        rate_limit_manager.set_limits("trading", rate_limits)
        
        # Execute function with rate limiting and type safety
        async def sample_api_call() -> str:
            return "API response"
        
        result: str = await rate_limit_manager.execute_with_limits(
            "trading", sample_api_call
        )
        tprint(f"Rate limited API call result: {result}", "INFO")
        
        # Check remaining requests with type safety
        remaining: int = rate_limit_manager.get_remaining_requests("trading")
        tprint(f"Remaining trading requests: {remaining}", "INFO")
        
        # 7. Manager Statistics with type safety
        tprint("\n=== Manager Statistics ===", "INFO")
        
        auth_status: Dict[str, Any] = auth_manager.get_status()
        market_status: Dict[str, Any] = market_manager.get_status()
        order_status: Dict[str, Any] = order_manager.get_status()
        risk_status: Dict[str, Any] = risk_manager.get_status()
        balance_status: Dict[str, Any] = balance_manager.get_status()
        rate_limit_status: Dict[str, Any] = rate_limit_manager.get_status()
        
        tprint(f"Auth status: {auth_status}", "INFO")
        tprint(f"Market status: {market_status}", "INFO")
        tprint(f"Order status: {order_status}", "INFO")
        tprint(f"Risk status: {risk_status}", "INFO")
        tprint(f"Balance status: {balance_status}", "INFO")
        tprint(f"Rate limit status: {rate_limit_status}", "INFO")
        
        return True
        
    except Exception as e:
        tprint(f"Error in main execution: {e}", "ERROR")
        return False
    
    finally:
        # Clean up with proper error handling
        try:
            auth_manager.close()
            market_manager.close()
            order_manager.close()
            risk_manager.close()
            balance_manager.close()
            rate_limit_manager.close()
            tprint("All managers closed successfully", "INFO")
        except Exception as e:
            tprint(f"Error closing managers: {e}", "ERROR")


@handle_async_errors(default_return=False)
async def advanced_typed_usage_example() -> bool:
    """Example of advanced usage with full type safety and error handling."""
    
    # Initialize managers with type annotations
    managers: Dict[str, Any] = {
        "auth": HighLevelAuthManager("okx"),
        "market": HighLevelMarketManager("okx"),
        "order": HighLevelOrderManager("okx"),
        "risk": HighLevelRiskManager("okx"),
        "balance": HighLevelBalanceManager("okx"),
        "rate_limit": HighLevelRateLimitManager("okx")
    }
    
    # Initialize all managers with error handling
    for name, manager in managers.items():
        try:
            manager.initialize()
            tprint(f"Initialized {name} manager", "DEBUG")
        except Exception as e:
            tprint(f"Failed to initialize {name} manager: {e}", "ERROR")
            return False
    
    try:
        # Configure rate limits with type safety
        trading_limits: Dict[str, int] = {
            "per_second": 2,
            "per_minute": 120,
            "per_hour": 7200
        }
        managers["rate_limit"].set_limits("trading", trading_limits)
        
        # Configure risk thresholds with type safety
        managers["risk"].risk_calculator.set_risk_thresholds(
            warning_ratio=0.7,
            critical_ratio=0.85,
            liquidation_ratio=0.95
        )
        
        # Example trading workflow with full type safety
        tprint("=== Advanced Trading Workflow ===", "INFO")
        
        # 1. Check authentication with type safety
        if not managers["auth"].is_authenticated():
            tprint("Not authenticated, please authenticate first", "WARNING")
            return False
        
        # 2. Get market data with type safety
        symbol: str = "BTCUSDT"
        price: Optional[float] = await managers["market"].get_price(symbol, DataSource.EXCHANGE)
        if not price:
            tprint(f"Could not get price for {symbol}", "ERROR")
            return False
        
        tprint(f"Current {symbol} price: ${price}", "INFO")
        
        # 3. Check balance with type safety
        balance: Optional[float] = await managers["balance"].get_balance("USDT", "spot")
        if not balance or balance < 100:
            tprint("Insufficient USDT balance", "WARNING")
            return False
        
        tprint(f"USDT balance: ${balance}", "INFO")
        
        # 4. Calculate position size based on risk with type safety
        max_position: float = managers["risk"].get_max_position_size(
            symbol=symbol,
            available_margin=balance * 0.1,  # Use 10% of balance
            risk_tolerance=0.8
        )
        
        position_size: float = min(max_position, 0.001)  # Cap at 0.001 BTC
        tprint(f"Calculated position size: {position_size}", "INFO")
        
        # 5. Validate order with type safety
        order_params: Dict[str, Any] = {
            "symbol": symbol,
            "side": "buy",
            "order_type": "limit",
            "quantity": position_size,
            "price": price * 0.99  # 1% below market
        }
        
        validation: ValidationResult = managers["order"].validate_order_params(order_params)
        if not validation.is_valid:
            tprint(f"Order validation failed: {validation.errors}", "ERROR")
            return False
        
        # 6. Execute order with rate limiting and type safety
        order_id: Optional[str] = await managers["rate_limit"].execute_with_limits(
            "trading",
            managers["order"].create_order,
            **order_params
        )
        
        if order_id:
            tprint(f"Order created successfully: {order_id}", "INFO")
            
            # Monitor order with type safety
            order_status: Optional[Dict[str, Any]] = await managers["order"].get_order_status(order_id)
            if order_status:
                tprint(f"Order status: {order_status}", "INFO")
        else:
            tprint("Failed to create order", "ERROR")
            return False
        
        return True
        
    except Exception as e:
        tprint(f"Error in advanced workflow: {e}", "ERROR")
        return False
    
    finally:
        # Clean up all managers with error handling
        for name, manager in managers.items():
            try:
                manager.close()
                tprint(f"Closed {name} manager", "DEBUG")
            except Exception as e:
                tprint(f"Error closing {name} manager: {e}", "ERROR")


@handle_errors(default_return=False)
def demonstrate_type_safety() -> bool:
    """Demonstrate type safety features."""
    
    tprint("=== Type Safety Demonstration ===", "INFO")
    
    # Type hints are enforced at development time
    auth_manager: HighLevelAuthManager = HighLevelAuthManager("test")
    
    # These will show type errors in IDEs that support type checking
    # auth_manager.authenticate("invalid")  # Type error: expects Dict[str, Any]
    # auth_manager.is_authenticated("extra_arg")  # Type error: expects no arguments
    
    # Proper usage with type safety
    credentials: Dict[str, Any] = {
        "api_key": "test",
        "api_secret": "test",
        "permissions": ["read"]
    }
    
    # This is properly typed
    auth_manager.initialize()
    is_auth: bool = auth_manager.is_authenticated()
    tprint(f"Type-safe authentication check: {is_auth}", "INFO")
    
    auth_manager.close()
    return True


if __name__ == "__main__":
    tprint("Running typed usage examples...", "INFO")
    
    # Run basic example
    asyncio.run(main())
    
    tprint("\n" + "="*50, "INFO")
    tprint("Running advanced typed usage example...", "INFO")
    
    # Run advanced example
    asyncio.run(advanced_typed_usage_example())
    
    tprint("\n" + "="*50, "INFO")
    tprint("Demonstrating type safety...", "INFO")
    
    # Demonstrate type safety
    demonstrate_type_safety()
    
    tprint("All examples completed!", "INFO")