"""
Example usage of high-level shared exchange utilities.

This example demonstrates how to use the high-level interfaces
for common exchange operations with consistent abstraction levels.
"""

import asyncio
from exchanges.shared import (
    HighLevelAuthManager,
    HighLevelMarketManager,
    HighLevelOrderManager,
    HighLevelRiskManager,
    HighLevelBalanceManager,
    HighLevelRateLimitManager,
    DataSource
)


async def main():
    """Example of using high-level exchange utilities."""
    
    # Initialize all managers
    auth_manager = HighLevelAuthManager("okx")
    market_manager = HighLevelMarketManager("okx")
    order_manager = HighLevelOrderManager("okx")
    risk_manager = HighLevelRiskManager("okx")
    balance_manager = HighLevelBalanceManager("okx")
    rate_limit_manager = HighLevelRateLimitManager("okx")
    
    # Initialize all managers
    auth_manager.initialize()
    market_manager.initialize()
    order_manager.initialize()
    risk_manager.initialize()
    balance_manager.initialize()
    rate_limit_manager.initialize()
    
    try:
        # 1. Authentication
        print("=== Authentication ===")
        credentials = {
            "api_key": "your_api_key",
            "api_secret": "your_api_secret",
            "passphrase": "your_passphrase",
            "permissions": ["read", "trade"],
            "auto_sync_time": True
        }
        
        auth_success = await auth_manager.authenticate(credentials)
        print(f"Authentication successful: {auth_success}")
        print(f"Has trade permission: {auth_manager.has_permission('trade')}")
        
        # 2. Market Data
        print("\n=== Market Data ===")
        
        # Get instrument information
        instrument_info = await market_manager.get_instrument_info("BTCUSDT")
        if instrument_info:
            print(f"Instrument info: {instrument_info}")
        
        # Get current price
        price = await market_manager.get_price("BTCUSDT", DataSource.CACHE)
        print(f"Current BTC price: ${price}")
        
        # Search for trading pairs
        trading_pairs = market_manager.search_instruments({
            "base_currency": "BTC",
            "type": "spot"
        })
        print(f"Found {len(trading_pairs)} BTC trading pairs")
        
        # 3. Balance Management
        print("\n=== Balance Management ===")
        
        # Get USDT balance
        usdt_balance = await balance_manager.get_balance("USDT", "spot")
        print(f"USDT balance: {usdt_balance}")
        
        # Get all balances
        all_balances = await balance_manager.get_all_balances("spot")
        print(f"All balances: {all_balances}")
        
        # Check if sufficient balance for trade
        has_balance = balance_manager.has_sufficient_balance("USDT", 100.0, "spot")
        print(f"Has sufficient USDT balance: {has_balance}")
        
        # 4. Risk Management
        print("\n=== Risk Management ===")
        
        # Calculate position risk
        position_risk = risk_manager.calculate_position_risk(
            symbol="BTCUSDT",
            position_size=0.001,
            current_price=50000.0,
            leverage=2.0
        )
        print(f"Position risk: {position_risk}")
        
        # Validate risk limits
        validation = risk_manager.validate_risk_limits(position_risk)
        print(f"Risk validation - Valid: {validation.is_valid}")
        if validation.warnings:
            print(f"Warnings: {validation.warnings}")
        if validation.errors:
            print(f"Errors: {validation.errors}")
        
        # Calculate max position size
        max_size = risk_manager.get_max_position_size(
            symbol="BTCUSDT",
            available_margin=1000.0,
            risk_tolerance=0.8
        )
        print(f"Max position size: {max_size}")
        
        # 5. Order Management
        print("\n=== Order Management ===")
        
        # Validate order parameters
        order_params = {
            "symbol": "BTCUSDT",
            "side": "buy",
            "order_type": "limit",
            "quantity": 0.001,
            "price": 49000.0
        }
        
        validation = order_manager.validate_order_params(order_params)
        print(f"Order validation - Valid: {validation.is_valid}")
        if validation.errors:
            print(f"Order errors: {validation.errors}")
        
        # Create order (if validation passes)
        if validation.is_valid:
            order_id = await order_manager.create_order(**order_params)
            print(f"Created order: {order_id}")
            
            # Get order status
            if order_id:
                order_status = await order_manager.get_order_status(order_id)
                print(f"Order status: {order_status}")
        
        # Get open orders
        open_orders = await order_manager.get_open_orders("BTCUSDT")
        print(f"Open BTCUSDT orders: {len(open_orders)}")
        
        # 6. Rate Limiting
        print("\n=== Rate Limiting ===")
        
        # Set rate limits
        rate_limit_manager.set_limits("trading", {
            "per_second": 5,
            "per_minute": 300,
            "per_hour": 18000,
            "burst": 10
        })
        
        # Execute function with rate limiting
        async def sample_api_call():
            return "API response"
        
        result = await rate_limit_manager.execute_with_limits(
            "trading", sample_api_call
        )
        print(f"Rate limited API call result: {result}")
        
        # Check remaining requests
        remaining = rate_limit_manager.get_remaining_requests("trading")
        print(f"Remaining trading requests: {remaining}")
        
        # 7. Manager Statistics
        print("\n=== Manager Statistics ===")
        
        print(f"Auth status: {auth_manager.get_status()}")
        print(f"Market status: {market_manager.get_status()}")
        print(f"Order status: {order_manager.get_status()}")
        print(f"Risk status: {risk_manager.get_status()}")
        print(f"Balance status: {balance_manager.get_status()}")
        print(f"Rate limit status: {rate_limit_manager.get_status()}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up
        auth_manager.close()
        market_manager.close()
        order_manager.close()
        risk_manager.close()
        balance_manager.close()
        rate_limit_manager.close()


async def advanced_usage_example():
    """Example of advanced usage with error handling and configuration."""
    
    # Initialize managers with error handling
    managers = {
        "auth": HighLevelAuthManager("okx"),
        "market": HighLevelMarketManager("okx"),
        "order": HighLevelOrderManager("okx"),
        "risk": HighLevelRiskManager("okx"),
        "balance": HighLevelBalanceManager("okx"),
        "rate_limit": HighLevelRateLimitManager("okx")
    }
    
    # Initialize all managers
    for manager in managers.values():
        manager.initialize()
    
    try:
        # Configure rate limits
        managers["rate_limit"].set_limits("trading", {
            "per_second": 2,
            "per_minute": 120,
            "per_hour": 7200
        })
        
        # Configure risk thresholds
        managers["risk"].risk_calculator.set_risk_thresholds(
            warning_ratio=0.7,
            critical_ratio=0.85,
            liquidation_ratio=0.95
        )
        
        # Example trading workflow
        print("=== Advanced Trading Workflow ===")
        
        # 1. Check authentication
        if not managers["auth"].is_authenticated():
            print("Not authenticated, please authenticate first")
            return
        
        # 2. Get market data
        symbol = "BTCUSDT"
        price = await managers["market"].get_price(symbol, DataSource.EXCHANGE)
        if not price:
            print(f"Could not get price for {symbol}")
            return
        
        print(f"Current {symbol} price: ${price}")
        
        # 3. Check balance
        balance = await managers["balance"].get_balance("USDT", "spot")
        if not balance or balance < 100:
            print("Insufficient USDT balance")
            return
        
        print(f"USDT balance: ${balance}")
        
        # 4. Calculate position size based on risk
        max_position = managers["risk"].get_max_position_size(
            symbol=symbol,
            available_margin=balance * 0.1,  # Use 10% of balance
            risk_tolerance=0.8
        )
        
        position_size = min(max_position, 0.001)  # Cap at 0.001 BTC
        print(f"Calculated position size: {position_size}")
        
        # 5. Validate order
        order_params = {
            "symbol": symbol,
            "side": "buy",
            "order_type": "limit",
            "quantity": position_size,
            "price": price * 0.99  # 1% below market
        }
        
        validation = managers["order"].validate_order_params(order_params)
        if not validation.is_valid:
            print(f"Order validation failed: {validation.errors}")
            return
        
        # 6. Execute order with rate limiting
        order_id = await managers["rate_limit"].execute_with_limits(
            "trading",
            managers["order"].create_order,
            **order_params
        )
        
        if order_id:
            print(f"Order created successfully: {order_id}")
            
            # Monitor order
            order_status = await managers["order"].get_order_status(order_id)
            print(f"Order status: {order_status}")
        else:
            print("Failed to create order")
    
    except Exception as e:
        print(f"Error in advanced workflow: {e}")
    
    finally:
        # Clean up all managers
        for manager in managers.values():
            manager.close()


if __name__ == "__main__":
    print("Running basic usage example...")
    asyncio.run(main())
    
    print("\n" + "="*50)
    print("Running advanced usage example...")
    asyncio.run(advanced_usage_example())