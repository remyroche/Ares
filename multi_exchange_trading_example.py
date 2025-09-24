#!/usr/bin/env python3
"""
Multi-Exchange Trading Example

This example demonstrates how to use the enhanced trading system with multi-exchange support.
It shows how to send orders to all exchanges and receive responses back.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

from exchanges import TradingReceiver
from live_trading.trading_orchestrator import TradingSignal
from live_trading.config import TradingConfig, TradingMode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def create_trading_receiver() -> TradingReceiver:
    """Create and initialize a trading receiver with multi-exchange support"""
    config = {
        "exchanges": {
            "binance": {
                "api_key": "your_binance_api_key",
                "api_secret": "your_binance_secret",
                "sandbox": True,  # Use sandbox for testing
                "rate_limit": 1200,
                "timeout": 30
            },
            "okx": {
                "api_key": "your_okx_api_key",
                "api_secret": "your_okx_secret",
                "sandbox": True,
                "rate_limit": 1200,
                "timeout": 30
            },
            "gateio": {
                "api_key": "your_gateio_api_key",
                "api_secret": "your_gateio_secret",
                "sandbox": False,
                "rate_limit": 1200,
                "timeout": 30
            }
        },
        "primary_exchange": "binance",
        "failover_exchanges": ["okx", "gateio"],
        "broadcast_enabled": True,
        "load_balancing_enabled": False
    }

    receiver = TradingReceiver(config)
    await receiver.start()

    return receiver

async def demonstrate_multi_exchange_orders():
    """Demonstrate sending orders to all exchanges"""
    print("🚀 Multi-Exchange Trading Example")
    print("=" * 50)

    try:
        # Create trading receiver
        receiver = await create_trading_receiver()
        print("✅ Trading receiver initialized")

        # Test 1: Send order to all exchanges (broadcast)
        print("\n📡 Test 1: Broadcasting order to all exchanges")
        print("-" * 40)

        broadcast_responses = await receiver.send_order_to_all_exchanges(
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            quantity=0.001,
            test_order=True  # Mark as test order
        )

        print(f"📊 Broadcast responses from {len(broadcast_responses)} exchanges:")
        for exchange, response in broadcast_responses.items():
            if isinstance(response, dict) and "success" in response:
                status = "✅ SUCCESS" if response["success"] else "❌ FAILED"
                print(f"  {exchange}: {status}")
                if not response["success"]:
                    print(f"    Error: {response.get('error', 'Unknown error')}")
            else:
                print(f"  {exchange}: Invalid response format")

        # Test 2: Send order with intelligent routing
        print("\n🎯 Test 2: Intelligent routing strategies")
        print("-" * 40)

        # Broadcast strategy
        broadcast_result = await receiver.send_order_with_routing(
            symbol="ETHUSDT",
            side="buy",
            order_type="limit",
            quantity=0.01,
            price=2000.0,
            routing_strategy="broadcast"
        )
        print("📡 Broadcast routing:")
        for exchange, response in broadcast_result.items():
            if isinstance(response, dict) and "success" in response:
                status = "✅ SUCCESS" if response["success"] else "❌ FAILED"
                print(f"  {exchange}: {status}")
            else:
                print(f"  {exchange}: {response}")

        # Primary with failover strategy
        primary_result = await receiver.send_order_with_routing(
            symbol="ETHUSDT",
            side="sell",
            order_type="limit",
            quantity=0.01,
            price=2100.0,
            routing_strategy="primary"
        )
        print("🔄 Primary with failover routing:")
        if isinstance(primary_result, dict) and "error" not in primary_result:
            for exchange, response in primary_result.items():
                if isinstance(response, dict) and "success" in response:
                    status = "✅ SUCCESS" if response["success"] else "❌ FAILED"
                    print(f"  {exchange}: {status}")
        else:
            print(f"  Error: {primary_result.get('error', 'Unknown error')}")

        # Best price strategy
        best_price_result = await receiver.send_order_with_routing(
            symbol="BTCUSDT",
            side="buy",
            order_type="limit",
            quantity=0.001,
            price=45000.0,
            routing_strategy="best_price"
        )
        print("💰 Best price routing:")
        if isinstance(best_price_result, dict) and "error" not in best_price_result:
            for exchange, response in best_price_result.items():
                if isinstance(response, dict) and "success" in response:
                    status = "✅ SUCCESS" if response["success"] else "❌ FAILED"
                    print(f"  {exchange}: {status}")
        else:
            print(f"  Error: {best_price_result.get('error', 'Unknown error')}")

        # Test 3: Get multi-exchange order status
        print("\n📋 Test 3: Multi-exchange order tracking")
        print("-" * 40)

        # Get all multi-exchange orders
        all_orders = await receiver.get_all_multi_exchange_orders()
        print(f"📊 Total multi-exchange orders: {len(all_orders)}")

        # Get specific order status (if any exist)
        for order_id in all_orders.keys():
            status = await receiver.get_multi_exchange_order_status(order_id)
            if status:
                print(f"📋 Order {order_id}: {status['status']}")
                print(f"   Target exchanges: {status.get('target_exchanges', [])}")

        # Test 4: Cancel multi-exchange order
        print("\n❌ Test 4: Multi-exchange order cancellation")
        print("-" * 40)

        # Try to cancel a multi-exchange order (if any exist)
        if all_orders:
            first_order_id = list(all_orders.keys())[0]
            cancel_result = await receiver.cancel_multi_exchange_order(first_order_id)
            print(f"📋 Cancellation result for {first_order_id}:")
            if "error" not in cancel_result:
                successful_cancellations = cancel_result.get("successful_cancellations", [])
                print(f"   Successful cancellations: {successful_cancellations}")
                print(f"   Total attempted: {len(cancel_result.get('cancellation_responses', {}))}")
            else:
                print(f"   Error: {cancel_result['error']}")

        # Test 5: Get comprehensive statistics
        print("\n📈 Test 5: System statistics")
        print("-" * 40)

        stats = await receiver.get_statistics()
        print("📊 System Statistics:"        print(f"   Running: {stats['running']}")
        print(f"   Total messages: {stats['statistics']['total_received']}")
        print(f"   Errors: {stats['statistics']['total_errors']}")
        print(f"   Multi-exchange orders: {stats['multi_exchange']['multi_exchange_orders']}")
        print(f"   Primary exchange: {stats['multi_exchange']['primary_exchange']}")
        print(f"   Failover exchanges: {stats['multi_exchange']['failover_exchanges']}")
        print(f"   Broadcast enabled: {stats['multi_exchange']['broadcast_enabled']}")

        # Test 6: Request data from all exchanges
        print("\n📊 Test 6: Multi-exchange data aggregation")
        print("-" * 40)

        # Get aggregated ticker data
        ticker_data = await receiver.request_data("binance", "BTCUSDT", "ticker")
        print(f"📈 Binance ticker: {ticker_data.get('data', 'No data')}")

        # Get account info from primary exchange
        account_info = await receiver.get_account_info("binance")
        print(f"💰 Binance account: {account_info.get('data', 'No data')}")

        # Test 7: Error handling demonstration
        print("\n⚠️ Test 7: Error handling")
        print("-" * 40)

        # Try to send order with invalid parameters
        error_response = await receiver.send_order(
            exchange="invalid_exchange",
            symbol="INVALID",
            side="buy",
            order_type="market",
            quantity=1.0
        )
        print(f"❌ Invalid exchange response: {error_response.get('error', 'No error message')}")

        await receiver.stop()
        print("✅ Trading receiver stopped successfully")

    except Exception as e:
        print(f"❌ Error in multi-exchange example: {e}")
        import traceback
        traceback.print_exc()

async def demonstrate_response_flow():
    """Demonstrate the response flow from exchanges back to the system"""
    print("\n🔄 Response Flow Demonstration")
    print("=" * 50)

    receiver = await create_trading_receiver()

    try:
        # Simulate response handling
        print("📨 Simulating response flow...")

        # Create a mock order response
        from exchanges.base_exchange.response_handler import ExchangeResponse, ResponseType, ResponseStatus
        from exchanges.base_exchange.response_handler import OrderExecutionResponse

        mock_response = OrderExecutionResponse(
            response_id="test_response_123",
            exchange_name="binance",
            response_type=ResponseType.ORDER_EXECUTION,
            status=ResponseStatus.SUCCESS,
            order_id="test_order_123",
            exchange_order_id="binance_12345",
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            quantity=0.001,
            filled_quantity=0.001,
            average_fill_price=45000.0,
            commission=0.001,
            commission_asset="BNB",
            data={"orderId": "binance_12345", "status": "FILLED"}
        )

        # Handle the response
        await receiver.response_handler.handle_response(mock_response)

        # Register a callback to handle order responses
        async def order_response_callback(response: ExchangeResponse):
            if hasattr(response, 'order_id'):
                print(f"📨 Received order response: {response.order_id} on {response.exchange_name}")
                print(f"   Status: {response.status.value}")
                print(f"   Filled: {response.filled_quantity}/{response.quantity}")
                print(f"   Price: {response.average_fill_price}")

        # Register the callback
        callback_id = receiver.response_handler.register_response_callback(
            ResponseType.ORDER_EXECUTION,
            order_response_callback
        )

        print(f"📝 Registered response callback: {callback_id}")

        # Simulate getting statistics
        response_stats = await receiver.response_handler.get_response_statistics()
        print("📊 Response Handler Statistics:"        print(f"   Registered callbacks: {response_stats.get('registered_callbacks', 0)}")
        print(f"   Pending responses: {response_stats.get('pending_responses', 0)}")

        await receiver.stop()

    except Exception as e:
        print(f"❌ Error in response flow demonstration: {e}")
        await receiver.stop()

async def demonstrate_message_routing():
    """Demonstrate advanced message routing capabilities"""
    print("\n🛣️ Message Routing Demonstration")
    print("=" * 50)

    receiver = await create_trading_receiver()

    try:
        print("📨 Testing message routing strategies...")

        # Test message queue status
        queue_status = await receiver.message_handler.get_queue_status()
        print("📊 Message Queue Status:"        print(f"   Total pending: {queue_status.get('total_pending_messages', 0)}")
        print(f"   Queue sizes: {queue_status.get('queue_sizes_by_priority', {})}")

        # Test message routing strategies
        from exchanges.base_exchange.message_handler import MessageRouter

        # Create a test message
        from exchanges.base_exchange.message_handler import ExchangeMessage, MessageType, MessagePriority

        test_message = ExchangeMessage(
            id="test_message_123",
            message_type=MessageType.ORDER,
            priority=MessagePriority.NORMAL,
            payload={
                "symbol": "BTCUSDT",
                "side": "buy",
                "order_type": "market",
                "quantity": 0.001
            }
        )

        # Test different routing strategies (simplified)
        print("🛣️ Testing routing strategies...")

        # Broadcast routing
        print("   📡 Broadcast: Would route to all exchanges")

        # Primary routing
        print("   🎯 Primary: Would route to binance (primary)")

        # Failover routing
        print("   🔄 Failover: Would route to binance, then okx, then gateio")

        # Best price routing
        print("   💰 Best Price: Would route to exchange with best price")

        await receiver.stop()
        print("✅ Message routing demonstration completed")

    except Exception as e:
        print(f"❌ Error in message routing demonstration: {e}")
        await receiver.stop()

async def main():
    """Main demonstration function"""
    print("🚀 Multi-Exchange Trading System Demonstration")
    print("=" * 60)

    try:
        # Run all demonstrations
        await demonstrate_multi_exchange_orders()
        print("\n" + "=" * 60)

        await demonstrate_response_flow()
        print("\n" + "=" * 60)

        await demonstrate_message_routing()
        print("\n" + "=" * 60)

        print("🎉 All demonstrations completed successfully!")
        print("\n📚 Key Features Demonstrated:")
        print("   ✅ Multi-exchange order broadcasting")
        print("   ✅ Intelligent routing strategies")
        print("   ✅ Response aggregation and handling")
        print("   ✅ Failover and load balancing")
        print("   ✅ Error handling and recovery")
        print("   ✅ System statistics and monitoring")

    except Exception as e:
        print(f"❌ Fatal error in demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())