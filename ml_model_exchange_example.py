#!/usr/bin/env python3
"""
ML Model Exchange Routing Example

This example demonstrates how to use ML model-based exchange routing.
Each ML model is associated with a specific exchange, and orders are sent
only to the exchange associated with the ML model.
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

async def create_trading_receiver_with_ml_models() -> TradingReceiver:
    """Create and initialize a trading receiver with ML model to exchange mappings"""
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
        "broadcast_enabled": False,  # ML model routing doesn't use broadcast
        "load_balancing_enabled": False,

        # ML model to exchange mappings
        "ml_model_exchanges": {
            "binance_prophet_model": "binance",
            "binance_lstm_model": "binance",
            "okx_random_forest": "okx",
            "okx_gradient_boost": "okx",
            "gateio_neural_net": "gateio",
            "gateio_svm_model": "gateio"
        },
        "default_ml_exchange": "binance"
    }

    receiver = TradingReceiver(config)

    # Register ML model associations
    receiver.register_ml_model_exchange("binance_prophet_model", "binance")
    receiver.register_ml_model_exchange("binance_lstm_model", "binance")
    receiver.register_ml_model_exchange("okx_random_forest", "okx")
    receiver.register_ml_model_exchange("okx_gradient_boost", "okx")
    receiver.register_ml_model_exchange("gateio_neural_net", "gateio")
    receiver.register_ml_model_exchange("gateio_svm_model", "gateio")

    await receiver.start()

    return receiver

async def demonstrate_ml_model_routing():
    """Demonstrate ML model-based exchange routing"""
    print("🤖 ML Model Exchange Routing Example")
    print("=" * 50)

    try:
        # Create trading receiver with ML model mappings
        receiver = await create_trading_receiver_with_ml_models()
        print("✅ Trading receiver initialized with ML model mappings")

        # Test 1: Orders routed to specific ML model exchanges
        print("\n🎯 Test 1: ML Model-Specific Exchange Routing")
        print("-" * 40)

        # Binance Prophet Model -> Binance Exchange
        print("📊 Binance Prophet Model → Binance Exchange")
        response1 = await receiver.send_order_for_ml_model(
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            quantity=0.001,
            ml_model_id="binance_prophet_model"
        )
        print(f"   Response: {response1.success}")
        if hasattr(response1, 'metadata') and response1.metadata:
            print(f"   Target Exchange: {response1.metadata.get('target_exchange', 'unknown')}")

        # OKX Random Forest Model -> OKX Exchange
        print("\n📊 OKX Random Forest Model → OKX Exchange")
        response2 = await receiver.send_order_for_ml_model(
            symbol="ETHUSDT",
            side="sell",
            order_type="limit",
            quantity=0.01,
            price=2000.0,
            ml_model_id="okx_random_forest"
        )
        print(f"   Response: {response2.success}")
        if hasattr(response2, 'metadata') and response2.metadata:
            print(f"   Target Exchange: {response2.metadata.get('target_exchange', 'unknown')}")

        # GateIO Neural Network Model -> GateIO Exchange
        print("\n📊 GateIO Neural Network Model → GateIO Exchange")
        response3 = await receiver.send_order_for_ml_model(
            symbol="ADAUSDT",
            side="buy",
            order_type="market",
            quantity=10.0,
            ml_model_id="gateio_neural_net"
        )
        print(f"   Response: {response3.success}")
        if hasattr(response3, 'metadata') and response3.metadata:
            print(f"   Target Exchange: {response3.metadata.get('target_exchange', 'unknown')}")

        # Test 2: Default ML exchange fallback
        print("\n🔄 Test 2: Default ML Exchange Fallback")
        print("-" * 40)

        print("📊 Unknown ML Model → Default Exchange (Binance)")
        response4 = await receiver.send_order_for_ml_model(
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            quantity=0.001,
            ml_model_id="unknown_model"  # This will use default exchange
        )
        print(f"   Response: {response4.success}")
        if hasattr(response4, 'metadata') and response4.metadata:
            print(f"   Target Exchange: {response4.metadata.get('target_exchange', 'unknown')}")

        # Test 3: Intelligent routing with ML models
        print("\n🧠 Test 3: Intelligent Routing with ML Models")
        print("-" * 40)

        # ML model routing strategy
        print("📊 Using ML Model Routing Strategy")
        response5 = await receiver.send_order_with_routing(
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            quantity=0.001,
            routing_strategy="ml_model",
            ml_model_id="binance_lstm_model"
        )
        print(f"   Response: {response5.success if isinstance(response5, object) and hasattr(response5, 'success') else 'Dict response'}")

        # Test 4: ML Model management
        print("\n⚙️ Test 4: ML Model Management")
        print("-" * 40)

        # Register new ML model
        print("📝 Registering new ML model...")
        success = receiver.register_ml_model_exchange("new_xgboost_model", "okx")
        print(f"   Registration successful: {success}")

        # Get ML model mappings
        print("📋 Current ML Model Mappings:")
        mappings = receiver.get_all_ml_model_exchanges()
        for ml_model, exchange in mappings.items():
            print(f"   {ml_model} → {exchange}")

        # Models by exchange
        print("📊 Models Grouped by Exchange:")
        models_by_exchange = receiver._get_ml_models_by_exchange()
        for exchange, models in models_by_exchange.items():
            print(f"   {exchange}: {len(models)} models - {models}")

        # Set default ML exchange
        print("🔧 Setting default ML exchange to OKX...")
        success = receiver.set_default_ml_exchange("okx")
        print(f"   Set successful: {success}")

        # Test 5: Statistics with ML model information
        print("\n📈 Test 5: ML Model Statistics")
        print("-" * 40)

        stats = await receiver.get_statistics()
        print("📊 ML Model Statistics:")
        print(f"   Registered ML models: {stats['ml_model']['registered_ml_models']}")
        print(f"   Default ML exchange: {stats['ml_model']['default_ml_exchange']}")
        print(f"   ML model to exchange mappings: {stats['ml_model']['ml_model_exchanges']}")
        print(f"   Models by exchange: {stats['ml_model']['ml_models_by_exchange']}")

        # Test 6: Error handling for invalid ML models
        print("\n⚠️ Test 6: Error Handling")
        print("-" * 40)

        print("📊 Invalid ML Model Test")
        invalid_response = await receiver.send_order_for_ml_model(
            symbol="BTCUSDT",
            side="buy",
            order_type="market",
            quantity=0.001,
            ml_model_id="nonexistent_model"  # This will use default
        )
        print(f"   Response: {invalid_response.success}")
        if hasattr(invalid_response, 'metadata') and invalid_response.metadata:
            print(f"   Target Exchange: {invalid_response.metadata.get('target_exchange', 'unknown')}")

        await receiver.stop()
        print("✅ Trading receiver stopped successfully")

    except Exception as e:
        print(f"❌ Error in ML model routing example: {e}")
        import traceback
        traceback.print_exc()

async def demonstrate_ml_model_signals():
    """Demonstrate how ML model signals are routed to specific exchanges"""
    print("\n🤖 ML Model Signal Routing Demonstration")
    print("=" * 50)

    receiver = await create_trading_receiver_with_ml_models()

    try:
        print("📨 Simulating ML model trading signals...")

        # Simulate different ML models sending trading signals
        ml_models = [
            {
                "id": "binance_prophet_model",
                "name": "Binance Prophet Model",
                "exchange": "binance",
                "symbol": "BTCUSDT",
                "signal": "BUY"
            },
            {
                "id": "okx_random_forest",
                "name": "OKX Random Forest",
                "exchange": "okx",
                "symbol": "ETHUSDT",
                "signal": "SELL"
            },
            {
                "id": "gateio_neural_net",
                "name": "GateIO Neural Network",
                "exchange": "gateio",
                "symbol": "ADAUSDT",
                "signal": "BUY"
            }
        ]

        for i, model_info in enumerate(ml_models, 1):
            print(f"\n📊 Signal {i}: {model_info['name']}")
            print(f"   ML Model: {model_info['id']}")
            print(f"   Expected Exchange: {model_info['exchange']}")
            print(f"   Symbol: {model_info['symbol']}")
            print(f"   Signal: {model_info['signal']}")

            # Send order using ML model routing
            response = await receiver.send_order_for_ml_model(
                symbol=model_info['symbol'],
                side=model_info['signal'].lower(),
                order_type="market",
                quantity=0.001,
                ml_model_id=model_info['id']
            )

            print(f"   ✅ Order Sent: {response.success}")
            if hasattr(response, 'metadata') and response.metadata:
                actual_exchange = response.metadata.get('target_exchange', 'unknown')
                print(f"   🎯 Target Exchange: {actual_exchange}")
                print(f"   ✅ Correct Exchange: {actual_exchange == model_info['exchange']}")

        await receiver.stop()
        print("✅ ML model signal routing demonstration completed")

    except Exception as e:
        print(f"❌ Error in ML model signal demonstration: {e}")
        await receiver.stop()

async def demonstrate_configuration_management():
    """Demonstrate ML model configuration management"""
    print("\n⚙️ ML Model Configuration Management")
    print("=" * 50)

    receiver = await create_trading_receiver_with_ml_models()

    try:
        print("📋 Initial Configuration:")
        print(f"   Default ML Exchange: {receiver.default_ml_exchange}")
        print(f"   ML Model Mappings: {receiver.get_all_ml_model_exchanges()}")

        # Demonstrate dynamic configuration changes
        print("\n🔄 Dynamic Configuration Changes:")

        # Change default exchange
        print("   Setting default ML exchange to OKX...")
        receiver.set_default_ml_exchange("okx")
        print(f"   ✅ New default: {receiver.default_ml_exchange}")

        # Register new model
        print("   Registering new ML model 'test_model' → GateIO...")
        receiver.register_ml_model_exchange("test_model", "gateio")
        print(f"   ✅ Registered: {receiver.get_ml_model_exchange('test_model')}")

        # Unregister a model
        print("   Unregistering 'test_model'...")
        receiver.unregister_ml_model_exchange("test_model")
        print(f"   ✅ Unregistered: {receiver.get_ml_model_exchange('test_model')}")

        # Show final configuration
        print("\n📋 Final Configuration:")
        print(f"   Default ML Exchange: {receiver.default_ml_exchange}")
        print(f"   ML Model Mappings: {receiver.get_all_ml_model_exchanges()}")
        print(f"   Models by Exchange: {receiver._get_ml_models_by_exchange()}")

        await receiver.stop()
        print("✅ Configuration management demonstration completed")

    except Exception as e:
        print(f"❌ Error in configuration management: {e}")
        await receiver.stop()

async def main():
    """Main demonstration function"""
    print("🚀 ML Model Exchange Routing System")
    print("=" * 60)

    try:
        # Run all demonstrations
        await demonstrate_ml_model_routing()
        print("\n" + "=" * 60)

        await demonstrate_ml_model_signals()
        print("\n" + "=" * 60)

        await demonstrate_configuration_management()
        print("\n" + "=" * 60)

        print("🎉 All demonstrations completed successfully!")
        print("\n📚 Key Features Demonstrated:")
        print("   ✅ ML model-specific exchange routing")
        print("   ✅ Default exchange fallback for unknown models")
        print("   ✅ Dynamic ML model registration/unregistration")
        print("   ✅ ML model configuration management")
        print("   ✅ Statistics and monitoring for ML models")
        print("   ✅ Error handling for invalid ML models")

        print("\n🔑 Key Benefits:")
        print("   🎯 Orders sent ONLY to ML model-associated exchanges")
        print("   🔄 No unnecessary broadcasting to all exchanges")
        print("   📊 Each ML model uses data from its specific exchange")
        print("   ⚙️ Easy configuration and management of ML model mappings")
        print("   📈 Comprehensive monitoring and statistics")

    except Exception as e:
        print(f"❌ Fatal error in demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())