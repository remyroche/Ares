#!/usr/bin/env python3
"""Test ExchangeInterface directly"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.trading.execution.exchange_interface import ExchangeInterface


async def main():
    print("🧪 Testing ExchangeInterface...")
    
    # Create config
    exchange_config = {
        'exchange_type': 'binance',
        'api_key': None,
        'api_secret': None,
        'testnet': False,
        'rate_limits': {}
    }
    
    print(f"📝 Config: {exchange_config}")
    
    # Create interface
    interface = ExchangeInterface(exchange_config)
    print(f"✅ Interface created")
    
    # Try to connect
    print(f"🔗 Attempting to connect...")
    try:
        result = await interface.connect()
        print(f"✅ Connection result: {result}")
        print(f"   Status: {interface.connection_status}")
        print(f"   Dispatcher: {interface.dispatcher}")
        
        if interface.dispatcher:
            print(f"   Dispatcher initialized: {interface.dispatcher._initialized}")
            print(f"   Exchange: {interface.dispatcher.exchange}")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Disconnect
    await interface.disconnect()
    print(f"👋 Disconnected")
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)

