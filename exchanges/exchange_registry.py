"""
Exchange Registry

Manages exchange instances and provides centralized access to exchanges.
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

from .factory import ExchangeFactory


class ExchangeRegistry:
    """Registry for managing exchange instances"""
    
    def __init__(self):
        self.exchanges: Dict[str, Any] = {}
        self.exchange_configs: Dict[str, Dict[str, Any]] = {}
        self.exchange_status: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Health monitoring
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Statistics
        self.registry_stats = {
            "total_registered": 0,
            "active_exchanges": 0,
            "failed_exchanges": 0,
            "health_checks": 0,
            "health_check_failures": 0
        }
    
    async def start(self) -> None:
        """Start exchange registry"""
        if self._running:
            return
            
        self._running = True
        self._health_check_task = asyncio.create_task(self._monitor_exchange_health())
        self.logger.info("Exchange registry started")
    
    async def stop(self) -> None:
        """Stop exchange registry"""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all exchange connections
        await self.close_all()
        
        self.logger.info("Exchange registry stopped")
    
    async def register_exchange(self, exchange_name: str, exchange_instance: Any) -> bool:
        """Register an exchange instance"""
        try:
            # Initialize exchange if needed
            if hasattr(exchange_instance, '_initialize_exchange'):
                await exchange_instance._initialize_exchange()
            
            # Store exchange instance
            self.exchanges[exchange_name] = exchange_instance
            
            # Initialize status tracking
            self.exchange_status[exchange_name] = {
                "status": "active",
                "last_health_check": datetime.now(),
                "health_status": "healthy",
                "error_count": 0,
                "last_error": None,
                "registered_at": datetime.now()
            }
            
            # Update statistics
            self.registry_stats["total_registered"] += 1
            self.registry_stats["active_exchanges"] += 1
            
            self.logger.info(f"Exchange registered: {exchange_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register exchange {exchange_name}: {e}")
            
            # Mark as failed
            self.exchange_status[exchange_name] = {
                "status": "failed",
                "last_health_check": datetime.now(),
                "health_status": "unhealthy",
                "error_count": 1,
                "last_error": str(e),
                "registered_at": datetime.now()
            }
            
            self.registry_stats["failed_exchanges"] += 1
            return False
    
    async def unregister_exchange(self, exchange_name: str) -> bool:
        """Unregister an exchange"""
        try:
            if exchange_name in self.exchanges:
                # Close exchange connection
                exchange = self.exchanges[exchange_name]
                if hasattr(exchange, 'close'):
                    await exchange.close()
                
                # Remove from registry
                del self.exchanges[exchange_name]
                del self.exchange_status[exchange_name]
                
                # Update statistics
                self.registry_stats["active_exchanges"] -= 1
                
                self.logger.info(f"Exchange unregistered: {exchange_name}")
                return True
            else:
                self.logger.warning(f"Exchange not found: {exchange_name}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error unregistering exchange {exchange_name}: {e}")
            return False
    
    async def get_exchange(self, exchange_name: str) -> Optional[Any]:
        """Get exchange instance by name"""
        exchange = self.exchanges.get(exchange_name)
        
        if exchange:
            # Check if exchange is healthy
            status = self.exchange_status.get(exchange_name, {})
            if status.get("health_status") == "unhealthy":
                self.logger.warning(f"Exchange {exchange_name} is unhealthy")
                return None
            
            return exchange
        
        return None
    
    async def get_active_exchanges(self) -> List[str]:
        """Get list of active exchange names"""
        active_exchanges = []
        
        for exchange_name, status in self.exchange_status.items():
            if status.get("status") == "active" and status.get("health_status") == "healthy":
                active_exchanges.append(exchange_name)
        
        return active_exchanges
    
    async def get_registered_exchanges(self) -> List[str]:
        """Get list of all registered exchange names"""
        return list(self.exchanges.keys())
    
    async def get_exchange_status(self, exchange_name: str) -> Optional[Dict[str, Any]]:
        """Get exchange status information"""
        return self.exchange_status.get(exchange_name)
    
    async def get_all_exchange_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all exchanges"""
        return dict(self.exchange_status)
    
    async def health_check_exchange(self, exchange_name: str) -> bool:
        """Perform health check on specific exchange"""
        try:
            exchange = self.exchanges.get(exchange_name)
            if not exchange:
                return False
            
            # Perform simple health check
            # Try to get ticker data or similar
            if hasattr(exchange, 'get_ticker'):
                await exchange.get_ticker("BTCUSDT")
            elif hasattr(exchange, 'get_account_info'):
                await exchange.get_account_info()
            else:
                # Basic connectivity check
                return True
            
            # Update health status
            status = self.exchange_status.get(exchange_name, {})
            status["health_status"] = "healthy"
            status["last_health_check"] = datetime.now()
            status["error_count"] = 0
            status["last_error"] = None
            
            self.registry_stats["health_checks"] += 1
            
            return True
            
        except Exception as e:
            # Update health status
            status = self.exchange_status.get(exchange_name, {})
            status["health_status"] = "unhealthy"
            status["last_health_check"] = datetime.now()
            status["error_count"] = status.get("error_count", 0) + 1
            status["last_error"] = str(e)
            
            self.registry_stats["health_check_failures"] += 1
            
            self.logger.warning(f"Health check failed for {exchange_name}: {e}")
            return False
    
    async def _monitor_exchange_health(self) -> None:
        """Monitor health of all registered exchanges"""
        while self._running:
            try:
                # Check health of all exchanges
                for exchange_name in list(self.exchanges.keys()):
                    await self.health_check_exchange(exchange_name)
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def close_all(self) -> None:
        """Close all exchange connections"""
        for exchange_name, exchange in self.exchanges.items():
            try:
                if hasattr(exchange, 'close'):
                    await exchange.close()
                self.logger.info(f"Closed connection to {exchange_name}")
            except Exception as e:
                self.logger.error(f"Error closing connection to {exchange_name}: {e}")
        
        self.exchanges.clear()
        self.exchange_status.clear()
        self.registry_stats["active_exchanges"] = 0
    
    async def restart_exchange(self, exchange_name: str) -> bool:
        """Restart an exchange connection"""
        try:
            # Get exchange configuration
            config = self.exchange_configs.get(exchange_name)
            if not config:
                self.logger.error(f"No configuration found for exchange {exchange_name}")
                return False
            
            # Unregister current instance
            await self.unregister_exchange(exchange_name)
            
            # Create new instance
            exchange_instance = ExchangeFactory.get_exchange(exchange_name)
            
            # Register new instance
            success = await self.register_exchange(exchange_name, exchange_instance)
            
            if success:
                self.logger.info(f"Exchange restarted: {exchange_name}")
            else:
                self.logger.error(f"Failed to restart exchange: {exchange_name}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error restarting exchange {exchange_name}: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "running": self._running,
            "statistics": self.registry_stats,
            "registered_exchanges": len(self.exchanges),
            "active_exchanges": len([s for s in self.exchange_status.values() 
                                   if s.get("status") == "active" and s.get("health_status") == "healthy"]),
            "unhealthy_exchanges": len([s for s in self.exchange_status.values() 
                                      if s.get("health_status") == "unhealthy"]),
            "timestamp": datetime.now().isoformat()
        }
    
    async def set_exchange_config(self, exchange_name: str, config: Dict[str, Any]) -> None:
        """Set configuration for an exchange"""
        self.exchange_configs[exchange_name] = config
    
    async def get_exchange_config(self, exchange_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for an exchange"""
        return self.exchange_configs.get(exchange_name)