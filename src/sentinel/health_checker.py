"""
Comprehensive health check service for all Ares components.
Integrates with sentinel monitoring and prometheus metrics.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import psutil
import logging

from src.utils.prometheus_metrics import metrics
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger


class ComponentHealthChecker:
    """Health checker for individual components."""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = system_logger.getChild(f"HealthChecker.{component_name}")
        self.start_time = time.time()
        self.last_check = None
        self.check_count = 0
        self.error_count = 0
        
    def get_uptime(self) -> float:
        """Get component uptime in seconds."""
        return time.time() - self.start_time
        
    async def check_health(self, component_instance: Any = None) -> Dict[str, Any]:
        """Perform health check on component."""
        start_time = time.time()
        self.check_count += 1
        
        try:
            health_data = {
                "component": self.component_name,
                "status": "healthy",
                "health_score": 100.0,
                "uptime_seconds": self.get_uptime(),
                "last_check": datetime.now().isoformat(),
                "check_count": self.check_count,
                "error_count": self.error_count
            }
            
            # Component-specific health checks
            if hasattr(component_instance, 'get_health_status'):
                component_health = component_instance.get_health_status()
                health_data.update(component_health)
            elif hasattr(component_instance, 'is_running'):
                health_data["is_running"] = component_instance.is_running
                if not component_instance.is_running:
                    health_data["status"] = "warning"
                    health_data["health_score"] = 70.0
            
            # Record metrics
            duration = time.time() - start_time
            metrics.record_health_check(
                component=self.component_name,
                status=health_data["status"],
                health_score=health_data["health_score"],
                duration=duration,
                alert_counts=health_data.get("alert_counts")
            )
            metrics.record_component_uptime(self.component_name, self.get_uptime())
            
            self.last_check = datetime.now()
            return health_data
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Health check failed for {self.component_name}: {e}")
            
            error_health = {
                "component": self.component_name,
                "status": "error",
                "health_score": 0.0,
                "error": str(e),
                "uptime_seconds": self.get_uptime(),
                "last_check": datetime.now().isoformat(),
                "check_count": self.check_count,
                "error_count": self.error_count
            }
            
            # Record error metrics
            duration = time.time() - start_time
            metrics.record_health_check(
                component=self.component_name,
                status="error",
                health_score=0.0,
                duration=duration
            )
            
            return error_health


class SystemHealthChecker:
    """Comprehensive system health checker."""
    
    def __init__(self):
        self.logger = system_logger.getChild("SystemHealthChecker")
        self.component_checkers: Dict[str, ComponentHealthChecker] = {}
        self.registered_components: Dict[str, Any] = {}
        self.system_start_time = time.time()
        
    def register_component(self, name: str, component_instance: Any = None):
        """Register a component for health checking."""
        self.component_checkers[name] = ComponentHealthChecker(name)
        self.registered_components[name] = component_instance
        self.logger.info(f"Registered component for health checking: {name}")
        
    def unregister_component(self, name: str):
        """Unregister a component from health checking."""
        if name in self.component_checkers:
            del self.component_checkers[name]
        if name in self.registered_components:
            del self.registered_components[name]
        self.logger.info(f"Unregistered component: {name}")
        
    async def check_component_health(self, component_name: str) -> Dict[str, Any]:
        """Check health of a specific component."""
        if component_name not in self.component_checkers:
            return {
                "component": component_name,
                "status": "error",
                "error": "Component not registered",
                "health_score": 0.0
            }
            
        checker = self.component_checkers[component_name]
        component_instance = self.registered_components.get(component_name)
        return await checker.check_health(component_instance)
        
    async def check_all_components(self) -> Dict[str, Any]:
        """Check health of all registered components."""
        component_results = {}
        
        # Check each component
        for component_name in self.component_checkers:
            component_results[component_name] = await self.check_component_health(component_name)
            
        # Calculate overall system health
        overall_health = self._calculate_overall_health(component_results)
        
        return {
            "system": overall_health,
            "components": component_results,
            "timestamp": datetime.now().isoformat()
        }
        
    def _calculate_overall_health(self, component_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall system health based on component health."""
        if not component_results:
            return {
                "status": "unknown",
                "health_score": 0.0,
                "message": "No components registered"
            }
            
        # Calculate weighted average health score
        total_score = 0.0
        total_components = 0
        critical_components = 0
        degraded_components = 0
        
        for component_name, health_data in component_results.items():
            score = health_data.get("health_score", 0.0)
            status = health_data.get("status", "error")
            
            total_score += score
            total_components += 1
            
            if status in ["error", "critical"]:
                critical_components += 1
            elif status in ["degraded", "warning"]:
                degraded_components += 1
                
        avg_score = total_score / total_components if total_components > 0 else 0.0
        
        # Determine overall status
        if critical_components > 0:
            status = "critical"
        elif degraded_components > total_components * 0.5:  # More than 50% degraded
            status = "degraded"
        elif avg_score >= 90:
            status = "healthy"
        elif avg_score >= 70:
            status = "warning"
        else:
            status = "degraded"
            
        return {
            "status": status,
            "health_score": round(avg_score, 2),
            "uptime_seconds": time.time() - self.system_start_time,
            "total_components": total_components,
            "critical_components": critical_components,
            "degraded_components": degraded_components,
            "healthy_components": total_components - critical_components - degraded_components
        }
        
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        try:
            # Get system resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Get process-specific metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                "cpu": {
                    "usage_percent": cpu_percent,
                    "count": psutil.cpu_count(),
                    "count_logical": psutil.cpu_count(logical=True)
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent,
                    "process_rss": process_memory.rss,
                    "process_vms": process_memory.vms
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                },
                "network": {
                    "connections": len(psutil.net_connections()),
                    "io_counters": psutil.net_io_counters()._asdict() if psutil.net_io_counters() else {}
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
    @handle_errors(exceptions=(Exception,), default_return={}, context="health check summary")
    async def get_health_summary(self) -> Dict[str, Any]:
        """Get a comprehensive health summary."""
        try:
            # Get component health
            health_check = await self.check_all_components()
            
            # Get system metrics
            system_metrics = await self.get_system_metrics()
            
            # Combine into summary
            return {
                "health": health_check,
                "system_metrics": system_metrics,
                "registered_components": list(self.component_checkers.keys()),
                "summary": {
                    "overall_status": health_check["system"]["status"],
                    "overall_score": health_check["system"]["health_score"],
                    "component_count": len(self.component_checkers),
                    "system_uptime": time.time() - self.system_start_time,
                    "cpu_usage": system_metrics.get("cpu", {}).get("usage_percent", 0),
                    "memory_usage": system_metrics.get("memory", {}).get("percent", 0),
                    "disk_usage": system_metrics.get("disk", {}).get("percent", 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error generating health summary: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }


# Global health checker instance
health_checker = SystemHealthChecker()


# Component-specific health check implementations
class AnalystHealthMixin:
    """Health check mixin for Analyst components."""
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get analyst health status."""
        try:
            health_score = 100.0
            status = "healthy"
            issues = []
            
            # Check if analyst is initialized
            if not getattr(self, '_initialized', False):
                health_score -= 50
                status = "critical"
                issues.append("Not initialized")
                
            # Check if required models are loaded
            if hasattr(self, 'models') and not self.models:
                health_score -= 30
                status = "degraded" if status == "healthy" else status
                issues.append("No models loaded")
                
            # Check data freshness
            if hasattr(self, 'last_data_update'):
                time_since_update = time.time() - getattr(self, 'last_data_update', 0)
                if time_since_update > 300:  # 5 minutes
                    health_score -= 20
                    status = "warning" if status == "healthy" else status
                    issues.append("Stale data")
                    
            return {
                "component": "analyst",
                "status": status,
                "health_score": max(0, health_score),
                "issues": issues,
                "models_loaded": len(getattr(self, 'models', [])),
                "last_analysis": getattr(self, 'last_analysis_time', None)
            }
            
        except Exception as e:
            return {
                "component": "analyst",
                "status": "error",
                "health_score": 0,
                "error": str(e)
            }


class StrategistHealthMixin:
    """Health check mixin for Strategist components."""
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get strategist health status."""
        try:
            health_score = 100.0
            status = "healthy" 
            issues = []
            
            # Check if strategist is running
            if not getattr(self, 'is_running', True):
                health_score -= 40
                status = "critical"
                issues.append("Not running")
                
            # Check strategy parameters
            if hasattr(self, 'strategy_config') and not self.strategy_config:
                health_score -= 30
                status = "degraded" if status == "healthy" else status
                issues.append("No strategy configuration")
                
            # Check recent decisions
            if hasattr(self, 'last_decision_time'):
                time_since_decision = time.time() - getattr(self, 'last_decision_time', 0)
                if time_since_decision > 600:  # 10 minutes
                    health_score -= 15
                    status = "warning" if status == "healthy" else status
                    issues.append("No recent decisions")
                    
            return {
                "component": "strategist",
                "status": status,
                "health_score": max(0, health_score),
                "issues": issues,
                "is_running": getattr(self, 'is_running', False),
                "last_decision": getattr(self, 'last_decision_time', None)
            }
            
        except Exception as e:
            return {
                "component": "strategist", 
                "status": "error",
                "health_score": 0,
                "error": str(e)
            }


class TacticianHealthMixin:
    """Health check mixin for Tactician components."""
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get tactician health status."""
        try:
            health_score = 100.0
            status = "healthy"
            issues = []
            
            # Check if tactician is active
            if not getattr(self, 'is_active', True):
                health_score -= 40
                status = "critical" 
                issues.append("Not active")
                
            # Check order management
            if hasattr(self, 'pending_orders'):
                pending_count = len(getattr(self, 'pending_orders', []))
                if pending_count > 10:  # Too many pending orders
                    health_score -= 20
                    status = "warning" if status == "healthy" else status
                    issues.append(f"High pending orders: {pending_count}")
                    
            # Check execution errors
            if hasattr(self, 'execution_errors'):
                error_count = getattr(self, 'execution_errors', 0)
                if error_count > 5:
                    health_score -= 25
                    status = "degraded" if status == "healthy" else status
                    issues.append(f"Recent execution errors: {error_count}")
                    
            # Check last execution
            if hasattr(self, 'last_execution_time'):
                time_since_execution = time.time() - getattr(self, 'last_execution_time', 0)
                if time_since_execution > 1800:  # 30 minutes
                    health_score -= 10
                    status = "warning" if status == "healthy" else status
                    issues.append("No recent executions")
                    
            return {
                "component": "tactician",
                "status": status,
                "health_score": max(0, health_score),
                "issues": issues,
                "is_active": getattr(self, 'is_active', False),
                "pending_orders": len(getattr(self, 'pending_orders', [])),
                "execution_errors": getattr(self, 'execution_errors', 0),
                "last_execution": getattr(self, 'last_execution_time', None)
            }
            
        except Exception as e:
            return {
                "component": "tactician",
                "status": "error", 
                "health_score": 0,
                "error": str(e)
            }


class ExchangeHealthMixin:
    """Health check mixin for Exchange components."""
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get exchange health status."""
        try:
            health_score = 100.0
            status = "healthy"
            issues = []
            
            # Check connection status
            if not getattr(self, 'is_connected', False):
                health_score -= 60
                status = "critical"
                issues.append("Not connected")
                
            # Check API rate limits
            if hasattr(self, 'rate_limit_remaining'):
                remaining = getattr(self, 'rate_limit_remaining', 100)
                if remaining < 10:
                    health_score -= 30
                    status = "warning" if status == "healthy" else status
                    issues.append(f"Low rate limit: {remaining}")
                    
            # Check recent API errors
            if hasattr(self, 'api_errors'):
                error_count = getattr(self, 'api_errors', 0)
                if error_count > 3:
                    health_score -= 25
                    status = "degraded" if status == "healthy" else status
                    issues.append(f"Recent API errors: {error_count}")
                    
            # Check latency
            if hasattr(self, 'avg_latency'):
                latency = getattr(self, 'avg_latency', 0)
                if latency > 1000:  # 1 second
                    health_score -= 20
                    status = "warning" if status == "healthy" else status
                    issues.append(f"High latency: {latency}ms")
                    
            return {
                "component": f"exchange_{getattr(self, 'exchange_name', 'unknown')}",
                "status": status,
                "health_score": max(0, health_score),
                "issues": issues,
                "is_connected": getattr(self, 'is_connected', False),
                "rate_limit_remaining": getattr(self, 'rate_limit_remaining', None),
                "api_errors": getattr(self, 'api_errors', 0),
                "avg_latency": getattr(self, 'avg_latency', None)
            }
            
        except Exception as e:
            return {
                "component": "exchange",
                "status": "error",
                "health_score": 0,
                "error": str(e)
            }