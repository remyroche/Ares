"""
Integration tests for System Coordinator functionality.

These tests verify the system-level coordination, health monitoring,
and recovery mechanisms.
"""

import asyncio
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.supervisor.coordinator import (
    CircuitBreaker,
    ComponentMonitor,
    HealthMonitor,
    OnlineLearningManager,
    RecoveryManager,
    SystemCoordinator,
)


class TestSystemCoordinatorIntegration:
    """Test suite for system coordinator integration."""

    @pytest.fixture
    def mock_config(self) -> Dict[str, Any]:
        """Provide test configuration."""
        return {
            "supervisor": {
                "supervision_interval": 1,
                "max_history": 10,
                "circuit_breaker_threshold": 3,
                "circuit_breaker_timeout": 5,
                "online_learning": {
                    "learning_rate": 0.1,
                    "min_weight": 0.1,
                    "max_weight": 0.8,
                },
                "max_recovery_attempts": 3,
                "recovery_cooldown": 5,
            }
        }

    @pytest.mark.asyncio
    async def test_system_coordinator_initialization(self, mock_config):
        """Test that system coordinator initializes properly."""
        coordinator = SystemCoordinator(mock_config)
        
        # Verify sub-components are created
        assert isinstance(coordinator.online_learning, OnlineLearningManager)
        assert isinstance(coordinator.component_monitor, ComponentMonitor)
        assert isinstance(coordinator.health_monitor, HealthMonitor)
        assert isinstance(coordinator.recovery_manager, RecoveryManager)
        
        # Initialize coordinator
        success = await coordinator.initialize()
        assert success
        assert coordinator.is_initialized

    @pytest.mark.asyncio
    async def test_circuit_breaker_functionality(self, mock_config):
        """Test circuit breaker behavior."""
        breaker = CircuitBreaker(failure_threshold=3, timeout=1)
        
        # Successful calls should work
        async def success_func():
            return "success"
        
        result = await breaker.call(success_func)
        assert result == "success"
        assert breaker.state == "CLOSED"
        
        # Failing calls should trigger circuit breaker
        async def failing_func():
            raise Exception("Test failure")
        
        # First 2 failures - circuit stays closed
        for i in range(2):
            with pytest.raises(Exception):
                await breaker.call(failing_func)
        assert breaker.state == "CLOSED"
        
        # Third failure opens circuit
        with pytest.raises(Exception):
            await breaker.call(failing_func)
        assert breaker.state == "OPEN"
        
        # Further calls should fail immediately
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await breaker.call(success_func)

    @pytest.mark.asyncio
    async def test_online_learning_weight_updates(self, mock_config):
        """Test online learning weight updates based on performance."""
        learning_manager = OnlineLearningManager(mock_config["supervisor"]["online_learning"])
        
        # Update performance for multiple models
        await learning_manager.update_model_performance("model_a", 0.8)
        await learning_manager.update_model_performance("model_b", 0.6)
        await learning_manager.update_model_performance("model_c", 0.4)
        
        # Check weights are calculated
        weights = learning_manager.get_model_weights()
        assert len(weights) == 3
        assert sum(weights.values()) == pytest.approx(1.0)
        assert weights["model_a"] > weights["model_b"] > weights["model_c"]

    @pytest.mark.asyncio
    async def test_component_monitoring(self, mock_config):
        """Test component monitoring functionality."""
        monitor = ComponentMonitor(mock_config["supervisor"])
        
        # Create mock components
        mock_analyst = Mock()
        mock_analyst.is_analyzing = True
        mock_analyst.analysis_history = [1, 2, 3]
        mock_analyst.analysis_results = {
            "ml_confidence": 0.85,
            "regime": "bullish",
            "timestamp": "2024-01-01T00:00:00"
        }
        
        # Monitor analyst
        features = monitor.monitor_analyst_features(mock_analyst)
        assert features["component"] == "analyst"
        assert features["is_analyzing"] == True
        assert features["analysis_count"] == 3
        assert features["model_confidence"] == 0.85
        assert features["regime"] == "bullish"
        
        # Check status
        status = monitor.get_component_status("analyst")
        assert status["status"] == "active"
        assert status["history_size"] == 1

    @pytest.mark.asyncio
    async def test_health_monitoring(self, mock_config):
        """Test health monitoring functionality."""
        monitor = HealthMonitor(mock_config["supervisor"])
        
        # Mock psutil functions
        with patch('psutil.cpu_percent', return_value=75.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            # Setup mock returns
            mock_memory.return_value = Mock(percent=80.0, available=1024*1024*1024)
            mock_disk.return_value = Mock(percent=60.0, free=100*1024*1024*1024)
            
            # Check health
            health_data = await monitor.check_system_health()
            
            assert health_data["overall_status"] == "healthy"
            assert health_data["resource_usage"]["cpu_percent"] == 75.0
            assert health_data["resource_usage"]["memory_percent"] == 80.0
            assert len(health_data["warnings"]) == 0
            assert len(health_data["errors"]) == 0

    @pytest.mark.asyncio
    async def test_health_monitoring_warnings(self, mock_config):
        """Test health monitoring with warnings."""
        monitor = HealthMonitor(mock_config["supervisor"])
        
        # Mock high resource usage
        with patch('psutil.cpu_percent', return_value=85.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_memory.return_value = Mock(percent=86.0, available=512*1024*1024)
            mock_disk.return_value = Mock(percent=70.0, free=50*1024*1024*1024)
            
            health_data = await monitor.check_system_health()
            
            assert health_data["overall_status"] == "warning"
            assert len(health_data["warnings"]) > 0
            assert any("CPU" in w for w in health_data["warnings"])
            assert any("memory" in w for w in health_data["warnings"])

    @pytest.mark.asyncio
    async def test_recovery_manager(self, mock_config):
        """Test recovery manager functionality."""
        recovery_manager = RecoveryManager(mock_config["supervisor"])
        
        # Test recovery attempt
        error_details = {"error_type": "connection_error", "message": "Connection lost"}
        success = await recovery_manager.handle_component_failure("test_component", error_details)
        
        # First attempt should try restart
        assert recovery_manager.recovery_attempts["test_component"] == 1
        assert success  # Mocked recovery succeeds
        
        # Check recovery status
        status = recovery_manager.get_recovery_status()
        assert status["total_recovery_attempts"] == 1
        assert len(status["recent_recoveries"]) == 1

    @pytest.mark.asyncio
    async def test_recovery_manager_max_attempts(self, mock_config):
        """Test recovery manager respects max attempts."""
        recovery_manager = RecoveryManager(mock_config["supervisor"])
        recovery_manager.max_recovery_attempts = 2
        
        # Exhaust recovery attempts
        for i in range(2):
            await recovery_manager.handle_component_failure(
                "failing_component",
                {"error_type": "critical", "message": f"Failure {i+1}"}
            )
        
        # Next attempt should fail
        success = await recovery_manager.handle_component_failure(
            "failing_component",
            {"error_type": "critical", "message": "Final failure"}
        )
        assert not success
        assert recovery_manager.recovery_attempts["failing_component"] == 2

    @pytest.mark.asyncio
    async def test_system_coordinator_component_registration(self, mock_config):
        """Test component registration in system coordinator."""
        coordinator = SystemCoordinator(mock_config)
        await coordinator.initialize()
        
        # Register mock components
        mock_analyst = Mock()
        mock_strategist = Mock()
        
        coordinator.register_component("analyst", mock_analyst)
        coordinator.register_component("strategist", mock_strategist)
        
        assert len(coordinator.components) == 2
        assert coordinator.components["analyst"] == mock_analyst
        assert coordinator.components["strategist"] == mock_strategist

    @pytest.mark.asyncio
    async def test_system_coordinator_supervision_cycle(self, mock_config):
        """Test a complete supervision cycle."""
        coordinator = SystemCoordinator(mock_config)
        await coordinator.initialize()
        
        # Register mock components
        mock_analyst = Mock()
        mock_analyst.get_performance_metrics = Mock(return_value={"performance_score": 0.85})
        
        coordinator.register_component("analyst", mock_analyst)
        
        # Perform one supervision cycle
        await coordinator._perform_supervision()
        
        # Check status was updated
        assert coordinator.status["is_running"] == False  # Not started via run()
        assert "timestamp" in coordinator.status
        assert coordinator.status["active_components"] == 1
        
        # Check history was updated
        assert len(coordinator.history) == 1

    @pytest.mark.asyncio
    async def test_component_coordination(self, mock_config):
        """Test component coordination functionality."""
        coordinator = SystemCoordinator(mock_config)
        await coordinator.initialize()
        
        # Create mock components with coordination attributes
        mock_analyst = Mock()
        mock_analyst.regime_classifier = True
        mock_analyst.regime_info = {"regime": "trending", "confidence": 0.9}
        
        mock_strategist = Mock()
        mock_strategist.current_regime = None
        mock_strategist.regime_confidence = None
        
        coordinator.register_component("analyst", mock_analyst)
        coordinator.register_component("strategist", mock_strategist)
        
        # Coordinate components
        await coordinator._coordinate_analyst_strategist()
        
        # Verify information was shared
        assert mock_strategist.current_regime == "trending"
        assert mock_strategist.regime_confidence == 0.9


if __name__ == "__main__":
    pytest.main([__file__, "-v"])