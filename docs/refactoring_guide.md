# Refactoring Guide: Supervisor-Strategist Separation

## Overview

This guide provides specific code changes needed to implement the separation between Supervisor and Strategist components, eliminating overlap and establishing clear boundaries.

## Phase 1: Remove Overlapping Methods from Supervisor

### 1.1 Remove Position Sizing Logic

**File**: `src/supervisor/supervisor.py`

**Remove these methods:**
```python
# REMOVE: Lines 692-697
def _tactician_calculate_position_size(self, confidence: float, leverage: float) -> float:
    """Calculate position size based on confidence and leverage."""
    base_size = confidence * 100  # Base size as percentage
    adjusted_size = base_size * leverage
    return min(adjusted_size, 100.0)  # Cap at 100%

# REMOVE: Lines 679-691
def _tactician_calculate_leverage(self, confidence: float) -> float:
    """Calculate leverage based on confidence."""
    # Move this logic to Tactician component
    pass

# REMOVE: Lines 698-711
def _tactician_calculate_entry_timing(self, market_data: pd.DataFrame, confidence: float) -> str:
    """Calculate optimal entry timing."""
    # Move this logic to Tactician component
    pass
```

**Update coordination method:**
```python
# UPDATE: Lines 1530-1557 in _coordinate_strategist_tactician()
async def _coordinate_strategist_tactician(self) -> None:
    """Coordinate between Strategist and Tactician using interface."""
    try:
        # Get strategist component
        strategist = self.components.get("strategist")
        if not strategist:
            self.logger.warning("Strategist component not available")
            return

        # Use interface to request strategy generation
        if hasattr(self, 'strategist_interface'):
            # Create strategy request
            strategy_request = StrategyRequest(
                market_data=self.current_market_data,
                current_price=self.current_price,
                analysis_results=self.analysis_results,
                system_status=self.get_system_status()
            )
            
            # Request strategy from strategist
            strategy_response = await self.strategist_interface.request_strategy_generation(strategy_request)
            
            # Handle strategy response
            if strategy_response and strategy_response.direction != "HOLD":
                # Pass strategy to tactician for execution
                await self._pass_strategy_to_tactician(strategy_response)
            else:
                self.logger.info("No actionable strategy generated")
                
    except Exception as e:
        self.logger.error(f"Error coordinating strategist-tactician: {e}")
```

### 1.2 Remove Strategy-Specific Risk Management

**File**: `src/supervisor/supervisor.py`

**Keep only portfolio-level risk management:**
```python
# KEEP: Lines 1917-1960 - _enforce_portfolio_guards()
# This is portfolio-level risk management, not strategy-specific

# REMOVE: Any strategy-specific risk calculations from coordination methods
# Strategy-specific risk should be handled by Strategist
```

**Update risk monitoring:**
```python
# UPDATE: Lines 1146-1225 in _calculate_analyst_risk_metrics()
async def _calculate_analyst_risk_metrics(self, ml_profit_data: dict[str, Any], barrier_analysis: dict[str, Any], market_data: pd.DataFrame) -> dict[str, Any]:
    """Calculate portfolio-level risk metrics, not strategy-specific."""
    try:
        # Focus on portfolio-level risk metrics
        risk_metrics = {
            "portfolio_risk_level": "MEDIUM",
            "max_drawdown": 0.0,
            "daily_var": 0.0,
            "portfolio_implications": {}
        }
        
        # Calculate portfolio implications
        if ml_profit_data and barrier_analysis:
            risk_metrics["portfolio_implications"] = {
                "profit_probability": ml_profit_data.get("profit_probability", 0.5),
                "barrier_risk": barrier_analysis.get("barrier_risk", 0.0),
                "portfolio_impact": "LOW"
            }
        
        return risk_metrics
        
    except Exception as e:
        self.logger.error(f"Error calculating portfolio risk metrics: {e}")
        return {"portfolio_risk_level": "HIGH", "error": str(e)}
```

### 1.3 Update Component Monitoring

**File**: `src/supervisor/supervisor.py`

**Update strategist monitoring to use interface:**
```python
# UPDATE: Lines 1315-1336 in _monitor_strategist_features()
def _monitor_strategist_features(self) -> None:
    """Monitor strategist features using interface."""
    try:
        strategist = self.components.get("strategist")
        if not strategist:
            return

        # Use interface to get strategist status
        if hasattr(self, 'strategist_interface'):
            # Request system status for strategist
            status_request = SystemStatusRequest(
                component_name="strategist",
                include_performance=True,
                include_health=True
            )
            
            # This would be handled by the interface
            # status_response = await self.strategist_interface.request_system_status(status_request)
            
            # Monitor strategist health and performance
            self.logger.debug("Strategist monitoring completed via interface")
            
    except Exception as e:
        self.logger.error(f"Error monitoring strategist features: {e}")
```

## Phase 2: Remove System-Level Functionality from Strategist

### 2.1 Remove Portfolio-Level Allocation

**File**: `src/strategist/volatility_targeting_strategy.py`

**Remove portfolio allocation method:**
```python
# REMOVE: Lines 319-369 - generate_portfolio_allocation()
def generate_portfolio_allocation(
    self,
    assets_data: Dict[str, pd.DataFrame],
    base_weights: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Generate volatility-targeted portfolio allocation."""
    # This is portfolio-level functionality - should be in Supervisor
    # Remove this method
```

**Keep only strategy-specific volatility targeting:**
```python
# KEEP: Lines 40-190 - calculate_position_multiplier() and calculate_volatility()
# These are strategy-specific position sizing methods
```

### 2.2 Update Strategy Generation

**File**: `src/strategist/strategist.py`

**Add interface integration:**
```python
# ADD: After line 32 in __init__()
def __init__(self, config: dict[str, Any]) -> None:
    # ... existing initialization code ...
    
    # Interface for supervisor communication
    self.supervisor_interface = None
    self.system_status_cache = {}
    self.last_status_update = None
    self.status_cache_duration = 60  # seconds
```

**Update strategy generation to use system context:**
```python
# UPDATE: Lines 162-218 in generate_strategy()
async def generate_strategy(
    self,
    market_data: pd.DataFrame,
    current_price: float,
    analysis_results: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Generate trading strategy with system context integration."""
    try:
        if not self._validate_market_data(market_data):
            self.logger.error("Invalid market data for strategy generation")
            return None

        self.logger.info("🎯 Generating trading strategy...")

        # Get system context if interface is available
        system_context = None
        if self.supervisor_interface:
            system_context = await self._get_system_context()

        # Extract key market indicators
        market_indicators = self._extract_market_indicators(market_data, current_price)

        # Generate base strategy
        base_strategy = await self._generate_base_strategy(market_indicators, current_price)

        # Integrate analysis results if available
        if analysis_results:
            base_strategy = await self._integrate_analysis_results(base_strategy, analysis_results)

        # Integrate system context if available
        if system_context:
            base_strategy = await self._integrate_system_context(base_strategy, system_context)

        # Apply strategy-specific risk management
        if self.enable_risk_management:
            base_strategy = await self._apply_risk_management(base_strategy, current_price)

        # Apply strategy-specific position sizing
        if self.enable_position_sizing:
            base_strategy = await self._apply_position_sizing(base_strategy, current_price)

        # Store strategy results
        await self._store_strategy_results(base_strategy)

        # Notify supervisor of strategy generation
        if self.supervisor_interface:
            await self.supervisor_interface.notify_strategy_event(
                "strategy_generated",
                {
                    "strategy_id": base_strategy.get("strategy_id"),
                    "direction": base_strategy.get("direction"),
                    "confidence": base_strategy.get("confidence")
                }
            )

        self.logger.info("✅ Strategy generation completed successfully")
        return base_strategy

    except Exception as e:
        self.logger.error(f"Error generating strategy: {e}")
        return None
```

**Add system context integration:**
```python
# ADD: New method after line 506
async def _get_system_context(self) -> dict[str, Any]:
    """Get system context from supervisor interface."""
    try:
        if not self.supervisor_interface:
            return {}

        # Check if we have cached status
        current_time = datetime.now()
        if (self.last_status_update and 
            (current_time - self.last_status_update).seconds < self.status_cache_duration):
            return self.system_status_cache

        # Request system status
        status_request = SystemStatusRequest(
            component_name="analyst",
            include_performance=True,
            include_health=True
        )
        
        status_response = await self.supervisor_interface.request_system_status(status_request)
        
        # Cache the response
        self.system_status_cache = {
            "analyst_health": status_response.is_healthy,
            "analyst_performance": status_response.performance_metrics,
            "system_status": status_response.health_status
        }
        self.last_status_update = current_time
        
        return self.system_status_cache
        
    except Exception as e:
        self.logger.error(f"Error getting system context: {e}")
        return {}

async def _integrate_system_context(self, strategy: dict[str, Any], system_context: dict[str, Any]) -> dict[str, Any]:
    """Integrate system context into strategy decisions."""
    try:
        # Adjust strategy based on system health
        if not system_context.get("analyst_health", True):
            strategy["confidence"] *= 0.8  # Reduce confidence if analyst is unhealthy
            strategy["reasoning"].append("Reduced confidence due to analyst health issues")

        # Adjust based on system performance
        analyst_performance = system_context.get("analyst_performance", {})
        if analyst_performance:
            # Use performance metrics to adjust strategy
            pass

        return strategy
        
    except Exception as e:
        self.logger.error(f"Error integrating system context: {e}")
        return strategy
```

## Phase 3: Update Component Initialization

### 3.1 Update Supervisor Initialization

**File**: `src/supervisor/supervisor.py`

**Add interface initialization:**
```python
# ADD: After line 257 in _initialize_components()
async def _initialize_components(self) -> None:
    try:
        # ... existing component initialization ...
        
        # Initialize strategist interface
        await self._initialize_strategist_interface()
        
        self.logger.info("Components initialized successfully")
    except Exception:
        self.print(initialization_error("Error initializing components: {e}"))

# ADD: New method after _initialize_components()
async def _initialize_strategist_interface(self) -> None:
    """Initialize the strategist interface."""
    try:
        from src.interfaces.supervisor_strategist_interface import SupervisorInterface
        
        # Create supervisor interface
        self.strategist_interface = SupervisorInterface(self)
        
        # If strategist component exists, connect the interface
        if 'strategist' in self.components and self.components['strategist']:
            strategist = self.components['strategist']
            if hasattr(strategist, 'supervisor_interface'):
                from src.interfaces.supervisor_strategist_interface import StrategistInterface
                strategist_interface = StrategistInterface(strategist)
                strategist_interface.set_supervisor_interface(self.strategist_interface)
                strategist.supervisor_interface = self.strategist_interface
        
        self.logger.info("Strategist interface initialized successfully")
        
    except Exception as e:
        self.logger.error(f"Error initializing strategist interface: {e}")
```

### 3.2 Update Strategist Initialization

**File**: `src/strategist/strategist.py`

**Add interface setup:**
```python
# UPDATE: Lines 73-102 in initialize()
async def initialize(self) -> bool:
    """Initialize strategist with interface support."""
    try:
        self.logger.info("Initializing Strategist...")

        # Validate configuration
        if not self._validate_configuration():
            self.logger.error(invalid("Invalid configuration for strategist"))
            return False

        # Initialize strategy components
        await self._initialize_strategy_components()
        
        # Initialize interface if supervisor interface is provided
        if hasattr(self, 'supervisor_interface') and self.supervisor_interface:
            await self._initialize_interface()

        self.logger.info("✅ Strategist initialized successfully")
        return True

    except Exception as e:
        self.logger.error(failed(f"❌ Strategist initialization failed: {e}"))
        return False

# ADD: New method after _initialize_strategy_components()
async def _initialize_interface(self) -> None:
    """Initialize the supervisor interface."""
    try:
        if self.supervisor_interface:
            # Test interface communication
            test_request = SystemStatusRequest(
                component_name="test",
                include_performance=False,
                include_health=True
            )
            
            # This would test the interface connection
            # await self.supervisor_interface.request_system_status(test_request)
            
            self.logger.info("✅ Supervisor interface initialized successfully")
        else:
            self.logger.warning("No supervisor interface available")
            
    except Exception as e:
        self.logger.error(f"Error initializing interface: {e}")
```

## Phase 4: Update Configuration

### 4.1 Add Interface Configuration

**File**: `src/config.py`

**Add interface configuration:**
```python
# ADD: After existing supervisor configuration
SUPERVISOR_CONFIG = {
    # ... existing supervisor config ...
    "interface": {
        "enable_strategist_interface": True,
        "strategy_request_timeout": 30,
        "performance_tracking_enabled": True,
        "event_notification_enabled": True
    }
}

STRATEGIST_CONFIG = {
    # ... existing strategist config ...
    "interface": {
        "enable_supervisor_interface": True,
        "system_status_cache_duration": 60,
        "event_notification_enabled": True,
        "performance_submission_enabled": True
    }
}
```

## Phase 5: Testing Updates

### 5.1 Update Supervisor Tests

**File**: `tests/test_supervisor.py`

**Add interface tests:**
```python
# ADD: New test methods
async def test_strategist_interface_initialization():
    """Test strategist interface initialization."""
    # Test interface setup
    pass

async def test_strategy_request_flow():
    """Test strategy request flow through interface."""
    # Test strategy request/response
    pass

async def test_system_status_provision():
    """Test system status provision to strategist."""
    # Test status requests
    pass
```

### 5.2 Update Strategist Tests

**File**: `tests/test_strategist.py`

**Add interface tests:**
```python
# ADD: New test methods
async def test_supervisor_interface_integration():
    """Test supervisor interface integration."""
    # Test interface communication
    pass

async def test_system_context_integration():
    """Test system context integration in strategy generation."""
    # Test context usage
    pass

async def test_event_notification():
    """Test event notification to supervisor."""
    # Test event notifications
    pass
```

## Validation Checklist

After implementing these changes, verify:

- [ ] No direct method calls between Supervisor and Strategist
- [ ] All communication goes through the interface
- [ ] Position sizing logic is only in Strategist
- [ ] Portfolio-level risk management is only in Supervisor
- [ ] Strategy-specific risk management is only in Strategist
- [ ] System health monitoring is only in Supervisor
- [ ] Strategy generation is only in Strategist
- [ ] Interface handles all error scenarios gracefully
- [ ] All existing functionality is preserved
- [ ] Performance impact is minimal

## Rollback Plan

If issues arise during implementation:

1. **Keep original methods as deprecated** for quick rollback
2. **Add feature flags** to enable/disable interface usage
3. **Maintain backward compatibility** during transition
4. **Monitor performance** and rollback if significant degradation
5. **Test thoroughly** before removing deprecated methods

## Next Steps

1. **Implement interface** (Phase 1)
2. **Refactor Supervisor** (Phase 2)
3. **Refactor Strategist** (Phase 3)
4. **Update initialization** (Phase 4)
5. **Add configuration** (Phase 5)
6. **Update tests** (Phase 6)
7. **Validate separation** (Phase 7)
8. **Remove deprecated code** (Phase 8)