# Manager Classes Responsibilities

This document clarifies the responsibilities and boundaries of each manager class in the Ares trading bot codebase to prevent confusion and overlapping responsibilities.

## Overview

The Ares trading bot uses several manager classes to handle different aspects of the system. Each manager has a specific, well-defined responsibility to maintain clean separation of concerns.

## Core Manager Classes

### 1. ModelManager (`src/utils/model_manager.py`)

**Primary Responsibility**: Manages ML model lifecycle, loading, and hot-swapping.

**Key Responsibilities**:
- ✅ **Model Loading**: Loads champion/challenger models from disk
- ✅ **Model Hot-Swapping**: Promotes challenger models to champion
- ✅ **Model Versioning**: Tracks model versions and metadata
- ✅ **Model Checkpointing**: Saves and loads model checkpoints
- ✅ **Core Module Instantiation**: Creates Analyst, Strategist, Tactician instances

**What it does NOT do**:
- ❌ Database operations (handled by database managers)
- ❌ State management (handled by StateManager)
- ❌ Training (handled by TrainingManager)
- ❌ Portfolio management (handled by GlobalPortfolioManager)

**Usage Example**:
```python
model_manager = ModelManager()
await model_manager.initialize()
analyst = model_manager.get_analyst()
strategist = model_manager.get_strategist()
tactician = model_manager.get_tactician()
```

### 2. StateManager (`src/utils/state_manager.py`)

**Primary Responsibility**: Manages operational state and persistence.

**Key Responsibilities**:
- ✅ **State Persistence**: Saves/loads state to/from JSON files
- ✅ **Trading Status**: Manages RUNNING/PAUSED states
- ✅ **Kill Switch**: Activates/deactivates emergency stop
- ✅ **Position Tracking**: Tracks current trading positions
- ✅ **Performance Metrics**: Stores live performance data
- ✅ **Risk Parameters**: Manages global risk settings

**What it does NOT do**:
- ❌ Model management (handled by ModelManager)
- ❌ Database operations (handled by database managers)
- ❌ Portfolio calculations (handled by GlobalPortfolioManager)
- ❌ Training state (handled by TrainingManager)

**Usage Example**:
```python
state_manager = StateManager("ares_state.json")
await state_manager.set_state("global_trading_status", "RUNNING")
is_paused = state_manager.get_state("is_trading_paused", False)
```

### 3. GlobalPortfolioManager (`src/supervisor/global_portfolio_manager.py`)

**Primary Responsibility**: Monitors overall portfolio health and enforces global risk policies.

**Key Responsibilities**:
- ✅ **Portfolio Monitoring**: Tracks total equity across all accounts
- ✅ **Global Risk Management**: Enforces portfolio-wide risk limits
- ✅ **Capital Allocation**: Manages position sizing across multiple bots
- ✅ **Risk Parameter Publishing**: Distributes risk settings to all bots
- ✅ **Exposure Monitoring**: Tracks total market exposure
- ✅ **Emergency Controls**: Implements portfolio-wide safety measures

**What it does NOT do**:
- ❌ Individual bot state (handled by StateManager)
- ❌ Model management (handled by ModelManager)
- ❌ Database operations (handled by database managers)
- ❌ Individual trade execution (handled by trading pipelines)

**Usage Example**:
```python
portfolio_manager = GlobalPortfolioManager(state_manager, db_manager)
await portfolio_manager.start()  # Starts monitoring loop
```

### 4. SQLiteManager (`src/database/sqlite_manager.py`)

**Primary Responsibility**: Manages local SQLite database operations.

**Key Responsibilities**:
- ✅ **Local Data Storage**: Stores structured data locally
- ✅ **Backup Management**: Creates and restores database backups
- ✅ **Migration Support**: Handles database schema migrations
- ✅ **Document Operations**: CRUD operations for documents
- ✅ **Collection Queries**: Query collections with filters
- ✅ **Data Persistence**: Ensures data durability

**What it does NOT do**:
- ❌ Cloud operations (handled by FirestoreManager)
- ❌ State management (handled by StateManager)
- ❌ Model storage (handled by ModelManager)
- ❌ Portfolio calculations (handled by GlobalPortfolioManager)

**Usage Example**:
```python
db_manager = SQLiteManager()
await db_manager.initialize()
await db_manager.set_document("trades", "trade_123", trade_data)
trades = await db_manager.get_collection("trades")
```

### 5. FirestoreManager (`src/database/firestore_manager.py`)

**Primary Responsibility**: Manages cloud Firestore database operations.

**Key Responsibilities**:
- ✅ **Cloud Data Storage**: Stores data in Google Cloud Firestore
- ✅ **User Authentication**: Handles Firebase authentication
- ✅ **Real-time Updates**: Supports real-time data synchronization
- ✅ **Document Operations**: CRUD operations for cloud documents
- ✅ **Collection Queries**: Query cloud collections with filters
- ✅ **Security Rules**: Enforces Firestore security policies

**What it does NOT do**:
- ❌ Local operations (handled by SQLiteManager)
- ❌ State management (handled by StateManager)
- ❌ Model storage (handled by ModelManager)
- ❌ Portfolio calculations (handled by GlobalPortfolioManager)

**Usage Example**:
```python
firestore_manager = FirestoreManager()
await firestore_manager.initialize()
await firestore_manager.set_document("users", "user_123", user_data)
users = await firestore_manager.get_collection("users")
```

## Specialized Manager Classes

### 6. RegularizationManager (`src/training/regularization.py`)

**Primary Responsibility**: Manages ML model regularization configuration.

**Key Responsibilities**:
- ✅ **Regularization Configuration**: Manages L1/L2 regularization parameters
- ✅ **Model-Specific Settings**: Adapts regularization for different ML frameworks
- ✅ **Ensemble Configuration**: Applies regularization to ensemble models
- ✅ **Parameter Validation**: Validates regularization settings
- ✅ **Framework Adaptation**: Converts settings for LightGBM, TensorFlow, etc.

**What it does NOT do**:
- ❌ Model training (handled by TrainingManager)
- ❌ Model storage (handled by ModelManager)
- ❌ State management (handled by StateManager)
- ❌ Database operations (handled by database managers)

**Usage Example**:
```python
reg_manager = RegularizationManager()
reg_manager.apply_regularization_to_ensembles(ensemble_orchestrator)
```

### 7. TrainingManager (`src/training/training_manager.py`)

**Primary Responsibility**: Orchestrates the model training process.

**Key Responsibilities**:
- ✅ **Training Orchestration**: Coordinates the entire training pipeline
- ✅ **Data Preparation**: Manages training data preprocessing
- ✅ **Model Training**: Executes model training workflows
- ✅ **Validation**: Handles cross-validation and testing
- ✅ **Model Evaluation**: Assesses model performance
- ✅ **Training State**: Tracks training progress and metadata

**What it does NOT do**:
- ❌ Model serving (handled by ModelManager)
- ❌ State persistence (handled by StateManager)
- ❌ Database operations (handled by database managers)
- ❌ Portfolio management (handled by GlobalPortfolioManager)

**Usage Example**:
```python
training_manager = TrainingManager()
await training_manager.train_model(symbol, exchange)
```

## Utility Manager Classes

### 8. AsyncFileManager (`src/utils/async_utils.py`)

**Primary Responsibility**: Manages asynchronous file operations.

**Key Responsibilities**:
- ✅ **Async File I/O**: Handles file reading/writing asynchronously
- ✅ **File Operations**: Create, read, write, delete files
- ✅ **JSON Operations**: Async JSON serialization/deserialization
- ✅ **Error Handling**: Robust error handling for file operations
- ✅ **Thread Pool Management**: Manages blocking operations in thread pools

**What it does NOT do**:
- ❌ Database operations (handled by database managers)
- ❌ State management (handled by StateManager)
- ❌ Model operations (handled by ModelManager)

### 9. AsyncProcessManager (`src/utils/async_utils.py`)

**Primary Responsibility**: Manages asynchronous process operations.

**Key Responsibilities**:
- ✅ **Process Management**: Creates and manages subprocesses
- ✅ **Async Process Execution**: Runs processes asynchronously
- ✅ **Process Monitoring**: Tracks process status and output
- ✅ **Resource Management**: Manages process resources
- ✅ **Error Handling**: Handles process failures gracefully

### 10. AsyncNetworkManager (`src/utils/async_utils.py`)

**Primary Responsibility**: Manages asynchronous network operations.

**Key Responsibilities**:
- ✅ **HTTP Requests**: Handles async HTTP client operations
- ✅ **WebSocket Management**: Manages WebSocket connections
- ✅ **Network Monitoring**: Tracks network status and performance
- ✅ **Connection Pooling**: Manages connection pools
- ✅ **Error Handling**: Handles network failures gracefully

### 11. AsyncLockManager (`src/utils/async_utils.py`)

**Primary Responsibility**: Manages asynchronous locks and synchronization.

**Key Responsibilities**:
- ✅ **Async Locks**: Provides async lock mechanisms
- ✅ **Resource Synchronization**: Manages concurrent access to resources
- ✅ **Deadlock Prevention**: Prevents deadlock scenarios
- ✅ **Lock Monitoring**: Tracks lock usage and performance
- ✅ **Error Handling**: Handles lock-related errors gracefully

## Manager Class Interactions

### Data Flow Between Managers

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ModelManager  │    │  StateManager   │    │GlobalPortfolio  │
│                 │    │                 │    │   Manager       │
│ • Load models   │    │ • Track state   │    │ • Monitor       │
│ • Hot-swap     │    │ • Persist data  │    │   portfolio     │
│ • Versioning    │    │ • Kill switch   │    │ • Risk mgmt     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ SQLiteManager   │    │FirestoreManager │    │TrainingManager  │
│                 │    │                 │    │                 │
│ • Local storage │    │ • Cloud storage │    │ • Train models  │
│ • Backups       │    │ • Auth          │    │ • Validate      │
│ • Migrations    │    │ • Real-time     │    │ • Evaluate      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Responsibility Boundaries

| Responsibility | Primary Manager | Supporting Managers |
|----------------|-----------------|-------------------|
| Model Loading | ModelManager | - |
| State Persistence | StateManager | SQLiteManager/FirestoreManager |
| Portfolio Monitoring | GlobalPortfolioManager | StateManager |
| Data Storage | SQLiteManager/FirestoreManager | - |
| Training | TrainingManager | RegularizationManager |
| File Operations | AsyncFileManager | - |
| Process Management | AsyncProcessManager | - |
| Network Operations | AsyncNetworkManager | - |
| Synchronization | AsyncLockManager | - |

## Best Practices

### 1. Single Responsibility Principle
Each manager should have one primary responsibility and avoid overlapping with other managers.

### 2. Clear Interfaces
Managers should expose clear, well-documented interfaces for their functionality.

### 3. Error Handling
Each manager should handle its own errors and not propagate them to other managers unnecessarily.

### 4. Async Operations
Managers should use async operations where appropriate to avoid blocking the main thread.

### 5. Configuration
Managers should read their configuration from the centralized CONFIG system.

### 6. Logging
Each manager should use the centralized logging system for consistent log output.

## Common Anti-Patterns to Avoid

### ❌ Don't: Mix Responsibilities
```python
# BAD: StateManager doing database operations
class StateManager:
    async def save_to_database(self, data):
        await self.db_manager.set_document("state", "current", data)
```

### ✅ Do: Clear Separation
```python
# GOOD: StateManager focuses on state, delegates storage
class StateManager:
    async def set_state(self, key, value):
        self._state_cache[key] = value
        await self._save_state_to_file()  # Uses AsyncFileManager
```

### ❌ Don't: Duplicate Functionality
```python
# BAD: Multiple managers doing the same thing
class ModelManager:
    async def save_model(self, model_data):
        # ModelManager doing file operations
        with open("model.json", "w") as f:
            json.dump(model_data, f)
```

### ✅ Do: Use Specialized Managers
```python
# GOOD: ModelManager delegates to appropriate managers
class ModelManager:
    async def save_model(self, model_data):
        await async_file_manager.write_json("model.json", model_data)
```

## Migration Guidelines

When adding new functionality:

1. **Identify the primary responsibility** - which manager should handle this?
2. **Check existing managers** - is there already a manager for this responsibility?
3. **Create new manager if needed** - only if the responsibility is truly new
4. **Update this documentation** - keep the responsibilities clear and current
5. **Add tests** - ensure the new manager works correctly
6. **Update examples** - show how to use the new manager

## Conclusion

This clear separation of responsibilities ensures:
- **Maintainability**: Each manager has a focused, well-defined purpose
- **Testability**: Managers can be tested independently
- **Scalability**: New functionality can be added without affecting existing managers
- **Clarity**: Developers know exactly which manager to use for each task
- **Reliability**: Reduced risk of conflicts between different parts of the system 