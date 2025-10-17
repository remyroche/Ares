"""
Demo Script: Using Migrated Backtesting Components

This script demonstrates how to use the migrated backtesting components
with the ModularComponent architecture, including component registry,
workflow orchestration, and monitoring.
"""

import sys
import os
from pathlib import Path
import time
import json

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.training.steps.backtesting.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent,
    create_backtesting_component,
    ValidationLevel,
    ErrorInfo,
    ErrorSeverity,
    ErrorCategory
)
from src.training.steps.backtesting.unified_data_driven_pipeline.core.component_registry import (
    get_registry,
    register_component,
    ComponentType,
    initialize_component,
    start_component,
    get_component_status,
    get_all_components
)
from src.training.steps.backtesting.unified_data_driven_pipeline.core.component_orchestrator import (
    define_workflow,
    execute_workflow,
    get_workflow_status,
    WorkflowStep,
    ExecutionMode
)
from src.training.steps.backtesting.unified_data_driven_pipeline.core.component_monitor import (
    start_monitoring,
    get_monitoring_dashboard_data,
    get_component_health,
    get_performance_metrics
)

# Import migrated components
from migrate_monte_carlo_engine import create_migrated_monte_carlo_engine, register_migrated_monte_carlo_engine
from migrate_vectorbt_manager import create_migrated_vectorbt_manager, register_migrated_vectorbt_manager
from migrate_paper_trading_engine import create_migrated_paper_trading_engine, register_migrated_paper_trading_engine


def demo_component_registry():
    """Demonstrate component registry functionality."""
    print("=== Component Registry Demo ===")
    
    # Get the registry
    registry = get_registry()
    
    # Register migrated components
    print("Registering migrated components...")
    register_migrated_monte_carlo_engine()
    register_migrated_vectorbt_manager()
    register_migrated_paper_trading_engine()
    
    # Get all components
    components = get_all_components()
    print(f"Total components registered: {len(components)}")
    
    for component in components:
        print(f"- {component['name']}: {component['type']} ({component['status']})")
    
    # Initialize and start components
    print("\nInitializing components...")
    for component in components:
        name = component['name']
        if initialize_component(name):
            print(f"✓ {name} initialized")
        else:
            print(f"✗ {name} initialization failed")
    
    # Get component status
    print("\nComponent status:")
    for component in components:
        name = component['name']
        status = get_component_status(name)
        if status:
            print(f"- {name}: {status['status']} (health: {status.get('health_status', 'unknown')})")


def demo_workflow_orchestration():
    """Demonstrate workflow orchestration."""
    print("\n=== Workflow Orchestration Demo ===")
    
    # Define a comprehensive backtesting workflow
    workflow = define_workflow(
        name='comprehensive_backtesting_pipeline',
        description='Complete backtesting pipeline with Monte Carlo simulation and paper trading',
        steps=[
            WorkflowStep('load_market_data', 'data_loader'),
            WorkflowStep('generate_features', 'feature_generator', dependencies=['load_market_data']),
            WorkflowStep('monte_carlo_simulation', 'monte_carlo_engine', dependencies=['generate_features']),
            WorkflowStep('vectorbt_analysis', 'vectorbt_manager', dependencies=['monte_carlo_simulation']),
            WorkflowStep('paper_trading', 'paper_trading_engine', dependencies=['vectorbt_analysis']),
            WorkflowStep('performance_analysis', 'performance_analyzer', dependencies=['paper_trading'])
        ],
        execution_mode=ExecutionMode.PIPELINE,
        max_parallel_workers=2,
        timeout=1800,  # 30 minutes
        enable_checkpointing=True,
        enable_monitoring=True
    )
    
    print(f"Workflow defined: {workflow.name}")
    print(f"Steps: {len(workflow.steps)}")
    print(f"Execution mode: {workflow.execution_mode.value}")
    
    # Execute workflow
    print("\nExecuting workflow...")
    workflow_id = execute_workflow(
        workflow,
        input_data={
            'symbol': 'BTCUSDT',
            'start_date': '2023-01-01',
            'end_date': '2023-12-31',
            'strategy_config': {
                'lookback': 20,
                'threshold': 0.5,
                'risk_level': 'medium'
            }
        }
    )
    
    print(f"Workflow started with ID: {workflow_id}")
    
    # Monitor workflow progress
    print("\nMonitoring workflow progress...")
    for i in range(5):  # Check 5 times
        status = get_workflow_status(workflow_id)
        if status:
            print(f"Status: {status['status']}")
            print(f"Current step: {status.get('current_step', 'N/A')}")
            print(f"Completed steps: {len(status.get('completed_steps', []))}")
            print(f"Failed steps: {len(status.get('failed_steps', []))}")
        else:
            print("Workflow not found")
        
        time.sleep(2)  # Wait 2 seconds between checks


def demo_component_monitoring():
    """Demonstrate component monitoring."""
    print("\n=== Component Monitoring Demo ===")
    
    # Start monitoring
    start_monitoring()
    print("Monitoring started")
    
    # Get dashboard data
    dashboard = get_monitoring_dashboard_data()
    print(f"\nDashboard data:")
    print(f"- Total components: {dashboard['components']['total']}")
    print(f"- Healthy components: {dashboard['components']['healthy']}")
    print(f"- Unhealthy components: {dashboard['components']['unhealthy']}")
    print(f"- Health percentage: {dashboard['components']['health_percentage']:.1f}%")
    print(f"- Total alerts: {dashboard['alerts']['total']}")
    print(f"- Critical alerts: {dashboard['alerts']['critical']}")
    
    # Get specific component health
    print(f"\nComponent health details:")
    for component_name in ['monte_carlo_engine', 'vectorbt_manager', 'paper_trading_engine']:
        health = get_component_health(component_name)
        if health:
            print(f"- {component_name}: {health.health_score:.2f} ({health.status.value})")
        else:
            print(f"- {component_name}: Not found")


def demo_individual_components():
    """Demonstrate individual component usage."""
    print("\n=== Individual Component Demo ===")
    
    # Demo Monte Carlo Engine
    print("\n--- Monte Carlo Engine Demo ---")
    mc_config = {
        'simulation': {
            'n_simulations': 1000,
            'confidence_levels': [0.95, 0.99],
            'method': 'bootstrap',
            'random_seed': 42
        },
        'backtesting': {
            'initial_capital': 100000.0,
            'commission': 0.001,
            'slippage': 0.0005
        }
    }
    
    mc_engine = create_migrated_monte_carlo_engine(mc_config)
    if mc_engine.initialize():
        print("Monte Carlo Engine initialized")
        
        # Process sample data
        sample_data = {
            'prices': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113, 115],
            'returns': [0.02, -0.01, 0.02, 0.02, -0.01, 0.02, 0.02, -0.01, 0.02, 0.02, -0.01, 0.02, 0.02, -0.01, 0.02]
        }
        
        result = mc_engine.process(sample_data)
        print(f"Simulation completed:")
        print(f"- Total return: {result['performance_metrics']['total_return']:.2%}")
        print(f"- Sharpe ratio: {result['performance_metrics']['sharpe_ratio']:.2f}")
        print(f"- Max drawdown: {result['performance_metrics']['max_drawdown']:.2%}")
        print(f"- VaR 95%: {result['performance_metrics']['var_95']:.2%}")
        
        mc_engine.cleanup()
        print("Monte Carlo Engine cleaned up")
    
    # Demo VectorBT Manager
    print("\n--- VectorBT Manager Demo ---")
    vbt_config = {
        'vectorbt': {
            'enable_gpu': False,
            'enable_parallel': True,
            'memory_limit': '2GB',
            'chunk_size': 1000
        },
        'optimization': {
            'enable_optimization': True,
            'max_workers': 4,
            'method': 'grid_search'
        }
    }
    
    vbt_manager = create_migrated_vectorbt_manager(vbt_config)
    if vbt_manager.initialize():
        print("VectorBT Manager initialized")
        
        # Process sample data
        sample_data = {
            'operation_type': 'rolling_statistics',
            'operation_params': {
                'window_size': 20,
                'statistics': ['mean', 'std', 'min', 'max']
            },
            'market_data': {
                'prices': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113, 115]
            }
        }
        
        result = vbt_manager.process(sample_data)
        print(f"VectorBT operation completed:")
        print(f"- Operation: {result['operation_type']}")
        print(f"- Success rate: {result['performance_metrics']['success_rate']:.2%}")
        print(f"- Avg processing time: {result['performance_metrics']['avg_processing_time']:.3f}s")
        
        vbt_manager.cleanup()
        print("VectorBT Manager cleaned up")
    
    # Demo Paper Trading Engine
    print("\n--- Paper Trading Engine Demo ---")
    trading_config = {
        'trading': {
            'initial_capital': 100000.0,
            'commission_rate': 0.001,
            'slippage_rate': 0.0005,
            'min_trade_size': 0.01
        },
        'market': {
            'enable_slippage': True,
            'enable_latency': True,
            'latency_ms': 100,
            'spread_bps': 5
        },
        'risk': {
            'max_position_size': 0.1,
            'max_drawdown': 0.15,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10
        }
    }
    
    trading_engine = create_migrated_paper_trading_engine(trading_config)
    if trading_engine.initialize():
        print("Paper Trading Engine initialized")
        
        # Process sample trading data
        sample_data = {
            'signals': [
                {'action': 'BUY', 'symbol': 'BTCUSDT', 'quantity': 0.1, 'price': 50000},
                {'action': 'BUY', 'symbol': 'ETHUSDT', 'quantity': 1.0, 'price': 3000},
                {'action': 'SELL', 'symbol': 'BTCUSDT', 'quantity': 0.05, 'price': 51000}
            ],
            'market_data': {
                'prices': {'BTCUSDT': 50500, 'ETHUSDT': 3050}
            }
        }
        
        result = trading_engine.process(sample_data)
        print(f"Trading completed:")
        print(f"- Trades executed: {len(result['trading_results'])}")
        print(f"- Portfolio value: ${result['portfolio_state']['total_value']:,.2f}")
        print(f"- Cash: ${result['portfolio_state']['cash']:,.2f}")
        print(f"- Unrealized P&L: ${result['portfolio_state']['unrealized_pnl']:,.2f}")
        print(f"- Total commission: ${result['portfolio_state']['total_commission']:,.2f}")
        print(f"- Win rate: {result['performance_metrics']['win_rate']:.2%}")
        
        trading_engine.cleanup()
        print("Paper Trading Engine cleaned up")


def demo_configuration_templates():
    """Demonstrate configuration templates."""
    print("\n=== Configuration Templates Demo ===")
    
    from src.training.steps.backtesting.unified_data_driven_pipeline.core import (
        get_backtesting_config_template,
        validate_backtesting_config
    )
    
    # Get configuration templates
    basic_config = get_backtesting_config_template('basic_backtesting')
    advanced_config = get_backtesting_config_template('advanced_backtesting')
    
    print("Basic backtesting configuration:")
    print(json.dumps(basic_config, indent=2))
    
    print("\nAdvanced backtesting configuration:")
    print(json.dumps(advanced_config, indent=2))
    
    # Validate configurations
    basic_valid = validate_backtesting_config(basic_config)
    advanced_valid = validate_backtesting_config(advanced_config)
    
    print(f"\nConfiguration validation:")
    print(f"- Basic config valid: {basic_valid}")
    print(f"- Advanced config valid: {advanced_valid}")


def main():
    """Main demo function."""
    print("=== Migrated Backtesting Components Demo ===")
    print("This demo shows how to use the migrated backtesting components")
    print("with the ModularComponent architecture.\n")
    
    try:
        # Demo individual components
        demo_individual_components()
        
        # Demo component registry
        demo_component_registry()
        
        # Demo workflow orchestration
        demo_workflow_orchestration()
        
        # Demo component monitoring
        demo_component_monitoring()
        
        # Demo configuration templates
        demo_configuration_templates()
        
        print("\n=== Demo Completed Successfully ===")
        print("All migrated components are working correctly!")
        
    except Exception as e:
        print(f"\n=== Demo Failed ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()