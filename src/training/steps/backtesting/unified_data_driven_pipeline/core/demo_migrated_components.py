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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.tree import Tree
from rich import print as rprint
from rich import box
from rich.syntax import Syntax
from rich.prompt import Confirm, Prompt

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

# Initialize Rich console
console = Console()


def demo_component_registry():
    """Demonstrate component registry functionality."""
    console.print(Panel.fit(
        "[bold blue]Component Registry Demo[/bold blue]\n"
        "Demonstrating component registry functionality",
        border_style="blue"
    ))
    
    # Get the registry
    registry = get_registry()
    
    # Register migrated components
    console.print("\n📝 [yellow]Registering migrated components...[/yellow]")
    register_migrated_monte_carlo_engine()
    register_migrated_vectorbt_manager()
    register_migrated_paper_trading_engine()
    
    # Get all components
    components = get_all_components()
    console.print(f"\n📊 [bold green]Total components registered:[/bold green] {len(components)}")
    
    # Display components table
    components_table = Table(title="Registered Components", box=box.ROUNDED)
    components_table.add_column("Name", style="cyan")
    components_table.add_column("Type", style="green")
    components_table.add_column("Status", style="yellow")
    
    for component in components:
        components_table.add_row(
            component['name'],
            component['type'],
            component['status']
        )
    
    console.print(components_table)
    
    # Initialize and start components
    console.print("\n🚀 [yellow]Initializing components...[/yellow]")
    init_table = Table(title="Component Initialization", box=box.ROUNDED)
    init_table.add_column("Component", style="cyan")
    init_table.add_column("Status", style="green")
    
    for component in components:
        name = component['name']
        if initialize_component(name):
            init_table.add_row(name, "✅ Initialized")
        else:
            init_table.add_row(name, "❌ Failed")
    
    console.print(init_table)
    
    # Get component status
    console.print("\n📊 [yellow]Component status:[/yellow]")
    status_table = Table(title="Component Status", box=box.ROUNDED)
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Health", style="yellow")
    
    for component in components:
        name = component['name']
        status = get_component_status(name)
        if status:
            health = status.get('health_status', 'unknown')
            health_style = "green" if health == "healthy" else "red" if health == "unhealthy" else "yellow"
            status_table.add_row(
                name,
                status['status'],
                f"[{health_style}]{health}[/{health_style}]"
            )
    
    console.print(status_table)


def demo_workflow_orchestration():
    """Demonstrate workflow orchestration."""
    console.print(Panel.fit(
        "[bold blue]Workflow Orchestration Demo[/bold blue]\n"
        "Demonstrating workflow orchestration capabilities",
        border_style="blue"
    ))
    
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
    
    # Display workflow info
    console.print(f"\n📋 [bold green]Workflow defined:[/bold green] {workflow.name}")
    console.print(f"📊 [cyan]Steps:[/cyan] {len(workflow.steps)}")
    console.print(f"⚙️  [yellow]Execution mode:[/yellow] {workflow.execution_mode.value}")
    
    # Display workflow steps
    steps_table = Table(title="Workflow Steps", box=box.ROUNDED)
    steps_table.add_column("Step", style="cyan")
    steps_table.add_column("Component", style="green")
    steps_table.add_column("Dependencies", style="yellow")
    
    for step in workflow.steps:
        deps = ", ".join(step.dependencies) if step.dependencies else "None"
        steps_table.add_row(step.name, step.component_type, deps)
    
    console.print(steps_table)
    
    # Execute workflow
    console.print("\n🚀 [yellow]Executing workflow...[/yellow]")
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
    
    console.print(f"✅ [bold green]Workflow started with ID:[/bold green] {workflow_id}")
    
    # Monitor workflow progress
    console.print("\n📊 [yellow]Monitoring workflow progress...[/yellow]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Monitoring workflow...", total=100)
        
        for i in range(5):  # Check 5 times
            status = get_workflow_status(workflow_id)
            if status:
                progress.update(task, description=f"Status: {status['status']}")
                console.print(f"\n📊 [cyan]Status:[/cyan] {status['status']}")
                console.print(f"📍 [yellow]Current step:[/yellow] {status.get('current_step', 'N/A')}")
                console.print(f"✅ [green]Completed steps:[/green] {len(status.get('completed_steps', []))}")
                console.print(f"❌ [red]Failed steps:[/red] {len(status.get('failed_steps', []))}")
            else:
                console.print("❌ [bold red]Workflow not found[/bold red]")
            
            time.sleep(2)  # Wait 2 seconds between checks
            progress.advance(task, 20)


def demo_component_monitoring():
    """Demonstrate component monitoring."""
    console.print(Panel.fit(
        "[bold blue]Component Monitoring Demo[/bold blue]\n"
        "Demonstrating component monitoring capabilities",
        border_style="blue"
    ))
    
    # Start monitoring
    start_monitoring()
    console.print("✅ [bold green]Monitoring started[/bold green]")
    
    # Get dashboard data
    dashboard = get_monitoring_dashboard_data()
    console.print("\n📊 [bold green]Dashboard data:[/bold green]")
    
    # Dashboard summary table
    dashboard_table = Table(title="Monitoring Dashboard", box=box.ROUNDED)
    dashboard_table.add_column("Metric", style="cyan")
    dashboard_table.add_column("Value", style="green")
    
    dashboard_table.add_row("Total Components", f"{dashboard['components']['total']}")
    dashboard_table.add_row("Healthy Components", f"{dashboard['components']['healthy']}")
    dashboard_table.add_row("Unhealthy Components", f"{dashboard['components']['unhealthy']}")
    dashboard_table.add_row("Health Percentage", f"{dashboard['components']['health_percentage']:.1f}%")
    dashboard_table.add_row("Total Alerts", f"{dashboard['alerts']['total']}")
    dashboard_table.add_row("Critical Alerts", f"{dashboard['alerts']['critical']}")
    
    console.print(dashboard_table)
    
    # Get specific component health
    console.print("\n🔍 [yellow]Component health details:[/yellow]")
    health_table = Table(title="Component Health", box=box.ROUNDED)
    health_table.add_column("Component", style="cyan")
    health_table.add_column("Health Score", style="green")
    health_table.add_column("Status", style="yellow")
    
    for component_name in ['monte_carlo_engine', 'vectorbt_manager', 'paper_trading_engine']:
        health = get_component_health(component_name)
        if health:
            status_style = "green" if health.status.value == "healthy" else "red" if health.status.value == "unhealthy" else "yellow"
            health_table.add_row(
                component_name,
                f"{health.health_score:.2f}",
                f"[{status_style}]{health.status.value}[/{status_style}]"
            )
        else:
            health_table.add_row(component_name, "N/A", "[red]Not found[/red]")
    
    console.print(health_table)


def demo_individual_components():
    """Demonstrate individual component usage."""
    console.print(Panel.fit(
        "[bold blue]Individual Component Demo[/bold blue]\n"
        "Demonstrating individual component usage",
        border_style="blue"
    ))
    
    # Demo Monte Carlo Engine
    console.print("\n🎲 [bold yellow]Monte Carlo Engine Demo[/bold yellow]")
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
        console.print("✅ [bold green]Monte Carlo Engine initialized[/bold green]")
        
        # Process sample data
        sample_data = {
            'prices': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109, 111, 110, 112, 114, 113, 115],
            'returns': [0.02, -0.01, 0.02, 0.02, -0.01, 0.02, 0.02, -0.01, 0.02, 0.02, -0.01, 0.02, 0.02, -0.01, 0.02]
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Running Monte Carlo simulation...", total=100)
            
            result = mc_engine.process(sample_data)
            progress.update(task, completed=100)
        
        # Display results
        console.print("\n📊 [bold green]Simulation completed:[/bold green]")
        mc_table = Table(title="Monte Carlo Results", box=box.ROUNDED)
        mc_table.add_column("Metric", style="cyan")
        mc_table.add_column("Value", style="green")
        
        metrics = result['performance_metrics']
        mc_table.add_row("Total Return", f"{metrics['total_return']:.2%}")
        mc_table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        mc_table.add_row("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        mc_table.add_row("VaR 95%", f"{metrics['var_95']:.2%}")
        mc_table.add_row("VaR 99%", f"{metrics['var_99']:.2%}")
        mc_table.add_row("Expected Shortfall 95%", f"{metrics['expected_shortfall_95']:.2%}")
        mc_table.add_row("Expected Shortfall 99%", f"{metrics['expected_shortfall_99']:.2%}")
        mc_table.add_row("Simulations", f"{metrics['n_simulations']}")
        
        console.print(mc_table)
        
        mc_engine.cleanup()
        console.print("🧹 [yellow]Monte Carlo Engine cleaned up[/yellow]")
    
    # Demo VectorBT Manager
    console.print("\n⚡ [bold yellow]VectorBT Manager Demo[/bold yellow]")
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
        console.print("✅ [bold green]VectorBT Manager initialized[/bold green]")
        
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
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Executing VectorBT operation...", total=100)
            
            result = vbt_manager.process(sample_data)
            progress.update(task, completed=100)
        
        # Display results
        console.print("\n📊 [bold green]VectorBT operation completed:[/bold green]")
        vbt_table = Table(title="VectorBT Results", box=box.ROUNDED)
        vbt_table.add_column("Property", style="cyan")
        vbt_table.add_column("Value", style="green")
        
        vbt_table.add_row("Operation Type", result['operation_type'])
        vbt_table.add_row("Success Rate", f"{result['performance_metrics']['success_rate']:.2%}")
        vbt_table.add_row("Total Operations", f"{result['performance_metrics']['total_operations']}")
        vbt_table.add_row("Successful Operations", f"{result['performance_metrics']['successful_operations']}")
        vbt_table.add_row("Failed Operations", f"{result['performance_metrics']['failed_operations']}")
        vbt_table.add_row("Avg Processing Time", f"{result['performance_metrics']['avg_processing_time']:.3f}s")
        
        console.print(vbt_table)
        
        vbt_manager.cleanup()
        console.print("🧹 [yellow]VectorBT Manager cleaned up[/yellow]")
    
    # Demo Paper Trading Engine
    console.print("\n💼 [bold yellow]Paper Trading Engine Demo[/bold yellow]")
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
        console.print("✅ [bold green]Paper Trading Engine initialized[/bold green]")
        
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
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Processing trading signals...", total=100)
            
            result = trading_engine.process(sample_data)
            progress.update(task, completed=100)
        
        # Display results
        console.print("\n📊 [bold green]Trading completed:[/bold green]")
        
        # Portfolio state table
        portfolio_table = Table(title="Portfolio State", box=box.ROUNDED)
        portfolio_table.add_column("Property", style="cyan")
        portfolio_table.add_column("Value", style="green")
        
        portfolio_state = result['portfolio_state']
        portfolio_table.add_row("Trades Executed", f"{len(result['trading_results'])}")
        portfolio_table.add_row("Total Value", f"${portfolio_state['total_value']:,.2f}")
        portfolio_table.add_row("Cash", f"${portfolio_state['cash']:,.2f}")
        portfolio_table.add_row("Unrealized P&L", f"${portfolio_state['unrealized_pnl']:,.2f}")
        portfolio_table.add_row("Realized P&L", f"${portfolio_state['realized_pnl']:,.2f}")
        portfolio_table.add_row("Total Commission", f"${portfolio_state['total_commission']:,.2f}")
        
        console.print(portfolio_table)
        
        # Performance metrics table
        performance_table = Table(title="Performance Metrics", box=box.ROUNDED)
        performance_table.add_column("Metric", style="cyan")
        performance_table.add_column("Value", style="green")
        
        performance_metrics = result['performance_metrics']
        performance_table.add_row("Total Trades", f"{performance_metrics['total_trades']}")
        performance_table.add_row("Winning Trades", f"{performance_metrics['winning_trades']}")
        performance_table.add_row("Losing Trades", f"{performance_metrics['losing_trades']}")
        performance_table.add_row("Win Rate", f"{performance_metrics['win_rate']:.2%}")
        performance_table.add_row("Avg Win", f"${performance_metrics['avg_win']:,.2f}")
        performance_table.add_row("Avg Loss", f"${performance_metrics['avg_loss']:,.2f}")
        performance_table.add_row("Profit Factor", f"{performance_metrics['profit_factor']:.2f}")
        
        console.print(performance_table)
        
        trading_engine.cleanup()
        console.print("🧹 [yellow]Paper Trading Engine cleaned up[/yellow]")


def demo_configuration_templates():
    """Demonstrate configuration templates."""
    console.print(Panel.fit(
        "[bold blue]Configuration Templates Demo[/bold blue]\n"
        "Demonstrating configuration templates and validation",
        border_style="blue"
    ))
    
    from src.training.steps.backtesting.unified_data_driven_pipeline.core import (
        get_backtesting_config_template,
        validate_backtesting_config
    )
    
    # Get configuration templates
    basic_config = get_backtesting_config_template('basic_backtesting')
    advanced_config = get_backtesting_config_template('advanced_backtesting')
    
    console.print("\n📋 [bold yellow]Basic backtesting configuration:[/bold yellow]")
    console.print(Syntax(json.dumps(basic_config, indent=2), "json", theme="monokai"))
    
    console.print("\n📋 [bold yellow]Advanced backtesting configuration:[/bold yellow]")
    console.print(Syntax(json.dumps(advanced_config, indent=2), "json", theme="monokai"))
    
    # Validate configurations
    basic_valid = validate_backtesting_config(basic_config)
    advanced_valid = validate_backtesting_config(advanced_config)
    
    console.print("\n✅ [bold green]Configuration validation:[/bold green]")
    validation_table = Table(title="Configuration Validation", box=box.ROUNDED)
    validation_table.add_column("Configuration", style="cyan")
    validation_table.add_column("Valid", style="green")
    
    validation_table.add_row(
        "Basic Config",
        "✅ Valid" if basic_valid else "❌ Invalid"
    )
    validation_table.add_row(
        "Advanced Config",
        "✅ Valid" if advanced_valid else "❌ Invalid"
    )
    
    console.print(validation_table)


def main():
    """Main demo function."""
    # Display main banner
    console.print(Panel.fit(
        "[bold blue]Migrated Backtesting Components Demo[/bold blue]\n"
        "This demo shows how to use the migrated backtesting components\n"
        "with the ModularComponent architecture.",
        border_style="blue"
    ))
    
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
        
        # Success message
        console.print(Panel.fit(
            "[bold green]Demo Completed Successfully![/bold green]\n"
            "All migrated components are working correctly!",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(Panel.fit(
            f"[bold red]Demo Failed[/bold red]\n"
            f"Error: {e}",
            border_style="red"
        ))
        import traceback
        console.print(Syntax(traceback.format_exc(), "python", theme="monokai"))


if __name__ == '__main__':
    main()