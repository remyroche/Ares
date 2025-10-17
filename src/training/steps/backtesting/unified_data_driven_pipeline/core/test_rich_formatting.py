#!/usr/bin/env python3
"""
Test Script for Rich Formatting in Migrated Components

This script demonstrates the rich formatting capabilities
in all migrated backtesting components.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.tree import Tree
from rich import print as rprint
from rich import box
from rich.syntax import Syntax
from rich.prompt import Confirm, Prompt

# Initialize Rich console
console = Console()

def test_rich_formatting():
    """Test rich formatting capabilities."""
    
    # Display main banner
    console.print(Panel.fit(
        "[bold blue]Rich Formatting Test[/bold blue]\n"
        "Testing rich formatting capabilities in migrated components",
        border_style="blue"
    ))
    
    # Test 1: Basic formatting
    console.print("\n🎨 [bold yellow]Test 1: Basic Formatting[/bold yellow]")
    console.print("This is [bold]bold text[/bold]")
    console.print("This is [italic]italic text[/italic]")
    console.print("This is [underline]underlined text[/underline]")
    console.print("This is [strike]strikethrough text[/strike]")
    console.print("This is [dim]dim text[/dim]")
    console.print("This is [red]red text[/red]")
    console.print("This is [green]green text[/green]")
    console.print("This is [blue]blue text[/blue]")
    console.print("This is [yellow]yellow text[/yellow]")
    console.print("This is [magenta]magenta text[/magenta]")
    console.print("This is [cyan]cyan text[/cyan]")
    
    # Test 2: Tables
    console.print("\n📊 [bold yellow]Test 2: Tables[/bold yellow]")
    table = Table(title="Sample Data Table", box=box.ROUNDED)
    table.add_column("Name", style="cyan")
    table.add_column("Age", style="green")
    table.add_column("City", style="yellow")
    table.add_column("Status", style="magenta")
    
    table.add_row("Alice", "25", "New York", "✅ Active")
    table.add_row("Bob", "30", "London", "❌ Inactive")
    table.add_row("Charlie", "35", "Paris", "✅ Active")
    table.add_row("Diana", "28", "Tokyo", "⚠️ Pending")
    
    console.print(table)
    
    # Test 3: Progress bars
    console.print("\n⏳ [bold yellow]Test 3: Progress Bars[/bold yellow]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Processing data...", total=100)
        
        for i in range(10):
            progress.update(task, description=f"Processing item {i+1}/10")
            progress.advance(task, 10)
            import time
            time.sleep(0.1)
    
    # Test 4: Syntax highlighting
    console.print("\n💻 [bold yellow]Test 4: Syntax Highlighting[/bold yellow]")
    sample_code = '''
def migrate_component(component_name: str) -> bool:
    """Migrate a component to ModularComponent architecture."""
    try:
        # Analyze component
        analysis = analyzer.analyze_component(component_name)
        
        # Migrate based on analysis
        if analysis.compatibility_score > 0.8:
            return migrate_direct(component_name)
        else:
            return migrate_wrapper(component_name)
            
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return False
'''
    
    console.print(Syntax(sample_code, "python", theme="monokai"))
    
    # Test 5: Trees
    console.print("\n🌳 [bold yellow]Test 5: Trees[/bold yellow]")
    tree = Tree("Backtesting Components")
    
    # Core components
    core_branch = tree.add("Core Components")
    core_branch.add("Monte Carlo Engine")
    core_branch.add("VectorBT Manager")
    core_branch.add("Paper Trading Engine")
    
    # ABC Testing components
    abc_branch = tree.add("ABC Testing Components")
    abc_branch.add("Performance Monitor")
    abc_branch.add("Risk Manager")
    abc_branch.add("Statistical Analyzer")
    
    # NAS TAS components
    nas_branch = tree.add("NAS TAS Components")
    nas_branch.add("Walk Forward Analyzer")
    nas_branch.add("Performance Attribution")
    nas_branch.add("Validation Orchestrator")
    
    console.print(tree)
    
    # Test 6: Panels
    console.print("\n📋 [bold yellow]Test 6: Panels[/bold yellow]")
    
    # Success panel
    console.print(Panel.fit(
        "[bold green]✅ Success![/bold green]\n"
        "Component migration completed successfully.",
        border_style="green"
    ))
    
    # Warning panel
    console.print(Panel.fit(
        "[bold yellow]⚠️ Warning[/bold yellow]\n"
        "Some components may require manual review.",
        border_style="yellow"
    ))
    
    # Error panel
    console.print(Panel.fit(
        "[bold red]❌ Error[/bold red]\n"
        "Migration failed for component 'invalid_component'.",
        border_style="red"
    ))
    
    # Info panel
    console.print(Panel.fit(
        "[bold blue]ℹ️ Information[/bold blue]\n"
        "Migration tools are ready for use.",
        border_style="blue"
    ))
    
    # Test 7: Prompts
    console.print("\n❓ [bold yellow]Test 7: Prompts[/bold yellow]")
    
    # Confirm prompt
    if Confirm.ask("Do you want to continue with the test?"):
        console.print("✅ [green]User confirmed[/green]")
    else:
        console.print("❌ [red]User cancelled[/red]")
    
    # Text prompt
    name = Prompt.ask("Enter your name", default="Anonymous")
    console.print(f"👋 [green]Hello, {name}![/green]")
    
    # Test 8: JSON formatting
    console.print("\n📄 [bold yellow]Test 8: JSON Formatting[/bold yellow]")
    sample_json = {
        "component_name": "monte_carlo_engine",
        "status": "migrated",
        "migration_strategy": "direct",
        "performance_metrics": {
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": 0.05
        },
        "dependencies": ["data_loader", "feature_generator"]
    }
    
    console.print(Syntax(str(sample_json).replace("'", '"'), "json", theme="monokai"))
    
    # Test 9: Status indicators
    console.print("\n📊 [bold yellow]Test 9: Status Indicators[/bold yellow]")
    status_table = Table(title="Component Status", box=box.ROUNDED)
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green")
    status_table.add_column("Health", style="yellow")
    status_table.add_column("Progress", style="magenta")
    
    status_table.add_row("Monte Carlo Engine", "✅ Migrated", "🟢 Healthy", "100%")
    status_table.add_row("VectorBT Manager", "✅ Migrated", "🟢 Healthy", "100%")
    status_table.add_row("Paper Trading Engine", "✅ Migrated", "🟢 Healthy", "100%")
    status_table.add_row("Risk Manager", "🔄 Migrating", "🟡 Warning", "75%")
    status_table.add_row("Performance Monitor", "⏳ Pending", "🔴 Unhealthy", "0%")
    
    console.print(status_table)
    
    # Test 10: Emoji and symbols
    console.print("\n🎭 [bold yellow]Test 10: Emoji and Symbols[/bold yellow]")
    console.print("🚀 [bold]Starting migration process[/bold]")
    console.print("📊 [bold]Analyzing components[/bold]")
    console.print("⚙️ [bold]Configuring settings[/bold]")
    console.print("🔧 [bold]Building components[/bold]")
    console.print("🧪 [bold]Testing functionality[/bold]")
    console.print("📝 [bold]Documenting changes[/bold]")
    console.print("✅ [bold]Migration completed[/bold]")
    console.print("🎉 [bold]Success![/bold]")
    
    # Final success message
    console.print(Panel.fit(
        "[bold green]🎉 Rich Formatting Test Completed Successfully![/bold green]\n"
        "All rich formatting capabilities are working correctly!",
        border_style="green"
    ))


if __name__ == '__main__':
    test_rich_formatting()