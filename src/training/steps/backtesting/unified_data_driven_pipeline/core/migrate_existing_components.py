"""
Migration Script for Existing Backtesting Components

This script provides automated migration of existing backtesting components
to use the ModularComponent architecture. It includes analysis, migration,
and validation capabilities.

Usage:
    python migrate_existing_components.py --analyze
    python migrate_existing_components.py --migrate --component real_monte_carlo_engine
    python migrate_existing_components.py --migrate-all
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import time
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.text import Text
from rich import print as rprint
from rich.prompt import Confirm, Prompt
from rich.syntax import Syntax
from rich.tree import Tree
from rich import box

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent.parent))

from src.training.steps.backtesting.unified_data_driven_pipeline.core.migration_utils import (
    BacktestingComponentAnalyzer,
    BacktestingComponentMigrator,
    create_backtesting_component_wrapper,
    validate_backtesting_migration_compatibility,
    generate_backtesting_migration_report
)
from src.training.steps.backtesting.unified_data_driven_pipeline.core.component_registry import (
    get_registry,
    register_component,
    ComponentType
)
from src.training.steps.backtesting.unified_data_driven_pipeline.core.modular_architecture import (
    ModularComponent,
    create_backtesting_component
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Rich console
console = Console()

class BacktestingComponentMigrationManager:
    """Manager for migrating existing backtesting components."""
    
    def __init__(self):
        self.analyzer = BacktestingComponentAnalyzer()
        self.migrator = BacktestingComponentMigrator()
        self.registry = get_registry()
        
        # Define components to migrate
        self.components_to_migrate = {
            # Core backtesting components
            'real_monte_carlo_engine': {
                'file': 'src/training/steps/backtesting/real_monte_carlo_engine.py',
                'class': 'RealMonteCarloEngine',
                'type': ComponentType.MONTE_CARLO_ENGINE,
                'dependencies': ['data_loader', 'feature_generator'],
                'priority': 'high'
            },
            'real_parameters_optimization': {
                'file': 'src/training/steps/backtesting/real_parameters_optimization.py',
                'class': 'RealParametersOptimizer',
                'type': ComponentType.PARAMETER_OPTIMIZER,
                'dependencies': ['data_loader', 'feature_generator'],
                'priority': 'high'
            },
            'real_reporting_engine': {
                'file': 'src/training/steps/backtesting/real_reporting_engine.py',
                'class': 'RealReportingEngine',
                'type': ComponentType.REPORTING_ENGINE,
                'dependencies': ['backtesting_engine', 'performance_analyzer'],
                'priority': 'high'
            },
            'vectorbt_unified_manager': {
                'file': 'src/training/steps/backtesting/vectorbt_unified_manager.py',
                'class': 'VectorBTUnifiedManager',
                'type': ComponentType.VECTORBT_MANAGER,
                'dependencies': ['data_loader'],
                'priority': 'high'
            },
            'final_parameters_optimization': {
                'file': 'src/training/steps/backtesting/final_parameters_optimization.py',
                'class': 'FinalParametersOptimizer',
                'type': ComponentType.PARAMETER_OPTIMIZER,
                'dependencies': ['data_loader', 'feature_generator'],
                'priority': 'medium'
            },
            
            # ABC Testing components
            'paper_trading_engine': {
                'file': 'src/training/steps/backtesting/abc_testing/paper_trading_engine.py',
                'class': 'PaperTradingEngine',
                'type': ComponentType.PAPER_TRADING_ENGINE,
                'dependencies': ['data_loader', 'risk_management'],
                'priority': 'high'
            },
            'performance_monitoring': {
                'file': 'src/training/steps/backtesting/abc_testing/performance_monitoring.py',
                'class': 'PerformanceMonitor',
                'type': ComponentType.PERFORMANCE_MONITOR,
                'dependencies': ['backtesting_engine'],
                'priority': 'medium'
            },
            'risk_management': {
                'file': 'src/training/steps/backtesting/abc_testing/risk_management.py',
                'class': 'RiskManager',
                'type': ComponentType.RISK_MANAGER,
                'dependencies': ['data_loader'],
                'priority': 'high'
            },
            'statistical_analysis': {
                'file': 'src/training/steps/backtesting/abc_testing/statistical_analysis.py',
                'class': 'StatisticalAnalyzer',
                'type': ComponentType.STATISTICAL_ANALYZER,
                'dependencies': ['data_loader'],
                'priority': 'medium'
            },
            
            # Note: Legacy components have been removed and replaced by modular components
        }
    
    def analyze_component(self, component_name: str) -> Dict[str, Any]:
        """Analyze a specific component."""
        if component_name not in self.components_to_migrate:
            raise ValueError(f"Component {component_name} not found in migration list")
        
        component_info = self.components_to_migrate[component_name]
        file_path = component_info['file']
        class_name = component_info['class']
        
        console.print(f"\n🔍 [bold blue]Analyzing component:[/bold blue] {component_name}")
        console.print(f"📁 [dim]File:[/dim] {file_path}")
        console.print(f"🏗️  [dim]Class:[/dim] {class_name}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            console.print(f"❌ [bold red]File not found:[/bold red] {file_path}")
            return {
                'component_name': component_name,
                'status': 'error',
                'error': f"File not found: {file_path}"
            }
        
        try:
            # Analyze the component
            analysis = self.analyzer.analyze_component(file_path, class_name)
            
            result = {
                'component_name': component_name,
                'file_path': file_path,
                'class_name': class_name,
                'status': 'analyzed',
                'analysis': analysis.__dict__,
                'migration_recommendation': self._get_migration_recommendation(analysis)
            }
            
            console.print(f"✅ [bold green]Analysis completed for[/bold green] {component_name}")
            console.print(f"📊 [cyan]Compatibility score:[/cyan] {analysis.compatibility_score:.2f}")
            console.print(f"⚙️  [yellow]Migration complexity:[/yellow] {analysis.migration_complexity}")
            
            return result
            
        except Exception as e:
            console.print(f"❌ [bold red]Error analyzing component[/bold red] {component_name}: {e}")
            return {
                'component_name': component_name,
                'status': 'error',
                'error': str(e)
            }
    
    def analyze_all_components(self) -> Dict[str, Any]:
        """Analyze all components."""
        console.print("\n🔍 [bold blue]Analyzing all components...[/bold blue]")
        
        results = {}
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Analyzing components...", total=len(self.components_to_migrate))
            
            for component_name in self.components_to_migrate:
                try:
                    progress.update(task, description=f"Analyzing {component_name}")
                    results[component_name] = self.analyze_component(component_name)
                    progress.advance(task)
                except Exception as e:
                    console.print(f"❌ [bold red]Error analyzing[/bold red] {component_name}: {e}")
                    results[component_name] = {
                        'component_name': component_name,
                        'status': 'error',
                        'error': str(e)
                    }
                    progress.advance(task)
        
        # Generate summary
        total_components = len(self.components_to_migrate)
        analyzed_components = sum(1 for r in results.values() if r['status'] == 'analyzed')
        error_components = sum(1 for r in results.values() if r['status'] == 'error')
        
        summary = {
            'total_components': total_components,
            'analyzed_components': analyzed_components,
            'error_components': error_components,
            'analysis_timestamp': time.time(),
            'results': results
        }
        
        console.print(f"\n✅ [bold green]Analysis completed:[/bold green] {analyzed_components}/{total_components} components analyzed")
        
        return summary
    
    def migrate_component(self, component_name: str, strategy: Optional[str] = None) -> Dict[str, Any]:
        """Migrate a specific component."""
        if component_name not in self.components_to_migrate:
            raise ValueError(f"Component {component_name} not found in migration list")
        
        component_info = self.components_to_migrate[component_name]
        file_path = component_info['file']
        class_name = component_info['class']
        
        console.print(f"\n🚀 [bold blue]Migrating component:[/bold blue] {component_name}")
        console.print(f"📁 [dim]File:[/dim] {file_path}")
        console.print(f"🏗️  [dim]Class:[/dim] {class_name}")
        
        try:
            # First analyze the component
            analysis = self.analyzer.analyze_component(file_path, class_name)
            
            # Determine migration strategy
            if strategy is None:
                strategy = self._get_migration_recommendation(analysis)
            
            console.print(f"🎯 [cyan]Using migration strategy:[/cyan] {strategy}")
            
            # Migrate the component
            result = self.migrator.migrate_component(file_path, class_name, strategy)
            
            if result.success:
                console.print(f"✅ [bold green]Migration successful for[/bold green] {component_name}")
                
                # Register the migrated component
                self._register_migrated_component(component_name, result, component_info)
                
                return {
                    'component_name': component_name,
                    'status': 'migrated',
                    'strategy': strategy,
                    'migration_result': result.__dict__,
                    'registered': True
                }
            else:
                console.print(f"❌ [bold red]Migration failed for[/bold red] {component_name}: {result.issues}")
                return {
                    'component_name': component_name,
                    'status': 'failed',
                    'strategy': strategy,
                    'migration_result': result.__dict__,
                    'error': result.issues
                }
                
        except Exception as e:
            console.print(f"❌ [bold red]Error migrating component[/bold red] {component_name}: {e}")
            return {
                'component_name': component_name,
                'status': 'error',
                'error': str(e)
            }
    
    def migrate_all_components(self, strategy: Optional[str] = None) -> Dict[str, Any]:
        """Migrate all components."""
        console.print("\n🚀 [bold blue]Migrating all components...[/bold blue]")
        
        results = {}
        successful_migrations = 0
        failed_migrations = 0
        
        # Sort components by priority
        sorted_components = sorted(
            self.components_to_migrate.items(),
            key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x[1]['priority']]
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Migrating components...", total=len(sorted_components))
            
            for component_name, component_info in sorted_components:
                try:
                    priority_color = {'high': 'red', 'medium': 'yellow', 'low': 'green'}[component_info['priority']]
                    progress.update(task, description=f"Migrating {component_name} ({component_info['priority']} priority)")
                    
                    result = self.migrate_component(component_name, strategy)
                    results[component_name] = result
                    
                    if result['status'] == 'migrated':
                        successful_migrations += 1
                        console.print(f"✅ [bold green]{component_name}[/bold green] migrated successfully")
                    else:
                        failed_migrations += 1
                        console.print(f"❌ [bold red]{component_name}[/bold red] migration failed")
                    
                    progress.advance(task)
                    
                except Exception as e:
                    console.print(f"❌ [bold red]Error migrating[/bold red] {component_name}: {e}")
                    results[component_name] = {
                        'component_name': component_name,
                        'status': 'error',
                        'error': str(e)
                    }
                    failed_migrations += 1
                    progress.advance(task)
        
        summary = {
            'total_components': len(self.components_to_migrate),
            'successful_migrations': successful_migrations,
            'failed_migrations': failed_migrations,
            'migration_timestamp': time.time(),
            'results': results
        }
        
        console.print(f"\n✅ [bold green]Migration completed:[/bold green] {successful_migrations}/{len(self.components_to_migrate)} components migrated successfully")
        
        return summary
    
    def _get_migration_recommendation(self, analysis) -> str:
        """Get migration recommendation based on analysis."""
        if analysis.compatibility_score >= 0.8:
            return 'direct'
        elif analysis.compatibility_score >= 0.6:
            return 'refactor'
        elif analysis.compatibility_score >= 0.4:
            return 'wrapper'
        else:
            return 'rewrite'
    
    def _register_migrated_component(self, component_name: str, migration_result, component_info: Dict[str, Any]) -> None:
        """Register the migrated component in the registry."""
        try:
            # Get the migrated component class
            migrated_component = migration_result.migrated_component
            
            # Register in the registry
            self.registry.register_component(
                name=component_name,
                component_type=component_info['type'],
                component_class=migrated_component,
                dependencies=component_info['dependencies'],
                metadata={
                    'migrated': True,
                    'original_file': component_info['file'],
                    'migration_strategy': migration_result.strategy,
                    'migration_timestamp': time.time()
                }
            )
            
            console.print(f"📝 [green]Component {component_name} registered in registry[/green]")
            
        except Exception as e:
            console.print(f"❌ [bold red]Error registering component[/bold red] {component_name}: {e}")
    
    def create_wrapper_component(self, component_name: str) -> Dict[str, Any]:
        """Create a wrapper component for backward compatibility."""
        if component_name not in self.components_to_migrate:
            raise ValueError(f"Component {component_name} not found in migration list")
        
        component_info = self.components_to_migrate[component_name]
        file_path = component_info['file']
        class_name = component_info['class']
        
        try:
            # Import the original component
            import importlib.util
            spec = importlib.util.spec_from_file_location(class_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            original_class = getattr(module, class_name)
            
            # Create wrapper
            wrapper_class = create_backtesting_component_wrapper(original_class)
            
            # Register wrapper
            self.registry.register_component(
                name=f"{component_name}_wrapper",
                component_type=component_info['type'],
                component_class=wrapper_class,
                dependencies=component_info['dependencies'],
                metadata={
                    'wrapper': True,
                    'original_file': file_path,
                    'original_class': class_name,
                    'creation_timestamp': time.time()
                }
            )
            
            console.print(f"🔧 [green]Wrapper created for component[/green] {component_name}")
            
            return {
                'component_name': component_name,
                'status': 'wrapped',
                'wrapper_class': wrapper_class,
                'registered': True
            }
            
        except Exception as e:
            console.print(f"❌ [bold red]Error creating wrapper for[/bold red] {component_name}: {e}")
            return {
                'component_name': component_name,
                'status': 'error',
                'error': str(e)
            }
    
    def generate_migration_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate a comprehensive migration report."""
        if output_file is None:
            output_file = f"migration_report_{int(time.time())}.json"
        
        # Generate the report
        report = generate_backtesting_migration_report(results)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        console.print(f"📄 [green]Migration report saved to:[/green] {output_file}")
        return output_file


def main():
    """Main function for the migration script."""
    # Display banner
    console.print(Panel.fit(
        "[bold blue]Backtesting Components Migration Tool[/bold blue]\n"
        "Migrate existing backtesting components to ModularComponent architecture",
        border_style="blue"
    ))
    
    parser = argparse.ArgumentParser(description='Migrate existing backtesting components to ModularComponent')
    parser.add_argument('--analyze', action='store_true', help='Analyze all components')
    parser.add_argument('--analyze-component', type=str, help='Analyze a specific component')
    parser.add_argument('--migrate', type=str, help='Migrate a specific component')
    parser.add_argument('--migrate-all', action='store_true', help='Migrate all components')
    parser.add_argument('--create-wrapper', type=str, help='Create wrapper for a specific component')
    parser.add_argument('--strategy', type=str, choices=['direct', 'wrapper', 'refactor', 'rewrite'], 
                       help='Migration strategy to use')
    parser.add_argument('--output', type=str, help='Output file for reports')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        console.print("🔍 [yellow]Verbose logging enabled[/yellow]")
    
    manager = BacktestingComponentMigrationManager()
    
    try:
        if args.analyze:
            results = manager.analyze_all_components()
            output_file = manager.generate_migration_report(results, args.output)
            
            # Display analysis summary
            table = Table(title="Analysis Summary", box=box.ROUNDED)
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Compatibility", style="yellow")
            table.add_column("Complexity", style="magenta")
            
            for component_name, result in results.items():
                if result['status'] == 'analyzed':
                    analysis = result['analysis']
                    table.add_row(
                        component_name,
                        "✅ Analyzed",
                        f"{analysis['compatibility_score']:.2f}",
                        analysis['migration_complexity']
                    )
                else:
                    table.add_row(component_name, "❌ Error", "N/A", "N/A")
            
            console.print(table)
            console.print(f"\n📄 [green]Analysis completed. Report saved to:[/green] {output_file}")
            
        elif args.analyze_component:
            result = manager.analyze_component(args.analyze_component)
            
            # Display component analysis
            if result['status'] == 'analyzed':
                analysis = result['analysis']
                console.print(f"\n📊 [bold blue]Component Analysis:[/bold blue] {args.analyze_component}")
                console.print(f"📁 File: {result['file_path']}")
                console.print(f"🏗️  Class: {result['class_name']}")
                console.print(f"📈 Compatibility Score: {analysis['compatibility_score']:.2f}")
                console.print(f"⚙️  Migration Complexity: {analysis['migration_complexity']}")
                console.print(f"🎯 Recommended Strategy: {result['migration_recommendation']}")
            else:
                console.print(f"❌ [bold red]Analysis failed:[/bold red] {result.get('error', 'Unknown error')}")
            
        elif args.migrate:
            result = manager.migrate_component(args.migrate, args.strategy)
            
            # Display migration result
            if result['status'] == 'migrated':
                console.print(f"✅ [bold green]Migration successful for[/bold green] {args.migrate}")
                console.print(f"🎯 Strategy: {result['strategy']}")
                console.print(f"📝 Registered: {result['registered']}")
            else:
                console.print(f"❌ [bold red]Migration failed for[/bold red] {args.migrate}")
                console.print(f"🎯 Strategy: {result['strategy']}")
                if 'error' in result:
                    console.print(f"💥 Error: {result['error']}")
            
        elif args.migrate_all:
            results = manager.migrate_all_components(args.strategy)
            output_file = manager.generate_migration_report(results, args.output)
            
            # Display migration summary
            table = Table(title="Migration Summary", box=box.ROUNDED)
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Strategy", style="yellow")
            table.add_column("Registered", style="magenta")
            
            for component_name, result in results.items():
                status_style = "green" if result['status'] == 'migrated' else "red"
                status_text = "✅ Migrated" if result['status'] == 'migrated' else f"❌ {result['status'].title()}"
                
                table.add_row(
                    component_name,
                    status_text,
                    result.get('strategy', 'N/A'),
                    "Yes" if result.get('registered', False) else "No"
                )
            
            console.print(table)
            console.print(f"\n📄 [green]Migration completed. Report saved to:[/green] {output_file}")
            
        elif args.create_wrapper:
            result = manager.create_wrapper_component(args.create_wrapper)
            
            if result['status'] == 'wrapped':
                console.print(f"✅ [bold green]Wrapper created for[/bold green] {args.create_wrapper}")
                console.print(f"📝 Registered: {result['registered']}")
            else:
                console.print(f"❌ [bold red]Wrapper creation failed for[/bold red] {args.create_wrapper}")
                if 'error' in result:
                    console.print(f"💥 Error: {result['error']}")
            
        else:
            console.print("\n[bold yellow]Available Commands:[/bold yellow]")
            console.print("  [cyan]--analyze[/cyan]              Analyze all components")
            console.print("  [cyan]--analyze-component NAME[/cyan]  Analyze specific component")
            console.print("  [cyan]--migrate NAME[/cyan]         Migrate specific component")
            console.print("  [cyan]--migrate-all[/cyan]          Migrate all components")
            console.print("  [cyan]--create-wrapper NAME[/cyan]  Create wrapper for component")
            console.print("\n[bold yellow]Options:[/bold yellow]")
            console.print("  [cyan]--strategy STRATEGY[/cyan]    Migration strategy (direct, wrapper, refactor, rewrite)")
            console.print("  [cyan]--output FILE[/cyan]          Output file for reports")
            console.print("  [cyan]--verbose[/cyan]              Enable verbose logging")
            
    except Exception as e:
        console.print(f"❌ [bold red]Error:[/bold red] {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()