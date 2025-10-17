"""
Migration Example: Real Monte Carlo Engine to ModularComponent

This script demonstrates how to migrate the existing RealMonteCarloEngine
to use the ModularComponent architecture.
"""

import sys
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich import print as rprint
from rich import box

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
    ComponentType
)

class MigratedMonteCarloEngine(ModularComponent):
    """
    Migrated Monte Carlo Engine using ModularComponent architecture.
    
    This is a migrated version of the RealMonteCarloEngine that inherits
    from ModularComponent and provides all the backtesting-specific features.
    """
    
    def __init__(self, config: dict = None, logger=None):
        super().__init__(
            name="monte_carlo_engine",
            config=config or {},
            logger=logger
        )
        
        # Monte Carlo specific configuration
        self._simulation_config = self.config.get('simulation', {})
        self._n_simulations = self._simulation_config.get('n_simulations', 1000)
        self._confidence_levels = self._simulation_config.get('confidence_levels', [0.95, 0.99])
        self._simulation_method = self._simulation_config.get('method', 'bootstrap')
        
        # Backtesting specific state
        self._portfolio_state = {}
        self._trade_history = []
        self._performance_metrics = {}
    
    def _initialize_resources(self) -> bool:
        """Initialize Monte Carlo simulation resources."""
        try:
            # Initialize simulation parameters
            self._simulation_config = {
                'n_simulations': self.config.get('simulation', {}).get('n_simulations', 1000),
                'confidence_levels': self.config.get('simulation', {}).get('confidence_levels', [0.95, 0.99]),
                'method': self.config.get('simulation', {}).get('method', 'bootstrap'),
                'random_seed': self.config.get('simulation', {}).get('random_seed', 42)
            }
            
            # Initialize portfolio state
            self._portfolio_state = {
                'initial_capital': self.config.get('backtesting', {}).get('initial_capital', 100000.0),
                'current_capital': self.config.get('backtesting', {}).get('initial_capital', 100000.0),
                'positions': {},
                'cash': self.config.get('backtesting', {}).get('initial_capital', 100000.0)
            }
            
            # Initialize performance tracking
            self._performance_metrics = {
                'total_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'var_95': 0.0,
                'var_99': 0.0,
                'expected_shortfall': 0.0
            }
            
            self.logger.info("Monte Carlo Engine resources initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Monte Carlo Engine resources: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup Monte Carlo simulation resources."""
        try:
            # Clear simulation data
            self._portfolio_state = {}
            self._trade_history = []
            self._performance_metrics = {}
            
            # Clear any cached data
            if hasattr(self, '_cached_simulations'):
                delattr(self, '_cached_simulations')
            
            self.logger.info("Monte Carlo Engine resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during Monte Carlo Engine cleanup: {e}")
    
    def _process_data(self, data: any, **kwargs) -> any:
        """Process market data through Monte Carlo simulation."""
        try:
            # Validate input data
            if not self._validate_market_data(data):
                raise ValueError("Invalid market data provided")
            
            # Extract market data
            prices = data.get('prices', [])
            returns = data.get('returns', [])
            
            if not prices and not returns:
                raise ValueError("No price or return data provided")
            
            # Run Monte Carlo simulation
            simulation_results = self._run_monte_carlo_simulation(prices, returns)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(simulation_results)
            
            # Update portfolio state
            self._update_portfolio_state(simulation_results)
            
            # Return results
            return {
                'simulation_results': simulation_results,
                'performance_metrics': performance_metrics,
                'portfolio_state': self._portfolio_state.copy(),
                'trade_history': self._trade_history.copy()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing data in Monte Carlo Engine: {e}")
            raise
    
    def _validate_market_data(self, data: any) -> bool:
        """Validate market data for Monte Carlo simulation."""
        if not isinstance(data, dict):
            return False
        
        # Check for required fields
        if 'prices' not in data and 'returns' not in data:
            return False
        
        # Validate prices if provided
        if 'prices' in data:
            prices = data['prices']
            if not isinstance(prices, (list, tuple)) or len(prices) == 0:
                return False
            
            # Check for valid price values
            for price in prices:
                if not isinstance(price, (int, float)) or price <= 0:
                    return False
        
        # Validate returns if provided
        if 'returns' in data:
            returns = data['returns']
            if not isinstance(returns, (list, tuple)) or len(returns) == 0:
                return False
        
        return True
    
    def _run_monte_carlo_simulation(self, prices: list, returns: list) -> dict:
        """Run Monte Carlo simulation on the provided data."""
        import numpy as np
        import random
        
        # Set random seed for reproducibility
        random.seed(self._simulation_config.get('random_seed', 42))
        np.random.seed(self._simulation_config.get('random_seed', 42))
        
        # Prepare data
        if returns:
            data = returns
        else:
            # Calculate returns from prices
            data = []
            for i in range(1, len(prices)):
                ret = (prices[i] - prices[i-1]) / prices[i-1]
                data.append(ret)
        
        n_simulations = self._simulation_config['n_simulations']
        n_periods = len(data)
        
        # Run simulations
        simulation_results = []
        
        for sim in range(n_simulations):
            # Generate random sample
            if self._simulation_method == 'bootstrap':
                # Bootstrap sampling
                sample = random.choices(data, k=n_periods)
            elif self._simulation_method == 'parametric':
                # Parametric sampling (normal distribution)
                mean_ret = np.mean(data)
                std_ret = np.std(data)
                sample = np.random.normal(mean_ret, std_ret, n_periods).tolist()
            else:
                # Default to bootstrap
                sample = random.choices(data, k=n_periods)
            
            # Calculate portfolio value over time
            portfolio_values = [self._portfolio_state['initial_capital']]
            for ret in sample:
                new_value = portfolio_values[-1] * (1 + ret)
                portfolio_values.append(new_value)
            
            simulation_results.append(portfolio_values)
        
        return {
            'simulation_results': simulation_results,
            'n_simulations': n_simulations,
            'n_periods': n_periods,
            'method': self._simulation_method
        }
    
    def _calculate_performance_metrics(self, simulation_results: dict) -> dict:
        """Calculate performance metrics from simulation results."""
        import numpy as np
        
        results = simulation_results['simulation_results']
        n_simulations = len(results)
        
        # Calculate final returns for each simulation
        final_returns = []
        for sim_result in results:
            if len(sim_result) > 1:
                final_return = (sim_result[-1] - sim_result[0]) / sim_result[0]
                final_returns.append(final_return)
        
        if not final_returns:
            return self._performance_metrics
        
        # Calculate metrics
        mean_return = np.mean(final_returns)
        std_return = np.std(final_returns)
        
        # Sharpe ratio (assuming risk-free rate of 0)
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        # Value at Risk
        var_95 = np.percentile(final_returns, 5)  # 95% VaR
        var_99 = np.percentile(final_returns, 1)  # 99% VaR
        
        # Expected Shortfall (Conditional VaR)
        es_95 = np.mean([r for r in final_returns if r <= var_95])
        es_99 = np.mean([r for r in final_returns if r <= var_99])
        
        # Maximum drawdown
        max_drawdown = 0
        for sim_result in results:
            peak = sim_result[0]
            for value in sim_result:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        
        # Update performance metrics
        self._performance_metrics = {
            'total_return': mean_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'n_simulations': n_simulations
        }
        
        return self._performance_metrics
    
    def _update_portfolio_state(self, simulation_results: dict) -> None:
        """Update portfolio state based on simulation results."""
        # This is a simplified update - in a real implementation,
        # you would update based on actual trading decisions
        self._portfolio_state['simulation_completed'] = True
        self._portfolio_state['last_simulation_time'] = self._get_current_time()
    
    def _get_validation_rules(self) -> dict:
        """Get validation rules for Monte Carlo simulation."""
        return {
            'market_data': {
                'required_fields': ['prices', 'returns'],
                'at_least_one': ['prices', 'returns'],
                'data_types': {
                    'prices': (list, tuple),
                    'returns': (list, tuple)
                }
            },
            'simulation_config': {
                'n_simulations': {
                    'type': int,
                    'min': 100,
                    'max': 10000
                },
                'confidence_levels': {
                    'type': list,
                    'min_length': 1,
                    'max_length': 5
                }
            }
        }
    
    def _validate_component_specific(self, data: any) -> bool:
        """Validate component-specific data."""
        if not isinstance(data, dict):
            return False
        
        # Check for market data
        if 'prices' not in data and 'returns' not in data:
            return False
        
        # Validate simulation configuration
        sim_config = self.config.get('simulation', {})
        n_simulations = sim_config.get('n_simulations', 1000)
        
        if not isinstance(n_simulations, int) or n_simulations < 100 or n_simulations > 10000:
            return False
        
        return True
    
    def get_simulation_config(self) -> dict:
        """Get current simulation configuration."""
        return self._simulation_config.copy()
    
    def update_simulation_config(self, config: dict) -> bool:
        """Update simulation configuration."""
        try:
            # Validate new configuration
            if 'n_simulations' in config:
                if not isinstance(config['n_simulations'], int) or config['n_simulations'] < 100:
                    return False
            
            # Update configuration
            self._simulation_config.update(config)
            self.config['simulation'].update(config)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating simulation config: {e}")
            return False
    
    def get_performance_metrics(self) -> dict:
        """Get current performance metrics."""
        return self._performance_metrics.copy()
    
    def get_portfolio_state(self) -> dict:
        """Get current portfolio state."""
        return self._portfolio_state.copy()


# Initialize Rich console
console = Console()

def create_migrated_monte_carlo_engine(config: dict = None) -> MigratedMonteCarloEngine:
    """Create a migrated Monte Carlo engine instance."""
    return MigratedMonteCarloEngine(config)


def register_migrated_monte_carlo_engine():
    """Register the migrated Monte Carlo engine in the component registry."""
    registry = get_registry()
    
    # Register the component
    registry.register_component(
        name='monte_carlo_engine',
        component_type=ComponentType.MONTE_CARLO_ENGINE,
        component_class=MigratedMonteCarloEngine,
        dependencies=['data_loader', 'feature_generator'],
        metadata={
            'migrated': True,
            'original_file': 'src/training/steps/backtesting/real_monte_carlo_engine.py',
            'migration_strategy': 'direct',
            'migration_timestamp': time.time()
        }
    )
    
    console.print("✅ [bold green]Migrated Monte Carlo Engine registered successfully[/bold green]")


if __name__ == '__main__':
    # Example usage
    import time
    
    # Create configuration
    config = {
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
    
    # Create migrated component
    engine = create_migrated_monte_carlo_engine(config)
    
    # Display banner
    console.print(Panel.fit(
        "[bold blue]Monte Carlo Engine Migration Demo[/bold blue]\n"
        "Demonstrating migrated Monte Carlo engine functionality",
        border_style="blue"
    ))
    
    # Initialize
    if engine.initialize():
        console.print("✅ [bold green]Monte Carlo Engine initialized successfully[/bold green]")
        
        # Example data
        sample_data = {
            'prices': [100, 102, 101, 103, 105, 104, 106, 108, 107, 109],
            'returns': [0.02, -0.01, 0.02, 0.02, -0.01, 0.02, 0.02, -0.01, 0.02]
        }
        
        # Process data
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Running Monte Carlo simulation...", total=100)
            
            result = engine.process(sample_data)
            progress.update(task, completed=100)
        
        # Display results
        console.print("\n📊 [bold blue]Simulation Results:[/bold blue]")
        table = Table(title="Performance Metrics", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        metrics = result['performance_metrics']
        table.add_row("Total Return", f"{metrics['total_return']:.2%}")
        table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
        table.add_row("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        table.add_row("VaR 95%", f"{metrics['var_95']:.2%}")
        table.add_row("VaR 99%", f"{metrics['var_99']:.2%}")
        table.add_row("Expected Shortfall 95%", f"{metrics['expected_shortfall_95']:.2%}")
        table.add_row("Expected Shortfall 99%", f"{metrics['expected_shortfall_99']:.2%}")
        table.add_row("Simulations", f"{metrics['n_simulations']}")
        
        console.print(table)
        
        # Cleanup
        engine.cleanup()
        console.print("🧹 [yellow]Monte Carlo Engine cleaned up[/yellow]")
    
    # Register in registry
    register_migrated_monte_carlo_engine()