"""
Monte Carlo Simulation Step.

This step performs Monte Carlo backtesting simulation using RealMonteCarloEngine.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint

# Real Monte Carlo Engine
from src.training.steps.backtesting.real_monte_carlo_engine import (
    RealMonteCarloEngine, RealMonteCarloConfig, MonteCarloMode
)

logger = logging.getLogger(__name__)


class MonteCarloSimulationStep(BaseStep):
    """
    Monte Carlo Simulation Step.

    Performs Monte Carlo simulation for robustness testing.
    """

    def __init__(self, step_name: str = "monte_carlo_simulation"):
        """Initialize the Monte Carlo simulation step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('MonteCarloSimulation')

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Monte Carlo simulation using RealMonteCarloEngine.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - direction: Trading direction ('long', 'short', 'both')
                - n_simulations: Number of simulations (default: 1000)
                - portfolio_value: Initial portfolio value (default: 100000)
                - data_dir: Data directory (default: 'historical_data')

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        symbol = config.get('symbol', 'UNKNOWN')
        tprint(f"🎲 Starting Monte Carlo simulation for {symbol}", "INFO")

        try:
            # Load market data using BaseStep pattern
            market_data = await self._load_market_data(config)
            
            if market_data is None or len(market_data) < 30:
                error_msg = f"Insufficient market data for Monte Carlo simulation: {len(market_data) if market_data is not None else 0} samples"
                tprint(f"❌ {error_msg}", "ERROR")
                return {
                    'success': False,
                    'artifacts': {},
                    'metrics': {},
                    'error': error_msg
                }
            
            # Calculate returns from market data
            returns_data = market_data['close'].pct_change().dropna()
            
            if len(returns_data) < 30:
                error_msg = f"Insufficient returns data for Monte Carlo simulation: {len(returns_data)} samples"
                tprint(f"❌ {error_msg}", "ERROR")
                return {
                    'success': False,
                    'artifacts': {},
                    'metrics': {},
                    'error': error_msg
                }
            
            tprint(f"📊 Loaded {len(returns_data)} return samples from market data", "INFO")
            
            # Configure Monte Carlo engine
            n_simulations = config.get('n_simulations', 1000)
            portfolio_value = config.get('portfolio_value', 100000.0)
            execution_mode = config.get('execution_mode', 'light')
            
            # Adjust simulations based on execution mode
            if execution_mode == 'light':
                n_simulations = min(n_simulations, 500)
            elif execution_mode == 'blank':
                n_simulations = min(n_simulations, 2000)
            
            mc_config = RealMonteCarloConfig(
                n_simulations=n_simulations,
                confidence_level=0.95,
                simulation_horizon=min(252, len(returns_data)),
                mode=MonteCarloMode.HYBRID,
                enable_gpu_acceleration=True,
                enable_memory_optimization=True,
                enable_parallel_processing=True,
                enable_data_validation=True,
                enable_leakage_detection=False,  # Disable for speed in light mode
                enable_cv_validation=False,  # Disable for speed
                save_results=False,  # Don't save to disk
                enable_detailed_logging=True
            )
            
            tprint(f"🚀 Initializing Monte Carlo Engine (mode: {mc_config.mode.value}, simulations: {n_simulations})", "INFO")
            
            # Initialize and run engine
            engine = RealMonteCarloEngine(mc_config)
            results = await engine.run_simulation(returns_data, portfolio_value)
            
            # Extract metrics from results
            metrics_dict = results['metrics'].to_dict() if hasattr(results['metrics'], 'to_dict') else results['metrics']
            risk_metrics = results.get('risk_metrics', {})
            
            # Build artifacts
            artifacts = {
                'monte_carlo_simulation': {
                    'simulation_method': mc_config.mode.value,
                    'n_simulations': n_simulations,
                    'portfolio_value': portfolio_value,
                    'risk_metrics': risk_metrics,
                    'execution_time': results.get('execution_time', 0.0),
                    'data_statistics': results.get('data_statistics', {}),
                    'metadata': {
                        'symbol': config['symbol'],
                        'exchange': config['exchange'],
                        'timeframe': config['timeframe'],
                        'direction': config.get('direction', 'long'),
                        'execution_mode': execution_mode,
                        'created_at': datetime.now().isoformat(),
                        'samples_used': len(returns_data)
                    }
                }
            }
            
            # Flatten metrics for easier access
            metrics = {
                'simulation_method': mc_config.mode.value,
                'n_simulations': n_simulations,
                'portfolio_value': portfolio_value,
                'execution_time': results.get('execution_time', 0.0),
                'samples_used': len(returns_data),
                'direction': config.get('direction', 'long'),
                'execution_mode': execution_mode,
                'success': True
            }
            
            # Add risk metrics to top level
            if isinstance(risk_metrics, dict):
                for category, category_metrics in risk_metrics.items():
                    if isinstance(category_metrics, dict):
                        for metric_name, metric_value in category_metrics.items():
                            metrics[f"{category}_{metric_name}"] = metric_value
            
            tprint(f"✅ Monte Carlo simulation completed successfully", "SUCCESS")
            tprint(f"   Simulations: {n_simulations:,}", "INFO")
            tprint(f"   Execution time: {results.get('execution_time', 0.0):.2f}s", "INFO")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics
            }

        except Exception as e:
            error_msg = f"Monte Carlo simulation failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg)
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

            return {
                'success': False,
                'artifacts': {},
                'metrics': {},
                'error': error_msg
            }
    
    async def _load_market_data(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Load market data using klines manager (BaseStep pattern).
        
        Args:
            config: Configuration dictionary
            
        Returns:
            DataFrame with market data or None if loading fails
        """
        try:
            tprint("📂 Loading market data for Monte Carlo simulation", "INFO")
            
            # Import klines manager
            from src.utils.data.klines_parquet import get_klines_manager
            
            # Get klines manager
            data_dir = config.get('data_dir', 'historical_data')
            klines_manager = get_klines_manager(data_dir=data_dir)
            
            # Parse date filters if provided
            start_date = None
            end_date = None
            
            if 'start_date' in config and config['start_date']:
                start_date = pd.to_datetime(config['start_date'])
                tprint(f"📅 Using start_date filter: {start_date}", "INFO")
            
            if 'end_date' in config and config['end_date']:
                end_date = pd.to_datetime(config['end_date'])
                tprint(f"📅 Using end_date filter: {end_date}", "INFO")
            
            # Load market data
            market_data = klines_manager.read_data(
                symbol=config['symbol'],
                interval=config['timeframe'],
                data_type="processed",
                start_date=start_date,
                end_date=end_date
            )
            
            if market_data is not None and len(market_data) > 0:
                tprint(f"✅ Loaded {len(market_data)} rows of market data", "SUCCESS")
                return market_data
            else:
                tprint("⚠️  No market data found", "WARNING")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load market data: {e}")
            tprint(f"❌ Failed to load market data: {e}", "ERROR")
            return None

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Register the step (only if not already registered by __init__.py)
def register_monte_carlo_simulation_step():
    """Register the Monte Carlo simulation step."""
    from src.training.steps.base_step import step_registry
    
    # Check if already registered to avoid duplicates
    if not step_registry.is_registered("monte_carlo_simulation"):
        step_registry.register("monte_carlo_simulation", MonteCarloSimulationStep)
        tprint("✅ Monte Carlo simulation step registered", "SUCCESS")


# Auto-register when module is imported
register_monte_carlo_simulation_step()
