"""
Migration Example: Paper Trading Engine to ModularComponent

This script demonstrates how to migrate the existing PaperTradingEngine
to use the ModularComponent architecture.
"""

import sys
import os
from pathlib import Path

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

class MigratedPaperTradingEngine(ModularComponent):
    """
    Migrated Paper Trading Engine using ModularComponent architecture.
    
    This is a migrated version of the PaperTradingEngine that inherits
    from ModularComponent and provides all the backtesting-specific features.
    """
    
    def __init__(self, config: dict = None, logger=None):
        super().__init__(
            name="paper_trading_engine",
            config=config or {},
            logger=logger
        )
        
        # Paper trading specific configuration
        self._trading_config = self.config.get('trading', {})
        self._market_config = self.config.get('market', {})
        self._risk_config = self.config.get('risk', {})
        
        # Trading state
        self._portfolio_state = {}
        self._trade_history = []
        self._order_book = {}
        self._position_tracker = {}
        self._performance_metrics = {}
    
    def _initialize_resources(self) -> bool:
        """Initialize paper trading engine resources."""
        try:
            # Initialize trading configuration
            self._trading_config = {
                'initial_capital': self.config.get('trading', {}).get('initial_capital', 100000.0),
                'commission_rate': self.config.get('trading', {}).get('commission_rate', 0.001),
                'slippage_rate': self.config.get('trading', {}).get('slippage_rate', 0.0005),
                'min_trade_size': self.config.get('trading', {}).get('min_trade_size', 0.01)
            }
            
            # Initialize market configuration
            self._market_config = {
                'enable_slippage': self.config.get('market', {}).get('enable_slippage', True),
                'enable_latency': self.config.get('market', {}).get('enable_latency', True),
                'latency_ms': self.config.get('market', {}).get('latency_ms', 100),
                'spread_bps': self.config.get('market', {}).get('spread_bps', 5)
            }
            
            # Initialize risk configuration
            self._risk_config = {
                'max_position_size': self.config.get('risk', {}).get('max_position_size', 0.1),
                'max_drawdown': self.config.get('risk', {}).get('max_drawdown', 0.15),
                'stop_loss_pct': self.config.get('risk', {}).get('stop_loss_pct', 0.05),
                'take_profit_pct': self.config.get('risk', {}).get('take_profit_pct', 0.10)
            }
            
            # Initialize portfolio state
            self._portfolio_state = {
                'cash': self._trading_config['initial_capital'],
                'total_value': self._trading_config['initial_capital'],
                'positions': {},
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'total_commission': 0.0
            }
            
            # Initialize performance metrics
            self._performance_metrics = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
            
            self.logger.info("Paper Trading Engine resources initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Paper Trading Engine resources: {e}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Cleanup paper trading engine resources."""
        try:
            # Clear trading state
            self._portfolio_state = {}
            self._trade_history = []
            self._order_book = {}
            self._position_tracker = {}
            self._performance_metrics = {}
            
            # Clear any cached data
            if hasattr(self, '_cached_prices'):
                delattr(self, '_cached_prices')
            
            self.logger.info("Paper Trading Engine resources cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during Paper Trading Engine cleanup: {e}")
    
    def _process_data(self, data: any, **kwargs) -> any:
        """Process trading data through paper trading engine."""
        try:
            # Validate input data
            if not self._validate_trading_data(data):
                raise ValueError("Invalid trading data provided")
            
            # Extract trading signals and market data
            signals = data.get('signals', [])
            market_data = data.get('market_data', {})
            current_prices = market_data.get('prices', {})
            
            # Process trading signals
            trading_results = self._process_trading_signals(signals, current_prices)
            
            # Update portfolio state
            self._update_portfolio_state(trading_results, current_prices)
            
            # Calculate performance metrics
            self._calculate_performance_metrics()
            
            # Return results
            return {
                'trading_results': trading_results,
                'portfolio_state': self._portfolio_state.copy(),
                'trade_history': self._trade_history[-10:],  # Last 10 trades
                'performance_metrics': self._performance_metrics.copy()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing data in Paper Trading Engine: {e}")
            raise
    
    def _validate_trading_data(self, data: any) -> bool:
        """Validate trading data."""
        if not isinstance(data, dict):
            return False
        
        # Check for required fields
        if 'signals' not in data and 'market_data' not in data:
            return False
        
        # Validate signals if provided
        if 'signals' in data:
            signals = data['signals']
            if not isinstance(signals, list):
                return False
            
            # Validate signal structure
            for signal in signals:
                if not isinstance(signal, dict):
                    return False
                if 'action' not in signal or 'symbol' not in signal:
                    return False
        
        # Validate market data if provided
        if 'market_data' in data:
            market_data = data['market_data']
            if not isinstance(market_data, dict):
                return False
        
        return True
    
    def _process_trading_signals(self, signals: list, current_prices: dict) -> list:
        """Process trading signals and execute trades."""
        trading_results = []
        
        for signal in signals:
            try:
                action = signal.get('action')
                symbol = signal.get('symbol')
                quantity = signal.get('quantity', 0)
                price = signal.get('price', 0)
                
                if not action or not symbol:
                    continue
                
                # Get current market price
                current_price = current_prices.get(symbol, price)
                if current_price <= 0:
                    continue
                
                # Apply slippage if enabled
                if self._market_config['enable_slippage']:
                    slippage = self._calculate_slippage(symbol, quantity, current_price)
                    execution_price = current_price * (1 + slippage)
                else:
                    execution_price = current_price
                
                # Execute trade
                trade_result = self._execute_trade(
                    action, symbol, quantity, execution_price
                )
                
                if trade_result:
                    trading_results.append(trade_result)
                    
            except Exception as e:
                self.logger.error(f"Error processing signal {signal}: {e}")
                continue
        
        return trading_results
    
    def _calculate_slippage(self, symbol: str, quantity: float, price: float) -> float:
        """Calculate slippage for a trade."""
        # Simple slippage model based on trade size and market conditions
        base_slippage = self._trading_config['slippage_rate']
        
        # Increase slippage for larger trades
        size_factor = min(quantity / 1000, 1.0)  # Normalize to 1000 units
        size_slippage = base_slippage * size_factor
        
        # Add random component
        import random
        random_slippage = random.uniform(-0.5, 0.5) * base_slippage
        
        total_slippage = size_slippage + random_slippage
        return max(0, total_slippage)  # No negative slippage
    
    def _execute_trade(self, action: str, symbol: str, quantity: float, price: float) -> dict:
        """Execute a trade."""
        try:
            # Check if we have enough cash for buy orders
            if action.upper() == 'BUY':
                required_cash = quantity * price * (1 + self._trading_config['commission_rate'])
                if required_cash > self._portfolio_state['cash']:
                    self.logger.warning(f"Insufficient cash for buy order: {symbol}")
                    return None
            
            # Check if we have enough position for sell orders
            elif action.upper() == 'SELL':
                current_position = self._portfolio_state['positions'].get(symbol, 0)
                if quantity > current_position:
                    self.logger.warning(f"Insufficient position for sell order: {symbol}")
                    return None
            
            # Calculate commission
            commission = quantity * price * self._trading_config['commission_rate']
            
            # Execute the trade
            if action.upper() == 'BUY':
                self._portfolio_state['cash'] -= (quantity * price + commission)
                self._portfolio_state['positions'][symbol] = self._portfolio_state['positions'].get(symbol, 0) + quantity
            elif action.upper() == 'SELL':
                self._portfolio_state['cash'] += (quantity * price - commission)
                self._portfolio_state['positions'][symbol] = self._portfolio_state['positions'].get(symbol, 0) - quantity
            
            # Update total commission
            self._portfolio_state['total_commission'] += commission
            
            # Create trade record
            trade_record = {
                'timestamp': self._get_current_time(),
                'action': action.upper(),
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'commission': commission,
                'total_value': quantity * price
            }
            
            # Add to trade history
            self._trade_history.append(trade_record)
            
            # Update performance metrics
            self._performance_metrics['total_trades'] += 1
            
            return trade_record
            
        except Exception as e:
            self.logger.error(f"Error executing trade: {e}")
            return None
    
    def _update_portfolio_state(self, trading_results: list, current_prices: dict) -> None:
        """Update portfolio state based on trading results."""
        # Calculate unrealized P&L
        unrealized_pnl = 0.0
        total_value = self._portfolio_state['cash']
        
        for symbol, position in self._portfolio_state['positions'].items():
            if position != 0 and symbol in current_prices:
                current_price = current_prices[symbol]
                position_value = position * current_price
                total_value += position_value
                
                # Calculate unrealized P&L (simplified)
                # In a real implementation, you would track entry prices
                unrealized_pnl += position_value * 0.01  # Simplified calculation
        
        # Update portfolio state
        self._portfolio_state['unrealized_pnl'] = unrealized_pnl
        self._portfolio_state['total_value'] = total_value
        
        # Calculate realized P&L from trade history
        realized_pnl = 0.0
        for trade in self._trade_history:
            if trade['action'] == 'SELL':
                # Simplified realized P&L calculation
                realized_pnl += trade['total_value'] * 0.01
        
        self._portfolio_state['realized_pnl'] = realized_pnl
    
    def _calculate_performance_metrics(self) -> None:
        """Calculate performance metrics."""
        if not self._trade_history:
            return
        
        # Calculate win/loss statistics
        winning_trades = 0
        losing_trades = 0
        total_wins = 0.0
        total_losses = 0.0
        
        for trade in self._trade_history:
            if trade['action'] == 'SELL':
                # Simplified win/loss calculation
                pnl = trade['total_value'] * 0.01  # Simplified
                if pnl > 0:
                    winning_trades += 1
                    total_wins += pnl
                else:
                    losing_trades += 1
                    total_losses += abs(pnl)
        
        # Update metrics
        self._performance_metrics['winning_trades'] = winning_trades
        self._performance_metrics['losing_trades'] = losing_trades
        self._performance_metrics['win_rate'] = winning_trades / max(winning_trades + losing_trades, 1)
        self._performance_metrics['avg_win'] = total_wins / max(winning_trades, 1)
        self._performance_metrics['avg_loss'] = total_losses / max(losing_trades, 1)
        self._performance_metrics['profit_factor'] = total_wins / max(total_losses, 1)
    
    def _get_validation_rules(self) -> dict:
        """Get validation rules for paper trading."""
        return {
            'trading_data': {
                'required_fields': ['signals', 'market_data'],
                'at_least_one': ['signals', 'market_data'],
                'data_types': {
                    'signals': list,
                    'market_data': dict
                }
            },
            'trading_config': {
                'initial_capital': {
                    'type': (int, float),
                    'min': 1000
                },
                'commission_rate': {
                    'type': (int, float),
                    'min': 0,
                    'max': 0.1
                }
            }
        }
    
    def _validate_component_specific(self, data: any) -> bool:
        """Validate component-specific data."""
        if not isinstance(data, dict):
            return False
        
        # Check for trading data
        if 'signals' not in data and 'market_data' not in data:
            return False
        
        # Validate trading configuration
        trading_config = self.config.get('trading', {})
        initial_capital = trading_config.get('initial_capital', 100000.0)
        
        if not isinstance(initial_capital, (int, float)) or initial_capital < 1000:
            return False
        
        return True
    
    def get_portfolio_state(self) -> dict:
        """Get current portfolio state."""
        return self._portfolio_state.copy()
    
    def get_trade_history(self, limit: int = 10) -> list:
        """Get trade history."""
        return self._trade_history[-limit:] if limit else self._trade_history
    
    def get_performance_metrics(self) -> dict:
        """Get current performance metrics."""
        return self._performance_metrics.copy()
    
    def get_trading_config(self) -> dict:
        """Get current trading configuration."""
        return self._trading_config.copy()
    
    def update_trading_config(self, config: dict) -> bool:
        """Update trading configuration."""
        try:
            # Validate new configuration
            if 'initial_capital' in config:
                if not isinstance(config['initial_capital'], (int, float)) or config['initial_capital'] < 1000:
                    return False
            
            # Update configuration
            self._trading_config.update(config)
            self.config['trading'].update(config)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating trading config: {e}")
            return False


def create_migrated_paper_trading_engine(config: dict = None) -> MigratedPaperTradingEngine:
    """Create a migrated paper trading engine instance."""
    return MigratedPaperTradingEngine(config)


def register_migrated_paper_trading_engine():
    """Register the migrated paper trading engine in the component registry."""
    registry = get_registry()
    
    # Register the component
    registry.register_component(
        name='paper_trading_engine',
        component_type=ComponentType.PAPER_TRADING_ENGINE,
        component_class=MigratedPaperTradingEngine,
        dependencies=['data_loader', 'risk_management'],
        metadata={
            'migrated': True,
            'original_file': 'src/training/steps/backtesting/abc_testing/paper_trading_engine.py',
            'migration_strategy': 'direct',
            'migration_timestamp': time.time()
        }
    )
    
    print("Migrated Paper Trading Engine registered successfully")


if __name__ == '__main__':
    # Example usage
    import time
    
    # Create configuration
    config = {
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
    
    # Create migrated component
    engine = create_migrated_paper_trading_engine(config)
    
    # Initialize
    if engine.initialize():
        print("Paper Trading Engine initialized successfully")
        
        # Example data
        sample_data = {
            'signals': [
                {'action': 'BUY', 'symbol': 'BTCUSDT', 'quantity': 0.1, 'price': 50000},
                {'action': 'SELL', 'symbol': 'BTCUSDT', 'quantity': 0.05, 'price': 51000}
            ],
            'market_data': {
                'prices': {'BTCUSDT': 50500}
            }
        }
        
        # Process data
        result = engine.process(sample_data)
        print(f"Trading completed: {len(result['trading_results'])} trades executed")
        print(f"Portfolio state: {result['portfolio_state']}")
        print(f"Performance metrics: {result['performance_metrics']}")
        
        # Cleanup
        engine.cleanup()
        print("Paper Trading Engine cleaned up")
    
    # Register in registry
    register_migrated_paper_trading_engine()