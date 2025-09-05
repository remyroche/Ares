from typing import Optional
from typing import Dict
from typing import Any
from typing import Dict, List, Optional, Union, Any, Tuple
from src.training.steps.model_training.step04_common_types import (
    StepResult, TripleBarrierResult, StepResultStatus, standardize_result
)
'Step 4: Triple Barrier Method.\n\nThis module applies the triple barrier method to create trading signals and labels.\nIt uses the optimized triple barrier labeling component and integrates with the pipeline.\n'
import asyncio
import sys
from pathlib import Path
from src.utils.common_operations import ensure_directory
import time
from datetime import datetime
from src.core.decorators import handles_errors, traced, validates
import numpy as np
import pandas as pd
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
from src.utils.logger import system_logger
import logging

try:
    from src.utils.enhanced_mlflow_integration import with_enhanced_mlflow_logging, log_step_report, log_step_metrics
except Exception:

    def with_enhanced_mlflow_logging(_name: str) -> None:

        def _decorator(fn: Any) -> None:
            return fn
        return _decorator

    def log_step_report(*args, **kwargs) -> None:
        return None

    def log_step_metrics(*args, **kwargs) -> None:
        return None

def with_tracing_span(_name: str) -> None:

    def _decorator(fn: Any) -> None:
        return fn
    return _decorator

def resource_monitor(fn: Any) -> None:
    return fn
logger = system_logger.getChild('Step4TripleBarrierMethod')

class TripleBarrierMethodStep:
    """Step 4: Triple Barrier Method with enhanced data quality management."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('TripleBarrierMethodStep')
        self.start_time = None
        self.step_timings = {}
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize triple barrier method components."""
        self.logger.info('🔧 Initializing triple barrier method components...')
        try:
            from src.training.steps.step06_labeling_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
            # Initialize with default parameters - will be overridden by optimized parameters
            default_config = self._get_triple_barrier_config()
            self.triple_barrier_labeler = OptimizedTripleBarrierLabeling(
                profit_take_multiplier=default_config['profit_take_multiplier'],
                stop_loss_multiplier=default_config['stop_loss_multiplier'],
                time_barrier_minutes=default_config['time_barrier_minutes'],
                max_lookahead=default_config['max_lookahead'],
                binary_classification=True
            )
            self.logger.info('✅ Optimized triple barrier labeler initialized successfully')
        except Exception as e:
            self.logger.warning(f'⚠️ Could not import OptimizedTripleBarrierLabeling: {e}')
            self.logger.info('📝 Proceeding without optimized triple barrier labeler')
            self.triple_barrier_labeler = None

    async def initialize(self) -> None:
        """Initialize the triple barrier method step."""
        self.start_time = time.time()
        self.logger.info('🚀 Initializing Triple Barrier Method Step...')
        self.logger.info('📋 Step 4 Configuration:')
        self.logger.info(f"   - Symbol: {self.config.get('SYMBOL', 'N/A')}")
        self.logger.info(f"   - Exchange: {self.config.get('EXCHANGE', 'N/A')}")
        self.logger.info(f"   - Timeframe: {self.config.get('TIMEFRAME', 'N/A')}")
        self.logger.info(f"   - Data Directory: {self.config.get('DATA_DIR', 'N/A')}")
        self.logger.info('✅ Triple Barrier Method Step initialized successfully')

    def _log_step_timing(self, step_name: str, start_time: float) -> None:
        """Log timing information for a step."""
        elapsed = time.time() - start_time
        self.step_timings[step_name] = elapsed
        self.logger.info(f'⏱️ {step_name} completed in {elapsed:.2f} seconds')

    @traced(span_name='execute_triple_barrier_method')
    @validates(min_quality_score=0.7, max_correlation=0.95, required_grade='C')
    @handles_errors()
    @validates()
    async def execute_triple_barrier_method(self, symbol: str, exchange: str, timeframe: str, data_dir: str='data_cache', force_rerun: bool=False) -> TripleBarrierResult:
        """Execute the triple barrier method step."""
        step_start = time.time()
        self.logger.info(f'🚀 Executing Triple Barrier Method for {symbol} on {exchange}')
        try:
            unified_data_path = Path(data_dir) / 'unified' / exchange / symbol / timeframe
            if not unified_data_path.exists():
                self.logger.error(f'❌ Unified data not found at {unified_data_path}')
                return TripleBarrierResult.failure_result(
                    error=f'Unified data not found at {unified_data_path}',
                    error_type='DataNotFoundError',
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir}
                )
            
            data_files = list(unified_data_path.glob('*.parquet'))
            if not data_files:
                self.logger.error(f'❌ No parquet files found in {unified_data_path}')
                return TripleBarrierResult.failure_result(
                    error=f'No parquet files found in {unified_data_path}',
                    error_type='DataNotFoundError',
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir}
                )
            
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f'📁 Loading data from {latest_file}')
            data = pd.read_parquet(latest_file)
            self.logger.info(f'✅ Loaded data with shape: {data.shape}')
            
            if self.triple_barrier_labeler:
                labeled_data = await self._apply_optimized_triple_barrier(data)
            else:
                labeled_data = await self._apply_basic_triple_barrier(data)
            
            if labeled_data is None:
                self.logger.error('❌ Failed to generate triple barrier labels')
                return TripleBarrierResult.failure_result(
                    error='Failed to generate triple barrier labels',
                    error_type='LabelingError',
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
                )
            
            output_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet'
            ensure_directory(output_path.parent)
            result_data = data.copy()
            result_data['label'] = labeled_data['label']
            if 'potential_profit_pct' in labeled_data.columns:
                result_data['potential_profit_pct'] = labeled_data['potential_profit_pct']
            result_data.to_parquet(output_path)
            self.logger.info(f'✅ Triple barrier labels saved to {output_path}')
            
            # Calculate label statistics
            label_stats = self._calculate_label_statistics(labeled_data)
            
            self._log_step_timing('Triple Barrier Method', step_start)
            
            return TripleBarrierResult.success_result(
                data=result_data,
                metadata={
                    'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe,
                    'input_file': str(latest_file), 'output_file': str(output_path),
                    'data_shape': data.shape, 'label_stats': label_stats
                },
                execution_time=time.time() - step_start
            )
            
        except Exception as e:
            self.logger.exception(f'❌ Error in triple barrier method: {e}')
            return TripleBarrierResult.failure_result(
                error=str(e),
                error_type=type(e).__name__,
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe},
                execution_time=time.time() - step_start
            )

    @with_enhanced_mlflow_logging('step04_5_triple_barrier_method')
    @with_tracing_span('step04_5_triple_barrier_execute')
    @handles_errors()
    def _calculate_label_statistics(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        labels = labeled_data.get('label')
        if labels is None:
            return {'total_labels': 0, 'buy_signals': 0, 'sell_signals': 0, 'no_action': 0}
        return {'total_labels': int(len(labels)), 'buy_signals': int((labels == 1).sum()), 'sell_signals': int((labels == -1).sum()), 'no_action': int((labels == 0).sum())}

    async def _load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to load data from {file_path}: {e}')
            return None

    async def _apply_triple_barrier(self, data: pd.DataFrame, profit_target: float, stop_loss: float, max_holding: int) -> Optional[pd.DataFrame]:
        try:
            if getattr(self, 'triple_barrier_labeler', None) is not None and hasattr(self.triple_barrier_labeler, 'label_data'):
                labeled = await self.triple_barrier_labeler.label_data(data, profit_target, stop_loss, max_holding)
                return labeled
            close_prices = data['close'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            labels = np.zeros(len(close_prices), dtype=np.int8)
            for i in range(len(close_prices) - 1):
                entry_price = close_prices[i]
                profit_barrier = entry_price * (1 + profit_target)
                stop_barrier = entry_price * (1 - stop_loss)
                for j in range(i + 1, min(i + max_holding, len(close_prices))):
                    if high_prices[j] >= profit_barrier:
                        labels[i] = 1
                        break
                    elif low_prices[j] <= stop_barrier:
                        labels[i] = -1
                        break
            return pd.DataFrame({'label': labels})
        except Exception as e:
            self.logger.exception(f'❌ Error applying triple barrier: {e}')
            return None

    async def _save_labeled_data(self, labeled_data: pd.DataFrame, output_path: Path) -> bool:
        try:
            ensure_directory(output_path.parent)
            labeled_data.to_parquet(output_path)
            return True
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to save labeled data: {e}')
            return False

    @with_enhanced_mlflow_logging('step04_5_triple_barrier_method')
    @with_tracing_span('step04_5_triple_barrier_execute')
    @handles_errors()
    async def execute(self, symbol: str, exchange: str, timeframe: str, data_dir: str='data_cache') -> Dict[str, Any]:
        try:
            unified_data_path = Path(data_dir) / 'unified' / exchange / symbol / timeframe
            latest_file = max(list(unified_data_path.glob('*.parquet')), key=lambda x: x.stat().st_mtime)
            data = await self._load_data(str(latest_file))
            if data is None:
                return {'success': False, 'error': 'data_load_failed'}
            labeled = await self._apply_triple_barrier(data, self.config.get('PROFIT_TARGET', 0.02), self.config.get('STOP_LOSS', 0.01), self.config.get('MAX_HOLDING_PERIOD', 100))
            if labeled is None:
                return {'success': False, 'error': 'labeling_failed'}
            output_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet'
            saved = await self._save_labeled_data(labeled, output_path)
            stats = self._calculate_label_statistics(pd.concat([data.reset_index(drop=True), labeled.reset_index(drop=True)], axis=1))
            log_step_report(config=self.config, step_name='step04_5_triple_barrier_method', report_data={'stats': stats})
            log_step_metrics(config=self.config, step_name='step04_5_triple_barrier_method', metrics={'total_labels': stats.get('total_labels', 0)})
            return {'success': bool(saved), 'labeled_data': labeled, 'label_stats': stats}
        except Exception as e:
            self.logger.exception(f'❌ Execute failed: {e}')
            return {'success': False, 'error': str(e)}

    def _get_triple_barrier_config(self) -> Dict[str, float]:
        """Extract triple barrier configuration parameters."""
        triple_barrier_config = self.config.get('triple_barrier', {})
        return {'profit_take_multiplier': triple_barrier_config.get('profit_take_multiplier', 0.002), 'stop_loss_multiplier': triple_barrier_config.get('stop_loss_multiplier', 0.001), 'time_barrier_minutes': triple_barrier_config.get('time_barrier_minutes', 30), 'max_lookahead': triple_barrier_config.get('max_lookahead', 100)}

    def _create_triple_barrier_labeler(self, config: Dict[str, float]) -> Optional[Any]:
        """Create triple barrier labeler with optimized parameters."""
        try:
            from src.training.steps.step06_labeling_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
            return OptimizedTripleBarrierLabeling(
                profit_take_multiplier=config['profit_take_multiplier'], 
                stop_loss_multiplier=config['stop_loss_multiplier'], 
                time_barrier_minutes=config['time_barrier_minutes'], 
                max_lookahead=config['max_lookahead'], 
                binary_classification=True
            )
        except ImportError as e:
            self.logger.warning(f'⚠️ Could not import OptimizedTripleBarrierLabeling: {e}')
            return None

    async def _apply_optimized_triple_barrier(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            config = self._get_triple_barrier_config()
            labeler = self._create_triple_barrier_labeler(config)
            if labeler is None:
                self.logger.warning('⚠️ Could not create optimized labeler, falling back to basic implementation')
                return await self._apply_basic_triple_barrier(data)
            labeled_data = labeler.apply_triple_barrier_labeling_vectorized(data)
            return labeled_data
        except Exception as e:
            self.logger.exception(f'❌ Error in optimized triple barrier: {e}')
            self.logger.info('📝 Falling back to basic triple barrier implementation')
            return await self._apply_basic_triple_barrier(data)

    async def _apply_basic_triple_barrier(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        try:
            self.logger.warning('⚠️ Using basic triple barrier implementation')
            close_prices = data['close'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            profit_take_multiplier = self.config.get('triple_barrier', {}).get('profit_take_multiplier', 0.002)
            stop_loss_multiplier = self.config.get('triple_barrier', {}).get('stop_loss_multiplier', 0.001)
            max_lookahead = self.config.get('triple_barrier', {}).get('max_lookahead', 100)
            labels = np.zeros(len(close_prices), dtype=np.int8)
            profit_pcts = np.zeros(len(close_prices), dtype=np.float64)
            for i in range(len(close_prices) - 1):
                entry_price = close_prices[i]
                profit_barrier = entry_price * (1 + profit_take_multiplier)
                stop_barrier = entry_price * (1 - stop_loss_multiplier)
                for j in range(i + 1, min(i + max_lookahead, len(close_prices))):
                    if high_prices[j] >= profit_barrier:
                        labels[i] = 1
                        profit_pcts[i] = profit_take_multiplier
                        break
                    elif low_prices[j] <= stop_barrier:
                        labels[i] = -1
                        profit_pcts[i] = -stop_loss_multiplier
                        break
            result_data = pd.DataFrame({'label': labels, 'potential_profit_pct': profit_pcts})
            return result_data
        except Exception as e:
            self.logger.exception(f'❌ Error in basic triple barrier: {e}')
            return None

async def run_step(symbol: str, exchange: str, timeframe: str, data_dir: str='data_cache', force_rerun: bool=False, config: Optional[Dict[str, Any]]=None) -> StepResult:
    """Run Step 4: Triple Barrier Method with standardized return types.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory
        force_rerun: Force rerun flag
        config: Configuration dictionary
        
    Returns:
        StepResult: Standardized result with success status and details
    """
    if config is None:
        config = {}
    
    step_config = {
        'SYMBOL': symbol, 'EXCHANGE': exchange, 'TIMEFRAME': timeframe, 'DATA_DIR': data_dir, 
        'triple_barrier': {
            'profit_take_multiplier': 0.002, 
            'stop_loss_multiplier': 0.001, 
            'time_barrier_minutes': 30, 
            'max_lookahead': 100
        }, 
        **config
    }
    
    step_start = time.time()
    try:
        step = TripleBarrierMethodStep(step_config)
        await step.initialize()
        result = await step.execute_triple_barrier_method(
            symbol=symbol, 
            exchange=exchange, 
            timeframe=timeframe, 
            data_dir=data_dir, 
            force_rerun=force_rerun
        )
        
        # Standardize the result if it's not already a StepResult
        standardized_result = standardize_result(result, "triple_barrier_method")
        
        if standardized_result.success:
            logger.info('✅ Step 4: Triple Barrier Method completed successfully')
        else:
            logger.error('❌ Step 4: Triple Barrier Method failed')
            logger.error(f'🔍 Error: {standardized_result.error}')
        
        return standardized_result
        
    except Exception as e:
        logger.exception(f'❌ Error in triple barrier method step: {e}')
        return StepResult.failure_result(
            error=str(e),
            error_type=type(e).__name__,
            metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir},
            execution_time=time.time() - step_start
        )
if __name__ == '__main__':

    async def test() -> None:
        success = await run_step(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir='data_cache')
        print(f'Step 4 result: {success}')
    asyncio.run(test())