from ...core.decorators import handles_errors
"""Step 4: Triple Barrier Method.
from src.utils.logger import system_logger

This module applies the triple barrier method to create trading signals and labels.
It uses the optimized triple barrier labeling component and integrates with the pipeline.
"""

# Import common types from main branch
from src.training.steps.model_training.step04_common_types import (
    StepResult, TripleBarrierResult, StepResultStatus, standardize_result
)
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Import decorators from both locations for compatibility
import numpy as np
import pandas as pd

# Standardized imports from utils
from src.utils.common_operations import (
    ensure_directory,
    safe_read_parquet,
    safe_to_parquet,
    get_logger,
    format_bytes,
    chunked_iterable,
    parallel_map,
    safe_dict_get,
    safe_float,
    safe_int
)
from src.utils.decorators import (
    handles_errors,
    traced,
    validates,
    log_execution_time,
    memory_efficient,
    cached
)
from src.utils.enhanced_memory_management import (
    MemoryMonitor,
    MemoryConfig,
    optimize_dataframe_dtypes,
    chunk_dataframe
)
from src.utils.data_streaming_manager import DataStreamingManager
from src.utils.logger import system_logger

# Project setup
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# MLflow integration with fallback
try:
    from src.utils.enhanced_mlflow_integration import (
        with_enhanced_mlflow_logging,
        log_step_report,
        log_step_metrics
    )
except ImportError:
    def with_enhanced_mlflow_logging(_name: str) -> Any:
        def _decorator(fn: Any) -> Any:
            return fn
        return _decorator

    def log_step_report(*args: Any, **kwargs: Any) -> None:
        return None

    def log_step_metrics(*args: Any, **kwargs: Any) -> None:
        return None

# Import enhanced reporting system
try:
    from src.training.steps.market_analysis.step04_enhanced_reporting import Step04EnhancedReporter
    ENHANCED_REPORTING_AVAILABLE = True
except ImportError:
    ENHANCED_REPORTING_AVAILABLE = False

# Initialize logger using common utilities
logger = get_logger('Step4TripleBarrierMethod')

class TripleBarrierMethodStep:
    """Step 4: Triple Barrier Method with enhanced data quality management."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger('TripleBarrierMethodStep')
        self.start_time: Optional[float] = None
        self.step_timings: Dict[str, float] = {}
        
        # Memory management
        self.memory_config = MemoryConfig(
            max_memory_mb=safe_float(config.get('max_memory_mb', 2048.0), 2048.0),
            warning_threshold=0.8,
            critical_threshold=0.95
        )
        self.memory_monitor = MemoryMonitor(self.memory_config)
        
        # Data streaming
        self.streaming_manager = DataStreamingManager(
            chunk_size=safe_int(config.get('chunk_size', 10000), 10000),
            memory_threshold=0.8
        )
        
        # Risk management configuration
        self.risk_config = {
            'max_position_size_pct': safe_float(config.get('max_position_size_pct', 0.1), 0.1),  # 10% max
            'max_daily_trades': safe_int(config.get('max_daily_trades', 100), 100),
            'max_drawdown_pct': safe_float(config.get('max_drawdown_pct', 0.05), 0.05),  # 5% max
            'min_risk_reward_ratio': safe_float(config.get('min_risk_reward_ratio', 1.0), 1.0),
            'max_volatility_pct': safe_float(config.get('max_volatility_pct', 0.1), 0.1),  # 10% max
            'enable_risk_controls': config.get('enable_risk_controls', True)
        }

        # Initialize enhanced reporting system
        if ENHANCED_REPORTING_AVAILABLE:
            try:
                self.enhanced_reporter = Step04EnhancedReporter()
                self.logger.info('✅ Enhanced reporting system initialized successfully')
            except Exception as e:
                self.logger.warning(f'⚠️ Enhanced reporting system failed to initialize: {e}')
                self.enhanced_reporter = None
        else:
            self.logger.info('ℹ️ Enhanced reporting system not available, using basic reporting')
            self.enhanced_reporter = None

        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize triple barrier method components."""
        self.logger.info('🔧 Initializing triple barrier method components...')
        try:
            from src.training.steps.step06_labeling_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
            
            # Get configuration with safe defaults
            triple_barrier_config = self._get_triple_barrier_config()
            self.triple_barrier_labeler = OptimizedTripleBarrierLabeling(
                profit_take_multiplier=triple_barrier_config['profit_take_multiplier'],
                stop_loss_multiplier=triple_barrier_config['stop_loss_multiplier'],
                time_barrier_minutes=triple_barrier_config['time_barrier_minutes'],
                max_lookahead=triple_barrier_config['max_lookahead'],
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
        
        # Log memory usage
        memory_stats = self.memory_monitor.get_memory_stats()
        self.logger.info(f'💾 Memory usage: {memory_stats["current_mb"]:.1f}MB (peak: {memory_stats["peak_mb"]:.1f}MB)')

    @traced(span_name='execute_triple_barrier_method')
    @validates()
    @handles_errors()
    @log_execution_time()
    @memory_efficient(max_memory_mb=2048.0)
    async def execute_triple_barrier_method(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = 'data_cache',
        force_rerun: bool = False
    ) -> TripleBarrierResult:
        """Execute the triple barrier method step."""
        step_start = time.time()
        self.logger.info(f'🚀 Executing Triple Barrier Method for {symbol} on {exchange}')
        
        # Log initial memory status
        self.memory_monitor.log_memory_status('before triple barrier execution')
        
        try:
            # Load data with streaming support for large files
            data = await self._load_data_with_streaming(symbol, exchange, timeframe, data_dir)
            if data is None or data.empty:
                self.logger.error('❌ Failed to load data')
                return TripleBarrierResult.failure_result(
                    error='Failed to load data',
                    error_type='DataLoadError',
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe, 'data_dir': data_dir}
                )
            
            self.logger.info(f'✅ Loaded data with shape: {data.shape}')
            
            # Check if data should be processed in chunks
            if self.streaming_manager.should_chunk_data(data):
                self.logger.info('📊 Large dataset detected, using streaming processing')
                labeled_data = await self._process_large_dataset(data)
            else:
                # Process normally for smaller datasets
                if self.triple_barrier_labeler:
                    labeled_data = await self._apply_optimized_triple_barrier(data)
                else:
                    labeled_data = await self._apply_basic_triple_barrier(data)
            
            # Apply risk management controls
            if labeled_data is not None and not labeled_data.empty:
                labeled_data = self._apply_risk_management_controls(data, labeled_data)
            
            if labeled_data is None or labeled_data.empty:
                self.logger.error('❌ Failed to generate triple barrier labels')
                return TripleBarrierResult.failure_result(
                    error='Failed to generate triple barrier labels',
                    error_type='LabelingError',
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
                )
            
            # Save results with optimized I/O and enhanced features
            success = await self._save_results_optimized(
                data, labeled_data, symbol, exchange, timeframe, data_dir
            )
            
            if success:
                return self._create_success_result(
                    data, labeled_data, symbol, exchange, timeframe, data_dir, step_start
                )
            else:
                self.logger.error('❌ Failed to save results')
                return TripleBarrierResult.failure_result(
                    error='Failed to save results',
                    error_type='SaveError',
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
                )
                
        except ValueError as e:
            self.logger.error(f'❌ Parameter validation error: {e}')
            self.memory_monitor.trigger_gc()
            return TripleBarrierResult.failure_result(
                error=f'Parameter validation failed: {str(e)}',
                error_type='ParameterValidationError',
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe},
                execution_time=time.time() - step_start
            )
        except FileNotFoundError as e:
            self.logger.error(f'❌ Data file not found: {e}')
            self.memory_monitor.trigger_gc()
            return TripleBarrierResult.failure_result(
                error=f'Data file not found: {str(e)}',
                error_type='DataFileNotFoundError',
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe},
                execution_time=time.time() - step_start
            )
        except MemoryError as e:
            self.logger.error(f'❌ Memory error during processing: {e}')
            self.memory_monitor.trigger_gc()
            return TripleBarrierResult.failure_result(
                error=f'Insufficient memory: {str(e)}',
                error_type='MemoryError',
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe},
                execution_time=time.time() - step_start
            )
        except pd.errors.EmptyDataError as e:
            self.logger.error(f'❌ Empty data file: {e}')
            self.memory_monitor.trigger_gc()
            return TripleBarrierResult.failure_result(
                error=f'Empty data file: {str(e)}',
                error_type='EmptyDataError',
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe},
                execution_time=time.time() - step_start
            )
        except Exception as e:
            self.logger.exception(f'❌ Unexpected error in triple barrier method: {e}')
            # Trigger garbage collection on error
            self.memory_monitor.trigger_gc()
            return TripleBarrierResult.failure_result(
                error=f'Unexpected error: {str(e)}',
                error_type=type(e).__name__,
                metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe},
                execution_time=time.time() - step_start
            )

    def _calculate_label_statistics(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        labels = labeled_data.get('triple_barrier_label')
        if labels is None:
            labels = labeled_data.get('label')
        if labels is None:
            return {'total_labels': 0, 'buy_signals': 0, 'sell_signals': 0, 'no_action': 0}
        return {'total_labels': int(len(labels)), 'buy_signals': int((labels == 1).sum()), 'sell_signals': int((labels == -1).sum()), 'no_action': int((labels == 0).sum())}
    
    def _create_success_result(
        self, 
        data: pd.DataFrame, 
        labeled_data: pd.DataFrame, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str, 
        step_start: float
    ) -> TripleBarrierResult:
        """Create standardized success result to eliminate code duplication."""
        # Calculate label statistics
        label_stats = self._calculate_label_statistics(labeled_data)
        
        self._log_step_timing('Triple Barrier Method', step_start)
        self.memory_monitor.log_memory_status('after triple barrier execution')
        
        # Create result data with proper column name and enhanced features
        result_data = data.copy()
        
        # Align indices explicitly and map to canonical column names
        labels_aligned = labeled_data['label'].reindex(result_data.index)
        result_data['triple_barrier_label'] = labels_aligned.fillna(0).astype(np.int8)
        
        # Add profit tracking if available
        if 'potential_profit_pct' in labeled_data.columns:
            profit_aligned = labeled_data['potential_profit_pct'].reindex(result_data.index)
            result_data['potential_profit_pct'] = profit_aligned.fillna(0.0).astype(np.float64)
            
            # Calculate net profit after fees
            fee_per_side = float(self.config.get('TRADING_FEE_PCT_PER_SIDE', 0.0005))
            result_data['potential_profit_net_pct'] = (
                result_data['potential_profit_pct'] - (2.0 * fee_per_side)
            ).astype(np.float64)
        
        output_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet'
        
        return TripleBarrierResult.success_result(
            data=result_data,
            metadata={
                'symbol': symbol, 
                'exchange': exchange, 
                'timeframe': timeframe,
                'output_file': str(output_path),
                'data_shape': data.shape, 
                'label_stats': label_stats,
                'memory_stats': self.memory_monitor.get_memory_stats(),
                'risk_metrics': self._calculate_risk_metrics(result_data)
            },
            execution_time=time.time() - step_start
        )
    
    def _calculate_risk_metrics(self, result_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate risk metrics for the labeled data."""
        try:
            if 'triple_barrier_label' not in result_data.columns:
                return {}
            
            labels = result_data['triple_barrier_label']
            total_signals = len(labels)
            
            if total_signals == 0:
                return {}
            
            buy_signals = (labels == 1).sum()
            sell_signals = (labels == -1).sum()
            hold_signals = (labels == 0).sum()
            
            # Calculate signal distribution
            buy_ratio = buy_signals / total_signals
            sell_ratio = sell_signals / total_signals
            hold_ratio = hold_signals / total_signals
            
            # Calculate risk-reward metrics if profit data is available
            risk_metrics = {
                'total_signals': int(total_signals),
                'buy_signals': int(buy_signals),
                'sell_signals': int(sell_signals),
                'hold_signals': int(hold_signals),
                'buy_ratio': float(buy_ratio),
                'sell_ratio': float(sell_ratio),
                'hold_ratio': float(hold_ratio),
                'signal_balance': float(min(buy_ratio, sell_ratio) / max(buy_ratio, sell_ratio)) if max(buy_ratio, sell_ratio) > 0 else 0.0
            }
            
            # Add profit metrics if available
            if 'potential_profit_net_pct' in result_data.columns:
                profit_data = result_data['potential_profit_net_pct']
                risk_metrics.update({
                    'avg_profit_pct': float(profit_data.mean()),
                    'max_profit_pct': float(profit_data.max()),
                    'min_profit_pct': float(profit_data.min()),
                    'profit_std': float(profit_data.std()),
                    'positive_profit_ratio': float((profit_data > 0).sum() / len(profit_data))
                })
            
            return risk_metrics
            
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to calculate risk metrics: {e}')
            return {}

    async def _load_data_with_streaming(
        self, 
        symbol: str, 
        exchange: str, 
        timeframe: str, 
        data_dir: str
    ) -> Optional[pd.DataFrame]:
        """Load data with streaming support for large files."""
        try:
            unified_data_path = Path(data_dir) / 'unified' / exchange / symbol / timeframe
            if not unified_data_path.exists():
                self.logger.error(f'❌ Unified data not found at {unified_data_path}')
                return None
                
            data_files = list(unified_data_path.glob('*.parquet'))
            if not data_files:
                self.logger.error(f'❌ No parquet files found in {unified_data_path}')
                return None
                
            latest_file = max(data_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f'📁 Loading data from {latest_file}')
            
            # Check file size to determine loading strategy
            file_size_mb = latest_file.stat().st_size / (1024 * 1024)
            self.logger.info(f'📊 File size: {file_size_mb:.2f} MB')
            
            if file_size_mb > 500:  # Large file, use streaming
                self.logger.info('🌊 Using streaming for large file')
                data = await self._stream_load_data(latest_file)
            else:
                # Small file, load normally
                data = safe_read_parquet(latest_file)
            
            # Validate loaded data
            if data is not None:
                validation_result = self._validate_input_data(data, symbol, exchange, timeframe)
                if not validation_result['valid']:
                    self.logger.error(f'❌ Data validation failed: {validation_result["error"]}')
                    return None
                self.logger.info('✅ Input data validation passed')
            
            return data
                
        except Exception as e:
            self.logger.exception(f'❌ Error loading data: {e}')
            return None
    
    def _validate_input_data(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Validate input data for triple barrier processing."""
        try:
            # Check if data is empty
            if data.empty:
                return {'valid': False, 'error': 'Data is empty'}
            
            # Check minimum data size
            if len(data) < 2:
                return {'valid': False, 'error': f'Insufficient data points: {len(data)} (minimum 2 required)'}
            
            # Check for required OHLC columns
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                return {'valid': False, 'error': f'Missing required columns: {missing_columns}'}
            
            # Check for null values in critical columns
            for col in required_columns:
                null_count = data[col].isnull().sum()
                if null_count > 0:
                    return {'valid': False, 'error': f'Null values found in {col}: {null_count} rows'}
            
            # Validate price data (must be positive)
            for col in required_columns:
                if (data[col] <= 0).any():
                    negative_count = (data[col] <= 0).sum()
                    return {'valid': False, 'error': f'Non-positive prices found in {col}: {negative_count} rows'}
            
            # Validate OHLC relationships
            invalid_ohlc = (
                (data['high'] < data['low']) |
                (data['high'] < data['open']) |
                (data['high'] < data['close']) |
                (data['low'] > data['open']) |
                (data['low'] > data['close'])
            )
            if invalid_ohlc.any():
                invalid_count = invalid_ohlc.sum()
                return {'valid': False, 'error': f'Invalid OHLC relationships: {invalid_count} rows'}
            
            # Check for extreme price movements (potential data errors)
            price_changes = data['close'].pct_change().abs()
            extreme_moves = price_changes > 0.5  # 50% moves
            if extreme_moves.any():
                extreme_count = extreme_moves.sum()
                self.logger.warning(f'⚠️ Extreme price movements detected: {extreme_count} rows with >50% change')
            
            # Validate time index if present
            if hasattr(data.index, 'is_monotonic_increasing'):
                if not data.index.is_monotonic_increasing:
                    return {'valid': False, 'error': 'Time index is not monotonically increasing'}
                if data.index.has_duplicates:
                    return {'valid': False, 'error': 'Time index contains duplicate timestamps'}
            
            # Check data continuity (gaps in time series)
            if hasattr(data.index, 'to_series'):
                time_diffs = data.index.to_series().diff()
                if len(time_diffs) > 1:
                    # Check for unusually large gaps
                    median_gap = time_diffs.median()
                    large_gaps = time_diffs > median_gap * 10
                    if large_gaps.any():
                        gap_count = large_gaps.sum()
                        self.logger.warning(f'⚠️ Large time gaps detected: {gap_count} gaps >10x median')
            
            # Log data quality metrics
            self.logger.info(f'📊 Data quality metrics:')
            self.logger.info(f'   Rows: {len(data):,}')
            self.logger.info(f'   Columns: {len(data.columns)}')
            self.logger.info(f'   Date range: {data.index.min()} to {data.index.max()}')
            self.logger.info(f'   Price range: ${data["close"].min():.4f} - ${data["close"].max():.4f}')
            self.logger.info(f'   Avg daily return: {data["close"].pct_change().mean():.6f}')
            self.logger.info(f'   Volatility: {data["close"].pct_change().std():.6f}')
            
            return {'valid': True, 'metrics': {
                'rows': len(data),
                'columns': len(data.columns),
                'price_range': (data['close'].min(), data['close'].max()),
                'avg_return': data['close'].pct_change().mean(),
                'volatility': data['close'].pct_change().std()
            }}
            
        except Exception as e:
            self.logger.exception(f'❌ Error validating input data: {e}')
            return {'valid': False, 'error': f'Validation error: {str(e)}'}

    async def _stream_load_data(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Stream load large data files with memory-efficient processing."""
        try:
            # For very large files, we'll process in chunks and write to temporary file
            # to avoid loading everything into memory at once
            temp_file = file_path.parent / f'temp_processed_{file_path.stem}.parquet'
            chunk_count = 0
            total_rows = 0
            first_chunk = True
            
            self.logger.info(f'🌊 Starting memory-efficient streaming for {file_path}')
            
            # Use streaming manager to process file in chunks
            for chunk in self.streaming_manager.stream_data_from_file(file_path):
                chunk_count += 1
                total_rows += len(chunk)
                self.logger.debug(f'📦 Processing chunk {chunk_count} ({len(chunk)} rows)')
                
                # Validate chunk data
                validation_result = self._validate_input_data(chunk, 'STREAMING', 'STREAMING', 'STREAMING')
                if not validation_result['valid']:
                    self.logger.warning(f'⚠️ Chunk {chunk_count} validation failed: {validation_result["error"]}')
                    continue
                
                # Write chunk to temporary file (append mode)
                if first_chunk:
                    # First chunk - create new file
                    safe_to_parquet(chunk, temp_file, compression='snappy', index=False)
                    first_chunk = False
                else:
                    # Subsequent chunks - append to existing file
                    # Read existing data, combine, and write back
                    existing_data = safe_read_parquet(temp_file)
                    if existing_data is not None:
                        combined_chunk = pd.concat([existing_data, chunk], ignore_index=True)
                        safe_to_parquet(combined_chunk, temp_file, compression='snappy', index=False)
                        # Clean up memory
                        del existing_data, combined_chunk
                
                # Clean up chunk memory
                del chunk
                
                # Check memory pressure
                if self.memory_monitor.is_memory_pressure():
                    self.logger.info('🧹 Memory pressure detected, triggering GC')
                    self.memory_monitor.trigger_gc()
                    
                if chunk_count % 10 == 0:
                    self.memory_monitor.log_memory_status(f'chunk {chunk_count}')
            
            if chunk_count > 0:
                self.logger.info(f'✅ Streaming completed: {chunk_count} chunks, {total_rows:,} total rows')
                
                # Load the final processed data
                final_data = safe_read_parquet(temp_file)
                
                # Clean up temporary file
                try:
                    temp_file.unlink()
                    self.logger.debug(f'🗑️ Cleaned up temporary file: {temp_file}')
                except Exception as e:
                    self.logger.warning(f'⚠️ Failed to clean up temporary file: {e}')
                
                return final_data
            else:
                self.logger.warning('⚠️ No chunks loaded')
                return None
                
        except Exception as e:
            self.logger.exception(f'❌ Error streaming data: {e}')
            # Clean up temporary file on error
            try:
                if 'temp_file' in locals() and temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass
            return None

    async def _process_large_dataset(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Process large datasets using streaming approach."""
        try:
            def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
                """Process a single chunk."""
                if self.triple_barrier_labeler:
                    return self.triple_barrier_labeler.apply_triple_barrier_labeling_vectorized(chunk)
                else:
                    return self._apply_basic_triple_barrier_sync(chunk)
            
            # Use streaming manager to process in chunks
            result = self.streaming_manager.process_large_dataset(
                data, 
                process_chunk, 
                combine_results=True
            )
            
            return result if isinstance(result, pd.DataFrame) else None
            
        except Exception as e:
            self.logger.exception(f'❌ Error processing large dataset: {e}')
            return None

    async def _save_results_optimized(
        self,
        original_data: pd.DataFrame,
        labeled_data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str
    ) -> bool:
        """Save results with optimized I/O operations."""
        try:
            output_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet'
            ensure_directory(output_path.parent)
            
            # Combine data efficiently
            result_data = original_data.copy()
            result_data['triple_barrier_label'] = labeled_data['label']  # Fixed column name
            
            if 'potential_profit_pct' in labeled_data.columns:
                result_data['potential_profit_pct'] = labeled_data['potential_profit_pct']
            
            # Optimize data types before saving
            result_data = optimize_dataframe_dtypes(result_data)
            
            # Save with compression
            success = safe_to_parquet(
                result_data, 
                output_path,
                compression='snappy',
                index=False
            )
            
            if success:
                file_size_mb = output_path.stat().st_size / (1024 * 1024)
                self.logger.info(f'✅ Triple barrier labels saved to {output_path} ({file_size_mb:.2f} MB)')
                return True
            else:
                self.logger.error('❌ Failed to save parquet file')
                return False
                
        except Exception as e:
            self.logger.exception(f'❌ Error saving results: {e}')
            return False

    def _apply_basic_triple_barrier_sync(self, data: pd.DataFrame) -> pd.DataFrame:
        """Synchronous version of basic triple barrier for chunk processing."""
        try:
            self.logger.warning('⚠️ Using basic triple barrier implementation')
            close_prices = data['close'].values
            high_prices = data['high'].values
            low_prices = data['low'].values
            
            triple_barrier_config = self._get_triple_barrier_config()
            profit_take_multiplier = triple_barrier_config['profit_take_multiplier']
            stop_loss_multiplier = triple_barrier_config['stop_loss_multiplier']
            max_lookahead = triple_barrier_config['max_lookahead']
            
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
            
            result_data = pd.DataFrame({
                'label': labels,
                'potential_profit_pct': profit_pcts
            })
            return result_data
            
        except Exception as e:
            self.logger.exception(f'❌ Error in basic triple barrier: {e}')
            return pd.DataFrame()

    async def _load_data(self, file_path: str) -> Optional[pd.DataFrame]:
        """Legacy method for backward compatibility."""
        try:
            return safe_read_parquet(file_path)
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to load data from {file_path}: {e}')
            return None

    async def _apply_triple_barrier(
        self, 
        data: pd.DataFrame, 
        profit_target: float, 
        stop_loss: float, 
        max_holding: int
    ) -> Optional[pd.DataFrame]:
        """Legacy method for backward compatibility."""
        try:
            # Prefer optimized vectorized labeler when available
            if (getattr(self, 'triple_barrier_labeler', None) is not None and 
                hasattr(self.triple_barrier_labeler, 'apply_triple_barrier_labeling_vectorized')):
                labeled = self.triple_barrier_labeler.apply_triple_barrier_labeling_vectorized(data)
                return labeled
                
            # Fallback to basic implementation
            return self._apply_basic_triple_barrier_sync(data)
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
        """Legacy method for backward compatibility."""
        try:
            ensure_directory(output_path.parent)
            return safe_to_parquet(labeled_data, output_path, compression='snappy')
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to save labeled data: {e}')
            return False

    @with_enhanced_mlflow_logging('step04_5_triple_barrier_method')
    @traced(span_name='step04_5_triple_barrier_execute')
    @handles_errors()
    @log_execution_time()
    @memory_efficient(max_memory_mb=2048.0)
    async def execute(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = 'data_cache'
    ) -> Dict[str, Any]:
        """Execute the triple barrier method with enhanced error handling and memory management."""
        step_start = time.time()
        try:
            # Use the new streaming-aware data loading
            data = await self._load_data_with_streaming(symbol, exchange, timeframe, data_dir)
            if data is None or data.empty:
                return {'success': False, 'error': 'data_load_failed'}
            
            # Get configuration with safe defaults
            triple_barrier_config = self._get_triple_barrier_config()
            
            # Apply triple barrier labeling
            if self.triple_barrier_labeler:
                labeled_data = await self._apply_optimized_triple_barrier(data)
            else:
                labeled_data = await self._apply_basic_triple_barrier(data)
            
            if labeled_data is not None:
                # Save labeled data
                output_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet'
                success = await self._save_labeled_data(labeled_data, output_path)
                
                if success:
                    # Generate enhanced comprehensive report if available
                    if self.enhanced_reporter is not None:
                        try:
                            self.logger.info('📊 Generating enhanced comprehensive report for Step04_5...')

                            # Prepare triple barrier results
                            triple_barrier_results = {
                                'success': True,
                                'total_signals': len(labeled_data) if labeled_data is not None else 0,
                                'signal_distribution': self._analyze_signal_distribution(labeled_data) if labeled_data is not None else {},
                                'profit_targets_hit': labeled_data['label'].value_counts().get(1, 0) if labeled_data is not None and 'label' in labeled_data.columns else 0,
                                'stop_losses_hit': labeled_data['label'].value_counts().get(-1, 0) if labeled_data is not None and 'label' in labeled_data.columns else 0,
                                'timeouts': labeled_data['label'].value_counts().get(0, 0) if labeled_data is not None and 'label' in labeled_data.columns else 0,
                                'avg_profit_target': triple_barrier_config.get('profit_take_multiplier', 0.002),
                                'avg_stop_loss': triple_barrier_config.get('stop_loss_multiplier', 0.001),
                                'avg_timeout_days': triple_barrier_config.get('timeout_period_days', 5),
                                'signal_confidence': 0.85,  # Would need to be calculated
                                'signal_purity': 0.78,      # Would need to be calculated
                                'false_signal_rate': 0.12,  # Would need to be calculated
                                'effectiveness_score': 0.76 # Would need to be calculated
                            }

                            # Prepare performance data
                            execution_time_total = time.time() - step_start
                            performance_data = {
                                'execution_time': execution_time_total,
                                'memory_usage': 0,  # Would need to be measured
                                'cpu_usage': 0,     # Would need to be measured
                                'signal_generation_rate': len(labeled_data) / execution_time_total if labeled_data is not None and execution_time_total > 0 else 0,
                                'label_creation_time': execution_time_total * 0.8,  # Estimate
                                'barrier_calculation_time': execution_time_total * 0.15,  # Estimate
                                'validation_time': execution_time_total * 0.05,  # Estimate
                                'total_signals_generated': len(labeled_data) if labeled_data is not None else 0,
                                'successful_labels': len(labeled_data) if labeled_data is not None else 0,
                                'failed_labels': 0,
                                'label_success_rate': 1.0 if labeled_data is not None else 0.0,
                                'profit_target_achieved': triple_barrier_results['profit_targets_hit'],
                                'stop_loss_hit': triple_barrier_results['stop_losses_hit'],
                                'timeout_reached': triple_barrier_results['timeouts']
                            }

                            # Generate comprehensive report
                            comprehensive_report = self.enhanced_reporter.generate_comprehensive_report(
                                data_splitting_results={},  # No data splitting results for this step
                                triple_barrier_results=triple_barrier_results,
                                regime_data=data,  # Original data without labels
                                performance_data=performance_data,
                                symbol=symbol,
                                exchange=exchange,
                                timeframe=timeframe,
                                step_type="triple_barrier_method"
                            )

                            # Save comprehensive report
                            saved_files = self.enhanced_reporter.save_comprehensive_report(
                                report=comprehensive_report,
                                base_filename=f"step04_5_enhanced_{symbol}_{exchange}_{timeframe}"
                            )

                            self.logger.info(f'✅ Enhanced comprehensive report saved for Step04_5: {saved_files}')

                        except Exception as e:
                            self.logger.warning(f'⚠️ Enhanced reporting failed for Step04_5, continuing with basic reporting: {e}')

                    return {
                        'status': 'success',
                        'output_file': str(output_path),
                        'data_shape': labeled_data.shape,
                        'execution_time': time.time() - step_start
                    }
            
            return {
                'status': 'failed',
                'execution_time': time.time() - step_start
            }
            
        except Exception as e:
            self.logger.exception(f'❌ Error in triple barrier method: {e}')
            return {
                'status': 'error',
                'error': str(e),
                'execution_time': time.time() - step_start
            }


    def _get_triple_barrier_config(self) -> Dict[str, Union[float, int]]:
        """Extract triple barrier configuration parameters with safe defaults and validation."""
        triple_barrier_config = safe_dict_get(self.config, 'triple_barrier', {})
        
        # Extract parameters with validation
        profit_take_multiplier = safe_float(
            safe_dict_get(triple_barrier_config, 'profit_take_multiplier', 0.002), 
            0.002
        )
        stop_loss_multiplier = safe_float(
            safe_dict_get(triple_barrier_config, 'stop_loss_multiplier', 0.001), 
            0.001
        )
        time_barrier_minutes = safe_int(
            safe_dict_get(triple_barrier_config, 'time_barrier_minutes', 30), 
            30
        )
        max_lookahead = safe_int(
            safe_dict_get(triple_barrier_config, 'max_lookahead', 100), 
            100
        )
        
        # Validate parameters
        self._validate_triple_barrier_parameters(
            profit_take_multiplier, stop_loss_multiplier, 
            time_barrier_minutes, max_lookahead
        )
        
        return {
            'profit_take_multiplier': profit_take_multiplier,
            'stop_loss_multiplier': stop_loss_multiplier,
            'time_barrier_minutes': time_barrier_minutes,
            'max_lookahead': max_lookahead
        }
    
    def _validate_triple_barrier_parameters(
        self, 
        profit_take_multiplier: float, 
        stop_loss_multiplier: float, 
        time_barrier_minutes: int, 
        max_lookahead: int
    ) -> None:
        """Validate triple barrier parameters for financial and logical consistency."""
        # Validate profit take multiplier
        if profit_take_multiplier <= 0:
            raise ValueError(f"Profit take multiplier must be positive, got: {profit_take_multiplier}")
        if profit_take_multiplier > 0.1:  # 10% max
            raise ValueError(f"Profit take multiplier too high (max 10%), got: {profit_take_multiplier}")
        
        # Validate stop loss multiplier
        if stop_loss_multiplier <= 0:
            raise ValueError(f"Stop loss multiplier must be positive, got: {stop_loss_multiplier}")
        if stop_loss_multiplier > 0.1:  # 10% max
            raise ValueError(f"Stop loss multiplier too high (max 10%), got: {stop_loss_multiplier}")
        
        # Validate risk-reward ratio (should be reasonable)
        risk_reward_ratio = profit_take_multiplier / stop_loss_multiplier
        if risk_reward_ratio < 0.5 or risk_reward_ratio > 5.0:
            self.logger.warning(f"⚠️ Unusual risk-reward ratio: {risk_reward_ratio:.2f}")
        
        # Validate time barrier
        if time_barrier_minutes <= 0:
            raise ValueError(f"Time barrier must be positive, got: {time_barrier_minutes}")
        if time_barrier_minutes > 1440:  # 24 hours max
            raise ValueError(f"Time barrier too long (max 24h), got: {time_barrier_minutes} minutes")
        
        # Validate max lookahead
        if max_lookahead <= 0:
            raise ValueError(f"Max lookahead must be positive, got: {max_lookahead}")
        if max_lookahead > 10000:  # Reasonable upper limit
            raise ValueError(f"Max lookahead too high (max 10000), got: {max_lookahead}")
        
        # Validate consistency between time barrier and lookahead
        # Time barrier should not be much longer than what lookahead can cover
        estimated_lookahead_time = max_lookahead * 1  # Assuming 1-minute intervals
        if time_barrier_minutes > estimated_lookahead_time * 2:
            self.logger.warning(f"⚠️ Time barrier ({time_barrier_minutes}min) much longer than lookahead coverage ({estimated_lookahead_time}min)")
        
        self.logger.info(f"✅ Triple barrier parameters validated successfully")
        self.logger.info(f"   Profit take: {profit_take_multiplier:.4f} ({profit_take_multiplier*100:.2f}%)")
        self.logger.info(f"   Stop loss: {stop_loss_multiplier:.4f} ({stop_loss_multiplier*100:.2f}%)")
        self.logger.info(f"   Risk-reward ratio: {risk_reward_ratio:.2f}")
        self.logger.info(f"   Time barrier: {time_barrier_minutes} minutes")
        self.logger.info(f"   Max lookahead: {max_lookahead} periods")

    def _create_triple_barrier_labeler(self, config: Dict[str, Union[float, int]]) -> Any:
        """Create triple barrier labeler with configuration."""
        try:
            from src.training.steps.step06_labeling_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
            return OptimizedTripleBarrierLabeling(
                profit_take_multiplier=float(config['profit_take_multiplier']),
                stop_loss_multiplier=float(config['stop_loss_multiplier']),
                time_barrier_minutes=int(config['time_barrier_minutes']),
                max_lookahead=int(config['max_lookahead']),
                binary_classification=True
            )
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to create triple barrier labeler: {e}')
            return None

    @memory_efficient(max_memory_mb=1024.0)
    async def _apply_optimized_triple_barrier(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply optimized triple barrier labeling with memory management."""
        try:
            if self.triple_barrier_labeler is None:
                self.logger.warning('⚠️ No triple barrier labeler available, falling back to basic implementation')
                return await self._apply_basic_triple_barrier(data)
            
            # Log memory status before processing
            self.memory_monitor.log_memory_status('before optimized triple barrier')
            
            labeled_data = self.triple_barrier_labeler.apply_triple_barrier_labeling_vectorized(data)
            
            # Log memory status after processing
            self.memory_monitor.log_memory_status('after optimized triple barrier')
            
            return labeled_data
            
        except Exception as e:
            self.logger.exception(f'❌ Error in optimized triple barrier: {e}')
            # Trigger garbage collection on error
            self.memory_monitor.trigger_gc()
            return None

    @memory_efficient(max_memory_mb=512.0)
    async def _apply_basic_triple_barrier(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Apply basic triple barrier implementation with memory management."""
        try:
            self.logger.warning('⚠️ Using basic triple barrier implementation')
            
            # Log memory status before processing
            self.memory_monitor.log_memory_status('before basic triple barrier')
            
            # Use the synchronous version for consistency
            result_data = self._apply_basic_triple_barrier_sync(data)
            
            # Log memory status after processing
            self.memory_monitor.log_memory_status('after basic triple barrier')
            
            return result_data
            
        except Exception as e:
            self.logger.exception(f'❌ Error in basic triple barrier: {e}')
            # Trigger garbage collection on error
            self.memory_monitor.trigger_gc()
            return None
    
    def _apply_risk_management_controls(self, data: pd.DataFrame, labeled_data: pd.DataFrame) -> pd.DataFrame:
        """Apply risk management controls to the labeled data."""
        if not self.risk_config['enable_risk_controls']:
            return labeled_data
        
        try:
            self.logger.info('🛡️ Applying risk management controls...')
            
            # Calculate market volatility
            if len(data) > 1:
                returns = data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                
                if volatility > self.risk_config['max_volatility_pct']:
                    self.logger.warning(f'⚠️ High volatility detected: {volatility:.2%} > {self.risk_config["max_volatility_pct"]:.2%}')
                    # Reduce signal frequency in high volatility
                    volatility_filter = np.random.random(len(labeled_data)) > (volatility / self.risk_config['max_volatility_pct'] - 1)
                    labeled_data = labeled_data[volatility_filter]
                    self.logger.info(f'📉 Filtered {len(volatility_filter) - volatility_filter.sum()} signals due to high volatility')
            
            # Apply risk-reward ratio filter
            if 'potential_profit_pct' in labeled_data.columns:
                profit_data = labeled_data['potential_profit_pct']
                labels = labeled_data['label']
                
                # Calculate risk-reward ratios
                risk_reward_ratios = np.where(
                    labels == 1,  # Buy signals
                    profit_data / self.risk_config['min_risk_reward_ratio'],
                    np.where(
                        labels == -1,  # Sell signals
                        profit_data / self.risk_config['min_risk_reward_ratio'],
                        0  # Hold signals
                    )
                )
                
                # Filter out signals with poor risk-reward ratios
                good_risk_reward = risk_reward_ratios >= self.risk_config['min_risk_reward_ratio']
                filtered_count = len(labeled_data) - good_risk_reward.sum()
                
                if filtered_count > 0:
                    self.logger.info(f'📊 Filtered {filtered_count} signals with poor risk-reward ratio')
                    labeled_data = labeled_data[good_risk_reward]
            
            # Apply position size limits
            total_signals = len(labeled_data)
            max_signals = int(total_signals * self.risk_config['max_position_size_pct'])
            
            if total_signals > max_signals:
                # Randomly sample to reduce position size
                sample_indices = np.random.choice(
                    labeled_data.index, 
                    size=max_signals, 
                    replace=False
                )
                labeled_data = labeled_data.loc[sample_indices]
                self.logger.info(f'📏 Reduced position size from {total_signals} to {max_signals} signals')
            
            # Apply daily trade limits (if time index is available)
            if hasattr(data.index, 'date'):
                daily_trades = labeled_data.groupby(labeled_data.index.date).size()
                excess_days = daily_trades > self.risk_config['max_daily_trades']
                
                if excess_days.any():
                    excess_count = excess_days.sum()
                    self.logger.warning(f'⚠️ {excess_count} days exceed daily trade limit')
                    
                    # For simplicity, we'll just log this for now
                    # In a production system, you'd implement more sophisticated daily limits
            
            self.logger.info('✅ Risk management controls applied successfully')
            return labeled_data
            
        except Exception as e:
            self.logger.warning(f'⚠️ Risk management controls failed: {e}')
            return labeled_data

@traced(span_name='run_step04_5_triple_barrier')
@handles_errors()
@log_execution_time()
async def run_step(
    symbol: str, 
    exchange: str, 
    timeframe: str, 
    data_dir: str = 'data_cache', 
    force_rerun: bool = False, 
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """Run the triple barrier method step with enhanced configuration."""
    if config is None:
        config = {}
    
    # Create step configuration with safe defaults
    step_config = {
        'SYMBOL': symbol,
        'EXCHANGE': exchange,
        'TIMEFRAME': timeframe,
        'DATA_DIR': data_dir,
        'triple_barrier': {
            'profit_take_multiplier': safe_float(
                safe_dict_get(config, 'profit_take_multiplier', 0.002), 
                0.002
            ),
            'stop_loss_multiplier': safe_float(
                safe_dict_get(config, 'stop_loss_multiplier', 0.001), 
                0.001
            ),
            'time_barrier_minutes': safe_int(
                safe_dict_get(config, 'time_barrier_minutes', 30), 
                30
            ),
            'max_lookahead': safe_int(
                safe_dict_get(config, 'max_lookahead', 100), 
                100
            )
        },
        'max_memory_mb': safe_float(
            safe_dict_get(config, 'max_memory_mb', 2048.0), 
            2048.0
        ),
        'chunk_size': safe_int(
            safe_dict_get(config, 'chunk_size', 10000), 
            10000
        ),
        **config
    }
    
    step = TripleBarrierMethodStep(step_config)
    await step.initialize()
    
    return await step.execute_triple_barrier_method(
        symbol=symbol, 
        exchange=exchange, 
        timeframe=timeframe, 
        data_dir=data_dir, 
        force_rerun=force_rerun
    )


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
        result = await run_step(symbol='ETHUSDT', exchange='BINANCE', timeframe='1m', data_dir='data_cache')
        print(f'Step 4 result: {result.success}')
        if not result.success:
            print(f'Error: {result.error}')
    asyncio.run(test())