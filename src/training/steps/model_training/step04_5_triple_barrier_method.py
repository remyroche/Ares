"""Step 4: Triple Barrier Method.

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
    safe_int,
    safe_json_dump,
    safe_json_load,
    optimize_dataframe_dtypes,
    validate_dataframe_schema,
    validate_data_quality
)

# Import ml_common utilities for enhanced functionality
from src.utils.ml_common.data_quality import DataQualityUtilities
from src.utils.ml_common.feature_selection import FeatureSelectionFramework
from src.utils.ml_common.model_evaluation import ModelEvaluationUtilities
from src.utils.ml_common.pipeline_orchestrator import MLPipelineOrchestrator
from src.utils.math_validation import (
    safe_divide,
    safe_log,
    safe_sqrt,
    safe_kelly_calculation,
    validate_positive,
    validate_range,
    MathValidationError
)
from src.utils.parquet_utils import get_parquet_utils
# Core decorators imports
from src.core.decorators import (
    handles_errors,
    traced,
    validates,
    log_execution_time,
    cached,
    error_boundary,
    timeout,
    retry,
    memory_efficient
)
# Core errors imports
from src.core.errors import (
    AppError,
    ValidationError,
    DataIntegrityError,
    NotFoundError,
    TimeoutError
)
from src.utils.enhanced_memory_management import (
    MemoryMonitor,
    MemoryConfig,
    chunk_dataframe
)
from src.utils.data_streaming_manager import DataStreamingManager

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

# Import financial metrics logging system
try:
    from .step04_5_financial_logging import Step04_5FinancialLogger
    FINANCIAL_LOGGING_AVAILABLE = True
except ImportError:
    FINANCIAL_LOGGING_AVAILABLE = False

# Initialize logger using common utilities
logger = get_logger('Step4TripleBarrierMethod')

class TripleBarrierMethodStep:
    """Step 4: Triple Barrier Method with enhanced data quality management."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = get_logger('TripleBarrierMethodStep')
        self.start_time: Optional[float] = None
        self.step_timings: Dict[str, float] = {}

        # Initialize parquet utilities
        self.parquet_utils = get_parquet_utils()

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

        # Initialize ml_common utilities
        self.data_quality_utils = DataQualityUtilities({
            'outlier_contamination': 0.05,  # More conservative for financial data
            'missing_threshold': 0.1,  # 10% threshold for financial data
            'drift_threshold': 0.05
        })

        self.feature_selector = FeatureSelectionFramework({
            'enable_gpu': True,
            'enable_parallel': True,
            'max_workers': 4,
            'random_state': 42
        })

        self.evaluator = ModelEvaluationUtilities({
            'enable_gpu': True,
            'enable_detailed_metrics': True,
            'confidence_thresholds': [0.5, 0.7, 0.9],
            'performance_stability_window': 30
        })

        # Initialize pipeline orchestrator for structured execution (optional)
        if config.get('use_pipeline_orchestrator', False):
            self.pipeline_orchestrator = MLPipelineOrchestrator({
                'max_workers': 4,
                'enable_parallel': True,
                'default_timeout': 3600,
                'enable_monitoring': True
            })
        else:
            self.pipeline_orchestrator = None
        
        # Risk management configuration
        self.risk_config = {
            'max_position_size_pct': safe_float(config.get('max_position_size_pct', 0.1), 0.1),  # 10% max
            'max_daily_trades': safe_int(config.get('max_daily_trades', 100), 100),
            'max_drawdown_pct': safe_float(config.get('max_drawdown_pct', 0.05), 0.05),  # 5% max
            'min_risk_reward_ratio': safe_float(config.get('min_risk_reward_ratio', 1.0), 1.0),
            'max_volatility_pct': safe_float(config.get('max_volatility_pct', 0.1), 0.1),  # 10% max
            'enable_risk_controls': config.get('enable_risk_controls', True)
        }

        # Initialize financial metrics logging system
        if FINANCIAL_LOGGING_AVAILABLE:
            try:
                self.financial_logger = None  # Will be initialized per execution
                self.logger.info('✅ Financial metrics logging system available')
            except Exception as e:
                self.logger.warning(f'⚠️ Financial metrics logging system failed to initialize: {e}')
                self.financial_logger = None
        else:
            self.logger.info('ℹ️ Financial metrics logging system not available, using basic reporting')
            self.financial_logger = None

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

            # Store data for volatility-based parameter calculation
            self._last_data = data

            # Enhanced data quality assessment using ml_common
            self.logger.info('🔍 Performing comprehensive data quality assessment...')
            quality_report = await self._perform_data_quality_assessment(data, symbol, exchange, timeframe)
            if quality_report.get('critical_issues', False):
                self.logger.warning('⚠️ Critical data quality issues detected')
                # Continue but log warnings

            # Feature selection and optimization if enabled
            if self.config.get('enable_feature_selection', True):
                self.logger.info('🎯 Performing feature selection and optimization...')
                data = await self._optimize_features(data, symbol, exchange, timeframe)
            
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
                # Enhanced evaluation using ml_common
                self.logger.info('📊 Performing comprehensive triple barrier evaluation...')
                evaluation_results = await self._perform_enhanced_evaluation(
                    data, labeled_data, symbol, exchange, timeframe
                )

                return self._create_enhanced_success_result(
                    data, labeled_data, symbol, exchange, timeframe, data_dir,
                    step_start, evaluation_results
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
        """Calculate comprehensive label statistics."""
        labels = labeled_data.get('triple_barrier_label')
        if labels is None:
            labels = labeled_data.get('label')
        if labels is None:
            return {'total_labels': 0, 'buy_signals': 0, 'sell_signals': 0, 'no_action': 0}

        stats = {
            'total_labels': int(len(labels)),
            'buy_signals': int((labels == 1).sum()),
            'sell_signals': int((labels == -1).sum()),
            'no_action': int((labels == 0).sum()),
            'signal_ratio': float(((labels == 1).sum() + (labels == -1).sum()) / len(labels)),
            'buy_ratio': float((labels == 1).sum() / len(labels)),
            'sell_ratio': float((labels == -1).sum() / len(labels)),
            'hold_ratio': float((labels == 0).sum() / len(labels))
        }
        return stats
    
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
        """Validate input data for triple barrier processing with fast fail checks."""
        try:
            # Fast fail checks for immediate error detection
            if data is None:
                return {'valid': False, 'error': 'Data is None'}
            if data.empty:
                return {'valid': False, 'error': 'Data is empty'}
            if len(data) < 100:  # Increased minimum for meaningful analysis
                return {'valid': False, 'error': f'Insufficient data points: {len(data)} (minimum 100 required)'}
            
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
            
            # FIXED: Prevent lookahead bias by using proper forward-looking validation
            for i in range(len(close_prices) - 1):
                entry_price = close_prices[i]
                profit_barrier = entry_price * (1 + profit_take_multiplier)
                stop_barrier = entry_price * (1 - stop_loss_multiplier)
                
                # Look ahead window (preventing lookahead bias by using only future data)
                lookahead_end = min(i + max_lookahead + 1, len(close_prices))
                future_highs = high_prices[i+1:lookahead_end]
                future_lows = low_prices[i+1:lookahead_end]
                
                if len(future_highs) == 0:
                    continue
                
                # Vectorized barrier hit detection
                profit_hits = future_highs >= profit_barrier
                stop_hits = future_lows <= stop_barrier
                
                # Find first hit
                profit_hit_idx = np.argmax(profit_hits) if np.any(profit_hits) else len(profit_hits)
                stop_hit_idx = np.argmax(stop_hits) if np.any(stop_hits) else len(stop_hits)
                
                # Determine outcome
                if profit_hit_idx < stop_hit_idx and np.any(profit_hits):
                    # Profit target hit first
                    labels[i] = 1
                    profit_pcts[i] = profit_take_multiplier
                elif stop_hit_idx < profit_hit_idx and np.any(stop_hits):
                    # Stop loss hit first
                    labels[i] = -1
                    profit_pcts[i] = -stop_loss_multiplier
                # If neither hit, label remains 0 (no action)
            
            result_data = pd.DataFrame({
                'label': labels,
                'potential_profit_pct': profit_pcts
            })
            return result_data
            
        except Exception as e:
            self.logger.exception(f'❌ Error in basic triple barrier: {e}')
            return pd.DataFrame()


    def _get_triple_barrier_config(self) -> Dict[str, Union[float, int]]:
        """Extract triple barrier configuration parameters with safe defaults, validation, and volatility-based suggestions."""
        triple_barrier_config = safe_dict_get(self.config, 'triple_barrier', {})
        
        # Check if volatility-based parameters should be used
        use_volatility_based = self.config.get('use_volatility_based_params', False)
        
        if use_volatility_based and hasattr(self, '_last_data') and self._last_data is not None:
            # Calculate volatility-based parameters
            volatility_params = self._calculate_volatility_based_parameters(self._last_data)
            triple_barrier_config.update(volatility_params)
            self.logger.info('📊 Using volatility-based parameters')
        
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
    
    def _calculate_volatility_based_parameters(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate optimal triple barrier parameters based on market volatility."""
        try:
            if len(data) < 30:
                return self._get_default_parameters()
            
            # Calculate rolling volatility
            returns = data['close'].pct_change().dropna()
            volatility = returns.rolling(window=30).std().iloc[-1]
            
            # Calculate ATR (Average True Range) for volatility measure
            high_low = data['high'] - data['low']
            high_close = np.abs(data['high'] - data['close'].shift())
            low_close = np.abs(data['low'] - data['close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            atr = true_range.rolling(window=30).mean().iloc[-1]
            
            # Calculate current price level
            current_price = data['close'].iloc[-1]
            
            # Volatility-based parameter calculation
            volatility_multiplier = min(max(volatility * 100, 0.5), 5.0)  # Clamp between 0.5% and 5%
            atr_multiplier = min(max(atr / current_price * 100, 0.1), 2.0)  # Clamp between 0.1% and 2%
            
            # Calculate optimal parameters
            profit_take_multiplier = max(volatility_multiplier * 0.8, 0.001)  # 80% of volatility
            stop_loss_multiplier = max(volatility_multiplier * 0.4, 0.0005)   # 40% of volatility
            
            # Time barrier based on volatility (higher volatility = shorter time barrier)
            base_time_minutes = 30
            volatility_time_factor = max(0.5, min(2.0, 1.0 / (volatility * 100 + 0.1)))
            time_barrier_minutes = int(base_time_minutes * volatility_time_factor)
            
            # Max lookahead based on volatility
            base_lookahead = 100
            volatility_lookahead_factor = max(0.5, min(2.0, 1.0 / (volatility * 100 + 0.1)))
            max_lookahead = int(base_lookahead * volatility_lookahead_factor)
            
            parameters = {
                'profit_take_multiplier': round(profit_take_multiplier, 6),
                'stop_loss_multiplier': round(stop_loss_multiplier, 6),
                'time_barrier_minutes': time_barrier_minutes,
                'max_lookahead': max_lookahead,
                'volatility': round(volatility, 6),
                'atr': round(atr, 6),
                'volatility_multiplier': round(volatility_multiplier, 6),
                'parameter_source': 'volatility_based'
            }
            
            self.logger.info(f'📊 Volatility-based parameters calculated:')
            self.logger.info(f'   Volatility: {volatility:.4f} ({volatility*100:.2f}%)')
            self.logger.info(f'   Profit Take: {profit_take_multiplier:.4f} ({profit_take_multiplier*100:.2f}%)')
            self.logger.info(f'   Stop Loss: {stop_loss_multiplier:.4f} ({stop_loss_multiplier*100:.2f}%)')
            self.logger.info(f'   Time Barrier: {time_barrier_minutes} minutes')
            self.logger.info(f'   Max Lookahead: {max_lookahead} periods')
            
            return parameters
            
        except Exception as e:
            self.logger.warning(f'⚠️ Failed to calculate volatility-based parameters: {e}')
            return self._get_default_parameters()
    
    def _get_default_parameters(self) -> Dict[str, float]:
        """Get default parameters when volatility calculation fails."""
        return {
            'profit_take_multiplier': 0.002,  # 0.2%
            'stop_loss_multiplier': 0.001,    # 0.1%
            'time_barrier_minutes': 30,
            'max_lookahead': 100,
            'volatility': 0.0,
            'atr': 0.0,
            'volatility_multiplier': 1.0,
            'parameter_source': 'default'
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
    
    async def _perform_data_quality_assessment(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Perform comprehensive data quality assessment using ml_common utilities."""
        try:
            self.logger.info('🔍 Running automated data quality analysis...')

            # Missing value analysis
            missing_analysis = self.data_quality_utils.missing_value_analysis(data)
            if missing_analysis.get('severity_assessment', {}).get('action_required', False):
                self.logger.warning(f'⚠️ Missing value issues detected: {missing_analysis["severity_assessment"]["severity_level"]}')

            # Outlier detection
            outlier_analysis = self.data_quality_utils.automated_outlier_detection(data)
            if outlier_analysis.get('summary', {}).get('outlier_percentage', 0) > 5:
                self.logger.warning(f'⚠️ High outlier percentage detected: {outlier_analysis["summary"]["outlier_percentage"]:.2f}%')

            # Feature correlation analysis
            correlation_analysis = self.data_quality_utils.feature_correlation_analysis(data)
            if correlation_analysis.get('multicollinearity_analysis', {}).get('highly_correlated_pairs_count', 0) > 0:
                self.logger.warning(f'⚠️ Multicollinearity detected: {correlation_analysis["multicollinearity_analysis"]["highly_correlated_pairs_count"]} pairs')

            # Compile quality report
            quality_report = {
                'missing_analysis': missing_analysis,
                'outlier_analysis': outlier_analysis,
                'correlation_analysis': correlation_analysis,
                'critical_issues': (
                    missing_analysis.get('severity_assessment', {}).get('severity_level') == 'critical' or
                    outlier_analysis.get('summary', {}).get('outlier_percentage', 0) > 10
                ),
                'recommendations': []
            }

            # Collect recommendations
            quality_report['recommendations'].extend(missing_analysis.get('recommendations', []))
            quality_report['recommendations'].extend(outlier_analysis.get('recommendations', []))
            quality_report['recommendations'].extend(correlation_analysis.get('recommendations', []))

            self.logger.info(f'✅ Data quality assessment completed - {len(quality_report["recommendations"])} recommendations')
            return quality_report

        except Exception as e:
            self.logger.warning(f'⚠️ Data quality assessment failed: {e}')
            return {'error': str(e), 'critical_issues': False}

    async def _optimize_features(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
        """Optimize features using ml_common feature selection utilities."""
        try:
            self.logger.info('🎯 Starting feature optimization...')

            # Prepare target variable (use returns for feature selection)
            if 'close' in data.columns:
                # Calculate returns as target for feature relevance
                data_with_target = data.copy()
                data_with_target['returns'] = data_with_target['close'].pct_change().fillna(0)

                # Get feature columns (exclude target and non-numeric)
                feature_cols = [col for col in data_with_target.columns
                              if col not in ['returns', 'timestamp', 'datetime'] and
                              data_with_target[col].dtype in ['float64', 'float32', 'int64', 'int32']]

                if len(feature_cols) > 10:  # Only if we have enough features
                    # Correlation-based filtering
                    correlation_results = self.feature_selector.correlation_based_filtering(
                        data_with_target[feature_cols].values,
                        feature_cols,
                        correlation_threshold=0.95
                    )

                    selected_features = correlation_results.get('selected_features', feature_cols)
                    self.logger.info(f'📊 Correlation filtering: {len(feature_cols)} -> {len(selected_features)} features')

                    # Update data to include only selected features plus required columns
                    required_cols = ['open', 'high', 'low', 'close', 'volume'] if 'volume' in data.columns else ['open', 'high', 'low', 'close']
                    optimized_data = data[required_cols + [col for col in selected_features if col in data.columns]]

                    self.logger.info(f'✅ Feature optimization completed: {data.shape[1]} -> {optimized_data.shape[1]} columns')
                    return optimized_data
                else:
                    self.logger.info('ℹ️ Insufficient features for optimization, using original data')
                    return data
            else:
                self.logger.warning('⚠️ No close price column found for feature optimization')
                return data

        except Exception as e:
            self.logger.warning(f'⚠️ Feature optimization failed: {e}')
            return data

    async def _perform_enhanced_evaluation(self, data: pd.DataFrame, labeled_data: pd.DataFrame,
                                         symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Perform comprehensive evaluation of triple barrier results using ml_common utilities."""
        try:
            self.logger.info('📊 Starting enhanced triple barrier evaluation...')

            # Extract labels for evaluation
            labels = labeled_data['label'].values if 'label' in labeled_data.columns else labeled_data.get('triple_barrier_label', pd.Series()).values

            # Create dummy target for evaluation (we evaluate signal quality)
            # Use future returns as pseudo-target for signal evaluation
            if len(data) > 1:
                future_returns = data['close'].shift(-1).pct_change().fillna(0).values[:len(labels)]
            else:
                future_returns = np.zeros(len(labels))

            # Multi-metric evaluation
            evaluation_results = self.evaluator.multi_metric_evaluation(
                y_true=(future_returns > 0).astype(int),  # Binary target: positive return or not
                y_pred=(labels > 0).astype(int),  # Binary prediction: buy signal or not
                task_type='classification'
            )

            # Class imbalance analysis
            imbalance_analysis = self.evaluator.class_imbalance_aware_metrics(
                y_true=(future_returns > 0).astype(int),
                y_pred=(labels > 0).astype(int)
            )

            # Signal quality assessment
            signal_quality = self._assess_signal_quality(data, labeled_data)

            # Compile enhanced evaluation
            enhanced_evaluation = {
                'multi_metric_evaluation': evaluation_results,
                'imbalance_analysis': imbalance_analysis,
                'signal_quality': signal_quality,
                'triple_barrier_metrics': self._calculate_triple_barrier_metrics(labeled_data),
                'evaluation_summary': {
                    'total_signals': len(labels),
                    'buy_signals': int((labels > 0).sum()),
                    'sell_signals': int((labels < 0).sum()),
                    'hold_signals': int((labels == 0).sum()),
                    'signal_distribution': {
                        'buy_ratio': float((labels > 0).mean()),
                        'sell_ratio': float((labels < 0).mean()),
                        'hold_ratio': float((labels == 0).mean())
                    }
                }
            }

            self.logger.info(f'✅ Enhanced evaluation completed - {len(labels)} signals analyzed')
            return enhanced_evaluation

        except Exception as e:
            self.logger.warning(f'⚠️ Enhanced evaluation failed: {e}')
            return {'error': str(e)}

    def _assess_signal_quality(self, data: pd.DataFrame, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess the quality of trading signals."""
        try:
            signals = labeled_data['label'].values if 'label' in labeled_data.columns else labeled_data.get('triple_barrier_label', pd.Series()).values

            # Calculate signal quality metrics
            signal_quality = {
                'signal_purity': float(np.mean(np.abs(signals) > 0)),  # Ratio of non-zero signals
                'signal_balance': float(min(np.mean(signals > 0), np.mean(signals < 0)) / max(np.mean(signals > 0), np.mean(signals < 0))) if max(np.mean(signals > 0), np.mean(signals < 0)) > 0 else 0.0,
                'signal_strength_distribution': {
                    'weak_signals': float(np.mean(np.abs(signals) < 0.5)),
                    'medium_signals': float(np.mean((np.abs(signals) >= 0.5) & (np.abs(signals) < 0.8))),
                    'strong_signals': float(np.mean(np.abs(signals) >= 0.8))
                }
            }

            # Profit potential analysis if available
            if 'potential_profit_pct' in labeled_data.columns:
                profits = labeled_data['potential_profit_pct'].values
                signal_quality['profitability_analysis'] = {
                    'avg_profit_buy_signals': float(profits[signals > 0].mean()) if np.any(signals > 0) else 0.0,
                    'avg_profit_sell_signals': float(profits[signals < 0].mean()) if np.any(signals < 0) else 0.0,
                    'profitable_signals_ratio': float(np.mean(profits > 0)),
                    'high_profit_signals': float(np.mean(profits > 0.01))  # >1% profit
                }

            return signal_quality

        except Exception as e:
            return {'error': str(e)}

    def _calculate_triple_barrier_metrics(self, labeled_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate specific triple barrier method metrics."""
        try:
            metrics = {}

            if 'potential_profit_pct' in labeled_data.columns:
                profits = labeled_data['potential_profit_pct'].values
                metrics['profit_distribution'] = {
                    'mean_profit': float(np.mean(profits)),
                    'std_profit': float(np.std(profits)),
                    'max_profit': float(np.max(profits)),
                    'min_profit': float(np.min(profits)),
                    'profit_percentiles': {
                        '25': float(np.percentile(profits, 25)),
                        '50': float(np.percentile(profits, 50)),
                        '75': float(np.percentile(profits, 75)),
                        '95': float(np.percentile(profits, 95))
                    }
                }

            # Barrier hit analysis
            labels = labeled_data['label'].values if 'label' in labeled_data.columns else labeled_data.get('triple_barrier_label', pd.Series()).values
            metrics['barrier_analysis'] = {
                'profit_barrier_hits': int(np.sum(labels == 1)),
                'stop_barrier_hits': int(np.sum(labels == -1)),
                'time_barrier_hits': int(np.sum(labels == 0)),
                'total_signals': len(labels)
            }

            return metrics

        except Exception as e:
            return {'error': str(e)}

    def _create_enhanced_success_result(
        self,
        data: pd.DataFrame,
        labeled_data: pd.DataFrame,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        step_start: float,
        evaluation_results: Dict[str, Any]
    ) -> TripleBarrierResult:
        """Create enhanced success result with comprehensive evaluation metrics."""
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

            # Calculate net profit after fees (corrected fee: 0.04% per side)
            fee_per_side = float(self.config.get('TRADING_FEE_PCT_PER_SIDE', 0.0004))  # 0.04% per side
            result_data['potential_profit_net_pct'] = (
                result_data['potential_profit_pct'] - (2.0 * fee_per_side)
            ).astype(np.float64)

        output_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_triple_barrier_labels.parquet'

        # Enhanced metadata with evaluation results
        enhanced_metadata = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'output_file': str(output_path),
            'data_shape': data.shape,
            'label_stats': label_stats,
            'memory_stats': self.memory_monitor.get_memory_stats(),
            'risk_metrics': self._calculate_risk_metrics(result_data),
            'evaluation_results': evaluation_results,
            'ml_common_enhanced': True
        }

        return TripleBarrierResult.success_result(
            data=result_data,
            metadata=enhanced_metadata,
            execution_time=time.time() - step_start
        )

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


    async def execute_with_pipeline_orchestrator(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str = 'data_cache',
        force_rerun: bool = False
    ) -> TripleBarrierResult:
        """Execute step04 using pipeline orchestrator for structured workflow management."""
        try:
            self.logger.info('🔧 Creating structured pipeline for triple barrier method...')

            # Initialize pipeline context
            pipeline_context = {}

            # Define pipeline steps with context passing
            def data_loading_step():
                return self._pipeline_data_loading(symbol, exchange, timeframe, data_dir, pipeline_context)

            def data_quality_step():
                return self._pipeline_data_quality(symbol, exchange, timeframe, pipeline_context)

            def feature_optimization_step():
                return self._pipeline_feature_optimization(symbol, exchange, timeframe, pipeline_context)

            def triple_barrier_step():
                return self._pipeline_triple_barrier_labeling(symbol, exchange, timeframe, data_dir, pipeline_context)

            def evaluation_step():
                return self._pipeline_enhanced_evaluation(symbol, exchange, timeframe, pipeline_context)

            def save_step():
                return self._pipeline_save_results(symbol, exchange, timeframe, data_dir, pipeline_context)

            # Execute pipeline steps sequentially with error handling
            try:
                # Step 1: Data Loading
                self.logger.info('📁 Executing: Data Loading')
                data_loading_result = data_loading_step()
                self.logger.info(f'✅ Data loading completed: {data_loading_result}')

                # Step 2: Data Quality Assessment
                self.logger.info('🔍 Executing: Data Quality Assessment')
                quality_result = data_quality_step()
                self.logger.info(f'✅ Data quality assessment completed: {len(quality_result.get("recommendations", []))} recommendations')

                # Step 3: Feature Optimization
                self.logger.info('🎯 Executing: Feature Optimization')
                feature_result = feature_optimization_step()
                self.logger.info(f'✅ Feature optimization completed: {feature_result}')

                # Step 4: Triple Barrier Labeling
                self.logger.info('🏷️ Executing: Triple Barrier Labeling')
                labeling_result = triple_barrier_step()
                self.logger.info(f'✅ Triple barrier labeling completed: {labeling_result}')

                # Step 5: Enhanced Evaluation
                self.logger.info('📊 Executing: Enhanced Evaluation')
                evaluation_result = evaluation_step()
                self.logger.info(f'✅ Enhanced evaluation completed: {len(evaluation_result)} metrics')

                # Step 6: Save Results
                self.logger.info('💾 Executing: Save Results')
                final_result = save_step()
                self.logger.info('✅ Pipeline execution completed successfully')

                return final_result

            except Exception as step_error:
                self.logger.error(f'❌ Pipeline step failed: {step_error}')
                return TripleBarrierResult.failure_result(
                    error=f'Pipeline step failed: {str(step_error)}',
                    error_type=type(step_error).__name__,
                    metadata={'symbol': symbol, 'exchange': exchange, 'timeframe': timeframe}
                )

        except Exception as e:
            self.logger.exception(f'❌ Pipeline orchestrator execution failed: {e}')
            return TripleBarrierResult.failure_result(
                error=str(e),
                error_type=type(e).__name__
            )

    # Pipeline step methods for orchestrator (static methods for pipeline execution)
    def _pipeline_data_loading(self, symbol: str, exchange: str, timeframe: str, data_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline step: Load data."""
        import asyncio
        async def async_load():
            self.logger.info('📁 Pipeline Step: Data Loading')
            data = await self._load_data_with_streaming(symbol, exchange, timeframe, data_dir)
            if data is None or data.empty:
                raise ValueError('Failed to load data')
            context['data'] = data
            return {'data_shape': data.shape, 'success': True}
        return asyncio.run(async_load())

    def _pipeline_data_quality(self, symbol: str, exchange: str, timeframe: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline step: Data quality assessment."""
        import asyncio
        async def async_quality():
            self.logger.info('🔍 Pipeline Step: Data Quality Assessment')
            data = context.get('data')
            if data is None:
                raise ValueError('No data available from previous step')

            quality_report = await self._perform_data_quality_assessment(data, symbol, exchange, timeframe)
            context['quality_report'] = quality_report
            return quality_report
        return asyncio.run(async_quality())

    def _pipeline_feature_optimization(self, symbol: str, exchange: str, timeframe: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline step: Feature optimization."""
        import asyncio
        async def async_optimize():
            self.logger.info('🎯 Pipeline Step: Feature Optimization')
            data = context.get('data')
            if data is None:
                raise ValueError('No data available from previous step')

            optimized_data = await self._optimize_features(data, symbol, exchange, timeframe)
            context['optimized_data'] = optimized_data
            return {'original_shape': data.shape, 'optimized_shape': optimized_data.shape}
        return asyncio.run(async_optimize())

    def _pipeline_triple_barrier_labeling(self, symbol: str, exchange: str, timeframe: str, data_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline step: Triple barrier labeling."""
        import asyncio
        async def async_label():
            self.logger.info('🏷️ Pipeline Step: Triple Barrier Labeling')
            data = context.get('optimized_data', context.get('data'))
            if data is None:
                raise ValueError('No data available for labeling')

            # Use streaming for large datasets
            if self.streaming_manager.should_chunk_data(data):
                labeled_data = await self._process_large_dataset(data)
            else:
                if self.triple_barrier_labeler:
                    labeled_data = await self._apply_optimized_triple_barrier(data)
                else:
                    labeled_data = await self._apply_basic_triple_barrier(data)

            if labeled_data is None or labeled_data.empty:
                raise ValueError('Failed to generate triple barrier labels')

            # Apply risk management
            labeled_data = self._apply_risk_management_controls(data, labeled_data)
            context['labeled_data'] = labeled_data
            return {'labeled_shape': labeled_data.shape, 'success': True}
        return asyncio.run(async_label())

    def _pipeline_enhanced_evaluation(self, symbol: str, exchange: str, timeframe: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Pipeline step: Enhanced evaluation."""
        import asyncio
        async def async_evaluate():
            self.logger.info('📊 Pipeline Step: Enhanced Evaluation')
            data = context.get('optimized_data', context.get('data'))
            labeled_data = context.get('labeled_data')

            if data is None or labeled_data is None:
                raise ValueError('Missing data or labeled data for evaluation')

            evaluation_results = await self._perform_enhanced_evaluation(data, labeled_data, symbol, exchange, timeframe)
            context['evaluation_results'] = evaluation_results
            return evaluation_results
        return asyncio.run(async_evaluate())

    def _pipeline_save_results(self, symbol: str, exchange: str, timeframe: str, data_dir: str, context: Dict[str, Any]) -> TripleBarrierResult:
        """Pipeline step: Save results."""
        import asyncio
        async def async_save():
            self.logger.info('💾 Pipeline Step: Save Results')
            data = context.get('optimized_data', context.get('data'))
            labeled_data = context.get('labeled_data')
            evaluation_results = context.get('evaluation_results')

            if data is None or labeled_data is None:
                raise ValueError('Missing data for saving results')

            # Save results
            success = await self._save_results_optimized(data, labeled_data, symbol, exchange, timeframe, data_dir)

            if success:
                # Create final result
                return self._create_enhanced_success_result(
                    data, labeled_data, symbol, exchange, timeframe, data_dir,
                    time.time(), evaluation_results
                )
            else:
                raise ValueError('Failed to save results')
        return asyncio.run(async_save())


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
        }
    }

    # Merge config if provided
    if config:
        step_config.update(config)
    
    step_start = time.time()
    try:
        step = TripleBarrierMethodStep(step_config)
        await step.initialize()

        # Choose execution mode based on config
        use_pipeline_orchestrator = config.get('use_pipeline_orchestrator', False) if config else False

        if use_pipeline_orchestrator:
            # Use structured pipeline execution
            result = await step.execute_with_pipeline_orchestrator(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun
            )
        else:
            # Use traditional execution with ml_common enhancements
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