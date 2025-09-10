
"""
Step 5: Labeling (Streamlined Implementation using ml_common utilities)

This module provides a streamlined `LabelingStep` that leverages the comprehensive
ml_common utilities for optimal performance, data quality, and parallel processing.
"""

from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import time
import datetime
import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.utils.lookahead_bias_detector import (
    get_global_detector, validate_no_future_data, LookaheadBiasError
)
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

# Import ml_common utilities
from src.utils.ml_common.data_quality import DataQualityUtilities
from src.utils.ml_common.pipeline_orchestrator import MLPipelineOrchestrator
from src.utils.ml_common.parallel_processing import ParallelProcessingCoordinator
from src.utils.ml_common.memory_optimization import MemoryEfficientTraining
from src.utils.ml_common.model_evaluation import ModelEvaluationUtilities
from src.utils.ml_common.validation_utils import (
    MLValidationSuite, ValidationError, create_validation_suite
)

# Import existing utilities that are still needed
from src.utils.math_validation import safe_divide, MathValidationError
try:
    from src.analyst import meta_labeling_system as meta_labeling_system
except Exception:
    meta_labeling_system = None

def ensure_directory(path: Path | str) -> Path:
    """Ensure directory exists and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _build_labeled_data_path(data_dir: str, symbol: str, exchange: str, timeframe: str) -> Path:
    """Build path for labeled data file."""
    return Path(data_dir) / 'training' / 'labeled_data' / f'{exchange}_{symbol}_{timeframe}_labeled_data.parquet'

class LabelingStep:
    """Streamlined labeling step using ml_common utilities.

    This implementation leverages comprehensive ml_common utilities for:
    - Data quality validation and cleaning
    - Parallel processing and memory optimization
    - Pipeline orchestration
    - Model evaluation and quality assessment
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.logger = system_logger.getChild('LabelingStep')
        self.start_time: Optional[float] = None
        self.meta_labeling_system: Optional[Any] = None

        # Initialize ML validation suite
        self.validation_suite = create_validation_suite(self.logger)

        # Fast fail: Validate configuration before initialization
        self.validation_suite.config_validator.validate_ml_config(config)

        # Initialize ml_common utilities
        self._init_ml_common_utilities()

        # Initialize meta-labeling system if available
        self._init_meta_labeling_system()


    def _init_ml_common_utilities(self) -> None:
        """Initialize ml_common utilities."""
        try:
            # Data quality utilities for validation and cleaning
            self.data_quality = DataQualityUtilities({
                'outlier_contamination': 0.1,
                'missing_threshold': 0.5,
                'correlation_method': 'spearman'
            })

            # Pipeline orchestrator for coordinated execution
            self.pipeline_orchestrator = MLPipelineOrchestrator({
                'max_workers': 4,
                'enable_parallel': True,
                'default_timeout': 3600
            })

            # Parallel processing coordinator
            self.parallel_processor = ParallelProcessingCoordinator({
                'max_workers': 4,
                'enable_parallel': True,
                'chunk_size': 1000
            })

            # Memory optimization utilities
            self.memory_optimizer = MemoryEfficientTraining({
                'chunk_size_mb': 500,
                'max_memory_usage': 0.8,
                'enable_gpu_memory_pool': True
            })

            # Model evaluation utilities
            self.evaluator = ModelEvaluationUtilities({
                'enable_gpu': True,
                'enable_detailed_metrics': True
            })

            self.logger.info('✅ ml_common utilities initialized successfully')

        except Exception as e:
            self.logger.warning(f'⚠️ Failed to initialize some ml_common utilities: {e}')
            # Initialize with fallbacks
            self.data_quality = None
            self.pipeline_orchestrator = None
            self.parallel_processor = None
            self.memory_optimizer = None
            self.evaluator = None

    def _init_meta_labeling_system(self) -> None:
        """Initialize meta-labeling system if available."""
        if meta_labeling_system is not None:
            try:
                self.meta_labeling_system = meta_labeling_system.MetaLabelingSystem(self.config)
                self.logger.info('✅ Meta-labeling system initialized')
            except Exception as e:
                self.logger.warning(f'⚠️ Meta-labeling system initialization failed: {e}')
                self.meta_labeling_system = None
        else:
            self.logger.info('ℹ️ Meta-labeling system not available')


    async def initialize(self) -> None:
        self.start_time = time.time()
        self.logger.info('LabelingStep initialized')

    async def _load_data_with_labels(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Load labeled data using ml_common utilities."""
        try:
            path = _build_labeled_data_path(data_dir, symbol, exchange, timeframe)

            # Use memory-efficient loading if available
            if self.memory_optimizer is not None:
                with self.memory_optimizer.memory_efficient_context():
                    df = standardized_parquet_handler.read_parquet_standardized(path)
            else:
                df = standardized_parquet_handler.read_parquet_standardized(path)

            if df is not None and 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df = df.set_index(pd.to_datetime(df['timestamp']))
                    self.logger.info('📅 Set datetime index')
                except Exception as e:
                    self.logger.warning(f'⚠️ Could not set datetime index: {e}')

            # Apply data quality checks if available
            if self.data_quality is not None and df is not None:
                quality_analysis = self.data_quality.missing_value_analysis(df)
                if quality_analysis.get('severity_assessment', {}).get('action_required', False):
                    self.logger.warning('⚠️ Data quality issues detected in loaded data')

            return df

        except FileNotFoundError:
            self.logger.warning('Labeled data file not found')
            return None
        except Exception as e:
            self.logger.exception(f'Failed to load labeled data: {e}')
            return None

    async def _create_meta_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create meta labels using ml_common utilities."""
        out = data.copy()

        if self.meta_labeling_system is not None:
            try:
                # Use parallel processing if data is large
                if self.parallel_processor is not None and len(out) > 10000:
                    self.logger.info("🔄 Using parallel processing for meta-labeling")

                    # Create tasks for parallel processing
                    chunk_size = 5000  # Reasonable chunk size
                    chunks = [out.iloc[i:i + chunk_size] for i in range(0, len(out), chunk_size)]

                    tasks = []
                    for i, chunk in enumerate(chunks):
                        tasks.append({
                            'function': self._process_meta_labeling_chunk,
                            'data': chunk,
                            'chunk_idx': i
                        })

                    # Execute in parallel
                    results = self.parallel_processor.parallel_processing(
                        tasks, lambda task: self._process_meta_labeling_chunk(task['data'])
                    )

                    # Combine results
                    if self.memory_optimizer is not None:
                        out = self.memory_optimizer.memory_efficient_concat(results)
                    else:
                        out = pd.concat(results, ignore_index=True)
                else:
                    # Standard processing
                    result = await self.meta_labeling_system.generate_meta_labels(out)
                meta = result.get('meta_labels')
                conf = result.get('confidence_scores')
                if meta is not None:
                    out['meta_label'] = np.asarray(meta)
                if conf is not None:
                    out['confidence'] = np.asarray(conf)

                return out

            except Exception as e:
                self.logger.warning(f'Meta-labeling failed, using fallback: {e}')

        # Fallback labeling
        return await self._create_fallback_labels(out)

    async def _process_meta_labeling_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """Process a chunk of data for meta-labeling."""
        try:
            result = await self.meta_labeling_system.generate_meta_labels(chunk)
            meta = result.get('meta_labels')
            conf = result.get('confidence_scores')

            if meta is not None:
                chunk = chunk.copy()
                chunk['meta_label'] = np.asarray(meta)
            if conf is not None:
                if 'meta_label' not in chunk.columns:
                    chunk = chunk.copy()
                chunk['confidence'] = np.asarray(conf)

            return chunk
        except Exception as e:
            self.logger.warning(f"Chunk processing failed: {e}")
            return chunk

    async def _create_fallback_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create fallback labels using simple heuristics."""
        out = data.copy()

        # Use existing labels if available
        if 'label' in out.columns:
            out['meta_label'] = out['label'].astype(float).fillna(0.0).values
        else:
            out['meta_label'] = np.zeros(len(out), dtype=float)

        # Calculate confidence based on price volatility if available
        if 'close' in out.columns:
            try:
                # Simple volatility-based confidence calculation
                volatility = out['close'].pct_change().abs().fillna(0.0)
                max_vol = volatility.max()
                if max_vol > 0:
                    confidence = (1.0 - safe_divide(volatility, max_vol + 1e-09, 1.0)).clip(0.0, 1.0)
                else:
                    confidence = np.ones_like(volatility) * 0.5  # Default moderate confidence

                out['confidence'] = np.asarray(confidence)
            except MathValidationError as e:
                self.logger.warning(f"Mathematical validation error in confidence calculation: {e}")
                out['confidence'] = np.ones(len(out), dtype=float) * 0.5
        else:
            out['confidence'] = np.ones(len(out), dtype=float) * 0.5

        return out
    def _calculate_label_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate statistics about the generated labels."""
        total = int(len(data))
        dist: Dict[str, int] = {}

        if 'label' in data.columns:
            vc = data['label'].value_counts().to_dict()
            dist = {str(int(k)) if isinstance(k, (int, np.integer)) else str(k): int(v) for k, v in vc.items()}
            buy = int((data['label'] == 1).sum())
            sell = int((data['label'] == -1).sum())
            flat = int((data['label'] == 0).sum())
        else:
            buy = sell = flat = 0

        # Calculate average confidence
        avg_conf = 0.0
        if 'confidence' in data.columns:
            try:
                confidence_series = data['confidence']
                avg_conf = float(confidence_series.mean()) if not confidence_series.empty else 0.0
                avg_conf = avg_conf if not np.isnan(avg_conf) else 0.0
            except Exception as e:
                self.logger.warning(f"Error calculating confidence: {e}")

        return {
            'total_samples': total,
            'buy_signals': buy,
            'sell_signals': sell,
            'no_action': flat,
            'avg_confidence': avg_conf,
            'label_distribution': dist
        }

    async def _save_labeled_data(self, data: pd.DataFrame, data_dir: str, symbol: str, exchange: str, timeframe: str) -> str:
        """Save labeled data using ml_common utilities."""
        out_dir = ensure_directory(Path(data_dir) / 'training' / 'labeled_data')
        data_to_save = data.copy()

        # Prepare labels
        if 'label' not in data_to_save.columns:
            if 'meta_label' in data_to_save.columns:
                try:
                    data_to_save['label'] = np.sign(data_to_save['meta_label']).astype(int)
                except Exception:
                    data_to_save['label'] = 0
            else:
                data_to_save['label'] = 0

        out_path = out_dir / f'{exchange}_{symbol}_{timeframe}_labeled_data.parquet'

        # Use memory-efficient saving if available
        if self.memory_optimizer is not None:
            with self.memory_optimizer.memory_efficient_context():
                standardized_parquet_handler.write_parquet_standardized(data_to_save, out_path)
        else:
            standardized_parquet_handler.write_parquet_standardized(data_to_save, out_path)

        # Create and save metadata
        try:
            metadata = self._create_labeling_metadata(data_to_save, symbol, exchange, timeframe)

            # Save metadata using centralized reporting system
            from src.training.reports import save_training_report
            report_path = save_training_report(
                data=metadata,
                step_name='step5_labeling',
                report_type='labeling_metadata',
                symbol=symbol,
                timeframe=timeframe,
                file_format='json'
            )

            self.logger.info(f'💾 Labeling metadata saved to: {report_path}')

        except Exception as e:
            self.logger.warning(f"Failed to save metadata: {e}")

        return str(out_path)

    def _create_labeling_metadata(self, data: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Create metadata for the labeling operation."""
        metadata = {
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'total_samples': int(len(data)),
            'labeling_method': 'meta_labeling' if 'meta_label' in data.columns else 'fallback',
            'labeling_timestamp': datetime.datetime.utcnow().isoformat(),
        }

        # Add label distribution if available
        if 'label' in data.columns:
            try:
                label_counts = data['label'].value_counts().to_dict()
                metadata['label_distribution'] = {
                    int(k) if isinstance(k, (int, np.integer)) else k: int(v)
                    for k, v in label_counts.items()
                }
            except Exception as e:
                self.logger.warning(f"Error creating label distribution: {e}")

        # Add confidence statistics if available
        if 'confidence' in data.columns:
            try:
                metadata['confidence_stats'] = {
                    'mean': float(data['confidence'].mean()),
                    'std': float(data['confidence'].std()),
                    'min': float(data['confidence'].min()),
                    'max': float(data['confidence'].max())
                }
            except Exception as e:
                self.logger.warning(f"Error calculating confidence stats: {e}")

        return metadata
    def _validate_labels(self, data: pd.DataFrame) -> bool:
        """Validate that required label columns are present."""
        return 'meta_label' in data.columns and 'confidence' in data.columns

    async def execute_labeling(self, *, symbol: str, exchange: str, timeframe: str, data_dir: str, force_rerun: bool = False) -> bool:
        """Execute labeling using ml_common utilities."""
        start_time = time.time()

        # Use pipeline orchestrator if available for coordinated execution
        if self.pipeline_orchestrator is not None:
            self.logger.info("🚀 Using pipeline orchestrator for labeling")
            return await self._execute_with_pipeline_orchestrator(symbol, exchange, timeframe, data_dir, force_rerun)

        # Fallback to standard execution
        return await self._execute_standard_labeling(symbol, exchange, timeframe, data_dir, force_rerun)

    async def _execute_with_pipeline_orchestrator(self, symbol: str, exchange: str, timeframe: str, data_dir: str, force_rerun: bool) -> bool:
        """Execute labeling using the ml_common pipeline orchestrator."""
        try:
            # Define pipeline steps
            pipeline_steps = [
                {
                    'name': 'load_data',
                    'function': self._load_data_with_labels,
                    'args': [symbol, exchange, timeframe, data_dir]
                },
                {
                    'name': 'validate_data',
                    'function': self._validate_input_data,
                    'dependencies': ['load_data']
                },
                {
                    'name': 'create_labels',
                    'function': self._create_meta_labels,
                    'dependencies': ['validate_data']
                },
                {
                    'name': 'validate_labels',
                    'function': self._validate_labels,
                    'dependencies': ['create_labels']
                },
                {
                    'name': 'calculate_stats',
                    'function': self._calculate_label_statistics,
                    'dependencies': ['validate_labels']
                },
                {
                    'name': 'save_data',
                    'function': self._save_labeled_data,
                    'args': [data_dir, symbol, exchange, timeframe],
                    'dependencies': ['calculate_stats']
                }
            ]

            # Create and execute pipeline
            pipeline_id = self.pipeline_orchestrator.create_training_pipeline(
                steps_config=pipeline_steps,
                error_handling='robust'
            )

            result = await self.pipeline_orchestrator.execute_pipeline(pipeline_id)

            if result.get('success', False):
                # Log results
                stats = result.get('results', {}).get('calculate_stats', {})
                self.logger.info(f"✅ Pipeline execution completed successfully")
                return True
            else:
                self.logger.error(f"❌ Pipeline execution failed: {result.get('errors', [])}")
                return False

        except Exception as e:
            self.logger.exception(f"Pipeline execution failed: {e}")
            return False

    async def _execute_standard_labeling(self, symbol: str, exchange: str, timeframe: str, data_dir: str, force_rerun: bool) -> bool:
        """Execute standard labeling workflow using ml_common utilities."""
        try:
            # Use memory-efficient context if available
            if self.memory_optimizer is not None:
                with self.memory_optimizer.memory_efficient_context():
                    df = await self._load_data_with_labels(symbol, exchange, timeframe, data_dir)
            else:
                df = await self._load_data_with_labels(symbol, exchange, timeframe, data_dir)

            if df is None or len(df) == 0:
                self.logger.error('No input data available for labeling')
                return False

            # Validate input data using data quality utilities
            if not await self._validate_input_data(df):
                self.logger.error('Input data validation failed')
                return False

            # Create labels
            labeled = await self._create_meta_labels(df)
            if not self._validate_labels(labeled):
                self.logger.error('Generated labels are invalid')
                return False

            # Calculate and log statistics
            stats = self._calculate_label_statistics(labeled)
            self.logger.info(f"🏷️ Labeling completed: {stats}")

            # Save labeled data
            await self._save_labeled_data(labeled, data_dir, symbol, exchange, timeframe)

            # Use model evaluation utilities for quality assessment
            if self.evaluator is not None:
                try:
                    # Perform comprehensive evaluation
                    evaluation_results = self.evaluator.multi_metric_evaluation(
                        y_true=labeled.get('label', pd.Series()),
                        y_pred=labeled.get('label', pd.Series()),
                        y_prob=labeled.get('confidence', pd.Series()).values.reshape(-1, 1) if 'confidence' in labeled.columns else None,
                        task_type='classification',
                        class_names=['sell', 'hold', 'buy']
                    )
                    self.logger.info(f"📊 Model evaluation completed: {evaluation_results.get('basic_metrics', {})}")
                except Exception as e:
                    self.logger.warning(f"Model evaluation failed: {e}")

            return True

        except Exception as e:
            self.logger.exception(f"Standard labeling execution failed: {e}")
            return False

    async def _validate_input_data(self, data: pd.DataFrame) -> bool:
        """Validate input data using ml_common validation utilities."""
        try:
            # Use the validation suite for comprehensive data validation
            validation_result = self.validation_suite.data_validator.validate_dataframe(
                data, validation_level="comprehensive"
            )
            return validation_result['passed']
        except ValidationError:
            return False

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute labeling step with comprehensive validation and monitoring."""
        start_time = time.time()

        try:
            self.logger.info('🏷️ Starting labeling step with comprehensive validation...')

            # Extract parameters
            symbol = training_input.get('symbol', 'ETHUSDT')
            exchange = training_input.get('exchange', 'BINANCE')
            timeframe = training_input.get('timeframe', '1m')
            data_dir = training_input.get('data_dir', 'data')
            force_rerun = training_input.get('force_rerun', False)

            # Use validation suite for comprehensive validation
            config_with_step = dict(self.config)
            config_with_step['step_name'] = 'step05_labeling'

            data = pipeline_state.get('dataframe') or pipeline_state.get('validated_data')

            # Pre-execution validation using validation suite
            await self.validation_suite.validate_step_execution(config_with_step, data)

            # Execute labeling with timeout protection
            timeout_seconds = self.config.get('timeout_seconds', 1800)  # 30 minutes default

            execution_coro = self.execute_labeling(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                data_dir=data_dir,
                force_rerun=force_rerun
            )

            success = await self.validation_suite.execution_validator.execute_with_timeout(
                execution_coro, timeout_seconds, "labeling"
            )

            execution_time = time.time() - start_time

            # Prepare result
            result = {
                'success': success,
                'step_name': 'step05_labeling',
                'execution_time': execution_time,
                'message': 'Labeling completed successfully' if success else 'Labeling failed',
                'ml_common_utilities_used': True
            }

            # Post-execution validation and metrics
            if success:
                labeled_data = pipeline_state.get('labeled_data')
                if labeled_data is not None:
                    # Use result validator for quality assessment
                    result_validation = self.validation_suite.result_validator.validate_labeling_results(labeled_data)
                    result['result_validation'] = result_validation

                    # Performance monitoring
                    if self.memory_optimizer is not None:
                        try:
                            memory_stats = self.memory_optimizer.memory_usage_monitoring('step05_labeling')
                            result['memory_stats'] = memory_stats
                        except Exception as e:
                            self.logger.debug(f"Memory monitoring failed: {e}")

                    # Performance metrics
                    result['performance_metrics'] = {
                        'execution_time_seconds': execution_time,
                        'samples_processed': len(labeled_data),
                        'processing_rate': len(labeled_data) / execution_time if execution_time > 0 else 0
                    }

            self.logger.info(f"🏷️ Labeling step completed in {execution_time:.2f}s")
            return result

        except ValidationError as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ FAST FAIL: Validation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'step_name': 'step05_labeling',
                'execution_time': execution_time,
                'error_type': e.error_type,
                'error_details': e.details
            }
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.exception(f'❌ Labeling step failed after {execution_time:.2f}s: {e}')

            return {
                'success': False,
                'error': str(e),
                'step_name': 'step05_labeling',
                'execution_time': execution_time
            }


__all__ = ['LabelingStep', 'ensure_directory', '_build_labeled_data_path']