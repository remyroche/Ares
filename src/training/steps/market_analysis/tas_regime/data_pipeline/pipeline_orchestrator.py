"""
Data Pipeline Orchestrator for TAS

Comprehensive orchestration system for the entire data pipeline including
data ingestion, preprocessing, feature engineering, regime detection, and storage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import pipeline components
from .data_ingestion import DataIngestionManager, DataIngestionConfig
from .data_preprocessing import DataPreprocessor, PreprocessingConfig
from .feature_engineering import FeatureEngineer, FeatureConfig
from .regime_detection import RegimeDetector, RegimeConfig
from .data_validation import DataValidator, ValidationConfig
from .data_storage import DataStorageManager, StorageConfig, StorageResult

logger = logging.getLogger(__name__)

class PipelineStage(Enum):
    """Pipeline stages."""
    INGESTION = "ingestion"
    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    REGIME_DETECTION = "regime_detection"
    VALIDATION = "validation"
    STORAGE = "storage"

class PipelineStatus(Enum):
    """Pipeline status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class PipelineConfig:
    """Configuration for data pipeline orchestrator."""

    # Pipeline stages
    enable_ingestion: bool = True
    enable_preprocessing: bool = True
    enable_feature_engineering: bool = True
    enable_regime_detection: bool = True
    enable_validation: bool = True
    enable_storage: bool = True

    # Pipeline options
    parallel_processing: bool = True
    max_workers: int = 4
    chunk_size: int = 1000
    enable_checkpointing: bool = True
    checkpoint_interval: int = 100

    # Data options
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT"])
    timeframes: List[str] = field(default_factory=lambda: ["1h", "4h", "1d"])
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Component configurations
    ingestion_config: DataIngestionConfig = field(default_factory=DataIngestionConfig)
    preprocessing_config: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    feature_engineering_config: FeatureConfig = field(default_factory=FeatureConfig)
    regime_detection_config: RegimeConfig = field(default_factory=RegimeConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    storage_config: StorageConfig = field(default_factory=StorageConfig)

    # Output configuration
    save_pipeline_info: bool = True
    output_directory: str = "pipeline_info"

@dataclass
class PipelineStageResult:
    """Result of a pipeline stage."""

    # Stage information
    stage: PipelineStage
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None

    # Data information
    data_shape: Optional[Tuple[int, int]] = None
    data_columns: Optional[List[str]] = None
    data_types: Optional[Dict[str, str]] = None

    # Stage-specific results
    stage_result: Optional[Any] = None
    stage_metadata: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None

    # Error information
    error_message: Optional[str] = None
    error_traceback: Optional[str] = None

    # Warnings
    warnings: List[str] = field(default_factory=list)

@dataclass
class PipelineResult:
    """Result of the entire data pipeline."""

    # Pipeline information
    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration: Optional[float] = None

    # Stage results
    stage_results: Dict[PipelineStage, PipelineStageResult] = field(default_factory=dict)

    # Data information
    final_data_shape: Optional[Tuple[int, int]] = None
    final_data_columns: Optional[List[str]] = None
    final_data_types: Optional[Dict[str, str]] = None

    # Pipeline metadata
    pipeline_metadata: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    total_memory_usage_mb: Optional[float] = None
    total_cpu_usage_percent: Optional[float] = None

    # Success metrics
    success_rate: float = 0.0
    failure_rate: float = 0.0

    # Error information
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Configuration
    config: PipelineConfig = field(default_factory=PipelineConfig)

class DataPipelineOrchestrator:
    """
    Comprehensive data pipeline orchestrator for TAS.

    Orchestrates the entire data pipeline from ingestion to storage
    for tree architecture search.
    """

    def __init__(self, config: PipelineConfig):
        """Initialize data pipeline orchestrator.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize pipeline components
        self._initialize_components()

        # Initialize pipeline state
        self.pipeline_id = f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_data = None
        self.stage_results = {}

        self.logger.info("✅ Data Pipeline Orchestrator initialized")
        self.logger.info(f"📊 Pipeline ID: {self.pipeline_id}")
        self.logger.info(f"📊 Symbols: {config.symbols}")
        self.logger.info(f"📊 Timeframes: {config.timeframes}")
        self.logger.info(f"📊 Parallel processing: {config.parallel_processing}")

    def _initialize_components(self):
        """Initialize pipeline components."""
        try:
            # Initialize data ingestion
            if self.config.enable_ingestion:
                self.data_ingestor = DataIngestionManager(self.config.ingestion_config)
                self.logger.info("✅ Data ingestion component initialized")

            # Initialize data preprocessing
            if self.config.enable_preprocessing:
                self.data_preprocessor = DataPreprocessor(self.config.preprocessing_config)
                self.logger.info("✅ Data preprocessing component initialized")

            # Initialize feature engineering
            if self.config.enable_feature_engineering:
                self.feature_engineer = FeatureEngineer(self.config.feature_engineering_config)
                self.logger.info("✅ Feature engineering component initialized")

            # Initialize regime detection
            if self.config.enable_regime_detection:
                self.regime_detector = RegimeDetector(self.config.regime_detection_config)
                self.logger.info("✅ Regime detection component initialized")

            # Initialize data validation
            if self.config.enable_validation:
                self.data_validator = DataValidator(self.config.validation_config)
                self.logger.info("✅ Data validation component initialized")

            # Initialize data storage
            if self.config.enable_storage:
                self.data_storage = DataStorageManager(self.config.storage_config)
                self.logger.info("✅ Data storage component initialized")

        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
            raise

    def run_pipeline(self, symbols: Optional[List[str]] = None,
                    timeframes: Optional[List[str]] = None,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None) -> PipelineResult:
        """
        Run the complete data pipeline.

        Args:
            symbols: Trading symbols to process
            timeframes: Data timeframes to process
            start_date: Start date for data
            end_date: End date for data

        Returns:
            Pipeline result
        """
        self.logger.info("🚀 Starting data pipeline")
        start_time = datetime.now()

        try:
            # Use provided parameters or config defaults
            symbols = symbols or self.config.symbols
            timeframes = timeframes or self.config.timeframes
            start_date = start_date or self.config.start_date
            end_date = end_date or self.config.end_date

            # Initialize pipeline result
            pipeline_result = PipelineResult(
                pipeline_id=self.pipeline_id,
                start_time=start_time,
                config=self.config
            )

            # Process each symbol and timeframe combination
            for symbol in symbols:
                for timeframe in timeframes:
                    self.logger.info(f"🔄 Processing {symbol} {timeframe}")

                    try:
                        # Run pipeline for this symbol/timeframe
                        symbol_result = self._run_symbol_pipeline(
                            symbol, timeframe, start_date, end_date
                        )

                        # Update pipeline result
                        pipeline_result.stage_results.update(symbol_result.stage_results)

                    except Exception as e:
                        self.logger.error(f"❌ Pipeline failed for {symbol} {timeframe}: {e}")
                        pipeline_result.errors.append(f"{symbol} {timeframe}: {str(e)}")
                        continue

            # Calculate final metrics
            pipeline_result.end_time = datetime.now()
            pipeline_result.total_duration = (pipeline_result.end_time - pipeline_result.start_time).total_seconds()
            pipeline_result.success_rate = self._calculate_success_rate(pipeline_result.stage_results)
            pipeline_result.failure_rate = 1.0 - pipeline_result.success_rate

            # Save pipeline info if configured
            if self.config.save_pipeline_info:
                self._save_pipeline_info(pipeline_result)

            self.logger.info(f"✅ Data pipeline completed in {pipeline_result.total_duration:.2f}s")
            self.logger.info(f"📊 Success rate: {pipeline_result.success_rate:.2%}")
            self.logger.info(f"📊 Failure rate: {pipeline_result.failure_rate:.2%}")

            return pipeline_result

        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed: {e}")
            raise

    def _run_symbol_pipeline(self, symbol: str, timeframe: str,
                           start_date: Optional[datetime],
                           end_date: Optional[datetime]) -> PipelineResult:
        """Run pipeline for a specific symbol and timeframe."""
        try:
            # Initialize symbol result
            symbol_result = PipelineResult(
                pipeline_id=f"{self.pipeline_id}_{symbol}_{timeframe}",
                start_time=datetime.now(),
                config=self.config
            )

            # Stage 1: Data Ingestion
            if self.config.enable_ingestion:
                ingestion_result = self._run_ingestion_stage(symbol, timeframe, start_date, end_date)
                symbol_result.stage_results[PipelineStage.INGESTION] = ingestion_result

                if ingestion_result.status == PipelineStatus.FAILED:
                    return symbol_result

                self.current_data = ingestion_result.stage_result

            # Stage 2: Data Preprocessing
            if self.config.enable_preprocessing and self.current_data is not None:
                preprocessing_result = self._run_preprocessing_stage()
                symbol_result.stage_results[PipelineStage.PREPROCESSING] = preprocessing_result

                if preprocessing_result.status == PipelineStatus.FAILED:
                    return symbol_result

                self.current_data = preprocessing_result.stage_result

            # Stage 3: Feature Engineering
            if self.config.enable_feature_engineering and self.current_data is not None:
                feature_result = self._run_feature_engineering_stage()
                symbol_result.stage_results[PipelineStage.FEATURE_ENGINEERING] = feature_result

                if feature_result.status == PipelineStatus.FAILED:
                    return symbol_result

                self.current_data = feature_result.stage_result

            # Stage 4: Regime Detection
            if self.config.enable_regime_detection and self.current_data is not None:
                regime_result = self._run_regime_detection_stage()
                symbol_result.stage_results[PipelineStage.REGIME_DETECTION] = regime_result

                if regime_result.status == PipelineStatus.FAILED:
                    return symbol_result

                self.current_data = regime_result.stage_result

            # Stage 5: Data Validation
            if self.config.enable_validation and self.current_data is not None:
                validation_result = self._run_validation_stage()
                symbol_result.stage_results[PipelineStage.VALIDATION] = validation_result

                if validation_result.status == PipelineStatus.FAILED:
                    return symbol_result

            # Stage 6: Data Storage
            if self.config.enable_storage and self.current_data is not None:
                storage_result = self._run_storage_stage(symbol, timeframe)
                symbol_result.stage_results[PipelineStage.STORAGE] = storage_result

                if storage_result.status == PipelineStatus.FAILED:
                    return symbol_result

            # Update final data information
            if self.current_data is not None:
                symbol_result.final_data_shape = self.current_data.shape
                symbol_result.final_data_columns = list(self.current_data.columns)
                symbol_result.final_data_types = self.current_data.dtypes.to_dict()

            symbol_result.end_time = datetime.now()
            symbol_result.total_duration = (symbol_result.end_time - symbol_result.start_time).total_seconds()

            return symbol_result

        except Exception as e:
            self.logger.error(f"❌ Symbol pipeline failed: {e}")
            raise

    def _run_ingestion_stage(self, symbol: str, timeframe: str,
                           start_date: Optional[datetime],
                           end_date: Optional[datetime]) -> PipelineStageResult:
        """Run data ingestion stage."""
        self.logger.info(f"🔄 Running ingestion stage for {symbol} {timeframe}")
        start_time = datetime.now()

        try:
            # Load historical data
            data = self.data_ingestor.load_historical_data(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            # Create stage result
            result = PipelineStageResult(
                stage=PipelineStage.INGESTION,
                status=PipelineStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                data_shape=data.shape,
                data_columns=list(data.columns),
                data_types=data.dtypes.to_dict(),
                stage_result=data,
                stage_metadata={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'start_date': start_date,
                    'end_date': end_date
                }
            )

            self.logger.info(f"✅ Ingestion stage completed in {result.duration:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Ingestion stage failed: {e}")
            return PipelineStageResult(
                stage=PipelineStage.INGESTION,
                status=PipelineStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                error_message=str(e)
            )

    def _run_preprocessing_stage(self) -> PipelineStageResult:
        """Run data preprocessing stage."""
        self.logger.info("🔄 Running preprocessing stage")
        start_time = datetime.now()

        try:
            # Preprocess data
            processed_data = self.data_preprocessor.preprocess_data(self.current_data)

            # Create stage result
            result = PipelineStageResult(
                stage=PipelineStage.PREPROCESSING,
                status=PipelineStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                data_shape=processed_data.shape,
                data_columns=list(processed_data.columns),
                data_types=processed_data.dtypes.to_dict(),
                stage_result=processed_data,
                stage_metadata={
                    'original_shape': self.current_data.shape,
                    'processed_shape': processed_data.shape
                }
            )

            self.logger.info(f"✅ Preprocessing stage completed in {result.duration:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Preprocessing stage failed: {e}")
            return PipelineStageResult(
                stage=PipelineStage.PREPROCESSING,
                status=PipelineStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                error_message=str(e)
            )

    def _run_feature_engineering_stage(self) -> PipelineStageResult:
        """Run feature engineering stage."""
        self.logger.info("🔄 Running feature engineering stage")
        start_time = datetime.now()

        try:
            # Generate features
            features_data = self.feature_engineer.generate_features(self.current_data)

            # Create stage result
            result = PipelineStageResult(
                stage=PipelineStage.FEATURE_ENGINEERING,
                status=PipelineStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                data_shape=features_data.shape,
                data_columns=list(features_data.columns),
                data_types=features_data.dtypes.to_dict(),
                stage_result=features_data,
                stage_metadata={
                    'original_shape': self.current_data.shape,
                    'features_shape': features_data.shape,
                    'feature_count': len(features_data.columns)
                }
            )

            self.logger.info(f"✅ Feature engineering stage completed in {result.duration:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Feature engineering stage failed: {e}")
            return PipelineStageResult(
                stage=PipelineStage.FEATURE_ENGINEERING,
                status=PipelineStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                error_message=str(e)
            )

    def _run_regime_detection_stage(self) -> PipelineStageResult:
        """Run regime detection stage."""
        self.logger.info("🔄 Running regime detection stage")
        start_time = datetime.now()

        try:
            # Detect and mark regimes
            regime_data = self.regime_detector.detect_and_mark_regimes(self.current_data)

            # Create stage result
            result = PipelineStageResult(
                stage=PipelineStage.REGIME_DETECTION,
                status=PipelineStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                data_shape=regime_data.shape,
                data_columns=list(regime_data.columns),
                data_types=regime_data.dtypes.to_dict(),
                stage_result=regime_data,
                stage_metadata={
                    'original_shape': self.current_data.shape,
                    'regime_data_shape': regime_data.shape,
                    'regime_columns': [col for col in regime_data.columns if 'regime' in col.lower()]
                }
            )

            self.logger.info(f"✅ Regime detection stage completed in {result.duration:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Regime detection stage failed: {e}")
            return PipelineStageResult(
                stage=PipelineStage.REGIME_DETECTION,
                status=PipelineStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                error_message=str(e)
            )

    def _run_validation_stage(self) -> PipelineStageResult:
        """Run data validation stage."""
        self.logger.info("🔄 Running validation stage")
        start_time = datetime.now()

        try:
            # Validate data
            validation_result = self.data_validator.validate_data(self.current_data)

            # Create stage result
            result = PipelineStageResult(
                stage=PipelineStage.VALIDATION,
                status=PipelineStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                data_shape=self.current_data.shape,
                data_columns=list(self.current_data.columns),
                data_types=self.current_data.dtypes.to_dict(),
                stage_result=validation_result,
                stage_metadata={
                    'validation_passed': validation_result.get('validation_passed', False),
                    'validation_score': validation_result.get('validation_score', 0.0)
                }
            )

            self.logger.info(f"✅ Validation stage completed in {result.duration:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Validation stage failed: {e}")
            return PipelineStageResult(
                stage=PipelineStage.VALIDATION,
                status=PipelineStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                error_message=str(e)
            )

    def _run_storage_stage(self, symbol: str, timeframe: str) -> PipelineStageResult:
        """Run data storage stage."""
        self.logger.info(f"🔄 Running storage stage for {symbol} {timeframe}")
        start_time = datetime.now()

        try:
            # Store data
            storage_result = self.data_storage.store_data(
                data=self.current_data,
                data_type="processed_with_regimes",
                symbol=symbol,
                timeframe=timeframe
            )

            # Create stage result
            result = PipelineStageResult(
                stage=PipelineStage.STORAGE,
                status=PipelineStatus.COMPLETED,
                start_time=start_time,
                end_time=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                data_shape=self.current_data.shape,
                data_columns=list(self.current_data.columns),
                data_types=self.current_data.dtypes.to_dict(),
                stage_result=storage_result,
                stage_metadata={
                    'storage_path': storage_result.storage_path,
                    'storage_size_mb': storage_result.storage_size_mb,
                    'compression_ratio': storage_result.compression_ratio
                }
            )

            self.logger.info(f"✅ Storage stage completed in {result.duration:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"❌ Storage stage failed: {e}")
            return PipelineStageResult(
                stage=PipelineStage.STORAGE,
                status=PipelineStatus.FAILED,
                start_time=start_time,
                end_time=datetime.now(),
                duration=(datetime.now() - start_time).total_seconds(),
                error_message=str(e)
            )

    def _calculate_success_rate(self, stage_results: Dict[PipelineStage, PipelineStageResult]) -> float:
        """Calculate pipeline success rate."""
        try:
            if not stage_results:
                return 0.0

            successful_stages = sum(1 for result in stage_results.values()
                                  if result.status == PipelineStatus.COMPLETED)
            total_stages = len(stage_results)

            return successful_stages / total_stages if total_stages > 0 else 0.0

        except Exception as e:
            self.logger.warning(f"⚠️ Success rate calculation failed: {e}")
            return 0.0

    def _save_pipeline_info(self, result: PipelineResult):
        """Save pipeline information."""
        try:
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pipeline_info_{timestamp}.json"
            filepath = output_dir / filename

            pipeline_info = {
                'pipeline_id': result.pipeline_id,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'total_duration': result.total_duration,
                'final_data_shape': result.final_data_shape,
                'final_data_columns': result.final_data_columns,
                'final_data_types': result.final_data_types,
                'pipeline_metadata': result.pipeline_metadata,
                'total_memory_usage_mb': result.total_memory_usage_mb,
                'total_cpu_usage_percent': result.total_cpu_usage_percent,
                'success_rate': result.success_rate,
                'failure_rate': result.failure_rate,
                'errors': result.errors,
                'warnings': result.warnings,
                'stage_results': {
                    stage.value: {
                        'status': result.status.value,
                        'start_time': result.start_time.isoformat(),
                        'end_time': result.end_time.isoformat() if result.end_time else None,
                        'duration': result.duration,
                        'data_shape': result.data_shape,
                        'data_columns': result.data_columns,
                        'data_types': result.data_types,
                        'stage_metadata': result.stage_metadata,
                        'memory_usage_mb': result.memory_usage_mb,
                        'cpu_usage_percent': result.cpu_usage_percent,
                        'error_message': result.error_message,
                        'warnings': result.warnings
                    }
                    for stage, result in result.stage_results.items()
                }
            }

            with open(filepath, 'w') as f:
                json.dump(pipeline_info, f, indent=2, default=str)

            self.logger.info(f"📁 Pipeline info saved to {filepath}")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save pipeline info: {e}")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        try:
            return {
                'pipeline_id': self.pipeline_id,
                'current_data_shape': self.current_data.shape if self.current_data is not None else None,
                'current_data_columns': list(self.current_data.columns) if self.current_data is not None else None,
                'stage_results_count': len(self.stage_results),
                'stage_results': {
                    stage.value: {
                        'status': result.status.value,
                        'duration': result.duration,
                        'data_shape': result.data_shape
                    }
                    for stage, result in self.stage_results.items()
                }
            }
        except Exception as e:
            self.logger.warning(f"⚠️ Pipeline status retrieval failed: {e}")
            return {}

    def export_pipeline_result(self, result: PipelineResult, filepath: str):
        """Export pipeline result to file."""
        try:
            pipeline_info = {
                'pipeline_id': result.pipeline_id,
                'start_time': result.start_time.isoformat(),
                'end_time': result.end_time.isoformat() if result.end_time else None,
                'total_duration': result.total_duration,
                'final_data_shape': result.final_data_shape,
                'final_data_columns': result.final_data_columns,
                'final_data_types': result.final_data_types,
                'pipeline_metadata': result.pipeline_metadata,
                'success_rate': result.success_rate,
                'failure_rate': result.failure_rate,
                'errors': result.errors,
                'warnings': result.warnings
            }

            with open(filepath, 'w') as f:
                json.dump(pipeline_info, f, indent=2, default=str)

            self.logger.info(f"📁 Pipeline result exported to {filepath}")

        except Exception as e:
            self.logger.error(f"❌ Failed to export pipeline result: {e}")
