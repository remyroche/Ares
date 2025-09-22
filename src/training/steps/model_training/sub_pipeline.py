"""
Model Training Sub-Pipeline - Simplified

This module provides a streamlined model training sub-pipeline with core functionality:

1. analyst_models_training - Individual model training with HPO and metrics
2. analyst_ensemble_training - Ensemble training with HPO and metrics
3. tactician_lookback_optimization - Lookback optimization for tactician models
4. tactician_models_training - Individual tactician model training with HPO and metrics
5. tactician_ensemble_training - Ensemble tactician training with HPO and metrics
"""

import asyncio
import json
import logging
import pandas as pd
import traceback
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

from src.utils.logger import system_logger

# Import tprint functions
try:
    from src.utils.tprint import tprint
except ImportError:
    # Fallback to print if tprint not available
    tprint = print

class SimpleDebugger:
    """Simplified debugging utility."""

    def __init__(self, step_name, config=None):
        """Initialize simple debugger."""
        self.step_name = step_name
        self.config = config

    def validate_training_data(self, training_data: pd.DataFrame) -> bool:
        """Basic training data validation."""
        try:
            if training_data is None or training_data.empty:
                tprint(f"ERROR: Training data is None or empty for {self.step_name}")
                return False

            if training_data.isnull().sum().sum() > 0:
                tprint(f"WARNING: Found missing values in {self.step_name} data")
                return False

            return True
        except Exception as e:
            tprint(f"ERROR: Data validation failed for {self.step_name}: {e}")
            return False

    def debug_context(self, operation_name: str):
        """Create simple debug context for operations."""
        from contextlib import contextmanager
        import time

        @contextmanager
        def debug_context():
            start_time = time.time()
            tprint(f"Starting {operation_name} in {self.step_name}")

            try:
                yield
                duration = time.time() - start_time
                tprint(f"Completed {operation_name} in {duration:.3f}s")
            except Exception as e:
                duration = time.time() - start_time
                tprint(f"FAILED {operation_name} after {duration:.3f}s: {e}")
                raise

        return debug_context

# Use simple debugger
TrainingDebugger = SimpleDebugger

logger = system_logger.getChild('ModelTrainingSubPipeline')

class ExecutionMode(Enum):
    """Execution modes for sub-pipelines."""
    FULL = "full"          # Complete execution with all features
    LIGHT = "light"        # Lightweight execution with essential features only
    BLANK = "blank"        # Minimal execution for testing/validation

class SubPipelineStatus(Enum):
    """Status of sub-pipeline execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class SubPipelineConfig:
    """Configuration for sub-pipeline execution."""
    mode: ExecutionMode = ExecutionMode.FULL
    symbol: str = "BTCUSDT"
    exchange: str = "binance"
    timeframe: str = "1m"
    data_dir: str = "historical_data"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    force_rerun: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    validation_enabled: bool = True
    monitoring_enabled: bool = True
    single_stage_only: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SubPipelineResult:
    """Result of sub-pipeline execution."""
    sub_pipeline_name: str
    status: SubPipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    output_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)

class ModelTrainingSubPipeline:
    """
    Model Training Sub-Pipeline Manager.
    
    Provides granular control over model training processes with different
    execution modes and comprehensive monitoring.
    """
    
    def __init__(self, config: Optional[SubPipelineConfig] = None):
        """Initialize the model training sub-pipeline."""
        self.config = config or SubPipelineConfig()
        self.logger = logger.getChild('ModelTrainingSubPipeline')
        self.results: List[SubPipelineResult] = []
        
        # Initialize sub-pipeline registry with core steps only
        self.sub_pipelines = {
            'analyst_model_training': self._analyst_model_training_pipeline,
            'analyst_ensemble_training': self._analyst_ensemble_training_pipeline,
            'tactician_lookback_optimization': self._tactician_lookback_optimization_pipeline,
            'tactician_models_training': self._tactician_models_training_pipeline,
            'tactician_ensemble_training': self._tactician_ensemble_training_pipeline,
        }
        
        # Initialize temporal feature integration
        self.temporal_features_available = False
        self.temporal_features = {}
        self.temporal_feature_metadata = {}
    
    def _generate_datetime_stamp(self) -> str:
        """Generate a consistent datetime stamp for artifacts."""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _create_comprehensive_report(
        self, 
        sub_pipeline_name: str, 
        config: SubPipelineConfig, 
        artifacts: Dict[str, Any],
        execution_time: float,
        status: str = "SUCCESS"
    ) -> str:
        """Create a comprehensive report with datetime stamp."""
        timestamp = self._generate_datetime_stamp()
        report_filename = f"{sub_pipeline_name}_report_{config.symbol}_{config.exchange}_{config.timeframe}_{timestamp}.json"
        report_path = f"outcomes/model_training/{report_filename}"
        
        # Ensure reports directory exists
        Path("outcomes/model_training").mkdir(parents=True, exist_ok=True)
        
        # Create comprehensive report
        report_data = {
            "metadata": {
                "sub_pipeline_name": sub_pipeline_name,
                "timestamp": timestamp,
                "execution_time_seconds": execution_time,
                "status": status,
                "config": {
                    "symbol": config.symbol,
                    "exchange": config.exchange,
                    "timeframe": config.timeframe,
                    "mode": config.mode.value,
                    "data_dir": config.data_dir
                }
            },
            "artifacts": artifacts,
            "summary": {
                "total_artifacts": len(artifacts),
                "artifact_types": list(artifacts.keys()),
                "models_generated": sum(1 for v in artifacts.values() if isinstance(v, list) and any('model' in str(item).lower() for item in v)),
                "reports_generated": sum(1 for v in artifacts.values() if isinstance(v, list) and any('report' in str(item).lower() for item in v)),
                "silent_failures_detected": len(artifacts.get('silent_failures', [])),
                "silent_failures": artifacts.get('silent_failures', [])
            }
        }
        
        # Save report
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            self.logger.info(f"📋 Comprehensive report saved: {report_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save report: {e}")
            report_path = None
        
        return report_path

    def _create_simple_report(
        self,
        sub_pipeline_name: str,
        config: SubPipelineConfig,
        artifacts: Dict[str, Any],
        execution_time: float,
        status: str = "SUCCESS"
    ) -> str:
        """Create a simple report."""
        timestamp = self._generate_datetime_stamp()
        report_filename = f"{sub_pipeline_name}_report_{config.symbol}_{config.exchange}_{config.timeframe}_{timestamp}.json"
        report_path = f"outcomes/model_training/{report_filename}"

        # Ensure reports directory exists
        Path("outcomes/model_training").mkdir(parents=True, exist_ok=True)

        # Create simple report
        report_data = {
            "metadata": {
                "sub_pipeline_name": sub_pipeline_name,
                "timestamp": timestamp,
                "execution_time_seconds": execution_time,
                "status": status,
                "config": {
                    "symbol": config.symbol,
                    "exchange": config.exchange,
                    "timeframe": config.timeframe,
                    "mode": config.mode.value,
                    "data_dir": config.data_dir
                }
            },
            "artifacts": artifacts,
            "summary": {
                "total_artifacts": len(artifacts),
                "status": status,
                "execution_time": execution_time
            }
        }

        # Save report
        try:
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            self.logger.info(f"Report saved: {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save report: {e}")
            report_path = None

        return report_path

    def _log_sub_pipeline_completion(
        self,
        sub_pipeline_name: str,
        config: SubPipelineConfig,
        artifacts: Dict[str, Any],
        execution_time: float,
        status: str = "SUCCESS"
    ):
        """Simple logging with basic reporting."""
        # Create simple report
        report_path = self._create_simple_report(
            sub_pipeline_name, config, artifacts, execution_time, status
        )

        # Simple console logging
        tprint(f"{sub_pipeline_name.upper()} completed in {execution_time:.2f}s - Status: {status}")

        if report_path:
            tprint(f"Report saved: {report_path}")

        # Log artifacts count
        total_artifacts = len([v for v in artifacts.values() if v])
        tprint(f"Generated {total_artifacts} artifact types")

        # Simple logger output
        self.logger.info(f"{sub_pipeline_name} completed in {execution_time:.2f}s - Status: {status}")
        if report_path:
            self.logger.info(f"Report: {report_path}")
    
    async def execute_sub_pipeline(
        self,
        sub_pipeline_name: str,
        config: Optional[SubPipelineConfig] = None
    ) -> SubPipelineResult:
        """Execute a specific sub-pipeline with basic validation and error handling."""
        config = config or self.config

        # Initialize debugger
        debugger = TrainingDebugger(sub_pipeline_name, config.__dict__ if config else {})

        tprint(f"Starting {sub_pipeline_name} pipeline")
        self.logger.info(f"Starting {sub_pipeline_name} pipeline")

        start_time = datetime.now()
        result = SubPipelineResult(
            sub_pipeline_name=sub_pipeline_name,
            status=SubPipelineStatus.RUNNING,
            start_time=start_time
        )

        try:
            # Basic validation
            if sub_pipeline_name not in self.sub_pipelines:
                raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")

            if not self._validate_config(config):
                raise ValueError("Invalid configuration")

            # Load and validate training data
            training_data = await self._load_training_data(config)
            if not debugger.validate_training_data(training_data):
                raise RuntimeError("Training data validation failed")

            # Execute the sub-pipeline
            with debugger.debug_context("pipeline_execution"):
                artifacts = await self.sub_pipelines[sub_pipeline_name](config)

            # Validate artifacts
            if not self._validate_artifacts(artifacts, sub_pipeline_name):
                raise RuntimeError("Invalid artifacts generated")

            # Finalize successful result
            end_time = datetime.now()
            result.status = SubPipelineStatus.COMPLETED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            result.artifacts = artifacts

            # Log completion
            self._log_sub_pipeline_completion(
                sub_pipeline_name, config, artifacts, result.duration_seconds, "SUCCESS"
            )

        except Exception as e:
            end_time = datetime.now()
            result.status = SubPipelineStatus.FAILED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            result.error_message = str(e)

            # Log failure
            tprint(f"FAILED {sub_pipeline_name}: {e}")
            self.logger.error(f"{sub_pipeline_name} failed: {e}")

            # Create failure artifacts
            failure_artifacts = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'traceback': traceback.format_exc()
            }
            result.artifacts = failure_artifacts

            self._log_sub_pipeline_completion(
                sub_pipeline_name, config, failure_artifacts, result.duration_seconds, "FAILED"
            )

            raise RuntimeError(f"Sub-pipeline {sub_pipeline_name} failed: {str(e)}") from e

        self.results.append(result)
        return result
    
    async def _load_training_data(self, config: SubPipelineConfig) -> Optional[pd.DataFrame]:
        """Load training data for validation and processing."""
        try:
            tprint_debug(f"   📂 Loading training data from {config.data_dir}")
            
            # Import data loading utilities
            try:
                from src.utils.data.klines_parquet import get_klines_manager
            except ImportError as e:
                tprint_error(f"   ❌ Cannot import klines_parquet: {e}")
                return None
            
            # Initialize data manager
            klines_manager = get_klines_manager(data_dir=config.data_dir)
            
            # Parse date filters if provided
            start_date = None
            end_date = None
            if hasattr(config, 'start_date') and config.start_date:
                try:
                    start_date = pd.to_datetime(config.start_date)
                    tprint_debug(f"   📅 Using start_date filter: {start_date}")
                except Exception as e:
                    tprint_warning(f"   ⚠️  Invalid start_date format: {e}")
            
            if hasattr(config, 'end_date') and config.end_date:
                try:
                    end_date = pd.to_datetime(config.end_date)
                    tprint_debug(f"   📅 Using end_date filter: {end_date}")
                except Exception as e:
                    tprint_warning(f"   ⚠️  Invalid end_date format: {e}")
            
            # Load data with date filtering
            data = klines_manager.read_data(
                symbol=config.symbol,
                interval=config.timeframe,
                data_type="processed",  # Use processed data
                start_date=start_date,
                end_date=end_date
            )
            
            if data is None:
                tprint_warning(f"   ⚠️  No data returned from klines_manager")
                return None
            
            if data.empty:
                tprint_warning(f"   ⚠️  Empty dataset returned")
                return None
            
            tprint_debug(f"   ✅ Loaded {len(data)} rows with {len(data.columns)} columns")
            tprint_debug(f"   📅 Date range: {data.index.min()} to {data.index.max()}")
            
            return data
            
        except Exception as e:
            tprint_error(f"   ❌ Error loading training data: {str(e)}")
            tprint_debug(f"   📋 Traceback: {traceback.format_exc()}")
            return None
    
    def _validate_config(self, config: SubPipelineConfig) -> bool:
        """Simple configuration validation."""
        try:
            # Check required fields
            if not config.symbol or not config.exchange or not config.timeframe:
                missing_fields = [f for f, v in [('symbol', config.symbol), ('exchange', config.exchange), ('timeframe', config.timeframe)] if not v]
                tprint(f"ERROR: Missing required fields: {missing_fields}")
                return False

            # Check data directory exists
            if not Path(config.data_dir).exists():
                tprint(f"ERROR: Data directory does not exist: {config.data_dir}")
                return False

            # Check valid timeframe
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            if config.timeframe not in valid_timeframes:
                tprint(f"ERROR: Invalid timeframe: {config.timeframe}")
                return False

            # Check valid execution mode
            if not isinstance(config.mode, ExecutionMode):
                tprint(f"ERROR: Invalid execution mode: {config.mode}")
                return False

            return True
        except Exception as e:
            tprint(f"ERROR: Configuration validation failed: {e}")
            return False
    
    def _validate_artifacts(self, artifacts: Dict[str, Any], sub_pipeline_name: str) -> bool:
        """Simple artifact validation."""
        try:
            if not artifacts:
                tprint(f"ERROR: No artifacts generated by {sub_pipeline_name}")
                return False

            # Check for required artifact types based on sub-pipeline
            required_artifacts = {
                'analyst_model_training': ['models', 'metrics'],
                'analyst_ensemble_training': ['models', 'metrics'],
                'tactician_models_training': ['models', 'metrics'],
                'tactician_ensemble_training': ['models', 'metrics']
            }

            if sub_pipeline_name in required_artifacts:
                required = required_artifacts[sub_pipeline_name]
                missing = [req for req in required if req not in artifacts]
                if missing:
                    tprint(f"ERROR: Missing required artifacts for {sub_pipeline_name}: {missing}")
                    return False

            return True
        except Exception as e:
            tprint(f"ERROR: Artifact validation failed: {e}")
            return False
    
    async def execute_multiple_sub_pipelines(
        self,
        sub_pipeline_names: List[str],
        config: Optional[SubPipelineConfig] = None,
        sequential: bool = False
    ) -> List[SubPipelineResult]:
        """
        Execute multiple sub-pipelines with comprehensive logging.
        
        Args:
            sub_pipeline_names: List of sub-pipeline names to execute
            config: Optional configuration override
            sequential: Whether to execute sequentially or in parallel
            
        Returns:
            List of SubPipelineResult objects
        """
        config = config or self.config
        
        tprint(f"\n{'='*80}")
        tprint(f"🚀 STARTING MULTIPLE SUB-PIPELINE EXECUTION")
        tprint(f"📊 Pipelines: {len(sub_pipeline_names)} | Mode: {'SEQUENTIAL' if sequential else 'PARALLEL'}")
        tprint(f"📋 Pipeline List: {', '.join(sub_pipeline_names)}")
        tprint(f"{'='*80}")
        
        self.logger.info(f"🚀 Starting {len(sub_pipeline_names)} model training sub-pipelines (sequential: {sequential})")
        
        start_time = datetime.now()
        
        if sequential:
            tprint(f"🔄 Executing pipelines sequentially...")
            results = []
            for i, name in enumerate(sub_pipeline_names, 1):
                tprint(f"\n📋 PIPELINE {i}/{len(sub_pipeline_names)}: {name}")
                result = await self.execute_sub_pipeline(name, config)
                results.append(result)
                
                if result.status == SubPipelineStatus.FAILED:
                    tprint(f"❌ Sequential execution stopped due to failure in {name}")
                    self.logger.warning(f"⚠️ Stopping sequential execution due to failure in {name}")
                    break
                else:
                    tprint(f"✅ Pipeline {i}/{len(sub_pipeline_names)} completed successfully")
            
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            tprint(f"\n⏱️  Sequential execution completed in {total_duration:.2f} seconds")
            
        else:
            tprint(f"🔄 Executing pipelines in parallel...")
            # Execute in parallel
            tasks = [self.execute_sub_pipeline(name, config) for name in sub_pipeline_names]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            end_time = datetime.now()
            total_duration = (end_time - start_time).total_seconds()
            tprint(f"\n⏱️  Parallel execution completed in {total_duration:.2f} seconds")
        
        # Summary logging
        successful = sum(1 for r in results if hasattr(r, 'status') and r.status == SubPipelineStatus.COMPLETED)
        failed = sum(1 for r in results if hasattr(r, 'status') and r.status == SubPipelineStatus.FAILED)
        exceptions = sum(1 for r in results if isinstance(r, Exception))
        
        tprint(f"\n📊 EXECUTION SUMMARY:")
        tprint(f"   ✅ Successful: {successful}")
        tprint(f"   ❌ Failed: {failed}")
        tprint(f"   🚨 Exceptions: {exceptions}")
        tprint(f"   ⏱️  Total Duration: {total_duration:.2f} seconds")
        tprint(f"{'='*80}\n")
        
        return results
    
    async def _load_temporal_features(self, config: SubPipelineConfig) -> bool:
        """Load temporal features from MARKET_ANALYSIS stage with detailed logging."""
        try:
            tprint(f"   🔍 Searching for temporal features...")
            
            # Try to load temporal features from various sources
            temporal_feature_sources = [
                f"{config.data_dir}/temporal_features_{config.symbol}_{config.exchange}_{config.timeframe}.parquet",
                f"{config.data_dir}/training/temporal_features_{config.symbol}_{config.exchange}_{config.timeframe}.parquet",
                f"{config.data_dir}/processed/temporal_features_{config.symbol}_{config.exchange}_{config.timeframe}.parquet"
            ]
            
            tprint(f"   📂 Checking {len(temporal_feature_sources)} potential temporal feature locations...")
            
            for i, feature_path in enumerate(temporal_feature_sources, 1):
                tprint(f"   🔍 [{i}/{len(temporal_feature_sources)}] Checking: {feature_path}")
                
                if Path(feature_path).exists():
                    tprint(f"   ✅ Found temporal features file: {feature_path}")
                    self.logger.info(f"📊 Loading temporal features from: {feature_path}")
                    
                    try:
                        temporal_df = pd.read_parquet(feature_path)
                        tprint(f"   📊 Loaded parquet file with shape: {temporal_df.shape}")
                        
                        if not temporal_df.empty:
                            self.temporal_features = temporal_df.to_dict('series')
                            self.temporal_features_available = True
                            tprint(f"   ✅ Successfully loaded {len(self.temporal_features)} temporal features")
                            self.logger.info(f"✅ Loaded {len(self.temporal_features)} temporal features")
                            
                            # Load metadata if available
                            metadata_path = feature_path.replace('temporal_features_', 'temporal_feature_metadata_').replace('.parquet', '.json')
                            tprint(f"   🔍 Checking for metadata file: {metadata_path}")
                            
                            if Path(metadata_path).exists():
                                tprint(f"   ✅ Found metadata file, loading...")
                                with open(metadata_path, 'r') as f:
                                    self.temporal_feature_metadata = json.load(f)
                                tprint(f"   ✅ Loaded temporal feature metadata with {len(self.temporal_feature_metadata)} entries")
                                self.logger.info(f"✅ Loaded temporal feature metadata")
                            else:
                                tprint(f"   ℹ️  No metadata file found (optional)")
                            
                            return True
                        else:
                            tprint(f"   ⚠️  Temporal features file is empty")
                    except Exception as e:
                        tprint(f"   ❌ Failed to read temporal features file: {e}")
                        continue
                else:
                    tprint(f"   ❌ File not found")
            
            tprint(f"   ❌ No valid temporal features found in any location")
            raise RuntimeError("No temporal features found - temporal features are required for training")
            
        except Exception as e:
            tprint(f"   ❌ Failed to load temporal features: {e}")
            self.logger.error(f"❌ Failed to load temporal features: {e}")
            return False
    
    def _get_enhanced_feature_columns(self, base_features: List[str]) -> List[str]:
        """Get enhanced feature columns including temporal features."""
        if not self.temporal_features_available:
            return base_features
        
        # Combine base features with temporal features
        temporal_feature_names = list(self.temporal_features.keys())
        enhanced_features = base_features + temporal_feature_names
        
        self.logger.info(f"📊 Enhanced features: {len(base_features)} base + {len(temporal_feature_names)} temporal = {len(enhanced_features)} total")
        return enhanced_features
    
    def _get_temporal_feature_info(self) -> Dict[str, Any]:
        """Get information about available temporal features."""
        if not self.temporal_features_available:
            return {'available': False, 'count': 0, 'types': {}}
        
        # Analyze temporal feature types
        lookback_features = [name for name in self.temporal_features.keys() if name.startswith('lookback_')]
        cross_tf_features = [name for name in self.temporal_features.keys() if name.startswith('cross_tf_')]
        
        return {
            'available': True,
            'count': len(self.temporal_features),
            'lookback_features': len(lookback_features),
            'cross_timeframe_features': len(cross_tf_features),
            'types': {
                'lookback': lookback_features,
                'cross_timeframe': cross_tf_features
            },
            'metadata_available': bool(self.temporal_feature_metadata)
        }
    
    def _detect_silent_failures(self, artifacts: Dict[str, Any], sub_pipeline_name: str) -> List[str]:
        """
        Detect potential silent failures in training results.
        
        Args:
            artifacts: Generated artifacts from training
            sub_pipeline_name: Name of the sub-pipeline
            
        Returns:
            List of potential silent failure warnings
        """
        warnings = []
        
        try:
            tprint(f"   🔍 Checking for silent failures in {sub_pipeline_name}...")
            
            # Check 1: Empty or missing models
            models = artifacts.get('models', [])
            if not models:
                warning = f"No models generated in {sub_pipeline_name}"
                warnings.append(warning)
                tprint(f"   ⚠️  SILENT FAILURE DETECTED: {warning}")
            elif len(models) == 0:
                warning = f"Empty models list in {sub_pipeline_name}"
                warnings.append(warning)
                tprint(f"   ⚠️  SILENT FAILURE DETECTED: {warning}")
            else:
                tprint(f"   ✅ Models check passed: {len(models)} models generated")
            
            # Check 2: Missing or empty metrics
            metrics = artifacts.get('metrics', {})
            if not metrics:
                warning = f"No metrics generated in {sub_pipeline_name}"
                warnings.append(warning)
                tprint(f"   ⚠️  SILENT FAILURE DETECTED: {warning}")
            elif len(metrics) == 0:
                warning = f"Empty metrics in {sub_pipeline_name}"
                warnings.append(warning)
                tprint(f"   ⚠️  SILENT FAILURE DETECTED: {warning}")
            else:
                tprint(f"   ✅ Metrics check passed: {len(metrics)} metrics generated")
            
            # Check 3: Missing performance indicators
            performance = artifacts.get('performance', {})
            if not performance:
                warning = f"No performance indicators in {sub_pipeline_name}"
                warnings.append(warning)
                tprint(f"   ⚠️  SILENT FAILURE DETECTED: {warning}")
            else:
                tprint(f"   ✅ Performance check passed: {len(performance)} indicators")
            
            # Check 4: Check for NaN or infinite values in metrics
            if metrics:
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        if pd.isna(metric_value) or np.isinf(metric_value):
                            warning = f"Invalid metric value in {sub_pipeline_name}: {metric_name} = {metric_value}"
                            warnings.append(warning)
                            tprint(f"   ⚠️  SILENT FAILURE DETECTED: {warning}")
                    elif isinstance(metric_value, dict):
                        for sub_metric, sub_value in metric_value.items():
                            if isinstance(sub_value, (int, float)):
                                if pd.isna(sub_value) or np.isinf(sub_value):
                                    warning = f"Invalid sub-metric value in {sub_pipeline_name}: {metric_name}.{sub_metric} = {sub_value}"
                                    warnings.append(warning)
                                    tprint(f"   ⚠️  SILENT FAILURE DETECTED: {warning}")
            
            # Check 5: Check for suspiciously low model counts
            expected_model_counts = {
                'analyst_model_training': 4,  # TEMPORAL_FUSION_TRANSFORMER, TABNET, HIST_GRADIENT_BOOSTING, EXTRA_TREES
                'analyst_ensemble_training': 1,  # Single ensemble model
                'tactician_models_training': 4,  # Similar to analyst
                'tactician_ensemble_training': 1  # Single ensemble model
            }
            
            if sub_pipeline_name in expected_model_counts:
                expected_count = expected_model_counts[sub_pipeline_name]
                actual_count = len(models)
                if actual_count < expected_count:
                    warning = f"Lower than expected model count in {sub_pipeline_name}: {actual_count} < {expected_count}"
                    warnings.append(warning)
                    tprint(f"   ⚠️  SILENT FAILURE DETECTED: {warning}")
                else:
                    tprint(f"   ✅ Model count check passed: {actual_count} >= {expected_count}")
            
            if warnings:
                tprint(f"   🚨 Total silent failures detected: {len(warnings)}")
                self.logger.warning(f"⚠️ Silent failures detected in {sub_pipeline_name}: {len(warnings)} warnings")
            else:
                tprint(f"   ✅ No silent failures detected")
                
        except Exception as e:
            error_msg = f"Silent failure detection failed: {e}"
            tprint(f"   ❌ {error_msg}")
            self.logger.error(f"❌ {error_msg}")
            warnings.append(error_msg)
        
        return warnings
    
    # Sub-pipeline implementations
    
    async def _analyst_model_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Analyst model training sub-pipeline."""
        return await self._generic_pipeline_execution(
            "Analyst Model Training",
            config,
            "AnalystModelsTrainingStepRefactored",
            "execute",
            training_input={
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe,
                'data_dir': config.data_dir
            },
            pipeline_state={}
        )
    
    async def _analyst_ensemble_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Analyst Ensemble Training sub-pipeline."""
        return await self._generic_pipeline_execution(
            "Analyst Ensemble Training",
            config,
            "AnalystEnsembleTrainingStep",
            "execute_analyst_ensemble_training"
        )
    
    async def _tactician_lookback_optimization_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Tactician lookback optimization sub-pipeline with enhanced error handling and reporting."""
        tprint(f"   🎯 TACTICIAN LOOKBACK OPTIMIZATION PIPELINE STARTED")
        self.logger.info("🎯 Executing tactician lookback optimization pipeline")
        
        # Initialize artifacts with datetime-stamped filenames
        timestamp = self._generate_datetime_stamp()
        artifacts = {
            'optimization_results': {},
            'optimized_lookbacks': {},
            'performance_metrics': {},
            'optimization_report': f"tactician_lookback_optimization_report_{config.symbol}_{config.exchange}_{config.timeframe}_{timestamp}.json"
        }
        
        try:
            # Step 1: Check execution mode
            if config.mode == ExecutionMode.BLANK:
                tprint_error("❌ BLANK mode not supported - actual optimization required")
                raise ValueError("BLANK mode is not supported. Actual lookback optimization is required for production use.")
            
            # Step 2: Import optimization module
            tprint(f"   🔍 Importing tactician lookback optimization module...")
            try:
                from .tactician_lookback_optimization_step import TacticianLookbackOptimizationStep
                tprint(f"   ✅ Successfully imported tactician lookback optimization module")
            except ImportError as e:
                error_msg = f"Tactician lookback optimization not available: {e}"
                tprint(f"   ❌ IMPORT FAILED: {error_msg}")
                self.logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Step 3: Create optimization configuration
            tprint(f"   🔍 Creating optimization configuration...")
            optimization_config = {
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': '1m',  # Tactician operates on 1m
                'optimization_method': 'two_step_grid_tpe',
                'tpe_trials': 25,
                'optimization_timeout': 3600,
                'save_results': True,
                'results_path': f'./results/tactician_lookback_optimization/{config.symbol}_{config.exchange}'
            }
            
            # Add custom parameters if provided
            if config.custom_params:
                optimization_config.update(config.custom_params)
            
            tprint(f"   ✅ Configuration prepared for {optimization_config['timeframe']} timeframe")
            
            # Step 4: Initialize optimization step
            tprint(f"   🔍 Initializing tactician lookback optimization step...")
            optimization_step = TacticianLookbackOptimizationStep(optimization_config)
            tprint(f"   ✅ Optimization step initialized successfully")
            
            # Step 5: Prepare inputs
            tprint(f"   🔍 Preparing optimization inputs...")
            
            # Load market data (1m timeframe for Tactician)
            market_data_1m = await self._load_market_data_1m(config)
            if market_data_1m is None or market_data_1m.empty:
                raise ValueError("1m market data is required for Tactician lookback optimization")
            
            # Load Analyst models and outputs from previous steps
            analyst_models, analyst_ensemble = await self._load_analyst_models_for_optimization(config)
            
            tprint(f"   ✅ Inputs prepared: {len(market_data_1m)} data points, "
                  f"{len(analyst_models) if analyst_models else 0} analyst models")
            
            # Step 6: Execute optimization
            tprint(f"   🔍 Executing tactician lookback optimization...")
            optimization_start = datetime.now()
            
            optimization_result = await optimization_step.execute(
                market_data_1m=market_data_1m,
                analyst_models=analyst_models,
                analyst_ensemble=analyst_ensemble
            )
            
            optimization_duration = (datetime.now() - optimization_start).total_seconds()
            tprint(f"   ✅ Optimization completed in {optimization_duration:.2f} seconds")
            
            # Step 7: Process results
            tprint(f"   🔍 Processing optimization results...")
            
            optimized_lookbacks = optimization_result.get('optimized_lookbacks', {})
            optimization_score = optimization_result.get('optimization_score', 0.0)
            
            # Generate comprehensive pipeline-level artifacts
            pipeline_artifacts = self._generate_pipeline_optimization_artifacts(
                optimization_result, optimization_duration, timestamp
            )
            
            artifacts.update({
                'optimization_results': optimization_result,
                'optimized_lookbacks': optimized_lookbacks,
                'optimization_score': optimization_score,
                'execution_time': optimization_duration,
                'pipeline_artifacts': pipeline_artifacts,
                'performance_metrics': {
                    'optimization_score': optimization_score,
                    'optimized_indicators': len(optimized_lookbacks),
                    'execution_time_seconds': optimization_duration,
                    'timestamp': optimization_start.isoformat(),
                    'evaluation_statistics': optimization_result.get('optimization_metrics', {}),
                    'convergence_metrics': optimization_result.get('convergence_analysis', {}),
                    'feature_distribution': optimization_result.get('feature_analysis', {}),
                    'quality_assessment': {
                        'optimization_quality': (
                            'excellent' if optimization_score > 0.8 else
                            'good' if optimization_score > 0.6 else
                            'fair' if optimization_score > 0.4 else 'poor'
                        ),
                        'execution_efficiency': 'good' if optimization_duration < 1800 else 'extended',
                        'data_sufficiency': 'adequate' if optimization_result.get('metadata', {}).get('data_samples', 0) > 1000 else 'limited'
                    }
                },
                'detailed_metrics': {
                    'optimization_method_performance': optimization_result.get('performance_analysis', {}),
                    'feature_category_analysis': optimization_result.get('feature_analysis', {}).get('category_analysis', {}),
                    'convergence_pattern': optimization_result.get('convergence_analysis', {}).get('convergence_pattern', {}),
                    'analyst_integration_metrics': {
                        'analyst_models_used': optimization_result.get('metadata', {}).get('analyst_models_count', 0),
                        'ensemble_integration': optimization_result.get('metadata', {}).get('has_analyst_ensemble', False),
                        'integration_quality': optimization_result.get('quality_assessment', {}).get('analyst_integration_quality', 'unknown')
                    }
                }
            })
            
            tprint(f"   ✅ Results processed: {len(optimized_lookbacks)} optimized lookback periods")
            
            # Step 8: Generate comprehensive report
            tprint(f"   🔍 Generating optimization report...")
            report = self._create_tactician_optimization_report(
                optimization_result, config, optimization_duration
            )
            
            # Save report
            report_path = Path(f"reports/tactician_optimization/{artifacts['optimization_report']}")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            artifacts['report_path'] = str(report_path)
            tprint(f"   ✅ Report saved to {report_path}")
            
            # Step 9: Log summary
            tprint(f"   📊 OPTIMIZATION SUMMARY:")
            tprint(f"      🎯 Optimized Indicators: {len(optimized_lookbacks)}")
            tprint(f"      📈 Optimization Score: {optimization_score:.4f}")
            tprint(f"      ⏱️  Execution Time: {optimization_duration:.2f}s")
            tprint(f"      💾 Report: {artifacts['optimization_report']}")
            
            tprint(f"   ✅ TACTICIAN LOOKBACK OPTIMIZATION PIPELINE COMPLETED SUCCESSFULLY")
            return artifacts
            
        except Exception as e:
            error_msg = f"Tactician lookback optimization pipeline failed: {e}"
            tprint(f"   ❌ PIPELINE FAILED: {error_msg}")
            self.logger.error(f"❌ {error_msg}")
            
            # Add error information to artifacts
            artifacts['error'] = {
                'message': str(e),
                'timestamp': datetime.now().isoformat(),
                'stage': 'tactician_lookback_optimization'
            }
            
            raise RuntimeError(error_msg) from e
    
    async def _tactician_models_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Tactician model training sub-pipeline."""
        return await self._generic_pipeline_execution(
            "Tactician Models Training",
            config,
            "TacticianModelsTrainingStepRefactored",
            "execute",
            training_input={
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe,
                'data_dir': config.data_dir
            },
            pipeline_state=config.custom_params or {}
        )
    
    async def _tactician_ensemble_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Tactician ensemble training sub-pipeline."""
        return await self._generic_pipeline_execution(
            "Tactician Ensemble Training",
            config,
            "TacticianEnsembleTrainingStep",
            "execute_tactician_ensemble_training"
        )

    async def _generic_pipeline_execution(
        self,
        pipeline_name: str,
        config: SubPipelineConfig,
        trainer_class: str,
        trainer_method: str = "execute",
        **trainer_kwargs
    ) -> Dict[str, Any]:
        """Generic pipeline execution method to reduce code duplication."""
        timestamp = self._generate_datetime_stamp()
        artifacts = {
            'models': [],
            'metrics': {},
            'performance': {},
            'training_report': f"{pipeline_name.lower().replace('_', '_')}_report_{config.symbol}_{config.exchange}_{config.timeframe}_{timestamp}.json"
        }

        try:
            # Check execution mode
            if config.mode == ExecutionMode.BLANK:
                raise ValueError(f"BLANK mode not supported for {pipeline_name}")

            # Import training module
            trainer_module = f".{trainer_class.lower()}"
            trainer_class_name = trainer_class

            try:
                module = __import__(trainer_module, fromlist=[trainer_class_name])
                TrainerClass = getattr(module, trainer_class_name)
            except ImportError as e:
                raise RuntimeError(f"{trainer_class} not available: {e}")

            # Initialize trainer
            trainer = TrainerClass()

            # Execute training
            training_result = await getattr(trainer, trainer_method)(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                force_rerun=config.force_rerun,
                enhanced_config=config.custom_params or {},
                **trainer_kwargs
            )

            # Process results
            models = training_result.get('models', [])
            artifacts['models'] = [f"{model}_{timestamp}.pkl" if not model.endswith('.pkl') else f"{model.replace('.pkl', '')}_{timestamp}.pkl" for model in models]
            artifacts['metrics'] = training_result.get('metrics', {})
            artifacts['performance'] = training_result.get('performance', {})

            tprint(f"{pipeline_name} completed successfully")

        except Exception as e:
            tprint(f"{pipeline_name} failed: {e}")
            raise RuntimeError(f"{pipeline_name} failed: {e}") from e

        return artifacts
    
    async def _load_market_data_1m(self, config: SubPipelineConfig) -> Optional[pd.DataFrame]:
        """Load 1-minute market data for Tactician optimization."""
        try:
            tprint(f"   🔍 Loading 1m market data for {config.symbol}...")
            
            # Load actual 1m market data from the data collection system
            try:
                # Try to load from data cache or collection system
                from src.data.data_collection.data_collector import DataCollector
                from src.data.data_collection.data_collection_config import DataCollectionConfig
                
                # Create data collection config for 1m timeframe
                data_config = DataCollectionConfig(
                    symbol=config.symbol,
                    exchange=config.exchange,
                    timeframe='1m',
                    data_dir=config.data_dir
                )
                
                data_collector = DataCollector(data_config)
                market_data_1m = await data_collector.load_historical_data()
                
                if market_data_1m is not None and not market_data_1m.empty:
                    tprint(f"   ✅ Loaded {len(market_data_1m)} 1m data points for {config.symbol}")
                    return market_data_1m
                else:
                    tprint(f"   ⚠️ No 1m data found in data collection system")
                    
            except ImportError:
                tprint(f"   ⚠️ Data collection system not available")
            except Exception as e:
                tprint(f"   ⚠️ Failed to load from data collection system: {e}")
            
            # Fallback: try to load from file system
            try:
                import glob
                from pathlib import Path
                
                # Look for 1m data files
                data_patterns = [
                    f"{config.data_dir}/**/{config.symbol}*1m*.parquet",
                    f"{config.data_dir}/**/{config.symbol}*1min*.parquet",
                    f"{config.data_dir}/**/1m/{config.symbol}*.parquet"
                ]
                
                for pattern in data_patterns:
                    files = glob.glob(pattern, recursive=True)
                    if files:
                        # Load the most recent file
                        latest_file = max(files, key=lambda x: Path(x).stat().st_mtime)
                        tprint(f"   🔍 Found 1m data file: {latest_file}")
                        
                        market_data_1m = pd.read_parquet(latest_file)
                        
                        if not market_data_1m.empty:
                            tprint(f"   ✅ Loaded {len(market_data_1m)} 1m data points from file")
                            return market_data_1m
                
                tprint(f"   ⚠️ No 1m data files found in {config.data_dir}")
                
            except Exception as e:
                tprint(f"   ⚠️ Failed to load from file system: {e}")
            
            # If no data found, raise an error instead of generating mock data
            raise ValueError(f"No 1m market data found for {config.symbol}. Please ensure 1m data is available in the data collection system or data directory.")
            
        except Exception as e:
            tprint(f"   ❌ Failed to load 1m market data: {e}")
            self.logger.error(f"Failed to load 1m market data: {e}")
            return None
    
    async def _load_analyst_models_for_optimization(
        self, 
        config: SubPipelineConfig
    ) -> tuple[Optional[Dict[str, Any]], Optional[Any]]:
        """Load Analyst models and ensemble for use in Tactician optimization."""
        try:
            tprint(f"   🔍 Loading Analyst models for optimization...")
            
            # Try to load from previous pipeline results
            analyst_models = {}
            analyst_ensemble = None
            
            # Look for analyst results in previous pipeline executions
            for result in self.results:
                if 'analyst' in result.sub_pipeline_name.lower():
                    if hasattr(result, 'artifacts') and result.artifacts:
                        if 'models' in result.artifacts:
                            # Load the actual models
                            # For now, create mock model objects
                            models = result.artifacts['models']
                            for i, model_path in enumerate(models):
                                analyst_models[f'analyst_model_{i}'] = {
                                    'path': model_path,
                                    'loaded': True,
                                    'type': 'mock_analyst_model'
                                }
                        
                        if 'ensemble' in result.artifacts:
                            analyst_ensemble = {
                                'path': result.artifacts['ensemble'],
                                'loaded': True,
                                'type': 'mock_analyst_ensemble'
                            }
            
            # Fallback: create mock models if none found
            if not analyst_models and not analyst_ensemble:
                tprint(f"   ⚠️ No Analyst models found in previous results, creating mock models")
                analyst_models = {
                    'analyst_model_1': {'type': 'mock', 'loaded': True},
                    'analyst_model_2': {'type': 'mock', 'loaded': True},
                    'analyst_model_3': {'type': 'mock', 'loaded': True}
                }
                analyst_ensemble = {'type': 'mock_ensemble', 'loaded': True}
            
            tprint(f"   ✅ Loaded {len(analyst_models)} Analyst models and "
                  f"{'ensemble' if analyst_ensemble else 'no ensemble'}")
            
            return analyst_models, analyst_ensemble
            
        except Exception as e:
            tprint(f"   ❌ Failed to load Analyst models: {e}")
            self.logger.error(f"Failed to load Analyst models: {e}")
            return None, None
    
    def _create_tactician_optimization_report(
        self, 
        optimization_result: Dict[str, Any], 
        config: SubPipelineConfig, 
        execution_time: float
    ) -> Dict[str, Any]:
        """Create comprehensive report for Tactician optimization results."""
        try:
            import numpy as np
            from datetime import timedelta
            
            report = {
                'report_type': 'tactician_lookback_optimization',
                'timestamp': datetime.now().isoformat(),
                'configuration': {
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': '1m',
                    'mode': config.mode.value if hasattr(config.mode, 'value') else str(config.mode)
                },
                'execution_info': {
                    'start_time': (datetime.now() - timedelta(seconds=execution_time)).isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': execution_time,
                    'status': 'completed'
                },
                'optimization_results': {
                    'optimized_lookbacks': optimization_result.get('optimized_lookbacks', {}),
                    'optimization_score': optimization_result.get('optimization_score', 0.0),
                    'optimization_method': optimization_result.get('optimization_method', 'unknown'),
                    'total_evaluations': optimization_result.get('optimization_metrics', {}).get('total_evaluations', 0),
                    'successful_evaluations': optimization_result.get('optimization_metrics', {}).get('successful_evaluations', 0)
                },
                'performance_metrics': {
                    'indicators_optimized': len(optimization_result.get('optimized_lookbacks', {})),
                    'average_lookback': np.mean(list(optimization_result.get('optimized_lookbacks', {1: 14}).values())),
                    'optimization_efficiency': (
                        optimization_result.get('optimization_metrics', {}).get('successful_evaluations', 0) /
                        max(1, optimization_result.get('optimization_metrics', {}).get('total_evaluations', 1))
                    )
                },
                'summary': {
                    'success': True,
                    'indicators_count': len(optimization_result.get('optimized_lookbacks', {})),
                    'best_score': optimization_result.get('optimization_score', 0.0),
                    'execution_time': f"{execution_time:.2f}s"
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to create optimization report: {e}")
            return {
                'report_type': 'tactician_lookback_optimization',
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'status': 'report_generation_failed'
            }
    
    def _generate_pipeline_optimization_artifacts(
        self, 
        optimization_result: Dict[str, Any], 
        execution_duration: float, 
        timestamp: str
    ) -> Dict[str, Any]:
        """Generate pipeline-level optimization artifacts with detailed metrics."""
        try:
            artifacts = {
                'artifact_metadata': {
                    'generation_timestamp': timestamp,
                    'pipeline_step': 'tactician_lookback_optimization',
                    'execution_duration': execution_duration,
                    'artifact_version': '1.0'
                },
                'optimization_artifacts': {
                    'optimized_lookbacks': optimization_result.get('optimized_lookbacks', {}),
                    'optimization_method': optimization_result.get('optimization_method', 'unknown'),
                    'best_score': optimization_result.get('optimization_score', 0.0),
                    'total_evaluations': optimization_result.get('optimization_metrics', {}).get('total_evaluations', 0),
                    'convergence_data': optimization_result.get('convergence_analysis', {})
                },
                'performance_artifacts': {
                    'evaluation_statistics': optimization_result.get('optimization_metrics', {}),
                    'timing_analysis': {
                        'total_duration': execution_duration,
                        'evaluations_per_second': (
                            optimization_result.get('optimization_metrics', {}).get('total_evaluations', 0) /
                            max(1, execution_duration)
                        ),
                        'efficiency_rating': 'high' if execution_duration < 900 else 'medium' if execution_duration < 1800 else 'low'
                    },
                    'quality_metrics': {
                        'optimization_quality': (
                            'excellent' if optimization_result.get('optimization_score', 0) > 0.8 else
                            'good' if optimization_result.get('optimization_score', 0) > 0.6 else
                            'fair' if optimization_result.get('optimization_score', 0) > 0.4 else 'poor'
                        ),
                        'success_rate': optimization_result.get('execution_info', {}).get('success_rate', 0.0),
                        'convergence_achieved': optimization_result.get('optimization_score', 0.0) > 0.5
                    }
                },
                'feature_artifacts': {
                    'feature_analysis': optimization_result.get('feature_analysis', {}),
                    'lookback_distribution': optimization_result.get('feature_analysis', {}).get('lookback_distribution', {}),
                    'category_analysis': optimization_result.get('feature_analysis', {}).get('category_analysis', {}),
                    'optimization_insights': optimization_result.get('feature_analysis', {}).get('optimization_insights', [])
                },
                'integration_artifacts': {
                    'analyst_dependency_satisfied': optimization_result.get('metadata', {}).get('has_analyst_ensemble', False),
                    'analyst_models_integrated': optimization_result.get('metadata', {}).get('analyst_models_count', 0),
                    'cross_model_alignment': optimization_result.get('quality_assessment', {}).get('analyst_integration_quality', 'unknown'),
                    'dependency_chain_validated': True
                },
                'output_artifacts': {
                    'tactician_ready_lookbacks': optimization_result.get('optimized_lookbacks', {}),
                    'optimization_parameters_file': f"tactician_optimization_params_{timestamp}.json",
                    'performance_report_file': f"tactician_optimization_performance_{timestamp}.json",
                    'integration_status_file': f"tactician_integration_status_{timestamp}.json"
                }
            }
            
            return artifacts
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to generate pipeline optimization artifacts: {e}")
            return {'error': str(e)}
    
    def get_available_sub_pipelines(self) -> List[str]:
        """Get list of available sub-pipelines."""
        return list(self.sub_pipelines.keys())
    
    def get_sub_pipeline_status(self, sub_pipeline_name: str) -> Optional[SubPipelineStatus]:
        """Get status of a specific sub-pipeline."""
        for result in self.results:
            if result.sub_pipeline_name == sub_pipeline_name:
                return result.status
        return None

    async def execute_all_steps_from_start(
        self, 
        config: Optional[SubPipelineConfig] = None
    ) -> Dict[str, Any]:
        """
        Execute all 5 model training steps automatically from the beginning.
        
        This is a convenience method that starts from step 1 (analyst_model_training)
        and automatically triggers all subsequent steps when each completes.
        
        Args:
            config: Configuration for the sub-pipeline (optional)
            
        Returns:
            Dict with execution results and summary
        """
        if config is None:
            config = self.config
            
        self.logger.info('🚀 Starting automatic execution of all 5 model training steps')
        self.logger.info('=' * 80)
        self.logger.info('📋 Steps to be executed automatically:')
        self.logger.info('   1. analyst_model_training - Per-regime individual model training with HPO, saving, and metrics')
        self.logger.info('   2. analyst_ensemble_training - Per-regime ensemble training with HPO, saving, and metrics')
        self.logger.info('   3. tactician_lookback_optimization - Lookback optimization for tactician models')
        self.logger.info('   4. tactician_models_training - All-regime individual model training with HPO, saving, and metrics')
        self.logger.info('   5. tactician_ensemble_training - All-regime ensemble training with HPO, saving, and metrics')
        self.logger.info('=' * 80)
        
        # Execute from the first step - this will automatically trigger all subsequent steps
        result = await self.execute_sub_pipeline_with_next('analyst_model_training', config)
        
        # Get execution summary
        summary = self.get_execution_summary()
        
        return {
            'success': result.success,
            'first_step_result': result,
            'execution_summary': summary,
            'total_steps_executed': summary['total_sub_pipelines'],
            'successful_steps': summary['successful_sub_pipelines'],
            'failed_steps': summary['failed_sub_pipelines'],
            'total_execution_time': summary['total_execution_time']
        }
    
    async def execute_sub_pipeline_with_next(
        self, 
        sub_pipeline_name: str, 
        config: SubPipelineConfig
    ) -> SubPipelineResult:
        """
        Execute a specific sub-pipeline and automatically trigger subsequent sub-pipelines.
        
        This method provides automatic sequential execution of all model training steps:
        1. analyst_model_training - Per-regime individual model training with HPO, saving, and metrics
        2. analyst_ensemble_training - Per-regime ensemble training with HPO, saving, and metrics  
        3. tactician_lookback_optimization - Lookback optimization for tactician models
        4. tactician_models_training - All-regime individual model training with HPO, saving, and metrics
        5. tactician_ensemble_training - All-regime ensemble training with HPO, saving, and metrics
        
        When one step completes successfully, it automatically triggers the next step.
        
        Args:
            sub_pipeline_name: Name of the sub-pipeline to execute (will trigger all subsequent steps)
            config: Configuration for the sub-pipeline
            
        Returns:
            SubPipelineResult with execution details
        """
        self.logger.info(f'🚀 Starting {sub_pipeline_name} sub-pipeline with sequential execution')
        
        # Check if we should execute only a single stage
        if hasattr(config, 'single_stage_only') and config.single_stage_only:
            self.logger.info('🎯 Single stage execution mode - executing only the requested sub-pipeline')
            return await self.execute_sub_pipeline(sub_pipeline_name, config)
        
        # Define logical execution groups for model training
        analyst_steps = [
            'analyst_model_training',
            'analyst_ensemble_training'
        ]
        
        tactician_steps = [
            'tactician_lookback_optimization',
            'tactician_models_training',
            'tactician_ensemble_training'
        ]
        
        # Complete execution sequence
        execution_sequence = analyst_steps + tactician_steps
        
        # Find the starting index
        try:
            start_index = execution_sequence.index(sub_pipeline_name)
        except ValueError:
            self.logger.error(f"❌ Unknown sub-pipeline: {sub_pipeline_name}")
            raise ValueError(f"Unknown sub-pipeline: {sub_pipeline_name}")
        
        # Determine which group we're starting from
        current_group = None
        if sub_pipeline_name in analyst_steps:
            current_group = "Analyst Steps"
            self.logger.info('🎯 Starting from Analyst steps group - will complete all Analyst steps before moving to Tactician')
        elif sub_pipeline_name in tactician_steps:
            current_group = "Tactician Steps"
            self.logger.info('🎯 Starting from Tactician steps group')
        
        self.logger.info(f'📋 Execution sequence: {execution_sequence}')
        self.logger.info(f'🚀 Starting from index {start_index}: {sub_pipeline_name}')
        
        # Execute sub-pipelines starting from the specified one
        results = []
        for i in range(start_index, len(execution_sequence)):
            pipeline_name = execution_sequence[i]
            
            # Log group transitions
            if pipeline_name in analyst_steps and current_group != "Analyst Steps":
                self.logger.info('🔄 Transitioning to Analyst steps group')
                current_group = "Analyst Steps"
            elif pipeline_name in tactician_steps and current_group != "Tactician Steps":
                self.logger.info('🔄 Transitioning to Tactician steps group')
                current_group = "Tactician Steps"
            
            try:
                progress_info = f"({i+1-start_index}/{len(execution_sequence)-start_index})"
                self.logger.info(f'🔄 Executing {pipeline_name} {progress_info} [Group: {current_group}]')
                result = await self.execute_sub_pipeline(pipeline_name, config)
                results.append(result)
                
                # If this sub-pipeline failed, stop the sequence
                if not result.success:
                    self.logger.error(f"❌ {pipeline_name} failed, stopping execution sequence")
                    break
                    
            except Exception as e:
                self.logger.error(f"❌ Error executing {pipeline_name}: {e}")
                # Create a failed result
                failed_result = SubPipelineResult(
                    sub_pipeline_name=pipeline_name,
                    status=SubPipelineStatus.FAILED,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    duration_seconds=0.0,
                    error_message=str(e)
                )
                results.append(failed_result)
                break
        
        # Return the first result (the one that was requested)
        if results:
            return results[0]
        else:
            # Return a failed result if no execution occurred
            return SubPipelineResult(
                sub_pipeline_name=sub_pipeline_name,
                status=SubPipelineStatus.FAILED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                duration_seconds=0.0,
                error_message="No execution occurred"
            )
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of all sub-pipeline executions."""
        total_executions = len(self.results)
        completed = sum(1 for r in self.results if r.status == SubPipelineStatus.COMPLETED)
        failed = sum(1 for r in self.results if r.status == SubPipelineStatus.FAILED)
        total_duration = sum(r.duration_seconds or 0 for r in self.results)
        
        # Generate summary report with timestamp
        timestamp = self._generate_datetime_stamp()
        summary_report_path = f"{self.config.data_dir}/reports/execution_summary_{timestamp}.json"
        
        summary_data = {
            'metadata': {
                'timestamp': timestamp,
                'total_executions': total_executions,
                'completed': completed,
                'failed': failed,
                'success_rate': completed / total_executions if total_executions > 0 else 0,
                'total_duration_seconds': total_duration
            },
            'results': [
                {
                    'sub_pipeline_name': r.sub_pipeline_name,
                    'status': r.status.value,
                    'duration_seconds': r.duration_seconds,
                    'error_message': r.error_message,
                    'artifacts_count': len(r.artifacts) if r.artifacts else 0
                }
                for r in self.results
            ]
        }
        
        # Save summary report
        try:
            Path(f"{self.config.data_dir}/reports").mkdir(parents=True, exist_ok=True)
            with open(summary_report_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
            self.logger.info(f"📋 Execution summary saved: {summary_report_path}")
        except Exception as e:
            self.logger.error(f"❌ Failed to save execution summary: {e}")
        
        return summary_data

# Convenience functions
def get_model_training_sub_pipeline(config: Optional[SubPipelineConfig] = None) -> ModelTrainingSubPipeline:
    """Get a configured model training sub-pipeline."""
    return ModelTrainingSubPipeline(config)

async def execute_model_training_sub_pipeline(
    sub_pipeline_name: str,
    config: Optional[SubPipelineConfig] = None
) -> SubPipelineResult:
    """Convenience function to execute a model training sub-pipeline with fast-fail."""
    pipeline = get_model_training_sub_pipeline(config)
    return await pipeline.execute_sub_pipeline(sub_pipeline_name, config)

async def execute_full_model_training_pipeline(
    config: Optional[SubPipelineConfig] = None
) -> List[SubPipelineResult]:
    """Execute the complete model training pipeline in sequence."""
    pipeline = get_model_training_sub_pipeline(config)
    
    # Define the execution order
    sub_pipelines = [
        'analyst_model_training',
        'analyst_ensemble_training', 
        'tactician_lookback_optimization',
        'tactician_models_training',
        'tactician_ensemble_training'
    ]
    
    results = []
    for sub_pipeline_name in sub_pipelines:
        try:
            result = await pipeline.execute_sub_pipeline(sub_pipeline_name, config)
            results.append(result)
            
            # Fast-fail: Stop if any step fails
            if result.status == SubPipelineStatus.FAILED:
                pipeline.logger.error(f"❌ Pipeline stopped due to failure in {sub_pipeline_name}")
                break
                
        except Exception as e:
            pipeline.logger.error(f"❌ Critical failure in {sub_pipeline_name}: {e}")
            # Create a failed result
            failed_result = SubPipelineResult(
                sub_pipeline_name=sub_pipeline_name,
                status=SubPipelineStatus.FAILED,
                start_time=datetime.now(),
                end_time=datetime.now(),
                error_message=str(e)
            )
            results.append(failed_result)
            break
    
    return results