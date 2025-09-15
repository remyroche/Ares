"""
Model Training Sub-Pipeline - Cleaned and Optimized

This module provides the final model training sub-pipeline with 4 core steps:

1. analyst_models_training - Per-regime individual model training with HPO, saving, and metrics
2. analyst_ensemble_training - Per-regime ensemble training with HPO, saving, and metrics  
3. tactician_models_training - All-regime individual model training with HPO, saving, and metrics
4. tactician_ensemble_training - All-regime ensemble training with HPO, saving, and metrics

Features:
- Fast-fail error handling
- Comprehensive datetime-stamped reports
- Clean, maintainable code structure
- Proper resource management
"""

import asyncio
import json
import logging
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field

from src.utils.logger import system_logger
from src.utils.tprint import tprint

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
        report_path = f"{config.data_dir}/reports/{report_filename}"
        
        # Ensure reports directory exists
        Path(f"{config.data_dir}/reports").mkdir(parents=True, exist_ok=True)
        
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
    
    def _log_sub_pipeline_completion(
        self, 
        sub_pipeline_name: str, 
        config: SubPipelineConfig, 
        artifacts: Dict[str, Any],
        execution_time: float,
        status: str = "SUCCESS"
    ):
        """Enhanced logging with comprehensive reporting."""
        # Create comprehensive report
        report_path = self._create_comprehensive_report(
            sub_pipeline_name, config, artifacts, execution_time, status
        )
        
        # Enhanced console logging
        tprint("\n" + "="*80)
        tprint(f"🎉 {sub_pipeline_name.upper().replace('_', ' ')} SUB-PIPELINE COMPLETED!")
        tprint(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        tprint(f"📊 Status: {status}")
        tprint("="*80)
        
        if report_path:
            tprint(f"📋 Comprehensive Report: {report_path}")
        
        tprint(f"📁 Generated Artifacts:")
        
        # Log different types of artifacts with appropriate emojis
        for key, value in artifacts.items():
            if isinstance(value, list) and value:
                if 'model' in key.lower():
                    for item in value:
                        tprint(f"   🤖 {key.title()}: {config.data_dir}/models/{item}")
                elif 'report' in key.lower():
                    for item in value:
                        tprint(f"   📋 {key.title()}: {config.data_dir}/{item}")
                else:
                    for item in value:
                        tprint(f"   📊 {key.title()}: {config.data_dir}/{item}")
            elif isinstance(value, dict) and value:
                tprint(f"   📊 {key.title()}: {config.data_dir}/{key}.json")
        
        tprint(f"📊 Total Artifacts: {len(artifacts)} types generated")
        
        # Report silent failures if any
        silent_failures = artifacts.get('silent_failures', [])
        if silent_failures:
            tprint(f"⚠️  Silent Failures Detected: {len(silent_failures)}")
            for i, failure in enumerate(silent_failures, 1):
                tprint(f"   {i}. {failure}")
        else:
            tprint(f"✅ No Silent Failures Detected")
        
        tprint("="*80 + "\n")
        
        # Enhanced logger output
        self.logger.info(f"🎉 {sub_pipeline_name.upper().replace('_', ' ')} SUB-PIPELINE COMPLETED!")
        self.logger.info(f"⏱️  Execution Time: {execution_time:.2f} seconds")
        self.logger.info(f"📊 Status: {status}")
        if report_path:
            self.logger.info(f"📋 Comprehensive Report: {report_path}")
        self.logger.info(f"📊 Total Artifacts: {len(artifacts)} types generated")
        if silent_failures:
            self.logger.warning(f"⚠️ Silent Failures: {len(silent_failures)} detected")
    
    async def execute_sub_pipeline(
        self,
        sub_pipeline_name: str,
        config: Optional[SubPipelineConfig] = None
    ) -> SubPipelineResult:
        """
        Execute a specific sub-pipeline with fast-fail error handling and comprehensive logging.
        
        Args:
            sub_pipeline_name: Name of the sub-pipeline to execute
            config: Optional configuration override
            
        Returns:
            SubPipelineResult with execution details
            
        Raises:
            ValueError: If sub-pipeline name is invalid
            RuntimeError: If execution fails (fast-fail)
        """
        config = config or self.config
        
        # Step 1: Initialization and validation
        tprint(f"\n{'='*80}")
        tprint(f"🚀 STARTING MODEL TRAINING SUB-PIPELINE: {sub_pipeline_name.upper()}")
        tprint(f"📊 Mode: {config.mode.value.upper()} | Symbol: {config.symbol} | Exchange: {config.exchange} | Timeframe: {config.timeframe}")
        tprint(f"{'='*80}")
        
        self.logger.info(f"🚀 Starting model training sub-pipeline: {sub_pipeline_name} (mode: {config.mode.value})")
        
        start_time = datetime.now()
        result = SubPipelineResult(
            sub_pipeline_name=sub_pipeline_name,
            status=SubPipelineStatus.RUNNING,
            start_time=start_time
        )
        
        try:
            # Step 2: Validate sub-pipeline exists
            tprint(f"🔍 STEP 1/6: Validating sub-pipeline name...")
            if sub_pipeline_name not in self.sub_pipelines:
                error_msg = f"Unknown sub-pipeline: {sub_pipeline_name}. Available: {list(self.sub_pipelines.keys())}"
                tprint(f"❌ VALIDATION FAILED: {error_msg}")
                self.logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            tprint(f"✅ Sub-pipeline '{sub_pipeline_name}' validated successfully")
            
            # Step 3: Validate configuration
            tprint(f"🔍 STEP 2/6: Validating configuration...")
            if not self._validate_config(config):
                error_msg = "Invalid configuration provided"
                tprint(f"❌ CONFIGURATION VALIDATION FAILED: {error_msg}")
                self.logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            tprint(f"✅ Configuration validated successfully")
            
            # Step 4: Execute the sub-pipeline
            tprint(f"🔍 STEP 3/6: Executing {sub_pipeline_name} pipeline...")
            pipeline_func = self.sub_pipelines[sub_pipeline_name]
            
            # Log execution start with timing
            execution_start = datetime.now()
            tprint(f"⏱️  Pipeline execution started at: {execution_start.strftime('%H:%M:%S')}")
            
            artifacts = await pipeline_func(config)
            
            execution_end = datetime.now()
            execution_duration = (execution_end - execution_start).total_seconds()
            tprint(f"⏱️  Pipeline execution completed in: {execution_duration:.2f} seconds")
            
            # Step 5: Validate artifacts
            tprint(f"🔍 STEP 4/6: Validating generated artifacts...")
            if not self._validate_artifacts(artifacts, sub_pipeline_name):
                error_msg = f"Invalid artifacts generated by {sub_pipeline_name}"
                tprint(f"❌ ARTIFACT VALIDATION FAILED: {error_msg}")
                self.logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg)
            tprint(f"✅ Artifacts validated successfully - {len(artifacts)} artifact types generated")
            
            # Step 6: Detect silent failures
            tprint(f"🔍 STEP 5/6: Detecting silent failures...")
            silent_failures = self._detect_silent_failures(artifacts, sub_pipeline_name)
            if silent_failures:
                tprint(f"⚠️  {len(silent_failures)} silent failures detected - see warnings above")
                # Add silent failures to artifacts for reporting
                artifacts['silent_failures'] = silent_failures
            else:
                tprint(f"✅ No silent failures detected")
            
            # Step 7: Finalize result
            tprint(f"🔍 STEP 6/6: Finalizing execution results...")
            end_time = datetime.now()
            result.status = SubPipelineStatus.COMPLETED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            result.artifacts = artifacts
            result.metadata = {
                'mode': config.mode.value,
                'symbol': config.symbol,
                'exchange': config.exchange,
                'timeframe': config.timeframe,
                'execution_duration_seconds': execution_duration
            }
            
            tprint(f"✅ All steps completed successfully!")
            
            # Enhanced logging with comprehensive reporting
            self._log_sub_pipeline_completion(
                sub_pipeline_name, config, artifacts, result.duration_seconds, "SUCCESS"
            )
            
        except Exception as e:
            end_time = datetime.now()
            result.status = SubPipelineStatus.FAILED
            result.end_time = end_time
            result.duration_seconds = (end_time - start_time).total_seconds()
            result.error_message = str(e)
            
            # Enhanced error logging with step information
            tprint(f"\n❌ SUB-PIPELINE EXECUTION FAILED!")
            tprint(f"⏱️  Failed after: {result.duration_seconds:.2f} seconds")
            tprint(f"🚨 Error: {e}")
            tprint(f"{'='*80}\n")
            
            self.logger.error(f"❌ Model training sub-pipeline {sub_pipeline_name} FAILED after {result.duration_seconds:.2f}s")
            self.logger.error(f"❌ Error: {e}")
            
            # Create failure report
            self._log_sub_pipeline_completion(
                sub_pipeline_name, config, {}, result.duration_seconds, "FAILED"
            )
            
            # Fast-fail: Re-raise the exception
            raise RuntimeError(f"Sub-pipeline {sub_pipeline_name} failed: {e}") from e
        
        self.results.append(result)
        return result
    
    def _validate_config(self, config: SubPipelineConfig) -> bool:
        """Validate configuration parameters with detailed logging."""
        try:
            tprint(f"   🔍 Checking required fields...")
            # Check required fields
            if not config.symbol or not config.exchange or not config.timeframe:
                missing_fields = []
                if not config.symbol:
                    missing_fields.append("symbol")
                if not config.exchange:
                    missing_fields.append("exchange")
                if not config.timeframe:
                    missing_fields.append("timeframe")
                tprint(f"   ❌ Missing required fields: {missing_fields}")
                self.logger.error(f"❌ Missing required config fields: {missing_fields}")
                return False
            tprint(f"   ✅ Required fields present: symbol={config.symbol}, exchange={config.exchange}, timeframe={config.timeframe}")
            
            tprint(f"   🔍 Checking data directory...")
            # Check data directory exists
            if not Path(config.data_dir).exists():
                tprint(f"   ❌ Data directory does not exist: {config.data_dir}")
                self.logger.error(f"❌ Data directory does not exist: {config.data_dir}")
                return False
            tprint(f"   ✅ Data directory exists: {config.data_dir}")
            
            tprint(f"   🔍 Validating timeframe...")
            # Check valid timeframe
            valid_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
            if config.timeframe not in valid_timeframes:
                tprint(f"   ❌ Invalid timeframe: {config.timeframe}. Valid options: {valid_timeframes}")
                self.logger.error(f"❌ Invalid timeframe: {config.timeframe}. Valid: {valid_timeframes}")
                return False
            tprint(f"   ✅ Timeframe is valid: {config.timeframe}")
            
            tprint(f"   🔍 Checking execution mode...")
            # Check valid execution mode
            if not isinstance(config.mode, ExecutionMode):
                tprint(f"   ❌ Invalid execution mode: {config.mode}")
                self.logger.error(f"❌ Invalid execution mode: {config.mode}")
                return False
            tprint(f"   ✅ Execution mode is valid: {config.mode.value}")
            
            return True
        except Exception as e:
            tprint(f"   ❌ Configuration validation failed with exception: {e}")
            self.logger.error(f"❌ Config validation failed: {e}")
            return False
    
    def _validate_artifacts(self, artifacts: Dict[str, Any], sub_pipeline_name: str) -> bool:
        """Validate generated artifacts with detailed logging."""
        try:
            tprint(f"   🔍 Checking if artifacts were generated...")
            if not artifacts:
                tprint(f"   ❌ No artifacts generated by {sub_pipeline_name}")
                self.logger.error(f"❌ No artifacts generated by {sub_pipeline_name}")
                return False
            tprint(f"   ✅ {len(artifacts)} artifact types generated")
            
            tprint(f"   🔍 Checking artifact types...")
            for artifact_type, artifact_value in artifacts.items():
                if isinstance(artifact_value, list):
                    tprint(f"   📊 {artifact_type}: {len(artifact_value)} items")
                elif isinstance(artifact_value, dict):
                    tprint(f"   📊 {artifact_type}: {len(artifact_value)} keys")
                else:
                    tprint(f"   📊 {artifact_type}: {type(artifact_value).__name__}")
            
            # Check for required artifact types based on sub-pipeline
            tprint(f"   🔍 Validating required artifacts for {sub_pipeline_name}...")
            required_artifacts = {
                'analyst_model_training': ['models', 'metrics'],
                'analyst_ensemble_training': ['models', 'metrics'],
                'tactician_models_training': ['models', 'metrics'],
                'tactician_ensemble_training': ['models', 'metrics']
            }
            
            if sub_pipeline_name in required_artifacts:
                required = required_artifacts[sub_pipeline_name]
                tprint(f"   📋 Required artifacts: {required}")
                
                missing = [req for req in required if req not in artifacts]
                if missing:
                    tprint(f"   ❌ Missing required artifacts: {missing}")
                    self.logger.error(f"❌ Missing required artifacts for {sub_pipeline_name}: {missing}")
                    return False
                
                # Check if required artifacts have content
                for req_artifact in required:
                    if req_artifact in artifacts:
                        artifact_content = artifacts[req_artifact]
                        if isinstance(artifact_content, list) and len(artifact_content) == 0:
                            tprint(f"   ⚠️  Warning: {req_artifact} is empty list")
                        elif isinstance(artifact_content, dict) and len(artifact_content) == 0:
                            tprint(f"   ⚠️  Warning: {req_artifact} is empty dict")
                        else:
                            tprint(f"   ✅ {req_artifact} has content")
                
                tprint(f"   ✅ All required artifacts present and validated")
            else:
                tprint(f"   ℹ️  No specific requirements defined for {sub_pipeline_name}")
            
            return True
        except Exception as e:
            tprint(f"   ❌ Artifact validation failed with exception: {e}")
            self.logger.error(f"❌ Artifact validation failed: {e}")
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
        """Analyst model training sub-pipeline with enhanced error handling and reporting."""
        tprint(f"   📊 ANALYST MODEL TRAINING PIPELINE STARTED")
        self.logger.info("📊 Executing analyst model training pipeline")
        
        # Initialize artifacts with datetime-stamped filenames
        timestamp = self._generate_datetime_stamp()
        artifacts = {
            'models': [],
            'metrics': {},
            'performance': {},
            'temporal_features_used': False,
            'temporal_feature_info': {},
            'training_report': f"analyst_models_training_report_{config.symbol}_{config.exchange}_{config.timeframe}_{timestamp}.json"
        }
        
        try:
            # Step 1: Load temporal features
            tprint(f"   🔍 Loading temporal features from MARKET_ANALYSIS stage...")
            temporal_loaded = await self._load_temporal_features(config)
            if temporal_loaded:
                temporal_info = self._get_temporal_feature_info()
                artifacts['temporal_features_used'] = True
                artifacts['temporal_feature_info'] = temporal_info
                tprint(f"   ✅ Loaded {temporal_info['count']} temporal features")
                self.logger.info(f"✅ Using {temporal_info['count']} temporal features in analyst model training")
            else:
                tprint(f"   ⚠️  No temporal features loaded - proceeding with base features only")
            
            # Step 2: Check execution mode
            if config.mode == ExecutionMode.BLANK:
                tprint(f"   🔄 BLANK MODE: Skipping actual analyst models training")
                self.logger.info("🔄 Blank mode: Skipping actual analyst models training")
                artifacts['models'] = [f"analyst_model_{timestamp}.pkl"]
                tprint(f"   ✅ Blank mode completed - generated mock model artifact")
                return artifacts
            
            # Step 3: Import training module
            tprint(f"   🔍 Importing analyst models training module...")
            try:
                from .analyst_models_training_refactored import AnalystModelsTrainingStepRefactored as AnalystModelsTrainingStep
                tprint(f"   ✅ Successfully imported analyst models training module")
            except ImportError as e:
                error_msg = f"Analyst models trainer not available: {e}"
                tprint(f"   ❌ IMPORT FAILED: {error_msg}")
                self.logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Step 4: Create enhanced configuration
            tprint(f"   🔍 Creating enhanced configuration...")
            enhanced_config = config.custom_params.copy() if config.custom_params else {}
            if temporal_loaded:
                enhanced_config['temporal_features_available'] = True
                enhanced_config['temporal_feature_columns'] = list(self.temporal_features.keys())
                enhanced_config['temporal_feature_metadata'] = self.temporal_feature_metadata
                tprint(f"   ✅ Enhanced config with {len(self.temporal_features)} temporal features")
            else:
                tprint(f"   ✅ Using base configuration (no temporal features)")
            
            # Step 5: Initialize trainer
            tprint(f"   🔍 Initializing analyst models trainer...")
            trainer = AnalystModelsTrainingStep()
            tprint(f"   ✅ Trainer initialized successfully")
            
            # Step 6: Execute training
            tprint(f"   🔍 Executing analyst model training...")
            training_start = datetime.now()
            training_result = await trainer.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe,
                    'data_dir': config.data_dir
                },
                pipeline_state={}
            )
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            tprint(f"   ✅ Training completed in {training_duration:.2f} seconds")
            
            # Step 7: Process results
            tprint(f"   🔍 Processing training results...")
            models = training_result.get('models', [])
            artifacts['models'] = [f"{model}_{timestamp}.pkl" if not model.endswith('.pkl') else f"{model.replace('.pkl', '')}_{timestamp}.pkl" for model in models]
            artifacts['metrics'] = training_result.get('metrics', {})
            artifacts['performance'] = training_result.get('performance', {})
            
            tprint(f"   ✅ Processed {len(artifacts['models'])} models, {len(artifacts['metrics'])} metrics, {len(artifacts['performance'])} performance indicators")
            self.logger.info(f"✅ Analyst model training completed with {len(artifacts['models'])} models")
            
        except Exception as e:
            error_msg = f"Analyst model training failed: {e}"
            tprint(f"   ❌ TRAINING FAILED: {error_msg}")
            self.logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
        
        tprint(f"   ✅ ANALYST MODEL TRAINING PIPELINE COMPLETED")
        return artifacts
    
    async def _analyst_ensemble_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Analyst Ensemble Training sub-pipeline with enhanced error handling and reporting."""
        tprint(f"   🎭 ANALYST ENSEMBLE TRAINING PIPELINE STARTED")
        self.logger.info("🎭 Executing analyst ensemble training pipeline (per-regime ensemble models)")
        
        # Initialize artifacts with datetime-stamped filenames
        timestamp = self._generate_datetime_stamp()
        artifacts = {
            'models': [],
            'metrics': {},
            'performance': {},
            'temporal_features_used': False,
            'temporal_feature_info': {},
            'training_report': f"analyst_ensemble_training_report_{config.symbol}_{config.exchange}_{config.timeframe}_{timestamp}.json"
        }
        
        try:
            # Step 1: Check execution mode
            if config.mode == ExecutionMode.BLANK:
                tprint(f"   🔄 BLANK MODE: Skipping actual analyst ensemble training")
                self.logger.info("🔄 Blank mode: Skipping actual analyst ensemble training")
                artifacts['models'] = [f"analyst_ensemble_{timestamp}.pkl"]
                tprint(f"   ✅ Blank mode completed - generated mock ensemble artifact")
                return artifacts
            
            # Step 2: Import training module
            tprint(f"   🔍 Importing analyst ensemble training module...")
            try:
                from .analyst_ensemble_training import AnalystEnsembleTrainingStep as AnalystEnsembleTrainer
                tprint(f"   ✅ Successfully imported analyst ensemble training module")
            except ImportError as e:
                error_msg = f"Analyst ensemble trainer not available: {e}"
                tprint(f"   ❌ IMPORT FAILED: {error_msg}")
                self.logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Step 3: Create enhanced configuration
            tprint(f"   🔍 Creating enhanced configuration...")
            enhanced_config = config.custom_params.copy() if config.custom_params else {}
            tprint(f"   ✅ Configuration prepared with {len(enhanced_config)} custom parameters")
            
            # Step 4: Initialize trainer
            tprint(f"   🔍 Initializing analyst ensemble trainer...")
            trainer = AnalystEnsembleTrainer()
            tprint(f"   ✅ Trainer initialized successfully")
            
            # Step 5: Execute training
            tprint(f"   🔍 Executing analyst ensemble training...")
            training_start = datetime.now()
            training_result = await trainer.execute_analyst_ensemble_training(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                force_rerun=config.force_rerun,
                enhanced_config=enhanced_config
            )
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            tprint(f"   ✅ Ensemble training completed in {training_duration:.2f} seconds")
            
            # Step 6: Process results
            tprint(f"   🔍 Processing ensemble training results...")
            models = training_result.get('models', [])
            artifacts['models'] = [f"{model}_{timestamp}.pkl" if not model.endswith('.pkl') else f"{model.replace('.pkl', '')}_{timestamp}.pkl" for model in models]
            artifacts['metrics'] = training_result.get('metrics', {})
            artifacts['performance'] = training_result.get('performance', {})
            
            tprint(f"   ✅ Processed {len(artifacts['models'])} ensemble models, {len(artifacts['metrics'])} metrics, {len(artifacts['performance'])} performance indicators")
            self.logger.info(f"✅ Analyst ensemble training completed with {len(artifacts['models'])} models")
            
        except Exception as e:
            error_msg = f"Analyst ensemble training failed: {e}"
            tprint(f"   ❌ ENSEMBLE TRAINING FAILED: {error_msg}")
            self.logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
        
        tprint(f"   ✅ ANALYST ENSEMBLE TRAINING PIPELINE COMPLETED")
        return artifacts
    
    async def _tactician_models_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Tactician model training sub-pipeline with enhanced error handling and reporting."""
        tprint(f"   ⚔️ TACTICIAN MODELS TRAINING PIPELINE STARTED")
        self.logger.info("⚔️ Executing tactician model training pipeline")
        
        # Initialize artifacts with datetime-stamped filenames
        timestamp = self._generate_datetime_stamp()
        artifacts = {
            'models': [],
            'metrics': {},
            'performance': {},
            'training_report': f"tactician_models_training_report_{config.symbol}_{config.exchange}_{config.timeframe}_{timestamp}.json"
        }
        
        try:
            # Step 1: Check execution mode
            if config.mode == ExecutionMode.BLANK:
                tprint(f"   🔄 BLANK MODE: Skipping actual tactician model training")
                self.logger.info("🔄 Blank mode: Skipping actual tactician model training")
                artifacts['models'] = [f"tactician_model_{timestamp}.pkl"]
                tprint(f"   ✅ Blank mode completed - generated mock tactician model artifact")
                return artifacts
            
            # Step 2: Import training module
            tprint(f"   🔍 Importing tactician models training module...")
            try:
                from .tactician_models_training_refactored import TacticianModelsTrainingStepRefactored as TacticianModelTrainer
                tprint(f"   ✅ Successfully imported tactician models training module")
            except ImportError as e:
                error_msg = f"Tactician model trainer not available: {e}"
                tprint(f"   ❌ IMPORT FAILED: {error_msg}")
                self.logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Step 3: Create enhanced configuration
            tprint(f"   🔍 Creating enhanced configuration...")
            enhanced_config = config.custom_params.copy() if config.custom_params else {}
            tprint(f"   ✅ Configuration prepared with {len(enhanced_config)} custom parameters")
            
            # Step 4: Initialize trainer
            tprint(f"   🔍 Initializing tactician models trainer...")
            trainer = TacticianModelTrainer()
            tprint(f"   ✅ Trainer initialized successfully")
            
            # Step 5: Execute training
            tprint(f"   🔍 Executing tactician model training...")
            training_start = datetime.now()
            training_result = await trainer.execute(
                training_input={
                    'symbol': config.symbol,
                    'exchange': config.exchange,
                    'timeframe': config.timeframe,
                    'data_dir': config.data_dir
                },
                pipeline_state=enhanced_config
            )
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            tprint(f"   ✅ Tactician training completed in {training_duration:.2f} seconds")
            
            # Step 6: Process results
            tprint(f"   🔍 Processing tactician training results...")
            models = training_result.get('models', [])
            artifacts['models'] = [f"{model}_{timestamp}.pkl" if not model.endswith('.pkl') else f"{model.replace('.pkl', '')}_{timestamp}.pkl" for model in models]
            artifacts['metrics'] = training_result.get('metrics', {})
            artifacts['performance'] = training_result.get('performance', {})
            
            tprint(f"   ✅ Processed {len(artifacts['models'])} tactician models, {len(artifacts['metrics'])} metrics, {len(artifacts['performance'])} performance indicators")
            self.logger.info(f"✅ Tactician model training completed with {len(artifacts['models'])} models")
            
        except Exception as e:
            error_msg = f"Tactician model training failed: {e}"
            tprint(f"   ❌ TACTICIAN TRAINING FAILED: {error_msg}")
            self.logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
        
        tprint(f"   ✅ TACTICIAN MODELS TRAINING PIPELINE COMPLETED")
        return artifacts
    
    async def _tactician_ensemble_training_pipeline(self, config: SubPipelineConfig) -> Dict[str, Any]:
        """Tactician ensemble training sub-pipeline with enhanced error handling and reporting."""
        tprint(f"   ⚔️🎯 TACTICIAN ENSEMBLE TRAINING PIPELINE STARTED")
        self.logger.info("⚔️🎯 Executing tactician ensemble training pipeline")
        
        # Initialize artifacts with datetime-stamped filenames
        timestamp = self._generate_datetime_stamp()
        artifacts = {
            'models': [],
            'metrics': {},
            'performance': {},
            'training_report': f"tactician_ensemble_training_report_{config.symbol}_{config.exchange}_{config.timeframe}_{timestamp}.json"
        }
        
        try:
            # Step 1: Check execution mode
            if config.mode == ExecutionMode.BLANK:
                tprint(f"   🔄 BLANK MODE: Skipping actual tactician ensemble training")
                self.logger.info("🔄 Blank mode: Skipping actual tactician ensemble training")
                artifacts['models'] = [f"tactician_ensemble_{timestamp}.pkl"]
                tprint(f"   ✅ Blank mode completed - generated mock tactician ensemble artifact")
                return artifacts
            
            # Step 2: Import training module
            tprint(f"   🔍 Importing tactician ensemble training module...")
            try:
                from .tactician_ensemble_training import TacticianEnsembleTrainingStep as TacticianEnsembleTrainer
                tprint(f"   ✅ Successfully imported tactician ensemble training module")
            except ImportError as e:
                error_msg = f"Tactician ensemble trainer not available: {e}"
                tprint(f"   ❌ IMPORT FAILED: {error_msg}")
                self.logger.error(f"❌ {error_msg}")
                raise RuntimeError(error_msg) from e
            
            # Step 3: Create enhanced configuration
            tprint(f"   🔍 Creating enhanced configuration...")
            enhanced_config = config.custom_params.copy() if config.custom_params else {}
            tprint(f"   ✅ Configuration prepared with {len(enhanced_config)} custom parameters")
            
            # Step 4: Initialize trainer
            tprint(f"   🔍 Initializing tactician ensemble trainer...")
            trainer = TacticianEnsembleTrainer()
            tprint(f"   ✅ Trainer initialized successfully")
            
            # Step 5: Execute training
            tprint(f"   🔍 Executing tactician ensemble training...")
            training_start = datetime.now()
            training_result = await trainer.execute_tactician_ensemble_training(
                symbol=config.symbol,
                exchange=config.exchange,
                timeframe=config.timeframe,
                data_dir=config.data_dir,
                force_rerun=config.force_rerun,
                enhanced_config=enhanced_config
            )
            training_end = datetime.now()
            training_duration = (training_end - training_start).total_seconds()
            tprint(f"   ✅ Tactician ensemble training completed in {training_duration:.2f} seconds")
            
            # Step 6: Process results
            tprint(f"   🔍 Processing tactician ensemble training results...")
            models = training_result.get('models', [])
            artifacts['models'] = [f"{model}_{timestamp}.pkl" if not model.endswith('.pkl') else f"{model.replace('.pkl', '')}_{timestamp}.pkl" for model in models]
            artifacts['metrics'] = training_result.get('metrics', {})
            artifacts['performance'] = training_result.get('performance', {})
            
            tprint(f"   ✅ Processed {len(artifacts['models'])} tactician ensemble models, {len(artifacts['metrics'])} metrics, {len(artifacts['performance'])} performance indicators")
            self.logger.info(f"✅ Tactician ensemble training completed with {len(artifacts['models'])} models")
            
        except Exception as e:
            error_msg = f"Tactician ensemble training failed: {e}"
            tprint(f"   ❌ TACTICIAN ENSEMBLE TRAINING FAILED: {error_msg}")
            self.logger.error(f"❌ {error_msg}")
            raise RuntimeError(error_msg) from e
        
        tprint(f"   ✅ TACTICIAN ENSEMBLE TRAINING PIPELINE COMPLETED")
        return artifacts
    
    
    
    
    def get_available_sub_pipelines(self) -> List[str]:
        """Get list of available sub-pipelines."""
        return list(self.sub_pipelines.keys())
    
    def get_sub_pipeline_status(self, sub_pipeline_name: str) -> Optional[SubPipelineStatus]:
        """Get status of a specific sub-pipeline."""
        for result in self.results:
            if result.sub_pipeline_name == sub_pipeline_name:
                return result.status
        return None
    
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