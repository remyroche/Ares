"""
Advanced Artifact Management for Unified Data-Driven Pipeline.

This module provides comprehensive artifact management and persistence infrastructure
similar to FeatureLookbackOptimizationComponent but adapted for the unified pipeline.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict

# Import utility modules
from src.utils.common_utilities import (
    CommonUtilities, safe_dataframe_operation, validate_dataframe_columns,
    analyze_nan_values_detailed, calculate_data_quality_metrics,
    create_data_quality_report, get_dataframe_info, create_summary_statistics,
    safe_convert_dtypes, safe_merge_dataframes, safe_drop_columns
)
from src.utils.serialization_utils import UniversalSerializer, JSONSerializer, PickleSerializer

try:
    from src.utils.tprint import tprint, tprint_error, tprint_warning, tprint_success, tprint_debug
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print("TPRINT:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)

import numpy as np
import pandas as pd


@dataclass
class ArtifactMetadata:
    """Metadata for an artifact."""
    name: str
    artifact_type: str
    created_at: str
    size_bytes: int
    checksum: str
    tags: Dict[str, str] = None
    description: str = ""


@dataclass
class ArtifactSaveReport:
    """Report of artifact saving operation."""
    success: bool
    correlation_id: str
    paths: Dict[str, str]
    total_size_bytes: int
    save_time_seconds: float
    artifacts_saved: int
    errors: List[str] = None


class AdvancedArtifactManager:
    """
    Advanced artifact manager for unified pipeline.
    
    Provides comprehensive artifact creation, storage, and retrieval capabilities
    similar to FeatureLookbackOptimizationComponent.
    """

    def __init__(self, base_dir: str = "artifacts", logger=None):
        """Initialize the advanced artifact manager."""
        self.logger = logger or logging.getLogger(__name__)
        self.common_utils = CommonUtilities()
        self.serializer = UniversalSerializer()
        self.json_serializer = JSONSerializer()
        self.pickle_serializer = PickleSerializer()

        # Set up artifact directories
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.dirs = {
            'results': self.base_dir / 'results',
            'models': self.base_dir / 'models',
            'data': self.base_dir / 'data',
            'reports': self.base_dir / 'reports',
            'logs': self.base_dir / 'logs',
            'temp': self.base_dir / 'temp'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)

        # Artifact tracking
        self.artifact_registry: Dict[str, ArtifactMetadata] = {}
        self.save_history: List[ArtifactSaveReport] = []

        tprint_success(f"✅ AdvancedArtifactManager initialized with base directory: {self.base_dir}")

    def analyze_artifact_quality(self, artifact_data: Any, artifact_name: str) -> Dict[str, Any]:
        """
        Analyze data quality of an artifact using enhanced utilities.
        
        Args:
            artifact_data: The artifact data to analyze
            artifact_name: Name of the artifact
            
        Returns:
            Dictionary with quality analysis results
        """
        tprint_debug(f"🔍 Analyzing quality of artifact: {artifact_name}")
        
        try:
            quality_analysis = {
                'artifact_name': artifact_name,
                'timestamp': datetime.now().isoformat(),
                'artifact_type': type(artifact_data).__name__,
                'analysis_results': {}
            }
            
            # Analyze DataFrame artifacts
            if isinstance(artifact_data, pd.DataFrame):
                tprint_debug(f"📊 Analyzing DataFrame artifact: {artifact_name}")
                
                # Validate columns
                required_columns = ['close'] if 'close' in artifact_data.columns else []
                column_validation = validate_dataframe_columns(artifact_data, required_columns)
                
                # Comprehensive data quality analysis
                nan_analysis = analyze_nan_values_detailed(artifact_data)
                quality_metrics = calculate_data_quality_metrics(artifact_data)
                quality_report = create_data_quality_report(artifact_data)
                dataframe_info = get_dataframe_info(artifact_data)
                summary_stats = create_summary_statistics(artifact_data)
                
                quality_analysis['analysis_results'] = {
                    'column_validation': column_validation,
                    'nan_analysis': nan_analysis,
                    'quality_metrics': quality_metrics,
                    'quality_report': quality_report,
                    'dataframe_info': dataframe_info,
                    'summary_statistics': summary_stats
                }
                
                tprint_success(f"✅ DataFrame analysis completed: {quality_metrics.get('missing_percentage', 0):.1f}% missing")
                
            elif isinstance(artifact_data, np.ndarray):
                tprint_debug(f"🔢 Analyzing NumPy array artifact: {artifact_name}")
                
                # Convert to DataFrame for analysis
                if artifact_data.ndim == 2:
                    df_for_analysis = pd.DataFrame(artifact_data)
                    nan_analysis = analyze_nan_values_detailed(df_for_analysis)
                    quality_metrics = calculate_data_quality_metrics(df_for_analysis)
                    
                    quality_analysis['analysis_results'] = {
                        'array_shape': artifact_data.shape,
                        'array_dtype': str(artifact_data.dtype),
                        'nan_analysis': nan_analysis,
                        'quality_metrics': quality_metrics
                    }
                    
                    tprint_success(f"✅ Array analysis completed: shape {artifact_data.shape}")
                else:
                    quality_analysis['analysis_results'] = {
                        'array_shape': artifact_data.shape,
                        'array_dtype': str(artifact_data.dtype),
                        'note': '1D array - limited analysis available'
                    }
                    
            else:
                # Basic analysis for other types
                quality_analysis['analysis_results'] = {
                    'type': type(artifact_data).__name__,
                    'size_bytes': self.common_utils.get_file_size(str(artifact_data)) if hasattr(artifact_data, '__len__') else 0,
                    'note': 'Limited analysis available for this data type'
                }
            
            return quality_analysis
            
        except Exception as e:
            tprint_error(f"❌ Quality analysis failed for {artifact_name}: {e}")
            return {
                'artifact_name': artifact_name,
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'analysis_results': {}
            }

    async def save_artifacts(self, artifacts: Dict[str, Any], 
                           metadata: Optional[Dict[str, Any]] = None) -> ArtifactSaveReport:
        """
        Save artifacts to persistent storage.
        
        Args:
            artifacts: Dictionary of artifacts to save
            metadata: Optional metadata for the save operation
            
        Returns:
            ArtifactSaveReport with save operation details
        """
        tprint_debug("💾 Starting artifact saving operation")
        start_time = datetime.now()
        
        correlation_id = f"save_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        saved_paths = {}
        errors = []
        total_size = 0
        artifacts_saved = 0

        try:
            for artifact_name, artifact_data in artifacts.items():
                try:
                    # Determine artifact type and save method
                    artifact_type = self._determine_artifact_type(artifact_data)
                    save_path = self._get_save_path(artifact_name, artifact_type)
                    
                    # Save the artifact
                    saved_size = await self._save_single_artifact(
                        artifact_data, save_path, artifact_type
                    )
                    
                    saved_paths[artifact_name] = str(save_path)
                    total_size += saved_size
                    artifacts_saved += 1
                    
                    # Register artifact
                    self._register_artifact(artifact_name, artifact_type, save_path, saved_size)
                    
                    tprint_debug(f"✅ Saved artifact {artifact_name} to {save_path}")
                    
                except Exception as e:
                    error_msg = f"Failed to save artifact {artifact_name}: {str(e)}"
                    errors.append(error_msg)
                    tprint_error(f"❌ {error_msg}")

            # Save metadata if provided
            if metadata:
                metadata_path = self.dirs['results'] / f"{correlation_id}_metadata.json"
                try:
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2, default=str)
                    saved_paths['metadata'] = str(metadata_path)
                    tprint_debug(f"✅ Saved metadata to {metadata_path}")
                except Exception as e:
                    error_msg = f"Failed to save metadata: {str(e)}"
                    errors.append(error_msg)
                    tprint_error(f"❌ {error_msg}")

            # Create save report
            save_time = (datetime.now() - start_time).total_seconds()
            report = ArtifactSaveReport(
                success=len(errors) == 0,
                correlation_id=correlation_id,
                paths=saved_paths,
                total_size_bytes=total_size,
                save_time_seconds=save_time,
                artifacts_saved=artifacts_saved,
                errors=errors if errors else None
            )
            
            self.save_history.append(report)
            
            if report.success:
                tprint_success(f"✅ Artifact saving completed: {artifacts_saved} artifacts, {total_size/1024/1024:.2f}MB")
            else:
                tprint_warning(f"⚠️ Artifact saving completed with errors: {len(errors)} errors")
            
            return report

        except Exception as e:
            tprint_error(f"❌ Artifact saving failed: {e}")
            return ArtifactSaveReport(
                success=False,
                correlation_id=correlation_id,
                paths={},
                total_size_bytes=0,
                save_time_seconds=(datetime.now() - start_time).total_seconds(),
                artifacts_saved=0,
                errors=[str(e)]
            )

    async def load_artifacts(self, correlation_id: str) -> Dict[str, Any]:
        """
        Load artifacts by correlation ID.
        
        Args:
            correlation_id: Correlation ID of the save operation
            
        Returns:
            Dictionary of loaded artifacts
        """
        tprint_debug(f"📥 Loading artifacts for correlation ID: {correlation_id}")
        
        # Find the save report
        save_report = None
        for report in self.save_history:
            if report.correlation_id == correlation_id:
                save_report = report
                break
        
        if not save_report:
            tprint_error(f"❌ No save report found for correlation ID: {correlation_id}")
            return {}
        
        loaded_artifacts = {}
        
        try:
            for artifact_name, artifact_path in save_report.paths.items():
                if artifact_name == 'metadata':
                    continue
                
                try:
                    artifact_data = await self._load_single_artifact(artifact_path)
                    loaded_artifacts[artifact_name] = artifact_data
                    tprint_debug(f"✅ Loaded artifact {artifact_name}")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to load artifact {artifact_name}: {e}")
            
            tprint_success(f"✅ Loaded {len(loaded_artifacts)} artifacts")
            return loaded_artifacts
            
        except Exception as e:
            tprint_error(f"❌ Artifact loading failed: {e}")
            return {}

    def create_optimization_artifacts(self, optimization_results: Dict[str, Any],
                                    pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create comprehensive artifacts from optimization results.
        
        Args:
            optimization_results: Results from optimization
            pipeline_state: Pipeline state information
            
        Returns:
            Dictionary of artifacts ready for saving
        """
        tprint_debug("📦 Creating optimization artifacts")
        
        artifacts = {}
        
        # Core optimization results
        if 'feature_results' in optimization_results:
            artifacts['feature_results'] = optimization_results['feature_results']
        
        if 'optimal_periods' in optimization_results:
            artifacts['optimal_periods'] = optimization_results['optimal_periods']
        
        if 'selected_features' in optimization_results:
            artifacts['selected_features'] = optimization_results['selected_features']
        
        # Performance metrics
        if 'performance_metrics' in optimization_results:
            artifacts['performance_metrics'] = optimization_results['performance_metrics']
        
        # Configuration
        if pipeline_state:
            artifacts['pipeline_config'] = {
                'symbol': pipeline_state.get('symbol', 'UNKNOWN'),
                'exchange': pipeline_state.get('exchange', 'UNKNOWN'),
                'timeframe': pipeline_state.get('timeframe', 'UNKNOWN'),
                'execution_mode': pipeline_state.get('execution_mode', 'UNKNOWN'),
                'optimization_direction': pipeline_state.get('direction', 'both')
            }
        
        # Summary statistics
        artifacts['summary'] = self._create_summary_statistics(optimization_results)
        
        # Feature importance
        if 'feature_importance' in optimization_results:
            artifacts['feature_importance'] = optimization_results['feature_importance']
        
        # Lookback optimization results
        if 'long_pipeline' in optimization_results.get('feature_results', {}):
            artifacts['long_pipeline_results'] = optimization_results['feature_results']['long_pipeline']
        
        if 'short_pipeline' in optimization_results.get('feature_results', {}):
            artifacts['short_pipeline_results'] = optimization_results['feature_results']['short_pipeline']
        
        tprint_success(f"✅ Created {len(artifacts)} optimization artifacts")
        return artifacts

    def create_outcome_report(self, optimization_results: Dict[str, Any],
                            performance_metrics: Dict[str, Any],
                            pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create comprehensive outcome report.
        
        Args:
            optimization_results: Results from optimization
            performance_metrics: Performance metrics
            pipeline_state: Pipeline state information
            
        Returns:
            Comprehensive outcome report
        """
        tprint_debug("📊 Creating comprehensive outcome report")
        
        # Extract feature statistics
        long_pipeline = optimization_results.get('feature_results', {}).get('long_pipeline', {})
        short_pipeline = optimization_results.get('feature_results', {}).get('short_pipeline', {})
        
        long_count = len(long_pipeline)
        short_count = len(short_pipeline)
        
        # Calculate optimization statistics
        long_lookbacks = [data.get('best_lookback_period', 0) for data in long_pipeline.values() 
                         if isinstance(data, dict) and 'best_lookback_period' in data]
        long_scores = [data.get('best_score', 0.0) for data in long_pipeline.values() 
                      if isinstance(data, dict) and 'best_score' in data]
        
        short_lookbacks = [data.get('best_lookback_period', 0) for data in short_pipeline.values() 
                          if isinstance(data, dict) and 'best_lookback_period' in data]
        short_scores = [data.get('best_score', 0.0) for data in short_pipeline.values() 
                       if isinstance(data, dict) and 'best_score' in data]
        
        # Create outcome report
        outcome_report = {
            'component': 'unified_data_driven_pipeline',
            'timestamp': datetime.now().isoformat(),
            'execution_time': performance_metrics.get('execution_time', 0.0),
            'configuration': {
                'symbol': pipeline_state.get('symbol', 'UNKNOWN') if pipeline_state else 'UNKNOWN',
                'exchange': pipeline_state.get('exchange', 'UNKNOWN') if pipeline_state else 'UNKNOWN',
                'timeframe': pipeline_state.get('timeframe', 'UNKNOWN') if pipeline_state else 'UNKNOWN',
                'execution_mode': pipeline_state.get('execution_mode', 'UNKNOWN') if pipeline_state else 'UNKNOWN',
                'optimization_direction': pipeline_state.get('direction', 'both') if pipeline_state else 'both'
            },
            'results': {
                'summary': {
                    'total_features_optimized': long_count + short_count,
                    'long_pipeline_features': long_count,
                    'short_pipeline_features': short_count,
                    'optimization_method': optimization_results.get('optimization_method', 'unknown'),
                    'execution_mode': optimization_results.get('execution_mode', 'unknown')
                },
                'optimization_statistics': {
                    'long_pipeline': {
                        'features_count': long_count,
                        'average_lookback': float(np.mean(long_lookbacks)) if long_lookbacks else 0.0,
                        'average_score': float(np.mean(long_scores)) if long_scores else 0.0,
                        'min_lookback': min(long_lookbacks) if long_lookbacks else 0,
                        'max_lookback': max(long_lookbacks) if long_lookbacks else 0
                    },
                    'short_pipeline': {
                        'features_count': short_count,
                        'average_lookback': float(np.mean(short_lookbacks)) if short_lookbacks else 0.0,
                        'average_score': float(np.mean(short_scores)) if short_scores else 0.0,
                        'min_lookback': min(short_lookbacks) if short_lookbacks else 0,
                        'max_lookback': max(short_lookbacks) if short_lookbacks else 0
                    }
                },
                'feature_results': {
                    'long_pipeline': long_pipeline,
                    'short_pipeline': short_pipeline
                }
            },
            'performance_metrics': performance_metrics,
            'status': 'success'
        }
        
        tprint_success("✅ Created comprehensive outcome report")
        return outcome_report

    async def _save_single_artifact(self, data: Any, path: Path, artifact_type: str) -> int:
        """Save a single artifact."""
        if artifact_type == 'dataframe':
            data.to_parquet(path)
            return path.stat().st_size
        elif artifact_type == 'json':
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return path.stat().st_size
        elif artifact_type == 'pickle':
            self.pickle_serializer.save(data, str(path))
            return path.stat().st_size
        else:
            # Default to JSON
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            return path.stat().st_size

    async def _load_single_artifact(self, path: str) -> Any:
        """Load a single artifact."""
        path_obj = Path(path)
        
        if path_obj.suffix == '.parquet':
            return pd.read_parquet(path)
        elif path_obj.suffix == '.json':
            with open(path, 'r') as f:
                return json.load(f)
        elif path_obj.suffix == '.pkl':
            return self.pickle_serializer.load(path)
        else:
            # Default to JSON
            with open(path, 'r') as f:
                return json.load(f)

    def _determine_artifact_type(self, data: Any) -> str:
        """Determine the type of artifact for appropriate serialization."""
        if isinstance(data, pd.DataFrame):
            return 'dataframe'
        elif isinstance(data, (dict, list)):
            return 'json'
        else:
            return 'pickle'

    def _get_save_path(self, artifact_name: str, artifact_type: str) -> Path:
        """Get the save path for an artifact."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if artifact_type == 'dataframe':
            return self.dirs['data'] / f"{artifact_name}_{timestamp}.parquet"
        elif artifact_type == 'json':
            return self.dirs['results'] / f"{artifact_name}_{timestamp}.json"
        else:
            return self.dirs['results'] / f"{artifact_name}_{timestamp}.pkl"

    def _register_artifact(self, name: str, artifact_type: str, path: Path, size_bytes: int):
        """Register an artifact in the registry."""
        metadata = ArtifactMetadata(
            name=name,
            artifact_type=artifact_type,
            created_at=datetime.now().isoformat(),
            size_bytes=size_bytes,
            checksum=str(hash(str(path))),  # Simple checksum
            description=f"Artifact saved at {path}"
        )
        
        self.artifact_registry[name] = metadata

    def _create_summary_statistics(self, optimization_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics from optimization results."""
        summary = {
            'total_features_optimized': 0,
            'optimization_method': optimization_results.get('optimization_method', 'unknown'),
            'execution_mode': optimization_results.get('execution_mode', 'unknown'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Count features
        feature_results = optimization_results.get('feature_results', {})
        long_pipeline = feature_results.get('long_pipeline', {})
        short_pipeline = feature_results.get('short_pipeline', {})
        
        summary['total_features_optimized'] = len(long_pipeline) + len(short_pipeline)
        summary['long_features'] = len(long_pipeline)
        summary['short_features'] = len(short_pipeline)
        
        return summary

    def get_artifact_registry(self) -> Dict[str, ArtifactMetadata]:
        """Get the artifact registry."""
        return self.artifact_registry.copy()

    def get_save_history(self) -> List[ArtifactSaveReport]:
        """Get the save history."""
        return self.save_history.copy()

    def cleanup_old_artifacts(self, days_to_keep: int = 30):
        """Clean up old artifacts."""
        tprint_debug(f"🧹 Cleaning up artifacts older than {days_to_keep} days")
        
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cleaned_count = 0
        
        for dir_path in self.dirs.values():
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        try:
                            file_path.unlink()
                            cleaned_count += 1
                        except Exception as e:
                            tprint_warning(f"⚠️ Failed to delete {file_path}: {e}")
        
        tprint_success(f"✅ Cleaned up {cleaned_count} old artifacts")