"""
Result Manager for NAS/TAS Systems

This module provides comprehensive result management, serialization, and handling
for both NAS and TAS implementations, consolidating result processing logic.
"""

import json
import pickle
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
from pathlib import Path
import hashlib
import uuid
from enum import Enum

from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error,
    tprint_success, tprint_progress, tprint_performance, tprint_timer,
    tprint_structured, tprint_with_level, tprint_logged, LogLevel
)

from ..evaluation.financial_metrics import TradingPerformanceMetrics, RiskMetrics
from ..evaluation.unified_evaluator import EvaluationResult

class ArchitectureType(Enum):
    """Types of architectures."""
    NEURAL = "neural"
    TREE = "tree"
    HYBRID = "hybrid"
    ENSEMBLE = "ensemble"

class ResultStatus(Enum):
    """Status of results."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ExecutionInfo:
    """Information about execution."""

    # Execution metadata
    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0

    # System information
    system_info: Dict[str, Any] = field(default_factory=dict)
    resource_usage: Dict[str, Any] = field(default_factory=dict)

    # Configuration
    config_hash: str = ""
    config_summary: Dict[str, Any] = field(default_factory=dict)

    # Status
    status: ResultStatus = ResultStatus.SUCCESS
    error_message: Optional[str] = None

    def __post_init__(self):
        """Calculate duration if end_time is set."""
        if self.end_time and self.start_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    def finish_execution(self, status: ResultStatus = ResultStatus.SUCCESS, error_message: Optional[str] = None):
        """Mark execution as finished."""
        self.end_time = datetime.now()
        self.status = status
        self.error_message = error_message
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'execution_id': self.execution_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration_seconds,
            'system_info': self.system_info,
            'resource_usage': self.resource_usage,
            'config_hash': self.config_hash,
            'config_summary': self.config_summary,
            'status': self.status.value,
            'error_message': self.error_message
        }

@dataclass
class ArchitectureResult:
    """Individual architecture result."""

    # Architecture information
    architecture_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    architecture_type: ArchitectureType = ArchitectureType.HYBRID
    architecture_config: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    # Financial metrics
    financial_metrics: Optional[TradingPerformanceMetrics] = None
    risk_metrics: Optional[RiskMetrics] = None

    # Evaluation results
    evaluation_result: Optional[EvaluationResult] = None

    # Model information
    model_info: Dict[str, Any] = field(default_factory=dict)
    model_size_mb: float = 0.0
    model_complexity: float = 0.0

    # Training information
    training_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'architecture_id': self.architecture_id,
            'architecture_type': self.architecture_type.value,
            'architecture_config': self.architecture_config,
            'performance_metrics': self.performance_metrics,
            'model_info': self.model_info,
            'model_size_mb': self.model_size_mb,
            'model_complexity': self.model_complexity,
            'training_info': self.training_info
        }

        if self.financial_metrics:
            result['financial_metrics'] = self.financial_metrics.to_dict()

        if self.risk_metrics:
            result['risk_metrics'] = self.risk_metrics.to_dict()

        if self.evaluation_result:
            result['evaluation_result'] = self.evaluation_result.to_dict()

        return result

@dataclass
class ComparisonResult:
    """Result of comparing two architectures."""

    # Comparison metadata
    comparison_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    comparison_timestamp: datetime = field(default_factory=datetime.now)

    # Architecture IDs being compared
    architecture_1_id: str = ""
    architecture_2_id: str = ""

    # Comparison metrics
    performance_comparison: Dict[str, float] = field(default_factory=dict)
    financial_comparison: Dict[str, float] = field(default_factory=dict)

    # Overall comparison
    better_architecture: str = ""
    improvement_percentage: float = 0.0
    significance_level: float = 0.05

    # Detailed comparison
    detailed_comparison: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'comparison_id': self.comparison_id,
            'comparison_timestamp': self.comparison_timestamp.isoformat(),
            'architecture_1_id': self.architecture_1_id,
            'architecture_2_id': self.architecture_2_id,
            'performance_comparison': self.performance_comparison,
            'financial_comparison': self.financial_comparison,
            'better_architecture': self.better_architecture,
            'improvement_percentage': self.improvement_percentage,
            'significance_level': self.significance_level,
            'detailed_comparison': self.detailed_comparison
        }

@dataclass
class UnifiedArchitectureResult:
    """
    Unified result structure for both NAS and TAS systems.

    This class consolidates result structures that were previously
    different between NAS and TAS implementations, providing a unified
    interface for result handling and analysis.
    """

    # Result metadata
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    result_timestamp: datetime = field(default_factory=datetime.now)
    result_version: str = "1.0.0"

    # Search information
    search_type: str = "unified"  # "nas", "tas", "hybrid"
    search_strategy: str = "hybrid"
    optimization_mode: str = "regime_aware"

    # Architecture results
    best_architecture: Optional[ArchitectureResult] = None
    all_architectures: List[ArchitectureResult] = field(default_factory=list)
    architecture_count: int = 0

    # Search performance
    search_performance: Dict[str, Any] = field(default_factory=dict)

    # Execution information
    execution_info: ExecutionInfo = field(default_factory=ExecutionInfo)

    # Regime analysis
    regime_analysis: Dict[str, Any] = field(default_factory=dict)

    # Financial analysis
    financial_analysis: Dict[str, Any] = field(default_factory=dict)

    # Performance analysis
    performance_analysis: Dict[str, Any] = field(default_factory=dict)

    # Recommendations
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization setup."""
        self.architecture_count = len(self.all_architectures)

        # Set best architecture if not set
        if self.best_architecture is None and self.all_architectures:
            self.best_architecture = self._select_best_architecture()

    def _select_best_architecture(self) -> Optional[ArchitectureResult]:
        """Select the best architecture based on performance."""
        if not self.all_architectures:
            return None

        # Simple selection based on performance metrics
        best_arch = None
        best_score = -float('inf')

        for arch in self.all_architectures:
            score = self._calculate_architecture_score(arch)
            if score > best_score:
                best_score = score
                best_arch = arch

        return best_arch

    def _calculate_architecture_score(self, architecture: ArchitectureResult) -> float:
        """Calculate overall score for an architecture."""
        score = 0.0

        # Performance score
        if 'accuracy' in architecture.performance_metrics:
            score += architecture.performance_metrics['accuracy'] * 0.4

        if 'f1_score' in architecture.performance_metrics:
            score += architecture.performance_metrics['f1_score'] * 0.3

        # Financial score
        if architecture.financial_metrics:
            score += architecture.financial_metrics.sharpe_ratio * 0.2
            score += (1 - abs(architecture.financial_metrics.max_drawdown)) * 0.1

        return score

    def add_architecture(self, architecture: ArchitectureResult):
        """Add an architecture to the results."""
        self.all_architectures.append(architecture)
        self.architecture_count = len(self.all_architectures)

        # Update best architecture if this one is better
        if self.best_architecture is None:
            self.best_architecture = architecture
        else:
            current_score = self._calculate_architecture_score(self.best_architecture)
            new_score = self._calculate_architecture_score(architecture)

            if new_score > current_score:
                self.best_architecture = architecture

    def get_top_architectures(self, n: int = 5) -> List[ArchitectureResult]:
        """Get top N architectures by score."""
        if not self.all_architectures:
            return []

        # Sort by score
        scored_architectures = [
            (arch, self._calculate_architecture_score(arch))
            for arch in self.all_architectures
        ]
        scored_architectures.sort(key=lambda x: x[1], reverse=True)

        return [arch for arch, _ in scored_architectures[:n]]

    def get_architecture_by_id(self, architecture_id: str) -> Optional[ArchitectureResult]:
        """Get architecture by ID."""
        for arch in self.all_architectures:
            if arch.architecture_id == architecture_id:
                return arch
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            'result_id': self.result_id,
            'result_timestamp': self.result_timestamp.isoformat(),
            'result_version': self.result_version,
            'search_type': self.search_type,
            'search_strategy': self.search_strategy,
            'optimization_mode': self.optimization_mode,
            'architecture_count': self.architecture_count,
            'search_performance': self.search_performance,
            'execution_info': self.execution_info.to_dict(),
            'regime_analysis': self.regime_analysis,
            'financial_analysis': self.financial_analysis,
            'performance_analysis': self.performance_analysis,
            'recommendations': self.recommendations,
            'metadata': self.metadata
        }

        # Add architectures
        result['all_architectures'] = [arch.to_dict() for arch in self.all_architectures]

        if self.best_architecture:
            result['best_architecture'] = self.best_architecture.to_dict()

        return result

    def save(self, filepath: Union[str, Path], format: str = "json") -> bool:
        """Save result to file."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            if format.lower() == "json":
                with open(filepath, 'w') as f:
                    json.dump(self.to_dict(), f, indent=2, default=str)
            elif format.lower() == "pickle":
                with open(filepath, 'wb') as f:
                    pickle.dump(self, f)
            else:
                raise ValueError(f"Unsupported format: {format}")

            tprint_success(f"Result saved to {filepath} ({format})")
            return True

        except Exception as e:
            tprint_error(f"Failed to save result: {e}")
            return False

    @classmethod
    def load(cls, filepath: Union[str, Path], format: str = "json") -> 'UnifiedArchitectureResult':
        """Load result from file."""
        try:
            filepath = Path(filepath)

            if format.lower() == "json":
                with open(filepath, 'r') as f:
                    data = json.load(f)
                return cls.from_dict(data)
            elif format.lower() == "pickle":
                with open(filepath, 'rb') as f:
                    return pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")

        except Exception as e:
            tprint_error(f"Failed to load result: {e}")
            raise

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedArchitectureResult':
        """Create result from dictionary."""
        # Extract architectures
        all_architectures = []
        for arch_data in data.get('all_architectures', []):
            arch = ArchitectureResult(
                architecture_id=arch_data.get('architecture_id', str(uuid.uuid4())),
                architecture_type=ArchitectureType(arch_data.get('architecture_type', 'hybrid')),
                architecture_config=arch_data.get('architecture_config', {}),
                performance_metrics=arch_data.get('performance_metrics', {}),
                model_info=arch_data.get('model_info', {}),
                model_size_mb=arch_data.get('model_size_mb', 0.0),
                model_complexity=arch_data.get('model_complexity', 0.0),
                training_info=arch_data.get('training_info', {})
            )

            # Load financial metrics if present
            if 'financial_metrics' in arch_data:
                arch.financial_metrics = TradingPerformanceMetrics(**arch_data['financial_metrics'])

            if 'risk_metrics' in arch_data:
                arch.risk_metrics = RiskMetrics(**arch_data['risk_metrics'])

            all_architectures.append(arch)

        # Create execution info
        exec_info_data = data.get('execution_info', {})
        execution_info = ExecutionInfo(
            execution_id=exec_info_data.get('execution_id', str(uuid.uuid4())),
            start_time=datetime.fromisoformat(exec_info_data.get('start_time', datetime.now().isoformat())),
            end_time=datetime.fromisoformat(exec_info_data['end_time']) if exec_info_data.get('end_time') else None,
            system_info=exec_info_data.get('system_info', {}),
            resource_usage=exec_info_data.get('resource_usage', {}),
            config_hash=exec_info_data.get('config_hash', ''),
            config_summary=exec_info_data.get('config_summary', {}),
            status=ResultStatus(exec_info_data.get('status', 'success')),
            error_message=exec_info_data.get('error_message')
        )
        execution_info.duration_seconds = exec_info_data.get('duration_seconds', 0.0)

        # Create result
        result = cls(
            result_id=data.get('result_id', str(uuid.uuid4())),
            result_timestamp=datetime.fromisoformat(data.get('result_timestamp', datetime.now().isoformat())),
            result_version=data.get('result_version', '1.0.0'),
            search_type=data.get('search_type', 'unified'),
            search_strategy=data.get('search_strategy', 'hybrid'),
            optimization_mode=data.get('optimization_mode', 'regime_aware'),
            all_architectures=all_architectures,
            search_performance=data.get('search_performance', {}),
            execution_info=execution_info,
            regime_analysis=data.get('regime_analysis', {}),
            financial_analysis=data.get('financial_analysis', {}),
            performance_analysis=data.get('performance_analysis', {}),
            recommendations=data.get('recommendations', []),
            metadata=data.get('metadata', {})
        )

        # Set best architecture
        if 'best_architecture' in data:
            best_arch_data = data['best_architecture']
            result.best_architecture = ArchitectureResult(
                architecture_id=best_arch_data.get('architecture_id', str(uuid.uuid4())),
                architecture_type=ArchitectureType(best_arch_data.get('architecture_type', 'hybrid')),
                architecture_config=best_arch_data.get('architecture_config', {}),
                performance_metrics=best_arch_data.get('performance_metrics', {}),
                model_info=best_arch_data.get('model_info', {}),
                model_size_mb=best_arch_data.get('model_size_mb', 0.0),
                model_complexity=best_arch_data.get('model_complexity', 0.0),
                training_info=best_arch_data.get('training_info', {})
            )

            if 'financial_metrics' in best_arch_data:
                result.best_architecture.financial_metrics = TradingPerformanceMetrics(**best_arch_data['financial_metrics'])

            if 'risk_metrics' in best_arch_data:
                result.best_architecture.risk_metrics = RiskMetrics(**best_arch_data['risk_metrics'])

        return result

    def compare(self, other: 'UnifiedArchitectureResult') -> ComparisonResult:
        """Compare with another result."""
        if not self.best_architecture or not other.best_architecture:
            raise ValueError("Both results must have best architectures to compare")

        comparison = ComparisonResult(
            architecture_1_id=self.result_id,
            architecture_2_id=other.result_id
        )

        # Performance comparison
        self_perf = self.best_architecture.performance_metrics
        other_perf = other.best_architecture.performance_metrics

        for metric in ['accuracy', 'f1_score', 'precision', 'recall']:
            if metric in self_perf and metric in other_perf:
                diff = self_perf[metric] - other_perf[metric]
                comparison.performance_comparison[f"{metric}_difference"] = diff

        # Financial comparison
        if self.best_architecture.financial_metrics and other.best_architecture.financial_metrics:
            self_fin = self.best_architecture.financial_metrics
            other_fin = other.best_architecture.financial_metrics

            comparison.financial_comparison = {
                'sharpe_difference': self_fin.sharpe_ratio - other_fin.sharpe_ratio,
                'max_drawdown_difference': self_fin.max_drawdown - other_fin.max_drawdown,
                'win_rate_difference': self_fin.win_rate - other_fin.win_rate,
                'profit_factor_difference': self_fin.profit_factor - other_fin.profit_factor
            }

        # Determine better architecture
        self_score = self._calculate_architecture_score(self.best_architecture)
        other_score = self._calculate_architecture_score(other.best_architecture)

        if self_score > other_score:
            comparison.better_architecture = self.result_id
            comparison.improvement_percentage = ((self_score - other_score) / other_score) * 100
        else:
            comparison.better_architecture = other.result_id
            comparison.improvement_percentage = ((other_score - self_score) / self_score) * 100

        return comparison

class ResultManager:
    """
    Result manager for NAS/TAS systems.

    This class consolidates result management logic that was previously
    scattered across NAS and TAS implementations, providing unified
    result handling, storage, and retrieval.
    """

    def __init__(self, base_directory: Union[str, Path] = "nas_tas_results"):
        """
        Initialize result manager.

        Args:
            base_directory: Base directory for storing results
        """
        tprint_info("Initializing Result Manager")

        self.base_directory = Path(base_directory)
        tprint_debug(f"Setting up base directory: {self.base_directory}")

        with tprint_timer("directory_setup", LogLevel.DEBUG):
            self.base_directory.mkdir(parents=True, exist_ok=True)

        tprint_success(f"Base directory created: {self.base_directory}")

        self.logger = logging.getLogger(self.__class__.__name__)

        # Log configuration
        tprint_structured({
            "result_manager_config": {
                "base_directory": str(self.base_directory),
                "directory_exists": self.base_directory.exists(),
                "directory_writable": self.base_directory.is_dir() and os.access(self.base_directory, os.W_OK)
            }
        }, LogLevel.INFO)

        tprint_success("Result Manager initialized successfully")

        # Result storage
        self.results: Dict[str, UnifiedArchitectureResult] = {}
        self.result_index: Dict[str, Dict[str, Any]] = {}

        # Statistics
        self.total_results = 0
        self.successful_results = 0
        self.failed_results = 0

        tprint_info(f"Result manager initialized with base directory: {self.base_directory}")

    @tprint_logged(LogLevel.INFO, include_args=True, include_result=True)
    def store_result(self, result: UnifiedArchitectureResult, save_to_disk: bool = True) -> bool:
        """Store a result."""
        tprint_info("Storing result")

        # Log result information
        tprint_structured({
            "result_storage": {
                "result_id": result.result_id,
                "search_type": result.search_type,
                "architecture_count": result.architecture_count,
                "execution_status": result.execution_info.status.value,
                "execution_duration": result.execution_info.duration_seconds,
                "save_to_disk": save_to_disk
            }
        }, LogLevel.INFO)

        try:
            # Store in memory
            tprint_debug("Storing result in memory")
            self.results[result.result_id] = result
            self.total_results += 1
            tprint_success("Result stored in memory")

            # Update statistics
            if result.execution_info.status == ResultStatus.SUCCESS:
                self.successful_results += 1
                tprint_success("Result marked as successful")
            else:
                self.failed_results += 1
                tprint_warning("Result marked as failed")

            # Update index
            tprint_debug("Updating result index")
            self.result_index[result.result_id] = {
                'timestamp': result.result_timestamp,
                'search_type': result.search_type,
                'search_strategy': result.search_strategy,
                'architecture_count': result.architecture_count,
                'status': result.execution_info.status.value,
                'duration': result.execution_info.duration_seconds
            }
            tprint_success("Result index updated")

            # Save to disk if requested
            if save_to_disk:
                tprint_debug("Saving result to disk")
                filename = f"result_{result.result_id}.json"
                filepath = self.base_directory / filename

                with tprint_timer("disk_save", LogLevel.DEBUG):
                    result.save(filepath, format="json")

                tprint_success(f"Result saved to disk: {filepath}")
            else:
                tprint_info("Skipping disk save (save_to_disk=False)")

            # Log storage statistics
            tprint_structured({
                "storage_statistics": {
                    "total_results": self.total_results,
                    "successful_results": self.successful_results,
                    "failed_results": self.failed_results,
                    "success_rate": self.successful_results / self.total_results if self.total_results > 0 else 0.0
                }
            }, LogLevel.INFO)

            tprint_success(f"Result stored successfully: {result.result_id}")
            return True

        except Exception as e:
            tprint_error(f"Failed to store result: {e}")
            tprint_structured({
                "storage_error": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "result_id": result.result_id,
                    "timestamp": datetime.now().isoformat()
                }
            }, LogLevel.ERROR)
            return False

    @tprint_logged(LogLevel.DEBUG, include_args=True, include_result=True)
    def get_result(self, result_id: str) -> Optional[UnifiedArchitectureResult]:
        """Get a result by ID."""
        tprint_debug(f"Retrieving result: {result_id}")

        if result_id in self.results:
            tprint_success(f"Result found in memory: {result_id}")
            return self.results[result_id]

        # Try to load from disk
        tprint_debug("Result not in memory, attempting to load from disk")
        try:
            filename = f"result_{result_id}.json"
            filepath = self.base_directory / filename

            if filepath.exists():
                tprint_debug(f"Loading result from disk: {filepath}")
                with tprint_timer("disk_load", LogLevel.DEBUG):
                    result = UnifiedArchitectureResult.load(filepath, format="json")
                self.results[result_id] = result
                tprint_success(f"Result loaded from disk: {result_id}")
                return result
            else:
                tprint_warning(f"Result file not found: {filepath}")
        except Exception as e:
            tprint_error(f"Failed to load result {result_id}: {e}")
            tprint_structured({
                "load_error": {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "result_id": result_id,
                    "filepath": str(filepath) if 'filepath' in locals() else None
                }
            }, LogLevel.ERROR)

        tprint_warning(f"Result not found: {result_id}")
        return None

    def list_results(self, search_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all results with optional filtering."""
        results = []

        for result_id, index_info in self.result_index.items():
            if search_type is None or index_info['search_type'] == search_type:
                results.append({
                    'result_id': result_id,
                    **index_info
                })

        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x['timestamp'], reverse=True)
        return results

    def compare_results(self, result_id_1: str, result_id_2: str) -> Optional[ComparisonResult]:
        """Compare two results."""
        result_1 = self.get_result(result_id_1)
        result_2 = self.get_result(result_id_2)

        if not result_1 or not result_2:
            tprint_error("One or both results not found")
            return None

        try:
            comparison = result_1.compare(result_2)

            # Store comparison
            comparison_filename = f"comparison_{comparison.comparison_id}.json"
            comparison_filepath = self.base_directory / comparison_filename

            with open(comparison_filepath, 'w') as f:
                json.dump(comparison.to_dict(), f, indent=2, default=str)

            tprint_success(f"Results compared: {result_id_1} vs {result_id_2}")
            return comparison

        except Exception as e:
            tprint_error(f"Failed to compare results: {e}")
            return None

    def get_best_result(self, search_type: Optional[str] = None) -> Optional[UnifiedArchitectureResult]:
        """Get the best result based on performance."""
        best_result = None
        best_score = -float('inf')

        for result in self.results.values():
            if search_type and result.search_type != search_type:
                continue

            if result.best_architecture:
                score = result._calculate_architecture_score(result.best_architecture)
                if score > best_score:
                    best_score = score
                    best_result = result

        return best_result

    def get_statistics(self) -> Dict[str, Any]:
        """Get result manager statistics."""
        return {
            'total_results': self.total_results,
            'successful_results': self.successful_results,
            'failed_results': self.failed_results,
            'success_rate': self.successful_results / max(self.total_results, 1),
            'results_by_type': self._get_results_by_type(),
            'average_duration': self._get_average_duration()
        }

    def _get_results_by_type(self) -> Dict[str, int]:
        """Get results count by type."""
        type_counts = {}
        for index_info in self.result_index.values():
            search_type = index_info['search_type']
            type_counts[search_type] = type_counts.get(search_type, 0) + 1
        return type_counts

    def _get_average_duration(self) -> float:
        """Get average execution duration."""
        durations = [info['duration'] for info in self.result_index.values() if info['duration'] > 0]
        return sum(durations) / len(durations) if durations else 0.0

    def cleanup_old_results(self, days_old: int = 30) -> int:
        """Clean up results older than specified days."""
        cutoff_date = datetime.now() - timedelta(days=days_old)
        cleaned_count = 0

        for result_id, index_info in list(self.result_index.items()):
            if index_info['timestamp'] < cutoff_date:
                try:
                    # Remove from memory
                    del self.results[result_id]
                    del self.result_index[result_id]

                    # Remove from disk
                    filename = f"result_{result_id}.json"
                    filepath = self.base_directory / filename
                    if filepath.exists():
                        filepath.unlink()

                    cleaned_count += 1

                except Exception as e:
                    tprint_error(f"Failed to clean up result {result_id}: {e}")

        if cleaned_count > 0:
            tprint_info(f"Cleaned up {cleaned_count} old results")

        return cleaned_count

    def export_results_summary(self, filepath: Union[str, Path]) -> bool:
        """Export results summary to CSV."""
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Prepare data
            data = []
            for result_id, index_info in self.result_index.items():
                result = self.results.get(result_id)
                row = {
                    'result_id': result_id,
                    'timestamp': index_info['timestamp'].isoformat(),
                    'search_type': index_info['search_type'],
                    'search_strategy': index_info['search_strategy'],
                    'architecture_count': index_info['architecture_count'],
                    'status': index_info['status'],
                    'duration_seconds': index_info['duration']
                }

                if result and result.best_architecture:
                    row.update(result.best_architecture.performance_metrics)
                    if result.best_architecture.financial_metrics:
                        row['sharpe_ratio'] = result.best_architecture.financial_metrics.sharpe_ratio
                        row['max_drawdown'] = result.best_architecture.financial_metrics.max_drawdown
                        row['win_rate'] = result.best_architecture.financial_metrics.win_rate

                data.append(row)

            # Create DataFrame and save
            df = pd.DataFrame(data)
            df.to_csv(filepath, index=False)

            tprint_success(f"Results summary exported to {filepath}")
            return True

        except Exception as e:
            tprint_error(f"Failed to export results summary: {e}")
            return False
