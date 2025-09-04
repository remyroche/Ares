#!/usr/bin/env python3
"""
Pipeline State Management and Checkpoint System

This module provides comprehensive state management and checkpoint validation
for the Ares trading pipeline, ensuring reliable execution and recovery.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set
import hashlib
import pickle

from src.core.decorators.errors import handles_errors, error_boundary
from src.core.decorators.logging import logs_execution
from src.utils.common_operations import (
    get_current_datetime,
    format_datetime,
    safe_file_exists,
    safe_json_load,
    safe_json_dump,
    ensure_directory
)


class CheckpointStatus(Enum):
    """Checkpoint status enumeration."""
    VALID = "valid"
    INVALID = "invalid"
    CORRUPTED = "corrupted"
    OUTDATED = "outdated"
    MISSING = "missing"


class PipelineState(Enum):
    """Pipeline execution state."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CheckpointMetadata:
    """Metadata for pipeline checkpoints."""
    checkpoint_id: str
    pipeline_name: str
    step_name: str
    timestamp: str
    state: PipelineState
    progress_percentage: float
    data_hash: str
    file_size: int
    dependencies: List[str] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CheckpointMetadata':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class PipelineExecutionState:
    """Complete pipeline execution state."""
    pipeline_id: str
    pipeline_name: str
    state: PipelineState
    start_time: str
    last_update: str
    current_step: Optional[str] = None
    progress_percentage: float = 0.0
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    checkpoints: List[CheckpointMetadata] = field(default_factory=list)
    configuration: Dict[str, Any] = field(default_factory=dict)
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineExecutionState':
        """Create from dictionary."""
        return cls(**data)


class CheckpointValidator:
    """Validates checkpoint integrity and dependencies."""
    
    def __init__(self):
        self.logger = logging.getLogger("checkpoint_validator")
    
    @handles_errors(Exception, fallback=CheckpointStatus.INVALID)
    def validate_checkpoint(
        self,
        checkpoint_path: str,
        metadata: CheckpointMetadata
    ) -> CheckpointStatus:
        """Validate a checkpoint file and its metadata."""
        
        try:
            # Check if file exists
            if not safe_file_exists(checkpoint_path):
                self.logger.warning(f"Checkpoint file not found: {checkpoint_path}")
                return CheckpointStatus.MISSING
            
            # Check file size
            actual_size = Path(checkpoint_path).stat().st_size
            if actual_size != metadata.file_size:
                self.logger.warning(f"File size mismatch: expected {metadata.file_size}, got {actual_size}")
                return CheckpointStatus.CORRUPTED
            
            # Check data hash
            actual_hash = self._calculate_file_hash(checkpoint_path)
            if actual_hash != metadata.data_hash:
                self.logger.warning(f"Data hash mismatch: expected {metadata.data_hash}, got {actual_hash}")
                return CheckpointStatus.CORRUPTED
            
            # Check timestamp freshness
            if self._is_checkpoint_outdated(metadata.timestamp):
                self.logger.warning(f"Checkpoint is outdated: {metadata.timestamp}")
                return CheckpointStatus.OUTDATED
            
            # Validate dependencies
            if not self._validate_dependencies(metadata.dependencies):
                self.logger.warning(f"Dependency validation failed for checkpoint {metadata.checkpoint_id}")
                return CheckpointStatus.INVALID
            
            self.logger.info(f"Checkpoint {metadata.checkpoint_id} is valid")
            return CheckpointStatus.VALID
            
        except Exception as e:
            self.logger.error(f"Checkpoint validation failed: {e}")
            return CheckpointStatus.INVALID
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file."""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def _is_checkpoint_outdated(self, timestamp: str, max_age_hours: int = 24) -> bool:
        """Check if checkpoint is too old."""
        try:
            from datetime import datetime, timedelta
            checkpoint_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            max_age = timedelta(hours=max_age_hours)
            return datetime.now() - checkpoint_time > max_age
        except Exception:
            return True  # If we can't parse timestamp, consider it outdated
    
    def _validate_dependencies(self, dependencies: List[str]) -> bool:
        """Validate that all dependencies exist and are valid."""
        for dep in dependencies:
            if not safe_file_exists(dep):
                self.logger.warning(f"Dependency not found: {dep}")
                return False
        return True


class CheckpointManager:
    """Manages pipeline checkpoints with validation and recovery."""
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.validator = CheckpointValidator()
        self.logger = logging.getLogger("checkpoint_manager")
        self.checkpoint_registry: Dict[str, CheckpointMetadata] = {}
        
        # Ensure checkpoint directory exists
        ensure_directory(self.checkpoint_dir)
    
    @handles_errors(Exception, fallback=False)
    @logs_execution("checkpoint_creation")
    def create_checkpoint(
        self,
        pipeline_id: str,
        step_name: str,
        data: Any,
        state: PipelineState,
        progress_percentage: float,
        dependencies: Optional[List[str]] = None,
        validation_results: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create a new checkpoint."""
        
        try:
            # Generate checkpoint ID
            checkpoint_id = f"{pipeline_id}_{step_name}_{int(time.time())}"
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            
            # Save data
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(data, f)
            
            # Calculate metadata
            file_size = checkpoint_path.stat().st_size
            data_hash = self.validator._calculate_file_hash(str(checkpoint_path))
            
            # Create metadata
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                pipeline_name=pipeline_id,
                step_name=step_name,
                timestamp=format_datetime(get_current_datetime()),
                state=state,
                progress_percentage=progress_percentage,
                data_hash=data_hash,
                file_size=file_size,
                dependencies=dependencies or [],
                validation_results=validation_results or {},
                performance_metrics=performance_metrics or {}
            )
            
            # Save metadata
            metadata_path = self.checkpoint_dir / f"{checkpoint_id}_metadata.json"
            safe_json_dump(metadata.to_dict(), metadata_path, indent=2)
            
            # Register checkpoint
            self.checkpoint_registry[checkpoint_id] = metadata
            
            self.logger.info(f"Created checkpoint {checkpoint_id} at {checkpoint_path}")
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            raise
    
    @handles_errors(Exception, fallback=None)
    def load_checkpoint(self, checkpoint_id: str) -> Optional[Any]:
        """Load checkpoint data."""
        
        try:
            # Get metadata
            metadata = self.checkpoint_registry.get(checkpoint_id)
            if not metadata:
                # Try to load from file
                metadata_path = self.checkpoint_dir / f"{checkpoint_id}_metadata.json"
                if safe_file_exists(metadata_path):
                    metadata_data = safe_json_load(metadata_path)
                    metadata = CheckpointMetadata.from_dict(metadata_data)
                    self.checkpoint_registry[checkpoint_id] = metadata
                else:
                    self.logger.error(f"Checkpoint metadata not found: {checkpoint_id}")
                    return None
            
            # Validate checkpoint
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            validation_status = self.validator.validate_checkpoint(str(checkpoint_path), metadata)
            
            if validation_status != CheckpointStatus.VALID:
                self.logger.error(f"Checkpoint validation failed: {validation_status}")
                return None
            
            # Load data
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            
            self.logger.info(f"Loaded checkpoint {checkpoint_id}")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_id}: {e}")
            return None
    
    @handles_errors(Exception, fallback=False)
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint and its metadata."""
        
        try:
            # Remove files
            checkpoint_path = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            metadata_path = self.checkpoint_dir / f"{checkpoint_id}_metadata.json"
            
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            if metadata_path.exists():
                metadata_path.unlink()
            
            # Remove from registry
            if checkpoint_id in self.checkpoint_registry:
                del self.checkpoint_registry[checkpoint_id]
            
            self.logger.info(f"Deleted checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoint {checkpoint_id}: {e}")
            return False
    
    @handles_errors(Exception, fallback=[])
    def list_checkpoints(self, pipeline_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """List all checkpoints, optionally filtered by pipeline ID."""
        
        try:
            checkpoints = []
            
            # Load from registry first
            for metadata in self.checkpoint_registry.values():
                if pipeline_id is None or metadata.pipeline_name == pipeline_id:
                    checkpoints.append(metadata)
            
            # Also check for metadata files not in registry
            for metadata_file in self.checkpoint_dir.glob("*_metadata.json"):
                checkpoint_id = metadata_file.stem.replace("_metadata", "")
                
                if checkpoint_id not in self.checkpoint_registry:
                    try:
                        metadata_data = safe_json_load(metadata_file)
                        metadata = CheckpointMetadata.from_dict(metadata_data)
                        self.checkpoint_registry[checkpoint_id] = metadata
                        
                        if pipeline_id is None or metadata.pipeline_name == pipeline_id:
                            checkpoints.append(metadata)
                    except Exception as e:
                        self.logger.warning(f"Could not load metadata from {metadata_file}: {e}")
            
            # Sort by timestamp
            checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
            
            return checkpoints
            
        except Exception as e:
            self.logger.error(f"Failed to list checkpoints: {e}")
            return []
    
    @handles_errors(Exception, fallback=None)
    def get_latest_checkpoint(self, pipeline_id: str, step_name: Optional[str] = None) -> Optional[CheckpointMetadata]:
        """Get the latest checkpoint for a pipeline and optionally a specific step."""
        
        try:
            checkpoints = self.list_checkpoints(pipeline_id)
            
            if step_name:
                checkpoints = [c for c in checkpoints if c.step_name == step_name]
            
            if not checkpoints:
                return None
            
            return checkpoints[0]  # Already sorted by timestamp
            
        except Exception as e:
            self.logger.error(f"Failed to get latest checkpoint: {e}")
            return None


class PipelineStateManager:
    """Comprehensive pipeline state management system."""
    
    def __init__(self, state_dir: str = "pipeline_states"):
        self.state_dir = Path(state_dir)
        self.checkpoint_manager = CheckpointManager()
        self.logger = logging.getLogger("pipeline_state_manager")
        self.active_states: Dict[str, PipelineExecutionState] = {}
        
        # Ensure state directory exists
        ensure_directory(self.state_dir)
    
    @handles_errors(Exception, fallback=False)
    @logs_execution("pipeline_state_initialization")
    def initialize_pipeline(
        self,
        pipeline_id: str,
        pipeline_name: str,
        configuration: Dict[str, Any]
    ) -> PipelineExecutionState:
        """Initialize a new pipeline execution state."""
        
        try:
            state = PipelineExecutionState(
                pipeline_id=pipeline_id,
                pipeline_name=pipeline_name,
                state=PipelineState.INITIALIZED,
                start_time=format_datetime(get_current_datetime()),
                last_update=format_datetime(get_current_datetime()),
                configuration=configuration
            )
            
            self.active_states[pipeline_id] = state
            self._save_pipeline_state(state)
            
            self.logger.info(f"Initialized pipeline state for {pipeline_id}")
            return state
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline state: {e}")
            raise
    
    @handles_errors(Exception, fallback=False)
    def update_pipeline_state(
        self,
        pipeline_id: str,
        state: PipelineState,
        current_step: Optional[str] = None,
        progress_percentage: Optional[float] = None,
        error: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update pipeline execution state."""
        
        try:
            if pipeline_id not in self.active_states:
                self.logger.error(f"Pipeline state not found: {pipeline_id}")
                return False
            
            pipeline_state = self.active_states[pipeline_id]
            
            # Update state
            pipeline_state.state = state
            pipeline_state.last_update = format_datetime(get_current_datetime())
            
            if current_step is not None:
                pipeline_state.current_step = current_step
            
            if progress_percentage is not None:
                pipeline_state.progress_percentage = progress_percentage
            
            # Handle step completion
            if state == PipelineState.COMPLETED and current_step:
                if current_step not in pipeline_state.completed_steps:
                    pipeline_state.completed_steps.append(current_step)
            
            # Handle step failure
            if state == PipelineState.FAILED and current_step:
                if current_step not in pipeline_state.failed_steps:
                    pipeline_state.failed_steps.append(current_step)
            
            # Add error to history
            if error:
                error_entry = {
                    "timestamp": format_datetime(get_current_datetime()),
                    "step": current_step,
                    "error": error
                }
                pipeline_state.error_history.append(error_entry)
            
            # Save state
            self._save_pipeline_state(pipeline_state)
            
            self.logger.info(f"Updated pipeline state for {pipeline_id}: {state.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update pipeline state: {e}")
            return False
    
    @handles_errors(Exception, fallback=False)
    def create_step_checkpoint(
        self,
        pipeline_id: str,
        step_name: str,
        data: Any,
        dependencies: Optional[List[str]] = None,
        validation_results: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Create a checkpoint for a specific step."""
        
        try:
            if pipeline_id not in self.active_states:
                self.logger.error(f"Pipeline state not found: {pipeline_id}")
                return None
            
            pipeline_state = self.active_states[pipeline_id]
            
            # Create checkpoint
            checkpoint_id = self.checkpoint_manager.create_checkpoint(
                pipeline_id=pipeline_id,
                step_name=step_name,
                data=data,
                state=pipeline_state.state,
                progress_percentage=pipeline_state.progress_percentage,
                dependencies=dependencies,
                validation_results=validation_results,
                performance_metrics=performance_metrics
            )
            
            # Add to pipeline state
            checkpoint_metadata = self.checkpoint_manager.checkpoint_registry[checkpoint_id]
            pipeline_state.checkpoints.append(checkpoint_metadata)
            
            # Update performance metrics
            if performance_metrics:
                pipeline_state.performance_metrics.update(performance_metrics)
            
            # Save state
            self._save_pipeline_state(pipeline_state)
            
            self.logger.info(f"Created step checkpoint {checkpoint_id} for {pipeline_id}")
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Failed to create step checkpoint: {e}")
            return None
    
    @handles_errors(Exception, fallback=None)
    def load_step_checkpoint(
        self,
        pipeline_id: str,
        step_name: str
    ) -> Optional[Any]:
        """Load checkpoint for a specific step."""
        
        try:
            # Get latest checkpoint for the step
            checkpoint_metadata = self.checkpoint_manager.get_latest_checkpoint(
                pipeline_id, step_name
            )
            
            if not checkpoint_metadata:
                self.logger.warning(f"No checkpoint found for {pipeline_id}:{step_name}")
                return None
            
            # Load checkpoint data
            data = self.checkpoint_manager.load_checkpoint(checkpoint_metadata.checkpoint_id)
            
            if data is not None:
                self.logger.info(f"Loaded step checkpoint for {pipeline_id}:{step_name}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load step checkpoint: {e}")
            return None
    
    @handles_errors(Exception, fallback=False)
    def resume_pipeline_from_checkpoint(
        self,
        pipeline_id: str,
        step_name: str
    ) -> bool:
        """Resume pipeline execution from a specific checkpoint."""
        
        try:
            if pipeline_id not in self.active_states:
                self.logger.error(f"Pipeline state not found: {pipeline_id}")
                return False
            
            pipeline_state = self.active_states[pipeline_id]
            
            # Load checkpoint
            checkpoint_data = self.load_step_checkpoint(pipeline_id, step_name)
            if checkpoint_data is None:
                self.logger.error(f"Cannot resume: no checkpoint found for {step_name}")
                return False
            
            # Update pipeline state
            pipeline_state.state = PipelineState.RUNNING
            pipeline_state.current_step = step_name
            pipeline_state.last_update = format_datetime(get_current_datetime())
            
            # Save state
            self._save_pipeline_state(pipeline_state)
            
            self.logger.info(f"Resumed pipeline {pipeline_id} from checkpoint at step {step_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to resume pipeline from checkpoint: {e}")
            return False
    
    @handles_errors(Exception, fallback=[])
    def get_pipeline_history(self, pipeline_id: str) -> List[PipelineExecutionState]:
        """Get execution history for a pipeline."""
        
        try:
            history = []
            state_files = self.state_dir.glob(f"{pipeline_id}_*.json")
            
            for state_file in sorted(state_files):
                try:
                    state_data = safe_json_load(state_file)
                    state = PipelineExecutionState.from_dict(state_data)
                    history.append(state)
                except Exception as e:
                    self.logger.warning(f"Could not load state from {state_file}: {e}")
            
            return history
            
        except Exception as e:
            self.logger.error(f"Failed to get pipeline history: {e}")
            return []
    
    @handles_errors(Exception, fallback=None)
    def _save_pipeline_state(self, state: PipelineExecutionState) -> None:
        """Save pipeline state to file."""
        
        try:
            state_file = self.state_dir / f"{state.pipeline_id}_{int(time.time())}.json"
            safe_json_dump(state.to_dict(), state_file, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save pipeline state: {e}")
    
    @handles_errors(Exception, fallback={})
    def get_pipeline_status_summary(self) -> Dict[str, Any]:
        """Get summary of all pipeline states."""
        
        try:
            summary = {
                "total_pipelines": len(self.active_states),
                "by_state": {},
                "by_pipeline": {},
                "checkpoint_summary": {
                    "total_checkpoints": len(self.checkpoint_manager.checkpoint_registry),
                    "by_pipeline": {}
                }
            }
            
            # Count by state
            for state in self.active_states.values():
                state_name = state.state.value
                summary["by_state"][state_name] = summary["by_state"].get(state_name, 0) + 1
                
                # Per-pipeline info
                summary["by_pipeline"][state.pipeline_id] = {
                    "state": state.state.value,
                    "progress": state.progress_percentage,
                    "current_step": state.current_step,
                    "completed_steps": len(state.completed_steps),
                    "failed_steps": len(state.failed_steps),
                    "checkpoints": len(state.checkpoints)
                }
            
            # Checkpoint summary
            for checkpoint in self.checkpoint_manager.checkpoint_registry.values():
                pipeline_id = checkpoint.pipeline_name
                if pipeline_id not in summary["checkpoint_summary"]["by_pipeline"]:
                    summary["checkpoint_summary"]["by_pipeline"][pipeline_id] = 0
                summary["checkpoint_summary"]["by_pipeline"][pipeline_id] += 1
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to get pipeline status summary: {e}")
            return {}


# Global state manager instance
pipeline_state_manager = PipelineStateManager()