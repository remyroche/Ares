"""
NAS-TAS Model Training Pipeline

Comprehensive model training pipeline that integrates regime detection with model training.
Provides regime-aware model training, selection, and management capabilities.
"""

from .regime_aware_trainer import RegimeAwareTrainer, RegimeAwareTrainingConfig
from .model_selector import ModelSelector, ModelSelectionConfig
from .training_orchestrator import TrainingOrchestrator, OrchestratorConfig
from .model_manager import ModelManager, ModelManagerConfig
from .performance_tracker import PerformanceTracker, PerformanceConfig

__all__ = [
    'RegimeAwareTrainer',
    'RegimeAwareTrainingConfig', 
    'ModelSelector',
    'ModelSelectionConfig',
    'TrainingOrchestrator',
    'OrchestratorConfig',
    'ModelManager',
    'ModelManagerConfig',
    'PerformanceTracker',
    'PerformanceConfig'
]