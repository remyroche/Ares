"""
Dynamic Configuration Calculator for ML Training

This module calculates optimal training parameters dynamically based on:
- Dataset size and complexity
- Available hardware resources
- Execution mode (light/full/production)
- Model type and architecture
- Timeframe and temporal requirements

Usage:
    calculator = DynamicConfigCalculator()
    config = calculator.calculate_all_parameters(
        total_samples=50000,
        n_features=100,
        timeframe='15m',
        execution_mode='full'
    )
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_success

# Import hardware utilities
from src.utils.hardware.unified_hardware_manager import (
    get_unified_hardware_manager, WorkloadType, OptimizationLevel
)
from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
from src.utils.hardware.m1_gpu_utils import get_m1_gpu_manager


@dataclass
class DynamicTrainingConfig:
    """Container for dynamically calculated training configuration."""
    # Data splits (already implemented)
    training_samples: int
    validation_samples: int
    test_samples: int
    cv_folds: int
    
    # Batch and epochs
    batch_size: int
    epochs: int
    early_stopping_patience: int
    
    # Model complexity
    n_estimators: int  # For gradient boosting
    iterations: int    # For CatBoost
    sequence_length: int  # For time series models
    
    # Learning
    learning_rate: float
    learning_rate_schedule: str
    
    # HPO
    hpo_max_trials: int
    hpo_time_budget_seconds: int
    
    # Hardware
    memory_limit_gb: float
    memory_limit_mb: int
    max_workers: int
    
    # Validation
    validation_frequency: int  # How often to validate during training
    checkpoint_frequency: int  # How often to save checkpoints


class DynamicConfigCalculator:
    """
    Calculate optimal training parameters dynamically.
    
    This class analyzes the dataset, hardware, and execution context
    to determine optimal hyperparameters for training.
    """
    
    def __init__(self):
        """Initialize the dynamic config calculator with hardware utilities."""
        self.logger = system_logger.getChild('DynamicConfigCalculator')
        
        # Initialize hardware managers
        try:
            self.hardware_manager = get_unified_hardware_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            self.gpu_manager = get_m1_gpu_manager()
        except Exception as e:
            self.logger.warning(f"Failed to initialize hardware managers: {e}")
            self.hardware_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
            self.gpu_manager = None
        
        self._hardware_info = self._get_hardware_info()
        
    def _get_hardware_info(self) -> Dict[str, Any]:
        """Get available hardware information using hardware utilities."""
        try:
            if self.hardware_manager:
                # Use unified hardware manager to get system info
                hw_status = self.hardware_manager.get_system_status()
                
                return {
                    'total_memory_gb': hw_status.get('memory', {}).get('total_gb', 16.0),
                    'available_memory_gb': hw_status.get('memory', {}).get('available_gb', 8.0),
                    'cpu_cores': hw_status.get('cpu', {}).get('total_cores', 8),
                    'cpu_threads': hw_status.get('cpu', {}).get('total_cores', 8),
                    'has_gpu': hw_status.get('gpu', {}).get('available', False),
                    'gpu_memory_gb': hw_status.get('gpu', {}).get('memory_gb', 0.0)
                }
            else:
                # Fallback to basic detection
                import psutil
                memory = psutil.virtual_memory()
                cpu_count = psutil.cpu_count(logical=False) or 4
                
                return {
                    'total_memory_gb': memory.total / (1024**3),
                    'available_memory_gb': memory.available / (1024**3),
                    'cpu_cores': cpu_count,
                    'cpu_threads': psutil.cpu_count(logical=True) or 8,
                    'has_gpu': False,
                    'gpu_memory_gb': 0.0
                }
        except Exception as e:
            self.logger.warning(f"Failed to get hardware info: {e}")
            return {
                'total_memory_gb': 8.0,
                'available_memory_gb': 4.0,
                'cpu_cores': 4,
                'cpu_threads': 8,
                'has_gpu': False,
                'gpu_memory_gb': 0.0
            }
    
    def calculate_all_parameters(
        self,
        total_samples: int,
        n_features: int,
        timeframe: str = '15m',
        execution_mode: str = 'full',
        model_type: str = 'ensemble',
        training_type: str = 'analyst_base',
        train_percentage: float = 0.70,
        validation_percentage: float = 0.15,
        test_percentage: float = 0.15
    ) -> DynamicTrainingConfig:
        """
        Calculate all dynamic parameters.
        
        Args:
            total_samples: Total number of samples in dataset
            n_features: Number of features
            timeframe: Trading timeframe (e.g., '15m', '1h')
            execution_mode: 'light', 'full', or 'production'
            model_type: Type of model being trained
            training_type: 'analyst_base', 'tactician_base', etc.
            train_percentage: Percentage for training split
            validation_percentage: Percentage for validation split
            test_percentage: Percentage for test split
            
        Returns:
            DynamicTrainingConfig with all calculated parameters
        """
        tprint_info(f"Calculating dynamic configuration for {total_samples} samples, {n_features} features")
        
        # Calculate data splits
        train_samples = int(total_samples * train_percentage)
        val_samples = int(total_samples * validation_percentage)
        test_samples = total_samples - train_samples - val_samples
        
        # Calculate CV folds
        cv_folds = self._calculate_cv_folds(total_samples, execution_mode)
        
        # Calculate batch size
        batch_size = self._calculate_batch_size(
            train_samples, self._hardware_info['available_memory_gb'], model_type
        )
        
        # Calculate epochs and patience
        epochs = self._calculate_epochs(total_samples, execution_mode, model_type)
        early_stopping_patience = self._calculate_early_stopping_patience(
            total_samples, epochs, cv_folds
        )
        
        # Calculate estimators for tree-based models
        n_estimators = self._calculate_estimators(total_samples, n_features, execution_mode)
        iterations = n_estimators  # Same for CatBoost
        
        # Calculate sequence length for time series models
        sequence_length = self._calculate_sequence_length(timeframe, training_type)
        
        # Calculate learning rate
        learning_rate = self._calculate_learning_rate(model_type, total_samples)
        learning_rate_schedule = self._determine_lr_schedule(total_samples, epochs)
        
        # Calculate HPO parameters
        hpo_max_trials = self._calculate_hpo_trials(
            execution_mode, self._estimate_model_complexity(n_features, model_type)
        )
        hpo_time_budget = self._calculate_hpo_time_budget(execution_mode, total_samples)
        
        # Calculate hardware limits
        memory_limit_gb = self._calculate_memory_limit()
        memory_limit_mb = int(memory_limit_gb * 1024)
        max_workers = self._calculate_max_workers()
        
        # Calculate validation and checkpoint frequencies
        validation_frequency = self._calculate_validation_frequency(train_samples, batch_size)
        checkpoint_frequency = self._calculate_checkpoint_frequency(epochs)
        
        config = DynamicTrainingConfig(
            training_samples=train_samples,
            validation_samples=val_samples,
            test_samples=test_samples,
            cv_folds=cv_folds,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            n_estimators=n_estimators,
            iterations=iterations,
            sequence_length=sequence_length,
            learning_rate=learning_rate,
            learning_rate_schedule=learning_rate_schedule,
            hpo_max_trials=hpo_max_trials,
            hpo_time_budget_seconds=hpo_time_budget,
            memory_limit_gb=memory_limit_gb,
            memory_limit_mb=memory_limit_mb,
            max_workers=max_workers,
            validation_frequency=validation_frequency,
            checkpoint_frequency=checkpoint_frequency
        )
        
        self._log_configuration(config)
        return config
    
    def _calculate_cv_folds(self, total_samples: int, execution_mode: str) -> int:
        """Calculate optimal number of CV folds based on data size and execution mode."""
        # Standard CV fold calculation based on data size (for full mode)
        if total_samples < 1000:
            base_folds = 3
        elif total_samples < 5000:
            base_folds = 5
        elif total_samples < 20000:
            base_folds = 7
        else:
            base_folds = 10  # Maximum for computational efficiency
        
        # Apply execution mode multiplier
        if execution_mode == 'light':
            # LIGHT mode: 10% of full (minimum 2 folds)
            adjusted_folds = max(2, int(base_folds * 0.1))
            tprint_info(f"LIGHT mode: Using 10% CV folds ({adjusted_folds} from {base_folds})")
            return adjusted_folds
        elif execution_mode == 'blank':
            # BLANK mode: 20% of full (minimum 2 folds)
            adjusted_folds = max(2, int(base_folds * 0.2))
            tprint_info(f"BLANK mode: Using 20% CV folds ({adjusted_folds} from {base_folds})")
            return adjusted_folds
        elif execution_mode == 'full':
            # FULL mode: 100% of folds
            tprint_info(f"FULL mode: Using 100% CV folds ({base_folds})")
            return base_folds
        else:  # production
            # PRODUCTION mode: Use full folds
            tprint_info(f"PRODUCTION mode: Using 100% CV folds ({base_folds})")
            return base_folds
    
    def _calculate_batch_size(
        self, 
        train_samples: int, 
        available_memory_gb: float,
        model_type: str
    ) -> int:
        """Calculate optimal batch size."""
        # Base calculation on data size
        if train_samples < 1000:
            base_batch = 32
        elif train_samples < 10000:
            base_batch = 64
        elif train_samples < 50000:
            base_batch = 128
        else:
            base_batch = 256
        
        # Adjust for neural networks (need larger batches for stability)
        if any(nn in model_type.lower() for nn in ['gru', 'lstm', 'tcn', 'transformer']):
            base_batch = max(base_batch, 64)
        
        # Adjust for available memory (assuming ~1GB per 100 batch size for neural nets)
        if 'neural' in model_type.lower() or 'tcn' in model_type.lower():
            max_batch_by_memory = int(available_memory_gb * 100)
            base_batch = min(base_batch, max_batch_by_memory)
        
        # Ensure power of 2 for optimal GPU utilization
        return 2 ** int(np.log2(base_batch))
    
    def _calculate_epochs(self, total_samples: int, execution_mode: str, model_type: str) -> int:
        """Calculate optimal number of training epochs."""
        # Base epochs on full mode (100%)
        base_epochs = 100  # Full mode baseline
        
        # Adjust based on data size (more data = fewer epochs needed)
        if total_samples > 50000:
            base_epochs = int(base_epochs * 0.75)
        elif total_samples < 5000:
            base_epochs = int(base_epochs * 1.5)
        
        # Tree-based models don't use epochs, but iterations
        if any(tree in model_type.lower() for tree in ['lgbm', 'catboost', 'xgboost']):
            return 0  # Will use n_estimators instead
        
        # Apply execution mode multiplier
        if execution_mode == 'light':
            # LIGHT mode: 10% of full
            adjusted_epochs = max(5, int(base_epochs * 0.1))
            tprint_info(f"LIGHT mode: Using 10% epochs ({adjusted_epochs} from {base_epochs})")
            return adjusted_epochs
        elif execution_mode == 'blank':
            # BLANK mode: 20% of full
            adjusted_epochs = max(10, int(base_epochs * 0.2))
            tprint_info(f"BLANK mode: Using 20% epochs ({adjusted_epochs} from {base_epochs})")
            return adjusted_epochs
        elif execution_mode == 'full':
            # FULL mode: 100% of epochs
            tprint_info(f"FULL mode: Using 100% epochs ({base_epochs})")
            return base_epochs
        else:  # production
            # PRODUCTION mode: 200% (extended training)
            production_epochs = int(base_epochs * 2.0)
            tprint_info(f"PRODUCTION mode: Using 200% epochs ({production_epochs})")
            return production_epochs
    
    def _calculate_early_stopping_patience(
        self, 
        total_samples: int, 
        epochs: int,
        cv_folds: int
    ) -> int:
        """Calculate early stopping patience."""
        # Base patience: 10% of total epochs
        if epochs > 0:
            base_patience = max(10, int(epochs * 0.1))
        else:
            base_patience = 100  # For tree-based models (iterations)
        
        # More data = can afford more patience
        if total_samples > 50000:
            base_patience = int(base_patience * 1.5)
        
        # More CV folds = more patience (each fold trains separately)
        patience_boost = (cv_folds - 3) * 2
        
        return max(5, base_patience + patience_boost)
    
    def _calculate_estimators(
        self, 
        total_samples: int, 
        n_features: int,
        execution_mode: str
    ) -> int:
        """Calculate optimal number of estimators for tree-based models."""
        # Base estimators for full mode (100%)
        base_estimators = 1000  # Full mode baseline
        
        # Scale with data size
        if total_samples > 50000:
            base_estimators = int(base_estimators * 1.5)
        elif total_samples < 5000:
            base_estimators = int(base_estimators * 0.7)
        
        # Scale with feature complexity
        if n_features > 200:
            base_estimators = int(base_estimators * 1.3)
        elif n_features < 50:
            base_estimators = int(base_estimators * 0.8)
        
        # Ensure minimum
        base_estimators = max(100, base_estimators)
        
        # Apply execution mode multiplier
        if execution_mode == 'light':
            # LIGHT mode: 10% of full
            adjusted_estimators = max(50, int(base_estimators * 0.1))
            tprint_info(f"LIGHT mode: Using 10% estimators ({adjusted_estimators} from {base_estimators})")
            return adjusted_estimators
        elif execution_mode == 'blank':
            # BLANK mode: 20% of full
            adjusted_estimators = max(100, int(base_estimators * 0.2))
            tprint_info(f"BLANK mode: Using 20% estimators ({adjusted_estimators} from {base_estimators})")
            return adjusted_estimators
        elif execution_mode == 'full':
            # FULL mode: 100% of estimators
            tprint_info(f"FULL mode: Using 100% estimators ({base_estimators})")
            return base_estimators
        else:  # production
            # PRODUCTION mode: 200% (extended training)
            production_estimators = int(base_estimators * 2.0)
            tprint_info(f"PRODUCTION mode: Using 200% estimators ({production_estimators})")
            return production_estimators
    
    def _calculate_sequence_length(self, timeframe: str, training_type: str) -> int:
        """Calculate optimal sequence length for time series models."""
        # Map timeframe to minutes
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440
        }
        
        minutes = timeframe_minutes.get(timeframe, 15)
        
        # Different lookback for analyst vs tactician
        if 'analyst' in training_type.lower():
            lookback_hours = 24  # Analyst looks at longer trends
        else:
            lookback_hours = 6   # Tactician focuses on shorter timing
        
        # Calculate sequences needed
        sequences = (lookback_hours * 60) // minutes
        
        # Clamp to reasonable range (20-200)
        return min(max(sequences, 20), 200)
    
    def _calculate_learning_rate(self, model_type: str, total_samples: int) -> float:
        """Calculate initial learning rate."""
        # Neural networks
        if any(nn in model_type.lower() for nn in ['gru', 'lstm', 'tcn', 'transformer']):
            base_lr = 0.001
            # Larger datasets can handle slightly larger LR
            if total_samples > 50000:
                base_lr = 0.002
        # Gradient boosting
        elif 'lgbm' in model_type.lower():
            base_lr = 0.05
            if total_samples < 5000:
                base_lr = 0.1
        elif 'catboost' in model_type.lower():
            base_lr = 0.08
            if total_samples < 5000:
                base_lr = 0.12
        else:
            base_lr = 0.01
        
        return base_lr
    
    def _determine_lr_schedule(self, total_samples: int, epochs: int) -> str:
        """Determine learning rate schedule."""
        if epochs == 0:
            return 'none'  # Tree-based models
        
        if total_samples < 5000:
            return 'constant'
        elif total_samples < 20000:
            return 'reduce_on_plateau'
        else:
            return 'cosine_annealing'
    
    def _calculate_hpo_trials(self, execution_mode: str, model_complexity: str) -> int:
        """Calculate number of HPO trials based on execution mode and model complexity."""
        trials_map = {
            'blank': {'low': 3, 'medium': 5, 'high': 8},     # Minimal HPO for blank mode
            'light': {'low': 5, 'medium': 10, 'high': 15},
            'full': {'low': 20, 'medium': 50, 'high': 100},
            'production': {'low': 50, 'medium': 100, 'high': 200}
        }
        
        mode_trials = trials_map.get(execution_mode, trials_map['light'])
        base_trials = mode_trials.get(model_complexity, 20)
        
        # Apply execution mode reduction
        if execution_mode == 'blank':
            # Already minimal in trials_map
            tprint_info(f"BLANK mode: Using minimal HPO trials ({base_trials})")
        elif execution_mode == 'light':
            # Already reduced in trials_map
            tprint_info(f"LIGHT mode: Using reduced HPO trials ({base_trials})")
        
        return base_trials
    
    def _calculate_hpo_time_budget(self, execution_mode: str, total_samples: int) -> int:
        """Calculate time budget for HPO in seconds."""
        # Base time budget
        if execution_mode == 'blank':
            base_time = 60  # 1 minute - minimal for blank mode
        elif execution_mode == 'light':
            base_time = 300  # 5 minutes
        elif execution_mode == 'full':
            base_time = 1800  # 30 minutes
        else:  # production
            base_time = 7200  # 2 hours
        
        # Scale with data size (more data = more time needed per trial)
        if total_samples > 50000:
            base_time = int(base_time * 1.5)
        
        return base_time
    
    def _estimate_model_complexity(self, n_features: int, model_type: str) -> str:
        """Estimate model complexity (low/medium/high)."""
        # Neural networks are inherently more complex
        if any(nn in model_type.lower() for nn in ['gru', 'lstm', 'tcn', 'transformer']):
            if n_features > 100:
                return 'high'
            else:
                return 'medium'
        
        # Tree-based models scale with features
        if n_features < 50:
            return 'low'
        elif n_features < 150:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_memory_limit(self) -> float:
        """Calculate memory limit for training using memory optimizer."""
        try:
            if self.memory_optimizer:
                # Use memory optimizer to get optimal memory allocation
                memory_stats = self.memory_optimizer.get_memory_stats()
                available_gb = memory_stats.get('available_memory_gb', 4.0)
                
                # Request optimal memory allocation for ML training
                optimal_memory = self.memory_optimizer.calculate_optimal_allocation(
                    workload_type='ml_training',
                    requested_gb=available_gb * 0.7
                )
                
                return min(max(optimal_memory, 1.0), 16.0)
            else:
                # Fallback calculation
                available_gb = self._hardware_info['available_memory_gb']
                training_memory = available_gb * 0.7
                return min(max(training_memory, 1.0), 16.0)
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate optimal memory: {e}")
            available_gb = self._hardware_info['available_memory_gb']
            return min(max(available_gb * 0.7, 1.0), 16.0)
    
    def _calculate_max_workers(self) -> int:
        """Calculate maximum number of parallel workers using CPU optimizer."""
        try:
            if self.cpu_optimizer:
                # Use CPU optimizer to get optimal worker count
                optimal_workers = self.cpu_optimizer.get_optimal_worker_count(
                    workload_type='ml_training'
                )
                return min(optimal_workers, 8)  # Cap at 8 for efficiency
            else:
                # Fallback calculation
                cpu_cores = self._hardware_info['cpu_cores']
                max_workers = max(1, int(cpu_cores * 0.75))
                return min(max_workers, 8)
                
        except Exception as e:
            self.logger.warning(f"Failed to calculate optimal workers: {e}")
            cpu_cores = self._hardware_info['cpu_cores']
            return min(max(1, int(cpu_cores * 0.75)), 8)
    
    def _calculate_validation_frequency(self, train_samples: int, batch_size: int) -> int:
        """Calculate how often to validate during training (in batches)."""
        # Validate every ~1000 samples or every 10 batches, whichever is larger
        samples_per_validation = 1000
        batches_per_validation = max(10, samples_per_validation // batch_size)
        
        return batches_per_validation
    
    def _calculate_checkpoint_frequency(self, epochs: int) -> int:
        """Calculate how often to save checkpoints (in epochs)."""
        if epochs == 0:
            return 100  # For tree-based (iterations)
        
        # Save every 10 epochs, or every 10% of training
        return max(5, min(10, int(epochs * 0.1)))
    
    def _log_configuration(self, config: DynamicTrainingConfig) -> None:
        """Log the calculated configuration."""
        tprint_success("✅ Dynamic configuration calculated:")
        tprint_info(f"  Data Splits: Train={config.training_samples}, Val={config.validation_samples}, Test={config.test_samples}")
        tprint_info(f"  CV Folds: {config.cv_folds}")
        tprint_info(f"  Batch Size: {config.batch_size}")
        tprint_info(f"  Epochs: {config.epochs if config.epochs > 0 else 'N/A (tree-based)'}")
        tprint_info(f"  Early Stopping Patience: {config.early_stopping_patience}")
        tprint_info(f"  Estimators: {config.n_estimators}")
        tprint_info(f"  Sequence Length: {config.sequence_length}")
        tprint_info(f"  Learning Rate: {config.learning_rate} ({config.learning_rate_schedule})")
        tprint_info(f"  HPO Trials: {config.hpo_max_trials} (Budget: {config.hpo_time_budget_seconds}s)")
        tprint_info(f"  Memory Limit: {config.memory_limit_gb:.2f} GB")
        tprint_info(f"  Max Workers: {config.max_workers}")
        tprint_info(f"  Validation Frequency: Every {config.validation_frequency} batches")
        tprint_info(f"  Checkpoint Frequency: Every {config.checkpoint_frequency} epochs/iterations")
    
    def to_dict(self, config: DynamicTrainingConfig) -> Dict[str, Any]:
        """
        Convert DynamicTrainingConfig to dictionary.
        
        Args:
            config: DynamicTrainingConfig instance
            
        Returns:
            Dictionary representation of the config
        """
        return asdict(config)


# Convenience function
def calculate_dynamic_config(
    total_samples: int,
    n_features: int,
    **kwargs
) -> DynamicTrainingConfig:
    """
    Convenience function to calculate dynamic configuration.
    
    Args:
        total_samples: Total number of samples
        n_features: Number of features
        **kwargs: Additional parameters
        
    Returns:
        DynamicTrainingConfig with all calculated parameters
    """
    calculator = DynamicConfigCalculator()
    return calculator.calculate_all_parameters(total_samples, n_features, **kwargs)

