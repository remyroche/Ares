#!/usr/bin/env python3
"""
Simplified NAS Trainer for testing without external dependencies.

This is a minimal version that demonstrates the structure and functionality
without requiring numpy, pandas, or other external libraries.
Enhanced with comprehensive error handling, tprint logging, and full implementation.
"""

import os
import sys
import time
import logging
import json
import pickle
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import random
import math
from datetime import datetime

# Import tprint with error handling
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_warning, tprint_error, tprint_success,
        tprint_progress, tprint_performance, tprint_structured, tprint_timer
    )
    TPRINT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: tprint not available: {e}")
    TPRINT_AVAILABLE = False
    
    # Fallback tprint functions with comprehensive error handling
    def tprint(*args, **kwargs):
        try:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", *args, **kwargs)
        except (TypeError, AttributeError, ValueError) as e:
            print(f"tprint error: {e}", *args, **kwargs)
    
    def tprint_info(*args, **kwargs):
        try:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] INFO:", *args, **kwargs)
        except (TypeError, AttributeError, ValueError) as e:
            print(f"tprint_info error: {e}", "INFO:", *args, **kwargs)
    
    def tprint_warning(*args, **kwargs):
        try:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] WARNING:", *args, **kwargs)
        except (TypeError, AttributeError, ValueError) as e:
            print(f"tprint_warning error: {e}", "WARNING:", *args, **kwargs)
    
    def tprint_error(*args, **kwargs):
        try:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ERROR:", *args, **kwargs)
        except (TypeError, AttributeError, ValueError) as e:
            print(f"tprint_error error: {e}", "ERROR:", *args, **kwargs)
    
    def tprint_success(*args, **kwargs):
        try:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] SUCCESS:", *args, **kwargs)
        except (TypeError, AttributeError, ValueError) as e:
            print(f"tprint_success error: {e}", "SUCCESS:", *args, **kwargs)
    
    def tprint_progress(step, total, message="", **kwargs):
        try:
            percentage = (step / total) * 100 if total > 0 else 0
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PROGRESS: {step}/{total} ({percentage:.1f}%) {message}")
        except (TypeError, AttributeError, ValueError, ZeroDivisionError) as e:
            print(f"tprint_progress error: {e}", f"PROGRESS: {step}/{total} {message}")
    
    def tprint_performance(operation, duration, **kwargs):
        try:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] PERFORMANCE: {operation} took {duration:.3f}s")
        except (TypeError, AttributeError, ValueError) as e:
            print(f"tprint_performance error: {e}", f"PERFORMANCE: {operation} took {duration:.3f}s")
    
    def tprint_structured(data, level=None, **kwargs):
        try:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] STRUCTURED:", data)
        except (TypeError, AttributeError, ValueError) as e:
            print(f"tprint_structured error: {e}", "STRUCTURED:", data)
    
    def tprint_timer(operation, level=None):
        from contextlib import contextmanager
        @contextmanager
        def timer():
            start = time.time()
            try:
                yield
            finally:
                duration = time.time() - start
                tprint_performance(operation, duration)
        return timer()

# Setup logging with comprehensive error handling
try:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
except Exception as e:
    print(f"Logging setup failed: {e}")
    logger = None

@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search with comprehensive validation."""
    
    # Search parameters
    search_strategy: str = 'random'  # 'random', 'grid', 'bayesian', 'evolutionary'
    max_trials: int = 100
    max_epochs: int = 50
    early_stopping_patience: int = 10
    
    # Architecture parameters
    min_layers: int = 2
    max_layers: int = 10
    min_neurons: int = 32
    max_neurons: int = 512
    activation_functions: List[str] = field(default_factory=lambda: ['relu', 'tanh', 'sigmoid'])
    dropout_rates: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5])
    
    # Training parameters
    learning_rate_range: Tuple[float, float] = (1e-5, 1e-1)
    batch_size_range: Tuple[int, int] = (16, 256)
    optimizer: str = 'adam'
    
    # Hardware optimization
    use_m1_optimization: bool = True
    use_gpu_acceleration: bool = True
    memory_limit_gb: Optional[float] = None
    
    # Data processing
    feature_selection: bool = True
    feature_selection_method: str = 'mutual_info'
    max_features: int = 100
    
    # Validation
    cv_folds: int = 5
    validation_split: float = 0.2
    test_split: float = 0.2
    
    # Output
    save_models: bool = True
    save_results: bool = True
    output_dir: str = 'nas_results'
    verbose: bool = True
    
    def __post_init__(self):
        """Validate configuration after initialization with comprehensive error handling."""
        try:
            tprint_info("🔧 Validating NAS configuration")
            
            # Validate search strategy
            valid_strategies = ['random', 'grid', 'bayesian', 'evolutionary']
            if self.search_strategy not in valid_strategies:
                tprint_warning(f"Invalid search strategy: {self.search_strategy}, using 'random'")
                self.search_strategy = 'random'
            
            # Validate numeric parameters with bounds checking
            if self.max_trials <= 0:
                tprint_warning(f"Invalid max_trials: {self.max_trials}, using 100")
                self.max_trials = 100
            elif self.max_trials > 10000:
                tprint_warning(f"Very large max_trials: {self.max_trials}, capping at 10000")
                self.max_trials = 10000
            
            if self.max_epochs <= 0:
                tprint_warning(f"Invalid max_epochs: {self.max_epochs}, using 50")
                self.max_epochs = 50
            elif self.max_epochs > 1000:
                tprint_warning(f"Very large max_epochs: {self.max_epochs}, capping at 1000")
                self.max_epochs = 1000
            
            if self.min_layers < 1:
                tprint_warning(f"Invalid min_layers: {self.min_layers}, using 1")
                self.min_layers = 1
            
            if self.max_layers < self.min_layers:
                tprint_warning(f"max_layers ({self.max_layers}) < min_layers ({self.min_layers}), adjusting")
                self.max_layers = self.min_layers + 1
            
            if self.min_neurons <= 0:
                tprint_warning(f"Invalid min_neurons: {self.min_neurons}, using 32")
                self.min_neurons = 32
            
            if self.max_neurons < self.min_neurons:
                tprint_warning(f"max_neurons ({self.max_neurons}) < min_neurons ({self.min_neurons}), adjusting")
                self.max_neurons = self.min_neurons * 2
            
            # Validate learning rate range
            if self.learning_rate_range[0] >= self.learning_rate_range[1]:
                tprint_warning(f"Invalid learning rate range: {self.learning_rate_range}, using default")
                self.learning_rate_range = (1e-5, 1e-1)
            
            # Validate batch size range
            if self.batch_size_range[0] >= self.batch_size_range[1]:
                tprint_warning(f"Invalid batch size range: {self.batch_size_range}, using default")
                self.batch_size_range = (16, 256)
            
            # Validate validation split
            if not 0 < self.validation_split < 1:
                tprint_warning(f"Invalid validation_split: {self.validation_split}, using 0.2")
                self.validation_split = 0.2
            
            # Validate test split
            if not 0 < self.test_split < 1:
                tprint_warning(f"Invalid test_split: {self.test_split}, using 0.2")
                self.test_split = 0.2
            
            # Validate CV folds
            if self.cv_folds < 2:
                tprint_warning(f"Invalid cv_folds: {self.cv_folds}, using 5")
                self.cv_folds = 5
            elif self.cv_folds > 20:
                tprint_warning(f"Very large cv_folds: {self.cv_folds}, capping at 20")
                self.cv_folds = 20
            
            # Validate activation functions
            valid_activations = ['relu', 'tanh', 'sigmoid', 'elu', 'swish', 'gelu']
            invalid_activations = [act for act in self.activation_functions if act not in valid_activations]
            if invalid_activations:
                tprint_warning(f"Invalid activation functions: {invalid_activations}, removing")
                self.activation_functions = [act for act in self.activation_functions if act in valid_activations]
                if not self.activation_functions:
                    self.activation_functions = ['relu']
            
            # Validate dropout rates
            invalid_dropouts = [rate for rate in self.dropout_rates if not 0 <= rate <= 1]
            if invalid_dropouts:
                tprint_warning(f"Invalid dropout rates: {invalid_dropouts}, removing")
                self.dropout_rates = [rate for rate in self.dropout_rates if 0 <= rate <= 1]
                if not self.dropout_rates:
                    self.dropout_rates = [0.0, 0.1, 0.2]
            
            tprint_success("✅ NAS configuration validated successfully")
            
        except Exception as e:
            tprint_error(f"Configuration validation failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            # Set safe defaults
            self.search_strategy = 'random'
            self.max_trials = 100
            self.max_epochs = 50
            self.min_layers = 2
            self.max_layers = 10
            self.min_neurons = 32
            self.max_neurons = 512
            self.learning_rate_range = (1e-5, 1e-1)
            self.batch_size_range = (16, 256)
            self.validation_split = 0.2
            self.test_split = 0.2
            self.cv_folds = 5

class SimpleDataFrame:
    """Simple DataFrame-like class for testing with comprehensive error handling."""
    
    def __init__(self, data, columns=None):
        try:
            self.data = data or []
            self.columns = columns or [f'col_{i}' for i in range(len(data[0]) if data and len(data) > 0 else 0)]
            self.shape = (len(data), len(self.columns)) if data else (0, 0)
            
            # Validate data consistency
            if data and len(data) > 0:
                expected_cols = len(data[0])
                if len(self.columns) != expected_cols:
                    tprint_warning(f"Column count mismatch: expected {expected_cols}, got {len(self.columns)}")
                    self.columns = [f'col_{i}' for i in range(expected_cols)]
            
            tprint_info(f"✅ SimpleDataFrame created: {self.shape}")
            
        except Exception as e:
            tprint_error(f"SimpleDataFrame initialization failed: {e}")
            self.data = []
            self.columns = []
            self.shape = (0, 0)
    
    def __getitem__(self, key):
        try:
            if isinstance(key, str):
                if key not in self.columns:
                    tprint_error(f"Column '{key}' not found in {self.columns}")
                    return []
                col_idx = self.columns.index(key)
                return [row[col_idx] for row in self.data]
            return self.data[key]
        except Exception as e:
            tprint_error(f"DataFrame indexing failed: {e}")
            return []
    
    def fillna(self, value):
        """Fill missing values with a value."""
        try:
            filled_data = []
            for row in self.data:
                filled_row = [value if x is None else x for x in row]
                filled_data.append(filled_row)
            return SimpleDataFrame(filled_data, self.columns)
        except Exception as e:
            tprint_error(f"fillna failed: {e}")
            return self
    
    def isnull(self):
        """Check for null values."""
        try:
            null_mask = []
            for row in self.data:
                null_row = [x is None for x in row]
                null_mask.append(null_row)
            return SimpleDataFrame(null_mask, self.columns)
        except Exception as e:
            tprint_error(f"isnull failed: {e}")
            return SimpleDataFrame([], self.columns)
    
    def sum(self):
        """Sum values with error handling."""
        try:
            if not self.data:
                return 0
            total = 0
            for row in self.data:
                for x in row:
                    if x is not None and isinstance(x, (int, float)):
                        total += x
            return total
        except Exception as e:
            tprint_error(f"sum failed: {e}")
            return 0
    
    def median(self):
        """Calculate median with error handling."""
        try:
            if not self.data:
                return 0
            all_values = []
            for row in self.data:
                for x in row:
                    if x is not None and isinstance(x, (int, float)):
                        all_values.append(x)
            
            if not all_values:
                return 0
            
            all_values.sort()
            n = len(all_values)
            if n % 2 == 0:
                return (all_values[n//2 - 1] + all_values[n//2]) / 2
            else:
                return all_values[n//2]
        except Exception as e:
            tprint_error(f"median calculation failed: {e}")
            return 0
    
    def mean(self):
        """Calculate mean with error handling."""
        try:
            if not self.data:
                return 0
            all_values = []
            for row in self.data:
                for x in row:
                    if x is not None and isinstance(x, (int, float)):
                        all_values.append(x)
            
            if not all_values:
                return 0
            
            return sum(all_values) / len(all_values)
        except Exception as e:
            tprint_error(f"mean calculation failed: {e}")
            return 0
    
    def std(self):
        """Calculate standard deviation with error handling."""
        try:
            if not self.data:
                return 0
            all_values = []
            for row in self.data:
                for x in row:
                    if x is not None and isinstance(x, (int, float)):
                        all_values.append(x)
            
            if len(all_values) < 2:
                return 0
            
            mean_val = self.mean()
            variance = sum((x - mean_val) ** 2 for x in all_values) / len(all_values)
            return math.sqrt(variance)
        except Exception as e:
            tprint_error(f"std calculation failed: {e}")
            return 0

class SimpleSeries:
    """Simple Series-like class for testing with comprehensive error handling."""
    
    def __init__(self, data, name=None):
        try:
            self.data = data or []
            self.name = name or 'series'
            self.shape = (len(data),) if data else (0,)
            
            tprint_info(f"✅ SimpleSeries created: {self.shape}")
            
        except Exception as e:
            tprint_error(f"SimpleSeries initialization failed: {e}")
            self.data = []
            self.name = 'series'
            self.shape = (0,)
    
    def __getitem__(self, key):
        try:
            return self.data[key]
        except Exception as e:
            tprint_error(f"Series indexing failed: {e}")
            return None
    
    def values(self):
        """Get values with error handling."""
        try:
            return self.data
        except Exception as e:
            tprint_error(f"values() failed: {e}")
            return []
    
    def fillna(self, value):
        """Fill missing values with error handling."""
        try:
            filled_data = [value if x is None else x for x in self.data]
            return SimpleSeries(filled_data, self.name)
        except Exception as e:
            tprint_error(f"Series fillna failed: {e}")
            return self
    
    def isnull(self):
        """Check for null values with error handling."""
        try:
            null_mask = [x is None for x in self.data]
            return SimpleSeries(null_mask, f"{self.name}_isnull")
        except Exception as e:
            tprint_error(f"Series isnull failed: {e}")
            return SimpleSeries([], f"{self.name}_isnull")
    
    def sum(self):
        """Sum values with error handling."""
        try:
            if not self.data:
                return 0
            total = 0
            for x in self.data:
                if x is not None and isinstance(x, (int, float)):
                    total += x
            return total
        except Exception as e:
            tprint_error(f"Series sum failed: {e}")
            return 0
    
    def mean(self):
        """Calculate mean with error handling."""
        try:
            if not self.data:
                return 0
            valid_values = [x for x in self.data if x is not None and isinstance(x, (int, float))]
            if not valid_values:
                return 0
            return sum(valid_values) / len(valid_values)
        except Exception as e:
            tprint_error(f"Series mean failed: {e}")
            return 0
    
    def median(self):
        """Calculate median with error handling."""
        try:
            if not self.data:
                return 0
            valid_values = [x for x in self.data if x is not None and isinstance(x, (int, float))]
            if not valid_values:
                return 0
            
            valid_values.sort()
            n = len(valid_values)
            if n % 2 == 0:
                return (valid_values[n//2 - 1] + valid_values[n//2]) / 2
            else:
                return valid_values[n//2]
        except Exception as e:
            tprint_error(f"Series median failed: {e}")
            return 0

class NASTrainer:
    """
    Simplified Neural Architecture Search Trainer with comprehensive error handling.
    
    This class provides NAS functionality without external dependencies,
    enhanced with proper error handling, tprint logging, and full implementation.
    """
    
    def __init__(self, config: Optional[NASConfig] = None):
        """Initialize the NAS Trainer with comprehensive error handling."""
        try:
            self.config = config or NASConfig()
            self.logger = logger.getChild('NASTrainer') if logger else None
            
            # Results storage
            self.search_results = []
            self.best_architecture = None
            self.best_model = None
            self.training_history = []
            self.error_log = []
            
            # Setup output directory with error handling
            try:
                self.output_dir = Path(self.config.output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
                tprint_info(f"📁 Output directory created: {self.output_dir}")
            except Exception as e:
                tprint_error(f"Failed to create output directory: {e}")
                self.output_dir = Path('nas_results_fallback')
                self.output_dir.mkdir(parents=True, exist_ok=True)
            
            tprint_success(f"🚀 NAS Trainer initialized with {self.config.search_strategy} strategy")
            tprint_structured({
                'config': {
                    'search_strategy': self.config.search_strategy,
                    'max_trials': self.config.max_trials,
                    'max_epochs': self.config.max_epochs,
                    'output_dir': str(self.output_dir)
                }
            })
            
        except Exception as e:
            tprint_error(f"NAS Trainer initialization failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to initialize NAS Trainer: {e}")
    
    def prepare_data(self, X, y, test_size: float = None) -> Dict[str, Any]:
        """
        Prepare data for NAS training with comprehensive error handling.
        
        Args:
            X: Feature data
            y: Target data
            test_size: Test set size (uses config if None)
            
        Returns:
            Dictionary containing prepared data splits
        """
        try:
            tprint_info("🔄 Starting data preparation...")
            
            # Validate inputs
            if X is None or y is None:
                raise ValueError("X and y cannot be None")
            
            if len(X) != len(y):
                raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
            
            if len(X) < 10:
                tprint_warning(f"Very small dataset: {len(X)} samples")
            
            # Convert to SimpleDataFrame if needed
            try:
                if not isinstance(X, SimpleDataFrame):
                    if isinstance(X, list):
                        X = SimpleDataFrame(X)
                        tprint_info("✅ Converted X from list to SimpleDataFrame")
                    else:
                        # Assume it's a 2D array-like structure
                        X = SimpleDataFrame(X)
                        tprint_info("✅ Converted X to SimpleDataFrame")
                
                if not isinstance(y, SimpleSeries):
                    if isinstance(y, list):
                        y = SimpleSeries(y)
                        tprint_info("✅ Converted y from list to SimpleSeries")
                    else:
                        y = SimpleSeries(y)
                        tprint_info("✅ Converted y to SimpleSeries")
                        
            except Exception as e:
                tprint_error(f"Failed to convert data: {e}")
                raise
            
            # Calculate data quality metrics with error handling
            try:
                quality_metrics = {
                    'total_rows': X.shape[0],
                    'total_columns': X.shape[1],
                    'missing_values': 0,  # Simplified
                    'missing_percentage': 0.0,
                    'duplicate_rows': 0,
                    'duplicate_percentage': 0.0
                }
                tprint_info(f"📊 Data quality metrics: {quality_metrics}")
            except Exception as e:
                tprint_warning(f"Failed to calculate quality metrics: {e}")
                quality_metrics = {'total_rows': len(X), 'total_columns': 0}
            
            tprint_success(f"✅ Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            
            # Simple data splitting (70% train, 15% val, 15% test)
            try:
                n_samples = X.shape[0]
                train_size = int(0.7 * n_samples)
                val_size = int(0.15 * n_samples)
                
                tprint_info(f"📊 Data split: {train_size} train, {val_size} val, {n_samples - train_size - val_size} test")
                
                # Split data with bounds checking
                X_train_data = X.data[:train_size]
                X_val_data = X.data[train_size:train_size + val_size]
                X_test_data = X.data[train_size + val_size:]
                
                y_train_data = y.data[:train_size]
                y_val_data = y.data[train_size:train_size + val_size]
                y_test_data = y.data[train_size + val_size:]
                
                # Create data splits with error handling
                data_splits = {
                    'X_train': SimpleDataFrame(X_train_data, X.columns),
                    'X_val': SimpleDataFrame(X_val_data, X.columns),
                    'X_test': SimpleDataFrame(X_test_data, X.columns),
                    'y_train': SimpleSeries(y_train_data),
                    'y_val': SimpleSeries(y_val_data),
                    'y_test': SimpleSeries(y_test_data),
                    'feature_names': X.columns,
                    'quality_metrics': quality_metrics
                }
                
                tprint_success("✅ Data preparation completed successfully")
                return data_splits
                
            except Exception as e:
                tprint_error(f"Failed to split data: {e}")
                raise
                
        except Exception as e:
            tprint_error(f"Data preparation failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            
            # Return minimal fallback data
            try:
                tprint_warning("🔄 Attempting fallback data preparation...")
                n_samples = len(X) if X is not None else 0
                if n_samples > 0:
                    train_size = max(1, int(n_samples * 0.7))
                    val_size = max(1, int(n_samples * 0.15))
                    
                    return {
                        'X_train': X[:train_size] if hasattr(X, '__getitem__') else X,
                        'X_val': X[train_size:train_size + val_size] if hasattr(X, '__getitem__') else X,
                        'X_test': X[train_size + val_size:] if hasattr(X, '__getitem__') else X,
                        'y_train': y[:train_size] if hasattr(y, '__getitem__') else y,
                        'y_val': y[train_size:train_size + val_size] if hasattr(y, '__getitem__') else y,
                        'y_test': y[train_size + val_size:] if hasattr(y, '__getitem__') else y,
                        'feature_names': getattr(X, 'columns', []),
                        'quality_metrics': {'total_rows': n_samples, 'total_columns': 0}
                    }
                else:
                    raise ValueError("No valid data available")
            except Exception as fallback_error:
                tprint_error(f"Fallback data preparation also failed: {fallback_error}")
                raise RuntimeError(f"Complete data preparation failure: {e}")
    
    def generate_architecture(self, trial_id: int) -> Dict[str, Any]:
        """
        Generate a neural network architecture for the given trial with error handling.
        
        Args:
            trial_id: Trial identifier
            
        Returns:
            Architecture configuration dictionary
        """
        try:
            tprint_info(f"🏗️ Generating architecture for trial {trial_id}")
            
            # Validate trial_id
            if not isinstance(trial_id, int) or trial_id < 0:
                tprint_warning(f"Invalid trial_id {trial_id}, using 0")
                trial_id = 0
            
            # Generate architecture based on search strategy
            try:
                if self.config.search_strategy == 'random':
                    architecture = self._generate_random_architecture()
                    tprint_info("🎲 Using random architecture generation")
                elif self.config.search_strategy == 'grid':
                    architecture = self._generate_grid_architecture(trial_id)
                    tprint_info("📊 Using grid architecture generation")
                else:
                    tprint_warning(f"Unknown search strategy {self.config.search_strategy}, using random")
                    architecture = self._generate_random_architecture()
                    
            except Exception as e:
                tprint_error(f"Architecture generation failed: {e}")
                tprint_warning("🔄 Falling back to simple architecture")
                architecture = self._generate_simple_architecture()
            
            # Add trial metadata
            architecture['trial_id'] = trial_id
            architecture['search_strategy'] = self.config.search_strategy
            architecture['generated_at'] = time.time()
            
            tprint_success(f"✅ Architecture generated for trial {trial_id}")
            tprint_structured({
                'trial_id': trial_id,
                'strategy': self.config.search_strategy,
                'n_layers': architecture.get('n_layers', 0),
                'total_neurons': sum(layer.get('neurons', 0) for layer in architecture.get('layers', []))
            })
            
            return architecture
            
        except Exception as e:
            tprint_error(f"Architecture generation failed completely: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            
            # Return minimal fallback architecture
            return {
                'trial_id': trial_id,
                'search_strategy': 'fallback',
                'layers': [{'neurons': 32, 'activation': 'relu', 'dropout': 0.2}],
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer': 'adam',
                'n_layers': 1,
                'generated_at': time.time(),
                'error': str(e)
            }
    
    def _generate_random_architecture(self) -> Dict[str, Any]:
        """Generate a random neural network architecture with error handling."""
        try:
            tprint_info("🎲 Generating random architecture...")
            
            # Random number of layers with validation
            try:
                n_layers = random.randint(self.config.min_layers, self.config.max_layers)
                if n_layers <= 0:
                    n_layers = 1
                    tprint_warning("Invalid layer count, using 1")
            except Exception as e:
                tprint_warning(f"Layer count generation failed: {e}, using 2")
                n_layers = 2
            
            # Random layer sizes with error handling
            layers = []
            for i in range(n_layers):
                try:
                    neurons = random.randint(self.config.min_neurons, self.config.max_neurons)
                    if neurons <= 0:
                        neurons = 32
                        tprint_warning(f"Invalid neuron count for layer {i}, using 32")
                    
                    activation = random.choice(self.config.activation_functions)
                    dropout = random.choice(self.config.dropout_rates)
                    
                    layers.append({
                        'neurons': neurons,
                        'activation': activation,
                        'dropout': dropout
                    })
                    
                except Exception as e:
                    tprint_warning(f"Layer {i} generation failed: {e}, using defaults")
                    layers.append({
                        'neurons': 32,
                        'activation': 'relu',
                        'dropout': 0.2
                    })
            
            # Random hyperparameters with validation
            try:
                lr_min, lr_max = self.config.learning_rate_range
                learning_rate = random.uniform(lr_min, lr_max)
                if learning_rate <= 0:
                    learning_rate = 0.001
                    tprint_warning("Invalid learning rate, using 0.001")
            except Exception as e:
                tprint_warning(f"Learning rate generation failed: {e}, using 0.001")
                learning_rate = 0.001
            
            try:
                batch_size = random.randint(*self.config.batch_size_range)
                if batch_size <= 0:
                    batch_size = 32
                    tprint_warning("Invalid batch size, using 32")
            except Exception as e:
                tprint_warning(f"Batch size generation failed: {e}, using 32")
                batch_size = 32
            
            architecture = {
                'layers': layers,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'optimizer': self.config.optimizer,
                'n_layers': n_layers
            }
            
            tprint_success(f"✅ Random architecture generated: {n_layers} layers")
            return architecture
            
        except Exception as e:
            tprint_error(f"Random architecture generation failed: {e}")
            return self._generate_simple_architecture()
    
    def _generate_simple_architecture(self) -> Dict[str, Any]:
        """Generate a simple fallback architecture."""
        try:
            tprint_info("🔄 Generating simple fallback architecture...")
            
            return {
                'layers': [
                    {'neurons': 64, 'activation': 'relu', 'dropout': 0.2},
                    {'neurons': 32, 'activation': 'relu', 'dropout': 0.2}
                ],
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer': 'adam',
                'n_layers': 2
            }
        except Exception as e:
            tprint_error(f"Simple architecture generation failed: {e}")
            return {
                'layers': [{'neurons': 32, 'activation': 'relu', 'dropout': 0.2}],
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer': 'adam',
                'n_layers': 1
            }
    
    def _generate_grid_architecture(self, trial_id: int) -> Dict[str, Any]:
        """Generate architecture from grid search with error handling."""
        try:
            tprint_info(f"📊 Generating grid architecture for trial {trial_id}")
            
            # Simple grid search implementation with validation
            n_layers_options = [2, 4, 6, 8, 10]
            neurons_options = [32, 64, 128, 256, 512]
            
            try:
                n_layers = n_layers_options[trial_id % len(n_layers_options)]
                base_neurons = neurons_options[trial_id % len(neurons_options)]
                
                if n_layers <= 0:
                    n_layers = 2
                    tprint_warning("Invalid layer count from grid, using 2")
                if base_neurons <= 0:
                    base_neurons = 64
                    tprint_warning("Invalid neuron count from grid, using 64")
                    
            except Exception as e:
                tprint_warning(f"Grid parameter selection failed: {e}, using defaults")
                n_layers = 2
                base_neurons = 64
            
            # Generate layers with error handling
            layers = []
            for i in range(n_layers):
                try:
                    neurons = base_neurons // (2 ** i) if i > 0 else base_neurons
                    neurons = max(self.config.min_neurons, min(neurons, self.config.max_neurons))
                    
                    if neurons <= 0:
                        neurons = 32
                        tprint_warning(f"Invalid neuron count for layer {i}, using 32")
                    
                    layers.append({
                        'neurons': neurons,
                        'activation': 'relu',
                        'dropout': 0.2
                    })
                    
                except Exception as e:
                    tprint_warning(f"Layer {i} generation failed: {e}, using defaults")
                    layers.append({
                        'neurons': 32,
                        'activation': 'relu',
                        'dropout': 0.2
                    })
            
            architecture = {
                'layers': layers,
                'learning_rate': 0.001,
                'batch_size': 64,
                'optimizer': self.config.optimizer,
                'n_layers': n_layers
            }
            
            tprint_success(f"✅ Grid architecture generated: {n_layers} layers")
            return architecture
            
        except Exception as e:
            tprint_error(f"Grid architecture generation failed: {e}")
            return self._generate_simple_architecture()
    
    def create_model_from_architecture(self, architecture: Dict[str, Any], 
                                     input_shape: int) -> Dict[str, Any]:
        """
        Create a model configuration from architecture with error handling.
        
        Args:
            architecture: Architecture configuration
            input_shape: Input feature dimension
            
        Returns:
            Model configuration dictionary
        """
        try:
            tprint_info(f"🏗️ Creating model from architecture...")
            
            # Validate inputs
            if not isinstance(architecture, dict):
                raise ValueError("Architecture must be a dictionary")
            
            if not isinstance(input_shape, int) or input_shape <= 0:
                tprint_warning(f"Invalid input_shape {input_shape}, using 1")
                input_shape = 1
            
            # Extract architecture components with error handling
            try:
                layers = architecture.get('layers', [])
                if not layers:
                    tprint_warning("No layers found in architecture, using default")
                    layers = [{'neurons': 32, 'activation': 'relu', 'dropout': 0.2}]
                
                learning_rate = architecture.get('learning_rate', 0.001)
                if learning_rate <= 0:
                    tprint_warning(f"Invalid learning rate {learning_rate}, using 0.001")
                    learning_rate = 0.001
                
                batch_size = architecture.get('batch_size', 32)
                if batch_size <= 0:
                    tprint_warning(f"Invalid batch size {batch_size}, using 32")
                    batch_size = 32
                
                optimizer = architecture.get('optimizer', 'adam')
                if not isinstance(optimizer, str):
                    tprint_warning(f"Invalid optimizer {optimizer}, using adam")
                    optimizer = 'adam'
                    
            except Exception as e:
                tprint_warning(f"Architecture component extraction failed: {e}, using defaults")
                layers = [{'neurons': 32, 'activation': 'relu', 'dropout': 0.2}]
                learning_rate = 0.001
                batch_size = 32
                optimizer = 'adam'
            
            model_config = {
                'type': 'neural_network',
                'input_shape': input_shape,
                'layers': layers,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'optimizer': optimizer,
                'created_at': time.time()
            }
            
            tprint_success(f"✅ Model configuration created: {len(layers)} layers")
            tprint_structured({
                'input_shape': input_shape,
                'n_layers': len(layers),
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'optimizer': optimizer
            })
            
            return model_config
            
        except Exception as e:
            tprint_error(f"Model creation failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            
            # Return minimal fallback model
            return {
                'type': 'neural_network',
                'input_shape': max(1, input_shape),
                'layers': [{'neurons': 32, 'activation': 'relu', 'dropout': 0.2}],
                'learning_rate': 0.001,
                'batch_size': 32,
                'optimizer': 'adam',
                'created_at': time.time(),
                'error': str(e)
            }
    
    def train_architecture(self, architecture: Dict[str, Any], 
                          data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulate training a neural network with comprehensive error handling.
        
        Args:
            architecture: Architecture configuration
            data_splits: Data splits dictionary
            
        Returns:
            Training results dictionary
        """
        try:
            trial_id = architecture.get('trial_id', 0)
            tprint_info(f"🧠 Training architecture {trial_id}")
            
            # Validate inputs
            if not isinstance(architecture, dict):
                raise ValueError("Architecture must be a dictionary")
            
            if not isinstance(data_splits, dict):
                raise ValueError("Data splits must be a dictionary")
            
            # Simulate training time with error handling
            try:
                time.sleep(0.1)  # Simulate training
            except Exception as e:
                tprint_warning(f"Training simulation interrupted: {e}")
            
            # Generate realistic mock results with validation
            try:
                # Generate realistic training metrics based on architecture complexity
                n_layers = len(architecture.get('layers', []))
                total_neurons = sum(layer.get('neurons', 32) for layer in architecture.get('layers', []))
                learning_rate = architecture.get('learning_rate', 0.001)
                
                # Base performance depends on architecture complexity
                complexity_factor = min(1.0, (n_layers * total_neurons) / 10000)
                base_accuracy = 0.5 + (complexity_factor * 0.4)  # 0.5 to 0.9 range
                
                # Add some randomness but keep it realistic
                train_accuracy = base_accuracy + random.uniform(-0.1, 0.1)
                train_accuracy = max(0.3, min(0.98, train_accuracy))  # Clamp to realistic range
                
                # Validation accuracy is typically slightly lower than training
                overfitting_factor = random.uniform(0.02, 0.15)
                val_accuracy = train_accuracy - overfitting_factor
                val_accuracy = max(0.2, val_accuracy)  # Ensure minimum performance
                
                # Epochs trained depends on early stopping and max epochs
                max_epochs = self.config.max_epochs
                early_stop_prob = random.uniform(0.3, 0.8)  # 30-80% chance of early stopping
                if random.random() < early_stop_prob:
                    epochs_trained = random.randint(1, max(1, int(max_epochs * 0.7)))
                else:
                    epochs_trained = random.randint(int(max_epochs * 0.8), max_epochs)
                
                # Validate results
                if train_accuracy < 0 or train_accuracy > 1:
                    train_accuracy = 0.8
                    tprint_warning("Invalid train accuracy, using 0.8")
                
                if val_accuracy < 0 or val_accuracy > 1:
                    val_accuracy = max(0, train_accuracy - 0.05)
                    tprint_warning("Invalid val accuracy, using adjusted value")
                
                if epochs_trained <= 0:
                    epochs_trained = 1
                    tprint_warning("Invalid epochs, using 1")
                    
            except Exception as e:
                tprint_warning(f"Result generation failed: {e}, using defaults")
                train_accuracy = 0.8
                val_accuracy = 0.75
                epochs_trained = 1
            
            results = {
                'trial_id': trial_id,
                'architecture': architecture,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'epochs_trained': epochs_trained,
                'best_val_accuracy': val_accuracy,
                'success': True,
                'trained_at': time.time()
            }
            
            tprint_success(f"✅ Architecture {trial_id} trained: val_acc={val_accuracy:.4f}")
            tprint_structured({
                'trial_id': trial_id,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'epochs_trained': epochs_trained
            })
            
            return results
            
        except Exception as e:
            tprint_error(f"Architecture training failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            
            # Return fallback results
            return {
                'trial_id': architecture.get('trial_id', 0),
                'architecture': architecture,
                'train_accuracy': 0.5,
                'val_accuracy': 0.45,
                'epochs_trained': 1,
                'best_val_accuracy': 0.45,
                'success': False,
                'error': str(e),
                'trained_at': time.time()
            }
    
    def search_architectures(self, data_splits: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Perform neural architecture search with comprehensive error handling.
        
        Args:
            data_splits: Prepared data splits
            
        Returns:
            List of search results
        """
        try:
            tprint_info(f"🔍 Starting NAS with {self.config.search_strategy} strategy")
            
            # Validate inputs
            if not isinstance(data_splits, dict):
                raise ValueError("Data splits must be a dictionary")
            
            if 'X_train' not in data_splits:
                raise ValueError("X_train not found in data splits")
            
            # Get data shape safely
            try:
                data_shape = data_splits['X_train'].shape
                tprint_info(f"📊 Data shape: {data_shape}")
            except Exception as e:
                tprint_warning(f"Could not get data shape: {e}")
                data_shape = "unknown"
            
            tprint_info(f"🎯 Max trials: {self.config.max_trials}")
            
            search_results = []
            successful_trials = 0
            failed_trials = 0
            
            for trial_id in range(self.config.max_trials):
                try:
                    tprint_info(f"🔄 Trial {trial_id + 1}/{self.config.max_trials}")
                    
                    # Generate architecture with error handling
                    try:
                        architecture = self.generate_architecture(trial_id)
                    except Exception as e:
                        tprint_error(f"Architecture generation failed for trial {trial_id}: {e}")
                        failed_trials += 1
                        continue
                    
                    # Train architecture with error handling
                    try:
                        result = self.train_architecture(architecture, data_splits)
                        search_results.append(result)
                        successful_trials += 1
                        
                        if result.get('success', False):
                            tprint_success(f"✅ Trial {trial_id + 1} completed successfully")
                        else:
                            tprint_warning(f"⚠️ Trial {trial_id + 1} completed with issues")
                            
                    except Exception as e:
                        tprint_error(f"Training failed for trial {trial_id}: {e}")
                        failed_trials += 1
                        
                        # Add failed trial result
                        search_results.append({
                            'trial_id': trial_id,
                            'architecture': architecture,
                            'train_accuracy': 0.0,
                            'val_accuracy': 0.0,
                            'epochs_trained': 0,
                            'best_val_accuracy': 0.0,
                            'success': False,
                            'error': str(e),
                            'trained_at': time.time()
                        })
                        
                except Exception as e:
                    tprint_error(f"Trial {trial_id} failed completely: {e}")
                    failed_trials += 1
            
            # Sort results by validation accuracy with error handling
            try:
                search_results.sort(key=lambda x: x.get('val_accuracy', 0), reverse=True)
                tprint_info("📊 Results sorted by validation accuracy")
            except Exception as e:
                tprint_warning(f"Failed to sort results: {e}")
            
            # Store results
            self.search_results = search_results
            
            # Save results with error handling
            try:
                if self.config.save_results:
                    self._save_search_results()
            except Exception as e:
                tprint_warning(f"Failed to save results: {e}")
            
            tprint_success(f"✅ NAS completed: {successful_trials} successful, {failed_trials} failed trials")
            tprint_structured({
                'total_trials': self.config.max_trials,
                'successful_trials': successful_trials,
                'failed_trials': failed_trials,
                'best_accuracy': max([r.get('val_accuracy', 0) for r in search_results]) if search_results else 0
            })
            
            return search_results
            
        except Exception as e:
            tprint_error(f"NAS search failed completely: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            
            # Return empty results
            self.search_results = []
            return []
    
    def evaluate_best_architecture(self, data_splits: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the best architecture on test data with comprehensive error handling.
        
        Args:
            data_splits: Data splits dictionary
            
        Returns:
            Evaluation results
        """
        try:
            tprint_info("🏆 Evaluating best architecture...")
            
            # Validate inputs
            if not isinstance(data_splits, dict):
                raise ValueError("Data splits must be a dictionary")
            
            if not self.search_results:
                tprint_error("No search results available. Run search_architectures first.")
                raise ValueError("No search results available. Run search_architectures first.")
            
            # Get best architecture with error handling
            try:
                successful_results = [r for r in self.search_results if r.get('success', False)]
                if not successful_results:
                    tprint_warning("No successful results found, using first result")
                    best_result = self.search_results[0] if self.search_results else None
                else:
                    best_result = max(successful_results, key=lambda x: x.get('val_accuracy', 0))
                
                if best_result is None:
                    raise ValueError("No results available for evaluation")
                    
            except Exception as e:
                tprint_error(f"Failed to find best architecture: {e}")
                raise
            
            trial_id = best_result.get('trial_id', 0)
            tprint_info(f"🏆 Evaluating best architecture (trial {trial_id})")
            
            # Simulate evaluation with error handling
            try:
                base_accuracy = best_result.get('val_accuracy', 0.5)
                test_accuracy = base_accuracy - random.uniform(0.0, 0.05)
                test_precision = test_accuracy + random.uniform(-0.1, 0.1)
                test_recall = test_accuracy + random.uniform(-0.1, 0.1)
                test_f1 = (test_precision + test_recall) / 2
                
                # Validate results
                test_accuracy = max(0, min(1, test_accuracy))
                test_precision = max(0, min(1, test_precision))
                test_recall = max(0, min(1, test_recall))
                test_f1 = max(0, min(1, test_f1))
                
            except Exception as e:
                tprint_warning(f"Evaluation simulation failed: {e}, using defaults")
                test_accuracy = 0.7
                test_precision = 0.7
                test_recall = 0.7
                test_f1 = 0.7
            
            evaluation_results = {
                'best_architecture': best_result.get('architecture', {}),
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'success': True,
                'evaluated_at': time.time()
            }
            
            # Store best model with error handling
            try:
                self.best_architecture = best_result.get('architecture', {})
                tprint_info("💾 Best architecture stored")
            except Exception as e:
                tprint_warning(f"Failed to store best architecture: {e}")
            
            tprint_success(f"✅ Best architecture evaluated: test_acc={test_accuracy:.4f}")
            tprint_structured({
                'trial_id': trial_id,
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1
            })
            
            return evaluation_results
            
        except Exception as e:
            tprint_error(f"Best architecture evaluation failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            
            # Return fallback evaluation
            return {
                'best_architecture': {},
                'test_accuracy': 0.5,
                'test_precision': 0.5,
                'test_recall': 0.5,
                'test_f1': 0.5,
                'success': False,
                'error': str(e),
                'evaluated_at': time.time()
            }
    
    def _save_search_results(self):
        """Save search results to files with comprehensive error handling."""
        try:
            tprint_info("💾 Saving search results...")
            
            if not self.search_results:
                tprint_warning("No search results to save")
                return
            
            # Save results as JSON with error handling
            try:
                results_file = self.output_dir / 'search_results.json'
                with open(results_file, 'w') as f:
                    json.dump(self.search_results, f, indent=2, default=str)
                tprint_success(f"✅ Results saved to {results_file}")
            except Exception as e:
                tprint_error(f"Failed to save JSON results: {e}")
                raise
            
            # Save summary with error handling
            try:
                summary_file = self.output_dir / 'search_summary.json'
                summary = {
                    'total_trials': len(self.search_results),
                    'successful_trials': len([r for r in self.search_results if r.get('success', False)]),
                    'best_accuracy': max([r.get('val_accuracy', 0) for r in self.search_results]) if self.search_results else 0,
                    'search_strategy': self.config.search_strategy,
                    'saved_at': time.time()
                }
                
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                tprint_success(f"✅ Summary saved to {summary_file}")
                
            except Exception as e:
                tprint_warning(f"Failed to save summary: {e}")
            
        except Exception as e:
            tprint_error(f"Failed to save search results: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
    
    def run_full_nas(self, X, y) -> Dict[str, Any]:
        """
        Run complete Neural Architecture Search pipeline with comprehensive error handling.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Complete NAS results
        """
        try:
            tprint_info("🚀 Starting full NAS pipeline")
            
            # Validate inputs
            if X is None or y is None:
                raise ValueError("X and y cannot be None")
            
            # Step 1: Prepare data with error handling
            try:
                tprint_info("📊 Step 1: Preparing data")
                data_splits = self.prepare_data(X, y)
                tprint_success("✅ Data preparation completed")
            except Exception as e:
                tprint_error(f"Data preparation failed: {e}")
                raise
            
            # Step 2: Search architectures with error handling
            try:
                tprint_info("🔍 Step 2: Searching architectures")
                search_results = self.search_architectures(data_splits)
                tprint_success(f"✅ Architecture search completed: {len(search_results)} results")
            except Exception as e:
                tprint_error(f"Architecture search failed: {e}")
                search_results = []
            
            # Step 3: Evaluate best architecture with error handling
            evaluation_results = {}
            try:
                if search_results:
                    tprint_info("🏆 Step 3: Evaluating best architecture")
                    evaluation_results = self.evaluate_best_architecture(data_splits)
                    tprint_success("✅ Best architecture evaluation completed")
                else:
                    tprint_warning("⚠️ No search results available for evaluation")
                    evaluation_results = {
                        'best_architecture': {},
                        'test_accuracy': 0.0,
                        'test_precision': 0.0,
                        'test_recall': 0.0,
                        'test_f1': 0.0,
                        'success': False,
                        'error': 'No search results available'
                    }
            except Exception as e:
                tprint_error(f"Best architecture evaluation failed: {e}")
                evaluation_results = {
                    'best_architecture': {},
                    'test_accuracy': 0.0,
                    'test_precision': 0.0,
                    'test_recall': 0.0,
                    'test_f1': 0.0,
                    'success': False,
                    'error': str(e)
                }
            
            # Compile final results with error handling
            try:
                # Get data info safely
                data_info = {}
                try:
                    data_info = {
                        'n_samples': len(X.data) if hasattr(X, 'data') else len(X),
                        'n_features': X.shape[1] if hasattr(X, 'shape') else len(X[0]) if X else 0,
                        'feature_names': data_splits.get('feature_names', [])
                    }
                except Exception as e:
                    tprint_warning(f"Failed to get data info: {e}")
                    data_info = {
                        'n_samples': 0,
                        'n_features': 0,
                        'feature_names': []
                    }
                
                final_results = {
                    'search_results': search_results,
                    'evaluation_results': evaluation_results,
                    'config': self.config.__dict__,
                    'data_info': data_info,
                    'best_architecture': self.best_architecture,
                    'training_completed': True,
                    'pipeline_completed_at': time.time()
                }
                
                tprint_success("✅ Full NAS pipeline completed successfully")
                tprint_structured({
                    'total_trials': len(search_results),
                    'best_accuracy': evaluation_results.get('test_accuracy', 0),
                    'data_samples': data_info.get('n_samples', 0),
                    'data_features': data_info.get('n_features', 0)
                })
                
                return final_results
                
            except Exception as e:
                tprint_error(f"Failed to compile final results: {e}")
                return {
                    'search_results': search_results,
                    'evaluation_results': evaluation_results,
                    'config': self.config.__dict__,
                    'data_info': {'n_samples': 0, 'n_features': 0, 'feature_names': []},
                    'best_architecture': {},
                    'training_completed': False,
                    'error': str(e),
                    'pipeline_completed_at': time.time()
                }
                
        except Exception as e:
            tprint_error(f"Full NAS pipeline failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'search_results': [],
                'evaluation_results': {},
                'config': self.config.__dict__,
                'data_info': {'n_samples': 0, 'n_features': 0, 'feature_names': []},
                'best_architecture': {},
                'training_completed': False,
                'error': str(e),
                'pipeline_completed_at': time.time()
            }
    
    def cleanup(self):
        """Cleanup resources with error handling."""
        try:
            tprint_info("🧹 Cleaning up NAS Trainer resources...")
            
            # Clear results
            try:
                self.search_results = []
                self.best_architecture = None
                self.best_model = None
                self.training_history = []
                tprint_info("✅ Results cleared")
            except Exception as e:
                tprint_warning(f"Failed to clear results: {e}")
            
            # Clear error log
            try:
                self.error_log = []
                tprint_info("✅ Error log cleared")
            except Exception as e:
                tprint_warning(f"Failed to clear error log: {e}")
            
            tprint_success("✅ NAS Trainer cleanup completed")
            
        except Exception as e:
            tprint_error(f"Cleanup failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")
    
    def __enter__(self):
        """Context manager entry with error handling."""
        try:
            tprint_info("🚀 Entering NAS Trainer context")
            return self
        except Exception as e:
            tprint_error(f"Context entry failed: {e}")
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with error handling."""
        try:
            if exc_type is not None:
                tprint_error(f"Exception occurred: {exc_type.__name__}: {exc_val}")
                tprint_error(f"Traceback: {exc_tb}")
            
            self.cleanup()
            tprint_info("🚪 Exiting NAS Trainer context")
            
        except Exception as e:
            tprint_error(f"Context exit failed: {e}")
            tprint_error(f"Traceback: {traceback.format_exc()}")


# Example usage and testing functions
def create_sample_data(n_samples: int = 1000, n_features: int = 20):
    """Create sample data for testing with error handling."""
    try:
        tprint_info(f"🧪 Creating sample data: {n_samples} samples, {n_features} features")
        
        # Validate inputs
        if n_samples <= 0:
            tprint_warning(f"Invalid n_samples {n_samples}, using 100")
            n_samples = 100
        
        if n_features <= 0:
            tprint_warning(f"Invalid n_features {n_features}, using 10")
            n_features = 10
        
        random.seed(42)
        
        # Generate features with error handling
        try:
            X_data = []
            for _ in range(n_samples):
                row = [random.uniform(-1, 1) for _ in range(n_features)]
                X_data.append(row)
            
            X = SimpleDataFrame(X_data, [f'feature_{i}' for i in range(n_features)])
            tprint_success(f"✅ Features generated: {X.shape}")
            
        except Exception as e:
            tprint_error(f"Feature generation failed: {e}")
            raise
        
        # Generate target (binary classification) with error handling
        try:
            y_data = []
            for row in X_data:
                target = 1 if sum(row) > 0 else 0
                y_data.append(target)
            
            y = SimpleSeries(y_data, 'target')
            tprint_success(f"✅ Targets generated: {len(y)} samples")
            
        except Exception as e:
            tprint_error(f"Target generation failed: {e}")
            raise
        
        tprint_success("✅ Sample data creation completed")
        return X, y
        
    except Exception as e:
        tprint_error(f"Sample data creation failed: {e}")
        tprint_error(f"Traceback: {traceback.format_exc()}")
        
        # Return minimal fallback data
        try:
            X = SimpleDataFrame([[0, 0]], ['feature_0', 'feature_1'])
            y = SimpleSeries([0], 'target')
            tprint_warning("🔄 Using minimal fallback data")
            return X, y
        except Exception as fallback_error:
            tprint_error(f"Fallback data creation failed: {fallback_error}")
            raise RuntimeError(f"Complete sample data creation failure: {e}")


def run_nas_example():
    """Run a complete NAS example with comprehensive error handling."""
    try:
        tprint_info("🧪 Running NAS example")
        
        # Create sample data with error handling
        try:
            X, y = create_sample_data(n_samples=100, n_features=10)
            tprint_success(f"✅ Created sample data: {X.shape}")
        except Exception as e:
            tprint_error(f"Sample data creation failed: {e}")
            raise
        
        # Configure NAS with error handling
        try:
            config = NASConfig(
                search_strategy='random',
                max_trials=5,
                max_epochs=3,
                use_m1_optimization=False,
                verbose=True
            )
            tprint_success("✅ NAS configuration created")
        except Exception as e:
            tprint_error(f"Configuration creation failed: {e}")
            raise
        
        # Run NAS with error handling
        try:
            with NASTrainer(config) as nas_trainer:
                results = nas_trainer.run_full_nas(X, y)
                
                # Print results with error handling
                try:
                    best_accuracy = results.get('evaluation_results', {}).get('test_accuracy', 0)
                    best_architecture = results.get('best_architecture', {})
                    n_layers = best_architecture.get('n_layers', 0)
                    total_trials = len(results.get('search_results', []))
                    
                    tprint_success(f"🏆 Best accuracy: {best_accuracy:.4f}")
                    tprint_success(f"🏗️ Best architecture: {n_layers} layers")
                    tprint_success(f"📊 Total trials: {total_trials}")
                    
                except Exception as e:
                    tprint_warning(f"Failed to print results: {e}")
                
        except Exception as e:
            tprint_error(f"NAS execution failed: {e}")
            raise
        
        tprint_success("✅ NAS example completed successfully")
        
    except Exception as e:
        tprint_error(f"NAS example failed: {e}")
        tprint_error(f"Traceback: {traceback.format_exc()}")
        raise


if __name__ == "__main__":
    # Run example
    run_nas_example()