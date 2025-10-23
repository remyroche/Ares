"""
Adaptive Optimization Engine for Apple Silicon.

This module provides intelligent optimization that learns from execution patterns
and automatically tunes hardware settings for optimal performance.
"""

import logging
import time
import threading
import json
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import asyncio
from collections import deque, defaultdict
import queue
import sqlite3
from datetime import datetime, timedelta
from functools import wraps
import hashlib
import psutil
import gc

# Optional dependencies
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Define the classes locally to avoid circular import
class LearningAlgorithm(Enum):
    """Learning algorithms for adaptive optimization."""
    GRADIENT_DESCENT = "gradient_descent"
    ADAM = "adam"
    RMSPROP = "rmsprop"
    MOMENTUM = "momentum"
    RANDOM_FOREST = "random_forest"

class OptimizationTarget(Enum):
    """Optimization targets."""
    PERFORMANCE = "performance"
    MEMORY = "memory"
    ENERGY = "energy"
    BALANCED = "balanced"

@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    accuracy: float = 0.0
    throughput: float = 0.0

@dataclass
class OptimizationSettings:
    """Settings for optimization."""
    learning_rate: float = 0.01
    batch_size: int = 32
    epochs: int = 100
    patience: int = 10

class LearningModel:
    """Base learning model."""
    def __init__(self):
        pass

class BaseAdaptiveOptimizationEngine:
    """Base adaptive optimization engine."""
    def __init__(self):
        pass

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Optimization strategies."""
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"
    ADAPTIVE = "adaptive"

class WorkloadCategory(Enum):
    """Workload categories for optimization."""
    MACHINE_LEARNING = "machine_learning"
    FINANCIAL_MODELING = "financial_modeling"
    DATA_PROCESSING = "data_processing"
    REAL_TIME_TRADING = "real_time_trading"
    BACKTESTING = "backtesting"
    FEATURE_ENGINEERING = "feature_engineering"
    MATRIX_OPERATIONS = "matrix_operations"
    NEURAL_INFERENCE = "neural_inference"

@dataclass
class EnhancedPerformanceMetrics(PerformanceMetrics):
    """Enhanced performance metrics with additional data."""
    # Hardware metrics
    cpu_frequency_ghz: float = 0.0
    gpu_utilization: float = 0.0
    memory_bandwidth_gbps: float = 0.0
    cache_hit_rate: float = 0.0
    
    # System metrics
    thermal_state: str = "cool"
    power_consumption_watts: float = 0.0
    memory_pressure: float = 0.0
    
    # Workload metrics
    workload_category: str = "general"
    data_size_mb: float = 0.0
    parallelization_efficiency: float = 0.0
    
    # Optimization metrics
    optimization_applied: str = "none"
    speedup_achieved: float = 1.0
    memory_savings_mb: float = 0.0

@dataclass
class OptimizationResult:
    """Result of optimization analysis."""
    optimization_id: str
    strategy: OptimizationStrategy
    target_metric: OptimizationTarget
    performance_improvement: float
    settings: OptimizationSettings
    confidence: float
    execution_time: float
    memory_usage_mb: float
    created_at: float = field(default_factory=time.time)

class MachineLearningOptimizer:
    """Machine learning-based optimization system."""
    
    def __init__(self):
        self.logger = logger.getChild('MachineLearningOptimizer')
        
        # Models for different optimization targets
        self.models: Dict[OptimizationTarget, Any] = {}
        self.scalers: Dict[OptimizationTarget, Any] = {}
        
        # Training data
        self.training_data: Dict[OptimizationTarget, List[Dict[str, Any]]] = {
            target: [] for target in OptimizationTarget
        }
        
        # Model performance tracking
        self.model_performance: Dict[OptimizationTarget, Dict[str, float]] = {
            target: {'accuracy': 0.0, 'last_updated': 0.0} for target in OptimizationTarget
        }
    
    def _prepare_features(self, metrics: EnhancedPerformanceMetrics) -> np.ndarray:
        """Prepare features for ML model."""
        features = [
            metrics.cpu_usage,
            metrics.memory_usage,
            metrics.gpu_usage,
            metrics.temperature,
            metrics.power_consumption,
            metrics.execution_time,
            metrics.throughput,
            metrics.cpu_frequency_ghz,
            metrics.gpu_utilization,
            metrics.memory_bandwidth_gbps,
            metrics.cache_hit_rate,
            metrics.power_consumption_watts,
            metrics.memory_pressure,
            metrics.data_size_mb,
            metrics.parallelization_efficiency
        ]
        return np.array(features).reshape(1, -1)
    
    def train_model(self, target: OptimizationTarget, training_data: List[Dict[str, Any]]):
        """Train ML model for specific optimization target."""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available, using simple heuristics")
            return
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for data_point in training_data:
                features = self._prepare_features(data_point['metrics'])
                target_value = data_point['target_value']
                
                X.append(features.flatten())
                y.append(target_value)
            
            X = np.array(X)
            y = np.array(y)
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model based on target
            if target == OptimizationTarget.PERFORMANCE:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif target == OptimizationTarget.MEMORY_USAGE:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            else:
                model = LinearRegression()
            
            model.fit(X_scaled, y)
            
            # Store model and scaler
            self.models[target] = model
            self.scalers[target] = scaler
            
            # Calculate accuracy
            y_pred = model.predict(X_scaled)
            accuracy = 1.0 - np.mean(np.abs(y - y_pred) / (np.abs(y) + 1e-8))
            
            self.model_performance[target] = {
                'accuracy': accuracy,
                'last_updated': time.time()
            }
            
            self.logger.info(f"🤖 Trained {target.value} model with accuracy: {accuracy:.3f}")
            
        except Exception as e:
            self.logger.error(f"Failed to train model for {target}: {e}")
    
    def predict_optimal_settings(self, metrics: EnhancedPerformanceMetrics, 
                               target: OptimizationTarget) -> OptimizationSettings:
        """Predict optimal settings using ML model."""
        if target not in self.models:
            return self._get_default_settings()
        
        try:
            # Prepare features
            features = self._prepare_features(metrics)
            features_scaled = self.scalers[target].transform(features)
            
            # Make prediction
            prediction = self.models[target].predict(features_scaled)[0]
            
            # Convert prediction to settings
            settings = self._prediction_to_settings(prediction, target, metrics)
            
            return settings
            
        except Exception as e:
            self.logger.warning(f"ML prediction failed: {e}")
            return self._get_default_settings()
    
    def _prediction_to_settings(self, prediction: float, target: OptimizationTarget, 
                               metrics: EnhancedPerformanceMetrics) -> OptimizationSettings:
        """Convert ML prediction to optimization settings."""
        # This is a simplified conversion - in practice, this would be more complex
        
        if target == OptimizationTarget.PERFORMANCE:
            cpu_cores_performance = min(8, max(2, int(prediction * 4)))
            cpu_cores_efficiency = max(2, 8 - cpu_cores_performance)
            cpu_frequency_scaling = min(1.0, max(0.5, prediction))
        else:
            cpu_cores_performance = 4
            cpu_cores_efficiency = 4
            cpu_frequency_scaling = 0.8
        
        return OptimizationSettings(
            cpu_cores_performance=cpu_cores_performance,
            cpu_cores_efficiency=cpu_cores_efficiency,
            cpu_frequency_scaling=cpu_frequency_scaling,
            memory_allocation_strategy="adaptive",
            gpu_acceleration_enabled=metrics.gpu_usage > 0.1,
            gpu_memory_pool_size=min(2048.0, max(512.0, prediction * 1000)),
            thermal_threshold=85.0,
            power_limit=25.0,
            optimization_level="adaptive"
        )
    
    def _get_default_settings(self) -> OptimizationSettings:
        """Get default optimization settings."""
        return OptimizationSettings(
            cpu_cores_performance=4,
            cpu_cores_efficiency=4,
            cpu_frequency_scaling=0.8,
            memory_allocation_strategy="balanced",
            gpu_acceleration_enabled=True,
            gpu_memory_pool_size=1024.0,
            thermal_threshold=80.0,
            power_limit=25.0,
            optimization_level="balanced"
        )

class NeuralNetworkOptimizer:
    """Neural network-based optimization system."""
    
    def __init__(self):
        self.logger = logger.getChild('NeuralNetworkOptimizer')
        
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available, neural network optimization disabled")
            self.enabled = False
            return
        
        self.enabled = True
        
        # Neural network models
        self.models: Dict[OptimizationTarget, nn.Module] = {}
        self.optimizers: Dict[OptimizationTarget, Any] = {}
        
        # Training data
        self.training_data: Dict[OptimizationTarget, List[Dict[str, Any]]] = {
            target: [] for target in OptimizationTarget
        }
    
    def _create_model(self, target: OptimizationTarget):
        """Create neural network model for optimization target."""
        if not TORCH_AVAILABLE:
            return None
            
        class OptimizationNet(torch.nn.Module):
            def __init__(self, input_size=15, hidden_size=64, output_size=1):
                super().__init__()
                self.network = torch.nn.Sequential(
                    torch.nn.Linear(input_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(hidden_size, hidden_size // 2),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.2),
                    torch.nn.Linear(hidden_size // 2, output_size)
                )
            
            def forward(self, x):
                return self.network(x)
        
        return OptimizationNet()
    
    def train_model(self, target: OptimizationTarget, training_data: List[Dict[str, Any]]):
        """Train neural network model."""
        if not self.enabled:
            return
        
        try:
            # Prepare training data
            X = []
            y = []
            
            for data_point in training_data:
                features = self._prepare_features(data_point['metrics'])
                target_value = data_point['target_value']
                
                X.append(features.flatten())
                y.append(target_value)
            
            X = np.array(X, dtype=np.float32)
            y = np.array(y, dtype=np.float32)
            
            # Convert to PyTorch tensors
            X_tensor = torch.from_numpy(X)
            y_tensor = torch.from_numpy(y).unsqueeze(1)
            
            # Create model if not exists
            if target not in self.models:
                self.models[target] = self._create_model(target)
                self.optimizers[target] = torch.optim.Adam(
                    self.models[target].parameters(), lr=0.001
                )
            
            model = self.models[target]
            optimizer = self.optimizers[target]
            criterion = torch.nn.MSELoss()
            
            # Training loop
            model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    self.logger.debug(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            self.logger.info(f"🧠 Trained neural network for {target.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to train neural network for {target}: {e}")
    
    def _prepare_features(self, metrics: EnhancedPerformanceMetrics) -> np.ndarray:
        """Prepare features for neural network."""
        features = [
            metrics.cpu_usage,
            metrics.memory_usage,
            metrics.gpu_usage,
            metrics.temperature,
            metrics.power_consumption,
            metrics.execution_time,
            metrics.throughput,
            metrics.cpu_frequency_ghz,
            metrics.gpu_utilization,
            metrics.memory_bandwidth_gbps,
            metrics.cache_hit_rate,
            metrics.power_consumption_watts,
            metrics.memory_pressure,
            metrics.data_size_mb,
            metrics.parallelization_efficiency
        ]
        return np.array(features)
    
    def predict_optimal_settings(self, metrics: EnhancedPerformanceMetrics, 
                               target: OptimizationTarget) -> OptimizationSettings:
        """Predict optimal settings using neural network."""
        if not self.enabled or target not in self.models:
            return self._get_default_settings()
        
        try:
            # Prepare features
            features = self._prepare_features(metrics)
            features_tensor = torch.from_numpy(features.astype(np.float32)).unsqueeze(0)
            
            # Make prediction
            model = self.models[target]
            model.eval()
            with torch.no_grad():
                prediction = model(features_tensor).item()
            
            # Convert prediction to settings
            settings = self._prediction_to_settings(prediction, target, metrics)
            
            return settings
            
        except Exception as e:
            self.logger.warning(f"Neural network prediction failed: {e}")
            return self._get_default_settings()
    
    def _prediction_to_settings(self, prediction: float, target: OptimizationTarget, 
                               metrics: EnhancedPerformanceMetrics) -> OptimizationSettings:
        """Convert neural network prediction to optimization settings."""
        # Similar to ML optimizer but with neural network-specific logic
        if target == OptimizationTarget.PERFORMANCE:
            cpu_cores_performance = min(8, max(2, int(prediction * 6)))
            cpu_cores_efficiency = max(2, 8 - cpu_cores_performance)
            cpu_frequency_scaling = min(1.0, max(0.5, prediction * 1.2))
        else:
            cpu_cores_performance = 4
            cpu_cores_efficiency = 4
            cpu_frequency_scaling = 0.8
        
        return OptimizationSettings(
            cpu_cores_performance=cpu_cores_performance,
            cpu_cores_efficiency=cpu_cores_efficiency,
            cpu_frequency_scaling=cpu_frequency_scaling,
            memory_allocation_strategy="neural_adaptive",
            gpu_acceleration_enabled=metrics.gpu_usage > 0.1,
            gpu_memory_pool_size=min(2048.0, max(512.0, prediction * 1200)),
            thermal_threshold=85.0,
            power_limit=25.0,
            optimization_level="neural_adaptive"
        )
    
    def _get_default_settings(self) -> OptimizationSettings:
        """Get default optimization settings."""
        return OptimizationSettings(
            cpu_cores_performance=4,
            cpu_cores_efficiency=4,
            cpu_frequency_scaling=0.8,
            memory_allocation_strategy="balanced",
            gpu_acceleration_enabled=True,
            gpu_memory_pool_size=1024.0,
            thermal_threshold=80.0,
            power_limit=25.0,
            optimization_level="balanced"
        )

class AdaptiveOptimizationEngine:
    """Enhanced adaptive optimization engine with ML and neural networks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger.getChild('AdaptiveOptimizationEngine')
        
        # Initialize optimizers
        self.ml_optimizer = MachineLearningOptimizer()
        self.nn_optimizer = NeuralNetworkOptimizer()
        
        # Performance tracking
        self.performance_history: List[EnhancedPerformanceMetrics] = []
        self.optimization_results: List[OptimizationResult] = []
        
        # Learning state
        self.learning_enabled = True
        self.learning_algorithm = LearningAlgorithm.RANDOM_FOREST  # Default
        
        # Optimization strategies
        self.current_strategy = OptimizationStrategy.ADAPTIVE
        self.workload_categories: Dict[str, WorkloadCategory] = {}
        
        # Start learning thread
        if self.learning_enabled:
            self._start_learning_thread()
        
        self.logger.info("🧠 Enhanced Adaptive Optimization Engine initialized")
    
    def _start_learning_thread(self):
        """Start learning thread for continuous optimization."""
        def learn():
            while True:
                try:
                    self._perform_learning_cycle()
                    time.sleep(60)  # Learn every minute
                except Exception as e:
                    self.logger.error(f"Learning thread error: {e}")
                    time.sleep(30)
        
        learning_thread = threading.Thread(target=learn, daemon=True)
        learning_thread.start()
        self.logger.info("📚 Learning thread started")
    
    def _perform_learning_cycle(self):
        """Perform one learning cycle."""
        if len(self.performance_history) < 10:
            return  # Need more data
        
        try:
            # Prepare training data for each optimization target
            for target in OptimizationTarget:
                training_data = self._prepare_training_data(target)
                
                if len(training_data) >= 5:  # Minimum training data
                    # Train ML model
                    self.ml_optimizer.train_model(target, training_data)
                    
                    # Train neural network
                    self.nn_optimizer.train_model(target, training_data)
            
            self.logger.debug("📚 Completed learning cycle")
            
        except Exception as e:
            self.logger.warning(f"Learning cycle failed: {e}")
    
    def _prepare_training_data(self, target: OptimizationTarget) -> List[Dict[str, Any]]:
        """Prepare training data for specific optimization target."""
        training_data = []
        
        for metrics in self.performance_history[-100:]:  # Use last 100 data points
            # Calculate target value based on optimization target
            if target == OptimizationTarget.PERFORMANCE:
                target_value = metrics.throughput / max(metrics.execution_time, 1e-8)
            elif target == OptimizationTarget.MEMORY_USAGE:
                target_value = 1.0 / max(metrics.memory_usage, 1e-8)
            elif target == OptimizationTarget.POWER_CONSUMPTION:
                target_value = 1.0 / max(metrics.power_consumption, 1e-8)
            else:
                target_value = metrics.performance_score
            
            training_data.append({
                'metrics': metrics,
                'target_value': target_value
            })
        
        return training_data
    
    def record_performance(self, metrics: EnhancedPerformanceMetrics):
        """Record performance metrics for learning."""
        self.performance_history.append(metrics)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-500:]
    
    def optimize_operation(self, operation_type: str, workload_category: WorkloadCategory,
                          data_size_mb: float, current_metrics: Optional[Dict[str, Any]] = None) -> OptimizationResult:
        """Optimize operation using adaptive learning."""
        
        # Create enhanced metrics
        if current_metrics is None:
            current_metrics = self._get_current_system_metrics()
        
        metrics = EnhancedPerformanceMetrics(
            timestamp=time.time(),
            workload_type=operation_type,
            optimization_level=self.current_strategy.value,
            cpu_usage=current_metrics.get('cpu_usage', 0.0),
            memory_usage=current_metrics.get('memory_usage', 0.0),
            gpu_usage=current_metrics.get('gpu_usage', 0.0),
            temperature=current_metrics.get('temperature', 50.0),
            power_consumption=current_metrics.get('power_consumption', 10.0),
            execution_time=current_metrics.get('execution_time', 1.0),
            throughput=current_metrics.get('throughput', 1.0),
            error_rate=current_metrics.get('error_rate', 0.0),
            optimization_target=OptimizationTarget.PERFORMANCE.value,
            settings_hash=hash(str(current_metrics)),
            performance_score=current_metrics.get('performance_score', 0.5),
            cpu_frequency_ghz=current_metrics.get('cpu_frequency_ghz', 3.2),
            gpu_utilization=current_metrics.get('gpu_utilization', 0.0),
            memory_bandwidth_gbps=current_metrics.get('memory_bandwidth_gbps', 100.0),
            cache_hit_rate=current_metrics.get('cache_hit_rate', 0.8),
            thermal_state=current_metrics.get('thermal_state', 'cool'),
            power_consumption_watts=current_metrics.get('power_consumption_watts', 15.0),
            memory_pressure=current_metrics.get('memory_pressure', 0.5),
            workload_category=workload_category.value,
            data_size_mb=data_size_mb,
            parallelization_efficiency=current_metrics.get('parallelization_efficiency', 0.7)
        )
        
        # Record metrics
        self.record_performance(metrics)
        
        # Determine optimization strategy based on workload category
        strategy = self._determine_strategy(workload_category)
        target = self._determine_target(workload_category)
        
        # Get optimal settings using learning algorithms
        if self.learning_algorithm == LearningAlgorithm.RANDOM_FOREST:
            settings = self.ml_optimizer.predict_optimal_settings(metrics, target)
        elif self.learning_algorithm == LearningAlgorithm.NEURAL_NETWORK:
            settings = self.nn_optimizer.predict_optimal_settings(metrics, target)
        else:
            settings = self._get_heuristic_settings(metrics, target)
        
        # Create optimization result
        optimization_id = f"opt_{int(time.time())}_{hash(operation_type)}"
        result = OptimizationResult(
            optimization_id=optimization_id,
            strategy=strategy,
            target_metric=target,
            performance_improvement=self._estimate_improvement(metrics, settings),
            settings=settings,
            confidence=self._calculate_confidence(metrics, settings),
            execution_time=0.0,  # Will be updated after execution
            memory_usage_mb=data_size_mb
        )
        
        # Store result
        self.optimization_results.append(result)
        
        # Keep only recent results
        if len(self.optimization_results) > 100:
            self.optimization_results = self.optimization_results[-50:]
        
        self.logger.info(f"🎯 Generated optimization for {operation_type} using {strategy.value} strategy")
        
        return result
    
    def _determine_strategy(self, workload_category: WorkloadCategory) -> OptimizationStrategy:
        """Determine optimization strategy based on workload category."""
        strategy_mapping = {
            WorkloadCategory.MACHINE_LEARNING: OptimizationStrategy.AGGRESSIVE,
            WorkloadCategory.FINANCIAL_MODELING: OptimizationStrategy.BALANCED,
            WorkloadCategory.DATA_PROCESSING: OptimizationStrategy.CONSERVATIVE,
            WorkloadCategory.REAL_TIME_TRADING: OptimizationStrategy.MAXIMUM,
            WorkloadCategory.BACKTESTING: OptimizationStrategy.AGGRESSIVE,
            WorkloadCategory.FEATURE_ENGINEERING: OptimizationStrategy.BALANCED,
            WorkloadCategory.MATRIX_OPERATIONS: OptimizationStrategy.AGGRESSIVE,
            WorkloadCategory.NEURAL_INFERENCE: OptimizationStrategy.MAXIMUM
        }
        
        return strategy_mapping.get(workload_category, OptimizationStrategy.ADAPTIVE)
    
    def _determine_target(self, workload_category: WorkloadCategory) -> OptimizationTarget:
        """Determine optimization target based on workload category."""
        target_mapping = {
            WorkloadCategory.MACHINE_LEARNING: OptimizationTarget.PERFORMANCE,
            WorkloadCategory.FINANCIAL_MODELING: OptimizationTarget.BALANCED,
            WorkloadCategory.DATA_PROCESSING: OptimizationTarget.MEMORY_USAGE,
            WorkloadCategory.REAL_TIME_TRADING: OptimizationTarget.PERFORMANCE,
            WorkloadCategory.BACKTESTING: OptimizationTarget.PERFORMANCE,
            WorkloadCategory.FEATURE_ENGINEERING: OptimizationTarget.BALANCED,
            WorkloadCategory.MATRIX_OPERATIONS: OptimizationTarget.PERFORMANCE,
            WorkloadCategory.NEURAL_INFERENCE: OptimizationTarget.PERFORMANCE
        }
        
        return target_mapping.get(workload_category, OptimizationTarget.BALANCED)
    
    def _get_heuristic_settings(self, metrics: EnhancedPerformanceMetrics, 
                               target: OptimizationTarget) -> OptimizationSettings:
        """Get settings using heuristic rules."""
        # Simple heuristic-based optimization
        if target == OptimizationTarget.PERFORMANCE:
            cpu_cores_performance = 6 if metrics.cpu_usage > 0.7 else 4
            cpu_cores_efficiency = 2 if metrics.cpu_usage > 0.7 else 4
            cpu_frequency_scaling = 1.0 if metrics.temperature < 70 else 0.8
        else:
            cpu_cores_performance = 4
            cpu_cores_efficiency = 4
            cpu_frequency_scaling = 0.8
        
        return OptimizationSettings(
            cpu_cores_performance=cpu_cores_performance,
            cpu_cores_efficiency=cpu_cores_efficiency,
            cpu_frequency_scaling=cpu_frequency_scaling,
            memory_allocation_strategy="heuristic",
            gpu_acceleration_enabled=metrics.gpu_usage > 0.1,
            gpu_memory_pool_size=1024.0,
            thermal_threshold=80.0,
            power_limit=25.0,
            optimization_level="heuristic"
        )
    
    def _estimate_improvement(self, metrics: EnhancedPerformanceMetrics, 
                            settings: OptimizationSettings) -> float:
        """Estimate performance improvement from settings."""
        # Simple estimation based on settings
        improvement = 1.0
        
        if settings.cpu_cores_performance > 4:
            improvement *= 1.2
        
        if settings.cpu_frequency_scaling > 0.9:
            improvement *= 1.1
        
        if settings.gpu_acceleration_enabled and metrics.gpu_usage > 0:
            improvement *= 1.3
        
        return improvement
    
    def _calculate_confidence(self, metrics: EnhancedPerformanceMetrics, 
                            settings: OptimizationSettings) -> float:
        """Calculate confidence in optimization settings."""
        # Simple confidence calculation
        confidence = 0.5  # Base confidence
        
        if len(self.performance_history) > 50:
            confidence += 0.2
        
        if metrics.performance_score > 0.7:
            confidence += 0.1
        
        if settings.optimization_level in ["adaptive", "neural_adaptive"]:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def _get_current_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        try:
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            return {
                'cpu_usage': cpu_percent / 100.0,
                'memory_usage': memory.percent / 100.0,
                'gpu_usage': 0.0,  # Would need GPU monitoring
                'temperature': 50.0,  # Would need thermal monitoring
                'power_consumption': 15.0,  # Would need power monitoring
                'execution_time': 1.0,
                'throughput': 1.0,
                'error_rate': 0.0,
                'performance_score': 0.5,
                'cpu_frequency_ghz': 3.2,
                'gpu_utilization': 0.0,
                'memory_bandwidth_gbps': 100.0,
                'cache_hit_rate': 0.8,
                'thermal_state': 'cool',
                'power_consumption_watts': 15.0,
                'memory_pressure': memory.percent / 100.0,
                'parallelization_efficiency': 0.7
            }
        except Exception as e:
            self.logger.warning(f"Failed to get system metrics: {e}")
            return {}
    
    def get_optimization_metrics(self) -> Dict[str, Any]:
        """Get optimization engine metrics."""
        return {
            'learning_enabled': self.learning_enabled,
            'learning_algorithm': self.learning_algorithm.value,
            'current_strategy': self.current_strategy.value,
            'performance_history_count': len(self.performance_history),
            'optimization_results_count': len(self.optimization_results),
            'ml_models_trained': len(self.ml_optimizer.models),
            'nn_models_trained': len(self.nn_optimizer.models) if self.nn_optimizer.enabled else 0,
            'recent_optimizations': [
                {
                    'id': result.optimization_id,
                    'strategy': result.strategy.value,
                    'improvement': result.performance_improvement,
                    'confidence': result.confidence
                }
                for result in self.optimization_results[-10:]
            ]
        }

# Global instance
_adaptive_optimization_engine: Optional[AdaptiveOptimizationEngine] = None

def get_adaptive_optimization_engine() -> AdaptiveOptimizationEngine:
    """Get or create the global adaptive optimization engine."""
    global _adaptive_optimization_engine
    
    if _adaptive_optimization_engine is None:
        _adaptive_optimization_engine = AdaptiveOptimizationEngine()
    
    return _adaptive_optimization_engine

def adaptive_feature_selection(data: Any, learn_from_execution: bool = True) -> Any:
    """Backward compatible function for adaptive feature selection."""
    engine = get_adaptive_optimization_engine()
    
    # Determine workload category
    workload_category = WorkloadCategory.FEATURE_ENGINEERING
    
    # Calculate data size
    data_size_mb = data.nbytes / (1024 * 1024) if hasattr(data, 'nbytes') else 100.0
    
    # Get optimization
    optimization = engine.optimize_operation(
        operation_type="feature_selection",
        workload_category=workload_category,
        data_size_mb=data_size_mb
    )
    
    # Apply optimization settings
    if optimization.settings.gpu_acceleration_enabled:
        # Use GPU acceleration
        pass
    
    if optimization.settings.memory_allocation_strategy == "adaptive":
        # Use adaptive memory allocation
        pass
    
    # Record performance for learning
    if learn_from_execution:
        # This would record actual performance metrics after execution
        pass
    
    return data

def get_adaptive_optimization_metrics() -> Dict[str, Any]:
    """Get adaptive optimization metrics."""
    engine = get_adaptive_optimization_engine()
    return engine.get_optimization_metrics()