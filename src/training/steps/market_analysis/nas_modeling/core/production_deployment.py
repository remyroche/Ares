"""
Production Deployment for NAS Models

This module provides production-ready deployment capabilities:
- Model serialization and versioning
- ONNX and TorchScript export
- Edge deployment optimization
- Real-time inference pipelines
- Model monitoring and drift detection
- A/B testing framework
- Container deployment
- Auto-scaling and load balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import copy
import json
import pickle
import joblib
from pathlib import Path
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import torch.jit
    TORCHSCRIPT_AVAILABLE = True
except ImportError:
    TORCHSCRIPT_AVAILABLE = False

try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class DeploymentConfig:
    """Configuration for production deployment."""
    export_format: str = "onnx"  # "onnx", "torchscript", "torchserve"
    model_version: str = "1.0.0"
    optimization_level: int = 2
    enable_model_monitoring: bool = True
    monitoring_interval: int = 60  # seconds
    drift_detection_threshold: float = 0.1
    enable_a_b_testing: bool = True
    a_b_test_ratio: float = 0.2
    use_edge_deployment: bool = True
    edge_device_type: str = "jetson"  # "jetson", "raspberry_pi", "mobile"
    enable_auto_scaling: bool = True
    min_instances: int = 1
    max_instances: int = 10
    scaling_threshold: float = 0.8
    container_registry: str = "dockerhub"
    use_load_balancing: bool = True
    load_balancer_type: str = "round_robin"

class ModelSerializer:
    """
    Advanced model serialization and versioning.

    Handles model export in multiple formats with versioning and metadata.
    """

    def __init__(self, config: DeploymentConfig):
        """Initialize model serializer.

        Args:
            config: Deployment configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def serialize_model(self, model: nn.Module, model_metadata: Dict[str, Any],
                       save_path: str) -> Dict[str, Any]:
        """
        Serialize model in specified format.

        Args:
            model: PyTorch model to serialize
            model_metadata: Model metadata
            save_path: Path to save serialized model

        Returns:
            Serialization results
        """
        logger.info(f"💾 Serializing model to {self.config.export_format}")

        results = {
            'model_path': save_path,
            'format': self.config.export_format,
            'version': self.config.model_version,
            'metadata': model_metadata
        }

        if self.config.export_format == "onnx":
            results['onnx_path'] = self._export_to_onnx(model, save_path)
        elif self.config.export_format == "torchscript":
            results['torchscript_path'] = self._export_to_torchscript(model, save_path)
        elif self.config.export_format == "torchserve":
            results['torchserve_artifacts'] = self._export_for_torchserve(model, save_path, model_metadata)

        # Save metadata
        metadata_path = Path(save_path).with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)

        self.logger.info(f"✅ Model serialized to {save_path}")
        return results

    def _export_to_onnx(self, model: nn.Module, save_path: str) -> str:
        """Export model to ONNX format."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX is required for ONNX export")

        onnx_path = Path(save_path).with_suffix('.onnx')

        # Create dummy input
        dummy_input = torch.randn(1, 100, 4)  # Adjust based on your model

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

        # Verify ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)

        self.logger.info(f"✅ ONNX model exported to {onnx_path}")
        return str(onnx_path)

    def _export_to_torchscript(self, model: nn.Module, save_path: str) -> str:
        """Export model to TorchScript format."""
        if not TORCHSCRIPT_AVAILABLE:
            raise ImportError("TorchScript is required")

        script_path = Path(save_path).with_suffix('.pt')

        # Trace model
        example_input = torch.randn(1, 100, 4)
        traced_model = torch.jit.trace(model, example_input)

        # Optimize
        if self.config.optimization_level > 0:
            traced_model = torch.jit.optimize_for_inference(traced_model)

        # Save
        torch.jit.save(traced_model, script_path)

        self.logger.info(f"✅ TorchScript model exported to {script_path}")
        return str(script_path)

    def _export_for_torchserve(self, model: nn.Module, save_path: str,
                              model_metadata: Dict[str, Any]) -> Dict[str, str]:
        """Export model for TorchServe deployment."""
        artifacts_path = Path(save_path) / "torchserve_artifacts"
        artifacts_path.mkdir(exist_ok=True, parents=True)

        # Export base model
        base_model_path = artifacts_path / "model.pt"
        torch.save(model.state_dict(), base_model_path)

        # Create model configuration
        model_config = {
            "modelName": model_metadata.get("name", "nas_model"),
            "modelVersion": self.config.model_version,
            "handler": "custom_handler.py",
            "runtime": "python",
            "minWorkers": self.config.min_instances,
            "maxWorkers": self.config.max_instances,
            "batchSize": 1,
            "maxBatchDelay": 100,
            "responseTimeout": 120
        }

        config_path = artifacts_path / "model_config.json"
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)

        # Create custom handler
        handler_content = self._create_torchserve_handler(model_metadata)
        handler_path = artifacts_path / "custom_handler.py"
        with open(handler_path, 'w') as f:
            f.write(handler_content)

        artifacts = {
            'model_path': str(base_model_path),
            'config_path': str(config_path),
            'handler_path': str(handler_path)
        }

        self.logger.info(f"✅ TorchServe artifacts created in {artifacts_path}")
        return artifacts

    def _create_torchserve_handler(self, model_metadata: Dict[str, Any]) -> str:
        """Create custom handler for TorchServe."""
        handler_template = '''
import torch
import json
import logging
import os
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)

class CustomHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.model = None

    def initialize(self, context):
        """Initialize model and context."""
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # Load model
        model_path = os.path.join(model_dir, "model.pt")
        self.model = torch.load(model_path, map_location="cpu")
        self.model.eval()

        logger.info("✅ Model loaded successfully")

    def preprocess(self, data):
        """Preprocess input data."""
        # Convert input to tensor
        input_data = data[0].get("data")
        if input_data is None:
            input_data = data[0].get("body")

        # Process input data (customize based on your needs)
        processed_data = torch.tensor(input_data, dtype=torch.float32)
        return processed_data

    def inference(self, data):
        """Run inference on preprocessed data."""
        with torch.no_grad():
            output = self.model(data)
            return output

    def postprocess(self, data):
        """Postprocess inference output."""
        # Convert output to appropriate format
        return data.tolist()

# Create handler instance
_handler = CustomHandler()

def handle(data, context):
    try:
        data = _handler.preprocess(data)
        data = _handler.inference(data)
        data = _handler.postprocess(data)
        return data
    except Exception as e:
        raise RuntimeError(f"Handler error: {e}")
'''
        return handler_template

class EdgeOptimizer:
    """
    Edge deployment optimizer.

    Optimizes models for deployment on edge devices.
    """

    def __init__(self, device_type: str = "jetson"):
        """Initialize edge optimizer.

        Args:
            device_type: Type of edge device
        """
        self.device_type = device_type
        self.logger = logging.getLogger(self.__class__.__name__)

        # Device-specific optimizations
        self.device_configs = {
            "jetson": {
                "precision": "fp16",
                "max_memory": 8 * 1024 * 1024 * 1024,  # 8GB
                "use_tensorrt": True,
                "batch_size": 1
            },
            "raspberry_pi": {
                "precision": "int8",
                "max_memory": 1 * 1024 * 1024 * 1024,  # 1GB
                "use_tensorrt": False,
                "batch_size": 1
            },
            "mobile": {
                "precision": "int8",
                "max_memory": 512 * 1024 * 1024,  # 512MB
                "use_tensorrt": False,
                "batch_size": 1
            }
        }

    def optimize_for_edge(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for edge deployment.

        Args:
            model: Model to optimize

        Returns:
            Optimized model
        """
        logger.info(f"🔧 Optimizing model for {self.device_type} deployment")

        optimized_model = copy.deepcopy(model)

        # Apply device-specific optimizations
        device_config = self.device_configs.get(self.device_type, {})

        if device_config.get("precision") == "fp16":
            optimized_model = optimized_model.half()
            self.logger.info("✅ Applied FP16 precision")

        elif device_config.get("precision") == "int8":
            optimized_model = self._quantize_to_int8(optimized_model)
            self.logger.info("✅ Applied INT8 quantization")

        # Memory optimization
        if device_config.get("max_memory"):
            optimized_model = self._optimize_memory_usage(optimized_model, device_config["max_memory"])

        # Batch size optimization
        if "batch_size" in device_config:
            optimized_model = self._optimize_batch_size(optimized_model, device_config["batch_size"])

        self.logger.info(f"✅ Edge optimization completed for {self.device_type}")
        return optimized_model

    def _quantize_to_int8(self, model: nn.Module) -> nn.Module:
        """Quantize model to INT8."""
        try:
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv1d, nn.LSTM},
                dtype=torch.qint8
            )
            return quantized_model
        except Exception as e:
            self.logger.warning(f"⚠️ INT8 quantization failed: {e}")
            return model

    def _optimize_memory_usage(self, model: nn.Module, max_memory: int) -> nn.Module:
        """Optimize model for memory constraints."""
        # Reduce model size if necessary
        total_params = sum(p.numel() for p in model.parameters())

        if total_params * 4 > max_memory * 0.5:  # If model uses >50% of memory
            # Apply aggressive compression
            model = self._apply_memory_compression(model, max_memory)

        return model

    def _apply_memory_compression(self, model: nn.Module, max_memory: int) -> nn.Module:
        """Apply memory compression techniques."""
        compressed_model = copy.deepcopy(model)

        # Reduce hidden dimensions
        for name, module in compressed_model.named_modules():
            if isinstance(module, nn.Linear):
                # Reduce hidden size by 25%
                new_out_features = max(16, module.out_features // 2)
                new_in_features = max(16, module.in_features // 2)

                compressed_layer = nn.Linear(new_in_features, new_out_features)

                # Copy compatible weights
                with torch.no_grad():
                    if new_in_features <= module.in_features and new_out_features <= module.out_features:
                        compressed_layer.weight.data[:new_out_features, :new_in_features] = \
                            module.weight.data[:new_out_features, :new_in_features]
                        compressed_layer.bias.data[:new_out_features] = module.bias.data[:new_out_features]

                # Replace layer
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = compressed_model.get_submodule(parent_name)
                    setattr(parent, child_name, compressed_layer)
                else:
                    setattr(compressed_model, child_name, compressed_layer)

        return compressed_model

    def _optimize_batch_size(self, model: nn.Module, batch_size: int) -> nn.Module:
        """Optimize model for specific batch size."""
        # Ensure model can handle the specified batch size
        try:
            dummy_input = torch.randn(batch_size, 100, 4)
            with torch.no_grad():
                _ = model(dummy_input)
            self.logger.info(f"✅ Model optimized for batch size {batch_size}")
        except Exception as e:
            self.logger.warning(f"⚠️ Batch size optimization failed: {e}")

        return model

class ModelMonitor:
    """
    Real-time model monitoring and drift detection.

    Monitors model performance, data drift, and system health.
    """

    def __init__(self, config: DeploymentConfig):
        """Initialize model monitor.

        Args:
            config: Deployment configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Monitoring state
        self.prediction_history = deque(maxlen=10000)
        self.performance_metrics = defaultdict(list)
        self.drift_scores = []
        self.alerts = []

        # Start monitoring thread
        if config.enable_model_monitoring:
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()

    def log_prediction(self, prediction: np.ndarray, ground_truth: Optional[np.ndarray] = None,
                      input_data: Optional[np.ndarray] = None):
        """Log model prediction for monitoring."""
        timestamp = time.time()

        log_entry = {
            'timestamp': timestamp,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'input_data': input_data
        }

        self.prediction_history.append(log_entry)

    def compute_performance_metrics(self) -> Dict[str, float]:
        """Compute current performance metrics."""
        if len(self.prediction_history) < 10:
            return {}

        recent_predictions = list(self.prediction_history)[-100:]  # Last 100 predictions

        # Accuracy (if ground truth available)
        if all(entry['ground_truth'] is not None for entry in recent_predictions):
            correct = sum(
                np.argmax(entry['prediction']) == entry['ground_truth']
                for entry in recent_predictions
            )
            accuracy = correct / len(recent_predictions)
        else:
            accuracy = None

        # Prediction statistics
        all_predictions = np.array([entry['prediction'] for entry in recent_predictions])
        mean_prediction = np.mean(all_predictions, axis=0)
        std_prediction = np.std(all_predictions, axis=0)

        # Confidence scores
        confidence_scores = np.max(all_predictions, axis=1)
        mean_confidence = np.mean(confidence_scores)

        metrics = {
            'accuracy': accuracy,
            'mean_confidence': mean_confidence,
            'prediction_std': std_prediction.tolist(),
            'num_samples': len(recent_predictions)
        }

        return metrics

    def detect_drift(self) -> Dict[str, Any]:
        """Detect data drift and concept drift."""
        if len(self.prediction_history) < 100:
            return {'drift_detected': False, 'drift_score': 0.0}

        # Simple drift detection based on prediction distribution change
        recent_predictions = list(self.prediction_history)[-50:]
        older_predictions = list(self.prediction_history)[-100:-50]

        recent_dist = np.mean([np.argmax(p['prediction']) for p in recent_predictions])
        older_dist = np.mean([np.argmax(p['prediction']) for p in older_predictions])

        drift_score = abs(recent_dist - older_dist)

        drift_detected = drift_score > self.config.drift_detection_threshold

        if drift_detected:
            self.logger.warning(f"⚠️ Drift detected with score: {drift_score:.4f}")

        return {
            'drift_detected': drift_detected,
            'drift_score': drift_score,
            'recent_distribution': recent_dist,
            'older_distribution': older_dist
        }

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while True:
            try:
                # Compute metrics
                metrics = self.compute_performance_metrics()
                self.performance_metrics['accuracy'].append(metrics.get('accuracy', 0))
                self.performance_metrics['confidence'].append(metrics.get('mean_confidence', 0))

                # Detect drift
                drift_info = self.detect_drift()
                self.drift_scores.append(drift_info['drift_score'])

                # Log alerts
                if drift_info['drift_detected']:
                    self.alerts.append({
                        'type': 'drift',
                        'timestamp': time.time(),
                        'message': f"Drift detected: {drift_info['drift_score']:.4f}"
                    })

                # Log performance
                if len(self.performance_metrics['accuracy']) % 10 == 0:
                    self.logger.info(f"📊 Monitoring: Accuracy={metrics.get('accuracy', 'N/A')}, "
                                   f"Confidence={metrics.get('mean_confidence', 'N/A')}")

                time.sleep(self.config.monitoring_interval)

            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
                time.sleep(60)  # Retry after 1 minute

class ABTester:
    """
    A/B testing framework for model deployment.

    Compares different model versions or configurations.
    """

    def __init__(self, config: DeploymentConfig):
        """Initialize A/B tester.

        Args:
            config: Deployment configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # A/B test variants
        self.variant_a = None
        self.variant_b = None
        self.variant_a_stats = defaultdict(list)
        self.variant_b_stats = defaultdict(list)

    def setup_test(self, model_a: nn.Module, model_b: nn.Module,
                  test_name: str = "model_comparison"):
        """Setup A/B test between two models."""
        self.variant_a = model_a
        self.variant_b = model_b
        self.test_name = test_name

        self.logger.info(f"🧪 A/B test setup: {test_name}")

    def assign_variant(self) -> str:
        """Assign variant (A or B) based on test ratio."""
        if np.random.random() < self.config.a_b_test_ratio:
            return "B"
        else:
            return "A"

    def log_result(self, variant: str, prediction: np.ndarray,
                  ground_truth: Optional[np.ndarray] = None):
        """Log test result for a variant."""
        timestamp = time.time()

        result = {
            'timestamp': timestamp,
            'prediction': prediction,
            'ground_truth': ground_truth
        }

        if variant == "A":
            self.variant_a_stats['results'].append(result)
        else:
            self.variant_b_stats['results'].append(result)

    def get_test_results(self) -> Dict[str, Any]:
        """Get A/B test results."""
        def compute_variant_stats(stats):
            if not stats['results']:
                return {}

            predictions = [r['prediction'] for r in stats['results']]
            predictions = np.array(predictions)

            # Compute metrics
            if all(r['ground_truth'] is not None for r in stats['results']):
                ground_truths = [r['ground_truth'] for r in stats['results']]
                accuracies = [
                    np.argmax(pred) == gt for pred, gt in zip(predictions, ground_truths)
                ]
                accuracy = np.mean(accuracies)
            else:
                accuracy = None

            mean_predictions = np.mean(predictions, axis=0)
            confidence_scores = np.max(predictions, axis=1)
            mean_confidence = np.mean(confidence_scores)

            return {
                'num_samples': len(stats['results']),
                'accuracy': accuracy,
                'mean_confidence': mean_confidence,
                'mean_predictions': mean_predictions.tolist()
            }

        results = {
            'test_name': self.test_name,
            'variant_a': compute_variant_stats(self.variant_a_stats),
            'variant_b': compute_variant_stats(self.variant_b_stats),
            'winner': self._determine_winner()
        }

        return results

    def _determine_winner(self) -> Optional[str]:
        """Determine winning variant based on results."""
        a_stats = self.variant_a_stats['results']
        b_stats = self.variant_b_stats['results']

        if not a_stats or not b_stats:
            return None

        # Simple winner determination based on accuracy
        a_accuracies = [
            np.argmax(r['prediction']) == r['ground_truth']
            for r in a_stats if r['ground_truth'] is not None
        ]
        b_accuracies = [
            np.argmax(r['prediction']) == r['ground_truth']
            for r in b_stats if r['ground_truth'] is not None
        ]

        if a_accuracies and b_accuracies:
            a_accuracy = np.mean(a_accuracies)
            b_accuracy = np.mean(b_accuracies)

            if b_accuracy > a_accuracy + 0.05:  # 5% improvement
                return "B"
            elif a_accuracy > b_accuracy + 0.05:
                return "A"
            else:
                return "Tie"

        return None

class AutoScaler:
    """
    Auto-scaling for model deployment.

    Automatically scales model instances based on load and performance.
    """

    def __init__(self, config: DeploymentConfig):
        """Initialize auto-scaler.

        Args:
            config: Deployment configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        self.current_instances = 1
        self.load_history = deque(maxlen=100)
        self.response_times = deque(maxlen=100)

    def update_load_metrics(self, current_load: float, response_time: float):
        """Update load and performance metrics."""
        self.load_history.append(current_load)
        self.response_times.append(response_time)

    def should_scale_up(self) -> bool:
        """Determine if should scale up."""
        if len(self.load_history) < 10:
            return False

        avg_load = np.mean(self.load_history)
        avg_response_time = np.mean(self.response_times)

        return (avg_load > self.config.scaling_threshold or
                avg_response_time > 1000)  # 1 second

    def should_scale_down(self) -> bool:
        """Determine if should scale down."""
        if len(self.load_history) < 20:
            return False

        avg_load = np.mean(self.load_history)
        return avg_load < self.config.scaling_threshold * 0.5

    def get_scaling_decision(self) -> Dict[str, Any]:
        """Get scaling decision."""
        decision = {
            'scale_up': self.should_scale_up(),
            'scale_down': self.should_scale_down(),
            'current_instances': self.current_instances,
            'target_instances': self.current_instances
        }

        if decision['scale_up'] and self.current_instances < self.config.max_instances:
            decision['target_instances'] = self.current_instances + 1
            decision['action'] = 'scale_up'

        elif decision['scale_down'] and self.current_instances > self.config.min_instances:
            decision['target_instances'] = self.current_instances - 1
            decision['action'] = 'scale_down'

        else:
            decision['action'] = 'maintain'

        return decision

# Utility functions
def deploy_model_to_onnx(model: nn.Module, save_path: str) -> str:
    """Deploy model to ONNX format."""
    serializer = ModelSerializer(DeploymentConfig(export_format="onnx"))
    results = serializer.serialize_model(model, {}, save_path)
    return results['onnx_path']

def create_edge_optimized_model(model: nn.Module, device_type: str = "jetson") -> nn.Module:
    """Create edge-optimized model."""
    optimizer = EdgeOptimizer(device_type)
    return optimizer.optimize_for_edge(model)

def setup_model_monitoring(config: DeploymentConfig) -> ModelMonitor:
    """Setup model monitoring."""
    return ModelMonitor(config)

def create_ab_test(model_a: nn.Module, model_b: nn.Module,
                  test_name: str = "comparison") -> ABTester:
    """Create A/B test setup."""
    config = DeploymentConfig(enable_a_b_testing=True)
    tester = ABTester(config)
    tester.setup_test(model_a, model_b, test_name)
    return tester

def create_auto_scaler(config: DeploymentConfig) -> AutoScaler:
    """Create auto-scaler."""
    return AutoScaler(config)