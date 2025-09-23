"""
Model Compression and Quantization for NAS

This module implements advanced model compression techniques:
- Quantization (post-training and quantization-aware training)
- Pruning (structured and unstructured)
- Knowledge distillation
- Model distillation
- Low-rank factorization
- Architecture search for efficient models
- Dynamic quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import prune
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from collections import OrderedDict
import copy
from pathlib import Path
import gc

logger = logging.getLogger(__name__)

@dataclass
class CompressionConfig:
    """Configuration for model compression."""
    use_quantization: bool = True
    quantization_bits: int = 8
    quantization_type: str = "static"  # "static", "dynamic", "qat"
    use_pruning: bool = True
    pruning_type: str = "structured"  # "structured", "unstructured"
    pruning_amount: float = 0.5
    pruning_schedule: str = "exponential"
    use_knowledge_distillation: bool = True
    distillation_temperature: float = 4.0
    distillation_alpha: float = 0.5
    use_low_rank: bool = True
    rank_factor: int = 4
    use_mixed_precision: bool = True
    mixed_precision_type: str = "fp16"
    compression_targets: List[str] = field(default_factory=lambda: ["accuracy", "size", "latency"])
    target_accuracy_drop: float = 0.02
    target_size_reduction: float = 0.5

class ModelCompressor:
    """
    Comprehensive model compression toolkit.

    Applies multiple compression techniques to optimize models for deployment.
    """

    def __init__(self, config: CompressionConfig):
        """Initialize model compressor.

        Args:
            config: Compression configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Track compression statistics
        self.compression_stats = {
            'original_size': 0,
            'compressed_size': 0,
            'original_params': 0,
            'compressed_params': 0,
            'compression_ratio': 1.0,
            'accuracy_drop': 0.0
        }

    def compress_model(self, model: nn.Module, train_loader: torch.utils.data.DataLoader = None,
                      val_loader: torch.utils.data.DataLoader = None) -> Dict[str, Any]:
        """
        Apply comprehensive model compression.

        Args:
            model: Model to compress
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Compression results
        """
        logger.info("🚀 Starting comprehensive model compression")

        # Store original model info
        original_model = copy.deepcopy(model)
        self._update_compression_stats(original_model, "original")

        compressed_model = copy.deepcopy(model)
        compression_steps = []

        # 1. Apply pruning
        if self.config.use_pruning:
            compressed_model, pruning_info = self._apply_pruning(compressed_model)
            compression_steps.append(f"pruning_{self.config.pruning_type}")
            self.logger.info(f"✅ Applied {self.config.pruning_type} pruning")

        # 2. Apply quantization-aware training if specified
        if self.config.use_quantization and self.config.quantization_type == "qat":
            compressed_model, qat_info = self._apply_qat(compressed_model, train_loader, val_loader)
            compression_steps.append("quantization_aware_training")
            self.logger.info("✅ Applied quantization-aware training")

        # 3. Apply low-rank factorization
        if self.config.use_low_rank:
            compressed_model, lowrank_info = self._apply_low_rank_factorization(compressed_model)
            compression_steps.append("low_rank_factorization")
            self.logger.info("✅ Applied low-rank factorization")

        # 4. Apply post-training quantization
        if self.config.use_quantization and self.config.quantization_type in ["static", "dynamic"]:
            compressed_model, quant_info = self._apply_quantization(compressed_model, train_loader)
            compression_steps.append(f"{self.config.quantization_type}_quantization")
            self.logger.info(f"✅ Applied {self.config.quantization_type} quantization")

        # 5. Apply mixed precision training
        if self.config.use_mixed_precision:
            compressed_model, mixed_info = self._apply_mixed_precision(compressed_model)
            compression_steps.append("mixed_precision_training")
            self.logger.info("✅ Applied mixed precision training")

        # Update compression statistics
        self._update_compression_stats(compressed_model, "compressed")

        results = {
            'original_model': original_model,
            'compressed_model': compressed_model,
            'compression_steps': compression_steps,
            'compression_stats': self.compression_stats,
            'config': self.config
        }

        self.logger.info(f"✅ Model compression completed")
        self.logger.info(f"📊 Compression ratio: {self.compression_stats['compression_ratio']:.2f}x")
        self.logger.info(f"📊 Accuracy drop: {self.compression_stats['accuracy_drop']:.4f}")

        return results

    def _apply_pruning(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply structured or unstructured pruning."""
        if self.config.pruning_type == "structured":
            pruned_model, prune_info = self._structured_pruning(model)
        else:
            pruned_model, prune_info = self._unstructured_pruning(model)

        return pruned_model, prune_info

    def _structured_pruning(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply structured pruning (remove entire filters/channels)."""
        pruned_model = copy.deepcopy(model)

        # Apply pruning to Linear layers
        for name, module in pruned_model.named_modules():
            if isinstance(module, nn.Linear):
                prune.l1_unstructured(module, name='weight', amount=self.config.pruning_amount)
                prune.remove(module, 'weight')

        prune_info = {
            'pruning_type': 'structured',
            'pruning_amount': self.config.pruning_amount,
            'schedule': self.config.pruning_schedule
        }

        return pruned_model, prune_info

    def _unstructured_pruning(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply unstructured pruning (remove individual weights)."""
        pruned_model = copy.deepcopy(model)

        # Apply pruning to all parameters
        parameters_to_prune = []
        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.config.pruning_amount,
        )

        # Remove pruning masks
        for module, param_name in parameters_to_prune:
            prune.remove(module, param_name)

        prune_info = {
            'pruning_type': 'unstructured',
            'pruning_amount': self.config.pruning_amount,
            'schedule': self.config.pruning_schedule
        }

        return pruned_model, prune_info

    def _apply_qat(self, model: nn.Module, train_loader: torch.utils.data.DataLoader,
                  val_loader: torch.utils.data.DataLoader) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply quantization-aware training."""
        qat_model = copy.deepcopy(model)

        # Configure quantization
        qat_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

        # Prepare for QAT
        torch.quantization.prepare_qat(qat_model, inplace=True)

        # Fine-tune with quantization
        qat_model = self._qat_fine_tuning(qat_model, train_loader, val_loader)

        qat_info = {
            'quantization_type': 'qat',
            'bits': self.config.quantization_bits
        }

        return qat_model, qat_info

    def _apply_quantization(self, model: nn.Module,
                           train_loader: torch.utils.data.DataLoader = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply post-training quantization."""
        quantized_model = copy.deepcopy(model)

        if self.config.quantization_type == "static":
            # Static quantization requires calibration
            quantized_model = self._static_quantization(quantized_model, train_loader)
        elif self.config.quantization_type == "dynamic":
            # Dynamic quantization
            quantized_model = self._dynamic_quantization(quantized_model)

        quant_info = {
            'quantization_type': self.config.quantization_type,
            'bits': self.config.quantization_bits
        }

        return quantized_model, quant_info

    def _apply_low_rank_factorization(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply low-rank factorization to linear layers."""
        factorized_model = copy.deepcopy(model)

        # Replace linear layers with low-rank approximations
        for name, module in factorized_model.named_modules():
            if isinstance(module, nn.Linear):
                factorized_layer = self._low_rank_approximation(module, self.config.rank_factor)
                # Replace the module in the model
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                if parent_name:
                    parent = factorized_model.get_submodule(parent_name)
                    setattr(parent, child_name, factorized_layer)
                else:
                    setattr(factorized_model, child_name, factorized_layer)

        lowrank_info = {
            'rank_factor': self.config.rank_factor,
            'method': 'svd'
        }

        return factorized_model, lowrank_info

    def _apply_mixed_precision(self, model: nn.Module) -> Tuple[nn.Module, Dict[str, Any]]:
        """Apply mixed precision training."""
        mixed_model = copy.deepcopy(model)

        # Convert appropriate layers to half precision
        if self.config.mixed_precision_type == "fp16":
            mixed_model = mixed_model.half()

        mixed_info = {
            'precision_type': self.config.mixed_precision_type,
            'conversion_type': 'automatic'
        }

        return mixed_model, mixed_info

    def _static_quantization(self, model: nn.Module,
                           train_loader: torch.utils.data.DataLoader) -> nn.Module:
        """Apply static quantization with calibration."""
        quantized_model = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv1d, nn.LSTM},
            dtype=torch.qint8
        )

        # Calibrate if train_loader is provided
        if train_loader is not None:
            quantized_model.eval()
            with torch.no_grad():
                for batch_x, _ in train_loader:
                    quantized_model(batch_x)

        return quantized_model

    def _dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization."""
        return torch.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype=torch.qint8
        )

    def _qat_fine_tuning(self, model: nn.Module,
                        train_loader: torch.utils.data.DataLoader,
                        val_loader: torch.utils.data.DataLoader) -> nn.Module:
        """Fine-tune model with quantization-aware training."""
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        # Training loop
        model.train()
        for epoch in range(5):  # Short fine-tuning
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        return model

    def _low_rank_approximation(self, layer: nn.Linear, rank_factor: int) -> nn.Module:
        """Apply low-rank approximation to a linear layer."""
        # SVD-based low-rank approximation
        weight = layer.weight.data
        U, s, V = torch.svd(weight)

        # Keep only top components
        rank = max(1, min(weight.size(0), weight.size(1)) // rank_factor)
        U_reduced = U[:, :rank]
        s_reduced = s[:rank]
        V_reduced = V.t()[:rank, :]

        # Create low-rank layers
        low_rank_layer = nn.Sequential(
            nn.Linear(layer.in_features, rank),
            nn.Linear(rank, layer.out_features)
        )

        # Initialize with SVD components
        with torch.no_grad():
            low_rank_layer[0].weight.data = V_reduced
            low_rank_layer[0].bias.data.zero_()
            low_rank_layer[1].weight.data = torch.diag(s_reduced) @ U_reduced.t()
            low_rank_layer[1].bias.data = layer.bias.data.clone()

        return low_rank_layer

    def _update_compression_stats(self, model: nn.Module, stage: str):
        """Update compression statistics."""
        total_params = sum(p.numel() for p in model.parameters())
        total_size = sum(p.numel() * p.element_size() for p in model.parameters())

        if stage == "original":
            self.compression_stats['original_params'] = total_params
            self.compression_stats['original_size'] = total_size
        else:
            self.compression_stats['compressed_params'] = total_params
            self.compression_stats['compressed_size'] = total_size

            if self.compression_stats['original_params'] > 0:
                param_ratio = self.compression_stats['original_params'] / total_params
                size_ratio = self.compression_stats['original_size'] / total_size

                self.compression_stats['compression_ratio'] = (param_ratio + size_ratio) / 2

class KnowledgeDistiller:
    """
    Knowledge distillation for model compression.

    Transfers knowledge from a large teacher model to a smaller student model.
    """

    def __init__(self, teacher_model: nn.Module, student_model: nn.Module,
                 config: CompressionConfig):
        """Initialize knowledge distiller.

        Args:
            teacher_model: Large teacher model
            student_model: Small student model
            config: Distillation configuration
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config

        self.logger = logging.getLogger(self.__class__.__name__)

        # Freeze teacher model
        for param in self.teacher_model.parameters():
            param.requires_grad = False

        # Setup distillation loss
        self.distillation_loss = self._setup_distillation_loss()

    def _setup_distillation_loss(self) -> nn.Module:
        """Setup knowledge distillation loss."""
        return nn.KLDivLoss(reduction='batchmean')

    def distill(self, train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader,
               num_epochs: int = 20) -> Dict[str, Any]:
        """
        Perform knowledge distillation.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of distillation epochs

        Returns:
            Distillation results
        """
        logger.info("🚀 Starting knowledge distillation")

        optimizer = optim.Adam(self.student_model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

        best_accuracy = 0.0
        history = []

        for epoch in range(num_epochs):
            # Training step
            train_loss = self._train_step(train_loader, optimizer)

            # Validation step
            val_accuracy = self._validate_step(val_loader)

            # Update learning rate
            scheduler.step()

            # Track best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                best_student = copy.deepcopy(self.student_model)

            history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_accuracy': val_accuracy
            })

            if epoch % 5 == 0:
                self.logger.info(f"📈 Epoch {epoch}: Loss = {train_loss:.4f}, Accuracy = {val_accuracy:.4f}")

        results = {
            'distilled_model': best_student,
            'teacher_model': self.teacher_model,
            'best_accuracy': best_accuracy,
            'training_history': history,
            'distillation_temperature': self.config.distillation_temperature,
            'distillation_alpha': self.config.distillation_alpha
        }

        self.logger.info(f"✅ Knowledge distillation completed with accuracy: {best_accuracy:.4f}")
        return results

    def _train_step(self, train_loader: torch.utils.data.DataLoader,
                   optimizer: optim.Optimizer) -> float:
        """Perform distillation training step."""
        self.student_model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()

            # Student predictions
            student_logits = self.student_model(batch_x)

            # Teacher predictions (soft targets)
            with torch.no_grad():
                teacher_logits = self.teacher_model(batch_x)

            # Distillation loss
            soft_targets = F.softmax(teacher_logits / self.config.distillation_temperature, dim=1)
            soft_student = F.log_softmax(student_logits / self.config.distillation_temperature, dim=1)

            distillation_loss = self.distillation_loss(soft_student, soft_targets) * (
                self.config.distillation_temperature ** 2
            )

            # Hard loss (cross-entropy with true labels)
            hard_loss = F.cross_entropy(student_logits, batch_y)

            # Combined loss
            loss = (self.config.distillation_alpha * distillation_loss +
                   (1 - self.config.distillation_alpha) * hard_loss)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate_step(self, val_loader: torch.utils.data.DataLoader) -> float:
        """Perform validation step."""
        self.student_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = self.student_model(batch_x)
                predictions = outputs.argmax(dim=1)
                correct += (predictions == batch_y).sum().item()
                total += batch_y.size(0)

        return correct / total

class ModelProfiler:
    """
    Model profiling for performance analysis.

    Measures model size, latency, throughput, and memory usage.
    """

    def __init__(self):
        """Initialize model profiler."""
        self.logger = logging.getLogger(self.__class__.__name__)

    def profile_model(self, model: nn.Module, input_size: Tuple[int, ...] = (1, 100, 4)) -> Dict[str, Any]:
        """
        Profile model performance.

        Args:
            model: Model to profile
            input_size: Size of input tensor

        Returns:
            Profiling results
        """
        logger.info("📊 Profiling model performance")

        # Model size
        param_count = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Memory usage
        memory_stats = self._get_memory_stats(model, input_size)

        # Latency and throughput
        latency_stats = self._get_latency_stats(model, input_size)

        # FLOPs and MACs
        flops_stats = self._get_flops_stats(model, input_size)

        profile_results = {
            'parameter_count': param_count,
            'trainable_parameters': trainable_params,
            'memory_stats': memory_stats,
            'latency_stats': latency_stats,
            'flops_stats': flops_stats,
            'model_size_mb': memory_stats['model_size_mb'],
            'inference_latency_ms': latency_stats['mean_latency_ms'],
            'throughput_samples_per_sec': latency_stats['throughput']
        }

        self.logger.info(f"✅ Model profiling completed")
        self.logger.info(f"📊 Parameters: {param_count","}, Size: {memory_stats['model_size_mb']".2f"} MB")
        self.logger.info(f"⚡ Latency: {latency_stats['mean_latency_ms']".2f"} ms, Throughput: {latency_stats['throughput']".1f"} samples/sec")

        return profile_results

    def _get_memory_stats(self, model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, float]:
        """Get memory usage statistics."""
        # Model size in memory
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024  # MB
        buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 / 1024  # MB

        # Peak memory during forward pass
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with torch.no_grad():
            x = torch.randn(input_size)
            if torch.cuda.is_available():
                x = x.cuda()
                model = model.cuda()

            _ = model(x)

            if torch.cuda.is_available():
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            else:
                peak_memory = 0.0

        return {
            'model_size_mb': param_memory + buffer_memory,
            'parameter_memory_mb': param_memory,
            'buffer_memory_mb': buffer_memory,
            'peak_memory_mb': peak_memory
        }

    def _get_latency_stats(self, model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, float]:
        """Get latency and throughput statistics."""
        latencies = []

        with torch.no_grad():
            x = torch.randn(input_size)
            if torch.cuda.is_available():
                x = x.cuda()
                model = model.cuda()

            # Warm up
            for _ in range(10):
                _ = model(x)

            # Measure latency
            for _ in range(100):
                start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
                end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

                if torch.cuda.is_available():
                    start_time.record()
                    _ = model(x)
                    end_time.record()
                    torch.cuda.synchronize()
                    latency = start_time.elapsed_time(end_time)
                else:
                    start_time = time.time()
                    _ = model(x)
                    end_time = time.time()
                    latency = (end_time - start_time) * 1000  # Convert to milliseconds

                latencies.append(latency)

        mean_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        throughput = 1000 / mean_latency  # samples per second

        return {
            'mean_latency_ms': mean_latency,
            'std_latency_ms': std_latency,
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'throughput': throughput
        }

    def _get_flops_stats(self, model: nn.Module, input_size: Tuple[int, ...]) -> Dict[str, float]:
        """Get FLOPs and MACs statistics."""
        try:
            from ptflops import get_model_complexity_info

            macs, params = get_model_complexity_info(
                model, input_size[1:], as_strings=False, print_per_layer_stat=False
            )

            flops = macs * 2  # FLOPs = 2 * MACs for most operations

            return {
                'macs': macs,
                'flops': flops,
                'gflops': flops / 1e9
            }

        except ImportError:
            self.logger.warning("⚠️ ptflops not available for FLOPs calculation")
            return {'macs': 0, 'flops': 0, 'gflops': 0}

# Utility functions
def compress_model_for_deployment(model: nn.Module,
                                train_loader: torch.utils.data.DataLoader = None,
                                val_loader: torch.utils.data.DataLoader = None,
                                config: CompressionConfig = None) -> Dict[str, Any]:
    """Compress model for deployment."""
    if config is None:
        config = CompressionConfig()

    compressor = ModelCompressor(config)
    return compressor.compress_model(model, train_loader, val_loader)

def distill_knowledge(teacher_model: nn.Module, student_model: nn.Module,
                     train_loader: torch.utils.data.DataLoader,
                     val_loader: torch.utils.data.DataLoader,
                     config: CompressionConfig = None) -> Dict[str, Any]:
    """Distill knowledge from teacher to student model."""
    if config is None:
        config = CompressionConfig()

    distiller = KnowledgeDistiller(teacher_model, student_model, config)
    return distiller.distill(train_loader, val_loader)

def profile_model_performance(model: nn.Module,
                            input_size: Tuple[int, ...] = (1, 100, 4)) -> Dict[str, Any]:
    """Profile model performance."""
    profiler = ModelProfiler()
    return profiler.profile_model(model, input_size)