"""
Hardware Acceleration and Optimization for NAS

This module provides advanced hardware acceleration features including:
- GPU utilization and mixed precision training
- Distributed training capabilities
- Model quantization and compression
- Optimized data loading and preprocessing
- Memory-efficient training strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from contextlib import nullcontext
import psutil
import GPUtil
import time
import os
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class HardwareConfig:
    """Configuration for hardware acceleration."""
    use_mixed_precision: bool = True
    use_gpu: bool = True
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    distributed_training: bool = False
    world_size: int = 1
    rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "12355"
    use_model_parallelism: bool = False
    use_data_parallelism: bool = True
    gradient_checkpointing: bool = True
    memory_efficient_attention: bool = True
    use_quantization: bool = True
    quantization_bits: int = 8
    use_optimizations: bool = True

@dataclass
class OptimizationConfig:
    """Configuration for training optimizations."""
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    scheduler_type: str = "cosine"
    use_swa: bool = True  # Stochastic Weight Averaging
    swa_start: int = 0.8
    use_ema: bool = True   # Exponential Moving Average
    ema_decay: float = 0.999
    use_amp: bool = True   # Automatic Mixed Precision
    amp_opt_level: str = "O1"
    use_fp16: bool = True
    use_bf16: bool = False

class HardwareAccelerator:
    """
    Hardware acceleration manager for NAS training.

    Manages GPU utilization, mixed precision training, and distributed training.
    """

    def __init__(self, config: HardwareConfig):
        """Initialize hardware accelerator.

        Args:
            config: Hardware configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self._setup_gpu()
        self._setup_mixed_precision()
        self._setup_distributed_training()

        self.logger.info("🚀 Hardware Accelerator initialized")

    def _setup_gpu(self):
        """Setup GPU configuration."""
        if not self.config.use_gpu or not torch.cuda.is_available():
            self.config.use_gpu = False
            self.logger.warning("⚠️ GPU not available, using CPU")
            return

        # Set GPU devices
        if len(self.config.gpu_ids) > 0:
            torch.cuda.set_device(self.config.gpu_ids[0])
            self.logger.info(f"📊 Using GPU: {torch.cuda.get_device_name()}")

        # Enable memory efficient attention if available
        if self.config.memory_efficient_attention:
            try:
                from torch.nn.functional import scaled_dot_product_attention
                self.use_efficient_attention = True
            except ImportError:
                self.use_efficient_attention = False
                self.logger.warning("⚠️ Efficient attention not available")

    def _setup_mixed_precision(self):
        """Setup mixed precision training."""
        if not self.config.use_mixed_precision:
            self.scaler = None
            self.use_amp = False
            return

        # Initialize gradient scaler for mixed precision
        if self.config.use_gpu and torch.cuda.is_available():
            self.scaler = amp.GradScaler()
            self.use_amp = True
        else:
            self.scaler = None
            self.use_amp = False
            self.logger.warning("⚠️ Mixed precision requires GPU")

    def _setup_distributed_training(self):
        """Setup distributed training."""
        if not self.config.distributed_training:
            self.is_distributed = False
            return

        try:
            # Initialize distributed process group
            os.environ['MASTER_ADDR'] = self.config.master_addr
            os.environ['MASTER_PORT'] = self.config.master_port

            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                world_size=self.config.world_size,
                rank=self.config.rank
            )

            self.is_distributed = True
            self.logger.info(f"🔗 Distributed training initialized (rank {self.config.rank}/{self.config.world_size})")

        except Exception as e:
            self.logger.error(f"❌ Failed to initialize distributed training: {e}")
            self.is_distributed = False

class OptimizedTrainer:
    """
    Optimized trainer with hardware acceleration.

    Implements advanced training techniques including mixed precision,
    gradient accumulation, and memory-efficient training.
    """

    def __init__(self, model: nn.Module, config: OptimizationConfig, hardware_config: HardwareConfig):
        """Initialize optimized trainer.

        Args:
            model: Neural network model
            config: Optimization configuration
            hardware_config: Hardware configuration
        """
        self.model = model
        self.config = config
        self.hardware_config = hardware_config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Setup components
        self.hardware_accelerator = HardwareAccelerator(hardware_config)
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        self.ema_model = None

        # Initialize SWA and EMA if enabled
        if self.config.use_swa:
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)
            self.swa_scheduler = torch.optim.swa_utils.SWALR(
                self.optimizer, swa_lrs=0.05, anneal_epochs=10
            )
        else:
            self.swa_model = None

        if self.config.use_ema:
            from copy import deepcopy
            self.ema_model = deepcopy(self.model).eval()
            self.ema_decay = config.ema_decay

        self.logger.info("🏋️ Optimized Trainer initialized")

    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimized optimizer."""
        if self.hardware_config.use_gpu and torch.cuda.is_available():
            # Use fused optimizers if available
            try:
                from apex.optimizers import FusedAdam
                optimizer = FusedAdam(
                    self.model.parameters(),
                    lr=1e-3,
                    weight_decay=1e-4,
                    eps=1e-6
                )
                self.logger.info("🚀 Using FusedAdam optimizer")
            except ImportError:
                optimizer = optim.AdamW(
                    self.model.parameters(),
                    lr=1e-3,
                    weight_decay=1e-4,
                    eps=1e-6,
                    fused=True if torch.cuda.is_available() else False
                )
                self.logger.info("🚀 Using AdamW optimizer with fused kernels")
        else:
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=1e-3,
                weight_decay=1e-4,
                eps=1e-6
            )

        return optimizer

    def _setup_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Setup learning rate scheduler."""
        if self.config.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=self.config.warmup_steps // 10,
                T_mult=2
            )
        elif self.config.scheduler_type == "linear":
            scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=self.config.warmup_steps
            )
        elif self.config.scheduler_type == "exponential":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer, gamma=0.9
            )
        else:
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1000, gamma=0.1
            )

        return scheduler

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor],
                   step: int, total_steps: int) -> Dict[str, float]:
        """Perform optimized training step.

        Args:
            batch: Training batch
            step: Current step
            total_steps: Total training steps

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        x, y = batch

        # Move to GPU if available
        if self.hardware_config.use_gpu and torch.cuda.is_available():
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Mixed precision training context
        context_manager = amp.autocast() if self.hardware_accelerator.use_amp else nullcontext()

        with context_manager:
            # Forward pass
            outputs = self.model(x)
            loss = self._compute_loss(outputs, y)

            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps

        # Backward pass with gradient scaling
        if self.hardware_accelerator.scaler is not None:
            self.hardware_accelerator.scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (step + 1) % self.config.gradient_accumulation_steps == 0:
            # Clip gradients
            if self.hardware_accelerator.scaler is not None:
                self.hardware_accelerator.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                # Optimizer step with gradient scaling
                self.hardware_accelerator.scaler.step(self.optimizer)
                self.hardware_accelerator.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

            self.optimizer.zero_grad(set_to_none=True)

            # Update learning rate
            if self.scheduler:
                self.scheduler.step()

            # Update EMA
            if self.ema_model is not None:
                self._update_ema()

        # Update SWA
        if self.swa_model is not None and step > total_steps * self.config.swa_start:
            self.swa_model.update_parameters(self.model)
            if self.swa_scheduler:
                self.swa_scheduler.step()

        return {"loss": loss.item() * self.config.gradient_accumulation_steps}

    def _compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute optimized loss function."""
        # Use label smoothing for better generalization
        if isinstance(outputs, torch.Tensor) and outputs.dim() > 1:
            loss = F.cross_entropy(outputs, targets, label_smoothing=0.1)
        else:
            loss = F.mse_loss(outputs, targets)

        # Add regularization terms
        l2_loss = sum(p.pow(2).sum() for p in self.model.parameters())
        loss = loss + 1e-4 * l2_loss

        return loss

    def _update_ema(self):
        """Update exponential moving average model."""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(model_param.data, alpha=1 - self.ema_decay)

class QuantizedModel:
    """
    Model quantization for efficient inference.

    Supports both static and dynamic quantization.
    """

    def __init__(self, model: nn.Module, bits: int = 8, quantize_type: str = "static"):
        """Initialize quantized model.

        Args:
            model: Original model
            bits: Quantization bits
            quantize_type: "static" or "dynamic"
        """
        self.original_model = model
        self.bits = bits
        self.quantize_type = quantize_type
        self.quantized_model = None

        self._quantize_model()

    def _quantize_model(self):
        """Quantize the model."""
        if self.quantize_type == "static":
            self.quantized_model = torch.quantization.quantize_dynamic(
                self.original_model,
                {nn.Linear, nn.Conv1d, nn.LSTM},
                dtype=torch.qint8
            )
        elif self.quantize_type == "dynamic":
            self.quantized_model = torch.quantization.quantize_dynamic(
                self.original_model,
                {nn.Linear},
                dtype=torch.qint8
            )
        else:
            self.logger.warning(f"⚠️ Unknown quantization type: {self.quantize_type}")
            self.quantized_model = self.original_model

        self.logger.info(f"🔧 Model quantized to {self.bits} bits ({self.quantize_type})")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through quantized model."""
        return self.quantized_model(x)

    def save_quantized_model(self, path: str):
        """Save quantized model."""
        torch.save({
            'model_state_dict': self.quantized_model.state_dict(),
            'bits': self.bits,
            'quantize_type': self.quantize_type
        }, path)

        self.logger.info(f"💾 Quantized model saved to {path}")

class MemoryEfficientLoader:
    """
    Memory-efficient data loader with advanced optimizations.

    Features:
    - Memory-mapped datasets
    - Lazy loading
    - Background prefetching
    - Optimized data types
    """

    def __init__(self, data: np.ndarray, batch_size: int = 32, num_workers: int = 4):
        """Initialize memory-efficient loader.

        Args:
            data: Dataset array
            batch_size: Batch size
            num_workers: Number of worker processes
        """
        self.data = data
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Memory mapping
        self._create_memory_map()

        # Optimize data types
        self._optimize_data_types()

        self.logger.info("📊 Memory-efficient data loader initialized")

    def _create_memory_map(self):
        """Create memory-mapped dataset."""
        # Save data to temporary file
        temp_file = Path("/tmp/nas_data.npy")
        np.save(temp_file, self.data)

        # Memory map the file
        self.mmap_data = np.load(temp_file, mmap_mode='r')

    def _optimize_data_types(self):
        """Optimize data types for memory efficiency."""
        # Convert to most memory-efficient types
        if self.data.dtype == np.float64:
            self.data = self.data.astype(np.float32)
        elif self.data.dtype == np.int64:
            self.data = self.data.astype(np.int32)

    def get_dataloader(self, shuffle: bool = True) -> DataLoader:
        """Get optimized data loader."""
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(self.data)
        )

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

class SystemMonitor:
    """
    System monitoring for training optimization.

    Monitors GPU memory, CPU usage, and system resources
    to optimize training performance.
    """

    def __init__(self, log_interval: int = 100):
        """Initialize system monitor.

        Args:
            log_interval: Logging interval in steps
        """
        self.log_interval = log_interval
        self.logger = logging.getLogger(self.__class__.__name__)

        # Track metrics
        self.gpu_memory_history = []
        self.cpu_memory_history = []
        self.step_times = []

    def log_system_status(self, step: int):
        """Log system status."""
        if step % self.log_interval == 0:
            # GPU memory
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                self.gpu_memory_history.append(gpu_memory)
                self.logger.info(f"🖥️ GPU Memory: {gpu_memory:.2f} GB")

            # CPU memory
            cpu_memory = psutil.virtual_memory().used / 1024**3  # GB
            self.cpu_memory_history.append(cpu_memory)
            self.logger.info(f"💾 CPU Memory: {cpu_memory:.2f} GB")

            # System load
            cpu_percent = psutil.cpu_percent()
            self.logger.info(f"⚡ CPU Usage: {cpu_percent}%")

    def get_optimization_suggestions(self) -> Dict[str, Any]:
        """Get optimization suggestions based on monitoring."""
        suggestions = {}

        if torch.cuda.is_available():
            avg_gpu_memory = np.mean(self.gpu_memory_history) if self.gpu_memory_history else 0
            if avg_gpu_memory > 10:  # GB
                suggestions['reduce_batch_size'] = True
                suggestions['enable_gradient_checkpointing'] = True

        avg_cpu_memory = np.mean(self.cpu_memory_history) if self.cpu_memory_history else 0
        if avg_cpu_memory > 30:  # GB
            suggestions['increase_swap'] = True
            suggestions['reduce_workers'] = True

        return suggestions

# Utility functions
def setup_distributed_training(rank: int, world_size: int):
    """Setup distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup_distributed_training():
    """Cleanup distributed training."""
    dist.destroy_process_group()

def get_memory_usage():
    """Get current memory usage."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    else:
        return psutil.virtual_memory().used / 1024**3  # GB

def optimize_model_for_inference(model: nn.Module):
    """Optimize model for efficient inference."""
    model.eval()

    # Fuse batch norm and conv layers if possible
    if hasattr(model, 'fuse_model'):
        model.fuse_model()

    # Convert to half precision if possible
    if torch.cuda.is_available():
        model.half()

    return model