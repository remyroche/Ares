#!/usr/bin/env python3
"""
HMM Hardware Integration Module

This module contains hardware optimization functionality for HMM models,
extracted from the monolithic hmm_composite_manager.py file. It handles
GPU acceleration, memory optimization, and CPU optimization.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..logger import system_logger

# Optional hardware optimization imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Import M1 optimization utilities
try:
    from ..hardware.m1_gpu_utils import M1GPUManager
    from ..hardware.m1_memory_optimizer import M1MemoryOptimizer
    from ..hardware.m1_cpu_optimizer import M1CPUOptimizer
    M1_UTILITIES_AVAILABLE = True
except ImportError:
    M1_UTILITIES_AVAILABLE = False

# Import matrix operations
try:
    from ..matrix_operations import get_unified_matrix_operations
    MATRIX_OPS_AVAILABLE = True
except ImportError:
    MATRIX_OPS_AVAILABLE = False

class HMMHardwareManager:
    """Manager for hardware-optimized HMM operations."""

    def __init__(self):
        """Initialize the hardware manager."""
        self.logger = system_logger.getChild('HMMHardwareManager')

        # Initialize hardware components
        self.gpu_manager = None
        self.memory_optimizer = None
        self.cpu_optimizer = None
        self.matrix_ops = None

        self._initialize_hardware_components()
        self._log_hardware_capabilities()

    def _initialize_hardware_components(self) -> None:
        """Initialize available hardware optimization components."""
        try:
            if M1_UTILITIES_AVAILABLE:
                self.gpu_manager = M1GPUManager()
                self.memory_optimizer = M1MemoryOptimizer()
                self.cpu_optimizer = M1CPUOptimizer()
                self.logger.info("M1 hardware utilities initialized")
            else:
                self.logger.warning("M1 utilities not available")

            if MATRIX_OPS_AVAILABLE:
                self.matrix_ops = get_unified_matrix_operations()
                self.logger.info("Matrix operations initialized")
            else:
                self.logger.warning("Matrix operations not available")

        except Exception as e:
            self.logger.error(f"Error initializing hardware components: {e}")

    def _log_hardware_capabilities(self) -> None:
        """Log available hardware capabilities."""
        capabilities = {
            'torch_available': TORCH_AVAILABLE,
            'm1_utilities': M1_UTILITIES_AVAILABLE,
            'matrix_ops': MATRIX_OPS_AVAILABLE,
            'gpu_manager': self.gpu_manager is not None,
            'memory_optimizer': self.memory_optimizer is not None,
            'cpu_optimizer': self.cpu_optimizer is not None
        }

        self.logger.info(f"Hardware capabilities: {capabilities}")

        if TORCH_AVAILABLE:
            self.logger.info(f"PyTorch version: {torch.__version__}")
            if torch.cuda.is_available():
                self.logger.info(f"CUDA available: {torch.cuda.device_count()} devices")
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.logger.info("MPS (Apple Silicon GPU) available")

    def optimize_data_for_training(self, data: np.ndarray) -> np.ndarray:
        """
        Optimize data array for HMM training.

        Args:
            data: Input data array

        Returns:
            Optimized data array
        """
        try:
            # Memory optimization
            if self.memory_optimizer:
                data = self.memory_optimizer.optimize_array(data)
                self.logger.debug("Applied memory optimization")

            # CPU optimization
            if self.cpu_optimizer:
                data = self.cpu_optimizer.optimize_array(data)
                self.logger.debug("Applied CPU optimization")

            # Ensure data is in correct format
            data = np.asarray(data, dtype=np.float32)

            return data

        except Exception as e:
            self.logger.error(f"Error optimizing data: {e}")
            return np.asarray(data, dtype=np.float32)

    def gpu_accelerated_hmm_training(
        self,
        data: np.ndarray,
        n_components: int,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Perform GPU-accelerated HMM training.

        Args:
            data: Training data
            n_components: Number of HMM components
            **kwargs: Additional parameters

        Returns:
            Training results
        """
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch not available for GPU acceleration")
            return self._cpu_hmm_training(data, n_components, **kwargs)

        try:
            self.logger.info("Starting GPU-accelerated HMM training")

            # Convert to PyTorch tensor
            device = self._get_best_device()
            data_tensor = torch.tensor(data, dtype=torch.float32, device=device)

            # Initialize parameters
            params = self._initialize_hmm_parameters_gpu(data_tensor, n_components)

            # Run EM algorithm
            trained_params = self._gpu_em_algorithm(data_tensor, params, **kwargs)

            # Convert back to CPU
            results = self._convert_hmm_results_to_cpu(trained_params, data.shape)

            self.logger.info("GPU-accelerated HMM training completed")
            return results

        except Exception as e:
            self.logger.error(f"GPU training failed: {e}, falling back to CPU")
            return self._cpu_hmm_training(data, n_components, **kwargs)

    def _get_best_device(self) -> torch.device:
        """Get the best available device for computation."""
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')

    def _initialize_hmm_parameters_gpu(
        self,
        data: torch.Tensor,
        n_components: int
    ) -> Dict[str, torch.Tensor]:
        """Initialize HMM parameters on GPU."""
        n_samples, n_features = data.shape
        device = data.device

        params = {
            'startprob': torch.ones(n_components, device=device) / n_components,
            'transmat': torch.ones(n_components, n_components, device=device) / n_components,
            'means': torch.randn(n_components, n_features, device=device),
            'covars': torch.eye(n_features, device=device).unsqueeze(0).repeat(n_components, 1, 1)
        }

        # Initialize means using k-means-like approach
        indices = torch.randperm(n_samples, device=device)[:n_components]
        params['means'] = data[indices]

        return params

    def _gpu_em_algorithm(
        self,
        data: torch.Tensor,
        params: Dict[str, torch.Tensor],
        max_iter: int = 100,
        tol: float = 1e-4,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Run EM algorithm on GPU."""
        prev_loglik = -torch.inf

        for iteration in range(max_iter):
            # E-step: Forward-backward algorithm
            log_probs, posteriors = self._gpu_forward_backward(data, params)

            # M-step: Update parameters
            params = self._gpu_update_parameters(data, posteriors, params)

            # Check convergence
            current_loglik = torch.sum(log_probs)

            if torch.abs(current_loglik - prev_loglik) < tol:
                self.logger.info(f"Converged after {iteration + 1} iterations")
                break

            prev_loglik = current_loglik

        return params

    def _gpu_forward_backward(
        self,
        data: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward-backward algorithm on GPU."""
        n_samples, n_features = data.shape
        n_components = params['means'].shape[0]
        device = data.device

        # Compute emission probabilities
        emission_probs = self._compute_emission_probs_gpu(data, params)

        # Forward pass
        alpha = torch.zeros(n_samples, n_components, device=device)
        alpha[0] = params['startprob'] * emission_probs[0]

        for t in range(1, n_samples):
            alpha[t] = torch.sum(alpha[t-1].unsqueeze(1) * params['transmat'], dim=0) * emission_probs[t]

        # Backward pass
        beta = torch.zeros(n_samples, n_components, device=device)
        beta[-1] = 1.0

        for t in range(n_samples - 2, -1, -1):
            beta[t] = torch.sum(params['transmat'] * emission_probs[t+1].unsqueeze(0) * beta[t+1].unsqueeze(0), dim=1)

        # Compute posteriors
        posteriors = alpha * beta
        posteriors = posteriors / torch.sum(posteriors, dim=1, keepdim=True)

        log_probs = torch.log(torch.sum(alpha, dim=1))

        return log_probs, posteriors

    def _compute_emission_probs_gpu(
        self,
        data: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute emission probabilities on GPU."""
        n_samples, n_features = data.shape
        n_components = params['means'].shape[0]
        device = data.device

        emission_probs = torch.zeros(n_samples, n_components, device=device)

        for k in range(n_components):
            diff = data - params['means'][k]

            # Compute multivariate normal probability
            try:
                inv_cov = torch.inverse(params['covars'][k])
                det_cov = torch.det(params['covars'][k])

                mahal_dist = torch.sum(diff @ inv_cov * diff, dim=1)

                emission_probs[:, k] = torch.exp(-0.5 * mahal_dist) / torch.sqrt(
                    (2 * torch.pi) ** n_features * det_cov
                )
            except:
                # Fallback to diagonal covariance
                var = torch.diag(params['covars'][k])
                emission_probs[:, k] = torch.exp(-0.5 * torch.sum(diff**2 / var, dim=1)) / torch.sqrt(
                    (2 * torch.pi) ** n_features * torch.prod(var)
                )

        return emission_probs

    def _gpu_update_parameters(
        self,
        data: torch.Tensor,
        posteriors: torch.Tensor,
        params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Update HMM parameters on GPU."""
        n_samples, n_features = data.shape
        n_components = posteriors.shape[1]

        # Update start probabilities
        params['startprob'] = posteriors[0] / torch.sum(posteriors[0])

        # Update transition matrix
        xi = torch.zeros(n_components, n_components, device=data.device)
        for t in range(n_samples - 1):
            xi += posteriors[t].unsqueeze(1) * posteriors[t+1].unsqueeze(0)

        params['transmat'] = xi / torch.sum(xi, dim=1, keepdim=True)

        # Update means
        gamma_sum = torch.sum(posteriors, dim=0)
        params['means'] = torch.sum(posteriors.unsqueeze(2) * data.unsqueeze(1), dim=0) / gamma_sum.unsqueeze(1)

        # Update covariances
        for k in range(n_components):
            diff = data - params['means'][k]
            weighted_diff = posteriors[:, k].unsqueeze(1) * diff
            params['covars'][k] = torch.mm(weighted_diff.t(), diff) / gamma_sum[k]

            # Add regularization
            params['covars'][k] += torch.eye(n_features, device=data.device) * 1e-6

        return params

    def _convert_hmm_results_to_cpu(
        self,
        params: Dict[str, torch.Tensor],
        data_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Convert GPU results back to CPU format."""
        cpu_params = {}

        for key, value in params.items():
            if isinstance(value, torch.Tensor):
                cpu_params[key] = value.detach().cpu().numpy()
            else:
                cpu_params[key] = value

        return cpu_params

    def _cpu_hmm_training(
        self,
        data: np.ndarray,
        n_components: int,
        **kwargs
    ) -> Dict[str, Any]:
        """Fallback CPU-based HMM training."""
        try:
            from hmmlearn import hmm

            model = hmm.GaussianHMM(
                n_components=n_components,
                covariance_type=kwargs.get('covariance_type', 'full'),
                n_iter=kwargs.get('n_iter', 100),
                tol=kwargs.get('tol', 1e-4),
                random_state=kwargs.get('random_state', 42)
            )

            model.fit(data)

            return {
                'startprob': model.startprob_,
                'transmat': model.transmat_,
                'means': model.means_,
                'covars': model.covars_,
                'score': model.score(data),
                'n_iter': model.n_iter_,
                'converged': model.monitor_.converged
            }

        except ImportError:
            self.logger.error("hmmlearn not available for CPU training")
            return {}
        except Exception as e:
            self.logger.error(f"CPU HMM training failed: {e}")
            return {}

    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        memory_info = {}

        if self.memory_optimizer:
            memory_info.update(self.memory_optimizer.get_memory_usage())

        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                memory_info['cuda_memory_allocated'] = torch.cuda.memory_allocated()
                memory_info['cuda_memory_cached'] = torch.cuda.memory_reserved()

            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                memory_info['mps_available'] = True

        return memory_info

    def cleanup_gpu_memory(self) -> None:
        """Clean up GPU memory."""
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("CUDA memory cache cleared")

            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS cleanup if available in future PyTorch versions
                pass
