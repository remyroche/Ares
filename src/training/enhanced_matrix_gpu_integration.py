# src/training/enhanced_matrix_gpu_integration.py

from src.core.decorators import (
    cached,
    circuit_breaker,
    handles_errors,
    log_call,
    log_execution_time,
    validates,
)
from src.core.domain import prevent_data_leakage, quality_gate, secure_data_processing

"""
Enhanced Matrix Operations with M1 GPU Integration.
Combines advanced matrix operations with Mac M1 GPU acceleration.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.training.gpu_acceleration_m1 import M1GPUAcceleration
from src.training.steps.step7_enhanced_matrix_operations import (
    EnhancedMatrixOperations,
)
from src.utils.logger import system_logger


class EnhancedMatrixGPUIntegration:
    """
    Enhanced matrix operations with M1 GPU integration.

    Combines advanced matrix operations with Mac M1 GPU acceleration
    for maximum performance and efficiency.
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize enhanced matrix GPU integration."""
        self.config = config
        self.logger = system_logger.getChild("EnhancedMatrixGPUIntegration")

        # Initialize enhanced matrix operations
        self.matrix_ops = EnhancedMatrixOperations(config)

        # Initialize M1 GPU acceleration
        self.gpu_accel = M1GPUAcceleration(config)

        # Integration state
        self.integration_results: dict[str, Any] = {}
        self.performance_metrics: dict[str, Any] = {}

    @secure_data_processing(encryption_level="high", data_validation=True)
    @prevent_data_leakage(temporal_validation=True, lookahead_bias_prevention=True)
    @log_execution_time(cpu_threshold_percent=85.0, memory_threshold_gb=16.0)
    @cached(chunk_size=5000, streaming_processing=True)
    @log_call(log_intermediate_results=True, save_debug_artifacts=True)
    @circuit_breaker(failure_threshold=3, recovery_timeout=600.0)
    @validates(required_files=[], data_quality_checks={"min_rows": 100})
    @quality_gate(
        model_performance_thresholds={},
        data_quality_metrics={"completeness": 0.9},
    )
    @handles_errors(exceptions=(ValueError, RuntimeError), default_return=None)
    async def enhanced_gpu_matrix_operations(
        self,
        features_df: pd.DataFrame,
        target: pd.Series | None = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        Apply enhanced matrix operations with GPU acceleration.

        Args:
            features_df: Input features DataFrame
            target: Target variable (optional)

        Returns:
            Enhanced features DataFrame and comprehensive metadata
        """
        try:
            start_time = time.time()
            self.logger.info("🚀 Starting Enhanced GPU Matrix Operations...")

            enhanced_df = features_df.copy()
            all_metadata: dict[str, Any] = {}

            # 1. GPU-accelerated matrix operations
            if self.gpu_accel.mps_available:
                self.logger.info("📊 Applying GPU-accelerated matrix operations...")

                # Convert to numpy for GPU operations
                features_array = features_df.values

                # GPU SVD decomposition
                U, S, Vt, svd_metadata = self.gpu_accel.gpu_svd_decomposition(
                    features_array,
                    k=50,
                )
                svd_features = U[:, :20]  # Use top 20 components
                svd_feature_names = [
                    f"gpu_svd_component_{i+1}" for i in range(svd_features.shape[1])
                ]
                svd_df = pd.DataFrame(
                    svd_features,
                    columns=svd_feature_names,
                    index=features_df.index,
                )
                enhanced_df = pd.concat([enhanced_df, svd_df], axis=1)
                all_metadata["gpu_svd"] = svd_metadata

                # GPU eigenvalue decomposition
                eigenvalues, eigenvectors, eigen_metadata = (
                    self.gpu_accel.gpu_eigenvalue_decomposition(
                        np.corrcoef(features_array.T),
                    )
                )
                eigen_features = (
                    features_array @ eigenvectors[:, :15]
                )  # Top 15 components
                eigen_feature_names = [
                    f"gpu_eigen_component_{i+1}" for i in range(eigen_features.shape[1])
                ]
                eigen_df = pd.DataFrame(
                    eigen_features,
                    columns=eigen_feature_names,
                    index=features_df.index,
                )
                enhanced_df = pd.concat([enhanced_df, eigen_df], axis=1)
                all_metadata["gpu_eigenvalue"] = eigen_metadata

                # GPU neural network features (if target provided)
                if target is not None:
                    nn_predictions, nn_metadata = (
                        self.gpu_accel.gpu_neural_network_operations(
                            features_array,
                            target.values,
                            hidden_layers=[100, 50],
                        )
                    )
                    nn_df = pd.DataFrame(
                        nn_predictions,
                        columns=["gpu_nn_prediction"],
                        index=features_df.index,
                    )
                    enhanced_df = pd.concat([enhanced_df, nn_df], axis=1)
                    all_metadata["gpu_neural_network"] = nn_metadata

                self.logger.info(
                    f"✅ GPU operations completed: +{len(enhanced_df.columns) - len(features_df.columns)} features",
                )
            else:
                self.logger.info("⚠️ GPU not available - using CPU operations")

            # 2. Enhanced matrix operations (CPU-based)
            self.logger.info("🔧 Applying enhanced matrix operations...")

            # Advanced decompositions
            enhanced_df, decomp_metadata = (
                self.matrix_ops.advanced_decomposition_techniques(enhanced_df)
            )
            all_metadata["enhanced_decompositions"] = decomp_metadata

            # Advanced clustering
            enhanced_df, cluster_metadata = (
                self.matrix_ops.advanced_clustering_features(enhanced_df)
            )
            all_metadata["enhanced_clustering"] = cluster_metadata

            # Advanced feature engineering
            enhanced_df, feature_metadata = (
                self.matrix_ops.advanced_feature_engineering(enhanced_df)
            )
            all_metadata["enhanced_feature_engineering"] = feature_metadata

            # 3. Quality assurance
            quality_check = self.matrix_ops.quality_assurance_checks(enhanced_df)
            all_metadata["quality_assurance"] = quality_check

            # 4. Performance metrics
            total_time = time.time() - start_time
            all_metadata["total_processing_time"] = total_time
            all_metadata["feature_count_increase"] = len(enhanced_df.columns) - len(
                features_df.columns,
            )

            # GPU performance summary
            if self.gpu_accel.mps_available:
                gpu_summary = self.gpu_accel.get_performance_summary()
                all_metadata["gpu_performance_summary"] = gpu_summary

            self.logger.info(
                f"✅ Enhanced GPU Matrix Operations completed in {total_time:.2f}s",
            )
            self.logger.info(
                f"📊 Features: {len(features_df.columns)} -> {len(enhanced_df.columns)} (+{all_metadata['feature_count_increase']})",
            )

            return enhanced_df, all_metadata

        except Exception as e:
            self.logger.exception(f"❌ Enhanced GPU Matrix Operations failed: {e}")
            return features_df, {"error": str(e)}

    @secure_data_processing(encryption_level="high", data_validation=True)
    @cached(chunk_size=3000, streaming_processing=True)
    @log_call(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.95})
    @handles_errors(exceptions=(ValueError, RuntimeError), default_return=None)
    async def gpu_optimized_training_pipeline(
        self,
        training_data: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        GPU-optimized training pipeline with enhanced matrix operations.

        Args:
            training_data: Training data dictionary

        Returns:
            Enhanced training data and metadata
        """
        try:
            start_time = time.time()
            self.logger.info("🚀 Starting GPU-Optimized Training Pipeline...")

            enhanced_data = training_data.copy()
            pipeline_metadata: dict[str, Any] = {}

            # Extract features and target
            if "features" in training_data:
                features_df = training_data["features"]
                target = training_data.get("target")

                # Apply enhanced GPU matrix operations
                enhanced_features, enhancement_metadata = (
                    await self.enhanced_gpu_matrix_operations(
                        features_df,
                        target,
                    )
                )
                enhanced_data["features"] = enhanced_features
                pipeline_metadata["enhancement"] = enhancement_metadata

                # GPU-accelerated batch operations
                if self.gpu_accel.mps_available:
                    self.logger.info("📊 Applying GPU batch operations...")

                    # Create batch of matrices for processing
                    feature_matrices = [
                        features_df[col].values.reshape(-1, 1)
                        for col in features_df.columns[:10]
                    ]

                    # GPU batch operations
                    batch_results, batch_metadata = self.gpu_accel.gpu_batch_operations(
                        feature_matrices,
                        operation="multiply",
                    )
                    pipeline_metadata["gpu_batch_operations"] = batch_metadata

                    # Create batch features
                    batch_features = np.column_stack(batch_results)
                    batch_feature_names = [
                        f"gpu_batch_feature_{i+1}"
                        for i in range(batch_features.shape[1])
                    ]
                    batch_df = pd.DataFrame(
                        batch_features,
                        columns=batch_feature_names,
                        index=features_df.index,
                    )
                    enhanced_data["features"] = pd.concat(
                        [enhanced_features, batch_df],
                        axis=1,
                    )

                # Performance optimization
                if target is not None:
                    self.logger.info(
                        "⚡ Applying GPU-optimized performance enhancements...",
                    )

                    # GPU-accelerated optimization algorithms (CPU-backed in matrix_ops)
                    optimized_features, opt_metadata = (
                        self.matrix_ops.optimization_algorithms(
                            enhanced_data["features"],
                            target,
                        )
                    )
                    enhanced_data["features"] = optimized_features
                    pipeline_metadata["optimization"] = opt_metadata

            # Pipeline performance metrics
            total_time = time.time() - start_time
            pipeline_metadata["total_pipeline_time"] = total_time
            pipeline_metadata["gpu_available"] = self.gpu_accel.mps_available
            pipeline_metadata["device_used"] = str(self.gpu_accel.device)

            self.logger.info(
                f"✅ GPU-Optimized Training Pipeline completed in {total_time:.2f}s",
            )
            return enhanced_data, pipeline_metadata

        except Exception as e:
            self.logger.exception(f"❌ GPU-Optimized Training Pipeline failed: {e}")
            return training_data, {"error": str(e)}

    @secure_data_processing(encryption_level="medium", data_validation=True)
    @cached(chunk_size=2000, streaming_processing=True)
    @log_call(log_intermediate_results=True)
    @quality_gate(data_quality_metrics={"completeness": 0.9})
    @handles_errors(exceptions=(ValueError, RuntimeError), default_return=None)
    async def benchmark_gpu_vs_cpu(
        self,
        features_df: pd.DataFrame,
        target: pd.Series | None = None,
    ) -> dict[str, Any]:
        """
        Benchmark GPU vs CPU performance for matrix operations.

        Args:
            features_df: Input features DataFrame
            target: Target variable (optional)

        Returns:
            Benchmark results and performance comparison
        """
        try:
            self.logger.info("📊 Benchmarking GPU vs CPU Performance...")

            benchmark_results: dict[str, Any] = {
                "gpu_available": self.gpu_accel.mps_available,
                "device_info": str(self.gpu_accel.device),
                "benchmarks": {},
            }

            # Test matrix multiplication
            A = features_df.values[:1000, : min(100, features_df.shape[1])]
            B = A.T

            # CPU benchmark
            cpu_start = time.time()
            cpu_result = np.matmul(A, B)
            cpu_time = time.time() - cpu_start

            # GPU benchmark
            if self.gpu_accel.mps_available:
                gpu_result, gpu_metadata = self.gpu_accel.gpu_matrix_multiplication(
                    A,
                    B,
                )
                gpu_time = gpu_metadata["processing_time"]

                # Verify results are similar
                result_diff = float(np.abs(cpu_result - gpu_result).max())

                benchmark_results["benchmarks"]["matrix_multiplication"] = {
                    "cpu_time": cpu_time,
                    "gpu_time": gpu_time,
                    "speedup": cpu_time / gpu_time if gpu_time > 0 else 0,
                    "result_difference": result_diff,
                    "gpu_metadata": gpu_metadata,
                }

            # Test SVD decomposition
            matrix = features_df.values[:500, : min(50, features_df.shape[1])]

            # CPU SVD
            cpu_start = time.time()
            U_cpu, S_cpu, Vt_cpu = np.linalg.svd(matrix, full_matrices=False)
            cpu_svd_time = time.time() - cpu_start

            # GPU SVD
            if self.gpu_accel.mps_available:
                U_gpu, S_gpu, Vt_gpu, gpu_metadata = (
                    self.gpu_accel.gpu_svd_decomposition(matrix, k=20)
                )
                gpu_svd_time = gpu_metadata["processing_time"]

                # Verify results
                s_diff = float(
                    np.abs(S_cpu[: min(len(S_cpu), len(S_gpu))] - S_gpu).max()
                )

                benchmark_results["benchmarks"]["svd_decomposition"] = {
                    "cpu_time": cpu_svd_time,
                    "gpu_time": gpu_svd_time,
                    "speedup": cpu_svd_time / gpu_svd_time if gpu_svd_time > 0 else 0,
                    "singular_value_difference": s_diff,
                    "gpu_metadata": gpu_metadata,
                }

            # Test neural network operations
            if target is not None:
                sample_features = features_df.values[
                    :1000, : min(20, features_df.shape[1])
                ]
                sample_target = target.values[:1000]

                # CPU neural network (simple linear regression)
                cpu_start = time.time()
                cpu_model = LinearRegression()
                cpu_model.fit(sample_features, sample_target)
                cpu_predictions = cpu_model.predict(sample_features)
                cpu_nn_time = time.time() - cpu_start

                # GPU neural network
                if self.gpu_accel.mps_available:
                    gpu_predictions, gpu_metadata = (
                        self.gpu_accel.gpu_neural_network_operations(
                            sample_features,
                            sample_target,
                            hidden_layers=[50, 25],
                        )
                    )
                    gpu_nn_time = gpu_metadata["processing_time"]

                    # Compare predictions
                    pred_diff = float(np.abs(cpu_predictions - gpu_predictions).max())

                    benchmark_results["benchmarks"]["neural_network"] = {
                        "cpu_time": cpu_nn_time,
                        "gpu_time": gpu_nn_time,
                        "speedup": cpu_nn_time / gpu_nn_time if gpu_nn_time > 0 else 0,
                        "prediction_difference": pred_diff,
                        "gpu_metadata": gpu_metadata,
                    }

            # Overall performance summary
            if self.gpu_accel.mps_available:
                gpu_summary = self.gpu_accel.get_performance_summary()
                benchmark_results["gpu_performance_summary"] = gpu_summary

            self.logger.info("✅ GPU vs CPU Benchmark completed")
            return benchmark_results

        except Exception as e:
            self.logger.exception(f"❌ GPU vs CPU Benchmark failed: {e}")
            return {"error": str(e)}

    def get_integration_summary(self) -> dict[str, Any]:
        """Get summary of integration operations and results."""
        try:
            return {
                "gpu_available": self.gpu_accel.mps_available,
                "device_info": str(self.gpu_accel.device),
                "integration_results": self.integration_results,
                "performance_metrics": self.performance_metrics,
                "gpu_performance": self.gpu_accel.get_performance_summary(),
            }

        except Exception as e:
            self.logger.exception(f"❌ Integration summary generation failed: {e}")
            return {"error": str(e)}

    def clear_gpu_memory(self) -> None:
        """Clear GPU memory cache."""
        self.gpu_accel.clear_gpu_memory()


async def demonstrate_gpu_integration() -> None:
    """Demonstrate GPU integration with enhanced matrix operations."""

    print("🚀 Enhanced Matrix Operations with M1 GPU Integration")
    print("=" * 60)

    # Create sample data
    print("\n📊 Creating sample financial data...")
    np.random.seed(42)

    features_df = pd.DataFrame(
        {
            "price": np.random.normal(100, 10, 2000),
            "volume": np.random.lognormal(10, 1, 2000),
            "returns": np.random.normal(0, 0.02, 2000),
            "volatility": np.random.gamma(2, 0.01, 2000),
            "momentum": np.random.normal(0, 0.1, 2000),
            "rsi": np.random.uniform(0, 100, 2000),
            "macd": np.random.normal(0, 0.5, 2000),
            "bollinger_upper": np.random.normal(110, 5, 2000),
            "bollinger_lower": np.random.normal(90, 5, 2000),
            "atr": np.random.gamma(1, 0.5, 2000),
        },
    )

    # Add more features
    for i in range(40):
        features_df[f"feature_{i+1}"] = np.random.normal(0, 1, 2000)

    target = pd.Series(np.random.binomial(1, 0.5, 2000), name="target")

    print(
        f"✅ Created dataset: {features_df.shape[0]} samples x {features_df.shape[1]} features",
    )

    # Initialize integration
    config = {
        "m1_gpu": {
            "enable_mps": True,
            "enable_mixed_precision": True,
            "batch_size": 1000,
            "cpu_threshold": 5000,
        },
        "matrix_operations": {
            "enable_svd_enhancement": True,
            "enable_nmf_enhancement": True,
            "enable_sparse_operations": True,
        },
    }

    integration = EnhancedMatrixGPUIntegration(config)

    # Benchmark GPU vs CPU
    print("\n📊 Benchmarking GPU vs CPU Performance...")
    benchmark_results = await integration.benchmark_gpu_vs_cpu(features_df, target)

    print(f"GPU Available: {benchmark_results.get('gpu_available')}")
    print(f"Device: {benchmark_results.get('device_info')}")

    if "benchmarks" in benchmark_results:
        for operation, results in benchmark_results["benchmarks"].items():
            print(f"\n{operation.upper()}:")
            print(f"  CPU Time: {results['cpu_time']:.4f}s")
            print(f"  GPU Time: {results['gpu_time']:.4f}s")
            print(f"  Speedup: {results['speedup']:.2f}x")

    # Apply enhanced GPU matrix operations
    print("\n🔧 Applying Enhanced GPU Matrix Operations...")
    enhanced_features, enhancement_metadata = (
        await integration.enhanced_gpu_matrix_operations(
            features_df,
            target,
        )
    )

    print(
        f"✅ Enhanced features: {len(features_df.columns)} -> {len(enhanced_features.columns)}",
    )
    print(
        f"📈 Feature increase: +{enhancement_metadata.get('feature_count_increase', 0)} features",
    )
    print(
        f"⏱️ Processing time: {enhancement_metadata.get('total_processing_time', 0):.2f}s",
    )

    # GPU performance summary
    if "gpu_performance_summary" in enhancement_metadata:
        gpu_summary = enhancement_metadata["gpu_performance_summary"]
        print("\n🎯 GPU Performance Summary:")
        print(f"  Operations: {gpu_summary.get('gpu_operations_count', 0)}")
        print(f"  Total Time: {gpu_summary.get('gpu_processing_time', 0):.2f}s")
        print(f"  Average Time: {gpu_summary.get('average_gpu_time', 0):.4f}s")

    # Clear GPU memory
    integration.clear_gpu_memory()

    print("\n🎉 GPU Integration demonstration completed!")
    print("✅ Enhanced matrix operations with M1 GPU acceleration")
    print("🔒 All operations secured with decorators")
    print("📊 Performance benchmarks completed")


if __name__ == "__main__":
    # Run GPU integration demonstration
    asyncio.run(demonstrate_gpu_integration())
