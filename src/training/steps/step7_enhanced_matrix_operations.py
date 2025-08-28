# src/training/steps/step7_enhanced_matrix_operations.py

"""Step 7: Enhanced Matrix Operations for Data Analysis.
This step performs advanced matrix operations for comprehensive data analysis after feature engineering.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.training.enhanced_matrix_operations import EnhancedMatrixOperations
from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.training_pipeline_decorators import (
    circuit_breaker_protection,
    debug_training_step,
    memory_efficient,
    prevent_data_leakage,
    quality_gate,
    resource_monitor,
    secure_data_processing,
    validate_step_output,
)


class Step7EnhancedMatrixOperations:
    """Step 7: Enhanced Matrix Operations for Data Analysis."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize Step 2.5 Enhanced Matrix Operations."""
        self.config = config
        self.logger = system_logger.getChild("Step7EnhancedMatrixOperations")
        
        # Initialize enhanced matrix operations
        self.matrix_ops = EnhancedMatrixOperations(config)
        
        # Step-specific configuration
        self.step_config = config.get("step7_enhanced_matrix_operations", {})
        self.output_dir = Path(self.step_config.get("output_dir", "data/matrix_operations"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @secure_data_processing(encryption_level="high", data_validation=True)
    @prevent_data_leakage(validate_inputs=True, sanitize_outputs=True)
    @resource_monitor(cpu_threshold_percent=90.0, memory_threshold_gb=16.0)
    @memory_efficient(chunk_size=5000, streaming_processing=True)
    @debug_training_step(log_intermediate_results=True, save_debug_artifacts=True)
    @circuit_breaker_protection(failure_threshold=3, recovery_timeout=300.0)
    @validate_step_output(
        required_files=["matrix_operations_config.json"],
        data_quality_checks={"min_operations": 1}
    )
    @quality_gate(
        model_performance_thresholds={},
        data_quality_metrics={"completeness": 0.95}
    )
    @handle_errors(exceptions=(ValueError, RuntimeError), default_return=False)
    async def execute(
        self,
        training_input: dict[str, Any],
        pipeline_state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Execute Step 7: Enhanced Matrix Operations.
        
        Args:
            training_input: Input data from previous steps
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with matrix operations results
        """
        try:
            start_time = datetime.now()
            self.logger.info("🚀 Starting Step 7: Enhanced Matrix Operations...")
            
            # Extract parameters
            symbol = training_input.get("symbol", "UNKNOWN")
            exchange = training_input.get("exchange", "UNKNOWN")
            timeframe = training_input.get("timeframe", "1m")
            
            # Load engineered features from step6
            features_train_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_train.parquet"
            features_val_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_val.parquet"
            
            if not os.path.exists(features_train_path):
                raise ValueError(f"Features train file not found: {features_train_path}")
            
            if not os.path.exists(features_val_path):
                raise ValueError(f"Features validation file not found: {features_val_path}")
            
            self.logger.info(f"📊 Loading engineered features from: {features_train_path}")
            
            # Load the engineered features (combine train and validation)
            df_train = pd.read_parquet(features_train_path)
            df_val = pd.read_parquet(features_val_path)
            df = pd.concat([df_train, df_val], ignore_index=True)
            
            self.logger.info(f"📈 Loaded {len(df)} rows of engineered features")
            self.logger.info(f"🔢 Features: {len(df.columns)} columns")

            
            # Prepare matrix operations configuration
            matrix_config = self._prepare_matrix_operations_config(df, symbol, exchange, timeframe)
            
            # Execute matrix operations
            matrix_results = await self._execute_matrix_operations(df, matrix_config)
            
            # Save results
            output_files = await self._save_matrix_operations_results(
                matrix_results, matrix_config, symbol, exchange, timeframe
            )
            
            # Update pipeline state
            pipeline_state["step7_enhanced_matrix_operations"] = {
                "status": "completed",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "output_files": output_files,
                "matrix_config": matrix_config,
                "matrix_results": matrix_results,
                "data_shape": df.shape,
                "symbol": symbol,
                "exchange": exchange,
                "timeframe": timeframe
            }
            
            self.logger.info("✅ Step 7: Enhanced Matrix Operations completed successfully")
            return pipeline_state
            
        except Exception as e:
            self.logger.error(f"❌ Step 7 failed: {str(e)}")
            pipeline_state["step7_enhanced_matrix_operations"] = {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            return pipeline_state

    def _prepare_matrix_operations_config(
        self, 
        df: pd.DataFrame, 
        symbol: str, 
        exchange: str, 
        timeframe: str
    ) -> dict[str, Any]:
        """Prepare configuration for matrix operations."""
        
        # Basic matrix operations configuration
        config = {
            "enable_gpu_acceleration": self.step_config.get("enable_gpu_acceleration", False),
            "enable_sparse_optimizations": self.step_config.get("enable_sparse_optimizations", True),
            "enable_memory_optimization": self.step_config.get("enable_memory_optimization", True),
            "enable_parallel_processing": self.step_config.get("enable_parallel_processing", True),
            
            # Quality thresholds
            "condition_number_threshold": self.step_config.get("condition_number_threshold", 1e12),
            "min_eigenvalue_threshold": self.step_config.get("min_eigenvalue_threshold", 1e-10),
            "correlation_threshold": self.step_config.get("correlation_threshold", 0.8),
            "memory_threshold_gb": self.step_config.get("memory_threshold_gb", 8.0),
            
            # Performance settings
            "batch_size": self.step_config.get("batch_size", 1000),
            "max_iterations": self.step_config.get("max_iterations", 1000),
            "tolerance": self.step_config.get("tolerance", 1e-6),
            
            # Data-specific settings
            "data_shape": df.shape,
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            
            # Operations to perform
            "operations": [
                "correlation_analysis",
                "condition_number_check",
                "eigenvalue_analysis",
                "singular_value_decomposition",
                "matrix_rank_analysis"
            ]
        }
        
        return config

    async def _execute_matrix_operations(
        self, 
        df: pd.DataFrame, 
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute matrix operations on the data."""
        
        results = {}
        
        # Get numeric columns for matrix operations
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            self.logger.warning("⚠️ No numeric columns found for matrix operations")
            return {"error": "No numeric columns available"}
        
        self.logger.info(f"🔢 Performing matrix operations on {len(numeric_df.columns)} numeric columns")
        
        # 1. Correlation Analysis
        if "correlation_analysis" in config["operations"]:
            self.logger.info("📊 Performing correlation analysis...")
            correlation_matrix = numeric_df.corr()
            results["correlation_analysis"] = {
                "correlation_matrix": correlation_matrix.to_dict(),
                "high_correlations": self._find_high_correlations(correlation_matrix, config["correlation_threshold"])
            }
        
        # 2. Condition Number Check
        if "condition_number_check" in config["operations"]:
            self.logger.info("🔍 Checking condition number...")
            condition_number = np.linalg.cond(numeric_df.values)
            results["condition_number_check"] = {
                "condition_number": float(condition_number),
                "is_well_conditioned": condition_number < config["condition_number_threshold"]
            }
        
        # 3. Eigenvalue Analysis
        if "eigenvalue_analysis" in config["operations"]:
            self.logger.info("📈 Performing eigenvalue analysis...")
            eigenvalues = np.linalg.eigvals(numeric_df.values)
            results["eigenvalue_analysis"] = {
                "eigenvalues": eigenvalues.tolist(),
                "min_eigenvalue": float(np.min(eigenvalues)),
                "max_eigenvalue": float(np.max(eigenvalues)),
                "eigenvalue_ratio": float(np.max(eigenvalues) / np.min(eigenvalues)),
                "small_eigenvalues": int(np.sum(np.abs(eigenvalues) < config["min_eigenvalue_threshold"]))
            }
        
        # 4. Singular Value Decomposition
        if "singular_value_decomposition" in config["operations"]:
            self.logger.info("🔧 Performing SVD analysis...")
            try:
                U, s, Vt = np.linalg.svd(numeric_df.values, full_matrices=False)
                results["singular_value_decomposition"] = {
                    "singular_values": s.tolist(),
                    "rank": int(np.sum(s > config["min_eigenvalue_threshold"])),
                    "condition_number_svd": float(s[0] / s[-1]) if len(s) > 1 else float('inf')
                }
            except Exception as e:
                self.logger.warning(f"⚠️ SVD failed: {str(e)}")
                results["singular_value_decomposition"] = {"error": str(e)}
        
        # 5. Matrix Rank Analysis
        if "matrix_rank_analysis" in config["operations"]:
            self.logger.info("📊 Analyzing matrix rank...")
            try:
                rank = np.linalg.matrix_rank(numeric_df.values)
                results["matrix_rank_analysis"] = {
                    "rank": int(rank),
                    "full_rank": rank == min(numeric_df.shape),
                    "rank_deficiency": min(numeric_df.shape) - rank
                }
            except Exception as e:
                self.logger.warning(f"⚠️ Rank analysis failed: {str(e)}")
                results["matrix_rank_analysis"] = {"error": str(e)}
        
        return results

    def _find_high_correlations(
        self, 
        correlation_matrix: pd.DataFrame, 
        threshold: float
    ) -> list[dict[str, Any]]:
        """Find high correlation pairs."""
        high_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_correlations.append({
                        "column1": correlation_matrix.columns[i],
                        "column2": correlation_matrix.columns[j],
                        "correlation": float(corr_value)
                    })
        
        return high_correlations

    async def _save_matrix_operations_results(
        self,
        results: dict[str, Any],
        config: dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str
    ) -> dict[str, str]:
        """Save matrix operations results to files."""
        
        output_files = {}
        
        # Save configuration
        config_file = self.output_dir / f"{exchange}_{symbol}_{timeframe}_matrix_operations_config.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        output_files["config"] = str(config_file)
        
        # Save results
        results_file = self.output_dir / f"{exchange}_{symbol}_{timeframe}_matrix_operations_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        output_files["results"] = str(results_file)
        
        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "operations_performed": list(results.keys()),
            "data_shape": config["data_shape"],
            "numeric_columns": len(config["numeric_columns"])
        }
        
        summary_file = self.output_dir / f"{exchange}_{symbol}_{timeframe}_matrix_operations_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        output_files["summary"] = str(summary_file)
        
        self.logger.info(f"💾 Saved matrix operations results to {self.output_dir}")
        return output_files


# Step execution function
async def run_step(
    symbol: str,
    exchange: str,
    timeframe: str = "1m",
    data_dir: str = "data_cache",
    force_rerun: bool = False,
    **kwargs: Any,
) -> bool:
    """
    Run Step 7: Enhanced Matrix Operations.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory
        force_rerun: Force rerun the step
        **kwargs: Additional arguments
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load configuration
        from src.config.training import get_training_config
        config = get_training_config()
        
        # Create step instance
        step = Step7EnhancedMatrixOperations(config)
        
        # Prepare training input
        training_input = {
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "data_dir": data_dir,
            "force_rerun": force_rerun,
            **kwargs
        }
        
        # Execute step
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)
        
        # Check if step was successful
        step_result = result.get("step7_enhanced_matrix_operations", {})
        return step_result.get("status") == "completed"
        
    except Exception as e:
        system_logger.error(f"❌ Step 7 failed: {str(e)}")
        return False


# Export the main class for external use
__all__ = ["Step7EnhancedMatrixOperations", "run_step"]