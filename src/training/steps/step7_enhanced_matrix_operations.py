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
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics(df, matrix_results)
            
            # Save results
            output_files = await self._save_matrix_operations_results(
                matrix_results, matrix_config, quality_metrics, symbol, exchange, timeframe
            )
            
            # Update pipeline state
            pipeline_state["step7_enhanced_matrix_operations"] = {
                "status": "completed",
                "start_time": start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "output_files": output_files,
                "matrix_config": matrix_config,
                "matrix_results": matrix_results,
                "quality_metrics": quality_metrics,
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
        
        # Identify SR features for specialized analysis
        sr_features = [col for col in df.columns if any(keyword in col.lower() for keyword in [
            "sr_", "support", "resistance", "proximity", "sr_distance",
            "sr_proximity", "sr_outcome", "normalized_distance", "sr_proximity_score",
            "strength_score", "clarity_factor", "directional_pressure", "sr_score",
            "delta_sr_score", "isolation_score", "sr_level", "sr_multi_timeframe", "support_", "resistance_"
        ])]
        
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
            
            # SR-specific settings
            "sr_features": sr_features,
            "sr_feature_count": len(sr_features),
            "enable_sr_analysis": len(sr_features) > 0,
            "sr_correlation_threshold": self.step_config.get("sr_correlation_threshold", 0.7),
            "sr_condition_number_threshold": self.step_config.get("sr_condition_number_threshold", 1e10),
        }
        
        self.logger.info(f"🔧 Matrix operations configuration prepared:")
        self.logger.info(f"   - Total features: {len(df.columns)}")
        self.logger.info(f"   - SR features: {len(sr_features)}")
        self.logger.info(f"   - Numeric features: {len(config['numeric_columns'])}")
        
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
        
        # Standard matrix operations
        results.update(await self._execute_standard_matrix_operations(numeric_df, config))
        
        # SR-specific matrix operations
        if config.get("enable_sr_analysis", False) and config.get("sr_features"):
            self.logger.info("🎯 Performing SR-specific matrix operations...")
            results["sr_analysis"] = await self._execute_sr_matrix_operations(df, config)
        
        return results

    async def _execute_standard_matrix_operations(
        self, 
        numeric_df: pd.DataFrame, 
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute standard matrix operations."""
        results = {}
        
        # 1. Correlation Analysis
        self.logger.info("📊 Performing correlation analysis...")
        correlation_matrix = numeric_df.corr()
        results["correlation_analysis"] = {
            "correlation_matrix": correlation_matrix.to_dict(),
            "high_correlations": self._find_high_correlations(correlation_matrix, config["correlation_threshold"])
        }
        
        # 2. Condition Number Check
        self.logger.info("🔍 Checking condition number...")
        condition_number = np.linalg.cond(numeric_df.values)
        results["condition_number_check"] = {
            "condition_number": float(condition_number),
            "is_well_conditioned": condition_number < config["condition_number_threshold"]
        }
        
        # 3. Eigenvalue Analysis
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

    async def _execute_sr_matrix_operations(
        self, 
        df: pd.DataFrame, 
        config: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute SR-specific matrix operations."""
        try:
            sr_features = config.get("sr_features", [])
            if not sr_features:
                return {"error": "No SR features found"}
            
            # Get SR feature columns
            sr_df = df[sr_features].select_dtypes(include=[np.number])
            
            if len(sr_df.columns) == 0:
                return {"error": "No numeric SR features found"}
            
            self.logger.info(f"🎯 Analyzing {len(sr_df.columns)} SR features")
            
            results = {}
            
            # 1. SR Feature Correlation Analysis
            self.logger.info("📊 Performing SR feature correlation analysis...")
            sr_correlation_matrix = sr_df.corr()
            results["sr_correlation_analysis"] = {
                "correlation_matrix": sr_correlation_matrix.to_dict(),
                "high_correlations": self._find_high_correlations(sr_correlation_matrix, config["sr_correlation_threshold"]),
                "sr_feature_count": len(sr_df.columns)
            }
            
            # 2. SR Feature Condition Number
            self.logger.info("🔍 Checking SR feature condition number...")
            sr_condition_number = np.linalg.cond(sr_df.values)
            results["sr_condition_number"] = {
                "condition_number": float(sr_condition_number),
                "is_well_conditioned": sr_condition_number < config["sr_condition_number_threshold"]
            }
            
            # 3. SR Feature Eigenvalue Analysis
            self.logger.info("📈 Performing SR feature eigenvalue analysis...")
            sr_eigenvalues = np.linalg.eigvals(sr_df.values)
            results["sr_eigenvalue_analysis"] = {
                "eigenvalues": sr_eigenvalues.tolist(),
                "min_eigenvalue": float(np.min(sr_eigenvalues)),
                "max_eigenvalue": float(np.max(sr_eigenvalues)),
                "eigenvalue_ratio": float(np.max(sr_eigenvalues) / np.min(sr_eigenvalues)),
                "small_eigenvalues": int(np.sum(np.abs(sr_eigenvalues) < config["min_eigenvalue_threshold"]))
            }
            
            # 4. SR Feature Clustering Analysis
            self.logger.info("🔧 Performing SR feature clustering analysis...")
            results["sr_clustering_analysis"] = self._analyze_sr_feature_clusters(sr_df)
            
            # 5. SR Feature Stability Analysis
            self.logger.info("📊 Analyzing SR feature stability...")
            results["sr_stability_analysis"] = self._analyze_sr_feature_stability(sr_df)
            
            # 6. SR Feature Importance Analysis
            self.logger.info("🎯 Analyzing SR feature importance...")
            results["sr_importance_analysis"] = self._analyze_sr_feature_importance(sr_df)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in SR matrix operations: {e}")
            return {"error": str(e)}

    def _analyze_sr_feature_clusters(self, sr_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze SR feature clusters."""
        try:
            # Simple clustering analysis based on correlation
            correlation_matrix = sr_df.corr()
            
            # Find feature groups with high correlation
            high_corr_groups = []
            processed_features = set()
            
            for i, feature1 in enumerate(sr_df.columns):
                if feature1 in processed_features:
                    continue
                    
                group = [feature1]
                processed_features.add(feature1)
                
                for feature2 in sr_df.columns[i+1:]:
                    if feature2 not in processed_features:
                        corr = abs(correlation_matrix.loc[feature1, feature2])
                        if corr > 0.8:  # High correlation threshold
                            group.append(feature2)
                            processed_features.add(feature2)
                
                if len(group) > 1:
                    high_corr_groups.append(group)
            
            return {
                "high_correlation_groups": high_corr_groups,
                "group_count": len(high_corr_groups),
                "total_grouped_features": sum(len(group) for group in high_corr_groups)
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _analyze_sr_feature_stability(self, sr_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze SR feature stability over time."""
        try:
            # Calculate stability metrics for each SR feature
            stability_metrics = {}
            
            for column in sr_df.columns:
                values = sr_df[column].dropna()
                if len(values) > 1:
                    # Coefficient of variation (lower = more stable)
                    cv = values.std() / abs(values.mean()) if values.mean() != 0 else float('inf')
                    
                    # Range stability
                    range_stability = 1.0 / (1.0 + (values.max() - values.min()))
                    
                    stability_metrics[column] = {
                        "coefficient_of_variation": float(cv),
                        "range_stability": float(range_stability),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max())
                    }
            
            # Overall stability metrics
            overall_stability = {
                "mean_cv": np.mean([metrics["coefficient_of_variation"] for metrics in stability_metrics.values()]),
                "mean_range_stability": np.mean([metrics["range_stability"] for metrics in stability_metrics.values()]),
                "stable_features": len([cv for cv in [metrics["coefficient_of_variation"] for metrics in stability_metrics.values()] if cv < 0.5]),
                "unstable_features": len([cv for cv in [metrics["coefficient_of_variation"] for metrics in stability_metrics.values()] if cv > 1.0])
            }
            
            return {
                "feature_stability": stability_metrics,
                "overall_stability": overall_stability
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _analyze_sr_feature_importance(self, sr_df: pd.DataFrame) -> dict[str, Any]:
        """Analyze SR feature importance based on variance and correlation."""
        try:
            # Calculate variance-based importance
            variances = sr_df.var()
            variance_importance = variances.sort_values(ascending=False)
            
            # Calculate correlation-based importance (inverse of average correlation)
            correlation_matrix = sr_df.corr()
            avg_correlations = correlation_matrix.abs().mean()
            correlation_importance = (1.0 / (1.0 + avg_correlations)).sort_values(ascending=False)
            
            # Combined importance score
            combined_importance = (variance_importance + correlation_importance) / 2
            combined_importance = combined_importance.sort_values(ascending=False)
            
            return {
                "variance_importance": variance_importance.to_dict(),
                "correlation_importance": correlation_importance.to_dict(),
                "combined_importance": combined_importance.to_dict(),
                "top_features": combined_importance.head(10).index.tolist()
            }
            
        except Exception as e:
            return {"error": str(e)}

    def _calculate_quality_metrics(self, df: pd.DataFrame, matrix_results: dict[str, Any]) -> dict[str, Any]:
        """Calculate comprehensive quality metrics for the feature matrix."""
        try:
            self.logger.info("📊 Calculating quality metrics...")
            
            numeric_df = df.select_dtypes(include=[np.number])
            quality_metrics = {}
            
            # 1. Data Completeness Metrics
            quality_metrics["completeness"] = {
                "total_cells": numeric_df.size,
                "missing_cells": numeric_df.isnull().sum().sum(),
                "missing_ratio": float(numeric_df.isnull().sum().sum() / numeric_df.size),
                "complete_rows": int(numeric_df.dropna().shape[0]),
                "complete_columns": int(numeric_df.dropna(axis=1).shape[1])
            }
            
            # 2. Feature Variance Metrics
            variances = numeric_df.var()
            quality_metrics["variance"] = {
                "mean_variance": float(variances.mean()),
                "median_variance": float(variances.median()),
                "min_variance": float(variances.min()),
                "max_variance": float(variances.max()),
                "low_variance_features": int((variances < 1e-6).sum()),
                "zero_variance_features": int((variances == 0).sum())
            }
            
            # 3. Feature Correlation Metrics
            if "correlation_analysis" in matrix_results:
                corr_matrix = pd.DataFrame(matrix_results["correlation_analysis"]["correlation_matrix"])
                high_corrs = matrix_results["correlation_analysis"]["high_correlations"]
                
                quality_metrics["correlation"] = {
                    "mean_correlation": float(corr_matrix.abs().mean().mean()),
                    "max_correlation": float(corr_matrix.abs().max().max()),
                    "high_correlation_pairs": len(high_corrs),
                    "correlation_threshold": 0.8
                }
            
            # 4. Numerical Stability Metrics
            if "condition_number_check" in matrix_results:
                quality_metrics["numerical_stability"] = {
                    "condition_number": matrix_results["condition_number_check"]["condition_number"],
                    "is_well_conditioned": matrix_results["condition_number_check"]["is_well_conditioned"],
                    "condition_threshold": 1e12
                }
            
            # 5. Dimensionality Metrics
            if "matrix_rank_analysis" in matrix_results:
                quality_metrics["dimensionality"] = {
                    "matrix_rank": matrix_results["matrix_rank_analysis"]["rank"],
                    "full_rank": matrix_results["matrix_rank_analysis"]["full_rank"],
                    "rank_deficiency": matrix_results["matrix_rank_analysis"]["rank_deficiency"],
                    "effective_dimensions": matrix_results["matrix_rank_analysis"]["rank"]
                }
            
            # 6. Feature Distribution Metrics
            quality_metrics["distribution"] = {
                "skewness_mean": float(numeric_df.skew().mean()),
                "skewness_std": float(numeric_df.skew().std()),
                "kurtosis_mean": float(numeric_df.kurtosis().mean()),
                "kurtosis_std": float(numeric_df.kurtosis().std()),
                "high_skew_features": int((abs(numeric_df.skew()) > 3).sum()),
                "high_kurtosis_features": int((numeric_df.kurtosis() > 10).sum())
            }
            
            # 7. Outlier Metrics
            quality_metrics["outliers"] = self._calculate_outlier_metrics(numeric_df)
            
            # 8. Memory Usage Metrics
            quality_metrics["memory"] = {
                "memory_usage_mb": float(numeric_df.memory_usage(deep=True).sum() / 1024 / 1024),
                "memory_per_feature_kb": float(numeric_df.memory_usage(deep=True).sum() / len(numeric_df.columns) / 1024),
                "data_types": numeric_df.dtypes.value_counts().to_dict()
            }
            
            # 9. Overall Quality Score
            quality_metrics["overall_score"] = self._calculate_overall_quality_score(quality_metrics)
            
            self.logger.info(f"✅ Quality metrics calculated. Overall score: {quality_metrics['overall_score']:.2f}")
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating quality metrics: {str(e)}")
            return {"error": str(e)}

    def _calculate_outlier_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate outlier metrics for features."""
        outlier_metrics = {}
        
        try:
            # IQR-based outlier detection
            outlier_counts = []
            outlier_ratios = []
            
            for col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                outlier_counts.append(outliers)
                outlier_ratios.append(outliers / len(df))
            
            outlier_metrics = {
                "total_outliers": sum(outlier_counts),
                "mean_outliers_per_feature": float(np.mean(outlier_counts)),
                "max_outliers_in_feature": max(outlier_counts),
                "mean_outlier_ratio": float(np.mean(outlier_ratios)),
                "high_outlier_features": int(sum(1 for ratio in outlier_ratios if ratio > 0.1))
            }
            
        except Exception as e:
            outlier_metrics = {"error": str(e)}
        
        return outlier_metrics

    def _calculate_overall_quality_score(self, quality_metrics: dict[str, Any]) -> float:
        """Calculate overall quality score from individual metrics."""
        try:
            score = 0.0
            max_score = 0.0
            
            # Completeness score (0-25 points)
            completeness = quality_metrics.get("completeness", {})
            if "missing_ratio" in completeness:
                completeness_score = max(0, 25 * (1 - completeness["missing_ratio"]))
                score += completeness_score
                max_score += 25
            
            # Variance score (0-20 points)
            variance = quality_metrics.get("variance", {})
            if "zero_variance_features" in variance:
                zero_var_ratio = variance["zero_variance_features"] / len(quality_metrics.get("completeness", {}).get("total_cells", 1))
                variance_score = max(0, 20 * (1 - zero_var_ratio))
                score += variance_score
                max_score += 20
            
            # Correlation score (0-20 points)
            correlation = quality_metrics.get("correlation", {})
            if "high_correlation_pairs" in correlation:
                corr_score = max(0, 20 * (1 - correlation["high_correlation_pairs"] / 100))  # Penalize high correlations
                score += corr_score
                max_score += 20
            
            # Numerical stability score (0-15 points)
            stability = quality_metrics.get("numerical_stability", {})
            if "is_well_conditioned" in stability:
                stability_score = 15 if stability["is_well_conditioned"] else 5
                score += stability_score
                max_score += 15
            
            # Dimensionality score (0-10 points)
            dimensionality = quality_metrics.get("dimensionality", {})
            if "rank_deficiency" in dimensionality:
                rank_score = max(0, 10 * (1 - dimensionality["rank_deficiency"] / 100))
                score += rank_score
                max_score += 10
            
            # Distribution score (0-10 points)
            distribution = quality_metrics.get("distribution", {})
            if "high_skew_features" in distribution:
                skew_penalty = min(10, distribution["high_skew_features"] / 10)
                distribution_score = max(0, 10 - skew_penalty)
                score += distribution_score
                max_score += 10
            
            return score / max_score if max_score > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating overall quality score: {str(e)}")
            return 0.0

    def _generate_detailed_quality_report(self, quality_metrics: dict[str, Any]) -> str:
        """Generate detailed quality report with recommendations."""
        try:
            report = []
            report.append("=" * 80)
            report.append("📊 DETAILED FEATURE MATRIX QUALITY REPORT")
            report.append("=" * 80)
            
            # Overall Score
            overall_score = quality_metrics.get("overall_score", 0.0)
            report.append(f"🎯 OVERALL QUALITY SCORE: {overall_score:.2f}/1.00")
            
            # Score interpretation
            if overall_score >= 0.9:
                report.append("✅ EXCELLENT - Feature matrix is of very high quality")
            elif overall_score >= 0.8:
                report.append("🟢 GOOD - Feature matrix is of good quality with minor issues")
            elif overall_score >= 0.7:
                report.append("🟡 ACCEPTABLE - Feature matrix has some quality issues")
            elif overall_score >= 0.6:
                report.append("🟠 POOR - Feature matrix has significant quality issues")
            else:
                report.append("🔴 CRITICAL - Feature matrix has severe quality issues")
            
            report.append("")
            
            # 1. Completeness Analysis
            completeness = quality_metrics.get("completeness", {})
            report.append("📋 1. DATA COMPLETENESS ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Total cells: {completeness.get('total_cells', 0):,}")
            report.append(f"   Missing cells: {completeness.get('missing_cells', 0):,}")
            report.append(f"   Missing ratio: {completeness.get('missing_ratio', 0):.2%}")
            report.append(f"   Complete rows: {completeness.get('complete_rows', 0):,}")
            report.append(f"   Complete columns: {completeness.get('complete_columns', 0):,}")
            
            if completeness.get('missing_ratio', 0) > 0.05:
                report.append("   ⚠️  RECOMMENDATION: High missing data ratio - consider imputation")
            else:
                report.append("   ✅ Data completeness is acceptable")
            report.append("")
            
            # 2. Variance Analysis
            variance = quality_metrics.get("variance", {})
            report.append("📊 2. FEATURE VARIANCE ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Mean variance: {variance.get('mean_variance', 0):.6f}")
            report.append(f"   Median variance: {variance.get('median_variance', 0):.6f}")
            report.append(f"   Min variance: {variance.get('min_variance', 0):.6f}")
            report.append(f"   Max variance: {variance.get('max_variance', 0):.6f}")
            report.append(f"   Low variance features: {variance.get('low_variance_features', 0)}")
            report.append(f"   Zero variance features: {variance.get('zero_variance_features', 0)}")
            
            if variance.get('zero_variance_features', 0) > 0:
                report.append("   ⚠️  RECOMMENDATION: Remove zero-variance features")
            else:
                report.append("   ✅ Feature variance is acceptable")
            report.append("")
            
            # 3. Correlation Analysis
            correlation = quality_metrics.get("correlation", {})
            report.append("🔗 3. FEATURE CORRELATION ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Mean correlation: {correlation.get('mean_correlation', 0):.4f}")
            report.append(f"   Max correlation: {correlation.get('max_correlation', 0):.4f}")
            report.append(f"   High correlation pairs: {correlation.get('high_correlation_pairs', 0)}")
            report.append(f"   Correlation threshold: {correlation.get('correlation_threshold', 0.8)}")
            
            if correlation.get('high_correlation_pairs', 0) > 10:
                report.append("   ⚠️  RECOMMENDATION: Many highly correlated features - consider feature selection")
            elif correlation.get('high_correlation_pairs', 0) > 0:
                report.append("   ⚠️  RECOMMENDATION: Some highly correlated features - review for redundancy")
            else:
                report.append("   ✅ Feature correlations are acceptable")
            report.append("")
            
            # 4. Numerical Stability Analysis
            stability = quality_metrics.get("numerical_stability", {})
            report.append("🔢 4. NUMERICAL STABILITY ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Condition number: {stability.get('condition_number', 0):.2e}")
            report.append(f"   Well-conditioned: {stability.get('is_well_conditioned', False)}")
            report.append(f"   Condition threshold: {stability.get('condition_threshold', 1e12):.2e}")
            
            if not stability.get('is_well_conditioned', False):
                report.append("   ⚠️  RECOMMENDATION: Matrix is ill-conditioned - consider regularization or feature scaling")
            else:
                report.append("   ✅ Numerical stability is good")
            report.append("")
            
            # 5. Dimensionality Analysis
            dimensionality = quality_metrics.get("dimensionality", {})
            report.append("📐 5. DIMENSIONALITY ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Matrix rank: {dimensionality.get('matrix_rank', 0)}")
            report.append(f"   Full rank: {dimensionality.get('full_rank', False)}")
            report.append(f"   Rank deficiency: {dimensionality.get('rank_deficiency', 0)}")
            report.append(f"   Effective dimensions: {dimensionality.get('effective_dimensions', 0)}")
            
            if dimensionality.get('rank_deficiency', 0) > 0:
                report.append("   ⚠️  RECOMMENDATION: Rank-deficient matrix - consider dimensionality reduction")
            else:
                report.append("   ✅ Matrix has full rank")
            report.append("")
            
            # 6. Distribution Analysis
            distribution = quality_metrics.get("distribution", {})
            report.append("📈 6. FEATURE DISTRIBUTION ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Mean skewness: {distribution.get('skewness_mean', 0):.4f}")
            report.append(f"   Skewness std: {distribution.get('skewness_std', 0):.4f}")
            report.append(f"   Mean kurtosis: {distribution.get('kurtosis_mean', 0):.4f}")
            report.append(f"   Kurtosis std: {distribution.get('kurtosis_std', 0):.4f}")
            report.append(f"   High skew features: {distribution.get('high_skew_features', 0)}")
            report.append(f"   High kurtosis features: {distribution.get('high_kurtosis_features', 0)}")
            
            if distribution.get('high_skew_features', 0) > 10:
                report.append("   ⚠️  RECOMMENDATION: Many skewed features - consider transformations")
            else:
                report.append("   ✅ Feature distributions are generally acceptable")
            report.append("")
            
            # 7. Outlier Analysis
            outliers = quality_metrics.get("outliers", {})
            report.append("🎯 7. OUTLIER ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Total outliers: {outliers.get('total_outliers', 0):,}")
            report.append(f"   Mean outliers per feature: {outliers.get('mean_outliers_per_feature', 0):.1f}")
            report.append(f"   Max outliers in feature: {outliers.get('max_outliers_in_feature', 0)}")
            report.append(f"   Mean outlier ratio: {outliers.get('mean_outlier_ratio', 0):.2%}")
            report.append(f"   High outlier features: {outliers.get('high_outlier_features', 0)}")
            
            if outliers.get('high_outlier_features', 0) > 5:
                report.append("   ⚠️  RECOMMENDATION: Many features with high outlier ratios - consider outlier handling")
            else:
                report.append("   ✅ Outlier levels are acceptable")
            report.append("")
            
            # 8. Memory Usage Analysis
            memory = quality_metrics.get("memory", {})
            report.append("💾 8. MEMORY USAGE ANALYSIS")
            report.append("-" * 40)
            report.append(f"   Total memory usage: {memory.get('memory_usage_mb', 0):.1f} MB")
            report.append(f"   Memory per feature: {memory.get('memory_per_feature_kb', 0):.1f} KB")
            report.append(f"   Data types: {memory.get('data_types', {})}")
            
            if memory.get('memory_usage_mb', 0) > 1000:
                report.append("   ⚠️  RECOMMENDATION: High memory usage - consider data type optimization")
            else:
                report.append("   ✅ Memory usage is reasonable")
            report.append("")
            
            # 9. Actionable Recommendations
            report.append("🚀 9. ACTIONABLE RECOMMENDATIONS")
            report.append("-" * 40)
            
            recommendations = []
            
            if completeness.get('missing_ratio', 0) > 0.05:
                recommendations.append("• Implement data imputation for missing values")
            
            if variance.get('zero_variance_features', 0) > 0:
                recommendations.append("• Remove zero-variance features")
            
            if correlation.get('high_correlation_pairs', 0) > 5:
                recommendations.append("• Apply feature selection to reduce multicollinearity")
            
            if not stability.get('is_well_conditioned', False):
                recommendations.append("• Apply feature scaling or regularization")
            
            if dimensionality.get('rank_deficiency', 0) > 0:
                recommendations.append("• Consider PCA or other dimensionality reduction techniques")
            
            if distribution.get('high_skew_features', 0) > 10:
                recommendations.append("• Apply log or power transformations to skewed features")
            
            if outliers.get('high_outlier_features', 0) > 5:
                recommendations.append("• Implement outlier detection and handling strategies")
            
            if memory.get('memory_usage_mb', 0) > 1000:
                recommendations.append("• Optimize data types to reduce memory usage")
            
            if not recommendations:
                recommendations.append("• No immediate actions required - feature matrix is in good condition")
            
            for rec in recommendations:
                report.append(f"   {rec}")
            
            report.append("")
            
            # 10. Summary
            report.append("📋 10. SUMMARY")
            report.append("-" * 40)
            report.append(f"   Overall Quality Score: {overall_score:.2f}/1.00")
            
            if overall_score >= 0.8:
                report.append("   Status: ✅ READY FOR MODEL TRAINING")
            elif overall_score >= 0.6:
                report.append("   Status: ⚠️  NEEDS IMPROVEMENT BEFORE TRAINING")
            else:
                report.append("   Status: 🔴 REQUIRES SIGNIFICANT IMPROVEMENT")
            
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating detailed quality report: {str(e)}")
            return f"Error generating report: {str(e)}"

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
        quality_metrics: dict[str, Any],
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
        
        # Save quality metrics
        quality_file = self.output_dir / f"{exchange}_{symbol}_{timeframe}_quality_metrics.json"
        with open(quality_file, 'w') as f:
            json.dump(quality_metrics, f, indent=2, default=str)
        output_files["quality_metrics"] = str(quality_file)
        
        # Generate and save detailed quality report
        detailed_report = self._generate_detailed_quality_report(quality_metrics)
        report_file = self.output_dir / f"{exchange}_{symbol}_{timeframe}_quality_report.txt"
        with open(report_file, 'w') as f:
            f.write(detailed_report)
        output_files["quality_report"] = str(report_file)
        
        # Log the detailed report
        self.logger.info("\n" + detailed_report)
        
        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "operations_performed": list(results.keys()),
            "data_shape": config["data_shape"],
            "numeric_columns": len(config["numeric_columns"]),
            "overall_quality_score": quality_metrics.get("overall_score", 0.0),
            "quality_summary": {
                "completeness_ratio": quality_metrics.get("completeness", {}).get("missing_ratio", 1.0),
                "zero_variance_features": quality_metrics.get("variance", {}).get("zero_variance_features", 0),
                "high_correlations": quality_metrics.get("correlation", {}).get("high_correlation_pairs", 0),
                "is_well_conditioned": quality_metrics.get("numerical_stability", {}).get("is_well_conditioned", False)
            }
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