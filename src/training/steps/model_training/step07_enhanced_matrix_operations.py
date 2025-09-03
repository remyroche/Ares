"""Step 7: Enhanced Matrix Operations - Refactored to use BaseStep.

This module performs advanced matrix operations for comprehensive data analysis
after feature engineering, with GPU/MPS acceleration support.
"""

from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.training.base_step import BaseStep
from src.utils.logger import system_logger
from src.core.decorators import handles_errors
from src.training.steps.model_training.matrix_components import (
    MatrixProcessor, DiverseLookbackIntegrator, MatrixOptimizer
)


class EnhancedMatrixOperationsStep(BaseStep):
    """Step 7: Enhanced Matrix Operations using standardized base class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced matrix operations step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "07", "enhanced_matrix_operations")
        
        # Step-specific configuration
        self.matrix_config = config.get("matrix_operations_config", {
            "use_gpu": True,
            "use_diverse_lookback": True,
            "optimization_level": "high",
            "batch_size": 1000,
            "feature_selection": {
                "method": "mutual_info",
                "top_k": 50,
                "min_importance": 0.01
            },
            "matrix_computations": {
                "correlation_matrix": True,
                "covariance_matrix": True,
                "feature_interaction_matrix": True,
                "regime_transition_matrix": True
            }
        })
        
        # Components
        self.matrix_processor = None
        self.lookback_integrator = None
        self.matrix_optimizer = None
        
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        try:
            # Initialize matrix processor
            self.matrix_processor = MatrixProcessor(
                use_gpu=self.matrix_config.get("use_gpu", True),
                batch_size=self.matrix_config.get("batch_size", 1000)
            )
            
            # Initialize diverse lookback integrator
            if self.matrix_config.get("use_diverse_lookback", True):
                self.lookback_integrator = DiverseLookbackIntegrator(self.config)
            
            # Initialize matrix optimizer
            self.matrix_optimizer = MatrixOptimizer(
                optimization_level=self.matrix_config.get("optimization_level", "high")
            )
            
            self.logger.info("✅ Enhanced matrix operations components initialized")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Some matrix components not available: {e}")
            # Will use fallback implementations
    
    def validate_inputs(
        self, 
        training_input: Dict[str, Any], 
        pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, list]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for engineered data
        if "engineered_data" not in pipeline_state:
            # Check for individual splits
            if not all(f"{split}_data" in pipeline_state for split in ["train", "val", "test"]):
                errors.append("No engineered data from step 6")
        
        # Check for selected features
        if "selected_features" not in pipeline_state:
            self.logger.warning("No selected features, will use all features")
        
        # Validate matrix computation requirements
        if self.matrix_config.get("matrix_computations", {}).get("regime_transition_matrix", False):
            if "regime_labels" not in pipeline_state:
                self.logger.warning("Regime labels not available for transition matrix")
        
        return len(errors) == 0, errors
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="enhanced matrix operations execution"
    )
    async def execute_logic(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute enhanced matrix operations logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state
        """
        self.logger.info("🔢 Starting enhanced matrix operations...")
        
        # Get data to process
        data_dict = self._get_data_to_process(pipeline_state)
        selected_features = pipeline_state.get("selected_features", [])
        
        # Apply diverse lookback optimization if enabled
        if self.lookback_integrator and selected_features:
            self.logger.info("🔄 Optimizing lookback periods...")
            lookback_results = await self._optimize_lookback_periods(
                data_dict, 
                selected_features
            )
            pipeline_state["lookback_optimization"] = lookback_results
        
        # Perform matrix computations
        matrix_results = {}
        
        for split_name, data in data_dict.items():
            self.logger.info(f"🧮 Computing matrices for {split_name} split...")
            
            # Compute various matrices
            split_matrices = await self._compute_matrices(
                data, 
                selected_features,
                pipeline_state
            )
            
            matrix_results[split_name] = split_matrices
        
        # Perform feature importance analysis
        self.logger.info("📊 Analyzing feature importance...")
        importance_results = await self._analyze_feature_importance(
            data_dict,
            selected_features,
            matrix_results
        )
        
        # Generate optimization insights
        optimization_insights = self._generate_optimization_insights(
            matrix_results,
            importance_results
        )
        
        # Generate reports
        reports = self._generate_matrix_reports(
            matrix_results,
            importance_results,
            optimization_insights
        )
        
        # Update pipeline state
        pipeline_state.update({
            "matrix_results": matrix_results,
            "feature_importance": importance_results,
            "optimization_insights": optimization_insights,
            "matrix_reports": reports,
            "matrix_config": self.matrix_config
        })
        
        # Save outputs
        await self._save_outputs(training_input, pipeline_state)
        
        return pipeline_state
    
    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check if matrix results exist
        if "matrix_results" not in pipeline_state:
            errors.append("No matrix results in pipeline state")
            return False, errors
        
        matrix_results = pipeline_state["matrix_results"]
        
        # Validate matrix computations
        for split_name, matrices in matrix_results.items():
            if not isinstance(matrices, dict):
                errors.append(f"Invalid matrix results for {split_name}")
                continue
            
            # Check for expected matrices based on config
            expected_matrices = []
            matrix_computations = self.matrix_config.get("matrix_computations", {})
            
            if matrix_computations.get("correlation_matrix", True):
                expected_matrices.append("correlation_matrix")
            if matrix_computations.get("covariance_matrix", True):
                expected_matrices.append("covariance_matrix")
            
            missing_matrices = set(expected_matrices) - set(matrices.keys())
            if missing_matrices:
                errors.append(f"Missing matrices for {split_name}: {missing_matrices}")
        
        # Check feature importance
        if "feature_importance" not in pipeline_state:
            errors.append("No feature importance analysis results")
        
        return len(errors) == 0, errors
    
    def _get_data_to_process(self, pipeline_state: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Get data splits to process.
        
        Args:
            pipeline_state: Current pipeline state
            
        Returns:
            Dictionary of data splits
        """
        data_dict = {}
        
        # Check for engineered data first
        if "engineered_data" in pipeline_state:
            return pipeline_state["engineered_data"]
        
        # Otherwise get individual splits
        for split in ["train", "val", "test"]:
            if f"{split}_data" in pipeline_state:
                data_dict[split] = pipeline_state[f"{split}_data"]
        
        return data_dict
    
    async def _optimize_lookback_periods(
        self,
        data_dict: Dict[str, pd.DataFrame],
        selected_features: List[str]
    ) -> Dict[str, Any]:
        """Optimize lookback periods using diverse lookback optimizer.
        
        Args:
            data_dict: Dictionary of data splits
            selected_features: List of selected features
            
        Returns:
            Lookback optimization results
        """
        if self.lookback_integrator:
            # Use training data for optimization
            train_data = data_dict.get("train", next(iter(data_dict.values())))
            
            return await self.lookback_integrator.optimize_lookback_periods(
                train_data,
                selected_features
            )
        else:
            # Return default lookback periods
            return {
                "optimized_periods": {
                    "short": [5, 10, 20],
                    "medium": [50, 100],
                    "long": [200]
                },
                "method": "default"
            }
    
    async def _compute_matrices(
        self,
        data: pd.DataFrame,
        selected_features: List[str],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Compute various matrices for the data.
        
        Args:
            data: Data to process
            selected_features: List of selected features
            pipeline_state: Pipeline state for additional context
            
        Returns:
            Dictionary of computed matrices
        """
        matrices = {}
        
        # Get feature data
        if selected_features:
            feature_data = data[selected_features]
        else:
            # Use all feature columns
            feature_cols = [col for col in data.columns if col.startswith("feature_")]
            feature_data = data[feature_cols]
        
        # Compute matrices based on configuration
        matrix_computations = self.matrix_config.get("matrix_computations", {})
        
        if matrix_computations.get("correlation_matrix", True):
            if self.matrix_processor:
                matrices["correlation_matrix"] = await self.matrix_processor.compute_correlation_matrix(
                    feature_data
                )
            else:
                matrices["correlation_matrix"] = feature_data.corr().values
        
        if matrix_computations.get("covariance_matrix", True):
            if self.matrix_processor:
                matrices["covariance_matrix"] = await self.matrix_processor.compute_covariance_matrix(
                    feature_data
                )
            else:
                matrices["covariance_matrix"] = feature_data.cov().values
        
        if matrix_computations.get("feature_interaction_matrix", True):
            matrices["feature_interaction_matrix"] = self._compute_interaction_matrix(
                feature_data
            )
        
        if matrix_computations.get("regime_transition_matrix", True) and "regime_label" in data.columns:
            matrices["regime_transition_matrix"] = self._compute_regime_transition_matrix(
                data["regime_label"]
            )
        
        return matrices
    
    def _compute_interaction_matrix(self, feature_data: pd.DataFrame) -> np.ndarray:
        """Compute feature interaction matrix.
        
        Args:
            feature_data: Feature data
            
        Returns:
            Interaction matrix
        """
        n_features = len(feature_data.columns)
        interaction_matrix = np.zeros((n_features, n_features))
        
        # Simple interaction: product of standardized features
        standardized = (feature_data - feature_data.mean()) / (feature_data.std() + 1e-8)
        
        for i in range(n_features):
            for j in range(i, n_features):
                interaction = (standardized.iloc[:, i] * standardized.iloc[:, j]).mean()
                interaction_matrix[i, j] = interaction
                interaction_matrix[j, i] = interaction
        
        return interaction_matrix
    
    def _compute_regime_transition_matrix(self, regime_labels: pd.Series) -> np.ndarray:
        """Compute regime transition matrix.
        
        Args:
            regime_labels: Series of regime labels
            
        Returns:
            Transition matrix
        """
        unique_regimes = sorted(regime_labels.unique())
        n_regimes = len(unique_regimes)
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        # Map regimes to indices
        regime_to_idx = {regime: idx for idx, regime in enumerate(unique_regimes)}
        
        # Count transitions
        for i in range(len(regime_labels) - 1):
            from_regime = regime_to_idx[regime_labels.iloc[i]]
            to_regime = regime_to_idx[regime_labels.iloc[i + 1]]
            transition_matrix[from_regime, to_regime] += 1
        
        # Normalize rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(
            transition_matrix, 
            row_sums, 
            where=row_sums != 0
        )
        
        return transition_matrix
    
    async def _analyze_feature_importance(
        self,
        data_dict: Dict[str, pd.DataFrame],
        selected_features: List[str],
        matrix_results: Dict[str, Dict[str, np.ndarray]]
    ) -> Dict[str, Any]:
        """Analyze feature importance using various methods.
        
        Args:
            data_dict: Dictionary of data splits
            selected_features: List of selected features
            matrix_results: Computed matrices
            
        Returns:
            Feature importance results
        """
        importance_results = {}
        
        # Use training data for importance analysis
        train_data = data_dict.get("train", next(iter(data_dict.values())))
        train_matrices = matrix_results.get("train", {})
        
        # Get feature columns
        if selected_features:
            feature_cols = selected_features
        else:
            feature_cols = [col for col in train_data.columns if col.startswith("feature_")]
        
        # Method 1: Correlation-based importance
        if "correlation_matrix" in train_matrices:
            corr_matrix = train_matrices["correlation_matrix"]
            # Average absolute correlation with target (if available)
            if "label" in train_data.columns:
                feature_data = train_data[feature_cols]
                target_corr = feature_data.corrwith(train_data["label"]).abs()
                importance_results["correlation_importance"] = target_corr.to_dict()
        
        # Method 2: Variance-based importance
        feature_data = train_data[feature_cols]
        variance_importance = feature_data.var()
        importance_results["variance_importance"] = variance_importance.to_dict()
        
        # Method 3: Matrix-based importance (eigenvalue decomposition)
        if "covariance_matrix" in train_matrices:
            cov_matrix = train_matrices["covariance_matrix"]
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            # Feature importance based on eigenvalue contribution
            eigenvalue_importance = np.abs(eigenvectors).dot(np.abs(eigenvalues))
            importance_results["eigenvalue_importance"] = {
                feature_cols[i]: float(eigenvalue_importance[i]) 
                for i in range(len(feature_cols))
            }
        
        # Aggregate importance scores
        aggregated_importance = self._aggregate_importance_scores(
            importance_results, 
            feature_cols
        )
        importance_results["aggregated_importance"] = aggregated_importance
        
        return importance_results
    
    def _aggregate_importance_scores(
        self,
        importance_results: Dict[str, Dict[str, float]],
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Aggregate multiple importance scores.
        
        Args:
            importance_results: Dictionary of importance scores by method
            feature_names: List of feature names
            
        Returns:
            Aggregated importance scores
        """
        aggregated = {}
        
        for feature in feature_names:
            scores = []
            
            for method, importance_dict in importance_results.items():
                if isinstance(importance_dict, dict) and feature in importance_dict:
                    score = importance_dict[feature]
                    if not np.isnan(score):
                        scores.append(score)
            
            if scores:
                # Normalize scores to [0, 1] for each method before averaging
                normalized_scores = []
                for method, importance_dict in importance_results.items():
                    if isinstance(importance_dict, dict) and feature in importance_dict:
                        values = list(importance_dict.values())
                        min_val = min(values)
                        max_val = max(values)
                        if max_val > min_val:
                            normalized = (importance_dict[feature] - min_val) / (max_val - min_val)
                            normalized_scores.append(normalized)
                
                if normalized_scores:
                    aggregated[feature] = np.mean(normalized_scores)
        
        return aggregated
    
    def _generate_optimization_insights(
        self,
        matrix_results: Dict[str, Dict[str, np.ndarray]],
        importance_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate optimization insights from matrix analysis.
        
        Args:
            matrix_results: Computed matrices
            importance_results: Feature importance results
            
        Returns:
            Optimization insights
        """
        insights = {
            "feature_recommendations": [],
            "matrix_insights": [],
            "optimization_suggestions": []
        }
        
        # Analyze feature importance
        if "aggregated_importance" in importance_results:
            aggregated = importance_results["aggregated_importance"]
            sorted_features = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
            
            # Recommend top features
            top_k = self.matrix_config.get("feature_selection", {}).get("top_k", 50)
            top_features = [f[0] for f in sorted_features[:top_k]]
            insights["feature_recommendations"] = top_features
            
            # Find low importance features
            min_importance = self.matrix_config.get("feature_selection", {}).get("min_importance", 0.01)
            low_importance = [f[0] for f in sorted_features if f[1] < min_importance]
            if low_importance:
                insights["optimization_suggestions"].append(
                    f"Consider removing {len(low_importance)} low-importance features"
                )
        
        # Analyze correlation matrices
        for split_name, matrices in matrix_results.items():
            if "correlation_matrix" in matrices:
                corr_matrix = matrices["correlation_matrix"]
                
                # Find highly correlated features
                high_corr_pairs = []
                n_features = corr_matrix.shape[0]
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        if abs(corr_matrix[i, j]) > 0.95:
                            high_corr_pairs.append((i, j, corr_matrix[i, j]))
                
                if high_corr_pairs:
                    insights["matrix_insights"].append(
                        f"{split_name}: Found {len(high_corr_pairs)} highly correlated feature pairs"
                    )
                    insights["optimization_suggestions"].append(
                        "Consider removing redundant features from highly correlated pairs"
                    )
        
        return insights
    
    def _generate_matrix_reports(
        self,
        matrix_results: Dict[str, Dict[str, np.ndarray]],
        importance_results: Dict[str, Any],
        optimization_insights: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate reports for matrix analysis.
        
        Args:
            matrix_results: Computed matrices
            importance_results: Feature importance results
            optimization_insights: Optimization insights
            
        Returns:
            Dictionary of reports
        """
        reports = {}
        
        # Summary report
        summary_lines = [
            "Enhanced Matrix Operations Summary",
            "=" * 40,
            "",
            "Matrix Computations:"
        ]
        
        for split_name, matrices in matrix_results.items():
            summary_lines.append(f"\n{split_name.upper()} split:")
            for matrix_name, matrix in matrices.items():
                if isinstance(matrix, np.ndarray):
                    summary_lines.append(
                        f"  {matrix_name}: {matrix.shape} "
                        f"(min={matrix.min():.3f}, max={matrix.max():.3f})"
                    )
        
        # Add feature importance summary
        if "aggregated_importance" in importance_results:
            aggregated = importance_results["aggregated_importance"]
            top_5 = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)[:5]
            
            summary_lines.extend([
                "",
                "Top 5 Important Features:"
            ])
            for feature, score in top_5:
                summary_lines.append(f"  {feature}: {score:.3f}")
        
        reports["summary"] = "\n".join(summary_lines)
        
        # Optimization report
        opt_lines = [
            "Optimization Insights",
            "=" * 40,
            ""
        ]
        
        if optimization_insights.get("feature_recommendations"):
            opt_lines.extend([
                f"Recommended features: {len(optimization_insights['feature_recommendations'])}",
                ""
            ])
        
        for insight in optimization_insights.get("matrix_insights", []):
            opt_lines.append(f"- {insight}")
        
        opt_lines.append("\nOptimization Suggestions:")
        for suggestion in optimization_insights.get("optimization_suggestions", []):
            opt_lines.append(f"- {suggestion}")
        
        reports["optimization"] = "\n".join(opt_lines)
        
        return reports
    
    async def _save_outputs(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> None:
        """Save step outputs to disk.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Pipeline state with results
        """
        output_dir = Path(training_input.get("output_dir", "output")) / "step07_matrix_operations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save matrix results
        if "matrix_results" in pipeline_state:
            for split_name, matrices in pipeline_state["matrix_results"].items():
                split_dir = output_dir / split_name
                split_dir.mkdir(exist_ok=True)
                
                for matrix_name, matrix in matrices.items():
                    if isinstance(matrix, np.ndarray):
                        np.save(split_dir / f"{matrix_name}.npy", matrix)
                
                self.logger.info(f"💾 Saved matrices for {split_name} split")
        
        # Save feature importance
        if "feature_importance" in pipeline_state:
            importance_path = output_dir / "feature_importance.json"
            with open(importance_path, 'w') as f:
                json.dump(pipeline_state["feature_importance"], f, indent=2)
            self.logger.info(f"💾 Saved feature importance to {importance_path}")
        
        # Save optimization insights
        if "optimization_insights" in pipeline_state:
            insights_path = output_dir / "optimization_insights.json"
            with open(insights_path, 'w') as f:
                json.dump(pipeline_state["optimization_insights"], f, indent=2)
            self.logger.info(f"💾 Saved optimization insights")
        
        # Save reports
        if "matrix_reports" in pipeline_state:
            for report_name, content in pipeline_state["matrix_reports"].items():
                report_path = output_dir / f"{report_name}_report.txt"
                with open(report_path, 'w') as f:
                    f.write(content)
                self.logger.info(f"💾 Saved {report_name} report")

        # Persist filtered train/val features for Step 08 legacy consumer
        try:
            symbol = training_input.get("symbol", "UNKNOWN")
            exchange = training_input.get("exchange", "UNKNOWN")
            timeframe = training_input.get("timeframe", "1m")
            data_dir = training_input.get("data_dir", "data/training")

            features_dir = Path(data_dir)
            features_dir.mkdir(parents=True, exist_ok=True)

            # Derive filtered features using selected_features if available
            selected_features = pipeline_state.get("selected_features", [])
            engineered_data = pipeline_state.get("engineered_data", {})

            def _save_split(df, split_name: str) -> None:
                if df is None:
                    return
                if selected_features:
                    available = [c for c in selected_features if c in df.columns]
                    if available:
                        df_to_save = df[available]
                    else:
                        df_to_save = df
                else:
                    df_to_save = df
                out_path = features_dir / f"{exchange}_{symbol}_{timeframe}_features_filtered_{split_name}.parquet"
                try:
                    df_to_save.to_parquet(out_path)
                    self.logger.info(f"💾 Saved filtered features: {out_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to save filtered {split_name} features: {e}")

            train_df = engineered_data.get("train") if isinstance(engineered_data, dict) else None
            val_df = engineered_data.get("val") if isinstance(engineered_data, dict) else None

            _save_split(train_df, "train")
            _save_split(val_df, "val")
        except Exception as e:
            self.logger.warning(f"⚠️ Skipped filtered feature persistence due to error: {e}")
    
    def get_required_inputs(self) -> list:
        """Get list of required inputs for this step."""
        return ["engineered_data or split data with features", "selected_features (optional)"]
    
    def get_produced_outputs(self) -> list:
        """Get list of outputs produced by this step."""
        return [
            "matrix_results", "feature_importance", 
            "optimization_insights", "matrix_reports"
        ]
    
    def get_dependencies(self) -> list:
        """Get list of step dependencies."""
        return ["06_feature_engineering"]