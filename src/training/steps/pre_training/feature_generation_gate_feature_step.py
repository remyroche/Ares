"""
Gate Feature Generation Step

This step generates gate features for quality protection and monitoring
in the machine learning pipeline. Gate features act as quality gates
and protection mechanisms.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field

from src.training.steps.base_step import BaseStep
from src.training.common.component_result import ComponentResult
from src.training.steps.pre_training.gate_feature_integration import (
    GateFeaturePipelineManager,
    get_gate_manager,
    create_gate_manager,
    GateFeatureConfig
)
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug,
    tprint_data_preview, tprint_data_format, tprint_performance, tprint_progress,
    tprint_structured, tprint_timer, tprint_exception
)


@dataclass
class GateFeatureGenerationResult(ComponentResult):
    """Result from gate feature generation step."""
    
    success: bool = False
    gate_features_generated: int = 0
    gate_feature_names: List[str] = field(default_factory=list)
    gate_evaluation_results: List[Dict[str, Any]] = field(default_factory=list)
    gate_manager_status: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    processing_time: float = 0.0


class FeatureGenerationGateFeatureStep(BaseStep):
    """
    Gate Feature Generation Step
    
    Generates gate features for quality protection and monitoring.
    These features act as quality gates and protection mechanisms
    in the machine learning pipeline.
    """
    
    def __init__(self, step_name: str = "feature_generation_gate_feature_step", config: Optional[Dict[str, Any]] = None):
        """Initialize the gate feature generation step."""
        super().__init__(step_name, config)
        
        # Load gate feature configuration
        self.gate_config = self._load_gate_config()
        
        # Initialize gate feature manager
        self.gate_manager = create_gate_manager(self.gate_config)
        
        tprint_info(f"🔧 Initialized {step_name}")
        self.logger.info(f"Initialized {step_name} with gate protection enabled: {self.gate_manager.is_gate_protection_enabled()}")
    
    def _load_gate_config(self) -> Dict[str, Any]:
        """Load gate feature configuration from YAML file."""
        try:
            import yaml
            config_path = "config/gate_feature_config.yaml"
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            tprint_info(f"✅ Loaded gate feature configuration from {config_path}")
            return config
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to load gate config from YAML: {e}")
            tprint_info("🔄 Using default gate configuration")
            
            # Return default configuration
            return {
                "gate_features": {
                    "enable_gate_protection": True,
                    "max_gate_features_per_base": 3,
                    "min_gate_ic_improvement": 0.005,
                    "min_gate_stability": 0.4
                },
                "quality_thresholds": {
                    "max_nan_ratio": 0.3,
                    "min_variance_threshold": 1e-8,
                    "max_correlation_threshold": 0.95,
                    "min_data_points": 100
                },
                "feature_selection": {
                    "enable_feature_importance_gates": True,
                    "enable_correlation_gates": True,
                    "enable_variance_gates": True,
                    "enable_outlier_gates": True
                },
                "integration": {
                    "enable_gate_integration": True,
                    "evaluation_frequency": 1,
                    "failure_threshold": 0.5,
                    "enable_corrective_measures": True
                }
            }
    
    async def execute(self, data: Dict[str, Any] = None) -> GateFeatureGenerationResult:
        """Execute the gate feature generation step."""
        start_time = datetime.now()
        
        try:
            tprint_info("🛡️ Starting gate feature generation...")
            
            if data is None:
                data = {}
            
            # Debug input data
            tprint_data_format(data, "gate_feature_input_data", level="DEBUG")
            
            # Extract features and targets from input data
            features_df = data.get('features')
            targets_series = data.get('targets')
            
            if features_df is None or targets_series is None:
                # Try to load from artifact manager
                tprint_info("🔍 Loading features and targets from artifact manager...")
                
                try:
                    from src.training.common.artifact_manager import artifact_manager
                    
                    # Try to get features from feature generation step
                    features_df = artifact_manager.get_dataframe('feature_generation_feature_generation_step', 'generated_features')
                    if features_df is None:
                        features_df = artifact_manager.get_dataframe('feature_generation_feature_generation_step', 'feature_dataframe')
                    if features_df is None:
                        features_df = artifact_manager.get_dataframe('feature_generation_feature_generation_step', 'features')
                    
                    # Try to get targets from labeling step
                    targets_series = artifact_manager.get_dataframe('feature_generation_labeling_integration_step', 'targets')
                    if targets_series is None:
                        targets_series = artifact_manager.get_dataframe('feature_generation_labeling_integration_step', 'y_train')
                    
                    if targets_series is not None and isinstance(targets_series, pd.DataFrame):
                        # Convert DataFrame to Series if needed
                        if len(targets_series.columns) == 1:
                            targets_series = targets_series.iloc[:, 0]
                        else:
                            # Use the first column as target
                            targets_series = targets_series.iloc[:, 0]
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to load from artifact manager: {e}")
            
            if features_df is None:
                return GateFeatureGenerationResult(
                    success=False,
                    error_message="No features available. Run feature_generation_feature_generation_step first."
                )
            
            if targets_series is None:
                return GateFeatureGenerationResult(
                    success=False,
                    error_message="No targets available. Run feature_generation_labeling_integration_step first."
                )
            
            tprint_info(f"📊 Processing {len(features_df)} samples with {len(features_df.columns)} features")
            tprint_data_preview(features_df, "gate_input_features", level="DEBUG")
            tprint_data_preview(targets_series, "gate_input_targets", level="DEBUG")
            
            # Ensure data alignment
            if len(features_df) != len(targets_series):
                tprint_warning(f"⚠️ Data length mismatch: features={len(features_df)}, targets={len(targets_series)}")
                # Align data by index
                common_index = features_df.index.intersection(targets_series.index)
                features_df = features_df.loc[common_index]
                targets_series = targets_series.loc[common_index]
                tprint_info(f"✅ Aligned data to {len(features_df)} samples")
            
            # Evaluate gate features
            tprint_info("🔍 Evaluating gate features...")
            gate_results = self.gate_manager.evaluate_gate_features(features_df, targets_series)
            
            # Select gate features
            tprint_info("🎯 Selecting gate features...")
            selected_gate_features = self.gate_manager.select_gate_features(features_df, targets_series)
            
            # Generate gate features based on selected features
            gate_features_df = self._generate_gate_features(features_df, targets_series, selected_gate_features)
            
            # Get gate manager status
            gate_status = self.gate_manager.get_gate_status()
            
            # Prepare evaluation results
            evaluation_results = []
            for result in gate_results:
                evaluation_results.append({
                    'feature_name': result.feature_name,
                    'gate_type': result.gate_type.value,
                    'status': result.status.value,
                    'score': result.score,
                    'threshold': result.threshold,
                    'message': result.message,
                    'timestamp': result.timestamp.isoformat()
                })
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Store results in artifact manager
            try:
                from src.training.common.artifact_manager import artifact_manager
                
                # Store gate features
                artifact_manager.store_dataframe('feature_generation_gate_feature_step', 'gate_features', gate_features_df)
                artifact_manager.store_artifact('feature_generation_gate_feature_step', 'gate_feature_names', list(gate_features_df.columns))
                artifact_manager.store_artifact('feature_generation_gate_feature_step', 'gate_evaluation_results', evaluation_results)
                artifact_manager.store_artifact('feature_generation_gate_feature_step', 'gate_manager_status', gate_status)
                
                tprint_success(f"✅ Stored {len(gate_features_df.columns)} gate features in artifact manager")
                
            except Exception as e:
                tprint_warning(f"⚠️ Failed to store in artifact manager: {e}")
            
            result = GateFeatureGenerationResult(
                success=True,
                gate_features_generated=len(gate_features_df.columns),
                gate_feature_names=list(gate_features_df.columns),
                gate_evaluation_results=evaluation_results,
                gate_manager_status=gate_status,
                processing_time=processing_time
            )
            
            tprint_success(f"✅ Gate feature generation completed: {len(gate_features_df.columns)} features generated in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            tprint_error(f"❌ Gate feature generation failed: {e}")
            tprint_exception(e)
            
            return GateFeatureGenerationResult(
                success=False,
                error_message=str(e),
                processing_time=processing_time
            )
    
    def _generate_gate_features(self, features_df: pd.DataFrame, targets_series: pd.Series, selected_features: List[str]) -> pd.DataFrame:
        """Generate gate features based on selected features."""
        tprint_info(f"🔧 Generating gate features from {len(selected_features)} selected features...")
        
        gate_features = {}
        
        for feature_name in selected_features:
            if feature_name not in features_df.columns:
                tprint_warning(f"⚠️ Selected feature '{feature_name}' not found in features")
                continue
            
            feature_values = features_df[feature_name]
            
            # Generate different types of gate features
            gate_features[f"{feature_name}_gate_quality"] = self._generate_quality_gate(feature_values, targets_series)
            gate_features[f"{feature_name}_gate_stability"] = self._generate_stability_gate(feature_values)
            gate_features[f"{feature_name}_gate_variance"] = self._generate_variance_gate(feature_values)
        
        # Add global gate features
        gate_features["global_data_quality_gate"] = self._generate_global_quality_gate(features_df, targets_series)
        gate_features["global_correlation_gate"] = self._generate_global_correlation_gate(features_df)
        gate_features["global_variance_gate"] = self._generate_global_variance_gate(features_df)
        
        gate_features_df = pd.DataFrame(gate_features, index=features_df.index)
        
        tprint_success(f"✅ Generated {len(gate_features_df.columns)} gate features")
        tprint_data_preview(gate_features_df, "generated_gate_features", level="DEBUG")
        
        return gate_features_df
    
    def _generate_quality_gate(self, feature_values: pd.Series, targets_series: pd.Series) -> pd.Series:
        """Generate quality gate feature."""
        # Check for NaN ratio
        nan_ratio = feature_values.isnull().sum() / len(feature_values)
        
        # Check for variance
        variance = feature_values.var()
        
        # Quality score (higher is better)
        quality_score = (1.0 - nan_ratio) * min(variance * 1000, 1.0)  # Scale variance
        
        return pd.Series([quality_score] * len(feature_values), index=feature_values.index)
    
    def _generate_stability_gate(self, feature_values: pd.Series) -> pd.Series:
        """Generate stability gate feature."""
        # Calculate rolling standard deviation as stability measure
        rolling_std = feature_values.rolling(window=min(20, len(feature_values)//4)).std()
        
        # Stability score (lower std is more stable)
        stability_score = 1.0 / (1.0 + rolling_std.fillna(0))
        
        return stability_score.fillna(0.5)  # Default score for NaN values
    
    def _generate_variance_gate(self, feature_values: pd.Series) -> pd.Series:
        """Generate variance gate feature."""
        # Calculate rolling variance
        rolling_var = feature_values.rolling(window=min(20, len(feature_values)//4)).var()
        
        # Variance score (moderate variance is good)
        variance_score = np.tanh(rolling_var.fillna(0) * 100)  # Tanh to bound between -1 and 1
        
        return variance_score.fillna(0.0)
    
    def _generate_global_quality_gate(self, features_df: pd.DataFrame, targets_series: pd.Series) -> pd.Series:
        """Generate global data quality gate."""
        # Overall data quality metrics
        nan_ratio = features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns))
        target_variance = targets_series.var()
        
        # Global quality score
        quality_score = (1.0 - nan_ratio) * min(target_variance * 100, 1.0)
        
        return pd.Series([quality_score] * len(features_df), index=features_df.index)
    
    def _generate_global_correlation_gate(self, features_df: pd.DataFrame) -> pd.Series:
        """Generate global correlation gate."""
        # Calculate average correlation between features
        corr_matrix = features_df.corr().abs()
        avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        
        # Correlation score (moderate correlation is good)
        correlation_score = 1.0 - avg_correlation  # Lower correlation is better
        
        return pd.Series([correlation_score] * len(features_df), index=features_df.index)
    
    def _generate_global_variance_gate(self, features_df: pd.DataFrame) -> pd.Series:
        """Generate global variance gate."""
        # Calculate average variance across features
        avg_variance = features_df.var().mean()
        
        # Variance score (moderate variance is good)
        variance_score = np.tanh(avg_variance * 1000)  # Tanh to bound between -1 and 1
        
        return pd.Series([variance_score] * len(features_df), index=features_df.index)


async def handle_feature_generation_gate_feature_step(
    step_name: str = "feature_generation_gate_feature_step",
    config: Optional[Dict[str, Any]] = None,
    data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Handle gate feature generation step execution.
    
    Args:
        step_name: Name of the step
        config: Step configuration
        data: Input data
        
    Returns:
        Step execution result
    """
    try:
        tprint_info(f"🚀 Starting {step_name}...")
        
        # Create step instance
        step = FeatureGenerationGateFeatureStep(step_name, config)
        
        # Execute step
        result = await step.execute(data)
        
        # Convert result to dictionary
        result_dict = {
            'success': result.success,
            'gate_features_generated': result.gate_features_generated,
            'gate_feature_names': result.gate_feature_names,
            'gate_evaluation_results': result.gate_evaluation_results,
            'gate_manager_status': result.gate_manager_status,
            'error_message': result.error_message,
            'processing_time': result.processing_time,
            'step_name': step_name,
            'timestamp': datetime.now().isoformat()
        }
        
        if result.success:
            tprint_success(f"✅ {step_name} completed successfully")
        else:
            tprint_error(f"❌ {step_name} failed: {result.error_message}")
        
        return result_dict
        
    except Exception as e:
        tprint_error(f"❌ {step_name} execution failed: {e}")
        tprint_exception(e)
        
        return {
            'success': False,
            'error_message': str(e),
            'step_name': step_name,
            'timestamp': datetime.now().isoformat()
        }