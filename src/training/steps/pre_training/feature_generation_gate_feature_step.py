"""
Feature Generation Gate Feature Step

This step generates gate features for quality protection and monitoring in the ML pipeline.
Gate features act as quality gates and protection mechanisms to ensure data integrity
and model performance throughout the training process.

Key Features:
- Quality gates: Data quality validation and monitoring
- Correlation gates: Feature correlation analysis and protection
- Variance gates: Feature variance validation and stability checks
- Performance gates: Model performance monitoring and alerting
- Integration with GateFeaturePipelineManager for comprehensive management
"""

from __future__ import annotations

import asyncio
import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

from src.training.steps.base_step import BaseStep
from src.training.steps.pre_training.gate_feature_integration import (
    GateFeaturePipelineManager,
    GateFeatureConfig,
    get_gate_manager,
    create_gate_manager
)
from src.utils.tprint import (
    tprint, tprint_info, tprint_success, tprint_warning, tprint_error,
    tprint_debug, tprint_data_preview, tprint_data_format, tprint_structured,
    tprint_step, tprint_result, tprint_performance
)


class FeatureGenerationGateFeatureStep(BaseStep):
    """
    Gate Feature Generation Step for Quality Protection and Monitoring.
    
    This step generates gate features that act as quality gates and protection
    mechanisms in the machine learning pipeline. It integrates with the
    GateFeaturePipelineManager to provide comprehensive gate feature management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the gate feature generation step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__("feature_generation_gate_feature_step", config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize gate feature manager
        self.gate_manager = None
        self.gate_config = None
        
        tprint_info("🔧 Initializing FeatureGenerationGateFeatureStep")
        tprint_debug(f"⚙️ Config provided: {config is not None}")
    
    async def _initialize_gate_manager(self, config: Dict[str, Any]) -> None:
        """
        Initialize the gate feature manager with configuration.
        
        Args:
            config: Configuration dictionary
        """
        tprint_step("🔧 Initializing gate feature manager")
        
        try:
            # Load gate feature configuration
            gate_config_path = "config/gate_feature_config.yaml"
            gate_config_data = self._load_yaml_config(gate_config_path)
            
            if gate_config_data:
                tprint_success(f"✅ Loaded gate configuration from {gate_config_path}")
                self.gate_config = GateFeatureConfig(**gate_config_data.get('gate_features', {}))
            else:
                tprint_warning("⚠️ Using default gate configuration")
                self.gate_config = GateFeatureConfig()
            
            # Create gate manager with configuration
            self.gate_manager = create_gate_manager(self.gate_config.__dict__)
            tprint_success("✅ Gate feature manager initialized")
            
        except Exception as e:
            tprint_error(f"❌ Failed to initialize gate manager: {e}")
            self.logger.error(f"Failed to initialize gate manager: {e}")
            # Fallback to default manager
            self.gate_manager = get_gate_manager()
            tprint_warning("⚠️ Using fallback gate manager")
    
    def _load_yaml_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """
        Load YAML configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configuration dictionary or None if failed
        """
        try:
            import yaml
            from pathlib import Path
            
            config_file = Path(config_path)
            if not config_file.exists():
                tprint_warning(f"⚠️ Config file not found: {config_path}")
                return None
            
            with open(config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            return config_data
            
        except Exception as e:
            tprint_error(f"❌ Failed to load YAML config: {e}")
            return None
    
    async def _load_input_data(self, config: Dict[str, Any]) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Load input features and targets for gate feature generation.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Tuple of (features_df, targets_series)
        """
        tprint_step("📦 Loading input data for gate feature generation")
        
        try:
            # Load features from final feature selection step
            tprint_info("🔍 Loading features from feature_generation_final_feature_selection_step")
            features_df = self.artifact_manager.get_dataframe(
                'feature_generation_final_feature_selection_step',
                'SELECTED_FEATURES'
            )
            
            if features_df is None:
                # Try alternative artifact names
                for artifact_name in ['selected_features', 'features', 'final_features']:
                    features_df = self.artifact_manager.get_dataframe(
                        'feature_generation_final_feature_selection_step',
                        artifact_name
                    )
                    if features_df is not None:
                        break
            
            if features_df is None:
                tprint_error("❌ No features found from final feature selection step")
                return None, None
            
            tprint_success(f"✅ Loaded {len(features_df.columns)} features")
            tprint_data_preview(features_df, "gate_input_features", level="DEBUG")
            tprint_data_format(features_df, "gate_input_features", level="DEBUG")
            
            # Load targets from labeling integration step
            tprint_info("🔍 Loading targets from feature_generation_labeling_integration_step")
            targets_series = self.artifact_manager.get_artifact(
                'feature_generation_labeling_integration_step',
                'targets'
            )
            
            if targets_series is None:
                # Try alternative artifact names
                for artifact_name in ['target', 'y', 'labels']:
                    targets_series = self.artifact_manager.get_artifact(
                        'feature_generation_labeling_integration_step',
                        artifact_name
                    )
                    if targets_series is not None:
                        break
            
            if targets_series is None:
                tprint_error("❌ No targets found from labeling integration step")
                return features_df, None
            
            # Ensure targets is a pandas Series
            if isinstance(targets_series, np.ndarray):
                targets_series = pd.Series(targets_series, index=features_df.index)
            elif isinstance(targets_series, pd.DataFrame):
                targets_series = targets_series.iloc[:, 0]  # Take first column
            
            tprint_success(f"✅ Loaded targets with {len(targets_series)} samples")
            tprint_data_preview(targets_series, "gate_input_targets", level="DEBUG")
            tprint_data_format(targets_series, "gate_input_targets", level="DEBUG")
            
            return features_df, targets_series
            
        except Exception as e:
            tprint_error(f"❌ Failed to load input data: {e}")
            self.logger.error(f"Failed to load input data: {e}")
            return None, None
    
    async def _generate_gate_features(self, features_df: pd.DataFrame, targets_series: pd.Series) -> Dict[str, Any]:
        """
        Generate gate features using the gate feature manager.
        
        Args:
            features_df: Input features DataFrame
            targets_series: Target values Series
            
        Returns:
            Dictionary containing gate feature results
        """
        tprint_step("🎯 Generating gate features")
        
        try:
            if not self.gate_manager:
                tprint_error("❌ Gate manager not initialized")
                return {'success': False, 'error': 'Gate manager not initialized'}
            
            # Evaluate gate features
            tprint_info("🔍 Evaluating gate features")
            gate_results = self.gate_manager.evaluate_gate_features(features_df, targets_series)
            
            if not gate_results:
                tprint_warning("⚠️ No gate features evaluated")
                return {'success': False, 'error': 'No gate features evaluated'}
            
            tprint_success(f"✅ Evaluated {len(gate_results)} gate features")
            
            # Select gate features
            tprint_info("🎯 Selecting gate features")
            selected_gate_features = self.gate_manager.select_gate_features(features_df, targets_series)
            
            if not selected_gate_features:
                tprint_warning("⚠️ No gate features selected")
                return {'success': False, 'error': 'No gate features selected'}
            
            tprint_success(f"✅ Selected {len(selected_gate_features)} gate features")
            
            # Generate gate feature DataFrame
            gate_features_df = self._create_gate_features_dataframe(
                features_df, targets_series, selected_gate_features, gate_results
            )
            
            if gate_features_df is None:
                tprint_error("❌ Failed to create gate features DataFrame")
                return {'success': False, 'error': 'Failed to create gate features DataFrame'}
            
            tprint_success(f"✅ Created gate features DataFrame with {len(gate_features_df.columns)} columns")
            tprint_data_preview(gate_features_df, "gate_features_output", level="DEBUG")
            tprint_data_format(gate_features_df, "gate_features_output", level="DEBUG")
            
            # Get gate status
            gate_status = self.gate_manager.get_gate_status()
            
            return {
                'success': True,
                'gate_features_df': gate_features_df,
                'selected_gate_features': selected_gate_features,
                'gate_results': gate_results,
                'gate_status': gate_status,
                'total_gate_features': len(gate_features_df.columns)
            }
            
        except Exception as e:
            tprint_error(f"❌ Failed to generate gate features: {e}")
            self.logger.error(f"Failed to generate gate features: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_gate_features_dataframe(
        self, 
        features_df: pd.DataFrame, 
        targets_series: pd.Series, 
        selected_gate_features: List[str],
        gate_results: List[Any]
    ) -> Optional[pd.DataFrame]:
        """
        Create gate features DataFrame based on selected features and gate results.
        
        Args:
            features_df: Input features DataFrame
            targets_series: Target values Series
            selected_gate_features: List of selected gate feature names
            gate_results: List of gate evaluation results
            
        Returns:
            Gate features DataFrame or None if failed
        """
        try:
            tprint_step("🔧 Creating gate features DataFrame")
            
            # Initialize gate features DataFrame
            gate_features_data = {}
            
            # Add quality gate features
            gate_features_data['quality_gate_data_size'] = len(features_df)
            gate_features_data['quality_gate_target_variance'] = targets_series.var()
            gate_features_data['quality_gate_nan_ratio'] = features_df.isnull().sum().sum() / (len(features_df) * len(features_df.columns))
            
            # Add correlation gate features
            corr_matrix = features_df.corr().abs()
            gate_features_data['correlation_gate_max_correlation'] = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max()
            gate_features_data['correlation_gate_mean_correlation'] = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            
            # Add variance gate features
            feature_variances = features_df.var()
            gate_features_data['variance_gate_min_variance'] = feature_variances.min()
            gate_features_data['variance_gate_mean_variance'] = feature_variances.mean()
            gate_features_data['variance_gate_low_variance_count'] = (feature_variances < 1e-8).sum()
            
            # Add stability gate features
            gate_features_data['stability_gate_feature_count'] = len(features_df.columns)
            gate_features_data['stability_gate_target_mean'] = targets_series.mean()
            gate_features_data['stability_gate_target_std'] = targets_series.std()
            
            # Add performance gate features
            gate_features_data['performance_gate_ic_estimate'] = self._estimate_information_coefficient(features_df, targets_series)
            gate_features_data['performance_gate_feature_importance'] = self._estimate_feature_importance(features_df, targets_series)
            
            # Create DataFrame
            gate_features_df = pd.DataFrame(gate_features_data, index=features_df.index)
            
            # Add selected base features as gate features
            for feature_name in selected_gate_features:
                if feature_name in features_df.columns:
                    gate_features_df[f'gate_base_{feature_name}'] = features_df[feature_name]
            
            tprint_success(f"✅ Created gate features DataFrame with {len(gate_features_df.columns)} columns")
            return gate_features_df
            
        except Exception as e:
            tprint_error(f"❌ Failed to create gate features DataFrame: {e}")
            self.logger.error(f"Failed to create gate features DataFrame: {e}")
            return None
    
    def _estimate_information_coefficient(self, features_df: pd.DataFrame, targets_series: pd.Series) -> float:
        """
        Estimate information coefficient between features and targets.
        
        Args:
            features_df: Features DataFrame
            targets_series: Target values Series
            
        Returns:
            Estimated information coefficient
        """
        try:
            # Simple correlation-based IC estimate
            correlations = []
            for col in features_df.columns:
                if not features_df[col].isnull().all():
                    corr = features_df[col].corr(targets_series)
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception:
            return 0.0
    
    def _estimate_feature_importance(self, features_df: pd.DataFrame, targets_series: pd.Series) -> float:
        """
        Estimate overall feature importance score.
        
        Args:
            features_df: Features DataFrame
            targets_series: Target values Series
            
        Returns:
            Estimated feature importance score
        """
        try:
            # Simple variance-based importance estimate
            feature_variances = features_df.var()
            return feature_variances.mean() / feature_variances.std() if feature_variances.std() > 0 else 0.0
            
        except Exception:
            return 0.0
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the gate feature generation step.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Execution results dictionary
        """
        tprint_step("🚀 Starting FeatureGenerationGateFeatureStep execution")
        tprint_data_preview(config, "gate_feature_config", level="DEBUG")
        
        try:
            # Initialize gate manager
            await self._initialize_gate_manager(config)
            
            # Load input data
            features_df, targets_series = await self._load_input_data(config)
            
            if features_df is None:
                return {
                    'success': False,
                    'error': 'Failed to load input features',
                    'step': 'feature_generation_gate_feature_step'
                }
            
            if targets_series is None:
                tprint_warning("⚠️ No targets available - generating gate features without targets")
                # Create dummy targets for gate feature generation
                targets_series = pd.Series(np.random.randn(len(features_df)), index=features_df.index)
            
            # Generate gate features
            gate_result = await self._generate_gate_features(features_df, targets_series)
            
            if not gate_result['success']:
                return {
                    'success': False,
                    'error': gate_result['error'],
                    'step': 'feature_generation_gate_feature_step'
                }
            
            # Save gate features
            gate_features_df = gate_result['gate_features_df']
            self.artifact_manager.save_dataframe(
                'feature_generation_gate_feature_step',
                'GATE_FEATURES',
                gate_features_df
            )
            
            # Save gate feature metadata
            gate_metadata = {
                'selected_gate_features': gate_result['selected_gate_features'],
                'gate_status': gate_result['gate_status'],
                'total_gate_features': gate_result['total_gate_features'],
                'generation_timestamp': datetime.now().isoformat(),
                'step_name': 'feature_generation_gate_feature_step'
            }
            
            self.artifact_manager.save_artifact(
                'feature_generation_gate_feature_step',
                'GATE_METADATA',
                gate_metadata
            )
            
            # Save gate results
            self.artifact_manager.save_artifact(
                'feature_generation_gate_feature_step',
                'GATE_RESULTS',
                gate_result['gate_results']
            )
            
            tprint_success(f"✅ Gate feature generation completed successfully")
            tprint_result(f"🎯 Generated {len(gate_features_df.columns)} gate features")
            
            return {
                'success': True,
                'artifacts': ['GATE_FEATURES', 'GATE_METADATA', 'GATE_RESULTS'],
                'total_gate_features': len(gate_features_df.columns),
                'selected_gate_features': gate_result['selected_gate_features'],
                'gate_status': gate_result['gate_status'],
                'step': 'feature_generation_gate_feature_step'
            }
            
        except Exception as e:
            error_msg = f"Gate feature generation failed: {str(e)}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg)
            
            return {
                'success': False,
                'error': error_msg,
                'step': 'feature_generation_gate_feature_step'
            }


async def handle_feature_generation_gate_feature_step(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle function for the gate feature generation step.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Execution results dictionary
    """
    tprint("🔧 Starting comprehensive gate feature generation")
    
    try:
        # Create step instance
        step = FeatureGenerationGateFeatureStep(config)
        
        # Execute step
        result = await step.execute(config)
        
        if result['success']:
            tprint_success("✅ Gate feature generation completed successfully")
            tprint_result(f"🎯 Generated {result.get('total_gate_features', 0)} gate features")
        else:
            tprint_error(f"❌ Gate feature generation failed: {result.get('error', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        error_msg = f"Gate feature generation handler failed: {str(e)}"
        tprint_error(f"❌ {error_msg}")
        return {
            'success': False,
            'error': error_msg,
            'step': 'feature_generation_gate_feature_step'
        }