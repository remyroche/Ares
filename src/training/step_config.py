"""Centralized configuration for all training pipeline steps.

This module defines the standardized step configuration, including:
- Step ordering
- Step metadata
- Dependencies
- Required inputs/outputs
"""

from typing import Dict, List, Any
from dataclasses import dataclass, field


@dataclass
class StepConfig:
    """Configuration for a single pipeline step."""
    
    step_number: str
    step_name: str
    description: str
    module_path: str
    class_name: str
    dependencies: List[str] = field(default_factory=list)
    required_inputs: List[str] = field(default_factory=list)
    produced_outputs: List[str] = field(default_factory=list)
    required_files: List[str] = field(default_factory=list)
    optional: bool = False
    enabled: bool = True
    
    @property
    def full_name(self) -> str:
        """Get full step name."""
        return f"step{self.step_number}_{self.step_name}"


# Define all pipeline steps in order
PIPELINE_STEPS: Dict[str, StepConfig] = {
    "01": StepConfig(
        step_number="01",
        step_name="data_collection",
        description="Download and consolidate market data",
        module_path="src.training.steps.step1_data_collection",
        class_name="DataCollectionStep",
        dependencies=[],
        required_inputs=["symbol", "exchange", "timeframe"],
        produced_outputs=["raw_market_data"],
        required_files=["data_cache/klines_*_*_1m_consolidated.parquet"]
    ),
    
    "01_5": StepConfig(
        step_number="01_5",
        step_name="data_converter",
        description="Convert data to unified format",
        module_path="src.training.steps.step1_5_data_converter",
        class_name="DataConverterStep",
        dependencies=["01"],
        required_inputs=["raw_market_data"],
        produced_outputs=["unified_market_data"],
        required_files=["data_cache/unified/*/*/*/*.parquet"]
    ),
    
    "02": StepConfig(
        step_number="02",
        step_name="feature_engineering",
        description="Generate basic technical indicators",
        module_path="src.training.steps.step2_feature_engineering",
        class_name="FeatureEngineeringStep",
        dependencies=["01_5"],
        required_inputs=["unified_market_data"],
        produced_outputs=["basic_features"],
        required_files=["data/training/*_features_basic.parquet"]
    ),
    
    "03": StepConfig(
        step_number="03",
        step_name="hmm_regime_discovery",
        description="Identify market regimes using HMM",
        module_path="src.training.steps.step3_hmm_regime_discovery",
        class_name="HMMRegimeDiscoveryStep",
        dependencies=["02"],
        required_inputs=["basic_features"],
        produced_outputs=["regime_labels", "regime_transitions"],
        required_files=["data/hmm_regimes/*_composite_clusters.parquet"]
    ),
    
    "04": StepConfig(
        step_number="04",
        step_name="regime_data_splitting",
        description="Split data based on regimes",
        module_path="src.training.steps.step4_regime_data_splitting",
        class_name="RegimeDataSplittingStep",
        dependencies=["03"],
        required_inputs=["regime_labels"],
        produced_outputs=["regime_splits"],
        required_files=["data/training/*_regime_splits.parquet"]
    ),
    
    "05": StepConfig(
        step_number="05",
        step_name="triple_barrier_labeling",
        description="Apply triple barrier method for labeling",
        module_path="src.training.steps.step5_triple_barrier_method",
        class_name="TripleBarrierStep",
        dependencies=["04"],
        required_inputs=["regime_splits"],
        produced_outputs=["labeled_data"],
        required_files=["data/training/*_triple_barrier_*.parquet"]
    ),
    
    "06": StepConfig(
        step_number="06",
        step_name="advanced_feature_engineering",
        description="Generate advanced features",
        module_path="src.training.steps.step6_feature_engineering",
        class_name="AdvancedFeatureEngineeringStep",
        dependencies=["05"],
        required_inputs=["labeled_data"],
        produced_outputs=["advanced_features"],
        required_files=["data/training/*_features_train.parquet", "data/training/*_features_val.parquet"]
    ),
    
    "07": StepConfig(
        step_number="07",
        step_name="feature_selection",
        description="Select optimal features using matrix operations",
        module_path="src.training.steps.step7_enhanced_matrix_operations",
        class_name="FeatureSelectionStep",
        dependencies=["06"],
        required_inputs=["advanced_features"],
        produced_outputs=["selected_features"],
        required_files=["data/matrix_operations/*_matrix_operations_*.json"]
    ),
    
    "08": StepConfig(
        step_number="08",
        step_name="regime_model_training",
        description="Train regime-specific models",
        module_path="src.training.steps.step8_regime_data_splitting",
        class_name="RegimeModelTrainingStep",
        dependencies=["07"],
        required_inputs=["selected_features", "regime_splits"],
        produced_outputs=["regime_models"],
        required_files=["data/training/*_regime_models.pkl"]
    ),
    
    "09": StepConfig(
        step_number="09",
        step_name="hmm_model_training",
        description="Train HMM-enhanced models",
        module_path="src.training.steps.step9_hmm_based_training",
        class_name="HMMModelTrainingStep",
        dependencies=["08"],
        required_inputs=["regime_models"],
        produced_outputs=["hmm_models"],
        required_files=["data/training/*_hmm_models.pkl"]
    ),
    
    "10": StepConfig(
        step_number="10",
        step_name="unified_intelligence",
        description="Create unified regime intelligence system",
        module_path="src.training.steps.step10_unified_regime_intelligence",
        class_name="UnifiedIntelligenceStep",
        dependencies=["09"],
        required_inputs=["hmm_models"],
        produced_outputs=["unified_system"],
        required_files=["data/training/*_unified_intelligence.pkl"]
    ),
    
    "11": StepConfig(
        step_number="11",
        step_name="analyst_creation",
        description="Create analyst models",
        module_path="src.training.steps.step11_analyst_creation",
        class_name="AnalystCreationStep",
        dependencies=["10"],
        required_inputs=["unified_system"],
        produced_outputs=["analyst_models"],
        required_files=["data/training/*_analyst_models.pkl"]
    ),
    
    "12": StepConfig(
        step_number="12",
        step_name="analyst_enhancement",
        description="Enhance analyst models",
        module_path="src.training.steps.step12_analyst_enhancement",
        class_name="AnalystEnhancementStep",
        dependencies=["11"],
        required_inputs=["analyst_models"],
        produced_outputs=["enhanced_analysts"],
        required_files=["data/training/*_enhanced_analyst_models.pkl"]
    ),
    
    "13": StepConfig(
        step_number="13",
        step_name="ensemble_creation",
        description="Create analyst ensemble",
        module_path="src.training.steps.step13_analyst_ensemble_creation",
        class_name="EnsembleCreationStep",
        dependencies=["12"],
        required_inputs=["enhanced_analysts"],
        produced_outputs=["analyst_ensemble"],
        required_files=["data/training/*_analyst_ensemble.pkl"]
    ),
    
    "14": StepConfig(
        step_number="14",
        step_name="tactician_labeling",
        description="Generate tactical trading labels",
        module_path="src.training.steps.step14_tactician_labeling",
        class_name="TacticianLabelingStep",
        dependencies=["13"],
        required_inputs=["analyst_ensemble"],
        produced_outputs=["tactical_labels"],
        required_files=["data/training/*_tactician_labels.parquet"]
    ),
    
    "15": StepConfig(
        step_number="15",
        step_name="tactician_training",
        description="Train tactical trading models",
        module_path="src.training.steps.step15_tactician_specialist_training",
        class_name="TacticianTrainingStep",
        dependencies=["14"],
        required_inputs=["tactical_labels"],
        produced_outputs=["tactician_models"],
        required_files=["data/training/*_tactician_models.pkl"]
    ),
    
    "16": StepConfig(
        step_number="16",
        step_name="confidence_calibration",
        description="Calibrate model confidence scores",
        module_path="src.training.steps.step16_confidence_calibration",
        class_name="ConfidenceCalibrationStep",
        dependencies=["15"],
        required_inputs=["tactician_models"],
        produced_outputs=["calibrated_models"],
        required_files=["data/training/*_calibrated_models.pkl"]
    ),
    
    "17": StepConfig(
        step_number="17",
        step_name="parameter_optimization",
        description="Optimize final model parameters",
        module_path="src.training.steps.step17_final_parameters_optimization",
        class_name="ParameterOptimizationStep",
        dependencies=["16"],
        required_inputs=["calibrated_models"],
        produced_outputs=["optimized_models"],
        required_files=["data/training/*_optimized_models.pkl"]
    ),
    
    "18": StepConfig(
        step_number="18",
        step_name="walk_forward_validation",
        description="Validate models using walk-forward analysis",
        module_path="src.training.steps.step18_walk_forward_validation",
        class_name="WalkForwardValidationStep",
        dependencies=["17"],
        required_inputs=["optimized_models"],
        produced_outputs=["walk_forward_results"],
        required_files=["data/training/*_walk_forward_results.json"]
    ),
    
    "19": StepConfig(
        step_number="19",
        step_name="monte_carlo_validation",
        description="Validate models using Monte Carlo simulation",
        module_path="src.training.steps.step19_monte_carlo_validation",
        class_name="MonteCarloValidationStep",
        dependencies=["18"],
        required_inputs=["optimized_models"],
        produced_outputs=["monte_carlo_results"],
        required_files=["data/training/*_monte_carlo_results.json"]
    ),
    
    "20": StepConfig(
        step_number="20",
        step_name="ab_testing",
        description="Compare model performance",
        module_path="src.training.steps.step20_ab_testing",
        class_name="ABTestingStep",
        dependencies=["19"],
        required_inputs=["optimized_models", "validation_results"],
        produced_outputs=["ab_test_results"],
        required_files=["data/training/*_ab_test_results.json"],
        optional=True
    ),
    
    "21": StepConfig(
        step_number="21",
        step_name="model_saving",
        description="Save all trained models and configurations",
        module_path="src.training.steps.step21_saving",
        class_name="ModelSavingStep",
        dependencies=["17"],  # Can run after optimization, doesn't need validation
        required_inputs=["all_models", "all_results"],
        produced_outputs=["saved_models"],
        required_files=["models/*_final_models.pkl"]
    ),
}


def get_step_config(step_number: str) -> StepConfig:
    """Get configuration for a specific step.
    
    Args:
        step_number: Step number (e.g., "01", "02", "03")
        
    Returns:
        StepConfig object
        
    Raises:
        KeyError: If step number not found
    """
    if step_number not in PIPELINE_STEPS:
        raise KeyError(f"Step {step_number} not found in pipeline configuration")
    return PIPELINE_STEPS[step_number]


def get_all_steps() -> List[StepConfig]:
    """Get all step configurations in order.
    
    Returns:
        List of StepConfig objects
    """
    return list(PIPELINE_STEPS.values())


def get_enabled_steps() -> List[StepConfig]:
    """Get all enabled step configurations.
    
    Returns:
        List of enabled StepConfig objects
    """
    return [step for step in PIPELINE_STEPS.values() if step.enabled]


def get_step_dependencies(step_number: str) -> List[StepConfig]:
    """Get all dependencies for a specific step.
    
    Args:
        step_number: Step number
        
    Returns:
        List of StepConfig objects that this step depends on
    """
    step = get_step_config(step_number)
    return [get_step_config(dep) for dep in step.dependencies]


def get_step_execution_order() -> List[str]:
    """Get the correct execution order for all steps.
    
    Returns:
        List of step numbers in execution order
    """
    # This is already in the correct order in PIPELINE_STEPS
    return list(PIPELINE_STEPS.keys())


def validate_step_sequence() -> Dict[str, Any]:
    """Validate the step sequence for consistency.
    
    Returns:
        Dictionary with validation results
    """
    issues = []
    
    # Check that all dependencies exist
    for step_num, step in PIPELINE_STEPS.items():
        for dep in step.dependencies:
            if dep not in PIPELINE_STEPS:
                issues.append(f"Step {step_num} depends on non-existent step {dep}")
    
    # Check for circular dependencies
    def has_circular_dependency(step_num: str, visited: set) -> bool:
        if step_num in visited:
            return True
        visited.add(step_num)
        step = PIPELINE_STEPS.get(step_num)
        if step:
            for dep in step.dependencies:
                if has_circular_dependency(dep, visited.copy()):
                    return True
        return False
    
    for step_num in PIPELINE_STEPS:
        if has_circular_dependency(step_num, set()):
            issues.append(f"Step {step_num} has circular dependencies")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "total_steps": len(PIPELINE_STEPS),
        "enabled_steps": len(get_enabled_steps())
    }