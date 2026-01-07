"""Meta-Labeling HPO Sample Weighted Step.

This step is now a wrapper around the Refactored Orchestrator (MetaLabelingHPOExperimentStep)
to ensure consistent execution of the proper De Prado Causal Framework pipeline
(Layers 0-5) regardless of which legacy step name is invoked.
"""

from src.training.steps.labeling.meta_labeling_hpo_experiment_step import MetaLabelingHPOExperimentStep

class MetaLabelingHPOSampleWeightedStep(MetaLabelingHPOExperimentStep):
    """
    Sample Weighted HPO Step.
    
    Inherits the full Orchestrator logic from MetaLabelingHPOExperimentStep.
    This ensures that 'meta_labeling_hpo_sample_weighted' executes the
    correct L0-L5 causal pipeline.
    """
    def __init__(self, step_name: str = "meta_labeling_hpo_sample_weighted", use_versioned_artifacts: bool = True):
        super().__init__(step_name=step_name, use_versioned_artifacts=use_versioned_artifacts)
