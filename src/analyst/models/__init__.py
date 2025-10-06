"""
Analyst Models Package

This package contains all the ML models for the Analyst component:
- A1: PatchTST-Embed + LightGBM
- A2: PatchTST-Embed + XGBoost  
- A3: FT-Transformer
- A4: PatchTST-Embed + CatBoost
- Stacker: LGBM Calibrated Meta-Learner

All models are designed for binary "green light" classification with
300+ features, regime posteriors, and cross-TF aggregates.
"""

from .analyst_a1_patchtst_lightgbm import (
    AnalystA1Model,
    PatchTSTConfig as A1PatchTSTConfig,
    LightGBMConfig,
    CalibrationConfig as A1CalibrationConfig,
    create_analyst_a1_model
)

from .analyst_a2_patchtst_xgboost import (
    AnalystA2Model,
    XGBoostConfig,
    CalibrationConfig as A2CalibrationConfig,
    create_analyst_a2_model
)

from .analyst_a3_ft_transformer import (
    AnalystA3Model,
    FTTransformerConfig,
    CalibrationConfig as A3CalibrationConfig,
    create_analyst_a3_model
)

from .analyst_a4_patchtst_catboost import (
    AnalystA4Model,
    PatchTSTConfig as A4PatchTSTConfig,
    CatBoostConfig,
    CalibrationConfig as A4CalibrationConfig,
    create_analyst_a4_model
)

from .stacker_lgbm_calibrated import (
    StackerLGBMCalibrated,
    StackerConfig,
    CalibrationConfig as StackerCalibrationConfig,
    create_stacker_lgbm_calibrated
)

from .analyst_models_orchestrator import (
    AnalystModelsOrchestrator,
    AnalystModelsConfig,
    create_analyst_models_orchestrator
)

__all__ = [
    # A1 Model
    'AnalystA1Model',
    'A1PatchTSTConfig',
    'LightGBMConfig',
    'A1CalibrationConfig',
    'create_analyst_a1_model',
    
    # A2 Model
    'AnalystA2Model',
    'XGBoostConfig',
    'A2CalibrationConfig',
    'create_analyst_a2_model',
    
    # A3 Model
    'AnalystA3Model',
    'FTTransformerConfig',
    'A3CalibrationConfig',
    'create_analyst_a3_model',
    
    # A4 Model
    'AnalystA4Model',
    'A4PatchTSTConfig',
    'CatBoostConfig',
    'A4CalibrationConfig',
    'create_analyst_a4_model',
    
    # Stacker Model
    'StackerLGBMCalibrated',
    'StackerConfig',
    'StackerCalibrationConfig',
    'create_stacker_lgbm_calibrated',
    
    # Orchestrator
    'AnalystModelsOrchestrator',
    'AnalystModelsConfig',
    'create_analyst_models_orchestrator'
]