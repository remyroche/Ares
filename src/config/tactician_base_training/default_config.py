#!/usr/bin/env python3
"""
Configuration Python par défaut pour Tactician Base Training

Ce fichier contient la configuration centralisée pour l'entraînement des modèles
tactician de base, au format Python pour une configuration programmatique dynamique.

Version: 1.0.0
Date: 2025-11-03T22:24:00.000Z
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime


@dataclass
class BaseModelConfig:
    """Configuration pour un modèle de base."""
    model_name: str
    class_name: str
    is_feature_generator: bool = False
    model_type: Optional[str] = None  # Optionnel, pour les modèles nécessitant un type spécifique
    params: Dict[str, Any] = field(default_factory=dict)
    hpo: Dict[str, Any] = field(default_factory=dict)
    optimal_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration d'entraînement."""
    enable_cross_validation: bool = True
    cv_folds: int = 3
    enable_early_stopping: bool = True
    early_stopping_patience: int = 10
    validation_split: float = 0.2
    test_split: float = 0.1
    # Deprecated - maintenant calculés en pourcentage
    training_samples: Optional[int] = None
    validation_samples: Optional[int] = None
    test_samples: Optional[int] = None


@dataclass
class PerformanceConfig:
    """Configuration de performance."""
    expected_accuracy: float = 0.82
    expected_sharpe_ratio: float = 1.45
    training_time_limit: int = 200  # secondes
    memory_limit_mb: int = 2048


@dataclass
class OutputConfig:
    """Configuration de sortie."""
    save_models: bool = True
    save_predictions: bool = True
    generate_reports: bool = True
    output_dir: str = "./tactician_base_models"


@dataclass
class PrimaryFeaturesConfig:
    """Configuration des features primaires."""
    source: str = "feature_generation_final_feature_selection_step"
    artifact_name: str = "tactician_features"
    initial_count: int = 300
    target_count: int = 100


@dataclass
class CrossTimeframeConfig:
    """Configuration des features cross-timeframe."""
    enable: bool = True
    base_timeframe: str = "15m"
    target_timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "30m", "1h", "4h", "1d"])
    feature_types: List[str] = field(default_factory=lambda: ["technical_indicators", "price_action", "volume_profile", "volatility"])
    optimized_lookback: bool = True


@dataclass
class RegimeFeaturesConfig:
    """Configuration des features de régimes."""
    enable: bool = True
    source: str = "regime_ml_models"
    feature_names: List[str] = field(default_factory=lambda: ["regime_prob_0", "regime_prob_1", "regime_prob_2", "regime_prob_3"])
    include_regime_outputs: bool = True


@dataclass
class AnalystEnsembleOutputsConfig:
    """Configuration des sorties analyst ensemble."""
    enable: bool = True
    source: str = "analyst_ensemble_models"
    features: List[str] = field(default_factory=lambda: [
        "analyst_ensemble_predictions",
        "analyst_ensemble_confidence",
        "analyst_ensemble_meta_learner_output"
    ])


@dataclass
class FeatureSelectionConfig:
    """Configuration de sélection de features."""
    method: str = "lasso"
    alpha: float = 0.005
    max_features: int = 100
    enable_recursive_elimination: bool = True
    enable_feature_importance: bool = True


@dataclass
class ScalingConfig:
    """Configuration de mise à l'échelle."""
    method: str = "robust"
    enable_outlier_handling: bool = True
    outlier_threshold: float = 2.5


@dataclass
class FeatureEngineeringConfig:
    """Configuration d'ingénierie des features."""
    primary_features: PrimaryFeaturesConfig = field(default_factory=PrimaryFeaturesConfig)
    cross_timeframe: CrossTimeframeConfig = field(default_factory=CrossTimeframeConfig)
    regime_features: RegimeFeaturesConfig = field(default_factory=RegimeFeaturesConfig)
    analyst_ensemble_outputs: AnalystEnsembleOutputsConfig = field(default_factory=AnalystEnsembleOutputsConfig)
    feature_selection: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)


@dataclass
class DataPreparationConfig:
    """Configuration de préparation des données."""
    time_series: Dict[str, Any] = field(default_factory=lambda: {
        'enable_temporal_features': True,
        'lookback_window': 100,
        'forecast_horizon': 1
    })
    target_generation: Dict[str, Any] = field(default_factory=lambda: {
        'method': 'momentum_based',
        'lookback_periods': [5, 10, 20],
        'momentum_threshold': 0.01,
        'enable_risk_adjustment': True,
        'risk_free_rate': 0.02
    })


@dataclass
class EvaluationConfig:
    """Configuration d'évaluation."""
    metrics: Dict[str, List[str]] = field(default_factory=lambda: {
        'regression': ["mse", "mae", "r2", "mape"],
        'classification': ["accuracy", "precision", "recall", "f1"],
        'trading': ["sharpe_ratio", "max_drawdown", "calmar_ratio", "sortino_ratio"]
    })
    cross_validation: Dict[str, Any] = field(default_factory=lambda: {
        'method': 'time_series_split',
        'n_splits': 5,
        'test_size': 0.2
    })
    comparison: Dict[str, Any] = field(default_factory=lambda: {
        'enable_model_ranking': True,
        'ranking_metric': 'sharpe_ratio',
        'enable_feature_importance': True,
        'enable_trading_metrics': True
    })


@dataclass
class RiskManagementConfig:
    """Configuration de gestion des risques."""
    enable_position_sizing: bool = True
    max_position_size: float = 0.1
    enable_stop_loss: bool = True
    enable_take_profit: bool = True
    enable_portfolio_limits: bool = True
    limits: Dict[str, float] = field(default_factory=lambda: {
        'max_drawdown': 0.05,
        'max_daily_loss': 0.02,
        'max_position_concentration': 0.3
    })
    monitoring: Dict[str, Any] = field(default_factory=lambda: {
        'enable_real_time_monitoring': True,
        'alert_thresholds': {
            'drawdown': 0.03,
            'daily_loss': 0.01,
            'position_size': 0.08
        }
    })


@dataclass
class HardwareConfig:
    """Configuration hardware."""
    enable_gpu_acceleration: bool = False
    enable_memory_optimization: bool = True
    enable_parallel_processing: bool = True
    memory_limit_gb: float = 2.0
    max_workers: Optional[int] = None


@dataclass
class LoggingConfig:
    """Configuration de logging."""
    level: str = "INFO"
    enable_detailed_logging: bool = True
    log_predictions: bool = True
    log_performance_metrics: bool = True
    log_feature_importance: bool = True
    log_risk_metrics: bool = True
    intervals: Dict[str, int] = field(default_factory=lambda: {
        'training_progress': 100,
        'prediction_summary': 1000,
        'performance_update': 5000,
        'risk_monitoring': 100
    })


@dataclass
class TacticianConfig:
    """Configuration principale du tactician."""
    model_name: str = "tactician_base"
    model_type: str = "separate_models"
    target: str = "entry_timing"
    base_timeframe: str = "15m"
    execution_timeframe: str = "15m"
    execution_frequency: str = "3m"
    price_change_target: float = 0.005
    base_models: List[BaseModelConfig] = field(default_factory=list)


@dataclass
class TacticianBaseTrainingConfig:
    """Configuration principale pour tactician_base_training."""
    version: str = "1.0.0"
    component_name: str = "tactician_base_training"
    description: str = "Configuration centralisée pour l'entraînement des modèles tactician de base"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Configuration principale
    tactician_config: TacticianConfig = field(default_factory=TacticianConfig)
    
    # Configuration des sections
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data_preparation: DataPreparationConfig = field(default_factory=DataPreparationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    risk_management: RiskManagementConfig = field(default_factory=RiskManagementConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Métadonnées
    _metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialisation post-construction."""
        if not self.tactician_config.base_models:
            self._setup_default_base_models()
        
        # Mettre à jour les métadonnées
        self._metadata.update({
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'component_name': self.component_name
        })
    
    def _setup_default_base_models(self):
        """Configurer les modèles de base par défaut."""
        # StandaloneGRU
        gru_model = BaseModelConfig(
            model_name="StandaloneGRU",
            class_name="src.models.standalone_gru_generator.StandaloneGRUGenerator",
            is_feature_generator=False,
            params={
                'sequence_length': 12,
                'num_features': 100,
                'hidden_units': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'learning_rate': 0.001,
                'training_params': {
                    'epochs': 40,
                    'batch_size': 256,
                    'verbose': 1,
                    'callbacks': [
                        {
                            'class': "tensorflow.keras.callbacks.EarlyStopping",
                            'params': {
                                'monitor': "val_loss",
                                'patience': 5,
                                'restore_best_weights': True
                            }
                        }
                    ]
                }
            },
            hpo={
                'enabled': True,
                'n_rounds': 2,
                'enable_final_refinement': True,
                'final_refinement_trials': 50,
                'search_space': {
                    'hidden_units': {'type': 'categorical', 'choices': [32, 64, 128, 256]},
                    'num_layers': {'type': 'int', 'low': 1, 'high': 4},
                    'dropout': {'type': 'float', 'low': 0.1, 'high': 0.5},
                    'learning_rate': {'type': 'float', 'low': 0.0001, 'high': 0.01, 'log': True},
                    'batch_size': {'type': 'categorical', 'choices': [64, 128, 256, 512]},
                    'sequence_length': {'type': 'int', 'low': 6, 'high': 24}
                },
                'optimal_params': {}
            }
        )
        
        # LGBM
        lgbm_model = BaseModelConfig(
            model_name="LGBM",
            class_name="lightgbm.LGBMRegressor",
            is_feature_generator=False,
            params={
                'n_estimators': 1000,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'objective': "regression",
                'metric': "rmse",
                'boosting_type': "gbdt",
                'n_jobs': -1,
                'verbose': -1,
                'training_params': {
                    'callbacks': ["lightgbm.early_stopping(100, verbose=False)"],
                    'eval_metric': "rmse"
                }
            },
            hpo={
                'enabled': True,
                'n_rounds': 2,
                'enable_final_refinement': True,
                'final_refinement_trials': 50,
                'search_space': {
                    'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.2, 'log': True},
                    'num_leaves': {'type': 'int', 'low': 20, 'high': 200},
                    'reg_alpha': {'type': 'float', 'low': 0.0, 'high': 5.0},
                    'reg_lambda': {'type': 'float', 'low': 0.0, 'high': 5.0},
                    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                    'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
                    'min_child_samples': {'type': 'int', 'low': 10, 'high': 100},
                    'min_gain_to_split': {'type': 'float', 'low': 0.1, 'high': 1.0}
                },
                'optimal_params': {}
            }
        )
        
        # CatBoost
        catboost_model = BaseModelConfig(
            model_name="CatBoost",
            class_name="catboost.CatBoostRegressor",
            is_feature_generator=False,
            params={
                'iterations': 500,
                'learning_rate': 0.1,
                'depth': 6,
                'l2_leaf_reg': 3.0,
                'subsample': 0.8,
                'colsample_bylevel': 0.8,
                'border_count': 128,
                'max_ctr_complexity': 2,
                'random_seed': 42,
                'verbose': False,
                'early_stopping_rounds': 50
            },
            hpo={
                'enabled': True,
                'n_rounds': 2,
                'enable_final_refinement': True,
                'final_refinement_trials': 50,
                'search_space': {
                    'iterations': {'type': 'int', 'low': 300, 'high': 1500},
                    'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                    'depth': {'type': 'int', 'low': 4, 'high': 10},
                    'l2_leaf_reg': {'type': 'float', 'low': 1.0, 'high': 10.0},
                    'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                    'colsample_bylevel': {'type': 'float', 'low': 0.6, 'high': 1.0},
                    'border_count': {'type': 'int', 'low': 64, 'high': 254},
                    'max_ctr_complexity': {'type': 'int', 'low': 1, 'high': 4}
                },
                'optimal_params': {}
            }
        )
        
        # ExtraTrees
        extratrees_model = BaseModelConfig(
            model_name="ExtraTrees",
            class_name="sklearn.ensemble.ExtraTreesClassifier",
            is_feature_generator=False,
            params={
                'n_estimators': 500,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': "sqrt",
                'bootstrap': True,
                'n_jobs': -1,
                'random_state': 42,
                'verbose': 0
            },
            hpo={
                'enabled': True,
                'n_rounds': 2,
                'enable_final_refinement': True,
                'final_refinement_trials': 50,
                'search_space': {
                    'n_estimators': {'type': 'int', 'low': 200, 'high': 1000},
                    'max_depth': {'type': 'int', 'low': 5, 'high': 20},
                    'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                    'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                    'max_features': {'type': 'categorical', 'choices': ["sqrt", "log2", 0.5, 0.7, 0.9]}
                },
                'optimal_params': {}
            }
        )
        
        # DepthwiseCNN
        cnn_model = BaseModelConfig(
            model_name="DepthwiseCNN",
            class_name="src.models.tcn_regressor.DepthwiseSeparableCNNRegressor",
            is_feature_generator=False,
            params={
                'filters': 64,
                'kernel_size': 3,
                'dropout': 0.2,
                'epochs': 50,
                'batch_size': 32,
                'learning_rate': 0.001,
                'validation_split': 0.2,
                'early_stopping_patience': 10,
                'reduce_lr_patience': 5,
                'use_batch_norm': False,
                'verbose': 0
            },
            hpo={
                'enabled': True,
                'n_rounds': 2,
                'enable_final_refinement': True,
                'final_refinement_trials': 50,
                'optimal_params': {}
            }
        )
        
        self.tactician_config.base_models = [
            gru_model, lgbm_model, catboost_model, extratrees_model, cnn_model
        ]
    
    def validate(self) -> bool:
        """Valider la configuration."""
        try:
            # Validation de base
            if not self.tactician_config.model_name:
                return False
            
            if not self.tactician_config.base_models:
                return False
            
            # Vérifier que tous les modèles ont les attributs requis
            for model in self.tactician_config.base_models:
                if not hasattr(model, 'model_name') or not hasattr(model, 'class_name'):
                    return False
            
            # Vérifier les configurations obligatoires
            if not self.feature_engineering.primary_features.artifact_name:
                return False
            
            return True
            
        except Exception:
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir en dictionnaire pour la sérialisation."""
        return {
            'version': self.version,
            'component_name': self.component_name,
            'description': self.description,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'tactician_config': {
                'model_name': self.tactician_config.model_name,
                'model_type': self.tactician_config.model_type,
                'target': self.tactician_config.target,
                'base_timeframe': self.tactician_config.base_timeframe,
                'execution_timeframe': self.tactician_config.execution_timeframe,
                'execution_frequency': self.tactician_config.execution_frequency,
                'price_change_target': self.tactician_config.price_change_target,
                'base_models': [
                    {
                        'model_name': model.model_name,
                        'class_name': model.class_name,
                        'is_feature_generator': model.is_feature_generator,
                        'params': model.params,
                        'hpo': model.hpo,
                        'optimal_params': model.optimal_params
                    }
                    for model in self.tactician_config.base_models
                ]
            },
            'feature_engineering': {
                'primary_features': {
                    'source': self.feature_engineering.primary_features.source,
                    'artifact_name': self.feature_engineering.primary_features.artifact_name,
                    'initial_count': self.feature_engineering.primary_features.initial_count,
                    'target_count': self.feature_engineering.primary_features.target_count
                },
                'cross_timeframe': {
                    'enable': self.feature_engineering.cross_timeframe.enable,
                    'base_timeframe': self.feature_engineering.cross_timeframe.base_timeframe,
                    'target_timeframes': self.feature_engineering.cross_timeframe.target_timeframes,
                    'feature_types': self.feature_engineering.cross_timeframe.feature_types,
                    'optimized_lookback': self.feature_engineering.cross_timeframe.optimized_lookback
                },
                'regime_features': {
                    'enable': self.feature_engineering.regime_features.enable,
                    'source': self.feature_engineering.regime_features.source,
                    'feature_names': self.feature_engineering.regime_features.feature_names,
                    'include_regime_outputs': self.feature_engineering.regime_features.include_regime_outputs
                },
                'analyst_ensemble_outputs': {
                    'enable': self.feature_engineering.analyst_ensemble_outputs.enable,
                    'source': self.feature_engineering.analyst_ensemble_outputs.source,
                    'features': self.feature_engineering.analyst_ensemble_outputs.features
                },
                'feature_selection': {
                    'method': self.feature_engineering.feature_selection.method,
                    'alpha': self.feature_engineering.feature_selection.alpha,
                    'max_features': self.feature_engineering.feature_selection.max_features,
                    'enable_recursive_elimination': self.feature_engineering.feature_selection.enable_recursive_elimination,
                    'enable_feature_importance': self.feature_engineering.feature_selection.enable_feature_importance
                },
                'scaling': {
                    'method': self.feature_engineering.scaling.method,
                    'enable_outlier_handling': self.feature_engineering.scaling.enable_outlier_handling,
                    'outlier_threshold': self.feature_engineering.scaling.outlier_threshold
                }
            },
            'training': {
                'enable_cross_validation': self.training.enable_cross_validation,
                'cv_folds': self.training.cv_folds,
                'enable_early_stopping': self.training.enable_early_stopping,
                'early_stopping_patience': self.training.early_stopping_patience,
                'validation_split': self.training.validation_split,
                'test_split': self.training.test_split,
                'training_samples': self.training.training_samples,
                'validation_samples': self.training.validation_samples,
                'test_samples': self.training.test_samples
            },
            'performance': {
                'expected_accuracy': self.performance.expected_accuracy,
                'expected_sharpe_ratio': self.performance.expected_sharpe_ratio,
                'training_time_limit': self.performance.training_time_limit,
                'memory_limit_mb': self.performance.memory_limit_mb
            },
            'output': {
                'save_models': self.output.save_models,
                'save_predictions': self.output.save_predictions,
                'generate_reports': self.output.generate_reports,
                'output_dir': self.output.output_dir
            },
            'hardware': {
                'enable_gpu_acceleration': self.hardware.enable_gpu_acceleration,
                'enable_memory_optimization': self.hardware.enable_memory_optimization,
                'enable_parallel_processing': self.hardware.enable_parallel_processing,
                'memory_limit_gb': self.hardware.memory_limit_gb,
                'max_workers': self.hardware.max_workers
            },
            'data_preparation': self.data_preparation,
            'evaluation': self.evaluation,
            'risk_management': self.risk_management,
            'logging': self.logging,
            '_metadata': self._metadata
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TacticianBaseTrainingConfig':
        """Créer une instance depuis un dictionnaire."""
        try:
            # Config tactician
            tactician_dict = config_dict.get('tactician_config', {})
            base_models = []
            
            for model_dict in tactician_dict.get('base_models', []):
                base_models.append(BaseModelConfig(**model_dict))
            
            tactician_config = TacticianConfig(
                model_name=tactician_dict.get('model_name', 'tactician_base'),
                model_type=tactician_dict.get('model_type', 'separate_models'),
                target=tactician_dict.get('target', 'entry_timing'),
                base_timeframe=tactician_dict.get('base_timeframe', '15m'),
                execution_timeframe=tactician_dict.get('execution_timeframe', '15m'),
                execution_frequency=tactician_dict.get('execution_frequency', '3m'),
                price_change_target=tactician_dict.get('price_change_target', 0.005),
                base_models=base_models
            )
            
            # Config feature engineering
            fe_dict = config_dict.get('feature_engineering', {})
            feature_engineering = FeatureEngineeringConfig(
                primary_features=PrimaryFeaturesConfig(**fe_dict.get('primary_features', {})),
                cross_timeframe=CrossTimeframeConfig(**fe_dict.get('cross_timeframe', {})),
                regime_features=RegimeFeaturesConfig(**fe_dict.get('regime_features', {})),
                analyst_ensemble_outputs=AnalystEnsembleOutputsConfig(**fe_dict.get('analyst_ensemble_outputs', {})),
                feature_selection=FeatureSelectionConfig(**fe_dict.get('feature_selection', {})),
                scaling=ScalingConfig(**fe_dict.get('scaling', {}))
            )
            
            # Config training
            training_dict = config_dict.get('training', {})
            training = TrainingConfig(**training_dict)
            
            # Autres configs
            performance = PerformanceConfig(**config_dict.get('performance', {}))
            output = OutputConfig(**config_dict.get('output', {}))
            hardware = HardwareConfig(**config_dict.get('hardware', {}))
            data_preparation = DataPreparationConfig(**config_dict.get('data_preparation', {}))
            evaluation = EvaluationConfig(**config_dict.get('evaluation', {}))
            risk_management = RiskManagementConfig(**config_dict.get('risk_management', {}))
            logging_config = LoggingConfig(**config_dict.get('logging', {}))
            
            # Créer l'instance
            return cls(
                version=config_dict.get('version', '1.0.0'),
                component_name=config_dict.get('component_name', 'tactician_base_training'),
                description=config_dict.get('description', ''),
                created_at=config_dict.get('created_at', datetime.now().isoformat()),
                updated_at=config_dict.get('updated_at', datetime.now().isoformat()),
                tactician_config=tactician_config,
                feature_engineering=feature_engineering,
                training=training,
                data_preparation=data_preparation,
                evaluation=evaluation,
                risk_management=risk_management,
                hardware=hardware,
                performance=performance,
                output=output,
                logging=logging_config,
                _metadata=config_dict.get('_metadata', {})
            )
            
        except Exception as e:
            raise ValueError(f"Erreur lors de la création de la configuration: {e}")
    
    def update_timestamp(self):
        """Mettre à jour le timestamp de modification."""
        self.updated_at = datetime.now().isoformat()
        if '_metadata' in self.__dict__:
            self._metadata['updated_at'] = self.updated_at


# Configuration factory function
def get_default_config() -> TacticianBaseTrainingConfig:
    """Obtenir la configuration par défaut."""
    return TacticianBaseTrainingConfig()


def create_custom_config(overrides: Dict[str, Any]) -> TacticianBaseTrainingConfig:
    """
    Créer une configuration personnalisée avec surcharge.
    
    Args:
        overrides: Paramètres à surcharger
        
    Returns:
        Configuration personnalisée
    """
    config = TacticianBaseTrainingConfig()
    
    # Appliquer les surcharges
    def update_nested_dict(base_dict: Dict, updates: Dict):
        for key, value in updates.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    # Convertir la config en dict pour faciliter les surcharges
    config_dict = config.to_dict()
    update_nested_dict(config_dict, overrides)
    
    # Recréer la config depuis le dict modifié
    return TacticianBaseTrainingConfig.from_dict(config_dict)


# Auto-test si exécuté directement
if __name__ == "__main__":
    print("🧪 Test de Configuration Tactician Base Training")
    print("=" * 60)
    
    # Test configuration par défaut
    config = get_default_config()
    print(f"✅ Configuration créée: {config.component_name}")
    print(f"   Version: {config.version}")
    print(f"   Description: {config.description}")
    print(f"   Modèles de base: {len(config.tactician_config.base_models)}")
    print(f"   Timeframe: {config.tactician_config.base_timeframe}")
    print(f"   Fréquence: {config.tactician_config.execution_frequency}")
    
    # Test validation
    is_valid = config.validate()
    print(f"   Validation: {'✅ Valide' if is_valid else '❌ Invalide'}")
    
    # Test conversion en dict
    config_dict = config.to_dict()
    print(f"   Conversion dict: ✅ ({len(config_dict)} sections)")
    
    # Test création configuration personnalisée
    custom_overrides = {
        'tactician_config': {
            'base_timeframe': '5m'
        },
        'performance': {
            'expected_accuracy': 0.85
        }
    }
    
    custom_config = create_custom_config(custom_overrides)
    print(f"✅ Configuration personnalisée créée:")
    print(f"   Timeframe modifié: {custom_config.tactician_config.base_timeframe}")
    print(f"   Précision personnalisée: {custom_config.performance.expected_accuracy}")
    
    print("\n🎯 Configuration prête pour l'intégration!")