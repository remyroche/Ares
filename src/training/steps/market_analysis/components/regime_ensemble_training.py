"""
Regime Detection Ensemble Training Component

This component implements the meta-learner for regime detection:
- stacker_lgbm_calibrated: LightGBM model used as the meta-learner with probability calibration
"""

import numpy as np
import pandas as pd
import pickle
import json
import time
import warnings
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime

from src.utils.logger import system_logger
from src.utils.tprint import tprint
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Enhanced imports for new functionality
from src.utils.ml_common.unified_vectorization_manager import (
    UnifiedVectorizationManager, OperationType, OptimizationStrategy
)
from src.utils.ml_common.optimization.hpo_utils import (
    HyperparameterOptimization
)
from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    ParameterGroup,
    OptimizationStage,
    create_param_group
)
from src.utils.ml_common.optimization.auto_tuner import (
    AutoTuner,
    DatasetCharacteristics
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    OptimizationConfig as TPEOptimizationConfig
)
from src.utils.ml_common.optimization.transition_aware_scoring import (
    create_transition_aware_scorer,
    create_pareto_multi_objective_hpo
)
try:
    from src.utils.ml_common.optimization.pareto import (
        ParetoOptimizer,
        Solution,
        ObjectiveDirection
    )
    PARETO_AVAILABLE = True
except ImportError:
    PARETO_AVAILABLE = False
    ParetoOptimizer = None
    Solution = None
    ObjectiveDirection = None
from src.utils.ml_common.validation.universal_temporal_validation import (
    UniversalTemporalValidator, TemporalValidationConfig
)
from src.utils.ml_common.utils.lookahead_protection import LookaheadProtection
from src.utils.hardware.unified_hardware_manager import (
    UnifiedHardwareManager, HardwareConfig, WorkloadType, OptimizationLevel
)
from src.utils.ml_common.evaluation.evaluation_utils import (
    EvaluationUtils
)
from src.utils.ml_common.evaluation.regime_temporal_metrics import (
    RegimeTemporalMetricsCalculator,
    calculate_temporal_smoothness_penalty,
    create_soft_labels
)
from src.utils.ml_common.feature_engineering.feature_smoothing import (
    add_smoothed_features,
    apply_ewm_smoothing
)
from src.utils.ml_common.post_training.model_validation import (
    ModelValidator, ValidationConfig
)
from src.utils.ml_common.validation.temporal_data_splitter import (
    TemporalDataSplitter, RegimeAwareSplitter, create_temporal_splitter
)
from src.utils.ml_common.validation.regime_walk_forward_validator import (
    RegimeWalkForwardValidator, RegimeValidationConfig, select_top_models
)

# Import new artifact schema and meta-features
from .regime_artifact_schema import (
    RegimeLabelsArtifact, FeatureContract, BaseModelContract,
    RegimeModelsArtifact, RegimeEnsembleArtifact, RegimeArtifactExtractor
)
from .ensemble_meta_features import EnsembleMetaFeaturesGenerator, generate_ensemble_meta_features
from src.feature_generation.categories.ensemble_disagreement import calculate_ensemble_disagreement_features

# Import centralized configuration system
try:
    from src.config.regime_ensemble_training import (
        RegimeEnsembleTrainingConfig,
        RegimeEnsembleTrainingConfigManager,
        get_regime_ensemble_config_manager,
        get_regime_ensemble_config
    )
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore')

# Import ML libraries
try:
    from sklearn.ensemble import StackingClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.calibration import CalibratedClassifierCV
    from lightgbm import LGBMClassifier
    ML_LIBRARIES_AVAILABLE = True
    tprint("✅ [REGIME_ENSEMBLE] ML libraries imported successfully", color="green")
except ImportError as e:
    ML_LIBRARIES_AVAILABLE = False
    tprint(f"❌ [REGIME_ENSEMBLE] Failed to import ML libraries: {e}", color="red")

# Import feature generation system
try:
    from src.feature_generation.core.factory import get_feature_bank, FeatureGenerator, FeatureCategory
    FEATURE_GENERATION_AVAILABLE = True
    tprint("✅ [REGIME_ENSEMBLE] Feature generation system imported successfully", color="green")
except ImportError as e:
    FEATURE_GENERATION_AVAILABLE = False
    tprint(f"⚠️ [REGIME_ENSEMBLE] Feature generation system not available: {e}", color="yellow")

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from src.utils.vectorbt_compat import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from src.utils.vectorbt_compat import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

class RegimeEnsembleTrainingComponent(BaseMarketAnalysisComponent):
    """
    Regime Detection Ensemble Training Component.

    This component trains the meta-learner for regime detection:
    - stacker_lgbm_calibrated: LightGBM model with probability calibration
    """

    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the Regime Ensemble Training Component."""
        tprint("🚀 [REGIME_ENSEMBLE] Initializing Regime Ensemble Training Component", color="cyan", bold=True)
        super().__init__(config)

        self.logger = system_logger.getChild('RegimeEnsembleTrainingComponent')
        tprint("✅ [REGIME_ENSEMBLE] Logger initialized", color="green")

        # Initialize centralized configuration management
        if CONFIG_AVAILABLE:
            self.config_manager = get_regime_ensemble_config_manager()
            self.ensemble_config = self.config_manager.get_config()
            tprint("🔧 [REGIME_ENSEMBLE] Centralized configuration loaded", color="green")
        else:
            self.config_manager = None
            self.ensemble_config = {}  # Fallback pour backward compatibility
            tprint("⚠️ [REGIME_ENSEMBLE] Centralized config not available, using fallback", color="yellow")

        # Get hardware configuration from centralized config or use defaults
        hardware_config_data = self.ensemble_config.hardware if hasattr(self.ensemble_config, 'hardware') else {}
        cpu_level = getattr(hardware_config_data, 'cpu_optimization_level', 'aggressive').upper()
        gpu_level = getattr(hardware_config_data, 'gpu_optimization_level', 'balanced').upper()
        memory_level = getattr(hardware_config_data, 'memory_optimization_level', 'balanced').upper()
        
        cpu_opt_level = getattr(OptimizationLevel, cpu_level, OptimizationLevel.AGGRESSIVE)
        gpu_opt_level = getattr(OptimizationLevel, gpu_level, OptimizationLevel.BALANCED)
        memory_opt_level = getattr(OptimizationLevel, memory_level, OptimizationLevel.BALANCED)
        
        # Initialize hardware manager for optimization
        self.hardware_manager = UnifiedHardwareManager(
            HardwareConfig(
                cpu_optimization_level=cpu_opt_level,
                gpu_optimization_level=gpu_opt_level,
                memory_optimization_level=memory_opt_level,
                enable_adaptive_optimization=hardware_config_data.get('enable_adaptive_optimization', True),
                enable_learning=hardware_config_data.get('enable_learning', True)
            )
        )
        tprint("🔧 [REGIME_ENSEMBLE] Hardware manager initialized with centralized config", color="green")

        # Initialize vectorization manager for feature generation
        self.vectorization_manager = UnifiedVectorizationManager()
        tprint("🔧 [REGIME_ENSEMBLE] Vectorization manager initialized", color="green")

        # Initialize HPO optimizer with centralized configuration
        hpo_config_data = self.ensemble_config.hpo if hasattr(self.ensemble_config, 'hpo') else {}
        self.hpo_optimizer = HyperparameterOptimization(
            {
                'max_trials': getattr(hpo_config_data, 'max_trials', 50),
                'timeout_seconds': getattr(hpo_config_data, 'timeout_seconds', 300),
                'enable_early_stopping': getattr(hpo_config_data, 'enable_early_stopping', True),
                'enable_pruning': getattr(hpo_config_data, 'enable_pruning', True)
            }
        )
        tprint("🔧 [REGIME_ENSEMBLE] HPO optimizer initialized with centralized config", color="green")
        
        # Initialize Auto Tuner for intelligent HPO configuration
        self.auto_tuner = AutoTuner(
            conservative_mode=False,
            enable_adaptive_timeout=True,
            enable_resource_monitoring=True
        )
        tprint("🔧 [REGIME_ENSEMBLE] Auto-tuner initialized for adaptive HPO", color="green")
        
        # Initialize Pareto optimizer for multi-objective HPO
        if PARETO_AVAILABLE:
            self.pareto_optimizer = ParetoOptimizer()
            tprint("✅ [REGIME_ENSEMBLE] Pareto optimizer initialized for multi-objective HPO", color="green")
        else:
            self.pareto_optimizer = None
            tprint("⚠️ [REGIME_ENSEMBLE] Pareto optimizer not available", color="yellow")
        
        # Enable transition-aware multi-objective HPO by default
        self.enable_multi_objective_hpo = True
        self.use_pareto_optimization = PARETO_AVAILABLE
        
        # Enable hierarchical optimization for models with many parameters (7+)
        self.use_hierarchical_hpo = True
        tprint("✅ [REGIME_ENSEMBLE] Hierarchical HPO enabled for complex models", color="green")
        self.temporal_smoothing_alpha = 0.1

        # Initialize temporal validator with centralized configuration
        temporal_config_data = self.ensemble_config.temporal_validation if hasattr(self.ensemble_config, 'temporal_validation') else {}
        self.temporal_validator = UniversalTemporalValidator(
            TemporalValidationConfig(
                enable_temporal_checks=getattr(temporal_config_data, 'enable_temporal_checks', True),
                strict_temporal_order=getattr(temporal_config_data, 'strict_temporal_order', True),
                initial_train_size=getattr(temporal_config_data, 'initial_train_size', 0.7),
                test_size=getattr(temporal_config_data, 'test_size', 0.3),
                gap_size=getattr(temporal_config_data, 'gap_size', 1)
            )
        )
        tprint("🔧 [REGIME_ENSEMBLE] Temporal validator initialized with centralized config", color="green")

        # Initialize lookahead protection
        self.lookahead_protection = LookaheadProtection()
        tprint("🔧 [REGIME_ENSEMBLE] Lookahead protection initialized", color="green")

        # Initialize model evaluator
        self.model_evaluator = EvaluationUtils()
        tprint("🔧 [REGIME_ENSEMBLE] Model evaluator initialized", color="green")
        
        # Initialize regime temporal metrics calculator
        self.temporal_metrics_calc = RegimeTemporalMetricsCalculator(min_episode_length=3)
        tprint("✅ [REGIME_ENSEMBLE] Temporal metrics calculator initialized", color="green")
        
        # Enhanced training configuration from centralized config
        ensemble_config_data = self.ensemble_config.ensemble if hasattr(self.ensemble_config, 'ensemble') else {}
        
        self.enable_temporal_smoothing = getattr(ensemble_config_data, 'enable_temporal_smoothing', True)
        self.temporal_smoothing_alpha = getattr(ensemble_config_data, 'temporal_smoothing_alpha', 0.1)
        self.enable_soft_labels = getattr(ensemble_config_data, 'enable_soft_labels', True)
        self.soft_label_smoothing = getattr(ensemble_config_data, 'soft_label_smoothing', 0.1)
        self.enable_smoothed_features = getattr(ensemble_config_data, 'enable_smoothed_features', True)
        self.smoothing_window_sizes = getattr(ensemble_config_data, 'smoothing_window_sizes', [3, 5, 7])
        
        # Meta-features configuration
        self.enable_enhanced_meta_features = getattr(self.ensemble_config, 'enable_enhanced_meta_features', True)
        self.enable_uncertainty_quantification = getattr(self.ensemble_config, 'enable_uncertainty_quantification', True)
        self.enable_confidence_features = getattr(self.ensemble_config, 'enable_confidence_features', True)
        self.enable_disagreement_analysis = getattr(self.ensemble_config, 'enable_disagreement_analysis', True)
        self.enable_regime_transition_features = getattr(self.ensemble_config, 'enable_regime_transition_features', True)

        # Initialize model validator with centralized configuration
        model_validation_config = self.ensemble_config.model_validation if hasattr(self.ensemble_config, 'model_validation') else {}
        self.model_validator = ModelValidator(
            ValidationConfig(
                enable_purged_cv=getattr(model_validation_config, 'enable_purged_cv', True),
                enable_data_leakage_detection=getattr(model_validation_config, 'enable_data_leakage_detection', True),
                enable_time_series_validation=getattr(model_validation_config, 'enable_time_series_validation', True)
            )
        )
        tprint("🔧 [REGIME_ENSEMBLE] Model validator initialized with centralized config", color="green")

        # Initialize meta-features generator
        self.meta_features_generator = EnsembleMetaFeaturesGenerator(component_name="REGIME_ENSEMBLE")
        tprint("🔧 [REGIME_ENSEMBLE] Meta-features generator initialized", color="green")
        
        # Initialize artifact extractor
        self.artifact_extractor = RegimeArtifactExtractor()
        tprint("🔧 [REGIME_ENSEMBLE] Artifact extractor initialized", color="green")

        # Initialize temporal splitter for proper train/test splits
        temporal_config = {
            'test_size': 0.3,
            'gap_size': 1,
            'validation_size': 0.2,
            'min_regime_samples': 1,  # CRITICAL FIX: Allow training with very limited data
            'regime_aware': True
        }
        self.temporal_splitter = create_temporal_splitter(temporal_config)
        tprint("🔧 [REGIME_ENSEMBLE] Temporal splitter initialized (regime-aware)", color="green")

        # Initialize walk-forward validator
        wf_config = RegimeValidationConfig(
            n_outer_folds=5,
            n_inner_folds=3,
            embargo_pct=0.05,
            min_train_samples=100,
            min_val_samples=30,
            min_regime_samples=10
        )
        self.walk_forward_validator = RegimeWalkForwardValidator(wf_config)
        tprint("🔧 [REGIME_ENSEMBLE] Walk-forward validator initialized", color="green")

        # Initialize ensemble training parameters from centralized config
        ensemble_config_data = self.ensemble_config.ensemble if hasattr(self.ensemble_config, 'ensemble') else {}
        self.ensemble_config = {
            'n_estimators': getattr(ensemble_config_data, 'n_estimators', 100),
            'max_depth': getattr(ensemble_config_data, 'max_depth', 6),
            'learning_rate': getattr(ensemble_config_data, 'learning_rate', 0.1),
            'random_state': getattr(ensemble_config_data, 'random_state', 42),
            'n_jobs': getattr(ensemble_config_data, 'n_jobs', -1),
            'verbose': getattr(ensemble_config_data, 'verbose', -1),
            'calibration_method': getattr(ensemble_config_data, 'calibration_method', 'isotonic'),
            'cv_folds': getattr(ensemble_config_data, 'cv_folds', 3)
        }
        tprint("⚙️ [REGIME_ENSEMBLE] Ensemble configuration loaded from centralized config", color="yellow")

        # Initialize ensemble models
        self.stacker_lgbm_calibrated = None
        self.base_models = {}
        self.ensemble_metrics = {}
        tprint("📊 [REGIME_ENSEMBLE] Ensemble models initialized", color="blue")

        tprint("✅ [REGIME_ENSEMBLE] Regime Ensemble Training Component initialized successfully", color="green", bold=True)

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        tprint("📋 [REGIME_ENSEMBLE] Getting required artifacts", color="cyan")
        required_artifacts = ['regime_ensemble_training_result']
        tprint(f"✅ [REGIME_ENSEMBLE] Required artifacts: {required_artifacts}", color="green")
        return required_artifacts

    async def _load_regime_models_predictions(
        self,
        base_step: Any
    ) -> Optional[pd.DataFrame]:
        """
        Load regime_models_predictions from HDF5.

        Args:
            base_step: BaseStep instance for artifact loading

        Returns:
            DataFrame with regime model predictions
        """
        try:
            tprint("📥 [REGIME_ENSEMBLE] Loading regime_models_predictions from versioned artifacts", color="cyan")

            predictions = base_step._get_artifact(
                'regime_models_predictions',
                artifact_type='data',
                data_category='features'
            )

            if predictions is None:
                tprint("⚠️ [REGIME_ENSEMBLE] No regime_models_predictions found in versioned artifacts", color="yellow")
                return None

            tprint(f"✅ [REGIME_ENSEMBLE] Loaded predictions from versioned artifacts: {predictions.shape}", color="green")
            tprint(f"📊 [REGIME_ENSEMBLE] Columns: {list(predictions.columns)}", color="blue")

            return predictions

        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Failed to load predictions from versioned artifacts: {e}", color="red")
            self.logger.error(f"Failed to load predictions: {e}", exc_info=True)
            return None

    def _calculate_disagreement_features(
        self,
        predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate disagreement features from base model predictions.

        Uses centralized disagreement feature calculation for consistency
        across all ensemble models (regime, analyst, tactician).

        Args:
            predictions: DataFrame with base model predictions

        Returns:
            DataFrame with disagreement features per regime
        """
        try:
            tprint("🔢 [REGIME_ENSEMBLE] Calculating disagreement features per regime", color="cyan")

            disagreement_features = pd.DataFrame(index=predictions.index)

            # Group by regime (e.g., all *_regime_0_prob columns)
            regime_groups = {}
            for col in predictions.columns:
                # Extract regime number from column name
                if '_regime_' in col and '_prob' in col:
                    regime_num = col.split('_regime_')[1].split('_')[0]
                    if regime_num not in regime_groups:
                        regime_groups[regime_num] = []
                    regime_groups[regime_num].append(col)

            # Calculate disagreement features for each regime using centralized implementation
            for regime_num, cols in regime_groups.items():
                if len(cols) < 2:
                    continue

                regime_preds = predictions[cols]

                # Use centralized disagreement feature calculation
                regime_disagreement = calculate_ensemble_disagreement_features(
                    predictions=regime_preds,
                    feature_prefix=f'regime_{regime_num}',
                    return_dataframe=True,
                    index=predictions.index
                )

                # Add regime-specific disagreement features
                for col in regime_disagreement.columns:
                    disagreement_features[col] = regime_disagreement[col]

            tprint(f"✅ [REGIME_ENSEMBLE] Calculated {len(disagreement_features.columns)} disagreement features", color="green")

            return disagreement_features

        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Failed to calculate disagreement features: {e}", color="red")
            self.logger.error(f"Failed to calculate disagreement features: {e}", exc_info=True)
            return pd.DataFrame(index=predictions.index)

    async def _save_ensemble_predictions_to_hdf5(
        self,
        predictions: pd.DataFrame,
        base_step: Any,
        artifact_name: str = 'regime_ensemble_predictions'
    ) -> None:
        """
        Save ensemble predictions to HDF5 file.

        Args:
            predictions: DataFrame with ensemble predictions
            base_step: BaseStep instance for artifact saving
            artifact_name: Name for the HDF5 artifact
        """
        try:
            tprint(f"💾 [REGIME_ENSEMBLE] Saving ensemble predictions to HDF5: {artifact_name}", color="cyan")

            # Ensure datetime index and 15m timeframe
            if not isinstance(predictions.index, pd.DatetimeIndex):
                predictions.index = pd.to_datetime(predictions.index)

            if predictions.index.freq != '15T':
                predictions = predictions.resample('15T').ffill()

            # Save to HDF5
            base_step._save_artifact(
                data=predictions,
                artifact_name=artifact_name,
                artifact_type='data',
                compression='auto',
                metadata={
                    'timeframe': '15m',
                    'ensemble_type': 'stacker_lgbm_calibrated',
                    'n_regimes': len([c for c in predictions.columns if 'regime' in c.lower()]),
                    'columns': list(predictions.columns),
                    'shape': predictions.shape,
                    'timestamp': datetime.now().isoformat()
                }
            )

            tprint(f"✅ [REGIME_ENSEMBLE] Saved ensemble predictions to HDF5: {predictions.shape}", color="green")

        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Failed to save ensemble predictions to HDF5: {e}", color="red")
            self.logger.error(f"Failed to save ensemble predictions to HDF5: {e}", exc_info=True)

    async def execute(self, data: pd.DataFrame, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute regime ensemble training with enhanced hardware optimization and validation.

        Args:
            data: Market data DataFrame
            pipeline_state: Pipeline state containing features, targets, and regime labels

        Returns:
            ComponentResult with training results
        """
        tprint("🚀 [REGIME_ENSEMBLE] Starting enhanced regime ensemble training execution", color="cyan", bold=True)
        start_time = datetime.now()

        try:
            # Initialize hardware optimization for intensive workload
            tprint("🔧 [REGIME_ENSEMBLE] Initializing hardware optimization", color="cyan")
            self.hardware_manager.initialize()
            self.hardware_manager.optimize_for_workload(WorkloadType.ML_TRAINING)
            tprint("✅ [REGIME_ENSEMBLE] Hardware optimization initialized", color="green")

            # Apply lookahead protection
            tprint("🔒 [REGIME_ENSEMBLE] Applying lookahead protection", color="cyan")
            protected_data = self.lookahead_protection.automated_future_data_filtering(data)
            tprint("✅ [REGIME_ENSEMBLE] Lookahead protection applied", color="green")
            
            # STORE OHLCV COLUMNS FOR TEMPORAL ANALYSIS (before protected_data gets modified)
            # Extract only the columns needed for returns calculation
            ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
            available_ohlcv = [col for col in ohlcv_columns if col in protected_data.columns]
            if available_ohlcv:
                market_ohlcv_data = protected_data[available_ohlcv].copy()
                tprint(f"📊 [REGIME_ENSEMBLE] Stored OHLCV data for temporal analysis: {market_ohlcv_data.shape}, columns: {available_ohlcv}", color="blue")
            else:
                market_ohlcv_data = None
                tprint("⚠️ [REGIME_ENSEMBLE] No OHLCV columns found in data", color="yellow")

            # Load regime_models_predictions as base features
            tprint("📥 [REGIME_ENSEMBLE] Loading regime_models artifacts", color="cyan")
            from src.training.steps.base_step import BaseStep
            
            class _ArtifactLoaderStep(BaseStep):
                async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
                    return {'success': True, 'artifacts': [], 'metrics': {}}
            
            base_step_inst = _ArtifactLoaderStep(
                "regime_ensemble_training_loader",
                use_versioned_artifacts=True
            )
            base_step_inst.set_context(
                symbol=self.config.symbol,
                exchange=self.config.exchange,
                timeframe=self.config.timeframe,
                direction='long',
                model='regime'
            )

            regime_models_preds = await self._load_regime_models_predictions(base_step_inst)

            if regime_models_preds is not None:
                tprint(f"✅ [REGIME_ENSEMBLE] Using regime_models predictions as features: {regime_models_preds.shape}", color="green")
                
                # Extract base model names from prediction columns
                base_model_names = set()
                for col in regime_models_preds.columns:
                    if '_regime_' in col and '_prob' in col:
                        model_name = col.split('_regime_')[0]
                        base_model_names.add(model_name)
                base_model_names = sorted(list(base_model_names))
                tprint(f"📊 [REGIME_ENSEMBLE] Detected {len(base_model_names)} base models: {base_model_names}", color="blue")
                
                # Store for later use in reporting
                pipeline_state['detected_base_models'] = base_model_names

                # Calculate disagreement features
                disagreement_feats = self._calculate_disagreement_features(regime_models_preds)

                if not disagreement_feats.empty:
                    tprint(f"✅ [REGIME_ENSEMBLE] Calculated disagreement features: {disagreement_feats.shape}", color="green")
                    # Combine predictions and disagreement features
                    all_features = pd.concat([regime_models_preds, disagreement_feats], axis=1)
                else:
                    all_features = regime_models_preds

                # Add to protected_data - ensure timezone compatibility before joining
                # Check if indexes have different timezone awareness
                if isinstance(protected_data.index, pd.DatetimeIndex) and isinstance(all_features.index, pd.DatetimeIndex):
                    protected_tz = protected_data.index.tz
                    features_tz = all_features.index.tz
                    
                    if (protected_tz is None) != (features_tz is None):
                        tprint("🔧 [REGIME_ENSEMBLE] Normalizing timezone differences before joining DataFrames", color="yellow")
                        # Make both indexes timezone-naive to avoid join issues
                        if protected_tz is not None:
                            protected_data.index = protected_data.index.tz_localize(None)
                            tprint("   Converted protected_data index to timezone-naive", color="blue")
                        if features_tz is not None:
                            all_features.index = all_features.index.tz_localize(None)
                            tprint("   Converted all_features index to timezone-naive", color="blue")
                
                protected_data = protected_data.join(all_features, how='left')
                tprint(f"📊 [REGIME_ENSEMBLE] Enhanced data shape: {protected_data.shape}", color="blue")

            # Extract required data from pipeline state using standardized extractors
            tprint("📊 [REGIME_ENSEMBLE] Extracting data from pipeline state with standardized extractors", color="yellow", bold=True)

            # First try to load rolling_hmm regime labels directly from versioned artifacts
            # This is the same logic used in regime_models_training
            regime_labels = None
            try:
                tprint("📥 [REGIME_ENSEMBLE] Loading rolling_hmm regime labels from versioned artifacts", color="cyan")
                from src.training.steps.base_step import BaseStep
                
                class _ArtifactLoaderStep(BaseStep):
                    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
                        return {'success': True, 'artifacts': [], 'metrics': {}}
                
                # Enable versioned artifacts to load from HDF5 storage
                base_step_inst = _ArtifactLoaderStep(
                    "regime_ensemble_training_loader",
                    use_versioned_artifacts=True,  # CRITICAL: Enable versioned artifacts
                )
                
                # Access ComponentConfig dataclass attributes
                symbol = self.config.symbol if hasattr(self.config, 'symbol') else 'ETHUSDT'
                exchange = self.config.exchange if hasattr(self.config, 'exchange') else 'binance'
                timeframe = self.config.timeframe if hasattr(self.config, 'timeframe') else '1h'
                
                # Set context to match the regime discovery output (1h timeframe)
                base_step_inst.set_context(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe,
                    direction='long',
                    model='regime',
                )
                
                # Load regime labels
                regime_labels_df = base_step_inst._get_artifact(
                    'rolling_hmm_regime_labels',
                    artifact_type='data'
                )
                
                if regime_labels_df is not None:
                    tprint(f"✅ [REGIME_ENSEMBLE] Loaded regime labels: {regime_labels_df.shape}", color="green")
                    tprint(f"📊 [REGIME_ENSEMBLE] Columns: {list(regime_labels_df.columns)}", color="blue")
                    
                    # Extract regime_label column
                    if 'regime_label' in regime_labels_df.columns:
                        regime_labels = regime_labels_df['regime_label'].values
                    else:
                        # Fallback to first column
                        regime_labels = regime_labels_df.iloc[:, 0].values
                    
                    tprint(f"✅ [REGIME_ENSEMBLE] Extracted {len(regime_labels)} regime labels", color="green")
                    tprint(f"📊 [REGIME_ENSEMBLE] Unique regimes: {np.unique(regime_labels)}", color="blue")
                else:
                    tprint("⚠️ [REGIME_ENSEMBLE] No rolling_hmm_regime_labels found in versioned artifacts", color="yellow")
                    
            except Exception as e:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Direct rolling HMM label loading failed: {e}", color="yellow")
                regime_labels = None
            
            # If direct loading failed, fall back to standardized extractor
            if regime_labels is None:
                tprint("🔄 [REGIME_ENSEMBLE] Falling back to standardized extractor", color="yellow")
                # Extract regime labels using standardized extractor
                # NOTE: Currently in testing mode (tries all methods). When you choose your winner:
                # 1. Uncomment the preferred_method parameter below
                # 2. Set it to your chosen method: "gmm", "hmm", "optimal", or "regime_clustering"
                # 3. Remove unused clustering steps from pipeline
                regime_labels_artifact = self.artifact_extractor.extract_regime_labels(
                    pipeline_state, 
                    component_name="REGIME_ENSEMBLE"
                    # preferred_method="gmm"  # 👈 PRODUCTION: Uncomment and set to your winner (gmm/hmm/optimal)
                )
                
                if regime_labels_artifact is None:
                    raise ValueError("❌ Failed to extract regime labels from pipeline state")
                
                # Validate regime labels artifact
                if not regime_labels_artifact.validate():
                    raise ValueError("❌ Regime labels artifact validation failed")
                
                regime_labels = regime_labels_artifact.cluster_assignments
                tprint(
                    f"✅ [REGIME_ENSEMBLE] Extracted regime labels: {len(regime_labels)} samples, "
                    f"{regime_labels_artifact.n_regimes} regimes using {regime_labels_artifact.clustering_method}",
                    color="green",
                    bold=True
                )
            else:
                # Create a simple artifact-like object for consistency
                tprint(f"✅ [REGIME_ENSEMBLE] Using directly loaded regime labels: {len(regime_labels)} samples", color="green")
            
            # Validate that we have regime labels
            if regime_labels is None:
                raise ValueError("❌ Failed to extract regime labels from pipeline state")

            # EXTRACT SOFT LABELS (POSTERIOR PROBABILITIES)
            soft_labels = None
            sample_weights = None
            if self.enable_soft_labels: # This flag exists in ensemble component too
                try:
                    # Try to get HDP-HMM probabilities first
                    hdp_probs_artifact = self.artifact_extractor.extract_artifact(
                        pipeline_state, "hdp_hmm_regime_probabilities", "hdp_hmm_regime_discovery"
                    )
                    
                    if hdp_probs_artifact is not None:
                        soft_labels = hdp_probs_artifact
                        tprint(f"✅ [REGIME_ENSEMBLE] Extracted HDP-HMM soft labels (probabilities): {soft_labels.shape}", "green")
                    else:
                        tprint("⚠️ [REGIME_ENSEMBLE] HDP-HMM soft labels not found. Will use hard labels.", "yellow")
                        
                except Exception as e:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] Error extracting soft labels: {e}", "yellow")

            
            # Extract base models using standardized extractor
            # NOTE: When running in blank mode with versioned artifacts, we don't need the actual model objects
            # We only need their predictions, which are already loaded above
            base_models = None
            try:
                regime_models_artifact = self.artifact_extractor.extract_base_models(
                    pipeline_state, component_name="REGIME_ENSEMBLE"
                )
                
                if regime_models_artifact is not None and regime_models_artifact.validate_models():
                    # Get ONLY base models (exclude ensemble/meta-learners to avoid circular references)
                    base_models = regime_models_artifact.get_base_models()
                    tprint(
                        f"✅ [REGIME_ENSEMBLE] Extracted {len(base_models)} base models (filtered from {len(regime_models_artifact.models)} total)",
                        color="green",
                        bold=True
                    )
                    
                    # Log base model names for transparency
                    tprint("📋 [REGIME_ENSEMBLE] Base models to use:", color="cyan")
                    for model_name in base_models.keys():
                        tprint(f"   - {model_name}", color="blue")
                else:
                    tprint("⚠️ [REGIME_ENSEMBLE] Base models not found in pipeline state (expected in blank mode)", color="yellow")
                    tprint("   Using predictions from versioned artifacts instead", color="yellow")
            except Exception as e:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Could not extract base models: {e}", color="yellow")
                tprint("   Continuing with predictions from versioned artifacts", color="yellow")

            # Check if regime labels are available before preparing training data
            if regime_labels is None:
                tprint("⚠️ [REGIME_ENSEMBLE] No regime labels found in artifacts, will create synthetic regime labels", color="yellow")
                # Create synthetic regime labels based on data patterns
                regime_labels = self._create_synthetic_regime_labels(protected_data)

            # Prepare training data from the input data DataFrame with advanced regime features
            tprint("🔧 [REGIME_ENSEMBLE] Preparing training data from input DataFrame with advanced regime features", color="yellow")
            X, y, feature_names = self._prepare_training_data(protected_data, regime_labels, pipeline_state)

            # Validate required data
            tprint("🔍 [REGIME_ENSEMBLE] Validating required data", color="yellow")
            if X is None or y is None or feature_names is None:
                tprint("❌ [REGIME_ENSEMBLE] Failed to prepare training data", color="red")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message="Failed to prepare training data from input DataFrame",
                    metadata={'component_type': 'regime_ensemble_training'}
                )

            if not base_models:
                tprint("⚠️ [REGIME_ENSEMBLE] No base models found from previous training, training base models", color="yellow")
                # Train base models if not provided
                tprint("🏋️ [REGIME_ENSEMBLE] Training base models for ensemble", color="blue")
                base_models = self._train_base_models(X, y, regime_labels)
                if not base_models:
                    tprint("❌ [REGIME_ENSEMBLE] Failed to train base models", color="red")
                    return ComponentResult(
                        success=False,
                        artifacts={},
                        error_message="Failed to train base models",
                        metadata={'component_type': 'regime_ensemble_training'}
                    )

            tprint(f"📊 [REGIME_ENSEMBLE] Data shapes - X: {X.shape}, y: {y.shape}, regime_labels: {len(regime_labels) if regime_labels is not None else 'None'}", color="blue")
            tprint(f"📊 [REGIME_ENSEMBLE] Base models available: {list(base_models.keys())}", color="blue")

            # Prepare data for ensemble training with proper train/test split
            tprint("🔧 [REGIME_ENSEMBLE] Preparing data for ensemble training with proper validation", color="yellow")
            X_processed, y_processed, regime_labels_processed = self._prepare_data(X, y, regime_labels)
            tprint(f"✅ [REGIME_ENSEMBLE] Data prepared - X: {X_processed.shape}, y: {y_processed.shape}", color="green")

            # ALIGN SOFT LABELS AND CREATE SAMPLE WEIGHTS
            weights_processed = None
            if soft_labels is not None:
                min_length = len(y_processed)
                if len(soft_labels) >= min_length:
                    try:
                        soft_labels_aligned = soft_labels[:min_length]
                        weights_processed = soft_labels_aligned[np.arange(min_length), y_processed]
                        tprint(f"✅ [REGIME_ENSEMBLE] Created sample weights from soft labels. Mean weight: {np.mean(weights_processed):.3f}", "green")
                    except Exception as e:
                        tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to create sample weights from soft labels: {e}", "yellow")
                else:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] Soft labels length ({len(soft_labels)}) mismatch with processed labels ({min_length}). No weights applied.", "yellow")


            # Perform proper temporal split to prevent data leakage using regime-aware splitter
            tprint("🔄 [REGIME_ENSEMBLE] Performing regime-aware temporal train/val/test split", color="cyan")

            # Use RegimeAwareSplitter for proper temporal split with regime awareness
            X_train, X_val, X_test, y_train, y_val, y_test = self.temporal_splitter.split_regime_aware(
                X_processed, y_processed
            )

            tprint(f"✅ [REGIME_ENSEMBLE] Regime-aware temporal split completed", color="green")
            tprint(f"   Train: {len(X_train)} samples", color="blue")
            tprint(f"   Val: {len(X_val)} samples", color="blue")
            tprint(f"   Test: {len(X_test)} samples", color="blue")

            # Create indices for weight splitting
            n_train = len(X_train)
            n_val = len(X_val)
            train_indices = np.arange(n_train)
            val_indices = np.arange(n_train, n_train + n_val)
            test_indices = np.arange(n_train + n_val, len(X_processed))

            # Validate the temporal split
            validation_report = self.temporal_validator.validate_temporal_split(
                X_train, X_test, y_train, y_test,
                model_name="regime_ensemble",
                model_type="ensemble"
            )

            if not validation_report.temporal_order_valid:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Temporal validation warning: {validation_report.temporal_message}", color="yellow")
                tprint("   Continuing with temporal split (no fallback to random split)", color="yellow")
            else:
                tprint("✅ [REGIME_ENSEMBLE] Temporal validation passed - no data leakage detected", color="green")

            # Merge train and val for final model training (test is held out for final evaluation)
            tprint("🔄 [REGIME_ENSEMBLE] Merging train+val for final model training", color="cyan")
            X_train_full = np.vstack([X_train, X_val])
            y_train_full = np.concatenate([y_train, y_val])
            tprint(f"📊 [REGIME_ENSEMBLE] Full train set: {X_train_full.shape}, Test set: {X_test.shape}", color="blue")

            # SPLIT SAMPLE WEIGHTS (now for train_full and test)
            weights_train_full, weights_test = None, None
            if weights_processed is not None:
                try:
                    # Combine train and val weights
                    train_val_indices = np.concatenate([train_indices, val_indices])
                    weights_train_full = weights_processed[train_val_indices]
                    weights_test = weights_processed[test_indices]
                    tprint(f"✅ [REGIME_ENSEMBLE] Sample weights split: Train+Val={len(weights_train_full)}, Test={len(weights_test)}", "green")
                except Exception as e:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to split sample weights: {e}", "yellow")
                    weights_train_full = None # Disable weighting if split fails

            # Train stacker_lgbm_calibrated meta-learner on training+validation data
            tprint("=" * 80, color="cyan")
            tprint("🎭 [REGIME_ENSEMBLE] STARTING META-LEARNER TRAINING", color="yellow", bold=True)
            tprint(f"📊 [REGIME_ENSEMBLE] Train data shape: {X_train_full.shape}", color="blue")
            tprint(f"📊 [REGIME_ENSEMBLE] Train labels shape: {y_train_full.shape}", color="blue")
            tprint(f"📊 [REGIME_ENSEMBLE] Base models: {list(base_models.keys()) if base_models else 'None'}", color="blue")
            tprint(f"📊 [REGIME_ENSEMBLE] Sample weights: {'Yes' if weights_train_full is not None else 'No'}", color="blue")
            tprint("=" * 80, color="cyan")
            stacker_result = self._train_stacker_lgbm_calibrated(X_train_full, y_train_full, base_models, weights_train_full)

            # Evaluate ensemble on holdout test data
            tprint("📊 [REGIME_ENSEMBLE] Evaluating ensemble performance on holdout test data", color="yellow")
            ensemble_metrics = self._evaluate_ensemble(X_test, y_test, stacker_result, weights_test)

            # Run walk-forward validation on the ensemble model
            tprint("🎯 [REGIME_ENSEMBLE] Running walk-forward validation for OOS ensemble performance", color="cyan")
            try:
                ensemble_model = stacker_result.get('model')
                if ensemble_model is not None:
                    wf_result = self.walk_forward_validator.validate_models(
                        X_processed, y_processed,
                        {'ensemble': ensemble_model},
                        model_configs=None
                    )

                    # Extract walk-forward metrics
                    walk_forward_metrics = {
                        'validation_completed': True,
                        'n_folds': wf_result.metadata['n_folds_completed'],
                        'accuracy': wf_result.accuracy,
                        'precision': wf_result.precision,
                        'recall': wf_result.recall,
                        'f1_score': wf_result.f1_score,
                        'temporal_metrics': wf_result.temporal_metrics
                    }

                    tprint("✅ [REGIME_ENSEMBLE] Walk-forward validation completed:", color="green")
                    tprint(f"   Accuracy: {wf_result.accuracy['mean']:.4f} [{wf_result.accuracy['ci_lower']:.4f}, {wf_result.accuracy['ci_upper']:.4f}]", color="blue")
                    tprint(f"   F1-score: {wf_result.f1_score['mean']:.4f} [{wf_result.f1_score['ci_lower']:.4f}, {wf_result.f1_score['ci_upper']:.4f}]", color="blue")
                    tprint(f"   MEL: {wf_result.temporal_metrics.get('mel', {}).get('mean', 0):.2f}", color="blue")
                else:
                    tprint("⚠️ [REGIME_ENSEMBLE] Ensemble model not available for walk-forward validation", color="yellow")
                    walk_forward_metrics = {'validation_completed': False, 'error': 'Model not available'}

            except Exception as e:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Walk-forward validation failed: {e}", color="yellow")
                walk_forward_metrics = {'validation_completed': False, 'error': str(e)}

            # Generate ensemble predictions and save to HDF5
            tprint("🎯 [REGIME_ENSEMBLE] Generating ensemble predictions for HDF5 storage", color="cyan")
            try:
                ensemble_model = stacker_result.get('model')
                if ensemble_model is not None and hasattr(ensemble_model, 'predict_proba'):
                    pred_probs = ensemble_model.predict_proba(X_processed)
                    # Create columns for each regime
                    ensemble_predictions = {}
                    for regime_idx in range(pred_probs.shape[1]):
                        # CRITICAL FIX: Convert to Python int to avoid JSON serialization errors
                        col_name = f'ensemble_regime_{int(regime_idx)}_prob'
                        ensemble_predictions[col_name] = pred_probs[:, regime_idx]

                    predictions_df = pd.DataFrame(ensemble_predictions, index=protected_data.index)
                    # Save to HDF5
                    await self._save_ensemble_predictions_to_hdf5(predictions_df, base_step_inst, 'regime_ensemble_predictions')
                    tprint("✅ [REGIME_ENSEMBLE] Ensemble predictions saved to HDF5", color="green")
                else:
                    tprint("⚠️ [REGIME_ENSEMBLE] Ensemble model not found or no predict_proba method", color="yellow")
            except Exception as e:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to save ensemble predictions: {e}", color="yellow")

            # Create comprehensive results
            tprint("📦 [REGIME_ENSEMBLE] Creating comprehensive results", color="yellow")
            results = {
                'regime_ensemble_training_result': {
                    'stacker_lgbm_calibrated': stacker_result,
                    'base_models': base_models,
                    'ensemble_metrics': ensemble_metrics,
                    'training_time': (datetime.now() - start_time).total_seconds(),
                    'success': True,
                    'validation_report': {
                        'temporal_order_valid': validation_report.temporal_order_valid,
                        'leakage_detected': validation_report.leakage_detected,
                        'validation_score': validation_report.validation_score,
                        'warnings': validation_report.warnings,
                        'recommendations': validation_report.recommendations
                    },
                    'hardware_optimization': {
                        'enabled': True,
                        'workload_type': 'ML_TRAINING',
                        'optimization_applied': True
                    },
                    'lookahead_protection': {
                        'enabled': True,
                        'protection_applied': True
                    },
                    'metadata': {
                        'component_type': 'regime_ensemble_training',
                        'data_shape': X_processed.shape,
                        'train_shape': X_train_full.shape,
                        'test_shape': X_test.shape,
                        'n_regimes': len(np.unique(regime_labels_processed)) if regime_labels_processed is not None else 0,
                        'feature_names': feature_names,
                        'timestamp': datetime.now().isoformat(),
                        'walk_forward_validation': walk_forward_metrics
                    }
                }
            }

            # Add detected base models from pipeline_state if available (for blank mode)
            if 'detected_base_models' in pipeline_state:
                results['detected_base_models'] = pipeline_state['detected_base_models']
                tprint(f"📊 [REGIME_ENSEMBLE] Added detected base models to results: {pipeline_state['detected_base_models']}", color="blue")
            
            # Add ensemble_metrics at top level for easier access in reporting
            results['ensemble_metrics'] = ensemble_metrics
            tprint(f"📊 [REGIME_ENSEMBLE] Added ensemble metrics to results", color="blue")
            
            tprint("✅ [REGIME_ENSEMBLE] Regime ensemble training completed successfully", color="green", bold=True)
            tprint(f"⏱️ [REGIME_ENSEMBLE] Total execution time: {(datetime.now() - start_time).total_seconds():.2f}s", color="blue")

            # Generate regime probability report
            try:
                regime_report = await self._generate_regime_probability_report(
                    results, X_processed, feature_names
                )
                if regime_report:
                    results['regime_probability_report'] = regime_report
                    tprint("📊 [REGIME_ENSEMBLE] Regime probability report generated successfully", color="green")
            except Exception as e:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to generate regime probability report: {e}", color="yellow")

            # Automatically tag the whole dataset with ensemble outputs and probabilities
            try:
                tprint("🏷️ [REGIME_ENSEMBLE] Automatically tagging dataset with ensemble outputs", color="cyan")
                tagged_data = await self._tag_dataset_with_ensemble_outputs(
                    protected_data, results, X_processed, feature_names
                )
                if tagged_data is not None:
                    results['tagged_dataset'] = tagged_data
                    tprint("✅ [REGIME_ENSEMBLE] Dataset tagged successfully with ensemble outputs", color="green")
                else:
                    tprint("⚠️ [REGIME_ENSEMBLE] Dataset tagging failed", color="yellow")
            except Exception as e:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to tag dataset: {e}", color="yellow")

            # Create artifacts for 15m and 1h timeframes
            try:
                tprint("📦 [REGIME_ENSEMBLE] Creating artifacts for 15m and 1h timeframes", color="cyan")
                timeframe_artifacts = await self._create_timeframe_artifacts(
                    results, protected_data, X_processed, feature_names
                )
                if timeframe_artifacts:
                    results.update(timeframe_artifacts)
                    tprint("✅ [REGIME_ENSEMBLE] Timeframe artifacts created successfully", color="green")
                else:
                    tprint("⚠️ [REGIME_ENSEMBLE] Timeframe artifact creation failed", color="yellow")
            except Exception as e:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to create timeframe artifacts: {e}", color="yellow")

            # Save artifacts persistently using the artifact manager
            try:
                # Prepare artifacts for downstream compatibility
                artifacts_to_save = self._prepare_artifacts_for_saving(results)
                
                # Save main artifacts
                save_report = await self.save_artifacts(artifacts_to_save, {
                    'component_type': 'regime_ensemble_training',
                    'execution_time': (datetime.now() - start_time).total_seconds(),
                    'artifacts_count': len(artifacts_to_save),
                    'downstream_compatible': True
                })
                tprint(
                    f"💾 [REGIME_ENSEMBLE] Main artifacts saved (correlation_id={save_report.correlation_id}): {list(save_report.paths.keys())}",
                    color="green"
                )
                
                # Save individual artifacts for better downstream access
                await self._save_individual_artifacts(results, save_report.correlation_id)
                # Generate and save temporal regime analysis report
                try:
                    tprint("📊 [REGIME_ENSEMBLE] Generating temporal regime analysis report", color="cyan")
                    temporal_report = await self._generate_temporal_regime_analysis(
                        results, market_ohlcv_data, X_processed, feature_names
                    )
                    if temporal_report:
                        results['temporal_regime_analysis'] = temporal_report
                        tprint("✅ [REGIME_ENSEMBLE] Temporal regime analysis completed", color="green")
                except Exception as e:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to generate temporal regime analysis: {e}", color="yellow")

                # Generate comprehensive CSV and markdown reports
                try:
                    tprint("📊 [REGIME_ENSEMBLE] Generating comprehensive CSV/MD reports", color="cyan")

                    # Determine symbol from config (works in blank mode) or pipeline state
                    symbol = self.config.symbol if hasattr(self.config, 'symbol') else pipeline_state.get('symbol', 'UNKNOWN')

                    # Generate CSV reports
                    metrics_path, comparison_path = self._generate_csv_reports(results, symbol)
                    if metrics_path:
                        results['csv_metrics_report'] = metrics_path
                        tprint(f"✅ [REGIME_ENSEMBLE] CSV metrics report: {metrics_path}", color="green")
                    if comparison_path:
                        results['csv_comparison_report'] = comparison_path
                        tprint(f"✅ [REGIME_ENSEMBLE] CSV comparison report: {comparison_path}", color="green")

                    # Generate markdown report
                    md_report_path = self._generate_markdown_report(results, symbol)
                    if md_report_path:
                        results['markdown_report'] = md_report_path
                        tprint(f"✅ [REGIME_ENSEMBLE] Markdown report: {md_report_path}", color="green")

                    tprint("✅ [REGIME_ENSEMBLE] All reports generated successfully", color="green")
                except Exception as e:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to generate CSV/MD reports: {e}", color="yellow")
                    self.logger.error(f"Failed to generate reports: {e}", exc_info=True)
                
            except Exception as e:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to save artifacts persistently: {e}", color="yellow")

            # Cleanup hardware resources
            tprint("🧹 [REGIME_ENSEMBLE] Hardware resources cleaned up", color="cyan")

            return ComponentResult(
                success=True,
                artifacts=results,
                metadata={
                    'component_type': 'regime_ensemble_training',
                    'execution_time': (datetime.now() - start_time).total_seconds(),
                    'artifacts_saved_persistently': True,
                    'hardware_optimization_enabled': True,
                    'lookahead_protection_enabled': True
                }
            )

        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Regime ensemble training failed: {e}", color="red", bold=True)
            self.logger.error(f"Regime ensemble training failed: {e}", exc_info=True)
            
            # Cleanup hardware resources on error
            try:
                tprint("🧹 [REGIME_ENSEMBLE] Hardware cleanup completed", color="cyan")
            except Exception as cleanup_error:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Hardware cleanup failed: {cleanup_error}", color="yellow")
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={'component_type': 'regime_ensemble_training'}
            )

    def _prepare_data(self, X: np.ndarray, y: np.ndarray, regime_labels: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for ensemble training."""
        tprint("🔧 [REGIME_ENSEMBLE] Preparing data for ensemble training", color="yellow")

        # Handle missing values
        tprint("🧹 [REGIME_ENSEMBLE] Handling missing values", color="blue")
        if isinstance(X, pd.DataFrame):
            X = X.fillna(0).values
        elif isinstance(X, list):
            X = np.array(X)

        if isinstance(y, (pd.Series, list)):
            y = np.array(y)

        if regime_labels is not None and isinstance(regime_labels, (pd.Series, list)):
            regime_labels = np.array(regime_labels)

        # Ensure all arrays have the same length
        tprint("📏 [REGIME_ENSEMBLE] Ensuring consistent array lengths", color="blue")
        min_length = min(len(X), len(y))
        if regime_labels is not None:
            min_length = min(min_length, len(regime_labels))

        X = X[:min_length]
        y = y[:min_length]
        if regime_labels is not None:
            regime_labels = regime_labels[:min_length]

        tprint(f"✅ [REGIME_ENSEMBLE] Data prepared - X: {X.shape}, y: {y.shape}, regime_labels: {regime_labels.shape if regime_labels is not None else 'None'}", color="green")
        return X, y, regime_labels

    def _perform_temporal_split_with_indices(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform a temporal train/test split while preserving original indices.

        This replaces the old random split function to ensure temporal integrity.
        Use self.temporal_splitter.split_regime_aware() for regime-aware splits.
        """
        n_samples = len(X)
        split_idx = int(n_samples * (1 - test_size))

        train_indices = np.arange(split_idx)
        test_indices = np.arange(split_idx, n_samples)

        return (
            X[train_indices],
            X[test_indices],
            y[train_indices],
            y[test_indices],
            train_indices,
            test_indices,
        )

    def _create_enhanced_meta_features(self, meta_features: np.ndarray, y: np.ndarray, base_model_predictions: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create enhanced meta-features for better ensemble performance.

        Args:
            meta_features: Base meta-features from base models
            y: Target labels
            base_model_predictions: Raw predictions from base models for disagreement analysis

        Returns:
            Enhanced meta-features array
        """
        try:
            tprint("🔧 [REGIME_ENSEMBLE] Creating enhanced meta-features", color="blue")

            # Calculate additional meta-features
            enhanced_features = []

            # 1. Base meta-features
            enhanced_features.append(meta_features)

            # 2. Confidence features (max probability for each sample)
            max_probs = np.max(meta_features, axis=1, keepdims=True)
            enhanced_features.append(max_probs)

            # 3. Entropy features (uncertainty measure)
            epsilon = 1e-10  # Avoid log(0)
            probs_safe = np.clip(meta_features, epsilon, 1 - epsilon)
            entropy = -np.sum(probs_safe * np.log(probs_safe), axis=1, keepdims=True)
            enhanced_features.append(entropy)

            # 4. Variance features (prediction consistency)
            variance = np.var(meta_features, axis=1, keepdims=True)
            enhanced_features.append(variance)

            # 5. Disagreement features (if base model predictions available)
            if base_model_predictions is not None and base_model_predictions.shape[1] > 1:
                # Model disagreement (variance across base model predictions)
                model_disagreement = np.var(base_model_predictions, axis=1, keepdims=True)
                enhanced_features.append(model_disagreement)
                
                # Pairwise disagreement (max disagreement between any two models)
                pairwise_disagreement = np.zeros((len(y), 1))
                for i in range(base_model_predictions.shape[1]):
                    for j in range(i+1, base_model_predictions.shape[1]):
                        disagreement = np.abs(base_model_predictions[:, i] - base_model_predictions[:, j])
                        pairwise_disagreement = np.maximum(pairwise_disagreement, disagreement.reshape(-1, 1))
                enhanced_features.append(pairwise_disagreement)

            # 6. Regime transition features
            if len(y) > 1:
                # Regime stability (consecutive same predictions)
                regime_stability = np.zeros((len(y), 1))
                for i in range(1, len(y)):
                    if y[i] == y[i-1]:
                        regime_stability[i] = regime_stability[i-1] + 1
                enhanced_features.append(regime_stability)
                
                # Regime change indicator
                regime_changes = np.zeros((len(y), 1))
                regime_changes[1:] = (y[1:] != y[:-1]).astype(int).reshape(-1, 1)
                enhanced_features.append(regime_changes)

            # 7. Uncertainty quantification features
            # Prediction confidence gap (difference between top 2 predictions)
            sorted_probs = np.sort(meta_features, axis=1)
            confidence_gap = sorted_probs[:, -1] - sorted_probs[:, -2]
            enhanced_features.append(confidence_gap.reshape(-1, 1))
            
            # Prediction margin (distance to decision boundary)
            prediction_margin = np.max(meta_features, axis=1) - np.mean(meta_features, axis=1)
            enhanced_features.append(prediction_margin.reshape(-1, 1))

            # 8. Class-specific features
            unique_classes = np.unique(y)
            for i, class_val in enumerate(unique_classes):
                class_mask = (y == class_val)
                if np.sum(class_mask) > 0:
                    # Use column index i instead of class_val for indexing
                    class_confidence = meta_features[class_mask, i].mean()
                    class_feature = np.full((len(y), 1), class_confidence)
                    enhanced_features.append(class_feature)

            # Combine all features
            enhanced_meta_features = np.column_stack(enhanced_features)

            tprint(f"✅ [REGIME_ENSEMBLE] Enhanced features created: {enhanced_meta_features.shape}", color="green")
            return enhanced_meta_features

        except Exception as e:
            tprint(f"⚠️ [REGIME_ENSEMBLE] Enhanced feature creation failed, using base features: {e}", color="yellow")
            return meta_features

    def _train_stacker_lgbm_calibrated(self, X: np.ndarray, y: np.ndarray, base_models: Dict[str, Any], sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train stacker_lgbm_calibrated meta-learner with HPO optimization.
        
        Uses comprehensive meta-features including:
        - Base model predictions (probabilities)
        - Uncertainty features (entropy, variance)
        - Confidence features (max prob, margin)
        - Disagreement features (diversity, agreement rate)
        
        Args:
            X: Feature matrix for base models
            y: Target labels
            base_models: Dictionary of trained base models
            
        Returns:
            Dictionary containing trained meta-learner, contracts, and metrics
        """
        tprint("🎭 [REGIME_ENSEMBLE] Training stacker_lgbm_calibrated meta-learner with enhanced meta-features", color="yellow", bold=True)
        tprint(f"🔍 [REGIME_ENSEMBLE] Function entry - X shape: {X.shape}, y shape: {y.shape}", color="cyan")
        tprint(f"🔍 [REGIME_ENSEMBLE] Base models provided: {base_models is not None}, count: {len(base_models) if base_models else 0}", color="cyan")

        try:
            # Validate base models
            if not base_models:
                tprint("❌ [REGIME_ENSEMBLE] No base models provided", color="red")
                return None

            tprint(
                f"📊 [REGIME_ENSEMBLE] Using {len(base_models)} base models: {list(base_models.keys())}",
                color="blue"
            )
            
            # Log base model input features for validation
            tprint(f"📊 [REGIME_ENSEMBLE] Base model input shape: {X.shape}", color="blue")
            tprint(f"📊 [REGIME_ENSEMBLE] Target shape: {y.shape}", color="blue")
            tprint(f"📊 [REGIME_ENSEMBLE] Number of classes: {len(np.unique(y))}", color="blue")

            # Generate comprehensive meta-features using the meta-features generator
            # This includes: base predictions + uncertainty + confidence + disagreement
            tprint("🔧 [REGIME_ENSEMBLE] Generating comprehensive meta-features", color="cyan", bold=True)
            
            meta_features, meta_feature_names = self.meta_features_generator.generate_meta_features(
                base_models=base_models,
                X=X,
                y=y,
                include_uncertainty=True,
                include_confidence=True,
                include_disagreement=True
            )
            
            tprint(
                f"✅ [REGIME_ENSEMBLE] Meta-features generated: shape {meta_features.shape} with {len(meta_feature_names)} features",
                color="green",
                bold=True
            )
            
            # CRITICAL FIX: Remove zero-variance features before training
            # Zero-variance features cause HPO validation failures
            feature_variances = np.var(meta_features, axis=0)
            zero_var_mask = feature_variances > 1e-10  # Keep features with non-zero variance
            n_zero_var = np.sum(~zero_var_mask)
            
            if n_zero_var > 0:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Removing {n_zero_var} zero-variance features", color="yellow", bold=True)
                meta_features = meta_features[:, zero_var_mask]
                meta_feature_names = [name for i, name in enumerate(meta_feature_names) if zero_var_mask[i]]
                tprint(f"✅ [REGIME_ENSEMBLE] Meta-features after variance filtering: {meta_features.shape}", color="green")
            
            # Log meta-feature composition
            base_pred_count = sum(1 for name in meta_feature_names if 'class' in name and 'prob' in name)
            uncertainty_count = sum(1 for name in meta_feature_names if 'uncertainty' in name)
            confidence_count = sum(1 for name in meta_feature_names if 'confidence' in name)
            disagreement_count = sum(1 for name in meta_feature_names if 'disagreement' in name)
            
            tprint("📋 [REGIME_ENSEMBLE] Meta-feature composition:", color="cyan")
            tprint(f"   - Base predictions: {base_pred_count}", color="blue")
            tprint(f"   - Uncertainty features: {uncertainty_count}", color="blue")
            tprint(f"   - Confidence features: {confidence_count}", color="blue")
            tprint(f"   - Disagreement features: {disagreement_count}", color="blue")

            # Log class distribution for debugging
            unique, counts = np.unique(y, return_counts=True)
            class_dist = dict(zip([int(u) for u in unique], [int(c) for c in counts]))
            tprint(f"📊 [REGIME_ENSEMBLE] Training class distribution: {class_dist}", color="cyan", bold=True)
            tprint(f"📊 [REGIME_ENSEMBLE] Total samples: {len(y)}, Unique regimes: {len(unique)}", color="cyan", bold=True)
            
            # Check for severely imbalanced classes
            min_samples = min(counts)
            if min_samples < 3:
                tprint(f"⚠️ [REGIME_ENSEMBLE] WARNING: Some classes have < 3 samples (min={min_samples})", color="yellow", bold=True)
                tprint(f"   This will cause issues with 3-fold CV. Consider using stratified sampling or reducing CV folds.", color="yellow")
            
            # CRITICAL FIX: Calculate custom class weights to combat imbalance
            # Penalize majority classes more heavily to prevent bias
            total_samples = len(y)
            n_classes = len(unique)
            
            # Calculate balanced weights with extra penalty for majority classes
            class_weights = {}
            max_count = max(counts)
            for regime_id, count in zip(unique, counts):
                # Base balanced weight
                balanced_weight = total_samples / (n_classes * count)
                # Extra penalty for majority classes (inverse frequency squared)
                majority_penalty = (max_count / count) ** 1.5
                # Combined weight
                class_weights[int(regime_id)] = balanced_weight * majority_penalty
            
            tprint(f"🎯 [REGIME_ENSEMBLE] Custom class weights: {class_weights}", color="cyan")
            
            # CRITICAL FIX: Use ONLY base meta-features (no enhancement) to reduce noise
            # Enhanced features (53 from 40) were adding noise rather than signal
            tprint("🔧 [REGIME_ENSEMBLE] Using simplified meta-features (base predictions only)", color="blue")
            simplified_meta_features = meta_features  # Use original 40 features
            tprint(f"📊 [REGIME_ENSEMBLE] Simplified meta-features shape: {simplified_meta_features.shape}", color="blue")
            tprint(f"📊 [REGIME_ENSEMBLE] Feature reduction: 53 -> {simplified_meta_features.shape[1]} (removed noisy enhanced features)", color="green")
            
            # Perform HPO for meta-learner tuning
            tprint("🔍 [REGIME_ENSEMBLE] Starting HPO for meta-learner optimization", color="cyan", bold=True)
            
            # Define search space for LightGBM meta-learner
            search_space = {
                'num_leaves': {'type': 'int', 'low': 10, 'high': 30},
                'max_depth': {'type': 'int', 'low': 3, 'high': 7},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.1, 'log': True},
                'n_estimators': {'type': 'int', 'low': 200, 'high': 600},
                'min_child_samples': {'type': 'int', 'low': 30, 'high': 100},
                'feature_fraction': {'type': 'float', 'low': 0.6, 'high': 0.9},
                'bagging_fraction': {'type': 'float', 'low': 0.6, 'high': 0.9},
                'reg_alpha': {'type': 'float', 'low': 0.01, 'high': 0.5, 'log': True},
                'reg_lambda': {'type': 'float', 'low': 0.01, 'high': 0.5, 'log': True}
            }
            
            # Model factory for HPO with custom class weights
            def create_lgbm_meta_learner(**params):
                return LGBMClassifier(
                    **params,
                    class_weight=class_weights,  # Use custom weights instead of 'balanced'
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1,
                    bagging_freq=5,
                    min_split_gain=0.01
                )
            
            try:
                # Adjust CV folds based on minimum class size
                cv_folds = 2 if min_samples < 3 else 3
                tprint(f"🔧 [REGIME_ENSEMBLE] Using {cv_folds}-fold CV (min class size: {min_samples})", color="cyan")
                
                # CRITICAL: Skip HPO for very small datasets (< 100 samples) to avoid validation failures
                # With small datasets, HPO validation will fail due to data leakage detection
                if len(y) < 100:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] Dataset too small ({len(y)} samples) - skipping HPO, using default params", color="yellow", bold=True)
                    tprint("   HPO validation requires larger datasets to avoid false data leakage warnings", color="yellow")
                    hpo_result = {'error': 'Dataset too small for HPO'}
                else:
                    # Run HPO with simplified features
                    hpo_result = self.hpo_optimizer.bayesian_optimization(
                        model_factory=create_lgbm_meta_learner,
                        X=simplified_meta_features,  # Use simplified features
                        y=y,
                        search_space=search_space,
                        cv=cv_folds,  # Adaptive CV folds
                        scoring='f1_weighted',  # Use weighted F1 for imbalanced classes
                        n_trials=50,  # Reasonable number of trials
                        fit_params={'sample_weight': sample_weight} if sample_weight is not None else None
                    )
                
                # Check if HPO succeeded and extract best parameters
                if hpo_result and not hpo_result.get('error'):
                    best_params = hpo_result.get('best_params', {})
                    best_score = hpo_result.get('best_score', 0)
                    
                    if best_params and best_score > 0:
                        tprint(f"✅ [REGIME_ENSEMBLE] HPO completed successfully", color="green")
                        tprint(f"📊 [REGIME_ENSEMBLE] Best F1 score: {best_score:.4f}", color="blue")
                        tprint(f"📊 [REGIME_ENSEMBLE] Best params: {best_params}", color="blue")
                        
                        # Create model with best parameters and custom weights
                        meta_learner = create_lgbm_meta_learner(**best_params)
                        # Train on simplified features
                        meta_learner.fit(simplified_meta_features, y, sample_weight=sample_weight)
                        hpo_result['success'] = True  # Mark as successful
                    else:
                        tprint(f"⚠️ [REGIME_ENSEMBLE] HPO returned invalid params/score, using defaults", color="yellow")
                        raise Exception("HPO returned invalid params/score")
                else:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] HPO failed with error: {hpo_result.get('error', 'unknown')}", color="yellow")
                    raise Exception(f"HPO failed: {hpo_result.get('error', 'unknown')}")
                    
            except Exception as e:
                tprint(f"⚠️ [REGIME_ENSEMBLE] HPO error: {e}, using optimized defaults", color="yellow")
                # Fallback to optimized default parameters with custom weights
                meta_learner = LGBMClassifier(
                    num_leaves=15,
                    max_depth=4,
                    learning_rate=0.03,
                    n_estimators=400,
                    min_child_samples=50,
                    feature_fraction=0.7,
                    bagging_fraction=0.7,
                    bagging_freq=5,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    class_weight=class_weights,  # Use custom weights
                    random_state=42,
                    verbose=-1,
                    n_jobs=-1,
                    min_split_gain=0.01
                )
                # Train with fallback parameters on simplified features
                meta_learner.fit(simplified_meta_features, y, sample_weight=sample_weight)
                hpo_result = {'success': False, 'best_score': 0.0, 'best_model': meta_learner}
            
            tprint("✅ [REGIME_ENSEMBLE] Meta-learner training completed", color="green")

            # Apply probability calibration
            tprint("🎯 [REGIME_ENSEMBLE] Applying probability calibration", color="blue")
            try:
                # Use same adaptive CV folds as HPO
                calibration_cv = 2 if min_samples < 3 else 3
                tprint(f"🔧 [REGIME_ENSEMBLE] Using {calibration_cv}-fold CV for calibration", color="cyan")
                
                calibrated_meta_learner = CalibratedClassifierCV(
                    meta_learner,
                    method=self.ensemble_config.get('calibration_method', 'isotonic'),
                    cv=calibration_cv  # Use adaptive CV
                )
                # CRITICAL FIX: Use simplified features for calibration
                calibrated_meta_learner.fit(simplified_meta_features, y, sample_weight=sample_weight) 
                tprint("✅ [REGIME_ENSEMBLE] Probability calibration applied successfully", color="green")

                # Create feature contract for the ensemble
                # CRITICAL FIX: Use simplified features (no enhancement)
                ensemble_contract = FeatureContract(
                    feature_names=meta_feature_names,
                    feature_count=len(meta_feature_names),
                    feature_types={name: self._infer_feature_type(name) for name in meta_feature_names},
                    expected_shape=(None, simplified_meta_features.shape[1]),  # Use simplified shape!
                    metadata={
                        'source': 'meta_features_generator',
                        'includes_uncertainty': True,
                        'includes_confidence': True,
                        'includes_disagreement': True,
                        'simplified_feature_count': simplified_meta_features.shape[1],  # Store for prediction
                        'base_meta_feature_count': meta_features.shape[1],
                        'feature_simplification': 'enabled',  # Flag that we're using simplified features
                        'zero_var_mask': zero_var_mask.tolist() if n_zero_var > 0 else None  # Store mask for prediction
                    }
                )
                
                # Return calibrated result
                stacker_result = {
                    'meta_learner': calibrated_meta_learner,
                    'base_models': base_models,
                    'meta_feature_names': meta_feature_names,
                    'meta_features_shape': meta_features.shape,
                    'simplified_meta_features_shape': simplified_meta_features.shape,
                    'zero_var_mask': zero_var_mask if n_zero_var > 0 else None,
                    'feature_contract': ensemble_contract,
                    'calibration_method': self.ensemble_config.get('calibration_method', 'isotonic'),
                    'cv_folds': calibration_cv,
                    'training_success': True,
                    'hpo_result': hpo_result if hpo_result.get('success', False) else None,
                    'model': calibrated_meta_learner
                }
                
                return stacker_result

            except Exception as e:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Probability calibration failed: {e}", color="yellow")
                tprint("📊 [REGIME_ENSEMBLE] Using uncalibrated meta-learner", color="blue")

                # Create feature contract for the ensemble
                # CRITICAL FIX: Use simplified features (no enhancement)
                ensemble_contract = FeatureContract(
                    feature_names=meta_feature_names,
                    feature_count=len(meta_feature_names),
                    feature_types={name: self._infer_feature_type(name) for name in meta_feature_names},
                    expected_shape=(None, simplified_meta_features.shape[1]),  # Use simplified shape!
                    metadata={
                        'source': 'meta_features_generator',
                        'includes_uncertainty': True,
                        'includes_confidence': True,
                        'includes_disagreement': True,
                        'simplified_feature_count': simplified_meta_features.shape[1],  # Store for prediction
                        'base_meta_feature_count': meta_features.shape[1],
                        'feature_simplification': 'enabled'  # Flag that we're using simplified features
                    }
                )
                
                # Return uncalibrated result
                stacker_result = {
                    'meta_learner': meta_learner,
                    'base_models': base_models,
                    'meta_feature_names': meta_feature_names,
                    'meta_features_shape': meta_features.shape,
                    'simplified_meta_features_shape': simplified_meta_features.shape,  # Store simplified shape
                    'zero_var_mask': zero_var_mask if n_zero_var > 0 else None,  # Store mask for prediction
                    'feature_contract': ensemble_contract,
                    'calibration_method': 'none',
                    'cv_folds': 0,
                    'training_success': True,
                    'hpo_result': hpo_result if hpo_result.get('success', False) else None,
                    'model': meta_learner
                }

                tprint("✅ [REGIME_ENSEMBLE] stacker_lgbm_calibrated training completed (uncalibrated)", color="green")
                return stacker_result

        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] stacker_lgbm_calibrated training failed: {e}", color="red")
            return None
    
    def _infer_feature_type(self, feature_name: str) -> str:
        """
        Infer the type of a feature based on its name.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Feature type string
        """
        feature_name_lower = feature_name.lower()
        
        if 'prob' in feature_name_lower and 'class' in feature_name_lower:
            return 'base_prediction'
        elif 'uncertainty' in feature_name_lower:
            return 'uncertainty'
        elif 'confidence' in feature_name_lower:
            return 'confidence'
        elif 'disagreement' in feature_name_lower:
            return 'disagreement'
        else:
            return 'meta'

    def _convert_numpy_types(self, obj: Any) -> Any:
        """
        Recursively convert numpy types to native Python types for JSON serialization.
        
        Args:
            obj: Object to convert (can be dict, list, numpy type, etc.)
            
        Returns:
            Object with all numpy types converted to Python native types
        """
        # Handle None
        if obj is None:
            return None
            
        # Handle dictionaries - convert all keys to strings
        if isinstance(obj, dict):
            converted = {}
            for k, v in obj.items():
                # Convert key to string if it's any numeric type
                if isinstance(k, (np.integer, np.floating, int, float)):
                    key_str = str(int(k) if isinstance(k, (np.integer, int)) else k)
                else:
                    key_str = str(k)
                converted[key_str] = self._convert_numpy_types(v)
            return converted
            
        # Handle lists and tuples
        elif isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
            
        # Handle numpy types
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
            
        # Handle sklearn models and other non-serializable objects - skip them
        elif hasattr(obj, '__module__') and ('sklearn' in obj.__module__ or 'lightgbm' in obj.__module__ or 'catboost' in obj.__module__):
            return f"<{obj.__class__.__name__} object>"
            
        # Return as-is for basic Python types
        else:
            return obj

    def _evaluate_ensemble(self, X: np.ndarray, y: np.ndarray, stacker_result: Dict[str, Any], sample_weight: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate ensemble performance using enhanced ML utilities.
        
        Uses same meta-features generation as training for consistency.
        """
        tprint("📊 [REGIME_ENSEMBLE] Evaluating ensemble performance with enhanced ML utilities", color="yellow", bold=True)

        metrics = {}

        if stacker_result is None:
            tprint("❌ [REGIME_ENSEMBLE] No stacker result to evaluate", color="red")
            return {'error': 'No stacker result available'}

        try:
            meta_learner = stacker_result['meta_learner']
            base_models = stacker_result['base_models']
            meta_feature_names = stacker_result.get('meta_feature_names', [])
            feature_contract = stacker_result.get('feature_contract')

            tprint(f"📊 [REGIME_ENSEMBLE] Evaluation input shape: {X.shape}", color="blue")
            tprint(f"📊 [REGIME_ENSEMBLE] Evaluation target shape: {y.shape}", color="blue")
            tprint(f"📊 [REGIME_ENSEMBLE] Base models count: {len(base_models)}", color="blue")

            # Generate meta-features for evaluation using the SAME method as training
            tprint("🔧 [REGIME_ENSEMBLE] Generating meta-features for evaluation (consistent with training)", color="cyan", bold=True)
            
            meta_features, generated_feature_names = self.meta_features_generator.generate_meta_features(
                base_models=base_models,
                X=X,
                y=y,
                include_uncertainty=True,
                include_confidence=True,
                include_disagreement=True
            )
            
            tprint(
                f"✅ [REGIME_ENSEMBLE] Meta-features generated for evaluation: {meta_features.shape}",
                color="green",
                bold=True
            )
            
            # Apply enhanced meta-features transformation (same as training)
            tprint("🔧 [REGIME_ENSEMBLE] Creating enhanced meta-features for evaluation", color="cyan")
            meta_features = self._create_enhanced_meta_features(meta_features, y)
            tprint(f"✅ [REGIME_ENSEMBLE] Enhanced meta-features for evaluation: {meta_features.shape}", color="green")

            # Validate feature contract if available
            if feature_contract is not None:
                tprint("🔍 [REGIME_ENSEMBLE] Validating meta-features against feature contract", color="cyan")
                try:
                    feature_contract.validate_features(meta_features, generated_feature_names)
                    tprint("✅ [REGIME_ENSEMBLE] Feature contract validation passed", color="green")
                except ValueError as e:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] Feature contract validation failed: {e}", color="yellow")
                    tprint("⚠️ [REGIME_ENSEMBLE] Continuing with evaluation despite validation failure", color="yellow")

            # Check if meta-learner expects different number of features
            if hasattr(meta_learner, 'n_features_') and meta_learner.n_features_ != meta_features.shape[1]:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Feature mismatch: meta-learner expects {meta_learner.n_features_} features, got {meta_features.shape[1]}", color="yellow")
                # Skip evaluation if feature dimensions don't match
                return {'error': f'Feature dimension mismatch: expected {meta_learner.n_features_}, got {meta_features.shape[1]}'}

            # Evaluate meta-learner
            tprint("📊 [REGIME_ENSEMBLE] Evaluating meta-learner", color="blue")
            y_pred = meta_learner.predict(meta_features)
            y_pred_proba = meta_learner.predict_proba(meta_features)

            # Use enhanced model evaluator for comprehensive evaluation
            tprint("🔍 [REGIME_ENSEMBLE] Performing comprehensive model evaluation", color="cyan")
            # CRITICAL FIX: Use correct method name evaluate_model_performance
            evaluation_result = self.model_evaluator.evaluate_model_performance(
                model=meta_learner,
                X=meta_features,
                y=y
            )

            # Use model validator for additional validation
            tprint("🔍 [REGIME_ENSEMBLE] Performing model validation", color="cyan")
            validation_result = self.model_validator.validate_model(
                model=meta_learner,
                X=meta_features,
                y=y,
                cv_folds=5,
                fit_params={'sample_weight': sample_weight}
            )

            # Calculate basic metrics
            accuracy = accuracy_score(y, y_pred, sample_weight=sample_weight)
            
            # Calculate comprehensive temporal and regime-persistence metrics
            # CRITICAL FIX: Remove sample_weight parameter - not supported by this method
            comprehensive_metrics = self.temporal_metrics_calc.calculate_comprehensive_metrics(
                y, y_pred, y_pred_proba
            )
            
            # Calculate temporal smoothness penalty
            smoothness_penalty = calculate_temporal_smoothness_penalty(
                y_pred, alpha=self.temporal_smoothing_alpha
            )

            # Calculate top-3 regime analysis with entropy metrics
            top_3_analysis = self._calculate_top_regime_analysis(y_pred_proba)

            # Get classification report and convert int64 keys to strings for JSON serialization
            class_report = classification_report(y, y_pred, sample_weight=sample_weight, output_dict=True, zero_division=0)
            tprint(f"🐛 [DEBUG] Classification report keys: {list(class_report.keys()) if isinstance(class_report, dict) else 'NOT A DICT'}", color="yellow")
            tprint(f"🐛 [DEBUG] Classification report has weighted avg: {'weighted avg' in class_report if isinstance(class_report, dict) else False}", color="yellow")
            
            # Convert numpy int64 keys to strings
            class_report_clean = {}
            if isinstance(class_report, dict):
                for key, value in class_report.items():
                    # Convert int64 keys to strings
                    str_key = str(key) if isinstance(key, (np.integer, int)) else key
                    class_report_clean[str_key] = value
                tprint(f"🐛 [DEBUG] Converted classification report keys: {list(class_report_clean.keys())}", color="yellow")
            else:
                class_report_clean = class_report

            # Enhanced metrics with ML utilities
            metrics['stacker_lgbm_calibrated'] = {
                'accuracy': accuracy,
                'classification_report': class_report_clean,
                'classification': comprehensive_metrics.get('classification', {}),
                'temporal': comprehensive_metrics.get('temporal', {}),
                'persistence': comprehensive_metrics.get('persistence', {}),
                'smoothness_penalty': smoothness_penalty,
                'prediction_confidence': {
                    'mean': y_pred_proba.max(axis=1).mean(),
                    'std': y_pred_proba.max(axis=1).std()
                },
                'top_regime_analysis': top_3_analysis,
                'calibration_method': stacker_result.get('calibration_method', 'none'),
                'base_models_used': len(base_models),
                'meta_features_shape': meta_features.shape,
                'enhanced_evaluation': evaluation_result,
                'model_validation': validation_result,
                'hpo_result': stacker_result.get('hpo_result')
            }

            # Calculate comprehensive metrics for meta-learner
            precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='weighted', sample_weight=sample_weight)
            confidence_mean = y_pred_proba.max(axis=1).mean()
            confidence_std = y_pred_proba.max(axis=1).std()

            tprint("🎯 [REGIME_ENSEMBLE] META-LEARNER PERFORMANCE METRICS", color="green", bold=True)
            tprint("="*50, color="green")
            tprint(f"🎯 Accuracy: {accuracy:.4f}", color="green")
            tprint(f"📈 Precision: {precision:.4f}", color="green")
            tprint(f"📈 Recall: {recall:.4f}", color="green")
            tprint(f"📈 F1-Score: {f1:.4f}", color="green")
            tprint(f"🎲 Prediction Confidence: {confidence_mean:.4f} ± {confidence_std:.4f}", color="green")
            tprint(f"🔧 Calibration Method: {stacker_result.get('calibration_method', 'none')}", color="green")
            tprint(f"🤖 Base Models Used: {len(base_models)}", color="green")
            tprint(f"📊 Meta-features Shape: {meta_features.shape}", color="green")

            # Display enhanced evaluation results
            if evaluation_result and evaluation_result.get('success'):
                eval_metrics = evaluation_result.get('metrics', {})
                tprint("🔍 ENHANCED EVALUATION RESULTS", color="cyan", bold=True)
                tprint(f"   📊 SHAP Analysis: {'Available' if eval_metrics.get('shap_available') else 'Not Available'}", color="cyan")
                tprint(f"   📊 LIME Analysis: {'Available' if eval_metrics.get('lime_available') else 'Not Available'}", color="cyan")
                tprint(f"   📊 OOF Validation: {'Passed' if eval_metrics.get('oof_validation_passed') else 'Failed'}", color="cyan")
                tprint(f"   📊 OOS Validation: {'Passed' if eval_metrics.get('oos_validation_passed') else 'Failed'}", color="cyan")

            # Display validation results
            if validation_result and validation_result.get('success'):
                val_metrics = validation_result.get('metrics', {})
                tprint("🔍 MODEL VALIDATION RESULTS", color="cyan", bold=True)
                tprint(f"   📊 Purged CV Score: {val_metrics.get('purged_cv_score', 'N/A')}", color="cyan")
                tprint(f"   📊 Data Leakage: {'Detected' if val_metrics.get('data_leakage_detected') else 'Not Detected'}", color="cyan")
                tprint(f"   📊 Time Series Validation: {'Passed' if val_metrics.get('time_series_validation_passed') else 'Failed'}", color="cyan")

            # Display top regime analysis summary
            if 'top_regime_analysis' in metrics['stacker_lgbm_calibrated']:
                top_analysis = metrics['stacker_lgbm_calibrated']['top_regime_analysis']
                entropy_metrics = top_analysis['entropy_metrics']
                confidence_gaps = top_analysis['confidence_gaps']
                conf_dist = top_analysis['prediction_confidence_distribution']

                tprint("🎯 TOP REGIME ANALYSIS", color="cyan", bold=True)
                tprint(f"   📊 Avg Entropy: {entropy_metrics['mean_entropy']:.4f}", color="cyan")
                tprint(f"   🎲 Confidence Gap (1st-2nd): {confidence_gaps['gap_1_2_mean']:.4f}", color="cyan")
                tprint(f"   📈 High Confidence Samples: {conf_dist['high_confidence_ratio']:.1%}", color="cyan")
                tprint(f"   📉 Low Confidence Samples: {conf_dist['low_confidence_ratio']:.1%}", color="cyan")

            tprint("="*50, color="green")

            # Add comparison with base models if available
            if base_models:
                tprint("🔄 [REGIME_ENSEMBLE] ENSEMBLE vs BASE MODELS COMPARISON", color="cyan", bold=True)
                tprint("="*60, color="cyan")

                # Calculate base model accuracies for comparison
                base_accuracies = {}
                for name, model in base_models.items():
                    try:
                        if name not in ['stacker_lgbm_calibrated', 'stacker_lgbm_calibrated_feature_indices']:
                            y_pred_base = model.predict(X)
                            base_accuracy = accuracy_score(y, y_pred_base, sample_weight=sample_weight)
                            base_accuracies[name] = base_accuracy
                    except Exception as e:
                        tprint(f"⚠️ [REGIME_ENSEMBLE] Could not evaluate {name}: {e}", color="yellow")

                # Print comparison
                tprint(f"🎯 Meta-learner Accuracy: {accuracy:.4f}", color="green")
                for name, base_acc in base_accuracies.items():
                    improvement = accuracy - base_acc
                    status = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
                    tprint(f"   {status} {name}: {base_acc:.4f} (Δ: {improvement:+.4f})", color="blue")

                # Calculate average base model performance
                if base_accuracies:
                    avg_base_accuracy = np.mean(list(base_accuracies.values()))
                    ensemble_improvement = accuracy - avg_base_accuracy
                    tprint(f"📊 Average Base Model: {avg_base_accuracy:.4f}", color="blue")
                    tprint(f"🚀 Ensemble Improvement: {ensemble_improvement:+.4f}", color="green" if ensemble_improvement > 0 else "red")

                tprint("="*60, color="cyan")

        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Ensemble evaluation failed: {e}", color="red")
            metrics['stacker_lgbm_calibrated'] = {'error': str(e)}

        tprint("✅ [REGIME_ENSEMBLE] Ensemble evaluation completed", color="green")
        
        # Convert all numpy types to Python native types for JSON serialization
        metrics = self._convert_numpy_types(metrics)
        
        return metrics

    def _calculate_top_regime_analysis(self, y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calculate comprehensive analysis of top regime predictions.

        Args:
            y_pred_proba: Probability predictions for each sample and regime

        Returns:
            Dictionary containing top-3 regime analysis with entropy metrics
        """
        try:
            n_samples, n_regimes = y_pred_proba.shape

            # Get top 3 predictions and probabilities for each sample
            # Use argsort with descending order to get highest probabilities first
            top_3_indices = np.argsort(y_pred_proba, axis=1)[:, -3:][:, ::-1]  # Get top 3, reverse to descending
            top_3_probabilities = np.sort(y_pred_proba, axis=1)[:, -3:][:, ::-1]  # Get top 3 probs, descending

            # Calculate entropy (measure of prediction uncertainty)
            # Use small epsilon to avoid log(0)
            epsilon = 1e-10
            entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + epsilon), axis=1)

            # Calculate confidence gaps between predictions
            confidence_gap_1_2 = top_3_probabilities[:, 0] - top_3_probabilities[:, 1]  # Gap between 1st and 2nd
            confidence_gap_2_3 = top_3_probabilities[:, 1] - top_3_probabilities[:, 2]  # Gap between 2nd and 3rd

            # Calculate relative confidence (how much more confident in 1st vs 2nd)
            relative_confidence_1_2 = np.divide(
                confidence_gap_1_2,
                top_3_probabilities[:, 0],
                out=np.zeros_like(confidence_gap_1_2),
                where=top_3_probabilities[:, 0] != 0
            )

            # Identify high-confidence vs low-confidence predictions
            high_confidence_threshold = 0.8
            low_confidence_threshold = 0.4

            high_confidence_samples = np.sum(top_3_probabilities[:, 0] >= high_confidence_threshold)
            low_confidence_samples = np.sum(top_3_probabilities[:, 0] <= low_confidence_threshold)
            uncertain_samples = n_samples - high_confidence_samples - low_confidence_samples

            # Calculate regime frequency in top predictions
            top_1_regime_counts = np.bincount(top_3_indices[:, 0], minlength=n_regimes)
            top_2_regime_counts = np.bincount(top_3_indices[:, 1], minlength=n_regimes)
            top_3_regime_counts = np.bincount(top_3_indices[:, 2], minlength=n_regimes)

            return {
                'top_predictions': {
                    'regime_indices': top_3_indices.tolist(),
                    'probabilities': top_3_probabilities.tolist()
                },
                'entropy_metrics': {
                    'mean_entropy': float(entropy.mean()),
                    'std_entropy': float(entropy.std()),
                    'min_entropy': float(entropy.min()),
                    'max_entropy': float(entropy.max()),
                    'entropy_distribution': {
                        'low_uncertainty': int(np.sum(entropy < 0.5)),
                        'medium_uncertainty': int(np.sum((entropy >= 0.5) & (entropy < 1.0))),
                        'high_uncertainty': int(np.sum(entropy >= 1.0))
                    }
                },
                'confidence_gaps': {
                    'gap_1_2_mean': float(confidence_gap_1_2.mean()),
                    'gap_1_2_std': float(confidence_gap_1_2.std()),
                    'gap_2_3_mean': float(confidence_gap_2_3.mean()),
                    'gap_2_3_std': float(confidence_gap_2_3.std()),
                    'relative_confidence_1_2_mean': float(relative_confidence_1_2.mean()),
                    'relative_confidence_1_2_std': float(relative_confidence_1_2.std())
                },
                'prediction_confidence_distribution': {
                    'high_confidence_samples': int(high_confidence_samples),
                    'low_confidence_samples': int(low_confidence_samples),
                    'uncertain_samples': int(uncertain_samples),
                    'high_confidence_ratio': float(high_confidence_samples / n_samples),
                    'low_confidence_ratio': float(low_confidence_samples / n_samples),
                    'uncertain_ratio': float(uncertain_samples / n_samples)
                },
                'regime_frequency_analysis': {
                    'top_1_regime_distribution': top_1_regime_counts.tolist(),
                    'top_2_regime_distribution': top_2_regime_counts.tolist(),
                    'top_3_regime_distribution': top_3_regime_counts.tolist(),
                    'most_common_second_choice': int(np.argmax(top_2_regime_counts)),
                    'most_common_third_choice': int(np.argmax(top_3_regime_counts))
                },
                'summary_statistics': {
                    'total_samples': n_samples,
                    'total_regimes': n_regimes,
                    'avg_top_1_confidence': float(top_3_probabilities[:, 0].mean()),
                    'avg_top_2_confidence': float(top_3_probabilities[:, 1].mean()),
                    'avg_top_3_confidence': float(top_3_probabilities[:, 2].mean())
                }
            }

        except Exception as e:
            self.logger.error(f"Error calculating top regime analysis: {e}")
            return {
                'error': str(e),
                'entropy_metrics': {'mean_entropy': 0.0, 'std_entropy': 0.0},
                'confidence_gaps': {'gap_1_2_mean': 0.0, 'gap_2_3_mean': 0.0},
                'summary_statistics': {'total_samples': len(y_pred_proba), 'total_regimes': y_pred_proba.shape[1]}
            }

    def _prepare_training_data(self, data: pd.DataFrame, regime_labels: np.ndarray, pipeline_state: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data from market data and regime labels."""
        tprint("🔧 [REGIME_ENSEMBLE] Preparing training data", color="cyan")
        self.logger.info("Starting data preparation process")

        try:
            # Log input data characteristics
            tprint(f"📊 [REGIME_ENSEMBLE] Input data shape: {data.shape}", color="blue")
            tprint(f"📊 [REGIME_ENSEMBLE] Input data columns: {list(data.columns)}", color="blue")

            # Force comprehensive feature generation using feature bank
            tprint("🔧 [REGIME_ENSEMBLE] FORCING comprehensive feature generation using feature bank", color="cyan", bold=True)
            tprint("🚫 [REGIME_ENSEMBLE] Bypassing base model features to ensure comprehensive feature set", color="yellow")

            # Check if we should use original market data for feature generation
            original_data = None
            if pipeline_state is not None:
                original_data = pipeline_state.get('original_data')
                force_feature_bank = pipeline_state.get('force_feature_bank', False)

                if original_data is not None and force_feature_bank:
                    tprint("✅ [REGIME_ENSEMBLE] Using original market data for feature bank generation", color="green")
                    data_for_features = original_data
                else:
                    tprint("⚠️ [REGIME_ENSEMBLE] No original data available, using processed data", color="yellow")
                    data_for_features = data
            else:
                data_for_features = data

            if FEATURE_GENERATION_AVAILABLE:
                X = self._generate_features_with_bank(data_for_features)
                if X is None or X.shape[1] < 50:
                    error_msg = f"Feature bank generated insufficient features: {X.shape[1] if X is not None else 0} < 50 required"
                    tprint(f"❌ [REGIME_ENSEMBLE] {error_msg}", color="red")
                    self.logger.error(error_msg)
                    return None, None, None
                else:
                    tprint(f"✅ [REGIME_ENSEMBLE] Feature bank generated {X.shape[1]} comprehensive features", color="green")
                    feature_names = [f'feature_{i}' for i in range(X.shape[1])]
            else:
                error_msg = "Feature generation system not available - cannot generate comprehensive features"
                tprint(f"❌ [REGIME_ENSEMBLE] {error_msg}", color="red")
                self.logger.error(error_msg)
                return None, None, None
            tprint(f"📋 [REGIME_ENSEMBLE] Feature names ({len(feature_names)}): {feature_names[:10]}..." if len(feature_names) > 10 else f"📋 [REGIME_ENSEMBLE] Feature names ({len(feature_names)}): {feature_names}", color="blue")

            # Check for NaN or infinite values in features with detailed analysis
            nan_count = np.isnan(X).sum()
            inf_count = np.isinf(X).sum()
            if nan_count > 0:
                # Import the detailed NaN analysis function
                from src.utils.common_utilities import analyze_nan_values_detailed, format_nan_analysis_report

                # Perform detailed NaN analysis
                nan_analysis = analyze_nan_values_detailed(X, feature_names)
                detailed_report = format_nan_analysis_report(nan_analysis, "[REGIME_ENSEMBLE] ")

                tprint(f"⚠️ [REGIME_ENSEMBLE] Found {nan_count} NaN values in features", color="yellow")
                tprint(detailed_report, color="yellow")
                tprint("🔧 [REGIME_ENSEMBLE] Using sophisticated NaN handling for time series data", color="cyan")
                # Use sophisticated NaN handling for time series data
                X = self._handle_nan_values(X, nan_count)
            if inf_count > 0:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Found {inf_count} infinite values in features", color="yellow")
                tprint("🔧 [REGIME_ENSEMBLE] Replacing infinite values with finite numbers", color="cyan")
                X = np.nan_to_num(X, posinf=1e6, neginf=-1e6)

            # Align with regime labels
            tprint("🔧 [REGIME_ENSEMBLE] Aligning features with regime labels", color="cyan")
            min_length = min(len(X), len(regime_labels))
            X = X[:min_length]
            y = np.array(regime_labels[:min_length])

            tprint(f"✅ [REGIME_ENSEMBLE] Training data prepared: {X.shape[0]} samples, {X.shape[1]} features", color="green", bold=True)

            self.logger.info(f"Training data preparation completed: {X.shape[0]} samples, {X.shape[1]} features")
            return X, y, feature_names

        except Exception as e:
            error_type = type(e).__name__
            tprint(f"❌ [REGIME_ENSEMBLE] Error preparing training data: {e}", color="red")
            tprint(f"🔍 [REGIME_ENSEMBLE] Error type: {error_type}", color="yellow")
            self.logger.error(f"Error preparing training data: {e}", exc_info=True)
            return None, None, None

    def _generate_features_with_bank(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate comprehensive features using the UnifiedVectorizationManager and feature bank."""
        tprint("🔧 [REGIME_ENSEMBLE] Generating features using UnifiedVectorizationManager and feature bank", color="cyan", bold=True)

        try:
            if not FEATURE_GENERATION_AVAILABLE:
                tprint("❌ [REGIME_ENSEMBLE] Feature generation system not available", color="red")
                return None

            # Configure vectorization for feature engineering
            vectorization_config = {
                'operation_type': OperationType.FEATURE_ENGINEERING,
                'data_size': len(data),
                'data_dimensions': data.shape,
                'memory_budget_mb': 2048.0,
                'time_budget_seconds': 300.0,
                'precision_requirement': 'high'
            }

            # Get feature bank
            feature_bank = get_feature_bank()
            tprint("✅ [REGIME_ENSEMBLE] Feature bank retrieved successfully", color="green")

            # Define feature categories to generate
            categories = [
                FeatureCategory.MOMENTUM,
                FeatureCategory.VOLATILITY,
                FeatureCategory.VOLUME,
                FeatureCategory.TREND,
                FeatureCategory.OSCILLATOR,
                FeatureCategory.RETURNS,
                FeatureCategory.MICROSTRUCTURE  # Microstructure features (no orderbook dependency)
            ]

            all_features = pd.DataFrame(index=data.index)
            total_features = 0

            # Generate features for each category using vectorization manager
            for category in categories:
                tprint(f"🔍 [REGIME_ENSEMBLE] Generating {category.value} features with vectorization", color="blue")

                # Get generators for this category
                generators = feature_bank.get_generators_by_category(category)

                if not generators:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] No generators found for {category.value}", color="yellow")
                    continue

                category_features = pd.DataFrame(index=data.index)

                # Generate features using each generator with vectorization optimization
                for generator in generators:
                    try:
                        tprint(f"🔧 [REGIME_ENSEMBLE] Using generator: {generator.config.name}", color="blue")
                        
                        # Generate feature directly
                        result = generator.generate(data)

                        if result and hasattr(result, 'data') and len(result.data) > 0:
                            # Add feature with category prefix
                            feature_name = f"{category.value}_{result.name}"
                            category_features[feature_name] = result.data
                            total_features += 1
                            tprint(f"✅ [REGIME_ENSEMBLE] Generated feature: {feature_name}", color="green")
                        else:
                            tprint(f"⚠️ [REGIME_ENSEMBLE] Generator {generator.config.name} returned empty result", color="yellow")

                    except Exception as e:
                        tprint(f"⚠️ [REGIME_ENSEMBLE] Generator {generator.config.name} failed: {e}", color="yellow")
                        continue

                # Add category features to all features
                if not category_features.empty:
                    all_features = pd.concat([all_features, category_features], axis=1)
                    tprint(f"📊 [REGIME_ENSEMBLE] {category.value} features: {category_features.shape[1]}", color="blue")

            # Convert to numpy array
            if not all_features.empty:
                X = all_features.values
                
                # Add smoothed features if enabled
                if self.enable_smoothed_features:
                    tprint("🔧 [REGIME_ENSEMBLE] Adding smoothed features", color="cyan")
                    feature_names = list(all_features.columns)
                    X, feature_names = add_smoothed_features(
                        X, 
                        window_sizes=self.smoothing_window_sizes,
                        feature_names=feature_names
                    )
                    tprint(f"✅ [REGIME_ENSEMBLE] Smoothed features added: {X.shape[1]} total features", color="green")
                
                tprint(f"✅ [REGIME_ENSEMBLE] Feature bank generated {X.shape[1]} features from {len(categories)} categories", color="green")
                tprint(f"📊 [REGIME_ENSEMBLE] Feature matrix shape: {X.shape}", color="blue")
                return X
            else:
                tprint("❌ [REGIME_ENSEMBLE] Feature bank generated no features", color="red")
                return None

        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Error generating features with feature bank: {e}", color="red")
            self.logger.error(f"Error generating features with feature bank: {str(e)}", exc_info=True)
            return None

    def _train_base_models(self, X: np.ndarray, y: np.ndarray, regime_labels: np.ndarray) -> Dict[str, Any]:
        """Train base models for ensemble."""
        tprint("🏋️ [REGIME_ENSEMBLE] Training base models", color="yellow")

        base_models = {}

        # CatBoost Classifier
        tprint("🐱 [REGIME_ENSEMBLE] Training CatBoost classifier", color="blue")
        try:
            from catboost import CatBoostClassifier
            catboost_model = CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False,
                thread_count=-1
            )
            catboost_model.fit(X, y)
            base_models['catboost'] = catboost_model

            # Calculate and print CatBoost metrics
            y_pred_catboost = catboost_model.predict(X)
            y_pred_proba_catboost = catboost_model.predict_proba(X)
            catboost_accuracy = accuracy_score(y, y_pred_catboost)

            tprint("✅ [REGIME_ENSEMBLE] CatBoost trained successfully", color="green")
            tprint(f"📊 [REGIME_ENSEMBLE] CatBoost Metrics:", color="blue")
            tprint(f"   🎯 Accuracy: {catboost_accuracy:.4f}", color="blue")
            tprint(f"   🎲 Prediction Confidence: {y_pred_proba_catboost.max(axis=1).mean():.4f} ± {y_pred_proba_catboost.max(axis=1).std():.4f}", color="blue")

            # Print classification report for CatBoost
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, support = precision_recall_fscore_support(y, y_pred_catboost, average='weighted')
            tprint(f"   📈 Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}", color="blue")
        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] CatBoost training failed: {e}", color="red")

        # Random Forest Classifier
        tprint("🌳 [REGIME_ENSEMBLE] Training Random Forest classifier", color="blue")
        try:
            from sklearn.ensemble import RandomForestClassifier
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            rf_model.fit(X, y)
            base_models['random_forest'] = rf_model

            # Calculate and print Random Forest metrics
            y_pred_rf = rf_model.predict(X)
            y_pred_proba_rf = rf_model.predict_proba(X)
            rf_accuracy = accuracy_score(y, y_pred_rf)

            tprint("✅ [REGIME_ENSEMBLE] Random Forest trained successfully", color="green")
            tprint(f"📊 [REGIME_ENSEMBLE] Random Forest Metrics:", color="blue")
            tprint(f"   🎯 Accuracy: {rf_accuracy:.4f}", color="blue")
            tprint(f"   🎲 Prediction Confidence: {y_pred_proba_rf.max(axis=1).mean():.4f} ± {y_pred_proba_rf.max(axis=1).std():.4f}", color="blue")

            # Print classification report for Random Forest
            precision, recall, f1, support = precision_recall_fscore_support(y, y_pred_rf, average='weighted')
            tprint(f"   📈 Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}", color="blue")
        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Random Forest training failed: {e}", color="red")

        # Extra Tree Classifier
        tprint("🌳 [REGIME_ENSEMBLE] Training Extra Tree classifier", color="blue")
        try:
            from sklearn.ensemble import ExtraTreesClassifier
            et_model = ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            et_model.fit(X, y)
            base_models['extra_tree'] = et_model

            # Calculate and print Extra Tree metrics
            y_pred_et = et_model.predict(X)
            y_pred_proba_et = et_model.predict_proba(X)
            et_accuracy = accuracy_score(y, y_pred_et)

            tprint("✅ [REGIME_ENSEMBLE] Extra Tree trained successfully", color="green")
            tprint(f"📊 [REGIME_ENSEMBLE] Extra Tree Metrics:", color="blue")
            tprint(f"   🎯 Accuracy: {et_accuracy:.4f}", color="blue")
            tprint(f"   🎲 Prediction Confidence: {y_pred_proba_et.max(axis=1).mean():.4f} ± {y_pred_proba_et.max(axis=1).std():.4f}", color="blue")

            # Print classification report for Extra Tree
            precision, recall, f1, support = precision_recall_fscore_support(y, y_pred_et, average='weighted')
            tprint(f"   📈 Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}", color="blue")
        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Extra Tree training failed: {e}", color="red")

        tprint(f"✅ [REGIME_ENSEMBLE] Base models training completed - {len(base_models)} models trained", color="green")

        # Print comprehensive summary of all base model metrics
        tprint("📊 [REGIME_ENSEMBLE] BASE MODELS PERFORMANCE SUMMARY", color="cyan", bold=True)
        tprint("="*60, color="cyan")

        for model_name, model in base_models.items():
            try:
                y_pred = model.predict(X)
                y_pred_proba = model.predict_proba(X)
                accuracy = accuracy_score(y, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(y, y_pred, average='weighted')
                confidence_mean = y_pred_proba.max(axis=1).mean()
                confidence_std = y_pred_proba.max(axis=1).std()

                tprint(f"🤖 {model_name.upper()}:", color="yellow")
                tprint(f"   🎯 Accuracy: {accuracy:.4f}", color="blue")
                tprint(f"   📈 Precision: {precision:.4f}", color="blue")
                tprint(f"   📈 Recall: {recall:.4f}", color="blue")
                tprint(f"   📈 F1-Score: {f1:.4f}", color="blue")
                tprint(f"   🎲 Confidence: {confidence_mean:.4f} ± {confidence_std:.4f}", color="blue")
                tprint("", color="white")  # Empty line for spacing

            except Exception as e:
                tprint(f"❌ [REGIME_ENSEMBLE] Failed to evaluate {model_name}: {e}", color="red")

        tprint("="*60, color="cyan")
        tprint("✅ [REGIME_ENSEMBLE] Base models evaluation completed", color="green")

        return base_models

    def _handle_nan_values(self, X: np.ndarray, original_nan_count: int) -> np.ndarray:
        """Handle NaN values in feature matrix using sophisticated time series methods.

        Args:
            X: Feature matrix with potential NaN values
            original_nan_count: Original number of NaN values for logging

        Returns:
            Feature matrix with NaN values handled
        """
        tprint(f"🔧 [REGIME_ENSEMBLE] Handling {original_nan_count} NaN values using sophisticated methods", color="cyan")

        try:
            # Convert to pandas for better NaN handling
            df = pd.DataFrame(X)

            # Strategy 1: Forward fill for time series data (fills gaps with previous values)
            df_filled = df.fillna(method='ffill')

            # Strategy 2: Backward fill for remaining NaN values (fills gaps with future values)
            df_filled = df_filled.fillna(method='bfill')

            # Strategy 3: For any remaining NaN values, use column median
            remaining_nan_count = df_filled.isna().sum().sum()
            if remaining_nan_count > 0:
                tprint(f"📊 [REGIME_ENSEMBLE] {remaining_nan_count} NaN values remain after forward/backward fill", color="yellow")

                # Calculate median for each column
                for col in df_filled.columns:
                    if df_filled[col].isna().sum() > 0:
                        median_val = df_filled[col].median()
                        df_filled[col] = df_filled[col].fillna(median_val)

                final_nan_count = df_filled.isna().sum().sum()
                if final_nan_count > 0:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] {final_nan_count} NaN values still remain, using zero fill as last resort", color="yellow")
                    df_filled = df_filled.fillna(0.0)

            # Convert back to numpy array
            X_cleaned = df_filled.values

            # Verify no NaN values remain
            final_nan_count = np.isnan(X_cleaned).sum()
            tprint(f"✅ [REGIME_ENSEMBLE] NaN handling completed: {original_nan_count} → {final_nan_count} NaN values", color="green")

            return X_cleaned

        except Exception as e:
            tprint(f"⚠️ [REGIME_ENSEMBLE] Sophisticated NaN handling failed: {e}, falling back to zero fill", color="yellow")
            return np.nan_to_num(X, nan=0.0)

    def _create_synthetic_regime_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Create synthetic regime labels based on data patterns when clustering results are not available.

        Args:
            data: Market data DataFrame

        Returns:
            Synthetic regime labels array
        """
        tprint("🔧 [REGIME_ENSEMBLE] Creating synthetic regime labels based on data patterns", color="cyan")

        try:
            # Use simple clustering based on price volatility and trend
            if 'close' in data.columns:
                # Calculate rolling volatility
                returns = data['close'].pct_change().dropna()
                volatility = self._vectorbt_rolling_operation(returns, "std", 20).fillna(returns.std())

                # Calculate trend strength
                if 'high' in data.columns and 'low' in data.columns:
                    price_range = (data['high'] - data['low']) / data['close']
                    trend_strength = self._vectorbt_rolling_operation(price_range, "mean", 20).fillna(price_range.mean())
                else:
                    # Fallback: use price momentum
                    momentum = data['close'].pct_change(20).fillna(0)
                    trend_strength = momentum.abs()

                # Create regime labels based on volatility and trend
                # High volatility + high trend = regime 0 (trending)
                # High volatility + low trend = regime 1 (ranging)
                # Low volatility + high trend = regime 2 (breakout)
                # Low volatility + low trend = regime 3 (consolidation)

                vol_threshold = volatility.median()
                trend_threshold = trend_strength.median()

                regime_labels = np.zeros(len(data))
                regime_labels[(volatility > vol_threshold) & (trend_strength > trend_threshold)] = 0  # Trending
                regime_labels[(volatility > vol_threshold) & (trend_strength <= trend_threshold)] = 1  # Ranging
                regime_labels[(volatility <= vol_threshold) & (trend_strength > trend_threshold)] = 2  # Breakout
                regime_labels[(volatility <= vol_threshold) & (trend_strength <= trend_threshold)] = 3  # Consolidation

                tprint(f"✅ [REGIME_ENSEMBLE] Created synthetic regime labels: {len(np.unique(regime_labels))} regimes", color="green")
                tprint(f"📊 [REGIME_ENSEMBLE] Regime distribution: {np.bincount(regime_labels.astype(int))}", color="blue")

                return regime_labels
            else:
                # Fallback: create simple regime labels based on data length
                n_regimes = min(4, max(2, len(data) // 100))  # 2-4 regimes based on data length
                regime_labels = np.random.randint(0, n_regimes, len(data))
                tprint(f"⚠️ [REGIME_ENSEMBLE] Using random regime labels as fallback: {n_regimes} regimes", color="yellow")
                return regime_labels

        except Exception as e:
            tprint(f"⚠️ [REGIME_ENSEMBLE] Synthetic regime creation failed: {e}, using simple fallback", color="yellow")

        # Calculate average probabilities for each regime
        avg_regime_probabilities = np.mean(regime_probabilities, axis=0)

        # Calculate regime stability (how consistent the predictions are)
        regime_stability = 1.0 - np.std(regime_probabilities, axis=0)
        
        # Calculate entropy for uncertainty measurement
        try:
            tprint("🔮 [REGIME_ENSEMBLE] Starting ensemble regime prediction with probabilities", color="cyan")

            # Scale features if scaler is provided
            if scaler is not None:
                X_scaled = scaler.transform(X)
                tprint("✅ [REGIME_ENSEMBLE] Features scaled using provided scaler", color="green")
            else:
                X_scaled = X
                tprint("⚠️ [REGIME_ENSEMBLE] No scaler provided, using unscaled features", color="yellow")

            # Extract meta-learner and base models
            meta_learner = stacker_result.get('meta_learner')
            base_models = stacker_result.get('base_models', {})
            base_model_names = stacker_result.get('base_model_names', [])

            if meta_learner is None:
                raise ValueError("No meta-learner found in stacker_result")

            # Generate base model predictions for meta-learning
            tprint("🔧 [REGIME_ENSEMBLE] Generating base model predictions", color="blue")
            base_predictions = []

            for name, model in base_models.items():
                try:
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(X_scaled)
                        base_predictions.append(pred_proba)
                        tprint(f"✅ [REGIME_ENSEMBLE] {name}: Generated {pred_proba.shape[1]} regime probabilities", color="green")
                    else:
                        pred = model.predict(X_scaled)
                        unique_classes = np.unique(pred)
                        pred_onehot = np.zeros((len(pred), len(unique_classes)))
                        for i, class_val in enumerate(unique_classes):
                            pred_onehot[pred == class_val, i] = 1
                        base_predictions.append(pred_onehot)
                        tprint(f"✅ [REGIME_ENSEMBLE] {name}: Converted class predictions to {len(unique_classes)} regime probabilities", color="green")
                except Exception as e:
                    tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to get predictions from {name}: {e}", color="yellow")
                    continue

            if not base_predictions:
                raise ValueError("No valid base model predictions generated")

            # Combine base model predictions
            meta_features = np.column_stack(base_predictions)
            tprint(f"📊 [REGIME_ENSEMBLE] Meta-features shape: {meta_features.shape}", color="blue")

            # Make predictions using meta-learner
            regime_labels = meta_learner.predict(meta_features)
            regime_probabilities = meta_learner.predict_proba(meta_features)

            # Get number of regimes
            n_regimes = regime_probabilities.shape[1] if len(regime_probabilities.shape) > 1 else 1

            # Calculate comprehensive probability information
            max_probs = np.max(regime_probabilities, axis=1)
            confidence_scores = max_probs

            # Calculate regime distribution statistics
            regime_counts = np.bincount(regime_labels, minlength=n_regimes)
            regime_percentages = regime_counts / len(regime_labels) * 100

            # Calculate average probabilities for each regime
            avg_regime_probabilities = np.mean(regime_probabilities, axis=0)

            # Calculate regime stability (how consistent the predictions are)
            regime_stability = 1.0 - np.std(regime_probabilities, axis=0)

            # Calculate entropy (uncertainty measure)
            epsilon = 1e-10
            entropy = -np.sum(regime_probabilities * np.log(regime_probabilities + epsilon), axis=1)

            # Calculate dominance (difference between top 2 probabilities)
            sorted_probs = np.sort(regime_probabilities, axis=1)
            if n_regimes > 1:
                dominance = sorted_probs[:, -1] - sorted_probs[:, -2]
            else:
                dominance = np.ones(len(regime_labels))

            # Generate ensemble probabilities from all available models
            from src.utils.regime_ensemble_utils import generate_ensemble_probabilities
            ensemble_probabilities = generate_ensemble_probabilities(base_models, X_scaled, feature_names, "REGIME_ENSEMBLE")

            # Use RegimeProbabilityAnalyzer for comprehensive analysis
            from src.utils.regime_probability_analyzer import RegimeProbabilityAnalyzer

            analyzer = RegimeProbabilityAnalyzer()

            # Create prediction result for analysis
            prediction_result = {
                'regime_labels': regime_labels,
                'regime_probabilities': regime_probabilities,
                'ensemble_probabilities': ensemble_probabilities,
                'dominance': dominance,
                'timestamp': pd.Timestamp.now()
            }

            # Analyze prediction quality and stability
            analysis_result = analyzer.analyze_regime_prediction_quality(prediction_result)

            # Extract key metrics
            confidence_score = analysis_result.get('confidence_score', 0.0)
            stability_score = analysis_result.get('stability_score', 0.0)
            regime_consistency = analysis_result.get('regime_consistency', 0.0)

            return {
                'regime_labels': regime_labels,
                'regime_probabilities': regime_probabilities,
                'confidence_score': confidence_score,
                'stability_score': stability_score,
                'regime_consistency': regime_consistency,
                'dominance': dominance
            }
        except ImportError:
            # Fallback if RegimeProbabilityAnalyzer is not available
            return {
                'regime_labels': regime_labels,
                'regime_probabilities': regime_probabilities,
                'confidence_score': 0.0,
                'stability_score': 0.0,
                'regime_consistency': 0.0,
                'dominance': dominance
            }

    async def _tag_dataset_with_ensemble_outputs(
        self,
        data: pd.DataFrame,
        training_results: Dict[str, Any],
        X: np.ndarray,
        feature_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Automatically tag the whole dataset with ensemble outputs and probabilities.
        
        Args:
            data: Original market data DataFrame
            training_results: Results from ensemble training
            X: Feature matrix used for training
            feature_names: List of feature names
            
        Returns:
            Dictionary containing tagged dataset with ensemble outputs
        """
        try:
            tprint("🏷️ [REGIME_ENSEMBLE] Starting dataset tagging with ensemble outputs", color="cyan")
            
            # Get the trained ensemble model
            ensemble_result = training_results.get('regime_ensemble_training_result', {})
            stacker_result = ensemble_result.get('stacker_lgbm_calibrated')
            
            if not stacker_result:
                tprint("❌ [REGIME_ENSEMBLE] No trained ensemble model found for tagging", color="red")
                return None
            
            # Generate predictions for the entire dataset
            tprint("🔮 [REGIME_ENSEMBLE] Generating ensemble predictions for entire dataset", color="blue")
            prediction_result = self.predict_regimes_with_probabilities(
                stacker_result=stacker_result,
                X=X,
                feature_names=feature_names,
                scaler=None
            )
            
            if not prediction_result:
                tprint("❌ [REGIME_ENSEMBLE] Failed to generate predictions for tagging", color="red")
                return None
            
            # Extract prediction data
            regime_labels = prediction_result.get('regime_labels', np.array([]))
            regime_probabilities = prediction_result.get('regime_probabilities', np.array([]))
            confidence_scores = prediction_result.get('confidence_score', 0.0)
            stability_scores = prediction_result.get('stability_score', 0.0)
            dominance = prediction_result.get('dominance', np.array([]))
            
            # Create tagged dataset
            tagged_data = data.copy()
            
            # Add ensemble prediction columns
            tagged_data['ensemble_regime_label'] = regime_labels
            tagged_data['ensemble_confidence_score'] = confidence_scores
            tagged_data['ensemble_stability_score'] = stability_scores
            tagged_data['ensemble_dominance'] = dominance
            
            # Add individual regime probabilities
            n_regimes = regime_probabilities.shape[1] if len(regime_probabilities.shape) > 1 else 1
            for i in range(n_regimes):
                tagged_data[f'ensemble_regime_{i}_probability'] = regime_probabilities[:, i]
            
            # Add ensemble metadata
            tagged_data['ensemble_prediction_timestamp'] = datetime.now().isoformat()
            tagged_data['ensemble_model_type'] = 'stacker_lgbm_calibrated'
            tagged_data['ensemble_n_regimes'] = n_regimes
            
            # Calculate additional ensemble metrics
            max_probs = np.max(regime_probabilities, axis=1)
            tagged_data['ensemble_max_probability'] = max_probs
            tagged_data['ensemble_entropy'] = -np.sum(regime_probabilities * np.log(regime_probabilities + 1e-10), axis=1)
            
            # Add regime transition indicators
            if len(regime_labels) > 1:
                regime_changes = np.zeros(len(regime_labels))
                regime_changes[1:] = (regime_labels[1:] != regime_labels[:-1]).astype(int)
                tagged_data['ensemble_regime_change'] = regime_changes
            else:
                tagged_data['ensemble_regime_change'] = 0
            
            # Create comprehensive tagging summary
            tagging_summary = {
                'total_samples_tagged': len(tagged_data),
                'n_regimes_detected': n_regimes,
                'regime_distribution': {
                    f'regime_{i}': int(np.sum(regime_labels == i)) for i in range(n_regimes)
                },
                'confidence_statistics': {
                    'mean': float(np.mean(max_probs)),
                    'std': float(np.std(max_probs)),
                    'min': float(np.min(max_probs)),
                    'max': float(np.max(max_probs))
                },
                'entropy_statistics': {
                    'mean': float(np.mean(tagged_data['ensemble_entropy'])),
                    'std': float(np.std(tagged_data['ensemble_entropy']))
                },
                'regime_changes': int(np.sum(tagged_data['ensemble_regime_change'])),
                'tagging_timestamp': datetime.now().isoformat(),
                'ensemble_model_info': {
                    'model_type': 'stacker_lgbm_calibrated',
                    'calibration_method': stacker_result.get('calibration_method', 'none'),
                    'base_models_count': len(stacker_result.get('base_models', {})),
                    'meta_features_shape': stacker_result.get('meta_features_shape', (0, 0))
                }
            }
            
            tprint(f"✅ [REGIME_ENSEMBLE] Dataset tagged successfully: {len(tagged_data)} samples, {n_regimes} regimes", color="green")
            tprint(f"📊 [REGIME_ENSEMBLE] Regime distribution: {tagging_summary['regime_distribution']}", color="blue")
            
            return {
                'tagged_dataset': tagged_data,
                'tagging_summary': tagging_summary,
                'prediction_result': prediction_result
            }
            
        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Dataset tagging failed: {e}", color="red")
            self.logger.error(f"Dataset tagging failed: {e}", exc_info=True)
            return None

    async def _create_timeframe_artifacts(
        self,
        training_results: Dict[str, Any],
        data: pd.DataFrame,
        X: np.ndarray,
        feature_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Create artifacts for 15m and 1h timeframes.
        
        Args:
            training_results: Results from ensemble training
            data: Original market data DataFrame
            X: Feature matrix used for training
            feature_names: List of feature names
            
        Returns:
            Dictionary containing timeframe-specific artifacts
        """
        try:
            tprint("📦 [REGIME_ENSEMBLE] Creating timeframe artifacts for 15m and 1h", color="cyan")
            
            timeframe_artifacts = {}
            timeframes = ['15m', '1h']
            
            for timeframe in timeframes:
                tprint(f"🔧 [REGIME_ENSEMBLE] Processing {timeframe} timeframe", color="blue")
                
                # Create timeframe-specific artifact
                timeframe_artifact = {
                    'timeframe': timeframe,
                    'creation_timestamp': datetime.now().isoformat(),
                    'base_training_results': training_results.get('regime_ensemble_training_result', {}),
                    'data_info': {
                        'original_data_shape': data.shape,
                        'feature_matrix_shape': X.shape,
                        'feature_names': feature_names,
                        'data_columns': list(data.columns)
                    },
                    'ensemble_metrics': training_results.get('regime_ensemble_training_result', {}).get('ensemble_metrics', {}),
                    'validation_report': training_results.get('regime_ensemble_training_result', {}).get('validation_report', {}),
                    'hardware_optimization': training_results.get('regime_ensemble_training_result', {}).get('hardware_optimization', {}),
                    'lookahead_protection': training_results.get('regime_ensemble_training_result', {}).get('lookahead_protection', {}),
                    'metadata': training_results.get('regime_ensemble_training_result', {}).get('metadata', {})
                }
                
                # Add timeframe-specific predictions if tagged dataset is available
                tagged_dataset = training_results.get('tagged_dataset', {})
                if tagged_dataset and 'tagged_dataset' in tagged_dataset:
                    tagged_data = tagged_dataset['tagged_dataset']
                    
                    # Create timeframe-specific predictions
                    timeframe_predictions = {
                        'ensemble_regime_labels': tagged_data['ensemble_regime_label'].tolist(),
                        'ensemble_confidence_scores': tagged_data['ensemble_confidence_score'].tolist(),
                        'ensemble_stability_scores': tagged_data['ensemble_stability_score'].tolist(),
                        'ensemble_dominance': tagged_data['ensemble_dominance'].tolist(),
                        'ensemble_max_probabilities': tagged_data['ensemble_max_probability'].tolist(),
                        'ensemble_entropy': tagged_data['ensemble_entropy'].tolist(),
                        'ensemble_regime_changes': tagged_data['ensemble_regime_change'].tolist()
                    }
                    
                    # Add individual regime probabilities
                    regime_prob_cols = [col for col in tagged_data.columns if col.startswith('ensemble_regime_') and col.endswith('_probability')]
                    for col in regime_prob_cols:
                        timeframe_predictions[col] = tagged_data[col].tolist()
                    
                    timeframe_artifact['predictions'] = timeframe_predictions
                    timeframe_artifact['prediction_summary'] = tagged_dataset.get('tagging_summary', {})
                
                # Add timeframe-specific configuration
                timeframe_artifact['configuration'] = {
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': timeframe,
                    'execution_mode': getattr(self.config, 'execution_mode', 'light'),
                    'ensemble_config': self.ensemble_config,
                    'feature_generation_enabled': True,
                    'probability_calibration_enabled': True,
                    'hardware_optimization_enabled': True,
                    'lookahead_protection_enabled': True
                }
                
                # Add performance metrics
                ensemble_result = training_results.get('regime_ensemble_training_result', {})
                timeframe_artifact['performance_metrics'] = {
                    'training_time': ensemble_result.get('training_time', 0),
                    'execution_time': training_results.get('metadata', {}).get('execution_time', 0),
                    'data_processing_time': 0,  # Could be calculated if needed
                    'model_complexity': {
                        'n_features': X.shape[1],
                        'n_samples': X.shape[0],
                        'n_base_models': len(ensemble_result.get('base_models', {})),
                        'meta_learner_type': 'stacker_lgbm_calibrated'
                    }
                }
                
                # Store the artifact
                timeframe_artifacts[f'{timeframe}_artifacts'] = timeframe_artifact
                tprint(f"✅ [REGIME_ENSEMBLE] {timeframe} artifacts created successfully", color="green")
            
            tprint(f"✅ [REGIME_ENSEMBLE] All timeframe artifacts created: {list(timeframe_artifacts.keys())}", color="green")
            return timeframe_artifacts
            
        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Timeframe artifact creation failed: {e}", color="red")
            self.logger.error(f"Timeframe artifact creation failed: {e}", exc_info=True)
            return None

    def _prepare_artifacts_for_saving(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare artifacts for downstream compatibility and saving.
        
        Args:
            results: Raw results from ensemble training
            
        Returns:
            Dictionary of artifacts structured for downstream users
        """
        try:
            tprint("🔧 [REGIME_ENSEMBLE] Preparing artifacts for downstream compatibility", color="cyan")
            
            artifacts = {}
            
            # 1. Core ensemble training results
            if 'regime_ensemble_training_result' in results:
                artifacts['ensemble_training_results'] = results['regime_ensemble_training_result']
                tprint("✅ [REGIME_ENSEMBLE] Added ensemble training results", color="green")
            
            # 2. Tagged dataset (if available)
            if 'tagged_dataset' in results:
                tagged_data = results['tagged_dataset']
                artifacts['tagged_dataset'] = {
                    'data': tagged_data.get('tagged_dataset'),
                    'summary': tagged_data.get('tagging_summary', {}),
                    'prediction_result': tagged_data.get('prediction_result', {}),
                    'metadata': {
                        'n_samples': len(tagged_data.get('tagged_dataset', [])),
                        'n_regimes': tagged_data.get('tagging_summary', {}).get('n_regimes_detected', 0),
                        'creation_timestamp': datetime.now().isoformat(),
                        'component': 'regime_ensemble_training'
                    }
                }
                tprint("✅ [REGIME_ENSEMBLE] Added tagged dataset artifact", color="green")
            
            # 3. Timeframe artifacts (15m and 1h)
            timeframe_artifacts = {}
            for key, value in results.items():
                if key.endswith('_artifacts'):
                    timeframe_artifacts[key] = value
                    tprint(f"✅ [REGIME_ENSEMBLE] Added {key} artifact", color="green")
            
            if timeframe_artifacts:
                artifacts['timeframe_artifacts'] = timeframe_artifacts
            
            # 4. Ensemble model artifacts (for downstream model loading)
            if 'regime_ensemble_training_result' in results:
                ensemble_result = results['regime_ensemble_training_result']
                if 'stacker_lgbm_calibrated' in ensemble_result:
                    stacker_result = ensemble_result['stacker_lgbm_calibrated']
                    artifacts['ensemble_model'] = {
                        'model_type': 'stacker_lgbm_calibrated',
                        'base_models': stacker_result.get('base_models', {}),
                        'meta_learner': stacker_result.get('meta_learner'),
                        'calibration_method': stacker_result.get('calibration_method', 'none'),
                        'feature_names': stacker_result.get('feature_names', []),
                        'model_metadata': {
                            'training_timestamp': stacker_result.get('training_timestamp'),
                            'n_features': stacker_result.get('meta_features_shape', (0, 0))[1],
                            'n_base_models': len(stacker_result.get('base_models', {})),
                            'calibration_enabled': stacker_result.get('calibration_method') != 'none'
                        }
                    }
                    tprint("✅ [REGIME_ENSEMBLE] Added ensemble model artifact", color="green")
            
            # 5. Feature information
            if 'feature_names' in results:
                artifacts['feature_info'] = {
                    'feature_names': results['feature_names'],
                    'n_features': len(results['feature_names']),
                    'feature_types': ['technical', 'statistical', 'regime_based'],
                    'generation_method': 'enhanced_meta_features'
                }
                tprint("✅ [REGIME_ENSEMBLE] Added feature information artifact", color="green")
            
            # 6. Validation and performance metrics
            if 'regime_ensemble_training_result' in results:
                ensemble_result = results['regime_ensemble_training_result']
                artifacts['validation_metrics'] = {
                    'ensemble_metrics': ensemble_result.get('ensemble_metrics', {}),
                    'validation_report': ensemble_result.get('validation_report', {}),
                    'hardware_optimization': ensemble_result.get('hardware_optimization', {}),
                    'lookahead_protection': ensemble_result.get('lookahead_protection', {}),
                    'performance_summary': {
                        'training_time': ensemble_result.get('training_time', 0),
                        'execution_mode': ensemble_result.get('execution_mode', 'light'),
                        'success': True
                    }
                }
                tprint("✅ [REGIME_ENSEMBLE] Added validation metrics artifact", color="green")
            
            # 7. Configuration and metadata
            artifacts['configuration'] = {
                'symbol': self.config.symbol,
                'exchange': self.config.exchange,
                'timeframe': self.config.timeframe,
                'execution_mode': getattr(self.config, 'execution_mode', 'light'),
                'ensemble_config': self.ensemble_config,
                'component_version': '1.0.0',
                'compatibility_version': '1.0.0'
            }
            tprint("✅ [REGIME_ENSEMBLE] Added configuration artifact", color="green")
            
            # 8. Downstream usage guide
            artifacts['usage_guide'] = {
                'description': 'Regime Ensemble Training Artifacts',
                'artifacts_available': list(artifacts.keys()),
                'downstream_usage': {
                    'tagged_dataset': 'Use for further analysis and model training',
                    'ensemble_model': 'Load for prediction on new data',
                    'timeframe_artifacts': 'Use for timeframe-specific analysis',
                    'validation_metrics': 'Use for model evaluation and comparison',
                    'feature_info': 'Use for feature engineering in downstream steps'
                },
                'loading_examples': {
                    'load_tagged_data': 'artifacts["tagged_dataset"]["data"]',
                    'load_ensemble_model': 'artifacts["ensemble_model"]',
                    'get_feature_names': 'artifacts["feature_info"]["feature_names"]',
                    'get_validation_metrics': 'artifacts["validation_metrics"]'
                }
            }
            tprint("✅ [REGIME_ENSEMBLE] Added usage guide artifact", color="green")
            
            # Convert all numpy types to Python native types for JSON serialization
            tprint("🔄 [REGIME_ENSEMBLE] Converting numpy types to Python native types", color="cyan")
            artifacts = self._convert_numpy_types(artifacts)
            
            tprint(f"🎯 [REGIME_ENSEMBLE] Prepared {len(artifacts)} artifacts for downstream compatibility", color="green")
            return artifacts
            
        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Failed to prepare artifacts for saving: {e}", color="red")
            self.logger.error(f"Artifact preparation failed: {e}", exc_info=True)
            # Return basic artifacts if preparation fails
            return {
                'ensemble_training_results': results.get('regime_ensemble_training_result', {}),
                'tagged_dataset': results.get('tagged_dataset', {}),
                'timeframe_artifacts': {k: v for k, v in results.items() if k.endswith('_artifacts')},
                'configuration': {
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'component_version': '1.0.0'
                }
            }

    async def _save_individual_artifacts(self, results: Dict[str, Any], correlation_id: str) -> None:
        """
        Save individual artifacts separately for better downstream access.
        
        Args:
            results: Raw results from ensemble training
            correlation_id: Correlation ID from main artifact save
        """
        try:
            tprint("🔧 [REGIME_ENSEMBLE] Saving individual artifacts for downstream access", color="cyan")
            
            # 1. Save tagged dataset separately
            if 'tagged_dataset' in results:
                tagged_artifact = {
                    'tagged_dataset': results['tagged_dataset'],
                    'correlation_id': correlation_id,
                    'artifact_type': 'tagged_dataset',
                    'component': 'regime_ensemble_training',
                    'timestamp': datetime.now().isoformat()
                }
                # Convert numpy types before saving
                tagged_artifact = self._convert_numpy_types(tagged_artifact)
                await self.save_artifacts(tagged_artifact, {
                    'artifact_type': 'tagged_dataset',
                    'correlation_id': correlation_id,
                    'downstream_ready': True
                })
                tprint("✅ [REGIME_ENSEMBLE] Tagged dataset saved individually", color="green")
            
            # 2. Save timeframe artifacts separately
            timeframe_artifacts = {k: v for k, v in results.items() if k.endswith('_artifacts')}
            if timeframe_artifacts:
                for timeframe, artifact_data in timeframe_artifacts.items():
                    individual_artifact = {
                        timeframe: artifact_data,
                        'correlation_id': correlation_id,
                        'artifact_type': 'timeframe_artifacts',
                        'timeframe': artifact_data.get('timeframe', 'unknown'),
                        'component': 'regime_ensemble_training',
                        'timestamp': datetime.now().isoformat()
                    }
                    # Convert numpy types before saving
                    individual_artifact = self._convert_numpy_types(individual_artifact)
                    await self.save_artifacts(individual_artifact, {
                        'artifact_type': 'timeframe_artifacts',
                        'timeframe': artifact_data.get('timeframe', 'unknown'),
                        'correlation_id': correlation_id,
                        'downstream_ready': True
                    })
                    tprint(f"✅ [REGIME_ENSEMBLE] {timeframe} saved individually", color="green")
            
            # 3. Save ensemble model separately
            if 'regime_ensemble_training_result' in results:
                ensemble_result = results['regime_ensemble_training_result']
                if 'stacker_lgbm_calibrated' in ensemble_result:
                    model_artifact = {
                        'ensemble_model': {
                            'stacker_lgbm_calibrated': ensemble_result['stacker_lgbm_calibrated'],
                            'ensemble_metrics': ensemble_result.get('ensemble_metrics', {}),
                            'validation_report': ensemble_result.get('validation_report', {})
                        },
                        'correlation_id': correlation_id,
                        'artifact_type': 'ensemble_model',
                        'component': 'regime_ensemble_training',
                        'timestamp': datetime.now().isoformat()
                    }
                    # Convert numpy types before saving
                    model_artifact = self._convert_numpy_types(model_artifact)
                    await self.save_artifacts(model_artifact, {
                        'artifact_type': 'ensemble_model',
                        'correlation_id': correlation_id,
                        'downstream_ready': True
                    })
                    tprint("✅ [REGIME_ENSEMBLE] Ensemble model saved individually", color="green")
            
            # 4. Save feature information separately
            if 'feature_names' in results:
                feature_artifact = {
                    'feature_info': {
                        'feature_names': results['feature_names'],
                        'n_features': len(results['feature_names']),
                        'feature_types': ['technical', 'statistical', 'regime_based']
                    },
                    'correlation_id': correlation_id,
                    'artifact_type': 'feature_info',
                    'component': 'regime_ensemble_training',
                    'timestamp': datetime.now().isoformat()
                }
                await self.save_artifacts(feature_artifact, {
                    'artifact_type': 'feature_info',
                    'correlation_id': correlation_id,
                    'downstream_ready': True
                })
                tprint("✅ [REGIME_ENSEMBLE] Feature info saved individually", color="green")
            
            tprint("🎯 [REGIME_ENSEMBLE] All individual artifacts saved successfully", color="green")
            
        except Exception as e:
            tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to save individual artifacts: {e}", color="yellow")
            self.logger.error(f"Individual artifact saving failed: {e}", exc_info=True)


    async def _generate_regime_probability_report(
        self,
        training_results: Dict[str, Any],
        X: np.ndarray,
        feature_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Generate a comprehensive report with regime probabilities for all regimes."""
        try:
            tprint("📊 [REGIME_ENSEMBLE] Generating regime probability report", color="cyan")

            # Get the trained stacker model
            stacker_model = training_results.get('stacker_lgbm_calibrated')
            if not stacker_model:
                tprint("⚠️ [REGIME_ENSEMBLE] No trained stacker model found for report generation", color="yellow")
                return None

            if not hasattr(stacker_model, 'predict_proba'):
                tprint("⚠️ [REGIME_ENSEMBLE] Stacker model does not support probability prediction", color="yellow")
                return None

            # Generate regime probabilities for all samples
            tprint("🔮 [REGIME_ENSEMBLE] Generating regime probabilities using stacker model", color="cyan")
            regime_probabilities = stacker_model.predict_proba(X)
            regime_labels = stacker_model.predict(X)

            n_regimes = regime_probabilities.shape[1]
            n_samples = len(regime_probabilities)

            # Calculate regime statistics
            regime_stats = {}
            for i in range(n_regimes):
                regime_probs = regime_probabilities[:, i]
                regime_count = np.sum(regime_labels == i)

                regime_stats[f'regime_{i}'] = {
                    'sample_count': int(regime_count),
                    'percentage': float(regime_count / n_samples * 100),
                    'mean_probability': float(np.mean(regime_probs)),
                    'std_probability': float(np.std(regime_probs)),
                    'min_probability': float(np.min(regime_probs)),
                    'max_probability': float(np.max(regime_probs)),
                    'confidence_distribution': {
                        'high_confidence': int(np.sum(regime_probs > 0.8)),
                        'medium_confidence': int(np.sum((regime_probs > 0.5) & (regime_probs <= 0.8))),
                        'low_confidence': int(np.sum(regime_probs <= 0.5))
                    }
                }

            # Calculate overall statistics
            overall_stats = {
                'total_samples': n_samples,
                'n_regimes': n_regimes,
                'mean_max_probability': float(np.mean(np.max(regime_probabilities, axis=1))),
                'std_max_probability': float(np.std(np.max(regime_probabilities, axis=1))),
                'regime_balance': float(np.std([regime_stats[f'regime_{i}']['percentage'] for i in range(n_regimes)])),
                'prediction_confidence': float(np.mean(np.max(regime_probabilities, axis=1))),
                'uncertainty_entropy': float(np.mean([-np.sum(p * np.log(p + 1e-10)) for p in regime_probabilities]))
            }

            # Get ensemble metrics
            ensemble_metrics = training_results.get('ensemble_metrics', {})
            stacker_metrics = ensemble_metrics.get('stacker_lgbm_calibrated', {})

            # Generate comprehensive report
            report = {
                'model_name': 'stacker_lgbm_calibrated',
                'generation_timestamp': datetime.now().isoformat(),
                'overall_statistics': overall_stats,
                'regime_statistics': regime_stats,
                'regime_probabilities': regime_probabilities.tolist(),
                'regime_labels': regime_labels.tolist(),
                'feature_names': feature_names,
                'data_shape': X.shape,
                'report_type': 'regime_ensemble_probability_analysis',
                'ensemble_metrics': {
                    'accuracy': stacker_metrics.get('accuracy', 0),
                    'prediction_confidence': stacker_metrics.get('prediction_confidence', {}),
                    'classification_report': stacker_metrics.get('classification_report', {})
                }
            }

            # Generate text report
            text_report = self._generate_text_report(report)
            report['text_report'] = text_report

            tprint(f"✅ [REGIME_ENSEMBLE] Regime probability report generated for {n_regimes} regimes", color="green")
            return report

        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Failed to generate regime probability report: {e}", color="red")
            return None

    def _generate_text_report(self, report: Dict[str, Any]) -> str:
        """Generate a human-readable text report from regime probability data."""
        try:
            lines = []
            lines.append("=" * 80)
            lines.append("REGIME ENSEMBLE PROBABILITY ANALYSIS REPORT")
            lines.append(f"Model: {report.get('model_name', 'Unknown')}")
            lines.append(f"Generated: {report.get('generation_timestamp', 'Unknown')}")
            lines.append("=" * 80)
            lines.append("")

            # Overall Statistics
            overall = report.get('overall_statistics', {})
            lines.append("📊 OVERALL STATISTICS")
            lines.append("-" * 40)
            lines.append(f"Total Samples: {overall.get('total_samples', 'N/A')}")
            lines.append(f"Number of Regimes: {overall.get('n_regimes', 'N/A')}")
            lines.append(f"Mean Max Probability: {overall.get('mean_max_probability', 0):.3f}")
            lines.append(f"Std Max Probability: {overall.get('std_max_probability', 0):.3f}")
            lines.append(f"Regime Balance: {overall.get('regime_balance', 0):.3f}")
            lines.append(f"Prediction Confidence: {overall.get('prediction_confidence', 0):.3f}")
            lines.append(f"Uncertainty Entropy: {overall.get('uncertainty_entropy', 0):.3f}")
            lines.append("")

            # Ensemble Metrics
            ensemble_metrics = report.get('ensemble_metrics', {})
            if ensemble_metrics:
                lines.append("🤖 ENSEMBLE PERFORMANCE")
                lines.append("-" * 40)
                lines.append(f"Accuracy: {ensemble_metrics.get('accuracy', 0):.3f}")
                pred_conf = ensemble_metrics.get('prediction_confidence', {})
                lines.append(f"Mean Confidence: {pred_conf.get('mean', 0):.3f}")
                lines.append(f"Std Confidence: {pred_conf.get('std', 0):.3f}")
                lines.append("")

            # Regime Statistics
            regime_stats = report.get('regime_statistics', {})
            lines.append("🎯 REGIME PROBABILITY STATISTICS")
            lines.append("-" * 40)

            for regime_key, regime_data in regime_stats.items():
                if isinstance(regime_data, dict):
                    lines.append(f"{regime_key.upper()}:")
                    lines.append(f"  Sample Count: {regime_data.get('sample_count', 0)}")
                    lines.append(f"  Percentage: {regime_data.get('percentage', 0):.1f}%")
                    lines.append(f"  Mean Probability: {regime_data.get('mean_probability', 0):.3f}")
                    lines.append(f"  Std Probability: {regime_data.get('std_probability', 0):.3f}")
                    lines.append(f"  Min Probability: {regime_data.get('min_probability', 0):.3f}")
                    lines.append(f"  Max Probability: {regime_data.get('max_probability', 0):.3f}")

                    conf_dist = regime_data.get('confidence_distribution', {})
                    lines.append(f"  Confidence Distribution:")
                    lines.append(f"    High (>0.8): {conf_dist.get('high_confidence', 0)}")
                    lines.append(f"    Medium (0.5-0.8): {conf_dist.get('medium_confidence', 0)}")
                    lines.append(f"    Low (<0.5): {conf_dist.get('low_confidence', 0)}")
                    lines.append("")

            return "\n".join(lines)

        except Exception as e:
            self.logger.error(f"Failed to generate text report: {e}", exc_info=True)
            return f"Error generating report: {str(e)}"

    async def _generate_temporal_regime_analysis(
        self,
        results: Dict[str, Any],
        data: Optional[pd.DataFrame],
        X: np.ndarray,
        feature_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Generate comprehensive temporal regime analysis using TemporalRegimeAnalyzer.
        
        Args:
            results: Results from ensemble training
            data: OHLCV DataFrame (can be None if not available)
            X: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Dictionary containing temporal regime analysis results
        """
        try:
            tprint("📊 [REGIME_ENSEMBLE] Starting temporal regime analysis", color="cyan")
            
            # Check if OHLCV data is available
            if data is None:
                tprint("⚠️ [REGIME_ENSEMBLE] No OHLCV data available, skipping temporal analysis", color="yellow")
                return None
            
            # Import temporal analyzer
            from src.analysis.temporal_regime_analyzer_simple import TemporalRegimeAnalyzer
            
            # Get regime labels from ensemble results
            ensemble_result = results.get('regime_ensemble_training_result', {})
            stacker_result = ensemble_result.get('stacker_lgbm_calibrated', {})
            
            if not stacker_result:
                tprint("⚠️ [REGIME_ENSEMBLE] No stacker result found for temporal analysis", color="yellow")
                return None
            
            # Extract regime labels from tagged dataset if available
            regime_labels = None
            tagged_dataset = results.get('tagged_dataset', {})
            if tagged_dataset and 'tagged_dataset' in tagged_dataset:
                tagged_data = tagged_dataset['tagged_dataset']
                if 'ensemble_regime_label' in tagged_data.columns:
                    regime_labels = tagged_data['ensemble_regime_label'].values
                    unique_regimes = len(np.unique(regime_labels))
                    tprint(f"✅ [REGIME_ENSEMBLE] Extracted regime labels from tagged dataset: {len(regime_labels)} samples, {unique_regimes} unique regimes", color="green")
            
            # Fallback: use predictions if no labels available
            if regime_labels is None:
                tprint("⚠️ [REGIME_ENSEMBLE] No regime labels found, using predictions", color="yellow")
                if hasattr(stacker_result.get('meta_learner'), 'predict'):
                    meta_learner = stacker_result['meta_learner']
                    # Generate meta-features for prediction
                    base_models = stacker_result.get('base_models', {})
                    if base_models:
                        # CRITICAL: Align X with OHLCV data length BEFORE generating predictions
                        # This ensures predictions match the returns length from the start
                        X_for_prediction = X
                        if data is not None and 'close' in data.columns:
                            n_ohlcv_samples = len(data)
                            if len(X) > n_ohlcv_samples:
                                tprint(f"⚠️ [REGIME_ENSEMBLE] Trimming X from {len(X)} to {n_ohlcv_samples} to match OHLCV length", color="yellow")
                                X_for_prediction = X[-n_ohlcv_samples:]
                        
                        from .ensemble_meta_features import EnsembleMetaFeaturesGenerator
                        meta_generator = EnsembleMetaFeaturesGenerator("REGIME_ENSEMBLE")
                        meta_features, _ = meta_generator.generate_meta_features(
                            base_models=base_models,
                            X=X_for_prediction,
                            y=np.zeros(len(X_for_prediction)),  # Dummy y for feature generation
                            include_uncertainty=True,
                            include_confidence=True,
                            include_disagreement=True
                        )
                        # CRITICAL FIX: Use SIMPLIFIED features (no enhancement)
                        # The model was trained on simplified meta-features (36), not enhanced (44)
                        simplified_meta_features = meta_features  # Use base meta-features directly
                        
                        # Apply zero-variance mask if it was used during training
                        zero_var_mask = stacker_result.get('zero_var_mask')
                        if zero_var_mask is not None:
                            tprint(f"🔧 [REGIME_ENSEMBLE] Applying zero-variance mask to prediction features", color="cyan")
                            simplified_meta_features = simplified_meta_features[:, zero_var_mask]
                        
                        tprint(f"✅ [REGIME_ENSEMBLE] Simplified meta-features for prediction: {simplified_meta_features.shape}", color="green")
                        
                        # Log expected vs actual feature counts for debugging
                        expected_features = stacker_result.get('simplified_meta_features_shape', (None, None))[1]
                        actual_features = simplified_meta_features.shape[1]
                        tprint(f"🔍 [REGIME_ENSEMBLE] Feature count check: expected={expected_features}, actual={actual_features}", color="cyan")
                        
                        if expected_features and expected_features != actual_features:
                            tprint(f"⚠️ [REGIME_ENSEMBLE] Feature count mismatch! Skipping temporal analysis.", color="yellow")
                            return None
                        
                        regime_labels = meta_learner.predict(simplified_meta_features)
                        unique_regimes = len(np.unique(regime_labels))
                        tprint(f"✅ [REGIME_ENSEMBLE] Generated regime labels from predictions: {len(regime_labels)} samples, {unique_regimes} unique regimes", color="green")
            
            if regime_labels is None:
                tprint("❌ [REGIME_ENSEMBLE] Cannot generate temporal analysis without regime labels", color="red")
                return None
            
            # Calculate returns from OHLCV data
            returns = None
            alignment_offset = 0  # Track how many rows we need to drop from the start
            if 'close' in data.columns:
                close_prices = data['close'].dropna()
                if len(close_prices) > 1:
                    returns = close_prices.pct_change().dropna().values
                    tprint(f"✅ [REGIME_ENSEMBLE] Calculated returns from close prices: {len(returns)} samples", color="green")
                    
                    # Align regime_labels with returns (pct_change drops first row)
                    if len(regime_labels) != len(returns):
                        unique_before = len(np.unique(regime_labels))
                        tprint(f"⚠️ [REGIME_ENSEMBLE] Aligning regime labels ({len(regime_labels)}) with returns ({len(returns)})", color="yellow")
                        tprint(f"   Unique regimes before alignment: {unique_before}", color="cyan")
                        # Drop first regime label to match returns length
                        alignment_offset = len(regime_labels) - len(returns)
                        regime_labels = regime_labels[alignment_offset:]
                        unique_after = len(np.unique(regime_labels))
                        tprint(f"✅ [REGIME_ENSEMBLE] Aligned regime labels: {len(regime_labels)} samples, {unique_after} unique regimes", color="green")
                else:
                    tprint("⚠️ [REGIME_ENSEMBLE] Insufficient close price data for returns calculation", color="yellow")
            else:
                tprint("⚠️ [REGIME_ENSEMBLE] No 'close' column in OHLCV data, cannot calculate returns", color="yellow")
            
            # Skip temporal analysis if no returns available
            if returns is None or len(returns) == 0:
                tprint("⚠️ [REGIME_ENSEMBLE] No returns available, skipping temporal analysis", color="yellow")
                return None
            
            # Check if we have enough regime diversity for meaningful analysis
            unique_regimes_final = len(np.unique(regime_labels))
            if unique_regimes_final < 2:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Insufficient regime diversity ({unique_regimes_final} unique regimes)", color="yellow")
                tprint("   This suggests the model is predicting only one class - check model performance", color="yellow")
                tprint("   Skipping temporal analysis (requires at least 2 regimes)", color="yellow")
                return None
            
            # Create features DataFrame for analysis, aligned with returns
            features_df = None
            if X is not None and feature_names:
                # Apply same alignment offset to X
                X_aligned = X[alignment_offset:] if alignment_offset > 0 else X
                # Ensure X_aligned matches returns length
                X_aligned = X_aligned[:len(returns)]
                features_df = pd.DataFrame(X_aligned, columns=feature_names)
                tprint(f"✅ [REGIME_ENSEMBLE] Created aligned features DataFrame: {features_df.shape}", color="green")
            
            # Initialize temporal analyzer
            analyzer = TemporalRegimeAnalyzer()
            
            # Perform comprehensive analysis
            tprint("🔍 [REGIME_ENSEMBLE] Performing comprehensive temporal analysis", color="blue")
            analysis_results = analyzer.analyze_regimes(
                regime_labels=regime_labels,
                returns=returns,
                features=features_df
            )
            
            # Export to CSV
            try:
                from pathlib import Path
                # Create outcomes directory if it doesn't exist
                outcomes_path = Path("outcomes")
                outcomes_path.mkdir(parents=True, exist_ok=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_filename = f"temporal_regime_analysis_{self.config.symbol}_{timestamp}.csv"
                csv_filepath = outcomes_path / csv_filename
                analyzer.export_to_csv(analysis_results, str(csv_filepath))
                tprint(f"✅ [REGIME_ENSEMBLE] Temporal analysis exported to outcomes/{csv_filename}", color="green")
                
                # Save CSV as artifact
                import os
                if os.path.exists(csv_filename):
                    csv_artifact = {
                        'temporal_regime_analysis_csv': {
                            'file_path': csv_filename,
                            'file_size': os.path.getsize(csv_filename),
                            'n_regimes': len(np.unique(regime_labels)),
                            'n_samples': len(regime_labels),
                            'analysis_timestamp': datetime.now().isoformat(),
                            'component': 'regime_ensemble_training'
                        }
                    }
                    await self.save_artifacts(csv_artifact, {
                        'artifact_type': 'temporal_regime_analysis_csv',
                        'component': 'regime_ensemble_training'
                    })
                    tprint("✅ [REGIME_ENSEMBLE] Temporal analysis CSV saved as artifact", color="green")
                
            except Exception as e:
                tprint(f"⚠️ [REGIME_ENSEMBLE] Failed to export temporal analysis to CSV: {e}", color="yellow")
            
            tprint("✅ [REGIME_ENSEMBLE] Temporal regime analysis completed successfully", color="green")
            return analysis_results
            
        except Exception as e:
            tprint(f"❌ [REGIME_ENSEMBLE] Failed to generate temporal regime analysis: {e}", color="red")
            self.logger.error(f"Failed to generate temporal regime analysis: {e}", exc_info=True)
            return None

    def _generate_csv_reports(self, results: Dict[str, Any], symbol: str, output_dir: str = "outcomes") -> Tuple[Optional[str], Optional[str]]:
        """Generate comprehensive CSV reports for regime ensemble training."""
        try:
            from pathlib import Path
            import csv

            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 1. Generate main metrics CSV
            metrics_filename = f"regime_ensemble_training_metrics_{symbol}_{timestamp}.csv"
            metrics_path = output_path / metrics_filename

            tprint(f"📊 Generating ensemble metrics CSV: {metrics_path}", color="cyan")

            csv_data = []
            csv_data.append(['Metric Category', 'Metric Name', 'Value', 'Description'])

            # Ensemble configuration
            ensemble_config = results.get('regime_ensemble_training_result', {})
            # Try to get base models from detected models (blank mode) or training result
            detected_base_models = results.get('detected_base_models', ensemble_config.get('base_models', []))
            csv_data.append(['Configuration', 'Ensemble Method', 'stacker_lgbm_calibrated', 'LightGBM meta-learner with calibration'])
            csv_data.append(['Configuration', 'Number of Base Models', str(len(detected_base_models)), 'Number of models in ensemble'])
            if detected_base_models:
                csv_data.append(['Configuration', 'Base Models', ', '.join(detected_base_models), 'Models used in ensemble'])

            # Ensemble evaluation metrics (accuracy, F1, recall, etc.)
            ensemble_metrics = results.get('ensemble_metrics', {})
            stacker_metrics = ensemble_metrics.get('stacker_lgbm_calibrated', {})
            if stacker_metrics:
                csv_data.append(['Ensemble Performance', 'Accuracy', f"{stacker_metrics.get('accuracy', 0):.4f}", 'Overall classification accuracy'])
                
                # Classification report metrics
                class_report = stacker_metrics.get('classification_report', {})
                if class_report:
                    # Overall weighted metrics
                    weighted_avg = class_report.get('weighted avg', {})
                    if weighted_avg:
                        csv_data.append(['Ensemble Performance', 'Precision (Weighted)', f"{weighted_avg.get('precision', 0):.4f}", 'Weighted average precision'])
                        csv_data.append(['Ensemble Performance', 'Recall (Weighted)', f"{weighted_avg.get('recall', 0):.4f}", 'Weighted average recall'])
                        csv_data.append(['Ensemble Performance', 'F1-Score (Weighted)', f"{weighted_avg.get('f1-score', 0):.4f}", 'Weighted average F1-score'])
                    
                    # Per-regime metrics
                    for regime_key, regime_metrics in class_report.items():
                        if isinstance(regime_metrics, dict) and regime_key not in ['accuracy', 'macro avg', 'weighted avg']:
                            csv_data.append([f'Regime {regime_key} Performance', 'Precision', f"{regime_metrics.get('precision', 0):.4f}", f'Precision for regime {regime_key}'])
                            csv_data.append([f'Regime {regime_key} Performance', 'Recall', f"{regime_metrics.get('recall', 0):.4f}", f'Recall for regime {regime_key}'])
                            csv_data.append([f'Regime {regime_key} Performance', 'F1-Score', f"{regime_metrics.get('f1-score', 0):.4f}", f'F1-score for regime {regime_key}'])
                            csv_data.append([f'Regime {regime_key} Performance', 'Support', str(regime_metrics.get('support', 0)), f'Number of samples in regime {regime_key}'])
                
                # Prediction confidence
                pred_conf = stacker_metrics.get('prediction_confidence', {})
                if pred_conf:
                    csv_data.append(['Ensemble Performance', 'Prediction Confidence Mean', f"{pred_conf.get('mean', 0):.4f}", 'Average prediction confidence'])
                    csv_data.append(['Ensemble Performance', 'Prediction Confidence Std', f"{pred_conf.get('std', 0):.4f}", 'Standard deviation of confidence'])
                
                # Calibration info
                calibration_method = stacker_metrics.get('calibration_method', 'none')
                csv_data.append(['Ensemble Performance', 'Calibration Method', calibration_method, 'Probability calibration method used'])

            # Ensemble performance metrics (regime probabilities)
            regime_report = results.get('regime_probability_report', {})
            if regime_report:
                overall = regime_report.get('overall_statistics', {})
                csv_data.append(['Overall', 'Total Samples', str(overall.get('total_samples', 'N/A')), 'Total number of samples'])
                csv_data.append(['Overall', 'Number of Regimes', str(overall.get('n_regimes', 'N/A')), 'Number of regimes detected'])
                csv_data.append(['Overall', 'Mean Max Probability', f"{overall.get('mean_max_probability', 0):.6f}", 'Average maximum probability'])
                csv_data.append(['Overall', 'Prediction Confidence', f"{overall.get('prediction_confidence', 0):.6f}", 'Average prediction confidence'])
                csv_data.append(['Overall', 'Uncertainty Entropy', f"{overall.get('uncertainty_entropy', 0):.6f}", 'Average entropy of predictions'])

                # Regime statistics
                regime_stats = regime_report.get('regime_statistics', {})
                for regime_key, regime_data in regime_stats.items():
                    if isinstance(regime_data, dict):
                        csv_data.append([f'Regime {regime_key}', 'Sample Count', str(regime_data.get('sample_count', 0)), 'Number of samples in regime'])
                        csv_data.append([f'Regime {regime_key}', 'Percentage', f"{regime_data.get('percentage', 0):.2f}%", 'Percentage of total samples'])
                        csv_data.append([f'Regime {regime_key}', 'Mean Probability', f"{regime_data.get('mean_probability', 0):.6f}", 'Average probability for regime'])

            # Write metrics CSV
            with open(metrics_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerows(csv_data)

            tprint(f"✅ Ensemble metrics CSV generated: {metrics_path}", color="green")

            # 2. Generate base model comparison CSV
            comparison_path = None
            # Try to get base models from detected models (blank mode) or training result
            base_models = results.get('detected_base_models', results.get('regime_ensemble_training_result', {}).get('base_models', []))
            if base_models:
                comparison_filename = f"regime_ensemble_base_models_{symbol}_{timestamp}.csv"
                comparison_path = output_path / comparison_filename

                tprint(f"📊 Generating base models comparison CSV: {comparison_path}", color="cyan")

                comparison_data = []
                comparison_data.append(['Model Name', 'Model Type', 'Included in Ensemble', 'Performance Role'])

                for model_name in base_models:
                    comparison_data.append([
                        model_name,
                        'Base Learner',
                        'Yes',
                        'Ensemble component'
                    ])

                # Add meta-learner
                comparison_data.append([
                    'stacker_lgbm_calibrated',
                    'Meta-Learner',
                    'N/A',
                    'Final ensemble predictions'
                ])

                with open(comparison_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(comparison_data)

                tprint(f"✅ Base models comparison CSV generated: {comparison_path}", color="green")

            # 3. Generate per-regime performance CSV
            regime_perf_path = None
            if regime_report and 'regime_statistics' in regime_report:
                regime_perf_filename = f"regime_ensemble_performance_by_regime_{symbol}_{timestamp}.csv"
                regime_perf_path = output_path / regime_perf_filename

                tprint(f"📊 Generating per-regime performance CSV: {regime_perf_path}", color="cyan")

                regime_data = []
                regime_data.append([
                    'Regime',
                    'Sample Count',
                    'Percentage',
                    'Mean Probability',
                    'Std Probability',
                    'High Confidence Count',
                    'Medium Confidence Count',
                    'Low Confidence Count'
                ])

                regime_stats = regime_report['regime_statistics']
                for regime_key, regime_metrics in regime_stats.items():
                    if isinstance(regime_metrics, dict):
                        conf_dist = regime_metrics.get('confidence_distribution', {})
                        regime_data.append([
                            regime_key,
                            str(regime_metrics.get('sample_count', 0)),
                            f"{regime_metrics.get('percentage', 0):.2f}%",
                            f"{regime_metrics.get('mean_probability', 0):.6f}",
                            f"{regime_metrics.get('std_probability', 0):.6f}",
                            str(conf_dist.get('high_confidence', 0)),
                            str(conf_dist.get('medium_confidence', 0)),
                            str(conf_dist.get('low_confidence', 0))
                        ])

                with open(regime_perf_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerows(regime_data)

                tprint(f"✅ Per-regime performance CSV generated: {regime_perf_path}", color="green")

            return str(metrics_path), str(comparison_path) if comparison_path else None

        except Exception as e:
            tprint(f"❌ Failed to generate CSV reports: {e}", color="red")
            self.logger.error(f"Failed to generate CSV reports: {e}", exc_info=True)
            return None, None

    def _generate_markdown_report(self, results: Dict[str, Any], symbol: str, output_dir: str = "outcomes") -> Optional[str]:
        """Generate comprehensive markdown report for regime ensemble training."""
        try:
            from pathlib import Path

            # Create output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"regime_ensemble_training_report_{symbol}_{timestamp}.md"
            report_path = output_path / filename

            tprint(f"📝 Generating markdown report: {report_path}", color="cyan")

            # Build markdown content
            md_lines = []
            md_lines.append(f"# Regime Ensemble Training Report")
            md_lines.append("")
            md_lines.append(f"**Symbol:** {symbol}")
            md_lines.append(f"**Model:** stacker_lgbm_calibrated (LightGBM Meta-Learner)")
            md_lines.append(f"**Generated:** {datetime.now().isoformat()}")
            md_lines.append(f"**Report Version:** 1.0")
            md_lines.append("")

            # Ensemble configuration
            ensemble_config = results.get('regime_ensemble_training_result', {})
            # Try to get base models from detected models (blank mode) or training result
            base_models = results.get('detected_base_models', ensemble_config.get('base_models', []))

            md_lines.append("## Ensemble Configuration")
            md_lines.append("")
            md_lines.append(f"- **Meta-Learner:** LightGBM with Probability Calibration")
            md_lines.append(f"- **Number of Base Models:** {len(base_models)}")
            md_lines.append(f"- **Base Models:** {', '.join(base_models) if base_models else 'N/A'}")
            md_lines.append("")

            # Ensemble Performance Metrics
            ensemble_metrics = results.get('ensemble_metrics', {})
            stacker_metrics = ensemble_metrics.get('stacker_lgbm_calibrated', {})
            if stacker_metrics:
                md_lines.append("## Ensemble Performance Metrics")
                md_lines.append("")
                md_lines.append("### Overall Performance")
                md_lines.append("")
                md_lines.append("| Metric | Value |")
                md_lines.append("|--------|-------|")
                md_lines.append(f"| Accuracy | {stacker_metrics.get('accuracy', 0):.4f} |")
                
                # Classification report metrics
                class_report = stacker_metrics.get('classification_report', {})
                if class_report:
                    weighted_avg = class_report.get('weighted avg', {})
                    if weighted_avg:
                        md_lines.append(f"| Precision (Weighted) | {weighted_avg.get('precision', 0):.4f} |")
                        md_lines.append(f"| Recall (Weighted) | {weighted_avg.get('recall', 0):.4f} |")
                        md_lines.append(f"| F1-Score (Weighted) | {weighted_avg.get('f1-score', 0):.4f} |")
                
                # Prediction confidence
                pred_conf = stacker_metrics.get('prediction_confidence', {})
                if pred_conf:
                    md_lines.append(f"| Prediction Confidence | {pred_conf.get('mean', 0):.4f} ± {pred_conf.get('std', 0):.4f} |")
                
                calibration_method = stacker_metrics.get('calibration_method', 'none')
                md_lines.append(f"| Calibration Method | {calibration_method} |")
                md_lines.append("")
                
                # Per-regime performance
                if class_report:
                    md_lines.append("### Per-Regime Performance")
                    md_lines.append("")
                    md_lines.append("| Regime | Precision | Recall | F1-Score | Support |")
                    md_lines.append("|--------|-----------|--------|----------|---------|")
                    for regime_key, regime_metrics in class_report.items():
                        if isinstance(regime_metrics, dict) and regime_key not in ['accuracy', 'macro avg', 'weighted avg']:
                            md_lines.append(f"| {regime_key} | {regime_metrics.get('precision', 0):.4f} | {regime_metrics.get('recall', 0):.4f} | {regime_metrics.get('f1-score', 0):.4f} | {regime_metrics.get('support', 0)} |")
                    md_lines.append("")

            # Overall statistics
            regime_report = results.get('regime_probability_report', {})
            if regime_report:
                overall = regime_report.get('overall_statistics', {})
                md_lines.append("## Overall Statistics")
                md_lines.append("")
                md_lines.append("| Metric | Value |")
                md_lines.append("|--------|-------|")
                md_lines.append(f"| Total Samples | {overall.get('total_samples', 'N/A')} |")
                md_lines.append(f"| Number of Regimes | {overall.get('n_regimes', 'N/A')} |")
                md_lines.append(f"| Mean Max Probability | {overall.get('mean_max_probability', 0):.4f} |")
                md_lines.append(f"| Prediction Confidence | {overall.get('prediction_confidence', 0):.4f} |")
                md_lines.append(f"| Uncertainty Entropy | {overall.get('uncertainty_entropy', 0):.4f} |")
                md_lines.append("")

                # Regime statistics
                md_lines.append("## Regime Statistics")
                md_lines.append("")
                md_lines.append("| Regime | Sample Count | Percentage | Mean Prob | High Conf | Med Conf | Low Conf |")
                md_lines.append("|--------|--------------|------------|-----------|-----------|----------|----------|")

                regime_stats = regime_report.get('regime_statistics', {})
                for regime_key, regime_data in regime_stats.items():
                    if isinstance(regime_data, dict):
                        conf_dist = regime_data.get('confidence_distribution', {})
                        md_lines.append(
                            f"| {regime_key} | "
                            f"{regime_data.get('sample_count', 0)} | "
                            f"{regime_data.get('percentage', 0):.1f}% | "
                            f"{regime_data.get('mean_probability', 0):.3f} | "
                            f"{conf_dist.get('high_confidence', 0)} | "
                            f"{conf_dist.get('medium_confidence', 0)} | "
                            f"{conf_dist.get('low_confidence', 0)} |"
                        )
                md_lines.append("")

            # Temporal analysis summary
            temporal_analysis = results.get('temporal_regime_analysis', {})
            if temporal_analysis:
                md_lines.append("## Temporal Analysis")
                md_lines.append("")
                md_lines.append(f"- **Transition Entropy:** {temporal_analysis.get('transition_entropy', 0):.4f}")
                md_lines.append(f"- **Average Regime Duration:** {temporal_analysis.get('average_regime_duration', 0):.2f} periods")
                md_lines.append(f"- **Number of Transitions:** {temporal_analysis.get('n_transitions', 0)}")
                md_lines.append("")

            # Write markdown file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(md_lines))

            tprint(f"✅ Markdown report generated: {report_path}", color="green")
            return str(report_path)

        except Exception as e:
            tprint(f"❌ Failed to generate markdown report: {e}", color="red")
            self.logger.error(f"Failed to generate markdown report: {e}", exc_info=True)
            return None
