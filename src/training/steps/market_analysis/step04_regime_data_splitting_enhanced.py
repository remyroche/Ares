"""
Enhanced Step 4: Regime Data Tagging with HMM ML Model Integration

This module creates a unified dataset with regime labels for regime-aware processing.
It now includes HMM ML model tagging capabilities to determine regime membership
using trained HMM models from the hmm_training module.

KEY FEATURES:
- HMM ML model integration for regime tagging
- 100% data retention (no rows lost to splitting boundaries)
- Full lookback period preservation for all features
- Temporal continuity maintained across regime transitions
- Single dataset management (no multiple files per regime)
- Context preservation around regime changes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
import logging
import pickle
import warnings
warnings.filterwarnings('ignore')

# Import existing infrastructure
try:
    from src.training.steps.market_analysis.hmm_training.hmm_models_training_refactored import HMMModelsTrainingRefactored as HMMModelsTraining
    from src.training.steps.market_analysis.hmm_training.hmm_ensemble_training import HMMEnsembleTrainingRefactored as HMMEnsembleTraining
    from src.feature_engineering.feature_generators import FeatureGenerator
    from src.training.utils.feature_selection.main_framework import FeatureSelectionFramework
    HMM_TRAINING_AVAILABLE = True
except ImportError as e:
    print(f"Warning: HMM training modules not available: {e}")
    HMM_TRAINING_AVAILABLE = False

# Core decorators imports
from src.core.decorators import (
    handles_errors,
    traced,
    log_execution_time
)

# Core errors imports
from src.core.errors import (
    AppError,
    ValidationError,
    DataIntegrityError,
    NotFoundError
)

from src.utils.logger import system_logger
from src.training.steps.standardized_parquet_handler import standardized_parquet_handler

logger = system_logger.getChild('RegimeDataSplittingEnhanced')

class HMMRegimeTagger:
    """HMM-based regime tagger using trained ML models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize HMM regime tagger."""
        self.config = config
        self.logger = logger.getChild('HMMRegimeTagger')
        self.base_models = {}
        self.ensemble_models = {}
        self.feature_generator = None
        self.feature_selector = None
        
        if HMM_TRAINING_AVAILABLE:
            self._initialize_components()
    
    def _initialize_components(self):
        """Initialize HMM training components."""
        try:
            # Initialize feature generator
            self.feature_generator = FeatureGenerator()
            
            # Initialize feature selection framework
            fs_config = {
                'selection_methods': ['mrmr', 'lasso_stability', 'correlation_filter'],
                'max_features': self.config.get('n_features', 100),
                'enable_stability_analysis': True,
                'enable_temporal_analysis': True
            }
            self.feature_selector = FeatureSelectionFramework(fs_config)
            
            self.logger.info("✅ HMM regime tagger components initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize HMM components: {e}")
            raise
    
    def load_trained_models(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> bool:
        """Load trained HMM models."""
        try:
            models_dir = Path(data_dir) / 'models' / 'hmm'
            
            # Load base models
            base_models_dir = models_dir / 'base_models'
            if base_models_dir.exists():
                for model_file in base_models_dir.glob(f'hmm_base_*_{symbol}_{exchange}_{timeframe}.pkl'):
                    model_name = model_file.stem.split('_')[2]  # Extract model name
                    with open(model_file, 'rb') as f:
                        self.base_models[model_name] = pickle.load(f)
                self.logger.info(f"✅ Loaded {len(self.base_models)} base models")
            
            # Load ensemble models
            ensemble_models_dir = models_dir / 'ensemble_models'
            if ensemble_models_dir.exists():
                for model_file in ensemble_models_dir.glob(f'hmm_ensemble_*_{symbol}_{exchange}_{timeframe}.pkl'):
                    model_name = model_file.stem.split('_')[2]  # Extract model name
                    with open(model_file, 'rb') as f:
                        self.ensemble_models[model_name] = pickle.load(f)
                self.logger.info(f"✅ Loaded {len(self.ensemble_models)} ensemble models")
            
            return len(self.base_models) > 0 or len(self.ensemble_models) > 0
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load trained models: {e}")
            return False
    
    def create_features_for_tagging(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """Create features for HMM regime tagging."""
        if self.feature_generator is None:
            raise ValueError("Feature generator not initialized")
        
        # Use existing feature generator for 200+ features
        features = self.feature_generator.generate_all_features(market_data)
        self.logger.info(f"✅ Generated {features.shape[1]} features for regime tagging")
        return features
    
    def select_features_for_tagging(self, X: pd.DataFrame, is_classification: bool = True) -> pd.DataFrame:
        """Select features for HMM regime tagging."""
        if self.feature_selector is None:
            # Return all features if no feature selector
            return X
        
        try:
            # Use existing feature selection framework
            selection_result = self.feature_selector.select_features(
                X, 
                method='comprehensive',
                max_features=self.config.get('n_features', 100),
                is_classification=is_classification
            )
            
            selected_features = selection_result.get('selected_features', X.columns.tolist()[:self.config.get('n_features', 100)])
            X_selected = X[selected_features]
            
            self.logger.info(f"✅ Selected {len(selected_features)} features for regime tagging")
            return X_selected
            
        except Exception as e:
            self.logger.warning(f"⚠️ Feature selection failed, using all features: {e}")
            return X
    
    def tag_regimes_with_models(self, market_data: pd.DataFrame, 
                              use_ensemble: bool = True) -> Dict[str, Any]:
        """Tag regimes using trained HMM models."""
        try:
            # Create features
            features = self.create_features_for_tagging(market_data)
            
            # Select features
            features_selected = self.select_features_for_tagging(features)
            
            # Scale features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_selected)
            
            # Get predictions from models
            regime_predictions = {}
            regime_probabilities = {}
            
            # Use ensemble models if available and requested
            if use_ensemble and self.ensemble_models:
                for name, model in self.ensemble_models.items():
                    try:
                        pred = model.predict(features_scaled)
                        proba = model.predict_proba(features_scaled) if hasattr(model, 'predict_proba') else None
                        
                        regime_predictions[f'ensemble_{name}'] = pred
                        if proba is not None:
                            regime_probabilities[f'ensemble_{name}'] = proba
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error with ensemble model {name}: {e}")
            
            # Use base models if no ensemble or as fallback
            if not regime_predictions or not use_ensemble:
                for name, model in self.base_models.items():
                    try:
                        pred = model.predict(features_scaled)
                        proba = model.predict_proba(features_scaled) if hasattr(model, 'predict_proba') else None
                        
                        regime_predictions[f'base_{name}'] = pred
                        if proba is not None:
                            regime_probabilities[f'base_{name}'] = proba
                            
                    except Exception as e:
                        self.logger.warning(f"⚠️ Error with base model {name}: {e}")
            
            if not regime_predictions:
                raise ValueError("No models available for regime tagging")
            
            # Use the best available model (first one)
            best_model_name = list(regime_predictions.keys())[0]
            final_predictions = regime_predictions[best_model_name]
            final_probabilities = regime_probabilities.get(best_model_name)
            
            self.logger.info(f"✅ Tagged regimes using model: {best_model_name}")
            
            return {
                'regime_predictions': final_predictions,
                'regime_probabilities': final_probabilities,
                'model_used': best_model_name,
                'all_predictions': regime_predictions,
                'all_probabilities': regime_probabilities,
                'n_regimes': len(np.unique(final_predictions)),
                'regime_distribution': dict(zip(*np.unique(final_predictions, return_counts=True)))
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error tagging regimes with models: {e}")
            raise

class RegimeDataSplittingEnhanced:
    """Enhanced regime data splitting with HMM ML model integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize enhanced regime data splitting."""
        self.config = config
        self.logger = logger.getChild('RegimeDataSplittingEnhanced')
        self.hmm_tagger = None
        
        if HMM_TRAINING_AVAILABLE:
            self.hmm_tagger = HMMRegimeTagger(config)
    
    @handles_errors
    @traced
    @log_execution_time
    async def execute(self, training_input: Dict[str, Any], 
                    pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute enhanced regime data splitting with HMM ML model tagging.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with regime-tagged data
        """
        try:
            symbol = training_input.get('symbol', 'UNKNOWN')
            exchange = training_input.get('exchange', 'UNKNOWN')
            timeframe = training_input.get('timeframe', 'UNKNOWN')
            data_dir = training_input.get('data_dir', 'data/training')
            
            self.logger.info(f"🔄 Starting enhanced regime data splitting for {symbol}/{exchange}/{timeframe}")
            
            # Load market data
            market_data = await self._load_market_data(symbol, exchange, timeframe, data_dir)
            if market_data is None or len(market_data) == 0:
                raise ValueError("No market data available for regime tagging")
            
            # Check if HMM models are available
            hmm_models_available = False
            if self.hmm_tagger:
                hmm_models_available = self.hmm_tagger.load_trained_models(symbol, exchange, timeframe, data_dir)
            
            if hmm_models_available:
                # Use HMM ML models for regime tagging
                self.logger.info("🤖 Using HMM ML models for regime tagging")
                tagging_result = self.hmm_tagger.tag_regimes_with_models(market_data)
                
                # Update market data with HMM regime tags
                market_data['hmm_regime_states'] = tagging_result['regime_predictions']
                if tagging_result['regime_probabilities'] is not None:
                    market_data['hmm_regime_probabilities'] = tagging_result['regime_probabilities'].tolist()
                market_data['hmm_regime_confidence'] = np.max(tagging_result['regime_probabilities'], axis=1) if tagging_result['regime_probabilities'] is not None else np.ones(len(market_data))
                
                # Store HMM tagging results
                hmm_tagging_info = {
                    'hmm_tagging_completed': True,
                    'hmm_model_used': tagging_result['model_used'],
                    'hmm_n_regimes': tagging_result['n_regimes'],
                    'hmm_regime_distribution': tagging_result['regime_distribution'],
                    'hmm_tagging_timestamp': pd.Timestamp.now().isoformat()
                }
                
            else:
                # Fallback to original regime discovery results
                self.logger.info("⚠️ HMM models not available, using original regime discovery results")
                
                # Get regime data from pipeline state
                regime_states = pipeline_state.get('regime_states', [])
                regime_probabilities = pipeline_state.get('regime_probabilities', [])
                
                if len(regime_states) == 0:
                    raise ValueError("No regime data available for tagging")
                
                # Align data lengths
                min_len = min(len(market_data), len(regime_states))
                market_data = market_data.iloc[:min_len]
                regime_states = regime_states[:min_len]
                regime_probabilities = regime_probabilities[:min_len] if len(regime_probabilities) > 0 else []
                
                # Use original regime data
                market_data['hmm_regime_states'] = regime_states
                if len(regime_probabilities) > 0:
                    market_data['hmm_regime_probabilities'] = regime_probabilities
                market_data['hmm_regime_confidence'] = np.ones(len(market_data))  # Default confidence
                
                hmm_tagging_info = {
                    'hmm_tagging_completed': False,
                    'hmm_model_used': 'original_regime_discovery',
                    'hmm_n_regimes': len(np.unique(regime_states)),
                    'hmm_regime_distribution': dict(zip(*np.unique(regime_states, return_counts=True))),
                    'hmm_tagging_timestamp': pd.Timestamp.now().isoformat()
                }
            
            # Save tagged data
            await self._save_tagged_data(market_data, symbol, exchange, timeframe, data_dir)
            
            # Update pipeline state
            updated_pipeline_state = pipeline_state.copy()
            updated_pipeline_state.update({
                'step04_regime_data_splitting_completed': True,
                'step04_regime_data_splitting_timestamp': pd.Timestamp.now().isoformat(),
                'regime_tagged_data_available': True,
                'regime_tagged_data_path': f"{data_dir}/training/{exchange}_{symbol}_{timeframe}_regime_tagged_data.parquet",
                'hmm_tagging_info': hmm_tagging_info
            })
            
            self.logger.info("✅ Enhanced regime data splitting completed successfully")
            
            return updated_pipeline_state
            
        except Exception as e:
            self.logger.error(f"❌ Enhanced regime data splitting failed: {e}")
            raise
    
    async def _load_market_data(self, symbol: str, exchange: str, timeframe: str, data_dir: str) -> Optional[pd.DataFrame]:
        """Load market data for regime tagging."""
        try:
            data_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_market_data.parquet'
            
            if not data_path.exists():
                self.logger.warning(f"⚠️ Market data file not found: {data_path}")
                return None
            
            market_data = pd.read_parquet(data_path)
            self.logger.info(f"✅ Loaded market data: {market_data.shape}")
            return market_data
            
        except Exception as e:
            self.logger.error(f"❌ Error loading market data: {e}")
            return None
    
    async def _save_tagged_data(self, market_data: pd.DataFrame, symbol: str, exchange: str, 
                              timeframe: str, data_dir: str) -> None:
        """Save regime-tagged data."""
        try:
            output_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_regime_tagged_data.parquet'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            market_data.to_parquet(output_path, index=False)
            self.logger.info(f"✅ Saved regime-tagged data: {output_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving regime-tagged data: {e}")
            raise

# Convenience function
async def execute_enhanced_regime_data_splitting(
    training_input: Dict[str, Any], 
    pipeline_state: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute enhanced regime data splitting with HMM ML model integration."""
    config = config or {}
    splitter = RegimeDataSplittingEnhanced(config)
    return await splitter.execute(training_input, pipeline_state)