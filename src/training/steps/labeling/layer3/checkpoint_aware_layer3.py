"""
Checkpoint-Aware Layer 3 Wrapper

Provides automatic checkpoint detection and resumption for Layer 3 execution.
Symbol-specific checkpoint management with intelligent resume logic.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from src.utils.tprint import tprint
from ..checkpoint_aware_runner import CheckpointAwareRunner, checkpoint_aware_step
from ..checkpoint_override_manager import CheckpointOverrideManager, create_checkpoint_override
from .core import (
    layer3_analyst_lgbm as core_layer3_analyst_lgbm,
    integrate_entropy_bars_into_layer3,
    generate_regime_aware_features,
    apply_mild_mp_clustering,
    select_best_model_per_task,
    prepare_layer3_features,
    prepare_layer3_targets_and_weights,
    process_layer3_results
)
from .model_training import train_dual_head_models
from .layer25_integration import integrate_layer25_into_layer3
from src.training.steps.labeling.irm_regime_pipeline import (
    build_env_indices_for_index,
    get_or_fit_regime_labels
)

logger = logging.getLogger(__name__)

class CheckpointAwareLayer3:
    """
    Checkpoint-aware wrapper for Layer 3 execution.
    
    Automatically:
    1. Detects available checkpoints for the symbol
    2. Resumes from the appropriate step
    3. Saves progress at each sub-step
    4. Provides detailed execution metadata
    """
    
    def __init__(self, symbol: str, checkpoint_dir: Optional[str] = None):
        """
        Initialize checkpoint-aware Layer 3.
        
        Args:
            symbol: Trading symbol (e.g., 'ETHUSDT')
            checkpoint_dir: Optional custom checkpoint directory
        """
        tprint("Starting CheckpointAwareLayer3.__init__...")
        self.symbol = symbol.upper()
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize checkpoint-aware runner
        self.runner = CheckpointAwareRunner('layer3', self.symbol, checkpoint_dir)
        
        logger.info(f"🔧 Initialized checkpoint-aware Layer 3 for {self.symbol}")
        logger.info(f"📍 Resume step: {self.runner.execution_plan.resume_step}")
        tprint("Finished CheckpointAwareLayer3.__init__")
    
    def run_with_override(
        self,
        override_step: str,
        oof_df: pd.DataFrame,
        base_model_cols: List[str],
        target_col: str,
        train_split_date: Optional[str] = None,
        sample_weight: Optional[np.ndarray] = None,
        layer1_weight: Optional[np.ndarray] = None,
        layer2_weight: Optional[np.ndarray] = None,
        layer2_weight_quality: Optional[np.ndarray] = None,
        net_returns: Optional[np.ndarray] = None,
        market_data: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
        outcomes_dir: Optional[str] = None,
        force_restart: bool = False,
        keep_earlier_checkpoints: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run Layer 3 with checkpoint override functionality.
        """
        tprint("Starting CheckpointAwareLayer3.run_with_override...")
        logger.info(f"🔄 Running Layer 3 with checkpoint override from '{override_step}'")
        logger.info(f"   Force restart: {force_restart}")
        logger.info(f"   Keep earlier checkpoints: {keep_earlier_checkpoints}")
        
        # Create override runner
        self.runner = create_checkpoint_override(
            layer='layer3',
            symbol=self.symbol,
            override_step=override_step,
            force_restart=force_restart,
            keep_earlier_checkpoints=keep_earlier_checkpoints,
            checkpoint_dir=self.checkpoint_dir
        )
        
        # Update execution plan
        self.execution_plan = self.runner.execution_plan
        
        # Run with override runner
        result = self.run(
            oof_df=oof_df,
            base_model_cols=base_model_cols,
            target_col=target_col,
            train_split_date=train_split_date,
            sample_weight=sample_weight,
            layer1_weight=layer1_weight,
            layer2_weight=layer2_weight,
            layer2_weight_quality=layer2_weight_quality,
            net_returns=net_returns,
            market_data=market_data,
            config=config,
            outcomes_dir=outcomes_dir
        )
        tprint("Finished CheckpointAwareLayer3.run_with_override")
        return result
    
    def run(
        self,
        oof_df: pd.DataFrame,
        base_model_cols: List[str],
        target_col: str,
        train_split_date: Optional[str] = None,
        sample_weight: Optional[np.ndarray] = None,
        layer1_weight: Optional[np.ndarray] = None,
        layer2_weight: Optional[np.ndarray] = None,
        layer2_weight_quality: Optional[np.ndarray] = None,
        net_returns: Optional[np.ndarray] = None,
        market_data: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
        outcomes_dir: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Run Layer 3 with automatic checkpoint management.
        """
        tprint("Starting CheckpointAwareLayer3.run...")
        # Prepare configuration
        config = config or {}
        config['symbol'] = self.symbol
        if outcomes_dir:
            config['outcomes_dir'] = outcomes_dir
        
        # Define step functions for checkpoint-aware execution
        step_functions = self._get_step_functions()
        
        # Prepare common arguments
        common_args = {
            'oof_df': oof_df,
            'base_model_cols': base_model_cols,
            'target_col': target_col,
            'train_split_date': train_split_date,
            'sample_weight': sample_weight,
            'layer1_weight': layer1_weight,
            'layer2_weight': layer2_weight,
            'layer2_weight_quality': layer2_weight_quality,
            'net_returns': net_returns,
            'market_data': market_data,
            'outcomes_dir': outcomes_dir,
            'symbol': self.symbol
        }
        
        # Run with checkpoint management
        result = self.runner.run_with_checkpoints(step_functions, config, **common_args)
        
        # Extract final results
        if 'final_processing' in result['results']:
            final_result = result['results']['final_processing']
            df_final = final_result['df']
            models_dict = final_result['models_dict']
        else:
            # Fallback to last available result
            logger.warning("⚠️ Final processing step not found, using latest available result")
            if result['results']:
                latest_step = max(result['results'].keys(), key=lambda k: self.runner.get_step_index(k))
                latest_result = result['results'][latest_step]

                if 'df' in latest_result:
                    df_final = latest_result['df']
                else:
                    df_final = oof_df  # Fallback to input

                models_dict = latest_result.get('models_dict', {})
            else:
                 df_final = oof_df
                 models_dict = {}
        
        # Add execution metadata
        models_dict['checkpoint_metadata'] = result['metadata']
        
        logger.info(f"🎉 Checkpoint-aware Layer 3 completed for {self.symbol}")
        logger.info(f"📊 Steps executed: {result['metadata']['steps_executed']}")
        logger.info(f"💾 Checkpoints saved: {len(result['metadata']['checkpoints_saved'])}")
        
        tprint("Finished CheckpointAwareLayer3.run")
        return df_final, models_dict
    
    def _get_step_functions(self) -> Dict[str, callable]:
        """Get step functions for checkpoint-aware execution."""
        tprint("Starting CheckpointAwareLayer3._get_step_functions...")
        functions = {
            'data_loading': self._step_data_loading,
            'entropy_bars_integration': self._step_entropy_bars_integration,
            'meta_features_engineering': self._step_meta_features_engineering,
            'feature_clustering': self._step_feature_clustering,
            'layer25_integration': self._step_layer25_integration,
            'dual_head_training': self._step_dual_head_training,
            'model_selection_12': self._step_model_selection_12,
            'model_selection_48': self._step_model_selection_48,
            'oof_predictions': self._step_oof_predictions,
            'race_reporting': self._step_race_reporting,
            'enhanced_reporting': self._step_enhanced_reporting,
            'final_processing': self._step_final_processing
        }
        tprint("Finished CheckpointAwareLayer3._get_step_functions")
        return functions
    
    @checkpoint_aware_step('data_loading')
    def _step_data_loading(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 0: Load and prepare data."""
        tprint("Starting CheckpointAwareLayer3._step_data_loading...")
        oof_df = kwargs['oof_df']
        base_model_cols = kwargs['base_model_cols']
        
        # We don't validate heavily here, validation happens in core utils
        tprint("Finished CheckpointAwareLayer3._step_data_loading")
        return {
            'oof_df': oof_df,
            'base_model_cols': base_model_cols
        }
    
    @checkpoint_aware_step('entropy_bars_integration')
    def _step_entropy_bars_integration(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 1: Integrate entropy bars and specialized features."""
        tprint("Starting CheckpointAwareLayer3._step_entropy_bars_integration...")

        df = results['data_loading']['oof_df']
        symbol = kwargs['symbol']
        exchange = config.get('exchange', 'binance')

        if config.get('use_entropy_bars', True):
            enhanced_df, entropy_bars_df = integrate_entropy_bars_into_layer3(
                df, symbol, exchange, config
            )
        else:
            enhanced_df = df
            entropy_bars_df = pd.DataFrame()

        tprint("Finished CheckpointAwareLayer3._step_entropy_bars_integration")
        return {
            'enhanced_df': enhanced_df,
            'entropy_bars_df': entropy_bars_df
        }
    
    @checkpoint_aware_step('meta_features_engineering')
    def _step_meta_features_engineering(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 2: Generate meta features."""
        tprint("Starting CheckpointAwareLayer3._step_meta_features_engineering...")
        
        df = results['entropy_bars_integration']['enhanced_df']
        base_model_cols = results['data_loading']['base_model_cols']
        symbol = kwargs['symbol']
        exchange = config.get('exchange', 'binance')
        market_data = kwargs.get('market_data')
        
        # Prepare features (Phase 1)
        df = prepare_layer3_features(
            df=df,
            base_model_cols=base_model_cols,
            symbol=symbol,
            exchange=exchange,
            config=config,
            market_data=market_data
        )
        
        tprint("Finished CheckpointAwareLayer3._step_meta_features_engineering")
        return {
            'df_with_features': df,
            'base_model_cols': base_model_cols
        }
    
    @checkpoint_aware_step('feature_clustering')
    def _step_feature_clustering(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 3: Target generation and Feature Clustering."""
        tprint("Starting CheckpointAwareLayer3._step_feature_clustering...")
        
        df = results['meta_features_engineering']['df_with_features']
        base_model_cols = results['data_loading']['base_model_cols']
        layer1_weight = kwargs.get('layer1_weight')
        net_returns = kwargs.get('net_returns')
        target_col = kwargs['target_col']
        
        # Generate Targets (Phase 2)
        targets_data = prepare_layer3_targets_and_weights(
            df=df,
            layer1_weight=layer1_weight,
            net_returns=net_returns,
            config=config
        )

        # Feature Clustering (Phase 3)
        # Select meta features
        safe_base_cols = [c for c in base_model_cols if c in df.columns]
        exclude = set(base_model_cols) | {target_col, 'close', 'high', 'low', 'volume', 'regime_label'}
        meta_features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64]]
        X_full = df[meta_features].copy()

        # Add base model cols back
        for col in safe_base_cols:
            if col in df.columns:
                X_full[col] = df[col].reindex(X_full.index)

        # Regime Aware Features (Phase 3.5)
        prob_cols = [c for c in X_full.columns if 'prob_' in c and '_oof' not in c]
        regime_feats = generate_regime_aware_features(X_full, 'volatility_20', prob_cols)
        X_full = pd.concat([X_full, regime_feats], axis=1)
        
        # Apply Clustering
        y_alpha_12_series = targets_data['y_alpha_12_series']
        X_clustered = apply_mild_mp_clustering(X_full, threshold=0.98, target=y_alpha_12_series)
        
        tprint("Finished CheckpointAwareLayer3._step_feature_clustering")
        return {
            'X_clustered': X_clustered,
            'targets_data': targets_data,
            'meta_features': meta_features,
            'df': df  # Pass full df along
        }
    
    @checkpoint_aware_step('layer25_integration')
    def _step_layer25_integration(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 4: Integrate Layer 2.5 chaser models."""
        tprint("Starting CheckpointAwareLayer3._step_layer25_integration...")

        df = results['feature_clustering']['df']
        X_clustered = results['feature_clustering']['X_clustered']

        layer25_enabled = config.get('layer25_chaser_enabled', True)
        layer25_results = config.get('layer25_chaser_results', None)
        
        if layer25_enabled and layer25_results is not None:
            tprint("🔗 Integrating Layer 2.5 Chaser Models...")
            try:
                # Need to construct a df compatible with integration logic (needs X_full basically)
                # But integrate_layer25_into_layer3 takes a DF.
                # We should update df and X_clustered with new features.

                # We can just pass the current df.
                # But X_clustered is what we train on.

                df_enhanced, integration_metadata = integrate_layer25_into_layer3(
                    df=df,
                    chaser_results=layer25_results,
                    symbol=kwargs['symbol'],
                    exchange=config.get('exchange', 'binance'),
                    timeframe=config.get('timeframe', '15m'),
                    top_n_models=config.get('layer25_top_models', 3),
                    outcomes_dir=kwargs.get('outcomes_dir')
                )
                
                # Add new features to X_clustered
                chaser_features = [col for col in df_enhanced.columns if col.startswith('chaser_')]
                for feature in chaser_features:
                    if feature not in X_clustered.columns:
                        X_clustered[feature] = df_enhanced[feature]
                        df[feature] = df_enhanced[feature]

                tprint("Finished CheckpointAwareLayer3._step_layer25_integration")
                return {
                    'X_clustered_enhanced': X_clustered,
                    'df_enhanced': df,
                    'integration_metadata': integration_metadata
                }
            except Exception as e:
                logger.warning(f"⚠️ Layer 2.5 integration failed: {e}")
                return {
                    'X_clustered_enhanced': X_clustered,
                    'df_enhanced': df,
                    'integration_metadata': {'status': 'failed'}
                }

        tprint("Finished CheckpointAwareLayer3._step_layer25_integration (Skipped)")
        return {
            'X_clustered_enhanced': X_clustered,
            'df_enhanced': df,
            'integration_metadata': {'status': 'skipped'}
        }
    
    @checkpoint_aware_step('dual_head_training')
    def _step_dual_head_training(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 5: Train all model families."""
        tprint("Starting CheckpointAwareLayer3._step_dual_head_training...")
        
        X_clustered = results['layer25_integration']['X_clustered_enhanced']
        targets_data = results['feature_clustering']['targets_data']
        df = results['layer25_integration']['df_enhanced']
        
        y_alpha_12 = targets_data['y_alpha_12']
        y_prob_12 = targets_data['y_prob_12']
        w_alpha = targets_data['w_alpha']
        
        # IRM Environment Detection
        irm_env_indices = []
        if config.get("irm_meta_enabled", True):
            gmm_dir = Path(config.get("irm_regime_dir", "artifacts/irm_regimes"))
            gmm_dir.mkdir(parents=True, exist_ok=True)
            try:
                regime_labels = get_or_fit_regime_labels(
                    df,
                    gmm_dir / "layer3_meta_gmm.pkl",
                    n_regimes=config.get("irm_meta_regimes", 2),
                    refit=config.get("irm_refit_regimes", False)
                )
                irm_env_indices = build_env_indices_for_index(regime_labels, X_clustered.index)
            except Exception as e:
                logger.warning(f"⚠️ IRM regime detection failed: {e}")

        # Configure IRM
        config["irm_env_indices"] = irm_env_indices
        config["irm_lambda"] = config.get("irm_meta_lambda", 2.0)
        config["y_alpha_48"] = targets_data['y_alpha_48']
        config["y_prob_48"] = targets_data['y_prob_48']

        # Train Models
        model_results = train_dual_head_models(
            X=X_clustered,
            y_alpha=y_alpha_12,
            y_prob=y_prob_12,
            w_alpha=w_alpha,
            w_prob=w_alpha,
            cv_splits=[],
            config=config,
            fast_mode=config.get('fast_mode', False)
        )
        
        tprint("Finished CheckpointAwareLayer3._step_dual_head_training")
        return {
            'model_results': model_results,
            'combined_models': model_results['models'],
            'irm_env_indices': irm_env_indices
        }
    
    @checkpoint_aware_step('model_selection_12')
    def _step_model_selection_12(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 6: Select best models for 12-bar horizon."""
        tprint("Starting CheckpointAwareLayer3._step_model_selection_12...")
        
        combined_models = results['dual_head_training']['combined_models']
        targets_data = results['feature_clustering']['targets_data']

        y_alpha_12 = targets_data['y_alpha_12']
        y_prob_12 = targets_data['y_prob_12']
        
        best_pred_12_reg, best_key_12_reg = select_best_model_per_task(
            combined_models, y_alpha_12, 'regression', '12'
        )
        best_pred_12_cls, best_key_12_cls = select_best_model_per_task(
            combined_models, y_prob_12, 'classification', '12'
        )
        
        tprint("Finished CheckpointAwareLayer3._step_model_selection_12")
        return {
            'best_models_12': {
                'regression': {'prediction': best_pred_12_reg, 'model_key': best_key_12_reg},
                'classification': {'prediction': best_pred_12_cls, 'model_key': best_key_12_cls}
            }
        }
    
    @checkpoint_aware_step('model_selection_48')
    def _step_model_selection_48(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 7: Select best models for 48-bar horizon."""
        tprint("Starting CheckpointAwareLayer3._step_model_selection_48...")
        
        combined_models = results['dual_head_training']['combined_models']
        targets_data = results['feature_clustering']['targets_data']

        y_alpha_48 = targets_data['y_alpha_48']
        y_prob_48 = targets_data['y_prob_48']
        
        best_pred_48_reg, best_key_48_reg = select_best_model_per_task(
            combined_models, y_alpha_48, 'regression', '48'
        )
        best_pred_48_cls, best_key_48_cls = select_best_model_per_task(
            combined_models, y_prob_48, 'classification', '48'
        )
        
        tprint("Finished CheckpointAwareLayer3._step_model_selection_48")
        return {
            'best_models_48': {
                'regression': {'prediction': best_pred_48_reg, 'model_key': best_key_48_reg},
                'classification': {'prediction': best_pred_48_cls, 'model_key': best_key_48_cls}
            }
        }
    
    @checkpoint_aware_step('oof_predictions')
    def _step_oof_predictions(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 8: Generate OOF predictions for all models."""
        tprint("Starting CheckpointAwareLayer3._step_oof_predictions...")

        combined_models = results['dual_head_training']['combined_models']
        df = results['layer25_integration']['df_enhanced']
        X_clustered = results['layer25_integration']['X_clustered_enhanced']
        
        best_models_info = {
            '12_reg': results['model_selection_12']['best_models_12']['regression']['model_key'],
            '12_cls': results['model_selection_12']['best_models_12']['classification']['model_key'],
            '48_reg': results['model_selection_48']['best_models_48']['regression']['model_key'],
            '48_cls': results['model_selection_48']['best_models_48']['classification']['model_key']
        }
        
        df_final = process_layer3_results(
            df=df,
            combined_models=combined_models,
            best_models_info=best_models_info,
            X_index=X_clustered.index
        )
        
        tprint("Finished CheckpointAwareLayer3._step_oof_predictions")
        return {
            'df_final': df_final,
            'best_models_info': best_models_info
        }
    
    @checkpoint_aware_step('race_reporting')
    def _step_race_reporting(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 9: Generate comprehensive model race reports."""
        tprint("Starting CheckpointAwareLayer3._step_race_reporting...")
        from .model_race_reporting import Layer3ModelRaceReporter
        
        combined_models = results['dual_head_training']['combined_models']
        targets_data = results['feature_clustering']['targets_data']
        
        # Generate race report
        try:
            reporter = Layer3ModelRaceReporter(outcomes_dir=kwargs.get('outcomes_dir'))
            reporter.generate_model_race_report(
                models_dict=combined_models,
                y_alpha_12=targets_data['y_alpha_12'],
                y_prob_12=targets_data['y_prob_12'],
                y_alpha_48=targets_data['y_alpha_48'],
                y_prob_48=targets_data['y_prob_48']
            )
        except Exception as e:
            logger.warning(f"⚠️ Model race reporting failed: {e}")
        
        tprint("Finished CheckpointAwareLayer3._step_race_reporting")
        return {'race_report_generated': True}
    
    @checkpoint_aware_step('enhanced_reporting')
    def _step_enhanced_reporting(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 10: Generate enhanced Layer 3 reports."""
        tprint("Starting CheckpointAwareLayer3._step_enhanced_reporting...")
        from .enhanced_reporting import EnhancedLayer3Reporter
        
        df_final = results['oof_predictions']['df_final']
        combined_models = results['dual_head_training']['combined_models']
        meta_features = results['feature_clustering']['meta_features']

        try:
            reporter = EnhancedLayer3Reporter(outcomes_dir=kwargs.get('outcomes_dir'))
            reporter.generate_all_reports(
                df=df_final,
                models={'models': combined_models},
                geometry_metrics=config.get('geometry_metrics', []),
                meta_features=meta_features,
                target_col='meta_prob',
                config=config
            )
        except Exception as e:
             logger.warning(f"⚠️ Enhanced reporting failed: {e}")
        
        tprint("Finished CheckpointAwareLayer3._step_enhanced_reporting")
        return {'enhanced_report_generated': True}
    
    @checkpoint_aware_step('final_processing')
    def _step_final_processing(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 11: Final validation and artifact saving."""
        tprint("Starting CheckpointAwareLayer3._step_final_processing...")

        df_final = results['oof_predictions']['df_final']
        best_models_info = results['oof_predictions']['best_models_info']
        combined_models = results['dual_head_training']['combined_models']
        meta_features = results['feature_clustering']['meta_features']
        entropy_bars_df = results['entropy_bars_integration']['entropy_bars_df']
        irm_env_indices = results['dual_head_training']['irm_env_indices']
        
        # Build models dictionary
        models_dict = {
            'all_models': combined_models,
            'best_models': best_models_info,
            'meta_features': meta_features,
            'entropy_bars': entropy_bars_df if not entropy_bars_df.empty else None,
            'irm_env_indices': irm_env_indices
        }
        
        # Add integration metadata if available
        if 'integration_metadata' in results['layer25_integration']:
            models_dict['layer25_integration'] = results['layer25_integration']['integration_metadata']
        
        tprint("Finished CheckpointAwareLayer3._step_final_processing")
        return {
            'df': df_final,
            'models_dict': models_dict
        }
    
    def get_checkpoint_status(self) -> Dict[str, Any]:
        """Get detailed checkpoint status."""
        tprint("Starting CheckpointAwareLayer3.get_checkpoint_status...")
        result = self.runner.get_checkpoint_status()
        tprint("Finished CheckpointAwareLayer3.get_checkpoint_status")
        return result
    
    def reset_checkpoints(self) -> int:
        """Reset all checkpoints for this symbol."""
        tprint("Starting CheckpointAwareLayer3.reset_checkpoints...")
        result = self.runner.reset_all_checkpoints()
        tprint("Finished CheckpointAwareLayer3.reset_checkpoints")
        return result

def layer3_analyst_lgbm_checkpoint_aware(
    oof_df: pd.DataFrame,
    base_model_cols: List[str],
    target_col: str,
    train_split_date: Optional[str] = None,
    sample_weight: Optional[np.ndarray] = None,
    layer1_weight: Optional[np.ndarray] = None,
    layer2_weight: Optional[np.ndarray] = None,
    layer2_weight_quality: Optional[np.ndarray] = None,
    net_returns: Optional[np.ndarray] = None,
    market_data: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None,
    outcomes_dir: Optional[str] = None,
    symbol: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    override_step: Optional[str] = None,
    force_restart: bool = False,
    keep_earlier_checkpoints: bool = False
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Checkpoint-aware wrapper for Layer 3 execution.
    """
    tprint("Starting layer3_analyst_lgbm_checkpoint_aware...")
    if symbol is None:
        raise ValueError("Symbol is required for checkpoint-aware execution")
    
    # Create checkpoint-aware Layer 3 instance
    checkpoint_layer3 = CheckpointAwareLayer3(symbol, checkpoint_dir)
    
    # Check if override is requested
    if override_step is not None:
        logger.info(f"🔄 Using checkpoint override from '{override_step}'")
        result = checkpoint_layer3.run_with_override(
            override_step=override_step,
            oof_df=oof_df,
            base_model_cols=base_model_cols,
            target_col=target_col,
            train_split_date=train_split_date,
            sample_weight=sample_weight,
            layer1_weight=layer1_weight,
            layer2_weight=layer2_weight,
            layer2_weight_quality=layer2_weight_quality,
            net_returns=net_returns,
            market_data=market_data,
            config=config,
            outcomes_dir=outcomes_dir,
            force_restart=force_restart,
            keep_earlier_checkpoints=keep_earlier_checkpoints
        )
    else:
        # Run with normal checkpoint management
        result = checkpoint_layer3.run(
            oof_df=oof_df,
            base_model_cols=base_model_cols,
            target_col=target_col,
            train_split_date=train_split_date,
            sample_weight=sample_weight,
            layer1_weight=layer1_weight,
            layer2_weight=layer2_weight,
            layer2_weight_quality=layer2_weight_quality,
            net_returns=net_returns,
            market_data=market_data,
            config=config,
            outcomes_dir=outcomes_dir
        )
    tprint("Finished layer3_analyst_lgbm_checkpoint_aware")
    return result
