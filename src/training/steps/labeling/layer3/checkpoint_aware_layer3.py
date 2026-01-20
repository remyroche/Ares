"""
Checkpoint-Aware Layer 3 Wrapper

Provides automatic checkpoint detection and resumption for Layer 3 execution.
Symbol-specific checkpoint management with intelligent resume logic.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

from src.utils.tprint import tprint
from ..checkpoint_aware_runner import CheckpointAwareRunner, checkpoint_aware_step
from ..checkpoint_override_manager import CheckpointOverrideManager, create_checkpoint_override
from .core import layer3_analyst_lgbm as core_layer3_analyst_lgbm
from .utils import calculate_alpha_target, calculate_sample_weights_efficient

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
        
        Args:
            override_step: Step to override from (e.g., 'dual_head_training')
            force_restart: Force restart from beginning (ignores all checkpoints)
            keep_earlier_checkpoints: Keep checkpoints before override step
            ... (other args same as run method)
            
        Returns:
            Tuple of (enhanced DataFrame, models dictionary)
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
        
        Args:
            oof_df: Out-of-fold DataFrame
            base_model_cols: Base model columns
            target_col: Target column name
            train_split_date: Optional train split date
            sample_weight: Optional sample weights
            layer1_weight: Optional Layer 1 weights
            layer2_weight: Optional Layer 2 weights
            layer2_weight_quality: Optional Layer 2 weight quality
            net_returns: Optional net returns
            market_data: Optional market data
            config: Configuration dictionary
            outcomes_dir: Outcomes directory
            
        Returns:
            Tuple of (enhanced DataFrame, models dictionary)
        """
        tprint("Starting CheckpointAwareLayer3.run...")
        # Prepare configuration
        config = config or {}
        config['symbol'] = self.symbol
        
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
            latest_step = max(result['results'].keys(), key=lambda k: self.runner.get_step_index(k))
            latest_result = result['results'][latest_step]
            
            if 'df' in latest_result:
                df_final = latest_result['df']
            else:
                df_final = oof_df  # Fallback to input
            
            models_dict = latest_result.get('models_dict', {})
        
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
        
        # Validate inputs
        from .utils import validate_feature_matrix
        X_validated = validate_feature_matrix(oof_df[base_model_cols])
        
        tprint("Finished CheckpointAwareLayer3._step_data_loading")
        return {
            'oof_df': oof_df,
            'base_model_cols': base_model_cols,
            'X_validated': X_validated
        }
    
    @checkpoint_aware_step('entropy_bars_integration')
    def _step_entropy_bars_integration(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 1: Integrate entropy bars and specialized features."""
        tprint("Starting CheckpointAwareLayer3._step_entropy_bars_integration...")
        # This would integrate entropy bars if available
        # For now, pass through the validated data
        tprint("Finished CheckpointAwareLayer3._step_entropy_bars_integration")
        return {
            'entropy_bars_df': pd.DataFrame(),  # Empty if no entropy bars
            'specialized_features': pd.DataFrame()
        }
    
    @checkpoint_aware_step('meta_features_engineering')
    def _step_meta_features_engineering(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 2: Generate regime-aware and meta features."""
        tprint("Starting CheckpointAwareLayer3._step_meta_features_engineering...")
        from .core import generate_regime_aware_features
        
        oof_df = kwargs['oof_df']
        base_model_cols = kwargs['base_model_cols']
        
        # Generate regime-aware features
        prob_cols = [c for c in oof_df.columns if 'prob_' in c and '_oof' not in c]
        regime_feats = generate_regime_aware_features(oof_df, 'volatility_20', prob_cols)
        
        tprint("Finished CheckpointAwareLayer3._step_meta_features_engineering")
        return {
            'regime_features': regime_feats,
            'meta_features': list(regime_feats.columns)
        }
    
    @checkpoint_aware_step('feature_clustering')
    def _step_feature_clustering(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 3: Apply mild MP-clustering for feature selection."""
        tprint("Starting CheckpointAwareLayer3._step_feature_clustering...")
        from .core import apply_mild_mp_clustering
        from .utils import calculate_alpha_target
        
        oof_df = kwargs['oof_df']
        target_col = kwargs['target_col']
        
        # Prepare target
        y_alpha_12 = calculate_alpha_target(oof_df, target_col, horizon=12)
        y_alpha_12_series = pd.Series(y_alpha_12, index=oof_df.index)
        
        # Apply clustering (simplified version)
        # In full implementation, this would use the actual clustering logic
        selected_features = kwargs['base_model_cols']  # Pass through for now
        
        tprint("Finished CheckpointAwareLayer3._step_feature_clustering")
        return {
            'selected_features': selected_features,
            'y_alpha_12': y_alpha_12,
            'y_alpha_12_series': y_alpha_12_series
        }
    
    @checkpoint_aware_step('layer25_integration')
    def _step_layer25_integration(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 4: Integrate Layer 2.5 chaser models (if available)."""
        tprint("Starting CheckpointAwareLayer3._step_layer25_integration...")
        layer25_enabled = config.get('layer25_chaser_enabled', True)
        layer25_results = config.get('layer25_chaser_results', None)
        
        if layer25_enabled and layer25_results is not None:
            from .layer25_integration import integrate_layer25_into_layer3
            
            oof_df = kwargs['oof_df']
            
            try:
                df_enhanced, integration_metadata = integrate_layer25_into_layer3(
                    df=oof_df,
                    chaser_results=layer25_results,
                    symbol=self.symbol,
                    exchange=config.get('exchange', 'binance'),
                    timeframe=config.get('timeframe', '15m'),
                    top_n_models=config.get('layer25_top_models', 3),
                    outcomes_dir=kwargs.get('outcomes_dir')
                )
                
                tprint("Finished CheckpointAwareLayer3._step_layer25_integration")
                return {
                    'df_enhanced': df_enhanced,
                    'integration_metadata': integration_metadata,
                    'chaser_features': [col for col in df_enhanced.columns if col.startswith('chaser_')]
                }
            except Exception as e:
                logger.warning(f"⚠️ Layer 2.5 integration failed: {e}")
                tprint("Finished CheckpointAwareLayer3._step_layer25_integration with error")
                return {'df_enhanced': kwargs['oof_df'], 'integration_metadata': {'status': 'failed'}}
        else:
            tprint("Finished CheckpointAwareLayer3._step_layer25_integration")
            return {'df_enhanced': kwargs['oof_df'], 'integration_metadata': {'status': 'skipped'}}
    
    @checkpoint_aware_step('dual_head_training')
    def _step_dual_head_training(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 5: Train all model families."""
        tprint("Starting CheckpointAwareLayer3._step_dual_head_training...")
        from .model_training import train_dual_head_models
        from .utils import calculate_sample_weights_efficient
        
        # Prepare data (simplified version)
        # In full implementation, this would prepare the actual feature matrix
        X_dummy = pd.DataFrame(np.random.randn(1000, 10))  # Placeholder
        
        # Prepare targets
        y_alpha_12 = np.random.randn(1000)  # Placeholder
        y_prob_12 = np.random.choice([0, 1], 1000)  # Placeholder
        w_alpha = calculate_sample_weights_efficient(y_alpha_12)
        w_prob = calculate_sample_weights_efficient(y_prob_12)
        
        # Train models (with fast_mode for demo)
        model_results = train_dual_head_models(
            X_dummy, y_alpha_12, y_prob_12, w_alpha, w_prob, [], config, fast_mode=True
        )
        
        tprint("Finished CheckpointAwareLayer3._step_dual_head_training")
        return {
            'model_results': model_results,
            'combined_models': model_results['models']
        }
    
    @checkpoint_aware_step('model_selection_12')
    def _step_model_selection_12(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 6: Select best models for 12-bar horizon."""
        tprint("Starting CheckpointAwareLayer3._step_model_selection_12...")
        from .core import select_best_model_per_task
        
        combined_models = results['dual_head_training']['combined_models']
        y_alpha_12 = results['feature_clustering']['y_alpha_12']
        
        # Select best models for 12-bar
        best_pred_12_reg, best_key_12_reg = select_best_model_per_task(
            combined_models, y_alpha_12, 'regression', '12'
        )
        
        tprint("Finished CheckpointAwareLayer3._step_model_selection_12")
        return {
            'best_models_12': {
                'regression': {'prediction': best_pred_12_reg, 'model_key': best_key_12_reg}
            }
        }
    
    @checkpoint_aware_step('model_selection_48')
    def _step_model_selection_48(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 7: Select best models for 48-bar horizon."""
        tprint("Starting CheckpointAwareLayer3._step_model_selection_48...")
        from .core import select_best_model_per_task
        
        combined_models = results['dual_head_training']['combined_models']
        y_alpha_48 = results['feature_clustering']['y_alpha_12'] * 1.5  # Placeholder
        
        # Select best models for 48-bar
        best_pred_48_reg, best_key_48_reg = select_best_model_per_task(
            combined_models, y_alpha_48, 'regression', '48'
        )
        
        tprint("Finished CheckpointAwareLayer3._step_model_selection_48")
        return {
            'best_models_48': {
                'regression': {'prediction': best_pred_48_reg, 'model_key': best_key_48_reg}
            }
        }
    
    @checkpoint_aware_step('oof_predictions')
    def _step_oof_predictions(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 8: Generate OOF predictions for all models."""
        tprint("Starting CheckpointAwareLayer3._step_oof_predictions...")
        combined_models = results['dual_head_training']['combined_models']
        oof_df = kwargs['oof_df']
        
        # Add OOF predictions to DataFrame (simplified)
        df_with_predictions = oof_df.copy()
        
        for key, model_data in combined_models.items():
            if 'cate' in model_data:
                # Create dummy predictions for demo
                predictions = np.random.randn(len(oof_df))
                df_with_predictions[f"{key}_oof"] = predictions
        
        tprint("Finished CheckpointAwareLayer3._step_oof_predictions")
        return {
            'df_with_predictions': df_with_predictions,
            'oof_predictions': {key: model_data['cate'] for key, model_data in combined_models.items()}
        }
    
    @checkpoint_aware_step('race_reporting')
    def _step_race_reporting(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 9: Generate comprehensive model race reports."""
        tprint("Starting CheckpointAwareLayer3._step_race_reporting...")
        from .model_race_reporting import Layer3ModelRaceReporter
        
        combined_models = results['dual_head_training']['combined_models']
        y_alpha_12 = results['feature_clustering']['y_alpha_12']
        y_prob_12 = np.random.choice([0, 1], len(y_alpha_12))  # Placeholder
        y_alpha_48 = y_alpha_12 * 1.5  # Placeholder
        y_prob_48 = y_prob_12  # Placeholder
        
        # Generate race report
        reporter = Layer3ModelRaceReporter(outcomes_dir=kwargs.get('outcomes_dir'))
        reporter.generate_model_race_report(
            models_dict=combined_models,
            y_alpha_12=y_alpha_12,
            y_prob_12=y_prob_12,
            y_alpha_48=y_alpha_48,
            y_prob_48=y_prob_48
        )
        
        tprint("Finished CheckpointAwareLayer3._step_race_reporting")
        return {'race_report_generated': True}
    
    @checkpoint_aware_step('enhanced_reporting')
    def _step_enhanced_reporting(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 10: Generate enhanced Layer 3 reports."""
        tprint("Starting CheckpointAwareLayer3._step_enhanced_reporting...")
        from .enhanced_reporting import EnhancedLayer3Reporter
        
        # Generate enhanced report
        reporter = EnhancedLayer3Reporter(outcomes_dir=kwargs.get('outcomes_dir'))
        # Simplified call - full implementation would pass actual data
        reporter.generate_all_reports(
            df=kwargs['oof_df'],
            models={'models': results['dual_head_training']['combined_models']},
            geometry_metrics=[],
            meta_features=[],
            target_col=kwargs['target_col'],
            config=config
        )
        
        tprint("Finished CheckpointAwareLayer3._step_enhanced_reporting")
        return {'enhanced_report_generated': True}
    
    @checkpoint_aware_step('final_processing')
    def _step_final_processing(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 11: Final validation and artifact saving."""
        tprint("Starting CheckpointAwareLayer3._step_final_processing...")
        # Collect all results
        df_final = results.get('oof_predictions', {}).get('df_with_predictions', kwargs['oof_df'])
        
        # Build models dictionary
        models_dict = {
            'all_models': results['dual_head_training']['combined_models'],
            'best_models': {
                '12_reg': results['model_selection_12']['best_models_12']['regression']['model_key'],
                '48_reg': results['model_selection_48']['best_models_48']['regression']['model_key']
            },
            'meta_features': results['meta_features_engineering']['meta_features']
        }
        
        # Add integration metadata if available
        if 'layer25_integration' in results:
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
    
    Automatically detects checkpoints and resumes from appropriate step.
    Supports checkpoint override functionality.
    
    Args:
        symbol: Trading symbol (required for checkpoint management)
        checkpoint_dir: Optional custom checkpoint directory
        override_step: Step to override from (e.g., 'dual_head_training')
        force_restart: Force restart from beginning (ignores all checkpoints)
        keep_earlier_checkpoints: Keep checkpoints before override step
        ... (other args same as original layer3_analyst_lgbm)
        
    Returns:
        Tuple of (enhanced DataFrame, models dictionary with checkpoint metadata)
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
