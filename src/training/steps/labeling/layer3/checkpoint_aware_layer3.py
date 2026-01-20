"""
Checkpoint-Aware Layer 3 Wrapper

Provides automatic checkpoint detection and resumption for Layer 3 execution.
Symbol-specific checkpoint management with intelligent resume logic.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from ..checkpoint_aware_runner import CheckpointAwareRunner, checkpoint_aware_step
from ..checkpoint_override_manager import CheckpointOverrideManager, create_checkpoint_override
from .core import (
    integrate_entropy_bars_into_layer3,
    generate_regime_aware_features,
    apply_mild_mp_clustering,
    select_best_model_per_task,
    get_or_fit_regime_labels,
    build_env_indices_for_index,
    should_use_cached_features,
    load_layer3_features_from_cache,
    save_layer3_features_to_cache
)
from .utils import (
    calculate_alpha_target,
    calculate_sample_weights_efficient,
    calculate_studentized_har_target,
    calculate_blended_forward_returns,
    validate_feature_matrix
)
from .model_training import train_dual_head_models
from .enhanced_reporting import EnhancedLayer3Reporter
from .model_race_reporting import Layer3ModelRaceReporter

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
        self.symbol = symbol.upper()
        self.checkpoint_dir = checkpoint_dir
        
        # Initialize checkpoint-aware runner
        self.runner = CheckpointAwareRunner('layer3', self.symbol, checkpoint_dir)
        
        logger.info(f"🔧 Initialized checkpoint-aware Layer 3 for {self.symbol}")
        logger.info(f"📍 Resume step: {self.runner.execution_plan.resume_step}")
    
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
        return self.run(
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
                df_final = latest_result.get('df', oof_df)
            else:
                df_final = oof_df  # Fallback to input
            
            models_dict = latest_result.get('models_dict', {})
        
        # Add execution metadata
        models_dict['checkpoint_metadata'] = result['metadata']
        
        logger.info(f"🎉 Checkpoint-aware Layer 3 completed for {self.symbol}")
        logger.info(f"📊 Steps executed: {result['metadata']['steps_executed']}")
        logger.info(f"💾 Checkpoints saved: {len(result['metadata']['checkpoints_saved'])}")
        
        return df_final, models_dict
    
    def _get_step_functions(self) -> Dict[str, callable]:
        """Get step functions for checkpoint-aware execution."""
        return {
            'data_loading': self._step_data_loading,
            'entropy_bars_integration': self._step_entropy_bars_integration,
            'meta_features_engineering': self._step_meta_features_engineering,
            'target_generation': self._step_target_generation,
            'feature_clustering': self._step_feature_clustering,
            'irm_regime_detection': self._step_irm_regime_detection,
            'dual_head_training': self._step_dual_head_training,
            'model_selection': self._step_model_selection,
            'oof_predictions': self._step_oof_predictions,
            'race_reporting': self._step_race_reporting,
            'enhanced_reporting': self._step_enhanced_reporting,
            'final_processing': self._step_final_processing
        }
    
    @checkpoint_aware_step('data_loading')
    def _step_data_loading(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 0: Load and prepare data."""
        oof_df = kwargs['oof_df']
        base_model_cols = kwargs['base_model_cols']
        
        # Use config for base model columns if not explicitly provided or empty, but prefer args
        if not base_model_cols and 'base_model_cols' in config:
            base_model_cols = config['base_model_cols']

        return {
            'oof_df': oof_df,
            'base_model_cols': base_model_cols
        }
    
    @checkpoint_aware_step('entropy_bars_integration')
    def _step_entropy_bars_integration(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 1: Integrate entropy bars and specialized features."""
        oof_df = results['data_loading']['oof_df']
        symbol = kwargs.get('symbol', 'ETHUSDT')
        # Try to get exchange from config or default to binance
        exchange = config.get('exchange', 'binance')

        if config.get('use_entropy_bars', True):
            try:
                df_enhanced, entropy_bars_df = integrate_entropy_bars_into_layer3(
                    oof_df, symbol, exchange, config
                )
                return {
                    'enhanced_df': df_enhanced,
                    'entropy_bars_df': entropy_bars_df
                }
            except Exception as e:
                logger.warning(f"⚠️ Entropy bars integration failed: {e}. Proceeding without.")
                return {
                    'enhanced_df': oof_df.copy(),
                    'entropy_bars_df': pd.DataFrame()
                }
        else:
            return {
                'enhanced_df': oof_df.copy(),
                'entropy_bars_df': pd.DataFrame()
            }
    
    @checkpoint_aware_step('meta_features_engineering')
    def _step_meta_features_engineering(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 2: Generate regime-aware and meta features."""
        df = results['entropy_bars_integration']['enhanced_df']
        base_model_cols = results['data_loading']['base_model_cols']
        market_data = kwargs.get('market_data')
        symbol = kwargs.get('symbol', 'ETHUSDT')
        exchange = config.get('exchange', 'binance')
        
        safe_base_cols = [c for c in base_model_cols if c in df.columns]

        # Try loading features from cache
        features_loaded = False
        try:
            if should_use_cached_features(config, symbol, exchange, config.get('timeframe', '15m'), 'long'):
                cached_features, _ = load_layer3_features_from_cache(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=config.get('timeframe', '15m'),
                    direction='long',
                    target_index=df.index,
                    market_data=market_data,
                    validate_hash=True
                )

                if cached_features is not None:
                    # Merge cached features into df
                    new_cols = [c for c in cached_features.columns if c not in df.columns]
                    if new_cols:
                        df = pd.concat([df, cached_features[new_cols]], axis=1)
                        features_loaded = True
        except Exception as e:
             logger.warning(f"⚠️ Cache loading failed: {e}")

        if not features_loaded:
            try:
                from src.feature_generation.categories.layer3_specific_features import generate_layer3_features
                df = generate_layer3_features(df, safe_base_cols)

                # Save to cache if enabled
                if config.get('use_layer3_feature_cache', True):
                    # Identify generated features (exclude base columns)
                    # We need original oof_df columns to know what was added, but strictly speaking
                    # generate_layer3_features adds specific known features.
                    # Simpler approach: save everything that is not in base_model_cols + basic OHLCV
                    exclude_cols = set(kwargs['oof_df'].columns) | set(base_model_cols) | {'close', 'high', 'low', 'open', 'volume'}
                    generated_cols = [c for c in df.columns if c not in exclude_cols]

                    if generated_cols:
                        feature_subset = df[generated_cols]
                        save_layer3_features_to_cache(
                            meta_features=feature_subset,
                            symbol=symbol,
                            exchange=exchange,
                            timeframe=config.get('timeframe', '15m'),
                            direction='long',
                            market_data=market_data,
                            config=config
                        )
            except Exception as e:
                logger.warning(f"⚠️ Feature generation failed: {e}")

        return {
            'df_with_features': df,
            'base_model_cols': base_model_cols
        }

    @checkpoint_aware_step('target_generation')
    def _step_target_generation(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 3: Generate targets and weights."""
        df = results['meta_features_engineering']['df_with_features']
        net_returns = kwargs.get('net_returns')
        target_col = kwargs['target_col']
        layer1_weight = kwargs.get('layer1_weight')
        
        # Calculate Returns and Volatility
        if net_returns is None:
            if 'close' in df.columns:
                net_returns = df['close'].pct_change().fillna(0)
            else:
                net_returns = pd.Series(0, index=df.index)
        
        ret_series = net_returns.reindex(df.index)
        vol_series = ret_series.rolling(24).std().fillna(0.001)

        # 12-bar targets
        if 'close' in df.columns:
            # Horizons: 16 (4h) and 24 (6h)
            blended_ret_series = calculate_blended_forward_returns(df['close'], [16, 24])
        else:
            blended_ret_series = ret_series

        y_alpha_12_series = calculate_studentized_har_target(blended_ret_series, vol_series)
        y_alpha_12 = y_alpha_12_series.values.astype(np.float32)
        y_prob_12 = (blended_ret_series.values > 0).astype(np.int32)
        
        # 48-bar targets
        if 'close' in df.columns:
            ret_48 = df['close'].shift(-48) / df['close'] - 1
            vol_48 = ret_series.rolling(48).std().fillna(0.001)
            y_alpha_48_series = calculate_studentized_har_target(ret_48.fillna(0), vol_48.fillna(0))
            y_alpha_48 = y_alpha_48_series.values.astype(np.float32)
            y_prob_48 = (ret_48.fillna(0) > 0).astype(np.int32)
        else:
            y_alpha_48 = y_alpha_12 * 1.5
            y_prob_48 = y_prob_12

        # Weights
        vol_values = df['volume'].values.astype(np.float32) if 'volume' in df.columns else None

        w_alpha = calculate_sample_weights_efficient(
            ret_series.values,
            vol_series.values,
            layer1_weights=layer1_weight.values if layer1_weight is not None else None,
            volume=vol_values
        )
        w_alpha = w_alpha.astype(np.float32)

        return {
            'y_alpha_12': y_alpha_12,
            'y_prob_12': y_prob_12,
            'y_alpha_48': y_alpha_48,
            'y_prob_48': y_prob_48,
            'w_alpha': w_alpha,
            'y_alpha_12_series': y_alpha_12_series  # Needed for clustering
        }

    @checkpoint_aware_step('feature_clustering')
    def _step_feature_clustering(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 4: Prepare final feature matrix and apply clustering."""
        df = results['meta_features_engineering']['df_with_features']
        base_model_cols = results['data_loading']['base_model_cols']
        y_alpha_12_series = results['target_generation']['y_alpha_12_series']
        target_col = kwargs['target_col']
        outcomes_dir = kwargs.get('outcomes_dir')

        exclude = set(base_model_cols) | {target_col, 'close', 'high', 'low', 'volume', 'regime_label'}
        meta_features = [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64]]
        X_full = df[meta_features].copy()

        # Add base model columns back for training
        safe_base_cols = [c for c in base_model_cols if c in df.columns]
        for col in safe_base_cols:
            if col in df.columns:
                X_full[col] = df[col].reindex(X_full.index)

        # Regime Aware Features
        prob_cols = [c for c in X_full.columns if 'prob_' in c and '_oof' not in c]
        regime_feats = generate_regime_aware_features(X_full, 'volatility_20', prob_cols)
        X_full = pd.concat([X_full, regime_feats], axis=1)

        # Layer 2.5 Integration
        layer25_enabled = config.get('layer25_chaser_enabled', True)
        layer25_results = config.get('layer25_chaser_results', None)
        
        integration_metadata = {'status': 'skipped'}

        if layer25_enabled and layer25_results is not None:
            try:
                from .layer25_integration import integrate_layer25_into_layer3

                # We need to construct a DF that looks like original but with X_full cols
                df_for_integration = df.copy()
                for col in X_full.columns:
                    if col not in df_for_integration.columns:
                        df_for_integration[col] = X_full[col]

                df_enhanced, integration_metadata = integrate_layer25_into_layer3(
                    df=df_for_integration,
                    chaser_results=layer25_results,
                    symbol=self.symbol,
                    exchange=config.get('exchange', 'binance'),
                    timeframe=config.get('timeframe', '15m'),
                    top_n_models=config.get('layer25_top_models', 3),
                    outcomes_dir=outcomes_dir
                )
                
                # Update X_full with new chaser features
                chaser_features = [col for col in df_enhanced.columns if col.startswith('chaser_')]
                for feature in chaser_features:
                    if feature not in X_full.columns:
                        X_full[feature] = df_enhanced[feature]
                        df[feature] = df_enhanced[feature] # Also update main df for posterity

            except Exception as e:
                logger.warning(f"⚠️ Layer 2.5 integration failed: {e}")
                integration_metadata = {'status': 'failed', 'error': str(e)}

        # Apply Clustering
        X_clustered = apply_mild_mp_clustering(X_full, threshold=0.98, target=y_alpha_12_series)

        return {
            'X_clustered': X_clustered,
            'meta_features': meta_features, # Original meta features
            'integration_metadata': integration_metadata,
            'final_feature_list': X_clustered.columns.tolist() # Save list of features for reproducibility
        }

    @checkpoint_aware_step('irm_regime_detection')
    def _step_irm_regime_detection(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 5: Detect IRM regimes."""
        df = results['meta_features_engineering']['df_with_features']
        X_clustered = results['feature_clustering']['X_clustered']

        irm_env_indices = []
        if config.get("irm_meta_enabled", True):
            try:
                gmm_dir = Path(config.get("irm_regime_dir", "artifacts/irm_regimes"))
                gmm_dir.mkdir(parents=True, exist_ok=True)
                regime_labels = get_or_fit_regime_labels(
                    df,
                    gmm_dir / "layer3_meta_gmm.pkl",
                    n_regimes=config.get("irm_meta_regimes", 2),
                    refit=config.get("irm_refit_regimes", False)
                )
                irm_env_indices = build_env_indices_for_index(regime_labels, X_clustered.index)
            except Exception as e:
                logger.warning(f"⚠️ IRM Regime detection failed: {e}. Proceeding without IRM.")

        return {'irm_env_indices': irm_env_indices}
    
    @checkpoint_aware_step('dual_head_training')
    def _step_dual_head_training(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 6: Train all model families."""
        X_clustered = results['feature_clustering']['X_clustered']

        y_alpha_12 = results['target_generation']['y_alpha_12']
        y_prob_12 = results['target_generation']['y_prob_12']
        y_alpha_48 = results['target_generation']['y_alpha_48']
        y_prob_48 = results['target_generation']['y_prob_48']
        
        w_alpha = results['target_generation']['w_alpha']
        
        irm_env_indices = results['irm_regime_detection']['irm_env_indices']
        
        # Prepare Config for training
        train_config = config.copy()
        train_config["irm_env_indices"] = irm_env_indices
        train_config["irm_lambda"] = config.get("irm_meta_lambda", 2.0)
        train_config['y_alpha_48'] = y_alpha_48
        train_config['y_prob_48'] = y_prob_48

        # Train models
        model_results = train_dual_head_models(
            X_clustered,
            y_alpha_12,
            y_prob_12,
            w_alpha,
            w_alpha, # Use same weights for prob for now, or derive w_prob if needed
            [], # cv_splits not used in current train_dual_head_models implementation (it does internal split)
            train_config,
            config.get('fast_mode', False)
        )
        
        return {
            'model_results': model_results,
            'combined_models': model_results['models']
        }
    
    @checkpoint_aware_step('model_selection')
    def _step_model_selection(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 7: Select best models."""
        combined_models = results['dual_head_training']['combined_models']
        y_alpha_12 = results['target_generation']['y_alpha_12']
        y_prob_12 = results['target_generation']['y_prob_12']
        y_alpha_48 = results['target_generation']['y_alpha_48']
        y_prob_48 = results['target_generation']['y_prob_48']
        
        best_pred_12_reg, best_key_12_reg = select_best_model_per_task(combined_models, y_alpha_12, 'regression', '12')
        best_pred_12_cls, best_key_12_cls = select_best_model_per_task(combined_models, y_prob_12, 'classification', '12')
        best_pred_48_reg, best_key_48_reg = select_best_model_per_task(combined_models, y_alpha_48, 'regression', '48')
        best_pred_48_cls, best_key_48_cls = select_best_model_per_task(combined_models, y_prob_48, 'classification', '48')
        
        return {
            'best_models': {
                '12_reg': {'prediction': best_pred_12_reg, 'model_key': best_key_12_reg},
                '12_cls': {'prediction': best_pred_12_cls, 'model_key': best_key_12_cls},
                '48_reg': {'prediction': best_pred_48_reg, 'model_key': best_key_48_reg},
                '48_cls': {'prediction': best_pred_48_cls, 'model_key': best_key_48_cls}
            }
        }
    
    @checkpoint_aware_step('oof_predictions')
    def _step_oof_predictions(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 8: Generate OOF predictions and finalize DataFrame."""
        combined_models = results['dual_head_training']['combined_models']
        best_models = results['model_selection']['best_models']
        df = results['meta_features_engineering']['df_with_features'].copy()
        X_clustered_index = results['feature_clustering']['X_clustered'].index
        
        # Helper to propagate predictions
        def propagate_simple(values, idx):
            return pd.Series(values, index=idx).reindex(df.index).fillna(0)

        # Save ALL models OOF predictions
        for key, res in combined_models.items():
            if 'cate' in res:
                pred = res['cate']
                df[f"{key}_oof"] = propagate_simple(pred, X_clustered_index)
        
        # Map best models to meta columns
        df['meta_alpha'] = propagate_simple(best_models['12_reg']['prediction'], X_clustered_index)
        df['meta_prob'] = propagate_simple(best_models['12_cls']['prediction'], X_clustered_index)
        df['meta_alpha_48'] = propagate_simple(best_models['48_reg']['prediction'], X_clustered_index)
        df['meta_prob_48'] = propagate_simple(best_models['48_cls']['prediction'], X_clustered_index)

        # Legacy compatibility
        df['orf_cate'] = df['meta_alpha']
        df['orf_se'] = df['meta_prob'] * 0.1 # Placeholder
        
        return {'df_final': df}
    
    @checkpoint_aware_step('race_reporting')
    def _step_race_reporting(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 9: Generate comprehensive model race reports."""
        combined_models = results['dual_head_training']['combined_models']
        y_alpha_12 = results['target_generation']['y_alpha_12']
        y_prob_12 = results['target_generation']['y_prob_12']
        y_alpha_48 = results['target_generation']['y_alpha_48']
        y_prob_48 = results['target_generation']['y_prob_48']
        
        outcomes_dir = kwargs.get('outcomes_dir')
        
        try:
            reporter = Layer3ModelRaceReporter(outcomes_dir=outcomes_dir)
            reporter.generate_model_race_report(
                models_dict=combined_models,
                y_alpha_12=y_alpha_12,
                y_prob_12=y_prob_12,
                y_alpha_48=y_alpha_48,
                y_prob_48=y_prob_48
            )
            return {'race_report_generated': True}
        except Exception as e:
            logger.warning(f"⚠️ Race reporting failed: {e}")
            return {'race_report_generated': False}
    
    @checkpoint_aware_step('enhanced_reporting')
    def _step_enhanced_reporting(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 10: Generate enhanced Layer 3 reports."""
        df = results['oof_predictions']['df_final']
        combined_models = results['dual_head_training']['combined_models']
        meta_features = results['feature_clustering']['meta_features']
        outcomes_dir = kwargs.get('outcomes_dir')
        
        try:
            reporter = EnhancedLayer3Reporter(outcomes_dir=outcomes_dir)
            reporter.generate_all_reports(
                df=df,
                models={'models': combined_models},
                geometry_metrics=config.get('geometry_metrics', []),
                meta_features=meta_features,
                target_col='meta_prob', # Primary target for report
                config=config
            )
            return {'enhanced_report_generated': True}
        except Exception as e:
             logger.warning(f"⚠️ Enhanced reporting failed: {e}")
             return {'enhanced_report_generated': False}
    
    @checkpoint_aware_step('final_processing')
    def _step_final_processing(self, results: Dict[str, Any], config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Step 11: Final validation and artifact saving."""
        df_final = results['oof_predictions']['df_final']
        combined_models = results['dual_head_training']['combined_models']
        best_models = results['model_selection']['best_models']
        meta_features = results['feature_clustering']['meta_features']
        irm_env_indices = results['irm_regime_detection']['irm_env_indices']
        entropy_bars_df = results['entropy_bars_integration']['entropy_bars_df']
        
        # Build models dictionary
        models_dict = {
            'all_models': combined_models,
            'best_models': {
                '12_reg': best_models['12_reg']['model_key'],
                '12_cls': best_models['12_cls']['model_key'],
                '48_reg': best_models['48_reg']['model_key'],
                '48_cls': best_models['48_cls']['model_key']
            },
            'meta_features': meta_features,
            'irm_env_indices': irm_env_indices,
            'entropy_bars': entropy_bars_df if not entropy_bars_df.empty else None
        }
        
        # Add integration metadata if available
        if 'layer25_integration' in results:
             if 'integration_metadata' in results['layer25_integration']:
                models_dict['layer25_integration'] = results['layer25_integration']['integration_metadata']
        
        return {
            'df': df_final,
            'models_dict': models_dict
        }
    
    def get_checkpoint_status(self) -> Dict[str, Any]:
        """Get detailed checkpoint status."""
        return self.runner.get_checkpoint_status()
    
    def reset_checkpoints(self) -> int:
        """Reset all checkpoints for this symbol."""
        return self.runner.reset_all_checkpoints()

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
    if symbol is None:
        raise ValueError("Symbol is required for checkpoint-aware execution")
    
    # Create checkpoint-aware Layer 3 instance
    checkpoint_layer3 = CheckpointAwareLayer3(symbol, checkpoint_dir)
    
    # Check if override is requested
    if override_step is not None:
        logger.info(f"🔄 Using checkpoint override from '{override_step}'")
        return checkpoint_layer3.run_with_override(
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
        return checkpoint_layer3.run(
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
