"""Meta-Labeling HPO Sample Weighted Step.

This step orchestrates the Layer 2 -> Layer 3 pipeline for label generation
and calibration.

Layer 2: Regime-Conditional Geometry Optimization (LabelBasedLayer2)
- Optimizes Barrier Geometries (TP/SL/Horizon) per barrier family.
- Selects diverse geometries.
- Generates Bagged OOF Labels and Weights.

Layer 3: Calibration & Meta-Model (LabelBasedLayer3)
- Feature Engineering on Layer 2 outputs (Disagreement, Volatility).
- Weights adjustment using Magnitude and Layer 1 weights.
- Calibrated Probability generation using LGBM + Isotonic Regression.

This replaces the legacy HierarchicalParameterOptimizer loop.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_info, tprint_error

from src.training.steps.labeling.label_based_layer_2 import LabelBasedLayer2
from src.training.steps.labeling.label_based_layer_3 import layer3_analyst_lgbm, plot_diagnostics
from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    create_meta_features,
)

class MetaLabelingHPOSampleWeightedStep(BaseStep):
    """
    Orchestrator for Layer 2 + Layer 3 Meta-Labeling Pipeline.
    """
    
    async def execute(self, config: dict) -> dict:
        """
        Execute the pipeline.
        
        Args:
            config: Configuration dictionary.
        """
        # Load market data (using standard BaseStep mechanism)
        market_data, _ = self.load_market_data_or_fail(config)
        
        # Load primary signals if available (or generate dummy if needed/implied)
        # Usually passed via pipeline_state or artifact, but here we assume self-contained logic if possible
        # For HPO, we often generate primary signals inside the step.
        # But `run_step` signature had `primary_signals`.
        # We'll need to generate them here if not provided.
        
        # Check for target_sample_weight in artifacts/pipeline state?
        # For now, let's proceed with minimal setup.
        
        # We need `run_step` to be compatible with legacy calls if any,
        # but `execute` is the main entry point for BaseStep.
        # We will wrap logic in `_run_pipeline`.
        
        # Generate primary signals (using default logic if not provided)
        from src.training.steps.labeling.feature_generation_meta_labeling_step import generate_primary_signals
        primary_signals = generate_primary_signals(market_data.copy())
        
        # Try load weights
        target_sample_weight = None
        # (Simulated for now as legacy loading is complex)
        
        return self.run_step(market_data, primary_signals, target_sample_weight)

    def run_step(self, market_data: pd.DataFrame, primary_signals: pd.DataFrame,
                 target_sample_weight: np.ndarray = None, **kwargs) -> dict:
        """
        Execute the pipeline (sync method for internal use).
        
        Args:
            market_data: OHLCV + features.
            primary_signals: 'consensus' column.
            target_sample_weight: Weights from Layer 1 (Uniqueness * Consistency).
        """
        config = self.config if hasattr(self, 'config') else kwargs.get('config', {})
        symbol = config.get("symbol", "UNKNOWN")
        exchange = config.get("exchange", "UNKNOWN")
        timeframe = config.get("timeframe", "15m")
        direction = config.get("direction", "long")
        
        run_timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        outcomes_dir = Path("outcomes") / f"meta_labeling_{symbol}_{run_timestamp}"
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        
        tprint_info(f"Starting Meta-Labeling Pipeline for {symbol} {direction}")
        
        # ---------------------------------------------------------
        # Data Preparation
        # ---------------------------------------------------------
        df = market_data.copy()
        
        if 'volatility_1d' not in df.columns:
            df['log_ret'] = np.log(df['close']).diff()
            df['volatility_1d'] = df['log_ret'].rolling(20).std()
            
        # ---------------------------------------------------------
        # LAYER 2: Geometry Optimization & Bagged Labeling
        # ---------------------------------------------------------
        tprint_info(">>> Executing Layer 2: Geometry Optimization...")
        
        layer2 = LabelBasedLayer2(
            transaction_cost=float(config.get('transaction_cost', 0.001)),
            n_trials=int(config.get('layer2_n_trials', 30)),
            n_splits=int(config.get('layer2_n_splits', 3)),
            verbose=True
        )
        
        l2_output = layer2.run(df)
        
        if not l2_output:
            tprint_error("Layer 2 produced no output. Exiting.")
            return {"success": False}
        
        # Unpack Layer 2 Artifacts
        l2_labels = l2_output['oof_labels']
        l2_returns = l2_output['oof_returns'] # Weighted average return of geometries
        l2_weights = l2_output['weights']
        individual_geos = l2_output['individual_geometries']
        events_df = l2_output['events_df']
        selected_trials = l2_output['selected_trials']
        
        # Save Layer 2 selection report
        with open(outcomes_dir / "layer2_selected_geometries.json", "w") as f:
            json.dump(selected_trials, f, indent=2, default=str)
            
        # ---------------------------------------------------------
        # Weight Calculation for Layer 3
        # ---------------------------------------------------------
        # Formula: W_t = W_L2 * log(1 + |R_composite|) * W_L1
        
        if target_sample_weight is not None:
             if len(target_sample_weight) == len(df):
                 w_l1_series = pd.Series(target_sample_weight, index=df.index)
                 w_l1_aligned = w_l1_series.reindex(events_df.index).fillna(1.0)
             else:
                 tprint_warning(f"Layer 1 weights length mismatch ({len(target_sample_weight)} vs {len(df)}). Using 1.0.")
                 w_l1_aligned = pd.Series(1.0, index=events_df.index)
        else:
             w_l1_aligned = pd.Series(1.0, index=events_df.index)
        
        magnitude_factor = np.log1p(l2_returns.abs().fillna(0))
        
        w_final_series = l2_weights * magnitude_factor * w_l1_aligned
        
        if w_final_series.mean() > 0:
            w_final_series /= w_final_series.mean()
        
        w_final = w_final_series.values
        
        # ---------------------------------------------------------
        # Data Assembly for Layer 3
        # ---------------------------------------------------------
        tprint_info(">>> Preparing Data for Layer 3...")
        
        geo_preds_df = pd.DataFrame(index=events_df.index)
        for uuid, preds in individual_geos.items():
            geo_preds_df[uuid] = preds.reindex(events_df.index)
            
        geo_cols = list(geo_preds_df.columns)
        
        l3_input_df = geo_preds_df.copy()
        
        context_cols = ['volatility_1d']
        for c in context_cols:
            if c in events_df.columns:
                l3_input_df[c] = events_df[c]
            elif c in df.columns:
                 l3_input_df[c] = df.loc[l3_input_df.index, c]
        
        target_col = 'l2_consensus_target'
        l3_input_df[target_col] = l2_labels
        
        # ---------------------------------------------------------
        # LAYER 3: Calibration & Meta-Model
        # ---------------------------------------------------------
        tprint_info(">>> Executing Layer 3: Calibration & Meta-Model...")
        
        val_export, calibrated_model = layer3_analyst_lgbm(
            oof_df=l3_input_df,
            base_model_cols=geo_cols,
            target_col=target_col,
            train_split_date=None,
            sample_weight=w_final
        )
        
        # Generate Diagnostics
        tprint_info(">>> Generating Layer 3 Diagnostics...")
        plot_diagnostics(
            y_true=val_export[target_col],
            y_prob=val_export['meta_prob'],
            output_path=str(outcomes_dir / "layer3_calibration_plot.png")
        )
        
        # ---------------------------------------------------------
        # Artifacts & Return
        # ---------------------------------------------------------
        
        val_export.to_csv(outcomes_dir / "layer3_validation_preds.csv")
        
        pd.DataFrame({'weight': w_final}).describe().to_csv(outcomes_dir / "layer3_weights_stats.csv")
        
        tprint_success(f"Pipeline Completed. Artifacts saved to {outcomes_dir}")
        
        return {
            "success": True,
            "outcomes_dir": str(outcomes_dir),
            "metrics": {
                "n_events": len(l3_input_df),
                "n_geometries": len(geo_cols)
            },
            "artifacts": {
                "val_preds": str(outcomes_dir / "layer3_validation_preds.csv"),
                "calibration_plot": str(outcomes_dir / "layer3_calibration_plot.png")
            }
        }

def register_meta_labeling_hpo_sample_weighted_step() -> None:
    """Register the meta-labeling HPO sample weighted step in the registry."""
    from src.training.steps.base_step import step_registry
    step_registry.register("meta_labeling_hpo_sample_weighted", MetaLabelingHPOSampleWeightedStep)
    # Aliases
    step_registry.register("meta_labeling_hpo_experiment", MetaLabelingHPOSampleWeightedStep)
    step_registry.register("sr_labeling_xgb", MetaLabelingHPOSampleWeightedStep)
    step_registry.register("sr_labeling_xgb_weighted", MetaLabelingHPOSampleWeightedStep)

register_meta_labeling_hpo_sample_weighted_step()
