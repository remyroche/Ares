"""Meta-Labeling HPO Sample Weighted Step.

This step orchestrates the Layer 2 -> Layer 3 pipeline for label generation
and calibration.

Layer 2: Regime-Conditional Geometry Optimization (LabelBasedLayer2)
- Optimizes Barrier Geometries (TP/SL/Horizon) per barrier family.
- Selects diverse geometries.
- Generates Bagged OOF Labels and Weights (K-Fold OOF for analytics).
- Also generates Production Geometries (Full Fit).

Layer 3: Calibration & Meta-Model (LabelBasedLayer3)
- Feature Engineering on Layer 2 outputs (Disagreement, Volatility).
- Weights adjustment using Magnitude and Layer 1 weights.
- Calibrated Probability generation using LGBM + Isotonic Regression (K-Fold OOF).
- Final Model training on full dataset.

Layer 4: Position Sizing & Diagnostics (LabelBasedLayer4)
- Converts calibrated probabilities to position sizes.
- Computes advanced diagnostics (Edge Monotonicity, Bet Efficiency).
- Generates final sized events for backtesting.

This replaces the legacy HierarchicalParameterOptimizer loop.
"""

from __future__ import annotations

from src.training.steps.labeling.multi_label_voting_utils import (
    TripleBarrierConfig,
    compute_multi_triple_barrier_outcomes_vectorized,
    compute_kalman_smoothed_price_and_volatility,
    compute_committee_voted_labels_full,
)
from src.training.steps.labeling.label_based_layer_0 import run_layer_0

from typing import Any, Dict, List, Tuple, Optional
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_success, tprint_warning, tprint_info, tprint_error

from src.training.steps.labeling.label_based_layer_2 import LabelBasedLayer2
from src.training.steps.labeling.label_based_layer_3 import layer3_analyst_lgbm, plot_diagnostics
# Import Layer 4
from src.training.steps.labeling.label_based_layer_4 import Layer4PositionSizer

from src.training.steps.labeling.feature_generation_meta_labeling_step import (
    create_meta_features,
)
from src.training.steps.labeling.lgbm_feature_selection import FeatureSetPersistence
from src.training.steps.labeling.label_config import (
    build_label_config,
    compute_label_config_id,
)

from src.utils.ml_common.optimization.hierarchical_parameter_optimizer import (
    HierarchicalParameterOptimizer,
    create_param_group,
    OptimizationStage,
)
from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
    BayesianTPEOptimizer,
    OptimizationConfig,
)
from src.utils.ml_common.standardized_xgb_trainer import (
    StandardizedXGBTrainer,
    XGBTrainingConfig,
)
from src.utils.ml_common.optimization.pareto import (
    Solution,
    ParetoFront,
    compute_pareto_front,
    select_knee_point,
)


class MetaLabelingHPOSampleWeightedStep(BaseStep):
    """
    Orchestrates the Layer 2 -> Layer 3 -> Layer 4 meta-labeling pipeline.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _load_market_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Load market data using the configured data loader."""
        # Use BaseStep's method if available or implement locally
        # Assuming standard implementation pattern:
        try:
            return super().load_market_data(config)
        except AttributeError:
            # Fallback if BaseStep doesn't expose load_market_data directly in this version
            from src.data.data_loader import DataLoader
            loader = DataLoader()
            return loader.load_data(config)

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the pipeline.
        
        Args:
            config: Configuration dictionary.
        """
        # Load market data (using standard BaseStep mechanism)
        market_data = self._load_market_data(config)
        
        if market_data is None or market_data.empty:
            tprint_error("Failed to load market data.")
            return {"success": False}

        # ---------------------------------------------------------
        # LAYER 2: Geometry Optimization & Bagged Labeling
        # ---------------------------------------------------------
        tprint_info(">>> Executing Layer 2: Geometry Optimization (OOF & Full)...")
        
        layer2 = LabelBasedLayer2(
            transaction_cost=float(config.get('transaction_cost', 0.001)),
            n_trials=int(config.get('layer2_n_trials', 30)),
            n_splits=int(config.get('layer2_n_splits', 3)),
            verbose=True
        )
        
        # This now returns OOF labels AND Production Geometries
        l2_output = layer2.execute(market_data, config)
        
        outcomes_dir = Path(config.get("outcomes_dir", "outcomes"))
        outcomes_dir.mkdir(parents=True, exist_ok=True)
        
        # Unpack Layer 2 Artifacts (OOF for Training/Analytics)
        l2_labels = l2_output['oof_labels']
        l2_returns = l2_output['oof_returns']
        l2_weights = l2_output['weights']
        individual_geos = l2_output['individual_geometries']
        events_df = l2_output['events_df']
        selected_trials = l2_output['selected_trials'] # Production Geometries
        
        # Save Layer 2 Production Geometries (Optimized on Full Data)
        with open(outcomes_dir / "layer2_selected_geometries.json", "w") as f:
            json.dump(selected_trials, f, indent=2, default=str)
            
        # ---------------------------------------------------------
        # Component Weights Preparation for Layer 3 Comparison
        # ---------------------------------------------------------
        
        # Try load weights from config or previous step if passed
        target_sample_weight = config.get('target_sample_weight')

        # Layer 1 Weights
        if target_sample_weight is not None:
             if len(target_sample_weight) == len(market_data):
                 w_l1_series = pd.Series(target_sample_weight, index=market_data.index)
                 w_l1_aligned = w_l1_series.reindex(events_df.index).fillna(1.0)
             else:
                 tprint_warning(f"Layer 1 weights length mismatch ({len(target_sample_weight)} vs {len(market_data)}). Using 1.0.")
                 w_l1_aligned = pd.Series(1.0, index=events_df.index)
        else:
             w_l1_aligned = pd.Series(1.0, index=events_df.index)
        
        # Layer 2 Weights (Composite from Geometry Bagging)
        w_l2_aligned = l2_weights # Already aligned to events_df
        
        # Net Returns (for Magnitude)
        l2_returns_aligned = l2_returns # Already aligned to events_df
        
        # ---------------------------------------------------------
        # Data Assembly for Layer 3
        # ---------------------------------------------------------
        tprint_info(">>> Preparing OOF Data for Layer 3...")
        
        # Assemble OOF predictions from individual geometries
        geo_preds_df = pd.DataFrame(index=events_df.index)
        for uuid, preds in individual_geos.items():
            # preds are already Series on the correct index (or reindex safe)
            geo_preds_df[uuid] = preds.reindex(events_df.index)
            
        geo_cols = list(geo_preds_df.columns)
        
        l3_input_df = geo_preds_df.copy()
        
        context_cols = ['volatility_1d']
        for c in context_cols:
            if c in events_df.columns:
                l3_input_df[c] = events_df[c]
            elif c in market_data.columns:
                 l3_input_df[c] = market_data.loc[l3_input_df.index, c]
        
        target_col = 'l2_consensus_target'
        l3_input_df[target_col] = l2_labels
        
        # ---------------------------------------------------------
        # LAYER 3: Calibration & Meta-Model (OOF & Final)
        # ---------------------------------------------------------
        tprint_info(">>> Executing Layer 3: Weighting Scheme Comparison & Training...")
        
        # Passes components to allow Layer 3 to compare 7 weighting schemes
        oof_export, final_model = layer3_analyst_lgbm(
            oof_df=l3_input_df,
            base_model_cols=geo_cols,
            target_col=target_col,
            train_split_date=None,
            layer1_weight=w_l1_aligned.values,
            layer2_weight=w_l2_aligned.values,
            net_returns=l2_returns_aligned.values
        )
        
        # Calculate final composite weight for artifact saving (using Scheme 7 logic as default/reference)
        # Note: The actual model training inside layer3 uses the BEST scheme found.
        # But for 'weights_stats.csv', we save the reference composite one.
        magnitude_factor = np.log1p(l2_returns.abs().fillna(0))
        w_final_series = w_l2_aligned * magnitude_factor * w_l1_aligned
        if w_final_series.mean() > 0:
            w_final_series /= w_final_series.mean()
        w_final = w_final_series.values

        # Generate Diagnostics (on OOF predictions)
        tprint_info(">>> Generating Layer 3 Diagnostics...")
        plot_diagnostics(
            y_true=oof_export[target_col],
            y_prob=oof_export['meta_prob'],
            output_path=str(outcomes_dir / "layer3_calibration_plot.png")
        )
        
        # Save OOF Predictions (Full History)
        layer3_oof_path = outcomes_dir / "layer3_oof_preds.csv"
        oof_export.to_csv(layer3_oof_path)
        
        # Save Weights
        pd.DataFrame({'weight': w_final}).describe().to_csv(outcomes_dir / "layer3_weights_stats.csv")
        
        # Save Final Model
        joblib.dump(final_model, outcomes_dir / "layer3_final_model.joblib")

        # ---------------------------------------------------------
        # LAYER 4: Position Sizing & Portfolio Diagnostics
        # ---------------------------------------------------------
        tprint_info(">>> Executing Layer 4: Position Sizing & Portfolio Diagnostics...")

        # Prepare Data for Layer 4
        # We need realized returns for backtesting. Layer 2 output 'oof_returns' is
        # the realized return of the *best geometry* for that event (or average).
        # We assume 'oof_export' is aligned with 'l2_returns'.

        l4_input = oof_export.copy()

        # Attach realized returns if not present (from Layer 2 OOF returns)
        if 'realized_return' not in l4_input.columns:
            l4_input['realized_return'] = l2_returns.reindex(l4_input.index).fillna(0.0)

        # Attach volatility if not present
        if 'volatility_1d' not in l4_input.columns and 'volatility_1d' in l3_input_df.columns:
            l4_input['volatility_1d'] = l3_input_df['volatility_1d']

        # Initialize Sizer
        sizer = Layer4PositionSizer(
            oof_df=l4_input,
            p_col='meta_prob',
            target_col=target_col,
            return_col='realized_return',
            transaction_cost=float(config.get('transaction_cost', 0.001)),
            gamma=1.2
        )

        # Run Backtest
        l4_metrics = sizer.run_backtest()

        # Save Layer 4 Artifacts
        sizer.save_artifacts(outcomes_dir)

        with open(outcomes_dir / "layer4_performance_metrics.json", "w") as f:
            json.dump(l4_metrics, f, indent=2, default=str)

        tprint_success(f"Layer 4 Completed. Metrics: {json.dumps(l4_metrics, indent=2, default=str)}")

        tprint_success(f"Pipeline Completed. Artifacts saved to {outcomes_dir}")
        
        return {
            "success": True,
            "outcomes_dir": str(outcomes_dir),
            "metrics": {
                "n_events": len(l3_input_df),
                "n_geometries": len(geo_cols),
                **l4_metrics
            },
            "artifacts": {
                "oof_preds": str(layer3_oof_path),
                "calibration_plot": str(outcomes_dir / "layer3_calibration_plot.png"),
                "final_model": str(outcomes_dir / "layer3_final_model.joblib"),
                "layer2_geometries": str(outcomes_dir / "layer2_selected_geometries.json"),
                "layer4_events": str(outcomes_dir / "layer4_sized_events.csv"),
                "layer4_metrics": str(outcomes_dir / "layer4_performance_metrics.json")
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
