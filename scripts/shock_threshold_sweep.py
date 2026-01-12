#!/usr/bin/env python3
"""
Shock Threshold HPO Tuner (Optuna-Optimized)

This script optimizes the GMM-Shock threshold parameters using Optuna.
It performs the computationally expensive GMM inference ONCE and then
rapidly evaluates thousands of threshold combinations using cached probability arrays.

Target:
- Maximize F1-Score (or Precision/Recall) for detecting significant market volatility/moves.

Usage:
    python3 scripts/shock_threshold_sweep.py --symbol ETHUSDT --trials 100
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json
from datetime import datetime
import warnings

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import TPrint
from src.utils.tprint import tprint_info, tprint_success, tprint_warning, tprint_error

# Import EnhancedGMMFeatures
from src.training.steps.market_analysis.gmm_enhanced_features import EnhancedGMMFeatures

# Import Optuna
try:
    import optuna
    from optuna.samplers import TPESampler
except ImportError:
    tprint_error("❌ Optuna not found. Please install: pip install optuna")
    sys.exit(1)

# Numba optimization for metric calculation
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def njit(func): return func  # Dummy decorator

# -----------------------------------------------------------------------------
# Optimized Logic
# -----------------------------------------------------------------------------

@njit
def calculate_shock_metrics_numba(
    prob_velocities: np.ndarray,
    z_fam_diff: np.ndarray,
    entropy_diff: np.ndarray,
    targets: np.ndarray,
    prob_thresh: float,
    z_thresh: float,
    ent_thresh: float
) -> Tuple[float, float, float]:
    """
    Numba-optimized metric calculation for a single trial.
    Calculates Precision, Recall, and F1 score.
    """
    n_samples = len(targets)
    tp = 0
    fp = 0
    fn = 0

    # Pre-calculate jump magnitudes
    # Assuming prob_velocities is (N, K), we take max jump across K?
    # Or L2 norm as in original code?
    # Original: np.linalg.norm(prob_velocities, axis=1)
    # We'll pass in the pre-calculated norm magnitude to keep this loop fast.

    for i in range(n_samples):
        # Check shock condition
        prob_shock = prob_velocities[i] > prob_thresh
        z_shock = z_fam_diff[i] > z_thresh
        ent_shock = entropy_diff[i] < -ent_thresh # Note: entropy drop is negative

        # Composite signal (Any shock detected)
        # Or should we require logical AND? Original code sums them (logical OR).
        is_shock = (prob_shock or z_shock or ent_shock)

        is_target = targets[i] > 0

        if is_shock and is_target:
            tp += 1
        elif is_shock and not is_target:
            fp += 1
        elif not is_shock and is_target:
            fn += 1

    # Metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

class GMMShockOptimizer(EnhancedGMMFeatures):
    """
    Subclass of EnhancedGMMFeatures designed for HPO.
    Exposes raw GMM arrays and separates inference from detection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.cached_gmm_data = None

    def prepare_data(self, config: Dict[str, Any]) -> None:
        """
        Run the heavy GMM inference pipeline once and cache results.
        """
        tprint_info("🏗️ Preparing GMM Data (One-Time Execution)...")

        # 1. Load Data
        try:
            market_data, _ = self.load_market_data_or_fail(config)
        except Exception as e:
            tprint_warning(f"Standard loader failed: {e}. Trying direct load...")
            # Fallback to direct load
            path = Path("data/historical/binance/ETHUSDT/15m.parquet")
            if path.exists():
                market_data = pd.read_parquet(path)
                tprint_success(f"Loaded data from {path}")
            else:
                raise ValueError("No market data found")

        if market_data is None:
            raise ValueError("Failed to load market data")

        returns = market_data['close'].pct_change()

        # Subsample for speed if needed (e.g. max 50k rows)
        if len(market_data) > 50000:
            tprint_warning(f"⚠️ Dataset large ({len(market_data)}), using last 50,000 samples for optimization.")
            market_data = market_data.tail(50000)
            returns = returns.tail(50000)

        # 2. Base Features
        # Mocking dummy signals
        dummy_signals = pd.DataFrame(index=market_data.index)

        # We need to manually import create_meta_features if not available in class scope
        from src.training.steps.labeling.mtf_feature_generation import create_meta_features
        base_features = create_meta_features(market_data, dummy_signals, volume_available=True)

        # 3. Preprocess
        self._initialize_original_pipeline()
        X_clean = self.original_pipeline._preprocess_features(base_features)

        # 4. Run GMM Inference with Data Capture
        # We define a capture hook to intercept the exact compressed features passed to the GMM.
        # This avoids the pitfall of re-running PCA/Clustering locally which leads to feature space mismatch.

        captured_data = {}

        # We need to ensure the RobustGMM class is available to patch
        from src.training.steps.market_analysis.gmm_based_features import RobustGMM

        original_predict = RobustGMM.predict

        def predict_capture_hook(self_gmm, X_input):
            # Capture the input array (compressed features)
            # We only want to capture it for the 'step_a_gmm' model
            # We can't easily check the model name from inside the instance without extra context,
            # but usually Step A is the first/primary one.
            # We'll capture the first call that matches the expected dimensionality or context.
            if 'compressed_features' not in captured_data:
                captured_data['compressed_features'] = X_input
                captured_data['gmm_instance'] = self_gmm

            return original_predict(self_gmm, X_input)

        # Apply the monkey patch
        RobustGMM.predict = predict_capture_hook

        try:
            tprint_info("   🧠 Running GMM pipeline with data capture hook...")
            _ = self.original_pipeline._step_a_macro_state(X_clean, returns)
        finally:
            # Restore original method immediately
            RobustGMM.predict = original_predict

        if 'compressed_features' not in captured_data:
            raise RuntimeError("Failed to capture GMM inputs. Pipeline flow may have changed.")

        # Retrieve captured data
        X_compressed = captured_data['compressed_features']
        gmm_model = captured_data['gmm_instance']

        tprint_success(f"   ✅ Captured compressed features: {X_compressed.shape}")

        # Generate predictions using the captured model and features
        # This ensures we use the EXACT model state and input space
        probs, z_fam, ent = gmm_model.predict(X_compressed)

        # Cache the arrays
        # Pre-calculate velocity/diff arrays to speed up Numba loop
        prob_velocities = np.diff(probs, axis=0, prepend=probs[:1])
        prob_mag = np.linalg.norm(prob_velocities, axis=1) # (N,)

        z_fam_diff = np.abs(np.diff(z_fam, prepend=z_fam[:1]))
        entropy_diff = np.diff(ent, prepend=ent[:1])

        # Define Targets: Forward Volatility / Large Moves
        # Target = 1 if forward 12-bar return is in top 5% absolute magnitude
        fwd_returns = returns.shift(-1).rolling(12).sum().shift(-11).fillna(0) # Approx 3h return
        abs_fwd_returns = np.abs(fwd_returns)
        threshold = np.percentile(abs_fwd_returns, 95)
        targets = (abs_fwd_returns > threshold).astype(int).values

        self.cached_gmm_data = {
            "prob_mag": prob_mag,
            "z_fam_diff": z_fam_diff,
            "entropy_diff": entropy_diff,
            "targets": targets,
            "threshold_value": threshold
        }

        tprint_success(f"✅ GMM Data Cached. Target Threshold (95%): {threshold:.4f}")
        tprint_info(f"   Target Prevalence: {np.mean(targets):.2%}")


def objective(trial, optimizer: GMMShockOptimizer):
    """Optuna Objective Function"""

    # Suggest parameters
    p_jump = trial.suggest_float("probability_jump_threshold", 0.1, 0.6)
    z_jump = trial.suggest_float("z_familiarity_jump_threshold", 1.0, 5.0)
    e_drop = trial.suggest_float("entropy_drop_threshold", 0.05, 0.5)

    # Get cached data
    data = optimizer.cached_gmm_data

    # Calculate Metrics (Numba Optimized)
    precision, recall, f1 = calculate_shock_metrics_numba(
        data["prob_mag"],
        data["z_fam_diff"],
        data["entropy_diff"],
        data["targets"],
        p_jump,
        z_jump,
        e_drop
    )

    # Store auxiliary metrics
    trial.set_user_attr("precision", precision)
    trial.set_user_attr("recall", recall)

    # Objective: Maximize F1
    # Penalty for very low recall (avoid trivial high precision on 0 samples)
    if recall < 0.01:
        return 0.0

    return f1

def main():
    parser = argparse.ArgumentParser(description="Optimize GMM Shock Thresholds")
    parser.add_argument("--symbol", type=str, default="ETHUSDT")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--direction", type=str, default="long")
    parser.add_argument("--outcomes-dir", type=str, default="outcomes")
    args = parser.parse_args()

    config = {
        "symbol": args.symbol,
        "exchange": args.exchange,
        "timeframe": args.timeframe,
        "direction": args.direction
    }

    # Initialize Optimizer
    tprint_info(f"🚀 Starting Shock Threshold Optimization for {args.symbol}...")
    optimizer = GMMShockOptimizer()

    try:
        # Prepare Data
        optimizer.prepare_data(config)

        # Run Optuna
        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
        study.optimize(lambda t: objective(t, optimizer), n_trials=args.trials)

        # Results
        best_params = study.best_params
        best_value = study.best_value
        best_trial = study.best_trial

        tprint_success(f"🎉 Optimization Complete!")
        tprint_info(f"   Best F1 Score: {best_value:.4f}")
        tprint_info(f"   Precision: {best_trial.user_attrs['precision']:.4f}")
        tprint_info(f"   Recall: {best_trial.user_attrs['recall']:.4f}")
        tprint_info(f"   Best Parameters: {best_params}")

        # Save Results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(args.outcomes_dir)
        out_dir.mkdir(exist_ok=True, parents=True)

        result_file = out_dir / f"shock_thresholds_{args.symbol}_{timestamp}.json"

        results = {
            "best_params": best_params,
            "metrics": {
                "f1": best_value,
                "precision": best_trial.user_attrs['precision'],
                "recall": best_trial.user_attrs['recall']
            },
            "config": config,
            "trials": args.trials
        }

        with open(result_file, "w") as f:
            json.dump(results, f, indent=2)

        tprint_success(f"💾 Results saved to {result_file}")

    except Exception as e:
        tprint_error(f"❌ Optimization Failed: {e}")
        import traceback
        tprint_error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
