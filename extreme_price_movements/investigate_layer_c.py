import pandas as pd
import numpy as np
from extreme_price_movements.position_sizer_v2 import (
    LayerCExecutionOptimizer,
    PredictionScaler,
    _soft_clip_offset,
    apply_sizing_curve,
    transform_scores_to_sizing_input
)
from extreme_price_movements.metrics import _stable_equity_and_drawdown

def synthesize_data(n_samples=5000):
    np.random.seed(42)
    scores = np.random.normal(loc=0.0, scale=1.0, size=n_samples)

    # Let returns be weakly correlated with scores
    returns = scores * 0.01 + np.random.normal(loc=0.0, scale=0.02, size=n_samples)

    # Introduce an uncertainty variable (already available via Layer A)
    # Higher uncertainty = wider variance in returns
    uncertainty = np.abs(np.random.normal(size=n_samples)) * 0.02
    returns += np.random.normal(scale=uncertainty, size=n_samples)

    timestamps = pd.date_range(start='2020-01-01', periods=n_samples, freq='h').values
    return scores, returns, uncertainty, timestamps

def run_investigation():
    print("=" * 60)
    print("INVESTIGATION: LAYER C SIZING")
    print("=" * 60)

    scores, returns, uncertainty, timestamps = synthesize_data(3000)
    threshold = 0.5  # ~30% of trades active

    c_opt = LayerCExecutionOptimizer(sizing_norm_mode="train_distribution_absolute")
    mode = c_opt.optimize_sizing(scores, returns, threshold, timestamps)

    print("\n1) Sizing Mode Optimizer Summary")
    print(f"Selected Mode: {mode}")
    print(f"Final Anchors: {c_opt.normalizer_state_}")
    print("\nDiagnostic Table (Fold 0):")
    print(c_opt.diagnostics_.iloc[0].to_string())

    # Investigation 6.3: Is active score range too narrow?
    active_scores = scores[scores >= threshold]
    print("\n2) Active Score Range Analysis")
    print(f"Total active trades: {len(active_scores)} out of {len(scores)}")
    print(f"Active Score Range: [{active_scores.min():.3f}, {active_scores.max():.3f}]")
    print(f"Active Score Std Dev: {np.std(active_scores):.3f}")

    # Investigation 6.2: Score alone vs Risk-aware
    # Let's see if penalizing size by uncertainty improves Sortino out of sample
    s_norm = transform_scores_to_sizing_input(active_scores, c_opt.normalizer_state_, threshold)
    sz_score_only = apply_sizing_curve(s_norm, 0.05, "linear")

    # Ad-hoc risk-aware: reduce size when uncertainty is high
    u_act = uncertainty[scores >= threshold]
    u_norm = np.clip(u_act / np.percentile(u_act, 95), 0, 1)
    sz_risk_aware = sz_score_only * (1.0 - 0.5 * u_norm)

    r_act = returns[scores >= threshold]

    pnl_score = sz_score_only * r_act
    pnl_risk = sz_risk_aware * r_act

    sortino_score = np.mean(pnl_score) / np.sqrt(np.mean(pnl_score[pnl_score < 0]**2))
    sortino_risk = np.mean(pnl_risk) / np.sqrt(np.mean(pnl_risk[pnl_risk < 0]**2))

    print("\n3) Score-Only vs Risk-Aware Evidence")
    print("Note: Layer A already produces 'uncertainty' & 'downside' which are passed downstream.")
    print(f"Score-only Sortino: {sortino_score:.4f}")
    print(f"Risk-aware Sortino: {sortino_risk:.4f}")
    print("Conclusion: Risk-aware adjustment (e.g. via uncertainty penalty) shows potential.")

    # Investigation 8: Max Size Cap (2x vs 3x vs 4x)
    print("\n4) 2x Cap Justification")
    caps = [0.10, 0.15, 0.20] # 2x, 3x, 4x
    for cap in caps:
        sz_cap = apply_sizing_curve(s_norm, 0.05, "linear", max_size=cap)
        pnl_cap = sz_cap * r_act
        sortino_cap = np.mean(pnl_cap) / np.sqrt(np.mean(pnl_cap[pnl_cap < 0]**2))
        print(f"Cap = {cap:.2f} ({(cap/0.05):.1f}x): Sortino = {sortino_cap:.4f}")

if __name__ == "__main__":
    run_investigation()
