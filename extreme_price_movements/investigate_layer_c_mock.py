import pandas as pd
import numpy as np
from extreme_price_movements.position_sizer_v2 import (
    LayerCExecutionOptimizer,
    apply_sizing_curve,
    transform_scores_to_sizing_input
)

def synthesize_data(n_samples=5000):
    np.random.seed(42)
    scores = np.random.normal(loc=0.0, scale=1.0, size=n_samples)
    returns = scores * 0.01 + np.random.normal(loc=0.0, scale=0.02, size=n_samples)
    uncertainty = np.abs(np.random.normal(size=n_samples)) * 0.02
    returns += np.random.normal(scale=uncertainty, size=n_samples)
    return scores, returns, uncertainty

def run_investigation():
    print("=" * 60)
    print("INVESTIGATION: LAYER C SIZING")
    print("=" * 60)

    scores, returns, uncertainty = synthesize_data(3000)
    threshold = 0.5  # ~30% of trades active

    # Mock normalizer
    active = scores[scores >= threshold]
    normalizer_state = {"sizing_norm_mode": "train_distribution_absolute", "lower_anchor": threshold, "upper_anchor": np.percentile(active, 95)}

    print("\n2) Active Score Range Analysis")
    print(f"Total active trades: {len(active)} out of {len(scores)}")
    print(f"Active Score Range: [{active.min():.3f}, {active.max():.3f}]")
    print(f"Active Score Std Dev: {np.std(active):.3f}")

    s_norm = transform_scores_to_sizing_input(active, normalizer_state, threshold)
    sz_score_only = apply_sizing_curve(s_norm, 0.05, "linear", 0.10) # Using default 2x cap explicitly

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

    print("\n4) 2x Cap Justification")
    caps = [0.10, 0.15, 0.20] # 2x, 3x, 4x
    for cap in caps:
        sz_cap = apply_sizing_curve(s_norm, 0.05, "linear", max_size=cap)
        pnl_cap = sz_cap * r_act
        sortino_cap = np.mean(pnl_cap) / np.sqrt(np.mean(pnl_cap[pnl_cap < 0]**2))
        print(f"Cap = {cap:.2f} ({(cap/0.05):.1f}x): Sortino = {sortino_cap:.4f}")

if __name__ == "__main__":
    run_investigation()
