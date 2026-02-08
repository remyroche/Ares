
import numpy as np
import pandas as pd
from extreme_price_movements.model_race import ModelRace
from xgboost import XGBClassifier

def debug_pos_weight():
    print("\n--- Debugging Pos Weight Calculation ---")
    np.random.seed(42)
    n_samples = 2000
    y = (np.random.rand(n_samples) < 0.135).astype(int)
    
    # Random sample weights
    weights = np.random.uniform(0.5, 1.5, size=n_samples)
    
    # Manually compute pos_weight using ModelRace logic
    race = ModelRace()
    pos_weight = race._compute_pos_weight(y)
    print(f"Computed pos_weight (unweighted formula): {pos_weight:.4f}")
    
    # Check effective pos_weight with sample weights?
    # XGBoost uses sum(w_neg) / sum(w_pos) if we passed weights? No, scale_pos_weight is a multiplier ON TOP of sample weights.
    # So if we provide sample weights, scale_pos_weight should still ideally be balanced based on counts or weighted counts?
    # Actually, standard formula (neg/pos) balances the total weight of positive vs negative class *assuming equal feature weight*.
    
    # If we use weighted training, maybe pos_weight needs to be calculated from weighted sums?
    neg_w = np.sum(weights[y==0])
    pos_w = np.sum(weights[y==1])
    weighted_pos_weight = neg_w / pos_w
    print(f"Weighted pos_weight (sum_neg / sum_pos): {weighted_pos_weight:.4f}")
    
    # Let's check what recalibration does with these values at 0.5 prob
    p = 0.5
    recal_unweighted = p / (p + (1-p)*pos_weight)
    recal_weighted = p / (p + (1-p)*weighted_pos_weight)
    print(f"Recalibrated 0.5 (unweighted w): {recal_unweighted:.4f}")
    print(f"Recalibrated 0.5 (weighted w): {recal_weighted:.4f}")
    
    # True prevalence
    print(f"True unweighted prev: {np.mean(y):.4f}")
    print(f"True weighted prev: {np.average(y, weights=weights):.4f}")

if __name__ == "__main__":
    debug_pos_weight()
