import numpy as np
import pandas as pd
from extreme_price_movements.model_race import ModelRace
from sklearn.datasets import make_classification
from extreme_price_movements.utils import tprint

def verify():
    # 1. Create imbalanced dataset
    # prevalence 10%
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, 
                               weights=[0.9, 0.1], flip_y=0.05, random_state=42)
    
    tprint(f"Dataset: n={len(y)}, prevalence={y.mean():.4f}")
    
    # 2. Run ModelRace
    # Use few splits to be fast
    race = ModelRace(n_splits=3, race_sample_frac=1.0)
    
    race.fit(X, y)
    
    # 3. Check metrics
    winner = race.best_model_name
    metrics = race.detailed_metrics[winner]
    
    tprint(f"\nWinner: {winner}")
    tprint(f"BSS: {metrics['BSS']:.4f}")
    tprint(f"BS: {metrics['BS']:.4f}")
    tprint(f"Ref: {metrics['BS_Ref']:.4f}")
    tprint(f"P10: {metrics['Prec10']:.4f}")
    tprint(f"P25: {metrics['Prec25']:.4f}")
    tprint(f"P40: {metrics['Prec40']:.4f}")
    
    if metrics['BSS'] < -0.05: # Allow slight negative noise
        tprint("FAILURE: BSS is significantly negative!")
    elif metrics['BSS'] > 0:
        tprint("SUCCESS: BSS is positive.")
    else:
        tprint("NEUTRAL: BSS near zero.")
        
    # Check if prediction mean matches prevalence (Validation of Mean Matching)
    oof_mean = np.mean(race.oof_probs)
    tprint(f"OOF Mean: {oof_mean:.4f} (Target: {y.mean():.4f})")
    
    if abs(oof_mean - y.mean()) > 0.05:
         tprint("WARNING: OOF mean drift > 5%")
    else:
         tprint("SUCCESS: OOF mean matches target.")

if __name__ == "__main__":
    verify()
