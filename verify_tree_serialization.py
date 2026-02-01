
import pandas as pd
import numpy as np
from src.training.steps.labeling.tree_based_causal_gates import StabilityRegimeTree

def test_serialization():
    print("Testing StabilityRegimeTree serialization...")
    
    # 1. Create dummy data
    rng = np.random.default_rng(42)
    n = 1000
    Z = pd.DataFrame({
        'feat1': rng.normal(size=n),
        'feat2': rng.normal(size=n)
    })
    preds = {
        'expert1': rng.normal(size=n),
        'expert2': rng.normal(size=n)
    }
    y = rng.integers(0, 2, size=n).astype(float)
    
    # 2. Fit tree
    folds = [(np.arange(n), np.arange(n))]  # Dummy fold
    tree = StabilityRegimeTree(max_depth=2, min_leaf_samples=0.1)
    tree.fit(Z, preds, y, folds)
    
    print(f"Original tree leaves: {len(tree.leaves_)}")
    
    # 3. Serialize
    tree_dict = tree.to_dict()
    print("Serialization successful.")
    
    # 4. Deserialize
    tree_restored = StabilityRegimeTree.from_dict(tree_dict)
    print("Deserialization successful.")
    
    # 5. Compare
    print(f"Restored tree leaves: {len(tree_restored.leaves_)}")
    if len(tree.leaves_) == len(tree_restored.leaves_):
        print("✅ Leaf count matches.")
    else:
        print("❌ Leaf count mismatch.")
        
    # Test prediction
    leaf_ids_orig = tree.predict_leaf_ids(Z)
    leaf_ids_restored = tree_restored.predict_leaf_ids(Z)
    
    if np.array_equal(leaf_ids_orig, leaf_ids_restored):
        print("✅ Predictions match.")
    else:
        print("❌ Predictions mismatch.")

if __name__ == "__main__":
    test_serialization()
