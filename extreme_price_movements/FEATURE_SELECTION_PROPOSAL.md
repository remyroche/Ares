# Feature Selection Review and Proposal

## 1. Review of Current Process

Currently, the `extreme_price_movements` pipeline uses an **implicit feature selection** strategy embedded within the `ModelRace` and `training.py` modules.

*   **Mechanism:** Large set of base features + interaction terms (e.g., `feature * G_VOL`) are fed into regularization-heavy models (ElasticNet) and tree ensembles (XGBoost, CatBoost).
*   **Collinearity Handling:** Relies on model internals (L1 shrinkage, random split selection).
*   **Drawbacks:** Instability in feature importance, dimensionality bloat, and inefficiency due to generating interactions for redundant features.

## 2. Alternative Proposal: Hierarchical Correlation Clustering (HCC)

To address **simplicity** and **collinearity**, we propose a pre-filtering step.

### The Algorithm

1.  **Correlation Matrix:** Compute **Spearman Rank Correlation** for all features.
2.  **Clustering:** Build a **Dendrogram** using Hierarchical Clustering (Ward's method).
3.  **Cut:** Define a similarity threshold (e.g., correlation > 0.85) and cut the tree to form clusters.
4.  **Select:** Pick **one** representative per cluster based on the highest Information Coefficient (IC) with the target.

### Benefits

*   **Simple:** Uses standard statistical libraries (`scipy.cluster`).
*   **Explicit Control:** Guarantees no two features have correlation > Threshold.
*   **Efficient:** Reduces dataset width before expensive training steps.
