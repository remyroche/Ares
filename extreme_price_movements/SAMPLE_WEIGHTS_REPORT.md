# Sample Weights in Meta Models & Alpha Models

This report details the exact formulation of sample weights used across the different model heads (TBM/Alpha, MAE/MFE/Utility, Meta Regressor, Meta Classifier, and Early Invalidation) during training.

## 1. Alpha Models / TBM (Triple Barrier Method) Base Models
The sample weights (`__w__`) for the base Alpha models are computed per-event during dataset generation in `build_hourly_training_set_and_weights()` via a multiplicative combination of several factors:

* **Uniqueness (`w_uniqueness`):** The square root of the number of concurrent active events, normalized to mean=1.
* **Realized Magnitude (`w_magnitude`):** A proxy for signal value, clipped at the 95th percentile, scaled to a baseline of `0.5 + abs_ret / p95`.
* **Excursion Quality (`w_mfe_mae`):** Calculated by `compute_mfe_mae_weights()`. Uses Max Favorable Excursion (MFE) and Max Adverse Excursion (MAE) relative to the TP and SL barriers respectively.
    * `r_mfe = MFE/TP`
    * `r_mae = MAE/SL`
    * A smooth intensity curve and an excursion dominance metric determine a decisiveness `quality` score.
    * For Mean Reversion (MR) strategies, an extra `_mr_path_w` multiplier rewards high MFE and low MAE.
* **Outcome Weighting (`w_outcome`):**
    * TP hits: Weighted by the event quality `qual_vals` (0.5 to 1.0).
    * SL hits: Weighted by `1.0 - qual_vals`.
    * Timeouts (TO): Dynamically weighted based on how close the timeout price was to SL vs TP, then scaled by `timeout_weight` (default 0.4).
* **Class Balance (`w_class`):** A mild inverse-frequency multiplier `(0.5 / p)^0.5` is applied to minority classes and hard-clipped to `[0.85, 1.25]`.
* **Geometry Consensus (`w_consensus`):** If multiple TP/SL grids vote similarly, weight is increased.

All components are multiplied, winsorized, and then cross-sectionally normalized so that the total weight mass across simultaneous events remains bounded. Finally, the square root of the aggregate weight (`np.sqrt(__w__)`) is passed to the models to prevent the `n_eff` collapsing due to extreme skewness.

## 2. Auxiliary Heads: MAE, MFE, Utility
The auxiliary regressors are trained using subset masks of valid, positive trades and apply specific tail-amplification multipliers.

* **Base Weights:** Valid rows from the trades mask are normalized and clipped to `[0.75, 1.25]`.
* **Tail Amplification (`_head_weight_vector`):**
    * **`symmetric_tail`:** If configured, amplifies weights linearly between the 50th and 95th percentile of targets (`_tail_multiplier`).
    * **`asymmetric_tail`:** Amplifies high values (above 70th percentile) and slightly penalizes low values (below 30th percentile) via `_tail_multiplier_asymmetric`.
    * **`top30_tail`:** Gives a flat 1.25x boost to values exceeding the 70th percentile.
    * A `weight_lambda` parameter interpolates between the base uniform weights and the fully tail-amplified weights.

## 3. Meta Regressor Head
The Meta Model regressor uses a combination of magnitude and excursion quality, ensuring the model focuses heavily on predicting "meaningful" trade setups.

* **Magnitude Weight (`w_mag`):** A sigmoid curve centered at the 60th percentile of absolute returns `_sigmoid((|y| - p60) / std)`. The top 40% receive ~1.1-1.2x weighting, and the bottom 60% receive ~1.0x, normalized to a mean of 1.
* **Excursion Weight (`w_exc`):** Derived from `max(MFE/TP, MAE/SL)` normalized to `[0.5, 1.0]` depending on decisiveness.
* **Optimization (`_optimize_training_sample_weights`):** If enabled, it runs an internal optimization routine combining base weights with features like time-recency (exponential decay) and ridge out-of-fold quality predictions, searching for the blend of component alphas that minimizes cross-validated Huber loss while maintaining a healthy `n_eff` (min 30% of N).

## 4. Meta Classifier Head
The meta classifier is trained precisely on the same weighting regime as the regression head (`w_meta_clf`), except its target is a dynamic utility outcome probability.

* Uses the exact same `w_mag_clf` (magnitude sigmoid) and `w_exc_clf` (MFE/MAE quality), calculated over an average of all available horizons instead of relying on a single horizon.
* Also utilizes `_optimize_training_sample_weights` via a hyper-parameter grid search across time-splits to fine-tune the component blends (magnitude, excursion, time-recency).

## 5. Other Heads
* **Early Invalidation Head:** Trained via a standard Logistic Regression model. It does **not** use custom sample weights. Instead, it relies strictly on `class_weight="balanced"` within `sklearn` to handle the target imbalance.
* **Position Sizers (EV Decomposition):** The Position Sizer models (win rate, loss size, and profit size quantiles) process a pre-aggregated dataset and do **not** use explicit sample weights (`sample_weight=None`).
