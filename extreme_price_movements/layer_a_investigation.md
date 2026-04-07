# Layer A Investigation Report: Target Race and Model 3 Value

## Target Race Winner Stability
**Objective:** Determine if the Model 1 (Edge) target race provides robust value or just introduces semantic drift between buckets and folds.

**Findings & Recommendations:**
- The target race currently evaluates raw PnL (`mean_net - 0.5 * std_net`) of the top decile. This metric is highly sensitive to outlier trades and tends to favor volatile, unconstrained targets (`log_clipped_winsorized_net`) over stable rank-based targets, even when the rank-based targets might produce a more capital-efficient downstream policy.
- Without a scale-standardized blend (which we have now introduced via `score_blend_mode="train_scaled_components"`), changing the target family drastically changes the `Edge` raw prediction scale, breaking the fixed `λ=0.5` downside penalty.
- **Recommendation:** Keep the race *constrained* for now. With the newly added scaling step (`train_scaled_components`), the downside of scale drift is neutralized. However, the race's objective function should be updated in a future iteration to penalize poor capital efficiency (e.g., using Top-Decile Sortino) to ensure the winner is truly economically superior, not just structurally lucky. The new ablation flag `model1_target_mode="fixed"` provides a safe baseline to run immediate A/B comparisons in production.

## Model 3 (Uncertainty) Value Investigation
**Objective:** Determine if Model 3 improves OOF score quality or downstream utility, or if it mostly tracks obvious proxies already in Edge/Downside.

**Findings & Recommendations:**
- Model 3 historically trained on an information-leaked target (residuals built against the *global* re-ranked target rather than the *fold-local* target). This leakage meant Model 3 was often predicting its own lookahead bias rather than true OOS uncertainty.
- In many buckets, Uncertainty is highly correlated with the absolute magnitude of Edge (high expected return = high expected residual), meaning it can inadvertently act as a momentum penalty if `η` is too high.
- **Recommendation:** With the leakage now fixed (Model 3 now strictly trains on the `oof_targets` paired to the `oof_preds`), Model 3's true value can be evaluated cleanly. Run `use_model3_uncertainty=False` against `True` in the offline optimiser. If the downstream `LayerBPolicyOptimizer` PnL/Sortino does not improve with Model 3 enabled, Model 3 should be deprioritized. It is likely that a simpler volatility penalty (already in Downside) captures 90% of the value.