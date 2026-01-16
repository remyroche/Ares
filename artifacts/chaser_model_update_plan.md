# Chaser Model Enhancement Plan

## Summary
Update the existing Layer 2.5 chaser implementation to use weak Huber constraints, strong regularization, and implement the classifier chaser workflow with BayesianRidge teacher → student architecture.

## Implementation Ready

Based on analysis of the existing code and Huber utilities, the implementation is ready to proceed with these specific changes:

### 1. Import Updates (lines 27-46)
- Add BayesianRidge to imports
- Import Huber constraint utilities with fallback
- Add uncertainty sanity check function

### 2. Strong Regularization Parameters (lines 248-270, 281-290, 314-325)
Update default parameters to user specifications:
- XGBoost: reg_lambda=50, min_child_weight=10, gamma=1.1, learning_rate=0.03
- LightGBM: reg_lambda=10, path_smooth=20, extra_trees=True  
- CatBoost: l2_leaf_reg=20, subsample=0.6, random_strength=5

### 3. Weak Constraints Integration (train_chaser_student function)
- Add monotone_constraints and interaction_constraints parameters
- Apply constraints from Huber analysis when available
- Use "weak" tier constraints from Huber

### 4. Enhanced Classifier Workflow
- Ensure proper base_margin usage for XGBoost
- Ensure proper baseline usage for CatBoost  
- Add teacher disagreement features |mu_B - mu_A|
- Improve uncertainty weighting bounds

### 5. Meta-Learner Features
- Add teacher baseline outputs (mu_B, std_B, margin_B)
- Add chaser correction signals (delta_margin_chaser, p_chaser_final)
- Prepare outputs for stacking without double counting

### 6. OOF Discipline & Validation
- Add sanity check for uncertainty signal
- Ensure proper OOF predictions
- Prevent leakage in residual calculations

Ready to implement these changes to layer2_5_chaser.py.
