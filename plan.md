1. **Understand the goal**: The user wants to fix the meta regression model target and sample weighting in `extreme_price_movements/training.py` inside `_build_meta_regression_target_bundle`. Specifically:
   - Calculate `raw_correction = y_ret_arr - base_reg_pred`
   - Set `meta_reg_target = raw_correction / vol_scale`
   - The sample weights should be symmetric, bypassing `_build_base_regression_sample_weight`, and instead rely purely on the economic source (`__u_policy_net__` or magnitude of `raw_correction`).

2. **Actions taken**: I modified `extreme_price_movements/training.py` using `sed`/`python` script to replace the implementation of `_build_meta_regression_target_bundle` to reflect the instructions.

3. **Verify actions**:
   - `git diff --staged extreme_price_movements/training.py` confirms the correct logic has been applied to `_build_meta_regression_target_bundle`. The previous implementation was successfully swapped out for the new target and symmetric weighting calculation without asymmetric positive/negative weighting.

4. **Pre commit instructions**:
   - Get instructions using `pre_commit_instructions` and follow them to make sure all standards are met before submit.

5. **Submit**:
   - Commit using git with descriptive message.
   - Use `submit` to push changes to branch.
