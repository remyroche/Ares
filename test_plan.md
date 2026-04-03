So the target logic is inside `extreme_price_movements/lgbm_based_mask_generation.py`.
Wait, the Ridge target used in `_compute_subset_ridge_details` is `fwd_ret`. But `fwd_ret` is what's passed in.
Wait! "Replace coarse Target NaN: N logging with reason-coded target-drop accounting. Target NaN: X ... horizon_exceeded, barrier_unresolved, etc."

Where in `extreme_price_movements/triad_targets.py` are the NaNs generated?
Ah, if I look at `get_bounded_triad` in `triad_targets.py`...
If the objective says "Refactor target-generation code to return both: target array, reason-code array for rows where target is NaN", where should I modify?
Is there a target-generation function that creates `target_eff`, `target_vame`?
Yes, `compute_triad_targets_for_horizons` in `triad_targets.py`.
Does it compute `target_eff`? Yes!

Wait, `fwd_ret` is passed into `run_mining_stage`. Let's look at `apply_robust_data_filtering`.
No, the Ridge model fits on `fwd_ret`!
Let's look at `_compute_subset_ridge_details` around line 11180:
```python
        y = np.asarray(fwd_ret, dtype=np.float32)
        y[~np.isfinite(fwd_ret)] = np.nan
```
If `y` is just `fwd_ret`, it's calculated from:
```python
    fwd_ret_wide = (
        panel["close"].pct_change(fwd_hours, fill_method=None).shift(-fwd_hours)
    )
```
Wait! "barrier_unresolved", "ambiguous_bar" - these are TBM specific drop reasons.
But TBM is NOT used here! In `extreme_price_movements/triad_targets.py`, targets are just continuous values.
Wait! Maybe `triad_targets.py` DOES use `_numba_triple_barrier`?
No, we grepped and found it in `labeling.py`.
Does `lgbm_based_mask_generation.py` import `compute_triple_barrier_labels`? Let's check!
