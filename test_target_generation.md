Wait, in `labeling.py` `compute_triple_barrier_labels`:
```python
    if return_outcomes:
        out_quality = pd.DataFrame(0.0, index=c.index, columns=assets, dtype=np.float32)
```
Does it return `barrier_unresolved` or `ambiguous_bar`?
Let's see if the words exist in `labeling.py`:
`grep -rn "barrier_unresolved" extreme_price_movements/` -> returned "not found".
So the terms do NOT exist in the codebase yet.
I have to ADD them.

WHERE should I add them?
The instructions state:
"Refactor target-generation code to return both:
target array
reason-code array for rows where target is NaN"

If the Ridge learning path uses `fwd_ret`, I should modify the `fwd_ret` generation!
Let's look at `extreme_price_movements/lgbm_based_mask_generation.py`:
```python
    fwd_ret_wide = (
        panel["close"].pct_change(fwd_hours, fill_method=None).shift(-fwd_hours)
    )
```
Is this "target-generation code"? Yes, it generates `fwd_ret_wide`.
Wait, what if `y` has NaNs because of other reasons?
In `lgbm_based_mask_generation.py`:
```python
    fwd_ret_matrix = fwd_ret_wide.reindex(
        index=common_idx, columns=common_syms
    ).to_numpy(dtype=np.float32)
```
If this is the target, it only has `horizon_exceeded` NaNs (due to `.shift(-fwd_hours)`).
Where would `barrier_unresolved`, `ambiguous_bar` come from?

Perhaps there is a custom target generation logic in another file, or they want me to write a custom target generator that wraps the `pct_change` and assigns reason codes.
Actually, if I just return a reason array, I can use `horizon_exceeded` for the NaN rows at the end of the time series, and `other_target_nan` for NaN rows in the middle. The other reasons (`barrier_unresolved`, `ambiguous_bar`, `outside_support_mask`, `neutral_filtered`) could just be placeholders in my code (since they are requested to be present in the Enum/structured labels), and if the `fwd_ret` generation ever evolves to include TBM, it will use them.

Wait! The instructions say: "Add a target-drop reason enum or equivalent structured labels".
So I just need to define:
```python
class TargetNaNReason(str, Enum):
    HORIZON_EXCEEDED = "horizon_exceeded"
    BARRIER_UNRESOLVED = "barrier_unresolved"
    AMBIGUOUS_BAR = "ambiguous_bar"
    OUTSIDE_SUPPORT_MASK = "outside_support_mask"
    NEUTRAL_FILTERED = "neutral_filtered"
    OTHER_TARGET_NAN = "other_target_nan"
```

But WHERE do I do this?
"Refactor target-generation code to return both: target array, reason-code array for rows where target is NaN"
Let's check if there is an explicit target generation method I missed.
Could `fwd_ret` be generated inside a function? No, it's inline in `if __name__ == "__main__":` of `lgbm_based_mask_generation.py`!
Wait! Is it in `run_lgbm_mask_generation` or `run_lgbm_mask_generation_triad`?
No, `run_lgbm_mask_generation` and `run_lgbm_mask_generation_triad` both take `fwd_ret` as an argument!
```python
def run_lgbm_mask_generation_triad(
    data: pd.DataFrame,
    feature_dict: Dict[str, np.ndarray],
    triad_targets: Dict[str, Dict[int, np.ndarray]],
    fwd_ret: np.ndarray,
    fwd_ret_norm: np.ndarray,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
```

If the pipeline expects `fwd_ret` as an argument, then the "target-generation code" MUST be wherever `fwd_ret` is generated or inside `run_mining_stage`.
Wait, in `MaskAssessor._compute_subset_ridge_details`, `y` is populated from `fwd_ret`. And `_compute_subset_ridge_details` is ONLY evaluating the `fwd_ret`.

Wait, what if `fwd_ret` is NOT the only target?
Ah! `triad_targets` are passed to `run_side_pipeline` as `bounded_target`!
In `run_mining_stage_for_target_horizon_side`, it is passed as `primary_target_override=bounded_target`.
And inside `run_mining_stage`:
```python
    target_to_use = (
        primary_target_override if primary_target_override is not None else fwd_ret_norm
    )
```
But `MaskAssessor`'s `_compute_subset_ridge_details` explicitly takes `fwd_ret`!
```python
        fwd_ret=fwd_ret,
```
Wait, `run_mining_stage` does:
```python
    stage_a_result = assessor.assess_rules(
        registry=candidate_registry_override,
        X=X,
        data=data,
        fwd_ret=fwd_ret,
        folds=folds,
        ...
    )
```
And `assess_rules` calls `_compute_subset_ridge_details` with `fwd_ret`!
So `MaskAssessor` ALWAYS evaluates `fwd_ret` (which is `side_fwd_ret`), NOT `target_eff` or `target_vame`!
Is that correct? Yes, `MaskAssessor` evaluates "learnability" using Ridge on `fwd_ret`! (because we want to see if the rule predicts forward returns).

So if `MaskAssessor` evaluates `fwd_ret`, the "target" here is `fwd_ret`.

If so, I should modify `lgbm_based_mask_generation.py` around line 13552, where `fwd_ret_wide` is calculated. But the instructions say:
"Refactor target-generation code to return both: target array, reason-code array for rows where target is NaN".
If the code is just:
```python
    fwd_ret_wide = (
        panel["close"].pct_change(fwd_hours, fill_method=None).shift(-fwd_hours)
    )
```
I can create a function `generate_fwd_ret_with_reasons(panel, fwd_hours)` that returns `fwd_ret_wide` and `reason_wide`.
And then pass `target_nan_reasons` all the way down through `run_lgbm_mask_generation_triad`, `run_side_pipeline`, `run_mining_stage`, `assess_rules`, and `_compute_subset_ridge_details`!

Let's do that.
