import re

with open('extreme_price_movements/training.py', 'r') as f:
    code = f.read()

# We need to make generate_label_datasets support multiple cores
old_gen = """def generate_label_datasets(
    panel, feats, mkt_gates, cfg, train_syms, ts_sig, p_exh_hist, horizons=None
):"""

new_gen = """def generate_label_datasets(
    panel, feats, mkt_gates, cfg, train_syms, ts_sig, p_exh_hist, horizons=None
):
    import warnings
    import gc
    from joblib import Parallel, delayed
    """

code = code.replace(old_gen, new_gen)

old_loop = """    for s_id, side, feature_keys, base_event_trigger in dynamic_rules:
        # Determine whether to use mr or tf target semantics based on strategy_id naming
        is_mr = "mr" in s_id.lower()

        # Build one row mask per strategy
        rule_mask = None
        if base_event_trigger:
            from extreme_price_movements.lgbm_based_mask_generation import (
                evaluate_rule_vectorized,
            )

            try:
                # Fast evaluation path for threshold rules
                if not base_event_trigger.startswith(
                    "price_up"
                ) and not base_event_trigger.startswith("price_down"):
                    rule_mask = evaluate_rule_vectorized(
                        base_event_trigger, panel, feats
                    )
            except Exception as e:
                tprint(
                    f"WARNING: failed to parse/evaluate trigger '{base_event_trigger}' for {s_id}: {e}"
                )"""

new_loop = """    def process_strategy(s_id, side, feature_keys, base_event_trigger, horizons, is_mr):
        import gc
        local_datasets = {}
        # Build one row mask per strategy
        rule_mask = None
        if base_event_trigger:
            from extreme_price_movements.lgbm_based_mask_generation import (
                evaluate_rule_vectorized,
            )

            try:
                # Fast evaluation path for threshold rules
                if not base_event_trigger.startswith(
                    "price_up"
                ) and not base_event_trigger.startswith("price_down"):
                    rule_mask = evaluate_rule_vectorized(
                        base_event_trigger, panel, feats
                    )
            except Exception as e:
                tprint(
                    f"WARNING: failed to parse/evaluate trigger '{base_event_trigger}' for {s_id}: {e}"
                )

        horizons_for_strat = run_horizons.get(s_id, horizons)
        if not horizons_for_strat:
            return local_datasets

        for H in horizons_for_strat:
            name = f"train_{s_id}_{H}"
            base_df = _build_dynamic_strategy_dataset(
                s_id,
                side,
                H,
                feature_keys,
                rule_mask,
                is_mr,
                balanced=False,  # Single unified dataset
                tight_sl=False,
                wide_sl=False,
            )
            if base_df is not None:
                local_datasets[name] = base_df

            for variant in base_geometry_archetypes:
                if variant == "balanced":
                    continue
                vname = f"train_{s_id}_{H}_{variant}"
                is_tight = variant == "tight"
                is_wide = variant == "wide"
                var_df = _build_dynamic_strategy_dataset(
                    s_id,
                    side,
                    H,
                    feature_keys,
                    rule_mask,
                    is_mr,
                    balanced=False,
                    tight_sl=is_tight,
                    wide_sl=is_wide,
                )
                if var_df is not None:
                    local_datasets[vname] = var_df

        return local_datasets

    # Run strategies in parallel to speed up generation
    max_workers = 2 # limited to 2 cores
    tprint(f"Generating label datasets with {max_workers} joblib workers")

    results = Parallel(n_jobs=max_workers)(
        delayed(process_strategy)(s_id, side, feature_keys, base_event_trigger, horizons, "mr" in s_id.lower())
        for s_id, side, feature_keys, base_event_trigger in dynamic_rules
    )

    for res in results:
        datasets.update(res)

    return datasets
"""

# Let's do a more targeted replace of the loop since it might be tricky
with open('extreme_price_movements/training.py', 'w') as f:
    f.write(code)
