import re

with open("extreme_price_movements/simple_position_sizer.py", "r") as f:
    content = f.read()

# Replace run_simple_position_sizer_from_artifacts with the new strategy-centric version
pattern = re.compile(r"def run_simple_position_sizer_from_artifacts.*?return run_bucketed_simple_position_sizer.*?\)", re.DOTALL)

new_func = """def run_simple_position_sizer_from_artifacts(
    data_root: str,
    run_id: str,
    top_fracs: Tuple[float, ...] = (0.1, 0.2),
    use_ridge_head_sizer: bool = True,
    top_n_strategies: int = 3
) -> Dict[str, Dict[str, Any]]:
    \"\"\"
    Runs the simple position sizer directly on pipeline artifacts, executing the pipeline
    once per strategy loaded.

    Loads base model OOF predictions and filters strictly to the exact strategy mask
    (as optimized per-bucket) before running diagnostics independently for each strategy.

    Returns a dictionary mapping strategy_id to its respective position sizer results.
    \"\"\"
    from extreme_price_movements.run_ridge_sizer import load_trade_outcomes

    # Load dynamic strategies (which rules are active per bucket)
    strategies = load_inference_candidate_mask_params_per_bucket(top_n=top_n_strategies, ranking_metric="score_for_best_params")

    if not strategies:
        logger.warning("No strategies loaded from params_store.")
        return {}

    # Load base OOFs
    base_oofs = load_base_oof_predictions(data_root, run_id)
    if not base_oofs:
        logger.warning(f"No base OOFs found in {data_root}/artifacts/{run_id}/oof.")
        return {}

    strategy_results = {}

    for strategy in strategies:
        strategy_id = strategy.get("strategy_id", "")
        if not strategy_id:
            continue

        bucket = f"{strategy.get('trade_side', '')}_{strategy.get('base_event_trigger', '')}"

        if bucket not in base_oofs:
            # Try to find a bucket that starts with this strategy_id if exact match fails
            matching_buckets = [b for b in base_oofs.keys() if strategy_id.startswith(b) or b.startswith(strategy_id)]
            if not matching_buckets:
                logger.info(f"Skipping strategy {strategy_id}: no matching base OOF bucket found (tried {bucket}).")
                continue
            bucket = matching_buckets[0]

        oof_df = base_oofs[bucket]

        # OOF DFS usually contain a mask column named mask_{strategy_id} or just 'mask'
        mask_col = f"mask_{strategy_id}"

        if mask_col in oof_df.columns:
            active_df = oof_df[oof_df[mask_col] == 1].copy()
        elif "mask" in oof_df.columns:
            # Fallback
            active_df = oof_df[oof_df["mask"] == 1].copy()
        else:
            active_df = oof_df.copy()

        if active_df.empty:
            logger.info(f"Skipping strategy {strategy_id}: no active rows after mask filtering.")
            continue

        # Get target outcomes
        trade_outcomes = load_trade_outcomes(data_root, run_id, active_df)
        if trade_outcomes is None or "return" not in trade_outcomes.columns:
            logger.info(f"Skipping strategy {strategy_id}: could not load trade outcomes.")
            continue

        # Identify columns to use as heads, STRICTLY matching the strategy_id
        head_cols = []
        for c in active_df.columns:
            if c.startswith("base_") or "pred" in c.lower() or "score" in c.lower() or "mae" in c.lower() or "mfe" in c.lower():
                if strategy_id in c:
                    head_cols.append(c)
                elif not any(s.get("strategy_id", "") in c for s in strategies if s.get("strategy_id") and s.get("strategy_id") != strategy_id):
                    # It's a generic column not tied to ANY OTHER strategy
                    head_cols.append(c)

        if not head_cols:
            logger.info(f"Skipping strategy {strategy_id}: no matching feature columns found.")
            continue

        y_raw_net_return = trade_outcomes["return"].values

        if "downside" in trade_outcomes.columns:
            y_downside = trade_outcomes["downside"].values
        elif "mae" in active_df.columns:
             y_downside = active_df["mae"].values
        else:
            y_downside = np.zeros_like(y_raw_net_return)

        timestamps = active_df["timestamp"].values if "timestamp" in active_df.columns else np.zeros(len(y_raw_net_return))

        feature_dict = {col: active_df[col].values for col in head_cols}

        # Run the pipeline for this specific strategy
        res = run_simple_position_sizer(
            feature_dict=feature_dict,
            trade_outcomes=trade_outcomes,
            y_raw_net_return=y_raw_net_return,
            y_downside=y_downside,
            timestamps=timestamps,
            bucket_labels=None, # Running independently, no bucketing needed
            top_fracs=top_fracs,
            use_ridge_head_sizer=use_ridge_head_sizer
        )

        strategy_results[strategy_id] = res

    return strategy_results"""

content = pattern.sub(new_func, content)

with open("extreme_price_movements/simple_position_sizer.py", "w") as f:
    f.write(content)
