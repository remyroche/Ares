with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

old_logic = """def build_hourly_training_set_and_weights(
    panel,
    feats,
    mkt_gates,
    cfg,
    syms,
    ts_end,
    p_exh_hist,
    H,
    model_kind,
    trend_filter=None,"""

new_logic = """def build_hourly_training_set_and_weights(
    panel,
    feats,
    mkt_gates,
    cfg,
    syms,
    ts_end,
    p_exh_hist,
    H,
    model_kind,
    trend_filter=None,
    strategy=None,"""

content = content.replace(old_logic, new_logic)


# Call site in build_hourly_training_set_and_weights:
old_call = """        weights = _optimize_training_sample_weights(
            df=df,
            X_frame=df[feature_cols_for_opt].fillna(0.0),
            y_ret=returns,
            label_times=label_times,
            base_weights=weights,
            cfg=cfg,
            stage="base",
            extra_components=None,  # Removed distance component per user request
        )"""

new_call = """        weights = _optimize_training_sample_weights(
            df=df,
            X_frame=df[feature_cols_for_opt].fillna(0.0),
            y_ret=returns,
            label_times=label_times,
            base_weights=weights,
            cfg=cfg,
            stage="base",
            extra_components=None,  # Removed distance component per user request
            strategy=strategy,
        )"""
content = content.replace(old_call, new_call)

with open('extreme_price_movements/training.py', 'w') as f:
    f.write(content)

print("Patched build_hourly_training_set_and_weights")
