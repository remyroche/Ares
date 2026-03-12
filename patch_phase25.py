import re

with open("extreme_price_movements/mask_optimiser.py", "r") as f:
    code = f.read()

target_str = "    feature_gain_vals: List[np.float32] = []\n    feature_pos_vals: List[np.float32] = []"

phase_2_5 = """
    # -------------------------------------------------------------------------
    # Phase 2.5: Ridge regime attribution
    # -------------------------------------------------------------------------
    tprint(f"Phase 2.5 ({mode}): Ridge regime attribution for top {len(df2)} candidates...")

    full_df_dict = {
        "timestamp": shared["timestamps"],
        "high": shared["high"],
        "low": shared["low"],
        "close": shared["close"],
    }
    if "open" in shared:
        full_df_dict["open"] = shared["open"]
    if "volume" in shared:
        full_df_dict["volume"] = shared["volume"]

    full_df = pd.DataFrame(full_df_dict)
    regime_features_df = build_regime_features(full_df)

    # Identify which features are binary vs continuous
    feature_types = {}
    for c in RIDGE_FEATURE_COLS:
        if c in regime_features_df.columns:
            u_vals = regime_features_df[c].dropna().unique()
            if len(u_vals) <= 2 and set(u_vals).issubset({0.0, 1.0, 0, 1}):
                feature_types[c] = "binary"
            else:
                feature_types[c] = "continuous"

    dynamic_conditioners: Dict[str, List[Dict[str, Any]]] = {}

    for _, row in df2.iterrows():
        base_name = str(row["name"])
        reg = candidate_registry[base_name]
        z = int(int(reg["z_hours"]) * bph)
        duration_bars = int(int(reg["duration_hours"]) * bph)
        if z not in global_z_cache:
            global_z_cache[z] = _compute_z_cache(
                high=shared["high"],
                low=shared["low"],
                close=shared["close"],
                ret_1=shared["ret_1"],
                vol_g=shared["vol_g"],
                asset_groups=shared["asset_groups"],
                z=z,
                bph=bph,
            )
        zc = global_z_cache[z]
        m_high, m_low = _generate_event_masks_fast(
            family=reg["family"],
            param_val=reg["param"],
            up_move=zc["up"],
            dn_move=zc["dn"],
            rolling_std_up=zc["std_up"],
            rolling_std_dn=zc["std_dn"],
            asset_groups=shared["asset_groups"],
            duration_bars=duration_bars,
        )
        side_mask = _get_side_mask(mode, m_high, m_low)

        fwd_2h_bars = int(2 * bph)
        if "close" in full_df:
            fwd_ret = full_df["close"].shift(-fwd_2h_bars) / full_df["close"] - 1.0
        else:
            fwd_ret = pd.Series(np.zeros(len(full_df)))

        ridge_df = regime_features_df.copy()
        ridge_df["event_mask"] = side_mask.astype(int)
        ridge_df["target_fwd_return"] = fwd_ret

        valid_feature_cols = [c for c in RIDGE_FEATURE_COLS if c in ridge_df.columns]

        res = fit_ridge_regime_scan(
            ridge_df,
            valid_feature_cols,
            "event_mask",
            "target_fwd_return",
            n_splits=max(2, len(folds))
        )

        cond_features = []
        if res is not None:
            ranked_features = res["ranked_features"]
            # Keep top max single features, e.g. 4
            max_vars = int(cfg.get("phase3_max_single_features", 4))
            top_vars = ranked_features.head(max_vars)
            for _, v_row in top_vars.iterrows():
                f_name = v_row["feature"]
                cond_features.append({
                    "feature": f_name,
                    "coef": v_row["coef"],
                    "abs_signed_importance": v_row["abs_signed_importance"],
                    "type": feature_types.get(f_name, "continuous")
                })

        dynamic_conditioners[base_name] = cond_features

"""

if target_str in code:
    code = code.replace(target_str, phase_2_5 + target_str)
    with open("extreme_price_movements/mask_optimiser.py", "w") as f:
        f.write(code)
    print("Patched successfully")
else:
    print("Target string not found")
