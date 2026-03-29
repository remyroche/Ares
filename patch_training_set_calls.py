with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

# Call inside training.py:
old_call = """        (
            X,
            y,
            y_ret,
            cols,
            w,
            meta_idx,
            lbl_vals,
        ) = build_hourly_training_set_and_weights(
            panel,
            feats,
            mkt_gates,
            cfg,
            syms,
            ts,
            p_exh_hist,
            H,
            k,
            trend_filter=move_bucket,
            feature_key=feat_key,
            extra_feature_keys=_meta_feature_keys_for_kind(cfg, strat),
            label_method="triple_barrier",
            fixed_tp=fixed_tp,
            fixed_sl=fixed_sl,
            side=side,
            _cached_cand_mask=mask_by_strategy.get(strategy_id),
            _cached_tb=tb_cache.get(_tb_key),
            _precomputed_events=_pre_h,
            _geom_frames=_geom,
        )"""

new_call = """        (
            X,
            y,
            y_ret,
            cols,
            w,
            meta_idx,
            lbl_vals,
        ) = build_hourly_training_set_and_weights(
            panel,
            feats,
            mkt_gates,
            cfg,
            syms,
            ts,
            p_exh_hist,
            H,
            k,
            trend_filter=move_bucket,
            strategy=strat,
            feature_key=feat_key,
            extra_feature_keys=_meta_feature_keys_for_kind(cfg, strat),
            label_method="triple_barrier",
            fixed_tp=fixed_tp,
            fixed_sl=fixed_sl,
            side=side,
            _cached_cand_mask=mask_by_strategy.get(strategy_id),
            _cached_tb=tb_cache.get(_tb_key),
            _precomputed_events=_pre_h,
            _geom_frames=_geom,
        )"""

content = content.replace(old_call, new_call)


old_call_meta = """            _w_opt = _optimize_training_sample_weights(
                df=pd.DataFrame({"ts": _meta_ts}),
                X_frame=X_meta_base.select_dtypes(include=[np.number]).fillna(0.0),
                y_ret=y_ret_raw_main,
                label_times=_meta_label_times,
                base_weights=w_meta_main,
                cfg={
                    **cfg,
                    "sample_weight_opt_trials": int(
                        cfg.get(
                            "meta_sample_weight_opt_trials",
                            cfg.get("sample_weight_opt_trials", 16),
                        )
                    ),
                },
                stage=f"meta_{k}",
                extra_components=_meta_extra,
            )"""

new_call_meta = """            _w_opt = _optimize_training_sample_weights(
                df=pd.DataFrame({"ts": _meta_ts}),
                X_frame=X_meta_base.select_dtypes(include=[np.number]).fillna(0.0),
                y_ret=y_ret_raw_main,
                label_times=_meta_label_times,
                base_weights=w_meta_main,
                cfg={
                    **cfg,
                    "sample_weight_opt_trials": int(
                        cfg.get(
                            "meta_sample_weight_opt_trials",
                            cfg.get("sample_weight_opt_trials", 16),
                        )
                    ),
                },
                stage=f"meta_{k}",
                extra_components=_meta_extra,
                strategy=strat,
            )"""

content = content.replace(old_call_meta, new_call_meta)

old_call_meta_clf = """                w_meta_clf = _optimize_training_sample_weights(
                    df=pd.DataFrame({"ts": _meta_ts_c}),
                    X_frame=X_meta_base.select_dtypes(include=[np.number]).fillna(
                        0.0
                    ),
                    y_ret=_y_per_h[_mid_h_c].astype(np.float64),
                    label_times=_meta_label_times_c,
                    base_weights=w_meta_clf,
                    cfg={
                        **cfg,
                        "sample_weight_opt_trials": int(
                            cfg.get(
                                "meta_sample_weight_opt_trials",
                                cfg.get("sample_weight_opt_trials", 16),
                            )
                        ),
                    },
                    stage=f"meta_clf_{k}",
                    extra_components={
                        "magnitude": w_mag_clf,
                        "excursion": w_exc_clf,
                    },
                )"""

new_call_meta_clf = """                w_meta_clf = _optimize_training_sample_weights(
                    df=pd.DataFrame({"ts": _meta_ts_c}),
                    X_frame=X_meta_base.select_dtypes(include=[np.number]).fillna(
                        0.0
                    ),
                    y_ret=_y_per_h[_mid_h_c].astype(np.float64),
                    label_times=_meta_label_times_c,
                    base_weights=w_meta_clf,
                    cfg={
                        **cfg,
                        "sample_weight_opt_trials": int(
                            cfg.get(
                                "meta_sample_weight_opt_trials",
                                cfg.get("sample_weight_opt_trials", 16),
                            )
                        ),
                    },
                    stage=f"meta_clf_{k}",
                    extra_components={
                        "magnitude": w_mag_clf,
                        "excursion": w_exc_clf,
                    },
                    strategy=strat,
                )"""

content = content.replace(old_call_meta_clf, new_call_meta_clf)

with open('extreme_price_movements/training.py', 'w') as f:
    f.write(content)
print("Patched all weight calls")
