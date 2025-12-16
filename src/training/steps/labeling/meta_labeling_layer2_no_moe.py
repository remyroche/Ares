from __future__ import annotations


def layer2_hpo_block():
    tprint_info(f"🔧 Layer 2 mode: {layer2_mode} (enable_committee_voting_hpo={enable_committee_voting_hpo})")

    committee_weight_factor_series: Optional[pd.Series] = None

    if enable_committee_voting_hpo or enable_committee_weight_factor:
        if enable_committee_voting_hpo:
            tprint_info("🧪 Layer 2: Optimizing Committee Consensus Weights...")

        # A. PRE-COMPUTE COMMITTEE LABEL MATRIX (The "Expert Panel")
        tprint_info("🏗️ Pre-computing Committee of 6 Label Matrix...")

        # 1. Define the 6 Profiles
        # Scalp: Tight (TP 1.2 / SL 0.6)
        # Swing: Balanced (TP 2.0 / SL 1.0)
        # Trend: Looser (TP 3.0 / SL 1.5)
        # Multipliers: Lower (0.8x), Upper (1.2x)

        # Base Multipliers (TP, SL)
        # Scalp: 1.2/0.6
        # Swing: 2.0/1.0
        # Trend: 3.0/1.5
        base_profiles = {
            "scalp": (1.2, 0.6, 8),
            "swing": (1.8, 0.9, 12),
            "trend": (2.4, 1.2, 24),
        }
        vol_scalars = {"lower": 0.8, "upper": 1.2}

        committee_configs = []
        committee_names = []

        for p_name, (tp_base, sl_base, h_base) in base_profiles.items():
            for v_name, v_scalar in vol_scalars.items():
                config_id = f"{p_name}_{v_name}"
                # Scale multipliers (effectively scaling volatility assumption)
                committee_configs.append(
                    TripleBarrierConfig(
                        tp_multiplier=tp_base * v_scalar,
                        sl_multiplier=sl_base * v_scalar,
                        horizon=h_base,
                    )
                )
                committee_names.append(config_id)

        # 2. Vectorized computation of all 6 outcomes
        try:
            # Add Kalman columns if not present (Stage 0 result)
            best_Q = best_kalman_params.get('kalman_Q', 1e-4)
            best_R = best_kalman_params.get('kalman_R', 0.01)

            # Re-compute Kalman smooth data for labeling (acausal/smooth for labeling)
            # using the optimized parameters
            kalman_price_smooth, kalman_vol_smooth = compute_kalman_smoothed_price_and_volatility(
                prices=market_data['close'],
                process_noise=best_Q,
                measurement_noise=best_R,
                vol_window=20
            )

            mk_data_voting = market_data.copy()
            mk_data_voting['kalman_price'] = kalman_price_smooth
            mk_data_voting['kalman_volatility'] = kalman_vol_smooth

            committee_results = compute_multi_triple_barrier_outcomes_vectorized(
                market_data=mk_data_voting,
                primary_signals=primary_signals,
                configs=committee_configs,
                transaction_cost=DEFAULT_TRANSACTION_COST,
            )

            # 3. Assemble Label Matrix (Rows=Events, Cols=6 Experts)
            # Find common events (primary_signals != 0)
            event_mask = primary_signals['consensus'] != 0
            event_idx = primary_signals[event_mask].index

            # Initialize matrices
            label_matrix_values = np.zeros((len(event_idx), len(committee_configs)), dtype=np.int8)
            returns_matrix_values = np.full((len(event_idx), len(committee_configs)), np.nan, dtype=np.float32)

            for i, res in enumerate(committee_results):
                # align to event_idx
                lbls = res['labels'].reindex(event_idx).fillna(0).values.astype(int)
                rets = res['returns'].reindex(event_idx).values.astype(np.float32)

                label_matrix_values[:, i] = lbls
                returns_matrix_values[:, i] = rets

            tprint_success(f"✅ Committee Matrices Built: {label_matrix_values.shape} (Events x Experts)")

            try:
                n_ev = int(label_matrix_values.shape[0])
                for j, name in enumerate(list(committee_names)):
                    col = np.asarray(label_matrix_values[:, j], dtype=float)
                    if col.size <= 0:
                        continue
                    frac_pos = float(np.mean(col > 0.0))
                    frac_neg = float(np.mean(col < 0.0))
                    frac_zero = float(np.mean(col == 0.0))
                    tprint_info(
                        f"   [committee expert] {name}: +={frac_pos:.2%}, -={frac_neg:.2%}, 0={frac_zero:.2%} (n={n_ev})"
                    )
            except Exception:
                pass

            try:
                fired_mask = (label_matrix_values != 0)
                ret_mat = np.asarray(returns_matrix_values, dtype=float)
                ret_mat = np.where(fired_mask, ret_mat, np.nan)

                abs_ret = np.abs(ret_mat)
                abs_ret_mean = np.nanmean(abs_ret, axis=1)
                abs_ret_mean = np.where(np.isfinite(abs_ret_mean), abs_ret_mean, 0.0)

                positive_abs = abs_ret_mean[abs_ret_mean > 0]
                abs_med = float(np.nanmedian(positive_abs)) if positive_abs.size > 0 else 0.0
                if np.isfinite(abs_med) and abs_med > 0:
                    mag_factor = abs_ret_mean / (abs_med + 1e-12)
                else:
                    mag_factor = np.ones_like(abs_ret_mean, dtype=float)

                sign_mat = np.asarray(label_matrix_values, dtype=float)
                sign_mat = np.where(fired_mask, np.sign(sign_mat), np.nan)
                mean_sign = np.nanmean(sign_mat, axis=1)
                agree = np.abs(mean_sign)
                agree = np.where(np.isfinite(agree), agree, 0.0)
                agree = np.clip(agree, 0.0, 1.0)

                alpha = float(
                    best_weighting_params.get(
                        "committee_agreement_alpha",
                        config.get("committee_agreement_alpha", 0.5),
                    )
                )
                mag_clip = float(
                    best_weighting_params.get(
                        "committee_mag_clip",
                        config.get("committee_mag_clip", 5.0),
                    )
                )
                mag_factor = np.where(np.isfinite(mag_factor), mag_factor, 1.0)
                mag_factor = np.clip(mag_factor, 0.0, mag_clip)

                factor = (1.0 + alpha * agree) * mag_factor
                factor_mean = float(np.nanmean(factor[np.isfinite(factor)])) if np.isfinite(factor).any() else 1.0
                if np.isfinite(factor_mean) and factor_mean > 0:
                    factor = factor / factor_mean
                else:
                    factor = np.ones_like(factor, dtype=float)

                committee_weight_factor_series = pd.Series(factor, index=event_idx)

                try:
                    if bool(config.get("log_committee_weight_factor", True)):
                        v = committee_weight_factor_series.values.astype(float)
                        v = v[np.isfinite(v)]
                        if v.size > 0:
                            tprint_info(
                                "   [committee weight factor] "
                                f"n={int(v.size)}, mean={float(np.mean(v)):.4f}, min={float(np.min(v)):.4f}, max={float(np.max(v)):.4f}"
                            )
                except Exception:
                    pass
            except Exception:
                committee_weight_factor_series = None

        except Exception as e:
            if allow_committee_fallback_to_standard:
                tprint_warning(
                    f"⚠️ Committee pre-computation failed: {e}. Falling back to standard HPO."
                )
                layer2_mode = "standard"
            else:
                tprint_error(
                    f"❌ Committee pre-computation failed: {e}. Aborting (set allow_committee_fallback_to_standard=true to fallback)."
                )
                return {"success": False, "error": str(e)}

    if layer2_mode == "committee":
        # Updated search space for Consensus Voting
        layer2_search_space = {
            "w_scalp": {"type": "float", "low": 0.0, "high": 2.0},
            "w_swing": {"type": "float", "low": 0.0, "high": 2.0},
            "w_trend": {"type": "float", "low": 0.0, "high": 2.0},
            "consensus_quantile": {"type": "float", "low": 0.50, "high": 0.85},
        }
    else:
        tprint_info("🧪 Layer 2: Optimizing Trading Parameters...")
        layer2_search_space = {
            "trail_distance_atr_mult": {"type": "float", "low": 0.5, "high": 3.0},
        }

    if layer2_mode == "committee":
        l2_trial_counter = 0

        def layer2_objective(trial_params: Dict[str, Any]) -> float:
            """
            Layer 2 objective using Committee Voting:
            - Optimizes weights for [Scalp, Swing, Trend] experts.
            - Computes consensus score.
            - Thresholds for binary labels.
            """
            nonlocal l2_trial_counter
            l2_trial_counter += 1

            w_scalp = trial_params["w_scalp"]
            w_swing = trial_params["w_swing"]
            w_trend = trial_params["w_trend"]
            threshold = float(trial_params.get("consensus_threshold", 0.5))
            cq = trial_params.get("consensus_quantile", None)

            try:
                metrics_trial = _compute_layer2_metrics_committee(trial_params)
                utility = float(metrics_trial.get("utility", -1.0))
                if not np.isfinite(float(utility)):
                    utility = -1.0
            except Exception:
                metrics_trial = {}
                utility = -1.0

            try:
                if bool(config.get("layer2_log_trials", True)):
                    every = int(config.get("layer2_trial_log_every", 1))
                    if every <= 0:
                        every = 1
                    if (l2_trial_counter % every) == 0:
                        try:
                            n_trades = int(metrics_trial.get("n_trades") or 0)
                        except Exception:
                            n_trades = 0

                        try:
                            trades_per_day = float(metrics_trial.get("trades_per_day"))
                        except Exception:
                            trades_per_day = float("nan")

                        try:
                            sharpe_mean = float(metrics_trial.get("sharpe_mean"))
                        except Exception:
                            sharpe_mean = float("nan")

                        try:
                            take_rate = float(metrics_trial.get("take_rate"))
                        except Exception:
                            take_rate = float("nan")

                        try:
                            cs_p50 = float(metrics_trial.get("consensus_p50"))
                        except Exception:
                            cs_p50 = float("nan")

                        try:
                            cs_p90 = float(metrics_trial.get("consensus_p90"))
                        except Exception:
                            cs_p90 = float("nan")

                        try:
                            trade_mean = float(metrics_trial.get("trade_mean_return"))
                        except Exception:
                            trade_mean = float("nan")

                        try:
                            trade_win = float(metrics_trial.get("trade_win_rate"))
                        except Exception:
                            trade_win = float("nan")

                        tprint_info(
                            "   [L2 committee trial] "
                            f"trial={l2_trial_counter}, utility={utility:.4f}, sharpe={sharpe_mean:.4f}, "
                            f"n_trades={n_trades}, tpd={trades_per_day:.2f}, "
                            f"thr={threshold:.3f}, q={cq}, thr_eff={metrics_trial.get('consensus_threshold_effective')}, "
                            f"w=({w_scalp:.3f},{w_swing:.3f},{w_trend:.3f}), "
                            f"take_rate={take_rate:.3f}, cs_p50={cs_p50:.3f}, cs_p90={cs_p90:.3f}, "
                            f"trade_mean={trade_mean:.4%}, win={trade_win:.2%}"
                        )
            except Exception:
                pass

            return float(utility)

    else:
        def layer2_objective(trial_params: Dict[str, Any]) -> float:
            try:
                metrics_trial = _compute_layer2_metrics(trial_params)
                return float(metrics_trial.get("utility", -1.0))
            except Exception:
                return -1.0

    meta_feature_cfg = config.get("meta_feature_engineering", {})
    volume_available = "volume" in market_data.columns

    # ------------------------------------------------------------------
    # OPTION: Use cached features from labeled_data artifact (Phase 3)
    # ------------------------------------------------------------------
    use_cached_features = bool(config.get("hpo_use_cached_features", False))
    cached_features_loaded = False

    if use_cached_features:
        try:
            # Attempt to load features from labeled_data artifact
            from src.artifacts.versioned_artifact_store import VersionedArtifactStore
            store = VersionedArtifactStore()
            labeled_data = store.get_artifact(
                f"labeled_data_{symbol}_{timeframe}",
                version="latest"
            )
            if labeled_data is not None and hasattr(labeled_data, 'columns'):
                # Extract feature columns (exclude label/return columns)
                exclude_cols = {
                    'binary_label', 'realized_return', 'target', 'target_long', 'target_short',
                    'meta_probability', 'meta_probability_ensemble', 'exit_reason', 'duration'
                }
                feature_cols = [c for c in labeled_data.columns if c not in exclude_cols]
                if len(feature_cols) > 10:
                    meta_features_full = labeled_data[feature_cols].copy()
                    cached_features_loaded = True
                    tprint_success(f"✅ Loaded {len(feature_cols)} cached features from labeled_data artifact")
        except Exception as cache_exc:
            tprint_warning(f"⚠️ Failed to load cached features: {cache_exc}. Regenerating.")

    if not cached_features_loaded:
        # PRE-CALCULATE META-FEATURES ONCE (Performance Optimization)
        # Use baseline returns/labels as proxy. The goal is to get X features.
        # Note: If meta-features rely heavily on exact realized_return of the specific TBM,
        # this is an approximation. But for HPO speed, it is necessary.
        # Most features (technicals, regime, kalman) depend only on market_data/signals.
        tprint_info("🏗️ Layer 2: Pre-calculating meta-features with optimized Kalman params...")
    mf_config_opt = meta_feature_cfg.copy()
    try:
        hpo_use_full_feature_set = bool(config.get("hpo_use_full_feature_set", True))
        if hpo_use_full_feature_set:
            mf_config_opt["enable_feature_selection"] = False
            if "max_features" in mf_config_opt:
                mf_config_opt.pop("max_features", None)
    except Exception:
        pass
    mf_config_opt['kalman_Q'] = best_kalman_params.get('kalman_Q', 1e-4)
    mf_config_opt['kalman_R'] = best_kalman_params.get('kalman_R', 0.01)

    # Generate dummy stop threshold for feature generation (won't affect independent features)
    dummy_stop_thr = (atr_frac * 1.0).astype(float).clip(lower=0.002)
    dummy_profit_thr = (atr_frac * 2.0).astype(float).clip(lower=0.008)

    _, meta_features_full, _, _ = build_meta_features_for_model(
        market_data=market_data,
        primary_signals=primary_signals,
        realized_returns=baseline_returns,
        binary_labels=binary_labels,
        event_durations=event_durations_raw,
        mfe_series=mfe_raw,
        mae_series=mae_raw,
        adaptive_stop_threshold=baseline_stop.reindex(market_data.index),
        horizon=12,
        volume_available=volume_available,
        meta_feature_cfg=mf_config_opt,
    )

    # ------------------------------------------------------------------
    # WEIGHTED PIPELINE: Add Kalman-based Features
    # ------------------------------------------------------------------
    # Uses the optimized Q and R from Stage 0 (RTS) in a CAUSAL Kalman Filter
    # for features that can be used in live trading.
    tprint_info("🏗️ Generating Kalman-based features (weighted pipeline)...")

    kalman_Q_opt = best_kalman_params.get('kalman_Q', 1e-4)
    kalman_R_opt = best_kalman_params.get('kalman_R', 0.01)

    try:
        kalman_features = generate_kalman_features(
            market_data=market_data,
            kalman_Q=kalman_Q_opt,
            kalman_R=kalman_R_opt,
        )

        # Merge Kalman features with existing meta features
        # Align indices and handle any missing data
        kalman_features_aligned = kalman_features.reindex(meta_features_full.index).fillna(0)

        # Add Kalman features to meta_features_full
        for col in kalman_features_aligned.columns:
            meta_features_full[col] = kalman_features_aligned[col]

        tprint_success(f"✅ Added {len(kalman_features.columns)} Kalman features")
    except Exception as kf_exc:
        tprint_warning(f"⚠️ Kalman feature generation failed: {kf_exc}. Continuing without Kalman features.")

    tprint_success(f"✅ Meta-features pre-calculated: {meta_features_full.shape[1]} columns")

    meta_features_full_raw = meta_features_full.copy()

    # ------------------------------------------------------------------
    # QUALITY-BASED FEATURE SELECTION (After Layer 0, Before HPO Loop)
    # ------------------------------------------------------------------
    # This solves the circular dependency: features are selected based on
    # unsupervised quality metrics (Signal/Noise ratio) rather than labels.
    #
    # Pipeline:
    # 1. Generate multi-horizon versions (Short/Medium/Long) for cross-timeframe
    # 2. Calculate Signal-to-Noise ratio for all features
    # 3. Reduce by correlation, keeping highest quality features
    #
    target_feature_count = int(config.get("target_feature_count", 70))
    feature_correlation_threshold = float(config.get("feature_correlation_threshold", 0.85))
    enable_multi_horizon = config.get("enable_multi_horizon_features", True)
    enable_cross_features = config.get("enable_cross_features", True)
    use_hierarchical_selection = config.get("use_hierarchical_selection", True)
    use_lgbm_sweep = config.get("use_lgbm_sweep", True)
    lgbm_lookahead = int(config.get("lgbm_sweep_lookahead", 4))
    lgbm_max_features = int(config.get("lgbm_max_features", 300))
    quality_drop_percentile = float(config.get("quality_drop_percentile", 20.0))
    use_feature_cache = config.get("use_feature_selection_cache", True)
    force_recompute_features = config.get("force_recompute_features", False)

    # Custom horizon configuration (can be overridden in config)
    horizon_config = config.get("feature_horizon_config", {
        "Short": 5,    # ~1.25 hours at 15m (fast signals)
        "Medium": 20,  # ~5 hours at 15m (medium signals)
        "Long": 60,    # ~15 hours at 15m (slow signals)
    })

    tprint_info("🔬 Running De Prado feature selection pipeline...")
    try:
        meta_features_full, feature_quality_scores = select_features_with_quality(
            df_features=meta_features_full,
            target_n=target_feature_count,
            correlation_threshold=feature_correlation_threshold,
            generate_horizons=enable_multi_horizon,
            horizon_config=horizon_config,
            enable_cross_features=enable_cross_features,
            market_data=market_data,
            config=config,
            # De Prado pipeline parameters
            use_hierarchical=use_hierarchical_selection,
            use_lgbm_sweep=use_lgbm_sweep,
            lgbm_lookahead=lgbm_lookahead,
            lgbm_max_features=lgbm_max_features,
            quality_drop_percentile=quality_drop_percentile,
            # Caching parameters
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            use_cache=use_feature_cache,
            force_recompute=force_recompute_features,
        )

        # Store quality scores for potential later use
        self._feature_quality_scores = feature_quality_scores

        tprint_success(
            f"✅ Feature selection complete: {len(meta_features_full.columns)} features "
            f"(target={target_feature_count})"
        )

        # Persist feature selection results immediately
        try:
            ts_fs = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            fs_artifact_path = Path("outcomes") / f"hpo_feature_selection_{symbol}_{timeframe}_{ts_fs}.json"
            fs_payload = {
                "selected_features": list(meta_features_full.columns),
                "quality_scores": feature_quality_scores,
                "target": target_feature_count,
                "timestamp": ts_fs,
            }
            fs_artifact_path.parent.mkdir(parents=True, exist_ok=True)
            with open(fs_artifact_path, "w") as f:
                json.dump(fs_payload, f, indent=2, default=str)
            tprint_info(f"   💾 Saved feature selection stage to {fs_artifact_path}")
        except Exception as fs_save_exc:
            tprint_warning(f"   ⚠️ Failed to save feature selection artifact: {fs_save_exc}")
    except Exception as fs_exc:
        tprint_warning(f"⚠️ Feature selection failed: {fs_exc}. Using all features.")
        self._feature_quality_scores = {}
        meta_features_full = meta_features_full_raw

    if False:
        # --- LEGACY RETENTION FOR INTERFACE COMPATIBILITY ---
        # Construct dummy objects if needed by downstream
        # Use Swing Upper (Index 3) as the " Representative" return for downstream analysis if needed,
        # but the HPO was driven by the Weighted P&L.
        l2_returns_clean = committee_results[3]['returns'].reindex(event_idx).fillna(0.0)
        valid_idx = np.ones(len(pnl_series), dtype=bool) # All events valid
        l2_labels_clean = pd.Series(l2_binary_labels, index=event_idx)
        l2_t_events = event_idx
        # ----------------------------------------------------


        # B. DYNAMIC WEIGHT GENERATION
        # Construct t1 (end times) Series for compute_uniqueness
        # Map start timestamps to integer locations
        t0_locs = pd.Series(np.arange(len(market_data)), index=market_data.index)
        start_locs = t0_locs.loc[l2_t_events].values
        # Get durations for these specific events
        dur_vals = l2_durations.loc[l2_t_events].values.astype(int)
        end_locs = np.minimum(start_locs + dur_vals, len(market_data) - 1)
        t1_vals = market_data.index[end_locs]
        t1_series = pd.Series(t1_vals, index=l2_t_events)

        batch_consistency = full_consistency.reindex(l2_t_events).fillna(1.0).values
        batch_volatility = full_volatility.reindex(l2_t_events).fillna(0).values
        batch_uniqueness = compute_uniqueness(t1_series, market_index=market_data.index)

        sample_weights = generate_weights_per_label(
            returns=l2_returns_clean.values,
            t_events=l2_t_events,
            close_series=None,
            consistency_scores=batch_consistency,
            uniqueness_scores=batch_uniqueness.values,
            vol_proxy=batch_volatility,
            **best_weighting_params
        )

        try:
            if committee_weight_factor_series is not None:
                cf = committee_weight_factor_series.reindex(l2_t_events).fillna(1.0).values.astype(float)
                cf = np.where(np.isfinite(cf) & (cf > 0.0), cf, 1.0)
                sample_weights = np.asarray(sample_weights, dtype=float) * cf
                sw_mean = float(np.mean(sample_weights)) if sample_weights.size else 1.0
                if np.isfinite(sw_mean) and sw_mean > 0:
                    sample_weights = sample_weights / sw_mean
        except Exception:
            pass

        # C. SUBSET META-FEATURES (Fast)
        X_trial = meta_features_full.loc[valid_idx].fillna(0)

        # D. FAST MODEL TRAINING WITH CV
        n_cv_folds = 5
        fast_model = lgb.LGBMClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.1, n_jobs=-1, verbose=-1, random_state=42
        )

        try:
            cv_preds, folds_sharpe, mean_brier, mean_ece = _cross_val_predict_proba_and_fold_sharpes_weighted(
                estimator=fast_model,
                X=X_trial,
                y=l2_labels_clean,
                sample_weight=sample_weights,
                n_splits=n_cv_folds,
                returns=l2_returns_clean.values.astype(float),
                direction=direction,
                prob_thr=0.5,
                use_calibration=True,
                enable_ev_gating=bool(config.get("enable_ev_gating", False)),
                ev_margin=config.get("ev_margin", 0.0),
            )
        except Exception:
            return -1.0

        # E. COMPUTE AUC (for trapezoidal gate)
        try:
            mean_auc = roc_auc_score(l2_labels_clean.values, cv_preds)
        except Exception:
            mean_auc = 0.5

        y_true_arr = l2_labels_clean.values.astype(float)
        returns_arr = l2_returns_clean.values.astype(float)

        # G. COMPUTE TRADES PER DAY (from predicted trades, not event count)
        # Use the same threshold as sizing to avoid density gate being constant.
        try:
            pred_trade_mask = np.asarray(cv_preds, dtype=float) >= 0.5
            n_pred_trades = int(np.sum(pred_trade_mask))
        except Exception:
            n_pred_trades = int(len(l2_returns_clean))
        trades_per_day = float(n_pred_trades) / float(max(days_span, 1))

        # H. CALCULATE UTILITY (Trapezoidal Gate + Stability)
        utility = calculate_hpo_utility(
            folds_sharpe=folds_sharpe,
            auc=mean_auc,
            trades_per_day=trades_per_day,
            lambda_vol=1.2,   # Penalty for Sharpe volatility across folds
            w_auc=0.8,        # Slightly looser AUC weighting
            w_den=0.5,        # Moderate density weight
            calibration_brier=mean_brier,
            calibration_ece=mean_ece,
            w_cal=0.0,
        )

        try:
            utility, q_details = _apply_hpo_quality_penalty(
                utility=utility,
                returns=l2_returns_clean.values,
                labels=l2_labels_clean.values,
                exit_reasons=l2_exit_reasons.loc[l2_t_events].values if l2_exit_reasons is not None else None,
                durations=l2_durations.loc[l2_t_events].values if l2_durations is not None else None,
                horizon=12,
                tx_cost=float(DEFAULT_TRANSACTION_COST),
                config=config,
            )
        except Exception:
            q_details = {}

        # Log objective components for traceability
        try:
            tprint_info(
                "   [L2 objective] "
                f"utility={utility:.4f}, auc={mean_auc:.4f}, "
                f"trades_per_day={trades_per_day:.2f}, "
                f"folds_sharpe_mean={float(np.mean(folds_sharpe)):.4f}, "
                f"folds_sharpe_std={float(np.std(folds_sharpe, ddof=1)) if len(folds_sharpe)>1 else 0.0:.4f}"
            )
        except Exception:
            pass

        return utility

    def _compute_layer2_metrics(params: Dict[str, Any]) -> Dict[str, Any]:
        """Single-shot computation of Layer 2 metrics for reporting."""
        trail_dist = float(params.get("trail_distance_atr_mult", 0.0))

        prof_thr = fixed_layer2_profit_thr
        stop_thr = fixed_layer2_stop_thr

        (
            l2_returns,
            l2_labels,
            l2_exit_reasons,
            l2_durations,
            l2_mfe,
            l2_mae,
            _, _
        ) = compute_realized_returns(
            market_data,
            primary_signals,
            profit_threshold=prof_thr,
            stop_threshold=stop_thr,
            horizon=12,
            transaction_cost=DEFAULT_TRANSACTION_COST,
            min_event_spacing=2,
            trail_distance_atr_mult=trail_dist,
            atr_series=atr_series,
        )

        valid_idx = ~l2_labels.isna()
        if valid_idx.sum() < 50:
            return {"valid_events": int(valid_idx.sum()), "utility": -1.0}

        l2_t_events = l2_returns.index[valid_idx]
        l2_returns_clean = l2_returns[valid_idx]
        l2_labels_clean = l2_labels[valid_idx]

        t0_locs = pd.Series(np.arange(len(market_data)), index=market_data.index)
        start_locs = t0_locs.loc[l2_t_events].values
        dur_vals = l2_durations.loc[l2_t_events].values.astype(int)
        end_locs = np.minimum(start_locs + dur_vals, len(market_data) - 1)
        t1_vals = market_data.index[end_locs]
        t1_series = pd.Series(t1_vals, index=l2_t_events)

        batch_consistency = full_consistency.reindex(l2_t_events).fillna(1.0).values
        batch_volatility = full_volatility.reindex(l2_t_events).fillna(0).values
        batch_uniqueness = compute_uniqueness(t1_series, market_index=market_data.index)

        sample_weights = generate_weights_per_label(
            returns=l2_returns_clean.values,
            t_events=l2_t_events,
            close_series=None,
            consistency_scores=batch_consistency,
            uniqueness_scores=batch_uniqueness.values,
            vol_proxy=batch_volatility,
            **best_weighting_params
        )

        try:
            if committee_weight_factor_series is not None:
                cf = committee_weight_factor_series.reindex(l2_t_events).fillna(1.0).values.astype(float)
                cf = np.where(np.isfinite(cf) & (cf > 0.0), cf, 1.0)
                sample_weights = np.asarray(sample_weights, dtype=float) * cf
                sw_mean = float(np.mean(sample_weights)) if sample_weights.size else 1.0
                if np.isfinite(sw_mean) and sw_mean > 0:
                    sample_weights = sample_weights / sw_mean
        except Exception:
            pass

        X_trial = meta_features_full.loc[valid_idx].fillna(0)
        n_cv_folds = 5
        fast_model = lgb.LGBMClassifier(
            n_estimators=60, max_depth=3, learning_rate=0.1, n_jobs=-1, verbose=-1, random_state=42
        )

        try:
            cv_preds, folds_sharpe, mean_brier, mean_ece = _cross_val_predict_proba_and_fold_sharpes_weighted(
                estimator=fast_model,
                X=X_trial,
                y=l2_labels_clean,
                sample_weight=sample_weights,
                n_splits=n_cv_folds,
                returns=l2_returns_clean.values.astype(float),
                direction=direction,
                prob_thr=0.5,
                use_calibration=True,
                enable_ev_gating=bool(config.get("enable_ev_gating", False)),
                ev_margin=config.get("ev_margin", 0.0),
            )
        except Exception:
            return {"valid_events": int(valid_idx.sum()), "utility": -1.0}

        try:
            mean_auc = roc_auc_score(l2_labels_clean.values, cv_preds)
        except Exception:
            mean_auc = 0.5

        y_true_arr = l2_labels_clean.values.astype(float)
        returns_arr = l2_returns_clean.values.astype(float)

        # Use predicted trade frequency (not raw event count) so density gating
        # reflects actual model aggressiveness.
        try:
            pred_trade_mask = np.asarray(cv_preds, dtype=float) >= 0.5
            n_pred_trades = int(np.sum(pred_trade_mask))
        except Exception:
            n_pred_trades = int(len(l2_returns_clean))
        trades_per_day = float(n_pred_trades) / float(max(days_span, 1))
        lambda_vol = 1.2
        w_auc = 1.0
        w_den = 0.5

        per_fold_metrics = []
        try:
            per_fold_metrics = _compute_fold_metrics_from_oof(
                X=X_trial,
                y_true=y_true_arr,
                probs=np.asarray(cv_preds, dtype=float),
                returns=np.asarray(returns_arr, dtype=float),
                threshold=0.5,
                days_span=float(days_span),
                transaction_cost=0.0,
            )
        except Exception:
            per_fold_metrics = []

        per_regime_metrics: Dict[str, Any] = {}
        try:
            regime_labels = _build_event_regime_labels(
                market_data=market_data,
                event_index=l2_t_events,
                config=config,
            )
            per_regime_metrics = {
                "volatility": _compute_metrics_by_regime(
                    y_true=y_true_arr,
                    probs=np.asarray(cv_preds, dtype=float),
                    returns=np.asarray(returns_arr, dtype=float),
                    base_thr=0.5,
                    transaction_cost=0.0,
                    regime_labels=regime_labels.get("volatility_regime"),
                    days_span=float(days_span),
                ),
                "trend": _compute_metrics_by_regime(
                    y_true=y_true_arr,
                    probs=np.asarray(cv_preds, dtype=float),
                    returns=np.asarray(returns_arr, dtype=float),
                    base_thr=0.5,
                    transaction_cost=0.0,
                    regime_labels=regime_labels.get("trend_regime"),
                    days_span=float(days_span),
                ),
                "combined": _compute_metrics_by_regime(
                    y_true=y_true_arr,
                    probs=np.asarray(cv_preds, dtype=float),
                    returns=np.asarray(returns_arr, dtype=float),
                    base_thr=0.5,
                    transaction_cost=0.0,
                    regime_labels=regime_labels.get("combined_regime"),
                    days_span=float(days_span),
                ),
            }
        except Exception:
            per_regime_metrics = {}

        avg_sharpe = float(np.mean(folds_sharpe))
        vol_sharpe = float(np.std(folds_sharpe, ddof=1)) if len(folds_sharpe) > 1 else 0.0
        base_score = avg_sharpe - (lambda_vol * vol_sharpe)
        try:
            base_norm = float(np.sign(base_score) * np.log1p(abs(float(base_score))))
        except Exception:
            base_norm = 0.0
        if not np.isfinite(base_norm):
            base_norm = 0.0
        phi_auc = trapezoidal_gate(mean_auc, lower=0.52, sweet_spot=(0.56, 0.66), upper=0.72)
        phi_density = trapezoidal_gate(
            float(trades_per_day),
            lower=0.5,
            sweet_spot=(1.5, 5.0),
            upper=8.0,
        )
        modifier = (phi_auc ** w_auc) * (phi_density ** w_den)

        utility = calculate_hpo_utility(
            folds_sharpe=folds_sharpe,
            auc=mean_auc,
            trades_per_day=trades_per_day,
            lambda_vol=lambda_vol,
            w_auc=w_auc,
            w_den=w_den,
            calibration_brier=mean_brier,
            calibration_ece=mean_ece,
            w_cal=0.0,
        )

        q_details: Dict[str, Any] = {}
        try:
            utility, q_details = _apply_hpo_quality_penalty(
                utility=float(utility),
                returns=returns_arr,
                labels=y_true_arr,
                exit_reasons=l2_exit_reasons.loc[l2_t_events].values if l2_exit_reasons is not None else None,
                durations=l2_durations.loc[l2_t_events].values if l2_durations is not None else None,
                horizon=12,
                tx_cost=float(DEFAULT_TRANSACTION_COST),
                config=config,
            )
        except Exception:
            pass

        return {
            "valid_events": int(valid_idx.sum()),
            "utility": float(utility),
            "quality_penalty": q_details,
            "auc": float(mean_auc),
            "trades_per_day": float(trades_per_day),
            "calibration_brier": float(mean_brier) if mean_brier is not None else None,
            "calibration_ece": float(mean_ece) if mean_ece is not None else None,
            "sharpe_mean": float(np.mean(folds_sharpe)),
            "sharpe_std": float(np.std(folds_sharpe, ddof=1)) if len(folds_sharpe) > 1 else 0.0,
            "sharpe_min": float(np.min(folds_sharpe)),
            "sharpe_max": float(np.max(folds_sharpe)),
            "folds_sharpe_values": [float(v) for v in folds_sharpe.tolist()] if isinstance(folds_sharpe, np.ndarray) else [],
            "per_fold_metrics": per_fold_metrics,
            "per_regime_metrics": per_regime_metrics,
            "lambda_vol": lambda_vol,
            "w_auc": w_auc,
            "w_den": w_den,
            "avg_sharpe": avg_sharpe,
            "vol_sharpe": vol_sharpe,
            "base_score": float(base_score),
            "base_norm": float(base_norm) if np.isfinite(base_norm) else float("nan"),
            "phi_auc": float(phi_auc),
            "phi_density": float(phi_density),
            "modifier": float(modifier),
        }

    def _compute_layer2_metrics_committee(params: Dict[str, Any]) -> Dict[str, Any]:
        try:
            w_scalp = float(params.get("w_scalp", 0.0))
            w_swing = float(params.get("w_swing", 0.0))
            w_trend = float(params.get("w_trend", 0.0))
            threshold = float(params.get("consensus_threshold", 0.5))
            consensus_quantile = params.get("consensus_quantile", None)
            consensus_quantile = float(consensus_quantile) if consensus_quantile is not None else None
        except Exception:
            return {"valid_events": int(len(event_idx)), "utility": -1.0}

        weights_vec = np.array(
            [w_scalp, w_scalp, w_swing, w_swing, w_trend, w_trend],
            dtype=float,
        )
        total_weight = float(np.sum(weights_vec)) + 1e-8
        if (not np.isfinite(total_weight)) or (total_weight <= 1e-8):
            return {"valid_events": int(len(event_idx)), "utility": -1.0}

        consensus_score = label_matrix_values.dot(weights_vec) / total_weight
        try:
            cs = np.asarray(consensus_score, dtype=float)
            cs = cs[np.isfinite(cs)]
            consensus_mean = float(np.mean(cs)) if cs.size > 0 else float("nan")
            consensus_std = float(np.std(cs, ddof=1)) if cs.size > 1 else 0.0
            consensus_p10 = float(np.quantile(cs, 0.10)) if cs.size > 0 else float("nan")
            consensus_p50 = float(np.quantile(cs, 0.50)) if cs.size > 0 else float("nan")
            consensus_p90 = float(np.quantile(cs, 0.90)) if cs.size > 0 else float("nan")
            consensus_p99 = float(np.quantile(cs, 0.99)) if cs.size > 0 else float("nan")
            consensus_min = float(np.min(cs)) if cs.size > 0 else float("nan")
            consensus_max = float(np.max(cs)) if cs.size > 0 else float("nan")
            frac_pos = float(np.mean(cs > 0.0)) if cs.size > 0 else float("nan")
            frac_neg = float(np.mean(cs < 0.0)) if cs.size > 0 else float("nan")
        except Exception:
            consensus_mean = float("nan")
            consensus_std = float("nan")
            consensus_p10 = float("nan")
            consensus_p50 = float("nan")
            consensus_p90 = float("nan")
            consensus_p99 = float("nan")
            consensus_min = float("nan")
            consensus_max = float("nan")
            frac_pos = float("nan")
            frac_neg = float("nan")

        thr_effective = float(threshold)
        try:
            cs_full = np.asarray(consensus_score, dtype=float)
            cs_full = np.where(np.isfinite(cs_full), cs_full, -np.inf)
            if consensus_quantile is not None and np.isfinite(consensus_quantile):
                q = float(np.clip(consensus_quantile, 0.0, 0.999999))
                n = int(cs_full.size)
                if n > 0:
                    k = int(np.ceil((1.0 - q) * float(n)))
                    k = int(np.clip(k, 1, n))
                    top_idx = np.argpartition(cs_full, n - k)[n - k :]
                    take_mask = np.zeros(n, dtype=bool)
                    take_mask[top_idx] = True
                    try:
                        thr_effective = float(np.min(cs_full[top_idx])) if top_idx.size > 0 else float(threshold)
                    except Exception:
                        thr_effective = float(threshold)
                else:
                    take_mask = np.zeros(0, dtype=bool)
            else:
                take_mask = cs_full > float(threshold)
        except Exception:
            take_mask = np.asarray(consensus_score, dtype=float) > float(threshold)

        n_trades = int(np.sum(take_mask))
        take_rate = float(n_trades) / float(len(event_idx)) if len(event_idx) > 0 else 0.0
        trades_per_day = float(n_trades) / float(max(days_span, 1))

        committee_expert_stats: Dict[str, Any] = {}
        sanity_checks: Dict[str, Any] = {"violations": [], "debug_tables": {}}
        try:
            ev_idx = pd.DatetimeIndex(event_idx)
            consensus_arr = np.asarray(consensus_score, dtype=float)
            for j, name in enumerate(list(committee_names)):
                lbl_col = np.asarray(label_matrix_values[:, j], dtype=float)
                ret_col = np.asarray(returns_matrix_values[:, j], dtype=float)
                fired = lbl_col != 0.0
                n_fired = int(np.sum(fired))
                pos_mask = lbl_col > 0.0
                neg_mask = lbl_col < 0.0
                out = {
                    "n_events": int(lbl_col.size),
                    "n_fired": int(n_fired),
                    "frac_fired": float(n_fired) / float(max(int(lbl_col.size), 1)),
                    "frac_pos": float(np.mean(pos_mask)) if lbl_col.size else 0.0,
                    "frac_neg": float(np.mean(neg_mask)) if lbl_col.size else 0.0,
                }
                if n_fired > 0:
                    r = ret_col[fired]
                    r = r[np.isfinite(r)]
                    out["mean_return_on_fired"] = float(np.mean(r)) if r.size else 0.0
                    out["win_rate_on_fired"] = float(np.mean(r > 0.0)) if r.size else 0.0
                else:
                    out["mean_return_on_fired"] = 0.0
                    out["win_rate_on_fired"] = 0.0

                r_pos = ret_col[pos_mask]
                r_pos = r_pos[np.isfinite(r_pos)]
                out["n_pos"] = int(np.sum(pos_mask))
                out["mean_return_on_pos"] = float(np.mean(r_pos)) if r_pos.size else 0.0
                out["win_rate_on_pos"] = float(np.mean(r_pos > 0.0)) if r_pos.size else 0.0

                r_neg = ret_col[neg_mask]
                r_neg = r_neg[np.isfinite(r_neg)]
                out["n_neg"] = int(np.sum(neg_mask))
                out["mean_return_on_neg"] = float(np.mean(r_neg)) if r_neg.size else 0.0
                out["win_rate_on_neg"] = float(np.mean(r_neg > 0.0)) if r_neg.size else 0.0

                try:
                    n_pos = int(out.get("n_pos", 0))
                    n_neg = int(out.get("n_neg", 0))
                    mean_pos = float(out.get("mean_return_on_pos", 0.0))
                    mean_neg = float(out.get("mean_return_on_neg", 0.0))
                    if n_pos >= 20 and n_neg >= 20 and np.isfinite(mean_pos) and np.isfinite(mean_neg) and mean_pos < mean_neg:
                        sanity_checks["violations"].append(
                            {
                                "expert": str(name),
                                "n_pos": int(n_pos),
                                "n_neg": int(n_neg),
                                "mean_return_on_pos": float(mean_pos),
                                "mean_return_on_neg": float(mean_neg),
                            }
                        )

                        dbg_n = int(config.get("layer2_sanity_debug_rows", 15))
                        bad_idx = np.where((lbl_col > 0.0) & np.isfinite(ret_col) & (ret_col < 0.0))[0]
                        if bad_idx.size > 0:
                            dbg_df = pd.DataFrame(
                                {
                                    "timestamp": ev_idx[bad_idx].astype(str),
                                    "consensus": consensus_arr[bad_idx].astype(float),
                                    "realized_return": ret_col[bad_idx].astype(float),
                                    "costs": float(DEFAULT_TRANSACTION_COST),
                                }
                            )
                            dbg_df = dbg_df.sort_values("realized_return", ascending=True).head(int(max(1, dbg_n)))
                            sanity_checks["debug_tables"][str(name)] = dbg_df.to_dict(orient="records")
                            try:
                                tprint_warning(
                                    f"⚠️ Layer2 sanity: inverted pos/neg returns for expert={str(name)} "
                                    f"mean_pos={mean_pos:.6f} < mean_neg={mean_neg:.6f}. Sample bad pos trades:\n"
                                    + dbg_df.to_string(index=False)
                                )
                            except Exception:
                                pass
                except Exception:
                    pass

                committee_expert_stats[str(name)] = out
        except Exception:
            committee_expert_stats = {}

        committee_overlap: Dict[str, Any] = {}
        try:
            n_exp = int(label_matrix_values.shape[1])
            for i in range(n_exp):
                for j in range(i + 1, n_exp):
                    name_i = str(list(committee_names)[i])
                    name_j = str(list(committee_names)[j])
                    li = np.asarray(label_matrix_values[:, i], dtype=float)
                    lj = np.asarray(label_matrix_values[:, j], dtype=float)
                    fi = li != 0.0
                    fj = lj != 0.0
                    inter = fi & fj
                    union = fi | fj
                    n_inter = int(np.sum(inter))
                    n_union = int(np.sum(union))
                    jacc = float(n_inter) / float(max(n_union, 1))
                    sign_agree = float(np.mean(np.sign(li[inter]) == np.sign(lj[inter]))) if n_inter > 0 else 0.0
                    committee_overlap[f"{name_i}__{name_j}"] = {
                        "n_intersection": int(n_inter),
                        "n_union": int(n_union),
                        "jaccard": float(jacc),
                        "sign_agreement": float(sign_agree),
                    }
        except Exception:
            committee_overlap = {}

        committee_drivers: Dict[str, Any] = {}
        try:
            take_mask_arr = np.asarray(take_mask, dtype=bool)
            for j, name in enumerate(list(committee_names)):
                lbl_col = np.asarray(label_matrix_values[:, j], dtype=float)
                pos_take = int(np.sum((lbl_col > 0.0) & take_mask_arr))
                neg_take = int(np.sum((lbl_col < 0.0) & take_mask_arr))
                fired_take = int(np.sum((lbl_col != 0.0) & take_mask_arr))
                committee_drivers[str(name)] = {
                    "pos_on_taken": int(pos_take),
                    "neg_on_taken": int(neg_take),
                    "fired_on_taken": int(fired_take),
                    "share_fired_on_taken": float(fired_take) / float(max(n_trades, 1)),
                }
        except Exception:
            committee_drivers = {}

        try:
            ret_mat = np.asarray(returns_matrix_values, dtype=float)
            finite_mask = np.isfinite(ret_mat)

            w_row = np.asarray(weights_vec, dtype=float).reshape(1, -1)
            denom = np.sum(finite_mask * w_row, axis=1).astype(float) + 1e-8
            numer = np.sum(np.where(finite_mask, ret_mat, 0.0) * w_row, axis=1).astype(float)
            weighted_returns = numer / denom
            weighted_returns = np.where(np.isfinite(weighted_returns), weighted_returns, 0.0)
        except Exception:
            weighted_returns = np.zeros(int(len(event_idx)), dtype=float)

        trade_returns = np.asarray(weighted_returns, dtype=float)[np.asarray(take_mask, dtype=bool)]
        trade_returns = trade_returns[np.isfinite(trade_returns)]

        per_fold_metrics: List[Dict[str, Any]] = []
        fold_sharpes: List[float] = []
        fold_aucs: List[float] = []
        try:
            cv_local = TimeSeriesSplit(n_splits=5)
            for fold_idx, (_, te_idx) in enumerate(cv_local.split(np.arange(len(event_idx)))):
                te_idx = np.asarray(te_idx, dtype=int)
                if te_idx.size <= 0:
                    continue
                tr_mask = np.asarray(take_mask, dtype=bool)[te_idx]
                tr_returns = np.asarray(weighted_returns, dtype=float)[te_idx]
                tr_returns = tr_returns[np.isfinite(tr_returns)]
                n_trades_fold = int(np.sum(tr_mask))

                fold_auc = 0.5
                try:
                    score_fold = np.asarray(consensus_score, dtype=float)[te_idx]
                    ret_fold = np.asarray(weighted_returns, dtype=float)[te_idx]
                    mm_auc = np.isfinite(score_fold) & np.isfinite(ret_fold)
                    if int(np.sum(mm_auc)) >= 20:
                        y_auc = (ret_fold[mm_auc] > 0.0).astype(int)
                        if int(np.unique(y_auc).size) >= 2:
                            fold_auc = float(roc_auc_score(y_auc, score_fold[mm_auc]))
                except Exception:
                    fold_auc = 0.5

                days_span_fold = 1.0
                try:
                    idx_fold = pd.DatetimeIndex(event_idx[te_idx])
                    if len(idx_fold) >= 2:
                        days_span_fold = max(
                            1.0,
                            float((idx_fold.max() - idx_fold.min()).total_seconds() / 86400.0),
                        )
                except Exception:
                    days_span_fold = float(max(days_span, 1))

                if n_trades_fold <= 0:
                    per_fold_metrics.append(
                        {
                            "fold": int(fold_idx),
                            "auc": float(fold_auc),
                            "n_test": int(len(te_idx)),
                            "n_trades": 0,
                            "trades_per_day": 0.0,
                            "mean_return": 0.0,
                            "net_pnl_per_trade": 0.0,
                            "win_rate": 0.0,
                            "sharpe": 0.0,
                        }
                    )
                    fold_aucs.append(float(fold_auc))
                    fold_sharpes.append(0.0)
                    continue

                fold_trade_returns = np.asarray(weighted_returns, dtype=float)[te_idx][tr_mask]
                fold_trade_returns = fold_trade_returns[np.isfinite(fold_trade_returns)]
                mean_ret = float(np.mean(fold_trade_returns)) if fold_trade_returns.size > 0 else 0.0
                win_rate = float(np.mean(fold_trade_returns > 0)) if fold_trade_returns.size > 0 else 0.0
                sharpe_fold = 0.0
                try:
                    idx_te = pd.DatetimeIndex(event_idx[te_idx])
                    idx_tr = idx_te[np.asarray(tr_mask, dtype=bool)]
                    if fold_trade_returns.size > 0 and len(idx_tr) == int(fold_trade_returns.size):
                        day_index = pd.date_range(
                            start=idx_te.min().normalize(),
                            end=idx_te.max().normalize(),
                            freq="D",
                        )
                        daily_pnl = pd.Series(fold_trade_returns, index=idx_tr).groupby(idx_tr.normalize()).sum()
                        daily_pnl = daily_pnl.reindex(day_index, fill_value=0.0)

                        daily_log = np.log1p(daily_pnl.astype(float).values)
                        daily_log = daily_log[np.isfinite(daily_log)]
                        if int(daily_log.size) > 1:
                            mu = float(np.mean(daily_log))
                            sd = float(np.std(daily_log, ddof=1))
                            if sd > 1e-12:
                                sharpe_fold = float(np.clip(mu / sd * np.sqrt(365.0), -20.0, 20.0))
                except Exception:
                    sharpe_fold = 0.0

                per_fold_metrics.append(
                    {
                        "fold": int(fold_idx),
                        "auc": float(fold_auc),
                        "n_test": int(len(te_idx)),
                        "n_trades": int(n_trades_fold),
                        "trades_per_day": float(n_trades_fold) / float(max(days_span_fold, 1.0)),
                        "mean_return": float(mean_ret),
                        "net_pnl_per_trade": float(mean_ret),
                        "win_rate": float(win_rate),
                        "sharpe": float(sharpe_fold),
                    }
                )
                fold_aucs.append(float(fold_auc))
                fold_sharpes.append(float(sharpe_fold))
        except Exception:
            per_fold_metrics = []
            fold_sharpes = []
            fold_aucs = []

        per_regime_metrics: Dict[str, Any] = {}
        try:
            regime_labels = _build_event_regime_labels(
                market_data=market_data,
                event_index=event_idx,
                config=config,
            )

            def _by_regime(reg: pd.Series) -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                if reg is None or reg.empty:
                    return out
                lab = reg.astype(object)
                for rv in pd.unique(lab.dropna()):
                    rm = (lab == rv).to_numpy(dtype=bool)
                    n_events_r = int(np.sum(rm))
                    if n_events_r < 20:
                        continue
                    tm = np.asarray(take_mask, dtype=bool) & rm
                    n_trades_r = int(np.sum(tm))
                    if n_trades_r <= 0:
                        out[str(rv)] = {"n_events": n_events_r, "n_trades": 0}
                        continue
                    rvals = np.asarray(weighted_returns, dtype=float)[tm]
                    rvals = rvals[np.isfinite(rvals)]
                    if rvals.size <= 0:
                        out[str(rv)] = {"n_events": n_events_r, "n_trades": 0}
                        continue
                    mean_r = float(np.mean(rvals))
                    win_r = float(np.mean(rvals > 0.0))
                    sharpe_r = 0.0
                    try:
                        idx_r = pd.DatetimeIndex(event_idx[tm])
                        if int(rvals.size) > 0 and int(idx_r.size) == int(rvals.size):
                            day_index_r = pd.date_range(
                                start=idx_r.min().normalize(),
                                end=idx_r.max().normalize(),
                                freq="D",
                            )
                            daily_pnl_r = pd.Series(rvals, index=idx_r).groupby(idx_r.normalize()).sum()
                            daily_pnl_r = daily_pnl_r.reindex(day_index_r, fill_value=0.0)

                            daily_log_r = np.log1p(daily_pnl_r.astype(float).values)
                            daily_log_r = daily_log_r[np.isfinite(daily_log_r)]
                            if int(daily_log_r.size) > 1:
                                mu_r = float(np.mean(daily_log_r))
                                sd_r = float(np.std(daily_log_r, ddof=1))
                                if sd_r > 1e-12:
                                    sharpe_r = mu_r / sd_r * np.sqrt(365.0)
                    except Exception:
                        sharpe_r = 0.0
                    out[str(rv)] = {
                        "n_events": int(n_events_r),
                        "n_trades": int(n_trades_r),
                        "trades_per_day": float(n_trades_r) / float(max(days_span, 1.0)),
                        "mean_return": float(mean_r),
                        "net_pnl_per_trade": float(mean_r),
                        "win_rate": float(win_r),
                        "sharpe": float(sharpe_r),
                    }
                return out

            per_regime_metrics = {
                "volatility": _by_regime(regime_labels.get("volatility_regime")),
                "trend": _by_regime(regime_labels.get("trend_regime")),
                "combined": _by_regime(regime_labels.get("combined_regime")),
            }
        except Exception:
            per_regime_metrics = {}

        sharpe = 0.0
        try:
            tm_all = np.asarray(take_mask, dtype=bool)
            wr_all = np.asarray(weighted_returns, dtype=float)
            tm_fin = tm_all & np.isfinite(wr_all)
            trade_idx_all = pd.DatetimeIndex(event_idx[tm_fin])
            trade_ret_all = wr_all[tm_fin]
            if int(trade_ret_all.size) > 0 and int(trade_idx_all.size) == int(trade_ret_all.size):
                day_index_all = pd.date_range(
                    start=trade_idx_all.min().normalize(),
                    end=trade_idx_all.max().normalize(),
                    freq="D",
                )
                daily_pnl_all = pd.Series(trade_ret_all, index=trade_idx_all).groupby(trade_idx_all.normalize()).sum()
                daily_pnl_all = daily_pnl_all.reindex(day_index_all, fill_value=0.0)

                daily_log_all = np.log1p(daily_pnl_all.astype(float).values)
                daily_log_all = daily_log_all[np.isfinite(daily_log_all)]
                if int(daily_log_all.size) > 1:
                    mu_all = float(np.mean(daily_log_all))
                    sd_all = float(np.std(daily_log_all, ddof=1))
                    if sd_all > 1e-12:
                        sharpe = float(mu_all / sd_all * np.sqrt(365.0))
        except Exception:
            sharpe = 0.0
        sharpe = float(np.clip(float(sharpe), -20.0, 20.0))

        utility = float(sharpe) * float(np.log1p(n_trades)) if n_trades >= 50 else -1.0
        if n_trades >= 50 and n_trades < 100:
            utility = float(utility) * (float(n_trades) / 100.0)
        if not np.isfinite(float(utility)):
            utility = -1.0

        try:
            trade_mean = float(np.mean(trade_returns)) if trade_returns.size > 0 else 0.0
            trade_win_rate = float(np.mean(trade_returns > 0.0)) if trade_returns.size > 0 else 0.0
        except Exception:
            trade_mean = 0.0
            trade_win_rate = 0.0

        auc_val = 0.5
        try:
            score_auc = np.asarray(consensus_score, dtype=float)
            wr = np.asarray(weighted_returns, dtype=float)

            m = np.isfinite(score_auc) & np.isfinite(wr)
            if int(np.sum(m)) >= 20:
                y_true_auc = (wr[m] > 0.0).astype(int)
                if int(np.unique(y_true_auc).size) >= 2:
                    auc_val = float(roc_auc_score(y_true_auc, score_auc[m]))
                else:
                    auc_val = 0.5
            else:
                auc_val = 0.5
        except Exception:
            auc_val = 0.5

        mean_auc = float(auc_val)
        try:
            fau = np.asarray(fold_aucs, dtype=float)
            fau = fau[np.isfinite(fau)]
            if fau.size > 0:
                mean_auc = float(np.mean(fau))
        except Exception:
            pass

        folds_sharpe_arr = np.asarray(fold_sharpes, dtype=float)
        folds_sharpe_arr = folds_sharpe_arr[np.isfinite(folds_sharpe_arr)]
        if folds_sharpe_arr.size <= 0:
            folds_sharpe_arr = np.asarray([float(sharpe)], dtype=float)

        lambda_vol = 1.2
        w_auc = 1.0
        w_den = 0.5

        avg_sharpe = float(np.mean(folds_sharpe_arr))
        vol_sharpe = float(np.std(folds_sharpe_arr, ddof=1)) if folds_sharpe_arr.size > 1 else 0.0
        base_score = avg_sharpe - (lambda_vol * vol_sharpe)
        try:
            base_norm = float(np.sign(base_score) * np.log1p(abs(float(base_score))))
        except Exception:
            base_norm = 0.0
        if not np.isfinite(base_norm):
            base_norm = 0.0

        phi_auc = trapezoidal_gate(mean_auc, lower=0.52, sweet_spot=(0.56, 0.66), upper=0.72)
        phi_density = trapezoidal_gate(
            float(trades_per_day),
            lower=0.5,
            sweet_spot=(1.5, 5.0),
            upper=8.0,
        )
        try:
            modifier = float((phi_auc ** w_auc) * (phi_density ** w_den))
        except Exception:
            modifier = 0.0
        if not np.isfinite(modifier):
            modifier = 0.0

        return {
            "valid_events": int(len(event_idx)),
            "utility": float(utility),
            "auc": float(mean_auc) if np.isfinite(float(mean_auc)) else 0.5,
            "trades_per_day": float(trades_per_day),
            "calibration_brier": None,
            "calibration_ece": None,
            "sharpe_mean": float(np.mean(folds_sharpe_arr)),
            "sharpe_std": float(np.std(folds_sharpe_arr, ddof=1)) if folds_sharpe_arr.size > 1 else 0.0,
            "sharpe_min": float(np.min(folds_sharpe_arr)),
            "sharpe_max": float(np.max(folds_sharpe_arr)),
            "folds_sharpe_values": [float(v) for v in folds_sharpe_arr.tolist()],
            "per_fold_metrics": per_fold_metrics,
            "per_regime_metrics": per_regime_metrics,
            "n_trades": int(n_trades),
            "trade_mean_return": float(trade_mean),
            "trade_win_rate": float(trade_win_rate),
            "take_rate": float(take_rate),
            "consensus_mean": float(consensus_mean),
            "consensus_std": float(consensus_std),
            "consensus_p10": float(consensus_p10),
            "consensus_p50": float(consensus_p50),
            "consensus_p90": float(consensus_p90),
            "consensus_p99": float(consensus_p99),
            "consensus_min": float(consensus_min),
            "consensus_max": float(consensus_max),
            "consensus_frac_pos": float(frac_pos),
            "consensus_frac_neg": float(frac_neg),
            "consensus_threshold_effective": float(thr_effective),
            "consensus_quantile": float(consensus_quantile) if consensus_quantile is not None else None,
            "committee_expert_stats": committee_expert_stats,
            "sanity_checks": sanity_checks,
            "committee_overlap": committee_overlap,
            "committee_drivers": committee_drivers,
            "lambda_vol": float(lambda_vol),
            "w_auc": float(w_auc),
            "w_den": float(w_den),
            "avg_sharpe": float(avg_sharpe),
            "vol_sharpe": float(vol_sharpe),
            "base_score": float(base_score),
            "base_norm": float(base_norm),
            "phi_auc": float(phi_auc),
            "phi_density": float(phi_density),
            "modifier": float(modifier),
        }

    layer2_loaded_from: Optional[str] = None
    if stage_rank["layer2"] < start_rank:
        loaded_params, loaded_path = _load_stage_best_params("layer2")
        best_trading_params = dict(loaded_params or {})
        layer2_loaded_from = str(loaded_path) if loaded_path is not None else None
        l2_result = {"best_params": dict(best_trading_params), "best_value": None, "history": []}
        best_l2_score = float("nan")
        tprint_info(
            f"♻️ Layer 2 skipped (start_at={start_at_canonical}); loaded best params from {layer2_loaded_from}"
        )
    else:
        l2_optimizer = BayesianTPEOptimizer(
            config=OptimizationConfig(
                n_trials=20,  # was 2; expand search
                execution_mode="full",
                direction="maximize",
                seed=42,
                enable_staged_optimization=False,
                enable_adaptive_grid_refinement=False,
                enable_adaptive_optimization=False,
                enable_vectorbt_optimization=False,
                enable_hardware_optimization=False,
                n_startup_trials=5,
                tpe_trials=20,
            )
        )
        l2_result = l2_optimizer.optimize(objective=layer2_objective, search_space=layer2_search_space)
        best_trading_params = l2_result.get("best_params", {})
        best_l2_score = l2_result.get("best_value", 0.0)
    ts_l2 = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    try:
        if layer2_mode == "committee":
            l2_metrics = _compute_layer2_metrics_committee(best_trading_params)
        else:
            l2_metrics = _compute_layer2_metrics(best_trading_params)
        try:
            if isinstance(l2_metrics, dict):
                l2_metrics["layer2_mode"] = layer2_mode
        except Exception:
            pass
        tprint_info(
            "   Layer 2 metrics: "
            f"utility={l2_metrics.get('utility', 0.0):.4f}, "
            f"auc={l2_metrics.get('auc', 0.0):.4f}, "
            f"trades_per_day={l2_metrics.get('trades_per_day', 0.0):.2f}, "
            f"sharpe_mean={l2_metrics.get('sharpe_mean', 0.0):.4f}, "
            f"sharpe_std={l2_metrics.get('sharpe_std', 0.0):.4f}"
        )
        tprint_info(
            "   Layer 2 gates: "
            f"base_score={l2_metrics.get('base_score', 0.0):.4f}, "
            f"phi_auc={l2_metrics.get('phi_auc', 0.0):.4f}, "
            f"phi_density={l2_metrics.get('phi_density', 0.0):.4f}, "
            f"modifier={l2_metrics.get('modifier', 0.0):.4f}"
        )
    except Exception as l2_diag_exc:
        l2_metrics = {}
        tprint_warning(f"   ⚠️ Failed to compute Layer 2 metrics breakdown: {l2_diag_exc}")
    tprint_success(f"✅ Layer 2 Complete. Best Score: {best_l2_score:.4f}")
    tprint_info(f"   Best Trading Params: {best_trading_params}")

    # Persist Layer 2 params immediately
    try:
        l2_path = Path("outcomes") / f"hpo_layer2_best_params_{symbol}_{timeframe}_{ts_l2}.json"
        l2_payload = {
            "best_params": best_trading_params,
            "best_score": best_l2_score,
            "timestamp": ts_l2,
        }
        l2_path.parent.mkdir(parents=True, exist_ok=True)
        with open(l2_path, "w") as f:
            json.dump(l2_payload, f, indent=2, default=str)
        tprint_info(f"   💾 Saved Layer 2 best params to {l2_path}")
    except Exception as l2_exc:
        tprint_warning(f"   ⚠️ Failed to save Layer 2 params: {l2_exc}")

    # Persist Layer 2 trial metrics for correlation analysis
    l2_trials_path: Optional[Path] = None
    try:
        trial_rows = []
        for trial in l2_result.get("history", []):
            params = trial.get("params", {}) if isinstance(trial, dict) else {}
            if layer2_mode == "committee":
                metrics_trial = _compute_layer2_metrics_committee(params)
            else:
                metrics_trial = _compute_layer2_metrics(params)
            row = {
                "valid_events": metrics_trial.get("valid_events"),
                "utility": metrics_trial.get("utility"),
                "auc": metrics_trial.get("auc"),
                "trades_per_day": metrics_trial.get("trades_per_day"),
                "calibration_brier": metrics_trial.get("calibration_brier"),
                "calibration_ece": metrics_trial.get("calibration_ece"),
                "sharpe_mean": metrics_trial.get("sharpe_mean"),
                "sharpe_std": metrics_trial.get("sharpe_std"),
                "sharpe_min": metrics_trial.get("sharpe_min"),
                "sharpe_max": metrics_trial.get("sharpe_max"),
                "n_trades": metrics_trial.get("n_trades"),
                "trade_mean_return": metrics_trial.get("trade_mean_return"),
                "trade_win_rate": metrics_trial.get("trade_win_rate"),
                "take_rate": metrics_trial.get("take_rate"),
                "consensus_mean": metrics_trial.get("consensus_mean"),
                "consensus_std": metrics_trial.get("consensus_std"),
                "consensus_p10": metrics_trial.get("consensus_p10"),
                "consensus_p50": metrics_trial.get("consensus_p50"),
                "consensus_p90": metrics_trial.get("consensus_p90"),
                # Optional per-fold Sharpe values for deeper correlation
                "folds_sharpe_values": json.dumps(metrics_trial.get("folds_sharpe_values", [])),
                "lambda_vol": metrics_trial.get("lambda_vol"),
                "w_auc": metrics_trial.get("w_auc"),
                "w_den": metrics_trial.get("w_den"),
                "avg_sharpe": metrics_trial.get("avg_sharpe"),
                "vol_sharpe": metrics_trial.get("vol_sharpe"),
                "base_score": metrics_trial.get("base_score"),
                "base_norm": metrics_trial.get("base_norm"),
                "phi_auc": metrics_trial.get("phi_auc"),
                "phi_density": metrics_trial.get("phi_density"),
                "modifier": metrics_trial.get("modifier"),
            }
            for k, v in params.items():
                row[f"param_{k}"] = v
            trial_rows.append(row)

        if trial_rows:
            l2_trials_path = Path("outcomes") / f"hpo_layer2_trials_{symbol}_{timeframe}_{ts_l2}.csv"
            pd.DataFrame(trial_rows).to_csv(l2_trials_path, index=False)
            tprint_info(f"   💾 Saved Layer 2 trial metrics to {l2_trials_path}")
    except Exception as l2_trials_exc:
        tprint_warning(f"   ⚠️ Failed to save Layer 2 trial metrics: {l2_trials_exc}")

    # ------------------------------------------------------------------
    # Layer 2 Debug Diagnostics (single re-evaluation with best params)
    # ------------------------------------------------------------------
    try:
        tprint_info("   🔍 Layer 2 debug: re-evaluating best params with diagnostics...")
        debug_trail = float(best_trading_params.get("trail_distance_atr_mult", 0.0))
        debug_prof_thr = fixed_layer2_profit_thr
        debug_stop_thr = fixed_layer2_stop_thr
        (
            dbg_returns,
            dbg_labels,
            _,
            dbg_durations,
            dbg_mfe,
            dbg_mae,
            _, _
        ) = compute_realized_returns(
            market_data,
            primary_signals,
            profit_threshold=debug_prof_thr,
            stop_threshold=debug_stop_thr,
            horizon=12,
            transaction_cost=DEFAULT_TRANSACTION_COST,
            min_event_spacing=2,
            trail_distance_atr_mult=debug_trail,
            atr_series=atr_series,
        )
        dbg_valid_idx = ~dbg_labels.isna()
        dbg_valid_events = int(dbg_valid_idx.sum())
        if dbg_valid_events < 50:
            tprint_info(
                f"   Layer 2 debug: valid_events={dbg_valid_events} (<50), "
                "objective would early-return -1.0"
            )
        else:
            dbg_t_events = dbg_returns.index[dbg_valid_idx]
            dbg_returns_clean = dbg_returns[dbg_valid_idx]
            dbg_labels_clean = dbg_labels[dbg_valid_idx]
            # Rebuild event horizons and uniqueness
            t0_locs_dbg = pd.Series(np.arange(len(market_data)), index=market_data.index)
            start_locs_dbg = t0_locs_dbg.loc[dbg_t_events].values
            dur_vals_dbg = dbg_durations.loc[dbg_t_events].values.astype(int)
            end_locs_dbg = np.minimum(start_locs_dbg + dur_vals_dbg, len(market_data) - 1)
            t1_vals_dbg = market_data.index[end_locs_dbg]
            t1_series_dbg = pd.Series(t1_vals_dbg, index=dbg_t_events)
            batch_consistency_dbg = full_consistency.reindex(dbg_t_events).fillna(1.0).values
            batch_volatility_dbg = full_volatility.reindex(dbg_t_events).fillna(0).values
            batch_uniqueness_dbg = compute_uniqueness(t1_series_dbg, market_index=market_data.index)
            sample_weights_dbg = generate_weights_per_label(
                returns=dbg_returns_clean.values,
                t_events=dbg_t_events,
                close_series=None,
                consistency_scores=batch_consistency_dbg,
                uniqueness_scores=batch_uniqueness_dbg.values,
                vol_proxy=batch_volatility_dbg,
                **best_weighting_params,
            )
            # Subset meta-features
            X_dbg = meta_features_full.loc[dbg_valid_idx].fillna(0)
            # Fast model + CV
            n_cv_folds_dbg = 5
            fast_model_dbg = lgb.LGBMClassifier(
                n_estimators=60,
                max_depth=3,
                learning_rate=0.1,
                n_jobs=-1,
                verbose=-1,
                random_state=42,
            )
            try:
                cv_preds_dbg, folds_sharpe_dbg, mean_brier_dbg, mean_ece_dbg = _cross_val_predict_proba_and_fold_sharpes_weighted(
                    estimator=fast_model_dbg,
                    X=X_dbg,
                    y=dbg_labels_clean,
                    sample_weight=sample_weights_dbg,
                    n_splits=n_cv_folds_dbg,
                    returns=dbg_returns_clean.values.astype(float),
                    direction=direction,
                    prob_thr=0.5,
                    use_calibration=True,
                    enable_ev_gating=bool(config.get("enable_ev_gating", False)),
                    ev_margin=config.get("ev_margin", 0.0),
                )
            except Exception as dbg_cv_exc:
                tprint_warning(f"   ⚠️ Layer 2 debug: CV failed: {dbg_cv_exc}")
                cv_preds_dbg = np.full(dbg_valid_events, 0.5, dtype=float)
                folds_sharpe_dbg = np.array([0.0], dtype=float)
                mean_brier_dbg = None
                mean_ece_dbg = None
            # AUC
            try:
                mean_auc_dbg = roc_auc_score(dbg_labels_clean.values, cv_preds_dbg)
            except Exception:
                mean_auc_dbg = 0.5
            trades_per_day_dbg = len(dbg_returns_clean) / max(days_span, 1)
            # Reconstruct base_score as in calculate_hpo_utility
            avg_sharpe_dbg = float(np.mean(folds_sharpe_dbg))
            vol_sharpe_dbg = float(np.std(folds_sharpe_dbg, ddof=1)) if len(folds_sharpe_dbg) > 1 else 0.0
            base_score_dbg = avg_sharpe_dbg - 1.2 * vol_sharpe_dbg
            utility_dbg = calculate_hpo_utility(
                folds_sharpe=folds_sharpe_dbg,
                auc=mean_auc_dbg,
                trades_per_day=trades_per_day_dbg,
                lambda_vol=1.2,
                w_auc=1.0,
                w_den=0.5,
                calibration_brier=mean_brier_dbg,
                calibration_ece=mean_ece_dbg,
                w_cal=0.0,
            )
            tprint_info(
                "   Layer 2 debug: "
                f"valid_events={dbg_valid_events}, "
                f"AUC={mean_auc_dbg:.4f}, "
                f"trades_per_day={trades_per_day_dbg:.2f}"
            )
            tprint_info(
                "   Layer 2 debug: "
                f"folds_sharpe={folds_sharpe_dbg.tolist()}, "
                f"base_score={base_score_dbg:.4f}, "
                f"utility={utility_dbg:.4f}"
            )
    except Exception as dbg_exc:
        tprint_warning(f"   ⚠️ Layer 2 debug diagnostics failed: {dbg_exc}")


    # Save Layer 2 History
    l2_history_path: Optional[Path] = None
    try:
        ts_l2_hist = config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        l2_history_path = Path("outcomes") / f"hpo_layer2_history_{symbol}_{timeframe}_{ts_l2_hist}.json"
        with open(l2_history_path, "w") as f:
            json.dump(l2_result.get("history", []), f, default=str, indent=4)
        tprint_info(f"   💾 Saved Layer 2 history to {l2_history_path}")
    except Exception as e:
        tprint_warning(f"   ⚠️ Failed to save Layer 2 history: {e}")

    try:
        l2_report = _write_hpo_stage_report(
            outcomes_dir=outcomes_dir,
            run_timestamp=str(config.get("run_timestamp") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")),
            stage_id="layer2_trading",
            symbol=symbol,
            exchange=exchange,
            timeframe=timeframe,
            direction=direction,
            best_params=dict(best_trading_params) if isinstance(best_trading_params, dict) else {},
            metrics={
                "best_score": best_l2_score,
                **(l2_metrics if isinstance(l2_metrics, dict) else {}),
            },
            search_space=layer2_search_space,
            trials_csv_path=l2_trials_path,
            history_json_path=l2_history_path,
        )
        hpo_stage_reports["layer2"] = l2_report
    except Exception as l2_report_exc:
        tprint_warning(f"   ⚠️ Failed to write Layer 2 report: {l2_report_exc}")

    # ------------------------------------------------------------------
