import re

with open("extreme_price_movements/mask_optimiser.py", "r") as f:
    code = f.read()

start_str = "    cond_rows: List[pd.Series] = []\n    if bool(cfg.get(\"enable_secondary_conditioners\", True)):"
end_str = "    if cond_rows:\n        df_short = pd.concat([df_short, pd.DataFrame(cond_rows)], ignore_index=True)"

idx_start = code.find(start_str)
idx_end = code.find(end_str)

new_phase3_logic = """    cond_rows: List[pd.Series] = []
    if bool(cfg.get("enable_secondary_conditioners", True)):
        # Configurable limits
        min_events = int(cfg.get("phase3_min_conditioned_event_count", 2000))
        min_fraction = float(cfg.get("phase3_min_event_fraction_of_base", 0.10))
        tier2_min_fraction = float(cfg.get("phase3_tier2_min_event_fraction", 0.05))
        max_singles = int(cfg.get("phase3_max_single_candidates_per_base", 4))
        max_pairs = int(cfg.get("phase3_max_pair_candidates", 10))

        for _, row in df_short.iterrows():
            cand_name = str(row["name"])
            reg = candidate_registry[cand_name]
            z = int(int(reg["z_hours"]) * bph)
            zc = global_z_cache[z]
            base_masks = candidate_masks[cand_name]
            base_side_mask = _get_side_mask(mode, base_masks["m_high"], base_masks["m_low"])
            base_event_count = int(np.sum(base_side_mask))

            # ---------------------------------------------------------
            # 3A. Generate Single-Regime Candidates (Tier-1)
            # ---------------------------------------------------------
            tier1_candidates = []
            top_vars = dynamic_conditioners.get(cand_name, [])

            for var_info in top_vars:
                var_name = var_info["feature"]
                coef = var_info["coef"]
                v_type = var_info["type"]
                family = var_info.get("family", "unknown")

                if var_name not in regime_features_df.columns:
                    continue

                feature_vals = regime_features_df[var_name].values
                valid_mask = np.isfinite(feature_vals)
                active_valid = base_side_mask & valid_mask
                if np.sum(active_valid) < 50:
                    continue

                if v_type == "binary":
                    target_val = 1 if coef > 0 else 0
                    cond_mask = valid_mask & (feature_vals == target_val)
                    tier1_candidates.append({
                        "name": f"{cand_name}_{var_name}_is_{target_val}",
                        "desc": f"{var_name} == {target_val}",
                        "mask": cond_mask,
                        "features": [var_name],
                        "families": [family]
                    })
                else:
                    direction = "gt" if coef > 0 else "lt"
                    thresholds_dict = var_info.get("thresholds")
                    if not thresholds_dict:
                        continue

                    quantiles_to_check = ["q50", "q60", "q70", "q80"] if coef > 0 else ["q50", "q40", "q30", "q20"]
                    for q_key in quantiles_to_check:
                        if q_key in thresholds_dict:
                            threshold = thresholds_dict[q_key]
                            if direction == "gt":
                                cond_mask = valid_mask & (feature_vals > threshold)
                                desc = f"{var_name} > {q_key}"
                            else:
                                cond_mask = valid_mask & (feature_vals < threshold)
                                desc = f"{var_name} < {q_key}"

                            tier1_candidates.append({
                                "name": f"{cand_name}_{var_name}_{desc.replace(' ', '').replace('>', 'gt').replace('<', 'lt')}",
                                "desc": desc,
                                "mask": cond_mask,
                                "features": [var_name],
                                "families": [family]
                            })

            # Base Evaluation Closure
            def eval_candidate(c_info, tier, parent_res=None):
                new_side_mask = base_side_mask & c_info["mask"]
                tot_events = int(np.sum(new_side_mask))

                req_fraction = min_fraction if tier == 1 else tier2_min_fraction
                if tot_events < min_events or (tot_events / base_event_count) < req_fraction:
                    return None

                coh = (
                    _coherence_metrics_single_side(new_side_mask, zc["b_up"], zc["s_up"], zc["m_up"])
                    if _mode_is_up(mode)
                    else _coherence_metrics_single_side(new_side_mask, zc["b_dn"], zc["s_dn"], zc["m_dn"])
                )

                valid_fwd_new = np.isfinite(global_signed_returns)
                non_event_new = (~new_side_mask) & valid_fwd_new
                basic_edge_new = (
                    float(np.nanmean(global_signed_returns[new_side_mask & valid_fwd_new]) - np.nanmean(global_signed_returns[non_event_new]))
                    if np.any(new_side_mask & valid_fwd_new) and np.any(non_event_new)
                    else 0.0
                )

                new_metrics = _compute_full_metrics_for_candidate(
                    mode,
                    new_side_mask,
                    shared,
                    feature_dict,
                    cfg,
                    float(coh["impulse_shape_dispersion"]),
                    float(basic_edge_new),
                )

                econ_metrics = _compute_tbm_economic_gain(shared, new_side_mask, mode, folds, cfg)
                mfe_metrics = _compute_mfe_coverage(shared, new_side_mask, cfg)
                new_econ = _metric_or_nan(econ_metrics.get("economic_gain_r"))
                new_mfe = _metric_or_nan(mfe_metrics.get("fixed_tp_mfe_coverage"))

                # In order to do base comparison, we need base_econ. If row doesn't have it, we must compute it.
                if "economic_gain_r" not in row:
                    base_econ_metrics = _compute_tbm_economic_gain(shared, base_side_mask, mode, folds, cfg)
                    base_econ = _metric_or_nan(base_econ_metrics.get("economic_gain_r"))
                    row["economic_gain_r"] = base_econ

                    base_mfe_metrics = _compute_mfe_coverage(shared, base_side_mask, cfg)
                    base_mfe = _metric_or_nan(base_mfe_metrics.get("fixed_tp_mfe_coverage"))
                    row["aggregate_mfe_coverage"] = base_mfe
                else:
                    base_econ = _metric_or_nan(row.get("economic_gain_r"))
                    base_mfe = _metric_or_nan(row.get("aggregate_mfe_coverage"))

                improves_econ = (new_econ > base_econ * 1.05)
                improves_mfe = (new_mfe > base_mfe * 1.05)

                # Check net regime value
                best_geom = econ_metrics.get("per_geometry_metrics", [{}])[0]
                labels_ER = best_geom.get("labels", np.array([]))

                base_best_geom = _compute_tbm_economic_gain(shared, base_side_mask, mode, folds, cfg).get("per_geometry_metrics", [{}])[0]
                labels_E = base_best_geom.get("labels", np.array([]))

                auc_ER = quick_ridge_auc(regime_features_df, labels_ER, new_side_mask, folds)
                auc_E = quick_ridge_auc(regime_features_df, labels_E, base_side_mask, folds)

                fwd_ret_ER = global_signed_returns[new_side_mask & valid_fwd_new]
                fwd_ret_E = global_signed_returns[base_side_mask & valid_fwd_new]

                nrv_score, nrv_diags = compute_net_regime_value(
                    returns_E=fwd_ret_E,
                    returns_ER=fwd_ret_ER,
                    delta_r_folds_E=np.array([float(np.nanmean(global_signed_returns[(base_side_mask & valid_fwd_new) & va])) for _, va in folds]),
                    delta_r_folds_ER=np.array([float(np.nanmean(global_signed_returns[(new_side_mask & valid_fwd_new) & va])) for _, va in folds]),
                    labels_E=labels_E[base_side_mask] if len(labels_E) == len(base_side_mask) else np.array([]),
                    labels_ER=labels_ER[new_side_mask] if len(labels_ER) == len(new_side_mask) else np.array([]),
                    auc_E=auc_E,
                    auc_ER=auc_ER,
                )

                new_metrics["net_regime_value"] = nrv_score

                # Stronger acceptance rules based on prompt
                der_ratio = nrv_diags["DER_ratio"]
                sr_ratio = nrv_diags["S_r_ratio"]

                # Check for deterioration
                is_stability_worse = (sr_ratio < 0.90)
                is_dispersion_worse = (der_ratio < 0.90)

                if tier == 1:
                    if not (improves_econ or improves_mfe or nrv_score > 1.05):
                        return None
                    if is_stability_worse or is_dispersion_worse:
                        return None

                if tier == 2:
                    # Compare against BEST single parent if provided
                    if parent_res is not None:
                        parent_econ = _metric_or_nan(parent_res.get("economic_gain_r"))
                        parent_mfe = _metric_or_nan(parent_res.get("aggregate_mfe_coverage"))
                        parent_nrv = _metric_or_nan(parent_res.get("net_regime_value"))

                        if not (new_econ > parent_econ * 1.05 or new_mfe > parent_mfe * 1.05 or nrv_score > parent_nrv * 1.05):
                            return None
                    else:
                        if not (new_econ > base_econ * 1.1 or new_mfe > base_mfe * 1.1 or nrv_score > 1.1):
                            return None
                    if is_stability_worse or is_dispersion_worse:
                        return None

                # Build row
                new_row = row.copy()
                new_row["name"] = c_info["name"]
                new_row["conditioner_mode"] = c_info["desc"]
                new_row["tier"] = tier
                new_row["total_events"] = tot_events
                new_row["impulse_shape_dispersion"] = float(coh["impulse_shape_dispersion"])

                for k, v in new_metrics.items():
                    new_row[k] = v

                new_row["delta_r_raw"] = float(basic_edge_new)
                new_row["delta_r_fallback"] = (
                    float(0.5 * new_row["incremental_information_delta_auc"])
                    if np.isfinite(new_row.get("incremental_information_delta_auc", np.nan))
                    else float("nan")
                )
                raw_val = _metric_or_nan(new_row["delta_r_raw"])
                new_row["delta_r"] = float(raw_val)

                return new_row

            # Evaluate Tier-1
            surviving_tier1 = []
            for c_info in tier1_candidates:
                eval_res = eval_candidate(c_info, tier=1)
                if eval_res is not None:
                    surviving_tier1.append((c_info, eval_res))

            # ---------------------------------------------------------
            # 3B. Select Top Single Regimes
            # ---------------------------------------------------------
            surviving_tier1.sort(key=lambda x: x[1].get("net_regime_value", 0.0), reverse=True)
            top_tier1 = surviving_tier1[:max_singles]

            for c_info, eval_res in top_tier1:
                cond_rows.append(eval_res)

            # ---------------------------------------------------------
            # 3C. Generate Two-Regime Candidates (Tier-2)
            # ---------------------------------------------------------
            tier2_candidates = []

            for i in range(len(top_tier1)):
                for j in range(i + 1, len(top_tier1)):
                    if len(tier2_candidates) >= max_pairs:
                        break
                    c1_info, r1 = top_tier1[i]
                    c2_info, r2 = top_tier1[j]

                    # Avoid redundant pairs (same feature)
                    if set(c1_info["features"]).intersection(set(c2_info["features"])):
                        continue

                    # Prefer cross-family combinations (skip if same family)
                    if set(c1_info["families"]).intersection(set(c2_info["families"])):
                        continue

                    combined_mask = c1_info["mask"] & c2_info["mask"]
                    # Determine best parent for relative comparison
                    best_parent_res = r1 if r1.get("net_regime_value", 0) > r2.get("net_regime_value", 0) else r2

                    tier2_candidates.append({
                        "name": f"{c1_info['name']}_AND_{c2_info['name'].replace(cand_name + '_', '')}",
                        "desc": f"{c1_info['desc']} AND {c2_info['desc']}",
                        "mask": combined_mask,
                        "features": c1_info["features"] + c2_info["features"],
                        "families": c1_info["families"] + c2_info["families"],
                        "best_parent_res": best_parent_res
                    })

            for c_info in tier2_candidates:
                parent_res = c_info.pop("best_parent_res")
                eval_res = eval_candidate(c_info, tier=2, parent_res=parent_res)
                if eval_res is not None:
                    cond_rows.append(eval_res)

"""

new_code = code[:idx_start] + new_phase3_logic + code[idx_end:]

with open("extreme_price_movements/mask_optimiser.py", "w") as f:
    f.write(new_code)
