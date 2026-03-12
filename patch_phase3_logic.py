import re

with open("extreme_price_movements/mask_optimiser.py", "r") as f:
    code = f.read()

start_str = '    cond_rows: List[pd.Series] = []'
end_str = '    if cond_rows:\n        df_short = pd.concat([df_short, pd.DataFrame(cond_rows)], ignore_index=True)'

idx_start = code.find(start_str)
idx_end = code.find(end_str)

if idx_start == -1 or idx_end == -1:
    print("Could not find boundaries")
    exit(1)

new_phase3_logic = """    cond_rows: List[pd.Series] = []
    if bool(cfg.get("enable_secondary_conditioners", True)):
        # Base limits
        min_events = int(cfg.get("phase3_min_conditioned_event_count", 2000))
        min_fraction = float(cfg.get("phase3_min_event_fraction_of_base", 0.10))
        tier2_min_fraction = float(cfg.get("phase3_tier2_min_event_fraction", 0.05))

        for _, row in df_short.iterrows():
            cand_name = str(row["name"])
            reg = candidate_registry[cand_name]
            z = int(int(reg["z_hours"]) * bph)
            zc = global_z_cache[z]
            base_masks = candidate_masks[cand_name]
            base_side_mask = _get_side_mask(mode, base_masks["m_high"], base_masks["m_low"])
            base_event_count = int(np.sum(base_side_mask))

            # Identify Tier-1 candidates
            tier1_candidates = []
            top_vars = dynamic_conditioners.get(cand_name, [])

            for var_info in top_vars:
                var_name = var_info["feature"]
                coef = var_info["coef"]
                v_type = var_info["type"]

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
                        "features": [var_name]
                    })
                else:
                    quantiles = [0.5, 0.6, 0.7, 0.8] if coef > 0 else [0.5, 0.4, 0.3, 0.2]
                    direction = "gt" if coef > 0 else "lt"
                    for q in quantiles:
                        threshold = np.nanpercentile(feature_vals[active_valid], q * 100)
                        if direction == "gt":
                            cond_mask = valid_mask & (feature_vals > threshold)
                            desc = f"{var_name} > p{int(q*100)}"
                        else:
                            cond_mask = valid_mask & (feature_vals < threshold)
                            desc = f"{var_name} < p{int(q*100)}"

                        tier1_candidates.append({
                            "name": f"{cand_name}_{var_name}_{desc.replace(' ', '').replace('>', 'gt').replace('<', 'lt')}",
                            "desc": desc,
                            "mask": cond_mask,
                            "features": [var_name]
                        })

            # Evaluate Tier-1
            surviving_tier1 = []

            def eval_candidate(c_info, tier):
                new_side_mask = base_side_mask & c_info["mask"]
                tot_events = int(np.sum(new_side_mask))

                req_fraction = min_fraction if tier == 1 else tier2_min_fraction
                if tot_events < min_events or (tot_events / base_event_count) < req_fraction:
                    return None

                # Full evaluation
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

                # Check acceptance rules
                base_econ = _metric_or_nan(row.get("economic_gain_r"))
                base_mfe = _metric_or_nan(row.get("aggregate_mfe_coverage"))
                new_econ = _metric_or_nan(new_metrics.get("economic_gain_r"))
                new_mfe = _metric_or_nan(new_metrics.get("aggregate_mfe_coverage"))

                improves_econ = (new_econ > base_econ * 1.05)
                improves_mfe = (new_mfe > base_mfe * 1.05)

                if tier == 1 and not (improves_econ or improves_mfe):
                    return None
                if tier == 2 and not (new_econ > base_econ * 1.1 or new_mfe > base_mfe * 1.1):
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

            for c_info in tier1_candidates:
                eval_res = eval_candidate(c_info, tier=1)
                if eval_res is not None:
                    surviving_tier1.append((c_info, eval_res))
                    cond_rows.append(eval_res)

            # Tier-2 Generation
            max_pairs = int(cfg.get("phase3_max_pair_candidates", 10))
            tier2_candidates = []

            for i in range(len(surviving_tier1)):
                for j in range(i + 1, len(surviving_tier1)):
                    if len(tier2_candidates) >= max_pairs:
                        break
                    c1_info, _ = surviving_tier1[i]
                    c2_info, _ = surviving_tier1[j]

                    # Avoid redundant pairs (same feature)
                    if set(c1_info["features"]).intersection(set(c2_info["features"])):
                        continue

                    combined_mask = c1_info["mask"] & c2_info["mask"]
                    tier2_candidates.append({
                        "name": f"{c1_info['name']}_AND_{c2_info['name'].replace(cand_name + '_', '')}",
                        "desc": f"{c1_info['desc']} AND {c2_info['desc']}",
                        "mask": combined_mask,
                        "features": c1_info["features"] + c2_info["features"]
                    })

            for c_info in tier2_candidates:
                eval_res = eval_candidate(c_info, tier=2)
                if eval_res is not None:
                    cond_rows.append(eval_res)

"""

new_code = code[:idx_start] + new_phase3_logic + code[idx_end:]

with open("extreme_price_movements/mask_optimiser.py", "w") as f:
    f.write(new_code)
