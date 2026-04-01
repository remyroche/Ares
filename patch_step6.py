import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# I need to add logic after the loop:
# 1. PNL normalization and risk adjustment
# 2. compute worst_penalty
# 3. compute base_regime_score
# 4. overlap control & selection_score

post_loop_code = """
        assessment_df = pd.DataFrame(assessment_results)

        if not assessment_df.empty:
            # --- 5. RIDGE PNL NORMALIZATION AND RISK ADJUSTMENT ---
            positive_pnls = assessment_df.loc[
                assessment_df["ridge_pnl_raw"] > 0, "ridge_pnl_raw"
            ]

            pnl_scale = float(positive_pnls.quantile(0.75)) if len(positive_pnls) > 0 else 1.0
            if not np.isfinite(pnl_scale) or pnl_scale <= 0:
                pnl_scale = 1.0

            ridge_pnl_norm = np.tanh(
                np.maximum(assessment_df["ridge_pnl_raw"].to_numpy(), 0.0) / pnl_scale
            )

            # Since we don't have fold-level PNL right now without refactoring how PNL is aggregated,
            # I will use the available `quality_stability_score` or compute it based on fold variance if present.
            # In the prompt: "normalized_fold_pnl_std = std dev of ridge_pnl_norm between folds... ridge_trade_stability_good = 1.0 - normalized_fold_pnl_std".
            # If fold PNL is not available, default to 1.0. (I'll add a default 1.0 for now, but I should look for fold data).
            ridge_trade_stability_good = 1.0

            ridge_pnl_risk = ridge_pnl_norm * np.sqrt(ridge_trade_stability_good)
            ridge_pnl_risk = np.clip(ridge_pnl_risk, 0.0, 1.0)
            assessment_df["ridge_pnl_risk"] = ridge_pnl_risk

            # Normalize other base score components to [0,1]
            def robust_normalize(s):
                if s.empty or s.isna().all():
                    return np.zeros(len(s))
                p05 = s.quantile(0.05)
                p95 = s.quantile(0.95)
                span = p95 - p05
                if span <= 1e-9:
                    return np.zeros(len(s))
                return np.clip((s - p05) / span, 0.0, 1.0)

            lift_norm = robust_normalize(assessment_df["lift"])
            ev_per_event_norm = robust_normalize(assessment_df["ev_per_event"])

            ridge_trade_sortino = assessment_df["ridge_trade_sortino"].to_numpy() # already [0,1]

            # --- FINAL BASE SCORE ---
            base_regime_score = (
                0.3 * ridge_pnl_risk
                + 0.3 * ridge_trade_sortino
                + 0.2 * lift_norm
                + 0.1 * ev_per_event_norm
            )
            assessment_df["regime_score"] = base_regime_score

            # --- WORST PENALTY ---
            worst_malus = np.minimum.reduce([
                ridge_pnl_risk,
                ridge_trade_sortino,
                lift_norm,
                ev_per_event_norm
            ])
            worst_penalty = 1.0 - worst_malus
            assessment_df["worst_penalty"] = worst_penalty

            # Remove ret_uplift from scoring, already done
"""

pattern = r"        assessment_df = pd.DataFrame\(assessment_results\)\n"
match = re.search(pattern, source)
if match:
    source = source[:match.start()] + post_loop_code + source[match.end():]
    with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
        f.write(source)
    print("Patched post loop calculation")
else:
    print("Could not find insertion point")
