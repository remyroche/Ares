import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# I will update the PNL Normalization and Risk Adjustment block in assess_rules.
old_block = r"""            # --- 5\. RIDGE PNL NORMALIZATION AND RISK ADJUSTMENT ---
            positive_pnls = assessment_df\.loc\[
                assessment_df\["ridge_pnl_raw"\] > 0, "ridge_pnl_raw"
            \]

            pnl_scale = float\(positive_pnls\.quantile\(0\.75\)\) if len\(positive_pnls\) > 0 else 1\.0
            if not np\.isfinite\(pnl_scale\) or pnl_scale <= 0:
                pnl_scale = 1\.0

            ridge_pnl_norm = np\.tanh\(
                np\.maximum\(assessment_df\["ridge_pnl_raw"\]\.to_numpy\(\), 0\.0\) / pnl_scale
            \)

            # Since we don't have fold-level PNL right now without refactoring how PNL is aggregated,
            # I will use the available `quality_stability_score` or compute it based on fold variance if present\.
            # In the prompt: "normalized_fold_pnl_std = std dev of ridge_pnl_norm between folds\.\.\. ridge_trade_stability_good = 1\.0 - normalized_fold_pnl_std"\.
            # If fold PNL is not available, default to 1\.0\. \(I'll add a default 1\.0 for now, but I should look for fold data\)\.
            ridge_trade_stability_good = 1\.0

            ridge_pnl_risk = ridge_pnl_norm \* np\.sqrt\(ridge_trade_stability_good\)"""

new_block = """            # --- 5. RIDGE PNL NORMALIZATION AND RISK ADJUSTMENT ---
            positive_pnls = assessment_df.loc[
                assessment_df["ridge_pnl_raw"] > 0, "ridge_pnl_raw"
            ]

            pnl_scale = float(positive_pnls.quantile(0.75)) if len(positive_pnls) > 0 else 1.0
            if not np.isfinite(pnl_scale) or pnl_scale <= 0:
                pnl_scale = 1.0

            ridge_pnl_norm = np.tanh(
                np.maximum(assessment_df["ridge_pnl_raw"].to_numpy(), 0.0) / pnl_scale
            )

            # Compute Fold Stability
            ridge_trade_stability_good = []
            for idx, row in assessment_df.iterrows():
                fold_pnl_dict = row.get("fold_pnl_raws", {})
                if not isinstance(fold_pnl_dict, dict) or len(fold_pnl_dict) < 2:
                    ridge_trade_stability_good.append(1.0)
                    continue

                # Normalize fold pnls with the same global pnl_scale
                fold_pnls = list(fold_pnl_dict.values())
                fold_norms = [np.tanh(max(float(p), 0.0) / pnl_scale) for p in fold_pnls]

                # std dev of ridge_pnl_norm between folds
                fold_std = float(np.std(fold_norms)) if len(fold_norms) > 1 else 0.0

                # normalize to [0,1]
                # Maximum std dev of values in [0,1] is 0.5 (e.g., [0,1]).
                # We can normalize it by dividing by 0.5 or cap it.
                normalized_fold_pnl_std = min(fold_std / 0.5, 1.0)
                ridge_trade_stability_good.append(1.0 - normalized_fold_pnl_std)

            ridge_trade_stability_good = np.array(ridge_trade_stability_good)

            ridge_pnl_risk = ridge_pnl_norm * np.sqrt(ridge_trade_stability_good)"""

source = re.sub(old_block, new_block, source, flags=re.DOTALL)

with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
    f.write(source)
    print("Patched normalization and fold stability logic.")
