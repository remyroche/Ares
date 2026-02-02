
import os

filepath = 'src/training/steps/labeling/label_based_layer_2.py'

new_code = r'''        # --- MULTI-REGIME VECTORIZED SPEARMAN FILTER (Optimized) ---
        tprint_info("   📊 Calculating Multi-Regime Conditional Spearman (Vol, Trend, Liq) - Vectorized...")
        mi_start = time.time()

        from scipy.stats import rankdata

        target_series = numeric_df[target_name]
        valid_mask = target_series.notna()
        # Enforce float32 for performance as requested
        X_clean = numeric_df[valid_mask].fillna(0).replace([np.inf, -np.inf], 0).astype(np.float32)
        y_clean = target_series[valid_mask].astype(np.float32)

        if len(X_clean) > self.mi_min_samples:
            # --- 1. Define Regimes (Proxies) ---
            regime_masks = {}

            # A) High Volatility (Stress)
            if 'volatility_1d' in X_clean.columns:
                vol = X_clean['volatility_1d']
            else:
                # Proxy if missing
                close_col = X_clean['close'] if 'close' in X_clean.columns else X_clean.iloc[:, 0]
                vol = close_col.pct_change().rolling(96).std().fillna(0)

            vol_thr = vol.quantile(0.75)
            regime_masks['HIGH_VOL'] = (vol >= vol_thr).values

            # B) Strong Trend (Momentum)
            if 'close' in X_clean.columns:
                ret_roll = X_clean['close'].pct_change().rolling(24).mean().abs().fillna(0)
                trend_thr = ret_roll.quantile(0.75)
                regime_masks['STRONG_TREND'] = (ret_roll >= trend_thr).values

            # C) Low Liquidity (Friction)
            if 'close' in X_clean.columns and 'volume' in X_clean.columns:
                dvol = (X_clean['close'] * X_clean['volume']).replace(0, 1)
                amihud = (X_clean['close'].pct_change().abs() / dvol).fillna(0)
                liq_thr = amihud.quantile(0.75)
                regime_masks['LOW_LIQ'] = (amihud >= liq_thr).values

            # --- 2. Vectorized Spearman Correlation ---
            def vectorized_spearman(X_arr, y_arr):
                # X_arr: (n, k), y_arr: (n,)
                n = len(y_arr)
                if n < 5: return np.zeros(X_arr.shape[1], dtype=np.float32)

                # Ranks (average method)
                # Apply along axis 0 for X (columns)
                Xr = np.apply_along_axis(rankdata, 0, X_arr, method="average").astype(np.float32)
                yr = rankdata(y_arr, method="average").astype(np.float32)

                # Center
                Xr_c = Xr - Xr.mean(axis=0, keepdims=True)
                yr_c = yr - yr.mean()

                # Correlation: (Xr_c . yr_c) / sqrt(sum(Xr_c^2) * sum(yr_c^2))
                # Numerator: (k, n) @ (n,) -> (k,)
                # Note: Xr_c is (n, k), so Xr_c.T is (k, n)
                num = Xr_c.T @ yr_c

                # Denominator
                # sum_sq_X: (k,)
                sum_sq_X = np.sum(Xr_c**2, axis=0)
                sum_sq_y = np.sum(yr_c**2)

                den = np.sqrt(sum_sq_X * sum_sq_y)

                # Avoid division by zero
                rho = np.divide(num, den, out=np.zeros_like(num), where=den > 1e-9)
                return np.abs(rho)

            # Global Scores
            tprint_info("      ⚡ Computing global Spearman scores...")
            final_scores = vectorized_spearman(X_clean.values, y_clean.values)
            feature_origins = {col: "GLOBAL" for col in X_clean.columns}

            # Regime Scores
            for r_name, r_mask in regime_masks.items():
                mask_sum = np.sum(r_mask)
                if mask_sum > 20:
                    X_r = X_clean.values[r_mask]
                    y_r = y_clean.values[r_mask]

                    r_scores = vectorized_spearman(X_r, y_r)

                    # Max-Pooling
                    improved_idx = np.where(r_scores > final_scores)[0]
                    if len(improved_idx) > 0:
                        final_scores[improved_idx] = r_scores[improved_idx]
                        for idx in improved_idx:
                            feature_origins[X_clean.columns[idx]] = r_name

            # --- 3. Noise Baseline ---
            # Shuffle y to destroy relationship
            y_shuffled = np.random.permutation(y_clean.values)
            noise_scores = vectorized_spearman(X_clean.values, y_shuffled)
            noise_mean = np.mean(noise_scores)

            threshold = 1.2 * noise_mean

            if self.verbose:
                 tprint_info(f"      - Spearman Stats: Floor={noise_mean:.5f} | Threshold={threshold:.5f}")
                 rescue_counts = defaultdict(int)
                 for i, col in enumerate(X_clean.columns):
                     if final_scores[i] > threshold and feature_origins[col] != "GLOBAL":
                         rescue_counts[feature_origins[col]] += 1
                 if rescue_counts:
                      tprint_info(f"      - 🛡️ Regime Rescues: {dict(rescue_counts)}")

            # --- 4. Filtering ---
            # Identify columns to drop
            to_drop_indices = np.where(final_scores <= threshold)[0]
            to_drop_cols = X_clean.columns[to_drop_indices]

            # Protect essential columns
            essential = [target_name, 'TARGET_RET_1', 'close', 'volume', 'log_ret']
            to_drop_noise = [c for c in to_drop_cols if c not in essential]

            if to_drop_noise:
                tprint_info(f"   📉 Pruning {len(to_drop_noise)} low-correlation features (Score <= {threshold:.5f})...")
                numeric_df = numeric_df.drop(columns=to_drop_noise)

            if self.verbose:
                tprint_info(f"   ✅ Spearman filtering complete in {time.time() - mi_start:.1f}s")

        else:
            tprint_warning("   ⚠️ Insufficient data for Correlation calculation, skipping filter.")'''

with open(filepath, 'r') as f:
    content = f.read()

start_marker = '        # Skip expensive MI filtering if requested'
end_marker = '            tprint_warning("   ⚠️ Insufficient data for MI calculation, skipping filter.")'

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1:
    print("Error: Start marker not found")
    exit(1)
if end_idx == -1:
    print("Error: End marker not found")
    exit(1)

# Include the end marker in replacement range (we replace it with the new else block end)
end_idx_inclusive = end_idx + len(end_marker)

# Replace
updated_content = content[:start_idx] + new_code + content[end_idx_inclusive:]

with open(filepath, 'w') as f:
    f.write(updated_content)

print("Successfully updated layer 2 file.")
