import re

with open('extreme_price_movements/features.py', 'r') as f:
    content = f.read()

target_str = '''        if candidate in feats:
            df = feats[candidate]
            if not isinstance(df, pd.DataFrame):
                continue

            # rank(axis=1, pct=True) computes cross-sectional percentile
            # We want to handle small cross sections safely.
            valid_counts = df.notna().sum(axis=1)

            # Compute rank
            cs_rank = df.rank(axis=1, pct=True)

            # Mask out timestamps with too few assets, emit neutral value 0.5
            mask = valid_counts < min_group_size
            if mask.any():
                cs_rank.loc[mask, :] = 0.5

            # Fill remaining NaNs with 0.5 (neutral)
            cs_rank = cs_rank.fillna(0.5).astype(np.float32)

            added_feats[f"cs_rank_{candidate}"] = cs_rank
            total_added += 1'''

new_str = '''        if candidate in feats:
            df = feats[candidate]
            if not isinstance(df, pd.DataFrame):
                continue

            # rank(axis=1, pct=True) computes cross-sectional percentile
            # We want to handle small cross sections safely.
            valid_counts = df.notna().sum(axis=1)
            mask = valid_counts < min_group_size

            # 1) Compute Cross-Sectional Rank
            cs_rank = df.rank(axis=1, pct=True)
            if mask.any():
                cs_rank.loc[mask, :] = 0.5
            cs_rank = cs_rank.fillna(0.5).astype(np.float32)
            added_feats[f"cs_rank_{candidate}"] = cs_rank

            # 2) Compute Cross-Sectional Robust Z-score (cs_rz)
            med = df.median(axis=1)
            mad = (df.sub(med, axis=0)).abs().median(axis=1)

            # MAD * 1.4826 for normal std proxy, bound by eps
            eps = 1e-6
            scale = (mad * 1.4826).clip(lower=eps)

            cs_rz = df.sub(med, axis=0).div(scale, axis=0)

            # Mask out insufficient group size with neutral 0.0
            if mask.any():
                cs_rz.loc[mask, :] = 0.0

            # Fill NaNs with 0.0 (neutral)
            cs_rz = cs_rz.fillna(0.0).astype(np.float32)
            added_feats[f"cs_rz_{candidate}"] = cs_rz

            total_added += 2'''

content = content.replace(target_str, new_str)

with open('extreme_price_movements/features.py', 'w') as f:
    f.write(content)
