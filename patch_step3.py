with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    code = f.read()

code = code.replace(
'''    def _get_mask_for_rule(self, key: str, X: np.ndarray) -> np.ndarray:
        """
        Parses '(F1==1)|(LOC1==0)|(*)' into a boolean mask.
        """
        parts = key.split('|')
        mask = np.ones(X.shape[0], dtype=bool)
        for p in parts:
            p = p.strip('()')
            if p == '*': continue
            if '==' in p:
                fname, val_part = p.split('==')
                val = int(val_part)
                # Find matching metadata for feature index
                f_idx = next(m.feature_index for m in self.metadata if m.feature_name == fname)
                mask &= (X[:, f_idx] == val)
        return mask''',
'''    def _get_mask_for_rule(self, key: str, X: np.ndarray) -> np.ndarray:
        """
        Parses '(F1==1)|(LOC1==0)|(*)' into a boolean mask.
        """
        parts = key.split('|')
        mask = np.ones(X.shape[0], dtype=bool)
        for p in parts:
            p = p.strip('()')
            if p == '*':
                continue
            for cond_str in p.split("&"):
                if '==' not in cond_str:
                    continue
                fname, val_part = cond_str.split('==')
                val = int(val_part)
                # Find matching metadata for feature index
                f_idx = next(m.feature_index for m in self.metadata if m.feature_name == fname)
                mask &= (X[:, f_idx] == val)
        return mask'''
)

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(code)
