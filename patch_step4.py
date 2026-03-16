with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    code = f.read()

code = code.replace(
'''        if unresolved:
            unresolved_groups = {g for g, _ in unresolved}
            unresolved_features = [f for _, f in unresolved]

            if not unresolved_groups.issubset({"location", "regime"}):
                raise KeyError(
                    f"Cannot resolve groups {unresolved_groups} for key {canonical_key}"
                )

            allow_context_fallback = all(f.startswith("ctx__") for f in unresolved_features)
            if not allow_context_fallback:
                raise KeyError(
                    f"Unresolved features {unresolved_features} in key {canonical_key}"
                )

            context_mask = self._resolve_context_parent_mask(canonical_key, indices)
            if context_mask is None:
                raise KeyError(f"Cannot map {canonical_key} to a saved Stage A context")
            mask &= context_mask''',
'''        if unresolved:
            unresolved_groups = {g for g, _ in unresolved}
            unresolved_features = [f for _, f in unresolved]

            if not unresolved_groups.issubset({"location", "regime"}):
                raise KeyError(
                    f"Cannot resolve groups {unresolved_groups} for key {canonical_key}"
                )

            # Stricter fallback safety: Allow context fallback if features explicitly
            # start with 'ctx__', OR if we successfully locate a parent context mask
            # mapped to this rule structure.
            context_mask = self._resolve_context_parent_mask(canonical_key, indices)
            allow_context_fallback = all(f.startswith("ctx__") for f in unresolved_features)

            if context_mask is None and not allow_context_fallback:
                raise KeyError(
                    f"Unresolved features {unresolved_features} in key {canonical_key}"
                )

            if context_mask is None:
                raise KeyError(f"Cannot map {canonical_key} to a saved Stage A context")

            mask &= context_mask'''
)

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(code)
