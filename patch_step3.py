with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    code = f.read()

code = code.replace(
'''        n_samples = self.X.shape[0] if indices is None else len(indices)
        mask = np.ones(n_samples, dtype=bool)
        unresolved_groups: List[str] = []

        for group, slot_value in slot_map.items():
            if slot_value == "*":
                continue

            for cond_str in slot_value.split("&"):
                if "==" not in cond_str:
                    raise ValueError(f"Malformed slot {cond_str} in {canonical_key}")
                feature_name, target_val_raw = cond_str.split("==")
                target_val = int(target_val_raw)
                if feature_name in self.name_to_idx or feature_name in self.context_lookup:
                    mask &= self._resolve_feature_mask(feature_name, target_val, indices)
                else:
                    unresolved_groups.append(group)

        if unresolved_groups:
            if not set(unresolved_groups).issubset({"location", "regime"}):
                raise KeyError(
                    f"Cannot resolve groups {unresolved_groups} for key {canonical_key}"
                )
            context_mask = self._resolve_context_parent_mask(canonical_key, indices)
            if context_mask is None:
                raise KeyError(f"Cannot map {canonical_key} to a saved Stage A context")
            mask &= context_mask

        return mask''',
'''        n_samples = self.X.shape[0] if indices is None else len(indices)
        mask = np.ones(n_samples, dtype=bool)
        unresolved: List[Tuple[str, str]] = []

        for group, slot_value in slot_map.items():
            if slot_value == "*":
                continue

            for cond_str in slot_value.split("&"):
                if "==" not in cond_str:
                    raise ValueError(f"Malformed slot {cond_str} in {canonical_key}")
                feature_name, target_val_raw = cond_str.split("==")
                target_val = int(target_val_raw)
                if feature_name in self.name_to_idx or feature_name in self.context_lookup:
                    mask &= self._resolve_feature_mask(feature_name, target_val, indices)
                else:
                    unresolved.append((group, feature_name))

        if unresolved:
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
            mask &= context_mask

        return mask'''
)

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(code)
