import re

with open("extreme_price_movements/lgbm_based_mask_generation.py", "r") as f:
    source = f.read()

# We need to pass folds into _find_threshold_star so we can map each trade to its fold_id.
# old sig: _find_threshold_star(oof_preds: np.ndarray, fwd_ret: np.ndarray, data: pd.DataFrame, horizon: int, round_fee: float = 0.0015, forbid_concurrent: bool = True)
# new sig: add folds parameter.

sig_pattern = r"def _find_threshold_star\(\n        oof_preds: np\.ndarray,\n        fwd_ret: np\.ndarray,\n        data: pd\.DataFrame,\n        horizon: int,\n        round_fee: float = 0\.0015,\n        forbid_concurrent: bool = True\n    \) -> Tuple\[Optional\[float\], List\[Dict\[str, Any\]\], Dict\[str, Any\]\]:"

new_sig = """def _find_threshold_star(
        oof_preds: np.ndarray,
        fwd_ret: np.ndarray,
        data: pd.DataFrame,
        folds: List[Tuple[np.ndarray, np.ndarray]],
        horizon: int,
        round_fee: float = 0.0015,
        forbid_concurrent: bool = True
    ) -> Tuple[Optional[float], List[Dict[str, Any]], Dict[str, Any]]:"""

source = re.sub(sig_pattern, new_sig, source)

# And inside _find_threshold_star map folds:
fold_map_pattern = r'                    "symbol": selected_sym\[i\],\n                    "fold_idx": idx # We will map fold_id later, or we can just keep idx and map it\.\n                \}\)'

fold_map_replacement = """                    "symbol": selected_sym[i],
                    "fold_id": fold_map[idx]
                })"""

# create fold map early
fold_map_creation = """        valid_mask = np.isfinite(oof_preds) & np.isfinite(fwd_ret)

        # Build fold map
        fold_map = np.full(len(oof_preds), -1, dtype=int)
        for f_idx, (tr_idx, va_idx) in enumerate(folds):
            fold_map[va_idx] = f_idx
"""

source = source.replace("        valid_mask = np.isfinite(oof_preds) & np.isfinite(fwd_ret)", fold_map_creation)
source = re.sub(fold_map_pattern, fold_map_replacement, source)

# And update the call to _find_threshold_star in assess_rules
call_pattern = r"_find_threshold_star\(\n                        oof_preds=oof_preds,\n                        fwd_ret=target_ret,\n                        data=data,\n                        horizon=horizon,\n                        round_fee=0\.0015,\n                        forbid_concurrent=True\n                    \)"

call_replacement = """_find_threshold_star(
                        oof_preds=oof_preds,
                        fwd_ret=target_ret,
                        data=data,
                        folds=folds,
                        horizon=horizon,
                        round_fee=0.0015,
                        forbid_concurrent=True
                    )"""

source = re.sub(call_pattern, call_replacement, source)

with open("extreme_price_movements/lgbm_based_mask_generation.py", "w") as f:
    f.write(source)
    print("Patched timestamp and folds mapping.")
