import re

with open("extreme_price_movements/position_sizer_v2.py", "r") as f:
    code = f.read()

# 1. Update _run_model1_target_race to explicitly return y_final
old_race = """        # Fit final Edge
        self.edge_model = Model1Edge(target_name=best_name)
        if best_name == "rank_style_target" and timestamps is None:
            y_final = build_rank_target(raw_returns, mode="fold_local")
        else:
            y_final = candidates[best_name]
        self.edge_model.fit(X, y_final, sample_weight)"""

new_race = """        # Fit final Edge
        self.edge_model = Model1Edge(target_name=best_name)
        if best_name == "rank_style_target" and timestamps is None:
            y_final = build_rank_target(raw_returns, mode="fold_local")
        else:
            y_final = candidates[best_name]
        self.edge_model.fit(X, y_final, sample_weight)

        # Save winning target for dimension-safe residual generation in Model 3
        self.model1_y_final_ = y_final"""

code = code.replace(old_race, new_race)

# 2. Update _run_model3_oof_eval signature and residual calculation
old_m3_eval = """    def _run_model3_oof_eval(
        self,
        X: np.ndarray,
        raw_returns: np.ndarray,
        timestamps: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ):
        \"\"\"OOF eval for Uncertainty. Fits on Model 1 OOF residuals.\"\"\"
        valid_oof = np.isfinite(self.model1_oof_pred_)
        if not np.any(valid_oof):
            return

        y_true_for_res = raw_returns[valid_oof]"""

new_m3_eval = """    def _run_model3_oof_eval(
        self,
        X: np.ndarray,
        y_winning_target: np.ndarray,
        timestamps: Optional[np.ndarray],
        sample_weight: Optional[np.ndarray]
    ):
        \"\"\"OOF eval for Uncertainty. Fits on Model 1 OOF residuals.\"\"\"
        valid_oof = np.isfinite(self.model1_oof_pred_)
        if not np.any(valid_oof):
            return

        y_true_for_res = y_winning_target[valid_oof]"""

code = code.replace(old_m3_eval, new_m3_eval)

# 3. Update the call to _run_model3_oof_eval inside fit()
old_m3_call = """        # e) fit final Model 3 on OOF residual target (includes its own OOF eval)
        self._run_model3_oof_eval(X3, y_raw_net_return, timestamps, sample_weight)"""

new_m3_call = """        # e) fit final Model 3 on OOF residual target using the dimensionally accurate winning target
        self._run_model3_oof_eval(X3, self.model1_y_final_, timestamps, sample_weight)"""

code = code.replace(old_m3_call, new_m3_call)

with open("extreme_price_movements/position_sizer_v2.py", "w") as f:
    f.write(code)
