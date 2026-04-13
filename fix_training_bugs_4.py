with open("extreme_price_movements/training.py", "r") as f:
    code = f.read()

import re

# Fix _PKF
code = code.replace("inner = _PKF(", "inner = PurgedKFold(")
code = code.replace("from sklearn.model_selection import PurgedKFold", "")
code = "from sklearn.model_selection import PurgedKFold\n" + code

# Fix _configure_meta_reg
code = code.replace(
    "m_mae_final = _configure_meta_reg(f\"mae_q70_final_{bucket_id}\", \"aux_mae_selector_cfg\")",
    "m_mae_final = _configure_meta_reg(\"mae_q70_final_\" + str(bucket_id), \"aux_mae_selector_cfg\")"
)
code = code.replace(
    "m_mfe_final = _configure_meta_reg(f\"mfe_final_{bucket_id}\", \"aux_mfe_selector_cfg\")",
    "m_mfe_final = _configure_meta_reg(\"mfe_final_\" + str(bucket_id), \"aux_mfe_selector_cfg\")"
)
code = code.replace(
    "m_asym_final = _configure_meta_reg(f\"asym_final_{bucket_id}\", \"aux_asym_selector_cfg\")",
    "m_asym_final = _configure_meta_reg(\"asym_final_\" + str(bucket_id), \"aux_asym_selector_cfg\")"
)

# Fix json referenced before assignment
code = code.replace(
    """                    try:
                        import json
                            with open(_meta_prev_path, "r", encoding="utf-8") as _f:
                            _meta_prev_sel = list(
                                (json.load(_f) or {}).get("selected_features", [])
                            )""",
    """                    try:
                        import json
                        with open(_meta_prev_path, "r", encoding="utf-8") as _f:
                            _meta_prev_sel = list(
                                (json.load(_f) or {}).get("selected_features", [])
                            )"""
)

# Clean up other variables
code = code.replace("dist_component = None", "pass")
code = code.replace("side_l = str(side).lower()", "")
code = code.replace("first_shape = _runs[0][\"lbl\"].values.shape", "")
code = code.replace("trades_day_30 = _avg_trades_per_day_global(", "pass #")
code = code.replace("tgt_top10 = float(np.mean(y_true[m_top10])) if m_top10.any() else float(\"nan\")", "")
code = code.replace("pred_top10 = float(np.mean(y_pred[m_top10])) if m_top10.any() else float(\"nan\")", "")
code = code.replace("alpha_half = max(1, len(alpha_models) // 2)", "")
code = code.replace("oof_u = np.full(n, np.nan, dtype=float)", "")
code = code.replace("p_oof = np.mean(p_oof_avg_parts, axis=0)", "")
code = code.replace("pred_logit = _logit_avg", "")

with open("extreme_price_movements/training.py", "w") as f:
    f.write(code)
