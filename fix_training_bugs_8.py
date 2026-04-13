with open("extreme_price_movements/training.py", "r") as f:
    code = f.read()

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
code = code.replace("inner = _PKF(", "from sklearn.model_selection import PurgedKFold\n                    inner = PurgedKFold(")

with open("extreme_price_movements/training.py", "w") as f:
    f.write(code)
