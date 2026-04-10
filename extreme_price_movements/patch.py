with open("extreme_price_movements/training.py", "r") as f:
    text = f.read()

text = text.replace("""            _head_name_asym = f"{_bucket_key}_asym_h{int(_h)}"
            _model_asym = _configure_meta_reg(_head_name_asym, "aux_mfe_selector_cfg")""", """            _head_name_asym = f"{_bucket_key}_asym_h{int(_h)}"
            _model_asym = _configure_meta_reg(_head_name_asym, "aux_asym_selector_cfg")""")

with open("extreme_price_movements/training.py", "w") as f:
    f.write(text)
