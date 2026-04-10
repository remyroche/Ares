with open("extreme_price_movements/training.py", "r") as f:
    text = f.read()

text = text.replace("""        def _asym_train_target_full():
            if asym_target_choice == "rank_pct":
                return _rank_pct_target(y_asym_fit)
            if asym_target_choice == "rank_tail_amp":
                return _rank_tail_amp_target(
                    y_asym_fit,
                    top_start=float(cfg.get("aux_head_rank_tail_start", 0.70)),
                    amp=float(cfg.get("aux_head_rank_tail_amp", 0.50)),
                )
            if asym_target_choice == "qbin_mid":
                return _qbin_mid_target(
                    y_asym_fit, n_bins=int(cfg.get("aux_asym_qbin_bins", 20))
                )
            return y_asym_fit""", """        def _asym_train_target_full():
            return y_asym_fit""")

with open("extreme_price_movements/training.py", "w") as f:
    f.write(text)
