import re

with open("extreme_price_movements/mask_optimiser.py", "r") as f:
    code = f.read()

# Replace the "keep_idx" dominance loop in Phase 4.
old_pruning_block = """        base_rows = df_short[df_short["conditioner_mode"] == "none"].set_index("name")
        keep_idx: List[int] = []
        for idx, row in df_short.iterrows():
            cond_mode = str(row.get("conditioner_mode", "none"))
            if cond_mode == "none":
                keep_idx.append(idx)
                continue
            base_name = str(row["name"]).rsplit("_", 1)[0]
            if base_name not in base_rows.index:
                continue
            base_row = base_rows.loc[base_name]
            if (
                _metric_or_nan(row.get("score_ml_trading"))
                > _metric_or_nan(base_row.get("score_ml_trading"))
                and _metric_or_nan(row.get("delta_r"))
                >= _metric_or_nan(base_row.get("delta_r"))
                and _metric_or_nan(row.get("S_r"))
                >= _metric_or_nan(base_row.get("S_r"))
            ):
                keep_idx.append(idx)
        df_short = (
            df_short.loc[keep_idx]
            .sort_values("score_ml_trading", ascending=False)
            .copy()
        )"""

new_pruning_block = """        # Apply complexity penalties
        phase4_single_regime_penalty = float(cfg.get("phase4_single_regime_penalty", 0.95))
        phase4_two_regime_penalty = float(cfg.get("phase4_two_regime_penalty", 0.85))

        penalties = []
        for _, row in df_short.iterrows():
            tier = row.get("tier", 0)
            if tier == 1:
                penalties.append(phase4_single_regime_penalty)
            elif tier == 2:
                penalties.append(phase4_two_regime_penalty)
            else:
                penalties.append(1.0)

        df_short["complexity_multiplier"] = np.array(penalties, dtype=np.float32)
        df_short["score_ml_trading"] = df_short["score_ml_trading"].astype(np.float32).values * df_short["complexity_multiplier"].astype(np.float32).values
        df_short["shortlist_score"] = df_short["score_ml_trading"].astype(np.float32)

        # Dominance Pruning
        base_rows = df_short[df_short["tier"] == 0].copy()

        keep_idx: List[int] = []
        tolerance = float(cfg.get("phase4_dominance_tolerance", 0.90))

        # We process each candidate and see if a simpler candidate strictly dominates it
        for idx, row in df_short.iterrows():
            tier = row.get("tier", 0)
            if tier == 0:
                keep_idx.append(idx)
                continue

            base_name = str(row["name"]).split("_" + row["conditioner_mode"].replace(" ", "").replace(">", "gt").replace("<", "lt"))[0]
            if "AND" in str(row["name"]):
                base_name = str(row["name"]).split("_AND_")[0].rsplit("_", 1)[0]

            dominated = False
            # Compare against all simpler candidates of the same base
            simpler_cands = df_short[(df_short["tier"] < tier)]

            for _, s_row in simpler_cands.iterrows():
                # Rough check if they share the same base name (ignoring conditioner suffixes)
                if not str(s_row["name"]).startswith(base_name):
                    continue

                # A dominates B if:
                if (
                    _metric_or_nan(s_row.get("score_ml_trading")) >= _metric_or_nan(row.get("score_ml_trading")) and
                    _metric_or_nan(s_row.get("economic_gain_r")) >= _metric_or_nan(row.get("economic_gain_r")) and
                    _metric_or_nan(s_row.get("S_r")) >= _metric_or_nan(row.get("S_r")) and
                    _metric_or_nan(s_row.get("total_events")) >= _metric_or_nan(row.get("total_events")) * tolerance
                ):
                    dominated = True
                    break

            if not dominated:
                keep_idx.append(idx)

        df_short = (
            df_short.loc[keep_idx]
            .sort_values("score_ml_trading", ascending=False)
            .copy()
        )"""

code = code.replace(old_pruning_block, new_pruning_block)

with open("extreme_price_movements/mask_optimiser.py", "w") as f:
    f.write(code)
